from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:
    torch = None

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import DirectRLEnv
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

    ISAACLAB_RUNTIME_AVAILABLE = True
except Exception:  # pragma: no cover - local fallback for skeleton stage
    ISAACLAB_RUNTIME_AVAILABLE = False

    class DirectRLEnv:
        def __init__(self, cfg, render_mode=None):
            self.cfg = cfg
            self.render_mode = render_mode


from badminton_intercept.envs.scene_builder import SceneEntityConfigs, build_scene
from badminton_intercept.control.attitude_pd import compute_rate_pid_angacc
from badminton_intercept.control.ctbr_decoder import decode_ctbr_action
from badminton_intercept.envs.intercept_env_cfg import CTBR_THRUST_SCALE
from badminton_intercept.control.rotor_mixer import (
    rotor_thrust_to_body_wrench,
)
from badminton_intercept.envs.launch_sampler import (
    build_launch_library_metadata,
    build_launch_sampling_spec,
    compare_launch_library_metadata,
    evaluate_launch_candidates as evaluate_launch_candidates_pure,
    get_launch_sampler_mode,
    launch_violation_score as launch_violation_score_pure,
    resolve_launch_library_stage_paths,
    sample_launch_candidates,
    validate_launch_library_payload,
)
from badminton_intercept.mdp.curriculum import CurriculumManager
from badminton_intercept.mdp.observations import build_actor_critic_observations
from badminton_intercept.mdp.rewards import compute_rewards
from badminton_intercept.mdp.terminations import compute_dones
from badminton_intercept.physics.air_params import AirYamlParams, load_air_yaml_params
from badminton_intercept.physics.shuttle_aero import compute_drag_force_tensor
from badminton_intercept.randomization.reset_manager import reset_subenvs


@dataclass
class StepOutput:
    observations: dict[str, Any]
    rewards: Any
    dones: Any
    infos: dict[str, Any]


class InterceptEnv(DirectRLEnv):
    """Interception task environment with IsaacLab scene wiring and local fallback mode."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._scene_cfgs: SceneEntityConfigs | None = None
        self._drone = None
        self._shuttlecock = None
        self._net = None
        self._racket_contact_sensor = None
        self._net_contact_sensor = None
        self._racket_body_ids = None
        self.last_actions = None

        # Runtime randomization/state buffers (initialized lazily once env sizes are known).
        self._drag_length_m = None
        self._shuttle_mass_kg = None
        self._racket_restitution = None
        self._drone_mass_scale = None
        self._drone_inertia_scale = None
        self._drone_thrust_scale = None
        self._launch_hit_plane_z = None
        self._launch_hit_time_s = None
        self._launch_hit_point_local = None
        self._actions = None
        self._prev_actions = None
        self._last_contact = None
        self._air_params: AirYamlParams | None = None
        self._max_rotor_thrust_n = None
        self._allocation_matrix_inv = None
        self._arm_lengths_m = None
        self._rotor_angles_rad = None
        self._rotor_directions = None
        self._force_constants = None
        self._moment_constants = None
        self._inertia_diag = None
        self._mass_kg = None
        self._motor_time_constant_s = None
        self._rotor_thrust_cmd_n = None
        self._filtered_body_rate_rad_s = None
        self._rate_integral = None
        self._rotor_throttle = None
        self._rotor_noise_scale = 0.0
        self._force_body_ids = None
        self._last_done_reasons = None
        self._last_launch_stats = {}
        # 扫掠接触检测缓冲区
        self._prev_ball_pos_w = None
        self._prev_ball_lin_vel_w = None
        self._prev_racket_contact_pos_w = None
        self._prev_racket_normal_w = None
        self._contact_reference_valid = None
        self.enable_post_hit_tracking = bool(getattr(cfg, "enable_post_hit_tracking", False))
        self.post_hit = None
        self._hit_drone_pos_w = None
        self._hit_drone_quat_w = None
        self._hit_drone_lin_vel_w = None
        self._hit_drone_ang_vel_w = None
        self._launch_sampler_mode = get_launch_sampler_mode(cfg)
        self._launch_libraries: dict[int, dict[str, Any]] | None = None
        self._curriculum = CurriculumManager(
            window_size=int(getattr(cfg, "curriculum_window_episodes", 100)),
            promote_threshold=float(getattr(cfg, "curriculum_promote_success_rate", 0.8)),
            promote_iteration_streak=int(getattr(cfg, "curriculum_promote_iteration_streak", 10)),
        )
        self._curriculum_fixed_stage = bool(getattr(cfg, "curriculum_fixed_stage", False))
        initial_stage_id = getattr(cfg, "curriculum_initial_stage_id", None)
        if initial_stage_id is not None:
            self._curriculum.set_stage_by_id(int(initial_stage_id))

        if ISAACLAB_RUNTIME_AVAILABLE:
            super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        else:
            super().__init__(cfg=cfg, render_mode=render_mode)
            self.scene = self._setup_scene()

        if self._launch_sampler_mode == "trajectory_library":
            self._ensure_launch_libraries_loaded()

    def _setup_scene(self):
        scene_obj = build_scene(config={"cfg": self.cfg})

        if not ISAACLAB_RUNTIME_AVAILABLE:
            return scene_obj

        if not isinstance(scene_obj, SceneEntityConfigs):
            raise RuntimeError("Expected SceneEntityConfigs in IsaacLab mode.")

        self._scene_cfgs = scene_obj

        if scene_obj.drone_kind == "rigid_object":
            self._drone = RigidObject(scene_obj.drone_cfg)
            self.scene.rigid_objects["drone"] = self._drone
        else:
            self._drone = Articulation(scene_obj.drone_cfg)
            self.scene.articulations["drone"] = self._drone

        self._shuttlecock = RigidObject(scene_obj.shuttlecock_cfg)
        self._net = RigidObject(scene_obj.net_cfg)
        self.scene.rigid_objects["shuttlecock"] = self._shuttlecock
        self.scene.rigid_objects["net"] = self._net

        ground_cfg = GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=float(getattr(self.cfg, "ground_restitution", 0.1)),
            )
        )
        ground_spawned = False
        try:
            spawn_ground_plane(prim_path="/World/ground", cfg=ground_cfg)
            ground_spawned = True
        except Exception as exc:
            # Offline/local runs may not have access to the default remote ground USD.
            print(f"[warn] ground plane spawn skipped: {exc}")

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu" and ground_spawned:
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # Get the contact sensor from the scene (it's automatically created from cfg)
        self._racket_contact_sensor = self.scene.sensors.get("contact_sensor", None)
        self._net_contact_sensor = self.scene.sensors.get("net_contact_sensor", None)
        self._racket_body_ids = None

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _resolve_force_body_ids(self):
        if self._drone is None or not hasattr(self._drone, "find_bodies"):
            self._force_body_ids = None
            return
        if self._force_body_ids is not None:
            return

        candidate_patterns = ["^base_link$", ".*base_link.*", "^base$"]
        body_ids = []
        body_names = []
        for pattern in candidate_patterns:
            try:
                body_ids, body_names = self._drone.find_bodies(pattern)
            except Exception:
                body_ids, body_names = [], []
            if len(body_ids) > 0:
                break

        if len(body_ids) == 0:
            # Fallback to body-0 when base_link regex doesn't resolve on a custom USD.
            self._force_body_ids = torch.tensor([0], device=self.device, dtype=torch.int32)
            print("[warn] base_link not found for force application; fallback to body_id=0.")
            return

        self._force_body_ids = torch.tensor([int(body_ids[0])], device=self.device, dtype=torch.int32)
        chosen_name = body_names[0] if len(body_names) > 0 else str(int(body_ids[0]))
        print(f"[info] force application body resolved: id={int(body_ids[0])}, name={chosen_name}")

    def _resolve_racket_body_ids(self):
        if self._drone is None or not hasattr(self._drone, "find_bodies"):
            self._racket_body_ids = None
            return
        if self._racket_body_ids is not None:
            return

        pattern = str(getattr(self.cfg, "racket_body_name_expr", ".*[Bb]at.*"))
        body_ids = []
        body_names = []
        try:
            body_ids, body_names = self._drone.find_bodies(pattern)
        except Exception:
            body_ids, body_names = [], []

        if len(body_ids) == 0:
            self._racket_body_ids = torch.empty((0,), device=self.device, dtype=torch.int32)
            print(f"[warn] racket body not found with pattern '{pattern}'; fallback to root+offset proxy.")
            return

        resolved_ids = [int(body_id) for body_id in body_ids]
        self._racket_body_ids = torch.tensor(resolved_ids, device=self.device, dtype=torch.int32)
        print(f"[info] racket body resolved: ids={resolved_ids}, names={body_names}")

    def _get_contact_proxy_radius(self) -> float:
        racket_proxy_radius = float(getattr(self.cfg, "contact_radius_m", 0.09))
        shuttle_radius = float(getattr(self.cfg, "shuttle_radius_m", 0.015))
        # Proxy is center-to-center, so include the shuttle radius on top of the racket-face proxy radius.
        return max(shuttle_radius, racket_proxy_radius + shuttle_radius)

    def _sample_uniform(self, value_range: tuple[float, float], count: int, device=None):
        lo, hi = value_range
        return lo + (hi - lo) * torch.rand(count, device=device)

    def _evaluate_launch_candidates(
        self,
        pos_local: "torch.Tensor",
        vel_local: "torch.Tensor",
        drag_length_m: "torch.Tensor",
        launch_spec,
        hit_plane_z: "torch.Tensor",
    ) -> dict[str, "torch.Tensor"]:
        return evaluate_launch_candidates_pure(
            pos_local=pos_local,
            vel_local=vel_local,
            drag_length_m=drag_length_m,
            spec=launch_spec,
            hit_plane_z=hit_plane_z,
        )

    def _launch_violation_score(
        self,
        eval_data: dict[str, "torch.Tensor"],
        launch_spec,
    ) -> "torch.Tensor":
        return launch_violation_score_pure(eval_data=eval_data, spec=launch_spec)

    def _ensure_launch_libraries_loaded(self) -> None:
        if self._launch_sampler_mode != "trajectory_library" or self._launch_libraries is not None:
            return

        stage_paths = resolve_launch_library_stage_paths(self.cfg)
        libraries: dict[int, dict[str, Any]] = {}
        for stage, stage_path in zip(self._curriculum.stages, stage_paths):
            if not stage_path.exists():
                raise FileNotFoundError(
                    f"Trajectory library file not found for stage {stage.stage_id} ({stage.name}): {stage_path}"
                )

            payload = torch.load(stage_path, map_location="cpu")
            validate_launch_library_payload(payload)

            expected_metadata = build_launch_library_metadata(build_launch_sampling_spec(self.cfg, stage))
            mismatches = compare_launch_library_metadata(payload["metadata"], expected_metadata)
            if mismatches:
                mismatch_text = "; ".join(mismatches)
                raise RuntimeError(
                    f"Trajectory library metadata mismatch for stage {stage.stage_id} ({stage.name}) at "
                    f"{stage_path}: {mismatch_text}"
                )

            libraries[int(stage.stage_id)] = {
                "pos_local": payload["pos_local"].contiguous(),
                "vel_local": payload["vel_local"].contiguous(),
                "drag_length_m": payload["drag_length_m"].contiguous(),
                "net_clearance_z": payload["net_clearance_z"].contiguous(),
                "landing_xy": payload["landing_xy"].contiguous(),
                "hit_plane_z": payload["hit_plane_z"].contiguous(),
                "hit_time_s": payload["hit_time_s"].contiguous(),
                "hit_point_local": payload["hit_point_local"].contiguous(),
                "metadata": payload["metadata"],
                "path": stage_path,
            }

        self._launch_libraries = libraries

    def _resolve_air_yaml_path(self) -> str | None:
        cfg_yaml = str(getattr(self.cfg, "drone_param_yaml_path", "")).strip()
        if cfg_yaml and Path(cfg_yaml).exists():
            return cfg_yaml

        usd_path = str(getattr(self.cfg, "drone_usd_path", "")).strip()
        if usd_path:
            candidate = Path(usd_path).with_suffix(".yaml")
            if candidate.exists():
                return str(candidate)
        return None

    def _quat_rotate_body_to_world(self, quat_wxyz, vec_body):
        w = quat_wxyz[:, 0]
        x = quat_wxyz[:, 1]
        y = quat_wxyz[:, 2]
        z = quat_wxyz[:, 3]
        q_xyz = torch.stack([x, y, z], dim=-1)
        t = 2.0 * torch.cross(q_xyz, vec_body, dim=-1)
        return vec_body + w.unsqueeze(-1) * t + torch.cross(q_xyz, t, dim=-1)

    def _quat_rotate_world_to_body(self, quat_wxyz, vec_world):
        wxyz_conj = torch.stack([quat_wxyz[:, 0], -quat_wxyz[:, 1], -quat_wxyz[:, 2], -quat_wxyz[:, 3]], dim=-1)
        return self._quat_rotate_body_to_world(wxyz_conj, vec_world)

    def _ensure_runtime_buffers(self):
        if not (ISAACLAB_RUNTIME_AVAILABLE and torch is not None):
            return
        if self._drag_length_m is not None:
            return

        self._drag_length_m = torch.full(
            (self.num_envs,), float(getattr(self.cfg, "nominal_drag_length_m", 4.1)), device=self.device
        )
        self._shuttle_mass_kg = torch.full(
            (self.num_envs,), float(getattr(self.cfg, "shuttle_mass_kg", 0.005)), device=self.device
        )
        self._racket_restitution = torch.full((self.num_envs,), 0.82, device=self.device)
        self._drone_mass_scale = torch.ones((self.num_envs,), device=self.device)
        self._drone_inertia_scale = torch.ones((self.num_envs,), device=self.device)
        self._drone_thrust_scale = torch.ones((self.num_envs,), device=self.device)
        self._launch_hit_plane_z = torch.zeros((self.num_envs,), device=self.device)
        self._launch_hit_time_s = torch.zeros((self.num_envs,), device=self.device)
        self._launch_hit_point_local = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_contact = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        self._air_params = load_air_yaml_params(self._resolve_air_yaml_path())
        ap = self._air_params

        self._mass_kg = torch.full((self.num_envs,), float(ap.mass_kg), device=self.device)
        self._inertia_diag = torch.tensor(ap.inertia_diag, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self._arm_lengths_m = torch.tensor(ap.arm_lengths_m, device=self.device)
        self._rotor_angles_rad = torch.tensor(ap.rotor_angles_rad, device=self.device)
        self._rotor_directions = torch.tensor(ap.directions, device=self.device)
        self._force_constants = torch.tensor(ap.force_constants, device=self.device)
        self._moment_constants = torch.tensor(ap.moment_constants, device=self.device)
        self._motor_time_constant_s = max(float(ap.motor_time_constant_s), 1.0e-3)
        self._rotor_noise_scale = float(ap.noise_scale)

        max_rot_vel = torch.tensor(ap.max_rot_vel_rad_s, device=self.device)
        self._max_rotor_thrust_n = self._force_constants * max_rot_vel.square()
        arm_length = float(self._arm_lengths_m[0].item())
        k_coef = max(float(getattr(self.cfg, "fm_k_coef", 0.0178)), 1.0e-6)
        s = (2.0**0.5) / (4.0 * arm_length)
        yaw = 1.0 / (4.0 * k_coef)
        self._allocation_matrix_inv = torch.tensor(
            [
                [0.25, s, -s, yaw],
                [0.25, -s, -s, -yaw],
                [0.25, -s, s, yaw],
                [0.25, s, s, -yaw],
            ],
            device=self.device,
        )

        num_rotors = int(self._max_rotor_thrust_n.shape[0])
        self._rotor_thrust_cmd_n = torch.zeros((self.num_envs, num_rotors), device=self.device)
        self._filtered_body_rate_rad_s = torch.zeros((self.num_envs, 3), device=self.device)
        self._rate_integral = torch.zeros((self.num_envs, 3), device=self.device)
        self._rotor_throttle = torch.zeros((self.num_envs, num_rotors), device=self.device)
        # 扫掠接触检测缓冲区
        self._prev_ball_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_ball_lin_vel_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_racket_contact_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_racket_normal_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._contact_reference_valid = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.post_hit = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._hit_drone_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._hit_drone_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self._hit_drone_quat_w[:, 0] = 1.0
        self._hit_drone_lin_vel_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._hit_drone_ang_vel_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._resolve_force_body_ids()
        self._resolve_racket_body_ids()

    def _get_drone_kinematics(self):
        if self._drone is None:
            zeros3 = torch.zeros((self.num_envs, 3), device=self.device)
            zeros4 = torch.zeros((self.num_envs, 4), device=self.device)
            zeros4[:, 0] = 1.0
            return zeros3, zeros4, zeros3, zeros3

        # For articulation assets, keep control-state aligned with the same body that receives wrench
        # (typically base_link). This avoids root-vs-link state mismatch in the feedback loop.
        if (
            self._force_body_ids is not None
            and hasattr(self._drone, "data")
            and hasattr(self._drone.data, "body_pos_w")
            and hasattr(self._drone.data, "body_quat_w")
            and hasattr(self._drone.data, "body_lin_vel_w")
            and hasattr(self._drone.data, "body_ang_vel_w")
        ):
            body_id = int(self._force_body_ids[0].item())
            return (
                self._drone.data.body_pos_w[:, body_id, :],
                self._drone.data.body_quat_w[:, body_id, :],
                self._drone.data.body_lin_vel_w[:, body_id, :],
                self._drone.data.body_ang_vel_w[:, body_id, :],
            )

        return (
            self._drone.data.root_pos_w,
            self._drone.data.root_quat_w,
            self._drone.data.root_lin_vel_w,
            self._drone.data.root_ang_vel_w,
        )

    def _get_racket_center_pos_w(self, drone_pos_w):
        self._resolve_racket_body_ids()
        if (
            self._racket_body_ids is not None
            and int(self._racket_body_ids.numel()) > 0
            and self._drone is not None
            and hasattr(self._drone, "data")
            and hasattr(self._drone.data, "body_pos_w")
        ):
            return self._drone.data.body_pos_w[:, self._racket_body_ids, :].mean(dim=1)

        offset_z = float(getattr(self.cfg, "racket_offset_z_m", 0.2))
        racket_pos = drone_pos_w.clone()
        racket_pos[:, 2] += offset_z
        return racket_pos

    def _get_racket_normal_w(self, drone_quat_w):
        self._resolve_racket_body_ids()
        if (
            self._racket_body_ids is not None
            and int(self._racket_body_ids.numel()) > 0
            and self._drone is not None
            and hasattr(self._drone, "data")
            and hasattr(self._drone.data, "body_quat_w")
        ):
            racket_quat_w = self._drone.data.body_quat_w[:, self._racket_body_ids[0], :]
            return self._quat_to_up_axis_w(racket_quat_w)
        return self._quat_to_up_axis_w(drone_quat_w)

    def _quat_to_yaw(self, quat_wxyz):
        w = quat_wxyz[:, 0]
        x = quat_wxyz[:, 1]
        y = quat_wxyz[:, 2]
        z = quat_wxyz[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    def _quat_to_up_axis_w(self, quat_wxyz):
        # Rotation matrix third column (world Z-axis of body frame).
        w = quat_wxyz[:, 0]
        x = quat_wxyz[:, 1]
        y = quat_wxyz[:, 2]
        z = quat_wxyz[:, 3]
        up_x = 2.0 * (x * z + w * y)
        up_y = 2.0 * (y * z - w * x)
        up_z = 1.0 - 2.0 * (x * x + y * y)
        return torch.stack([up_x, up_y, up_z], dim=-1)

    def _compute_contact_proxy(self, racket_pos_w, ball_pos_w):
        radius = self._get_contact_proxy_radius()
        dist = torch.linalg.norm(ball_pos_w - racket_pos_w, dim=-1)
        return dist <= radius, dist

    def _compute_swept_contact_proxy(self, racket_pos_w, ball_pos_w):
        """计算扫掠接触检测，处理高速穿模问题"""
        instant_contact, instant_dist = self._compute_contact_proxy(racket_pos_w, ball_pos_w)

        # 检查前一帧状态是否有效
        if (
            self._prev_ball_pos_w is None
            or self._prev_racket_contact_pos_w is None
            or self._contact_reference_valid is None
            or not bool(torch.any(self._contact_reference_valid))
        ):
            return instant_contact, instant_dist

        # 计算相对运动
        rel_prev = self._prev_ball_pos_w - self._prev_racket_contact_pos_w
        rel_curr = ball_pos_w - racket_pos_w
        rel_delta = rel_curr - rel_prev

        # 计算沿运动方向的最近点参数 t
        denom = torch.sum(rel_delta * rel_delta, dim=-1)
        numer = -torch.sum(rel_prev * rel_delta, dim=-1)
        t_star = torch.where(denom > 1.0e-9, (numer / denom).clamp(0.0, 1.0), torch.zeros_like(denom))

        # 求解最近点位置
        closest_rel = rel_prev + t_star.unsqueeze(-1) * rel_delta
        swept_dist = torch.linalg.norm(closest_rel, dim=-1)

        # 扫掠接触判定
        radius = self._get_contact_proxy_radius()
        swept_contact = swept_dist <= radius

        # 只在有效的前一帧状态下使用扫掠结果
        valid = self._contact_reference_valid
        swept_contact = swept_contact & valid
        swept_dist = torch.where(valid, swept_dist, instant_dist)

        return swept_contact, swept_dist

    def _compute_hit_velocity_confirmation(self, ball_lin_vel_w, racket_normal_w, sensor_contact):
        if (
            ball_lin_vel_w is None
            or racket_normal_w is None
            or self._prev_ball_lin_vel_w is None
            or self._prev_racket_normal_w is None
            or self._contact_reference_valid is None
        ):
            return sensor_contact

        valid = self._contact_reference_valid
        if not bool(torch.any(valid)):
            return sensor_contact

        normal_ref = self._prev_racket_normal_w
        normal_ref = normal_ref / torch.linalg.norm(normal_ref, dim=-1, keepdim=True).clamp_min(1.0e-6)
        vn_prev = torch.sum(self._prev_ball_lin_vel_w * normal_ref, dim=-1)
        vn_post = torch.sum(ball_lin_vel_w * normal_ref, dim=-1)
        delta_vn = torch.abs(vn_post - vn_prev)
        threshold = float(getattr(self.cfg, "hit_normal_velocity_delta_threshold_mps", 1.0))
        confirmed = delta_vn > threshold
        confirmed = confirmed & valid
        return torch.where(valid, confirmed, sensor_contact)

    def _compute_contact_signal(self, racket_pos_w, ball_pos_w, ball_lin_vel_w=None, racket_normal_w=None):
        # Stable hit detection combines PhysX contacts with a continuous relative-motion sweep.
        proxy_contact, instant_dist = self._compute_contact_proxy(racket_pos_w, ball_pos_w)
        swept_contact, swept_dist = self._compute_swept_contact_proxy(racket_pos_w, ball_pos_w)
        # 综合判定：瞬时接触 OR 扫掠接触
        geometric_contact = proxy_contact | swept_contact
        d_axis = torch.minimum(instant_dist, swept_dist)

        sensor_contact = torch.zeros_like(geometric_contact)

        if self._racket_contact_sensor is None:
            candidate_contact = geometric_contact
        else:
            try:
                threshold = float(getattr(self.cfg, "contact_force_threshold", 0.1))
                force_matrix_hist = getattr(self._racket_contact_sensor.data, "force_matrix_w_history", None)
                if force_matrix_hist is not None:
                    sensor_force_mag = torch.norm(force_matrix_hist, dim=-1)  # (N, H, B, F)
                    sensor_contact = sensor_force_mag.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0] > threshold
                else:
                    net_forces_hist = getattr(self._racket_contact_sensor.data, "net_forces_w_history", None)
                    if net_forces_hist is not None:
                        if self._racket_body_ids is None or int(self._racket_body_ids.numel()) == 0:
                            force_mag = torch.norm(net_forces_hist, dim=-1)  # (N, H, B)
                        else:
                            force_mag = torch.norm(net_forces_hist[:, :, self._racket_body_ids, :], dim=-1)  # (N, H, Br)
                        sensor_contact = force_mag.max(dim=1)[0].max(dim=1)[0] > threshold
            except Exception:
                sensor_contact = torch.zeros_like(geometric_contact)
            candidate_contact = sensor_contact | geometric_contact

        velocity_confirmed = self._compute_hit_velocity_confirmation(
            ball_lin_vel_w=ball_lin_vel_w,
            racket_normal_w=racket_normal_w,
            sensor_contact=sensor_contact,
        )
        contact = candidate_contact & velocity_confirmed
        
        # 任务二模式：更新 post-hit 标志并保存 hit 时刻的无人机状态
        # 只有当击球姿态正确（球拍朝向对方场地，normal_x < 0）时才追踪球
        if self.enable_post_hit_tracking and self.post_hit is not None and contact.any():
            from badminton_intercept.mdp.rewards import compute_racket_normal_x_component
            drone_pos_w, drone_quat_w, drone_lin_vel_w, drone_ang_vel_w = self._get_drone_kinematics()
            normal_x = compute_racket_normal_x_component(drone_quat_w)
            correct_posture = normal_x < 0.0
            
            newly_hit = contact & (~self.post_hit) & correct_posture
            if newly_hit.any():
                self.post_hit[newly_hit] = True
                self._hit_drone_pos_w[newly_hit] = drone_pos_w[newly_hit]
                self._hit_drone_quat_w[newly_hit] = drone_quat_w[newly_hit]
                self._hit_drone_lin_vel_w[newly_hit] = drone_lin_vel_w[newly_hit]
                self._hit_drone_ang_vel_w[newly_hit] = drone_ang_vel_w[newly_hit]
        
        return contact, d_axis

    def _capture_contact_reference_state(self, env_ids=None):
        """捕获当前帧状态作为下一帧的参考，用于扫掠接触检测"""
        if not (ISAACLAB_RUNTIME_AVAILABLE and torch is not None):
            return
        if (
            self._shuttlecock is None
            or not hasattr(self._shuttlecock, "data")
            or self._prev_ball_pos_w is None
            or self._prev_racket_contact_pos_w is None
            or self._contact_reference_valid is None
        ):
            return

        env_ids = self._resolve_env_ids(env_ids)
        if env_ids is None:
            return

        drone_pos_w, drone_quat_w, _, _ = self._get_drone_kinematics()
        racket_contact_pos_w = self._get_racket_center_pos_w(drone_pos_w)
        ball_pos_w = self._shuttlecock.data.root_pos_w
        ball_lin_vel_w = self._shuttlecock.data.root_lin_vel_w
        racket_normal_w = self._get_racket_normal_w(drone_quat_w)
        self._prev_ball_pos_w[env_ids] = ball_pos_w[env_ids]
        self._prev_ball_lin_vel_w[env_ids] = ball_lin_vel_w[env_ids]
        self._prev_racket_contact_pos_w[env_ids] = racket_contact_pos_w[env_ids]
        self._prev_racket_normal_w[env_ids] = racket_normal_w[env_ids]
        self._contact_reference_valid[env_ids] = True

    def _compute_net_contact_signal(self):
        if self._net_contact_sensor is None:
            return torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        try:
            threshold = float(getattr(self.cfg, "net_contact_force_threshold", 0.05))

            force_matrix_hist = getattr(self._net_contact_sensor.data, "force_matrix_w_history", None)
            if force_matrix_hist is not None:
                force_mag = torch.norm(force_matrix_hist, dim=-1)
                return force_mag.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0] > threshold

            net_forces_hist = getattr(self._net_contact_sensor.data, "net_forces_w_history", None)
            if net_forces_hist is not None:
                force_mag = torch.norm(net_forces_hist, dim=-1)
                return force_mag.max(dim=1)[0].max(dim=1)[0] > threshold
        except Exception:
            pass

        return torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    def _compute_bound_distance(self, drone_pos_w):
        # Boundaries are defined in each env-local court frame (not global world frame).
        # 使用无人机的最下面点来计算边界（中心高度 - 无人机底部距中心的高度）
        if hasattr(self.scene, "env_origins"):
            drone_pos_w = drone_pos_w - self.scene.env_origins
        # 假设无人机底部距中心约 0.13m (臂长 0.125m + 螺旋桨半径)
        drone_bottom_offset = 0.13
        drone_bottom_z = drone_pos_w[:, 2] - drone_bottom_offset

        x_min, x_max = 0.0, 6.7
        y_min, y_max = -3.05, 3.05
        z_min, z_max = 0.5, 3.0  # 按最下面计算：低于50cm开始惩罚

        dx = torch.maximum(torch.zeros_like(drone_pos_w[:, 0]), x_min - drone_pos_w[:, 0]) + torch.maximum(
            torch.zeros_like(drone_pos_w[:, 0]), drone_pos_w[:, 0] - x_max
        )
        dy = torch.maximum(torch.zeros_like(drone_pos_w[:, 1]), y_min - drone_pos_w[:, 1]) + torch.maximum(
            torch.zeros_like(drone_pos_w[:, 1]), drone_pos_w[:, 1] - y_max
        )
        dz = torch.maximum(torch.zeros_like(drone_bottom_z), z_min - drone_bottom_z) + torch.maximum(
            torch.zeros_like(drone_bottom_z), drone_bottom_z - z_max
        )
        return torch.sqrt(dx * dx + dy * dy + dz * dz)

    def _enforce_linear_speed_limit(self):
        if self._drone is None or not hasattr(self._drone, "write_root_velocity_to_sim"):
            return
        max_speed = float(getattr(self.cfg, "max_linear_speed_mps", 20.0))
        if max_speed <= 0.0:
            return

        lin_vel_w = self._drone.data.root_lin_vel_w
        speed = torch.linalg.norm(lin_vel_w, dim=-1, keepdim=True)
        over = speed > max_speed
        if not bool(torch.any(over)):
            return

        scale = torch.clamp(max_speed / speed.clamp_min(1.0e-6), max=1.0)
        clamped_lin_vel_w = lin_vel_w * scale
        ang_vel_w = self._drone.data.root_ang_vel_w
        root_vel_w = torch.cat([clamped_lin_vel_w, ang_vel_w], dim=-1)
        env_ids = torch.nonzero(over.squeeze(-1), as_tuple=False).squeeze(-1)
        self._drone.write_root_velocity_to_sim(root_vel_w[env_ids], env_ids)

    def _pre_physics_step(self, actions):
        self.last_actions = actions
        if ISAACLAB_RUNTIME_AVAILABLE and torch is not None:
            if self._actions is None:
                self._actions = torch.zeros((self.num_envs, 4), device=self.device)
                self._prev_actions = torch.zeros_like(self._actions)
            self._prev_actions.copy_(self._actions)
            self._actions.copy_(actions)
            # 捕获当前帧状态，用于下一帧的扫掠接触检测
            self._capture_contact_reference_state()

    def _apply_action(self):
        if not (ISAACLAB_RUNTIME_AVAILABLE and torch is not None):
            self._apply_external_forces()
            return None
        if self._drone is None or self._actions is None:
            self._apply_external_forces()
            return None

        self._ensure_runtime_buffers()
        self._enforce_linear_speed_limit()

        _, drone_quat_w, _, drone_ang_vel_w = self._get_drone_kinematics()
        body_rate_rad_s = self._quat_rotate_world_to_body(drone_quat_w, drone_ang_vel_w)

        effective_mass = self._mass_kg * self._drone_mass_scale + float(getattr(self.cfg, "fm_extra_mass_kg", 0.147))
        target_body_rate, target_thrust_ref = decode_ctbr_action(
            self._actions,
            rate_scale_rad_s=float(self.cfg.ctbr_rate_max_rad_s),
            thrust_scale=float(getattr(self.cfg, "ctbr_thrust_scale", CTBR_THRUST_SCALE)),
        )
        axis_sign = torch.tensor(
            getattr(self.cfg, "ctbr_body_rate_axis_sign", (-1.0, -1.0, 1.0)),
            device=self.device,
            dtype=target_body_rate.dtype,
        ).unsqueeze(0)
        target_body_rate = target_body_rate * axis_sign

        dt = float(getattr(self.cfg.sim, "dt", 1.0 / 200.0))
        p_gain = torch.tensor(getattr(self.cfg, "fm_p_gain", (0.11, 0.11, 0.2)), device=self.device).unsqueeze(0)
        i_gain = torch.tensor(getattr(self.cfg, "fm_i_gain", (0.008, 0.008, 0.01)), device=self.device).unsqueeze(0)
        d_gain = torch.tensor(getattr(self.cfg, "fm_d_gain", (0.00075, 0.00075, 0.0)), device=self.device).unsqueeze(0)
        k_gain = torch.tensor(getattr(self.cfg, "fm_rate_k", (1.0, 1.0, 1.0)), device=self.device).unsqueeze(0)
        int_lim = torch.tensor(getattr(self.cfg, "fm_int_lim", (1.0, 1.0, 1.0)), device=self.device).unsqueeze(0)

        angacc_des, self._rate_integral, self._filtered_body_rate_rad_s = compute_rate_pid_angacc(
            target_body_rate_rad_s=target_body_rate,
            current_body_rate_rad_s=body_rate_rad_s,
            prev_filtered_body_rate_rad_s=self._filtered_body_rate_rad_s,
            integ_error=self._rate_integral,
            dt=dt,
            p_gain=p_gain,
            i_gain=i_gain,
            d_gain=d_gain,
            k_gain=k_gain,
            integ_limit=int_lim,
            lpf_alpha=float(getattr(self.cfg, "fm_rate_lpf_alpha", 0.803307)),
            lpf_beta=float(getattr(self.cfg, "fm_rate_lpf_beta", 0.196693)),
        )
        angacc_limit = float(self.cfg.max_angular_accel_rad_s2)
        if angacc_limit > 0.0:
            angacc_des = angacc_des.clamp(-angacc_limit, angacc_limit)

        collective_thrust_n = target_thrust_ref * effective_mass
        thrust_and_ang = torch.cat([collective_thrust_n.unsqueeze(-1), angacc_des], dim=-1)
        rotor_thrust_target = thrust_and_ang @ self._allocation_matrix_inv.T
        rotor_thrust_target = torch.minimum(
            rotor_thrust_target.clamp_min(0.0),
            self._max_rotor_thrust_n.unsqueeze(0),
        )

        motor_cmd = 2.0 * rotor_thrust_target / self._max_rotor_thrust_n.unsqueeze(0).clamp_min(1.0e-9) - 1.0
        throttle_target = torch.sqrt(((motor_cmd + 1.0) * 0.5).clamp(0.0, 1.0))
        tau_up = max(float(self._motor_time_constant_s), 1.0e-4)
        tau_down = max(float(self._motor_time_constant_s), 1.0e-4)
        tau = torch.where(throttle_target >= self._rotor_throttle, dt / tau_up, dt / tau_down)
        self._rotor_throttle = self._rotor_throttle + tau * (throttle_target - self._rotor_throttle)
        throttle_sq = self._rotor_throttle.square()
        if self._rotor_noise_scale > 0.0:
            throttle_sq = throttle_sq + self._rotor_noise_scale * torch.randn_like(throttle_sq)
        throttle_sq = throttle_sq.clamp(0.0, 1.0)
        self._rotor_thrust_cmd_n = throttle_sq * self._max_rotor_thrust_n.unsqueeze(0)

        body_force, body_torque = rotor_thrust_to_body_wrench(
            rotor_thrust_n=self._rotor_thrust_cmd_n,
            arm_lengths=self._arm_lengths_m,
            rotor_angles=self._rotor_angles_rad,
            directions=self._rotor_directions,
            force_constants=self._force_constants,
            moment_constants=self._moment_constants,
        )
        force_w = self._quat_rotate_body_to_world(drone_quat_w, body_force)
        torque_w = self._quat_rotate_body_to_world(drone_quat_w, body_torque)

        self._drone.instantaneous_wrench_composer.set_forces_and_torques(
            forces=force_w.unsqueeze(1),
            torques=torque_w.unsqueeze(1),
            is_global=True,
            body_ids=self._force_body_ids,
        )
        self._apply_external_forces()
        return None

    def _apply_external_forces(self):
        if not (ISAACLAB_RUNTIME_AVAILABLE and torch is not None):
            return None
        if self._shuttlecock is None:
            return None

        self._ensure_runtime_buffers()

        ball_vel_w = self._shuttlecock.data.root_lin_vel_w
        drag_forces = compute_drag_force_tensor(ball_vel_w, self._shuttle_mass_kg, self._drag_length_m)

        forces = drag_forces.unsqueeze(1)
        torques = torch.zeros_like(forces)
        self._shuttlecock.instantaneous_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            is_global=True,
        )
        return None

    def _get_observations(self):
        if ISAACLAB_RUNTIME_AVAILABLE and torch is not None:
            self._ensure_runtime_buffers()
            # Enforce a hard post-step speed cap so logged/observed states remain bounded.
            self._enforce_linear_speed_limit()
            drone_pos_w, drone_quat_w, drone_lin_vel_w, drone_ang_vel_w = self._get_drone_kinematics()
            ball_pos_w = self._shuttlecock.data.root_pos_w
            ball_lin_vel_w = self._shuttlecock.data.root_lin_vel_w
            
            # 任务二模式：post-hit 阶段使用 hit 时刻的无人机状态
            if self.enable_post_hit_tracking and self.post_hit is not None and self.post_hit.any():
                post_hit_mask = self.post_hit
                drone_pos_w = torch.where(post_hit_mask.unsqueeze(-1), self._hit_drone_pos_w, drone_pos_w)
                drone_quat_w = torch.where(post_hit_mask.unsqueeze(-1), self._hit_drone_quat_w, drone_quat_w)
                drone_lin_vel_w = torch.where(post_hit_mask.unsqueeze(-1), self._hit_drone_lin_vel_w, drone_lin_vel_w)
                drone_ang_vel_w = torch.where(post_hit_mask.unsqueeze(-1), self._hit_drone_ang_vel_w, drone_ang_vel_w)
            
            drone_pos_local = drone_pos_w
            ball_pos_local = ball_pos_w
            if hasattr(self.scene, "env_origins"):
                drone_pos_local = drone_pos_w - self.scene.env_origins
                ball_pos_local = ball_pos_w - self.scene.env_origins
            hit_point_w = self._launch_hit_point_local + self.scene.env_origins
            hit_point_rel_w = hit_point_w - drone_pos_w

            time_norm = (self.episode_length_buf.float() / float(self.max_episode_length)).clamp(0.0, 1.0)
            privileged = {
                "time_norm": time_norm,
                "drag_length": self._drag_length_m,
                "ball_mass": self._shuttle_mass_kg,
            }
            obs = build_actor_critic_observations(
                drone_pos=drone_pos_local,
                drone_quat_w=drone_quat_w,
                drone_lin_vel=drone_lin_vel_w,
                drone_ang_vel=drone_ang_vel_w,
                ball_pos=ball_pos_local,
                ball_lin_vel=ball_lin_vel_w,
                hit_time_s=self._launch_hit_time_s,
                hit_point_rel=hit_point_rel_w,
                privileged=privileged,
                add_actor_noise=True,
                noise_std=0.01,
            )
            if isinstance(obs, dict):
                for k, v in obs.items():
                    obs[k] = torch.nan_to_num(v, nan=0.0, posinf=1.0e3, neginf=-1.0e3).clamp(-1.0e3, 1.0e3)
            return obs
        return build_actor_critic_observations()

    def _get_rewards(self):
        if ISAACLAB_RUNTIME_AVAILABLE and torch is not None:
            drone_pos_w, drone_quat_w, drone_lin_vel_w, drone_ang_vel_w = self._get_drone_kinematics()
            racket_pos_w = self._get_racket_center_pos_w(drone_pos_w)
            ball_pos_w = self._shuttlecock.data.root_pos_w
            ball_lin_vel_w = self._shuttlecock.data.root_lin_vel_w
            racket_normal_w = self._get_racket_normal_w(drone_quat_w)
            contact, d_axis = self._compute_contact_signal(
                racket_pos_w,
                ball_pos_w,
                ball_lin_vel_w=ball_lin_vel_w,
                racket_normal_w=racket_normal_w,
            )
            yaw = self._quat_to_yaw(drone_quat_w)
            drone_up_w = self._quat_to_up_axis_w(drone_quat_w)
            bound_dist = self._compute_bound_distance(drone_pos_w)
            
            # 任务二模式：使用 compute_rewards_task2
            if self.enable_post_hit_tracking:
                from badminton_intercept.mdp.rewards import compute_rewards_task2, compute_racket_normal_x_component
                normal_x = compute_racket_normal_x_component(drone_quat_w)
                
                rewards = compute_rewards_task2(
                    racket_pos_w=racket_pos_w,
                    ball_pos_w=ball_pos_w,
                    ball_vel_w=ball_lin_vel_w,
                    contact=contact,
                    action=self._actions,
                    prev_action=self._prev_actions,
                    yaw=yaw,
                    d_axis=d_axis,
                    bound_dist=bound_dist,
                    drone_up_w=drone_up_w,
                    drone_ang_vel_w=drone_ang_vel_w,
                    drone_lin_vel_w=drone_lin_vel_w,
                    drone_pos_w=drone_pos_w,
                    drone_quat_w=drone_quat_w,
                    episode_length_buf=self.episode_length_buf,
                    max_episode_length=self.max_episode_length,
                    has_hit_ball=self.post_hit,
                    c_ang_vel=float(getattr(self.cfg, "reward_c_ang_vel", 0.05)),
                    c_vert_vel=float(getattr(self.cfg, "reward_c_vert_vel", 0.1)),
                )
                return rewards["total"]
            
            # 任务一模式：使用原有奖励函数
            rewards = compute_rewards(
                racket_pos_w=racket_pos_w,
                ball_pos_w=ball_pos_w,
                contact=contact,
                action=self._actions,
                prev_action=self._prev_actions,
                yaw=yaw,
                d_axis=d_axis,
                bound_dist=bound_dist,
                drone_up_w=drone_up_w,
                drone_ang_vel_w=drone_ang_vel_w,
                drone_lin_vel_w=drone_lin_vel_w,
                drone_pos_w=drone_pos_w,
                episode_length_buf=self.episode_length_buf,
                max_episode_length=self.max_episode_length,
                c_ang_vel=float(getattr(self.cfg, "reward_c_ang_vel", 0.05)),
                c_vert_vel=float(getattr(self.cfg, "reward_c_vert_vel", 0.1)),
            )
            return rewards["total"]
        return compute_rewards()

    def _get_dones(self):
        if ISAACLAB_RUNTIME_AVAILABLE and torch is not None:
            drone_pos_w, drone_quat_w, _, _ = self._get_drone_kinematics()
            racket_pos_w = self._get_racket_center_pos_w(drone_pos_w)
            ball_pos_w = self._shuttlecock.data.root_pos_w
            ball_lin_vel_w = self._shuttlecock.data.root_lin_vel_w
            racket_normal_w = self._get_racket_normal_w(drone_quat_w)
            contact, _ = self._compute_contact_signal(
                racket_pos_w,
                ball_pos_w,
                ball_lin_vel_w=ball_lin_vel_w,
                racket_normal_w=racket_normal_w,
            )
            net_contact = self._compute_net_contact_signal()
            self._last_contact = contact.clone()
            drone_up_w = self._quat_to_up_axis_w(drone_quat_w)
            drone_pos_local = drone_pos_w
            racket_pos_local = racket_pos_w
            ball_pos_local = ball_pos_w
            if hasattr(self.scene, "env_origins"):
                drone_pos_local = drone_pos_w - self.scene.env_origins
                racket_pos_local = racket_pos_w - self.scene.env_origins
                ball_pos_local = ball_pos_w - self.scene.env_origins
            # 使用无人机的最下面点来判断（中心高度 - 0.13m）
            drone_bottom_z = drone_pos_local[:, 2] - 0.13
            safe_bounds = {"x": (0.0, 6.7), "y": (-3.05, 3.05), "z": (0.2, 3.0)}  # 最下面不低于20cm
            
            # 任务二模式：post-hit 阶段使用新的 termination 逻辑
            if self.enable_post_hit_tracking and self.post_hit is not None and self.post_hit.any():
                from badminton_intercept.mdp.terminations import compute_dones_task2
                
                post_hit_envs = self.post_hit
                terminated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
                truncated = torch.zeros_like(terminated)
                reason_masks = {}
                
                # post-hit 环境：使用 compute_dones_task2
                if post_hit_envs.any():
                    task2_court_bounds = {"x": (-6.7, 6.7), "y": (-3.05, 3.05)}
                    terminated_task2, rewards_task2, reason_masks_task2 = compute_dones_task2(
                        ball_pos_w=ball_pos_local[post_hit_envs],
                        net_contact=net_contact[post_hit_envs],
                        z_threshold=0.1,
                        max_ball_height=7.0,
                        court_bounds=task2_court_bounds,
                        return_reason_masks=True,
                    )
                    terminated[post_hit_envs] = terminated_task2
                    
                    # 转换 reason_masks 到完整形状
                    for key, mask in reason_masks_task2.items():
                        full_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
                        full_mask[post_hit_envs] = mask
                        reason_masks[key] = full_mask
                
                # 非 post-hit 环境：使用原有 compute_dones 逻辑
                non_post_hit = ~post_hit_envs
                if non_post_hit.any():
                    terminated_non, truncated_non, reason_masks_non = compute_dones(
                        ball_pos_w=ball_pos_local[non_post_hit],
                        racket_pos_w=racket_pos_local[non_post_hit],
                        drone_pos_w=drone_pos_local[non_post_hit],
                        drone_bottom_z=drone_bottom_z[non_post_hit],
                        contact=contact[non_post_hit],
                        net_contact=net_contact[non_post_hit],
                        drone_up_w=drone_up_w[non_post_hit],
                        episode_length_buf=self.episode_length_buf[non_post_hit],
                        max_episode_length=self.max_episode_length,
                        safe_bounds=safe_bounds,
                        z_threshold=0.1,
                        return_reason_masks=True,
                    )
                    terminated[non_post_hit] = terminated_non
                    truncated[non_post_hit] = truncated_non
                    
                    # 合并 reason_masks
                    for key, mask in reason_masks_non.items():
                        if key not in reason_masks:
                            reason_masks[key] = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
                        reason_masks[key][non_post_hit] = mask
                
                self._last_done_reasons = reason_masks
            else:
                # 任务一模式或任务二 pre-hit 阶段：使用原有逻辑
                terminated, truncated, reason_masks = compute_dones(
                    ball_pos_w=ball_pos_local,
                    racket_pos_w=racket_pos_local,
                    drone_pos_w=drone_pos_local,
                    drone_bottom_z=drone_bottom_z,  # 传递最下面高度用于z边界判定
                    contact=contact,
                    net_contact=net_contact,
                    drone_up_w=drone_up_w,
                    episode_length_buf=self.episode_length_buf,
                    max_episode_length=self.max_episode_length,
                    safe_bounds=safe_bounds,
                    z_threshold=0.1,
                    return_reason_masks=True,
                )
                self._last_done_reasons = reason_masks
            
            if hasattr(self, "extras"):
                # 只有当有 env 真正结束时才上报 episode 统计
                num_resets = int((terminated | truncated).sum().item())
                if num_resets > 0:
                    counts = {k: int(v.sum().item()) for k, v in reason_masks.items()}
                    summary = dict(counts)
                    summary["num_resets"] = num_resets
                    # Provide normalized rates for training logs; raw counts are batch aggregates.
                    summary["success_rate"] = counts.get("success_contact", 0) / num_resets
                    summary["failure_ball_drop_rate"] = counts.get("failure_ball_drop", 0) / num_resets
                    summary["failure_out_of_bounds_rate"] = counts.get("failure_out_of_bounds", 0) / num_resets
                    summary["failure_tilt_rate"] = counts.get("failure_tilt", 0) / num_resets
                    summary["timeout_rate"] = counts.get("timeout", 0) / num_resets
                    if "failure_net_contact" in counts:
                        summary["failure_net_contact_rate"] = counts.get("failure_net_contact", 0) / num_resets
                    if "failure_server_side_grounded" in counts:
                        summary["failure_server_side_grounded_rate"] = (
                            counts.get("failure_server_side_grounded", 0) / num_resets
                        )
                    # 添加课程学习阶段信息
                    current_stage = self._curriculum.current_stage
                    summary["curriculum_stage_id"] = current_stage.stage_id
                    # RSL-RL logger 只读取 extras["episode"] 或 extras["log"]
                    self.extras["episode"] = summary
                else:
                    # 没有 env 结束时，清除 episode 统计避免被错误平均
                    if "episode" in self.extras:
                        del self.extras["episode"]
            return terminated, truncated
        return compute_dones()

    def _resolve_env_ids(self, env_ids):
        if env_ids is None:
            if self._drone is not None and hasattr(self._drone, "_ALL_INDICES"):
                return self._drone._ALL_INDICES
            if torch is not None:
                return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            return None
        return env_ids

    def _apply_domain_randomization(self, env_ids):
        if not (ISAACLAB_RUNTIME_AVAILABLE and torch is not None):
            return

        self._ensure_runtime_buffers()

        if not bool(getattr(self.cfg, "randomization_enabled", True)):
            return

        num_ids = int(len(env_ids))
        env_ids_cpu = env_ids.detach().to(device="cpu")

        # Sample episode-wise random physics parameters.
        racket_restitution = self._sample_uniform(getattr(self.cfg, "dr_racket_restitution_range", (0.75, 0.90)), num_ids, self.device)
        shuttle_mass = self._sample_uniform(getattr(self.cfg, "dr_ball_mass_range", (0.0045, 0.0055)), num_ids, self.device)
        drone_mass_scale = self._sample_uniform(getattr(self.cfg, "dr_quad_mass_scale_range", (0.95, 1.05)), num_ids, self.device)
        drone_inertia_scale = self._sample_uniform(getattr(self.cfg, "dr_quad_inertia_scale_range", (0.95, 1.05)), num_ids, self.device)
        drone_thrust_scale = self._sample_uniform(getattr(self.cfg, "dr_thrust_coeff_scale_range", (0.95, 1.05)), num_ids, self.device)

        self._racket_restitution[env_ids] = racket_restitution
        self._shuttle_mass_kg[env_ids] = shuttle_mass
        if self._launch_sampler_mode == "online":
            drag_length = self._sample_uniform(getattr(self.cfg, "dr_drag_length_range", (3.8, 4.4)), num_ids, self.device)
            self._drag_length_m[env_ids] = drag_length
        self._drone_mass_scale[env_ids] = drone_mass_scale
        self._drone_inertia_scale[env_ids] = drone_inertia_scale
        self._drone_thrust_scale[env_ids] = drone_thrust_scale

        if self._shuttlecock is not None:
            masses = self._shuttlecock.root_physx_view.get_masses()
            default_mass = self._shuttlecock.data.default_mass
            shuttle_mass_cpu = shuttle_mass.detach().cpu()

            if masses.ndim == 1:
                masses[env_ids_cpu] = shuttle_mass_cpu
                ratio = shuttle_mass_cpu / default_mass[env_ids_cpu].clamp_min(1.0e-6)
            else:
                masses[env_ids_cpu] = shuttle_mass_cpu.unsqueeze(-1)
                ratio = shuttle_mass_cpu.unsqueeze(-1) / default_mass[env_ids_cpu].clamp_min(1.0e-6)

            self._shuttlecock.root_physx_view.set_masses(masses, env_ids_cpu)

            inertias = self._shuttlecock.root_physx_view.get_inertias()
            default_inertia = self._shuttlecock.data.default_inertia
            if inertias.ndim == 2:
                inertias[env_ids_cpu] = default_inertia[env_ids_cpu] * ratio
            elif inertias.ndim == 3:
                inertias[env_ids_cpu] = default_inertia[env_ids_cpu] * ratio.unsqueeze(-1)
            self._shuttlecock.root_physx_view.set_inertias(inertias, env_ids_cpu)

        if self._drone is not None and hasattr(self._drone, "root_physx_view"):
            masses = self._drone.root_physx_view.get_masses()
            default_mass = self._drone.data.default_mass
            mass_scale_cpu = drone_mass_scale.detach().cpu()

            if masses.ndim == 1:
                masses[env_ids_cpu] = default_mass[env_ids_cpu] * mass_scale_cpu
            else:
                masses[env_ids_cpu] = default_mass[env_ids_cpu] * mass_scale_cpu.unsqueeze(-1)
            self._drone.root_physx_view.set_masses(masses, env_ids_cpu)

            inertias = self._drone.root_physx_view.get_inertias()
            default_inertia = self._drone.data.default_inertia
            inertia_scale_cpu = drone_inertia_scale.detach().cpu()

            if inertias.ndim == 2:
                inertias[env_ids_cpu] = default_inertia[env_ids_cpu] * inertia_scale_cpu.unsqueeze(-1)
            elif inertias.ndim == 3:
                inertias[env_ids_cpu] = default_inertia[env_ids_cpu] * inertia_scale_cpu.unsqueeze(-1).unsqueeze(-1)
            self._drone.root_physx_view.set_inertias(inertias, env_ids_cpu)

    def _reset_drone_state(self, env_ids):
        if self._drone is None or not hasattr(self._drone, "data"):
            return

        default_root_state = self._drone.data.default_root_state[env_ids].clone()
        origins = self.scene.env_origins[env_ids]
        num_ids = int(len(env_ids))

        # 根据课程阶段获取无人机初始位置范围
        current_stage = self._curriculum.current_stage
        x_range = current_stage.uav_init_x_range
        y_range = current_stage.uav_init_y_range
        z_range = current_stage.uav_init_z_range
        x = self._sample_uniform(x_range, num_ids, self.device)
        y = self._sample_uniform(y_range, num_ids, self.device)
        z = self._sample_uniform(z_range, num_ids, self.device)
        vel_noise = self._sample_uniform(getattr(self.cfg, "uav_init_lin_vel_range", (-0.1, 0.1)), num_ids * 3, self.device).view(num_ids, 3)

        default_root_state[:, 0] = origins[:, 0] + x
        default_root_state[:, 1] = origins[:, 1] + y
        default_root_state[:, 2] = origins[:, 2] + z
        default_root_state[:, 3] = 1.0
        default_root_state[:, 4:7] = 0.0
        default_root_state[:, 7:10] = vel_noise
        default_root_state[:, 10:13] = 0.0

        self._drone.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._drone.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        if hasattr(self._drone.data, "default_joint_pos") and hasattr(self._drone, "write_joint_state_to_sim"):
            joint_pos = self._drone.data.default_joint_pos[env_ids]
            joint_vel = self._drone.data.default_joint_vel[env_ids]
            self._drone.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_shuttlecock_state(self, env_ids):
        if self._shuttlecock is None or not hasattr(self._shuttlecock, "data"):
            return {}

        self._ensure_runtime_buffers()

        origins = self.scene.env_origins[env_ids]
        num_ids = int(len(env_ids))
        current_stage = self._curriculum.current_stage
        launch_spec = build_launch_sampling_spec(self.cfg, current_stage)

        if self._launch_sampler_mode == "trajectory_library":
            self._ensure_launch_libraries_loaded()
            stage_id = int(current_stage.stage_id)
            library = None if self._launch_libraries is None else self._launch_libraries.get(stage_id)
            if library is None:
                raise RuntimeError(f"Trajectory library for stage {stage_id} is not loaded.")

            num_entries = int(library["pos_local"].shape[0])
            sample_idx = torch.randint(num_entries, (num_ids,), device="cpu")
            pos_local = library["pos_local"].index_select(0, sample_idx).to(device=self.device)
            vel_local = library["vel_local"].index_select(0, sample_idx).to(device=self.device)
            drag_length_m = library["drag_length_m"].index_select(0, sample_idx).to(device=self.device)
            hit_plane_z = library["hit_plane_z"].index_select(0, sample_idx).to(device=self.device)
            hit_time_s = library["hit_time_s"].index_select(0, sample_idx).to(device=self.device)
            hit_point_local = library["hit_point_local"].index_select(0, sample_idx).to(device=self.device)
            self._drag_length_m[env_ids] = drag_length_m
            stats = {
                "launch_source": "library",
                "launch_valid_rate": 1.0,
                "launch_resample_rounds_mean": 0.0,
                "launch_fallback_count": 0,
                "launch_no_cross_count": 0,
                "launch_oob_count": 0,
                "launch_no_hit_plane_count": 0,
            }
        else:
            max_rounds = int(getattr(self.cfg, "launch_max_resample_rounds", 2))
            vx_lo, vx_hi = launch_spec.v_forward_range
            vy_lo, vy_hi = launch_spec.v_side_range
            vz_lo, vz_hi = launch_spec.v_up_range
            vx_cross_lo, vx_cross_hi = launch_spec.vx_cross_range
            vx_mid = 0.5 * (vx_lo + vx_hi)
            vy_mid = 0.5 * (vy_lo + vy_hi)
            vz_mid = 0.5 * (vz_lo + vz_hi)

            sampled = sample_launch_candidates(num_ids, launch_spec, self.device, include_drag_length=False)
            pos_local = sampled["pos_local"]
            vel_local = sampled["vel_local"]
            hit_plane_z = sampled["hit_plane_z"]
            drag_length_m = self._drag_length_m[env_ids]

            eval_data = self._evaluate_launch_candidates(
                pos_local=pos_local,
                vel_local=vel_local,
                drag_length_m=drag_length_m,
                launch_spec=launch_spec,
                hit_plane_z=hit_plane_z,
            )
            valid = eval_data["valid"].clone()
            resample_counts = torch.zeros((num_ids,), dtype=torch.int32, device=self.device)

            for _round in range(max_rounds):
                invalid_mask = ~valid
                if not bool(torch.any(invalid_mask)):
                    break

                invalid_idx = torch.nonzero(invalid_mask, as_tuple=False).squeeze(-1)
                k = int(invalid_idx.numel())
                resample_counts[invalid_idx] += 1

                vx_new = self._sample_uniform(launch_spec.v_forward_range, k, self.device)
                vy_new = self._sample_uniform(launch_spec.v_side_range, k, self.device)
                vz_new = self._sample_uniform(launch_spec.v_up_range, k, self.device)

                local_net_fail = (~eval_data["net_ok"][invalid_idx]) | eval_data["no_cross"][invalid_idx]
                local_x_short = eval_data["landing_x_short"][invalid_idx]
                local_x_long = eval_data["landing_x_long"][invalid_idx]
                local_y_out = eval_data["landing_y_out"][invalid_idx]
                local_no_landing = eval_data["no_landing"][invalid_idx]

                if bool(torch.any(local_net_fail)):
                    c = int(local_net_fail.sum().item())
                    vx_new[local_net_fail] = self._sample_uniform((vx_cross_lo, vx_cross_hi), c, self.device)
                    vz_new[local_net_fail] = self._sample_uniform((max(vz_mid, vz_lo), vz_hi), c, self.device)
                    vy_new[local_net_fail] = self._sample_uniform((0.45 * vy_lo, 0.45 * vy_hi), c, self.device)

                if bool(torch.any(local_x_short)):
                    c = int(local_x_short.sum().item())
                    vx_new[local_x_short] = self._sample_uniform((vx_cross_lo, vx_cross_hi), c, self.device)
                if bool(torch.any(local_x_long)):
                    c = int(local_x_long.sum().item())
                    vx_new[local_x_long] = self._sample_uniform((vx_lo, min(vx_mid, vx_hi)), c, self.device)
                if bool(torch.any(local_y_out)):
                    c = int(local_y_out.sum().item())
                    vy_new[local_y_out] = self._sample_uniform((0.35 * vy_lo, 0.35 * vy_hi), c, self.device)
                if bool(torch.any(local_no_landing)):
                    c = int(local_no_landing.sum().item())
                    vz_new[local_no_landing] = self._sample_uniform((max(vz_mid, vz_lo), vz_hi), c, self.device)
                    vx_new[local_no_landing] = self._sample_uniform((vx_cross_lo, vx_cross_hi), c, self.device)

                vel_local[invalid_idx, 0] = vx_new
                vel_local[invalid_idx, 1] = vy_new
                vel_local[invalid_idx, 2] = vz_new

                eval_sub = self._evaluate_launch_candidates(
                    pos_local=pos_local[invalid_idx],
                    vel_local=vel_local[invalid_idx],
                    drag_length_m=drag_length_m[invalid_idx],
                    launch_spec=launch_spec,
                    hit_plane_z=hit_plane_z[invalid_idx],
                )
                for key, value in eval_sub.items():
                    eval_data[key][invalid_idx] = value
                valid = eval_data["valid"]

            invalid_final = ~valid
            fallback_env_count = int(invalid_final.sum().item())
            if fallback_env_count > 0:
                fail_idx = torch.nonzero(invalid_final, as_tuple=False).squeeze(-1)
                m = int(fail_idx.numel())
                unresolved = torch.ones((m,), dtype=torch.bool, device=self.device)
                best_score = torch.full((m,), float("inf"), device=self.device)
                best_vel = vel_local[fail_idx].clone()

                candidate_bank = torch.tensor(
                    [
                        [vx_cross_lo, 0.0, vz_mid],
                        [0.95 * vx_cross_hi, 0.0, vz_mid],
                        [vx_cross_lo, 0.0, 0.95 * vz_hi],
                        [0.95 * vx_cross_hi, 0.0, 0.9 * vz_hi],
                        [0.9 * vx_cross_hi, 0.20 * vy_hi, 0.9 * vz_hi],
                        [0.9 * vx_cross_hi, 0.20 * vy_lo, 0.9 * vz_hi],
                        [vx_cross_lo, 0.12 * vy_hi, max(vz_mid, vz_lo)],
                        [vx_cross_lo, 0.12 * vy_lo, max(vz_mid, vz_lo)],
                        [vx_mid, 0.0, vz_mid],
                        [0.75 * vx_cross_hi, 0.0, max(vz_mid, vz_lo)],
                    ],
                    device=self.device,
                    dtype=vel_local.dtype,
                )
                candidate_bank[:, 0] = candidate_bank[:, 0].clamp(vx_lo, vx_hi)
                candidate_bank[:, 1] = candidate_bank[:, 1].clamp(vy_lo, vy_hi)
                candidate_bank[:, 2] = candidate_bank[:, 2].clamp(vz_lo, vz_hi)

                for candidate in candidate_bank:
                    unresolved_idx_local = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
                    if unresolved_idx_local.numel() == 0:
                        break
                    env_sel = fail_idx[unresolved_idx_local]
                    cand_vel = candidate.unsqueeze(0).repeat(int(unresolved_idx_local.numel()), 1)

                    eval_cand = self._evaluate_launch_candidates(
                        pos_local=pos_local[env_sel],
                        vel_local=cand_vel,
                        drag_length_m=drag_length_m[env_sel],
                        launch_spec=launch_spec,
                        hit_plane_z=hit_plane_z[env_sel],
                    )
                    score = self._launch_violation_score(
                        eval_data=eval_cand,
                        launch_spec=launch_spec,
                    )
                    better = score < best_score[unresolved_idx_local]
                    if bool(torch.any(better)):
                        best_score[unresolved_idx_local[better]] = score[better]
                        best_vel[unresolved_idx_local[better]] = cand_vel[better]

                    found_valid = eval_cand["valid"]
                    if bool(torch.any(found_valid)):
                        resolved_local = unresolved_idx_local[found_valid]
                        vel_local[fail_idx[resolved_local]] = cand_vel[found_valid]
                        unresolved[resolved_local] = False

                if bool(torch.any(unresolved)):
                    unresolved_idx_local = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
                    vel_local[fail_idx[unresolved_idx_local]] = best_vel[unresolved_idx_local]

                eval_fallback = self._evaluate_launch_candidates(
                    pos_local=pos_local[fail_idx],
                    vel_local=vel_local[fail_idx],
                    drag_length_m=drag_length_m[fail_idx],
                    launch_spec=launch_spec,
                    hit_plane_z=hit_plane_z[fail_idx],
                )
                for key, value in eval_fallback.items():
                    eval_data[key][fail_idx] = value

            final_valid = eval_data["valid"]
            stats = {
                "launch_source": "online",
                "launch_valid_rate": float(final_valid.float().mean().item()),
                "launch_resample_rounds_mean": float(resample_counts.float().mean().item()),
                "launch_fallback_count": int(fallback_env_count),
                "launch_no_cross_count": int(eval_data["no_cross"].sum().item()),
                "launch_oob_count": int((~eval_data["landing_ok"]).sum().item()),
                "launch_no_hit_plane_count": int(eval_data["no_hit_plane"].sum().item()),
            }
            hit_time_s = eval_data["hit_time_s"]
            hit_point_local = eval_data["hit_point_local"]

        invalid_hit = (~torch.isfinite(hit_time_s)) | (~torch.isfinite(hit_point_local).all(dim=-1))
        if bool(torch.any(invalid_hit)):
            hit_time_s = hit_time_s.clone()
            hit_point_local = hit_point_local.clone()
            hit_time_s[invalid_hit] = 0.0
            hit_point_local[invalid_hit] = pos_local[invalid_hit]

        self._launch_hit_plane_z[env_ids] = hit_plane_z
        self._launch_hit_time_s[env_ids] = hit_time_s
        self._launch_hit_point_local[env_ids] = hit_point_local

        pos_start_world = pos_local + origins

        shuttle_state = self._shuttlecock.data.default_root_state[env_ids].clone()
        shuttle_state[:, 0] = pos_start_world[:, 0]
        shuttle_state[:, 1] = pos_start_world[:, 1]
        shuttle_state[:, 2] = pos_start_world[:, 2]
        shuttle_state[:, 3] = 1.0
        shuttle_state[:, 4:7] = 0.0

        shuttle_state[:, 7] = vel_local[:, 0]
        shuttle_state[:, 8] = vel_local[:, 1]
        shuttle_state[:, 9] = vel_local[:, 2]
        shuttle_state[:, 10:13] = 0.0

        self._shuttlecock.write_root_pose_to_sim(shuttle_state[:, :7], env_ids=env_ids)
        self._shuttlecock.write_root_velocity_to_sim(shuttle_state[:, 7:], env_ids=env_ids)

        self._last_launch_stats = stats
        return stats

    def _update_curriculum_from_resets(self, env_ids):
        if self._last_contact is None:
            return
        if self._curriculum_fixed_stage:
            return
        success_flags = self._last_contact[env_ids].detach().cpu().tolist()
        self._curriculum.record_episode_outcomes([bool(s) for s in success_flags])

    def _reset_idx(self, env_ids):
        if ISAACLAB_RUNTIME_AVAILABLE:
            env_ids = self._resolve_env_ids(env_ids)
            if env_ids is None:
                return
            if bool(getattr(self.cfg, "termination_debug_print", False)) and self._last_done_reasons is not None:
                counts = {}
                for reason, mask in self._last_done_reasons.items():
                    counts[reason] = int(mask[env_ids].sum().item())
                counts_str = ", ".join([f"{k}={v}" for k, v in counts.items() if v > 0]) or "none"
                print(f"[done] reset_envs={int(len(env_ids))}, reasons: {counts_str}")
            self._update_curriculum_from_resets(env_ids)
            super()._reset_idx(env_ids)

            if self._drone is not None and hasattr(self._drone, "reset"):
                self._drone.reset(env_ids)
            if self._shuttlecock is not None and hasattr(self._shuttlecock, "reset"):
                self._shuttlecock.reset(env_ids)
            if self._racket_contact_sensor is not None and hasattr(self._racket_contact_sensor, "reset"):
                self._racket_contact_sensor.reset(env_ids)
            if self._net_contact_sensor is not None and hasattr(self._net_contact_sensor, "reset"):
                self._net_contact_sensor.reset(env_ids)

            self._apply_domain_randomization(env_ids)
            self._reset_drone_state(env_ids)
            launch_stats = self._reset_shuttlecock_state(env_ids)
            if hasattr(self, "extras") and isinstance(self.extras, dict):
                self.extras["launch"] = launch_stats
            if self._actions is not None:
                self._actions[env_ids] = 0.0
            if self._prev_actions is not None:
                self._prev_actions[env_ids] = 0.0
            if self._rotor_thrust_cmd_n is not None:
                self._rotor_thrust_cmd_n[env_ids] = 0.0
            if self._filtered_body_rate_rad_s is not None:
                self._filtered_body_rate_rad_s[env_ids] = 0.0
            if self._rate_integral is not None:
                self._rate_integral[env_ids] = 0.0
            if self._rotor_throttle is not None:
                # Seed motors near hover throttle on reset to avoid immediate drop/tilt transients.
                num_rotors = int(self._max_rotor_thrust_n.shape[0]) if self._max_rotor_thrust_n is not None else 4
                if self._max_rotor_thrust_n is not None and self._mass_kg is not None and self._drone_mass_scale is not None:
                    eff_mass = self._mass_kg[env_ids] * self._drone_mass_scale[env_ids] + float(
                        getattr(self.cfg, "fm_extra_mass_kg", 0.147)
                    )
                    hover_total_thrust = 9.81 * eff_mass
                    hover_per_rotor = (hover_total_thrust / float(max(num_rotors, 1))).unsqueeze(-1)
                    max_rotor = self._max_rotor_thrust_n.unsqueeze(0).expand(int(len(env_ids)), -1).clamp_min(1.0e-6)
                    hover_throttle = torch.sqrt((hover_per_rotor / max_rotor).clamp(0.0, 1.0))
                    self._rotor_throttle[env_ids] = hover_throttle
                    if self._rotor_thrust_cmd_n is not None:
                        self._rotor_thrust_cmd_n[env_ids] = hover_throttle.square() * self._max_rotor_thrust_n.unsqueeze(0)
                else:
                    self._rotor_throttle[env_ids] = 0.0
            # 重置扫掠接触检测的参考状态
            if self._contact_reference_valid is not None:
                self._contact_reference_valid[env_ids] = False
            if self.post_hit is not None:
                self.post_hit[env_ids] = False
            # 重置 hit 时刻保存的无人机状态
            if self._hit_drone_pos_w is not None:
                self._hit_drone_pos_w[env_ids] = 0.0
            if self._hit_drone_quat_w is not None:
                self._hit_drone_quat_w[env_ids] = 0.0
                self._hit_drone_quat_w[env_ids, 0] = 1.0
            if self._hit_drone_lin_vel_w is not None:
                self._hit_drone_lin_vel_w[env_ids] = 0.0
            if self._hit_drone_ang_vel_w is not None:
                self._hit_drone_ang_vel_w[env_ids] = 0.0
            return

        return reset_subenvs(env_ids)

    def reset(self, *args, **kwargs):
        if ISAACLAB_RUNTIME_AVAILABLE:
            return super().reset(*args, **kwargs)
        return self._get_observations()

    def step(self, actions):
        if ISAACLAB_RUNTIME_AVAILABLE:
            return super().step(actions)

        self._pre_physics_step(actions)
        self._apply_action()
        self._apply_external_forces()
        obs = self._get_observations()
        rew = self._get_rewards()
        done = self._get_dones()
        info = {"status": "skeleton_step"}
        return StepOutput(observations=obs, rewards=rew, dones=done, infos=info)
