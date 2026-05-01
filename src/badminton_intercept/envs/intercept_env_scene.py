from __future__ import annotations

from pathlib import Path
from typing import Any

from badminton_intercept.envs.intercept_env_common import (
    Articulation,
    ContactSensor,
    GroundPlaneCfg,
    ISAACLAB_RUNTIME_AVAILABLE,
    RigidObject,
    activate_contact_sensors,
    sim_utils,
    spawn_ground_plane,
    torch,
)
from badminton_intercept.envs.launch_sampler import (
    build_launch_library_metadata,
    build_launch_sampling_spec,
    compare_launch_library_metadata,
    evaluate_launch_candidates as evaluate_launch_candidates_pure,
    launch_violation_score as launch_violation_score_pure,
    resolve_launch_library_stage_paths,
    validate_launch_library_payload,
)
from badminton_intercept.envs.scene_builder import SceneEntityConfigs, build_scene


class InterceptEnvSceneMixin:
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
            print(f"[warn] ground plane spawn skipped: {exc}")

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu" and ground_spawned:
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self._activate_contact_reporters()

        for sensor_name in (
            "contact_sensor",
            "net_contact_sensor",
            "drone_net_contact_sensor",
            "net_drone_contact_sensor",
        ):
            sensor_cfg = getattr(self.cfg, sensor_name, None)
            if sensor_cfg is not None and sensor_name not in self.scene.sensors:
                self.scene.sensors[sensor_name] = ContactSensor(sensor_cfg)

        self._racket_contact_sensor = self.scene.sensors.get("contact_sensor", None)
        self._net_contact_sensor = self.scene.sensors.get("net_contact_sensor", None)
        self._drone_net_contact_sensor = self.scene.sensors.get("drone_net_contact_sensor", None)
        self._racket_net_contact_sensor = self.scene.sensors.get("racket_net_contact_sensor", None)
        self._net_drone_contact_sensor = self.scene.sensors.get("net_drone_contact_sensor", None)
        self._racket_body_ids = None
        if bool(getattr(self.cfg, "drone_net_contact_debug_print", False)):
            print(f"[DEBUG] scene sensor keys={list(self.scene.sensors.keys())}")
            print(
                "[DEBUG] contact sensors: "
                f"racket_ball={self._racket_contact_sensor is not None}, "
                f"shuttle_net={self._net_contact_sensor is not None}, "
                f"drone={self._drone_net_contact_sensor is not None}, "
                f"racket={self._racket_net_contact_sensor is not None}, "
                f"net={self._net_drone_contact_sensor is not None}"
            )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _activate_contact_reporters(self):
        """Ensure PhysX contact reporter API exists before ContactSensor initialization."""
        if not ISAACLAB_RUNTIME_AVAILABLE:
            return

        prim_path_exprs = (
            "/World/envs/env_.*/Drone.*",
            "/World/envs/env_.*/ShuttlecockProxy",
            "/World/envs/env_.*/Net",
        )
        activated_count = 0
        for prim_path_expr in prim_path_exprs:
            try:
                prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
            except Exception as exc:
                print(f"[warn] contact reporter path lookup failed for '{prim_path_expr}': {exc}")
                continue

            if len(prim_paths) == 0:
                print(f"[warn] contact reporter path lookup found no prims for '{prim_path_expr}'.")
                continue

            for prim_path in prim_paths:
                try:
                    activate_contact_sensors(prim_path, threshold=0.0)
                    activated_count += 1
                except Exception as exc:
                    print(f"[warn] contact reporter activation skipped for '{prim_path}': {exc}")

        if bool(getattr(self.cfg, "drone_net_contact_debug_print", False)):
            print(f"[DEBUG] contact reporter activation attempted on {activated_count} prim roots")

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
        return max(shuttle_radius, racket_proxy_radius + shuttle_radius)

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
