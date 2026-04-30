from __future__ import annotations

from badminton_intercept.control.x152b_params import DEFAULT_X152B_PARAMS
from badminton_intercept.envs.intercept_env_common import ISAACLAB_RUNTIME_AVAILABLE, torch
from badminton_intercept.physics.air_params import load_air_yaml_params
from badminton_intercept.physics.shuttle_aero import compute_drag_force_tensor


class InterceptEnvRuntimeMixin:
    def _sample_uniform(self, value_range: tuple[float, float], count: int, device=None):
        lo, hi = value_range
        return lo + (hi - lo) * torch.rand(count, device=device)

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
        self._last_sensor_hit = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._last_geometric_hit = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._last_weak_hit_failure = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._pending_weak_hit = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._pending_weak_hit_age = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._weak_hit_termination_rewards = torch.zeros((self.num_envs,), device=self.device)

        if str(getattr(self.cfg, "drone_control_mode", "")).lower() == "x152b_airgym":
            xp = DEFAULT_X152B_PARAMS
            self._air_params = None
            self._mass_kg = torch.full((self.num_envs,), float(xp.mass_kg), device=self.device)
            self._inertia_diag = torch.tensor(xp.inertia_diag, device=self.device).unsqueeze(0).repeat(
                self.num_envs, 1
            )
            self._arm_lengths_m = None
            self._rotor_angles_rad = None
            self._rotor_directions = torch.tensor(xp.rotor_directions, device=self.device)
            self._force_constants = None
            self._moment_constants = None
            self._motor_time_constant_s = 0.0
            self._rotor_noise_scale = 0.0
            self._max_rotor_thrust_n = torch.full((4,), float(xp.airgym_thrust_scale_n), device=self.device)
            num_rotors = 4
            self._allocation_matrix_inv = None
            self._rotor_thrust_cmd_n = torch.zeros((self.num_envs, num_rotors), device=self.device)
            self._filtered_body_rate_rad_s = torch.zeros((self.num_envs, 3), device=self.device)
            self._rate_integral = torch.zeros((self.num_envs, 3), device=self.device)
            self._rotor_throttle = torch.zeros((self.num_envs, num_rotors), device=self.device)
            self._prev_ball_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
            self._prev_ball_lin_vel_w = torch.zeros((self.num_envs, 3), device=self.device)
            self._prev_racket_contact_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
            self._prev_racket_normal_w = torch.zeros((self.num_envs, 3), device=self.device)
            self._contact_reference_valid = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.post_hit = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self._just_entered_post_hit = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self._hit_drone_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
            self._hit_drone_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
            self._hit_drone_quat_w[:, 0] = 1.0
            self._hit_drone_lin_vel_w = torch.zeros((self.num_envs, 3), device=self.device)
            self._hit_drone_ang_vel_w = torch.zeros((self.num_envs, 3), device=self.device)
            self._hit_ball_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
            self._hit_ball_vel_w = torch.zeros((self.num_envs, 3), device=self.device)
            self._hit_time_elapsed = torch.zeros((self.num_envs,), device=self.device)
            self._task2_server_ground_hold_active = torch.zeros(
                (self.num_envs,), dtype=torch.bool, device=self.device
            )
            self._task2_server_ground_landing_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
            self._task2_server_ground_hold_elapsed_s = torch.zeros((self.num_envs,), device=self.device)
            self._resolve_force_body_ids()
            self._resolve_racket_body_ids()
            return

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
        self._just_entered_post_hit = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._hit_drone_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._hit_drone_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self._hit_drone_quat_w[:, 0] = 1.0
        self._hit_drone_lin_vel_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._hit_drone_ang_vel_w = torch.zeros((self.num_envs, 3), device=self.device)
        # Analytical trajectory: ball state at hit moment
        self._hit_ball_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._hit_ball_vel_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._hit_time_elapsed = torch.zeros((self.num_envs,), device=self.device)
        self._task2_server_ground_hold_active = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._task2_server_ground_landing_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        self._task2_server_ground_hold_elapsed_s = torch.zeros((self.num_envs,), device=self.device)
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
        sensor_max_force = torch.zeros_like(geometric_contact, dtype=torch.float32)

        has_racket_sensor = self._racket_contact_sensor is not None
        if not has_racket_sensor:
            candidate_contact = geometric_contact
        else:
            try:
                threshold = float(getattr(self.cfg, "contact_force_threshold", 0.1))
                force_matrix_hist = getattr(self._racket_contact_sensor.data, "force_matrix_w_history", None)
                if force_matrix_hist is not None:
                    sensor_force_mag = torch.norm(force_matrix_hist, dim=-1)  # (N, H, B, F)
                    sensor_max_force = sensor_force_mag.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0]
                    sensor_contact = sensor_force_mag.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0] > threshold
                else:
                    net_forces_hist = getattr(self._racket_contact_sensor.data, "net_forces_w_history", None)
                    if net_forces_hist is not None:
                        force_mag = torch.norm(net_forces_hist, dim=-1)  # (N, H, sensor bodies)
                        sensor_max_force = force_mag.max(dim=1)[0].max(dim=1)[0]
                        sensor_contact = force_mag.max(dim=1)[0].max(dim=1)[0] > threshold
            except Exception:
                sensor_contact = torch.zeros_like(geometric_contact)
            # Always use sensor OR geometric contact; weak_hit logic is disabled
            candidate_contact = sensor_contact | geometric_contact

        velocity_confirmed = self._compute_hit_velocity_confirmation(
            ball_lin_vel_w=ball_lin_vel_w,
            racket_normal_w=racket_normal_w,
            sensor_contact=sensor_contact,
        )
        contact = candidate_contact & velocity_confirmed

        prev_weak_hit_failure = (
            self._last_weak_hit_failure.clone()
            if self._last_weak_hit_failure is not None
            else torch.zeros_like(contact)
        )
        weak_hit_failure = torch.zeros_like(contact)
        weak_hit_failure_start_steps = torch.zeros_like(contact, dtype=torch.long)
        if self.enable_post_hit_tracking and self.post_hit is not None:
            from badminton_intercept.mdp.rewards import compute_racket_normal_x_component

            _, drone_quat_w_for_hit, _, _ = self._get_drone_kinematics()
            normal_x_for_hit = compute_racket_normal_x_component(drone_quat_w_for_hit)
            correct_posture_for_hit = normal_x_for_hit < 0.0
            pre_hit_mask = ~self.post_hit
            weak_hit_candidate = (
                geometric_contact
                & (~sensor_contact)
                & velocity_confirmed
                & correct_posture_for_hit
                & pre_hit_mask
                & has_racket_sensor
            )
            sensor_hit = sensor_contact & correct_posture_for_hit & pre_hit_mask
            if self._pending_weak_hit is not None and self._pending_weak_hit_age is not None:
                pending_sensor_hit = self._pending_weak_hit & sensor_hit
                contact = contact | pending_sensor_hit
                cancel_pending = self._pending_weak_hit & (sensor_hit | self.post_hit)
                self._pending_weak_hit[cancel_pending] = False
                self._pending_weak_hit_age[cancel_pending] = 0

                new_pending = weak_hit_candidate & (~self._pending_weak_hit)
                self._pending_weak_hit[new_pending] = True
                self._pending_weak_hit_age[new_pending] = int(self.common_step_counter)
                if new_pending.any() and bool(getattr(self.cfg, "drone_net_contact_debug_print", False)):
                    env_id = int(torch.nonzero(new_pending, as_tuple=False)[0].item())
                    print(
                        "[DEBUG] weak_hit_pending_enter: "
                        f"env={env_id}, step={int(self.common_step_counter)}, "
                        f"sensor={bool(sensor_contact[env_id].item())}, "
                        f"geom={bool(geometric_contact[env_id].item())}, "
                        f"instant_dist={float(instant_dist[env_id].item()):.4f}, "
                        f"swept_dist={float(swept_dist[env_id].item()):.4f}, "
                        f"racket_ball_force={float(sensor_max_force[env_id].item()):.4f}"
                    )

                active_pending = self._pending_weak_hit & pre_hit_mask & (~sensor_hit)
                grace_steps = int(getattr(self.cfg, "weak_hit_sensor_grace_steps", 4))
                pending_elapsed_steps = int(self.common_step_counter) - self._pending_weak_hit_age
                weak_hit_failure = active_pending & (pending_elapsed_steps >= max(grace_steps, 0))
                weak_hit_failure_start_steps = torch.where(
                    weak_hit_failure,
                    self._pending_weak_hit_age,
                    weak_hit_failure_start_steps,
                )
                self._pending_weak_hit[weak_hit_failure] = False
                self._pending_weak_hit_age[weak_hit_failure] = 0
            else:
                weak_hit_failure = weak_hit_candidate
                weak_hit_failure_start_steps = torch.where(
                    weak_hit_failure,
                    torch.full_like(weak_hit_failure_start_steps, int(self.common_step_counter)),
                    weak_hit_failure_start_steps,
                )

        newly_weak_hit_failure = weak_hit_failure & (~prev_weak_hit_failure)
        weak_hit_failure = weak_hit_failure | prev_weak_hit_failure

        self._last_sensor_hit = sensor_contact & velocity_confirmed
        self._last_geometric_hit = geometric_contact & velocity_confirmed
        self._last_weak_hit_failure = weak_hit_failure

        if newly_weak_hit_failure.any() and bool(getattr(self.cfg, "drone_net_contact_debug_print", False)):
            env_id = int(torch.nonzero(newly_weak_hit_failure, as_tuple=False)[0].item())
            delta_vn = torch.zeros_like(contact, dtype=torch.float32)
            if (
                ball_lin_vel_w is not None
                and racket_normal_w is not None
                and self._prev_ball_lin_vel_w is not None
                and self._prev_racket_normal_w is not None
            ):
                normal_ref = self._prev_racket_normal_w
                normal_ref = normal_ref / torch.linalg.norm(normal_ref, dim=-1, keepdim=True).clamp_min(1.0e-6)
                vn_prev = torch.sum(self._prev_ball_lin_vel_w * normal_ref, dim=-1)
                vn_post = torch.sum(ball_lin_vel_w * normal_ref, dim=-1)
                delta_vn = torch.abs(vn_post - vn_prev)
            print(
                "[DEBUG] weak_hit_failure: "
                f"env={env_id}, pending_start_step={int(weak_hit_failure_start_steps[env_id].item())}, "
                f"step={int(self.common_step_counter)}, "
                f"current_sensor={bool(sensor_contact[env_id].item())}, "
                f"current_geom={bool(geometric_contact[env_id].item())}, "
                f"instant_dist={float(instant_dist[env_id].item()):.4f}, "
                f"swept_dist={float(swept_dist[env_id].item()):.4f}, "
                f"racket_ball_force={float(sensor_max_force[env_id].item()):.4f}, "
                f"delta_vn={float(delta_vn[env_id].item()):.4f}"
            )

        # 任务二模式：更新 post-hit 标志并保存 hit 时刻的状态
        # 只有当击球姿态正确（球拍朝向对方场地，normal_x < 0）时才追踪球
        if self.enable_post_hit_tracking and self.post_hit is not None and contact.any():
            from badminton_intercept.mdp.rewards import compute_racket_normal_x_component
            drone_pos_w, drone_quat_w, drone_lin_vel_w, drone_ang_vel_w = self._get_drone_kinematics()
            normal_x = compute_racket_normal_x_component(drone_quat_w)
            correct_posture = normal_x < 0.0

            newly_hit = contact & (~self.post_hit) & correct_posture
            if newly_hit.any():
                if bool(getattr(self.cfg, "drone_net_contact_debug_print", False)):
                    env_id = int(torch.nonzero(newly_hit, as_tuple=False)[0].item())
                    delta_vn = torch.zeros_like(contact, dtype=torch.float32)
                    if (
                        ball_lin_vel_w is not None
                        and racket_normal_w is not None
                        and self._prev_ball_lin_vel_w is not None
                        and self._prev_racket_normal_w is not None
                    ):
                        normal_ref = self._prev_racket_normal_w
                        normal_ref = normal_ref / torch.linalg.norm(normal_ref, dim=-1, keepdim=True).clamp_min(1.0e-6)
                        vn_prev = torch.sum(self._prev_ball_lin_vel_w * normal_ref, dim=-1)
                        vn_post = torch.sum(ball_lin_vel_w * normal_ref, dim=-1)
                        delta_vn = torch.abs(vn_post - vn_prev)
                    print(
                        "[DEBUG] newly_hit: "
                        f"env={env_id}, step={int(self.common_step_counter)}, "
                        f"sensor={bool(sensor_contact[env_id].item())}, "
                        f"geom={bool(geometric_contact[env_id].item())}, "
                        f"instant_dist={float(instant_dist[env_id].item()):.4f}, "
                        f"swept_dist={float(swept_dist[env_id].item()):.4f}, "
                        f"racket_ball_force={float(sensor_max_force[env_id].item()):.4f}, "
                        f"delta_vn={float(delta_vn[env_id].item()):.4f}, "
                        f"normal_x={float(normal_x[env_id].item()):.4f}"
                    )
                self.post_hit[newly_hit] = True
                if self._just_entered_post_hit is not None:
                    self._just_entered_post_hit[newly_hit] = True
                self._hit_drone_pos_w[newly_hit] = drone_pos_w[newly_hit]
                self._hit_drone_quat_w[newly_hit] = drone_quat_w[newly_hit]
                self._hit_drone_lin_vel_w[newly_hit] = drone_lin_vel_w[newly_hit]
                self._hit_drone_ang_vel_w[newly_hit] = drone_ang_vel_w[newly_hit]

                # 保存球的位置和速度作为轨迹起点
                ball_pos_w = self._shuttlecock.data.root_pos_w
                ball_vel_w = self._shuttlecock.data.root_lin_vel_w
                self._hit_ball_pos_w[newly_hit] = ball_pos_w[newly_hit]
                self._hit_ball_vel_w[newly_hit] = ball_vel_w[newly_hit]

                # 重置轨迹计时器
                self._hit_time_elapsed[newly_hit] = 0.0
                if self._task2_server_ground_hold_active is not None:
                    self._task2_server_ground_hold_active[newly_hit] = False
                if self._task2_server_ground_landing_pos_local is not None:
                    self._task2_server_ground_landing_pos_local[newly_hit] = 0.0
                if self._task2_server_ground_hold_elapsed_s is not None:
                    self._task2_server_ground_hold_elapsed_s[newly_hit] = 0.0

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

    def _compute_drone_net_contact_signal(self):
        sensor_contact = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        threshold = float(getattr(self.cfg, "drone_net_contact_force_threshold", 0.1))
        if bool(getattr(self.cfg, "drone_net_contact_debug_print", False)) and not self._debug_drone_net_compute_printed:
            print(
                "[DEBUG] _compute_drone_net_contact_signal active: "
                f"threshold={threshold:.6f}, "
                f"drone_sensor={self._drone_net_contact_sensor is not None}, "
                f"racket_sensor={self._racket_net_contact_sensor is not None}, "
                f"net_sensor={self._net_drone_contact_sensor is not None}"
            )
            self._debug_drone_net_compute_printed = True
        for sensor_name, sensor in (
            ("drone_net_contact_sensor", self._drone_net_contact_sensor),
            ("racket_net_contact_sensor", self._racket_net_contact_sensor),
            ("net_drone_contact_sensor", self._net_drone_contact_sensor),
        ):
            if sensor is None:
                continue
            try:
                sensor_has_data = False
                sensor_max_force = 0.0

                force_matrix = getattr(sensor.data, "force_matrix_w", None)
                if force_matrix is not None:
                    sensor_has_data = True
                    force_mag = torch.norm(force_matrix, dim=-1)
                    sensor_max_force = max(sensor_max_force, float(force_mag.max().item()))
                    sensor_contact = sensor_contact | (force_mag.max(dim=1)[0].max(dim=1)[0] > threshold)

                net_forces = getattr(sensor.data, "net_forces_w", None)
                if net_forces is not None:
                    sensor_has_data = True
                    force_mag = torch.norm(net_forces, dim=-1)
                    sensor_max_force = max(sensor_max_force, float(force_mag.max().item()))
                    sensor_contact = sensor_contact | (force_mag.max(dim=1)[0] > threshold)

                if bool(getattr(self.cfg, "drone_net_contact_debug_print", False)):
                    step = int(self.common_step_counter)
                    if step % 100 == 0 or sensor_max_force > threshold:
                        print(f"[DEBUG] {sensor_name}: max_force={sensor_max_force:.6f}, threshold={threshold:.6f}")

                if not sensor_has_data and not hasattr(self, f"_debug_{sensor_name}_none_printed"):
                    print(f"[DEBUG] {sensor_name} data has no force_matrix_w or net_forces_w")
                    setattr(self, f"_debug_{sensor_name}_none_printed", True)
            except Exception as e:
                if not bool(getattr(self, "_debug_drone_net_error_printed", False)):
                    print(f"[DEBUG] {sensor_name} exception: {e}")
                    self._debug_drone_net_error_printed = True

        if sensor_contact.any() and hasattr(self, '_debug_drone_net_count'):
            self._debug_drone_net_count += int(sensor_contact.sum())
        return sensor_contact

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

