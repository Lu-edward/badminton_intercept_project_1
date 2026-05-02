from __future__ import annotations

from badminton_intercept.control.attitude_pd import compute_rate_pid_angacc
from badminton_intercept.control.ctbr_decoder import decode_ctbr_action
from badminton_intercept.control.rotor_mixer import rotor_thrust_to_body_wrench
from badminton_intercept.control.x152b_ctbr_controller import compute_x152b_ctbr_wrench
from badminton_intercept.control.x152b_params import DEFAULT_X152B_PARAMS
from badminton_intercept.envs.intercept_env_cfg import CTBR_THRUST_SCALE
from badminton_intercept.envs.intercept_env_common import ISAACLAB_RUNTIME_AVAILABLE, torch
from badminton_intercept.physics.shuttle_aero import compute_drag_force_tensor


class InterceptEnvControlMixin:
    def _pre_physics_step(self, actions):
        self.last_actions = actions
        # Reset trajectory update guard at start of each step
        self._post_hit_trajectory_updated = False
        self._task2_server_ground_hold_updated = False
        if self._just_entered_post_hit is not None:
            self._just_entered_post_hit.zero_()
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
        dt = float(getattr(self.cfg.sim, "dt", 1.0 / 200.0))
        effective_mass = self._mass_kg * self._drone_mass_scale + float(getattr(self.cfg, "fm_extra_mass_kg", 0.147))

        if str(getattr(self.cfg, "drone_control_mode", "")).lower() == "x152b_airgym":
            control_out = compute_x152b_ctbr_wrench(
                actions=self._actions,
                body_rate_rad_s=body_rate_rad_s,
                prev_filtered_body_rate_rad_s=self._filtered_body_rate_rad_s,
                rate_integral=self._rate_integral,
                dt=dt,
                effective_mass_kg=effective_mass,
                params=DEFAULT_X152B_PARAMS,
                rate_scale_rad_s=float(self.cfg.ctbr_rate_max_rad_s),
                thrust_scale=float(getattr(self.cfg, "ctbr_thrust_scale", CTBR_THRUST_SCALE)),
                body_rate_axis_sign=tuple(getattr(self.cfg, "ctbr_body_rate_axis_sign", (1.0, 1.0, 1.0))),
            )
            self._rate_integral = control_out.rate_integral
            self._filtered_body_rate_rad_s = control_out.filtered_body_rate_rad_s
            self._rotor_thrust_cmd_n = control_out.rotor_force_n

            force_w = self._quat_rotate_body_to_world(drone_quat_w, control_out.body_force)
            torque_w = self._quat_rotate_body_to_world(drone_quat_w, control_out.body_torque)
            self._drone.instantaneous_wrench_composer.set_forces_and_torques(
                forces=force_w.unsqueeze(1),
                torques=torque_w.unsqueeze(1),
                is_global=True,
                body_ids=self._force_body_ids,
            )
            self._apply_external_forces()
            return None

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

    def _apply_post_hit_analytical_trajectory(self):
        """Override ball position using analytical trajectory for post-hit environments.

        After a hit is detected, the ball follows a hybrid analytical trajectory:
        - Horizontal (X-Y): Logarithmic solution for quadratic drag
        - Vertical (Z): Exponential solution for linearized drag

        Guard: Only updates once per step to avoid time being incremented multiple times.
        """
        if not (ISAACLAB_RUNTIME_AVAILABLE and torch is not None):
            return
        if self._shuttlecock is None:
            return
        if not self.enable_post_hit_tracking:
            return
        if self.post_hit is None or not self.post_hit.any():
            return
        if self._post_hit_trajectory_updated:
            return

        post_hit_mask = self.post_hit
        sim_dt = self._get_step_dt_s()

        # Increment time elapsed since hit for post-hit environments
        self._hit_time_elapsed[post_hit_mask] += sim_dt

        trajectory_mode = getattr(self.cfg, "post_hit_trajectory_mode", "physx")
        if trajectory_mode != "analytical":
            self._post_hit_trajectory_updated = True
            return

        from badminton_intercept.physics.shuttle_aero import (
            get_trajectory_point_batch,
            get_trajectory_velocity_batch,
        )

        hit_pos = self._hit_ball_pos_w[post_hit_mask]
        hit_vel = self._hit_ball_vel_w[post_hit_mask]
        t = self._hit_time_elapsed[post_hit_mask]

        # Compute analytical trajectory positions and velocities
        new_positions = get_trajectory_point_batch(hit_pos, hit_vel, t)
        new_velocities = get_trajectory_velocity_batch(hit_pos, hit_vel, t)

        # Override ball position and velocity in simulation
        env_ids = torch.nonzero(post_hit_mask, as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            # Write position (translation + rotation as quaternion)
            current_quats = self._shuttlecock.data.root_quat_w[env_ids]
            pos_with_quat = torch.cat([new_positions, current_quats], dim=-1)
            self._shuttlecock.write_root_pose_to_sim(pos_with_quat, env_ids=env_ids)
            # Write velocity (linear + angular)
            current_ang_vel = self._shuttlecock.data.root_ang_vel_w[env_ids]
            vel_with_ang = torch.cat([new_velocities, current_ang_vel], dim=-1)
            self._shuttlecock.write_root_velocity_to_sim(vel_with_ang, env_ids=env_ids)

        self._post_hit_trajectory_updated = True

    def _get_serve_hover_policy_mask(self, device=None) -> "torch.Tensor":
        target_device = self.device if device is None else device
        if torch is None:
            return None
        if self.post_hit is None:
            return torch.zeros((self.num_envs,), dtype=torch.bool, device=target_device)

        delay_steps = int(getattr(self.cfg, "serve_hover_policy_delay_steps", 1) or 0)
        if delay_steps <= 0 or self._hit_time_elapsed is None:
            return self.post_hit.to(device=target_device)

        delay_s = delay_steps * self._get_step_dt_s()
        return (self.post_hit & (self._hit_time_elapsed >= (delay_s - 1.0e-9))).to(device=target_device)

    def _get_step_dt_s(self) -> float:
        sim_cfg = getattr(self.cfg, "sim", None)
        if sim_cfg is not None and hasattr(sim_cfg, "dt"):
            return float(sim_cfg.dt)
        return float(getattr(self.cfg, "sim_dt", 0.005))

    def _is_task2_server_ground_hold_enabled(self) -> bool:
        return bool(getattr(self.cfg, "enable_task2_server_ground_hold", False))

    def _get_task2_server_ground_hold_duration_s(self) -> float:
        return float(getattr(self.cfg, "task2_server_ground_hold_duration_s", 3.0))

    def _update_task2_server_ground_hold_state(self, ball_pos_local, z_threshold: float = 0.1):
        if (
            ball_pos_local is None
            or not self.enable_post_hit_tracking
            or not self._is_task2_server_ground_hold_enabled()
            or self.enable_serve_hover
            or self.post_hit is None
            or self._task2_server_ground_hold_active is None
            or self._task2_server_ground_landing_pos_local is None
            or self._task2_server_ground_hold_elapsed_s is None
        ):
            return

        in_server_half = ball_pos_local[:, 0] <= 0.0
        server_grounded_now = self.post_hit & in_server_half & (ball_pos_local[:, 2] <= z_threshold)
        newly_grounded = server_grounded_now & (~self._task2_server_ground_hold_active)

        if newly_grounded.any():
            landing_pos_local = ball_pos_local[newly_grounded].clone()
            landing_pos_local[:, 2] = 0.0
            self._task2_server_ground_landing_pos_local[newly_grounded] = landing_pos_local
            self._task2_server_ground_hold_elapsed_s[newly_grounded] = 0.0
            self._task2_server_ground_hold_active[newly_grounded] = True

        active_before_step = self._task2_server_ground_hold_active & (~newly_grounded)
        if (not self._task2_server_ground_hold_updated) and bool(torch.any(active_before_step)):
            self._task2_server_ground_hold_elapsed_s[active_before_step] += self._get_step_dt_s()
        self._task2_server_ground_hold_updated = True

    def _get_task2_ball_state_for_policy(self, ball_pos_local, ball_lin_vel_w=None):
        if (
            ball_pos_local is None
            or not self.enable_post_hit_tracking
            or not self._is_task2_server_ground_hold_enabled()
            or self.enable_serve_hover
            or self._task2_server_ground_hold_active is None
        ):
            return ball_pos_local, ball_lin_vel_w

        self._update_task2_server_ground_hold_state(ball_pos_local, z_threshold=0.1)
        hold_active = self._task2_server_ground_hold_active
        if not bool(torch.any(hold_active)):
            return ball_pos_local, ball_lin_vel_w

        effective_ball_pos_local = ball_pos_local.clone()
        effective_ball_pos_local[hold_active] = self._task2_server_ground_landing_pos_local[hold_active]

        if ball_lin_vel_w is None:
            return effective_ball_pos_local, None

        effective_ball_lin_vel_w = ball_lin_vel_w.clone()
        effective_ball_lin_vel_w[hold_active] = 0.0
        return effective_ball_pos_local, effective_ball_lin_vel_w
