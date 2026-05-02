from __future__ import annotations

from typing import Any

from badminton_intercept.envs.launch_sampler import get_launch_sampler_mode
from badminton_intercept.mdp.curriculum import CurriculumManager


class InterceptEnvStateMixin:
    def _configure_policy_modes(self, cfg) -> None:
        self.enable_serve_hover = bool(getattr(cfg, "enable_serve_hover", False))
        if self.enable_serve_hover:
            cfg.enable_post_hit_tracking = True
            cfg.observation_space = 23
            cfg.state_space = 23

        self.enable_post_hit_tracking = bool(getattr(cfg, "enable_post_hit_tracking", False))

    def _initialize_env_state(self, cfg) -> None:
        self._scene_cfgs = None
        self._drone = None
        self._shuttlecock = None
        self._net = None
        self._racket_contact_sensor = None
        self._net_contact_sensor = None
        self._drone_net_contact_sensor = None
        self._racket_net_contact_sensor = None
        self._net_drone_contact_sensor = None
        self._racket_body_ids = None
        self.last_actions = None
        self._debug_drone_net_count = 0
        self._debug_drone_net_error_printed = False
        self._debug_drone_net_setup_printed = False
        self._debug_drone_net_compute_printed = False

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
        self._air_params = None
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
        self._last_sensor_hit = None
        self._last_geometric_hit = None
        self._prev_ball_pos_w = None
        self._prev_ball_lin_vel_w = None
        self._prev_racket_contact_pos_w = None
        self._prev_racket_normal_w = None
        self._contact_reference_valid = None
        self.post_hit = None
        self._just_entered_post_hit = None
        self._serve_hover_termination_rewards = None
        self._hit_drone_pos_w = None
        self._hit_drone_quat_w = None
        self._hit_drone_lin_vel_w = None
        self._hit_drone_ang_vel_w = None
        self._hit_ball_pos_w = None
        self._hit_ball_vel_w = None
        self._hit_time_elapsed = None
        self._post_hit_trajectory_updated = False
        self._task2_server_ground_hold_active = None
        self._task2_server_ground_landing_pos_local = None
        self._task2_server_ground_hold_elapsed_s = None
        self._task2_server_ground_hold_updated = False
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
