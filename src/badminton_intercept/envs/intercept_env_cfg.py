from dataclasses import dataclass, field
from typing import Literal

# === 统一修改点：无人机最大推力 (N) ===
# 修改此值即可自动更新 ctbr_thrust_scale
MAX_THRUST_NEWTON = 20
_DRONE_MASS_KG = 0.9505  # 无人机+球拍总质量
CTBR_THRUST_SCALE = MAX_THRUST_NEWTON / _DRONE_MASS_KG  

try:
    import isaaclab.sim as sim_utils
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.sim import PhysxCfg, SimulationCfg
    from isaaclab.utils import configclass

    ISAACLAB_CFG_AVAILABLE = True
except Exception:
    ISAACLAB_CFG_AVAILABLE = False


@dataclass
class ArenaCfg:
    court_length: float = 13.4
    court_width: float = 6.1
    net_height: float = 1.55


if ISAACLAB_CFG_AVAILABLE:

    @configclass
    class InterceptEnvCfg(DirectRLEnvCfg):
        # env
        decimation = 4
        episode_length_s = 4.0
        action_space = 4
        observation_space = 31
        state_space = 34

        # scene
        scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False
        )
        
        # Contact sensor for racket-shuttlecock collision detection
        contact_sensor: ContactSensorCfg = ContactSensorCfg(
            # Attach to the shuttlecock and filter the bat so this detects
            # racket-ball contact, not body-ball contact.
            prim_path="/World/envs/env_.*/ShuttlecockProxy",
            filter_prim_paths_expr=["/World/envs/env_.*/Drone.*/[Bb]at"],
            history_length=3,
            update_period=0.0,
            track_air_time=False,
            force_threshold=1.0,
        )
        net_contact_sensor: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Net",
            filter_prim_paths_expr=["/World/envs/env_.*/ShuttlecockProxy"],
            history_length=3,
            update_period=0.0,
            track_air_time=False,
            force_threshold=0.05,
        )
        drone_net_contact_sensor: ContactSensorCfg = ContactSensorCfg(
            # Support both /Drone/base_link and /Drone/TransferAsset/base_link layouts.
            prim_path="/World/envs/env_.*/Drone.*/base_link",
            filter_prim_paths_expr=["/World/envs/env_.*/Net"],
            history_length=3,
            update_period=0.0,
            track_air_time=False,
            force_threshold=15.0,
        )
        racket_net_contact_sensor: ContactSensorCfg = ContactSensorCfg(
            # Not registered by InterceptEnv by default: some bat prims are
            # collision children without contact reporter API. Net-side sensing
            # covers racket-net contact via net_drone_contact_sensor.
            prim_path="/World/envs/env_.*/Drone.*/[Bb]at",
            filter_prim_paths_expr=["/World/envs/env_.*/Net"],
            history_length=3,
            update_period=0.0,
            track_air_time=False,
            force_threshold=15.0,
        )
        net_drone_contact_sensor: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Net",
            filter_prim_paths_expr=[
                "/World/envs/env_.*/Drone.*/base_link",
            ],
            history_length=3,
            update_period=0.0,
            track_air_time=False,
            force_threshold=15.0,
        )

        # sim
        sim: SimulationCfg = SimulationCfg(
            dt=1 / 400,
            render_interval=decimation,
              physx=PhysxCfg(
                enable_ccd=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )

        # task-specific params
        env_name: str = "intercept"
        drone_asset_kind: str = "articulation"
        drone_usd_path: str = "F:/eai/transfer/assets/air.usd"
        drone_param_yaml_path: str = ""
        drone_init_pos: tuple[float, float, float] = (2.0, 0.0, 1.2)
        drone_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        # FM-style CTBR scaling: rate*2pi, thrust*15.
        ctbr_rate_max_rad_s: float = 3.14  # 2π
        ctbr_thrust_scale: float = CTBR_THRUST_SCALE
        # Axis sign correction from CTBR command frame to simulation body-rate frame.
        # For current air.usd articulation, roll/pitch need sign flip to keep negative feedback.
        ctbr_body_rate_axis_sign: tuple[float, float, float] = (-1.0, -1.0, 1.0)
        max_angular_accel_rad_s2: float = 3
        ctbr_max_thrust_ratio: float = 1.0
        fm_p_gain: tuple[float, float, float] = (0.11, 0.11, 0.2)
        fm_i_gain: tuple[float, float, float] = (0.008, 0.008, 0.01)
        fm_d_gain: tuple[float, float, float] = (0.00075, 0.00075, 0.0)
        fm_rate_k: tuple[float, float, float] = (1.0, 1.0, 1.0)
        fm_int_lim: tuple[float, float, float] = (1.0, 1.0, 1.0)
        fm_k_coef: float = 0.0178
        fm_extra_mass_kg: float = 0.147
        fm_rate_lpf_alpha: float = 0.803307
        fm_rate_lpf_beta: float = 0.196693
        max_linear_speed_mps: float = 20.0
        termination_debug_print: bool = False
        reward_c_tilt: float = 2.0
        reward_c_ang_vel: float = 0.05
        rate_p_scale: float = 1.0
        rate_d_scale: float = 0.15
        rate_i_scale: float = 0.05
        rate_integral_limit: float = 3.0
        shuttle_radius_m: float = 0.025
        shuttle_mass_kg: float = 0.005
        shuttle_init_pos: tuple[float, float, float] = (-2.0, 0.0, 1.6)
        racket_offset_z_m: float = 0.2
        contact_radius_m: float = 0.09
        racket_body_name_expr: str = ".*[Bb]at.*"
        contact_force_threshold: float = 1.0
        hit_normal_velocity_delta_threshold_mps: float = 1.0
        net_thickness_m: float = 0.03
        net_contact_force_threshold: float = 0.05
        drone_net_contact_force_threshold: float = 15.0
        drone_net_contact_debug_print: bool = False
        weak_hit_reward: float = 5
        weak_hit_sensor_grace_steps: int = 2
        net_width_m: float = 6.5
        ground_restitution: float = 0.1
        nominal_drag_length_m: float = 4.1
        policy_hz: int = 50
        physics_hz: int = 400
        curriculum_window_episodes: int = 100
        curriculum_promote_success_rate: float = 0.85
        curriculum_promote_iteration_streak: int = 10
        curriculum_initial_stage_id: int | None = None
        curriculum_fixed_stage: bool = False
        randomization_enabled: bool = True
        # initialization ranges (now primarily controlled by curriculum stages)
        # UAV init ranges are read from curriculum.current_stage
        uav_init_lin_vel_range: tuple[float, float] = (-0.02, 0.02)
        # Ball init ranges are read from curriculum.current_stage
        # (ball_vel_x_range and ball_vel_z_range are also in curriculum)
        launch_prediction_dt: float = 0.01
        launch_prediction_horizon_s: float = 3.0
        launch_max_resample_rounds: int = 2
        launch_net_clearance_margin_m: float = 0.20
        launch_landing_margin_x_m: float = 0.1
        launch_landing_margin_y_m: float = 0.1
        launch_use_near_net_init_x: bool = True
        launch_ball_init_x_range: tuple[float, float] = (-4.5, -1.8)
        launch_vx_min_for_cross: float = 6.0
        launch_vx_max_for_cross: float = 12.0
        launch_hit_plane_z_range: tuple[float, float] = (1.0, 1.5)#击球平面设置
        launch_center_ball_init_z_on_net: bool = True
        launch_ball_init_z_half_span_m: float = 0.3
        launch_sampler_mode: Literal["online", "trajectory_library"] = "trajectory_library"
        launch_library_dir: str = "assets/trajectory_library"
        launch_library_stage_files: tuple[str, str, str] = ("serve_stage_1.pt", "serve_stage_2.pt", "serve_stage_3.pt")
        # domain randomization ranges
        dr_racket_restitution_range: tuple[float, float] = (0.75, 0.90)
        dr_ball_mass_range: tuple[float, float] = (0.0045, 0.0055)
        dr_drag_length_range: tuple[float, float] = (3.8, 4.4)
        dr_quad_mass_scale_range: tuple[float, float] = (0.95, 1.05)
        dr_quad_inertia_scale_range: tuple[float, float] = (0.95, 1.05)
        dr_thrust_coeff_scale_range: tuple[float, float] = (0.95, 1.05)
        arena: ArenaCfg = field(default_factory=ArenaCfg)
        enable_post_hit_tracking: bool = False
        enable_serve_hover: bool = False
        enable_task2_server_ground_hold: bool = False
        task2_server_ground_hold_duration_s: float = 3.0
        post_hit_trajectory_mode: str = "physx"  # "physx" or "analytical"
        serve_hover_min_height: float = 0.1
        serve_hover_max_height: float = 4.0

else:

    @dataclass
    class InterceptEnvCfg:
        env_name: str = "intercept"
        num_envs: int = 1024
        sim_dt: float = 0.005
        episode_length_s: float = 4.0
        drone_asset_kind: str = "articulation"
        drone_usd_path: str = "F:/eai/transfer/assets/air.usd"
        drone_param_yaml_path: str = ""
        drone_init_pos: tuple[float, float, float] = (2.0, 0.0, 1.2)
        drone_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        ctbr_rate_max_rad_s: float = 3.14  # π
        ctbr_thrust_scale: float = CTBR_THRUST_SCALE
        ctbr_body_rate_axis_sign: tuple[float, float, float] = (-1.0, -1.0, 1.0)
        max_angular_accel_rad_s2: float = 3
        ctbr_max_thrust_ratio: float = 1.0
        fm_p_gain: tuple[float, float, float] = (0.11, 0.11, 0.2)
        fm_i_gain: tuple[float, float, float] = (0.008, 0.008, 0.01)
        fm_d_gain: tuple[float, float, float] = (0.00075, 0.00075, 0.0)
        fm_rate_k: tuple[float, float, float] = (1.0, 1.0, 1.0)
        fm_int_lim: tuple[float, float, float] = (1.0, 1.0, 1.0)
        fm_k_coef: float = 0.0178
        fm_extra_mass_kg: float = 0.147
        fm_rate_lpf_alpha: float = 0.803307
        fm_rate_lpf_beta: float = 0.196693
        max_linear_speed_mps: float = 20.0
        termination_debug_print: bool = False
        reward_c_tilt: float = 2.0
        reward_c_ang_vel: float = 0.05
        rate_p_scale: float = 1.0
        rate_d_scale: float = 0.15
        rate_i_scale: float = 0.05
        rate_integral_limit: float = 3.0
        shuttle_radius_m: float = 0.025
        shuttle_mass_kg: float = 0.005
        shuttle_init_pos: tuple[float, float, float] = (-2.0, 0.0, 1.6)
        racket_offset_z_m: float = 0.2
        contact_radius_m: float = 0.09
        racket_body_name_expr: str = ".*[Bb]at.*"
        contact_force_threshold: float = 1.0
        hit_normal_velocity_delta_threshold_mps: float = 1.0
        net_thickness_m: float = 0.03
        net_contact_force_threshold: float = 0.05
        drone_net_contact_force_threshold: float = 15.0
        drone_net_contact_debug_print: bool = False
        weak_hit_reward: float = 5
        weak_hit_sensor_grace_steps: int = 2
        net_width_m: float = 6.5
        ground_restitution: float = 0.1
        nominal_drag_length_m: float = 4.1
        policy_hz: int = 50
        physics_hz: int = 400
        curriculum_window_episodes: int = 100
        curriculum_promote_success_rate: float = 0.85
        curriculum_promote_iteration_streak: int = 10
        curriculum_initial_stage_id: int | None = None
        curriculum_fixed_stage: bool = False
        # initialization ranges (now primarily controlled by curriculum stages)
        # UAV init ranges are read from curriculum.current_stage
        uav_init_lin_vel_range: tuple[float, float] = (-0.02, 0.02)
        # Ball init ranges are read from curriculum.current_stage
        # (ball_vel_x_range and ball_vel_z_range are also in curriculum)
        launch_prediction_dt: float = 0.01
        launch_prediction_horizon_s: float = 3.0
        launch_max_resample_rounds: int = 2
        launch_net_clearance_margin_m: float = 0.20
        launch_landing_margin_x_m: float = 0.1
        launch_landing_margin_y_m: float = 0.1
        launch_use_near_net_init_x: bool = True
        launch_ball_init_x_range: tuple[float, float] = (-4.5, -1.8)
        launch_vx_min_for_cross: float = 6.0
        launch_vx_max_for_cross: float = 12.0
        launch_hit_plane_z_range: tuple[float, float] = (1.0, 1.5)
        launch_center_ball_init_z_on_net: bool = True
        launch_ball_init_z_half_span_m: float = 0.3
        launch_sampler_mode: Literal["online", "trajectory_library"] = "trajectory_library"
        launch_library_dir: str = "assets/trajectory_library"
        launch_library_stage_files: tuple[str, str, str] = ("serve_stage_1.pt", "serve_stage_2.pt", "serve_stage_3.pt")
        dr_racket_restitution_range: tuple[float, float] = (0.75, 0.90)
        dr_ball_mass_range: tuple[float, float] = (0.0045, 0.0055)
        dr_drag_length_range: tuple[float, float] = (3.8, 4.4)
        dr_quad_mass_scale_range: tuple[float, float] = (0.95, 1.05)
        dr_quad_inertia_scale_range: tuple[float, float] = (0.95, 1.05)
        dr_thrust_coeff_scale_range: tuple[float, float] = (0.95, 1.05)
        arena: ArenaCfg = field(default_factory=ArenaCfg)
        randomization_enabled: bool = True
        enable_post_hit_tracking: bool = False
        enable_serve_hover: bool = False
        enable_task2_server_ground_hold: bool = False
        task2_server_ground_hold_duration_s: float = 3.0
        post_hit_trajectory_mode: str = "physx"  # "physx" or "analytical"
        serve_hover_min_height: float = 0.1
        serve_hover_max_height: float = 4.0


DEFAULT_ENV_CFG = InterceptEnvCfg()
