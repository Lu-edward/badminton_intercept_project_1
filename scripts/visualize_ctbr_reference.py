"""Visual reference test: drive periodic CTBR commands and inspect drone motion near the net."""

from __future__ import annotations

import argparse
import math
import time

import torch
from isaaclab.app import AppLauncher


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Visualize deterministic periodic CTBR reference motion.")
    parser.add_argument("--task", type=str, default="Isaac-Badminton-Intercept-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drone_kind", type=str, default="articulation", choices=["rigid_object", "articulation"])
    parser.add_argument("--mode", type=str, default="reference", choices=["reference", "fixed_ctbr"])
    parser.add_argument("--fixed_wx", type=float, default=0.0, help="Fixed body-rate wx [rad/s] for fixed_ctbr mode.")
    parser.add_argument("--fixed_wy", type=float, default=0.0, help="Fixed body-rate wy [rad/s] for fixed_ctbr mode.")
    parser.add_argument("--fixed_wz", type=float, default=0.0, help="Fixed body-rate wz [rad/s] for fixed_ctbr mode.")
    parser.add_argument("--fixed_thrust", type=float, default=9.81, help="Fixed CTBR thrust ref [m/s^2] for fixed_ctbr mode.")
    parser.add_argument("--freq_hz", type=float, default=0.15)
    parser.add_argument("--x_center", type=float, default=1.5)
    parser.add_argument("--x_amp", type=float, default=0.4)
    parser.add_argument("--y_amp", type=float, default=0.3)
    parser.add_argument("--z_ref", type=float, default=1.6)
    parser.add_argument("--z_amp", type=float, default=0.05)
    parser.add_argument("--max_tilt_deg", type=float, default=10.0)
    parser.add_argument("--max_rate_rad_s", type=float, default=1.2)
    parser.add_argument("--kp_pos", type=float, default=0.6)
    parser.add_argument("--kd_pos", type=float, default=0.7)
    parser.add_argument("--kp_att", type=float, default=2.5)
    parser.add_argument("--fm_gain_scale", type=float, default=1.0, help="Scale FM PID (P/I/D) gains for debug.")
    parser.add_argument("--max_angacc", type=float, default=1.4, help="Max angular acceleration [rad/s^2].")
    parser.add_argument("--speed", type=float, default=0.5, help="Playback speed. 1.0 = realtime.")
    parser.add_argument("--print_every", type=int, default=100)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_known_args()


args_cli, _hydra_args = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from badminton_intercept.envs.intercept_env_cfg import InterceptEnvCfg
from badminton_intercept.train.register_task import TASK_ID, register_task


def quat_to_roll_pitch(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
    return roll, pitch


def main() -> None:
    register_task()

    env_cfg = InterceptEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.drone_asset_kind = args_cli.drone_kind
    env_cfg.randomization_enabled = False
    env_cfg.termination_debug_print = True
    env_cfg.max_angular_accel_rad_s2 = float(args_cli.max_angacc)
    gain_scale = float(args_cli.fm_gain_scale)
    if gain_scale > 0.0:
        env_cfg.fm_p_gain = tuple(float(v) * gain_scale for v in env_cfg.fm_p_gain)
        env_cfg.fm_i_gain = tuple(float(v) * gain_scale for v in env_cfg.fm_i_gain)
        env_cfg.fm_d_gain = tuple(float(v) * gain_scale for v in env_cfg.fm_d_gain)

    # Stable reset near the net for visual verification.
    env_cfg.uav_init_x_range = (args_cli.x_center - 0.1, args_cli.x_center + 0.1)
    env_cfg.uav_init_y_range = (-0.1, 0.1)
    env_cfg.uav_init_z_range = (args_cli.z_ref - 0.05, args_cli.z_ref + 0.05)
    env_cfg.uav_init_lin_vel_range = (0.0, 0.0)

    # Keep shuttle motion minimal so focus stays on drone control chain.
    env_cfg.ball_init_x_range = (-6.0, -6.0)
    env_cfg.ball_init_y_range = (0.0, 0.0)
    env_cfg.ball_init_z_range = (4.0, 4.0)
    env_cfg.ball_vel_x_range = (0.0, 0.0)
    env_cfg.ball_vel_y_range = (0.0, 0.0)
    env_cfg.ball_vel_z_range = (0.0, 0.0)

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task or TASK_ID, cfg=env_cfg)

    try:
        obs, _ = env.reset()
        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs
        step_dt = float(env.unwrapped.step_dt)
        max_tilt = math.radians(args_cli.max_tilt_deg)
        g = 9.81
        if args_cli.mode == "fixed_ctbr":
            print(
                "[info] fixed_ctbr mode: "
                f"wx={args_cli.fixed_wx:.3f}, wy={args_cli.fixed_wy:.3f}, wz={args_cli.fixed_wz:.3f}, thrust={args_cli.fixed_thrust:.3f}"
            )

        for step in range(args_cli.steps):
            t = step * step_dt

            drone_pos, drone_quat, drone_lin_vel, _ = env.unwrapped._get_drone_kinematics()
            roll, pitch = quat_to_roll_pitch(drone_quat)

            if args_cli.mode == "fixed_ctbr":
                wx_cmd = torch.full((num_envs,), float(args_cli.fixed_wx), device=device)
                wy_cmd = torch.full((num_envs,), float(args_cli.fixed_wy), device=device)
                wz_cmd = torch.full((num_envs,), float(args_cli.fixed_wz), device=device)
                thrust_ref = torch.full((num_envs,), float(args_cli.fixed_thrust), device=device).clamp(0.0, 15.0)
            else:
                x_ref = args_cli.x_center + args_cli.x_amp * math.sin(2.0 * math.pi * args_cli.freq_hz * t)
                y_ref = args_cli.y_amp * math.sin(2.0 * math.pi * args_cli.freq_hz * t + math.pi / 2.0)
                z_ref = args_cli.z_ref + args_cli.z_amp * math.sin(2.0 * math.pi * 0.5 * args_cli.freq_hz * t)

                pos_ref = torch.tensor([x_ref, y_ref, z_ref], device=device).unsqueeze(0).repeat(num_envs, 1)
                vel_ref = torch.zeros_like(pos_ref)

                acc_cmd = args_cli.kp_pos * (pos_ref - drone_pos) + args_cli.kd_pos * (vel_ref - drone_lin_vel)
                ax = acc_cmd[:, 0]
                ay = acc_cmd[:, 1]
                az = acc_cmd[:, 2]

                roll_des = torch.clamp(-ay / g, -max_tilt, max_tilt)
                pitch_des = torch.clamp(ax / g, -max_tilt, max_tilt)

                wx_cmd = torch.clamp(args_cli.kp_att * (roll_des - roll), -args_cli.max_rate_rad_s, args_cli.max_rate_rad_s)
                wy_cmd = torch.clamp(args_cli.kp_att * (pitch_des - pitch), -args_cli.max_rate_rad_s, args_cli.max_rate_rad_s)
                wz_cmd = torch.zeros_like(wx_cmd)

                thrust_ref = torch.clamp(g + az, 0.0, 15.0)

            wx_hat = torch.clamp(wx_cmd / math.pi, -0.95, 0.95)
            wy_hat = torch.clamp(wy_cmd / math.pi, -0.95, 0.95)
            wz_hat = torch.clamp(wz_cmd / math.pi, -0.95, 0.95)
            thrust_hat = torch.clamp((2.0 * thrust_ref / 15.0) - 1.0, -0.95, 0.95)
            action_hat = torch.stack([wx_hat, wy_hat, wz_hat, thrust_hat], dim=-1)
            action_raw = torch.atanh(action_hat)

            obs, rew, terminated, truncated, info = env.step(action_raw)

            if step % max(1, args_cli.print_every) == 0 or step == args_cli.steps - 1:
                mean_pos = drone_pos.mean(dim=0).detach().cpu().tolist()
                mean_roll_deg = float(torch.rad2deg(roll).mean().item())
                mean_pitch_deg = float(torch.rad2deg(pitch).mean().item())
                mean_ang_speed = float(torch.linalg.norm(env.unwrapped._drone.data.root_ang_vel_w, dim=-1).mean().item())
                thrust_msg = ""
                if hasattr(env.unwrapped, "_rotor_thrust_cmd_n") and env.unwrapped._rotor_thrust_cmd_n is not None:
                    mean_rotor_thrust = float(env.unwrapped._rotor_thrust_cmd_n.mean().item())
                    thrust_msg = f" rotor_thrust={mean_rotor_thrust:.2f}N"
                print(
                    f"[step {step:04d}] pos=({mean_pos[0]:.2f},{mean_pos[1]:.2f},{mean_pos[2]:.2f}) "
                    f"roll={mean_roll_deg:.1f}deg pitch={mean_pitch_deg:.1f}deg "
                    f"ang_speed={mean_ang_speed:.2f}rad/s "
                    f"cmd=({float(wx_cmd.mean()):.2f},{float(wy_cmd.mean()):.2f},{float(wz_cmd.mean()):.2f},{float(thrust_ref.mean()):.2f}) "
                    f"mean_reward={float(rew.mean()):.3f}{thrust_msg}"
                )

            if bool(torch.any(terminated | truncated)):
                num_done = int((terminated | truncated).sum().item())
                print(f"[info] reset triggered for {num_done} env(s) at step={step}")
                if isinstance(info, dict) and "termination" in info:
                    print(f"[info] termination summary: {info['termination']}")

            if args_cli.speed > 0:
                time.sleep(step_dt / args_cli.speed)

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
