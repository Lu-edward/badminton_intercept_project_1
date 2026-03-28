"""Sign-consistency check for CTBR control chain.

This script validates whether commanded body rates (wx/wy/wz) produce
matching-sign body angular velocity and mixer torque in simulation.
"""

from __future__ import annotations

import argparse
import time

import torch
from isaaclab.app import AppLauncher
from badminton_intercept.envs.intercept_env_cfg import CTBR_THRUST_SCALE


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Check CTBR sign consistency (command -> torque -> body rate).")
    parser.add_argument("--task", type=str, default="Isaac-Badminton-Intercept-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drone_kind", type=str, default="articulation", choices=["rigid_object", "articulation"])
    parser.add_argument("--steps_per_phase", type=int, default=180)
    parser.add_argument("--warmup_steps", type=int, default=40)
    parser.add_argument("--rate_cmd", type=float, default=0.35, help="Commanded body-rate magnitude [rad/s].")
    parser.add_argument("--thrust_cmd", type=float, default=9.81, help="CTBR thrust reference [m/s^2].")
    parser.add_argument("--x_center", type=float, default=1.5)
    parser.add_argument("--z_ref", type=float, default=1.6)
    parser.add_argument("--fm_gain_scale", type=float, default=1.0)
    parser.add_argument("--max_angacc", type=float, default=1.0)
    parser.add_argument("--axis_sign_x", type=float, default=-1.0)
    parser.add_argument("--axis_sign_y", type=float, default=-1.0)
    parser.add_argument("--axis_sign_z", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=0.0, help="Playback speed. 0 = no sleep.")
    parser.add_argument("--print_every", type=int, default=60)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_known_args()


args_cli, _hydra_args = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from badminton_intercept.control.rotor_mixer import rotor_thrust_to_body_wrench
from badminton_intercept.envs.intercept_env_cfg import InterceptEnvCfg
from badminton_intercept.train.register_task import TASK_ID, register_task


@torch.no_grad()
def main() -> None:
    register_task()

    env_cfg = InterceptEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.drone_asset_kind = args_cli.drone_kind
    env_cfg.randomization_enabled = False
    env_cfg.termination_debug_print = True
    env_cfg.max_angular_accel_rad_s2 = float(args_cli.max_angacc)
    env_cfg.ctbr_body_rate_axis_sign = (
        float(args_cli.axis_sign_x),
        float(args_cli.axis_sign_y),
        float(args_cli.axis_sign_z),
    )

    scale = float(args_cli.fm_gain_scale)
    if scale > 0.0:
        env_cfg.fm_p_gain = tuple(float(v) * scale for v in env_cfg.fm_p_gain)
        env_cfg.fm_i_gain = tuple(float(v) * scale for v in env_cfg.fm_i_gain)
        env_cfg.fm_d_gain = tuple(float(v) * scale for v in env_cfg.fm_d_gain)

    # Stable initial conditions.
    env_cfg.uav_init_x_range = (args_cli.x_center - 0.05, args_cli.x_center + 0.05)
    env_cfg.uav_init_y_range = (-0.05, 0.05)
    env_cfg.uav_init_z_range = (args_cli.z_ref - 0.03, args_cli.z_ref + 0.03)
    env_cfg.uav_init_lin_vel_range = (0.0, 0.0)
    env_cfg.ball_init_x_range = (-6.0, -6.0)
    env_cfg.ball_init_y_range = (0.0, 0.0)
    env_cfg.ball_init_z_range = (4.0, 4.0)
    env_cfg.ball_vel_x_range = (0.0, 0.0)
    env_cfg.ball_vel_y_range = (0.0, 0.0)
    env_cfg.ball_vel_z_range = (0.0, 0.0)

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task or TASK_ID, cfg=env_cfg)

    phases = [
        ("hover", (0.0, 0.0, 0.0)),
        ("roll_pos", (args_cli.rate_cmd, 0.0, 0.0)),
        ("roll_neg", (-args_cli.rate_cmd, 0.0, 0.0)),
        ("pitch_pos", (0.0, args_cli.rate_cmd, 0.0)),
        ("pitch_neg", (0.0, -args_cli.rate_cmd, 0.0)),
        ("yaw_pos", (0.0, 0.0, args_cli.rate_cmd)),
        ("yaw_neg", (0.0, 0.0, -args_cli.rate_cmd)),
    ]

    axis_names = ["wx", "wy", "wz"]
    rate_scale = float(env.unwrapped.cfg.ctbr_rate_max_rad_s)
    thrust_scale = float(getattr(env.unwrapped.cfg, "ctbr_thrust_scale", CTBR_THRUST_SCALE))

    def build_action_raw(wx: float, wy: float, wz: float, thrust: float, num_envs: int, device) -> torch.Tensor:
        wx_hat = torch.full((num_envs,), float(wx / rate_scale), device=device).clamp(-0.95, 0.95)
        wy_hat = torch.full((num_envs,), float(wy / rate_scale), device=device).clamp(-0.95, 0.95)
        wz_hat = torch.full((num_envs,), float(wz / rate_scale), device=device).clamp(-0.95, 0.95)
        thrust_hat = torch.full((num_envs,), float((2.0 * thrust / thrust_scale) - 1.0), device=device).clamp(-0.95, 0.95)
        action_hat = torch.stack([wx_hat, wy_hat, wz_hat, thrust_hat], dim=-1)
        return torch.atanh(action_hat)

    try:
        _obs, _ = env.reset()
        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs
        step_dt = float(env.unwrapped.step_dt)

        print(
            f"[info] sign-check started | drone_kind={args_cli.drone_kind} "
            f"num_envs={num_envs} rate_cmd={args_cli.rate_cmd:.3f} thrust_cmd={args_cli.thrust_cmd:.3f} "
            f"axis_sign={env_cfg.ctbr_body_rate_axis_sign}"
        )

        results = []
        for phase_name, (cmd_wx, cmd_wy, cmd_wz) in phases:
            accum_rate = torch.zeros(3, device=device)
            accum_torque = torch.zeros(3, device=device)
            valid_steps = 0
            resets_in_phase = 0

            for step in range(args_cli.steps_per_phase):
                action_raw = build_action_raw(
                    wx=cmd_wx,
                    wy=cmd_wy,
                    wz=cmd_wz,
                    thrust=float(args_cli.thrust_cmd),
                    num_envs=num_envs,
                    device=device,
                )

                _obs, rew, terminated, truncated, info = env.step(action_raw)

                pos_w, quat_w, _lin_vel_w, ang_vel_w = env.unwrapped._get_drone_kinematics()
                body_rate = env.unwrapped._quat_rotate_world_to_body(quat_w, ang_vel_w)

                rotor_thrust = env.unwrapped._rotor_thrust_cmd_n
                _, body_torque = rotor_thrust_to_body_wrench(
                    rotor_thrust_n=rotor_thrust,
                    arm_lengths=env.unwrapped._arm_lengths_m,
                    rotor_angles=env.unwrapped._rotor_angles_rad,
                    directions=env.unwrapped._rotor_directions,
                    force_constants=env.unwrapped._force_constants,
                    moment_constants=env.unwrapped._moment_constants,
                )

                if step >= args_cli.warmup_steps:
                    accum_rate += body_rate.mean(dim=0)
                    accum_torque += body_torque.mean(dim=0)
                    valid_steps += 1

                if bool(torch.any(terminated | truncated)):
                    resets_in_phase += int((terminated | truncated).sum().item())

                if step % max(1, args_cli.print_every) == 0 or step == args_cli.steps_per_phase - 1:
                    mean_pos = pos_w.mean(dim=0).detach().cpu().tolist()
                    mean_rate = body_rate.mean(dim=0).detach().cpu().tolist()
                    mean_torque = body_torque.mean(dim=0).detach().cpu().tolist()
                    print(
                        f"[{phase_name} step {step:03d}] pos=({mean_pos[0]:.2f},{mean_pos[1]:.2f},{mean_pos[2]:.2f}) "
                        f"rate=({mean_rate[0]:.3f},{mean_rate[1]:.3f},{mean_rate[2]:.3f}) "
                        f"torque=({mean_torque[0]:.4f},{mean_torque[1]:.4f},{mean_torque[2]:.4f}) "
                        f"reward={float(rew.mean()):.3f}"
                    )

                if args_cli.speed > 0:
                    time.sleep(step_dt / args_cli.speed)

            denom = max(valid_steps, 1)
            mean_rate = (accum_rate / denom).detach().cpu()
            mean_torque = (accum_torque / denom).detach().cpu()
            cmd_vec = torch.tensor([cmd_wx, cmd_wy, cmd_wz], dtype=torch.float32)

            axis = int(torch.argmax(torch.abs(cmd_vec)).item()) if float(torch.abs(cmd_vec).sum()) > 0 else -1
            cmd_axis = float(cmd_vec[axis].item()) if axis >= 0 else 0.0
            meas_axis = float(mean_rate[axis].item()) if axis >= 0 else 0.0
            torque_axis = float(mean_torque[axis].item()) if axis >= 0 else 0.0

            rate_sign_ok = True if axis < 0 else (cmd_axis * meas_axis > 0.0)
            torque_sign_ok = True if axis < 0 else (cmd_axis * torque_axis > 0.0)

            results.append(
                {
                    "phase": phase_name,
                    "cmd": (cmd_wx, cmd_wy, cmd_wz),
                    "mean_rate": mean_rate.tolist(),
                    "mean_torque": mean_torque.tolist(),
                    "axis": axis_names[axis] if axis >= 0 else "none",
                    "rate_sign_ok": bool(rate_sign_ok),
                    "torque_sign_ok": bool(torque_sign_ok),
                    "resets": resets_in_phase,
                }
            )

        print("\n=== CTBR Sign Check Summary ===")
        for r in results:
            cmd = r["cmd"]
            rate = r["mean_rate"]
            torque = r["mean_torque"]
            print(
                f"{r['phase']:>10} | cmd=({cmd[0]: .3f},{cmd[1]: .3f},{cmd[2]: .3f}) "
                f"| mean_rate=({rate[0]: .3f},{rate[1]: .3f},{rate[2]: .3f}) "
                f"| mean_torque=({torque[0]: .4f},{torque[1]: .4f},{torque[2]: .4f}) "
                f"| axis={r['axis']} rate_sign_ok={r['rate_sign_ok']} torque_sign_ok={r['torque_sign_ok']} "
                f"| resets={r['resets']}"
            )

        bad_phases = [r["phase"] for r in results if not (r["rate_sign_ok"] and r["torque_sign_ok"])]
        if len(bad_phases) == 0:
            print("RESULT: PASS (command signs are consistent with torque and measured body rates).")
        else:
            print(f"RESULT: FAIL (sign inconsistency detected in phases: {bad_phases}).")

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
