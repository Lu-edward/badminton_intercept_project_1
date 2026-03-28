"""Local smoke test for control chain: action -> control -> wrench -> drone motion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
try:
    from isaaclab.app import AppLauncher
except Exception as exc:
    raise SystemExit(
        "Cannot import isaaclab. Please activate Isaac Lab Python env first "
        "(for this project, run scripts/activate_env.ps1), then rerun this script."
    ) from exc


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Smoke-test whether the drone physically moves under control actions.")
    parser.add_argument("--task", type=str, default="Isaac-Badminton-Intercept-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--warmup_steps", type=int, default=40)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--move_threshold_m", type=float, default=0.10)
    parser.add_argument("--speed_threshold_mps", type=float, default=0.20)
    parser.add_argument("--disable_randomization", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_known_args()


args_cli, _hydra_args = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from badminton_intercept.envs.intercept_env_cfg import InterceptEnvCfg
from badminton_intercept.train.register_task import TASK_ID, register_task


@dataclass
class MotionStats:
    displacement_m: float
    max_speed_mps: float
    mean_speed_mps: float
    max_height_delta_m: float


def _as_scalar(x: torch.Tensor) -> float:
    return float(x.detach().float().item())


def _collect_motion_stats(env, initial_pos_w: torch.Tensor, speed_hist: list[torch.Tensor]) -> MotionStats:
    drone = env.unwrapped._drone
    pos_w = drone.data.root_pos_w
    vel_w = drone.data.root_lin_vel_w

    displacement = torch.linalg.norm(pos_w - initial_pos_w, dim=-1).mean()
    speed = torch.linalg.norm(vel_w, dim=-1)
    speed_hist_t = torch.cat(speed_hist, dim=0) if len(speed_hist) > 0 else speed
    mean_speed = speed_hist_t.mean()
    max_speed = speed_hist_t.max()
    max_height_delta = (pos_w[:, 2] - initial_pos_w[:, 2]).abs().max()

    return MotionStats(
        displacement_m=_as_scalar(displacement),
        max_speed_mps=_as_scalar(max_speed),
        mean_speed_mps=_as_scalar(mean_speed),
        max_height_delta_m=_as_scalar(max_height_delta),
    )


def main() -> None:
    register_task()

    env_cfg = InterceptEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.disable_randomization:
        env_cfg.randomization_enabled = False

    task_id = args_cli.task or TASK_ID
    env = gym.make(task_id, cfg=env_cfg)

    try:
        obs, _ = env.reset()
        drone = env.unwrapped._drone
        if drone is None:
            raise RuntimeError("Drone asset is not initialized in environment.")

        initial_pos_w = drone.data.root_pos_w.clone()
        speed_hist: list[torch.Tensor] = []

        n = env.unwrapped.num_envs
        device = env.unwrapped.device

        # FM-style CTBR order is [wx, wy, wz, thrust].
        # Phase A: near-hover thrust with zero body-rate commands.
        action_a = torch.tensor([[0.0, 0.0, 0.0, 0.35]], device=device).repeat(n, 1)
        # Phase B: stronger thrust + body-rate commands to ensure visible motion.
        action_b = torch.tensor([[0.5, -0.3, 0.2, 0.85]], device=device).repeat(n, 1)

        total_steps = max(args_cli.steps, args_cli.warmup_steps + 1)
        for step in range(total_steps):
            if step < args_cli.warmup_steps:
                action = action_a
            else:
                action = action_b

            obs, rew, terminated, truncated, extras = env.step(action)
            vel_w = drone.data.root_lin_vel_w
            speed = torch.linalg.norm(vel_w, dim=-1)
            speed_hist.append(speed.detach().cpu())

            if step % max(1, args_cli.print_every) == 0 or step == total_steps - 1:
                pos_w = drone.data.root_pos_w
                mean_h = _as_scalar(pos_w[:, 2].mean())
                mean_speed = _as_scalar(speed.mean())
                print(f"[step {step:04d}] mean_z={mean_h:.3f} m, mean_speed={mean_speed:.3f} m/s")

            if bool(torch.any(terminated | truncated)):
                env_ids = torch.nonzero(terminated | truncated, as_tuple=False).squeeze(-1)
                print(f"[info] reset triggered for {int(env_ids.numel())} env(s) at step={step}")

        stats = _collect_motion_stats(env, initial_pos_w, speed_hist)
        print("\n=== Control Chain Smoke Test ===")
        print(f"displacement_m   : {stats.displacement_m:.4f}")
        print(f"max_speed_mps    : {stats.max_speed_mps:.4f}")
        print(f"mean_speed_mps   : {stats.mean_speed_mps:.4f}")
        print(f"max_height_delta : {stats.max_height_delta_m:.4f}")

        moved = (stats.displacement_m >= args_cli.move_threshold_m) or (
            stats.max_speed_mps >= args_cli.speed_threshold_mps
        )
        if moved:
            print("RESULT: PASS (drone motion detected)")
        else:
            print("RESULT: FAIL (no significant motion detected)")
            raise SystemExit(2)

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
