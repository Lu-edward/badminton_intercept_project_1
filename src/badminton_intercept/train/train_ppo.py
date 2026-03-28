import argparse
import json
from pathlib import Path
from typing import Any

from badminton_intercept.envs.intercept_env import InterceptEnv
from badminton_intercept.envs.intercept_env_cfg import DEFAULT_ENV_CFG
from badminton_intercept.train.callbacks import TrainCallbacks


def _load_yaml_with_fallback(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default

    try:
        import yaml  # type: ignore
    except Exception:
        print(f"[warn] PyYAML not available; using defaults for {path.name}")
        return default

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        return default
    return loaded


def _unpack_reset(reset_out: Any) -> Any:
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out[0]
    return reset_out


def _to_scalar(x: Any) -> float:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return float(x.float().mean().item())
    except Exception:
        pass
    if isinstance(x, (int, float)):
        return float(x)
    return 0.0


def _sample_zero_actions(obs: Any) -> Any:
    try:
        import torch

        if isinstance(obs, dict) and "policy" in obs and isinstance(obs["policy"], torch.Tensor):
            n = obs["policy"].shape[0]
            return torch.zeros((n, 4), device=obs["policy"].device)
    except Exception:
        pass
    return [0.0, 0.0, 0.0, 0.0]


def _extract_step(step_out: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    # IsaacLab DirectRLEnv: (obs, reward, terminated, time_out, extras)
    if isinstance(step_out, tuple) and len(step_out) == 5:
        return step_out

    # Fallback skeleton dataclass
    if hasattr(step_out, "observations"):
        obs = step_out.observations
        rew = step_out.rewards
        done = step_out.dones
        terminated = done.get("terminated", False) if isinstance(done, dict) else False
        time_out = done.get("truncated", False) if isinstance(done, dict) else False
        extras = step_out.infos if hasattr(step_out, "infos") else {}
        return obs, rew, terminated, time_out, extras

    return {}, 0.0, False, False, {}


def _collect_rollout(env: InterceptEnv, obs: Any, rollout_steps: int) -> tuple[Any, dict[str, Any]]:
    transitions = []
    reward_acc = 0.0
    done_acc = 0.0

    for _ in range(rollout_steps):
        action = _sample_zero_actions(obs)
        next_obs, reward, terminated, time_out, extras = _extract_step(env.step(action))
        transitions.append((obs, action, reward, next_obs, terminated, time_out, extras))
        reward_acc += _to_scalar(reward)
        done_acc += _to_scalar(terminated)
        obs = next_obs

    stats = {
        "mean_reward": reward_acc / max(1, rollout_steps),
        "mean_done": done_acc / max(1, rollout_steps),
        "transitions": len(transitions),
    }
    return obs, stats


def run_train(project_root: Path, smoke_test: bool) -> None:
    ppo_cfg = _load_yaml_with_fallback(
        project_root / "configs" / "train" / "ppo.yaml",
        {
            "total_steps": 2_000_000_000,
            "num_envs": 4096,
            "rollout_steps": 32,
            "update_epochs": 5,
            "num_minibatches": 32,
        },
    )
    env = InterceptEnv(cfg=DEFAULT_ENV_CFG, render_mode=None)
    callbacks = TrainCallbacks()

    callbacks.on_train_start()
    obs = _unpack_reset(env.reset())
    obs_keys = list(obs.keys()) if isinstance(obs, dict) else [type(obs).__name__]

    print(f"[train] reset observation keys: {obs_keys}")
    print(f"[train] loaded ppo config: {json.dumps(ppo_cfg, ensure_ascii=True)}")

    # Skeleton PPO loop structure:
    # 1) parallel rollout collection
    # 2) GAE/value targets (TODO)
    # 3) clipped actor update + critic MSE update (TODO)
    outer_iters = 2 if smoke_test else 5
    rollout_steps = int(ppo_cfg.get("rollout_steps", 32))

    for i in range(outer_iters):
        obs, rollout_stats = _collect_rollout(env, obs, rollout_steps)

        # TODO: Compute GAE and returns from collected rollout.
        # TODO: Optimize actor with PPO clipped objective and critic with value loss.

        metrics = {
            "rollout_mean_reward": rollout_stats["mean_reward"],
            "rollout_mean_done": rollout_stats["mean_done"],
            "rollout_transitions": rollout_stats["transitions"],
        }

        curriculum = getattr(env, "_curriculum", None)
        if curriculum is not None:
            metrics["curr_stage"] = curriculum.current_stage.stage_id
            metrics["curr_success_rate"] = curriculum.success_rate

        callbacks.on_iteration_end(i, metrics)

    callbacks.on_train_end(project_root / "outputs")
    print("[train] PPO pipeline skeleton run complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO skeleton for interception.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    run_train(project_root=args.project_root, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
