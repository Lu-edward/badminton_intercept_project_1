"""Official IsaacLab RSL-RL training entrypoint for badminton interception."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

import wandb


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train with IsaacLab official RSL-RL runner.")
    parser.add_argument("--task", type=str, default="Isaac-Badminton-Intercept-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--episode_length_s", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, default="badminton_intercept_direct")
    parser.add_argument("--num_steps_per_env", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_mini_batches", type=int, default=None)
    parser.add_argument("--num_learning_epochs", type=int, default=None)
    parser.add_argument("--entropy_coef", type=float, default=None)
    parser.add_argument("--clip_param", type=float, default=None)
    parser.add_argument("--desired_kl", type=float, default=None)
    parser.add_argument("--policy_init_noise_std", type=float, default=None, help="Override init noise std when constructing a fresh policy.")
    parser.add_argument("--reset_action_std", type=float, default=None, help="Reset policy action std after loading a checkpoint.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a prior run, including optimizer and iteration.")
    parser.add_argument("--load_run", type=str, default=None, help="Regex or exact run directory under logs/rsl_rl/<experiment_name>.")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint filename regex, exact filename, or absolute checkpoint path.")
    parser.add_argument("--init_from_checkpoint", type=str, default=None, help="Initialize model weights from a checkpoint for finetuning without loading optimizer state.")
    parser.add_argument("--curriculum_stage_id", type=int, default=None, help="1-based curriculum stage id to start from, e.g. 2 for stage2.")
    parser.add_argument("--init_curriculum_stage", type=int, default=None, help="Deprecated 0-based alias: 0=stage1, 1=stage2, 2=stage3.")
    parser.add_argument("--freeze_curriculum_stage", action="store_true", default=False, help="Keep training fixed at the selected curriculum stage without auto-promotion.")
    parser.add_argument("--enable_post_hit_tracking", action="store_true", default=False, help="Enable post-hit tracking mode for task 2")
    parser.add_argument("--enable_serve_hover", action="store_true", default=False, help="Enable two-stage serve-hover training")
    parser.add_argument("--prehit_checkpoint", type=str, default=None, help="Task2 checkpoint used to initialize the frozen pre-hit actor/critic")
    parser.add_argument("--serve_hover_actor_freeze_iterations", type=int, default=0, help="Freeze serve-hover actor updates for the first N PPO iterations.")
    parser.add_argument("--serve_hover_actor_lr_scale", type=float, default=1.0, help="Scale serve-hover actor LR after the freeze period.")
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="badminton_intercept", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (team/username)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_known_args()


args_cli, _hydra_args = parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

from badminton_intercept.envs.intercept_env_cfg import InterceptEnvCfg
from badminton_intercept.train.register_task import TASK_ID, register_task
from badminton_intercept.train.rsl_rl_ppo_cfg import BadmintonInterceptPPORunnerCfg


def _get_runner_policy(runner: OnPolicyRunner):
    alg = runner.alg
    if hasattr(alg, "policy"):
        return alg.policy
    if hasattr(alg, "actor_critic"):
        return alg.actor_critic
    if hasattr(alg, "actor"):
        return alg.actor
    raise AttributeError(f"Unsupported PPO runner interface: no 'policy', 'actor_critic', or 'actor' on {type(alg).__name__}.")


def _resolve_requested_stage_id(args: argparse.Namespace) -> int | None:
    if args.curriculum_stage_id is not None and args.init_curriculum_stage is not None:
        raise ValueError("Use either --curriculum_stage_id or --init_curriculum_stage, not both.")

    if args.curriculum_stage_id is not None:
        if args.curriculum_stage_id < 1:
            raise ValueError("--curriculum_stage_id must be >= 1.")
        return int(args.curriculum_stage_id)

    if args.init_curriculum_stage is not None:
        if args.init_curriculum_stage < 0:
            raise ValueError("--init_curriculum_stage must be >= 0.")
        print("[WARN] --init_curriculum_stage is deprecated; please switch to --curriculum_stage_id.")
        return int(args.init_curriculum_stage) + 1

    return None


def _resolve_checkpoint_path(log_root: str, load_run: str | None, checkpoint: str | None) -> str:
    if checkpoint is None:
        raise ValueError("Checkpoint path is required but no checkpoint was specified.")

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_file():
        return str(checkpoint_path.resolve())
    if checkpoint_path.is_absolute():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    log_root_path = Path(log_root)
    if not log_root_path.exists():
        raise FileNotFoundError(f"Log root does not exist: {log_root_path}")

    run_pattern = load_run or ".*"
    candidate_runs = [path for path in log_root_path.iterdir() if path.is_dir() and re.match(run_pattern, path.name)]
    candidate_runs.sort()
    if not candidate_runs:
        raise FileNotFoundError(f"No run under {log_root_path} matches pattern '{run_pattern}'.")

    run_path = candidate_runs[-1]
    checkpoint_matches = [path for path in run_path.iterdir() if path.is_file() and re.match(checkpoint, path.name)]
    checkpoint_matches.sort(key=lambda path: f"{path.name:0>15}")
    if not checkpoint_matches:
        raise FileNotFoundError(f"No checkpoint in {run_path} matches pattern '{checkpoint}'.")
    return str(checkpoint_matches[-1].resolve())


def _extract_model_state_dict(checkpoint_payload: dict[str, Any]) -> dict[str, Any]:
    if "model_state_dict" in checkpoint_payload:
        return checkpoint_payload["model_state_dict"]
    if "state_dict" in checkpoint_payload and isinstance(checkpoint_payload["state_dict"], dict):
        return checkpoint_payload["state_dict"]

    keys = list(checkpoint_payload.keys())
    if keys and all(
        key.startswith(("actor.", "critic.", "log_std", "std", "actor_obs_normalizer", "critic_obs_normalizer"))
        for key in keys
    ):
        return checkpoint_payload

    if "actor_state_dict" in checkpoint_payload and "critic_state_dict" in checkpoint_payload:
        model_state_dict: dict[str, Any] = {}
        
        actor_state = checkpoint_payload["actor_state_dict"]
        critic_state = checkpoint_payload["critic_state_dict"]
        
        actor_keys = list(actor_state.keys())
        if actor_keys and any(k.startswith("mlp.") for k in actor_keys):
            for key, value in actor_state.items():
                if key.startswith("mlp."):
                    model_state_dict[key] = value
                elif key.startswith("obs_normalizer."):
                    model_state_dict[key] = value
                elif key == "distribution.log_std_param":
                    model_state_dict["log_std"] = value
                else:
                    model_state_dict[key] = value
        else:
            for key, value in actor_state.items():
                model_state_dict[key if key.startswith("actor.") else f"actor.{key}"] = value
        
        critic_keys = list(critic_state.keys())
        if critic_keys and any(k.startswith("mlp.") for k in critic_keys):
            for key, value in critic_state.items():
                if key.startswith("mlp."):
                    model_state_dict[key] = value
                elif key.startswith("obs_normalizer."):
                    model_state_dict[key] = value
                else:
                    model_state_dict[key] = value
        else:
            for key, value in critic_state.items():
                model_state_dict[key if key.startswith("critic.") else f"critic.{key}"] = value
        
        if "actor_obs_normalizer_state_dict" in checkpoint_payload:
            for key, value in checkpoint_payload["actor_obs_normalizer_state_dict"].items():
                model_state_dict[f"actor_obs_normalizer.{key}"] = value
        if "critic_obs_normalizer_state_dict" in checkpoint_payload:
            for key, value in checkpoint_payload["critic_obs_normalizer_state_dict"].items():
                model_state_dict[f"critic_obs_normalizer.{key}"] = value
        if "log_std" in checkpoint_payload:
            model_state_dict["log_std"] = checkpoint_payload["log_std"]
        if "std" in checkpoint_payload:
            model_state_dict["std"] = checkpoint_payload["std"]
        return model_state_dict

    raise RuntimeError(
        "Unsupported checkpoint format. Expected 'model_state_dict', 'state_dict', or split actor/critic state dicts."
    )


def _set_policy_action_std(policy, target_std: float) -> None:
    if target_std <= 0.0:
        raise ValueError("--reset_action_std must be > 0.")

    with torch.no_grad():
        if hasattr(policy, "set_action_std"):
            policy.set_action_std(target_std)
            return
        # Try direct attributes first (older rsl-rl versions)
        if hasattr(policy, "log_std"):
            policy.log_std.fill_(math.log(target_std))
            return
        if hasattr(policy, "std"):
            policy.std.fill_(target_std)
            return
        # Try distribution attribute (rsl-rl 5.0+)
        if hasattr(policy, "distribution"):
            dist = policy.distribution
            if hasattr(dist, "std"):
                dist.std.fill_(target_std)
                return
            if hasattr(dist, "log_std_param"):
                dist.log_std_param.fill_(math.log(target_std))
                return
        # Try output_std (rsl-rl 5.0+ alternative)
        if hasattr(policy, "output_std"):
            policy.output_std.fill_(target_std)
            return
        raise RuntimeError("Policy does not expose a supported action std parameter.")


def _get_policy_action_std_mean(policy) -> float:
    if hasattr(policy, "get_action_std_mean"):
        return float(policy.get_action_std_mean())
    # Try direct attributes first (older rsl-rl versions)
    if hasattr(policy, "log_std"):
        return float(torch.exp(policy.log_std).mean().item())
    if hasattr(policy, "std"):
        return float(policy.std.mean().item())
    # Try distribution attribute (rsl-rl 5.0+)
    if hasattr(policy, "distribution"):
        dist = policy.distribution
        if hasattr(dist, "std"):
            return float(dist.std.mean().item())
        if hasattr(dist, "log_std_param"):
            return float(torch.exp(dist.log_std_param).mean().item())
    # Try output_std (rsl-rl 5.0+ alternative)
    if hasattr(policy, "output_std"):
        return float(policy.output_std.mean().item())
    raise RuntimeError("Policy does not expose a supported action std parameter.")


def _load_finetune_checkpoint(runner: OnPolicyRunner, checkpoint_path: str) -> None:
    """Load checkpoint for finetuning - loads actor and critic weights without optimizer/iteration."""
    checkpoint_payload = torch.load(checkpoint_path, weights_only=False, map_location=runner.device)
    
    # rsl-rl 5.0+ 使用 alg.actor 和 alg.critic 分别加载
    if hasattr(runner, 'alg') and hasattr(runner.alg, 'actor') and hasattr(runner.alg, 'critic'):
        if "actor_state_dict" in checkpoint_payload:
            runner.alg.actor.load_state_dict(checkpoint_payload["actor_state_dict"], strict=True)
            print(f"[INFO] Loaded actor weights from: {checkpoint_path}")
        if "critic_state_dict" in checkpoint_payload:
            runner.alg.critic.load_state_dict(checkpoint_payload["critic_state_dict"], strict=True)
            print(f"[INFO] Loaded critic weights from: {checkpoint_path}")
    else:
        # 回退到旧版本的方式
        model_state_dict = _extract_model_state_dict(checkpoint_payload)
        _get_runner_policy(runner).load_state_dict(model_state_dict, strict=True)


def _patch_runner_log_for_wandb(runner: OnPolicyRunner) -> None:
    # rsl-rl 5.0+ uses logger.log instead of runner.log
    original_log = None
    if hasattr(runner, "log"):
        original_log = runner.log
    elif hasattr(runner, "logger") and hasattr(runner.logger, "log"):
        original_log = runner.logger.log
    else:
        print("[WARN] Could not find log method for wandb patching. Skipping wandb log.")
        return

    def _wandb_log(**locs) -> None:
        # Call original log with same kwargs
        original_log(**locs)
        policy = _get_runner_policy(runner)

        # Try different std attributes (rsl-rl 5.0+ uses output_std or distribution.std)
        mean_std = 0.0
        if hasattr(policy, "output_std"):
            mean_std = float(policy.output_std.mean().item())
        elif hasattr(policy, "distribution") and hasattr(policy.distribution, "std"):
            mean_std = float(policy.distribution.std.mean().item())
        elif hasattr(policy, "action_std"):
            mean_std = float(policy.action_std.mean().item())

        wandb_metrics: dict[str, float] = {
            "iteration": float(locs.get("it", 0)),
            "collect_time": float(locs.get("collect_time", 0)),
            "learn_time": float(locs.get("learn_time", 0)),
            "learning_rate": float(locs.get("learning_rate", 0)),
            "Policy/mean_noise_std": mean_std,
        }
        loss_dict = locs.get("loss_dict", {})
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                wandb_metrics[f"Loss/{key}"] = float(value)

        if locs.get("mean_reward") is not None:
            wandb_metrics["Train/mean_reward"] = float(locs["mean_reward"])
        if locs.get("mean_episode_length") is not None:
            wandb_metrics["Train/mean_episode_length"] = float(locs["mean_episode_length"])

        if locs.get("extras"):
            extras = locs["extras"]
            if isinstance(extras, dict):
                for key in (
                    "success_rate",
                    "success_contact",
                    "post_hit_rate",
                    "wrong_hit_rate",
                    "num_resets",
                    "curriculum_stage_id",
                    "failure_server_side_grounded",
                    "failure_server_side_grounded_rate",
                    "failure_net_contact_rate",
                    "failure_ball_drop_rate",
                    "failure_out_of_bounds_rate",
                    "failure_tilt_rate",
                    "timeout_rate",
                ):
                    if key in extras:
                        value = extras[key]
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        wandb_metrics[f"Episode/{key}"] = float(value)

        wandb.log(wandb_metrics, step=int(locs.get("it", 0)))

    # Patch the correct location
    if hasattr(runner, "log"):
        runner.log = _wandb_log
    elif hasattr(runner.logger, "log"):
        runner.logger.log = _wandb_log


def _patch_runner_log_for_curriculum(runner: OnPolicyRunner, curriculum, freeze_stage: bool = False) -> None:
    """Patch the runner's logger to trigger curriculum promotion after each iteration.

    The success_rate is extracted from runner.logger.ep_extras buffer, which is populated
    by Logger.process_env_step() during environment rollouts. The log() method itself
    does not receive extras/ep_infos as parameters.

    IMPORTANT: We must extract ep_extras BEFORE calling original_log(), because
    Logger.log() clears ep_extras at the end of its execution.
    """
    if curriculum is None or freeze_stage:
        return

    # Get the logger object (not the log method)
    logger_obj = getattr(runner, "logger", None)
    if logger_obj is None:
        print("[WARN] Runner has no logger. Iteration-based curriculum promotion disabled.")
        return

    original_log = None
    patch_target = None
    if hasattr(runner, "log"):
        original_log = runner.log
        patch_target = "runner"
    elif hasattr(logger_obj, "log"):
        original_log = logger_obj.log
        patch_target = "logger"
    else:
        print("[WARN] Could not find log method for curriculum patching. Iteration-based curriculum promotion disabled.")
        return

    def _extract_iteration_success_rate_from_ep_extras(ep_extras: list) -> float | None:
        """Extract mean success_rate from ep_extras buffer.

        The ep_extras buffer contains episode-level statistics collected during
        the rollout via Logger.process_env_step(). Each entry is a dict with
        keys like 'success_rate', 'curriculum_stage_id', etc.
        """
        if not ep_extras or not isinstance(ep_extras, list):
            return None

        values: list[float] = []
        for info in ep_extras:
            if not isinstance(info, dict) or "success_rate" not in info:
                continue
            value = info["success_rate"]
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    continue
                values.extend(float(v) for v in value.detach().flatten().cpu().tolist())
            else:
                values.append(float(value))

        if values:
            return float(sum(values) / len(values))
        return None

    def _curriculum_log(*args, **kwargs):
        # CRITICAL: Extract ep_extras BEFORE calling original_log(),
        # because Logger.log() clears ep_extras at the end!
        ep_extras_snapshot = list(getattr(logger_obj, "ep_extras", []))
        iteration_success_rate = _extract_iteration_success_rate_from_ep_extras(ep_extras_snapshot)

        # Now call the original log method (which will clear ep_extras)
        result = original_log(*args, **kwargs)

        # Update curriculum with the pre-extracted success rate
        if iteration_success_rate is not None:
            curriculum.update_iteration(iteration_success_rate)
        return result

    if patch_target == "runner":
        runner.log = _curriculum_log
    else:
        logger_obj.log = _curriculum_log


def _patch_runner_log_for_task2_product_json(
    runner: OnPolicyRunner,
    log_dir: str,
    threshold: float = 0.85,
) -> None:
    """Track task2 metrics and append iteration info to JSON when product improves.

    Rule:
    1) Start tracking only after both metrics are > threshold.
    2) After that, whenever metric_a * metric_b exceeds previous best product,
       append this step information to a JSON file.
    """
    logger_obj = getattr(runner, "logger", None)
    if logger_obj is None:
        print("[WARN] Runner has no logger. Task2 product JSON tracking disabled.")
        return

    original_log = None
    patch_target = None
    if hasattr(runner, "log"):
        original_log = runner.log
        patch_target = "runner"
    elif hasattr(logger_obj, "log"):
        original_log = logger_obj.log
        patch_target = "logger"
    else:
        print("[WARN] Could not find log method for task2 product JSON tracking. Disabled.")
        return

    metric_a_key = "post_hit_server_half_grounded_rate"
    metric_b_key = "pre_hit_success_hit_rate"
    json_path = os.path.join(log_dir, "task2_product_improve_steps.json")
    best_product = float("-inf")
    threshold_armed = False
    saved_steps: list[dict[str, float | int]] = []

    def _extract_metric_mean(ep_extras: list, key: str) -> float | None:
        if not ep_extras:
            return None
        values: list[float] = []
        for info in ep_extras:
            if not isinstance(info, dict) or key not in info:
                continue
            value = info[key]
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    continue
                values.extend(float(v) for v in value.detach().flatten().cpu().tolist())
            else:
                values.append(float(value))
        if not values:
            return None
        return float(sum(values) / len(values))

    def _task2_product_log(*args, **kwargs):
        nonlocal best_product, threshold_armed, saved_steps

        # Must snapshot before original log clears ep_extras.
        ep_extras_snapshot = list(getattr(logger_obj, "ep_extras", []))
        metric_a = _extract_metric_mean(ep_extras_snapshot, metric_a_key)
        metric_b = _extract_metric_mean(ep_extras_snapshot, metric_b_key)
        iteration = kwargs.get("it", getattr(runner, "current_learning_iteration", 0))

        result = original_log(*args, **kwargs)

        if metric_a is None or metric_b is None:
            return result

        if (metric_a > threshold) and (metric_b > threshold):
            threshold_armed = True

        if not threshold_armed:
            return result

        product = metric_a * metric_b
        if product > best_product:
            best_product = product
            step_info: dict[str, float | int] = {
                "iteration": int(iteration),
                metric_a_key: float(metric_a),
                metric_b_key: float(metric_b),
                "product": float(product),
            }
            saved_steps.append(step_info)
            with open(json_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "threshold": float(threshold),
                        "metric_a": metric_a_key,
                        "metric_b": metric_b_key,
                        "best_product": float(best_product),
                        "saved_steps": saved_steps,
                    },
                    file,
                    indent=2,
                )
            print(
                "[task2-json] saved improved step: "
                f"iter={int(iteration)}, {metric_a_key}={metric_a:.4f}, "
                f"{metric_b_key}={metric_b:.4f}, product={product:.6f}"
            )
        return result

    if patch_target == "runner":
        runner.log = _task2_product_log
    else:
        logger_obj.log = _task2_product_log


def _save_curriculum_summary(runner: OnPolicyRunner, log_dir: str) -> None:
    logger = getattr(runner, "logger", None)
    if logger is None or not hasattr(logger, "rewbuffer") or len(logger.rewbuffer) == 0:
        print("[summary] Skipped last-10 summary because this rsl-rl runner does not expose episode buffers.")
        return

    last_n = 10
    rew_list = list(logger.rewbuffer)
    len_list = list(logger.lenbuffer)
    last_rewards = rew_list[-last_n:] if len(rew_list) >= last_n else rew_list
    last_lengths = len_list[-last_n:] if len(len_list) >= last_n else len_list

    ep_extras = logger.ep_extras if hasattr(logger, "ep_extras") else []
    last_extras = ep_extras[-last_n:] if len(ep_extras) >= last_n else ep_extras
    success_rates = [e.get("success_rate", 0) for e in last_extras if isinstance(e, dict)]
    curriculum_stages = [e.get("curriculum_stage_id", 0) for e in last_extras if isinstance(e, dict)]
    success_contacts = [e.get("success_contact", 0) for e in last_extras if isinstance(e, dict)]
    num_resets_list = [e.get("num_resets", 0) for e in last_extras if isinstance(e, dict)]

    summary = {
        "last_10_mean_reward": statistics.mean(last_rewards) if last_rewards else 0.0,
        "last_10_mean_episode_length": statistics.mean(last_lengths) if last_lengths else 0.0,
        "last_10_rewards": last_rewards,
        "last_10_episode_lengths": last_lengths,
        "last_10_success_rates": success_rates if success_rates else [0.0] * min(last_n, len(ep_extras)),
        "last_10_mean_success_rate": statistics.mean(success_rates) if success_rates else 0.0,
        "last_10_curriculum_stages": curriculum_stages if curriculum_stages else [1] * min(last_n, len(ep_extras)),
        "last_10_success_contacts": success_contacts if success_contacts else [0] * min(last_n, len(ep_extras)),
        "last_10_num_resets": num_resets_list if num_resets_list else [0] * min(last_n, len(ep_extras)),
        "total_iterations": runner.current_learning_iteration,
    }

    summary_path = os.path.join(log_dir, "last_10_iterations_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=float)
    print(f"[summary] Last 10 iterations saved to {summary_path}")
    print(f"[summary] Mean reward: {summary['last_10_mean_reward']:.4f}")
    print(f"[summary] Mean episode length: {summary['last_10_mean_episode_length']:.2f}")
    print(f"[summary] Mean success rate: {summary['last_10_mean_success_rate']:.4f}")
    if curriculum_stages:
        print(f"[summary] Curriculum stage: {statistics.mode(curriculum_stages)}")


def main() -> None:
    register_task()

    requested_stage_id = _resolve_requested_stage_id(args_cli)
    checkpoint_mode = "fresh"
    if args_cli.resume and args_cli.init_from_checkpoint is not None:
        raise ValueError("Use either --resume or --init_from_checkpoint, not both.")
    if args_cli.resume:
        checkpoint_mode = "resume"
    elif args_cli.init_from_checkpoint is not None:
        checkpoint_mode = "finetune"

    env_cfg = InterceptEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    elif args_cli.enable_serve_hover:
        env_cfg.episode_length_s = 10.0
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if requested_stage_id is not None:
        env_cfg.curriculum_initial_stage_id = requested_stage_id
    env_cfg.curriculum_fixed_stage = bool(args_cli.freeze_curriculum_stage)

    if args_cli.enable_post_hit_tracking:
        env_cfg.enable_post_hit_tracking = True
        env_cfg.curriculum_initial_stage_id = requested_stage_id if requested_stage_id is not None else 1
        env_cfg.curriculum_fixed_stage = True

    if args_cli.enable_serve_hover:
        env_cfg.enable_serve_hover = True
        env_cfg.enable_post_hit_tracking = True
        env_cfg.curriculum_fixed_stage = True

    runner_cfg = BadmintonInterceptPPORunnerCfg()
    runner_cfg.max_iterations = args_cli.max_iterations
    runner_cfg.seed = args_cli.seed
    runner_cfg.experiment_name = args_cli.experiment_name
    if args_cli.device is not None:
        runner_cfg.device = args_cli.device
    if args_cli.num_steps_per_env is not None:
        runner_cfg.num_steps_per_env = args_cli.num_steps_per_env
    if args_cli.save_interval is not None:
        runner_cfg.save_interval = args_cli.save_interval
    if args_cli.policy_init_noise_std is not None:
        runner_cfg.policy.init_noise_std = args_cli.policy_init_noise_std
    configured_policy_init_noise_std = float(runner_cfg.policy.init_noise_std)

    if args_cli.learning_rate is not None:
        runner_cfg.algorithm.learning_rate = args_cli.learning_rate
    if args_cli.num_mini_batches is not None:
        runner_cfg.algorithm.num_mini_batches = args_cli.num_mini_batches
    if args_cli.num_learning_epochs is not None:
        runner_cfg.algorithm.num_learning_epochs = args_cli.num_learning_epochs
    if args_cli.entropy_coef is not None:
        runner_cfg.algorithm.entropy_coef = args_cli.entropy_coef
    if args_cli.clip_param is not None:
        runner_cfg.algorithm.clip_param = args_cli.clip_param
    if args_cli.desired_kl is not None:
        runner_cfg.algorithm.desired_kl = args_cli.desired_kl

    if args_cli.enable_post_hit_tracking:
        if args_cli.policy_init_noise_std is None:
            runner_cfg.policy.init_noise_std = 0.3
        if args_cli.learning_rate is None:
            runner_cfg.algorithm.learning_rate = 1e-4

    if args_cli.enable_serve_hover:
        runner_cfg.enable_serve_hover = True
        runner_cfg.prehit_checkpoint_path = args_cli.prehit_checkpoint or ""
        runner_cfg.serve_hover_actor_freeze_iterations = args_cli.serve_hover_actor_freeze_iterations
        runner_cfg.serve_hover_actor_lr_scale = args_cli.serve_hover_actor_lr_scale
        if args_cli.learning_rate is None:
            runner_cfg.algorithm.learning_rate = 1e-4

    if checkpoint_mode == "resume":
        runner_cfg.resume = True
    if args_cli.load_run is not None:
        runner_cfg.load_run = args_cli.load_run
    if args_cli.load_checkpoint is not None:
        runner_cfg.load_checkpoint = args_cli.load_checkpoint

    if args_cli.wandb:
        wandb.init(
            project=args_cli.wandb_project,
            entity=args_cli.wandb_entity,
            name=args_cli.wandb_run_name or runner_cfg.experiment_name,
            config={
                "task": args_cli.task,
                "num_envs": args_cli.num_envs,
                "max_iterations": args_cli.max_iterations,
                "episode_length_s": args_cli.episode_length_s,
                "seed": args_cli.seed,
                "num_steps_per_env": runner_cfg.num_steps_per_env,
                "learning_rate": runner_cfg.algorithm.learning_rate,
                "num_mini_batches": runner_cfg.algorithm.num_mini_batches,
                "num_learning_epochs": runner_cfg.algorithm.num_learning_epochs,
                "entropy_coef": runner_cfg.algorithm.entropy_coef,
                "clip_param": runner_cfg.algorithm.clip_param,
                "desired_kl": runner_cfg.algorithm.desired_kl,
                "policy_init_noise_std": configured_policy_init_noise_std,
                "reset_action_std": args_cli.reset_action_std,
                "checkpoint_mode": checkpoint_mode,
                "curriculum_stage_id": requested_stage_id,
                "freeze_curriculum_stage": args_cli.freeze_curriculum_stage,
                "enable_serve_hover": args_cli.enable_serve_hover,
                "prehit_checkpoint": args_cli.prehit_checkpoint,
                "serve_hover_actor_freeze_iterations": args_cli.serve_hover_actor_freeze_iterations,
                "serve_hover_actor_lr_scale": args_cli.serve_hover_actor_lr_scale,
                "curriculum_promote_success_rate": env_cfg.curriculum_promote_success_rate,
                "curriculum_promote_iteration_streak": env_cfg.curriculum_promote_iteration_streak,
            },
        )
        print(f"[wandb] Logging to project: {args_cli.wandb_project}")

    task_id = args_cli.task or TASK_ID
    env = gym.make(task_id, cfg=env_cfg)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", runner_cfg.experiment_name))
    os.makedirs(log_root, exist_ok=True)
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root, run_dir)
    os.makedirs(log_dir, exist_ok=True)

    resume_path = None
    if checkpoint_mode == "resume":
        resume_path = _resolve_checkpoint_path(log_root, args_cli.load_run, args_cli.load_checkpoint)
    elif checkpoint_mode == "finetune":
        if args_cli.init_from_checkpoint is not None:
            resume_path = _resolve_checkpoint_path(log_root, None, args_cli.init_from_checkpoint)
        else:
            resume_path = _resolve_checkpoint_path(log_root, args_cli.load_run, args_cli.load_checkpoint)

    if args_cli.enable_serve_hover and checkpoint_mode == "fresh" and not args_cli.prehit_checkpoint:
        raise ValueError("--prehit_checkpoint is required when starting fresh serve-hover training.")

    env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    # 处理rsl-rl版本兼容性 - 必须在to_dict()之前调用，因为函数需要操作配置对象的属性
    handle_deprecated_rsl_rl_cfg(runner_cfg, "5.0.1")
    runner_cfg_dict = runner_cfg.to_dict()
    runner_cls = OnPolicyRunner
    if args_cli.enable_serve_hover:
        from badminton_intercept.train.serve_hover_runner import ServeHoverOnPolicyRunner

        runner_cls = ServeHoverOnPolicyRunner
    runner = runner_cls(env, runner_cfg_dict, log_dir=log_dir, device=runner_cfg.device)

    if checkpoint_mode == "resume" and resume_path is not None:
        print(f"[INFO] Resuming runner from checkpoint: {resume_path}")
        runner.load(resume_path)
    elif checkpoint_mode == "finetune" and resume_path is not None:
        print(f"[INFO] Initializing policy weights from checkpoint: {resume_path}")
        if args_cli.enable_serve_hover:
            runner.load(resume_path, load_optimizer=False)
        else:
            _load_finetune_checkpoint(runner, resume_path)

    if args_cli.reset_action_std is not None:
        _set_policy_action_std(_get_runner_policy(runner), args_cli.reset_action_std)
        print(f"[INFO] Reset action std to {args_cli.reset_action_std:.4f}")

    curriculum_log_path = os.path.join(log_dir, "curriculum_log.json")
    curriculum_events = []

    def curriculum_logger(msg: str) -> None:
        print(msg)
        if "Manually set" in msg:
            match = re.search(r"stage (\d+): (\w+)", msg)
            if match:
                curriculum_events.append(
                    {
                        "iteration": runner.current_learning_iteration,
                        "event": "manual_set",
                        "stage_id": int(match.group(1)),
                        "stage_name": match.group(2),
                        "message": msg,
                    }
                )
        elif "Promoted from" in msg:
            match = re.search(r"from stage (\d+) to stage (\d+): (\w+)", msg)
            if match:
                curriculum_events.append(
                    {
                        "iteration": runner.current_learning_iteration,
                        "event": "promotion",
                        "from_stage": int(match.group(1)),
                        "to_stage": int(match.group(2)),
                        "stage_name": match.group(3),
                        "message": msg,
                    }
                )
        with open(curriculum_log_path, "w", encoding="utf-8") as file:
            json.dump(curriculum_events, file, indent=2)

    curriculum = getattr(env.unwrapped, "_curriculum", None)
    if curriculum is not None:
        curriculum.set_logger(curriculum_logger)
        if requested_stage_id is not None:
            curriculum.set_stage_by_id(requested_stage_id)
            print(f"[INFO] Set initial curriculum stage to {requested_stage_id} ({curriculum.current_stage.name})")
        if args_cli.freeze_curriculum_stage or args_cli.enable_post_hit_tracking:
            print(f"[INFO] Curriculum auto-promotion disabled at stage {curriculum.current_stage.stage_id}.")
    else:
        print("[WARN] Curriculum not found in env.")

    if args_cli.wandb:
        _patch_runner_log_for_wandb(runner)
    freeze_curriculum = bool(
        args_cli.freeze_curriculum_stage or args_cli.enable_post_hit_tracking or args_cli.enable_serve_hover
    )
    _patch_runner_log_for_curriculum(runner, curriculum, freeze_stage=freeze_curriculum)
    if args_cli.enable_post_hit_tracking:
        _patch_runner_log_for_task2_product_json(runner, log_dir=log_dir, threshold=0.85)

    if checkpoint_mode == "finetune" and args_cli.reset_action_std is None:
        current_std = _get_policy_action_std_mean(_get_runner_policy(runner))
        print(
            f"[INFO] Finetune initialized from checkpoint with current action std {current_std:.4f}. "
            "If stage2/stage3 feels too noisy, try --reset_action_std 0.20~0.35."
        )

    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()
    _save_curriculum_summary(runner, log_dir)

    if args_cli.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
    simulation_app.close()
