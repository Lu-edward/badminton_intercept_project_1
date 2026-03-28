"""Evaluate a trained policy in headless mode with fixed curriculum stage."""

from __future__ import annotations

import argparse
import os
import sys
import time
import torch
import numpy as np
from collections import defaultdict

from isaaclab.app import AppLauncher


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Evaluate trained policy in headless mode.")
    parser.add_argument("--task", type=str, default="Isaac-Badminton-Intercept-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=3000, help="Total number of episodes to evaluate")
    parser.add_argument("--curriculum_stage", type=int, default=3, help="Fixed curriculum stage (1, 2, or 3)")
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    
    # Force headless mode for evaluation
    args.headless = True
    
    return args, hydra_args


args_cli, _hydra_args = parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Add project root to path before importing badminton_intercept
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from badminton_intercept.envs.intercept_env_cfg import InterceptEnvCfg
from badminton_intercept.train.register_task import TASK_ID, register_task
from badminton_intercept.train.rsl_rl_ppo_cfg import BadmintonInterceptPPORunnerCfg


def build_combined_model_state_from_split_checkpoint(checkpoint: dict[str, object]) -> dict[str, torch.Tensor]:
    model_state: dict[str, torch.Tensor] = {}

    for key, value in checkpoint["actor_state_dict"].items():
        if key == "obs_normalizer._mean":
            model_state["actor_obs_normalizer._mean"] = value
        elif key == "obs_normalizer._var":
            model_state["actor_obs_normalizer._var"] = value
        elif key == "obs_normalizer._std":
            model_state["actor_obs_normalizer._std"] = value
        elif key == "obs_normalizer.count":
            model_state["actor_obs_normalizer.count"] = value
        elif key == "distribution.log_std_param":
            model_state["log_std"] = value
        elif key.startswith("mlp."):
            model_state["actor." + key[4:]] = value
        else:
            model_state[key] = value

    for key, value in checkpoint["critic_state_dict"].items():
        if key == "obs_normalizer._mean":
            model_state["critic_obs_normalizer._mean"] = value
        elif key == "obs_normalizer._var":
            model_state["critic_obs_normalizer._var"] = value
        elif key == "obs_normalizer._std":
            model_state["critic_obs_normalizer._std"] = value
        elif key == "obs_normalizer.count":
            model_state["critic_obs_normalizer.count"] = value
        elif key.startswith("mlp."):
            model_state["critic." + key[4:]] = value
        else:
            model_state[key] = value

    return model_state


def infer_checkpoint_obs_dims(checkpoint_path: str) -> tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "actor_state_dict" in checkpoint and "critic_state_dict" in checkpoint:
        state_dict = build_combined_model_state_from_split_checkpoint(checkpoint)
    else:
        state_dict = checkpoint
    try:
        actor_dim = int(state_dict["actor.0.weight"].shape[1])
        critic_dim = int(state_dict["critic.0.weight"].shape[1])
    except Exception as exc:
        raise RuntimeError(
            "Could not infer actor/critic observation dimensions from checkpoint. "
            "Expected keys like 'actor.0.weight' and 'critic.0.weight'."
        ) from exc
    return actor_dim, critic_dim


class CheckpointObsAdapter:
    """Adapt environment observations to match checkpoint dimensions."""

    def __init__(self, env, actor_dim: int, critic_dim: int):
        self.env = env
        self.actor_dim = actor_dim
        self.critic_dim = critic_dim
        self.num_envs = env.num_envs
        self.device = env.device

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        obs = self._adapt_obs(obs)
        return obs, rewards, dones, infos

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        obs = self._adapt_obs(obs)
        return obs, info

    def _adapt_obs(self, obs: TensorDict) -> TensorDict:
        adapted = {}
        for key, value in obs.items():
            if key == "policy":
                adapted[key] = self._resize_last_dim(value, self.actor_dim)
            elif key == "critic":
                adapted[key] = self._resize_last_dim(value, self.critic_dim)
            else:
                adapted[key] = value
        return TensorDict(adapted, batch_size=obs.batch_size, device=obs.device)

    @staticmethod
    def _resize_last_dim(value: torch.Tensor, target_dim: int) -> torch.Tensor:
        current_dim = int(value.shape[-1])
        if current_dim == target_dim:
            return value
        if current_dim > target_dim:
            return value[..., :target_dim]
        pad_shape = (*value.shape[:-1], target_dim - current_dim)
        padding = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
        return torch.cat([value, padding], dim=-1)


def main() -> None:
    register_task()

    # Create environment configuration
    env_cfg = InterceptEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.termination_debug_print = False  # Disable debug print for cleaner output
    env_cfg.randomization_enabled = False  # Deterministic evaluation
    env_cfg.uav_init_lin_vel_range = (0.0, 0.0)
    
    # Fixed curriculum stage settings
    env_cfg.curriculum_fixed_stage = True
    env_cfg.curriculum_initial_stage_id = args_cli.curriculum_stage
    


    print(f"[INFO] Creating environment with {args_cli.num_envs} parallel envs...")
    print(f"[INFO] Fixed curriculum stage: {args_cli.curriculum_stage}")
    print(f"[INFO] Target episodes: {args_cli.num_episodes}")

    # Create environment
    task_id = args_cli.task or TASK_ID
    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    # Load the trained policy
    print(f"[INFO] Loading checkpoint from: {args_cli.checkpoint}")
    
    if not os.path.exists(args_cli.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args_cli.checkpoint}")

    checkpoint = torch.load(args_cli.checkpoint, weights_only=False, map_location=env.device)
    ckpt_actor_dim, ckpt_critic_dim = infer_checkpoint_obs_dims(args_cli.checkpoint)
    print(f"[INFO] Checkpoint observation dims: actor={ckpt_actor_dim}, critic={ckpt_critic_dim}")

    # Create policy network
    runner_cfg = BadmintonInterceptPPORunnerCfg()
    env_actor_dim = int(np.prod(env.unwrapped.single_observation_space["policy"].shape))
    env_critic_dim = int(np.prod(env.unwrapped.single_observation_space["critic"].shape))
    if (env_actor_dim, env_critic_dim) != (ckpt_actor_dim, ckpt_critic_dim):
        print(
            "[INFO] Adapting observations to checkpoint dims: "
            f"env_actor={env_actor_dim}, env_critic={env_critic_dim}"
        )
        env = CheckpointObsAdapter(env, actor_dim=ckpt_actor_dim, critic_dim=ckpt_critic_dim)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=env.device)
    
    # Load the checkpoint
    if "model_state_dict" in checkpoint:
        runner.load(args_cli.checkpoint, load_optimizer=False)
    elif "actor_state_dict" in checkpoint and "critic_state_dict" in checkpoint:
        print("[INFO] Loading custom checkpoint format (actor_state_dict / critic_state_dict)")
        model_state = build_combined_model_state_from_split_checkpoint(checkpoint)
        runner.alg.policy.load_state_dict(model_state, strict=False)
    else:
        raise RuntimeError(
            "Unsupported checkpoint format. Expected either 'model_state_dict' or "
            "'actor_state_dict' + 'critic_state_dict'."
        )
    
    print("[INFO] Loaded policy successfully")
    print("[INFO] Starting evaluation...")

    # Reset environment
    obs, _ = env.reset()
    
    # Statistics tracking
    episode_count = 0
    step_count = 0
    reason_counts = defaultdict(int)
    stage_episode_counts = {1: 0, 2: 0, 3: 0}
    stage_success_counts = {1: 0, 2: 0, 3: 0}
    
    start_time = time.time()
    
    while episode_count < args_cli.num_episodes:
        with torch.no_grad():
            actions = runner.get_inference_policy(device=env.device)(obs)
        
        obs, rewards, dones, infos = env.step(actions)
        step_count += 1
        
        # Collect statistics from completed episodes
        if dones.any():
            num_done = int(dones.sum().item())
            
            # Get termination reasons from extras
            if isinstance(infos, dict) and "episode" in infos:
                episode_stats = infos["episode"]
                
                # Count termination reasons
                for reason in ["success_contact", "failure_ball_drop", "failure_out_of_bounds", 
                              "failure_tilt", "failure_net_contact", "failure_server_side_grounded", "timeout"]:
                    if reason in episode_stats:
                        reason_counts[reason] += int(episode_stats[reason])
                
                # Track stage-specific success rates
                stage_id = episode_stats.get("curriculum_stage_id", args_cli.curriculum_stage)
                stage_episode_counts[stage_id] = num_done
                if "success_contact" in episode_stats:
                    stage_success_counts[stage_id] += int(episode_stats["success_contact"])
            
            episode_count += num_done
            
            # Print progress every 100 episodes
            if episode_count % 100 < num_done or episode_count >= args_cli.num_episodes:
                elapsed = time.time() - start_time
                eps_per_sec = episode_count / elapsed if elapsed > 0 else 0
                print(f"[INFO] Episodes: {episode_count}/{args_cli.num_episodes} "
                      f"({100*episode_count/args_cli.num_episodes:.1f}%) | "
                      f"Speed: {eps_per_sec:.1f} eps/s | "
                      f"Steps: {step_count}")
    
    # Final statistics
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"Curriculum Stage: {args_cli.curriculum_stage}")
    print(f"Total Episodes: {episode_count}")
    print(f"Total Steps: {step_count}")
    print(f"Elapsed Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Throughput: {episode_count/elapsed:.1f} episodes/s")
    print()
    
    # Termination reason breakdown
    print("Termination Reasons:")
    total_resets = sum(reason_counts.values())
    if total_resets > 0:
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            percentage = 100 * count / total_resets
            print(f"  {reason:40s}: {count:5d} ({percentage:5.1f}%)")
    print()
    
    # Success rate
    success_count = reason_counts.get("success_contact", 0)
    success_rate = 100 * success_count / total_resets if total_resets > 0 else 0
    print(f"SUCCESS RATE: {success_rate:.2f}% ({success_count}/{total_resets})")
    print("="*70)
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
