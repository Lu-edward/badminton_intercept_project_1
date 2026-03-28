"""Play/visualize a trained policy for badminton interception."""

from __future__ import annotations

import argparse
import os
import time
import torch

from isaaclab.app import AppLauncher


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Play trained policy with visualization.")
    parser.add_argument("--task", type=str, default="Isaac-Badminton-Intercept-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=0.7, help="Playback speed multiplier (0.7 = slower)")
    parser.add_argument("--num_episodes", type=int, default=30, help="Number of episodes to record (10 per curriculum stage)")
    parser.add_argument("--video_length", type=int, default=1000, help="Length of each video in steps")
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    
    # Override headless to False for visualization
    args.headless = False
    
    return args, hydra_args


args_cli, _hydra_args = parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import cv2
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
    """Adapts env observations to the actor/critic dims expected by a checkpoint."""

    def __init__(self, env, actor_dim: int, critic_dim: int):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.actor_dim = int(actor_dim)
        self.critic_dim = int(critic_dim)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs, extras = self.env.reset()
        return self._adapt_obs(obs), extras

    def get_observations(self):
        return self._adapt_obs(self.env.get_observations())

    def step(self, actions):
        obs, rewards, dones, extras = self.env.step(actions)
        return self._adapt_obs(obs), rewards, dones, extras

    def close(self):
        return self.env.close()

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
    env_cfg.termination_debug_print = True
    # Keep evaluation deterministic and easier to debug.
    env_cfg.randomization_enabled = False
    env_cfg.uav_init_lin_vel_range = (0.0, 0.0)
    # Enable fixed curriculum stage mode for visualization
    env_cfg.curriculum_fixed_stage = True
    env_cfg.curriculum_initial_stage_id = 1  # Start with stage 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Create environment
    task_id = args_cli.task or TASK_ID
    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env)

    # Setup video recording
    checkpoint_name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
    video_dir = os.path.join("videos", checkpoint_name)
    os.makedirs(video_dir, exist_ok=True)
    print(f"[INFO] Videos will be saved to: {video_dir}")

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
            "[INFO] Adapting visualization observations to checkpoint dims: "
            f"env_actor={env_actor_dim}, env_critic={env_critic_dim}"
        )
        env = CheckpointObsAdapter(env, actor_dim=ckpt_actor_dim, critic_dim=ckpt_critic_dim)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=env.device)
    
    # Load the checkpoint using runner's load method
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
    
    print(f"[INFO] Loaded policy successfully")
    print(f"[INFO] Starting visualization with {args_cli.num_envs} environment(s)...")
    print(f"[INFO] Playback speed: {args_cli.speed}x")
    print(f"[INFO] Will record {args_cli.num_episodes} episodes")
    print("[INFO] Press Ctrl+C to stop")

    # Reset environment
    obs, _ = env.reset()
    
    # Calculate sleep time for desired playback speed
    # Physics runs at dt=0.005s per step, with decimation=4, so env step = 0.02s
    env_dt = 0.02  # seconds per environment step
    sleep_time = env_dt / args_cli.speed if args_cli.speed > 0 else 0
    
    # Run policy with video recording - single video for all episodes
    # Curriculum stages: 1-10 episodes = stage 1, 11-20 = stage 2, 21-30 = stage 3
    EPISODES_PER_STAGE = 10
    episode_count = 0
    step_count = 0
    all_video_frames = []
    
    def get_stage_for_episode(ep_idx):
        """Determine curriculum stage based on episode index (0-based)."""
        if ep_idx < EPISODES_PER_STAGE:
            return 1  # Stage 1: episodes 0-9
        elif ep_idx < EPISODES_PER_STAGE * 2:
            return 2  # Stage 2: episodes 10-19
        else:
            return 3  # Stage 3: episodes 20-29
    
    def set_curriculum_stage(stage_id):
        """Manually set the curriculum stage in the environment."""
        if hasattr(env, 'env') and hasattr(env.env, '_curriculum'):
            env.env._curriculum.set_stage_by_id(stage_id)
            stage = env.env._curriculum.current_stage
            print(f"[INFO] Switched to curriculum stage {stage_id}: {stage.name}")
            print(f"[INFO]   UAV ranges: x={stage.uav_init_x_range}, y={stage.uav_init_y_range}, z={stage.uav_init_z_range}")
            print(f"[INFO]   Ball velocity: x={stage.ball_vel_x_range}, z={stage.ball_vel_z_range}")
    
    # Set initial stage
    current_stage = get_stage_for_episode(0)
    set_curriculum_stage(current_stage)
    
    try:
        while episode_count < args_cli.num_episodes:
            # Check if we need to switch curriculum stage
            next_stage = get_stage_for_episode(episode_count)
            if next_stage != current_stage:
                current_stage = next_stage
                set_curriculum_stage(current_stage)
            
            with torch.no_grad():
                # Use runner's get_inference_policy to get actions
                actions = runner.get_inference_policy(device=env.device)(obs)
            
            obs, rewards, dones, infos = env.step(actions)
            step_count += 1
            
            # Capture frame for video
            if hasattr(env, 'env') and hasattr(env.env, 'render'):
                frame = env.env.render()
                if frame is not None:
                    all_video_frames.append(frame)
            
            # Sleep to control playback speed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Print statistics when episodes complete
            if dones.any():
                episode_count += dones.sum().item()
                stage_info = f"[Stage {current_stage}]"
                print(f"[INFO] Episode {episode_count}/{args_cli.num_episodes} {stage_info} completed at step {step_count}")
                if isinstance(infos, dict) and ("termination" in infos):
                    print(f"[INFO] Termination summary: {infos['termination']}")
            
    except KeyboardInterrupt:
        print("\n[INFO] Visualization stopped by user")
    
    # Save single video with all episodes
    if all_video_frames:
        video_path = os.path.join(video_dir, f"{checkpoint_name}_all_episodes.mp4")
        if len(all_video_frames) > 0:
            height, width = all_video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            for frame in all_video_frames:
                if frame.shape[2] == 3:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_rgb = frame
                out.write(frame_rgb)
            out.release()
            print(f"[INFO] Saved single video with all episodes: {video_path}")
            print(f"[INFO] Total frames: {len(all_video_frames)}, Total episodes: {episode_count}")
    
    env.close()
    print(f"[INFO] Total episodes: {episode_count}, Total steps: {step_count}")


if __name__ == "__main__":
    main()
    simulation_app.close()
