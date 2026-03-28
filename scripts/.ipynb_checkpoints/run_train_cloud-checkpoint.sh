#!/usr/bin/env bash
set -euo pipefail

# -------------------------------
# Cloud single-GPU training setup
# -------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Set this on cloud if IsaacLab is not in the default path.
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/IsaacLab}"
ISAACLAB_SH="${ISAACLAB_SH:-$ISAACLAB_ROOT/isaaclab.sh}"

# Core run parameters (edit here)
TASK_ID="Isaac-Badminton-Intercept-Direct-v0"
DEVICE="cuda:0"
HEADLESS="--headless"

NUM_ENVS=4096
EPISODE_LENGTH_S=4.0
TOTAL_FRAMES=2000000000
SEED=42
EXPERIMENT_NAME="badminton_intercept_cloud"

# Key PPO parameters (edit here)
NUM_STEPS_PER_ENV=32
SAVE_INTERVAL=1000
LEARNING_RATE=3e-4
NUM_MINI_BATCHES=16
NUM_LEARNING_EPOCHS=5
ENTROPY_COEF=0.005
CLIP_PARAM=0.15
DESIRED_KL=0.005

# Derived iterations for the target total frames.
FRAMES_PER_ITER=$((NUM_ENVS * NUM_STEPS_PER_ENV))
MAX_ITERATIONS=$((TOTAL_FRAMES / FRAMES_PER_ITER))
if (( TOTAL_FRAMES % FRAMES_PER_ITER != 0 )); then
  MAX_ITERATIONS=$((MAX_ITERATIONS + 1))
fi

if [[ ! -x "$ISAACLAB_SH" ]]; then
  echo "[error] isaaclab.sh not found or not executable: $ISAACLAB_SH"
  echo "        Set ISAACLAB_ROOT or ISAACLAB_SH before running this script."
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

"$ISAACLAB_SH" -p "$PROJECT_ROOT/scripts/train_rsl_official.py" \
  $HEADLESS \
  --task "$TASK_ID" \
  --device "$DEVICE" \
  --num_envs "$NUM_ENVS" \
  --episode_length_s "$EPISODE_LENGTH_S" \
  --max_iterations "$MAX_ITERATIONS" \
  --seed "$SEED" \
  --experiment_name "$EXPERIMENT_NAME" \
  --num_steps_per_env "$NUM_STEPS_PER_ENV" \
  --save_interval "$SAVE_INTERVAL" \
  --learning_rate "$LEARNING_RATE" \
  --num_mini_batches "$NUM_MINI_BATCHES" \
  --num_learning_epochs "$NUM_LEARNING_EPOCHS" \
  --entropy_coef "$ENTROPY_COEF" \
  --clip_param "$CLIP_PARAM" \
  --desired_kl "$DESIRED_KL"
