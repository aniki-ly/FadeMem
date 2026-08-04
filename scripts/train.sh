#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG_PATH="${1:-configs/longlive_train_long.yaml}"
NUM_GPUS="${NUM_GPUS:-8}"
LOG_DIR="${LOG_DIR:-outputs/train}"
WANDB_DIR="${WANDB_DIR:-outputs/wandb}"

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  train.py \
  --config_path "$CONFIG_PATH" \
  --logdir "$LOG_DIR" \
  --wandb-save-dir "$WANDB_DIR" \
  --disable-wandb \
  --no-one-logger \
  --no_visualize
