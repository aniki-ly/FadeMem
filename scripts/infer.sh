#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG_PATH="${1:-configs/inference.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  inference.py --config_path "$CONFIG_PATH"
