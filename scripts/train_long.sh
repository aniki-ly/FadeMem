#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export LOG_DIR="${LOG_DIR:-outputs/train_long}"

exec bash scripts/train.sh configs/longlive_train_long.yaml
