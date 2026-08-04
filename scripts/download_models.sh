#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-inference}"

if [[ "$MODE" != "inference" && "$MODE" != "training" ]]; then
  echo "Usage: $0 [inference|training]" >&2
  exit 2
fi

if command -v hf >/dev/null 2>&1; then
  HF_DOWNLOAD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DOWNLOAD=(huggingface-cli download)
else
  echo "Hugging Face CLI is missing. Install requirements first." >&2
  exit 1
fi

"${HF_DOWNLOAD[@]}" Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir wan_models/Wan2.1-T2V-1.3B
"${HF_DOWNLOAD[@]}" Efficient-Large-Model/LongLive-1.3B \
  --local-dir longlive_models

if [[ "$MODE" == "training" ]]; then
  "${HF_DOWNLOAD[@]}" Wan-AI/Wan2.1-T2V-14B \
    --local-dir wan_models/Wan2.1-T2V-14B
fi
