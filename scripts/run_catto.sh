#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# All SeeSR sampling/model settings retain their original CLI defaults.
PYTHON="${PYTHON:-python}"
MODEL_ROOT="${MODEL_ROOT:-preset/models}"
CONFIG="${CONFIG:-configs/setting.json}"
INPUT="${INPUT:-preset/datasets/input}"
OUTPUT="${OUTPUT:-outputs/catto}"
SEED="${SEED:-5}"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
exec "$PYTHON" -B test_seesr.py \
  --pretrained_model_path "${SD_MODEL:-$MODEL_ROOT/stable-diffusion-2-base}" \
  --seesr_model_path "${SEESR_MODEL:-$MODEL_ROOT/seesr}" \
  --ram_model_path "${RAM_MODEL:-$MODEL_ROOT/ram_swin_large_14m.pth}" \
  --ram_ft_path "${DAPE_MODEL:-$MODEL_ROOT/DAPE.pth}" \
  --image_path "$INPUT" --output_dir "$OUTPUT" --seed "$SEED" \
  --catto-config "$CONFIG" "$@"
