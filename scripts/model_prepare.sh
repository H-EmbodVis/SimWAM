#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${PROJECT_ROOT}/checkpoints}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

cd "${PROJECT_ROOT}"
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/simwam_navsim.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim_3outdim.pt \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  "$@"
