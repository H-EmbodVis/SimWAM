#!/usr/bin/env bash
# Predict all 1505 official Waymo test frames and write predictions.jsonl.
# Usage: CKPT=./weights/SimWAM-Waymo-RL.pt NPROC_PER_NODE=8 \
#          bash experiments/waymo/run_predict_waymo_test.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export ACCELERATE_USE_DEEPSPEED=false

# All defaults are repository-relative; every command runs from the project root.
export WAYMO_DATA_ROOT="${WAYMO_DATA_ROOT:-./data/waymo}"
export WAYMO_TEST_JSONL="${WAYMO_TEST_JSONL:-./data/waymo_test_front_current_input.jsonl}"
export WAYMO_STATS_PATH="${WAYMO_STATS_PATH:-./data/waymo_dataset_stats.json}"
export WAYMO_TEXT_EMBED_CACHE="${WAYMO_TEXT_EMBED_CACHE:-${NAVSIM_TEXT_EMBED_CACHE:-./data/text_embeds_cache/navsim}}"

CKPT="${WAYMO_TEST_CHECKPOINT:-./weights/SimWAM-Waymo-RL.pt}"
OUTPUT_DIR="${WAYMO_TEST_OUTPUT_DIR:-}"
DRY_RUN=false
EXTRA_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=true ;;
    ckpt=*) CKPT="${arg#ckpt=}" ;;
    PREDICTION.output_dir=*) OUTPUT_DIR="${arg#PREDICTION.output_dir=}" ;;
    *) EXTRA_ARGS+=("${arg}") ;;
  esac
done
if [[ -z "${CKPT}" ]]; then
  echo "Error: pass ckpt=/path/to/step_XXXXXX.pt or set WAYMO_TEST_CHECKPOINT." >&2; exit 1
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  CHECKPOINT_NAME="${CKPT##*/}"
  CHECKPOINT_TAG="${CHECKPOINT_NAME%.pt}"
  if [[ "${CKPT}" == */checkpoints/weights/*.pt ]]; then
    MODEL_TAG="$(basename "${CKPT%/checkpoints/weights/*}")"
  else
    MODEL_TAG="waymo"
  fi
  OUTPUT_DIR="./runs/waymo_test/${MODEL_TAG}_${CHECKPOINT_TAG}"
fi
# Content-addressed cache of the fixed T5 prompt: sha256(FIXED_PROMPT).t5_len256.wan22ti2v5b.pt
PROMPT_CACHE="${WAYMO_TEXT_EMBED_CACHE}/2b876e249d9af4dff512d9878eefbd1ec46967807fae945c746c66a89b0ddbda.t5_len256.wan22ti2v5b.pt"
for required in "${CKPT}" "${WAYMO_TEST_JSONL}" "${WAYMO_STATS_PATH}" "${PROMPT_CACHE}"; do
  if [[ ! -f "${required}" ]]; then echo "Missing file: ${required}" >&2; exit 1; fi
done
if [[ ! -d "${WAYMO_DATA_ROOT}" ]]; then
  echo "Error: required directory does not exist: ${WAYMO_DATA_ROOT}" >&2; exit 1
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-29506}"
for name in NPROC_PER_NODE NNODES MASTER_PORT; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: ${name} must be positive." >&2; exit 1
  fi
done
if [[ ! "${NODE_RANK}" =~ ^(0|[1-9][0-9]*)$ ]] || (( NODE_RANK >= NNODES || MASTER_PORT > 65535 )); then
  echo "Invalid node rank or port." >&2; exit 1
fi
if (( NNODES > 1 )) && [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "Multi-node inference requires MASTER_ADDR." >&2; exit 1
fi
COMMAND=(torchrun --nnodes "${NNODES}" --nproc_per_node "${NPROC_PER_NODE}"
  --node_rank "${NODE_RANK}" --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT}"
  "${SCRIPT_DIR}/predict_waymo_test.py" "ckpt=${CKPT}"
  "PREDICTION.output_dir=${OUTPUT_DIR}" "${EXTRA_ARGS[@]}")
printf '[launch] '; printf '%q ' "${COMMAND[@]}"; printf '\n'
if [[ "${DRY_RUN}" == true ]]; then exit 0; fi
exec "${COMMAND[@]}"
