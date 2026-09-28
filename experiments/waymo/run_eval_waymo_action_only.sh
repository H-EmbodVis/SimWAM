#!/usr/bin/env bash
# Full 479-row Waymo val; one trajectory per sample; no future image reads.
# Usage: CKPT=./weights/SimWAM-Waymo-IL.pt NPROC_PER_NODE=8 \
#          bash experiments/waymo/run_eval_waymo_action_only.sh
# Override with ckpt=/path/to/step_XXXXXX.pt and EVALUATION.output_dir=/path/to/results.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export ACCELERATE_USE_DEEPSPEED=false

# All defaults are repository-relative; every command runs from the project root.
export WAYMO_DATA_ROOT="${WAYMO_DATA_ROOT:-./data/waymo}"
export WAYMO_VAL_JSONL="${WAYMO_VAL_JSONL:-./data/waymo_val_front_current_traj5s_4hz_xy_pref_only.jsonl}"
export WAYMO_STATS_PATH="${WAYMO_STATS_PATH:-./data/waymo_dataset_stats.json}"
export WAYMO_TEXT_EMBED_CACHE="${WAYMO_TEXT_EMBED_CACHE:-${NAVSIM_TEXT_EMBED_CACHE:-./data/text_embeds_cache/navsim}}"

CKPT="${CKPT:-${WAYMO_IL_CHECKPOINT:-./weights/SimWAM-Waymo-IL.pt}}"
OUTPUT_DIR="${WAYMO_EVAL_OUTPUT_DIR:-}"
DRY_RUN=false
EXTRA_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=true ;;
    ckpt=*) CKPT="${arg#ckpt=}" ;;
    EVALUATION.output_dir=*) OUTPUT_DIR="${arg#EVALUATION.output_dir=}" ;;
    *) EXTRA_ARGS+=("${arg}") ;;
  esac
done
if [[ -z "${CKPT}" ]]; then
  echo "Error: pass ckpt=/path/to/Waymo/checkpoints/weights/step_XXXXXX.pt or set CKPT." >&2; exit 1
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  CHECKPOINT_NAME="${CKPT##*/}"
  CHECKPOINT_TAG="${CHECKPOINT_NAME%.pt}"
  if [[ "${CKPT}" == */checkpoints/weights/*.pt ]]; then
    OUTPUT_DIR="${CKPT%/checkpoints/weights/*}/eval/${CHECKPOINT_TAG}"
  else
    OUTPUT_DIR="./runs/waymo_eval/${CHECKPOINT_TAG}"
  fi
fi
# Content-addressed cache of the fixed T5 prompt: sha256(FIXED_PROMPT).t5_len256.wan22ti2v5b.pt
PROMPT_CACHE="${WAYMO_TEXT_EMBED_CACHE}/2b876e249d9af4dff512d9878eefbd1ec46967807fae945c746c66a89b0ddbda.t5_len256.wan22ti2v5b.pt"
for required in "${CKPT}" "${WAYMO_VAL_JSONL}" "${WAYMO_STATS_PATH}" "${PROMPT_CACHE}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Error: required file does not exist: ${required}" >&2; exit 1
  fi
done
if [[ ! -d "${WAYMO_DATA_ROOT}" ]]; then
  echo "Error: required directory does not exist: ${WAYMO_DATA_ROOT}" >&2; exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-29504}"
for name in NPROC_PER_NODE NNODES MASTER_PORT; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: ${name} must be a positive integer." >&2; exit 1
  fi
done
if [[ ! "${NODE_RANK}" =~ ^(0|[1-9][0-9]*)$ ]] || (( NODE_RANK >= NNODES || MASTER_PORT > 65535 )); then
  echo "Error: require 0 <= NODE_RANK < NNODES and 1 <= MASTER_PORT <= 65535." >&2; exit 1
fi
if (( NNODES > 1 )) && [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "Error: multi-node evaluation requires MASTER_ADDR." >&2; exit 1
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

COMMAND=(torchrun --nnodes "${NNODES}" --nproc_per_node "${NPROC_PER_NODE}"
  --node_rank "${NODE_RANK}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}"
  "${SCRIPT_DIR}/eval_waymo_action_only.py" "ckpt=${CKPT}"
  "EVALUATION.output_dir=${OUTPUT_DIR}" "${EXTRA_ARGS[@]}")
printf '[launch] '; printf '%q ' "${COMMAND[@]}"; printf '\n'
if [[ "${DRY_RUN}" == true ]]; then exit 0; fi
exec "${COMMAND[@]}"
