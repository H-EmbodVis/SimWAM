#!/usr/bin/env bash
# Waymo supervised training: five FRONT images at 1 Hz, native 20x2 future actions at 4 Hz.
# Usage: NPROC_PER_NODE=8 bash scripts/train_waymo_zero1_torchrun.sh [Hydra overrides]
#
# W&B is optional and off by default (training logs to TensorBoard). If you want W&B,
# export WANDB_API_KEY in your shell before launching -- do NOT hardcode secrets here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
# Another checkout may be installed editable in the active environment.
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1

# All defaults are repository-relative; every command runs from the project root.
export WAYMO_DATA_ROOT="${WAYMO_DATA_ROOT:-./data/waymo}"
export WAYMO_TRAIN_JSONL="${WAYMO_TRAIN_JSONL:-./data/waymo_training_front_video4s_traj5s_4hz_xy.jsonl}"
export WAYMO_VAL_JSONL="${WAYMO_VAL_JSONL:-./data/waymo_val_front_current_traj5s_4hz_xy_pref_only.jsonl}"
export WAYMO_STATS_PATH="${WAYMO_STATS_PATH:-./data/waymo_dataset_stats.json}"
export WAYMO_TEXT_EMBED_CACHE="${WAYMO_TEXT_EMBED_CACHE:-${NAVSIM_TEXT_EMBED_CACHE:-./data/text_embeds_cache/navsim}}"

TASK="waymo_uncond_front_512x480_1e-4"
DRY_RUN=false
EXTRA_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=true ;;
    task=*) TASK="${arg#task=}"; TASK="${TASK%.yaml}" ;;
    configs/task/*.yaml) TASK="${arg##*/}"; TASK="${TASK%.yaml}" ;;
    *) EXTRA_ARGS+=("${arg}") ;;
  esac
done

NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_NUM:-1}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-${MAIN_PROCESS_PORT:-29503}}"
for name in NPROC_PER_NODE NNODES MASTER_PORT; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: ${name} must be a positive integer." >&2; exit 1
  fi
done
if [[ ! "${NODE_RANK}" =~ ^(0|[1-9][0-9]*)$ ]] || (( NODE_RANK >= NNODES || MASTER_PORT > 65535 )); then
  echo "Error: require 0 <= NODE_RANK < NNODES and 1 <= MASTER_PORT <= 65535." >&2; exit 1
fi
if (( NNODES > 1 )) && [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "Error: multi-node training requires MASTER_ADDR (and the same RUN_ID on every node)." >&2; exit 1
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE="${ACCELERATE_DEEPSPEED_CONFIG_FILE:-${PROJECT_ROOT}/scripts/ds_configs/ds_zero1_config.json}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Content-addressed cache of the fixed T5 prompt: sha256(FIXED_PROMPT).t5_len256.wan22ti2v5b.pt
# Regenerate with `bash scripts/precomput_text_embed.sh` if it is missing.
PROMPT_CACHE="${WAYMO_TEXT_EMBED_CACHE}/2b876e249d9af4dff512d9878eefbd1ec46967807fae945c746c66a89b0ddbda.t5_len256.wan22ti2v5b.pt"
for required in "${WAYMO_TRAIN_JSONL}" "${WAYMO_VAL_JSONL}" "${WAYMO_STATS_PATH}" "${PROMPT_CACHE}" "${ACCELERATE_DEEPSPEED_CONFIG_FILE}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Error: required file does not exist: ${required}" >&2; exit 1
  fi
done
if [[ ! -d "${WAYMO_DATA_ROOT}" ]]; then
  echo "Error: required directory does not exist: ${WAYMO_DATA_ROOT}" >&2; exit 1
fi

COMMAND=(torchrun --nnodes "${NNODES}" --nproc_per_node "${NPROC_PER_NODE}"
  --node_rank "${NODE_RANK}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}"
  "${PROJECT_ROOT}/scripts/train.py" "task=${TASK}"
  "output_dir=./runs/${TASK}/${RUN_ID}" "wandb.name=${TASK}" "${EXTRA_ARGS[@]}")
echo "[waymo] width=480 height=512 action=20x2@4Hz train_frames=5 val_frames=1"
echo "[waymo] image_root=${WAYMO_DATA_ROOT}"
echo "[waymo] train=${WAYMO_TRAIN_JSONL}"
echo "[waymo] val=${WAYMO_VAL_JSONL} stats=${WAYMO_STATS_PATH}"
printf '[launch] '; printf '%q ' "${COMMAND[@]}"; printf '\n'
if [[ "${DRY_RUN}" == true ]]; then exit 0; fi
exec "${COMMAND[@]}"
