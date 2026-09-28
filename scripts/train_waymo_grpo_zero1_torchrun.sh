#!/usr/bin/env bash
# Waymo action LoRA GRPO on the 479 scored val rows, with the official RFS reward.
# Usage: NPROC_PER_NODE=8 bash scripts/train_waymo_grpo_zero1_torchrun.sh
#
# W&B is optional and off by default (training logs to TensorBoard). If you want W&B,
# export WANDB_API_KEY in your shell before launching -- do NOT hardcode secrets here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1

# All defaults are repository-relative; every command runs from the project root.
export WAYMO_DATA_ROOT="${WAYMO_DATA_ROOT:-./data/waymo}"
export WAYMO_GRPO_JSONL="${WAYMO_GRPO_JSONL:-${WAYMO_VAL_JSONL:-./data/waymo_val_front_current_traj5s_4hz_xy_pref_only.jsonl}}"
export WAYMO_IL_CHECKPOINT="${WAYMO_IL_CHECKPOINT:-./weights/SimWAM-Waymo-IL.pt}"
export WAYMO_STATS_PATH="${WAYMO_STATS_PATH:-./data/waymo_dataset_stats.json}"
export WAYMO_TEXT_EMBED_CACHE="${WAYMO_TEXT_EMBED_CACHE:-${NAVSIM_TEXT_EMBED_CACHE:-./data/text_embeds_cache/navsim}}"

TASK="waymo_grpo_rfs"
DRY_RUN=false
EXTRA_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=true ;;
    task=*) TASK="${arg#task=}"; TASK="${TASK%.yaml}" ;;
    model.checkpoint_path=*) export WAYMO_IL_CHECKPOINT="${arg#model.checkpoint_path=}" ;;
    data.train.dataset_jsonl=*) export WAYMO_GRPO_JSONL="${arg#data.train.dataset_jsonl=}" ;;
    *) EXTRA_ARGS+=("${arg}") ;;
  esac
done

NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_NUM:-8}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_PORT="${MASTER_PORT:-29505}"
for name in NPROC_PER_NODE NNODES MASTER_PORT; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: ${name} must be a positive integer." >&2; exit 1
  fi
done
if [[ ! "${NODE_RANK}" =~ ^(0|[1-9][0-9]*)$ ]] || (( NODE_RANK >= NNODES || MASTER_PORT > 65535 )); then
  echo "Error: require 0 <= NODE_RANK < NNODES and 1 <= MASTER_PORT <= 65535." >&2; exit 1
fi
if (( NNODES > 1 )) && [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "Error: multi-node GRPO requires MASTER_ADDR." >&2; exit 1
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE="${ACCELERATE_DEEPSPEED_CONFIG_FILE:-${PROJECT_ROOT}/scripts/ds_configs/ds_zero1_config.json}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Content-addressed cache of the fixed T5 prompt: sha256(FIXED_PROMPT).t5_len256.wan22ti2v5b.pt
PROMPT_CACHE="${WAYMO_TEXT_EMBED_CACHE}/2b876e249d9af4dff512d9878eefbd1ec46967807fae945c746c66a89b0ddbda.t5_len256.wan22ti2v5b.pt"
for required in "${WAYMO_GRPO_JSONL}" "${WAYMO_IL_CHECKPOINT}" "${WAYMO_STATS_PATH}" "${PROMPT_CACHE}" "${ACCELERATE_DEEPSPEED_CONFIG_FILE}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Error: required file does not exist: ${required}" >&2; exit 1
  fi
done
if [[ ! -d "${WAYMO_DATA_ROOT}" ]]; then
  echo "Error: required directory does not exist: ${WAYMO_DATA_ROOT}" >&2; exit 1
fi

COMMAND=(torchrun --nnodes "${NNODES}" --nproc_per_node "${NPROC_PER_NODE}"
  --node_rank "${NODE_RANK}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}"
  "${PROJECT_ROOT}/scripts/train_grpo.py" --config-name train_waymo_grpo "task=${TASK}"
  "output_dir=./runs_grpo/${TASK}/${RUN_ID}" "wandb.name=${TASK}" "${EXTRA_ARGS[@]}")
echo "[waymo RL] data=${WAYMO_GRPO_JSONL}"
echo "[waymo RL] image_root=${WAYMO_DATA_ROOT}"
echo "[waymo RL] IL=${WAYMO_IL_CHECKPOINT} normalization=${WAYMO_STATS_PATH}"
printf '[launch] '; printf '%q ' "${COMMAND[@]}"; printf '\n'
if [[ "${DRY_RUN}" == true ]]; then exit 0; fi
exec "${COMMAND[@]}"
