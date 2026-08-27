#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="./src:${PYTHONPATH:-}"
export PHYSICALAI_TRAIN_JSONL="${PHYSICALAI_TRAIN_JSONL:-./data/physicalai_train.jsonl}"
export PHYSICALAI_STATS_PATH="${PHYSICALAI_STATS_PATH:-./data/physicalai_dataset_stats.json}"
export NAVSIM_TEXT_EMBED_CACHE="${NAVSIM_TEXT_EMBED_CACHE:-./data/text_embeds_cache/navsim}"

for required_file in "${PHYSICALAI_TRAIN_JSONL}" "${PHYSICALAI_STATS_PATH}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Error: required file does not exist: ${required_file}" >&2
    exit 1
  fi
done

first_non_empty() {
  for value in "$@"; do
    if [[ -n "${value}" ]]; then
      echo "${value}"
      return 0
    fi
  done
  return 1
}

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

NPROC_PER_NODE="$(first_non_empty "${NPROC_PER_NODE:-}" "${GPU_NUM:-}" "1")"
MASTER_ADDR="$(first_non_empty "${MASTER_ADDR:-}" "127.0.0.1")"
MASTER_PORT="$(first_non_empty "${MASTER_PORT:-}" "29500")"
NODE_RANK="$(first_non_empty "${NODE_RANK:-}" "${RANK:-}" "0")"
NNODES="$(first_non_empty "${NNODES:-}" "1")"

if ! is_integer "${NPROC_PER_NODE}" || ! is_integer "${MASTER_PORT}" || \
   ! is_integer "${NODE_RANK}" || ! is_integer "${NNODES}"; then
  echo "NPROC_PER_NODE, MASTER_PORT, NODE_RANK and NNODES must be integers." >&2
  exit 1
fi

TASK_BASENAME="physicalai_uncond_front_384x672_1e-4"
EXTRA_ARGS=("$@")
HAS_TASK=false
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == task=* ]]; then
    TASK_BASENAME="${arg#task=}"
    TASK_BASENAME="${TASK_BASENAME%.yaml}"
    HAS_TASK=true
  fi
done
if [[ "${HAS_TASK}" == false ]]; then
  EXTRA_ARGS=("task=${TASK_BASENAME}" "${EXTRA_ARGS[@]}")
fi

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE="${ACCELERATE_DEEPSPEED_CONFIG_FILE:-./scripts/ds_configs/ds_zero1_config.json}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[launch] task=${TASK_BASENAME} nproc_per_node=${NPROC_PER_NODE} nnodes=${NNODES} run_id=${RUN_ID}"
echo "[physicalai] dataset=${PHYSICALAI_TRAIN_JSONL}"
echo "[physicalai] stats=${PHYSICALAI_STATS_PATH}"

torchrun \
  --nnodes "${NNODES}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  --node_rank "${NODE_RANK}" \
  "./scripts/train.py" \
  "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"
