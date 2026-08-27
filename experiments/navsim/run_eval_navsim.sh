#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${CKPT:?Set CKPT to a supervised or merged FlowGRPO checkpoint.}"

TASK="${TASK:-navsim_uncond_front_384x672_1e-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-27890}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${TASK}_eval}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${PROJECT_ROOT}/data/navsim/maps}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${PROJECT_ROOT}/navsim}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${PROJECT_ROOT}/data/navsim}"
export NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/trainval}"
export NAVSIM_SENSOR_BLOBS_PATH="${NAVSIM_SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/trainval}"
export NAVSIM_SPLIT="${NAVSIM_SPLIT:-navtest}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${PROJECT_ROOT}/evaluate_results}"
export PYTHONPATH="${PROJECT_ROOT}/src:${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

cd "${PROJECT_ROOT}"
torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes 1 \
  --master_port "${MASTER_PORT}" \
  experiments/navsim/eval_navsim.py \
  "task=${TASK}" \
  "ckpt=${CKPT}" \
  "experiment_name=${EXPERIMENT_NAME}" \
  "$@"
