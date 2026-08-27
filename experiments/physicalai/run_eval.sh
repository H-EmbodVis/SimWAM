#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="./src:./experiments/navsim:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-27895}"
N_TRAJ="${N_TRAJ:-6}"
NUM_STEPS="${NUM_STEPS:-10}"
CKPT="${CKPT:-./weights/checkpoint.pt}"
EXP_NAME="${EXP_NAME:-physicalai_eval}"

torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes 1 \
  --master-port "${MASTER_PORT}" \
  experiments/physicalai/eval.py \
  task=physicalai_uncond_front_384x672_1e-4 \
  ckpt="${CKPT}" \
  experiment_name="${EXP_NAME}" \
  EVALUATION.num_traj_samples="${N_TRAJ}" \
  EVALUATION.num_inference_steps="${NUM_STEPS}" \
  "$@"
