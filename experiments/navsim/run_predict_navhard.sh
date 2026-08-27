#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-./data/navsim/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-./data/navsim}"
export NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-./data/navsim/navsim_logs/test}"
export NAVSIM_SENSOR_BLOBS_PATH="${NAVSIM_SENSOR_BLOBS_PATH:-./data/navsim/sensor_blobs/test}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH="./navsim_v2:./src:./experiments/navsim:${PYTHONPATH:-}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-27891}"
CKPT="${CKPT:-./weights/checkpoint.pt}"
EXP_NAME="${EXP_NAME:-simwam_navhard}"
PRED_DIR="${PRED_DIR:-./evaluate_results_v2/navhard/${EXP_NAME}/pred_actions}"
SCENE_FILTER="${SCENE_FILTER:-./navsim_v2/navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml}"

export NAVHARD_SYNTHETIC_SENSOR_PATH="${NAVHARD_SYNTHETIC_SENSOR_PATH:-./data/navsim/navhard_two_stage/sensor_blobs}"
export NAVHARD_SYNTHETIC_SCENES_PATH="${NAVHARD_SYNTHETIC_SCENES_PATH:-./data/navsim/navhard_two_stage/synthetic_scene_pickles}"

echo "===== Predicting NAVSIM v2 navhard: ${CKPT} (${EXP_NAME}) ====="
torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes 1 \
  --master-port "${MASTER_PORT}" \
  experiments/navsim/predict_navhard.py \
  task=navsim_uncond_front_384x672_1e-4 \
  ckpt="${CKPT}" \
  experiment_name="${EXP_NAME}" \
  EVALUATION.scene_filter="${SCENE_FILTER}" \
  EVALUATION.action_output_dir="${PRED_DIR}" \
  EVALUATION.synthetic_sensor_path="${NAVHARD_SYNTHETIC_SENSOR_PATH}" \
  EVALUATION.synthetic_scenes_path="${NAVHARD_SYNTHETIC_SCENES_PATH}" \
  "$@"

echo "Predictions written to: ${PRED_DIR}"
