#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-./data/navsim/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-./data/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-./evaluate_results_v2}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH="./navsim_v2:./src:${PYTHONPATH:-}"

SPLIT="${SPLIT:-both}"
WORKER="${WORKER:-sequential}"
EXP_NAME="${EXP_NAME:-simwam_v2}"
NAVTEST_CACHE="${NAVTEST_CACHE:-./data/metric_cache_navtest_v2}"
NAVHARD_CACHE="${NAVHARD_CACHE:-./data/metric_cache_navhard_two_stage}"
NAVTEST_PRED_DIR="${NAVTEST_PRED_DIR:-${NAVSIM_EXP_ROOT}/navtest/${EXP_NAME}/pred_actions}"
NAVHARD_PRED_DIR="${NAVHARD_PRED_DIR:-${NAVSIM_EXP_ROOT}/navhard/${EXP_NAME}/pred_actions}"
SYNTHETIC_SENSOR_PATH="${SYNTHETIC_SENSOR_PATH:-./data/navsim/navhard_two_stage/sensor_blobs}"
SYNTHETIC_SCENES_PATH="${SYNTHETIC_SCENES_PATH:-./data/navsim/navhard_two_stage/synthetic_scene_pickles}"

score_navtest() {
  echo "===== Scoring NAVSIM v2 navtest: ${EXP_NAME} ====="
  python -u navsim_v2/navsim/planning/script/run_pdm_score_one_stage.py \
    train_test_split=navtest \
    worker="${WORKER}" \
    agent=npy_trajectory_agent \
    agent.pred_actions_path="${NAVTEST_PRED_DIR}" \
    traffic_agents=reactive \
    metric_cache_path="${NAVTEST_CACHE}" \
    experiment_name="${EXP_NAME}_navtest_v2" \
    "$@"
}

score_navhard() {
  echo "===== Scoring NAVSIM v2 navhard: ${EXP_NAME} ====="
  python -u navsim_v2/navsim/planning/script/run_pdm_score.py \
    train_test_split=navhard_two_stage \
    worker="${WORKER}" \
    agent=npy_trajectory_agent \
    agent.pred_actions_path="${NAVHARD_PRED_DIR}" \
    metric_cache_path="${NAVHARD_CACHE}" \
    synthetic_sensor_path="${SYNTHETIC_SENSOR_PATH}" \
    synthetic_scenes_path="${SYNTHETIC_SCENES_PATH}" \
    experiment_name="${EXP_NAME}_navhard_v2" \
    "$@"
}

case "${SPLIT}" in
  navtest)
    score_navtest "$@"
    ;;
  navhard)
    score_navhard "$@"
    ;;
  both)
    score_navtest "$@"
    score_navhard "$@"
    ;;
  *)
    echo "SPLIT must be one of: navtest, navhard, both" >&2
    exit 2
    ;;
esac
