#!/usr/bin/env bash
set -euo pipefail

# LIBERO evaluation launcher. Configure with environment variables:
#
#   CHECKPOINT_PATH=/path/to/checkpoint bash scripts/run_libero.sh
#   bash scripts/run_libero.sh /path/to/checkpoint
#   CHECKPOINT_PATH=/path/to/checkpoint SMOKE=1 bash scripts/run_libero.sh
#   CONFIG=/path/to/eval.yaml bash scripts/run_libero.sh
#
# Optional knobs:
#   CUDA_ID=0
#   DRIVER_MODE=in_process
#   NUM_WORKERS=1
#   MAX_BATCH_SIZE=1
#   TASK_SUITE_NAME=libero_spatial
#   NUM_TRIALS_PER_TASK=50
#   TASK_INDICES=0,1,2
#   MAX_INFER_TIMES=52
#   NORM_KEY=libero_all
#   LOG_DIR=/tmp/harrix_libero_eval

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -gt 1 ]]; then
    echo "Usage: bash scripts/run_libero.sh [CHECKPOINT_PATH]" >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    CHECKPOINT_PATH="$1"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_ID:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"
if [[ -d "${REPO_ROOT}/third_party/harrix/python" ]]; then
    export PYTHONPATH="${REPO_ROOT}/third_party/harrix/python:${PYTHONPATH:-}"
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

args=()
if [[ -n "${CONFIG:-}" ]]; then
    args+=(--config "${CONFIG}")
else
    if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
        echo "CHECKPOINT_PATH is required when CONFIG is not set." >&2
        exit 2
    fi
    args+=(--checkpoint-path "${CHECKPOINT_PATH}")
fi

if [[ -n "${TRAIN_CONFIG_PATH:-}" ]]; then
    args+=(--train-config-path "${TRAIN_CONFIG_PATH}")
fi
if [[ "${SMOKE:-0}" == "1" ]]; then
    args+=(--smoke)
fi
if [[ "${DET:-0}" == "1" ]]; then
    args+=(--deterministic-model)
fi
if [[ -n "${TASK_INDICES:-}" ]]; then
    args+=(--task-indices "${TASK_INDICES}")
fi
if [[ -n "${MAX_INFER_TIMES:-}" ]]; then
    args+=(--max-infer-times "${MAX_INFER_TIMES}")
fi

args+=(--driver-mode "${DRIVER_MODE:-in_process}")
args+=(--num-workers "${NUM_WORKERS:-1}")
args+=(--max-batch-size "${MAX_BATCH_SIZE:-1}")
args+=(--task-suite-name "${TASK_SUITE_NAME:-libero_spatial}")
args+=(--num-trials-per-task "${NUM_TRIALS_PER_TASK:-50}")
args+=(--norm-key "${NORM_KEY:-libero_all}")
args+=(--log-dir "${LOG_DIR:-/tmp/harrix_libero_eval}")

cd "${REPO_ROOT}"
python scripts/infer_libero.py "${args[@]}"
