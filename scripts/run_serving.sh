#!/bin/bash
set -euo pipefail

# WebSocket inference server launcher. Configure with environment variables:
#
#   CHECKPOINT_PATH=/path/to/checkpoint \
#   TRAIN_CONFIG_PATH=/path/to/my_config.yml \
#   bash scripts/run_serving.sh

export CUDA_VISIBLE_DEVICES="${CUDA_ID:-0}"
export WALLX_SMOOTH_ACTION="${WALLX_SMOOTH_ACTION:-true}"
export ENABLE_CUDA_SYNC_IN_TIMER="${ENABLE_CUDA_SYNC_IN_TIMER:-true}"
export ENABLE_FAST_PREPROCESS="${ENABLE_FAST_PREPROCESS:-true}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/path/to/checkpoint}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-/path/to/my_config.yml}"
PORT="${PORT:-43166}"

python -m wall_x.serving.launch_serving \
  --env X2ROBOT \
  --port "${PORT}" \
  --serialize-actions \
  --max-batch-size 1 \
  --enable-experimental-engine \
  --image-passing-mode numpy \
  --enable-cuda-graph \
  model-config:server-model-config \
  --model-config.checkpoint-path "${CHECKPOINT_PATH}" \
  --model-config.train-config-path "${TRAIN_CONFIG_PATH}"
