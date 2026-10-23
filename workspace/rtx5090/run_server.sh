#!/bin/bash
# Start Wall-X serving on RTX 5090.
# Usage (from repo root): bash workspace/rtx5090/run_server.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${SCRIPT_DIR}/local/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/local/env.sh"
elif [[ -f "${SCRIPT_DIR}/env.example" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/env.example"
else
  echo "Missing ${SCRIPT_DIR}/local/env.sh (copy from env.example)" >&2
  exit 1
fi

: "${CONDA_SH:?Set CONDA_SH in local/env.sh}"
: "${CONDA_ENV:?Set CONDA_ENV in local/env.sh}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH in local/env.sh}"

# shellcheck source=/dev/null
source "${CONDA_SH}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
conda activate "${CONDA_ENV}"

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include:${CPATH:-}"
export PYTHON_BIN="${CONDA_PREFIX}/bin/python"
export HOST="0.0.0.0"

export ENABLE_FAST_PREPROCESS="${ENABLE_FAST_PREPROCESS:-false}"
export WALLX_VISION_ATTN_IMPLEMENTATION="${WALLX_VISION_ATTN_IMPLEMENTATION:-flash_attention_2}"
export ENABLE_CUDA_GRAPH=1
export ENABLE_EXPERIMENTAL_ENGINE=1

cd "${REPO_ROOT}"
exec bash scripts/run_serving.sh \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --train-config-path "${CHECKPOINT_PATH}/config.yml" \
  --port "${PORT}" \
  --cuda-id "${CUDA_ID}" \
  --robot-type ex001 \
  --serialize-actions \
  --enable-cuda-graph \
  --enable-experimental-engine \
  -- \
  --model-config.norm-key ex_normal
