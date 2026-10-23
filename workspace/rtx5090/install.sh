#!/bin/bash
# Install Wall-X conda env on RTX 5090.
# Usage (from repo root): bash workspace/rtx5090/install.sh
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
: "${LEROBOT_DIR:?Set LEROBOT_DIR in local/env.sh}"

# shellcheck source=/dev/null
source "${CONDA_SH}"

if [[ "${CONDA_ENV}" == */* ]]; then
  PY="${CONDA_ENV}/bin/python"
else
  PY="$(conda run -n "${CONDA_ENV}" which python 2>/dev/null || true)"
fi

if [[ ! -x "${PY}" ]]; then
  echo "[1/6] Creating conda env: ${CONDA_ENV}"
  if [[ "${CONDA_ENV}" == */* ]]; then
    conda create --prefix "${CONDA_ENV}" python=3.10 -y
  else
    conda create -n "${CONDA_ENV}" python=3.10 -y
  fi
else
  echo "[1/6] Using existing env: ${CONDA_ENV}"
fi

conda activate "${CONDA_ENV}"
PY="${CONDA_PREFIX}/bin/python"
PIP="${CONDA_PREFIX}/bin/pip"

echo "[2/6] Installing requirements.txt (use env pip, not pyenv)"
"${PIP}" install -r "${REPO_ROOT}/requirements.txt"

echo "[3/6] Installing dmuon"
"${PIP}" install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git"

echo "[4/6] Installing lerobot==0.4.4"
if [[ ! -d "${LEROBOT_DIR}/.git" ]]; then
  git clone https://github.com/huggingface/lerobot.git "${LEROBOT_DIR}"
fi
git -C "${LEROBOT_DIR}" fetch --tags
git -C "${LEROBOT_DIR}" checkout v0.4.4
"${PIP}" install --no-deps -e "${LEROBOT_DIR}"

echo "[5/6] Installing CUDA 12.8 nvcc (match torch cu128)"
conda install -p "${CONDA_PREFIX}" -y -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8 || true

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include:${CPATH:-}"
export FLASH_ATTN_CUDA_ARCHS=120

echo "[5b/6] Installing flash-attn (required for vision attention)"
MAX_JOBS=4 "${PIP}" install flash-attn==2.8.3 --no-build-isolation

echo "[6/6] Installing wall-x"
cd "${REPO_ROOT}"
MAX_JOBS=8 "${PIP}" install --no-build-isolation -e .

echo "Done. Activate with:"
if [[ "${CONDA_ENV}" == */* ]]; then
  echo "  conda activate ${CONDA_ENV}"
else
  echo "  conda activate ${CONDA_ENV}"
fi
echo "Verify:"
echo "  ${PY} -c \"import torch, flash_attn, lerobot, wall_x; print(torch.__version__, lerobot.__version__)\""
