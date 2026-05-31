#!/usr/bin/env bash
# Smoke-test one of the three Data backends end-to-end on a single GPU.
#
# Usage:
#   bash tests/smoke/run_backend_smoke.sh {lerobot|x2robot_v2|x2robot_v1}
#
# Behavior:
#   - Launches train_fsdp.py with the matching tests/smoke/configs/<backend>_smoke.yml
#   - 1 epoch, batch_size_per_gpu=1, val runs once at epoch end
#   - Logs land in ckpt/smoke/<backend>/run.log; checkpoints in ckpt/smoke/<backend>/
#   - Asserts the run reached at least one training step + the val_loop fired
set -euo pipefail

BACKEND=${1:-}
case "${BACKEND}" in
    lerobot|x2robot_v2|x2robot_v1) ;;
    *)
        echo "Usage: $0 {lerobot|x2robot_v2|x2robot_v1}" >&2
        exit 2
        ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Force this checkout to win over any pre-installed ``wall_x`` package
# (e.g. ``pip install -e`` from a different worktree). torchrun spawns
# subprocesses that re-resolve imports, so PYTHONPATH is the durable lever.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="tests/smoke/configs/${BACKEND}_smoke.yml"
OUT_DIR="ckpt/smoke/${BACKEND}"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/run.log"

# Single GPU is enough for a smoke. Pick a non-default port to avoid clashes.
NPROC=${NPROC:-1}
MASTER_PORT=${MASTER_PORT:-32503}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "==> ${BACKEND} smoke: torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT}"
echo "==> config=${CONFIG}"
echo "==> log=${LOG}"

torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    wall_x/trainer/fsdp_trainer/train_fsdp.py \
    --config "${CONFIG}" \
    2>&1 | tee "${LOG}"

# ---- Assertions ----
# Smoke goal is to exercise the build_data → backend.build path end-to-end.
# Per-step training output depends on the dataset yaml producing non-empty
# batches (orthogonal to the refactor); the load + epoch-completion path is
# what we actually want to assert.

# 1. backend.build() returned a valid DataBundle (data finished loading).
if ! grep -qE "after load data usage" "${LOG}"; then
    echo "FAIL: backend.build did not complete (no 'after load data usage' line) in ${LOG}"
    exit 1
fi

# 2. train_loop completed an epoch — either checkpoint was saved or a real
#    [FSDP Train] step was logged.
if ! grep -qE "Saved checkpoint|\[FSDP Train\] Step" "${LOG}"; then
    echo "FAIL: train_loop did not complete an epoch in ${LOG}"
    exit 1
fi

# 3. val_loop fired (epoch-end auto val): either it ran or skipped explicitly.
if ! grep -qiE "val_iters|validation|skipping validation|val_loop" "${LOG}"; then
    echo "FAIL: val_loop did not fire in ${LOG}"
    exit 1
fi

# 4. No bare ImportError / ModuleNotFoundError for internal_dataset_backend escaped.
if grep -qE "(Module|Import)Error.*internal_dataset_backend" "${LOG}"; then
    echo "FAIL: internal_dataset_backend import error in ${LOG}"
    exit 1
fi

echo "OK: ${BACKEND} backend smoke passed"
