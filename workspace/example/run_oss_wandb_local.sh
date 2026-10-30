#!/bin/bash
# ============================================================================
# Launch Wall-OSS FSDP training (single node or multi-node).
#
# Usage:
#   CONDA_HOME=/path/to/miniconda3 \
#   CONFIG=workspace/example/arrange_3_flowers_wrc_red.yml \
#     bash workspace/example/run_oss_wandb_local.sh
#
#   DEBUG=1 ...           # short smoke run
#   WANDB_OFFLINE=1 ...   # offline wandb
#
# Notes:
#   Install multi-node RDMA / InfiniBand in the cluster image. This script no longer downloads private packages.
# ============================================================================
set -euo pipefail
nvidia-smi || true

# ── Repository root (workspace/example/ → two levels up) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Config path (env CONFIG > $1 > default) ──
CONFIG="${CONFIG:-${1:-arrange_3_flowers_wrc_red.yml}}"
if [[ "${CONFIG}" != *"/"* ]]; then
    CONFIG="workspace/example/${CONFIG}"
fi

# ── conda environment ──
CONDA_HOME="${CONDA_HOME:?Set CONDA_HOME to your miniconda root}"
CONDA_ENV="${CONDA_ENV:-wallx}"
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── environment variables ──
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/tmp}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# Wall-OSS imports wall_x from this repo; drop any stale PYTHONPATH.
export PYTHONPATH="${REPO_DIR}"

# ── wandb (optional; configured through environment variables) ──
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY 2>/dev/null || true
if [[ "${WANDB_OFFLINE:-0}" != "1" && "${WANDB_OFFLINE:-}" != "true" ]]; then
    # Set WANDB_BASE_URL / WANDB_API_KEY / WANDB_ENTITY if using a custom wandb server.
    export WANDB_BASE_URL="${WANDB_BASE_URL:-}"
    export WANDB_API_KEY="${WANDB_API_KEY:-}"
    export WANDB_ENTITY="${WANDB_ENTITY:-}"
fi

# ── DEBUG=1: short smoke run ──
if [[ "${DEBUG:-0}" == "1" || "${DEBUG:-}" == "true" ]]; then
    _DBG_CONFIG="/tmp/oss_debug_$(basename "${CONFIG%.*}").yml"
    python - "${CONFIG}" "${_DBG_CONFIG}" <<'PY'
import sys, os, yaml
src, dst = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open(src))
hp = c.setdefault("hyperparams", {})
sch = hp.setdefault("scheduler", {})
sch["num_training_steps"] = 30
sch["num_warmup_steps"] = 2
hp["num_epoch"] = 1
log = c.setdefault("logging", {})
log["save_interval"] = 10
log["val_interval"] = 1000000
log["epoch_save_interval"] = 1
log["log_name"] = "DEBUG_" + str(log.get("log_name", "run"))
ck = c.setdefault("checkpoint", {})
if ck.get("save_path"):
    ck["save_path"] = os.path.join(
        os.path.dirname(ck["save_path"]), "DEBUG_" + os.path.basename(ck["save_path"])
    )
yaml.safe_dump(c, open(dst, "w"), allow_unicode=True, sort_keys=False)
PY
    CONFIG="${_DBG_CONFIG}"
    echo "[DEBUG] smoke config: ${CONFIG}"
fi

# ── distributed args (prefer values injected by the cluster) ──
if [[ -n "${NPROC_PER_NODE:-}" ]]; then :;
elif [[ -n "${LOCAL_WORLD_SIZE:-}" ]]; then NPROC_PER_NODE="${LOCAL_WORLD_SIZE}";
elif [[ -n "${TOTAL_PROCS:-}" && -n "${WORLD_SIZE:-}" ]]; then NPROC_PER_NODE=$((TOTAL_PROCS / WORLD_SIZE));
else NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l); : "${NPROC_PER_NODE:=8}"; fi
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-41119}"

EXTRA_ARGS=()
[[ "${WANDB_OFFLINE:-0}" == "1" || "${WANDB_OFFLINE:-}" == "true" ]] && EXTRA_ARGS+=("--wandb_offline" "true")

echo "================================================================"
echo "  REPO           = ${REPO_DIR}"
echo "  CONDA_ENV      = ${CONDA_ENV}"
echo "  CONFIG         = ${CONFIG}"
echo "  NPROC_PER_NODE = ${NPROC_PER_NODE} | NNODES = ${NNODES} | NODE_RANK = ${NODE_RANK}"
echo "  MASTER         = ${MASTER_ADDR}:${MASTER_PORT}"
echo "================================================================"

cd "${REPO_DIR}"
mkdir -p "${CKPT_ROOT:-/path/to/ckpt}/arrange_3_flowers_wrc_red"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    wall_x/trainer/fsdp_trainer/train_fsdp.py \
    --config "${CONFIG}" \
    --log_to_file \
    "${EXTRA_ARGS[@]}"
