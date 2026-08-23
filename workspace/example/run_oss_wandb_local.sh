#!/bin/bash
# ============================================================================
# Wall-OSS FSDP 训练启动 (PAI DLC / 单机均可)
#
# 用法:
#   CONFIG=workspace/example/arrange_3_flowers_wrc_red.yml \
#     bash workspace/example/run_oss_wandb_local.sh
#
#   DEBUG=1 ...           # 快速冒烟
#   WANDB_OFFLINE=1 ...   # 离线 wandb
#   SKIP_RDMA_INSTALL=1   # 跳过 RDMA 安装
# ============================================================================
set -euo pipefail
nvidia-smi || true

# ── RDMA 用户态驱动库自装 + 探测 ──
if [[ "${SKIP_RDMA_INSTALL:-0}" != "1" && "${SKIP_RDMA_INSTALL:-}" != "true" ]]; then
  (
    echo ""
    echo "===== install RDMA user-space libs ====="
    apt-get update && \
      apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        libnl-3-dev libnl-route-3-dev libnl-3-200 libnl-route-3-200 \
        iproute2 udev dmidecode ethtool && \
      apt-get clean && \
      rm -rf /var/lib/apt/lists/*

    cd /tmp/ && \
      wget -q http://pythonrun.oss-cn-zhangjiakou.aliyuncs.com/rdma/nic-libs-mellanox-rdma-5.2-2/nic-lib-rdma-core-installer-ubuntu.tar.gz && \
      tar xzf nic-lib-rdma-core-installer-ubuntu.tar.gz && \
      cd nic-lib-rdma-core-installer-ubuntu && \
      echo Y | /bin/bash install.sh && \
      cd .. && \
      rm -rf nic-lib-rdma-core-installer-ubuntu nic-lib-rdma-core-installer-ubuntu.tar.gz

    echo ""
    echo "===== RDMA probe ====="
    ls -la /usr/lib/x86_64-linux-gnu/libibverbs.so* 2>&1 | head -5 || echo "libibverbs.so NOT FOUND"
    ls /sys/class/infiniband/ 2>&1 | head -10 || echo "/sys/class/infiniband empty"
    which ibv_devinfo && ibv_devinfo 2>&1 | head -20 || echo "ibv_devinfo not in PATH"
  ) || echo "[RDMA] 安装/探测出错, 已忽略(不中断训练)"
fi

# ── 仓库根目录 (workspace/example/ → 上溯 2 级) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── 配置路径 (env CONFIG > $1 > 默认) ──
CONFIG="${CONFIG:-${1:-arrange_3_flowers_wrc_red.yml}}"
if [[ "${CONFIG}" != *"/"* ]]; then
    CONFIG="workspace/example/${CONFIG}"
fi

# ── conda 环境 ──
CONDA_HOME="${CONDA_HOME:-/mnt/cpfs/zbl-cpfs-new/USERS/zane/miniconda3}"
CONDA_ENV="${CONDA_ENV:-wallx_oss}"
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── 环境变量 ──
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/tmp}"
export CUDA_HOME="${CUDA_HOME:-/mnt/cpfs/zbl-cpfs-new/USERS/zane/cuda-12.6}"
# Wall-OSS 用 repo 内 wall_x；清掉可能污染的 PYTHONPATH
export PYTHONPATH="${REPO_DIR}"

# ── wandb (PAI 内网) ──
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY 2>/dev/null || true
if [[ "${WANDB_OFFLINE:-0}" != "1" && "${WANDB_OFFLINE:-}" != "true" ]]; then
    export WANDB_BASE_URL="${WANDB_BASE_URL:-http://192.168.17.255:30880}"
    export WANDB_API_KEY="${WANDB_API_KEY:-}"
    export WANDB_ENTITY="${WANDB_ENTITY:-x2robot}"
fi

# ── DEBUG=1: 小步数冒烟 ──
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
    echo "[DEBUG] 快速冒烟配置: ${CONFIG}"
fi

# ── 分布式参数 (优先集群注入) ──
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
mkdir -p /mnt/cpfs/zbl-cpfs-new/USERS/zane/wall-oss-05/ckpt/arrange_3_flowers_wrc_red

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
