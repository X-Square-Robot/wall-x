#!/usr/bin/env bash
# Serve a Wall-X flow checkpoint to RoboDojo (XPolicyLab WebSocket protocol).
#
# Usage:
#   bash scripts/run_robodojo_serving.sh --checkpoint-path /ckpt --norm-key KEY [--port 19000] [...]
#
# All arguments are forwarded to `python -m wall_x.serving.robodojo.serve`;
# run it with --help for the full list. CUDA_ID selects the GPU (default 0).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_ID:-0}"

exec "${PYTHON_BIN:-python}" -m wall_x.serving.robodojo.serve "$@"
