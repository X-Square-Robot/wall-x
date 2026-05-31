#!/usr/bin/env bash
set -euo pipefail

# LIBERO evaluation launcher. Configure with environment variables:
#
# Single suite (default):
#   CHECKPOINT_PATH=/path/to/checkpoint bash scripts/run_libero.sh
#   CHECKPOINT_PATH=/path/to/checkpoint SMOKE=1 bash scripts/run_libero.sh
#   CONFIG=/path/to/eval.yaml bash scripts/run_libero.sh
#
# All suites (spatial -> object -> goal -> libero_10):
#   bash scripts/run_libero.sh --all
#   ALL_SUITES=1 bash scripts/run_libero.sh
#   CHECKPOINT_PATH=... TRAIN_CONFIG_PATH=... bash scripts/run_libero.sh --all
#   EVAL_ROOT=/path/to/output bash scripts/run_libero.sh --all
#   TASK_SUITES="libero_object libero_goal" bash scripts/run_libero.sh --all
#
# Optional knobs:
#   CUDA_ID=0
#   DRIVER_MODE=in_process
#   NUM_WORKERS=1
#   MAX_BATCH_SIZE=1
#   TASK_SUITE_NAME=libero_spatial
#   NUM_TRIALS_PER_TASK=50
#   TASK_INDICES=0,1,2
#   NORM_KEY=libero_all
#   LOG_DIR=/path/to/harrix_libero_eval
#   ROLLOUT_DIR=./libero_eval/rollouts   # save MP4 replays per episode


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/LIBERO}"

ALL_SUITES="${ALL_SUITES:-0}"
for arg in "$@"; do
    case "${arg}" in
        --all|--all-suites)
            ALL_SUITES=1
            ;;
    esac
done

setup_pythonpath() {
    export CUDA_VISIBLE_DEVICES="${CUDA_ID:-0}"
    if [[ -d "${REPO_ROOT}/third_party/harrix/python" ]]; then
        export PYTHONPATH="${REPO_ROOT}/third_party/harrix/python:${PYTHONPATH:-}"
    fi
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    if [[ -d "${LIBERO_PYTHON}" ]]; then
        export PYTHONPATH="${PYTHONPATH}:${LIBERO_PYTHON}"
    fi
}

build_infer_args() {
    local -n _args=$1

    _args=()
    if [[ -n "${CONFIG:-}" ]]; then
        _args+=(--config "${CONFIG}")
    else
        if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
            echo "CHECKPOINT_PATH is required when CONFIG is not set." >&2
            exit 2
        fi
        _args+=(--checkpoint-path "${CHECKPOINT_PATH}")
    fi

    if [[ -n "${TRAIN_CONFIG_PATH:-}" ]]; then
        _args+=(--train-config-path "${TRAIN_CONFIG_PATH}")
    fi
    if [[ "${SMOKE:-0}" == "1" ]]; then
        _args+=(--smoke)
    fi
    if [[ "${DET:-0}" == "1" ]]; then
        _args+=(--deterministic-model)
    fi
    if [[ -n "${TASK_INDICES:-}" ]]; then
        _args+=(--task-indices "${TASK_INDICES}")
    fi

    _args+=(--driver-mode "${DRIVER_MODE:-in_process}")
    _args+=(--num-workers "${NUM_WORKERS:-1}")
    _args+=(--max-batch-size "${MAX_BATCH_SIZE:-1}")
    _args+=(--task-suite-name "${TASK_SUITE_NAME:-libero_spatial}")
    _args+=(--num-trials-per-task "${NUM_TRIALS_PER_TASK:-50}")
    _args+=(--norm-key "${NORM_KEY:-libero_all}")
    _args+=(--log-dir "${LOG_DIR:-/path/to/harrix_libero_eval}")

    if [[ -n "${ROLLOUT_DIR:-}" ]]; then
        _args+=(--rollout-dir "${ROLLOUT_DIR}")
    elif [[ -n "${WALLX_ROLLOUT_DIR:-}" ]]; then
        _args+=(--rollout-dir "${WALLX_ROLLOUT_DIR}")
    fi
    if [[ "${DISABLE_ROLLOUT:-0}" == "1" ]]; then
        _args+=(--disable-rollout)
    fi
}

run_single_suite() {
    local args=()
    build_infer_args args

    cd "${REPO_ROOT}"
    python scripts/infer_libero.py "${args[@]}"
}

append_suite_summary() {
    local suite="$1"
    local report_path="$2"
    local summary_json="$3"

    python3 - "${suite}" "${report_path}" "${summary_json}" <<'PY'
import json, sys
from pathlib import Path

suite, report_path, summary_path = sys.argv[1:4]
with open(report_path) as f:
    report = json.load(f)

entry = {
    "task_suite": suite,
    "report_path": report_path,
    "overall": report.get("overall", {}),
    "per_task": report.get("per_task", {}),
}
summary = Path(summary_path)
records = []
if summary.exists():
    with open(summary) as f:
        records = json.load(f)
records.append(entry)
with open(summary, "w") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

overall = entry["overall"]
print(
    f"[{suite}] success_rate={overall.get('success_rate', 0)*100:.2f}% "
    f"({overall.get('successes', '?')}/{overall.get('attempted', '?')})"
)
PY
}

write_final_summary() {
    local summary_json="$1"
    local summary_txt="$2"

    python3 - "${summary_json}" "${summary_txt}" <<'PY'
import json, sys
from pathlib import Path

summary_json, summary_txt = sys.argv[1:3]
path = Path(summary_json)
if not path.exists():
    print("No suite finished successfully; summary not written.")
    sys.exit(0)

with open(path) as f:
    records = json.load(f)

lines = ["LIBERO sequential eval summary", "=" * 40]
rates = []
for rec in records:
    suite = rec["task_suite"]
    o = rec.get("overall", {})
    rate = float(o.get("success_rate", 0.0))
    rates.append(rate)
    lines.append(
        f"{suite:16s}  {rate*100:6.2f}%  "
        f"({o.get('successes', '?')}/{o.get('attempted', '?')})"
    )

if rates:
    avg = sum(rates) / len(rates)
    lines.append("-" * 40)
    lines.append(f"{'average':16s}  {avg*100:6.2f}%  ({len(rates)} suites)")

text = "\n".join(lines) + "\n"
Path(summary_txt).write_text(text)
print(text, end="")
PY
}

run_all_suites() {
    export WALLX_SMOOTH_ACTION="${WALLX_SMOOTH_ACTION:-false}"
    export WALLX_SMOOTH_GRIPPER="${WALLX_SMOOTH_GRIPPER:-false}"

    export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/path/to/checkpoint}"
    export TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-/path/to/libero.yml}"

    local run_ts
    run_ts="$(date +%Y%m%d_%H%M%S)"
    local eval_root="${EVAL_ROOT:-${REPO_ROOT}/libero_eval3/run_${run_ts}}"
    local rollout_root="${eval_root}/rollouts"
    local console_dir="${eval_root}/console"
    local log_root="${eval_root}/logs"

    mkdir -p "${rollout_root}" "${console_dir}" "${log_root}"

    local -a suites
    if [[ -n "${TASK_SUITES:-}" ]]; then
        # shellcheck disable=SC2206
        suites=(${TASK_SUITES})
    else
        suites=(libero_spatial libero_object libero_goal libero_10)
    fi

    local summary_json="${eval_root}/summary.json"
    local summary_txt="${eval_root}/summary.txt"

    echo "Eval root: ${eval_root}"
    echo "Checkpoint: ${CHECKPOINT_PATH}"
    echo "Suites (sequential): ${suites[*]}"
    echo

    setup_pythonpath

    local suite
    for suite in "${suites[@]}"; do
        echo "========== [$(date '+%F %T')] start ${suite} =========="

        export TASK_SUITE_NAME="${suite}"
        export LOG_DIR="${log_root}/${suite}"
        export WALLX_ROLLOUT_DIR="${rollout_root}"
        mkdir -p "${LOG_DIR}"

        local console_log="${console_dir}/${suite}.log"
        set +e
        run_single_suite 2>&1 | tee "${console_log}"
        local exit_code=${PIPESTATUS[0]}
        set -e

        if [[ ${exit_code} -ne 0 ]]; then
            echo "[ERROR] ${suite} failed with exit code ${exit_code}" | tee -a "${eval_root}/errors.log"
            continue
        fi

        if [[ -f "${LOG_DIR}/report.json" ]]; then
            append_suite_summary "${suite}" "${LOG_DIR}/report.json" "${summary_json}"
        else
            echo "[WARN] missing report.json for ${suite}"
        fi
        echo
    done

    write_final_summary "${summary_json}" "${summary_txt}"

    echo "Done."
    echo "  Per-suite logs : ${log_root}/<suite>/{state.jsonl,report.json}"
    echo "  Rollout videos : ${rollout_root}/<suite>/"
    echo "  Console logs   : ${console_dir}/"
    echo "  Aggregated     : ${summary_json}"
    echo "                   ${summary_txt}"
}

if [[ "${ALL_SUITES}" == "1" ]]; then
    run_all_suites
else
    setup_pythonpath
    run_single_suite
fi
