#!/usr/bin/env python3
import yaml
import os
from pathlib import Path
import csv
import tempfile


def check_baseline_dump(
    baseline_path: str,
    id_val: str,
    name_val: str,
    total_time: float,
    total_episodes_run: int,
    total_successes: int,
):
    """
    Compare current eval results against a baseline CSV and dump details to YAML on regression.

    Set output directory via TARSFER_DUMP_PATH.
    Dump only when total_successes decreases or final_success_rate drops.
    """

    if total_episodes_run <= 0:
        print("✅ total_episodes_run=0, no episodes ran; skipping regression check.")
        return

    baseline_path = Path(baseline_path)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    fieldnames = [
        "id",
        "name",
        "total_time",
        "total_episodes_run",
        "total_successes",
        "final_success_rate",
    ]

    baseline_row = None
    with open(baseline_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames) != set(fieldnames):
            raise ValueError(
                f"Baseline header mismatch. expected: {fieldnames}, got: {reader.fieldnames}"
            )
        for row in reader:
            if row["id"] == str(id_val):
                baseline_row = row
                break

    if baseline_row is None:
        raise ValueError(f"No baseline row found for id='{id_val}'")

    try:
        base_total_time = float(baseline_row["total_time"])
        base_episodes = int(baseline_row["total_episodes_run"])
        base_successes = int(baseline_row["total_successes"])
        base_rate_str = baseline_row["final_success_rate"]
        if base_rate_str.endswith("%"):
            base_success_rate = float(base_rate_str.rstrip("%")) / 100.0
        else:
            base_success_rate = (
                float(base_rate_str) / 100.0
                if "." in base_rate_str
                else float(base_rate_str)
            )
    except (ValueError, KeyError) as e:
        raise ValueError(f"Failed to parse baseline row: {e}, row={baseline_row}")

    current_success_rate = (
        total_successes / total_episodes_run if total_episodes_run > 0 else 0.0
    )

    should_dump = False
    messages = []

    if current_success_rate < base_success_rate - 1e-1:
        should_dump = True
        messages.append(
            f"Success rate dropped: baseline={base_success_rate:.4f} ({base_successes}/{base_episodes}) "
            f"→ current={current_success_rate:.4f} ({total_successes}/{total_episodes_run})"
            f"Success count decreased: baseline={base_successes} → current={total_successes}"
        )

    if total_time > base_total_time:
        print(
            f"⚠️  Warning: runtime increased: baseline={base_total_time:.2f}s → current={total_time:.2f}s"
        )

    if not should_dump:
        print("✅ No success-rate regression detected; skipping dump.")
        return

    dump_dir = os.environ.get("TARSFER_DUMP_PATH")
    if not dump_dir:
        raise EnvironmentError("TARSFER_DUMP_PATH is not set; cannot save dump file")

    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_file = dump_dir / f"{id_val}.yml"

    dump_data = {
        "id": id_val,
        "name": name_val,
        "comparison_reason": messages,
        "baseline": {
            "total_time": base_total_time,
            "total_episodes_run": base_episodes,
            "total_successes": base_successes,
            "final_success_rate": base_success_rate,
            "final_success_rate_percent_str": baseline_row["final_success_rate"],
        },
        "current": {
            "total_time": total_time,
            "total_episodes_run": total_episodes_run,
            "total_successes": total_successes,
            "final_success_rate": current_success_rate,
        },
        "regression_detected": True,
    }

    with open(dump_file, "w", encoding="utf-8") as f:
        yaml.dump(dump_data, f, indent=2, allow_unicode=True)

    print(f"🚨 Regression detected! Dumped comparison to: {dump_file}")


def update_baseline(
    baseline_path: str,
    id_val: str,
    name_val: str,
    total_time: float,
    total_episodes_run: int,
    total_successes: int,
):
    """
    Update baseline CSV:
    - Create file with header if missing
    - Update row when id matches
    - Otherwise append a new row
    """
    final_success_rate = (
        total_successes / total_episodes_run if total_episodes_run > 0 else 0
    )
    baseline_path = Path(baseline_path)
    fieldnames = [
        "id",
        "name",
        "total_time",
        "total_episodes_run",
        "total_successes",
        "final_success_rate",
    ]

    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    found = False

    if baseline_path.exists():
        with open(baseline_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if set(reader.fieldnames) != set(fieldnames):
                raise ValueError(
                    f"Baseline file {baseline_path} header mismatch. "
                    f"expected: {fieldnames}, got: {reader.fieldnames}"
                )
            for row in reader:
                if row["id"] == str(id_val):
                    row["name"] = str(name_val)
                    row["total_time"] = f"{total_time:.6f}"
                    row["total_episodes_run"] = str(total_episodes_run)
                    row["total_successes"] = str(total_successes)
                    row["final_success_rate"] = f"{final_success_rate * 100:.2f}%"
                    found = True
                rows.append(row)

    if not found:
        rows.append(
            {
                "id": str(id_val),
                "name": str(name_val),
                "total_time": f"{total_time:.6f}",
                "total_episodes_run": str(total_episodes_run),
                "total_successes": str(total_successes),
                "final_success_rate": f"{final_success_rate * 100:.2f}%",
            }
        )

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv", dir=baseline_path.parent
    ) as tmp_f:
        writer = csv.DictWriter(tmp_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = tmp_f.name

    os.replace(tmp_path, baseline_path)
    print(f"✅ Baseline updated at: {baseline_path}")
