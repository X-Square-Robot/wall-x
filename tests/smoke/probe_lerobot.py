"""Time each phase of LeRobotDataset construction at increasing episode counts.

Output is a single timing table that reveals which phase scales with the
number of episodes. Skip 1693 (full) here; that case is profiled live with
py-spy because it deadlocks the smoke harness.

Usage::

    python tests/smoke/probe_lerobot.py
"""

from __future__ import annotations

import gc
import sys
import time

REPO = "/path/to/libero_all"


def fmt(secs: float) -> str:
    return f"{secs:7.2f}s"


def probe(ep_count: int) -> dict[str, float]:
    """Return per-phase wallclock for a given episode subset size."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import load_nested_dataset

    timing: dict[str, float] = {}

    # --- Phase 0: metadata only ---
    t0 = time.time()
    meta = LeRobotDatasetMetadata(REPO, root=None)
    timing["meta"] = time.time() - t0

    episodes = list(range(min(ep_count, meta.total_episodes)))

    # --- Phase 1: load_nested_dataset (the main suspect) ---
    from pathlib import Path

    t0 = time.time()
    hf = load_nested_dataset(
        Path(REPO) / "data",
        features=None,
        episodes=episodes,
    )
    timing["load_nested_dataset"] = time.time() - t0
    timing["n_frames"] = float(len(hf))

    # --- Phase 2: build absolute->relative idx map ---
    t0 = time.time()
    _ = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(hf["index"])}
    timing["abs_to_rel_idx"] = time.time() - t0

    # --- Phase 3: full LeRobotDataset.__init__ ---
    del hf
    gc.collect()
    t0 = time.time()
    ds = LeRobotDataset(
        repo_id=REPO,
        root=None,
        episodes=episodes,
        delta_timestamps=None,
        video_backend="pyav",
    )
    timing["LeRobotDataset.init"] = time.time() - t0
    timing["init_num_frames"] = float(ds.num_frames)

    del ds
    gc.collect()
    return timing


def main() -> None:
    counts = [10, 50, 200, 500, 1000]
    if "--full" in sys.argv:
        counts.append(1693)

    rows = []
    for c in counts:
        print(f"=== probing ep_count={c} ===", flush=True)
        try:
            row = probe(c)
            row["ep_count"] = c
            rows.append(row)
            for k, v in row.items():
                print(f"  {k:30s} {v}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            rows.append({"ep_count": c, "error": str(e)})

    print()
    print("=" * 80)
    print(
        f"{'ep':>5} {'meta':>9} {'load_nested':>12} {'abs->rel':>10} "
        f"{'LR.init':>9} {'frames':>8}"
    )
    print("=" * 80)
    for r in rows:
        if "error" in r:
            print(f"{r['ep_count']:>5} ERROR {r['error']}")
            continue
        print(
            f"{int(r['ep_count']):>5} "
            f"{fmt(r['meta'])} "
            f"{fmt(r['load_nested_dataset'])} "
            f"{fmt(r['abs_to_rel_idx'])} "
            f"{fmt(r['LeRobotDataset.init'])} "
            f"{int(r['n_frames']):>8}"
        )


if __name__ == "__main__":
    main()
