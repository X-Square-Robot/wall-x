"""Time each step inside LeRobotDataset.__init__ via subclass instrumentation.

Reveals whether the long init time comes from load_hf_dataset, the cache
check (.unique()), the abs->rel idx build, or the metadata re-init.
"""

from __future__ import annotations

import time

REPO = "/path/to/libero_all"


def fmt(s):
    return f"{s:7.2f}s"


def make_instrumented(LeRobotDataset):
    timings = {}

    orig_load_hf = LeRobotDataset.load_hf_dataset
    orig_check = LeRobotDataset._check_cached_episodes_sufficient

    def timed_load_hf(self):
        t0 = time.time()
        out = orig_load_hf(self)
        timings.setdefault("load_hf_dataset", []).append(time.time() - t0)
        return out

    def timed_check(self):
        t0 = time.time()
        out = orig_check(self)
        timings.setdefault("_check_cached", []).append(time.time() - t0)
        return out

    LeRobotDataset.load_hf_dataset = timed_load_hf
    LeRobotDataset._check_cached_episodes_sufficient = timed_check

    return timings


def probe(ep_count):
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

    print(f"\n=== ep_count={ep_count} ===", flush=True)

    timings = make_instrumented(LeRobotDataset)
    timings.clear()

    # Phase 0: meta only (probe-side)
    t0 = time.time()
    meta = LeRobotDatasetMetadata(REPO, root=None)
    print(f"  [probe] meta(no rev)         {fmt(time.time() - t0)}")

    episodes = list(range(min(ep_count, meta.total_episodes)))

    # Phase 1: full init with timing hooks
    t0 = time.time()
    ds = LeRobotDataset(
        repo_id=REPO,
        root=None,
        episodes=episodes,
        delta_timestamps=None,
        video_backend="pyav",
    )
    total = time.time() - t0
    print(f"  total LeRobotDataset.__init__{fmt(total)}  rows={ds.num_frames}")

    for k, v_list in timings.items():
        v = sum(v_list)
        print(f"    └─ {k:25s} {fmt(v)}  (calls={len(v_list)})")

    # Phase 2: time the abs->rel idx build (post-init, on the same hf_dataset)
    t0 = time.time()
    _ = {
        abs_idx.item() if hasattr(abs_idx, "item") else abs_idx: rel_idx
        for rel_idx, abs_idx in enumerate(ds.hf_dataset["index"])
    }
    print(f"  [probe] abs->rel rebuild     {fmt(time.time() - t0)}")

    # Phase 3: hf_dataset.unique on its own
    t0 = time.time()
    _ = ds.hf_dataset.unique("episode_index")
    print(f"  [probe] unique('episode_idx'){fmt(time.time() - t0)}")

    # Phase 4: arrow column access cost
    t0 = time.time()
    col = ds.hf_dataset["index"]
    print(
        f"  [probe] hf_dataset['index']  {fmt(time.time() - t0)}  type={type(col).__name__}"
    )


def main():
    for c in [10, 50, 200]:
        probe(c)


if __name__ == "__main__":
    main()
