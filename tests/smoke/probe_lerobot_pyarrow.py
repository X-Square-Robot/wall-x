"""Compare ``Dataset.from_parquet(filters=...)`` (HF datasets) against
pure ``pa_ds.dataset(...)`` for the same episode-filter query.

If pure PyArrow is materially faster, the HF wrapper is the bottleneck and
patching ``load_nested_dataset`` to bypass HF datasets is the cheapest fix.
"""

from __future__ import annotations

import time
from pathlib import Path

REPO = Path("/path/to/libero_all/data")


def fmt(s: float) -> str:
    return f"{s:7.2f}s"


def main() -> None:
    import pyarrow.dataset as pa_ds
    from datasets import Dataset

    paths = sorted(REPO.glob("*/*.parquet"))
    print(f"parquet count: {len(paths)}")

    for ep_count in (50, 200, 500, 1000):
        episodes = list(range(ep_count))
        flt = pa_ds.field("episode_index").isin(episodes)

        # --- pure pyarrow dataset API ---
        t0 = time.time()
        tbl = pa_ds.dataset([str(p) for p in paths], format="parquet").to_table(
            filter=flt
        )
        pa_t = time.time() - t0
        pa_rows = tbl.num_rows
        del tbl

        # --- HF Dataset.from_parquet ---
        t0 = time.time()
        hf = Dataset.from_parquet([str(p) for p in paths], filters=flt)
        hf_t = time.time() - t0
        hf_rows = len(hf)
        del hf

        print(
            f"ep={ep_count:>4}  "
            f"pyarrow={fmt(pa_t)} ({pa_rows:>6} rows)  "
            f"hf_datasets={fmt(hf_t)} ({hf_rows:>6} rows)  "
            f"speedup={hf_t / max(pa_t, 1e-6):>5.1f}x"
        )


if __name__ == "__main__":
    main()
