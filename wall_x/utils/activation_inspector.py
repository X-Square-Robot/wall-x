import torch
import torch.distributed as dist
from dataclasses import dataclass, asdict
from typing import Any, Dict, List
from collections import defaultdict
import json
import csv


@dataclass
class ActivationRecord:
    idx: int
    module_name: str
    module_class: str
    shape: tuple
    dtype: str
    numel: int
    size_mb: float
    requires_grad: bool


class ActivationInspector:
    """
    Plug-and-play activation inspector using forward hooks.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        only_requires_grad: bool = True,
        min_size_mb: float = 0.0,
        rank0_only: bool = True,
        enabled: bool = True,
    ):
        self.model = model
        self.only_requires_grad = only_requires_grad
        self.min_size_mb = min_size_mb
        self.rank0_only = rank0_only
        self.enabled = enabled

        self.records: List[ActivationRecord] = []
        self._hooks = []
        self._counter = 0

        self._is_rank0 = (
            not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        )

    # -------------------------
    # internal helpers
    # -------------------------
    @staticmethod
    def _sizeof_mb(t: torch.Tensor) -> float:
        return t.numel() * t.element_size() / 1024 / 1024

    def _should_record(self, t: torch.Tensor) -> bool:
        if self.only_requires_grad and not t.requires_grad:
            return False
        if self._sizeof_mb(t) < self.min_size_mb:
            return False
        return True

    def _make_hook(self, module_name: str, module: torch.nn.Module):
        def hook(mod, inputs, outputs):
            if not self.enabled:
                return
            if self.rank0_only and not self._is_rank0:
                return

            def handle(obj: Any):
                if torch.is_tensor(obj):
                    if not self._should_record(obj):
                        return
                    self.records.append(
                        ActivationRecord(
                            idx=self._counter,
                            module_name=module_name,
                            module_class=module.__class__.__name__,
                            shape=tuple(obj.shape),
                            dtype=str(obj.dtype),
                            numel=obj.numel(),
                            size_mb=self._sizeof_mb(obj),
                            requires_grad=obj.requires_grad,
                        )
                    )
                    self._counter += 1
                elif isinstance(obj, (list, tuple)):
                    for o in obj:
                        handle(o)
                elif isinstance(obj, dict):
                    for o in obj.values():
                        handle(o)

            handle(outputs)

        return hook

    # -------------------------
    # context manager API
    # -------------------------
    def __enter__(self):
        if not self.enabled:
            return self

        for name, module in self.model.named_modules():
            h = module.register_forward_hook(self._make_hook(name, module))
            self._hooks.append(h)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # -------------------------
    # reporting utilities
    # -------------------------
    def summary_by_module(self) -> Dict[str, float]:
        summary = defaultdict(float)
        for r in self.records:
            key = f"{r.module_name} ({r.module_class})"
            summary[key] += r.size_mb
        return dict(summary)

    def print_records(self, limit: int | None = None):
        recs = self.records if limit is None else self.records[:limit]
        for r in recs:
            print(
                f"[{r.idx:04d}] {r.module_name}\n"
                f"       class={r.module_class}\n"
                f"       shape={r.shape}\n"
                f"       dtype={r.dtype}\n"
                f"       size={r.size_mb:.2f} MB\n"
                f"       requires_grad={r.requires_grad}\n"
            )

    def print_top_modules(self, topk: int = 20):
        summary = self.summary_by_module()
        items = sorted(summary.items(), key=lambda x: -x[1])

        if topk is not None:
            items = items[:topk]

        total_mb = sum(size for _, size in summary.items())
        shown_mb = sum(size for _, size in items)

        print("\n=== Top activation modules ===")
        for name, size in items:
            print(f"{name:80s} {size:8.2f} MB")

        print("-" * 96)
        print(f"{'TOTAL (all modules)':80s} {total_mb:8.2f} MB")
        if topk is not None:
            print(f"{f'SHOWN TOP-{topk}':80s} {shown_mb:8.2f} MB")

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)

    def to_csv(self, path: str):
        """
        Export ALL activation records to a CSV file.

        Each row corresponds to one activation tensor produced during forward.
        """
        if not self.records:
            print("[ActivationInspector] No activation records to export.")
            return

        fieldnames = [
            "idx",
            "module_name",
            "module_class",
            "shape",
            "dtype",
            "numel",
            "size_mb",
            "requires_grad",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in self.records:
                row = {
                    "idx": r.idx,
                    "module_name": r.module_name,
                    "module_class": r.module_class,
                    # store shape as str for pandas / Excel export
                    "shape": "x".join(map(str, r.shape)),
                    "dtype": r.dtype,
                    "numel": r.numel,
                    "size_mb": round(r.size_mb, 4),
                    "requires_grad": r.requires_grad,
                }
                writer.writerow(row)

        print(
            f"[ActivationInspector] Exported {len(self.records)} activations to {path}"
        )
