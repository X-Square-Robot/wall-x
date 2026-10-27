"""Per-model MFU calculators + shared GPU hardware specs.

One module per model family (``vga_mfu``; qwen still lives in its
adapters). ``hardware`` holds the shared peak-FLOPS / bandwidth tables.
"""

from wall_x.trainer.metrics.mfu.hardware import GpuSpecs, resolve_gpu_specs

__all__ = ["GpuSpecs", "resolve_gpu_specs"]
