"""GPU hardware specs shared by all per-model MFU calculators.

Peak dense BF16 TFLOPS (no sparsity) and memory bandwidth per device
family. Matched by substring against ``torch.cuda.get_device_name()``.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch

PEAK_TFLOPS = {
    "A100": 312,
    "A800": 312,
    "H100": 989,
    "H800": 989,
    "H20": 148,
    "A30": 165,
    "A40": 150,
    "L40S": 362,
    "L40": 362,
    "4090": 330,
    "3090": 142,
    "B200": 2250,
    "GB200": 2250,
}

MEM_BW_GBS = {
    "A100": 2039,
    "A800": 2039,
    "H100": 3350,
    "H800": 3350,
    "H20": 4000,
    "A30": 933,
    "A40": 696,
    "L40S": 864,
    "L40": 864,
    "4090": 1008,
    "3090": 936,
    "B200": 8000,
    "GB200": 8000,
}

# Fallback when the device name matches nothing: A800 numbers (the fleet
# default), same behavior as the previous in-adapter table.
_DEFAULT_PEAK_TFLOPS = 312
_DEFAULT_MEM_BW_GBS = 2039


class GpuSpecs(NamedTuple):
    gpu_name: str
    peak_flops: float  # FLOPs/s
    mem_bw_gbs: float

    @property
    def ops_byte_ratio(self) -> float:
        return self.peak_flops / (self.mem_bw_gbs * 1e9)


def resolve_gpu_specs(gpu_name: Optional[str] = None) -> GpuSpecs:
    """Look up peak FLOPS / memory bandwidth for a GPU.

    ``gpu_name`` defaults to the current CUDA device's name (empty string
    when CUDA is unavailable, which resolves to the A800 fallback).
    """
    if gpu_name is None:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    peak_tflops = _DEFAULT_PEAK_TFLOPS
    mem_bw = _DEFAULT_MEM_BW_GBS
    for key, tflops in PEAK_TFLOPS.items():
        if key in gpu_name:
            peak_tflops = tflops
            mem_bw = MEM_BW_GBS.get(key, _DEFAULT_MEM_BW_GBS)
            break
    return GpuSpecs(gpu_name, peak_tflops * 1e12, mem_bw)
