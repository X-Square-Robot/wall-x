import os
import torch
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cwd = Path(os.path.dirname(os.path.abspath(__file__)))

nvcc_flags = [
    "-std=c++17",  # NOTE: CUTLASS requires c++17
    "-DENABLE_BF16",  # Enable BF16 for cuda_version >= 11
]

env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)

if env_arch_list:
    # Let PyTorch builder to choose device to target for.
    device_capability = ""
else:
    device_capability = torch.cuda.get_device_capability()
    device_capability = f"{device_capability[0]}{device_capability[1]}"

if device_capability:
    nvcc_flags.extend(
        [
            f"--generate-code=arch=compute_{device_capability},code=sm_{device_capability}",
            f"-DGROUPED_GEMM_DEVICE_CAPABILITY={device_capability}",
        ]
    )

ext_modules = [
    CUDAExtension(
        "wallx_csrc",
        [
            "csrc/ops.cu",
            "csrc/dual_asym_grouped_gemm.cu",
            "csrc/permute.cu",
            "csrc/rope.cu",
            "csrc/rope_index.cu",
            "csrc/rot_pos.cu",
            "csrc/window_index.cu",
        ],
        include_dirs=[f"{cwd}/3rdparty/cutlass/include/", f"{cwd}/csrc"],
        extra_compile_args={
            "cxx": ["-fopenmp", "-fPIC", "-Wno-strict-aliasing"],
            "nvcc": nvcc_flags,
        },
    )
]

setup(
    name="wall_x",
    version="1.0.1",
    author="X2Robot Team",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
