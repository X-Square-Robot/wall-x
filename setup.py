from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CSRC = Path("wall_x") / "model" / "core" / "ops" / "csrc"

# PyTorch extensions disable the CUDA half/bfloat16 operators by default; the
# kernels use them, so re-enable them.
_HALF_OPERATORS = [
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
]
FAST = ["-O3", "-DNDEBUG", "--use_fast_math", "-std=c++17", *_HALF_OPERATORS]
# rope and pos_embed_interp must round exactly like PyTorch: no FMA contraction,
# exact division/sqrt, denormals kept.
EXACT = [*FAST, "--fmad=false", "-prec-div=true", "-prec-sqrt=true", "--ftz=false"]


def extension(name, binding, nvcc_flags):
    if not (CSRC / binding).exists():
        raise RuntimeError(f"CUDA operator sources are missing: {CSRC / binding}")
    return CUDAExtension(
        name=f"wall_x.model.core.ops.{name}",
        sources=[str(CSRC / binding)],
        include_dirs=[str(CSRC.resolve()), str((CSRC / "common").resolve())],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": nvcc_flags},
    )


setup(
    name="wall_x",
    packages=find_packages(),
    ext_modules=[
        extension("_cuda_ext_bin", "binding.cu", FAST),
        extension("_cuda_ext_exact_bin", "binding_exact.cu", EXACT),
    ],
    cmdclass={"build_ext": BuildExtension},
)
