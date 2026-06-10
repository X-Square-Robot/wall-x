from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
OPS_DIR = Path("wall_x") / "model" / "core" / "ops"
CSRC_DIR = OPS_DIR / "csrc"

PUBLIC_SCRIPTS = [
    "scripts/compute_norm_stats.py",
    "scripts/draw_openloop_plot.py",
    "scripts/fake_inference.py",
    "scripts/infer_libero.py",
    "scripts/merge_sharded_weights.py",
    "scripts/merge_tokenizer.py",
    "scripts/run_libero.sh",
    "scripts/run_serving.sh",
]


def read_readme() -> str:
    readme = ROOT / "README.md"
    return readme.read_text(encoding="utf-8") if readme.exists() else ""


def build_ext_modules():
    binding = CSRC_DIR / "binding.cu"
    if not (ROOT / binding).exists():
        raise RuntimeError(
            "CUDA operator sources are missing. The OSS export must include "
            "wall_x/model/core/ops/csrc/binding.cu."
        )

    cuda_sources = [binding] + sorted(
        path for path in (ROOT / CSRC_DIR).rglob("*.cu") if path.name != "binding.cu"
    )
    cuda_sources = [
        path if path == binding else path.relative_to(ROOT) for path in cuda_sources
    ]

    return [
        CUDAExtension(
            name="wall_x.model.core.ops._cuda_ext_bin",
            sources=[str(path) for path in cuda_sources],
            include_dirs=[str(CSRC_DIR), str(CSRC_DIR / "common")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
            },
        )
    ]


setup(
    name="wall_x",
    version="1.1.0",
    description="Training and inference code for WALL open-source embodied models.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="X-Square Robot",
    url="https://github.com/X-Square-Robot/wall-x",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "tests.*")),
    scripts=PUBLIC_SCRIPTS,
    ext_modules=build_ext_modules(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
