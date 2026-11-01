"""Install-time compiled CUDA kernels for wall_x ops.

Two extension modules are built by setup.py: ``_cuda_ext_bin`` (``-O3
--use_fast_math``) and ``_cuda_ext_exact_bin`` (IEEE rounding, for the rope and
position-embedding kernels that must match PyTorch bit for bit). ``load()``
returns one namespace over both.
"""

_module = None


class _Kernels:
    def __init__(self, *modules):
        self._modules = modules

    def __getattr__(self, name):
        for module in self._modules:
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(name)


def load():
    """Load the CUDA extension modules built by setup.py."""
    global _module
    if _module is not None:
        return _module

    try:
        from wall_x.model.core.ops import _cuda_ext_bin, _cuda_ext_exact_bin
    except ImportError as exc:
        raise ImportError(
            "Wall-X CUDA operators were not built. Install requirements first, "
            "then reinstall Wall-X with: "
            "MAX_JOBS=8 pip install --no-build-isolation -e ."
        ) from exc

    _module = _Kernels(_cuda_ext_bin, _cuda_ext_exact_bin)
    return _module


def is_available() -> bool:
    """Check whether the install-time CUDA extensions can be imported."""
    try:
        load()
        return True
    except Exception:
        return False
