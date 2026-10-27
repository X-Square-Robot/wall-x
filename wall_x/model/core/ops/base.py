"""Base class for operator proxies with lazy backend resolution."""

import logging
import threading

logger = logging.getLogger(__name__)

_kernels = None
_kernels_loaded = False
_kernels_lock = threading.Lock()


def load_kernels():
    """The compiled wall_x CUDA operators (``_cuda_wrappers``), or None.

    They are built by ``setup.py`` from ``csrc/``. Without a build (or when the
    build fails to load) every operator runs its PyTorch fallback.
    """
    global _kernels, _kernels_loaded
    if _kernels_loaded:
        return _kernels
    with _kernels_lock:
        if not _kernels_loaded:
            try:
                from wall_x.model.core.ops import _cuda_ext

                if _cuda_ext.is_available():
                    from wall_x.model.core.ops import _cuda_wrappers

                    _kernels = _cuda_wrappers
                else:
                    logger.info("wall_x CUDA operators not built; using PyTorch fallbacks")
            except Exception as exc:  # built but unusable: fall back, but say so
                logger.warning("wall_x CUDA operators failed to load: %s", exc)
                _kernels = None
            _kernels_loaded = True
    return _kernels


class OpsProxy:
    """Callable proxy that lazily resolves to the best available backend.

    Fallback order:
      Level 1: compiled CUDA operator (``csrc/``), looked up by
               ``_external_accel_name`` and adapted by ``_probe_external_accel``
      Level 2: CUDA kernel a subclass provides through ``_get_cuda_kernel``
      Level 3: Pure PyTorch fallback
    """

    #: Name of this operator in the compiled operator module; None = no kernel.
    _external_accel_name = None

    def __init__(self):
        self._resolved_fn = None
        self._backend = None
        self._resolve_lock = threading.Lock()

    def _resolve(self):
        """Resolve the best available backend. Called once on first use."""
        if self._resolved_fn is not None:
            return
        with self._resolve_lock:
            if self._resolved_fn is not None:
                return

            kernel_fn = self._probe_external_accel()
            if kernel_fn is not None:
                self._backend = "cuda"
                self._resolved_fn = kernel_fn
                return

            cuda_fn = self._get_cuda_kernel()
            if cuda_fn is not None:
                self._backend = "cuda_inline"
                self._resolved_fn = cuda_fn
                return

            pytorch_fn = self._get_pytorch_fallback()
            if pytorch_fn is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}: no available operator backend. "
                    f"Available: {self.available_backends()}"
                )
            self._backend = "pytorch"
            self._resolved_fn = pytorch_fn
            logger.info("%s: using PyTorch fallback", self.__class__.__name__)

    def _probe_external_accel(self):
        """The compiled kernel for this operator, or None.

        Subclasses override this to adapt the kernel's signature; they call
        ``super()._probe_external_accel()`` for the raw kernel.
        """
        name = self._external_accel_name
        if not name:
            return None
        return getattr(load_kernels(), name, None)

    def _get_cuda_kernel(self):
        """Override in subclasses to provide a CUDA inline kernel."""
        return None

    def _pytorch_fallback(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} has no PyTorch fallback")

    def _get_pytorch_fallback(self):
        """Return the PyTorch fallback only when a subclass implements one."""
        if type(self)._pytorch_fallback is OpsProxy._pytorch_fallback:
            return None
        return self._pytorch_fallback

    def __call__(self, *args, **kwargs):
        if self._resolved_fn is None:
            self._resolve()
        return self._resolved_fn(*args, **kwargs)

    @property
    def backend(self) -> str:
        if self._resolved_fn is None:
            self._resolve()
        return self._backend

    def has_external_accel(self) -> bool:
        """Public capability query: is a CUDA kernel available for this op?"""
        return self._probe_external_accel() is not None or self._get_cuda_kernel() is not None

    def available_backends(self):
        """Return available backend names for this operator."""
        backends = []
        if self._probe_external_accel() is not None:
            backends.append("cuda")
        if self._get_cuda_kernel() is not None:
            backends.append("cuda_inline")
        if self._get_pytorch_fallback() is not None:
            backends.append("pytorch")
        return backends

    def _get_backend_fn(self, backend):
        """Return the callable for a specific backend, or None if unavailable."""
        if backend == "pytorch":
            return self._get_pytorch_fallback()
        if backend == "cuda":
            return self._probe_external_accel()
        if backend == "cuda_inline":
            return self._get_cuda_kernel()
        raise ValueError(f"Unknown backend: {backend}")

    def call_with_backend(self, backend, *args, **kwargs):
        """Call this operator using a specific backend."""
        fn = self._get_backend_fn(backend)
        if fn is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: backend '{backend}' not available. "
                f"Available: {self.available_backends()}"
            )
        return fn(*args, **kwargs)

    def __repr__(self):
        backend = self._backend or "unresolved"
        return f"<{self.__class__.__name__} backend={backend}>"
