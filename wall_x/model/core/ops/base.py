"""Base class for operator proxies with lazy backend resolution."""

import logging
import threading

logger = logging.getLogger(__name__)


class OpsProxy:
    """Callable proxy that lazily resolves to the best available backend.

    Public fallback order:
      Level 1: CUDA inline kernel, when available
      Level 2: Pure PyTorch fallback
    """

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

    def available_backends(self):
        """Return available backend names for this operator."""
        backends = []
        if self._get_cuda_kernel() is not None:
            backends.append("cuda_inline")
        if self._get_pytorch_fallback() is not None:
            backends.append("pytorch")
        return backends

    def _get_backend_fn(self, backend):
        """Return the callable for a specific backend, or None if unavailable."""
        if backend == "pytorch":
            return self._get_pytorch_fallback()
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
