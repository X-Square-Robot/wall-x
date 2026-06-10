"""Base class for operator proxies with lazy backend resolution."""

import importlib
import logging
import threading

logger = logging.getLogger(__name__)


class OpsProxy:
    """Callable proxy that lazily resolves to the best available backend.

    Three-level fallback:
      Level 1: external_accel (fastest, requires external_accel package)
      Level 2: CUDA inline kernel (JIT compiled, requires CUDA)
      Level 3: Pure PyTorch (always available)

    Subclasses define _external_accel_name, _pytorch_fallback, and optionally _cuda_kernel.

    Set _external_accel_name to None to skip external_accel resolution (e.g., when the
    external_accel API is incompatible).
    """

    def __init__(self):
        self._resolved_fn = None
        self._backend = None
        self._resolve_lock = threading.Lock()

    def _resolve(self):
        """Resolve the best available backend. Called once on first use (thread-safe)."""
        if self._resolved_fn is not None:
            return
        with self._resolve_lock:
            # Double-check after acquiring lock
            if self._resolved_fn is not None:
                return

            # Level 1: try external_accel (skip if _external_accel_name is None)
            fn = self._probe_external_accel()
            if fn is not None:
                self._backend = "external_accel"
                self._resolved_fn = fn
                return

            # Level 2: try CUDA inline kernel (subclass override)
            cuda_fn = self._get_cuda_kernel()
            if cuda_fn is not None:
                self._backend = "cuda_inline"
                self._resolved_fn = cuda_fn
                return

            # Level 3: PyTorch fallback
            self._backend = "pytorch"
            self._resolved_fn = self._pytorch_fallback
            logger.info(f"{self.__class__.__name__}: using PyTorch fallback")

    def _get_cuda_kernel(self):
        """Override in subclass to provide Level 2 CUDA kernel. Returns None if unavailable."""
        return None

    @property
    def _external_accel_name(self):
        """Return external_accel attribute name, or None to skip external_accel resolution.

        Base returns None (skip external_accel). Subclasses override to return the
        attribute name when external_accel integration is desired.
        """
        return None

    def _pytorch_fallback(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} has no PyTorch fallback")

    def __call__(self, *args, **kwargs):
        if self._resolved_fn is None:
            self._resolve()
        return self._resolved_fn(*args, **kwargs)

    @property
    def backend(self) -> str:
        if self._resolved_fn is None:
            self._resolve()
        return self._backend

    # ------------------------------------------------------------------
    # Multi-backend API (for testing / benchmarking)
    # ------------------------------------------------------------------

    def _probe_external_accel(self):
        """Return the external_accel function for this op, or None if unavailable."""
        if self._external_accel_name is None:
            return None
        try:
            mod = importlib.import_module("external_accel")
            return getattr(mod, self._external_accel_name)
        except (ImportError, AttributeError):
            return None

    def available_backends(self):
        """Return list of available backend names for this operator."""
        backends = ["pytorch"]
        if self._probe_external_accel() is not None:
            backends.append("external_accel")
        if self._get_cuda_kernel() is not None:
            backends.append("cuda_inline")
        return backends

    def _get_backend_fn(self, backend):
        """Return the callable for a specific backend, or None if unavailable."""
        if backend == "pytorch":
            return self._pytorch_fallback
        elif backend == "external_accel":
            return self._probe_external_accel()
        elif backend == "cuda_inline":
            return self._get_cuda_kernel()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def call_with_backend(self, backend, *args, **kwargs):
        """Call this operator using a specific backend.

        This does NOT change the default backend used by ``__call__``.
        The ``backend`` property still reflects the auto-resolved default.

        Args:
            backend: One of "external_accel", "cuda_inline", "pytorch".

        Returns:
            Operator output from the specified backend.

        Raises:
            RuntimeError: If the requested backend is not available.
        """
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
