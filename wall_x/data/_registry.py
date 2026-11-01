"""Data backend registry + multi-verb dispatch.

Each backend registers an entire **module** (not just a single function).
Consumers obtain a backend via :func:`backend_for` and then call any
operation the backend supports (``backend.build`` or optional helper
operations published by that backend). Operations a backend does *not*
implement raise
:class:`MissingOperationError` with a list of backends that do.

Three failure shapes, each pointing at the right next step:

- **Unknown name** (typo / never registered): :class:`KeyError`
  ``Unknown backend 'foo_bar'. Known: [...]``
- **Known but failed to import** (optional dependency missing):
  :class:`RuntimeError`
  ``Backend 'example' failed to import: <ImportError>. Install its
  dependency or switch to one of [...]``
- **Backend exists but operation missing**: :class:`MissingOperationError`
  ``Backend 'example' does not implement 'make_processor'.
  Supported by: ['other_backend']``
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Any, Dict

from wall_x.data._bundle import DataBundle
from wall_x.data._protocol import BuildContext

logger = logging.getLogger(__name__)

_BACKENDS: Dict[str, ModuleType] = {}
_import_errors: Dict[str, BaseException] = {}

# The single, process-wide active backend. Set exactly once at config load
# time by :func:`_set_data_backend`; all consumer code reads it via
# :func:`data_backend`. Keeping this as module-level state (rather than
# threading cfg through every callsite) is the whole point of the design.
_DATA_BACKEND: str | None = None


# --- Registration --------------------------------------------------------


def register_module(dataset_type: str, module: ModuleType) -> None:
    """Register a backend module under ``dataset_type``.

    The module must expose a ``build(cfg, ctx) -> DataBundle`` callable.
    Other optional operations are
    discovered lazily via ``getattr`` when ``backend_for(name).<op>`` is
    accessed; backends only declare what they support.
    """
    if not hasattr(module, "build"):
        raise TypeError(
            f"Backend module for {dataset_type!r} must expose a "
            f"``build(cfg, ctx) -> DataBundle`` callable; got {module!r}."
        )
    if dataset_type in _BACKENDS:
        logger.warning("overwriting existing backend registration for %r", dataset_type)
    _BACKENDS[dataset_type] = module


def register(dataset_type: str, build_callable) -> None:
    """Legacy single-callable registration. Wraps ``build_callable`` in a
    minimal module so the new dispatch path still works.

    New backends should prefer :func:`register_module` so they can publish
    multiple operations.
    """
    shim = ModuleType(f"_legacy_backend_shim_{dataset_type}")
    shim.build = build_callable  # type: ignore[attr-defined]
    register_module(dataset_type, shim)


def record_import_error(dataset_type: str, error: BaseException) -> None:
    """Stash the exception that prevented a backend from registering."""
    _import_errors[dataset_type] = error


def available_backends() -> list[str]:
    """Return the names of currently-registered backends."""
    return sorted(_BACKENDS)


# --- Lookup --------------------------------------------------------------


class MissingOperationError(NotImplementedError):
    """Raised when a backend module does not implement a requested op."""

    def __init__(self, backend_name: str, op_name: str) -> None:
        impls = [n for n, mod in _BACKENDS.items() if hasattr(mod, op_name)]
        if impls:
            hint = f"Supported by: {impls}."
        else:
            hint = (
                f"No registered backend implements {op_name!r} — check the "
                f"spelling or add it to the backend module."
            )
        super().__init__(
            f"Backend {backend_name!r} does not implement {op_name!r}. {hint}"
        )
        self.backend_name = backend_name
        self.op_name = op_name


class _BackendProxy:
    """Thin wrapper that forwards ``proxy.<op>(...)`` to the backend module.

    Wrapping (instead of returning the module directly) lets us emit
    ``MissingOperationError`` with a useful "supported by" hint instead
    of plain ``AttributeError``.
    """

    __slots__ = ("_name", "_module")

    def __init__(self, name: str, module: ModuleType) -> None:
        self._name = name
        self._module = module

    def __repr__(self) -> str:
        return f"<BackendProxy name={self._name!r}>"

    def __getattr__(self, op: str):
        attr = getattr(self._module, op, None)
        if attr is None:
            raise MissingOperationError(self._name, op)
        return attr

    def supports(self, op: str) -> bool:
        """Cheap predicate: does this backend implement ``op``?"""
        return hasattr(self._module, op)


def backend_for(name: str) -> _BackendProxy:
    """Look up a backend by ``dataset_type`` name.

    Raises :class:`RuntimeError` (with chained ImportError) if the
    backend is known but failed to import; :class:`KeyError` if the
    name was never registered.
    """
    if name in _BACKENDS:
        return _BackendProxy(name, _BACKENDS[name])
    if name in _import_errors:
        err = _import_errors[name]
        raise RuntimeError(
            f"Backend {name!r} failed to import: {err}. "
            f"Install its dependency or switch to one of "
            f"{available_backends()}."
        ) from err
    raise KeyError(f"Unknown backend {name!r}. Known: {available_backends()}.")


# --- Convenience helpers -------------------------------------------------


def resolve_dataset_type(cfg_or_yaml: Any, default: str = "lerobot") -> str:
    """Pull ``dataset_type`` out of either a typed TrainConfig or a raw yaml dict.

    Lookup order:
    1. ``cfg.data.dataset_type`` (typed TrainConfig)
    2. ``yaml_dict["dataset_type"]`` (legacy flat yaml)
    3. ``default``
    """
    data = getattr(cfg_or_yaml, "data", None)
    if data is not None:
        v = getattr(data, "dataset_type", None)
        if v:
            return v
    if isinstance(cfg_or_yaml, dict):
        v = cfg_or_yaml.get("dataset_type")
        if v:
            return v
    return default


def build_data(cfg: Any, ctx: BuildContext) -> DataBundle:
    """Dispatch ``backend.build(cfg, ctx)`` to the backend named by cfg.

    ``cfg`` is expected to be a typed :class:`TrainConfig` — that's what
    ``wall_x.config.load_config`` returns and what every trainer entry
    point passes in. Raw yaml dicts are not supported here on purpose:
    the typed schema is the contract that gives backends their
    ``cfg.data.dataset_type`` access. If you have a raw dict, use
    :func:`resolve_dataset_type` + :func:`backend_for` directly.
    """
    if not hasattr(cfg, "data") or not hasattr(cfg.data, "dataset_type"):
        raise TypeError(
            f"build_data() expects a typed TrainConfig, got {type(cfg).__name__}. "
            f"Load via wall_x.config.load_config() instead of passing a raw dict."
        )
    return backend_for(cfg.data.dataset_type).build(cfg, ctx)


# --- Active backend (process-global) ------------------------------------


def _set_data_backend(name: str) -> None:
    """Internal — only config loaders should call this.

    Strict semantics: first call wins; same-value re-set is a no-op;
    different-value re-set raises. The error is loud on purpose — it means
    two config loaders disagreed about which backend to use, which is
    almost always a bug (e.g. business code calling this directly, or two
    yamls being loaded in the same process).
    """
    global _DATA_BACKEND
    if _DATA_BACKEND is None:
        _DATA_BACKEND = name
        return
    if _DATA_BACKEND == name:
        return  # idempotent on same value
    raise RuntimeError(
        f"Active backend already set to {_DATA_BACKEND!r}, refusing to "
        f"overwrite with {name!r}. This usually means a config loader was "
        f"called twice with different dataset_type, or business code called "
        f"_set_data_backend directly. Use _reset_data_backend() in tests "
        f"if you need to switch."
    )


def _reset_data_backend() -> None:
    """Internal — clear the active backend. Tests only."""
    global _DATA_BACKEND
    _DATA_BACKEND = None


def has_data_backend() -> bool:
    """Whether a backend has been registered for this process."""
    return _DATA_BACKEND is not None


def data_backend() -> _BackendProxy:
    """Return the active backend proxy. Raises if no config has been loaded.

    Direct construction of ``TrainConfig(...)`` does **not** register a
    backend — that's intentional. Business code must go through
    :func:`wall_x.config.load_config`, which registers the backend before
    returning.
    """
    if _DATA_BACKEND is None:
        raise RuntimeError(
            "No active data backend. Load a TrainConfig via "
            "wall_x.config.load_config() before accessing backend verbs. "
            "Direct TrainConfig() construction does not register a backend."
        )
    return backend_for(_DATA_BACKEND)


__all__ = [
    "register",
    "register_module",
    "record_import_error",
    "available_backends",
    "backend_for",
    "build_data",
    "resolve_dataset_type",
    "MissingOperationError",
    "data_backend",
    "has_data_backend",
]
