"""Public data facade for backend selection and dataset construction."""

from wall_x.data import backends  # noqa: F401  (side-effect registration)
from wall_x.data._bundle import DataBundle
from wall_x.data._protocol import BuildContext, DatasetBackend
from wall_x.data._registry import (
    MissingOperationError,
    available_backends,
    backend_for,
    build_data,
    data_backend,
    has_data_backend,
    register,
    register_module,
    resolve_dataset_type,
)

__all__ = [
    "DataBundle",
    "BuildContext",
    "DatasetBackend",
    "MissingOperationError",
    "data_backend",
    "available_backends",
    "backend_for",
    "build_data",
    "has_data_backend",
    "register",
    "register_module",
    "resolve_dataset_type",
]
