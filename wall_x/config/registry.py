"""Runtime registries for typed config variants.

Core Wall-X keeps only base config types in ``wall_x.config``. Optional
datasets, model families, and optimizer variants register their dataclasses
from their own packages, so trimmed distributions do not leave dead config
entries behind.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar("T")

_DATA_CONFIG_REGISTRY: dict[str, type[Any]] = {}
_MODEL_CONFIG_REGISTRY: dict[str, type[Any]] = {}
_OPTIMIZER_CONFIG_REGISTRY: dict[str, type[Any]] = {}
_SCHEDULER_CONFIG_REGISTRY: dict[str, type[Any]] = {}


def _register(
    registry: dict[str, type[Any]], names: tuple[str, ...], cls: type[T]
) -> type[T]:
    for name in names:
        if not name:
            raise ValueError("Config registry name must be non-empty")
        existing = registry.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Config name {name!r} is already registered by "
                f"{existing.__module__}.{existing.__name__}"
            )
        registry[name] = cls
    return cls


def register_data_config(*names: str) -> Callable[[type[T]], type[T]]:
    return lambda cls: _register(_DATA_CONFIG_REGISTRY, names, cls)


def register_model_config(*names: str) -> Callable[[type[T]], type[T]]:
    return lambda cls: _register(_MODEL_CONFIG_REGISTRY, names, cls)


def register_optimizer_config(*names: str) -> Callable[[type[T]], type[T]]:
    return lambda cls: _register(_OPTIMIZER_CONFIG_REGISTRY, names, cls)


def register_scheduler_config(*names: str) -> Callable[[type[T]], type[T]]:
    return lambda cls: _register(_SCHEDULER_CONFIG_REGISTRY, names, cls)


def get_data_config(name: str) -> type[Any] | None:
    return _DATA_CONFIG_REGISTRY.get(name)


def get_model_config(name: str) -> type[Any] | None:
    return _MODEL_CONFIG_REGISTRY.get(name)


def get_optimizer_config(name: str) -> type[Any] | None:
    return _OPTIMIZER_CONFIG_REGISTRY.get(name)


def get_scheduler_config(name: str) -> type[Any] | None:
    return _SCHEDULER_CONFIG_REGISTRY.get(name)


def registered_data_configs() -> list[str]:
    return sorted(_DATA_CONFIG_REGISTRY)


def registered_model_configs() -> list[str]:
    return sorted(_MODEL_CONFIG_REGISTRY)


def registered_optimizer_configs() -> list[str]:
    return sorted(_OPTIMIZER_CONFIG_REGISTRY)


def registered_scheduler_configs() -> list[str]:
    return sorted(_SCHEDULER_CONFIG_REGISTRY)


__all__ = [
    "get_data_config",
    "get_model_config",
    "get_optimizer_config",
    "get_scheduler_config",
    "register_data_config",
    "register_model_config",
    "register_optimizer_config",
    "register_scheduler_config",
    "registered_data_configs",
    "registered_model_configs",
    "registered_optimizer_configs",
    "registered_scheduler_configs",
]
