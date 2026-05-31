"""Inference adapter registry and factory."""

from __future__ import annotations

from typing import Type

from wall_x._vendor.harrix.adapters.base import BaseInferAdapter
from wall_x._vendor.harrix.eval_config import EvalConfig


ADAPTER_REGISTRY: dict[str, Type[BaseInferAdapter]] = {}


def register_adapter(name: str):
    def deco(cls: Type[BaseInferAdapter]):
        if name in ADAPTER_REGISTRY:
            raise ValueError(
                f"adapter {name!r} is already registered "
                f"(cls={ADAPTER_REGISTRY[name].__name__})"
            )
        ADAPTER_REGISTRY[name] = cls
        return cls

    return deco


def build_adapter(cfg: EvalConfig) -> BaseInferAdapter:
    arch = cfg.model.architecture
    cls = ADAPTER_REGISTRY.get(arch)
    if cls is None:
        raise ValueError(
            f"unknown architecture={arch!r}, registered={sorted(ADAPTER_REGISTRY)}"
        )
    return cls(cfg)


def registered_architectures() -> list[str]:
    """Return registered architecture names for diagnostics."""
    return sorted(ADAPTER_REGISTRY)
