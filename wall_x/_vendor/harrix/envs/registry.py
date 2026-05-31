"""Environment registry and factory."""

from __future__ import annotations

from typing import Type

from wall_x._vendor.harrix.eval_config import EvalConfig
from wall_x._vendor.harrix.envs.base import BaseEnv


_REGISTRY: dict[str, Type[BaseEnv]] = {}


def register_env(name: str):
    def deco(cls: Type[BaseEnv]):
        if name in _REGISTRY:
            raise ValueError(
                f"env {name!r} already registered (cls={_REGISTRY[name].__name__})"
            )
        _REGISTRY[name] = cls
        return cls

    return deco


def _get_class(cfg: EvalConfig) -> Type[BaseEnv]:
    t = cfg.env.type
    cls = _REGISTRY.get(t)
    if cls is None:
        if t == "libero":
            import wall_x._vendor.harrix.envs as _envs

            exc = getattr(_envs, "_LIBERO_IMPORT_ERROR", None)
            if exc is not None:
                raise RuntimeError(
                    "LIBERO evaluation dependencies are not installed. "
                    "Install LIBERO/robosuite and their simulator dependencies "
                    "before using env.type='libero'."
                ) from exc
        raise ValueError(f"unknown env type={t!r}, registered: {sorted(_REGISTRY)}")
    return cls


def build_env(cfg: EvalConfig, worker_id: int) -> BaseEnv:
    return _get_class(cfg)(cfg, worker_id)


def enumerate_episodes_for(cfg: EvalConfig) -> list[tuple]:
    return _get_class(cfg).enumerate_episodes(cfg)


def registered_envs() -> list[str]:
    return sorted(_REGISTRY)
