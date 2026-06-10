"""Decorator-based model adapter registry."""

from __future__ import annotations

from typing import Dict, Type

# Global registry mapping model_type string -> adapter class
_MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(model_type: str):
    """Decorator to register an adapter class for a given model_type.

    Can be stacked to register the same class for multiple model_types:

        @register_model("qwen2_5")
        class QActAdapter: ...
    """

    def decorator(cls):
        if model_type in _MODEL_REGISTRY:
            existing = _MODEL_REGISTRY[model_type]
            if existing is not cls:
                raise ValueError(
                    f"model_type '{model_type}' already registered to "
                    f"{existing.__name__}, cannot re-register to {cls.__name__}"
                )
        _MODEL_REGISTRY[model_type] = cls
        return cls

    return decorator


def get_adapter(model_type: str, **kwargs):
    """Instantiate the adapter registered for model_type.

    Args:
        model_type: Registered model type string (e.g. "qwen2_5").
        **kwargs: Passed to the adapter constructor.

    Returns:
        An adapter instance.

    Raises:
        KeyError: If model_type is not registered.
    """
    if model_type not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {available}. "
            f"Did you forget to import the adapter module?"
        )
    return _MODEL_REGISTRY[model_type](**kwargs)


def list_registered_models() -> list:
    """Return sorted list of registered model_type strings."""
    return sorted(_MODEL_REGISTRY.keys())


def clear_registry():
    """Clear all registrations. Intended for testing only."""
    _MODEL_REGISTRY.clear()
