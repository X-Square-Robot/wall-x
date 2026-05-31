"""Public model adapter registry."""

from importlib import import_module

from wall_x.trainer.adapters.base_adapter import ModelAdapter
from wall_x.trainer.adapters.vla_model_adapter import VLAdapter


ADAPTER_REGISTRY: dict[str, type[ModelAdapter]] = {}
ADAPTER_IMPORT_ERRORS: dict[str, str] = {}

_OSS_ADAPTER_SPECS: dict[str, tuple[str, str]] = {
    "qwen2_5": ("wall_x.model.qact.qwen2_5.adapter", "Qwen2_5Adapter"),
}


def _register_adapter(model_type: str, adapter_cls: type[ModelAdapter]) -> None:
    existing = ADAPTER_REGISTRY.get(model_type)
    if existing is not None and existing is not adapter_cls:
        raise ValueError(
            f"model_type {model_type!r} already registered to "
            f"{existing.__name__}, cannot re-register to {adapter_cls.__name__}"
        )
    declared_model_type = getattr(adapter_cls, "MODEL_TYPE", model_type)
    if declared_model_type and declared_model_type != model_type:
        raise ValueError(
            f"adapter {adapter_cls.__name__} declares MODEL_TYPE="
            f"{declared_model_type!r}, but is registered as {model_type!r}"
        )
    ADAPTER_REGISTRY[model_type] = adapter_cls


def _load_optional_adapter(
    module_name: str,
    class_name: str,
    model_type: str,
) -> type[ModelAdapter] | None:
    try:
        module = import_module(module_name)
        adapter_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        ADAPTER_IMPORT_ERRORS[model_type] = f"{type(exc).__name__}: {exc}"
        return None
    _register_adapter(model_type, adapter_cls)
    ADAPTER_IMPORT_ERRORS.pop(model_type, None)
    return adapter_cls


def resolve_adapter(model_type: str) -> type[ModelAdapter]:
    """Return a registered adapter class, loading lazily when needed."""
    adapter_cls = ADAPTER_REGISTRY.get(model_type)
    if adapter_cls is not None:
        return adapter_cls
    spec = _OSS_ADAPTER_SPECS.get(model_type)
    if spec is not None:
        adapter_cls = _load_optional_adapter(*spec, model_type)
    if adapter_cls is None:
        raise ValueError(format_adapter_error(model_type))
    return adapter_cls


def format_adapter_error(model_type: str) -> str:
    msg = (
        f"Unsupported model type: {model_type}. "
        f"Registered: {sorted(ADAPTER_REGISTRY)}"
    )
    if model_type in ADAPTER_IMPORT_ERRORS:
        msg += f". Import failed: {ADAPTER_IMPORT_ERRORS[model_type]}"
    return msg


Qwen2_5Adapter = _load_optional_adapter(
    "wall_x.model.qact.qwen2_5.adapter",
    "Qwen2_5Adapter",
    "qwen2_5",
)


__all__ = [
    "ModelAdapter",
    "VLAdapter",
    "Qwen2_5Adapter",
    "ADAPTER_REGISTRY",
    "ADAPTER_IMPORT_ERRORS",
    "format_adapter_error",
    "resolve_adapter",
]
