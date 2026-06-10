"""Public model adapter registry."""

from importlib import import_module

from wall_x.trainer.adapters.base_adapter import ModelAdapter
from wall_x.trainer.adapters.vla_model_adapter import VLAdapter

_ADAPTER_SPECS = {
    "qwen2_5": ("wall_x.model.qact.qwen2_5.adapter", "Qwen2_5Adapter"),
}
_ADAPTER_CLASS_EXPORTS = {
    "Qwen2_5Adapter": "qwen2_5",
}
ADAPTER_IMPORT_ERRORS: dict[str, str] = {}
_ADAPTER_LOADING: set[str] = set()


class _AdapterRegistry(dict[str, type[ModelAdapter]]):
    def get(self, model_type: str, default=None):
        _load_public_adapter(model_type)
        return dict.get(self, model_type, default)

    def __contains__(self, model_type: object) -> bool:
        if isinstance(model_type, str):
            _load_public_adapter(model_type)
        return dict.__contains__(self, model_type)

    def __getitem__(self, model_type: str) -> type[ModelAdapter]:
        _load_public_adapter(model_type)
        return dict.__getitem__(self, model_type)

    def __iter__(self):
        _load_all_public_adapters()
        return dict.__iter__(self)

    def keys(self):
        _load_all_public_adapters()
        return dict.keys(self)

    def items(self):
        _load_all_public_adapters()
        return dict.items(self)

    def values(self):
        _load_all_public_adapters()
        return dict.values(self)


ADAPTER_REGISTRY: _AdapterRegistry = _AdapterRegistry()


def _register_adapter(model_type: str, adapter_cls: type[ModelAdapter]) -> None:
    existing = dict.get(ADAPTER_REGISTRY, model_type)
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
    dict.__setitem__(ADAPTER_REGISTRY, model_type, adapter_cls)


def _load_public_adapter(model_type: str) -> type[ModelAdapter] | None:
    if dict.__contains__(ADAPTER_REGISTRY, model_type):
        return dict.__getitem__(ADAPTER_REGISTRY, model_type)
    spec = _ADAPTER_SPECS.get(model_type)
    if spec is None or model_type in _ADAPTER_LOADING:
        return None
    module_name, class_name = spec
    _ADAPTER_LOADING.add(model_type)
    try:
        module = import_module(module_name)
        adapter_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        ADAPTER_IMPORT_ERRORS[model_type] = f"{type(exc).__name__}: {exc}"
        return None
    finally:
        _ADAPTER_LOADING.discard(model_type)
    _register_adapter(model_type, adapter_cls)
    return adapter_cls


def _load_all_public_adapters() -> None:
    for model_type in _ADAPTER_SPECS:
        _load_public_adapter(model_type)


def format_adapter_error(model_type: str) -> str:
    msg = (
        f"Unsupported model type: {model_type}. "
        f"Registered: {sorted(ADAPTER_REGISTRY)}"
    )
    if model_type in ADAPTER_IMPORT_ERRORS:
        msg += f". Import failed: {ADAPTER_IMPORT_ERRORS[model_type]}"
    return msg


def resolve_adapter(model_type: str) -> type[ModelAdapter]:
    """Return a registered adapter class, loading lazily when needed."""
    adapter_cls = ADAPTER_REGISTRY.get(model_type)
    if adapter_cls is None:
        raise ValueError(format_adapter_error(model_type))
    return adapter_cls


def __getattr__(name: str):
    model_type = _ADAPTER_CLASS_EXPORTS.get(name)
    if model_type is not None:
        adapter_cls = _load_public_adapter(model_type)
        if adapter_cls is not None:
            return adapter_cls
    raise AttributeError(name)


__all__ = [
    "ModelAdapter",
    "VLAdapter",
    "Qwen2_5Adapter",
    "ADAPTER_REGISTRY",
    "ADAPTER_IMPORT_ERRORS",
    "format_adapter_error",
    "resolve_adapter",
]
