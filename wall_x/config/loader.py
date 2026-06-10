"""Wall-X config loader."""

import dataclasses
import os
import shutil
from typing import Any, Type, TypeVar

import yaml

from .data_config import LeRobotDataConfig
from .hyperparams_config import (
    AdamWConfig,
    LRGroupConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainHyperParams,
)
from .infra_config import (
    CheckpointConfig,
    DebugConfig,
    DistributedConfig,
    LoggingConfig,
)
from .model_config import ModelConfig, QActModelConfig
from .registry import (
    get_data_config,
    get_model_config,
    get_optimizer_config,
    get_scheduler_config,
    registered_data_configs,
    registered_model_configs,
    registered_optimizer_configs,
    registered_scheduler_configs,
)
from .task_config import TaskConfig
from .train_config import TrainConfig

T = TypeVar("T")

_CONFIG_PLUGINS_LOADED = False


class _TrainConfigSafeLoader(yaml.SafeLoader):
    pass


def _construct_python_tuple(loader: yaml.SafeLoader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node))


_TrainConfigSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", _construct_python_tuple
)


def _ensure_config_plugins_loaded() -> None:
    """Load optional internal config plugins when they are present."""
    global _CONFIG_PLUGINS_LOADED
    if _CONFIG_PLUGINS_LOADED:
        return
    _CONFIG_PLUGINS_LOADED = True
    try:
        from .internal_plugins import register_internal_config_plugins
    except ImportError:
        return
    register_internal_config_plugins()


def load_config(config_path: str, cli_args: Any = None) -> TrainConfig:
    """Load a ``TrainConfig`` from a YAML file."""
    _ensure_config_plugins_loaded()

    with open(config_path, "r") as f:
        raw = yaml.load(f, Loader=_TrainConfigSafeLoader)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")

    model_type = raw.get("model_type")
    if model_type is None:
        raise ValueError(f"Missing required field 'model_type' in {config_path}")

    config = TrainConfig(
        model_type=model_type,
        task=_build_dataclass(TaskConfig, raw.get("task", {})),
        model=_build_model_config(model_type, raw.get("model", {})),
        data=_build_data_config(raw.get("data", {})),
        hyperparams=_build_hyperparams(raw.get("hyperparams", {})),
        distributed=_build_dataclass(DistributedConfig, raw.get("distributed", {})),
        logging=_build_dataclass(LoggingConfig, raw.get("logging", {})),
        checkpoint=_build_dataclass(CheckpointConfig, raw.get("checkpoint", {})),
        debug=_build_dataclass(DebugConfig, raw.get("debug", {})),
        _raw_data=raw.get("data", {}),
        _raw_yaml=raw,
        dataset_path=raw.get("dataset_path"),
    )

    if cli_args is not None:
        _apply_cli_overrides(config, cli_args)

    _validate(config)

    # Register the active data backend now that cfg is fully resolved
    # (post-CLI-override, post-validate). This is the single source of
    # truth for "which dataset backend is this run using".
    from wall_x.data._registry import _set_data_backend

    _set_data_backend(config.data.dataset_type)

    return config


def save_config(config: TrainConfig, save_dir: str) -> str:
    """Save ``TrainConfig`` to ``config.yml`` under ``save_dir``."""
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.yml")

    data = _sanitize_for_yaml(dataclasses.asdict(config))
    with open(config_path, "w") as f:
        yaml.dump(
            data, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

    dataset_config_path = getattr(config.data, "dataset_config_path", None)
    if dataset_config_path and os.path.exists(dataset_config_path):
        dst = os.path.join(save_dir, "dataset_config.yml")
        shutil.copy(dataset_config_path, dst)

    return config_path


def _sanitize_for_yaml(obj: Any) -> Any:
    """Convert dataclass output to YAML-safe containers."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_yaml(v) for v in obj]
    return obj


def _build_dataclass(cls: Type[T], raw: dict) -> T:
    """Build a dataclass from a dict, ignoring unknown keys."""
    if not raw:
        return cls()

    field_names = {f.name for f in dataclasses.fields(cls)}
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    filtered = {}

    for k, v in raw.items():
        if k not in field_names:
            continue
        ft = field_types[k]
        if (
            isinstance(ft, type)
            and dataclasses.is_dataclass(ft)
            and isinstance(v, dict)
        ):
            filtered[k] = _build_dataclass(ft, v)
        else:
            filtered[k] = v

    return cls(**filtered)


def _build_model_config(model_type: str, raw: dict) -> ModelConfig:
    """Build the registered model config for ``model_type``."""
    cls = get_model_config(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Supported: {registered_model_configs()}"
        )
    return _build_dataclass(cls, raw)


def _build_data_config(raw: dict):
    """Build the registered data config for ``dataset_type``."""
    if not raw:
        return LeRobotDataConfig()

    dataset_type = raw.get("dataset_type", "lerobot")
    cls = get_data_config(dataset_type)
    if cls is None:
        raise ValueError(
            f"Unknown dataset_type: '{dataset_type}'. "
            f"Supported: {registered_data_configs()}"
        )

    return _build_dataclass(cls, raw)


def _build_optimizer_config(raw: dict) -> OptimizerConfig:
    """Build the registered optimizer config for ``optimizer_type``."""
    if not raw:
        return AdamWConfig()

    raw = dict(raw)
    optimizer_type = raw.get("optimizer_type", "adamw")
    cls = get_optimizer_config(optimizer_type)
    if cls is None:
        raise ValueError(
            f"Unknown optimizer_type: '{optimizer_type}'. "
            f"Supported: {registered_optimizer_configs()}"
        )

    if "betas" in raw and isinstance(raw["betas"], list):
        raw["betas"] = tuple(raw["betas"])
    if "adamw_betas" in raw and isinstance(raw["adamw_betas"], list):
        raw["adamw_betas"] = tuple(raw["adamw_betas"])
    if raw.get("lr_groups") is not None:
        raw["lr_groups"] = [
            _build_dataclass(LRGroupConfig, group) for group in raw["lr_groups"]
        ]

    return _build_dataclass(cls, raw)


def _build_scheduler_config(raw: dict) -> SchedulerConfig:
    """Build the registered scheduler config for ``scheduler_type``."""
    if not raw:
        cls = get_scheduler_config("cosine")
        if cls is None:
            raise ValueError("Scheduler config 'cosine' is not registered")
        return cls()

    scheduler_type = raw.get("scheduler_type", "cosine")
    cls = get_scheduler_config(scheduler_type)
    if cls is None:
        raise ValueError(
            f"Unknown scheduler_type: '{scheduler_type}'. "
            f"Supported: {registered_scheduler_configs()}"
        )
    return _build_dataclass(cls, raw)


def _build_hyperparams(raw: dict) -> TrainHyperParams:
    """Build ``TrainHyperParams`` with polymorphic optimizer/scheduler config."""
    if not raw:
        return TrainHyperParams()

    raw = dict(raw)  # shallow copy to avoid mutating caller's dict
    optimizer_raw = raw.pop("optimizer", {})
    scheduler_raw = raw.pop("scheduler", {})

    optimizer = _build_optimizer_config(optimizer_raw)
    scheduler = _build_scheduler_config(scheduler_raw)

    hp = _build_dataclass(TrainHyperParams, raw)
    hp.optimizer = optimizer
    hp.scheduler = scheduler
    return hp


def _apply_cli_overrides(config: TrainConfig, args: Any) -> None:
    """Apply CLI overrides to the loaded config."""
    if getattr(args, "fsdp_sharding_strategy", None) is not None:
        config.distributed.fsdp_sharding_strategy = args.fsdp_sharding_strategy

    if getattr(args, "debug", False):
        config.logging.log_name = "debug"
        config.logging.log_project = "debug"
        config.checkpoint.save_path = "./ckpt/debug"

    if getattr(args, "visualize", False):
        config.debug.visualize_sample = True

    if getattr(args, "wandb_offline", None) is not None:
        config.logging.wandb_offline = args.wandb_offline in ("true", "True", "1")


def _validate(config: TrainConfig) -> None:
    """Validate required fields and model-specific config."""
    if not config.model_type:
        raise ValueError("model_type is required")

    if not config.task.dof_config:
        raise ValueError("task.dof_config is required (cannot be empty)")

    if isinstance(config.model, QActModelConfig):
        model = config.model
        if not model.config_path:
            raise ValueError("model.config_path is required for QAct models")
        if not model.processor_path:
            raise ValueError("model.processor_path is required for QAct models")
