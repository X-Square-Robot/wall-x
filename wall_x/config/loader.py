"""Wall-X config loader."""

import dataclasses
import math
import os
import shutil
import warnings
from typing import Any, Optional, Type, TypeVar

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
    RECOMPUTE_CONFIG_BY_MODEL_TYPE,
    CheckpointConfig,
    DebugConfig,
    DistributedConfig,
    LoggingConfig,
    RecomputeConfig,
    ServingConfig,
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
    with open(config_path, "r") as f:
        raw = yaml.load(f, Loader=_TrainConfigSafeLoader)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if raw.get("portable_bundle_version"):
        _resolve_config_relative_paths(raw, config_path)

    config = _build_train_config(raw, source=config_path, cli_args=cli_args)

    # Register the active data backend now that cfg is fully resolved
    # (post-CLI-override, post-validate). This is the single source of
    # truth for "which dataset backend is this run using".
    from wall_x.data._registry import _set_data_backend

    _set_data_backend(config.data.dataset_type)

    return config


def load_train_config_from_checkpoint(
    checkpoint_path: str,
    train_config_path: Optional[str] = None,
) -> TrainConfig:
    """The single inference-side entry point for loading a train config.

    Locates the yaml (explicit ``train_config_path`` wins, else
    ``<ckpt>/config.yml(.yaml)``), builds the typed ``TrainConfig`` through the
    same pipeline as training, applies checkpoint-local processor/tokenizer
    overlays, and registers the data backend.
    """
    yml_path = train_config_path
    if yml_path is None:
        for fname in ("config.yml", "config.yaml"):
            candidate = os.path.join(checkpoint_path, fname)
            if os.path.exists(candidate):
                yml_path = candidate
                break
        if yml_path is None:
            raise FileNotFoundError(
                f"No config.yml/config.yaml in {checkpoint_path}; "
                "pass train_config_path explicitly."
            )
    elif not os.path.exists(yml_path):
        raise FileNotFoundError(f"train_config_path not found: {yml_path}")

    with open(yml_path, "r") as f:
        raw = yaml.load(f, Loader=_TrainConfigSafeLoader)
    if raw is None:
        raise ValueError(f"Config file is empty: {yml_path}")

    if raw.get("portable_bundle_version"):
        _resolve_config_relative_paths(raw, yml_path)

    config = _build_train_config(raw, source=yml_path)
    _apply_checkpoint_overlays(config, checkpoint_path)

    from wall_x.data._registry import _set_data_backend

    _set_data_backend(config.data.dataset_type)

    return config


def _resolve_config_relative_paths(raw: dict, config_path: str) -> None:
    """Resolve declared filesystem fields relative to their YAML file.

    Dataset URIs and arbitrary strings are deliberately excluded.
    """
    base = os.path.dirname(os.path.abspath(config_path))

    def resolve(container: dict, key: str) -> None:
        value = container.get(key)
        if not isinstance(value, str) or not value or os.path.isabs(value):
            return
        container[key] = os.path.normpath(os.path.join(base, value))

    model = raw.get("model")
    if isinstance(model, dict):
        for key in (
            "config_path",
            "processor_path",
            "pretrained_path",
            "action_tokenizer_path",
            "action_tokenizer_checkpoint_path",
            "action_tokenizer_config_dir",
        ):
            resolve(model, key)

    data = raw.get("data")
    if isinstance(data, dict):
        resolve(data, "dataset_config_path")
        resolve(data, "norm_stats_path")
        norm = data.get("normalizer_config")
        if isinstance(norm, dict):
            resolve(norm, "customized_action_statistic_dof")

    checkpoint = raw.get("checkpoint")
    if isinstance(checkpoint, dict):
        for key in (
            "save_path",
            "resume_from",
            "resume_model",
            "resume_optimizer",
            "resume_scheduler",
            "resume_ema",
            "resume_rng",
            "resume_data",
            "resume_step",
        ):
            resolve(checkpoint, key)


def _apply_checkpoint_overlays(config: TrainConfig, checkpoint_path: str) -> None:
    """Point processor/tokenizer paths at checkpoint-local copies so the
    checkpoint stays portable when the training-machine paths are gone.
    No-op for model configs without these fields."""
    if not os.path.isdir(checkpoint_path):
        return

    model = config.model
    if hasattr(model, "processor_path") and all(
        os.path.exists(os.path.join(checkpoint_path, name))
        for name in ("preprocessor_config.json", "tokenizer_config.json")
    ):
        model.processor_path = checkpoint_path

    orig_tok = getattr(model, "action_tokenizer_path", None)
    if orig_tok is not None and not os.path.exists(orig_tok):
        if os.path.exists(
            os.path.join(checkpoint_path, "tokenizer.json")
        ) and os.path.exists(os.path.join(checkpoint_path, "tokenizer_config.json")):
            model.action_tokenizer_path = checkpoint_path


def _build_train_config(raw: dict, *, source: str, cli_args: Any = None) -> TrainConfig:
    """Build a validated ``TrainConfig`` from a typed-schema raw dict.
    Data backend registration is left to the public entry points."""
    _ensure_config_plugins_loaded()

    model_type = raw.get("model_type")
    if model_type is None:
        raise ValueError(f"Missing required field 'model_type' in {source}")

    hyperparams = _build_hyperparams(
        raw.get("hyperparams", {}),
        model_type=model_type,
    )
    distributed = _build_distributed_config(
        raw.get("distributed", {}),
        optimizer_type=hyperparams.optimizer.optimizer_type,
        model_type=model_type,
    )

    config = TrainConfig(
        model_type=model_type,
        task=_build_dataclass(
            TaskConfig, raw.get("task", {}), where=f"task: section of {source}"
        ),
        model=_build_model_config(model_type, raw.get("model", {})),
        data=_build_data_config(raw.get("data", {})),
        hyperparams=hyperparams,
        distributed=distributed,
        logging=_build_dataclass(LoggingConfig, raw.get("logging", {})),
        checkpoint=_build_dataclass(CheckpointConfig, raw.get("checkpoint", {})),
        debug=_build_dataclass(DebugConfig, raw.get("debug", {})),
        serving=_build_dataclass(ServingConfig, raw.get("serving", {})),
        _raw_data=raw.get("data", {}),
        _raw_yaml=raw,
        dataset_path=raw.get("dataset_path"),
    )

    if cli_args is not None:
        _apply_cli_overrides(config, cli_args)

    _validate(config)

    return config


def dump_config(config: TrainConfig) -> dict:
    """Serialize the effective ``TrainConfig`` back into the typed yaml schema.

    Inverse of :func:`load_config`: a yaml-safe dict in the same schema as
    hand-written configs, no ``_raw_*`` private fields, load/dump roundtrip
    stable. Checkpoints persist this form -- the *effective* config (defaults
    materialized, CLI overrides applied) -- so reloads never re-derive values
    from future defaults.
    """
    distributed = dataclasses.asdict(config.distributed)
    # These fields double as legacy DDP aliases and would warn on every
    # reload; omit them when they equal the default (non-default values are
    # kept so the roundtrip stays lossless).
    _dist_defaults = {
        f.name: f.default for f in dataclasses.fields(type(config.distributed))
    }
    for legacy_alias in (
        "find_unused_parameters",
        "broadcast_buffers",
        "bucket_cap_mb",
    ):
        if distributed.get(legacy_alias) == _dist_defaults.get(legacy_alias):
            distributed.pop(legacy_alias, None)

    out = {
        "model_type": config.model_type,
        "task": dataclasses.asdict(config.task),
        "model": dataclasses.asdict(config.model),
        "data": dataclasses.asdict(config.data),
        "hyperparams": dataclasses.asdict(config.hyperparams),
        "distributed": distributed,
        "logging": dataclasses.asdict(config.logging),
        "checkpoint": dataclasses.asdict(config.checkpoint),
        "debug": dataclasses.asdict(config.debug),
        "serving": dataclasses.asdict(config.serving),
    }
    if config.dataset_path is not None:
        out["dataset_path"] = config.dataset_path
    return _sanitize_for_yaml(out)


def save_config(config: TrainConfig, save_dir: str) -> str:
    """Save ``TrainConfig`` to ``config.yml`` under ``save_dir``."""
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.yml")

    with open(config_path, "w") as f:
        yaml.safe_dump(
            dump_config(config),
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    dataset_config_path = getattr(config.data, "dataset_config_path", None)
    if dataset_config_path and os.path.exists(dataset_config_path):
        dst = os.path.join(save_dir, "dataset_config.yml")
        shutil.copy(dataset_config_path, dst)

    return config_path


def _sanitize_for_yaml(obj: Any) -> Any:
    """Convert dataclass/config output to YAML-safe containers so it can be
    written with ``yaml.safe_dump`` (no ``!!python/*`` tags). tuples/sets ->
    lists; numpy scalars -> python scalars; any other non-primitive leaf -> str
    (config consumers read these as plain scalars/sequences)."""
    if isinstance(obj, dict):
        return {
            (k if isinstance(k, str) else _sanitize_for_yaml(k)): _sanitize_for_yaml(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_yaml(v) for v in obj]
    if isinstance(obj, (str, bool, int, float)) or obj is None:
        return obj
    item = getattr(obj, "item", None)  # e.g. numpy scalar -> python scalar
    if callable(item):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _build_dataclass(cls: Type[T], raw: dict, *, where: Optional[str] = None) -> T:
    """Build a dataclass from a dict, ignoring unknown keys.

    Fields declared without a default are required: missing ones raise a
    ValueError instead of letting the dataclass default paper over them.
    """
    raw = raw or {}
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

    missing = [
        f.name
        for f in dataclasses.fields(cls)
        if f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
        and f.name not in filtered
    ]
    if missing:
        loc = f" ({where})" if where else ""
        raise ValueError(f"{cls.__name__} is missing required keys {missing}{loc}")

    return cls(**filtered)


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"distributed.{field} must be a boolean, got {value!r}")


def _warn_distributed_alias(old: str, new: str) -> None:
    warnings.warn(
        "[config deprecation] "
        f"distributed.{old} is deprecated; use distributed.{new}. "
        "The alias will be removed in the next Wall-X config cleanup release.",
        FutureWarning,
        stacklevel=3,
    )


def _set_nested_alias(
    raw: dict,
    *,
    old_key: str,
    section: str,
    new_key: str,
) -> None:
    if old_key not in raw:
        return
    _warn_distributed_alias(old_key, f"{section}.{new_key}")
    nested = raw.setdefault(section, {})
    if not isinstance(nested, dict):
        raise ValueError(f"distributed.{section} must be a mapping")
    value = raw[old_key]
    if new_key in nested and nested[new_key] != value:
        raise ValueError(
            f"distributed.{old_key} conflicts with distributed." f"{section}.{new_key}"
        )
    nested[new_key] = value


def _normalize_distributed_config(
    raw: dict | None,
    *,
    optimizer_type: str | None = None,
) -> dict:
    raw = dict(raw or {})
    backend = raw.get("backend")
    if backend is not None:
        backend = str(backend)

    dmuon_strategy = raw.get("dmuon_strategy")
    if dmuon_strategy is not None:
        strategy = str(dmuon_strategy)
        if strategy == "pure":
            _warn_distributed_alias("dmuon_strategy: pure", "backend: dmuon")
            if backend is not None and backend != "dmuon":
                raise ValueError(
                    "distributed.dmuon_strategy=pure conflicts with "
                    f"distributed.backend={backend!r}"
                )
            backend = "dmuon"
        elif strategy == "fsdp":
            raise ValueError(
                "distributed.dmuon_strategy=fsdp has been removed. "
                "Use distributed.backend=dmuon for pure DMuon, or use a "
                "non-DMuon optimizer with distributed.backend=fsdp2."
            )
        else:
            raise ValueError(
                "distributed.dmuon_strategy only accepted legacy value 'pure'; "
                f"got {strategy!r}"
            )

    if backend is None and optimizer_type == "dmuon":
        _warn_distributed_alias(
            "implicit backend from optimizer_type=dmuon",
            "backend: dmuon",
        )
        backend = "dmuon"

    if "use_fsdp" in raw:
        _warn_distributed_alias("use_fsdp", "backend")
        if not _as_bool(raw["use_fsdp"], field="use_fsdp"):
            if backend is not None and backend != "ddp":
                raise ValueError(
                    "distributed.use_fsdp=false conflicts with "
                    f"distributed.backend={backend!r}"
                )
            backend = "ddp"
        elif backend is None:
            backend = "fsdp2"

    if backend is None:
        backend = "fsdp2"
    if backend not in {"fsdp2", "dmuon", "ddp"}:
        raise ValueError(
            f"Unknown distributed.backend={backend!r}; expected 'fsdp2', "
            "'dmuon', or 'ddp'."
        )
    raw["backend"] = backend

    active_section = "dmuon" if backend == "dmuon" else "fsdp"
    if backend != "ddp":
        _set_nested_alias(
            raw,
            old_key="fsdp_sharding_strategy",
            section=active_section,
            new_key="sharding_strategy",
        )
        _set_nested_alias(
            raw,
            old_key="fsdp_hsdp_replicate_size",
            section=active_section,
            new_key="hsdp_replicate_size",
        )

    fsdp_aliases = {
        "fsdp_backward_prefetch": "backward_prefetch",
        "fsdp_cpu_offload": "cpu_offload",
        "fsdp_use_orig_params": "use_orig_params",
        "fsdp_limit_all_gathers": "limit_all_gathers",
        "fsdp_forward_prefetch": "forward_prefetch",
        "fsdp_sync_module_states": "sync_module_states",
        "fsdp_save_policy": "save_policy",
        "fsdp_reduce_dtype": "reduce_dtype",
    }
    for old_key, new_key in fsdp_aliases.items():
        if old_key in raw:
            if backend == "dmuon":
                _warn_distributed_alias(
                    old_key,
                    f"fsdp.{new_key} (ignored by backend=dmuon)",
                )
                continue
            if backend == "ddp":
                _warn_distributed_alias(
                    old_key,
                    f"fsdp.{new_key} (ignored by backend=ddp)",
                )
                continue
            _set_nested_alias(raw, old_key=old_key, section="fsdp", new_key=new_key)

    ddp_aliases = {
        "find_unused_parameters": "find_unused_parameters",
        "broadcast_buffers": "broadcast_buffers",
        "bucket_cap_mb": "bucket_cap_mb",
    }
    for old_key, new_key in ddp_aliases.items():
        if old_key in raw:
            if backend != "ddp":
                _warn_distributed_alias(
                    old_key,
                    f"ddp.{new_key} (ignored by backend={backend})",
                )
                continue
            _set_nested_alias(raw, old_key=old_key, section="ddp", new_key=new_key)

    return raw


def _build_distributed_config(
    raw: dict | None,
    *,
    optimizer_type: str | None = None,
    model_type: str | None = None,
) -> DistributedConfig:
    cls = DistributedConfig
    raw = dict(raw or {})
    # recompute has a per-model schema (RECOMPUTE_CONFIG_BY_MODEL_TYPE) and
    # rejects unknown keys loudly, so it is hydrated out-of-band from the
    # ignore-unknown-keys _build_dataclass path.
    recompute_raw = raw.pop("recompute", None)
    recompute_cls = RECOMPUTE_CONFIG_BY_MODEL_TYPE.get(model_type, RecomputeConfig)
    config = _build_dataclass(
        cls,
        _normalize_distributed_config(raw, optimizer_type=optimizer_type),
    )
    config.recompute = recompute_cls.build(
        recompute_raw, where=f"distributed.recompute (model_type={model_type})"
    )
    return config


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


def _resolve_hyperparams_cls(algorithm: str, model_type: str):
    """Resolve the TrainHyperParams subclass for (algorithm, model_type).

    Non-sl algorithms route to RL hyperparams subclasses via the routing table
    RL owns in ``wall_x.rl.hyperparams`` (lazy import keeps base config loading
    free of an RL dependency). ``algorithm="sl"`` (default) → base
    TrainHyperParams. An unknown non-sl algorithm is a config error (raise),
    not a silent fallback — otherwise a typo would drop all RL hyperparams.
    """
    if algorithm and algorithm != "sl":
        from wall_x.rl.hyperparams import resolve_hyperparams_cls

        cls = resolve_hyperparams_cls(algorithm, model_type)
        if cls is None:
            raise ValueError(
                f"Unknown hyperparams.algorithm={algorithm!r} for "
                f"model_type={model_type!r}. Known RL algorithms are routed in "
                f"wall_x.rl.hyperparams.HYPERPARAMS_ROUTING; use 'sl' for "
                f"supervised training."
            )
        return cls
    return TrainHyperParams


def _build_hyperparams(raw: dict, model_type: str = "") -> TrainHyperParams:
    """Build ``TrainHyperParams`` (incl. RL subclasses) with polymorphic
    optimizer/scheduler config + (algorithm, model_type) routing.

    The discriminator ``hyperparams.algorithm`` lives in the section it
    multiplexes (symmetric with ``model_type`` ↔ ``model``); it is read from
    the raw dict before the typed object is built.
    """
    algorithm = (raw or {}).get("algorithm", "sl")
    cls = _resolve_hyperparams_cls(algorithm, model_type)
    if not raw:
        return cls()

    raw = dict(raw)  # shallow copy to avoid mutating caller's dict
    optimizer_raw = raw.pop("optimizer", {})
    scheduler_raw = raw.pop("scheduler", {})

    optimizer = _build_optimizer_config(optimizer_raw)
    scheduler = _build_scheduler_config(scheduler_raw)

    hp = _build_dataclass(cls, raw)
    hp.optimizer = optimizer
    hp.scheduler = scheduler
    return hp


def _apply_cli_overrides(config: TrainConfig, args: Any) -> None:
    """Apply CLI overrides to the loaded config."""
    if getattr(args, "fsdp_sharding_strategy", None) is not None:
        if config.distributed.backend == "dmuon":
            config.distributed.dmuon.sharding_strategy = args.fsdp_sharding_strategy
        elif config.distributed.backend == "fsdp2":
            config.distributed.fsdp.sharding_strategy = args.fsdp_sharding_strategy

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

    backend = config.distributed.backend
    if backend not in {"fsdp2", "dmuon", "ddp"}:
        raise ValueError(
            "distributed.backend must be 'fsdp2', 'dmuon', or 'ddp', "
            f"got {backend!r}"
        )
    optimizer_type = config.hyperparams.optimizer.optimizer_type
    if optimizer_type == "naive_fsdp2_muon" and backend != "fsdp2":
        raise ValueError(
            "optimizer_type=naive_fsdp2_muon requires distributed.backend=fsdp2."
        )
    if backend == "dmuon" and optimizer_type != "dmuon":
        raise ValueError(
            "distributed.backend=dmuon requires "
            "hyperparams.optimizer.optimizer_type=dmuon."
        )
    if backend == "fsdp2" and optimizer_type == "dmuon":
        raise ValueError(
            "FSDP2 + DMuon dedicated params is no longer supported. "
            "Use distributed.backend=dmuon for pure DMuon."
        )
    if backend == "ddp" and optimizer_type == "dmuon":
        raise ValueError(
            "distributed.backend=ddp is not compatible with optimizer_type=dmuon. "
            "Use distributed.backend=dmuon for DMuon."
        )
    valid_sharding = {
        "full_shard",
        "shard_grad_op",
        "hybrid_shard",
        "_hybrid_shard_zero2",
    }
    if backend != "ddp":
        backend_section = "dmuon" if backend == "dmuon" else "fsdp"
        sharding = getattr(config.distributed, backend_section).sharding_strategy
        if sharding not in valid_sharding:
            raise ValueError(
                f"distributed.{backend_section}.sharding_strategy={sharding!r} is invalid; "
                f"expected one of {sorted(valid_sharding)}."
            )

    if not config.task.dof_config:
        raise ValueError("task.dof_config is required (cannot be empty)")

    ar_action_loss_weight = config.model.ar_action_loss_weight
    if ar_action_loss_weight is not None:
        if config.model_type != "qwen3_5":
            raise ValueError(
                "model.ar_action_loss_weight is supported only for "
                "model_type=qwen3_5"
            )
        if isinstance(ar_action_loss_weight, dict):
            expected_keys = {"initial", "target", "warmup_steps"}
            unknown_keys = set(ar_action_loss_weight) - expected_keys
            missing_keys = expected_keys - set(ar_action_loss_weight)
            if unknown_keys or missing_keys:
                raise ValueError(
                    "model.ar_action_loss_weight warmup requires exactly "
                    "initial, target, and warmup_steps; "
                    f"missing={sorted(missing_keys)}, unknown={sorted(unknown_keys)}"
                )
            initial = ar_action_loss_weight["initial"]
            target = ar_action_loss_weight["target"]
            warmup_steps = ar_action_loss_weight["warmup_steps"]
            for name, value in (("initial", initial), ("target", target)):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(
                        f"model.ar_action_loss_weight.{name} must be a finite "
                        "non-negative number"
                    )
            if float(initial) > float(target):
                raise ValueError(
                    "model.ar_action_loss_weight.initial must not exceed target"
                )
            if (
                isinstance(warmup_steps, bool)
                or not isinstance(warmup_steps, int)
                or warmup_steps <= 0
            ):
                raise ValueError(
                    "model.ar_action_loss_weight.warmup_steps must be a positive integer"
                )
        elif (
            isinstance(ar_action_loss_weight, bool)
            or not isinstance(ar_action_loss_weight, (int, float))
            or not math.isfinite(float(ar_action_loss_weight))
            or float(ar_action_loss_weight) < 0.0
        ):
            raise ValueError(
                "model.ar_action_loss_weight must be a finite non-negative number "
                "or a warmup mapping"
            )

    if isinstance(config.model, QActModelConfig):
        model = config.model
        if not model.config_path:
            raise ValueError("model.config_path is required for QAct models")
        if not model.processor_path:
            raise ValueError("model.processor_path is required for QAct models")
        if model.action_tokenizer_type in {
            "wall_vlm_rvq_v3_2",
            "wall_vlm_rvq_v3_3_ar",
        }:
            if config.model_type != "qwen3_5":
                raise ValueError(
                    f"{model.action_tokenizer_type} is supported only for "
                    "model_type=qwen3_5"
                )
            if model.rvq_token_protocol != "wall_vlm_start_v1":
                raise ValueError(
                    f"{model.action_tokenizer_type} requires "
                    "model.rvq_token_protocol=wall_vlm_start_v1"
                )
            from wall_x.model.qact.rvq_position import normalize_rvq_token_layout

            normalize_rvq_token_layout(model.rvq_token_layout)
            if model.rvq_position_encoding != "time_level_2d":
                raise ValueError(
                    f"{model.action_tokenizer_type} requires "
                    "model.rvq_position_encoding=time_level_2d"
                )
            if model.rvq_action_horizon is None or model.rvq_action_horizon <= 0:
                raise ValueError(
                    f"{model.action_tokenizer_type} requires a positive "
                    "model.rvq_action_horizon"
                )
            if model.rvq_checkpoint_contract not in {"strict", "bootstrap"}:
                raise ValueError(
                    "model.rvq_checkpoint_contract must be 'strict' or 'bootstrap'"
                )
