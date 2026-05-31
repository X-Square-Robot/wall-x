"""Load checkpoint-side train config and construct runtime config objects.

Inference adapters use this module to load the train config, apply checkpoint
overrides for moved processor/tokenizer files, build model config, and build the
data config required by preprocessing.
"""

from __future__ import annotations

import os
import logging

import yaml

logger = logging.getLogger(__name__)


def resolve_use_state_string_representation(train_config: dict) -> bool:
    """Read ``use_state_string_representation`` from flat, data, or task YAML."""
    if train_config.get("use_state_string_representation") is not None:
        return bool(train_config["use_state_string_representation"])
    data = train_config.get("data") or {}
    if data.get("use_state_string_representation") is not None:
        return bool(data["use_state_string_representation"])
    task = train_config.get("task") or {}
    return bool(task.get("use_state_string_representation", False))


def resolve_state_bins(train_config: dict, default: int = 256) -> int:
    """Read discretization bin count from flat or nested train config."""
    if train_config.get("state_bins") is not None:
        return int(train_config["state_bins"])
    data = train_config.get("data") or {}
    if data.get("state_bins") is not None:
        return int(data["state_bins"])
    return default


# Fields ``load_wallx_processors`` and model construction read from a flat dict.
_INFERENCE_MODEL_KEYS = (
    "processor_path",
    "pretrained_path",
    "config_path",
    "action_tokenizer_type",
    "action_tokenizer_path",
    "action_tokenizer_checkpoint_path",
    "action_tokenizer_config_dir",
    "ar_loss_weight",
    "attn_deterministic",
    "flow_loss_weight",
    "use_ema",
)


def _merge_model_fields(target: dict, source: dict, *, overwrite: bool = False) -> None:
    """Copy model-related keys from ``source`` into flat ``target``."""
    if not source:
        return
    for key in _INFERENCE_MODEL_KEYS:
        value = source.get(key)
        if value is None:
            continue
        if overwrite or target.get(key) in (None, ""):
            target[key] = value


def _load_ckpt_config_overlay(checkpoint_path: str) -> dict:
    """Read ``config.yml`` saved beside a training checkpoint."""
    ckpt_yml = os.path.join(checkpoint_path, "config.yml")
    if not os.path.exists(ckpt_yml):
        return {}
    with open(ckpt_yml, "r") as f:
        raw = yaml.load(f, Loader=yaml.FullLoader) or {}

    overlay: dict = {}
    _merge_model_fields(overlay, raw.get("model") or {}, overwrite=True)
    raw_yaml = raw.get("_raw_yaml") or {}
    _merge_model_fields(overlay, raw_yaml.get("model") or {}, overwrite=False)
    _merge_model_fields(overlay, raw, overwrite=False)
    return overlay


def load_train_config_with_ckpt_overlay(
    train_config_path: str,
    checkpoint_path: str,
) -> dict:
    """Load train YAML and apply checkpoint-local processor/tokenizer overlays.

    This lets a checkpoint remain portable when the original training-machine
    processor or action-tokenizer paths are no longer available.
    """
    with open(train_config_path, "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    preprocessor_file = os.path.join(checkpoint_path, "preprocessor_config.json")
    if os.path.exists(preprocessor_file):
        train_config["processor_path"] = checkpoint_path

    orig_action_tok = train_config.get("action_tokenizer_path", None)
    if orig_action_tok is not None and not os.path.exists(orig_action_tok):
        tokenizer_file = os.path.join(checkpoint_path, "tokenizer.json")
        tokenizer_config_file = os.path.join(checkpoint_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_file) and os.path.exists(tokenizer_config_file):
            train_config["action_tokenizer_path"] = checkpoint_path

    ckpt_overlay = _load_ckpt_config_overlay(checkpoint_path)
    _merge_model_fields(train_config, ckpt_overlay, overwrite=False)
    if isinstance(train_config.get("model"), dict):
        _merge_model_fields(train_config["model"], ckpt_overlay, overwrite=False)

    return train_config


def normalize_train_config_for_inference(
    train_config: dict,
    train_config_path: str,
) -> dict:
    """Flatten typed TrainConfig YAML into the legacy dict inference expects.

    Serving and ``update_model_config`` still read top-level ``dof_config`` and
    nested ``data.*`` fields. New training yamls store those under ``task:``.
    """
    try:
        from wall_x.config.loader import load_config

        typed_cfg = load_config(train_config_path)
    except Exception as e:
        logger.debug(
            "Train config is not typed TrainConfig schema (%s): %s",
            train_config_path,
            e,
        )
        return train_config

    import dataclasses

    normalized = typed_cfg.build_data_loader_dict()
    model_dict = dataclasses.asdict(typed_cfg.model)
    _merge_model_fields(normalized, model_dict, overwrite=True)

    normalized["data"] = dict(normalized.get("data") or {})
    if isinstance(train_config.get("model"), dict):
        _merge_model_fields(normalized, train_config["model"], overwrite=False)
    for key, value in train_config.items():
        if key in _INFERENCE_MODEL_KEYS or key == "qwen_vl_act_config_path":
            if value is not None:
                normalized[key] = value
        elif key == "data" and isinstance(value, dict):
            normalized["data"].update(value)

    return normalized


def register_data_backend(train_config: dict) -> None:
    """Register the data backend before typed data config construction."""
    from wall_x.data._registry import _set_data_backend

    data_section = train_config.get("data", {})
    dataset_type = train_config.get("dataset_type") or data_section.get(
        "dataset_type", "lerobot"
    )
    _set_data_backend(dataset_type)


def build_model_config(
    config_class,
    checkpoint_path: str,
    train_config: dict,
    train_config_path: str | None = None,
):
    """Build the HF model config and inject train_config-derived fields.

    ``config_class`` is supplied by the variant adapter. If the checkpoint does
    not include ``config.json``, this falls back to
    ``train_config["qwen_vl_act_config_path"]``.
    """
    ckpt_config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(ckpt_config_path):
        resolved = ckpt_config_path
    else:
        resolved = train_config.get("qwen_vl_act_config_path")
        if resolved is None or not os.path.exists(resolved):
            raise ValueError(
                f"cannot load model config: checkpoint file {ckpt_config_path} "
                f"does not exist, and fallback qwen_vl_act_config_path={resolved!r} "
                "does not exist either"
            )
    train_config["qwen_vl_act_config_path"] = resolved

    if resolved.endswith(".json"):
        model_config = config_class.from_json_file(resolved)
    else:
        model_config = config_class.from_pretrained(resolved)

    legacy_train_config = (
        normalize_train_config_for_inference(train_config, train_config_path)
        if train_config_path
        else train_config
    )
    model_config.update_model_config(legacy_train_config)
    model_config._attn_implementation = "sdpa"
    model_config.vision_config._attn_implementation = "flash_attention_2"

    return model_config


def build_data_config(train_config_path: str, train_config: dict):
    """Build the data config used by image resizing and preprocessing.

    New checkpoints use the typed schema. Older checkpoints may still use a flat
    legacy schema, so this falls back to the active data backend.
    """
    from wall_x.config.loader import load_config
    from wall_x.data import data_backend

    try:
        typed_cfg = load_config(train_config_path)
        backend = data_backend()
        if backend.supports("load_trainer_data_config"):
            return backend.load_trainer_data_config(typed_cfg)
    except Exception as e:
        logger.warning(
            "typed TrainConfig loading failed; falling back to raw data config: %s",
            e,
        )

    backend = data_backend()
    if backend.supports("load_trainer_data_config_from_yaml_dict"):
        return backend.load_trainer_data_config_from_yaml_dict(train_config)
    raise RuntimeError(
        f"active data backend {backend!r} cannot build a trainer data config"
    )
