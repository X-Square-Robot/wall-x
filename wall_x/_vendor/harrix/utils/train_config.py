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


_TASK_INFERENCE_KEYS = (
    "dof_config",
    "agent_pos_config",
    "ar_dof_config",
    "action_horizon",
    "action_horizon_flow",
    "noise_scheduler",
    "use_state_string_representation",
)

_VIRTUAL_TAIL_KEYS = frozenset(("action_padding",))


def _move_virtual_keys_to_tail(layout: dict | None) -> dict | None:
    """Keep real LeRobot dims before virtual padding dims in inference layouts."""
    if not isinstance(layout, dict):
        return layout
    head = {k: v for k, v in layout.items() if k not in _VIRTUAL_TAIL_KEYS}
    tail = {k: v for k, v in layout.items() if k in _VIRTUAL_TAIL_KEYS}
    if not tail:
        return layout
    return {**head, **tail}


def _canonicalize_task_layouts(target: dict) -> None:
    if os.environ.get("WALLX_CANONICALIZE_VIRTUAL_DOF_ORDER", "1") == "0":
        return
    task = target.get("task")
    data = target.get("data")
    for key in ("dof_config", "agent_pos_config", "ar_dof_config"):
        value = _move_virtual_keys_to_tail(target.get(key))
        if value is not None:
            target[key] = value
        if isinstance(task, dict):
            task_value = _move_virtual_keys_to_tail(task.get(key))
            if task_value is not None:
                task[key] = task_value
        if isinstance(data, dict):
            data_value = _move_virtual_keys_to_tail(data.get(key))
            if data_value is not None:
                data[key] = data_value


def resolve_use_state_string_representation(train_config: dict) -> bool:
    """Read ``use_state_string_representation`` with task YAML as authority."""
    task = train_config.get("task") or {}
    if task.get("use_state_string_representation") is not None:
        return bool(task["use_state_string_representation"])
    if train_config.get("use_state_string_representation") is not None:
        return bool(train_config["use_state_string_representation"])
    data = train_config.get("data") or {}
    if data.get("use_state_string_representation") is not None:
        return bool(data["use_state_string_representation"])
    return False


def resolve_state_bins(train_config: dict, default: int = 256) -> int:
    """Read discretization bin count from flat or nested train config."""
    if train_config.get("state_bins") is not None:
        return int(train_config["state_bins"])
    data = train_config.get("data") or {}
    if data.get("state_bins") is not None:
        return int(data["state_bins"])
    return default


def resolve_agent_pos_config(train_config: dict) -> dict:
    """Return ``agent_pos_config`` from flat or ``task``-nested train YAML."""
    task = train_config.get("task") or {}
    agent_pos_config = train_config.get("agent_pos_config") or task.get(
        "agent_pos_config"
    )
    if not agent_pos_config:
        raise KeyError(
            "agent_pos_config missing from train config "
            "(expected top-level or task.agent_pos_config)"
        )
    return dict(agent_pos_config)


def resolve_dof_config(train_config: dict) -> dict:
    """Return ``dof_config`` from flat or ``task``-nested train YAML."""
    task = train_config.get("task") or {}
    dof_config = train_config.get("dof_config") or task.get("dof_config")
    if not dof_config:
        raise KeyError(
            "dof_config missing from train config (expected top-level or task.dof_config)"
        )
    return dict(dof_config)


def resolve_cam_names_from_train_config(train_config: dict) -> list[str] | None:
    """Infer model camera keys from ``data.key_mappings.camera`` (e.g. LIBERO)."""
    data = train_config.get("data") or {}
    key_mappings = data.get("key_mappings") or {}
    camera = key_mappings.get("camera") or {}
    if not camera:
        return None
    names: list[str] = []
    for value in camera.values():
        name = str(value)
        if name not in names:
            names.append(name)
    return names or None


def resolve_camera_label(cam_name: str, camera_name_mapping: dict | None = None) -> str:
    """Match training ``get_wallx_normal_text`` camera display names."""
    mapping = camera_name_mapping or {}
    return mapping.get(cam_name, cam_name)


def resolve_max_length(train_config: dict, default: int = 768) -> int:
    """Read ``max_length`` from flat or nested train YAML (collator default 768)."""
    data = train_config.get("data") or {}
    if train_config.get("max_length") is not None:
        return int(train_config["max_length"])
    if data.get("max_length") is not None:
        return int(data["max_length"])
    raw = train_config.get("_raw_data") or {}
    if raw.get("max_length") is not None:
        return int(raw["max_length"])
    return default


# Fields ``load_wallx_processors`` and model construction read from a flat dict.
_ACTION_TOKENIZER_KEYS = (
    "action_tokenizer_type",
    "action_tokenizer_path",
    "action_tokenizer_checkpoint_path",
    "action_tokenizer_config_dir",
    "action_tokenizer",
)

_INFERENCE_MODEL_KEYS = (
    "processor_path",
    "pretrained_path",
    "config_path",
    *_ACTION_TOKENIZER_KEYS,
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


def _mirror_task_fields(target: dict, task: dict) -> None:
    """Copy typed task fields into the flat/data legacy inference mirrors."""
    if not isinstance(task, dict):
        return
    data = target.setdefault("data", {})
    if not isinstance(data, dict):
        data = {}
        target["data"] = data
    for key in _TASK_INFERENCE_KEYS:
        value = task.get(key)
        if value is None:
            continue
        target[key] = value
        data[key] = value


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


def strip_action_tokenizer_fields(train_config: dict) -> None:
    """Drop action-tokenizer fields so flow inference can skip AR tokenizer setup."""
    for key in _ACTION_TOKENIZER_KEYS:
        train_config.pop(key, None)
    model = train_config.get("model")
    if isinstance(model, dict):
        for key in _ACTION_TOKENIZER_KEYS:
            model.pop(key, None)


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
        _mirror_task_fields(train_config, train_config.get("task") or {})
        _canonicalize_task_layouts(train_config)
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

    _mirror_task_fields(normalized, dataclasses.asdict(typed_cfg.task))
    _canonicalize_task_layouts(normalized)
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
