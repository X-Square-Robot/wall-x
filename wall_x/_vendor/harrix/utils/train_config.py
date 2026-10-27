"""Load checkpoint-side train config and construct runtime config objects.

Inference adapters use this module to load the train config through wall_x's
single inference-side loader (typed schema only; migrate legacy files with
``scripts/migrate_legacy_yml.py``), build the HF model config, and build the
data config required by preprocessing.
"""

from __future__ import annotations

import dataclasses
import logging
import os

from wall_x._vendor.harrix.utils.dataset_v2 import (
    build_inference_data_config,
    merge_private_data_config,
    normalize_dataset_config_for_inference,
)

logger = logging.getLogger(__name__)

# Task keys mirrored into the data: section, matching the training-side dict
# shape (``TrainConfig.build_data_loader_dict``).
_TASK_DATA_SECTION_KEYS = (
    "dof_config",
    "agent_pos_config",
    "action_horizon",
    "action_horizon_flow",
    "ar_dof_config",
    "use_state_string_representation",
)


def load_train_config_with_ckpt_overlay(
    train_config_path: str,
    checkpoint_path: str,
):
    """Load the checkpoint train config through the single wall_x loader.

    Returns ``(cfg, train_config)``: the typed ``TrainConfig`` (ckpt overlays
    applied; the single source consumers should read) plus its dict view
    shaped like the training-side ``TrainConfig.build_data_loader_dict``
    output -- kept solely for the dict-contract consumers listed above.
    """
    from wall_x.config.loader import dump_config, load_train_config_from_checkpoint

    cfg = load_train_config_from_checkpoint(checkpoint_path, train_config_path)
    train_config = dump_config(cfg)
    merge_private_data_config(cfg, train_config)
    task_dict = dataclasses.asdict(cfg.task)
    train_config.update(task_dict)
    data_section = train_config.setdefault("data", {})
    for key in _TASK_DATA_SECTION_KEYS:
        value = task_dict.get(key)
        if value is not None:
            data_section.setdefault(key, value)
    return cfg, train_config


def build_model_config(config_class, checkpoint_path: str, cfg, train_config: dict):
    """Build the HF model config and inject train_config-derived fields.

    ``config_class`` is supplied by the variant adapter. A checkpoint-local
    ``config.json`` wins; otherwise ``cfg.model.config_path`` is used.
    """
    ckpt_config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(ckpt_config_path):
        resolved = ckpt_config_path
    else:
        resolved = getattr(cfg.model, "config_path", None)
        if not resolved or not os.path.exists(resolved):
            raise ValueError(
                f"cannot load model config: checkpoint file {ckpt_config_path} "
                f"does not exist, and fallback cfg.model.config_path={resolved!r} "
                "does not exist either"
            )

    if resolved.endswith(".json"):
        model_config = config_class.from_json_file(resolved)
    else:
        model_config = config_class.from_pretrained(resolved)

    model_config.update_model_config(train_config)
    model_config._attn_implementation = "sdpa"
    model_config.vision_config._attn_implementation = "flash_attention_2"

    return model_config


def build_data_config(cfg, train_config: dict):
    """Build the data config used by image resizing and preprocessing.

    x2robot_v2 / lerobot: ``load_x2r_data_config`` returns None for these
    backends, so project the checkpoint-embedded effective processor contract.
    This is shared with the serving stack and does not depend on the removed
    ``x2robot_utils.vision_params`` shim.
    """
    if cfg.data.dataset_type in ("x2robot_v2", "lerobot"):
        processors, source = normalize_dataset_config_for_inference(train_config)
        return build_inference_data_config(train_config, processors, source)

    from wall_x.trainer.adapters.base_adapter import load_trainer_data_config

    return load_x2r_data_config(cfg)
