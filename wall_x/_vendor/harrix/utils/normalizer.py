"""Build action/proprio normalizers and resolve the effective norm key.

Public inference artifacts must carry their own normalization data. This module
uses checkpoint-local ``norm_stats.json`` first, then checkpoint-side normalizer
state dicts, and finally an explicit ``customized_action_statistic_dof`` path.
It does not fall back to internal default action statistics.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch

from wall_x.data.backends.lerobot.utils import NormStats
from wall_x.model.core.action.normalizer import (
    NORM_KEY_SEP,
    Normalizer,
    make_norm_key,
)

logger = logging.getLogger(__name__)


def _load_norm_stats(norm_stats_path: str, action_key: str) -> NormStats:
    with open(norm_stats_path, "r") as f:
        norm_stats = json.load(f)
    q01 = torch.tensor(norm_stats["norm_stats"][action_key]["q01"])
    q99 = torch.tensor(norm_stats["norm_stats"][action_key]["q99"])
    return NormStats(min=q01, max=q99, delta=q99 - q01)


def build_normalizers(
    checkpoint_path: str,
    cfg,
    norm_key: str,
    end_effector_type: Optional[str] = None,
) -> tuple[Normalizer, Normalizer, str]:
    """Return action/proprio normalizers and the resolved norm key.

    ``cfg`` is the typed ``TrainConfig``.
    """
    norm_cfg = getattr(cfg.data, "normalizer_config", None) or {}

    def _norm_opt(key, default=None):
        value = norm_cfg.get(key)
        if value is None:
            value = getattr(cfg.data, key, None)
        return default if value is None else value

    norm_stats_path = os.path.join(checkpoint_path, "norm_stats.json")
    if os.path.exists(norm_stats_path):
        propri_stats = _load_norm_stats(norm_stats_path, "observation.state")
        action_stats = _load_norm_stats(norm_stats_path, "action")
        normalizer_propri = Normalizer.from_lerobot_norm_stats(propri_stats, norm_key)
        normalizer_action = Normalizer.from_lerobot_norm_stats(action_stats, norm_key)
    else:
        action_pth = os.path.join(checkpoint_path, "normalizer_action.pth")
        propri_pth = os.path.join(checkpoint_path, "normalizer_propri.pth")
        custom = _norm_opt("customized_action_statistic_dof")
        if not custom and not (os.path.exists(action_pth) and os.path.exists(propri_pth)):
            raise FileNotFoundError(
                "Public inference requires normalization data. Expected one of: "
                f"{norm_stats_path}; checkpoint-side normalizer_action.pth and "
                "normalizer_propri.pth; or data.normalizer_config."
                "customized_action_statistic_dof."
            )
        custom_stats = None
        if custom:
            with open(custom, "r") as f:
                custom_stats = json.load(f)
        min_key = _norm_opt("min_key", "min")
        delta_key = _norm_opt("delta_key", "delta")

        if os.path.exists(action_pth):
            normalizer_action = Normalizer.from_ckpt(action_pth)
        else:
            normalizer_action = Normalizer(
                custom_stats, cfg.task.dof_config, min_key=min_key, delta_key=delta_key
            )
        if os.path.exists(propri_pth):
            normalizer_propri = Normalizer.from_ckpt(propri_pth)
        else:
            normalizer_propri = Normalizer(
                custom_stats,
                cfg.task.agent_pos_config,
                min_key=min_key,
                delta_key=delta_key,
            )

    resolved = _resolve_norm_key(
        norm_key,
        normalizer_action,
        normalizer_propri,
        end_effector_type=end_effector_type,
    )
    return normalizer_action, normalizer_propri, resolved


def _resolve_norm_key(
    norm_key: str,
    normalizer_action: Normalizer,
    normalizer_propri: Normalizer,
    end_effector_type: Optional[str] = None,
) -> str:
    """Resolve a requested norm key against normalizer keys.

    Composite keys use ``dataset/end_effector_type``; a single composite match
    is selected automatically, while multiple matches require an explicit type.
    """
    available = sorted(
        set(normalizer_action.min.keys()) & set(normalizer_propri.min.keys())
    )
    if end_effector_type:
        composite = make_norm_key(norm_key, end_effector_type)
        if composite in available:
            return composite
        logger.warning(
            "composite norm key %r not found; available=%s", composite, available
        )
    if norm_key in available:
        return norm_key
    if not available:
        return norm_key

    prefix_matches = [k for k in available if k.startswith(f"{norm_key}_")]
    if len(prefix_matches) == 1:
        logger.warning(
            "norm_key=%r not found; using prefix fallback %r",
            norm_key,
            prefix_matches[0],
        )
        return prefix_matches[0]
    composite_matches = [
        k for k in available if k.startswith(f"{norm_key}{NORM_KEY_SEP}")
    ]
    if len(composite_matches) == 1:
        return composite_matches[0]
    if len(composite_matches) > 1:
        raise RuntimeError(
            f"norm_key={norm_key!r} matches multiple end_effector_types "
            f"{composite_matches}; set model.end_effector_type explicitly."
        )
    if len(available) == 1:
        logger.warning(
            "norm_key=%r not found; using the only available key %r",
            norm_key,
            available[0],
        )
        return available[0]

    logger.warning(
        "norm_key=%r not found; available=%s; returning the requested key unchanged",
        norm_key,
        available,
    )
    return norm_key
