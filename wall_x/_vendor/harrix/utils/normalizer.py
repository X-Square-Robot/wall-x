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

import torch

from wall_x.data.backends.lerobot.utils import NormStats
from wall_x.model.core.action.normalizer import Normalizer, pad_normalizer_to_dim
from wall_x._vendor.harrix.utils.train_config import (
    resolve_agent_pos_config,
    resolve_dof_config,
)

logger = logging.getLogger(__name__)


def _load_norm_stats(norm_stats_path: str, action_key: str) -> NormStats:
    with open(norm_stats_path, "r") as f:
        norm_stats = json.load(f)
    q01 = torch.tensor(norm_stats["norm_stats"][action_key]["q01"])
    q99 = torch.tensor(norm_stats["norm_stats"][action_key]["q99"])
    return NormStats(min=q01, max=q99, delta=q99 - q01)


def _load_custom_action_stats(train_config: dict) -> dict | None:
    custom = train_config.get("customized_action_statistic_dof", None)
    if not custom:
        return None
    with open(custom, "r") as f:
        return json.load(f)


def _normalizer_from_stats(action_stats: dict, train_config: dict, key: str) -> Normalizer:
    return Normalizer(
        action_stats,
        train_config[key],
        min_key=train_config.get("min_key", "min"),
        delta_key=train_config.get("delta_key", "delta"),
    )


def _missing_normalizer_error(checkpoint_path: str, train_config: dict) -> FileNotFoundError:
    custom = train_config.get("customized_action_statistic_dof", None)
    return FileNotFoundError(
        "Public inference requires normalization data. Expected one of: "
        f"{os.path.join(checkpoint_path, 'norm_stats.json')}; checkpoint-side "
        "normalizer_action.pth and normalizer_propri.pth; or an explicit "
        f"customized_action_statistic_dof path. Current customized_action_statistic_dof={custom!r}."
    )


def build_normalizers(
    checkpoint_path: str,
    train_config: dict,
    norm_key: str,
) -> tuple[Normalizer, Normalizer, str]:
    """Return action/proprio normalizers and the resolved norm key."""
    norm_stats_path = os.path.join(checkpoint_path, "norm_stats.json")
    if os.path.exists(norm_stats_path):
        propri_stats = _load_norm_stats(norm_stats_path, "observation.state")
        action_stats = _load_norm_stats(norm_stats_path, "action")
        normalizer_propri = Normalizer.from_lerobot_norm_stats(propri_stats, norm_key)
        normalizer_action = Normalizer.from_lerobot_norm_stats(action_stats, norm_key)
    else:
        action_pth = os.path.join(checkpoint_path, "normalizer_action.pth")
        propri_pth = os.path.join(checkpoint_path, "normalizer_propri.pth")
        custom_stats = _load_custom_action_stats(train_config)
        if custom_stats is None and (not os.path.exists(action_pth) or not os.path.exists(propri_pth)):
            raise _missing_normalizer_error(checkpoint_path, train_config)

        if os.path.exists(action_pth):
            normalizer_action = Normalizer.from_ckpt(action_pth)
        else:
            normalizer_action = _normalizer_from_stats(custom_stats, train_config, "dof_config")

        if os.path.exists(propri_pth):
            normalizer_propri = Normalizer.from_ckpt(propri_pth)
        else:
            normalizer_propri = _normalizer_from_stats(custom_stats, train_config, "agent_pos_config")

    action_dim = sum(resolve_dof_config(train_config).values())
    propri_dim = sum(resolve_agent_pos_config(train_config).values())
    pad_normalizer_to_dim(normalizer_action, action_dim, "action")
    pad_normalizer_to_dim(normalizer_propri, propri_dim, "propri")

    resolved = _resolve_norm_key(norm_key, normalizer_action, normalizer_propri)
    return normalizer_action, normalizer_propri, resolved


def _resolve_norm_key(
    norm_key: str,
    normalizer_action: Normalizer,
    normalizer_propri: Normalizer,
) -> str:
    """Resolve a requested norm key against normalizer keys."""
    available = sorted(
        set(normalizer_action.min.keys()) & set(normalizer_propri.min.keys())
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
