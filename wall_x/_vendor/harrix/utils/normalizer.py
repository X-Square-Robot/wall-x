"""Build action/proprio normalizers and resolve the effective norm key.

Public inference artifacts must carry their own normalization data. This module
uses checkpoint-local ``normalizer_action.pth`` / ``normalizer_propri.pth`` first,
then ``norm_stats.json``, and finally an explicit ``customized_action_statistic_dof``
path. It does not fall back to internal default action statistics.
"""

from __future__ import annotations

import json
import logging
import os


from wall_x.data.backends.lerobot.utils import load_norm_stats
from wall_x.model.core.action.normalizer import Normalizer
from wall_x._vendor.harrix.utils.ckpt_load import resolve_checkpoint_dir

logger = logging.getLogger(__name__)


def _resolve_norm_stats_path(checkpoint_path: str, train_config: dict) -> str | None:
    ckpt_stats = os.path.join(checkpoint_path, "norm_stats.json")
    if os.path.exists(ckpt_stats):
        return ckpt_stats
    for key in ("norm_stats_path",):
        path = train_config.get(key)
        if path and os.path.exists(path):
            return path
    data = train_config.get("data") or {}
    path = data.get("norm_stats_path")
    if path and os.path.exists(path):
        return path
    return None


def _layout_configs(train_config: dict) -> tuple[dict, dict, dict]:
    task = train_config.get("task") or {}
    dof_config = dict(train_config.get("dof_config") or task.get("dof_config") or {})
    agent_pos_config = dict(
        train_config.get("agent_pos_config") or task.get("agent_pos_config") or {}
    )
    data = train_config.get("data") or {}
    key_mappings = dict(
        data.get("key_mappings") or {"action": "action", "state": "observation.state"}
    )
    return dof_config, agent_pos_config, key_mappings


def _load_custom_action_stats(train_config: dict) -> dict | None:
    custom = train_config.get("customized_action_statistic_dof", None)
    if not custom:
        return None
    with open(custom, "r") as f:
        return json.load(f)


def _normalizer_from_stats(
    action_stats: dict, train_config: dict, key: str
) -> Normalizer:
    return Normalizer(
        action_stats,
        train_config[key],
        min_key=train_config.get("min_key", "min"),
        delta_key=train_config.get("delta_key", "delta"),
    )


def _missing_normalizer_error(
    checkpoint_path: str, train_config: dict, resolved_ckpt: str | None = None
) -> FileNotFoundError:
    custom = train_config.get("customized_action_statistic_dof", None)
    data = train_config.get("data") or {}
    yaml_stats = train_config.get("norm_stats_path") or data.get("norm_stats_path")
    ckpt = resolved_ckpt or checkpoint_path
    return FileNotFoundError(
        "Public inference requires normalization data. Expected one of: "
        f"{os.path.join(ckpt, 'normalizer_action.pth')} and "
        f"{os.path.join(ckpt, 'normalizer_propri.pth')}; "
        f"{os.path.join(ckpt, 'norm_stats.json')}; train-config "
        f"norm_stats_path={yaml_stats!r}; or explicit "
        f"customized_action_statistic_dof. Current customized_action_statistic_dof={custom!r}."
    )


def build_normalizers(
    checkpoint_path: str,
    train_config: dict,
    norm_key: str,
) -> tuple[Normalizer, Normalizer, str]:
    """Return action/proprio normalizers and the resolved norm key."""
    resolved_ckpt = resolve_checkpoint_dir(checkpoint_path)
    action_pth = os.path.join(resolved_ckpt, "normalizer_action.pth")
    propri_pth = os.path.join(resolved_ckpt, "normalizer_propri.pth")

    if os.path.exists(action_pth) and os.path.exists(propri_pth):
        logger.info(
            "Loading normalizers from checkpoint: %s, %s",
            action_pth,
            propri_pth,
        )
        normalizer_action = Normalizer.from_ckpt(action_pth)
        normalizer_propri = Normalizer.from_ckpt(propri_pth)
    else:
        dof_config, agent_pos_config, key_mappings = _layout_configs(train_config)
        norm_stats_path = _resolve_norm_stats_path(resolved_ckpt, train_config)
        if norm_stats_path:
            if norm_stats_path != os.path.join(resolved_ckpt, "norm_stats.json"):
                logger.info("Using norm stats from %s", norm_stats_path)
            else:
                logger.info("Using norm stats from checkpoint: %s", norm_stats_path)
            stats = load_norm_stats(
                norm_stats_path,
                key_mappings,
                dof_config=dof_config,
                agent_pos_config=agent_pos_config,
            )
            normalizer_propri = Normalizer.from_lerobot_norm_stats(
                stats["state"], norm_key
            )
            normalizer_action = Normalizer.from_lerobot_norm_stats(
                stats["action"], norm_key
            )
        else:
            custom_stats = _load_custom_action_stats(train_config)
            if custom_stats is None and (
                not os.path.exists(action_pth) or not os.path.exists(propri_pth)
            ):
                raise _missing_normalizer_error(
                    checkpoint_path, train_config, resolved_ckpt
                )

            if os.path.exists(action_pth):
                normalizer_action = Normalizer.from_ckpt(action_pth)
            else:
                normalizer_action = _normalizer_from_stats(
                    custom_stats, train_config, "dof_config"
                )

            if os.path.exists(propri_pth):
                normalizer_propri = Normalizer.from_ckpt(propri_pth)
            else:
                normalizer_propri = _normalizer_from_stats(
                    custom_stats, train_config, "agent_pos_config"
                )

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
