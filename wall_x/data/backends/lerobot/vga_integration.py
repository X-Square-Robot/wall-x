"""VGA(wan) ops for the LeRobot backend, dispatched via data_backend().<op>()."""

from __future__ import annotations

import os
from typing import Any, Tuple


def build_vga_train_config(cfg: Any) -> dict:
    """Assemble the LerobotVGADataset config dict from a typed TrainConfig."""
    d = cfg.data
    lr = d.lerobot_config or {}
    m = cfg.model
    if not lr.get("repo_id"):
        raise ValueError(
            "lerobot requires data.lerobot_config.repo_id to be set "
            "(HuggingFace LeRobot dataset id)."
        )
    return {
        "repo_id": lr.get("repo_id"),
        "root": lr.get("root"),
        "episodes": lr.get("episodes"),
        "norm_stats_path": d.norm_stats_path,
        "key_mappings": d.key_mappings,
        "num_views": m.num_views,
        "image_history_length": d.image_history_length,
        "future_image_length": d.future_image_length,
        "future_image_interval": d.future_image_interval,
        "action_history_length": getattr(d, "action_history_length", 0),
        "action_horizon": cfg.task.action_horizon,
        "resolution": (int(m.height), int(m.width)),
        "dof_config": cfg.task.dof_config,
        "agent_pos_config": cfg.task.agent_pos_config,
        "batch_size": cfg.hyperparams.batch_size_per_gpu,
        "num_workers": d.num_workers,
        "seed": getattr(cfg.hyperparams, "seed", 42),
        "train_test_split": d.train_test_split,
    }


def build_vga_train_data(cfg: Any, rank: int, world_size: int, logger: Any = None):
    """Build the LeRobot->VGA dataset; returns (dataset, dataloader, train_num)."""
    from wall_x.data.backends.lerobot.vga_video_dataset import (
        make_lerobot_vga_dataset,
    )

    ds_cfg = build_vga_train_config(cfg)
    dataset = make_lerobot_vga_dataset(ds_cfg)
    train_num = dataset.global_train_iters.value
    train_dataloader = dataset.get_train_dataloader()
    if rank == 0 and logger is not None:
        logger.info(
            "\n"
            + "=" * 50
            + "\n[lerobot->VGA] Data Loading:\n"
            + f"✦ repo_id: {ds_cfg['repo_id']}\n"
            + f"✦ batch_size: {ds_cfg['batch_size']}, world_size: {world_size}\n"
            + f"✦ action_horizon: {ds_cfg['action_horizon']}, "
            + f"future_image_length: {ds_cfg['future_image_length']}, "
            + f"interval: {ds_cfg['future_image_interval']}\n"
            + f"✦ BATCHES PER RANK PER EPOCH (train_num): {train_num}\n"
            + "=" * 50
        )
    return dataset, train_dataloader, train_num


def build_vga_normalizers(
    dataload_config: dict,
    checkpoint_path: str,
    norm_key: str,
    action_dim: int,
) -> Tuple[Any, Any, str]:
    """(action, proprio) normalizers from a LeRobot norm_stats file; returns
    (normalizer_action, normalizer_propri, norm_stats_path)."""
    from wall_x.data.backends.lerobot.utils import load_norm_stats
    from wall_x.model.core.action.normalizer import (
        create_normalizers_from_lerobot_norm_stats,
    )

    config_stats_path = dataload_config.get("norm_stats_path")
    norm_stats_path = config_stats_path
    if not norm_stats_path or not os.path.exists(norm_stats_path):
        norm_stats_path = os.path.join(checkpoint_path, "norm_stats.json")
        if not os.path.exists(norm_stats_path):
            raise FileNotFoundError(
                "No LeRobot norm stats found: data.norm_stats_path="
                f"{config_stats_path!r} and checkpoint-local "
                f"{norm_stats_path} both missing."
            )
    key_mappings = dataload_config.get("key_mappings")
    propri_dim = (
        sum((dataload_config.get("agent_pos_config") or {}).values()) or action_dim
    )
    norm_stats = load_norm_stats(norm_stats_path, key_mappings)
    normalizer_action, normalizer_propri = create_normalizers_from_lerobot_norm_stats(
        norm_stats, norm_key, action_dim, propri_dim
    )
    return normalizer_action, normalizer_propri, norm_stats_path
