"""LeRobot data loading bridge for typed training configs."""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist

from wall_x.data.backends.lerobot.config import LerobotConfig

logger = logging.getLogger(__name__)


class _LerobotDatasetWrapper:
    """Trainer-facing wrapper aligning PreprocessedDataset with v1 API.

    PreprocessedDataset internally switches ``self._dataset`` between
    its train/val splits via ``_train()`` / ``_eval()``. Its
    ``get_train_dataloader`` / ``get_val_dataloader`` return
    ``(dataloader, sampler)`` tuples and no-argument calls are supported
    (they read rank/world_size/seed from the inner object itself).

    This wrapper:
    - Caches the rebuilt train dataloader / sampler so
      ``set_epoch(epoch)`` can reset shuffling per-epoch.
    - Owns the val dataloader so the trainer's ``val_loop`` can do
      ``self.dataset.get_val_dataloader()`` and iterate directly (matching
      what the v1/v2 wrappers return).
    """

    def __init__(
        self,
        inner,
        train_dataloader: torch.utils.data.DataLoader,
        train_sampler,
        train_num: int,
        val_dataloader: torch.utils.data.DataLoader = None,
        val_num: int = 0,
    ):
        self._inner = inner
        self._train_dataloader = train_dataloader
        self._train_sampler = train_sampler
        self._train_num = train_num
        self._val_dataloader = val_dataloader
        self.global_train_iters = mp.Value("i", train_num)
        self.global_val_iters = mp.Value("i", val_num)

    def __len__(self) -> int:
        return self._train_num

    def _activate_train_split(self) -> None:
        if hasattr(self._inner, "_train"):
            self._inner._train()

    def get_train_dataloader(self):
        self._activate_train_split()
        return self._train_dataloader

    def get_val_dataloader(self):
        # PreprocessedDataset shares one ``_dataset`` pointer between its
        # train and val splits (flipped by ``_train()`` / ``_eval()``).
        # The val DataLoader's DistributedSampler caches total_size sized
        # to the val split but ``__iter__`` reads ``len(self.dataset)``
        # live — if a preceding train_loop left the pointer at train, that
        # live len is ~20× total_size and DistributedSampler asserts.
        # Rebuild each time so ``_eval()`` runs and a fresh sampler is
        # snapped to the current (val) split length. Mirrors the train-side
        # rebuild-on-every-epoch pattern.
        if self._val_dataloader is None:
            return None
        self._val_dataloader, _ = self._inner.get_val_dataloader()
        return self._val_dataloader

    def set_epoch(self, epoch: int) -> None:
        """Seed the per-epoch shuffle in the train DistributedSampler."""
        self._activate_train_split()
        if self._train_sampler is not None and hasattr(
            self._train_sampler, "set_epoch"
        ):
            self._train_sampler.set_epoch(epoch)


def load_trainer_data_config(cfg: Any) -> LerobotConfig:
    """Build the inference/trainer data config from a typed TrainConfig."""
    raw_yaml = dict(getattr(cfg, "_raw_yaml", {}) or {})
    raw_data = dict(getattr(cfg, "_raw_data", {}) or {})
    data = getattr(cfg, "data", None)

    data_section = dict(raw_yaml.get("data", {}) or {})
    data_section.update(raw_data)

    if getattr(cfg, "task", None) is not None:
        data_section.setdefault(
            "use_state_string_representation",
            cfg.task.use_state_string_representation,
        )
        data_section.setdefault("state_bins", raw_data.get("state_bins", 256))

    if data is not None:
        for key in (
            "resolution",
            "train_test_split",
            "priority_order",
            "camera_name_mapping",
            "use_state_string_representation",
            "state_bins",
        ):
            value = getattr(data, key, None)
            if value is not None:
                data_section.setdefault(key, value)

    raw_yaml["data"] = data_section
    raw_yaml.setdefault("model_type", getattr(cfg, "model_type", "qwen2_5"))
    return load_trainer_data_config_from_yaml_dict(raw_yaml)


def load_trainer_data_config_from_yaml_dict(yaml_dict: Dict[str, Any]) -> LerobotConfig:
    """Build the LeRobot runtime config from a raw training YAML dict."""
    return LerobotConfig.from_yaml_dict(yaml_dict)


def _build_flat_config(cfg: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Map typed TrainConfig → (flat_config, lerobot_config) for legacy entry.

    ``load_lerobot_data`` expects a 2509-style flat dict plus a separate
    ``lerobot_config`` carrying ``repo_id`` / ``root``. This function is
    the one place that translation lives; keep it surgical so future
    field additions on ``LeRobotDataConfig`` do not require touching the
    legacy loader.
    """
    model = cfg.model
    data = cfg.data
    hp = cfg.hyperparams
    raw = getattr(cfg, "_raw_yaml", {}) or {}
    raw_data = dict(getattr(cfg, "_raw_data", {}) or {})

    lerobot_cfg = dict(data.lerobot_config or {})
    if "repo_id" not in lerobot_cfg:
        raise ValueError(
            "lerobot requires data.lerobot_config.repo_id to be set "
            "(HuggingFace LeRobot dataset id)."
        )

    data_section: Dict[str, Any] = {
        "key_mappings": data.key_mappings,
        "action_horizon": cfg.task.action_horizon,
        "train_test_split": data.train_test_split,
        "seed": hp.seed,
        "resolution": data.resolution,
    }
    if data.priority_order is not None:
        data_section["priority_order"] = data.priority_order
    if data.camera_name_mapping is not None:
        data_section["camera_name_mapping"] = data.camera_name_mapping
    data_section.setdefault(
        "use_state_string_representation",
        cfg.task.use_state_string_representation,
    )
    data_section.setdefault(
        "state_bins",
        raw_data.get("state_bins", raw.get("state_bins", 256)),
    )

    # Dof/agent_pos totals for the collator's zero-pad step. When resuming
    # from a checkpoint trained on a larger action space (e.g. bus2602 26-dim
    # dof while libero data is 7-dim), task.dof_config should include an
    # ``action_padding`` key that absorbs the diff; the collator right-pads
    # action/agent_pos tensors to these totals with dof_mask/agent_pos_mask
    # zeroed on padded dims so loss doesn't flow through them.
    dof_total = int(sum((cfg.task.dof_config or {}).values()))
    agent_pos_total = int(sum((cfg.task.agent_pos_config or {}).values()))

    flat: Dict[str, Any] = {
        "model_type": cfg.model_type,
        "processor_path": getattr(model, "processor_path", "") or "",
        "norm_stats_path": data.norm_stats_path or raw.get("norm_stats_path"),
        "batch_size_per_gpu": hp.batch_size_per_gpu,
        "eval_batch_size_per_gpu": raw.get(
            "eval_batch_size_per_gpu", hp.batch_size_per_gpu
        ),
        "num_workers": data.num_workers,
        "padding_side": data.padding_side,
        "use_fast_tokenizer": data.use_fast_tokenizer,
        "action_tokenizer_path": data.action_tokenizer_path,
        "noise_scheduler": data.noise_scheduler or {},
        "dof_total_dim": dof_total,
        "agent_pos_total_dim": agent_pos_total,
        "dof_config": dict(cfg.task.dof_config or {}),
        "agent_pos_config": dict(cfg.task.agent_pos_config or {}),
        "use_state_string_representation": cfg.task.use_state_string_representation,
        "state_bins": int(
            raw_data.get("state_bins")
            or data_section.get("state_bins")
            or raw.get("state_bins")
            or 256
        ),
        "data": data_section,
    }
    return flat, lerobot_cfg


def load_lerobot_v2(
    cfg: Any,
) -> Tuple[_LerobotDatasetWrapper, torch.utils.data.DataLoader, int]:
    """Build lerobot (wrapper, dataloader, train_num) from TrainConfig.

    The third return value ``train_num`` is a snapshot of
    ``len(train_dataloader)`` at construction time. It matches
    ``wrapper.global_train_iters.value`` initially but does not track
    subsequent rebuilds inside ``set_epoch`` — callers doing dynamic
    resampling should read from the mp.Value, not from this snapshot.
    """
    from wall_x.data.backends.lerobot.loader import load_lerobot_data

    flat_cfg, lerobot_cfg = _build_flat_config(cfg)

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    seed = cfg.hyperparams.seed
    inner, _ = load_lerobot_data(
        flat_cfg,
        lerobot_cfg,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )

    # PreprocessedDataset.get_*_dataloader returns (dataloader, sampler).
    # Build val first, train second, so the inner ``_dataset`` pointer is
    # left at the train split when we finish — workers fork from that
    # state on first iteration.
    val_dataloader, _ = inner.get_val_dataloader()
    val_num = len(val_dataloader) if val_dataloader is not None else 0

    train_dataloader, train_sampler = inner.get_train_dataloader()
    train_num = len(train_dataloader)

    if rank == 0:
        logger.info(
            "\n%s\nLeRobot Data Loading Configuration:\n"
            "  RANK: %d\n  WORLD SIZE: %d\n"
            "  BATCH SIZE PER DEVICE: %d\n  GLOBAL BATCH SIZE: %d\n"
            "  TRAIN BATCHES: %d\n  VAL BATCHES: %d\n"
            "  NUM WORKERS: %d\n  REPO ID: %s\n%s",
            "=" * 50,
            rank,
            world_size,
            flat_cfg["batch_size_per_gpu"],
            flat_cfg["batch_size_per_gpu"] * world_size,
            train_num,
            val_num,
            flat_cfg["num_workers"],
            lerobot_cfg.get("repo_id"),
            "=" * 50,
        )

    wrapper = _LerobotDatasetWrapper(
        inner,
        train_dataloader,
        train_sampler,
        train_num,
        val_dataloader=val_dataloader,
        val_num=val_num,
    )
    return wrapper, train_dataloader, train_num


def build(cfg, ctx):
    """Backend Protocol entry — returns a ``DataBundle``.

    Wraps ``load_lerobot_v2`` (which returns the trainer-facing triple)
    into the unified ``DataBundle`` shape every backend exposes.
    """
    from wall_x.data._bundle import DataBundle

    wrapper, train_dataloader, train_num = load_lerobot_v2(cfg)

    # PreprocessedDataset shares one ``self._dataset`` pointer between
    # train and val splits (flipped by ``_train()`` / ``_eval()``).
    # ``wrapper.get_val_dataloader()`` flips the pointer to val. Flip back once
    # here so the initial train loop starts from the right split even if callers
    # inspect the raw ``train_dataloader`` before invoking ``set_epoch``.
    val_loader = wrapper.get_val_dataloader()
    inner = wrapper._inner
    if hasattr(inner, "_train"):
        inner._train()

    return DataBundle(
        dataset=wrapper,
        train_loader=train_dataloader,
        val_loader=val_loader,
        train_iters=train_num,
        val_iters=wrapper.global_val_iters.value,
        set_epoch=wrapper.set_epoch,
    )
