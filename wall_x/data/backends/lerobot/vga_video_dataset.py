"""Standalone LeRobot -> VGA(wan) video dataset.

This module produces *temporal video* batches for the wan video-diffusion model
(``wall_x/model/vga/modeling/modeling_wan_act.py``), NOT single-frame VLA
batches.  The emitted batch dict matches exactly what
``WanModel.forward_preprocess`` / ``_process_video_data`` consume:

  - ``obs``      : {viewK: [B, Th, H, W, C] uint8 [0,255], "agent_pos": [B, Th, Dstate] norm}
  - ``next_obs`` : {viewK: [B, Tf, H, W, C] uint8 [0,255]}
  - ``camera_names`` : List[List[str]] (len B), each = model view names in slot order
  - ``action``         : [B, action_horizon, action_dim] normalized
  - ``action_history`` : [B, Hh, action_dim] normalized PAST actions
                         a_{t-Hh}..a_{t-1}; only present when
                         action_history_length > 0 (never the target chunk)
  - ``dof_mask``       : [B, action_horizon, action_dim] (1.0 valid / 0.0 padded)
  - ``instruction``    : List[str] (len B) - the lerobot task string
  - ``dataset_name``   : List[str] (len B) - norm-stats dataset key

Frames flow into ``tensor_video_to_vae_input`` (data_utils.py:62) which accepts
``[B, T, H, W, C]`` in range [0,255]; we emit uint8 to mirror the x2robot
``vg_processor`` path (torchcodec VideoDecoder output, see vg_processor.py:1583).

Normalization mirrors ``wall_x/data/backends/lerobot/loader.py``: min-max with
min=q01, delta=q99-q01, scaled to [-1, 1] and clamped (utils.load_norm_stats +
DataCollator._normalize).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Normalization helpers (mirror wall_x/data/backends/lerobot/{utils,loader}.py)
# --------------------------------------------------------------------------- #
def _load_minmax_stats(
    norm_stats_path: str, action_key: str, state_key: str
) -> Tuple[Dict[str, torch.Tensor], str]:
    """Load min/delta norm stats from a libero_norm_stats.json-style file.

    Returns (stats_dict, dataset_key) where dataset_key is the top-level
    norm_stats container key chosen for the ``dataset_name`` field.

    Matches ``load_norm_stats``: min = q01, delta = q99 - q01.
    """
    with open(norm_stats_path, "r") as f:
        raw = json.load(f)

    container = raw["norm_stats"]

    def _stat(key: str) -> Dict[str, torch.Tensor]:
        q01 = torch.tensor(container[key]["q01"], dtype=torch.float32)
        q99 = torch.tensor(container[key]["q99"], dtype=torch.float32)
        delta = q99 - q01
        return {"min": q01, "delta": delta}

    stats = {"action": _stat(action_key), "state": _stat(state_key)}
    # The file has a single ``norm_stats`` container key; the model's
    # dataset-name resolver expects a dataset name string. There is no
    # explicit per-dataset key inside libero_norm_stats.json, so the dataset
    # name is supplied separately by the caller (repo_id). The returned key is
    # only informational.
    return stats, "norm_stats"


def _normalize(
    x: torch.Tensor, min_stat: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    """Min-max normalize to [-1, 1] then clamp (DataCollator._normalize)."""
    delta = torch.where(delta == 0, torch.ones_like(delta), delta)
    out = (x - min_stat) / delta
    out = out * 2 - 1
    return torch.clamp(out, -1, 1)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class _IntValue:
    """Tiny holder so callers can read ``.global_train_iters.value``."""

    def __init__(self, value: int):
        self.value = int(value)


class LerobotVGADataset(Dataset):
    """LeRobot-backed dataset emitting VGA(wan) temporal video batches.

    See module docstring for the exact emitted batch contract.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)

        # ---- repo / paths -------------------------------------------------
        self.repo_id: str = config["repo_id"]
        self.root: Optional[str] = config.get("root", None)
        self.norm_stats_path: str = config["norm_stats_path"]

        # ---- key mappings (lerobot key -> model view / state / action) ----
        # key_mappings = {
        #   "camera": {<lerobot_img_key>: <model_view_name>, ...},  # ordered
        #   "state": <lerobot_state_key>,
        #   "action": <lerobot_action_key>,
        # }
        self.key_mappings: Dict[str, Any] = config["key_mappings"]
        self.cam_mapping: Dict[str, str] = dict(self.key_mappings["camera"])
        self.state_key: str = self.key_mappings["state"]
        self.action_key: str = self.key_mappings["action"]

        # Camera order is the insertion order of key_mappings["camera"].
        self.lerobot_cam_keys: List[str] = list(self.cam_mapping.keys())
        self.model_view_names: List[str] = [
            self.cam_mapping[k] for k in self.lerobot_cam_keys
        ]
        self.num_views: int = int(config.get("num_views", len(self.lerobot_cam_keys)))
        if self.num_views != len(self.lerobot_cam_keys):
            raise ValueError(
                f"num_views={self.num_views} != #cameras in key_mappings "
                f"({len(self.lerobot_cam_keys)})"
            )
        self.view_slot_keys: List[str] = [f"view{i + 1}" for i in range(self.num_views)]

        # ---- temporal sampling -------------------------------------------
        self.image_history_length: int = int(config.get("image_history_length", 1))
        self.future_image_length: int = int(config.get("future_image_length", 16))
        self.future_image_interval: int = int(config.get("future_image_interval", 2))
        self.action_horizon: int = int(config.get("action_horizon", 32))
        # Past actions fed as the model's action_history tokens. 0 = no history
        # (matches inference, which has no executed-action plumbing). When > 0,
        # the history is the PAST actions a_{t-Hh}..a_{t-1} -- NEVER the current
        # chunk's own actions: a_t is the first prediction target, and feeding it
        # as history lets the model copy near-constant dims (e.g. the libero
        # gripper) instead of learning them (openloop grip corr 0.97 with the
        # leaked token vs 0.20 without, ckpt 16_220000).
        self.action_history_length: int = int(config.get("action_history_length", 0))
        if self.image_history_length != 1:
            # The contract (and event_level=False openloop path) assumes Th=1.
            logger.warning(
                "image_history_length=%s != 1; obs/agent_pos will carry %s frames.",
                self.image_history_length,
                self.image_history_length,
            )

        # ---- resolution (per-view target H,W) -----------------------------
        # resolution: int | (H, W) | {view_name_or_lerobot_key: int|(H,W)}
        self.resolution = config.get("resolution", 256)

        # ---- dof / agent_pos config (optional tail-pad target widths) -----
        self.dof_config: Optional[Dict[str, int]] = config.get("dof_config", None)
        self.agent_pos_config: Optional[Dict[str, int]] = config.get(
            "agent_pos_config", None
        )
        self.dof_total_dim: Optional[int] = (
            int(sum(self.dof_config.values())) if self.dof_config else None
        )
        self.agent_pos_total_dim: Optional[int] = (
            int(sum(self.agent_pos_config.values())) if self.agent_pos_config else None
        )

        # ---- loader knobs -------------------------------------------------
        self.batch_size: int = int(config.get("batch_size", 1))
        self.num_workers: int = int(config.get("num_workers", 0))
        self.seed: int = int(config.get("seed", 42))
        self.train_test_split: float = float(config.get("train_test_split", 0.95))

        # ---- norm stats ---------------------------------------------------
        self.norm_stats, self.norm_dataset_key = _load_minmax_stats(
            self.norm_stats_path, self.action_key, self.state_key
        )
        # dataset_name used by the model normalizer resolver = repo_id.
        self.dataset_name: str = config.get("dataset_name", self.repo_id)

        # ---- build LeRobotDataset ----------------------------------------
        meta = LeRobotDatasetMetadata(self.repo_id, root=self.root)
        self.fps: float = float(meta.fps)
        total_episodes: int = int(meta.total_episodes)

        # episode train/test split (deterministic, episode-level)
        split_idx = int(total_episodes * self.train_test_split)
        if split_idx < 1:
            raise ValueError(
                f"train_test_split={self.train_test_split} on {total_episodes} "
                f"episodes yields 0 train episodes."
            )
        episodes_override = config.get("episodes", None)
        if episodes_override is not None:
            all_eps = list(episodes_override)
            split_idx = max(1, int(len(all_eps) * self.train_test_split))
            self.train_episodes = all_eps[:split_idx]
        else:
            self.train_episodes = list(range(split_idx))

        self.delta_timestamps = self._build_delta_timestamps()

        self.dataset = LeRobotDataset(
            repo_id=self.repo_id,
            root=self.root,
            episodes=self.train_episodes,
            delta_timestamps=self.delta_timestamps,
            video_backend="pyav",
        )

        # ---- valid frame index list (drop frames where future window runs
        #      past the episode end). max future offset spans the future image
        #      (frame + future_image_length*future_image_interval) and the
        #      action horizon (frame + action_horizon - 1). A frame is valid
        #      iff frame_index + max_future_offset <= ep_len - 1.
        self.max_future_offset = max(
            self.future_image_length * self.future_image_interval,
            self.action_horizon - 1,
        )
        self.valid_indices = self._compute_valid_indices()
        if len(self.valid_indices) == 0:
            raise ValueError("No valid frames after dropping episode-end frames.")

        self.global_train_iters = _IntValue(
            len(self.valid_indices) // self._world_size() // self.batch_size
        )

    @staticmethod
    def _world_size() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    # ------------------------------------------------------------------ #
    def _build_delta_timestamps(self) -> Dict[str, List[float]]:
        fps = self.fps
        # obs frames: image_history_length consecutive frames ending at 0.
        # i=0 -> oldest, ..., last -> current (delta 0).
        hist_deltas = [
            (i - (self.image_history_length - 1)) / fps
            for i in range(self.image_history_length)
        ]
        # future frames: deltas [interval, 2*interval, ...] (Tf frames).
        future_deltas = [
            (j + 1) * self.future_image_interval / fps
            for j in range(self.future_image_length)
        ]
        img_deltas = hist_deltas + future_deltas

        dt: Dict[str, List[float]] = {}
        for cam_key in self.lerobot_cam_keys:
            dt[cam_key] = img_deltas
        # agent_pos: Th frames (delta hist), same as obs history.
        dt[self.state_key] = hist_deltas
        # action: [past history | prediction target]. History (when enabled) is
        # the PAST actions a_{t-Hh}..a_{t-1} at negative deltas; the target chunk
        # is a_t..a_{t+H-1}. lerobot edge-pads out-of-range past steps with a_0
        # (leaks a_0 only at each episode's very first frames -- standard).
        dt[self.action_key] = [
            t / fps for t in range(-self.action_history_length, self.action_horizon)
        ]
        return dt

    def _compute_valid_indices(self) -> List[int]:
        hf = self.dataset.hf_dataset
        episode_index = np.asarray(hf["episode_index"])
        frame_index = np.asarray(hf["frame_index"])
        valid = []
        for ep in np.unique(episode_index):
            mask = episode_index == ep
            ep_len = int(mask.sum())
            ep_positions = np.nonzero(mask)[0]
            ep_frame_idx = frame_index[mask]
            # keep frames whose full future window fits in-episode.
            keep = ep_frame_idx + self.max_future_offset <= (ep_len - 1)
            valid.extend(ep_positions[keep].tolist())
        valid.sort()
        return valid

    # ------------------------------------------------------------------ #
    def _resolve_target_hw(self, lerobot_cam_key: str) -> Optional[Tuple[int, int]]:
        view_name = self.cam_mapping[lerobot_cam_key]
        res = self.resolution
        if res is None:
            return None
        if isinstance(res, dict):
            r = res.get(view_name, res.get(lerobot_cam_key, None))
        else:
            r = res
        if r is None:
            return None
        if isinstance(r, (list, tuple)):
            return int(r[0]), int(r[1])
        return int(r), int(r)

    def _frames_to_uint8_hwc(
        self, frames: torch.Tensor, lerobot_cam_key: str
    ) -> torch.Tensor:
        """Convert lerobot frames [T, C, H, W] float[0,1] -> [T, H, W, C] uint8[0,255].

        Resizes to the per-view target resolution. Mirrors the x2robot
        vg_processor path which feeds uint8 [0,255] frames in [T,H,W,C]
        (vg_processor.py:1583) into ``tensor_video_to_vae_input``.
        """
        if frames.dim() != 4:
            raise ValueError(f"expected [T,C,H,W], got shape {tuple(frames.shape)}")
        # lerobot returns float in [0,1]; some formats may return uint8.
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        else:
            frames = frames.float()

        target = self._resolve_target_hw(lerobot_cam_key)
        if target is not None and (
            frames.shape[2] != target[0] or frames.shape[3] != target[1]
        ):
            frames = F.interpolate(
                frames, size=target, mode="bilinear", align_corners=False
            )
        frames = frames.clamp(0.0, 1.0)
        frames = (frames * 255.0).round().to(torch.uint8)
        # [T, C, H, W] -> [T, H, W, C]
        return frames.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _tail_pad(
        x: torch.Tensor, mask: torch.Tensor, total_dim: Optional[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tail-pad last dim to ``total_dim`` (zeros) and zero the mask there.

        Mirrors DataCollator.__call__ dof_total / agent_pos_total padding.
        """
        if total_dim is None:
            return x, mask
        cur = x.shape[-1]
        if cur >= total_dim:
            return x, mask
        pad = total_dim - cur
        x = F.pad(x, (0, pad))
        mask = F.pad(mask, (0, pad))
        return x, mask

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        real_idx = self.valid_indices[i]
        item = self.dataset[real_idx]

        Th = self.image_history_length
        Tf = self.future_image_length

        obs: Dict[str, Any] = {}
        next_obs: Dict[str, Any] = {}
        for slot, cam_key in zip(self.view_slot_keys, self.lerobot_cam_keys):
            frames = self._frames_to_uint8_hwc(item[cam_key], cam_key)  # [Th+Tf,H,W,C]
            if frames.shape[0] != Th + Tf:
                raise ValueError(
                    f"frame count mismatch for {cam_key}: got {frames.shape[0]}, "
                    f"expected {Th + Tf}"
                )
            obs[slot] = frames[:Th]  # [Th, H, W, C]
            next_obs[slot] = frames[Th:]  # [Tf, H, W, C]

        # ---- agent_pos (normalized proprioception) [Th, Dstate] ----------
        agent_pos = item[self.state_key].float()  # [Th, Dstate]
        if agent_pos.dim() == 1:
            agent_pos = agent_pos.unsqueeze(0)
        ap_mask = (~torch.isnan(agent_pos)).float()
        agent_pos = torch.nan_to_num(agent_pos, nan=0.0)
        agent_pos = _normalize(
            agent_pos,
            self.norm_stats["state"]["min"],
            self.norm_stats["state"]["delta"],
        )
        agent_pos, _ = self._tail_pad(agent_pos, ap_mask, self.agent_pos_total_dim)
        obs["agent_pos"] = agent_pos

        # ---- action [Hh + H, action_dim]: past history | target chunk ----
        action = item[self.action_key].float()  # [Hh+H, Da]
        if action.dim() == 1:
            action = action.unsqueeze(0)
        dof_mask = (~torch.isnan(action)).float()
        action = torch.nan_to_num(action, nan=0.0)
        action = _normalize(
            action, self.norm_stats["action"]["min"], self.norm_stats["action"]["delta"]
        )
        action, dof_mask = self._tail_pad(action, dof_mask, self.dof_total_dim)

        Hh = self.action_history_length
        sample = {
            "obs": obs,
            "next_obs": next_obs,
            "action": action[Hh:],  # [H, Da] prediction target
            "dof_mask": dof_mask[Hh:],  # [H, Da]
            "camera_names": list(self.model_view_names),  # List[str]
            "instruction": item.get("task", ""),  # str
            "dataset_name": self.dataset_name,  # str
        }
        if Hh > 0:
            # PAST actions only (a_{t-Hh}..a_{t-1}); the current chunk's own
            # actions must never appear here (label leakage, see __init__).
            sample["action_history"] = action[:Hh]  # [Hh, Da]
        return sample

    # ------------------------------------------------------------------ #
    @staticmethod
    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Default-style collate producing batch-first tensors + lists.

        - obs/next_obs view slots -> [B, T, H, W, C]
        - obs["agent_pos"]        -> [B, Th, Dstate]
        - action/action_history/dof_mask -> [B, *, *]
        - camera_names -> List[List[str]] (len B)
        - instruction / dataset_name -> List[str] (len B)
        """
        out: Dict[str, Any] = {}
        view_keys = [k for k in batch[0]["obs"].keys() if k != "agent_pos"]

        out["obs"] = {
            vk: torch.stack([s["obs"][vk] for s in batch], dim=0) for vk in view_keys
        }
        out["obs"]["agent_pos"] = torch.stack(
            [s["obs"]["agent_pos"] for s in batch], dim=0
        )
        out["next_obs"] = {
            vk: torch.stack([s["next_obs"][vk] for s in batch], dim=0)
            for vk in view_keys
        }
        out["action"] = torch.stack([s["action"] for s in batch], dim=0)
        if "action_history" in batch[0]:
            out["action_history"] = torch.stack(
                [s["action_history"] for s in batch], dim=0
            )
        out["dof_mask"] = torch.stack([s["dof_mask"] for s in batch], dim=0)
        out["camera_names"] = [list(s["camera_names"]) for s in batch]
        out["instruction"] = [s["instruction"] for s in batch]
        out["dataset_name"] = [s["dataset_name"] for s in batch]
        return out

    # ------------------------------------------------------------------ #
    def get_train_dataloader(self) -> DataLoader:
        if self._world_size() > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                seed=self.seed,
                drop_last=True,
            )
        else:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            sampler = torch.utils.data.RandomSampler(self, generator=generator)
        return DataLoader(
            self,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def get_val_dataloader(self):
        """No held-out val split for the lerobot path.

        The trainer calls this once per epoch (base_trainer.fit -> val_loop);
        returning None makes val_loop skip ("No val split configured"). The
        VGA openloop/video-gen validation is x2robot-episode oriented and isn't
        wired for lerobot, so per-epoch validation is intentionally disabled here.
        """
        return None


def make_lerobot_vga_dataset(config: Dict[str, Any]) -> LerobotVGADataset:
    """Factory mirroring the x2robot dynamic-dataset construction style."""
    return LerobotVGADataset(config)
