#!/usr/bin/env python3
"""Open-loop evaluation over WebSocket using LeRobot-format datasets.

Loads episodes from a local LeRobot v3 dataset, sends observations to a running
Wall-X websocket server, collects predicted action chunks, and plots them against
ground-truth trajectories.

Reference: ``infer_openloop_websocket.py`` (websocket client + open-loop loop).

For single-arm LIBERO checkpoints, start the server with raw model output::

    python -m wall_x._vendor.harrix.serving.launch_serving \\
      --env X2ROBOT --port 32194 \\
      --no-serialize-actions \\
      model-config:server-model-config \\
      --model-config.checkpoint-path /path/to/ckpt \\
      --model-config.train-config-path /path/to/libero.yml \\
      --model-config.action-horizon 10 \\
      --model-config.robot-type desktop
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Open-loop image encoding requires opencv-python-headless. "
            "Install the repository requirements first."
        ) from exc
    return cv2


def require_msgpack():
    try:
        import msgpack
        import msgpack_numpy as m
    except ImportError as exc:
        raise RuntimeError(
            "Open-loop websocket evaluation requires msgpack-numpy. "
            "Install the repository requirements first."
        ) from exc
    m.patch()
    return msgpack


def require_websockets():
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError(
            "Open-loop websocket evaluation requires websockets. "
            "Install the repository requirements first."
        ) from exc
    return websockets


_DEFAULT_CAM_WS_KEYS = {
    "observation.images.faceImg": "face_view",
    "observation.images.rightImg": "right_wrist_view",
    "observation.images.leftImg": "left_wrist_view",
    "observation.images.move1Img": "move1_view",
}


@dataclass
class EpisodeArrays:
    """Episode state/action plus lazily decoded camera frames."""

    episode_index: int
    instruction: str
    states: np.ndarray  # (T, D_state)
    actions: np.ndarray  # (T, D_action)
    camera_keys: list[str]
    frame_offset: int = 0
    num_steps: int = 0
    _dataset: Any = field(default=None, repr=False)
    _image_cache: dict[int, dict[str, np.ndarray]] = field(
        default_factory=dict, repr=False
    )

    def get_frame_images(self, frame_idx: int) -> dict[str, np.ndarray]:
        """Decode camera frames for one observation index (cached)."""
        if frame_idx in self._image_cache:
            return self._image_cache[frame_idx]
        if self._dataset is None:
            raise RuntimeError("Episode image loader is not initialized.")
        if not self._image_cache:
            logger.info(
                "Decoding camera frames on demand (first obs_idx=%d)", frame_idx
            )
        item = self._dataset[frame_idx]
        images = {
            cam_key: tensor_to_rgb_uint8(item[cam_key])
            for cam_key in self.camera_keys
            if cam_key in item
        }
        self._image_cache[frame_idx] = images
        return images


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_source_root_if_needed() -> None:
    repo_root = _repo_root()
    if (repo_root / "wall_x").is_dir() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_train_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader) or {}
    if "data" not in cfg:
        cfg["data"] = {}
    cfg["data"]["model_type"] = cfg.get("model_type")
    return cfg


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _task_block(train_config: dict[str, Any]) -> dict[str, Any]:
    return train_config.get("task") or {}


def _dof_config(train_config: dict[str, Any]) -> dict[str, int]:
    return (
        train_config.get("dof_config")
        or _task_block(train_config).get("dof_config")
        or {}
    )


def _agent_pos_config(train_config: dict[str, Any]) -> dict[str, int]:
    return (
        train_config.get("agent_pos_config")
        or _task_block(train_config).get("agent_pos_config")
        or {}
    )


def real_vector_dim(config_block: dict[str, int]) -> int:
    """Sum dof/agent dims excluding ``action_padding``."""
    return sum(d for k, d in config_block.items() if k != "action_padding" and d > 0)


def slice_by_config(vec: np.ndarray, config_block: dict[str, int]) -> np.ndarray:
    """Keep only non-padding dims from a 1-D vector (state / single action row)."""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if not config_block:
        return vec
    parts: list[np.ndarray] = []
    start = 0
    for key, dim in config_block.items():
        if key == "action_padding":
            start += dim
            continue
        end = start + dim
        if end <= vec.shape[0]:
            parts.append(vec[start:end])
        start = end
    if parts:
        return np.concatenate(parts, axis=0).astype(np.float32)
    return vec


def strip_padding_columns(arr: np.ndarray, config_block: dict[str, int]) -> np.ndarray:
    """Drop ``action_padding`` columns from a [T, D] action chunk."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return slice_by_config(arr, config_block)
    if not config_block:
        return arr
    parts: list[np.ndarray] = []
    start = 0
    for key, dim in config_block.items():
        if key == "action_padding":
            start += dim
            continue
        parts.append(arr[:, start : start + dim])
        start += dim
    if parts:
        return np.concatenate(parts, axis=1).astype(np.float32)
    return arr


def tensor_to_rgb_uint8(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def encode_image_rgb(image: np.ndarray) -> str:
    """JPEG base64; input must be RGB uint8."""
    cv2 = require_cv2()
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buffer).decode("utf-8")


def build_camera_ws_mapping(train_config: dict[str, Any]) -> dict[str, str]:
    key_mappings = (train_config.get("data") or {}).get("key_mappings") or {}
    cam_map = key_mappings.get("camera") or {}
    if cam_map:
        return dict(cam_map)
    return dict(_DEFAULT_CAM_WS_KEYS)


def resolve_feature_keys(
    train_config: dict[str, Any],
    state_key: str | None,
    action_key: str | None,
) -> tuple[str, str]:
    key_mappings = (train_config.get("data") or {}).get("key_mappings") or {}
    sk = state_key or key_mappings.get("state") or "observation.state"
    ak = action_key or key_mappings.get("action") or "action"
    return sk, ak


def _arm_follow_pos_from_agent_cfg(
    state_vec: np.ndarray,
    agent_cfg: dict[str, int],
    arm_prefix: str,
) -> list[float] | None:
    """Pack one arm into websocket ``follow{1,2}_pos`` layout: pos3 + rpy3 + grip1."""
    from wall_x._vendor.x2robot_utils import geometry as geom

    pos = rot = grip = None
    start = 0
    for key, dim in agent_cfg.items():
        if key == "action_padding":
            start += int(dim)
            continue
        end = start + int(dim)
        chunk = (
            state_vec[start:end]
            if end <= state_vec.shape[0]
            else np.zeros(int(dim), dtype=np.float32)
        )
        start = end
        if not key.startswith(arm_prefix):
            continue
        if "cartesian_pos" in key:
            pos = chunk.reshape(-1)
        elif "rotation_6d" in key.lower():
            rot = geom.so3_to_euler_zyx_batch_nb(chunk.reshape(1, -1)).reshape(-1)
        elif "rotation" in key:
            rot = chunk.reshape(-1)
        elif "gripper" in key:
            grip = chunk.reshape(-1)

    if pos is None or rot is None or grip is None:
        return None
    packed = np.concatenate([pos[:3], rot[:3], grip[:1]], axis=0).astype(np.float32)
    if packed.shape[0] != 7:
        return None
    return packed.tolist()


def build_state_payload(
    state_vec: np.ndarray,
    train_config: dict[str, Any],
) -> dict[str, list[float]]:
    """Build websocket ``state`` dict from a flat proprio vector."""
    state_vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)

    # LeRobot dual-arm export in this workflow is already raw follow-pos 14D:
    # [L_pos3, L_rot3, L_grip1, R_pos3, R_rot3, R_grip1].
    # Prefer direct passthrough and avoid parsing with agent_pos_config
    # (which may be 6D/padded and would mis-slice a 14D state).
    if state_vec.shape[0] >= 14:
        return {
            "follow1_pos": state_vec[:7].tolist(),
            "follow2_pos": state_vec[7:14].tolist(),
        }
    if state_vec.shape[0] >= 7:
        return {"follow2_pos": state_vec.tolist()}

    agent_cfg = _agent_pos_config(train_config)

    if agent_cfg:
        has_left = any(k.startswith("follow_left_") for k in agent_cfg)
        has_right = any(k.startswith("follow_right_") for k in agent_cfg)
        payload: dict[str, list[float]] = {}
        if has_left:
            left = _arm_follow_pos_from_agent_cfg(state_vec, agent_cfg, "follow_left_")
            if left is not None:
                payload["follow1_pos"] = left
        if has_right:
            right = _arm_follow_pos_from_agent_cfg(
                state_vec, agent_cfg, "follow_right_"
            )
            if right is not None:
                payload["follow2_pos"] = right
        if payload:
            return payload

        # Legacy single-arm LIBERO path: only follow_right_* in agent_pos_config.
        real_dim = real_vector_dim(agent_cfg)
        if real_dim > 0 and state_vec.shape[0] >= real_dim:
            sliced = slice_by_config(state_vec, agent_cfg)
            if sliced.shape[0] == 0:
                sliced = state_vec[:real_dim]
            right = _arm_follow_pos_from_agent_cfg(
                state_vec, agent_cfg, "follow_right_"
            )
            if right is not None:
                return {"follow2_pos": right}
            return {"follow2_pos": sliced.tolist()}
    raise ValueError(f"Unsupported state dimension: {state_vec.shape[0]}")


def decode_predict_action(
    predict_action: Any,
    train_config: dict[str, Any],
) -> np.ndarray:
    """Convert raw ``predict_action`` [H, D_model] to flat action rows per step."""
    if hasattr(predict_action, "detach"):
        predict_action = predict_action.detach().cpu().numpy()
    pa = np.asarray(predict_action, dtype=np.float32)
    if pa.ndim == 3:
        pa = pa[0]

    dof_cfg = _dof_config(train_config)
    if dof_cfg:
        parts: list[np.ndarray] = []
        start = 0
        for key, dim in dof_cfg.items():
            if key == "action_padding":
                start += dim
                continue
            parts.append(pa[:, start : start + dim])
            start += dim
        if parts:
            return np.concatenate(parts, axis=1).astype(np.float32)

    try:
        from wall_x._vendor.harrix.envs.libero_common import decode_chunk

        chunk = decode_chunk(pa, train_config)
        if chunk.ndim == 2 and chunk.shape[1] > 0:
            return chunk.astype(np.float32)
    except Exception:
        pass

    return pa.astype(np.float32)


def extract_follow_pos_14d_from_response(result: dict[str, Any]) -> np.ndarray:
    """Read serialized ``follow{1,2}_pos`` action rows (skip state row 0) as ``[H, 14]``."""
    if "follow2_pos" not in result or "follow1_pos" not in result:
        raise KeyError("Response missing follow1_pos/follow2_pos for 14D eval.")
    left = np.asarray(result["follow1_pos"], dtype=np.float32)
    right = np.asarray(result["follow2_pos"], dtype=np.float32)
    if left.ndim == 1:
        left = left.reshape(1, -1)
    if right.ndim == 1:
        right = right.reshape(1, -1)
    if left.shape[0] < 2 or right.shape[0] < 2:
        raise ValueError(
            f"follow_pos response must include state+action rows, got "
            f"left={left.shape}, right={right.shape}"
        )
    horizon = min(left.shape[0] - 1, right.shape[0] - 1)
    return np.concatenate([left[1 : 1 + horizon], right[1 : 1 + horizon]], axis=1)


def extract_action_chunk_from_response(
    result: dict[str, Any],
    train_config: dict[str, Any],
) -> np.ndarray:
    if "predict_action" in result:
        return decode_predict_action(result["predict_action"], train_config)

    if "follow2_pos" in result and "follow1_pos" in result:
        return extract_follow_pos_14d_from_response(result)

    if "follow2_pos" in result:
        right = np.asarray(result["follow2_pos"], dtype=np.float32)
        if right.ndim == 1:
            right = right.reshape(1, -1)
        return right[1:] if right.shape[0] > 1 else right

    if "action" in result:
        action = np.asarray(result["action"], dtype=np.float32)
        if action.ndim == 1:
            action = action[np.newaxis, :]
        return action

    raise KeyError(
        "Response has no action fields. For single-arm LIBERO, restart the server with "
        "`--no-serialize-actions` so responses include `predict_action`."
    )


def _hf_row_to_numpy(row: Any, key: str) -> np.ndarray:
    value = row[key]
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _episode_instruction(ds: Any, row_index: int = 0) -> str:
    row = ds.hf_dataset[row_index]
    task_idx = row["task_index"]
    if hasattr(task_idx, "item"):
        task_idx = task_idx.item()
    return str(ds.meta.tasks.iloc[task_idx].name)


def resolve_lerobot_dataset_paths(dataset_root: str | Path) -> tuple[str, Path]:
    """Return ``(repo_id, root)`` for a local LeRobot dataset directory."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LeRobot dataset root not found: {root}")
    return root.name, root


def open_local_lerobot_dataset(
    dataset_root: str | Path,
    *,
    episodes: list[int] | None = None,
    video_backend: str = "pyav",
) -> Any:
    """Open a local LeRobot v3 dataset without HuggingFace Hub fallback."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id, root = resolve_lerobot_dataset_paths(dataset_root)
    try:
        return LeRobotDataset(
            repo_id,
            root=root,
            episodes=episodes,
            video_backend=video_backend,
        )
    except TypeError:
        pass

    import torch
    from lerobot.datasets.lerobot_dataset import (
        CODEBASE_VERSION,
        LeRobotDatasetMetadata,
    )
    from lerobot.datasets.utils import (
        get_hf_features_from_features,
        hf_transform_to_torch,
        load_nested_dataset,
    )
    from lerobot.datasets.video_utils import get_safe_default_codec

    meta = LeRobotDatasetMetadata(repo_id, root=root)

    features = get_hf_features_from_features(meta.features)
    hf_dataset = load_nested_dataset(
        root / "data", features=features, episodes=episodes
    )
    hf_dataset.set_transform(hf_transform_to_torch)

    if episodes is not None:
        available = {
            int(ep.item()) if hasattr(ep, "item") else int(ep)
            for ep in hf_dataset.unique("episode_index")
        }
        missing = set(episodes) - available
        if missing:
            raise ValueError(
                f"Episodes {sorted(missing)} not found under {root}. "
                f"Available in loaded parquet: {sorted(available)[:10]}"
                f"{'...' if len(available) > 10 else ''}"
            )

    if meta.video_keys:
        check_eps = (
            episodes if episodes is not None else list(range(meta.total_episodes))
        )
        for ep_idx in check_eps:
            for vid_key in meta.video_keys:
                video_path = root / meta.get_video_file_path(ep_idx, vid_key)
                if not video_path.exists():
                    raise FileNotFoundError(
                        "LeRobot camera videos are missing for open-loop eval.\n"
                        f"  dataset root: {root}\n"
                        f"  first missing file: {video_path}\n"
                        "Parquet state/action may exist, but this script also needs "
                        "mp4 files under videos/."
                    )

    ds = LeRobotDataset.__new__(LeRobotDataset)
    ds.repo_id = repo_id
    ds.root = root
    ds.image_transforms = None
    ds.delta_timestamps = None
    ds.episodes = episodes
    ds.tolerance_s = 1e-4
    ds.revision = CODEBASE_VERSION
    ds.video_backend = video_backend or get_safe_default_codec()
    ds.delta_indices = None
    ds.meta = meta
    ds.hf_dataset = hf_dataset
    ds._lazy_loading = False
    ds._absolute_to_relative_idx = None
    if episodes is not None:
        ds._absolute_to_relative_idx = {
            abs_idx.item() if isinstance(abs_idx, torch.Tensor) else abs_idx: rel_idx
            for rel_idx, abs_idx in enumerate(hf_dataset["index"])
        }
    ds.image_writer = None
    ds.episode_buffer = None
    ds.writer = None
    ds.latest_episode = None
    ds._current_file_start_frame = None
    ds._streaming_encoder = None
    ds.batch_encoding_size = 1
    ds.episodes_since_last_encoding = 0
    return ds


def load_lerobot_metadata(dataset_root: str | Path) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    repo_id, root = resolve_lerobot_dataset_paths(dataset_root)
    return LeRobotDatasetMetadata(repo_id, root=root)


def plan_eval_frame_range(
    num_steps: int,
    start_idx: int,
    step_stride: int,
    action_horizon: int,
    max_inferences: int | None,
) -> tuple[int, int, int]:
    """Return ``(tabular_start, tabular_end, expected_video_decodes)``."""
    tabular_start = max(0, int(start_idx))
    if max_inferences is None:
        return tabular_start, num_steps, -1

    tabular_end = tabular_start
    idx = tabular_start
    infer_count = 0
    video_decodes = 0
    while idx <= num_steps - action_horizon - 1:
        if infer_count >= max_inferences:
            break
        record_n = min(step_stride, action_horizon, num_steps - idx)
        if record_n <= 0:
            break
        tabular_end = max(tabular_end, idx + record_n)
        video_decodes += 1
        idx += record_n
        infer_count += 1
    return tabular_start, min(num_steps, tabular_end), video_decodes


def load_episode_arrays(
    dataset_root: str | Path,
    episode_index: int,
    state_key: str,
    action_key: str,
    camera_keys: list[str],
    *,
    frame_start: int = 0,
    frame_end: int | None = None,
    preload_all_images: bool = False,
) -> EpisodeArrays:
    ds = open_local_lerobot_dataset(
        dataset_root,
        episodes=[episode_index],
        video_backend="pyav",
    )
    num_steps = len(ds.hf_dataset)
    if num_steps == 0:
        raise ValueError(f"Episode {episode_index} is empty under {dataset_root}")

    start = max(0, int(frame_start))
    end = num_steps if frame_end is None else min(num_steps, int(frame_end))
    states = [_hf_row_to_numpy(ds.hf_dataset[i], state_key) for i in range(start, end)]
    actions = [
        _hf_row_to_numpy(ds.hf_dataset[i], action_key) for i in range(start, end)
    ]
    instruction = _episode_instruction(ds, start)

    episode = EpisodeArrays(
        episode_index=episode_index,
        instruction=instruction,
        states=np.stack(states, axis=0),
        actions=np.stack(actions, axis=0),
        camera_keys=list(camera_keys),
        frame_offset=start,
        num_steps=num_steps,
        _dataset=ds,
    )
    if preload_all_images:
        for frame_idx in range(start, end):
            episode.get_frame_images(frame_idx)
    return episode


def build_obs_payload(
    episode: EpisodeArrays,
    frame_idx: int,
    train_config: dict[str, Any],
    cam_ws_mapping: dict[str, str],
    extra_view_keys: list[str],
) -> dict[str, Any]:
    local_idx = frame_idx - episode.frame_offset
    if local_idx < 0 or local_idx >= episode.states.shape[0]:
        raise IndexError(
            f"Frame {frame_idx} is outside loaded tabular range "
            f"[{episode.frame_offset}, {episode.frame_offset + episode.states.shape[0]})."
        )
    state = episode.states[local_idx]
    frame_images = episode.get_frame_images(frame_idx)
    views: dict[str, str] = {}

    ref_shape: tuple[int, int, int] | None = None
    for cam_key, ws_key in cam_ws_mapping.items():
        if cam_key not in frame_images:
            continue
        rgb = frame_images[cam_key]
        ref_shape = rgb.shape
        views[ws_key] = encode_image_rgb(rgb)

    for ws_key in extra_view_keys:
        if ws_key in views:
            continue
        if ref_shape is None:
            ref_shape = (256, 256, 3)
        views[ws_key] = encode_image_rgb(np.zeros(ref_shape, dtype=np.uint8))

    return {
        "state": build_state_payload(state, train_config),
        "views": views,
        "instruction": episode.instruction,
    }


def _dim_label(i: int, dim: int) -> str:
    arm7 = [
        "pos_x",
        "pos_y",
        "pos_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "gripper",
    ]
    if dim == 14:
        prefix = "L_" if i < 7 else "R_"
        return prefix + arm7[i % 7]
    if dim == 7:
        return arm7[i] if i < len(arm7) else f"dim_{i}"
    if dim == 12:
        prefix = "L_" if i < 6 else "R_"
        return f"{prefix}rot6_{i % 6}"
    if dim == 10:
        names = [
            "pos_x",
            "pos_y",
            "pos_z",
            "rot6_0",
            "rot6_1",
            "rot6_2",
            "rot6_3",
            "rot6_4",
            "rot6_5",
            "gripper",
        ]
        return names[i] if i < len(names) else f"dim_{i}"
    return f"dim_{i}"


def follow_pos_14d_to_rotation_6d(rows: list | np.ndarray) -> np.ndarray:
    """Convert ``follow1_pos + follow2_pos`` rows from euler rotation to rot6D only."""
    from wall_x._vendor.x2robot_utils import geometry as geom

    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.shape[1] < 14:
        raise ValueError(f"Expected follow_pos_14d rows, got shape {arr.shape}")

    left_rot6 = geom.euler_to_matrix_zyx_6d_nb(arr[:, 3:6]).astype(np.float32)
    right_rot6 = geom.euler_to_matrix_zyx_6d_nb(arr[:, 10:13]).astype(np.float32)
    return np.concatenate([left_rot6, right_rot6], axis=1).astype(np.float32)


def plot_openloop(
    action_pred_list: list | np.ndarray,
    action_gt_list: list | np.ndarray,
    save_path: Path,
    *,
    title: str = "",
    action_l1: float | None = None,
) -> None:
    """Plot aligned GT vs predicted action trajectories."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_rows = np.asarray(action_gt_list, dtype=np.float32)
    pred_rows = np.asarray(action_pred_list, dtype=np.float32)
    assert (
        gt_rows.shape == pred_rows.shape
    ), f"shape mismatch: pred={pred_rows.shape} gt={gt_rows.shape}"
    n, dim = gt_rows.shape

    fig, axes = plt.subplots(dim, 1, figsize=(12, 3.5 * dim), sharex=True)
    if dim == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(gt_rows[:, i], label="Ground Truth", color="blue", linewidth=1.5)
        ax.plot(pred_rows[:, i], label="Model Output", color="orange", linewidth=1.5)
        ax.set_title(_dim_label(i, dim))
        ax.set_ylabel("Action Value")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time Step")
    axes[0].set_xticks(np.arange(0, n, step=max(1, min(10, max(n // 15, 1)))))
    l1_note = f"Mean L1: {action_l1:.6f}" if action_l1 is not None else ""
    if title and l1_note:
        fig.suptitle(f"{title}\n{l1_note}")
    elif title:
        fig.suptitle(title)
    elif l1_note:
        fig.suptitle(l1_note)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    out = save_path.with_suffix(".jpg") if save_path.suffix != ".jpg" else save_path
    fig.savefig(out, dpi=200)
    plt.close(fig)
    logger.info("Saved plot -> %s", out)


def compute_second_diff(arr: np.ndarray) -> np.ndarray:
    d1 = arr[1:] - arr[:-1]
    return d1[1:] - d1[:-1]


async def run_openloop_eval(
    uri: str,
    dataset_root: str,
    train_config_path: str,
    save_dir: str,
    episode_indices: list[int] | None,
    start_ratio: float,
    stride: int | None,
    max_inferences: int | None,
    state_key: str | None,
    action_key: str | None,
    extra_view_keys: list[str],
    preload_all_images: bool = False,
    plot_rotation_6d: bool = False,
) -> None:
    train_config = load_train_config(train_config_path)
    state_key, action_key = resolve_feature_keys(train_config, state_key, action_key)
    cam_ws_mapping = build_camera_ws_mapping(train_config)
    camera_keys = list(cam_ws_mapping.keys())

    action_horizon = (
        train_config.get("action_horizon_flow")
        or _task_block(train_config).get("action_horizon_flow")
        or _task_block(train_config).get("action_horizon")
        or train_config.get("action_horizon")
        or 10
    )
    step_stride = stride if stride is not None else int(action_horizon)

    meta = load_lerobot_metadata(dataset_root)
    if episode_indices is None:
        episode_indices = [0]
    episode_indices = [i for i in episode_indices if 0 <= i < meta.total_episodes]
    if not episode_indices:
        raise ValueError("No valid episode indices to evaluate.")

    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    msgpack = require_msgpack()
    websockets = require_websockets()

    async with websockets.connect(
        uri,
        ping_interval=None,
        ping_timeout=None,
        max_size=None,
    ) as websocket:
        metadata = msgpack.unpackb(await websocket.recv())
        logger.info("Connected to %s, server metadata: %s", uri, metadata)

        for ep_idx in episode_indices:
            ep_meta = meta.episodes[ep_idx]
            num_steps = int(
                ep_meta.get("length")
                if isinstance(ep_meta, dict)
                else getattr(ep_meta, "length", 0)
            )
            if num_steps <= 0:
                raise ValueError(f"Episode {ep_idx} has invalid length metadata.")

            start_idx = int(start_ratio * num_steps)
            tabular_start, tabular_end, expected_decodes = plan_eval_frame_range(
                num_steps,
                start_idx,
                step_stride,
                int(action_horizon),
                max_inferences,
            )
            logger.info(
                "Loading episode %d tabular frames [%d, %d) / %d "
                "(images on demand%s)...",
                ep_idx,
                tabular_start,
                tabular_end,
                num_steps,
                f", ~{expected_decodes} obs frames" if expected_decodes >= 0 else "",
            )
            episode = load_episode_arrays(
                dataset_root,
                ep_idx,
                state_key=state_key,
                action_key=action_key,
                camera_keys=camera_keys,
                frame_start=tabular_start,
                frame_end=tabular_end,
                preload_all_images=preload_all_images,
            )
            logger.info(
                "Episode %d tabular ready: %d frames loaded, video decode per inference",
                ep_idx,
                episode.states.shape[0],
            )

            gt_full = np.asarray(episode.actions, dtype=np.float32)
            if gt_full.ndim == 1:
                gt_full = gt_full[np.newaxis, :]
            pred_full = np.full_like(gt_full, np.nan)
            aligned_gt: list[list[float]] = []
            aligned_pred: list[list[float]] = []
            obs_infer_points: list[tuple[int, int]] = []

            idx = start_idx
            infer_count = 0
            while idx <= num_steps - action_horizon - 1:
                if max_inferences is not None and infer_count >= max_inferences:
                    break

                payload = build_obs_payload(
                    episode,
                    idx,
                    train_config,
                    cam_ws_mapping,
                    extra_view_keys=extra_view_keys,
                )
                await websocket.send(msgpack.packb(payload, use_bin_type=True))
                raw = await websocket.recv()
                if isinstance(raw, str):
                    hint = ""
                    if "get_serialized_actions" in raw:
                        hint = (
                            "\n\nHint: restart the server with `--no-serialize-actions` "
                            "so responses include `predict_action`."
                        )
                    raise RuntimeError(f"Server error at frame {idx}:\n{raw}{hint}")

                result = msgpack.unpackb(raw, raw=False)
                pred_chunk = extract_action_chunk_from_response(result, train_config)
                if pred_chunk.ndim == 1:
                    pred_chunk = pred_chunk[np.newaxis, :]
                if pred_chunk.shape[1] != gt_full.shape[1]:
                    raise ValueError(
                        f"Prediction action dim {pred_chunk.shape[1]} does not match "
                        f"dataset action dim {gt_full.shape[1]}."
                    )
                record_n = min(
                    step_stride, action_horizon, num_steps - idx, pred_chunk.shape[0]
                )
                if record_n <= 0:
                    break

                local_idx = idx - episode.frame_offset
                gt_chunk = gt_full[local_idx : local_idx + record_n]
                pred_chunk = pred_chunk[:record_n]
                record_n = min(record_n, pred_chunk.shape[0], gt_chunk.shape[0])
                if record_n <= 0:
                    break

                row0 = len(aligned_gt)
                obs_infer_points.append((row0, idx))

                pred_full[local_idx : local_idx + record_n] = pred_chunk[:record_n]
                aligned_gt.extend(gt_chunk[:record_n].tolist())
                aligned_pred.extend(pred_chunk[:record_n].tolist())

                logger.info(
                    "episode=%d obs_idx=%d row0=%d record_n=%d pred_shape=%s gt_dim=%d",
                    ep_idx,
                    idx,
                    row0,
                    record_n,
                    pred_chunk.shape,
                    gt_chunk.shape[1],
                )

                idx += record_n
                infer_count += 1

            if not obs_infer_points:
                logger.warning("Episode %d: no inference steps executed.", ep_idx)
                continue

            ep_tag = f"ep{ep_idx}"
            valid_mask = ~np.isnan(pred_full).any(axis=1)
            action_l1 = float(
                np.mean(np.abs(pred_full[valid_mask] - gt_full[valid_mask]))
            )
            summary = {
                "episode_index": ep_idx,
                "instruction": episode.instruction,
                "num_steps": num_steps,
                "obs_infer_points": obs_infer_points,
                "action_horizon": action_horizon,
                "stride": step_stride,
                "mean_l1": action_l1,
                "state_key": state_key,
                "action_key": action_key,
            }
            summary_path = save_root / f"{ep_tag}_summary.json"
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            logger.info("Episode %d mean L1 = %.6f", ep_idx, action_l1)

            ml = min(len(aligned_gt), len(aligned_pred))
            if ml < 3:
                logger.warning(
                    "Episode %d: too few aligned rows (%d) to plot.", ep_idx, ml
                )
                continue

            gt_ml = np.asarray(aligned_gt[:ml], dtype=np.float32)
            pred_ml = np.asarray(aligned_pred[:ml], dtype=np.float32)
            title = f"Episode {ep_idx}: {episode.instruction[:80]}"
            plot_openloop(
                pred_ml,
                gt_ml,
                save_root / ep_tag,
                title=title,
                action_l1=action_l1,
            )
            if plot_rotation_6d:
                gt_rot6d = follow_pos_14d_to_rotation_6d(gt_ml)
                pred_rot6d = follow_pos_14d_to_rotation_6d(pred_ml)
                rot6d_l1 = float(np.mean(np.abs(pred_rot6d - gt_rot6d)))
                plot_openloop(
                    pred_rot6d,
                    gt_rot6d,
                    save_root / f"{ep_tag}_rot6d",
                    title=f"{title} (rotation 6D)",
                    action_l1=rot6d_l1,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open-loop websocket inference on LeRobot datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri", default="ws://127.0.0.1:32194")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Local LeRobot dataset root.",
    )
    parser.add_argument(
        "--train-config",
        required=True,
        help="Training YAML for key mappings / dof layout.",
    )
    parser.add_argument(
        "--save-dir",
        default="./openloop_lerobot_plots",
    )
    parser.add_argument(
        "--episode-indices",
        default="0",
        help="Comma-separated episode indices, e.g. 0,1,2",
    )
    parser.add_argument(
        "--start-ratio",
        type=float,
        default=0.0,
        help="Start open-loop from this fraction of the episode length.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Frames to advance between inferences (default: action_horizon).",
    )
    parser.add_argument(
        "--max-inferences",
        type=int,
        default=None,
        help=(
            "Cap inference requests per episode; also limits tabular/video loading "
            "to the evaluated frame range."
        ),
    )
    parser.add_argument(
        "--preload-all-images",
        action="store_true",
        help="Decode all camera frames up front (slow; old behavior).",
    )
    parser.add_argument(
        "--plot-rotation-6d",
        action="store_true",
        help="Also save a rotation-6D-only plot as ep*_rot6d.jpg.",
    )
    parser.add_argument(
        "--state-key",
        default=None,
        help="LeRobot state feature key (default: from train config key_mappings).",
    )
    parser.add_argument(
        "--action-key",
        default=None,
        help="LeRobot action feature key (default: from train config key_mappings).",
    )
    parser.add_argument(
        "--extra-view-keys",
        default="left_wrist_view",
        help="Websocket view keys to fill with black images when missing from data.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    _add_source_root_if_needed()

    extra_view_keys = [k.strip() for k in args.extra_view_keys.split(",") if k.strip()]
    episode_indices = parse_int_list(args.episode_indices)

    try:
        asyncio.run(
            run_openloop_eval(
                uri=args.uri,
                dataset_root=args.dataset_root,
                train_config_path=args.train_config,
                save_dir=args.save_dir,
                episode_indices=episode_indices,
                start_ratio=args.start_ratio,
                stride=args.stride,
                max_inferences=args.max_inferences,
                state_key=args.state_key,
                action_key=args.action_key,
                extra_view_keys=extra_view_keys,
                preload_all_images=args.preload_all_images,
                plot_rotation_6d=args.plot_rotation_6d,
            )
        )
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
