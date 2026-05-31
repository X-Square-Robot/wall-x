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
      --model-config.train-config-path /path/to/my_config.yml \\
      --model-config.action-horizon 10 \\
      --model-config.robot-type desktop

Example::

    python scripts/draw_openloop_plot.py \\
      --uri ws://127.0.0.1:32194 \\
      --dataset-root /path/to/libero_all \\
      --train-config workspace/example/fintune_bus2602/libero2_6d.yml \\
      --episode-indices 0,1 \\
      --save-dir ./openloop_lerobot_plots
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import msgpack
import msgpack_numpy as m
import numpy as np
import websockets
import yaml

m.patch()

logger = logging.getLogger(__name__)

_DEFAULT_CAM_WS_KEYS = {
    "observation.images.faceImg": "face_view",
    "observation.images.rightImg": "right_wrist_view",
    "observation.images.leftImg": "left_wrist_view",
    "observation.images.move1Img": "move1_view",
}


@dataclass
class EpisodeArrays:
    """Preloaded episode tensors for fast open-loop iteration."""

    episode_index: int
    instruction: str
    states: np.ndarray  # (T, D_state)
    actions: np.ndarray  # (T, D_action)
    images: dict[str, np.ndarray]  # cam_key -> (T, H, W, 3) uint8 RGB


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_train_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
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


def build_state_payload(
    state_vec: np.ndarray,
    train_config: dict[str, Any],
) -> dict[str, list[float]]:
    """Build websocket ``state`` dict from a flat proprio vector."""
    state_vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)
    agent_cfg = _agent_pos_config(train_config)

    if agent_cfg:
        real_dim = real_vector_dim(agent_cfg)
        if real_dim > 0 and state_vec.shape[0] >= real_dim:
            follow2 = slice_by_config(
                state_vec[: max(state_vec.shape[0], real_dim)], agent_cfg
            )
            if follow2.shape[0] == 0:
                follow2 = state_vec[:real_dim]
            return {"follow2_pos": follow2.tolist()}

    if state_vec.shape[0] >= 14:
        return {
            "follow1_pos": state_vec[:7].tolist(),
            "follow2_pos": state_vec[7:14].tolist(),
        }
    if state_vec.shape[0] >= 7:
        return {"follow2_pos": state_vec[:7].tolist()}
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


def align_pred_to_gt(
    pred_chunk: np.ndarray,
    gt_chunk: np.ndarray,
    train_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Align predicted and GT action chunks to the same dimensionality."""
    pred = np.asarray(pred_chunk, dtype=np.float32)
    gt = np.asarray(gt_chunk, dtype=np.float32)
    if pred.ndim == 1:
        pred = pred[np.newaxis, :]
    if gt.ndim == 1:
        gt = gt[np.newaxis, :]

    dof_cfg = _dof_config(train_config)
    model_real_dim = real_vector_dim(dof_cfg) if dof_cfg else pred.shape[1]
    # Only strip padding columns; never flatten a [T, D] chunk (that collapses T -> 1).
    if dof_cfg and pred.shape[1] > model_real_dim:
        pred = strip_padding_columns(pred, dof_cfg)

    if pred.shape[1] != gt.shape[1]:
        n = min(pred.shape[1], gt.shape[1])
        pred = pred[:, :n]
        gt = gt[:, :n]
    return pred, gt


def extract_action_chunk_from_response(
    result: dict[str, Any],
    train_config: dict[str, Any],
) -> np.ndarray:
    if "predict_action" in result:
        return decode_predict_action(result["predict_action"], train_config)

    if "follow2_pos" in result:
        right = np.asarray(result["follow2_pos"], dtype=np.float32)
        if "follow1_pos" in result:
            left = np.asarray(result["follow1_pos"], dtype=np.float32)
            if left.shape == right.shape:
                return np.concatenate([left, right], axis=1)
        return right

    if "action" in result:
        action = np.asarray(result["action"], dtype=np.float32)
        if action.ndim == 1:
            action = action[np.newaxis, :]
        return action

    raise KeyError(
        "Response has no action fields. For single-arm LIBERO, restart the server with "
        "`--no-serialize-actions` so responses include `predict_action`."
    )


def load_episode_arrays(
    dataset_root: str | Path,
    episode_index: int,
    state_key: str,
    action_key: str,
    camera_keys: list[str],
) -> EpisodeArrays:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        str(dataset_root),
        root=None,
        episodes=[episode_index],
        video_backend="pyav",
    )
    if len(ds) == 0:
        raise ValueError(f"Episode {episode_index} is empty under {dataset_root}")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    images: dict[str, list[np.ndarray]] = {k: [] for k in camera_keys}
    instruction = ""

    for i in range(len(ds)):
        item = ds[i]
        states.append(np.asarray(item[state_key], dtype=np.float32).reshape(-1))
        actions.append(np.asarray(item[action_key], dtype=np.float32).reshape(-1))
        if not instruction:
            instruction = str(item.get("task", ""))
        for cam_key in camera_keys:
            if cam_key in item:
                images[cam_key].append(tensor_to_rgb_uint8(item[cam_key]))

    stacked_images = {k: np.stack(v, axis=0) for k, v in images.items() if v}
    return EpisodeArrays(
        episode_index=episode_index,
        instruction=instruction,
        states=np.stack(states, axis=0),
        actions=np.stack(actions, axis=0),
        images=stacked_images,
    )


def build_obs_payload(
    episode: EpisodeArrays,
    frame_idx: int,
    train_config: dict[str, Any],
    cam_ws_mapping: dict[str, str],
    extra_view_keys: list[str],
) -> dict[str, Any]:
    state = episode.states[frame_idx]
    views: dict[str, str] = {}

    ref_shape: tuple[int, int, int] | None = None
    for cam_key, ws_key in cam_ws_mapping.items():
        if cam_key not in episode.images:
            continue
        rgb = episode.images[cam_key][frame_idx]
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
    if dim == 7:
        names = [
            "pos_x",
            "pos_y",
            "pos_z",
            "rot_x",
            "rot_y",
            "rot_z",
            "gripper",
        ]
        return names[i] if i < len(names) else f"dim_{i}"
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

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(str(dataset_root), root=None)
    if episode_indices is None:
        episode_indices = [0]
    episode_indices = [i for i in episode_indices if 0 <= i < meta.total_episodes]
    if not episode_indices:
        raise ValueError("No valid episode indices to evaluate.")

    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    async with websockets.connect(
        uri,
        ping_interval=None,
        ping_timeout=None,
        max_size=None,
    ) as websocket:
        metadata = msgpack.unpackb(await websocket.recv())
        logger.info("Connected to %s, server metadata: %s", uri, metadata)

        for ep_idx in episode_indices:
            logger.info("Loading episode %d ...", ep_idx)
            episode = load_episode_arrays(
                dataset_root,
                ep_idx,
                state_key=state_key,
                action_key=action_key,
                camera_keys=camera_keys,
            )
            num_steps = episode.states.shape[0]
            start_idx = int(start_ratio * num_steps)

            gt_full = episode.actions.copy()
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
                record_n = min(
                    step_stride, action_horizon, num_steps - idx, pred_chunk.shape[0]
                )
                if record_n <= 0:
                    break

                gt_chunk = episode.actions[idx : idx + record_n]
                pred_chunk, gt_chunk = align_pred_to_gt(
                    pred_chunk[:record_n], gt_chunk, train_config
                )
                record_n = min(record_n, pred_chunk.shape[0], gt_chunk.shape[0])
                if record_n <= 0:
                    break

                row0 = len(aligned_gt)
                obs_infer_points.append((row0, idx))

                pred_full[idx : idx + record_n] = pred_chunk[:record_n]
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

            # pred_diff = compute_second_diff(pred_ml)
            # gt_diff = compute_second_diff(gt_ml)
            # mld = min(len(pred_diff), len(gt_diff))
            # if mld >= 3:
            #     plot_openloop(
            #         pred_diff[:mld],
            #         gt_diff[:mld],
            #         save_root / f"{ep_tag}_diff",
            #         title=f"{title} (2nd diff)",
            #     )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open-loop websocket inference on LeRobot datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri", default="ws://127.0.0.1:32194")
    parser.add_argument(
        "--dataset-root",
        default="/path/to/libero_all",
        help="Local LeRobot dataset root.",
    )
    parser.add_argument(
        "--train-config",
        default="/path/to/my_config.yml",
        help="Training YAML for key mappings / dof layout.",
    )
    parser.add_argument(
        "--save-dir",
        default="/path/to/openloop_plots",
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
        help="Cap the number of inference requests per episode.",
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

    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

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
            )
        )
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
