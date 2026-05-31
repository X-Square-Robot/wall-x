#!/usr/bin/env python3
"""Compute LeRobot normalization stats (mean, std, q01, q99) for training.

Writes JSON in the format expected by wall-x training configs::

    {"norm_stats": {
        "observation.state": {"mean": [...], "std": [...], "q01": [...], "q99": [...]},
        "action": {"mean": [...], "std": [...], "q01": [...], "q99": [...]}
    }}

When ``--train_config`` is provided, the script reads ``data.lerobot_config.repo_id``,
``norm_stats_path``, ``task.dof_config``, ``task.agent_pos_config``, and
``task.action_horizon`` from the YAML. Per-DOF slices are aggregated separately;
keys ending with ``_relative`` use the same relative-pose logic as the LeRobot loader.
If the config specifies ``rotation_6D`` (6 dims) but the dataset still stores 3D Euler,
rotation slices are converted to 6D before computing stats (same as ``loader.py``).

Usage
-----
Recommended: pass a finetune YAML (paths in the config can be placeholders; override
with CLI flags if needed)::

    python scripts/compute_norm_stats.py \\
        --train_config workspace/example/fintune_bus2602/libero.yml

CVPR / multi-task example::

    python scripts/compute_norm_stats.py \\
        --train_config workspace/example/fintune_bus2602/cvpr_example.yml

Override dataset or output path from the command line::

    python scripts/compute_norm_stats.py \\
        --train_config workspace/example/fintune_bus2602/libero.yml \\
        --data_root /path/to/repo_id \\
        --output_path /path/to/norm_stats_path

Without a train config (global stats only, no per-DOF relative slices)::

    python scripts/compute_norm_stats.py \\
        --data_root /path/to/lerobot_dataset \\
        --output_path /path/to/norm_stats.json

Requirements
------------
- Local LeRobot v3 dataset at ``--data_root`` (or ``data.lerobot_config.repo_id``)
- ``lerobot>=0.3``, ``numpy``, ``pyyaml``, ``tqdm``
- Run from the wall-x repo root (or set ``PYTHONPATH`` so ``wall_x`` imports resolve)

After running, set ``norm_stats_path`` in your training YAML to the generated JSON.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from numba import jit, prange
from tqdm import tqdm

from wall_x._vendor.x2robot_utils.geometry import (
    canonicalize_euler_zyx_batch_nb,
    euler_to_matrix_zyx_batch_nb,
    matrix_to_euler_zyx_batch_nb,
    so3_to_matrix_batch_nb,
)
from wall_x.data.backends.lerobot.rotation_layout import (
    euler_layout_dim,
    layout_uses_6d_rotation,
    maybe_convert_euler_to_6d,
)

SKIP_DOF_KEYS = frozenset(
    {"velocity_decomposed", "height", "head_actions", "action_padding"}
)


@dataclass
class TrainConfigContext:
    data_root: Path
    output_path: Path
    state_key: str
    action_key: str
    propri_ranges: dict[str, list[int]]
    action_ranges: dict[str, list[int]]
    action_chunk: int
    dof_config: dict[str, int]
    agent_pos_config: dict[str, int]
    convert_action_euler_to_6d: bool
    convert_state_euler_to_6d: bool


def write_norm_stats(path: Path, norm_stats: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"norm_stats": norm_stats}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def compute_vector_stats(values: np.ndarray) -> dict[str, list[float]]:
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _apply_slice_stats(
    full_stats: dict[str, list[float]],
    index_range: list[int],
    slice_stats: dict[str, list[float]],
) -> None:
    start, end = index_range
    for field in ("mean", "std", "q01", "q99"):
        full_stats[field][start:end] = slice_stats[field]


def config_to_index_ranges(config: dict[str, int]) -> dict[str, list[int]]:
    ranges: dict[str, list[int]] = {}
    cur = 0
    for key, dim in config.items():
        if key in SKIP_DOF_KEYS:
            continue
        ranges[key] = [cur, cur + int(dim)]
        cur += int(dim)
    return ranges


def load_train_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    if not isinstance(config, dict):
        raise ValueError(f"train config must be a YAML mapping, got {type(config)}")
    return config


def parse_train_config(config: dict[str, Any]) -> TrainConfigContext:
    task = config.get("task")
    if not isinstance(task, dict):
        raise ValueError("train config must contain a 'task' section")

    dof_config = task.get("dof_config")
    agent_pos_config = task.get("agent_pos_config")
    if not isinstance(dof_config, dict) or not dof_config:
        raise ValueError("task.dof_config is required in train config")
    if not isinstance(agent_pos_config, dict) or not agent_pos_config:
        raise ValueError("task.agent_pos_config is required in train config")

    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("train config must contain a 'data' section")

    lerobot_config = data_cfg.get("lerobot_config")
    if not isinstance(lerobot_config, dict):
        raise ValueError("data.lerobot_config is required in train config")

    repo_id = lerobot_config.get("repo_id")
    if not repo_id:
        raise ValueError("data.lerobot_config.repo_id is required in train config")

    norm_stats_path = config.get("norm_stats_path") or data_cfg.get("norm_stats_path")
    if not norm_stats_path:
        raise ValueError(
            "norm_stats_path is required in train config " "(top-level or under data)"
        )

    key_mappings = data_cfg.get("key_mappings")
    if not isinstance(key_mappings, dict):
        raise ValueError("data.key_mappings is required in train config")

    state_key = key_mappings.get("state", "observation.state")
    action_key = key_mappings.get("action", "action")

    action_chunk = int(
        task.get("action_horizon")
        or task.get("action_horizon_flow")
        or data_cfg.get("action_horizon")
        or 32
    )

    return TrainConfigContext(
        data_root=Path(repo_id),
        output_path=Path(norm_stats_path),
        state_key=state_key,
        action_key=action_key,
        propri_ranges=config_to_index_ranges(agent_pos_config),
        action_ranges=config_to_index_ranges(dof_config),
        action_chunk=action_chunk,
        dof_config={k: int(v) for k, v in dof_config.items()},
        agent_pos_config={k: int(v) for k, v in agent_pos_config.items()},
        convert_action_euler_to_6d=layout_uses_6d_rotation(dof_config),
        convert_state_euler_to_6d=layout_uses_6d_rotation(agent_pos_config),
    )


def _load_state_action_table(
    data_root: Path,
    state_key: str,
    action_key: str,
):
    dataset = LeRobotDataset(str(data_root), root=None, video_backend="pyav")
    non_image_columns = [
        col for col in dataset.features if "image" not in col and col not in {"task"}
    ]
    if state_key not in non_image_columns or action_key not in non_image_columns:
        raise ValueError(
            f"Expected keys {state_key!r} and {action_key!r} in dataset columns, "
            f"got {non_image_columns}"
        )
    table = dataset.hf_dataset.select_columns([state_key, action_key])
    return table, state_key, action_key


def _table_to_arrays(
    table,
    state_key: str,
    action_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load full columns once; avoids O(N*chunk) random row access."""
    logging.info("Loading state/action columns into memory...")
    try:
        states = np.asarray(table[state_key], dtype=np.float32)
        actions = np.asarray(table[action_key], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        logging.warning(
            "Column-wise load failed (%s); falling back to per-row stack.", exc
        )
        states = np.stack(
            [
                np.asarray(table[i][state_key], dtype=np.float32)
                for i in range(len(table))
            ]
        )
        actions = np.stack(
            [
                np.asarray(table[i][action_key], dtype=np.float32)
                for i in range(len(table))
            ]
        )
    if states.ndim == 1:
        states = states.reshape(-1, 1)
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    logging.info(
        "  frames=%d state_dim=%d action_dim=%d",
        len(states),
        states.shape[1],
        actions.shape[1],
    )
    return states, actions


def _collect_relative_cartesian(
    actions: np.ndarray,
    states: np.ndarray,
    index_range: list[int],
    action_chunk: int,
) -> np.ndarray:
    start, end = index_range
    max_start = max(0, len(actions) - action_chunk)
    chunks = []
    anchor_states = states[:max_start, start:end]
    for offset in range(action_chunk):
        chunks.append(actions[offset : offset + max_start, start:end] - anchor_states)
    return np.concatenate(chunks, axis=0)


def _compute_delta_from_state_and_abs_rot(
    rotations: np.ndarray, state: np.ndarray
) -> np.ndarray:
    """Relative rotation: R_rel = R_abs @ R_state^T (same convention as the loader)."""
    if rotations.shape[-1] == 3:
        rotations_matrix = euler_to_matrix_zyx_batch_nb(rotations)
        out_is_euler = True
    elif rotations.shape[-1] == 6:
        rotations_matrix = so3_to_matrix_batch_nb(rotations)
        out_is_euler = False
    else:
        raise ValueError(
            f"Only 3D euler or 6D rotation supported, got {rotations.shape[-1]}D"
        )

    if state.shape[-1] == 3:
        state_matrix = euler_to_matrix_zyx_batch_nb(state[np.newaxis, :])[0]
    elif state.shape[-1] == 6:
        state_matrix = so3_to_matrix_batch_nb(state[np.newaxis, :])[0]
    else:
        raise ValueError(
            f"Only 3D euler or 6D rotation supported, got {state.shape[-1]}D"
        )

    return _abs_rot_to_delta(rotations_matrix, state_matrix, out_is_euler)


@jit(nopython=True, parallel=True)
def _abs_rot_to_delta(
    rotations_matrix: np.ndarray,
    state_matrix: np.ndarray,
    out_is_euler: bool,
) -> np.ndarray:
    st = np.empty((3, 3), dtype=np.float64)
    st[0, 0] = state_matrix[0, 0]
    st[0, 1] = state_matrix[1, 0]
    st[0, 2] = state_matrix[2, 0]
    st[1, 0] = state_matrix[0, 1]
    st[1, 1] = state_matrix[1, 1]
    st[1, 2] = state_matrix[2, 1]
    st[2, 0] = state_matrix[0, 2]
    st[2, 1] = state_matrix[1, 2]
    st[2, 2] = state_matrix[2, 2]

    n = rotations_matrix.shape[0]
    r_rel = np.empty((n, 3, 3), dtype=np.float64)
    for i in prange(n):
        a00 = rotations_matrix[i, 0, 0]
        a01 = rotations_matrix[i, 0, 1]
        a02 = rotations_matrix[i, 0, 2]
        a10 = rotations_matrix[i, 1, 0]
        a11 = rotations_matrix[i, 1, 1]
        a12 = rotations_matrix[i, 1, 2]
        a20 = rotations_matrix[i, 2, 0]
        a21 = rotations_matrix[i, 2, 1]
        a22 = rotations_matrix[i, 2, 2]

        r_rel[i, 0, 0] = a00 * st[0, 0] + a01 * st[1, 0] + a02 * st[2, 0]
        r_rel[i, 0, 1] = a00 * st[0, 1] + a01 * st[1, 1] + a02 * st[2, 1]
        r_rel[i, 0, 2] = a00 * st[0, 2] + a01 * st[1, 2] + a02 * st[2, 2]
        r_rel[i, 1, 0] = a10 * st[0, 0] + a11 * st[1, 0] + a12 * st[2, 0]
        r_rel[i, 1, 1] = a10 * st[0, 1] + a11 * st[1, 1] + a12 * st[2, 1]
        r_rel[i, 1, 2] = a10 * st[0, 2] + a11 * st[1, 2] + a12 * st[2, 2]
        r_rel[i, 2, 0] = a20 * st[0, 0] + a21 * st[1, 0] + a22 * st[2, 0]
        r_rel[i, 2, 1] = a20 * st[0, 1] + a21 * st[1, 1] + a22 * st[2, 1]
        r_rel[i, 2, 2] = a20 * st[0, 2] + a21 * st[1, 2] + a22 * st[2, 2]

    if out_is_euler:
        d_euler = matrix_to_euler_zyx_batch_nb(r_rel)
        return canonicalize_euler_zyx_batch_nb(d_euler)

    out6 = np.empty((n, 6), dtype=np.float64)
    for i in prange(n):
        out6[i, 0] = r_rel[i, 0, 0]
        out6[i, 1] = r_rel[i, 0, 1]
        out6[i, 2] = r_rel[i, 0, 2]
        out6[i, 3] = r_rel[i, 1, 0]
        out6[i, 4] = r_rel[i, 1, 1]
        out6[i, 5] = r_rel[i, 1, 2]
    return out6


def _collect_relative_rotation(
    actions: np.ndarray,
    states: np.ndarray,
    index_range: list[int],
    action_chunk: int,
) -> np.ndarray:
    start, end = index_range
    max_start = max(0, len(actions) - action_chunk)
    chunks = []
    for anchor_idx in tqdm(
        range(max_start),
        desc="  relative rotation anchors",
        leave=False,
    ):
        action_clip = actions[anchor_idx : anchor_idx + action_chunk, start:end]
        proprio_clip = states[anchor_idx, start:end]
        rel = _compute_delta_from_state_and_abs_rot(
            action_clip.astype(np.float64), proprio_clip.astype(np.float64)
        ).astype(np.float32)
        chunks.append(rel)
    return np.concatenate(chunks, axis=0)


def collect_dof_vectors_from_arrays(
    states: np.ndarray,
    actions: np.ndarray,
    propri_ranges: dict[str, list[int]],
    action_ranges: dict[str, list[int]],
    action_chunk: int = 32,
) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}

    absolute_action_keys = {
        key: index_range
        for key, index_range in action_ranges.items()
        if not key.endswith("_relative")
    }
    relative_action_keys = {
        key: index_range
        for key, index_range in action_ranges.items()
        if key.endswith("_relative")
    }

    for sub_key, index_range in propri_ranges.items():
        start, end = index_range
        vectors[sub_key] = states[:, start:end]

    for sub_key, index_range in absolute_action_keys.items():
        start, end = index_range
        vectors[sub_key] = actions[:, start:end]

    if relative_action_keys:
        logging.info(
            "Computing relative action slices (chunk=%d, anchors=%d)...",
            action_chunk,
            max(0, len(actions) - action_chunk),
        )
        for sub_key, index_range in tqdm(
            relative_action_keys.items(), desc="Relative action keys"
        ):
            if "rotation" in sub_key:
                vectors[sub_key] = _collect_relative_rotation(
                    actions, states, index_range, action_chunk
                )
            else:
                vectors[sub_key] = _collect_relative_cartesian(
                    actions, states, index_range, action_chunk
                )

    return vectors


def _apply_euler_to_6d_if_needed(
    states: np.ndarray,
    actions: np.ndarray,
    dof_config: dict[str, int],
    agent_pos_config: dict[str, int],
    convert_action: bool,
    convert_state: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if convert_state:
        states = maybe_convert_euler_to_6d(states, agent_pos_config, True)
    if convert_action:
        actions = maybe_convert_euler_to_6d(actions, dof_config, True)
    if convert_state or convert_action:
        logging.info(
            "Euler→6D rotation applied for norm stats "
            "(action=%s, state=%s; action dim %s→%s, state dim %s→%s)",
            convert_action,
            convert_state,
            euler_layout_dim(dof_config) if convert_action else "-",
            actions.shape[1] if convert_action else "-",
            euler_layout_dim(agent_pos_config) if convert_state else "-",
            states.shape[1] if convert_state else "-",
        )
    return states, actions


def compute_norm_stats_with_dof_config(
    data_root: Path,
    output_path: Path,
    propri_ranges: dict[str, list[int]],
    action_ranges: dict[str, list[int]],
    state_key: str = "observation.state",
    action_key: str = "action",
    action_chunk: int = 32,
    dof_config: dict[str, int] | None = None,
    agent_pos_config: dict[str, int] | None = None,
    convert_action_euler_to_6d: bool = False,
    convert_state_euler_to_6d: bool = False,
) -> dict[str, dict]:
    table, state_key, action_key = _load_state_action_table(
        data_root, state_key, action_key
    )
    states, actions = _table_to_arrays(table, state_key, action_key)
    if dof_config and agent_pos_config:
        states, actions = _apply_euler_to_6d_if_needed(
            states,
            actions,
            dof_config,
            agent_pos_config,
            convert_action_euler_to_6d,
            convert_state_euler_to_6d,
        )
    norm_stats = {
        state_key: compute_vector_stats(states),
        action_key: compute_vector_stats(actions),
    }

    vectors = collect_dof_vectors_from_arrays(
        states=states,
        actions=actions,
        propri_ranges=propri_ranges,
        action_ranges=action_ranges,
        action_chunk=action_chunk,
    )

    for sub_key, index_range in propri_ranges.items():
        if sub_key not in vectors:
            logging.warning("No samples collected for propri key %s, skipping", sub_key)
            continue
        slice_stats = compute_vector_stats(vectors[sub_key])
        _apply_slice_stats(norm_stats[state_key], index_range, slice_stats)
        logging.info("  %s (agent_pos): dim=%d", sub_key, len(slice_stats["mean"]))

    for sub_key, index_range in action_ranges.items():
        if sub_key not in vectors:
            logging.warning("No samples collected for action key %s, skipping", sub_key)
            continue
        slice_stats = compute_vector_stats(vectors[sub_key])
        _apply_slice_stats(norm_stats[action_key], index_range, slice_stats)
        mode = "relative" if sub_key.endswith("_relative") else "absolute"
        logging.info("  %s (dof, %s): dim=%d", sub_key, mode, len(slice_stats["mean"]))

    write_norm_stats(output_path, norm_stats)
    return norm_stats


def load_vectors(
    data_root: Path,
    state_key: str,
    action_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    table, state_key, action_key = _load_state_action_table(
        data_root, state_key, action_key
    )
    return _table_to_arrays(table, state_key, action_key)


def compute_norm_stats(
    data_root: Path,
    output_path: Path,
    state_key: str = "observation.state",
    action_key: str = "action",
    train_ctx: TrainConfigContext | None = None,
) -> dict[str, dict]:
    if train_ctx is None:
        states, actions = load_vectors(data_root, state_key, action_key)
        norm_stats = {
            state_key: compute_vector_stats(states),
            action_key: compute_vector_stats(actions),
        }
        write_norm_stats(output_path, norm_stats)
        return norm_stats

    return compute_norm_stats_with_dof_config(
        data_root=data_root,
        output_path=output_path,
        propri_ranges=train_ctx.propri_ranges,
        action_ranges=train_ctx.action_ranges,
        state_key=state_key,
        action_key=action_key,
        action_chunk=train_ctx.action_chunk,
        dof_config=train_ctx.dof_config,
        agent_pos_config=train_ctx.agent_pos_config,
        convert_action_euler_to_6d=train_ctx.convert_action_euler_to_6d,
        convert_state_euler_to_6d=train_ctx.convert_state_euler_to_6d,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute norm stats for a local LeRobot v3 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --train_config workspace/example/fintune_bus2602/libero.yml
  %(prog)s --train_config workspace/example/fintune_bus2602/cvpr_example.yml \\
      --data_root /path/to/repo_id --output_path /path/to/norm_stats_path
  %(prog)s --data_root /path/to/lerobot_dataset --output_path /path/to/norm_stats.json
""",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default=None,
        help=(
            "Training YAML config (e.g. cvpr_example.yml). When set, reads "
            "data.lerobot_config.repo_id, data.norm_stats_path, task.dof_config, "
            "task.agent_pos_config and task.action_horizon. Action keys ending "
            "with '_relative' use the same relative-pose logic as lerobot loader."
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Local LeRobot dataset directory (overrides train config)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output json path (overrides train config)",
    )
    parser.add_argument(
        "--state_key",
        type=str,
        default=None,
        help="Dataset column for proprioception (overrides train config)",
    )
    parser.add_argument(
        "--action_key",
        type=str,
        default=None,
        help="Dataset column for action (overrides train config)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    train_ctx: TrainConfigContext | None = None
    if args.train_config:
        config_path = Path(args.train_config)
        if not config_path.exists():
            raise FileNotFoundError(f"train config not found: {config_path}")
        train_ctx = parse_train_config(load_train_config(config_path))
        logging.info("train_config: %s", config_path)
        logging.info("  dof_config keys: %s", list(train_ctx.action_ranges))
        logging.info("  agent_pos_config keys: %s", list(train_ctx.propri_ranges))
        logging.info("  action_chunk: %d", train_ctx.action_chunk)
        if train_ctx.convert_action_euler_to_6d or train_ctx.convert_state_euler_to_6d:
            logging.info(
                "  euler→6d: action=%s state=%s",
                train_ctx.convert_action_euler_to_6d,
                train_ctx.convert_state_euler_to_6d,
            )

    data_root = Path(args.data_root or (train_ctx.data_root if train_ctx else ""))

    output_path = Path(args.output_path or (train_ctx.output_path if train_ctx else ""))
    state_key = args.state_key or (
        train_ctx.state_key if train_ctx else "observation.state"
    )
    action_key = args.action_key or (train_ctx.action_key if train_ctx else "action")

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset not found: {data_root}")

    logging.info("dataset: %s", data_root)
    logging.info("output:  %s", output_path)

    norm_stats = compute_norm_stats(
        data_root=data_root,
        output_path=output_path,
        state_key=state_key,
        action_key=action_key,
        train_ctx=train_ctx,
    )

    for key, stats in norm_stats.items():
        logging.info("  %s: dim=%d", key, len(stats["mean"]))

    logging.info("Saved norm stats to %s", output_path)


if __name__ == "__main__":
    main()
