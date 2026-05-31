#!/usr/bin/env python3
"""Run one Wall-X VLA inference pass through the harrix adapter.

This is a lightweight smoke test for the inference path. It builds a synthetic
LIBERO-style observation, loads the checkpoint through harrix, and prints the
predicted action chunk shape.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _ensure_local_harrix_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    harrix_python = repo_root / "third_party" / "harrix" / "python"
    if harrix_python.is_dir():
        sys.path.insert(0, str(harrix_python))


def _build_fake_observation(seed: int, image_size: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "eef_pos": rng.normal(size=(3,)).astype(np.float32),
        "eef_axisangle": rng.normal(size=(3,)).astype(np.float32),
        "gripper": rng.normal(size=(1,)).astype(np.float32),
        "face_view": rng.integers(0, 256, (image_size, image_size, 3), dtype=np.uint8),
        "wrist_view": rng.integers(0, 256, (image_size, image_size, 3), dtype=np.uint8),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path", required=True, help="Checkpoint directory."
    )
    parser.add_argument(
        "--train-config-path",
        default=None,
        help="Optional training config path. Defaults to config.yml/config.yaml next to the checkpoint.",
    )
    parser.add_argument("--norm-key", default="libero_all")
    parser.add_argument("--architecture", default="qwen2_5")
    parser.add_argument("--action-mode", default="flow")
    parser.add_argument(
        "--cam-names",
        nargs="+",
        default=["face_view", "right_wrist_view"],
        help="Camera names expected by the checkpoint.",
    )
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--instruction", default="pick up the object")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _ensure_local_harrix_on_path()

    from wall_x._vendor.harrix.adapters.registry import build_adapter
    from wall_x._vendor.harrix.eval_config import (
        EvalConfig,
        LiberoEnvParams,
        autofill_from_checkpoint,
    )

    import wall_x._vendor.harrix.adapters  # noqa: F401  register model adapters

    cfg = EvalConfig()
    cfg.model.checkpoint_path = args.checkpoint_path
    cfg.model.train_config_path = args.train_config_path
    cfg.model.norm_key = args.norm_key
    cfg.model.cam_names = list(args.cam_names)
    cfg.model.action_horizon = args.action_horizon
    cfg.model.architecture = args.architecture
    cfg.model.action_mode = args.action_mode
    cfg.env.libero = LiberoEnvParams(num_trials_per_task=1, task_indices=[0])
    cfg = autofill_from_checkpoint(cfg)

    adapter = build_adapter(cfg)
    payload = {
        "observation": _build_fake_observation(args.seed, args.image_size),
        "instruction": args.instruction,
        "noise": None,
    }
    actions = adapter.predict_batch([payload])
    action = np.asarray(actions[0])

    print("Fake inference succeeded.")
    print(f"action shape: {action.shape}")
    print(f"action dtype: {action.dtype}")
    print(f"action min/max: {float(action.min()):.6f} / {float(action.max()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
