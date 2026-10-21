"""Serve a Wall-X flow checkpoint to RoboDojo over the XPolicyLab WebSocket protocol.

Example:
    python -m wall_x.serving.robodojo.serve \\
        --checkpoint-path /path/to/ckpt --norm-key ex_normal --port 43865

Keyframe-memory checkpoints (trained with the ``keyframe_memory_packing``
epilogue) are detected from their config and served with client-held memory;
see ``wall_x.serving.robodojo.memory_protocol`` for the request format.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from wall_x.serving.robodojo.policy import (
    RoboDojoPolicyConfig,
    WallXRoboDojoPolicy,
    build_infer_config,
)
from wall_x.serving.robodojo.server import PolicyServer, PolicyServerConfig


def optional_float(value: str) -> float | None:
    return None if value.lower() in ("none", "null") else float(value)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--train-config-path", help="Defaults to <checkpoint>/config.yml.")
    p.add_argument("--norm-key", required=True, help="Normalizer key in the checkpoint's norm stats.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=19000)
    p.add_argument("--model-device", default="cuda")
    p.add_argument("--robot-type", default="desktop")
    p.add_argument("--action-horizon", type=int, help="Defaults to the checkpoint's action horizon.")
    p.add_argument("--num-inference-timesteps", type=int, default=10)
    p.add_argument("--action-start-index", type=int, default=0)
    p.add_argument(
        "--replan-steps",
        type=int,
        default=16,
        help="Steps the client advances per request (platform replan_steps).",
    )
    p.add_argument(
        "--action-chunk-length",
        type=int,
        help="Actions returned per request. Defaults to --replan-steps for the "
        "plain policy and to the checkpoint's full action horizon for "
        "keyframe-memory checkpoints.",
    )
    p.add_argument("--default-instruction", default="", help="Used when an observation has no instruction.")
    p.add_argument("--enable-jpeg-ndarray-v1", action="store_true")
    p.add_argument("--ws-ping-interval-s", type=optional_float, default=None)
    p.add_argument("--ws-ping-timeout-s", type=optional_float, default=None)
    p.add_argument("--ws-compression", choices=("deflate", "none"), default="deflate")
    p.add_argument("--debug", action="store_true")
    memory = p.add_argument_group("keyframe memory")
    memory.add_argument(
        "--memory",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto: on for checkpoints trained with keyframe_memory_packing.",
    )
    memory.add_argument("--memory-capacity", type=int, help="Memory bank size.")
    memory.add_argument("--memory-candidates", type=int, help="Candidate frames per selection.")
    memory.add_argument("--memory-select-stride", type=int, help="Frames between selections.")
    memory.add_argument(
        "--memory-tasks",
        help="Comma-separated tasks that use memory (default: all tasks).",
    )
    return p.parse_args(argv)


def trained_with_keyframe_memory(infer_config) -> bool:
    epilogue = getattr(infer_config.data_config, "epilogue_processor_type", "")
    return epilogue == "keyframe_memory_packing"


def memory_overrides(args: argparse.Namespace) -> dict:
    overrides = {
        "capacity": args.memory_capacity,
        "num_candidates": args.memory_candidates,
        "select_stride": args.memory_select_stride,
    }
    if args.memory_tasks:
        overrides["memory_tasks"] = frozenset(
            task.strip() for task in args.memory_tasks.split(",") if task.strip()
        )
    if args.memory_select_stride is not None:
        overrides["write_tail"] = args.memory_select_stride
    return {k: v for k, v in overrides.items() if v is not None}


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    policy_config = RoboDojoPolicyConfig(
        checkpoint_path=args.checkpoint_path,
        train_config_path=args.train_config_path,
        norm_key=args.norm_key,
        model_device=args.model_device,
        robot_type=args.robot_type,
        action_horizon=args.action_horizon,
        num_inference_timesteps=args.num_inference_timesteps,
        action_start_index=args.action_start_index,
        replan_steps=args.replan_steps,
        action_chunk_length=args.action_chunk_length,
        default_instruction=args.default_instruction,
    )
    # Configs only (no weights); handed to the policy so it is loaded once.
    infer_config = build_infer_config(policy_config)
    use_memory = args.memory == "on" or (
        args.memory == "auto" and trained_with_keyframe_memory(infer_config)
    )
    overrides = memory_overrides(args)
    if use_memory:
        from wall_x.serving.robodojo.memory_policy import KeyframeMemoryRoboDojoPolicy

        policy = KeyframeMemoryRoboDojoPolicy(
            policy_config, infer_config, memory_overrides=overrides
        )
    else:
        if overrides:
            raise SystemExit(
                f"--memory-* options given, but keyframe memory is off "
                f"(--memory {args.memory}): {sorted(overrides)}"
            )
        policy = WallXRoboDojoPolicy(policy_config, infer_config)
    server = PolicyServer(
        policy,
        PolicyServerConfig(
            host=args.host,
            port=args.port,
            jpeg_ndarray_v1=args.enable_jpeg_ndarray_v1,
            ws_ping_interval_s=args.ws_ping_interval_s,
            ws_ping_timeout_s=args.ws_ping_timeout_s,
            ws_compression=None if args.ws_compression == "none" else "deflate",
        ),
    )
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
