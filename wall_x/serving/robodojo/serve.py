"""Serve a Wall-X flow checkpoint to RoboDojo over the XPolicyLab WebSocket protocol.

Example:
    python -m wall_x.serving.robodojo.serve \\
        --checkpoint-path /path/to/ckpt --norm-key ex_normal --port 43865
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from wall_x.serving.robodojo.policy import RoboDojoPolicyConfig, WallXRoboDojoPolicy
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
    p.add_argument("--replan-steps", type=int, default=16, help="Actions returned per request.")
    p.add_argument("--default-instruction", default="", help="Used when an observation has no instruction.")
    p.add_argument("--enable-jpeg-ndarray-v1", action="store_true")
    p.add_argument("--ws-ping-interval-s", type=optional_float, default=None)
    p.add_argument("--ws-ping-timeout-s", type=optional_float, default=None)
    p.add_argument("--ws-compression", choices=("deflate", "none"), default="deflate")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    policy = WallXRoboDojoPolicy(
        RoboDojoPolicyConfig(
            checkpoint_path=args.checkpoint_path,
            train_config_path=args.train_config_path,
            norm_key=args.norm_key,
            model_device=args.model_device,
            robot_type=args.robot_type,
            action_horizon=args.action_horizon,
            num_inference_timesteps=args.num_inference_timesteps,
            action_start_index=args.action_start_index,
            replan_steps=args.replan_steps,
            default_instruction=args.default_instruction,
        )
    )
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
