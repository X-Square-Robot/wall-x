#!/usr/bin/env python3
# Copyright (c) 2025, X-Square-Robot
# SPDX-License-Identifier: BSD-3-Clause

"""
Server script for Wall-X model.

This script serves a Wall-X model using a websocket server, allowing
clients to connect and get action predictions from observations.

Based on the OpenPI serve_policy.py script structure.
"""

import dataclasses
from dataclasses import field
import enum
import logging
import socket
import sys
import yaml
from pathlib import Path
from typing import List

import tyro

from wall_x.serving.policy.wall_x_policy import WallXPolicy
from wall_x.serving.websocket_policy_server import WebsocketPolicyServer

logger = logging.getLogger(__name__)


class EnvMode(enum.Enum):
    """Supported environments/datasets."""

    LIBERO = "libero"
    ALOHA = "aloha"


@dataclasses.dataclass
class ModelConfig:
    """Configuration for loading a Wall-X model."""

    # Path to the pretrained model checkpoint
    model_path: str
    # Path to the action tokenizer
    action_tokenizer_path: str
    # Path to train config yaml
    train_config_path: str
    # Action dimension for the environment
    action_dim: int = 7
    # State dimension for the environment
    state_dim: int = 8
    # Prediction horizon (number of future actions to predict)
    pred_horizon: int = 32
    # Device to run model on
    device: str = "cuda"
    # Model dtype (bfloat16, float16, float32)
    dtype: str = "bfloat16"
    # Prediction mode (fast or slow)
    predict_mode: str = "fast"
    # Camera key for the environment
    camera_key: List[str] = field(
        default_factory=lambda: ["front_view", "left_wrist_view", "right_wrist_view"]
    )


@dataclasses.dataclass
class Args:
    """Arguments for the serve_wall_x script."""

    # Environment mode (used for default configurations)
    env: EnvMode = EnvMode.LIBERO

    # Model configuration. If not provided, uses default config for the environment
    model_config: ModelConfig | None = None

    # Default text prompt to use if not provided in observation
    default_prompt: str | None = None

    # Port to serve the policy on
    port: int = 8000

    # Host to bind the server to
    host: str = "0.0.0.0"

    # Enable debug logging
    debug: bool = False


# Default model configurations for each environment
DEFAULT_CONFIGS: dict[EnvMode, ModelConfig] = {
    EnvMode.LIBERO: ModelConfig(
        model_path="/path/to/model",
        action_tokenizer_path="/path/to/action_tokenizer",
        train_config_path="/path/to/train_config",
        state_dim=8,
        action_dim=7,
        pred_horizon=32,
        device="cuda",
        dtype="bfloat16",
        predict_mode="fast",
        camera_key=["front_view", "left_wrist_view"],
    ),
    EnvMode.ALOHA: ModelConfig(
        model_path="/path/to/model",
        action_tokenizer_path="/path/to/action_tokenizer",
        train_config_path="/path/to/train_config",
        state_dim=14,
        action_dim=14,
        pred_horizon=32,
        device="cuda",
        dtype="bfloat16",
        predict_mode="fast",
        camera_key=["face_view", "left_wrist_view", "right_wrist_view"],
    ),
}


def get_model_config(args: Args) -> ModelConfig:
    """Get model configuration from args or defaults."""
    if args.model_config is not None:
        return args.model_config

    if config := DEFAULT_CONFIGS.get(args.env):
        logger.info(f"Using default configuration for {args.env.value}")
        return config

    raise ValueError(
        f"No default configuration for {args.env.value}. "
        f"Please provide --model-config with model_path and action_tokenizer_path."
    )


def create_policy(args: Args) -> WallXPolicy:
    """Create a Wall-X policy from the given arguments."""
    config = get_model_config(args)
    logger.info(f"Creating Wall-X policy with config: {config}")

    # Validate paths
    if not Path(config.model_path).exists():
        logger.warning(f"Model path does not exist: {config.model_path}")

    if not Path(config.action_tokenizer_path).exists():
        logger.warning(
            f"Action tokenizer path does not exist: {config.action_tokenizer_path}"
        )

    with open(config.train_config_path, "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    policy = WallXPolicy(
        model_path=config.model_path,
        train_config=train_config,
        action_tokenizer_path=config.action_tokenizer_path,
        action_dim=config.action_dim,
        agent_pos_dim=config.state_dim,
        pred_horizon=config.pred_horizon,
        device=config.device,
        dtype=config.dtype,
        predict_mode=config.predict_mode,
        default_prompt=args.default_prompt,
        camera_key=config.camera_key,
    )

    return policy


def main(args: Args) -> None:
    """Main function to start the Wall-X model server."""
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting Wall-X model server")
    logger.info(f"Environment: {args.env.value}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Host: {args.host}")

    # Create policy
    try:
        policy = create_policy(args)
    except Exception as e:
        logger.error(f"Failed to create policy: {e}")
        sys.exit(1)

    # Get policy metadata
    policy_metadata = policy.metadata
    policy_metadata["env"] = args.env.value

    # Get network info
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "unknown"

    logger.info(f"Server hostname: {hostname}")
    logger.info(f"Server IP: {local_ip}")
    logger.info(f"Server will be available at: ws://{args.host}:{args.port}")
    logger.info(f"Health check endpoint: http://{args.host}:{args.port}/healthz")

    # Create and start server
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
    )

    logger.info("Starting server...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
