#!/usr/bin/env python3
"""
Server script for Wall-X model.

This script serves a Wall-X model using a websocket server, allowing
clients to connect and get action predictions from observations.

Based on the OpenPI serve_policy.py script structure.
"""

import dataclasses
import enum
import logging
import os
import socket
import sys
import yaml
import traceback
import tyro

from wall_x._vendor.harrix.serving.websocket_policy_server import WebsocketPolicyServer


def _infer_config_kwargs(model_config, config_cls):
    """ServerModelConfig fields are the union across the three policies; filter by the target InferConfig signature."""
    import inspect

    accepted = set(inspect.signature(config_cls.__init__).parameters)
    return {
        k: v
        for k, v in vars(model_config).items()
        if v is not None and k in accepted
    }


def get_wallx_policy(model_config, image_passing_mode, serialize_actions=True):
    from wall_x._vendor.harrix.serving.policy.wall_x_policy import WallXPolicy
    from wall_x._vendor.harrix.infer.infer_config import InferConfig

    config = InferConfig(**_infer_config_kwargs(model_config, InferConfig))
    # default_infer_mode is a ServerModelConfig field, not an InferConfig one
    # (so _infer_config_kwargs filters it out); pass it to the policy directly.
    return WallXPolicy(
        config=config,
        image_passing_mode=image_passing_mode,
        default_infer_mode=getattr(model_config, "default_infer_mode", "flow"),
        serialize_actions=serialize_actions,
        ar_do_sample=getattr(model_config, "ar_do_sample", False),
        ar_temperature=getattr(model_config, "ar_temperature", 1.0),
        instruction_override=getattr(model_config, "instruction_override", None),
    )


logger = logging.getLogger(__name__)


class EnvMode(enum.Enum):
    """Supported environments/datasets."""

    X2ROBOT = "x2robot"
    LIBERO = "libero"


@dataclasses.dataclass
class ServerModelConfig:
    """Configuration for loading a Wall-X model."""

    checkpoint_path: str | None = None
    train_config_path: str | None = None
    # robot_host: str = '0.0.0.0'
    # robot_port: int = 33723
    robot_type: str = "desktop"  # ["desktop", "turtle", "ex001"]
    robot_id: str | None = None  # Defaults from robot_type; may be overridden.
    robot_action_start_ratio: float = (
        0.0  # proportion of action execution to start from
    )
    robot_action_end_ratio: float = 1.0  # proportion of action execution to end at
    robot_action_interpolate_multiplier: int = 10  # action interpolation multiplier
    robot_use_joint_angle_control: bool = (
        False  # use joint angle control (model must predict joints)
    )
    turtle_as_desktop: bool = (
        False  # use turtle platform as desktop with fixed base/head/camera/height
    )
    ex001_as_desktop: bool = False
    # Shared fixed actuator targets for mobile robots in desktop mode.
    fixed_head_pos: tuple[float, float] = (0.0, -1.0)  # yaw, pitch
    fixed_lift: float = 0.4
    # Base commands retain platform-specific protocol semantics.
    fixed_car_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, yaw
    fixed_base_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)  # vx, vy, wz
    action_horizon: int = 32  # specify the correct horizon for the model
    action_dim: int | None = None
    default_infer_mode: str = "flow"
    ar_do_sample: bool = False
    ar_temperature: float = 1.0
    # Server-side prompt hack: if set, REPLACES the client-sent obs["instruction"]
    instruction_override: str | None = None
    model_device: str = "cuda"
    num_inference_timesteps: int = 10
    num_inference_steps: int | None = None
    cfg_scale: float | None = None
    seed: int | None = None
    save_video_dir: str = "./videos"

    # Please specify explicitly if the checkpoint was not trained on the x2robot dataset.
    norm_key: str | None = None
    # Dataset V2 normalizer discriminator. None keeps legacy flat-key behavior
    # and auto-resolves a single unambiguous composite key.
    end_effector_type: str | None = None
    # Optional serving-host override for Dataset V2 stats.
    normalizer_stats_path: str | None = None
    # Remove the discretized state string from the prompt while preserving the
    # checkpoint's state-string model topology.
    state_drop: bool = False
    # Robot camera keys in incoming websocket observations.
    camera_front_key: str = "camera_front"
    camera_left_key: str = "camera_left"
    camera_right_key: str = "camera_right"

@dataclasses.dataclass
class Args:
    """Arguments for the serve_wall_x script."""

    # Environment mode (used for default configurations)
    env: EnvMode = EnvMode.X2ROBOT

    # Model configuration. If not provided, uses default config for the environment
    model_config: ServerModelConfig | None = None

    # Default text prompt to use if not provided in observation
    default_prompt: str | None = None

    # Port to serve the policy on
    port: int = 43007

    # Host to bind the server to
    host: str = "0.0.0.0"

    # Enable debug logging
    debug: bool = False

    # Image passing mode
    image_passing_mode: str = "base64"  # ["numpy", "base64"]

    # Model type
    model_type: str = "wallx"

    # Serialize actions via robot_preprocessor (True for robot control, False for raw output)
    serialize_actions: bool = True

    # ── Dynamic batching ─────────────────────────────────────────
    # Set max_batch_size to enable dynamic batching. None = single mode.
    max_batch_size: int | None = None
    max_wait_time_ms: float = 0
    max_queue_size: int = 100
    timeout_ms: float = 30000



# Default model configurations for each environment
DEFAULT_CONFIGS: dict[EnvMode, ServerModelConfig] = {}

def get_model_config(args: Args) -> ServerModelConfig:
    """Get model configuration from args or defaults."""
    if args.model_config is not None:
        return args.model_config

    if config := DEFAULT_CONFIGS.get(args.env):
        logger.info(f"Using default configuration for {args.env.value}")
        return config

    raise ValueError(
        f"No default configuration for {args.env.value}. "
        "Pass model-config:server-model-config --model-config.checkpoint-path PATH."
    )


def create_policy(args: Args):
    """Create a policy from the given arguments."""
    model_config = get_model_config(args)
    if args.model_type == "wallx":
        policy = get_wallx_policy(model_config, args.image_passing_mode, args.serialize_actions)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    return policy


def main(args: Args) -> None:
    """Main function to start the Wall-X model server."""
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting model server")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Environment: {args.env.value}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Host: {args.host}")

    # Create policy
    try:
        policy = create_policy(args)
    except Exception as e:
        logger.error(f"Failed to create policy: {e}")
        logger.error(traceback.format_exc())
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

    batching_str = (
        f"batch_size={args.max_batch_size}, wait={args.max_wait_time_ms}ms"
        if args.max_batch_size
        else "disabled"
    )
    logger.info(f"Batching: {batching_str}")

    # Create and start server
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
        max_batch_size=args.max_batch_size,
        max_wait_time_ms=args.max_wait_time_ms,
        max_queue_size=args.max_queue_size,
        timeout_ms=args.timeout_ms,
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
