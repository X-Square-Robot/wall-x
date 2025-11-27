#!/usr/bin/env python3
"""
Example client for Wall-X model server with sync support.

This script demonstrates how to connect to a Wall-X server and request
action predictions from observations in both sync and async contexts.
"""

import asyncio
import logging
from typing import Dict, List
import numpy as np
import threading
import yaml
import torch
import matplotlib.pyplot as plt
import os

from wall_x.data.utils import update_action_statistics
from wall_x.utils.constant import action_statistic_dof
from wall_x.model.action_head import Normalizer

try:
    import msgpack
    import msgpack_numpy as m

    m.patch()
except ImportError:
    print("Please install msgpack-numpy: pip install msgpack-numpy")
    exit(1)

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WallXClient:
    """Client for connecting to Wall-X model server."""

    def __init__(
        self,
        config_path: str,
        uri: str = "ws://localhost:8000",
        norm_stats_path: str = "x2_norm_stats.json",
    ):
        """Initialize client.

        Args:
            config_path: Path to train config file
            uri: WebSocket URI of the server (e.g., ws://localhost:8000)
            norm_stats_path: Path to normalization stats file
        """
        self.uri = uri
        self.websocket = None
        self.metadata = None
        self._loop = None
        self._thread = None
        self.norm_stats_path = norm_stats_path

        with open(config_path, "r") as f:
            self.train_config = yaml.load(f, Loader=yaml.FullLoader)

        self.init_normalizer(self.train_config)

    async def connect(self):
        """Connect to the server and receive metadata."""
        logger.info(f"Connecting to {self.uri}...")
        self.websocket = await websockets.connect(
            self.uri,
            ping_interval=None,
            ping_timeout=None,
            max_size=None,
        )

        self.metadata = msgpack.unpackb(await self.websocket.recv())
        logger.info(f"Connected! Server metadata: {self.metadata}")

    async def predict(self, obs: Dict) -> Dict:
        """Get action prediction from observation.

        Args:
            obs: Observation dictionary containing:
                - 'image': Image array (H, W, C)
                - 'prompt': Optional text prompt
                - 'state': Optional robot state

        Returns:
            Dictionary with:
                - 'action': Predicted action array
                - 'server_timing': Timing information
        """
        if self.websocket is None:
            raise RuntimeError("Not connected. Call connect() first.")

        await self.websocket.send(msgpack.packb(obs))
        response = msgpack.unpackb(await self.websocket.recv())
        return response

    async def close(self):
        """Close the connection."""
        if self.websocket:
            await self.websocket.close()
            logger.info("Connection closed")

    async def reset(self):
        """Reset the policy (if supported)."""
        pass

    # ============ Synchronous methods (using independent thread event loop) ============

    def _start_background_loop(self):
        """Start event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _ensure_loop(self):
        """Ensure background event loop is running."""
        if self._loop is None or not self._loop.is_running():
            self._thread = threading.Thread(
                target=self._start_background_loop, daemon=True
            )
            self._thread.start()
            # Wait for loop to start
            import time

            while self._loop is None:
                time.sleep(0.01)

    def _run_async(self, coro):
        """Run coroutine in background event loop."""
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def connect_sync(self):
        """Synchronously connect to server."""
        return self._run_async(self.connect())

    def norm_state(
        self,
        state: np.ndarray,
        dataset_names: List[str],
        state_mask: torch.Tensor = None,
    ) -> np.ndarray:
        """Normalize state."""
        return self.normalizer_propri.normalize_data(state, dataset_names, state_mask)

    def predict_sync(self, obs: Dict) -> Dict:
        """Synchronous prediction method.

        Args:
            obs: Observation dictionary

        Returns:
            Prediction result dictionary
        """
        return self._run_async(self.predict(obs))

    def close_sync(self):
        """Synchronously close connection."""
        result = self._run_async(self.close())
        # Stop event loop
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        return result

    def init_normalizer(self, train_config):
        # Define default configurations
        dof_config = {"biarm_eed_with_base": 20}

        agent_pos_config = {"biarm_eed_with_base": 20}

        update_action_statistics(
            action_statistic_dof=action_statistic_dof,
            norm_stats_path=self.norm_stats_path,
            repo_id="x2",
            dof_config=dof_config,
            agent_pos_config=agent_pos_config,
        )

        self.normalizer_action = Normalizer(action_statistic_dof, dof_config)
        self.normalizer_propri = Normalizer(action_statistic_dof, agent_pos_config)

        print("Normalizer initialized")


def prepare_batch_sync(data, normalizer_action, normalizer_propri, dataset_names):
    """Synchronous version of prepare_batch."""
    image = (data["image"].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    wrist_image = (
        (data["wrist_image"].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    )
    prompt = data["task"]

    state = data["state"].to("cuda")
    if state.dim() == 1:
        state = state.unsqueeze(0)

    state_mask = torch.ones([1, 32, 20]).to("cuda")
    state_mask[:, :, 8:] = 0

    state = normalizer_propri.normalize_data(state, dataset_names, state_mask)
    state = state.cpu().numpy().astype(np.float32)

    obs = {
        "front_view": image,
        "left_wrist_view": wrist_image,
        "prompt": prompt,
        "state": state,
        "dataset_names": dataset_names,
    }
    return obs


def init_serving_sample_dataset(train_config):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    repo_id = train_config["data"]["lerobot_config"]["repo_id"]

    meta_info = LeRobotDatasetMetadata(repo_id)
    dataset_fps = meta_info.fps
    delta_timestamps = {
        "actions": [t / dataset_fps for t in range(32)],
    }
    dataset = LeRobotDataset(
        repo_id,
        episodes=[0],
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

    return dataset, repo_id


# ============ Synchronous version of main function ============


def main_sync(args):
    """Synchronous version of main function."""

    # Create client and connect
    client = WallXClient(
        args.config_path, uri=args.uri, norm_stats_path=args.norm_stats_path
    )
    client.connect_sync()

    dataset, repo_id = init_serving_sample_dataset(client.train_config)

    total_frames = len(dataset)
    gt_traj = np.zeros((total_frames, args.action_dim))
    pred_traj = np.zeros((total_frames, args.action_dim))
    import torch

    dof_mask = torch.ones([1, 32, 20]).to("cuda")
    dof_mask[:, :, args.action_dim :] = 0

    # Synchronous processing
    for idx, data in enumerate(dataset):
        if idx % args.pred_horizon == 0 and idx + args.pred_horizon < total_frames:
            print(f"Processing frame {idx}")
            obs = prepare_batch_sync(
                data,
                client.normalizer_action,
                client.normalizer_propri,
                dataset_names=[repo_id],
            )
            response = client.predict_sync(obs)
            pred_action = response["action"]
            pred_traj[idx : idx + args.pred_horizon] = pred_action
            gt_traj[idx : idx + args.pred_horizon] = data["actions"]

    # Draw plot
    timesteps = gt_traj.shape[0]
    fig, axs = plt.subplots(
        args.action_dim, 1, figsize=(15, 5 * args.action_dim), sharex=True
    )
    fig.suptitle("Action Comparison for lerobot", fontsize=16)

    for i in range(args.action_dim):
        axs[i].plot(range(timesteps), gt_traj[:, i], label="Ground Truth")
        axs[i].plot(range(timesteps), pred_traj[:, i], label="Prediction")
        axs[i].set_ylabel(f"Action Dim {i+1}")
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "lerobot_comparison_serving.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

    # Close connection
    client.close_sync()


# ============ Asynchronous version of main function (keep original functionality) ============


async def main(args):
    client = WallXClient(
        args.config_path, uri=args.uri, norm_stats_path=args.norm_stats_path
    )
    await client.connect()
    dataset, repo_id = init_serving_sample_dataset(client.train_config)

    total_frames = len(dataset)
    gt_traj = np.zeros((total_frames, args.action_dim))
    pred_traj = np.zeros((total_frames, args.action_dim))

    for idx, data in enumerate(dataset):
        if idx % args.pred_horizon == 0 and idx + args.pred_horizon < total_frames:
            print(f"Processing frame {idx}")
            obs = prepare_batch_sync(
                data,
                client.normalizer_action,
                client.normalizer_propri,
                dataset_names=[repo_id],
            )
            response = await client.predict(obs)
            pred_action = response["action"]
            print(pred_action.shape)
            pred_traj[idx : idx + args.pred_horizon] = pred_action
            gt_traj[idx : idx + args.pred_horizon] = data["actions"]

    timesteps = gt_traj.shape[0]

    fig, axs = plt.subplots(
        args.action_dim, 1, figsize=(15, 5 * args.action_dim), sharex=True
    )
    fig.suptitle("Action Comparison for lerobot", fontsize=16)

    for i in range(args.action_dim):
        axs[i].plot(range(timesteps), gt_traj[:, i], label="Ground Truth")
        axs[i].plot(range(timesteps), pred_traj[:, i], label="Prediction")
        axs[i].set_ylabel(f"Action Dim {i+1}")
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "lerobot_comparison_serving.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    """Asynchronous version of main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Wall-X client examples")
    parser.add_argument(
        "--example",
        choices=["single", "multiple", "benchmark"],
        default="single",
        help="Example to run",
    )
    parser.add_argument(
        "--uri",
        default="ws://localhost:8000",
        help="Server URI",
    )
    parser.add_argument(
        "--pred_horizon", type=int, default=32, help="Prediction horizon"
    )
    parser.add_argument("--action_dim", type=int, default=7, help="Action dimension")
    parser.add_argument(
        "--config_path",
        default="config_from_qwen_libero.yml",
        help="Train config path",
    )
    parser.add_argument(
        "--save_dir",
        default="libero",
        help="Save directory",
    )
    parser.add_argument(
        "--norm_stats_path",
        default="x2_norm_stats.json",
        help="Normalization stats path",
    )
    args = parser.parse_args()

    # Synchronous mode
    main_sync(args)

    # Asynchronous mode
    # asyncio.run(main(args))
