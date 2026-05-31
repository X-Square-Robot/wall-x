#!/usr/bin/env python3
"""
Example client for Wall-X model server with sync support.

This script demonstrates how to connect to a Wall-X server and request
action predictions from observations in both sync and async contexts.
"""

import asyncio
import logging
from typing import Dict
import threading
import time
import numpy as np
import torch
import os


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


class RobotClient:
    """Client for connecting to Wall-X model server."""

    def __init__(self, config_path: str, uri: str = "ws://localhost:8000"):
        """Initialize client.

        Args:
            uri: WebSocket URI of the server (e.g., ws://localhost:8000)
        """
        self.uri = uri
        self.websocket = None
        self.metadata = None
        self._loop = None
        self._thread = None

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

        ziped = msgpack.packb(obs)
        size_bytes = len(ziped)
        print(f"{size_bytes / 1024:.2f} KB")
        print(f"{size_bytes / 1024 / 1024:.2f} MB")
        await self.websocket.send(ziped)
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


# ============ Synchronous version of main function ============


def main_sync(args):
    """Synchronous version of main function."""

    # Create client and connect
    client = RobotClient(args.config_path, uri=args.uri)
    client.connect_sync()
    import pickle

    with open(args.config_path, "rb") as f:
        all_views = pickle.load(f)

    with open(args.config_path, "rb") as f:
        all_actions = pickle.load(f)
    print(all_actions.keys())
    print(all_views.keys())

    pos = {
        "follow1_pos": all_actions["follow1_pos"],
        "follow2_pos": all_actions["follow2_pos"],
    }
    use_base64 = False
    if use_base64:
        import cv2
        import base64

        for k, v in all_views.items():
            v = v.squeeze(0)
            encoded_img = cv2.imencode(".jpg", v)[1]
            print(encoded_img.shape)

            base64_str = base64.b64encode(encoded_img).decode("utf-8")
            all_views[k] = base64_str
        obs = {
            "state": pos,
            "views": all_views,
            "instruction": "Pick up the cup.",
        }
    else:
        path = args.config_path
        follow_pos_1 = torch.load(os.path.join(path, "state_follow1_pos.pt")).numpy()
        follow_pos_2 = torch.load(os.path.join(path, "state_follow2_pos.pt")).numpy()
        camera_front = torch.load(os.path.join(path, "views_camera_front.pt")).numpy()
        camera_left = torch.load(os.path.join(path, "views_camera_left.pt")).numpy()
        camera_right = torch.load(os.path.join(path, "views_camera_right.pt")).numpy()
        obs = {
            "state": {
                "follow1_pos": follow_pos_1,
                "follow2_pos": follow_pos_2,
            },
            "views": {
                "camera_front": camera_front,
                "camera_left": camera_left,
                "camera_right": camera_right,
            },
            "instruction": "Pick up the cup.",
        }
    for _ in range(1):
        start_time = time.perf_counter()
        response = client.predict_sync(obs)
        print(response["action"].shape)
        print(response["action"])
        end_time = time.perf_counter()
        print(f"Time taken: {(end_time - start_time) * 1000} ms")
    action_label = torch.load(args.config_path).numpy()
    np.testing.assert_array_equal(response["action"], action_label)
    print("compare success")
    client.close_sync()


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
        default="/path/to/example_data",
        help="Train config path",
    )
    parser.add_argument(
        "--save_dir",
        default="/path/to/example_data",
        help="Save directory",
    )
    args = parser.parse_args()

    # Synchronous mode
    main_sync(args)

    # Asynchronous mode
    # asyncio.run(main(args))
