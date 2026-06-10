"""Base environment abstraction.

Two execution granularities are supported:

- Episode-level ``run_episode``: caller supplies a predict callback and the env
  owns the full episode loop.
- Chunk-level ``reset_episode`` + ``execute_chunk``: caller runs the model
  between chunks and feeds action chunks back to the env.

Subclasses must implement the chunk-level primitives. The default episode loop
is built on top of those primitives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from wall_x._vendor.harrix.eval_config import EvalConfig


class BaseEnv(ABC):

    @abstractmethod
    def __init__(self, cfg: EvalConfig, worker_id: int) -> None:
        """Perform env-specific setup."""

    @classmethod
    @abstractmethod
    def enumerate_episodes(cls, cfg: EvalConfig) -> list[tuple]:
        """Return episode ids to seed JobState before env instances are built.

        JobState treats the returned tuples as opaque ids.
        """

    @property
    @abstractmethod
    def robot_spec(self) -> dict:
        """Return robot metadata for driver/checkpoint validation.

        Fields:
          - dof_layout: dict[str, int]
          - cam_names:  list[str]
          - norm_key:   str
        """

    # ---- chunk-level primitives ----

    @abstractmethod
    def reset_episode(self, ep_id: tuple) -> dict:
        """Start a new episode and return the first fresh observation.

        Returns {"obs": dict, "instruction": str, "task_desc": str (optional)}.
        """

    @abstractmethod
    def execute_chunk(self, actions: np.ndarray) -> dict:
        """Execute one action chunk with shape ``(H, action_dim)``.

        Returns {"obs": dict, "done": bool, "steps": int}.
        """

    def shutdown(self) -> None:
        """Release resources. Subclasses may override."""

    # ---- episode-level default implementation ----

    def run_episode(
        self,
        ep_id: tuple,
        predict: Callable[[dict, str, int], np.ndarray],
    ) -> dict:
        """Episode loop built from reset, predict, and execute_chunk.

        ``predict(observation, instruction, step)`` returns one action chunk.
        The env remains unaware of the model transport.
        """
        import time

        from wall_x._vendor.harrix.envs.libero_common import encode_raw_obs

        t_ep_start = time.time()
        initial = self.reset_episode(ep_id)
        obs = initial["obs"]
        instruction = initial["instruction"]
        task_desc = initial.get("task_desc", "")

        max_rounds = self._max_infer_rounds()
        success = False
        steps_total = 0
        for round_idx in range(max_rounds):
            encoded = encode_raw_obs(obs)
            chunk = predict(encoded, instruction, round_idx)
            result = self.execute_chunk(chunk)
            obs = result["obs"]
            steps_total += result["steps"]
            if result["done"]:
                success = True
                break

        self.finalize_episode(success)
        return {
            "success": bool(success),
            "steps": steps_total,
            "elapsed_sec": round(time.time() - t_ep_start, 3),
            "task_desc": task_desc,
        }

    def finalize_episode(self, success: bool) -> None:
        """Hook for env-specific cleanup after an episode (e.g. save rollouts)."""

    def _max_infer_rounds(self) -> int:
        """Return the maximum number of model chunks for one episode."""
        raise NotImplementedError(
            f"{type(self).__name__} must override _max_infer_rounds when using "
            "the default run_episode implementation"
        )
