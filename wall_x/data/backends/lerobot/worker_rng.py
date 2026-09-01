"""Process-local random number generation for LeRobot data workers."""

from __future__ import annotations

import operator
import os

import numpy as np
import torch

FLOW_TIMESTEP_TRAIN_STREAM = 0
FLOW_TIMESTEP_VALIDATION_STREAM = 1


def _as_nonnegative_int(value, name: str) -> int:
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}"
        ) from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return int(value)


class WorkerNumpyRNG:
    """Lazily create a deterministic NumPy generator per stream and worker.

    DataLoader copies the collator into each worker process. Creating a NumPy
    generator in the collator constructor would therefore copy the same RNG
    state into every worker. Waiting until the first worker-side call lets us
    use PyTorch's distinct worker seed instead. ``stream`` separates consumers
    such as training and validation even when they run in the main process and
    therefore share the same PyTorch seed.
    """

    def __init__(self, *, rank: int = 0, stream: int):
        self.rank = _as_nonnegative_int(rank, "rank")
        self.stream = _as_nonnegative_int(stream, "stream")
        self._seed_key: tuple[int, int, int, int] | None = None
        self._generator: np.random.Generator | None = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_seed_key"] = None
        state["_generator"] = None
        return state

    def generator(self) -> np.random.Generator:
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = int(
            worker_info.seed if worker_info is not None else torch.initial_seed()
        )
        if worker_seed < 0:
            raise ValueError(f"worker seed must be non-negative, got {worker_seed}")
        seed_key = (os.getpid(), worker_seed, self.rank, self.stream)

        if self._generator is None or self._seed_key != seed_key:
            seed_sequence = np.random.SeedSequence(
                [worker_seed, self.rank, self.stream]
            )
            self._generator = np.random.default_rng(seed_sequence)
            self._seed_key = seed_key

        return self._generator
