import logging
import os
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from functools import wraps
from typing import List

import torch
from torch.cuda import nvtx

logger = logging.getLogger(__name__)

ENABLE_PERFORMANCE_TIMING = (
    os.environ.get("ENABLE_PERFORMANCE_TIMING", "True").lower() == "true"
)

ENABLE_CUDA_SYNC_IN_TIMER = (
    os.environ.get("ENABLE_CUDA_SYNC_IN_TIMER", "False").lower() == "true"
)


class ScopeTimerContext:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        if ENABLE_CUDA_SYNC_IN_TIMER and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if ENABLE_CUDA_SYNC_IN_TIMER and torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        cost_ms = (end_time - self.start_time) * 1e3
        logger.info("%s took %.3f ms to execute", self.msg, cost_ms)


ScopeTimer = ScopeTimerContext if ENABLE_PERFORMANCE_TIMING else nullcontext


def timer(func, msg=None):
    """Decorator to measure function execution time."""
    if msg is None:
        msg = func.__name__
    else:
        msg = f"{func.__name__:} {msg}"

    @wraps(func)
    def wrapper(*args, **kwargs):
        with ScopeTimer(msg):
            result = func(*args, **kwargs)
        return result

    return wrapper


def _is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _get_world_size():
    if _is_distributed():
        return torch.distributed.get_world_size()
    return 1


def _get_rank():
    if _is_distributed():
        return torch.distributed.get_rank()
    return 0


def _barrier(group=None):
    if _is_distributed():
        torch.distributed.barrier(group=group)


if torch.distributed.is_available():
    try:
        dist_all_gather_func = torch.distributed.all_gather_into_tensor
    except AttributeError:
        dist_all_gather_func = torch.distributed.all_gather
else:
    dist_all_gather_func = None


class TimerBase(ABC):
    """Timer base class."""

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def start(self, barrier=False):
        """Start the timer, optionally syncing all ranks with a barrier first."""
        pass

    @abstractmethod
    def stop(self, barrier=False):
        """Stop the timer, optionally syncing all ranks with a barrier first."""
        pass

    @abstractmethod
    def reset(self):
        """Reset accumulated elapsed time to zero."""
        pass

    @abstractmethod
    def elapsed(self, reset=True, barrier=False):
        """Return accumulated elapsed time in seconds; reset if reset=True."""
        pass


class DummyTimer(TimerBase):
    """Dummy Timer - no-op placeholder used when log level exceeds threshold."""

    def __init__(self):
        super().__init__("dummy timer")

    def start(self, barrier=False, nvtx_push=False, sync=False, **kwargs):
        return

    def stop(self, barrier=False, sync=False, **kwargs):
        return

    def reset(self):
        return

    def elapsed(self, reset=True, barrier=False):
        raise Exception(
            "dummy timer should not be used to calculate elapsed time, "
            "check if timer's log_level <= self._log_level."
        )

    def active_time(self):
        raise Exception(
            "active timer should not be used to calculate elapsed time, "
            "check if timer's log_level <= self._log_level."
        )


class Timer(TimerBase):
    """
    Timer class with ability to start/stop.

    Comment on using `barrier`: If this flag is passed, then all
    the caller processes will wait till all reach the timing routine.
    It is up to the user to make sure all the ranks in `barrier_group`
    call it otherwise, it will result in a hang.
    Comment on `barrier_group`: By default it is set to None which
    in torch distributed land, it will result in the global communicator.
    """

    def __init__(self, name):
        super().__init__(name)
        self._elapsed = 0.0
        self._active_time = 0.0
        self._started = False
        self._barrier_group = None
        self._start_time = time.time()
        self.nvtx = False

    def set_barrier_group(self, barrier_group):
        self._barrier_group = barrier_group

    def start(self, barrier=False, nvtx_push=False, sync=False):
        assert not self._started, "timer has already been started"
        if barrier:
            _barrier(group=self._barrier_group)
        if torch.cuda.is_available() and sync:
            torch.cuda.synchronize()
        self._start_time = time.time()
        self._started = True
        if nvtx_push:
            nvtx.range_push("{}".format(self.name))
            self.nvtx = True

    def stop(self, barrier=False, sync=False):
        if self.nvtx:
            nvtx.range_pop()
        assert self._started, "timer is not started"
        if barrier:
            _barrier(group=self._barrier_group)
        if torch.cuda.is_available() and sync:
            torch.cuda.synchronize()
        elapsed = time.time() - self._start_time
        self._elapsed += elapsed
        self._active_time += elapsed
        self._started = False

    def reset(self):
        self._elapsed = 0.0
        self._started = False

    def elapsed(self, reset=True, barrier=False):
        _started = self._started
        if self._started:
            self.stop(barrier=barrier)
        _elapsed = self._elapsed
        if reset:
            self.reset()
        if _started:
            self.start(barrier=barrier)
        return _elapsed

    def active_time(self):
        return self._active_time


class Timers:
    """Class for a group of Timers."""

    def __init__(self, log_level, log_option):
        self._log_level = log_level
        allowed_log_options = set(["max", "minmax", "all"])
        assert (
            log_option in allowed_log_options
        ), "input log option {} is invalid. It must be one of {}".format(
            log_option, allowed_log_options
        )
        self._log_option = log_option
        self._timers = {}
        self._log_levels = {}
        self._dummy_timer = DummyTimer()
        self._max_log_level = 2

    def __call__(self, name, log_level=None):
        if name in self._timers:
            if log_level is not None:
                assert log_level == self._log_levels[name], (
                    "input log level {} does not match already existing "
                    "log level {} for {} timer".format(
                        log_level, self._log_levels[name], name
                    )
                )
            return self._timers[name]
        if log_level is None:
            log_level = self._max_log_level
        assert (
            log_level <= self._max_log_level
        ), "log level {} is larger than max supported log level {}".format(
            log_level, self._max_log_level
        )
        if log_level > self._log_level:
            return self._dummy_timer
        self._timers[name] = Timer(name)
        self._log_levels[name] = log_level
        return self._timers[name]

    def _get_elapsed_time_all_ranks(self, names, reset, barrier):
        if barrier:
            _barrier()

        world_size = _get_world_size()
        rank = _get_rank()

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = torch.device("cpu")

        rank_name_to_time = torch.zeros(
            (world_size, len(names)), dtype=torch.float, device=device
        )

        for i, name in enumerate(names):
            if name in self._timers:
                rank_name_to_time[rank, i] = self._timers[name].elapsed(reset=reset)

        if world_size > 1 and _is_distributed() and dist_all_gather_func is not None:
            try:
                dist_all_gather_func(
                    rank_name_to_time.view(-1), rank_name_to_time[rank, :].view(-1)
                )
            except Exception as e:
                logger.warning("all_gather failed: %s. Using single rank timing.", e)

        return rank_name_to_time

    def _get_global_min_max_time(self, names, reset, barrier, normalizer):
        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier)
        name_to_min_max_time = {}
        for i, name in enumerate(names):
            rank_to_time = rank_name_to_time[:, i]
            rank_to_time = rank_to_time[rank_to_time > 0.0]
            if rank_to_time.numel() > 0:
                name_to_min_max_time[name] = (
                    rank_to_time.min().item() / normalizer,
                    rank_to_time.max().item() / normalizer,
                )
        return name_to_min_max_time

    def _get_global_min_max_time_string(
        self, names, reset, barrier, normalizer, max_only
    ):
        name_to_min_max_time = self._get_global_min_max_time(
            names, reset, barrier, normalizer
        )
        if not name_to_min_max_time:
            return None

        world_size = _get_world_size()
        if world_size == 1:
            output_string = "time (ms):"
            for name in name_to_min_max_time:
                _, max_time = name_to_min_max_time[name]
                output_string += "\n    {}: {:.2f}".format(
                    (name + " ").ljust(48, "."), max_time
                )
        else:
            if max_only:
                output_string = "max time across ranks (ms):"
            else:
                output_string = "(min, max) time across ranks (ms):"
            for name in name_to_min_max_time:
                min_time, max_time = name_to_min_max_time[name]
                if max_only:
                    output_string += "\n    {}: {:.2f}".format(
                        (name + " ").ljust(48, "."), max_time
                    )
                else:
                    output_string += "\n    {}: ({:.2f}, {:.2f})".format(
                        (name + " ").ljust(48, "."), min_time, max_time
                    )
        return output_string

    def _get_all_ranks_time_string(self, names, reset, barrier, normalizer):
        rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier)
        world_size = _get_world_size()

        output_string = "times across ranks (ms):"
        no_reported_timing = True
        for i, name in enumerate(names):
            not_yet_found = True
            for rank in range(world_size):
                if rank_name_to_time[rank, i] > 0:
                    no_reported_timing = False
                    if not_yet_found:
                        not_yet_found = False
                        output_string += "\n  {}:".format(name)
                    if world_size == 1:
                        output_string += "\n     {:.2f}".format(
                            rank_name_to_time[rank, i] / normalizer
                        )
                    else:
                        output_string += "\n     rank {:2d}: {:.2f}".format(
                            rank, rank_name_to_time[rank, i] / normalizer
                        )
        if no_reported_timing:
            return None
        return output_string

    def get_all_timers_string(
        self,
        names: List[str] = None,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """Return a formatted timing string for the given timer names.

        Args:
            names: Timers to include; defaults to all registered timers.
            normalizer: Divide raw seconds by this value (e.g. 1000 for ms output).
            reset: Reset each timer after reading its elapsed time.
            barrier: Synchronize across ranks before gathering times.
        """
        if names is None:
            names = list(self._timers.keys())

        assert normalizer > 0.0
        if self._log_option in ["max", "minmax"]:
            max_only = self._log_option == "max"
            output_string = self._get_global_min_max_time_string(
                names, reset, barrier, normalizer / 1000.0, max_only
            )
        elif self._log_option == "all":
            output_string = self._get_all_ranks_time_string(
                names, reset, barrier, normalizer / 1000.0
            )
        else:
            raise Exception("unknown timing log option {}".format(self._log_option))
        return output_string

    def log(
        self,
        names: List[str],
        rank: int = None,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """Print timing results for the given names to stdout on one rank.

        Args:
            names: Timer names to log.
            rank: Rank that prints; defaults to the last rank (world_size - 1).
            normalizer: Divide raw seconds by this value before printing.
            reset: Reset each timer after reading.
            barrier: Synchronize across ranks first.
        """
        output_string = self.get_all_timers_string(names, normalizer, reset, barrier)
        world_size = _get_world_size()
        current_rank = _get_rank()

        if rank is None:
            rank = world_size - 1
        if rank == current_rank and output_string is not None:
            logger.info("%s", output_string)

    def write(
        self,
        names: List[str],
        writer,
        iteration: int,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """Write per-timer max times as TensorBoard scalars.

        Args:
            names: Timer names to write.
            writer: TensorBoard SummaryWriter instance.
            iteration: Global step value for the scalar.
            normalizer: Divide raw seconds by this value.
            reset: Reset each timer after reading.
            barrier: Synchronize across ranks first.
        """
        assert normalizer > 0.0
        name_to_min_max_time = self._get_global_min_max_time(
            names, reset, barrier, normalizer
        )
        if writer is not None:
            for name in name_to_min_max_time:
                _, max_time = name_to_min_max_time[name]
                writer.add_scalar(name + "-time", max_time, iteration)
