"""Rank-aware text logger for distributed training."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from torch.distributed import get_rank, is_initialized


class DistributedLogger:
    def __init__(
        self,
        name: str = "wallx",
        save_path: Optional[str] = None,
        level: int = logging.INFO,
    ):
        if is_initialized():
            self._rank = get_rank()
        else:
            self._rank = int(os.environ.get("RANK", 0))
            logging.warning(
                "DistributedLogger created before init_process_group; "
                "falling back to RANK env var (rank=%d).",
                self._rank,
            )
        logger = logging.getLogger(f"{name}.rank{self._rank}")
        logger.setLevel(level)
        logger.propagate = False
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)

        fmt = logging.Formatter(
            f"%(asctime)s - [rank{self._rank}] - %(levelname)s - %(message)s"
        )

        # All ranks write a file log (if a save_path was given).
        if save_path:
            log_dir = Path(save_path) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"rank_{self._rank}.log")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

        # Only rank 0 writes to stdout.
        if self._rank == 0:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(fmt)
            logger.addHandler(stream_handler)

        self._logger = logger

    # Thin pass-throughs — callers use standard logging verbs.
    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    @property
    def rank(self) -> int:
        return self._rank

    # --- backward-compatible shim ----------------------------------------
    # Some legacy call sites still invoke `.log(msg, level=...,
    # main_process_only=...)`. Forward those to the standard logger so we
    # don't break them during the migration; new code should use .info /
    # .warning / .error / .debug directly.
    def log(
        self,
        message,
        level: int = logging.INFO,
        main_process_only: bool = False,
    ):
        if main_process_only and self._rank != 0:
            return
        self._logger.log(level, message)

    # Pytorch's ``accelerate``-style fallback for code that still passes an
    # accelerator object to the old constructor; accept and ignore it.
    # Older codepaths can be migrated incrementally.
    @classmethod
    def legacy(cls, name: str, level: int = logging.INFO, accelerator=None):
        del accelerator  # ignored
        return cls(name=name, save_path=None, level=level)
