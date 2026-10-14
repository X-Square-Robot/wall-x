"""Atomic publication helpers for directory training checkpoints.

Writers populate a hidden sibling directory and publish it with one rename
only after every rank has finished writing.  New-format checkpoints are valid
only when their completion marker is present; directories without the new
trainer metadata remain readable as legacy checkpoints.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


COMPLETION_MARKER = "_SUCCESS"
TRAINER_STATE_FILE = "trainer_state.pth"
TRANSACTION_FORMAT_VERSION = 1


def begin(final_path: str) -> str:
    """Create and return a unique hidden staging directory."""
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(final):
        raise FileExistsError(f"Checkpoint already exists: {final}")

    staging = final.parent / f".{final.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    return str(staging)


def publish(staging_path: str, final_path: str) -> None:
    """Fsync a completion marker and atomically rename staging to final."""
    staging = Path(staging_path)
    final = Path(final_path)
    if staging.parent != final.parent:
        raise ValueError("Checkpoint staging and final paths must be siblings")
    if not staging.is_dir():
        raise FileNotFoundError(f"Checkpoint staging directory not found: {staging}")
    if os.path.lexists(final):
        raise FileExistsError(f"Checkpoint already exists: {final}")

    marker = staging / COMPLETION_MARKER
    with marker.open("w", encoding="utf-8") as stream:
        json.dump(
            {"transaction_format_version": TRANSACTION_FORMAT_VERSION}, stream
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

    _fsync_directory(staging)
    os.replace(staging, final)
    _fsync_directory(final.parent)


def validate_for_resume(checkpoint_path: str) -> None:
    """Reject incomplete new-format directories while accepting legacy ones."""
    checkpoint = Path(checkpoint_path)
    trainer_state = checkpoint / TRAINER_STATE_FILE
    marker = checkpoint / COMPLETION_MARKER
    if trainer_state.exists() and not marker.is_file():
        raise RuntimeError(
            f"Incomplete checkpoint (missing {COMPLETION_MARKER}): {checkpoint}"
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
