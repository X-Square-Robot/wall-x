"""Subprocess env handle used by the in-process driver.

The model remains in the driver process. Each env subprocess receives reset and
execute-chunk commands through a pipe.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random

import numpy as np


def _subproc_main(cfg, worker_id, child_conn):
    """Env subprocess entry point."""
    # Spawned children inherit env vars, but set these explicitly for launchers
    # that did not configure them. Robosuite validates the EGL id against the
    # CUDA_VISIBLE_DEVICES environment string, so keep the same visible id here.
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    # EGL device index is always 0 after CUDA_VISIBLE_DEVICES remapping.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ.get("MUJOCO_EGL_DEVICE_ID") or "0"

    # Env workers run robosuite only. Do not import torch here; otherwise many
    # subprocesses may initialize CUDA contexts and compete with the driver model.
    seed = cfg.env.seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import wall_x._vendor.harrix.envs  # noqa: F401  trigger register
    from wall_x._vendor.harrix.envs.registry import build_env

    env = build_env(cfg, worker_id)
    try:
        while True:
            try:
                cmd, args = child_conn.recv()
            except EOFError:
                break
            try:
                if cmd == "reset_episode":
                    result = env.reset_episode(tuple(args))
                elif cmd == "execute_chunk":
                    result = env.execute_chunk(args)
                elif cmd == "finalize_episode":
                    env.finalize_episode(bool(args))
                    result = None
                elif cmd == "shutdown":
                    env.shutdown()
                    child_conn.send(("ok", None))
                    break
                else:
                    raise ValueError(f"unknown cmd {cmd!r}")
                child_conn.send(("ok", result))
            except Exception as e:
                import traceback

                child_conn.send(("err", f"{e}\n{traceback.format_exc()}"))
    finally:
        try:
            child_conn.close()
        except Exception:
            pass


class SubprocEnvHandle:
    """Synchronous pipe wrapper around one env subprocess.

    Usage:
        h.submit_reset(ep_id)
        ...
        h.wait_reset()       # -> {"obs", "instruction", "task_desc"}
        h.submit_execute_chunk(actions)
        ...
        h.wait_execute_chunk()  # -> {"obs", "done", "steps"}
    """

    def __init__(self, cfg, worker_id: int):
        # Use spawn instead of fork because the driver may already hold a CUDA
        # context for the model.
        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(
            target=_subproc_main,
            args=(cfg, worker_id, child_conn),
            daemon=False,
            name=f"infer-subproc-w{worker_id}",
        )
        self._proc.start()
        child_conn.close()
        self._has_pending = False

    def _send(self, cmd: str, args):
        self._parent_conn.send((cmd, args))
        self._has_pending = True

    def _recv(self):
        if not self._has_pending:
            raise RuntimeError("no pending request to wait for")
        status, payload = self._parent_conn.recv()
        self._has_pending = False
        if status == "err":
            raise RuntimeError(f"subproc env error (w={self._proc.name}): {payload}")
        return payload

    def submit_reset(self, ep_id):
        self._send("reset_episode", ep_id)

    def wait_reset(self):
        return self._recv()

    def submit_execute_chunk(self, actions):
        self._send("execute_chunk", np.asarray(actions, dtype=np.float32))

    def wait_execute_chunk(self):
        return self._recv()

    def submit_finalize_episode(self, success: bool):
        self._send("finalize_episode", success)

    def wait_finalize_episode(self):
        return self._recv()

    def finalize_episode(self, success: bool):
        """Blocking convenience wrapper."""
        self.submit_finalize_episode(success)
        return self.wait_finalize_episode()

    def reset_episode(self, ep_id):
        """Blocking convenience wrapper."""
        self.submit_reset(ep_id)
        return self.wait_reset()

    def execute_chunk(self, actions):
        """Blocking convenience wrapper."""
        self.submit_execute_chunk(actions)
        return self.wait_execute_chunk()

    def shutdown(self) -> None:
        try:
            self._send("shutdown", None)
            self._recv()
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2)
