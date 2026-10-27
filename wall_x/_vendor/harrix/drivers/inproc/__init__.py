"""In-process driver: DirectModelHandle + N x SubprocEnvHandle.

The model lives in the driver process while simulator envs live in subprocesses.
Evaluation proceeds in chunk-level lockstep:
    1. Seed the driver process and construct the model handle.
    2. Start one subprocess env handle per worker.
    3. Claim a task-local frame from JobState.
    4. Reset envs, batch active observations, run model.predict_batch, then
       fan out action chunks to env subprocesses.
    5. Complete all episodes in the frame and move to the next frame.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

from wall_x._vendor.harrix.eval_config import EvalConfig
from wall_x._vendor.harrix.drivers.inproc.env_handle import DirectEnvHandle, SubprocEnvHandle
from wall_x._vendor.harrix.drivers.inproc.model_handle import DirectModelHandle
from wall_x._vendor.harrix.drivers.job_state import JobState

logger = logging.getLogger(__name__)


def run(cfg: EvalConfig) -> None:
    from wall_x._vendor.harrix.utils.seed import set_seed_everywhere

    cfg.runtime.log_dir = os.path.abspath(cfg.runtime.log_dir)

    import wall_x._vendor.harrix.envs  # noqa: F401  trigger env register
    import wall_x._vendor.harrix.adapters  # noqa: F401  trigger adapter register
    from wall_x._vendor.harrix.envs.registry import enumerate_episodes_for

    # 1) Seed the driver process.
    set_seed_everywhere(cfg.env.seed)

    # 2) Build DirectModelHandle; the model is loaded in the driver process.
    logger.info("Constructing DirectModelHandle in the driver process")
    t_model = time.time()
    model_handle = DirectModelHandle(cfg)
    logger.info("Model loaded in %.1fs", time.time() - t_model)

    # 3) Start env handles: inline in the driver process (DirectEnvHandle) or in
    #    spawned subprocesses (SubprocEnvHandle).
    env_handle_cls = DirectEnvHandle if cfg.runtime.inline_env else SubprocEnvHandle
    logger.info(
        "Starting %s %s(s)", cfg.runtime.num_workers, env_handle_cls.__name__
    )
    env_handles = [
        env_handle_cls(cfg, worker_id=i) for i in range(cfg.runtime.num_workers)
    ]
    completed_ok = False

    # 4) JobState in FIFO mode: a worker claims the next pending episode the
    #    instant it finishes, so the inference batch stays full (continuous
    #    batching) instead of collapsing onto the slowest episode of a lockstep
    #    frame. This is what makes batched inference actually pay off.
    os.makedirs(cfg.runtime.log_dir, exist_ok=True)
    log_path = os.path.join(cfg.runtime.log_dir, "state.jsonl")
    report_path = os.path.join(cfg.runtime.log_dir, "report.json")
    episodes = enumerate_episodes_for(cfg)
    logger.info("env.type=%r; scheduled %s episodes", cfg.env.type, len(episodes))
    state = JobState(episodes, log_path, batch_sync_mode=False)

    try:
        # 5) Continuous-batching main loop.
        t0 = time.time()
        _run_continuous(state, model_handle, env_handles, cfg)

        elapsed = time.time() - t0
        logger.info(
            "All episodes finished in %.1fs (%.1f min)", elapsed, elapsed / 60
        )

        state.dump_final(report_path)
        with open(report_path) as f:
            report = json.load(f)
        overall = report["overall"]
        logger.info(
            "attempted=%s, successes=%s, success_rate=%.2f%%, failed=%s",
            overall["attempted"],
            overall["successes"],
            overall["success_rate"] * 100,
            overall["failed"],
        )
        logger.info("Report: %s", report_path)
        logger.info("State log: %s", log_path)
        # Exit non-zero when any episode errored (crash/infra), so an
        # orchestrator can distinguish a degraded shard from a genuinely low
        # score. Plain task failures (completed, success=False) keep exit 0.
        exit_code = 0 if overall["failed"] == 0 else 1
        completed_ok = True
    finally:
        for h in env_handles:
            h.shutdown()
        model_handle.shutdown()


def _run_continuous(
    state: JobState,
    model_handle: DirectModelHandle,
    env_handles: list,
    cfg: EvalConfig,
) -> None:
    """Continuous-batching scheduler.

    Every env worker is kept busy: the instant a worker finishes its episode it
    claims the next pending one, so the inference batch stays ~num_workers wide
    until the queue drains. This is the key difference from the old lockstep
    frame path, whose batch collapsed to 1 whenever one episode outran the rest.

    A just-finished worker's reset (submit_reset) is issued *before* the active
    batch and its completion is collected *after* the batch's predict+execute, so
    the expensive expert-filter reset overlaps inference instead of stalling it.

    Requires the env to accept episodes of different tasks across resets when the
    schedule mixes tasks (see RoboTwinRobotEnv multi-task support).
    """
    from wall_x._vendor.harrix.envs.registry import encode_observation_for, max_infer_rounds_for

    n = len(env_handles)
    max_rounds = max_infer_rounds_for(cfg)

    def _blank():
        return {
            "ep": None, "obs": None, "instr": "", "steps": 0, "rounds": 0,
            "task_desc": "", "phase": "idle",
        }

    W = [_blank() for _ in range(n)]
    t_last = time.time()
    done_count = 0

    def _finish(i: int, success: bool, error: str | None = None) -> None:
        nonlocal done_count
        w = W[i]
        res = {"success": bool(success), "steps": w["steps"], "task_desc": w["task_desc"]}
        if error is not None:
            res["_error"] = error
            state.fail(w["ep"], str(error))
        else:
            state.complete(w["ep"], res)
        W[i] = _blank()
        done_count += 1

    while True:
        # (a) Refill idle workers from the queue; submit resets async so they
        #     overlap the active batch below.
        for i in range(n):
            if W[i]["phase"] != "idle":
                continue
            ep = state.claim(i)
            if ep is None:
                continue
            W[i]["ep"] = ep
            try:
                env_handles[i].submit_reset(tuple(ep))
                W[i]["phase"] = "resetting"
            except Exception as e:
                _finish(i, False, str(e))

        # (b) Batch all active workers and advance each one chunk.
        active = [i for i in range(n) if W[i]["phase"] == "active"]
        if active:
            payloads = [
                {
                    "observation": encode_observation_for(cfg, W[i]["obs"]),
                    "instruction": W[i]["instr"],
                    "noise": None,
                }
                for i in active
            ]
            chunks = model_handle.predict_batch(payloads)
            for k, i in enumerate(active):
                env_handles[i].submit_execute_chunk(chunks[k])
            for k, i in enumerate(active):
                try:
                    r = env_handles[i].wait_execute_chunk()
                except Exception as e:
                    _finish(i, False, str(e))
                    continue
                W[i]["obs"] = r["obs"]
                W[i]["steps"] += r["steps"]
                W[i]["rounds"] += 1
                if r["done"]:
                    _finish(i, True)
                elif W[i]["rounds"] >= max_rounds:
                    _finish(i, False)  # hit the chunk cap: unsuccessful

        # (c) Collect the resets from (a); they overlapped the batch in (b).
        for i in range(n):
            if W[i]["phase"] != "resetting":
                continue
            try:
                init = env_handles[i].wait_reset()
                W[i].update(
                    obs=init["obs"], instr=init["instruction"],
                    task_desc=init.get("task_desc", ""),
                    phase="active", steps=0, rounds=0,
                )
            except Exception as e:
                _finish(i, False, str(e))

        now = time.time()
        if now - t_last >= 30.0:
            in_flight = sum(1 for i in range(n) if W[i]["phase"] != "idle")
            logger.info(
                "continuous: batch=%s in_flight=%s done=%s %s",
                len(active), in_flight, done_count, state.progress(),
            )
            t_last = now

        # (d) Done when nothing is in flight and the queue is drained.
        if not any(W[i]["phase"] != "idle" for i in range(n)) and state.is_drained():
            break
