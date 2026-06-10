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
import time

import numpy as np

from wall_x._vendor.harrix.eval_config import EvalConfig
from wall_x._vendor.harrix.drivers.inproc.env_handle import SubprocEnvHandle
from wall_x._vendor.harrix.drivers.inproc.model_handle import DirectModelHandle
from wall_x._vendor.harrix.drivers.job_state import JobState

logger = logging.getLogger(__name__)


def run(cfg: EvalConfig) -> None:
    import wall_x._vendor.harrix.envs  # noqa: F401  trigger env register
    import wall_x._vendor.harrix.adapters  # noqa: F401  trigger adapter register
    from wall_x._vendor.harrix.envs.registry import enumerate_episodes_for
    from wall_x._vendor.harrix.utils.seed import set_seed_everywhere

    # 1) Seed the driver process.
    set_seed_everywhere(cfg.env.seed)

    # 2) Build DirectModelHandle; the model is loaded in the driver process.
    logger.info("Constructing DirectModelHandle in the driver process")
    t_model = time.time()
    model_handle = DirectModelHandle(cfg)
    logger.info("Model loaded in %.1fs", time.time() - t_model)

    # 3) Start env subprocesses.
    logger.info("Starting %s SubprocEnvHandle(s)", cfg.runtime.num_workers)
    env_handles = [
        SubprocEnvHandle(cfg, worker_id=i) for i in range(cfg.runtime.num_workers)
    ]

    # 4) JobState runs in frame-sync mode for lockstep evaluation.
    os.makedirs(cfg.runtime.log_dir, exist_ok=True)
    log_path = os.path.join(cfg.runtime.log_dir, "state.jsonl")
    report_path = os.path.join(cfg.runtime.log_dir, "report.json")
    episodes = enumerate_episodes_for(cfg)
    logger.info("env.type=%r; scheduled %s episodes", cfg.env.type, len(episodes))
    state = JobState(
        episodes,
        log_path,
        batch_sync_mode=True,
        batch_size=cfg.runtime.num_workers,
    )

    # 5) Main loop, one frame at a time.
    t0 = time.time()
    frame_idx = 0
    while not state.is_drained():
        frame_eps = state.claim_frame()
        if not frame_eps:
            # Previous frame still has in-flight episodes.
            time.sleep(0.05)
            continue

        results = _run_frame(frame_eps, model_handle, env_handles, cfg, frame_idx)
        for ep, res in zip(frame_eps, results):
            if "_error" in res:
                state.fail(ep, str(res["_error"]))
            else:
                state.complete(ep, res)
        frame_idx += 1

    elapsed = time.time() - t0
    logger.info("All episodes finished in %.1fs (%.1f min)", elapsed, elapsed / 60)

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

    for h in env_handles:
        h.shutdown()
    model_handle.shutdown()


def _run_frame(
    frame_eps: list,
    model_handle: DirectModelHandle,
    env_handles: list,
    cfg: EvalConfig,
    frame_idx: int,
) -> list[dict]:
    """Run one task-local frame with chunk-level lockstep.

    Active workers are batched together at each chunk boundary. Workers that
    already finished no longer participate in later forwards.
    """
    from wall_x._vendor.harrix.envs.libero_common import encode_raw_obs

    n = len(frame_eps)
    t_frame_start = time.time()

    # ---- a) reset: fan out, then gather ----
    for i in range(n):
        env_handles[i].submit_reset(tuple(frame_eps[i]))
    initials = [env_handles[i].wait_reset() for i in range(n)]

    obs_list = [r["obs"] for r in initials]
    instr_list = [r["instruction"] for r in initials]
    status = [
        {
            "done": False,
            "success": False,
            "steps": 0,
            "task_desc": initials[i].get("task_desc", ""),
        }
        for i in range(n)
    ]

    max_rounds = cfg.env.libero.max_infer_times

    # ---- b) chunk lockstep ----
    for round_idx in range(max_rounds):
        active = [i for i in range(n) if not status[i]["done"]]
        if not active:
            break

        payloads = [
            {
                "observation": encode_raw_obs(obs_list[i]),
                "instruction": instr_list[i],
                "noise": None,
            }
            for i in active
        ]
        chunks = model_handle.predict_batch(payloads)

        # fan-out submit
        for k, i in enumerate(active):
            env_handles[i].submit_execute_chunk(chunks[k])
        # gather
        for k, i in enumerate(active):
            try:
                r = env_handles[i].wait_execute_chunk()
            except Exception as e:
                status[i]["_error"] = str(e)
                status[i]["done"] = True
                continue
            obs_list[i] = r["obs"]
            status[i]["steps"] += r["steps"]
            if r["done"]:
                status[i]["done"] = True
                status[i]["success"] = True

    for i in range(n):
        env_handles[i].submit_finalize_episode(status[i]["success"])
    for i in range(n):
        env_handles[i].wait_finalize_episode()

    elapsed_frame = time.time() - t_frame_start
    logger.info(
        "frame=%s n=%s succ=%s/%s elapsed=%.1fs",
        frame_idx,
        n,
        sum(s["success"] for s in status),
        n,
        elapsed_frame,
    )

    return [
        {
            "success": s["success"],
            "steps": s["steps"],
            "elapsed_sec": round(elapsed_frame / max(1, n), 3),
            "task_desc": s["task_desc"],
            **({"_error": s["_error"]} if "_error" in s else {}),
        }
        for s in status
    ]
