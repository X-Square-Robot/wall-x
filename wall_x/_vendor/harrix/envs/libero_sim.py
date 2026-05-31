"""LIBERO benchmark and robosuite engine helpers.

This module may import robosuite and LIBERO. Adapter-side code should use
``libero_common.py`` instead, which only depends on NumPy.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import numpy as np
from robosuite.wrappers import VisualizationWrapper

logger = logging.getLogger(__name__)


# ============================================================
# One-time side effect: auto-create ~/.libero/config.yaml so importing LIBERO
# does not trigger an interactive prompt.
# ============================================================


def _ensure_libero_config() -> None:
    import yaml as _yaml

    libero_config_path = os.environ.get(
        "LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")
    )
    config_file = os.path.join(libero_config_path, "config.yaml")

    if not os.path.exists(config_file):
        os.makedirs(libero_config_path, exist_ok=True)
        import libero.libero as _libero_pkg

        benchmark_root = os.path.dirname(os.path.abspath(_libero_pkg.__file__))
        default_paths = {
            "benchmark_root": benchmark_root,
            "bddl_files": os.path.join(benchmark_root, "./bddl_files"),
            "init_states": os.path.join(benchmark_root, "./init_files"),
            "datasets": os.path.join(benchmark_root, "../datasets"),
            "assets": os.path.join(benchmark_root, "./assets"),
        }
        with open(config_file, "w") as f:
            _yaml.dump(default_paths, f)
        logger.info("Auto-created LIBERO config: %s", config_file)


_ensure_libero_config()


# ============================================================
# task-suite entry point
# ============================================================


def get_task_suite(task_suite_name: str):
    """Load a LIBERO task suite."""
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[task_suite_name]()


# ============================================================
# actions
# ============================================================


def get_libero_dummy_action() -> list[float]:
    """Return the 7-dof dummy action used for episode warmup."""
    return [0, 0, 0, 0, 0, 0, -1]


# ============================================================
# robosuite engine factory
# ============================================================


def create_libero_engine(
    task_id: int,
    task_suite_name: str,
    resolution: int = 256,
    seed: int = 7,
) -> Any:
    """Construct one LIBERO robosuite engine."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = get_task_suite(task_suite_name)
    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    # The seed still affects object poses even when an initial state is fixed.
    env.seed(seed)
    env.env = VisualizationWrapper(env.env)
    env.env.set_visualization_setting(setting="grippers", visible=False)
    return env


# ============================================================
# task metadata / initial states
# ============================================================


def load_initial_states(initial_states_path: str) -> Optional[dict]:
    """Load custom initial states, or return None for suite defaults."""
    if initial_states_path == "DEFAULT":
        return None
    with open(initial_states_path, "r") as f:
        return json.load(f)


def resolve_task_info(task_suite, task_id: int) -> tuple[str, Any]:
    """Return ``(task_desc, default_initial_states)`` for one task id."""
    num_tasks = task_suite.n_tasks
    if task_id < 0 or task_id >= num_tasks:
        raise ValueError(f"invalid task_id={task_id}, num_tasks={num_tasks}")
    task = task_suite.get_task(task_id)
    return task.language, task_suite.get_task_init_states(task_id)


def pick_initial_state(
    initial_states_path: str,
    custom_initial_states: Optional[dict],
    task_desc: str,
    default_states: Any,
    episode_idx: int,
) -> np.ndarray:
    """Pick one initial state from suite defaults or a custom states file."""
    if initial_states_path == "DEFAULT":
        if default_states is None:
            raise ValueError("default states missing for DEFAULT mode")
        return default_states[episode_idx]

    if custom_initial_states is None:
        raise ValueError(f"custom initial states not loaded for {initial_states_path}")
    key = task_desc.replace(" ", "_")
    ep_key = f"demo_{episode_idx}"
    record = custom_initial_states[key][ep_key]
    if not record["success"]:
        raise ValueError(f"expert demo failed for {ep_key}")
    return np.array(record["initial_state"])


def get_instruction(task_desc: str) -> str:
    """Return the instruction text for a LIBERO task description."""
    return task_desc


# ============================================================
# render-skip: directly assign obs._enabled to avoid set_enabled() side effects.
# ============================================================


def find_image_observables(env) -> list:
    """Find image observables along the env.env wrapper chain."""
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "_observables") and isinstance(cur._observables, dict):
            return [
                obs
                for obs in cur._observables.values()
                if getattr(obs, "modality", None) == "image"
            ]
        cur = getattr(cur, "env", None)
    return []


def set_render_enabled(image_obs_list, enabled: bool) -> None:
    for obs in image_obs_list:
        obs._enabled = enabled
