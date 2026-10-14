"""Typed YAML-driven configuration for inference and evaluation.

The driver, model handle, and environment workers share one ``EvalConfig`` so
configuration is parsed once and then passed through explicitly.

YAML schema:
    model:
      checkpoint_path: <required>     # checkpoint directory
      train_config_path: null         # null = <ckpt>/config.yml
      norm_key: libero_all
      state_drop: false               # true = remove discretized state string
      robot_type: desktop             # prompt embodiment for native X2 keys
      cam_names: [face_view, right_wrist_view]
      action_horizon: null            # null = read data.action_horizon_flow
      architecture: qwen2_5           # adapter registry key
      action_mode: flow               # flow / ar / dllm / vqa / subtask
    env:
      type: libero                    # env registry key
      seed: 42
      libero:
        task_suite_name: libero_spatial
        initial_states_path: DEFAULT
        num_trials_per_task: 50
        task_indices: null            # null = all tasks
        max_infer_times: 22
        skip_intermediate_render: true
        force_render_task_indices: null   # e.g. [5] always render that task
        chunk_granular_render_toggle: false
        rebuild_env_per_episode: false
    runtime:
      num_workers: 1
      max_batch_size: 1
      ws_port: 8765
      log_dir: /tmp/wallx_v5_log
      batch_sync_mode: false
    debug:
      deterministic_model: false

Driver flow:
    cfg = load_eval_config(yaml_path)
    cfg = autofill_from_checkpoint(cfg)

Unknown YAML fields are rejected to avoid silent typos.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ModelSection:
    checkpoint_path: str = ""
    train_config_path: Optional[str] = None
    norm_key: str = "libero_all"
    # Runtime robot family used to recover Dataset V2 embodiment metadata for
    # native X2 checkpoints. External datasets ignore this field.
    robot_type: str = "desktop"
    # Dataset V2 normalizer discriminator. Optional for legacy flat keys and
    # checkpoints with a single unambiguous end-effector type.
    end_effector_type: Optional[str] = None
    # Remove the discretized state string from the prompt. Default preserves
    # the historical inference behavior.
    state_drop: bool = False
    cam_names: list = field(
        default_factory=lambda: ["face_view", "right_wrist_view"]
    )
    action_horizon: Optional[int] = None
    # Adapter implementation key in harrix.adapters.registry.ADAPTER_REGISTRY.
    architecture: str = "qwen2_5"
    # Inference algorithm. Each adapter validates its supported subset.
    action_mode: str = "flow"
    num_inference_steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    seed: Optional[int] = None


@dataclass
class LiberoEnvParams:
    """LIBERO-specific env settings used when ``env.type == "libero"``."""

    task_suite_name: str = "libero_spatial"
    initial_states_path: str = "DEFAULT"
    num_trials_per_task: int = 50
    task_indices: Optional[list] = None
    max_infer_times: int = 22
    # Render-skip is enabled by default. Listed task ids always render every
    # simulator step for contact-sensitive tasks.
    skip_intermediate_render: bool = True
    force_render_task_indices: Optional[list] = None
    # If enabled, render observables are toggled only when the chunk boundary
    # actually changes the desired state.
    chunk_granular_render_toggle: bool = False
    # Rebuild the simulator for every episode instead of only on task changes.
    # This is mainly a debugging option because it changes the simulator RNG path.
    rebuild_env_per_episode: bool = False


@dataclass
class EnvSection:
    # Env implementation key in harrix.envs.registry.
    type: str = "libero"
    seed: int = 42
    libero: LiberoEnvParams = field(default_factory=LiberoEnvParams)


@dataclass
class RuntimeSection:
    num_workers: int = 1
    max_batch_size: int = 1
    ws_port: int = 8765
    log_dir: str = "/tmp/wallx_v5_log"
    # Run fixed task-local frames instead of dynamic work stealing. This improves
    # reproducibility at the cost of possible idle workers inside a frame.
    batch_sync_mode: bool = False
    # Public Wall-X evaluation uses the in-process driver. Other drivers may be
    # enabled by downstream/internal integrations.
    driver_mode: str = "in_process"
    # In-process driver only: run each env inline in the driver process
    # (DirectEnvHandle) instead of a spawned subprocess (SubprocEnvHandle).
    inline_env: bool = False


@dataclass
class DebugSection:
    # Enable deterministic torch backend options for variance debugging.
    deterministic_model: bool = False


@dataclass
class EvalConfig:
    model: ModelSection = field(default_factory=ModelSection)
    env: EnvSection = field(default_factory=EnvSection)
    runtime: RuntimeSection = field(default_factory=RuntimeSection)
    debug: DebugSection = field(default_factory=DebugSection)


def _build_dataclass(cls, raw):
    """Build a dataclass from a dict and reject unknown fields."""
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected dict for {cls.__name__}, got {type(raw).__name__}"
        )
    field_names = {f.name for f in dataclasses.fields(cls)}
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    unknown = set(raw.keys()) - field_names
    if unknown:
        raise ValueError(
            f"Unknown field(s) {sorted(unknown)} in {cls.__name__}; "
            f"expected one of {sorted(field_names)}"
        )
    built = {}
    for k, v in raw.items():
        ft = field_types[k]
        if (
            isinstance(ft, type)
            and dataclasses.is_dataclass(ft)
            and isinstance(v, dict)
        ):
            built[k] = _build_dataclass(ft, v)
        else:
            built[k] = v
    return cls(**built)


def load_eval_config(yaml_path: str) -> EvalConfig:
    """Load ``EvalConfig`` from YAML and validate checkpoint_path."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Empty YAML: {yaml_path}")

    cfg = _build_dataclass(EvalConfig, raw)

    if not cfg.model.checkpoint_path:
        raise ValueError(f"model.checkpoint_path is required in {yaml_path}")
    if not _checkpoint_path_exists(cfg.model.checkpoint_path):
        raise FileNotFoundError(
            f"model.checkpoint_path does not exist: {cfg.model.checkpoint_path}"
        )

    return cfg


def _checkpoint_path_exists(path: str) -> bool:
    parts = [p.strip() for p in path.split(",")] if "," in path else [path]
    return all(os.path.exists(p) for p in parts)


def autofill_from_checkpoint(cfg: EvalConfig) -> EvalConfig:
    """Fill omitted model fields from the checkpoint-side train config.

    This resolves ``model.train_config_path`` and ``model.action_horizon``.
    """
    if cfg.model.train_config_path is None:
        if "," in cfg.model.checkpoint_path or not os.path.isdir(
            cfg.model.checkpoint_path
        ):
            raise FileNotFoundError(
                "model.train_config_path is required when checkpoint_path is not a checkpoint directory"
            )
        for fname in ("config.yml", "config.yaml"):
            cand = os.path.join(cfg.model.checkpoint_path, fname)
            if os.path.exists(cand):
                cfg.model.train_config_path = cand
                break
        else:
            raise FileNotFoundError(
                f"No config.yml/config.yaml in {cfg.model.checkpoint_path}; "
                "set model.train_config_path explicitly in YAML"
            )

    if cfg.model.action_horizon is None:
        # Typed task contract: the qwen flow path emits
        # task.action_horizon_flow steps per chunk.
        from wall_x.config.loader import load_train_config_from_checkpoint

        train_cfg = load_train_config_from_checkpoint(
            cfg.model.checkpoint_path, cfg.model.train_config_path
        )
        cfg.model.action_horizon = int(train_cfg.task.action_horizon_flow)

    return cfg
