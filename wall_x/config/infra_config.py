"""Infrastructure config: distributed runtime, logging, checkpoints, and debug."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DistributedConfig:
    # FSDP
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "full_shard"
    fsdp_backward_prefetch: str = "backward_pre"
    fsdp_cpu_offload: bool = False
    fsdp_use_orig_params: bool = True
    fsdp_limit_all_gathers: bool = True
    fsdp_forward_prefetch: bool = False
    fsdp_sync_module_states: bool = True
    fsdp_save_policy: str = "full"
    fsdp_hsdp_replicate_size: Optional[int] = None
    # Mixed precision
    use_mixed_precision: bool = True
    bf16: bool = True
    fsdp_reduce_dtype: str = "bf16"
    use_amp: bool = False
    use_gradient_checkpointing: bool = False
    use_gradient_checkpointing_offload: bool = False
    use_selective_recompute: bool = False
    # DDP fallback
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25


@dataclass
class LoggingConfig:
    log_name: str = "exp"
    log_project: str = "wallx"
    log_entity: Optional[str] = None
    use_wandb: bool = True
    wandb_offline: bool = False
    log_interval: int = 1
    save_interval: int = 1000
    val_interval: int = 4000
    epoch_save_interval: int = 1
    gc_interval_steps: int = 1000
    ignore_until_interval: int = 0
    # Rolling-window for smoothing per-step training metrics displayed on the
    # console (and reused by tqdm). 1 = raw per-step (historical behavior).
    # 10 = DZ-style 10-step rolling average — diffusion losses are dominated
    # by timestep-sampling noise per step; the smoothing only changes display,
    # not training. Independent of log_interval (which buffers for wandb).
    loss_log_smooth_window: int = 1


@dataclass
class CheckpointConfig:
    """Checkpoint save and resume options.

    ``resume_from`` is shorthand for setting every component-specific resume
    path to the same checkpoint. Component-specific fields take precedence.
    """

    save_path: str = "./ckpt"
    validate_first: bool = False
    # Shorthand path for all components.
    resume_from: Optional[str] = None
    # Component-specific resume paths.
    resume_model: Optional[str] = None
    resume_optimizer: Optional[str] = None
    resume_scheduler: Optional[str] = None
    resume_ema: Optional[str] = None
    resume_rng: Optional[str] = None
    resume_data: Optional[str] = None
    resume_step: Optional[str] = None

    def get_resume_path(self, component: str) -> Optional[str]:
        """Return the resume path for one checkpoint component."""
        specific = getattr(self, f"resume_{component}", None)
        if specific is not None:
            return specific
        return self.resume_from


@dataclass
class DebugConfig:
    profile: bool = False
    profile_save_path: str = "./profile"
    profile_wait_iters: int = 1
    profile_warmup_iters: int = 1
    profile_active_iters: int = 3
    show_time_details: bool = False
    visualize_sample: bool = False
    save_debug_batch_path: Optional[str] = None
    nvtx: bool = False
    # Formula MFU uses the local FLOPs estimate and measured step time.
    enable_mfu: bool = False
    # Optional FLOPs profiling runs an extra forward at step 0.
    enable_mfu_profile: bool = False
