"""Infrastructure config: distributed runtime, logging, checkpoints, debug, and serving."""

from dataclasses import dataclass, field
from dataclasses import fields as _dc_fields
from dataclasses import is_dataclass as _dc_is_dataclass
from typing import Optional


@dataclass
class ModelServiceConfig:
    """Model-service push config (used by CkptPusher — trainer pushes new actor
    weights back to serving)."""

    model_service_url: str = ""
    push_max_retries: int = 3


@dataclass
class RLDataServiceConfig:
    """RLDataSource ctor kwargs for RL off-policy ingest (see ADR-0007).

    Collects RLDataSource's 5 kwargs under the yaml ``cfg.serving.data`` block so
    ``train_rl.py::_build_trainer`` can pass them straight to ``RLDataSource()``.
    (Episode physical location comes from the data service's discover ``data_uri``,
    not configured here — the trainer does not couple to storage layout.)"""

    data_service_url: str = ""
    tag: str = ""
    trainer_encoder_version: str = ""
    trainer_data_pipeline_version: str = ""


@dataclass
class ServingConfig:
    """Unified config for the two serving counterparts the trainer talks to.

    ``model`` = model service (trainer → serving actor-weight push); ``data`` =
    data service (robot → DS upload, trainer ← DS ingest). Both are serving
    components the training process talks to over HTTP, so they are grouped under
    ``serving:``."""

    model: ModelServiceConfig = field(default_factory=ModelServiceConfig)
    data: RLDataServiceConfig = field(default_factory=RLDataServiceConfig)


@dataclass
class ShardedBackendConfig:
    """Shared 1D/2D sharding layout options."""

    sharding_strategy: str = "full_shard"
    hsdp_replicate_size: Optional[int] = None

    @property
    def reshard_after_forward(self) -> bool:
        return self.sharding_strategy not in ("shard_grad_op", "_hybrid_shard_zero2")


@dataclass
class FSDPBackendConfig(ShardedBackendConfig):
    """PyTorch FSDP2-specific distributed backend options."""

    backward_prefetch: str = "backward_pre"
    cpu_offload: bool = False
    use_orig_params: bool = True
    limit_all_gathers: bool = True
    forward_prefetch: bool = False
    sync_module_states: bool = True
    save_policy: str = "full"
    reduce_dtype: str = "bf16"


@dataclass
class DMuonBackendConfig(ShardedBackendConfig):
    """Pure DMuon distributed backend options."""

    sharding_strategy: str = "shard_grad_op"


@dataclass
class DDPBackendConfig:
    """PyTorch DistributedDataParallel backend options."""

    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25


_RECOMPUTE_MODES = ("full", "attn_only", "none")


@dataclass
class RecomputeOverride:
    """One decoder recompute exception.

    ``layers``: explicit layer indices, or the string "gdn" (every
    linear-attention layer). ``mode``: how much of each listed layer is
    recomputed -- "full" (whole layer), "attn_only" (attention section
    recomputes, MLP activations stay resident), "none" (whole layer keeps
    its activations).
    """

    layers: object = None
    mode: str = "full"

    def __post_init__(self):
        if self.mode not in _RECOMPUTE_MODES:
            raise ValueError(
                f"recompute override mode must be one of {_RECOMPUTE_MODES}, "
                f"got {self.mode!r}"
            )
        ok = self.layers == "gdn" or (
            isinstance(self.layers, list)
            and len(self.layers) > 0
            and all(isinstance(i, int) and not isinstance(i, bool) for i in self.layers)
        )
        if not ok:
            raise ValueError(
                "recompute override layers must be a non-empty list of ints "
                f'or the string "gdn", got {self.layers!r}'
            )


@dataclass
class DecoderRecomputeConfig:
    default: str = "full"
    overrides: list = field(default_factory=list)

    def __post_init__(self):
        if self.default not in _RECOMPUTE_MODES:
            raise ValueError(
                f"recompute.decoder.default must be one of {_RECOMPUTE_MODES}, "
                f"got {self.default!r}"
            )
        hydrated = []
        for item in self.overrides:
            if isinstance(item, RecomputeOverride):
                hydrated.append(item)
            elif isinstance(item, dict):
                unknown = set(item) - {"layers", "mode"}
                if unknown:
                    raise ValueError(
                        f"recompute.decoder.overrides entry has unknown keys "
                        f"{sorted(unknown)} (expected layers/mode)"
                    )
                hydrated.append(RecomputeOverride(**item))
            else:
                raise ValueError(
                    f"recompute.decoder.overrides entries must be mappings, "
                    f"got {item!r}"
                )
        self.overrides = hydrated


@dataclass
class RecomputeConfig:
    """Selective activation recompute, config-driven (no env vars).

    ``enable: true`` puts the model in recompute-by-default (memory-minimal);
    model-specific subclasses add the exception sections that buy time back
    with memory, measured as ms saved per GB spent. Each
    model family maintains its own subclass; the loader picks it via
    ``RECOMPUTE_CONFIG_BY_MODEL_TYPE``. Unknown yaml keys are rejected
    loudly: a typo here silently changing the memory plan is exactly the
    failure mode this replaces (WALLX_CKPT_* env vars).
    """

    enable: bool = False

    @classmethod
    def build(cls, raw, *, where: str):
        raw = dict(raw or {})
        field_names = {f.name for f in _dc_fields(cls)}
        unknown = set(raw) - field_names
        if unknown:
            raise ValueError(
                f"{where}: unknown recompute keys {sorted(unknown)} for "
                f"{cls.__name__} (known: {sorted(field_names)})"
            )
        kwargs = {}
        for f in _dc_fields(cls):
            if f.name not in raw:
                continue
            value = raw[f.name]
            if isinstance(f.type, type) and _dc_is_dataclass(f.type):
                if not isinstance(value, dict):
                    raise ValueError(f"{where}: recompute.{f.name} must be a mapping")
                sub_unknown = set(value) - {sf.name for sf in _dc_fields(f.type)}
                if sub_unknown:
                    raise ValueError(
                        f"{where}: unknown recompute.{f.name} keys "
                        f"{sorted(sub_unknown)}"
                    )
                kwargs[f.name] = f.type(**value)
            else:
                kwargs[f.name] = value
        return cls(**kwargs)


@dataclass
class Qwen35RecomputeConfig(RecomputeConfig):
    """qwen3_5 family: decoder-layer granularity (full/attn_only/none)."""

    decoder: DecoderRecomputeConfig = field(default_factory=DecoderRecomputeConfig)

    def __post_init__(self):
        if not self.enable and (
            self.decoder.overrides or self.decoder.default != "full"
        ):
            raise ValueError(
                "recompute.decoder configured but recompute.enable is false"
            )


# Each model family maintains its own recompute schema here; absent entries
# get the base (enable-only) config.
RECOMPUTE_CONFIG_BY_MODEL_TYPE = {
    "qwen3_5": Qwen35RecomputeConfig,
}


@dataclass
class DistributedConfig:
    # Public distributed backend. DMuon means pure DMuon; FSDP2+DMuon dedicated
    # params is intentionally not a separate backend. DDP is kept as a
    # lightweight debug/small-model backend.
    backend: str = "fsdp2"
    fsdp: FSDPBackendConfig = field(default_factory=FSDPBackendConfig)
    dmuon: DMuonBackendConfig = field(default_factory=DMuonBackendConfig)
    ddp: DDPBackendConfig = field(default_factory=DDPBackendConfig)
    # Mixed precision
    use_mixed_precision: bool = True
    bf16: bool = True
    use_amp: bool = False
    use_gradient_checkpointing: bool = False
    use_gradient_checkpointing_offload: bool = False
    # Partial recompute: number of leading transformer layers to checkpoint.
    # None = checkpoint all layers (legacy behavior when use_gradient_checkpointing=True).
    # N in [0, num_layers] = checkpoint only the first N layers, keep the rest's
    # activations resident in HBM. Clamped to [0, num_layers] at apply time.
    # Ignored when use_gradient_checkpointing=False.
    gradient_checkpoint_num_layers: Optional[int] = None
    use_selective_recompute: bool = False
    # Selective recompute plan; concrete subclass chosen per model_type
    # by the loader (RECOMPUTE_CONFIG_BY_MODEL_TYPE).
    recompute: RecomputeConfig = field(default_factory=RecomputeConfig)
    # DDP fallback
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25

    @property
    def use_fsdp(self) -> bool:
        """Compatibility property for code paths that require FSDP2 wrappers."""

        return self.backend == "fsdp2"


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
    enable_validation: bool = True
    gc_interval_steps: int = 1000
    ignore_until_interval: int = 0
    # 前几步 step_time / samples_per_sec 不写 wandb：cuDNN benchmark warm-up、
    # dataloader 首次预取、lazy init 让 step 0~N 远比稳态慢，写进去会把 y 轴拉爆。
    # 控制台 line 仍然显示原值，便于现场看到 warm-up 发生。
    perf_log_skip_steps: int = 5
    param_norm_interval: int = 100
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
    # Restore optimizer and LR-scheduler state when resuming from a checkpoint
    # directory. Set to False for a weight/data/step resume with a freshly
    # initialized optimizer and scheduler.
    resume_optimizer_state: bool = True
    # Extra optimizer steps worth of per-rank sampler data to skip after
    # restoring an in-epoch dataset cursor. Model/optimizer/global_step are
    # unchanged; only the resumed data cursor moves forward.
    resume_data_skip_steps: int = 0
    # Extra per-rank sampler indices to skip after restoring an in-epoch
    # dataset cursor. This is the cursor's native unit and supports variable
    # batch sizes. Mutually exclusive with resume_data_skip_steps.
    resume_data_skip_samples_per_rank: int = 0
    # When True, passes try_harder=True to reshape_compatible_state_dict so that
    # action-preprocessor weights are padded/sliced to match the current model shape.
    try_harder: bool = False
    # CkptPusher: path to the JSON log of already-pushed weight versions.
    weight_version_persistence_path: str = ""

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
    # Number of profile cycles to record. Default 1 = capture once and stop.
    # torch.profiler.schedule's own default is 0 = repeat forever, which was
    # the silent behavior when this field was missing (the yaml key was being
    # dropped by the typed loader). Each cycle dumps a ~500MB trace per worker
    # and adds a multi-second stall, so 1 is the right default for ad-hoc runs.
    profile_repeat: int = 1
    # When True, traces are gzip-compressed by torch.profiler's
    # tensorboard_trace_handler. Output files become *.pt.trace.json.gz
    # (~10× smaller). Perfetto / chrome://tracing accept .gz directly.
    profile_use_gzip: bool = False
    # torch.profiler.profile() kwargs. Defaults tuned for "post-hoc analysis
    # via visualize_wallx" rather than the legacy "tensorboard + py-spy stack"
    # workflow:
    #   - with_stack=False saves a lot of trace size and CPU overhead BUT it
    #     drops the `nn.Module: <ClassName>` annotations (which `with_stack`
    #     inferred indirectly from python frames). wall-x compensates by
    #     emitting explicit record_function zones — see wall_x/utils/profile_zones.py.
    #   - with_modules=True has no effect in eager mode (PyTorch limits it
    #     to TorchScript), so the explicit-zone approach is the only path.
    #   - record_shapes keeps GEMM input shapes (visualize uses it for FLOPS)
    #   - profile_memory off by default — overlaps with enable_memory_snapshot
    profile_with_stack: bool = False
    profile_with_modules: bool = False  # no-op in eager mode — keep off
    profile_record_shapes: bool = True
    profile_memory: bool = False
    # When True, only rank 0 records a trace. Avoids dumping N copies of the
    # same timeline on multi-GPU runs (16-rank × ~80MB each = 1.3GB per cycle).
    profile_only_rank0: bool = True
    # CUDA memory snapshot. When True, rank 0 records the allocator history
    # starting at step 0 and dumps a .pickle at step `memory_snapshot_until_step`.
    # Output drops into `memory_snapshot_dir/memory_snapshot.pickle`; open it at
    # https://pytorch.org/memory_viz to see per-allocation timeline + peak +
    # which Python frame owned each block.
    enable_memory_snapshot: bool = False
    memory_snapshot_dir: str = "./memory_snapshot"
    # Recording starts in trainer __init__ (covers model load + FSDP wrap +
    # optimizer creation); dump fires right after global_step == until_step.
    # 3 is enough to see steady-state per-step alloc pattern without making
    # the .pickle huge.
    memory_snapshot_until_step: int = 3
    # Hard step cap for probe/diagnostic jobs (0 = unlimited). Jobs whose only
    # purpose is a snapshot/profile MUST set this so they self-terminate
    # instead of relying on someone remembering to stop them.
    max_steps: int = 0
    memory_snapshot_max_entries: int = 100000
    show_time_details: bool = False
    visualize_sample: bool = False
    save_debug_batch_path: Optional[str] = None
    nvtx: bool = False
    # Formula MFU uses the local FLOPs estimate and measured step time.
    enable_mfu: bool = False
    # Optional FLOPs profiling runs an extra forward at step 0.
    enable_mfu_profile: bool = False
