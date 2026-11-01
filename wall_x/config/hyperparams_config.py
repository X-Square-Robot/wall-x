"""Training hyperparameter config dataclasses."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .registry import register_optimizer_config, register_scheduler_config


@dataclass
class LRGroupConfig:
    """Named optimizer group matched by parameter-name substrings.

    Despite the legacy ``LRGroupConfig`` name, a group may override learning
    rate, weight decay, or both.  Leaving a value unset preserves the
    optimizer's base/route-specific default.
    """

    name: str
    lr: Optional[float] = None
    include: List[str] = field(default_factory=list)
    fail_on_empty: bool = True
    # Keep this after the legacy fields so positional construction of
    # LRGroupConfig(name, lr, include, fail_on_empty) remains compatible.
    weight_decay: Optional[float] = None


@dataclass
class OptimizerConfig:
    """Base optimizer config. ``optimizer_type`` selects a registered subclass."""

    optimizer_type: str = "adamw"
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    enable_grad_clip: bool = True
    # Named parameter groups with optional learning-rate and weight-decay
    # overrides. Unmatched trainable parameters remain in the base group.
    lr_groups: Optional[List[LRGroupConfig]] = None
    # Optional action-expert LR split.
    train_action_expert_only: bool = False
    action_expert_learning_rate: Optional[float] = None
    action_lr_keywords: Optional[List[str]] = None

    @property
    def use_dedicated_params(self) -> bool:
        return False


@register_optimizer_config("adamw")
@dataclass
class AdamWConfig(OptimizerConfig):
    """AdamW optimizer config."""

    optimizer_type: str = "adamw"
    betas: Tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 1e-8
    eps: float = 1e-8
    fused: bool = True
    foreach: Optional[bool] = None


@register_optimizer_config("dmuon")
@dataclass
class DMuonConfig(OptimizerConfig):
    """DMuon optimizer config."""

    optimizer_type: str = "dmuon"
    muon_lr: float = 0.02
    momentum: float = 0.95
    ns_steps: int = 5
    muon_weight_decay: float = 0.0
    adamw_lr: float = 1e-3
    adamw_betas: Tuple[float, float] = (0.9, 0.999)
    adamw_weight_decay: float = 0.01
    adamw_eps: float = 1e-8
    ns_backend: str = "gram"
    ns_coefficients: str = "default"
    nesterov: bool = True
    owner_strategy: str = "lpt"
    owner_cost_model: str = "optimizer"
    owner_pack_small_params: bool = True
    hsdp_column_balance: bool = True
    record_step_profile: bool = False
    first_step_progress_log_limit: int = 8
    replicate_async: bool = True
    group_prepare_ahead: bool = True
    forward_prefetch_depth: int = 1

    @property
    def use_dedicated_params(self) -> bool:
        return True


@register_optimizer_config("naive_fsdp2_muon")
@dataclass
class NaiveFSDP2MuonConfig(DMuonConfig):
    """Temporary FSDP2 baseline: every rank runs full-matrix Muon locally."""

    optimizer_type: str = "naive_fsdp2_muon"

    @property
    def use_dedicated_params(self) -> bool:
        return False


@dataclass
class SchedulerConfig:
    """Base scheduler config. ``scheduler_type`` selects a registered subclass."""

    scheduler_type: str = "cosine"
    # Optional training-step cap. When > 0, the trainer sets
    # loss_guard_should_stop=True once global_step >= num_training_steps,
    # regardless of scheduler type. Cosine reads it for its own decay
    # horizon; constant / step schedulers use it only for the stop signal.
    num_training_steps: int = 0


@register_scheduler_config("cosine")
@dataclass
class CosineSchedulerConfig(SchedulerConfig):
    """Cosine annealing with warmup."""

    scheduler_type: str = "cosine"
    num_warmup_steps: int = 0
    num_training_steps: int = 0
    min_lr: Optional[float] = None  # None means 0.1 * learning_rate at runtime.


@register_scheduler_config("constant")
@dataclass
class ConstantSchedulerConfig(SchedulerConfig):
    """Constant learning rate, optionally after linear warmup."""

    scheduler_type: str = "constant"
    num_warmup_steps: int = 0


@register_scheduler_config("step")
@dataclass
class StepSchedulerConfig(SchedulerConfig):
    """Step decay scheduler."""

    scheduler_type: str = "step"
    step_size: int = 10000
    gamma: float = 0.1


@dataclass
class TrainHyperParams:
    # 训练算法判别器（依附于它所多态的 hyperparams section，与 model_type↔model 对称）：
    # 选 TrainHyperParams 子类 + DDP trainer 路径。
    #   "sl"(默认)=监督学习（model_type=rlt 时为 encoder warmup）; "td3"=off-policy TD3。
    # loader._resolve_hyperparams_cls 按 (algorithm, model_type) 惰性路由到 RL 子类。
    algorithm: str = "sl"
    num_epoch: int = 1
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 1
    seed: int = 42
    # opt-in 严格确定性：调 torch.use_deterministic_algorithms 让 same-seed +
    # same-yaml 两次 run 的 step 0 forward byte-equal。caller 必须
    # 同步在 launch 脚本里 export CUBLAS_WORKSPACE_CONFIG=:4096:8（库代码不
    # silently setdefault env，避免 fork 出来的子进程 env 难追溯）。warn_only
    # 内置 True：bf16 训练里总有几个 op 没确定实现，warn 比 crash 实用。
    # default False 保兼容（公共 utility，部分 model 可能含非 deterministic
    # op，强开会抛 RuntimeError）。
    # 这是确定性的**唯一开关**：除 torch.use_deterministic_algorithms 外，还派生
    # model_config.attn_deterministic（qwen2_5 / qwen3_5 的 attention
    # 内核）——旧的独立 ``attn_deterministic`` yaml 字段已下线（MR !556 review）。
    deterministic_algorithms: bool = False
    optimizer: OptimizerConfig = field(default_factory=AdamWConfig)
    scheduler: SchedulerConfig = field(default_factory=CosineSchedulerConfig)


__all__ = [
    "AdamWConfig",
    "ConstantSchedulerConfig",
    "CosineSchedulerConfig",
    "DMuonConfig",
    "LRGroupConfig",
    "NaiveFSDP2MuonConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "StepSchedulerConfig",
    "TrainHyperParams",
]
