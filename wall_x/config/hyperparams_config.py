"""Training hyperparameter config dataclasses."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .registry import register_optimizer_config, register_scheduler_config


@dataclass
class LRGroupConfig:
    """Named optimizer LR group matched by parameter-name substrings."""

    name: str
    lr: float
    include: List[str] = field(default_factory=list)
    fail_on_empty: bool = True


@dataclass
class OptimizerConfig:
    """Base optimizer config. ``optimizer_type`` selects a registered subclass."""

    optimizer_type: str = "adamw"
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    enable_grad_clip: bool = True
    # Named parameter groups with independent learning rates. Unmatched
    # trainable parameters remain in the base group using ``learning_rate``.
    lr_groups: Optional[List[LRGroupConfig]] = None
    # Optional action-expert LR split.
    train_action_expert_only: bool = False
    action_expert_learning_rate: Optional[float] = None
    action_lr_keywords: Optional[List[str]] = None


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
    """Constant learning rate with no decay."""

    scheduler_type: str = "constant"


@register_scheduler_config("step")
@dataclass
class StepSchedulerConfig(SchedulerConfig):
    """Step decay scheduler."""

    scheduler_type: str = "step"
    step_size: int = 10000
    gamma: float = 0.1


@dataclass
class TrainHyperParams:
    num_epoch: int = 1
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 1
    seed: int = 42
    optimizer: OptimizerConfig = field(default_factory=AdamWConfig)
    scheduler: SchedulerConfig = field(default_factory=CosineSchedulerConfig)


__all__ = [
    "AdamWConfig",
    "ConstantSchedulerConfig",
    "CosineSchedulerConfig",
    "DMuonConfig",
    "LRGroupConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "StepSchedulerConfig",
    "TrainHyperParams",
]
