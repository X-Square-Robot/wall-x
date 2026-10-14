"""Typed training configuration system."""

from .data_config import DataConfig, LeRobotDataConfig
from .hyperparams_config import (
    AdamWConfig,
    ConstantSchedulerConfig,
    CosineSchedulerConfig,
    DMuonConfig,
    NaiveFSDP2MuonConfig,
    OptimizerConfig,
    SchedulerConfig,
    StepSchedulerConfig,
    TrainHyperParams,
)
from .infra_config import (
    CheckpointConfig,
    DebugConfig,
    DMuonBackendConfig,
    DistributedConfig,
    FSDPBackendConfig,
    LoggingConfig,
    ShardedBackendConfig,
)
from .loader import load_config, save_config
from .model_config import ModelConfig, QActModelConfig
from .registry import (
    register_data_config,
    register_model_config,
    register_optimizer_config,
    register_scheduler_config,
)
from .task_config import TaskConfig
from .train_config import TrainConfig

__all__ = [
    # Core
    "TrainConfig",
    "TaskConfig",
    # Model configs
    "ModelConfig",
    "QActModelConfig",
    # Data configs
    "DataConfig",
    "LeRobotDataConfig",
    # Hyperparams
    "TrainHyperParams",
    "OptimizerConfig",
    "AdamWConfig",
    "DMuonConfig",
    "NaiveFSDP2MuonConfig",
    "SchedulerConfig",
    "CosineSchedulerConfig",
    "ConstantSchedulerConfig",
    "StepSchedulerConfig",
    # Infra configs
    "ShardedBackendConfig",
    "FSDPBackendConfig",
    "DMuonBackendConfig",
    "DistributedConfig",
    "LoggingConfig",
    "CheckpointConfig",
    "DebugConfig",
    # Functions
    "load_config",
    "save_config",
    "register_data_config",
    "register_model_config",
    "register_optimizer_config",
    "register_scheduler_config",
]
