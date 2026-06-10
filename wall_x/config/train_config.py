"""Top-level training config."""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict

from .data_config import DataConfig
from .hyperparams_config import TrainHyperParams
from .infra_config import (
    CheckpointConfig,
    DebugConfig,
    DistributedConfig,
    LoggingConfig,
)
from .model_config import ModelConfig
from .task_config import TaskConfig


@dataclass
class TrainConfig:
    """Top-level Wall-X training config."""

    model_type: str = "qwen2_5"
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hyperparams: TrainHyperParams = field(default_factory=TrainHyperParams)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    # Raw YAML sections preserved verbatim for backend APIs that read fields
    # not captured by the typed DataConfig dataclass.
    _raw_data: Dict[str, Any] = field(default_factory=dict)
    # Full raw YAML dict for backend-specific compatibility paths.
    _raw_yaml: Dict[str, Any] = field(default_factory=dict)
    # dataset_path lives at top level in YAML, consumed by data loaders directly
    dataset_path: Any = None

    @property
    def action_dim(self) -> int:
        return sum(self.task.dof_config.values())

    @property
    def propri_dim(self) -> int:
        return sum(self.task.agent_pos_config.values())

    def build_data_loader_dict(self) -> Dict[str, Any]:
        """Build the raw dict consumed by backend-specific compatibility paths.

        Merges ``_raw_data`` (verbatim YAML ``data:`` section) with task
        fields (dof_config, action_horizon, etc.) and hyperparams
        (batch_size). Backend compatibility APIs may read fields that the
        typed DataConfig dataclass does not carry.

        This keeps legacy flat configs working while the main config surface
        stays typed.
        """
        # Start with the raw YAML data section, then add typed task defaults.
        data_dict = dict(self._raw_data)
        task_dict = dataclasses.asdict(self.task)
        for key in (
            "dof_config",
            "agent_pos_config",
            "action_horizon",
            "action_horizon_flow",
            "ar_dof_config",
            "use_state_string_representation",
        ):
            if key in task_dict and task_dict[key] is not None:
                data_dict.setdefault(key, task_dict[key])
        data_dict.setdefault("batch_size_per_gpu", self.hyperparams.batch_size_per_gpu)
        data_dict.setdefault("batch_size", self.hyperparams.batch_size_per_gpu)
        result: Dict[str, Any] = {
            "model_type": self.model_type,
            "data": data_dict,
            **task_dict,
        }
        if self.dataset_path is not None:
            result["dataset_path"] = self.dataset_path
        return result
