"""Task config shared by model and data code."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TaskConfig:
    """Robot task definition shared by model construction and data slicing."""

    dof_config: Dict[str, int] = field(default_factory=dict)
    agent_pos_config: Dict[str, int] = field(default_factory=dict)
    ar_dof_config: Optional[Dict[str, int]] = None
    action_horizon: int = 32
    action_horizon_flow: int = 32
    noise_scheduler: Optional[Dict[str, Any]] = None
    use_state_string_representation: bool = False
