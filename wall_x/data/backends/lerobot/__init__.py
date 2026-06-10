"""LeRobot backend registration."""

from __future__ import annotations

import sys

try:
    from wall_x.data._registry import register_module
    from wall_x.data.backends.lerobot.build import (
        build,
        load_trainer_data_config,
        load_trainer_data_config_from_yaml_dict,
    )

    register_module("lerobot", sys.modules[__name__])
except ImportError as _e:
    from wall_x.data._registry import record_import_error

    record_import_error("lerobot", _e)


__all__ = [
    "build",
    "load_trainer_data_config",
    "load_trainer_data_config_from_yaml_dict",
]
