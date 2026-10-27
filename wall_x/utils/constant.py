"""OSS-safe compatibility constants.

The internal repository keeps canonical robot, dataset, action-key, and action
statistics tables in ``x2robot_utils.constants``. Those tables are intentionally
not bundled in the public export. This shim keeps the public VLA code importable
without exposing private dataset catalogues or default normalization stats.
"""

from __future__ import annotations

from collections.abc import Iterable


ACTION_KEY_FULL_MAPPING: dict[str, str] = {}
_ACTION_KEY_FULL_MAPPING = ACTION_KEY_FULL_MAPPING

ACTION_DATASET_NAMES: tuple[str, ...] = ()
MULTIMODAL_DATASET_NAMES: tuple[str, ...] = ()
_ACTION_DATASET_NAMES = ACTION_DATASET_NAMES
_MULTIMODAL_DATASET_NAMES = MULTIMODAL_DATASET_NAMES

VIEW_SLOT_KEYS = ["view1", "view2", "view3"]
PHYSICAL_VIEW_IDS: dict[str, int] = {}


def get_view_slot_keys(num_views: int) -> list[str]:
    """Return canonical view slot names for ``num_views`` cameras."""
    return [f"view{i + 1}" for i in range(max(0, int(num_views)))]


def is_multimodal_dataset_name(dataset_name: str | None) -> bool:
    """Return whether a dataset is marked multimodal in the public shim."""
    return dataset_name in MULTIMODAL_DATASET_NAMES


def is_action_dataset_name(dataset_name: str | None) -> bool:
    """Return whether a dataset row should be treated as an action sample."""
    return dataset_name is not None and not is_multimodal_dataset_name(dataset_name)


def iter_action_dataset_names(dataset_names: Iterable[str]) -> list[str]:
    """Filter a sequence down to action dataset names."""
    return [name for name in dataset_names if is_action_dataset_name(name)]


__all__ = [
    "ACTION_KEY_FULL_MAPPING",
    "_ACTION_KEY_FULL_MAPPING",
    "ACTION_DATASET_NAMES",
    "MULTIMODAL_DATASET_NAMES",
    "_ACTION_DATASET_NAMES",
    "_MULTIMODAL_DATASET_NAMES",
    "VIEW_SLOT_KEYS",
    "PHYSICAL_VIEW_IDS",
    "get_view_slot_keys",
    "is_multimodal_dataset_name",
    "is_action_dataset_name",
    "iter_action_dataset_names",
]
