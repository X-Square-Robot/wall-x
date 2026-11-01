import json
import logging

import torch
import torch.nn as nn

from wall_x.utils.constant import _ACTION_KEY_FULL_MAPPING as _MODEL_KEY_TO_RAW_KEY

logger = logging.getLogger(__name__)


def resolve_normalizer_dataset_names(
    dataset_names,
    normalizer,
    batch_size,
    *,
    source="dataset_names",
    allow_skip_names=None,
):
    """Validate dataset names before indexing a normalizer ParameterDict.

    The normalizer key is data-dependent. Callers should pass the
    dataset names carried by the batch/model inputs, matching the qact/VLA
    path. This helper intentionally only validates presence, batch-size
    alignment, and key membership; it does not remap unknown names or fall
    back to a default key, since silent fallback can apply the wrong action
    scale without failing loudly. Pass ``allow_skip_names`` explicitly only
    for names that should bypass normalizer lookup, such as multimodal-only
    rows that do not carry action values.
    """
    if dataset_names is None:
        raise KeyError(f"Missing {source}; cannot choose normalizer key.")

    if isinstance(dataset_names, str):
        names = [dataset_names]
    elif isinstance(dataset_names, (list, tuple)):
        names = [str(name) for name in dataset_names]
    else:
        raise TypeError(
            f"{source} must be a string or sequence of strings, got "
            f"{type(dataset_names).__name__}"
        )

    if len(names) != batch_size:
        raise ValueError(
            f"{source} count ({len(names)}) does not match batch size "
            f"({batch_size}). names={names}"
        )

    available = list(getattr(normalizer, "delta", {}).keys())
    if not available:
        raise KeyError("Normalizer has no registered dataset keys.")

    allowed_skip = set(allow_skip_names or ())
    missing = sorted(
        {name for name in names if name not in available and name not in allowed_skip}
    )
    if missing:
        raise KeyError(
            f"{source} contains keys not present in normalizer: {missing}. "
            f"available={available}"
        )

    return names


def _normalizer_dataset_name_list(dataset_names):
    if dataset_names is None:
        return []
    if isinstance(dataset_names, str):
        return [dataset_names]
    return list(dataset_names)


def _normalizer_width(normalizer, dataset_names):
    names = _normalizer_dataset_name_list(dataset_names)
    if not names:
        return None
    if any(name not in normalizer.delta for name in names):
        return None
    widths = [int(normalizer.delta[name].shape[0]) for name in names]
    if not widths or len(set(widths)) != 1:
        return None
    return widths[0]


def _has_uniform_virtual_tail_mask(dof_mask, width):
    if dof_mask is None:
        return False
    mask = dof_mask.reshape(-1, dof_mask.shape[-1]).bool()
    if mask.shape[-1] <= width:
        return False
    prefix_active = mask[:, :width].all(dim=1)
    tail_inactive = (~mask[:, width:]).all(dim=1)
    return bool((prefix_active & tail_inactive).all().item())


def normalize_data_with_virtual_tail(normalizer, values, dataset_names, dof_mask):
    """Normalize real prefix dims when model tensors include a virtual tail."""
    width = _normalizer_width(normalizer, dataset_names)
    if width is None or values.shape[-1] == width:
        return normalizer.normalize_data(values, dataset_names)
    if values.shape[-1] < width or not _has_uniform_virtual_tail_mask(dof_mask, width):
        return normalizer.normalize_data(values, dataset_names)

    out = values.clone()
    out[..., :width] = normalizer.normalize_data(values[..., :width], dataset_names)
    out[..., width:] = 0
    return out


def unnormalize_data_with_virtual_tail(normalizer, values, dataset_names, dof_mask):
    """Unnormalize real prefix dims when model tensors include a virtual tail."""
    width = _normalizer_width(normalizer, dataset_names)
    if width is None or values.shape[-1] == width:
        return normalizer.unnormalize_data(values, dataset_names, None)
    if values.shape[-1] < width or not _has_uniform_virtual_tail_mask(dof_mask, width):
        return normalizer.unnormalize_data(values, dataset_names, dof_mask)

    out = values.clone()
    out[..., :width] = normalizer.unnormalize_data(
        values[..., :width], dataset_names, None
    )
    out[..., width:] = 0
    return out


def print_rank_last(message):
    """If distributed is initialized, log only on last rank."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1):
            logger.info(message)
    else:
        logger.info(message)


class Normalizer(nn.Module):
    @classmethod
    def from_ckpt(cls, ckpt_path):
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        instance.min = nn.ParameterDict()
        instance.delta = nn.ParameterDict()
        instance.min_key = "min"
        instance.delta_key = "delta"

        ckpt = torch.load(ckpt_path, map_location="cpu")

        for key, value in ckpt.items():
            # parse key: "min.robot_name" -> prefix="min", name="robot_name"
            try:
                prefix, name = key.split(".", 1)
                if hasattr(instance, prefix):
                    getattr(instance, prefix)[name] = nn.Parameter(
                        value, requires_grad=False
                    )
            except ValueError:
                continue

        return instance

    @classmethod
    def from_lerobot_norm_stats(
        cls,
        action_stats,
        dataset_name,
    ):

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        # Set containers
        instance.min = nn.ParameterDict()
        instance.delta = nn.ParameterDict()

        # Fill dataset entry
        instance.min[dataset_name] = nn.Parameter(action_stats.min, requires_grad=False)
        instance.delta[dataset_name] = nn.Parameter(
            action_stats.delta, requires_grad=False
        )

        # Record keys
        instance.min_key = "min"
        instance.delta_key = "delta"

        return instance

    @classmethod
    def from_vectors(cls, min_by_key, delta_by_key):
        """Build a Normalizer directly from precomputed per-dataset min/delta.

        ``min_by_key`` / ``delta_by_key`` are ``{dataset_name: 1-D array/tensor}``
        with matching keys. Used by :func:`create_normalizers_from_v2_stats` so
        the model-side normalizer carries the exact statistics the
        v2 data pipeline used.
        """
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        instance.min = nn.ParameterDict()
        instance.delta = nn.ParameterDict()
        instance.min_key = "min"
        instance.delta_key = "delta"

        for name, vec in min_by_key.items():
            instance.min[name] = nn.Parameter(
                torch.as_tensor(vec, dtype=torch.float32), requires_grad=False
            )
        for name, vec in delta_by_key.items():
            instance.delta[name] = nn.Parameter(
                torch.as_tensor(vec, dtype=torch.float32), requires_grad=False
            )

        return instance

    def __init__(
        self,
        action_statistic_dof,
        dof_config,
        min_key="min",
        delta_key="delta",
        name="normalizer",
    ):
        super(Normalizer, self).__init__()

        self.min_key = min_key
        self.delta_key = delta_key

        action_statistic = {}
        normalizer_missing_information = []
        for robot_name in action_statistic_dof.keys():
            action_statistic[robot_name] = {}
            all_dof_min = []
            all_dof_delta = []
            for config_key, dof_dim in dof_config.items():
                # Use a fallback stats key without mutating the model dof key.
                stat_key = config_key
                follow_key = config_key.replace("master_", "follow_", 1)
                if (
                    stat_key not in action_statistic_dof[robot_name]
                    and follow_key in action_statistic_dof[robot_name]
                ):
                    stat_key = follow_key

                if stat_key in action_statistic_dof[robot_name]:
                    if (
                        min_key in action_statistic_dof[robot_name][stat_key]
                        and delta_key in action_statistic_dof[robot_name][stat_key]
                    ):
                        all_dof_min.extend(
                            action_statistic_dof[robot_name][stat_key][min_key]
                        )
                        all_dof_delta.extend(
                            action_statistic_dof[robot_name][stat_key][delta_key]
                        )
                    else:
                        normalizer_missing_information.append(
                            f"Normalizer (Warning): min_key {min_key} or delta_key {delta_key} not in action_statistic_dof[{robot_name}][{stat_key}], use default min 0.0 and delta 1.0"
                        )
                        all_dof_min.extend([0.0] * dof_dim)
                        all_dof_delta.extend([1.0] * dof_dim)
                else:
                    # config_key is a model key; action_statistic_dof may store raw keys.
                    raw_key = _MODEL_KEY_TO_RAW_KEY.get(config_key)
                    if (
                        raw_key is not None
                        and raw_key in action_statistic_dof[robot_name]
                    ):
                        stat = action_statistic_dof[robot_name][raw_key]
                        if min_key in stat and delta_key in stat:
                            all_dof_min.extend(stat[min_key])
                            all_dof_delta.extend(stat[delta_key])
                        else:
                            normalizer_missing_information.append(
                                f"Normalizer (Warning): Action {config_key} not in action_statistic_dof for {robot_name}, use default min 0.0 and delta 1.0"
                            )
                            all_dof_min.extend([0.0] * dof_dim)
                            all_dof_delta.extend([1.0] * dof_dim)
                    else:
                        normalizer_missing_information.append(
                            f"Normalizer (Warning): Action {config_key} not in action_statistic_dof for {robot_name}, use default min 0.0 and delta 1.0"
                        )
                        all_dof_min.extend([0.0] * dof_dim)
                        all_dof_delta.extend([1.0] * dof_dim)
            all_dof_min = torch.tensor(all_dof_min)
            all_dof_delta = torch.tensor(all_dof_delta)
            action_statistic[robot_name][min_key] = all_dof_min
            action_statistic[robot_name][delta_key] = all_dof_delta

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == (
            torch.distributed.get_world_size() - 1
        ):
            # export normalizer_missing_information to file
            with open(f"normalizer_missing_information_{name}.txt", "w") as f:
                for info in normalizer_missing_information:
                    f.write(info + "\n")
            logger.info(
                "Normalizer missing information saved to normalizer_missing_information_%s.txt",
                name,
            )

        self.min = nn.ParameterDict(
            {
                k: nn.Parameter(action_statistic[k][min_key], requires_grad=False)
                for k in action_statistic.keys()
            }
        )
        self.delta = nn.ParameterDict(
            {
                k: nn.Parameter(action_statistic[k][delta_key], requires_grad=False)
                for k in action_statistic.keys()
            }
        )

    def normalize_data(self, xs, dataset_names):
        new_xs = []
        dataset_names = [name for name in dataset_names if name != "x2_multimodal"]
        for x, dataset_name in zip(xs, dataset_names):
            # if dataset_name == "ex_normal":
            # dataset_name = "x2_normal"
            delta = self.delta[dataset_name]
            delta = torch.where(delta == 0, torch.ones_like(delta), delta)
            x = (x - self.min[dataset_name]) / delta
            x = x * 2 - 1
            x = torch.clamp(x, -1, 1)
            new_xs.append(x)
        new_xs = torch.stack(new_xs)
        return new_xs

    def unnormalize_data(self, xs, dataset_names, dof_mask=None):
        new_xs = []
        dataset_names = [name for name in dataset_names if name != "x2_multimodal"]
        dof_mask = dof_mask if dof_mask is not None else [None] * len(xs)
        for x, dataset_name, mask in zip(xs, dataset_names, dof_mask):
            x = (x + 1) / 2
            if mask is not None:
                mask = mask[0].bool()
                action_space_delta = self.delta[dataset_name][mask]
                action_space_min = self.min[dataset_name][mask]
            else:
                action_space_delta = self.delta[dataset_name]
                action_space_min = self.min[dataset_name]
            x = x * action_space_delta + action_space_min
            new_xs.append(x)
        new_xs = torch.stack(new_xs)
        return new_xs


def _pad_lerobot_stats(stats, target_dim, label):
    """Tail-pad flat LeRobot stats to the model's configured action/state dim."""
    target_dim = int(target_dim)
    current_dim = int(stats.min.numel())
    if current_dim == target_dim:
        return stats
    if current_dim > target_dim:
        raise ValueError(
            f"LeRobot {label} norm stats dim ({current_dim}) exceeds configured "
            f"{label} dim ({target_dim})"
        )

    pad_dim = target_dim - current_dim
    min_stat = torch.cat(
        [stats.min.flatten(), torch.zeros(pad_dim, dtype=stats.min.dtype)]
    )
    delta = torch.cat(
        [stats.delta.flatten(), torch.ones(pad_dim, dtype=stats.delta.dtype)]
    )
    max_stat = torch.cat(
        [stats.max.flatten(), torch.ones(pad_dim, dtype=stats.max.dtype)]
    )
    logger.info(
        "Padded LeRobot %s normalizer stats from %d to %d dims",
        label,
        current_dim,
        target_dim,
    )
    return type(stats)(min=min_stat, max=max_stat, delta=delta)


def create_normalizers_from_lerobot_norm_stats(
    norm_stats, dataset_name, action_dim, propri_dim
):
    """Create model normalizers from LeRobot flat action/state norm stats."""
    action_stats = _pad_lerobot_stats(norm_stats["action"], action_dim, "action")
    propri_stats = _pad_lerobot_stats(norm_stats["state"], propri_dim, "state")
    return (
        Normalizer.from_lerobot_norm_stats(action_stats, dataset_name),
        Normalizer.from_lerobot_norm_stats(propri_stats, dataset_name),
    )


def create_normalizers(config, action_statistic_dof=None):
    """Create action and proprioception normalizers from explicit stats.

    Normalization statistics must come from the config, checkpoint, or dataset.
    Public Wall-X builds intentionally do not bundle private default stats.
    """

    norm_cfg = config.get("normalizer_config") or {}
    if action_statistic_dof is None:
        custom_path = norm_cfg.get("customized_action_statistic_dof")
        if custom_path:
            with open(custom_path, "r") as f:
                action_statistic_dof = json.load(f)
        else:
            raise ValueError(
                "create_normalizers requires action statistics. Provide "
                "`customized_action_statistic_dof` in data.normalizer_config or "
                "pass `action_statistic_dof` from the checkpoint/dataset."
            )

    min_key = norm_cfg.get("min_key", "min")
    delta_key = norm_cfg.get("delta_key", "delta")

    # Required keys for constructing action/proprio normalizers.
    missing_required = []
    if config.get("dof_config") is None:
        missing_required.append("dof_config")
    if config.get("agent_pos_config") is None:
        missing_required.append("agent_pos_config")
    if missing_required:
        raise KeyError(
            "create_normalizers requires non-None config keys: "
            + ", ".join(missing_required)
        )

    normalizer_action = Normalizer(
        action_statistic_dof,
        config["dof_config"],
        min_key=min_key,
        delta_key=delta_key,
    )
    normalizer_propri = Normalizer(
        action_statistic_dof,
        config["agent_pos_config"],
        min_key=min_key,
        delta_key=delta_key,
    )
    return normalizer_action, normalizer_propri, action_statistic_dof


NORM_KEY_SEP = "/"
V2_DEFAULT_MIN_KEY = "q001"
V2_DEFAULT_DELTA_KEY = "delta_001"


def make_norm_key(dataset_type, end_effector_type=None):
    """Compose the model/serving normalizer key from a ``(dataset_type,
    end_effector_type)`` pair.

    Returns the bare ``dataset_type`` when there is no end_effector_type
    (flat/legacy stats, e.g. lerobot / x2robot_v1).
    """
    if end_effector_type is None or end_effector_type == "":
        return dataset_type
    return f"{dataset_type}{NORM_KEY_SEP}{end_effector_type}"


def _list_v2_end_effector_types(ds_stats, min_key, delta_key):
    """List normalizer scopes under one dataset.

    Supported layout:

        stats[dataset_type][action_key] = stat_dict
        stats[dataset_type][end_effector_type][action_key] = stat_dict

    The flat / legacy scope is represented by None.

    Example:
        {
            "follow_right_ee_cartesian_pos": {...},  # flat
            "gripper-G": {
                "follow_right_gripper": {...},
            },
            "gripper-H": {
                "follow_right_gripper": {...},
            },
        }

    Returns:
        [None, "gripper-G", "gripper-H"]
    """

    if not isinstance(ds_stats, dict) or not ds_stats:
        return [None]

    def _is_stat_leaf(value):
        if not isinstance(value, dict):
            return False

        return (
            min_key in value or delta_key in value or "min" in value or "q001" in value
        )

    scopes = []

    for key, value in ds_stats.items():

        # ---------------------------------------------------------
        # Flat / legacy:
        #
        # ds_stats[action_key] = stat_dict
        # ---------------------------------------------------------
        if _is_stat_leaf(value):
            if None not in scopes:
                scopes.append(None)
            continue

        # ---------------------------------------------------------
        # End-effector-specific:
        #
        # ds_stats[ee_type][action_key] = stat_dict
        # ---------------------------------------------------------
        if isinstance(value, dict):
            scopes.append(key)
            continue

        raise ValueError(
            f"Invalid normalizer entry for key={key!r}: "
            f"expected stat dict or end-effector dict, "
            f"got {type(value).__name__}"
        )

    return scopes or [None]


def normalizer_is_identity(normalizer):
    """True if every dataset entry is the identity transform (min==0, delta==1)."""
    min_dict = getattr(normalizer, "min", None)
    delta_dict = getattr(normalizer, "delta", None)
    if not min_dict or not delta_dict:
        return False
    for k in min_dict.keys():
        if not torch.allclose(min_dict[k], torch.zeros_like(min_dict[k])):
            return False
        if k in delta_dict and not torch.allclose(
            delta_dict[k], torch.ones_like(delta_dict[k])
        ):
            return False
    return True

