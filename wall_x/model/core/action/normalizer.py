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
            for k in dof_config:
                if (
                    k not in action_statistic_dof[robot_name]
                    and k.replace("master_", "follow_", 1)
                    in action_statistic_dof[robot_name]
                ):
                    k = k.replace("master_", "follow_", 1)
                if k in action_statistic_dof[robot_name]:
                    if (
                        min_key in action_statistic_dof[robot_name][k]
                        and delta_key in action_statistic_dof[robot_name][k]
                    ):
                        all_dof_min.extend(action_statistic_dof[robot_name][k][min_key])
                        all_dof_delta.extend(
                            action_statistic_dof[robot_name][k][delta_key]
                        )
                    else:
                        normalizer_missing_information.append(
                            f"Normalizer (Warning): min_key {min_key} or delta_key {delta_key} not in action_statistic_dof[{robot_name}][{k}], use default min 0.0 and delta 1.0"
                        )
                        all_dof_min.extend([0.0] * dof_config[k])
                        all_dof_delta.extend([1.0] * dof_config[k])
                else:
                    # k is a model key; action_statistic_dof may store raw keys — try fallback lookup
                    raw_key = _MODEL_KEY_TO_RAW_KEY.get(k)
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
                                f"Normalizer (Warning): Action {k} not in action_statistic_dof for {robot_name}, use default min 0.0 and delta 1.0"
                            )
                            all_dof_min.extend([0.0] * dof_config[k])
                            all_dof_delta.extend([1.0] * dof_config[k])
                    else:
                        normalizer_missing_information.append(
                            f"Normalizer (Warning): Action {k} not in action_statistic_dof for {robot_name}, use default min 0.0 and delta 1.0"
                        )
                        all_dof_min.extend([0.0] * dof_config[k])
                        all_dof_delta.extend([1.0] * dof_config[k])
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
            norm_dim = self.min[dataset_name].shape[0]
            if x.shape[-1] > norm_dim:
                real = x[..., :norm_dim]
                tail = x[..., norm_dim:]
                real = (real - self.min[dataset_name]) / (self.delta[dataset_name])
                real = real * 2 - 1
                real = torch.clamp(real, -1, 1)
                x = torch.cat([real, tail], dim=-1)
            else:
                x = (x - self.min[dataset_name]) / (self.delta[dataset_name])
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
            norm_dim = self.delta[dataset_name].shape[0]
            if x.shape[-1] > norm_dim:
                real = x[..., :norm_dim]
                tail = x[..., norm_dim:]
                real = real * self.delta[dataset_name] + self.min[dataset_name]
                x = torch.cat([real, tail], dim=-1)
            elif mask is not None:
                mask = mask[0].bool()
                action_space_delta = self.delta[dataset_name][mask]
                action_space_min = self.min[dataset_name][mask]
                x = x * action_space_delta + action_space_min
            else:
                action_space_delta = self.delta[dataset_name]
                action_space_min = self.min[dataset_name]
                x = x * action_space_delta + action_space_min
            new_xs.append(x)
        new_xs = torch.stack(new_xs)
        return new_xs


def create_normalizers(config, action_statistic_dof=None):
    """Create action and proprioception normalizers from explicit stats.

    Normalization statistics must come from the config, checkpoint, or dataset.
    Public Wall-X builds intentionally do not bundle private default stats.
    """

    if action_statistic_dof is None:
        custom_path = config.get("customized_action_statistic_dof")
        if custom_path:
            with open(custom_path, "r") as f:
                action_statistic_dof = json.load(f)
        else:
            raise ValueError(
                "create_normalizers requires action statistics. Provide "
                "`customized_action_statistic_dof` in the config or pass "
                "`action_statistic_dof` from the checkpoint/dataset."
            )

    min_key = config.get("min_key", "min")
    delta_key = config.get("delta_key", "delta")

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
