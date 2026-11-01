"""Shared Dataset Config V2 projection used by Harrix inference stacks."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_RELATIVE_ACTION_FRAMES = frozenset({"base", "body"})

_QWEN25_FACTOR = 28
_QWEN25_MIN_PIXELS = 56 * 56
_QWEN25_MAX_PIXELS = 1280 * 28 * 28
_QWEN35_FACTOR = 32
_QWEN35_MIN_PIXELS = 4 * 32 * 32
_QWEN35_MAX_PIXELS = 16384 * 32 * 32
_STATE_SCHEMA_LABELS = {
    "follow_left_ee_cartesian_pos": "left_xyz",
    "follow_left_ee_rotation_6D": "left_rotation_6d",
    "follow_left_gripper": "left_gripper",
    "follow_right_ee_cartesian_pos": "right_xyz",
    "follow_right_ee_rotation_6D": "right_rotation_6d",
    "follow_right_gripper": "right_gripper",
    "velocity_decomposed": "base_velocity",
    "height": "height",
    "head_actions": "head_actions",
}


@dataclass
class InferenceDataConfig:
    """Processor semantics needed to reproduce Dataset V2 inputs online."""

    resolution: dict[str, int] = field(default_factory=dict)
    model_type: str = "qwen2_5"
    dataset_type: str = ""
    max_pixels: int = _QWEN25_MAX_PIXELS
    min_pixels: int = _QWEN25_MIN_PIXELS
    image_factor: int = _QWEN25_FACTOR
    use_relative_action: bool = False
    predict_action_keys: list[str] = field(default_factory=list)
    obs_action_keys: list[str] = field(default_factory=list)
    text_processor_type: str = ""
    text_processor_params: dict[str, Any] = field(default_factory=dict)
    vision_processor_type: str = ""
    vision_processor_params: dict[str, Any] = field(default_factory=dict)
    is_dino_vision: bool = False
    action_processor_type: str = ""
    action_processor_params: dict[str, Any] = field(default_factory=dict)
    epilogue_processor_type: str = ""
    epilogue_params: dict[str, Any] = field(default_factory=dict)
    state_bins: int = 256
    max_seq_length: int = 2048
    padding_side: str = "left"
    use_state_string_representation: bool = False
    dataset_config_source: str = "legacy"

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)


def resolve_relative_action_frame(data_config: Any) -> str:
    """Resolve the checkpoint's relative EEF convention, defaulting legacy saves.

    Dataset V2 stores this under ``processors.action.params``.  Inference keeps
    those params intact on :class:`InferenceDataConfig`; dict support is kept for
    older serving wrappers.
    """
    if isinstance(data_config, dict):
        params = data_config.get("action_processor_params") or {}
        if not params:
            params = (
                ((data_config.get("processors") or {}).get("action") or {}).get(
                    "params"
                )
                or {}
            )
        if not params:
            effective = data_config.get("dataset_config_effective") or {}
            dataset = (
                effective.get("dataset", effective)
                if isinstance(effective, dict)
                else {}
            )
            params = (
                (((dataset.get("processors") or {}).get("action") or {}).get(
                    "params"
                ))
                or {}
            )
    else:
        params = getattr(data_config, "action_processor_params", {}) or {}

    frame = params.get("relative_action_frame", "base")
    if frame not in _RELATIVE_ACTION_FRAMES:
        raise ValueError(
            "relative_action_frame must be one of "
            f"{sorted(_RELATIVE_ACTION_FRAMES)}, got {frame!r}"
        )
    return frame


def merge_private_data_config(cfg, train_config: dict) -> dict:
    """Restore backend fields omitted by ``wall_x.config.loader.dump_config``."""
    raw_data = copy.deepcopy(getattr(cfg, "_raw_data", {}) or {})
    raw_data.update(train_config.get("data", {}) or {})
    train_config["data"] = raw_data
    return raw_data


def normalize_dataset_config_for_inference(
    train_config: dict,
) -> tuple[dict[str, Any], str]:
    """Overlay effective Dataset V2 processor values onto ``data``.

    The checkpoint-embedded effective config is authoritative.  The external
    YAML path is retained only as a compatibility fallback for older saves.
    """
    data = train_config.setdefault("data", {})
    dataset: dict[str, Any] = {}
    source = "legacy"
    effective = data.get("dataset_config_effective") or {}
    if isinstance(effective, dict) and effective:
        dataset = effective.get("dataset") or effective
        source = "checkpoint:data.dataset_config_effective"
    else:
        dataset_config_path = data.get("dataset_config_path")
        if dataset_config_path and os.path.exists(dataset_config_path):
            try:
                with open(dataset_config_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                dataset = raw.get("dataset") or {}
                source = dataset_config_path
            except Exception as exc:
                logger.warning(
                    "Failed to load dataset_config_path=%s: %s",
                    dataset_config_path,
                    exc,
                )

    if not isinstance(dataset, dict) or not dataset:
        return {}, source

    window = dataset.get("window") or {}
    if window.get("action_horizon") is not None:
        data["action_horizon"] = window["action_horizon"]
        data["action_horizon_flow"] = window["action_horizon"]
    if window.get("action_horizon_ar") is not None:
        data["action_horizon_ar"] = window["action_horizon_ar"]

    processors = copy.deepcopy(dataset.get("processors") or {})
    vision = processors.get("vision") or {}
    vision_params = vision.get("params") or {}
    for key in ("resolution", "image_factor", "min_pixels", "max_pixels"):
        if vision_params.get(key) is not None:
            data[key] = copy.deepcopy(vision_params[key])
    if vision.get("type") in ("qwen3", "qwen3_5"):
        data["image_factor"] = vision_params.get("image_factor", _QWEN35_FACTOR)
        data["min_pixels"] = vision_params.get("min_pixels", _QWEN35_MIN_PIXELS)
        data["max_pixels"] = vision_params.get("max_pixels", _QWEN35_MAX_PIXELS)

    action_params = (processors.get("action") or {}).get("params") or {}
    for key in ("predict_action_keys", "obs_action_keys"):
        if action_params.get(key) is not None:
            data[key] = copy.deepcopy(action_params[key])

    text_params = (processors.get("text") or {}).get("params") or {}
    if text_params.get("use_relative_action") is not None:
        data["use_relative_action"] = text_params["use_relative_action"]

    epilogue_params = (processors.get("epilogue") or {}).get("params") or {}
    for key in (
        "state_bins",
        "max_seq_length",
        "padding_side",
        "use_state_string_representation",
    ):
        if epilogue_params.get(key) is not None:
            data[key] = epilogue_params[key]

    return processors, source


def build_inference_data_config(
    train_config: dict,
    processors: dict[str, Any],
    source: str,
) -> InferenceDataConfig:
    """Build the common runtime view from normalized train config data."""
    data = train_config.get("data", {}) or {}

    def get(key: str, default: Any) -> Any:
        return data.get(key, train_config.get(key, default))

    text = processors.get("text") or {}
    vision = processors.get("vision") or {}
    action = processors.get("action") or {}
    epilogue = processors.get("epilogue") or {}
    text_params = copy.deepcopy(text.get("params") or {})
    vision_params = copy.deepcopy(vision.get("params") or {})
    action_params = copy.deepcopy(action.get("params") or {})
    epilogue_params = copy.deepcopy(epilogue.get("params") or {})
    dof_config = train_config.get("dof_config") or {}
    agent_pos_config = train_config.get("agent_pos_config") or {}

    def key_list(value: Any, fallback: dict[str, Any]) -> list[str]:
        if isinstance(value, dict):
            return list(value.keys())
        if value:
            return list(value)
        return list(fallback.keys())

    dino_image_factor = _resolve_dino_image_factor(train_config)
    dataset_type = get("dataset_type", train_config.get("dataset_type", ""))
    result = InferenceDataConfig(
        resolution=copy.deepcopy(get("resolution", {}) or {}),
        model_type=get("model_type", train_config.get("model_type", "qwen2_5")),
        dataset_type=str(dataset_type),
        max_pixels=int(get("max_pixels", _QWEN25_MAX_PIXELS)),
        min_pixels=int(get("min_pixels", _QWEN25_MIN_PIXELS)),
        image_factor=int(get("image_factor", _QWEN25_FACTOR)),
        use_relative_action=bool(get("use_relative_action", False)),
        predict_action_keys=key_list(get("predict_action_keys", None), dof_config),
        obs_action_keys=key_list(get("obs_action_keys", None), agent_pos_config),
        text_processor_type=str(text.get("type", "")),
        text_processor_params=text_params,
        vision_processor_type=str(vision.get("type", "")),
        vision_processor_params=vision_params,
        is_dino_vision=dino_image_factor is not None,
        action_processor_type=str(action.get("type", "")),
        action_processor_params=action_params,
        epilogue_processor_type=str(epilogue.get("type", "")),
        epilogue_params=epilogue_params,
        state_bins=int(get("state_bins", 256)),
        max_seq_length=int(get("max_seq_length", 2048)),
        # The checkpoint records this under the epilogue processor params
        # (``data.dataset_config_effective.dataset.processors.epilogue.params``);
        # an explicit ``data.padding_side`` still wins if one is present.
        padding_side=str(get("padding_side", epilogue_params.get("padding_side", "left"))),
        use_state_string_representation=bool(
            get("use_state_string_representation", False)
        ),
        dataset_config_source=source,
    )
    if (
        result.model_type == "qwen3_5"
        and dataset_type == "x2robot_v2"
        and result.vision_processor_type == "default"
        and result.image_factor == _QWEN35_FACTOR
        and _has_compatible_qwen35_image_processor(train_config)
    ):
        # Older runs used VideoVisionProcessor (default) with factor=32.
        # Select the Qwen inference path only after resolving resize values:
        # relabeling the saved config earlier would change min/max defaults.
        result.vision_processor_type = "qwen3_5"

    qwen35_vision_type_is_valid = result.vision_processor_type in (
        "qwen3",
        "qwen3_5",
    )
    if result.vision_processor_type == "default" and dino_image_factor is not None:
        qwen35_vision_type_is_valid = result.image_factor == dino_image_factor
        if not qwen35_vision_type_is_valid:
            raise ValueError(
                "DINO inference image_factor must match patch_size * merge_size: "
                f"configured={result.image_factor}, expected={dino_image_factor}"
            )
    if (
        train_config.get("model_type") == "qwen3_5"
        and dataset_type == "x2robot_v2"
        and not qwen35_vision_type_is_valid
    ):
        raise ValueError(
            "qwen3_5 inference requires Dataset V2 vision processor type "
            "qwen3_5/qwen3, or type default backed by a compatible Qwen3.5 "
            "image processor with image_factor=32, or a compatible DINO "
            f"processor; got {result.vision_processor_type!r} from "
            f"{result.dataset_config_source}"
        )
    return result


def _has_compatible_qwen35_image_processor(train_config: dict) -> bool:
    """Recognize legacy stock Qwen processors without loading model weights."""
    model = train_config.get("model") or {}
    processor_path = model.get("processor_path") or train_config.get("processor_path")
    if not processor_path:
        return False
    for filename in ("processor_config.json", "preprocessor_config.json"):
        path = os.path.join(os.fspath(processor_path), filename)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as stream:
                config = json.load(stream)
            image = config.get("image_processor", config)
            if image.get("image_processor_type") in (
                "Qwen2VLImageProcessor",
                "Qwen2VLImageProcessorFast",
                "Qwen3VLImageProcessor",
                "Qwen3VLImageProcessorFast",
            ) and (
                int(image.get("patch_size", 0)),
                int(image.get("merge_size", 0)),
                int(image.get("temporal_patch_size", 0)),
            ) == (
                16,
                2,
                2,
            ):
                return True
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            logger.warning("Failed to inspect processor config %s: %s", path, exc)
    return False


def _resolve_dino_image_factor(train_config: dict) -> int | None:
    """Return the DINO patch/merge quantum declared by the saved processor."""
    model = train_config.get("model") or {}
    processor_path = model.get("processor_path") or train_config.get(
        "processor_path"
    )
    if not processor_path:
        return None
    config_path = os.path.join(os.fspath(processor_path), "processor_config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, encoding="utf-8") as stream:
            processor_config = json.load(stream)
    except (OSError, ValueError) as exc:
        logger.warning("Failed to inspect processor config %s: %s", config_path, exc)
        return None
    image_config = processor_config.get("image_processor") or {}
    if image_config.get("image_processor_type") != "PluggableImageProcessor":
        return None
    patch_size = int(image_config.get("patch_size", 0))
    merge_size = int(image_config.get("merge_size", 0))
    if patch_size <= 0 or merge_size <= 0:
        raise ValueError(
            "DINO processor must declare positive patch_size and merge_size in "
            f"{config_path}"
        )
    return patch_size * merge_size


def dataset_v2_smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Bit-for-bit sizing logic from Dataset V2's vision processor."""
    if height < 1 or width < 1:
        return factor, factor
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        scale = math.sqrt(max_pixels / (height * width))
        h_bar = max(factor, round(height * scale / factor) * factor)
        w_bar = max(factor, round(width * scale / factor) * factor)
    if h_bar * w_bar < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h_bar = max(factor, round(height * scale / factor) * factor)
        w_bar = max(factor, round(width * scale / factor) * factor)
    return h_bar, w_bar


def build_dataset_v2_text_processor(data_config: InferenceDataConfig):
    """Public builds have no Dataset V2 text processor; use generic prompts."""
    return None


def replace_state_schema_placeholder(
    text: str,
    data_config: InferenceDataConfig,
) -> str:
    """Apply Dataset V2 WallxEpilogue's state-schema replacement online."""
    placeholder = "<|state_schema|>"
    if placeholder not in text:
        return text
    schema_keys = data_config.epilogue_params.get("state_schema_keys")
    if not isinstance(schema_keys, dict) or not schema_keys:
        raise ValueError(
            "Dataset V2 enables the state-schema prompt, but "
            "epilogue.params.state_schema_keys is missing"
        )
    parts = []
    for key, dim in schema_keys.items():
        if not isinstance(key, str) or not key or int(dim) <= 0:
            raise ValueError(f"invalid state_schema_keys entry: {key!r}={dim!r}")
        parts.append(f"{_STATE_SCHEMA_LABELS.get(key, key)}({int(dim)})")
    return text.replace(placeholder, " + ".join(parts))


def inject_dataset_v2_state_strings(
    prefix_text: list[str],
    norm_state: Any,
    agent_pos_mask: Any,
    dataset_names: list[str],
    *,
    state_bins: int,
    keep_dims_by_dataset: dict[str, list[int]],
    state_drop: bool = False,
) -> list[str]:
    """Inject state strings using checkpoint keep-dim semantics.

    Dataset names are normalizer keys and may include an end-effector suffix.
    Exact mapping keys win; otherwise the longest dataset-name prefix is used.
    """
    state_np = (
        norm_state.detach().cpu().numpy()
        if hasattr(norm_state, "detach")
        else np.asarray(norm_state)
    )
    mask_np = (
        agent_pos_mask.detach().cpu().numpy()
        if hasattr(agent_pos_mask, "detach")
        else np.asarray(agent_pos_mask)
    )
    if state_np.ndim == 3:
        state_np = state_np[:, 0, :]
    if mask_np.ndim == 3:
        mask_np = mask_np[:, 0, :]
    if state_np.ndim != 2 or mask_np.shape != state_np.shape:
        raise ValueError(
            "Dataset V2 state injection requires matching [B, D] or [B, T, D] "
            f"state/mask arrays; got state={state_np.shape}, mask={mask_np.shape}"
        )
    if len(dataset_names) != state_np.shape[0]:
        raise ValueError(
            "Dataset V2 state injection dataset-name count does not match state "
            f"rows: {len(dataset_names)} != {state_np.shape[0]}"
        )

    discretized = (
        np.digitize(
            state_np,
            bins=np.linspace(-1, 1, int(state_bins) + 1)[:-1],
        )
        - 1
    )
    valid_mask = mask_np.astype(bool)
    output = list(prefix_text)
    state_row = 0
    for text_row, row_text in enumerate(output):
        if "<|propri|>" not in row_text:
            continue
        if state_row >= discretized.shape[0]:
            raise ValueError("More <|propri|> placeholders than state rows")
        if state_drop:
            state_string = ""
        else:
            normalizer_key = str(dataset_names[state_row])
            mapping_key = (
                normalizer_key
                if normalizer_key in keep_dims_by_dataset
                else None
            )
            if mapping_key is None:
                matches = [
                    key
                    for key in keep_dims_by_dataset
                    if normalizer_key.startswith(f"{key}/")
                ]
                if matches:
                    mapping_key = max(matches, key=len)

            if mapping_key is None:
                values = discretized[state_row, valid_mask[state_row]]
            else:
                width = discretized.shape[1]
                keep_dims = keep_dims_by_dataset[mapping_key]
                invalid_dims = [
                    dim for dim in keep_dims if dim < 0 or dim >= width
                ]
                if invalid_dims:
                    raise ValueError(
                        "state_string_keep_dims_by_dataset contains out-of-range "
                        f"dimensions for dataset {mapping_key!r}: {invalid_dims}; "
                        f"state width is {width}"
                    )
                values = np.full(width, -1, dtype=discretized.dtype)
                for dim in keep_dims:
                    if valid_mask[state_row, dim]:
                        values[dim] = discretized[state_row, dim]
            state_string = " ".join(map(str, values))
        output[text_row] = row_text.replace("<|propri|>", state_string)
        state_row += 1
    return output


def validate_inference_data_semantics(
    data_config: InferenceDataConfig,
    action_horizon: int,
) -> None:
    """Fail fast when saved Dataset V2 semantics cannot be reproduced."""
    if data_config.text_processor_type not in (
        "",
        "wallx",
        "wallx_task_instruction",
    ):
        raise NotImplementedError(
            "flow inference cannot reproduce Dataset V2 text processor "
            f"{data_config.text_processor_type!r}"
        )
    trained_horizon = data_config.text_processor_params.get("action_chunk_size")
    if trained_horizon is not None and int(trained_horizon) != int(action_horizon):
        raise ValueError(
            "inference action_horizon does not match Dataset V2 "
            "processors.text.params.action_chunk_size: "
            f"{action_horizon} != {trained_horizon}"
        )
    if (
        data_config.vision_processor_params.get("resize_mode", "smart_resize")
        != "smart_resize"
    ):
        raise NotImplementedError(
            "flow inference currently supports only Dataset V2 "
            "vision resize_mode='smart_resize'"
        )
    unsupported = []
    for key in (
        "enable_image_history",
        "fixed_pad_to",
    ):
        if data_config.epilogue_params.get(key):
            unsupported.append(key)
    if unsupported:
        raise NotImplementedError(
            "flow inference cannot yet reproduce these Dataset V2 epilogue "
            f"settings: {unsupported}"
        )
