"""RTC (Real-Time Chunking) protocol helpers, independent of any policy.

Protocol parsing, delay clamping, the absolute-to-relative reference-frame
conversion, and response fields live here; the module depends only on numpy
and geometry helpers. The DoF layout is injected through ``segments``, which
each policy computes from its own config. Prefix pinning inside the model is
implemented on the model side.

Requests take the RTC path (training-time hard pinning) when the observation
carries:
    use_rtc           : bool
    prev_action_chunk : np.ndarray [N, 14], absolute dual-arm end-effector pose
                        [left_pos(3), left_rpy(3), left_grip(1),
                         right_pos(3), right_rpy(3), right_grip(1)]
    inference_delay   : int, number of leading steps pinned to the prefix
    num_flow_steps    : int, optional (preferred name)
    num_steps         : int, optional (equivalent alias)
    rtc_mode          : "training_time" | "inference_time" (the latter falls
                        back to training_time with a warning)
Responses additionally carry:
    action_pred       : np.ndarray [*, 14] = concat(follow1_pos, follow2_pos, -1)
    server_timing     : {"infer_ms": float}
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from wall_x._vendor.x2robot_utils import geometry as _geom

RTC_KEY_USE = "use_rtc"
RTC_KEY_PREV_CHUNK = "prev_action_chunk"
RTC_KEY_DELAY = "inference_delay"
RTC_KEY_NUM_STEPS = "num_steps"
RTC_KEY_NUM_FLOW_STEPS = "num_flow_steps"
RTC_KEY_MODE = "rtc_mode"

RTC_EE_CHUNK_DIM = 14
RTC_EE_PER_ARM_DIM = RTC_EE_CHUNK_DIM // 2
RTC_DEFAULT_MAX_RTC_STEPS = 10

USE_CHUNK_STATE_ENV = os.environ.get("WALLX_USE_CHUNK_STATE", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
if USE_CHUNK_STATE_ENV:
    logger.warning(
        "[rtc] WALLX_USE_CHUNK_STATE=1: RTC ref switches to prev_action_chunk[0] "
        "(= master_abs[t_new]) instead of obs.state.follow — ablation only "
        "(re-introduces master-vs-follow servo lag offset)."
    )

DofSegments = List[Tuple[str, str, int, int]]


def get_training_max_rtc_steps(
    train_config: Optional[Dict[str, Any]],
    model_config: Any = None,
    default: int = RTC_DEFAULT_MAX_RTC_STEPS,
) -> int:
    """Return the maximum RTC delay seen in training, or the client default.

    The server clamps the requested delay to this value (rtc_train.max_delay
    and friends); longer prefixes would be out of the training distribution.
    """
    train_config = train_config or {}
    candidates = [
        train_config.get("max_rtc_steps"),
        train_config.get("rtc_max_steps"),
        train_config.get("max_inference_delay"),
    ]
    rtc_train = train_config.get("rtc_train", {})
    if isinstance(rtc_train, dict):
        candidates.append(rtc_train.get("max_delay"))
        candidates.append(rtc_train.get("max_rtc_steps"))
    for section_name in ("data", "rtc", "rtc_config", "epilogue"):
        section = train_config.get(section_name, {})
        if isinstance(section, dict):
            candidates.extend(
                [
                    section.get("max_rtc_steps"),
                    section.get("rtc_max_steps"),
                    section.get("max_inference_delay"),
                ]
            )
            params = section.get("params", {})
            if isinstance(params, dict):
                inner = params.get("rtc_train", {})
                if isinstance(inner, dict):
                    candidates.append(inner.get("max_delay"))
    if model_config is not None:
        candidates.extend(
            [
                getattr(model_config, "max_rtc_steps", None),
                getattr(model_config, "rtc_max_steps", None),
                getattr(model_config, "max_inference_delay", None),
            ]
        )
    for raw_value in candidates:
        if raw_value is None:
            continue
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return default


def clamp_rtc_delay(
    requested_delay: int,
    prev_len: int,
    action_horizon: int,
    max_rtc_steps: int,
) -> int:
    """Clamp the client delay to the range accepted by the server and training."""
    max_allowed = min(prev_len, action_horizon, max_rtc_steps)
    if max_allowed <= 0:
        return 0
    delay = min(int(requested_delay), max_allowed)
    if delay != requested_delay:
        logger.warning(
            "RTC: clamped inference_delay from %d to %d "
            "(prev_len=%d, action_horizon=%d, max_rtc_steps=%d)",
            requested_delay,
            delay,
            prev_len,
            action_horizon,
            max_rtc_steps,
        )
    return delay


def parse_rtc_obs(
    obs: Dict,
    action_horizon: int,
    max_rtc_steps: int,
) -> Optional[Dict[str, Any]]:
    """Parse RTC fields from ``obs``; return None (standard flow) when RTC is off or incomplete.

    A valid RTC observation has a truthy use_rtc, prev_action_chunk [N, 14]
    (or [1, N, 14]), and inference_delay > 0. The delay is clamped against
    N - 1 because chunk[0] is consumed as the state reference.
    """
    if not obs.get(RTC_KEY_USE):
        return None

    rtc_mode = obs.get(RTC_KEY_MODE)
    if rtc_mode == "inference_time":
        logger.warning(
            "RTC: client requested rtc_mode='inference_time', but server only "
            "implements training_time (hard prefix substitution). Falling back; "
            "prefix_attention_* / max_guidance_weight are ignored."
        )

    prev = obs.get(RTC_KEY_PREV_CHUNK)
    if prev is None:
        return None
    try:
        delay = int(obs.get(RTC_KEY_DELAY, 0) or 0)
    except (TypeError, ValueError):
        logger.warning(
            "RTC: non-numeric inference_delay %r, fallback to non-RTC",
            obs.get(RTC_KEY_DELAY),
        )
        return None
    if delay <= 0:
        return None
    prev_arr = np.asarray(prev)
    if prev_arr.ndim == 3 and prev_arr.shape[0] == 1:
        prev_arr = prev_arr[0]
    if prev_arr.ndim != 2 or prev_arr.shape[-1] != RTC_EE_CHUNK_DIM:
        logger.warning(
            "RTC: prev_action_chunk has invalid shape %s "
            "(expected [N, %d] abs EE pose), fallback to non-RTC",
            prev_arr.shape,
            RTC_EE_CHUNK_DIM,
        )
        return None
    delay = clamp_rtc_delay(
        delay, prev_arr.shape[0] - 1, action_horizon, max_rtc_steps
    )
    if delay <= 0:
        return None

    out: Dict[str, Any] = {
        "prev_action_chunk": prev_arr,
        "inference_delay": delay,
    }
    num_steps_raw = obs.get(RTC_KEY_NUM_FLOW_STEPS)
    if num_steps_raw is None:
        num_steps_raw = obs.get(RTC_KEY_NUM_STEPS)
    if num_steps_raw is not None:
        try:
            num_steps = int(num_steps_raw)
        except (TypeError, ValueError):
            logger.warning(
                "RTC: non-numeric num_flow_steps %r, using server default",
                num_steps_raw,
            )
            num_steps = 0
        if num_steps > 0:
            out["num_steps"] = num_steps
    return out


def build_state_dict_from_abs_ref(ref_ee14: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert a 14-dim absolute end-effector reference into follow1_pos / follow2_pos state."""
    ref = np.asarray(ref_ee14, dtype=np.float32)
    return {
        "follow1_pos": ref[0:7].copy(),
        "follow2_pos": ref[7:14].copy(),
    }


def transform_abs_chunk_to_relative(
    prev_abs_ee14: np.ndarray,
    sensor_state_ee14: np.ndarray,
    segments: DofSegments,
    use_chunk_state: bool = USE_CHUNK_STATE_ENV,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Convert a 14-dim absolute end-effector chunk into (relative+6D server layout, state_override).

    * The client sends the previous action_pred shifted left by the executed
      horizon; chunk[0] is the current (already elapsed) step and is dropped,
      chunk[1:] is the action sequence.
    * Training uses rel[k] = master[t+1+k] - follow[t], so the reference is the
      sensor state (obs.state.follow1/2_pos) by default.
    * use_chunk_state=True uses chunk[0] as the reference (ablation only).
    * Positions are offset by the reference position; rotations become 6D
      deltas against the reference rotation; grippers are copied. Other
      segments (velocity/height/head/action_padding) carry no data in the
      14-dim protocol and stay 0.
    """
    if prev_abs_ee14.ndim != 2 or prev_abs_ee14.shape[-1] != RTC_EE_CHUNK_DIM:
        raise ValueError(
            f"prev_action_chunk must be 2D [N, {RTC_EE_CHUNK_DIM}] "
            f"(absolute dual-arm EE pose), got shape={tuple(prev_abs_ee14.shape)}"
        )
    if prev_abs_ee14.shape[0] < 2:
        raise ValueError(
            f"prev_action_chunk length must be >= 2 (chunk[0] discarded as "
            f"current-time action), got N={prev_abs_ee14.shape[0]}"
        )
    if sensor_state_ee14.shape[-1] != RTC_EE_CHUNK_DIM:
        raise ValueError(
            f"sensor_state_ee14 must have last dim {RTC_EE_CHUNK_DIM}, "
            f"got shape={tuple(sensor_state_ee14.shape)}"
        )

    if use_chunk_state:
        ref_flat = np.asarray(prev_abs_ee14[0], dtype=np.float32).reshape(14)
    else:
        ref_flat = np.asarray(sensor_state_ee14, dtype=np.float32).reshape(14)
    state_override = build_state_dict_from_abs_ref(ref_flat)

    chunk_f64 = prev_abs_ee14[1:].astype(np.float64)
    n_action = chunk_f64.shape[0]  # = N - 1
    left_pos_abs = chunk_f64[:, 0:3]
    left_rpy_abs = chunk_f64[:, 3:6]
    left_grip = chunk_f64[:, 6:7]
    right_pos_abs = chunk_f64[:, 7:10]
    right_rpy_abs = chunk_f64[:, 10:13]
    right_grip = chunk_f64[:, 13:14]

    sensor_f64 = ref_flat.astype(np.float64)
    state_pos_l = sensor_f64[0:3].reshape(1, 3)
    state_rpy_l = sensor_f64[3:6]
    state_pos_r = sensor_f64[7:10].reshape(1, 3)
    state_rpy_r = sensor_f64[10:13]
    state_rot6d_l = _geom.euler_to_matrix_zyx_6d_nb(state_rpy_l[None])[0]
    state_rot6d_r = _geom.euler_to_matrix_zyx_6d_nb(state_rpy_r[None])[0]

    left_pos_rel = left_pos_abs - state_pos_l
    right_pos_rel = right_pos_abs - state_pos_r
    left_rot6d_abs = _geom.euler_to_matrix_zyx_6d_nb(left_rpy_abs)
    right_rot6d_abs = _geom.euler_to_matrix_zyx_6d_nb(right_rpy_abs)
    left_rot6d_rel = _geom.compute_delta_from_state_and_abs_rot(
        left_rot6d_abs, state_rot6d_l
    )
    right_rot6d_rel = _geom.compute_delta_from_state_and_abs_rot(
        right_rot6d_abs, state_rot6d_r
    )
    left_rpy_rel = _geom.compute_delta_from_state_and_abs_rot(
        left_rpy_abs, state_rpy_l
    )
    right_rpy_rel = _geom.compute_delta_from_state_and_abs_rot(
        right_rpy_abs, state_rpy_r
    )

    expected_dim = segments[-1][3] if segments else 0
    rel_chunk = np.zeros((n_action, expected_dim), dtype=np.float32)

    dispatch = {
        "left_ee_cartesian_pos_relative": left_pos_rel,
        "left_ee_cartesian_pos": left_pos_abs,
        "left_ee_rotation_6D_relative": left_rot6d_rel,
        "left_ee_rotation_6D": left_rot6d_abs,
        "left_ee_rotation_relative": left_rpy_rel,
        "left_ee_rotation": left_rpy_abs,
        "left_gripper": left_grip,
        "right_ee_cartesian_pos_relative": right_pos_rel,
        "right_ee_cartesian_pos": right_pos_abs,
        "right_ee_rotation_6D_relative": right_rot6d_rel,
        "right_ee_rotation_6D": right_rot6d_abs,
        "right_ee_rotation_relative": right_rpy_rel,
        "right_ee_rotation": right_rpy_abs,
        "right_gripper": right_grip,
    }

    for _, stripped, start, end in segments:
        seg = dispatch.get(stripped)
        if seg is None:
            continue
        rel_chunk[:, start:end] = seg.astype(np.float32)

    return rel_chunk, state_override


def strip_role_prefix(key: str) -> str:
    """Strip the follow_/master_ prefix, matching base_dataclass.save_action_data_with_key."""
    return key.replace("follow_", "").replace("master_", "")


def dof_segments_from_keys(
    predict_action_keys: List[str],
    dof_dims: Dict[str, int],
    train_dof_config: Optional[Dict[str, int]] = None,
) -> DofSegments:
    """Map predict_action_keys to (raw_key, stripped_key, start, end) segments.

    The action_padding width comes from train_config.dof_config; it is a
    virtual key that the abs/rel conversion skips while advancing the cursor.
    """
    train_dof = train_dof_config or {}
    segments: DofSegments = []
    cursor = 0
    for raw_key in predict_action_keys:
        if raw_key == "action_padding":
            dof_dim = int(train_dof.get("action_padding", 0))
            segments.append((raw_key, raw_key, cursor, cursor + dof_dim))
            cursor += dof_dim
            continue
        stripped = strip_role_prefix(raw_key)
        if stripped not in dof_dims:
            raise RuntimeError(
                f"RTC: unknown predict_action_key {raw_key!r} "
                f"(stripped={stripped!r}); not in dof_dims"
            )
        dof_dim = int(dof_dims[stripped])
        segments.append((raw_key, stripped, cursor, cursor + dof_dim))
        cursor += dof_dim
    return segments


def apply_state_override(
    obs: Dict, state_override: Dict[str, np.ndarray]
) -> Dict:
    """Return a new obs (shallow copy with merged state); ``obs`` is not modified.

    Only fields present in the override are replaced. With the default sensor
    reference the override equals obs.state, so this is a no-op (kept for the
    WALLX_USE_CHUNK_STATE ablation).
    """
    new_obs = dict(obs)
    new_state = dict(obs.get("state") or {})
    for k, v in state_override.items():
        new_state[k] = np.asarray(v)
    new_obs["state"] = new_state
    return new_obs


def attach_rtc_outputs(
    serialized: Dict[str, Any],
    infer_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Attach RTC protocol fields (action_pred and server_timing) to serialized actions.

    action_pred = concat(follow1_pos, follow2_pos, -1). The client stores it and
    sends it back as prev_action_chunk next round; without it the client never
    starts RTC. It is attached only in end-pose mode (follow1/2_pos of shape
    [*, 7]); joint mode does not support RTC.
    """
    if "follow1_pos" in serialized and "follow2_pos" in serialized:
        f1 = np.asarray(serialized["follow1_pos"], dtype=np.float32)
        f2 = np.asarray(serialized["follow2_pos"], dtype=np.float32)
        if (
            f1.ndim == 2
            and f2.ndim == 2
            and f1.shape == f2.shape
            and f1.shape[-1] == RTC_EE_PER_ARM_DIM
        ):
            serialized["action_pred"] = np.concatenate([f1, f2], axis=-1)
        else:
            logger.warning(
                "RTC: cannot attach action_pred — follow1_pos shape=%s, "
                "follow2_pos shape=%s (expected matching [H, %d])",
                f1.shape,
                f2.shape,
                RTC_EE_PER_ARM_DIM,
            )
    if infer_ms is not None:
        serialized["server_timing"] = {"infer_ms": float(infer_ms)}
    return serialized


__all__ = [
    "RTC_KEY_USE",
    "RTC_KEY_PREV_CHUNK",
    "RTC_KEY_DELAY",
    "RTC_KEY_NUM_STEPS",
    "RTC_KEY_NUM_FLOW_STEPS",
    "RTC_KEY_MODE",
    "RTC_EE_CHUNK_DIM",
    "RTC_EE_PER_ARM_DIM",
    "RTC_DEFAULT_MAX_RTC_STEPS",
    "USE_CHUNK_STATE_ENV",
    "get_training_max_rtc_steps",
    "clamp_rtc_delay",
    "parse_rtc_obs",
    "build_state_dict_from_abs_ref",
    "transform_abs_chunk_to_relative",
    "strip_role_prefix",
    "dof_segments_from_keys",
    "apply_state_override",
    "attach_rtc_outputs",
]
