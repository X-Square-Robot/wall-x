# Copyright (c) 2025, X-Square-Robot
# SPDX-License-Identifier: BSD-3-Clause

import os
from scipy.fft import dct
from scipy.fft import idct
import yaml
import torch
import numpy as np
import dataclasses
import copy
import json
from PIL import Image
from safetensors.torch import load_file
from qwen_vl_utils.vision_process import smart_resize
from transformers import BatchFeature, AutoProcessor

from wall_x.model.action_head import Normalizer
from wall_x.utils.constant import action_statistic_dof as default_action_statistic_dof
from numba import jit, prange

try:
    from spatial_tokenizer.spatial_tokenizer_kdisk import SpatialActionTokenizer
except ImportError:
    SpatialActionTokenizer = None

device = "cuda"

dof_config = {
    "follow_left_ee_cartesian_pos": 3,
    "follow_left_ee_rotation": 3,
    "follow_left_gripper": 1,
    "follow_right_ee_cartesian_pos": 3,
    "follow_right_ee_rotation": 3,
    "follow_right_gripper": 1,
    "head_actions": 2,
    "height": 1,
    "car_pose": 3,
    "velocity_decomposed": 3,
}
_CAM_NAME_MAPPING = {
    "face_view": "front view",
    "left_wrist_view": "left wrist view",
    "right_wrist_view": "right wrist view",
    "move1_view": "move view",
    "move2_view": "move view",
    "wall_view": "wall view",
    "top_view": "top view",
    "side_view": "side view",
    "global_view": "global view",
}
camera_to_view_mapping = {
    "camera_front": "face_view",
    "camera_left": "left_wrist_view",
    "camera_right": "right_wrist_view",
    "camera_side": "side_view",
    "camera_global": "global_view",
}

action_key_mapping = {
    "follow_left_ee_cartesian_pos": "follow1_pos[:3]",
    "follow_left_ee_rotation": "follow1_pos[3:6]",
    "follow_left_gripper": "follow1_pos[6:7]",
    "follow_right_ee_cartesian_pos": "follow2_pos[:3]",
    "follow_right_ee_rotation": "follow2_pos[3:6]",
    "follow_right_gripper": "follow2_pos[6:7]",
    "head_actions": "head_pos",
    "height": "lift",
    "velocity_decomposed": "velocity_decomposed",
    "follow_left_arm_joint_cur": "follow1_joints_cur[-1:]",
    "follow_right_arm_joint_cur": "follow2_joints_cur[-1:]",
    "follow_left_arm_joint_pos": "follow1_pos",
    "follow_right_arm_joint_pos": "follow2_pos",
}

dim_dof_config = {
    "right_xyz": {"rpy": (0, 3), "so3": (0, 3)},
    "right_rot": {"rpy": (3, 6), "so3": (3, 9)},
    "right_gripper": {"rpy": (6, 7), "so3": (9, 10)},
}
SINGLE_ARM_DIM = 7


@jit(nopython=True, parallel=True)
def euler_to_matrix_zyx_batch_nb(eulers):
    N = eulers.shape[0]
    R = np.empty((N, 3, 3), dtype=np.float64)
    for i in prange(N):
        roll = eulers[i, 0]
        pitch = eulers[i, 1]
        yaw = eulers[i, 2]

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        R[i, 0, 0] = cy * cp
        R[i, 0, 1] = cy * sp * sr - sy * cr
        R[i, 0, 2] = cy * sp * cr + sy * sr

        R[i, 1, 0] = sy * cp
        R[i, 1, 1] = sy * sp * sr + cy * cr
        R[i, 1, 2] = sy * sp * cr - cy * sr

        R[i, 2, 0] = -sp
        R[i, 2, 1] = cp * sr
        R[i, 2, 2] = cp * cr
    return R


@jit(nopython=True, parallel=True)
def compose_state_and_delta_to_abs_rpy(delta, state):
    """
    Input:
        delta: (N,3) -> Δrpy(ZYX) or (N,6) -> Δ6D (first two rows flattened)
        state: (3,) -> rpy(ZYX) or (6,) -> 6D (first two rows flattened)
    Output:
        abs_rpy: (N,3) Absolute pose rpy(ZYX, radians), normalized to (-π, π]
    """
    if delta.shape[-1] == 3:
        R_delta = euler_to_matrix_zyx_batch_nb(delta)  # (N,3,3)
    elif delta.shape[-1] == 6:
        R_delta = so3_to_matrix_batch_nb(delta)  # (N,3,3)
    else:
        raise ValueError(f"delta last dim must be 3 or 6, got {delta.shape[-1]}")

    if state.shape[-1] == 3:
        R_state = euler_to_matrix_zyx_batch_nb(state[np.newaxis, :])[0]  # (3,3)
    elif state.shape[-1] == 6:
        R_state = so3_to_matrix_batch_nb(state[np.newaxis, :])[0]  # (3,3)
    else:
        raise ValueError(f"state last dim must be 3 or 6, got {state.shape[-1]}")

    N = R_delta.shape[0]
    R_abs = np.empty((N, 3, 3), dtype=np.float64)

    S00 = R_state[0, 0]
    S01 = R_state[0, 1]
    S02 = R_state[0, 2]
    S10 = R_state[1, 0]
    S11 = R_state[1, 1]
    S12 = R_state[1, 2]
    S20 = R_state[2, 0]
    S21 = R_state[2, 1]
    S22 = R_state[2, 2]

    for i in prange(N):
        A00 = R_delta[i, 0, 0]
        A01 = R_delta[i, 0, 1]
        A02 = R_delta[i, 0, 2]
        A10 = R_delta[i, 1, 0]
        A11 = R_delta[i, 1, 1]
        A12 = R_delta[i, 1, 2]
        A20 = R_delta[i, 2, 0]
        A21 = R_delta[i, 2, 1]
        A22 = R_delta[i, 2, 2]

        R_abs[i, 0, 0] = A00 * S00 + A01 * S10 + A02 * S20
        R_abs[i, 0, 1] = A00 * S01 + A01 * S11 + A02 * S21
        R_abs[i, 0, 2] = A00 * S02 + A01 * S12 + A02 * S22

        R_abs[i, 1, 0] = A10 * S00 + A11 * S10 + A12 * S20
        R_abs[i, 1, 1] = A10 * S01 + A11 * S11 + A12 * S21
        R_abs[i, 1, 2] = A10 * S02 + A11 * S12 + A12 * S22

        R_abs[i, 2, 0] = A20 * S00 + A21 * S10 + A22 * S20
        R_abs[i, 2, 1] = A20 * S01 + A21 * S11 + A22 * S21
        R_abs[i, 2, 2] = A20 * S02 + A21 * S12 + A22 * S22


@jit(nopython=True, parallel=True)
def so3_to_matrix_batch_nb(batch_so3):
    N = batch_so3.shape[0]
    R_all = np.empty((N, 3, 3), dtype=np.float64)
    eps = 1e-12
    for i in prange(N):
        r1x, r1y, r1z = batch_so3[i, 0], batch_so3[i, 1], batch_so3[i, 2]
        r2x, r2y, r2z = batch_so3[i, 3], batch_so3[i, 4], batch_so3[i, 5]

        # normalize r1
        n1 = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z) + eps
        r1x /= n1
        r1y /= n1
        r1z /= n1

        # orthogonalize r2 to r1, then normalize
        dot12 = r1x * r2x + r1y * r2y + r1z * r2z
        r2x -= dot12 * r1x
        r2y -= dot12 * r1y
        r2z -= dot12 * r1z
        n2 = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z) + eps
        r2x /= n2
        r2y /= n2
        r2z /= n2

        # r3 = r1 x r2
        r3x = r1y * r2z - r1z * r2y
        r3y = r1z * r2x - r1x * r2z
        r3z = r1x * r2y - r1y * r2x

        R_all[i, 0, 0] = r1x
        R_all[i, 0, 1] = r1y
        R_all[i, 0, 2] = r1z
        R_all[i, 1, 0] = r2x
        R_all[i, 1, 1] = r2y
        R_all[i, 1, 2] = r2z
        R_all[i, 2, 0] = r3x
        R_all[i, 2, 1] = r3y
        R_all[i, 2, 2] = r3z
    return R_all


@jit(nopython=True, parallel=True)
def matrix_to_euler_zyx_batch_nb(Rs):
    """
    R = Rz(yaw) * Ry(pitch) * Rx(roll)
    extract:
      pitch = asin(-R[2,0])
      roll  = atan2(R[2,1], R[2,2])
      yaw   = atan2(R[1,0], R[0,0])
    """
    N = Rs.shape[0]
    eulers = np.empty((N, 3), dtype=np.float64)
    for i in prange(N):
        r00 = Rs[i, 0, 0]
        # r01 = Rs[i, 0, 1]
        # r02 = Rs[i, 0, 2]
        r10 = Rs[i, 1, 0]
        # r11 = Rs[i, 1, 1]
        # r12 = Rs[i, 1, 2]
        r20 = Rs[i, 2, 0]
        r21 = Rs[i, 2, 1]
        r22 = Rs[i, 2, 2]

        x = -r20
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0

        pitch = np.arcsin(x)
        roll = np.arctan2(r21, r22)
        yaw = np.arctan2(r10, r00)

        eulers[i, 0] = roll
        eulers[i, 1] = pitch
        eulers[i, 2] = yaw
    return eulers


@jit(nopython=True, parallel=True)
def canonicalize_euler_zyx_batch_nb(rpy_batch):
    """
    Batch ZYX Euler Angle Normalization (Parallel Version)
    Input: rpy_batch (N, 3) [roll, pitch, yaw] (radians)
    Output: out (N, 3) Constrained to the same branch with each component in (-π, π]
    Rules:
    1) First, wrap each component to (-π, π]
    2) If p > π/2: p = π - p; r += π; y += π
        If p <= -π/2: p = -π - p; r += π; y += π
    3) Finally, wrap each component to (-π, π] again.
    """
    N = rpy_batch.shape[0]
    out = np.empty_like(rpy_batch)
    two_pi = 2.0 * np.pi

    for i in prange(N):
        r = rpy_batch[i, 0]
        p = rpy_batch[i, 1]
        y = rpy_batch[i, 2]

        r = (r + np.pi) % two_pi - np.pi
        p = (p + np.pi) % two_pi - np.pi
        y = (y + np.pi) % two_pi - np.pi

        if p > np.pi / 2.0:
            p = np.pi - p
            r = r + np.pi
            y = y + np.pi
        elif p <= -np.pi / 2.0:
            p = -np.pi - p
            r = r + np.pi
            y = y + np.pi

        r = (r + np.pi) % two_pi - np.pi
        p = (p + np.pi) % two_pi - np.pi
        y = (y + np.pi) % two_pi - np.pi

        out[i, 0] = r
        out[i, 1] = p
        out[i, 2] = y

    return out


def so3_to_euler_zyx_batch_nb(batch_so3):
    matrix = so3_to_matrix_batch_nb(batch_so3)
    eulers = matrix_to_euler_zyx_batch_nb(matrix)
    return canonicalize_euler_zyx_batch_nb(eulers)


def update_model_config(train_config, model_config):
    model_config.use_state_string_representation = train_config["data"].get(
        "use_state_string_representation", False
    )
    model_config.flow_loss_weight = train_config.get("flow_loss_weight", 1.0)

    model_config.dof_config = train_config["dof_config"]
    model_config.agent_pos_config = train_config["agent_pos_config"]

    model_config.action_horizon_flow = train_config["data"].get(
        "action_horizon_flow", 32
    )

    if train_config.get("_attn_implementation", None) is not None:
        model_config._attn_implementation = train_config["_attn_implementation"]

    return model_config


def move_to_cuda(obj, device=device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (dict, BatchFeature)):
        return {k: move_to_cuda(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cuda(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cuda(v, device) for v in obj)
    else:
        return obj


def extract_components(data, config, is_rpy):
    mode = "rpy" if is_rpy else "so3"
    result = {}
    for key, slice_config in config.items():
        slice_range = slice_config[mode]
        result[key] = data[:, slice_range[0] : slice_range[1]]
    return result


@dataclasses.dataclass
class WallxInferArgs:
    config_path: str | None = None
    checkpoint_path: str | None = None
    action_mode: str = "diffusion"  # "ar" or "diffusion"

    max_time_step: int = 1000
    action_start_ratio: float = 0
    action_end_ratio: float = 0.6

    model_action_dim: int = 14
    action_horizon: int = 32

    action_dim: int = 14

    interpolate_action: bool = False
    interpolate_multiplier: int = 1
    turtle_as_desktop: bool = False
    generate_subtask: bool = False
    subtask_interval: int = 0
    with_cur: bool = False
    state_str: bool = True  ### NOTE
    wostate: bool = False
    delta_action: bool = False
    state_rpy: bool = True
    action_rpy: bool = True

    dataset_name: str = "robochallenge_aloha"
    use_hard_prompt: bool = True
    dct_scale: float = -1


class WallxModelWrapper:
    def __init__(self, args: WallxInferArgs):
        self.args = args
        self.get_model_and_processor()
        self.action_predict_mode = "ar" if args.action_mode == "ar" else "diffusion"
        print("action_predict_mode", self.action_predict_mode, flush=True)

    def get_model_and_processor(self):
        if self.args.config_path is None:
            self.args.config_path = os.path.join(
                self.args.checkpoint_path, "config.yml"
            )
        with open(self.args.config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.config = config
        self.dof_config = config["dof_config"]
        self.agent_pos_config = config["agent_pos_config"]
        self.obs_action_keys = [
            "follow_left_ee_cartesian_pos",
            "follow_left_ee_rotation",
            "follow_left_gripper",
            "follow_right_ee_cartesian_pos",
            "follow_right_ee_rotation",
            "follow_right_gripper",
        ]
        print("obs_action_keys", self.obs_action_keys, flush=True)
        self.action_tokenizer_type = config.get("action_tokenizer_type", None)
        config_path = config["qwen_vl_act_config_path"]
        self.processor = AutoProcessor.from_pretrained(
            config["processor_path"], use_fast=True
        )
        self.processor.tokenizer.padding_side = "left"
        new_tokens = ["<|propri|>", "<|action|>"]

        # load fast tokenizer
        self.action_tokenizer_type = config.get("action_tokenizer_type", None)
        print("self.action_tokenizer_type", self.action_tokenizer_type, flush=True)
        self.action_tokenizer = None
        self.action_mapper = None
        if self.action_tokenizer_type:
            # fast
            if self.action_tokenizer_type == "fast":
                print("Using fast tokenizer")
                self.action_tokenizer = AutoProcessor.from_pretrained(
                    config["action_tokenizer_path"], trust_remote_code=True
                )
            elif self.action_tokenizer_type == "spatialvla":
                print("Using spatialvla tokenizer")
                assert (
                    SpatialActionTokenizer is not None
                ), "SpatialActionTokenizer is not installed"
                self.action_tokenizer = SpatialActionTokenizer()
            else:
                raise ValueError(
                    f"Unsupported action tokenizer type: {self.action_tokenizer_type}"
                )
            new_tokens += [
                f"<|action_token_{i}|>" for i in range(self.action_tokenizer.vocab_size)
            ]

        # num_added_tokens = self.processor.tokenizer.add_tokens(new_tokens)

        # define action mapper
        if self.action_tokenizer_type:
            self.action_mapper = {}
            for i in range(self.action_tokenizer.vocab_size):
                token = f"<|action_token_{i}|>"
                token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
                self.action_mapper[token_id] = i

        # action & propri normalizer
        self._register_normalizers()

        model_type = config["model_type"]

        if model_type == "qwen2_5":
            print("Using qwen2_5 model as base model")
            from wall_x.model.qwen2_5_based import (
                Qwen2_5_VLMoEForAction,
                Qwen2_5_VLConfig,
            )

            ModelClass = Qwen2_5_VLMoEForAction
            ConfigClass = Qwen2_5_VLConfig

        model_config = ConfigClass.from_pretrained(config_path)
        model_config = update_model_config(config, model_config)

        print("model_config", model_config, flush=True)

        # if self.args.action_mode == "ar":
        #     model_config._attn_implementation = "flash_attention_2"
        # else
        model_config._attn_implementation = "sdpa"
        model_config.vision_config._attn_implementation = "flash_attention_2"

        if model_config.model_type == "qwen3_vl":
            self.MAX_PIXELS = 16384 * 32 * 32
            self.MIN_PIXELS = 4 * 32 * 32
            self.IMAGE_FACTOR = 32
        elif model_config.model_type == "qwen2_5_vl":
            self.MAX_PIXELS = 16384 * 28 * 28
            self.MIN_PIXELS = 4 * 28 * 28
            self.IMAGE_FACTOR = 28

        model = ModelClass(
            model_config,
            self.action_tokenizer_type,
            self.processor,
            self.action_tokenizer,
            self.action_mapper,
        )
        model.resize_token_embeddings(len(self.processor.tokenizer))
        state_dict = load_file(
            self.args.checkpoint_path + "/model.safetensors", device="cpu"
        )
        if os.path.exists(os.path.join(self.args.checkpoint_path, "global_step.pth")):
            global_step = torch.load(
                os.path.join(self.args.checkpoint_path, "global_step.pth")
            )["global_step"]
            print("Loaded global step:", global_step)
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        model.eval()
        model.set_normalizer(
            copy.deepcopy(self.normalizer_action), copy.deepcopy(self.normalizer_propri)
        )
        model.to(device)
        model.to_bfloat16_for_selected_params()

        self.model = model
        print("self.args.dataset_name", self.args.dataset_name, flush=True)
        print(
            "normalizer_action min",
            self.normalizer_action.min.__getattr__(self.args.dataset_name),
            flush=True,
        )
        print(
            "normalizer_action delta",
            self.normalizer_action.delta.__getattr__(self.args.dataset_name),
            flush=True,
        )

    def _register_normalizers(self):
        if self.config.get("customized_action_statistic_dof", None):
            action_statistic_dof = json.load(
                open(self.config["customized_action_statistic_dof"], "r")
            )
        else:
            action_statistic_dof = default_action_statistic_dof

        if os.path.exists(self.args.checkpoint_path + "/normalizer_action.pth"):
            print(
                "Loading normalizer_action from checkpoint",
                self.args.checkpoint_path + "/normalizer_action.pth",
                flush=True,
            )
            self.normalizer_action = Normalizer.from_ckpt(
                self.args.checkpoint_path + "/normalizer_action.pth"
            )
        else:
            self.normalizer_action = Normalizer(
                action_statistic_dof,
                self.config["dof_config"],
                min_key=self.config.get("min_key", "min"),
                delta_key=self.config.get("delta_key", "delta"),
            )

        # print("action_statistic_dof",action_statistic_dof)

        if os.path.exists(self.args.checkpoint_path + "/normalizer_propri.pth"):
            print(
                "Loading normalizer_propri from checkpoint",
                self.args.checkpoint_path + "/normalizer_propri.pth",
                flush=True,
            )
            self.normalizer_propri = Normalizer.from_ckpt(
                self.args.checkpoint_path + "/normalizer_propri.pth"
            )
        else:
            self.normalizer_propri = Normalizer(
                action_statistic_dof,
                self.config["agent_pos_config"],
                min_key=self.config.get("min_key", "min"),
                delta_key=self.config.get("delta_key", "delta"),
            )

        print("self.args.dataset_name", self.args.dataset_name, flush=True)
        print(
            "normalizer_propri min",
            self.normalizer_propri.min.__getattr__(self.args.dataset_name),
            flush=True,
        )
        print(
            "normalizer_propri delta",
            self.normalizer_propri.delta.__getattr__(self.args.dataset_name),
            flush=True,
        )
        print(
            "normalizer_action min",
            self.normalizer_action.min.__getattr__(self.args.dataset_name),
            flush=True,
        )
        print(
            "normalizer_action delta",
            self.normalizer_action.delta.__getattr__(self.args.dataset_name),
            flush=True,
        )

    def get_text_ar(self, instruction, camera_names, norm_state=None, state_mask=None):
        role_start_symbol = "<|im_start|>"
        role_end_symbol = "<|im_end|>"
        vision_start_symbol = "<|vision_start|>"
        vision_end_symbol = "<|vision_end|>"
        image_pad_symbol = "<|image_pad|>"
        propri_symbol = "<|propri|>"

        prologue = f"{role_start_symbol}system\nYou are a helpful assistant.{role_end_symbol}\n"
        user_request = f"{role_start_symbol}user\nObservation:"
        print("camera_names", camera_names, flush=True)
        for cam_name in camera_names:
            user_request += f" {_CAM_NAME_MAPPING[cam_name]}: {vision_start_symbol}{image_pad_symbol}{vision_end_symbol}"
        user_request += "\nInstruction:"
        if self.args.state_str:
            assert norm_state is not None
            if isinstance(norm_state, torch.Tensor):
                if state_mask is not None:
                    if isinstance(state_mask, torch.Tensor):
                        mask_1d = state_mask[0, 0].to(
                            dtype=torch.bool, device=norm_state.device
                        )
                    else:
                        mask_1d = torch.as_tensor(state_mask, device=norm_state.device)[
                            0, 0
                        ].to(dtype=torch.bool)

                    norm_state = norm_state[..., mask_1d]

                norm_state = norm_state.detach().cpu().numpy()
                print("norm_state", norm_state, flush=True)
            discretized_state = (
                np.digitize(norm_state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            )
            propri = " ".join(map(str, discretized_state[0, 0]))
        elif self.args.wostate:
            propri = ""
        else:
            propri = propri_symbol
        text_prompt = (
            f"\nPredict the next action in robot action.\nProprioception: {propri}\n"
        )
        user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"
        assistant_message = f"{role_start_symbol}assistant\n"
        text = prologue + user_message + assistant_message

        return text

    def get_text_flow(
        self,
        instruction,
        camera_names,
        action_chunk_size,
        norm_state=None,
        state_mask=None,
    ):
        role_start_symbol = "<|im_start|>"
        role_end_symbol = "<|im_end|>"
        vision_start_symbol = "<|vision_start|>"
        vision_end_symbol = "<|vision_end|>"
        image_pad_symbol = "<|image_pad|>"
        propri_symbol = "<|propri|>"
        action_symbol = "<|action|>"
        action_space = "Rel EEF" if self.args.delta_action else "Abs EEF"
        _camera = ", ".join([_CAM_NAME_MAPPING[cam_name] for cam_name in camera_names])
        prologue = f"<|im_start|>system\nYou are an embodied vision-language-action (VLA) model controlling the robot with language instructions.\n Embodiment: {self.args.dataset_name.split('_')[-1]}\n Camera Setup: {_camera},\n Frequency: 32HZ\n Action Space: {action_space}\n<|im_end|>\n"

        user_request = f"{role_start_symbol}user\nObservation:"
        print("camera_names", camera_names, flush=True)
        for cam_name in camera_names:
            user_request += f" {_CAM_NAME_MAPPING[cam_name]}: {vision_start_symbol}{image_pad_symbol}{vision_end_symbol}"
        user_request += "\nInstruction:"
        if self.args.state_str:
            assert norm_state is not None
            if isinstance(norm_state, torch.Tensor):
                if state_mask is not None:
                    if isinstance(state_mask, torch.Tensor):
                        mask_1d = state_mask[0, 0].to(
                            dtype=torch.bool, device=norm_state.device
                        )
                    else:
                        mask_1d = torch.as_tensor(state_mask, device=norm_state.device)[
                            0, 0
                        ].to(dtype=torch.bool)
                    norm_state = norm_state[..., mask_1d]
                norm_state = norm_state.detach().cpu().numpy()
            discretized_state = (
                np.digitize(norm_state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            )
            propri = " ".join(map(str, discretized_state[0, 0]))
        elif self.args.wostate:
            propri = ""
        else:
            propri = propri_symbol
        text_prompt = (
            f"\nPredict the next action in robot action.\nProprioception: {propri}\n"
        )
        user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"
        assistant_message = f"{role_start_symbol}assistant\n"
        action = f"{action_symbol * action_chunk_size}"
        text = prologue + user_message + assistant_message + action

        return text

    def resize_images(self, observation):
        image_inputs = []
        view_candidates = [
            "face_view",
            "left_wrist_view",
            "right_wrist_view",
            "side_view",
            "global_view",
        ]
        for key in observation.keys():
            if key not in view_candidates:
                print("!!! key not in view_candidates", key, flush=True)
                continue
            # 1. Get the original image
            current_obs = observation[key]
            img_pil = Image.fromarray(current_obs)
            orig_width, orig_height = img_pil.size

            # 2. Apply resolution limits (if the configuration is not -1)
            target_size = 256
            if target_size != -1:
                # Logic for maintaining aspect ratio constraints
                if orig_width > orig_height:
                    new_width = target_size
                    new_height = int(target_size * orig_height / orig_width)
                else:
                    new_height = target_size
                    new_width = int(target_size * orig_width / orig_height)
                img_pil = img_pil.resize((new_width, new_height))

            # 3. Apply intelligent scaling
            current_width, current_height = img_pil.size
            resized_height, resized_width = smart_resize(
                current_height,
                current_width,
                factor=self.IMAGE_FACTOR,
                min_pixels=self.MIN_PIXELS,
                max_pixels=self.MAX_PIXELS,
            )
            resized_img = img_pil.resize((resized_width, resized_height))
            print("resized_img", resized_img.size, flush=True)

            image_inputs.append(resized_img)

        return image_inputs

    def _construct_input(
        self,
        observation,
        instruction,
        camera_names,
        valid_action_dim=7,
        mode="ar",
        single_image=False,
    ):
        additional_inputs = {}

        agent_pos = torch.from_numpy(observation["agent_pos"])
        agent_pos_mask = torch.from_numpy(observation["agent_pos_mask"])
        dof_mask = torch.from_numpy(observation["dof_mask"])
        additional_inputs["dof_mask"] = dof_mask
        print("before normalizing agent_pos", agent_pos, flush=True)

        if self.normalizer_propri is not None:

            agent_pos = self.normalizer_propri.normalize_data(
                agent_pos, [self.args.dataset_name]
            )
            additional_inputs["proprioception"] = agent_pos
            additional_inputs["agent_pos_mask"] = agent_pos_mask

            print(
                f"normalizing agent_pos: {agent_pos}, {self.args.dataset_name}",
                flush=True,
            )
        print("agent_pos_mask", agent_pos_mask, flush=True)
        print("dof_mask", dof_mask[0, 0], flush=True)
        if mode == "ar":
            text = self.get_text_ar(
                instruction, camera_names, agent_pos, agent_pos_mask
            )
        elif mode == "diffusion":
            text = self.get_text_flow(
                instruction,
                camera_names,
                self.args.action_horizon,
                agent_pos,
                agent_pos_mask,
            )
        elif mode == "subtask":
            text = self.get_text_subtask(instruction, single_view=single_image)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        text = [text]

        image_inputs = self.resize_images(observation)
        if single_image:
            image_inputs = [
                image_inputs[0]
            ]  # single view subtask/vqa use head view only
        image_inputs = self.processor.image_processor(
            images=image_inputs, videos=None, return_tensors="pt"
        )
        image_grid_thw = image_inputs["image_grid_thw"]
        # Processing image placeholder tokens in the text
        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while "<|image_pad|>" in text[i]:
                    # Replace image placeholders with actual quantities.
                    text[i] = text[i].replace(
                        "<|image_pad|>",
                        "<|placeholder|>"
                        * (image_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")

        text_inputs = self.processor.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        inputs = BatchFeature(data={**text_inputs, **image_inputs})

        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")
        additional_inputs["moe_token_types"] = inputs.input_ids == action_token_id
        additional_inputs["dataset_names"] = [self.args.dataset_name]

        inputs.update(additional_inputs)
        inputs = move_to_cuda(inputs, device)
        print("inputs", inputs.keys(), flush=True)
        for k in inputs.keys():
            if isinstance(inputs[k], torch.Tensor):
                print(k, inputs[k].shape, flush=True)
        return inputs

    def preprocess(self, state, views, valid_action_dim):
        # print("state",state, flush=True)
        # state: dict, keys: follow1_pos, follow2_pos
        model_action_dim = sum(self.dof_config.values())

        if valid_action_dim not in (SINGLE_ARM_DIM, 2 * SINGLE_ARM_DIM):
            raise ValueError(
                f"Invalid valid_action_dim: {valid_action_dim}, expect 7 or 14"
            )

        # 1) First, prepare a container of size (1, 1, D), where D = model_action_dim.
        agent_data = np.zeros((1, 1, model_action_dim), dtype=np.float32)
        agent_pos_mask = np.zeros((1, 1, model_action_dim), dtype=np.float32)
        dof_mask = np.zeros(
            (1, self.args.action_horizon, model_action_dim), dtype=np.float32
        )

        # 2) Determine the interval to be filled [start:end)
        if valid_action_dim == SINGLE_ARM_DIM:
            start = 0 if model_action_dim == SINGLE_ARM_DIM else SINGLE_ARM_DIM
            end = start + SINGLE_ARM_DIM

            if end > model_action_dim:
                raise ValueError(
                    f"model_action_dim={model_action_dim} too small for valid_action_dim=7 "
                    f"(need end={end})"
                )

            follow2 = np.asarray(state["follow2_pos"], dtype=np.float32).reshape(
                1, 1, SINGLE_ARM_DIM
            )
            agent_data[:, :, start:end] = follow2
            agent_pos_mask[:, :, start:end] = 1
            dof_mask[:, :, start:end] = 1

        else:  # valid_action_dim == 14
            end = 2 * SINGLE_ARM_DIM
            if end > model_action_dim:
                raise ValueError(
                    f"model_action_dim={model_action_dim} too small for valid_action_dim=14"
                )

            follow1 = np.asarray(state["follow1_pos"], dtype=np.float32).reshape(
                1, 1, SINGLE_ARM_DIM
            )
            follow2 = np.asarray(state["follow2_pos"], dtype=np.float32).reshape(
                1, 1, SINGLE_ARM_DIM
            )
            agent_data[:, :, :end] = np.concatenate([follow1, follow2], axis=-1)
            agent_pos_mask[:, :, :end] = 1
            dof_mask[:, :, :end] = 1

        observation = {
            camera_to_view_mapping[key]: views[key][0] for key in views.keys()
        }
        observation["agent_pos"] = agent_data
        observation["agent_pos_mask"] = agent_pos_mask
        observation["dof_mask"] = dof_mask
        return observation

    def model_output_process(self, action_pred, state):

        if not self.args.delta_action and self.args.action_rpy:
            return action_pred

        pred_components = extract_components(
            action_pred, dim_dof_config, self.args.action_rpy
        )

        pred_right_xyz = pred_components["right_xyz"]
        pred_right_rot = pred_components["right_rot"]
        pred_right_gripper = pred_components["right_gripper"]
        if "left_xyz" in pred_components:
            pred_left_xyz = pred_components["left_xyz"]
            pred_left_rot = pred_components["left_rot"]
            pred_left_gripper = pred_components["left_gripper"]
        else:
            pred_left_xyz = np.zeros((self.args.action_horizon, 3))
            pred_left_rot = np.zeros((self.args.action_horizon, 3))
            pred_left_gripper = np.zeros((self.args.action_horizon, 1))

        post_action_pred = np.zeros((self.args.action_horizon, self.args.action_dim))
        if self.args.delta_action:
            assert (
                self.args.action_dim == 14
            ), "Delta robot support and testing are not yet available."

            state_components = extract_components(
                state, dim_dof_config, self.args.state_rpy
            )
            left_xyz = state_components["left_xyz"]
            left_rot = state_components["left_rot"]
            right_xyz = state_components["right_xyz"]
            right_rot = state_components["right_rot"]

            post_action_pred[:, :3] = pred_left_xyz + left_xyz
            post_action_pred[:, 3:6] = compose_state_and_delta_to_abs_rpy(
                pred_left_rot, left_rot[0]
            )
            post_action_pred[:, 6:7] = pred_left_gripper
            post_action_pred[:, 7:10] = pred_right_xyz + right_xyz
            post_action_pred[:, 10:13] = compose_state_and_delta_to_abs_rpy(
                pred_right_rot, right_rot[0]
            )
            post_action_pred[:, 13:14] = pred_right_gripper

        elif not self.args.action_rpy:
            post_action_pred[:, :3] = pred_left_xyz
            post_action_pred[:, 3:6] = so3_to_euler_zyx_batch_nb(pred_left_rot)
            post_action_pred[:, 6:7] = pred_left_gripper
            post_action_pred[:, 7:10] = pred_right_xyz
            post_action_pred[:, 10:13] = so3_to_euler_zyx_batch_nb(pred_right_rot)
            post_action_pred[:, 13:14] = pred_right_gripper
        else:
            post_action_pred = action_pred
        return post_action_pred

    def postprocess(self, action_pred, interpolate_multiplier=None):

        if interpolate_multiplier is None:
            interpolate_multiplier = self.args.interpolate_multiplier

        if isinstance(action_pred, torch.Tensor):
            action_pred = action_pred.to(torch.float32).cpu().squeeze(0).numpy()
        left_action_pred = action_pred[:, :7]  # (32, 7)
        right_action_pred = action_pred[:, 7:14]  # (32, 7)

        start_frame = int(self.args.action_start_ratio * len(left_action_pred))
        end_frame = int(self.args.action_end_ratio * len(left_action_pred))
        left_action_pred = left_action_pred[start_frame:end_frame]
        right_action_pred = right_action_pred[start_frame:end_frame]

        print("left_action_pred", left_action_pred[-1], flush=True)
        print("right_action_pred", right_action_pred[-1], flush=True)

        left_action_pred = left_action_pred.tolist()
        right_action_pred = right_action_pred.tolist()

        serialized_actions = {
            "follow1_pos": left_action_pred,
            "follow2_pos": right_action_pred,
            ## for joint-control
            # "follow1_joints":left_action_pred,
            # "follow2_joints":right_action_pred,
        }

        return serialized_actions

    def predict_action_rtc(
        self,
        state,
        views,
        instruction=None,
        valid_action_dim=7,
        update_subtask=False,
        action_predict_mode=None,
    ):
        if action_predict_mode is not None:
            self.action_predict_mode = action_predict_mode

        observation = self.preprocess(state, views, valid_action_dim)
        print("use instruction", instruction, flush=True)
        camera_names = [camera_to_view_mapping[key] for key in views.keys()]
        # camera_names = ["right_wrist_view", "global_view", "side_view"]
        print("mode:", self.action_predict_mode, flush=True)
        inputs = self._construct_input(
            observation,
            instruction,
            camera_names=camera_names,
            valid_action_dim=valid_action_dim,
            mode=self.action_predict_mode,
        )
        model_action_dim = sum(self.dof_config.values())
        padding = torch.zeros((1, model_action_dim))
        norm_padding = self.normalizer_action.normalize_data(
            padding, [self.args.dataset_name]
        )
        inputs["padding_action"] = norm_padding
        inputs = move_to_cuda(inputs, device="cuda:0")
        agent_data = observation["agent_pos"][..., : self.args.model_action_dim][0]
        print("before generate_flow_action", flush=True)
        print(inputs.keys(), flush=True)
        print(inputs["dataset_names"], flush=True)
        print(self.processor.tokenizer.decode(inputs["input_ids"][0]), flush=True)
        print(inputs["agent_pos_mask"][0], flush=True)
        print(inputs["dof_mask"][0, 0], flush=True)
        if self.action_predict_mode == "ar":
            action_pred = self.generate_ar_action(inputs)
        else:
            action_pred = self.generate_flow_action(inputs)
        print("after generate_flow_action", flush=True)
        if isinstance(action_pred, torch.Tensor):
            action_pred = action_pred.float().cpu().squeeze(0).numpy()

        if action_pred is None:
            return None

        if self.args.dct_scale > 0:
            scale = self.args.dct_scale
            dct_coeff = dct(action_pred, axis=0, norm="ortho")
            dct_coeff = np.around(dct_coeff * scale)
            action_pred = idct(dct_coeff / scale, axis=0, norm="ortho")

        if action_pred.shape[-1] == 7:
            print("Before concat action_pred", action_pred.shape, flush=True)
            right_action_pred = action_pred
            left_action_pred = np.zeros_like(right_action_pred)
            action_pred = np.concatenate([left_action_pred, right_action_pred], axis=1)
            print("After concat action_pred", action_pred.shape, flush=True)
            # unnorm action_pred
        # print("action_pred", action_pred[:, 3], flush=True)
        action_pred = (
            self.normalizer_action.unnormalize_data(
                torch.tensor(action_pred).unsqueeze(0), [self.args.dataset_name]
            )
            .squeeze(0)
            .numpy()
        )
        print("After unnormalize_data action_pred", action_pred.shape, flush=True)

        action_pred = self.model_output_process(action_pred, agent_data)
        action_pred = self.postprocess(action_pred)

        return action_pred

    def generate_flow_action(
        self,
        inputs,
        last_action_chunk=None,
        max_guidance_weight=20.0,
        num_inference_timesteps=10,
        sigma_action=0.2,
    ):
        model_action_dim = sum(self.dof_config.values())
        if last_action_chunk is None:
            output = self.model.generate_flow_action(
                action_horizon=self.args.action_horizon,
                action_dim=model_action_dim,
                num_inference_timesteps=num_inference_timesteps,
                unnorm=False,
                **inputs,
            )
        else:
            output = self.model.generate_flow_action_rtc(
                action_horizon=self.args.action_horizon,
                action_dim=model_action_dim,
                num_inference_timesteps=num_inference_timesteps,
                inference_delay=self.args.rtc_inference_delay,
                execution_horizon=self.args.rtc_execution_horizon - 1,
                max_guidance_weight=max_guidance_weight,
                last_action_chunk=last_action_chunk,
                sigma_action=sigma_action,
                unnorm=False,
                **inputs,
            )
        action_pred = output["predict_action"]  # (b, action_horizon, action_dim)
        return action_pred

    def generate_text(self, inputs):
        return self.model.generate_text(**inputs)

    def _preprocess_ar_batch(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        moe_token_types = batch["moe_token_types"]
        labels = batch.get("labels", None)
        prefix_length = batch.get("prefix_length", None)

        generation_prompt_ids = torch.tensor(
            [151644, 77091], device=input_ids.device, dtype=input_ids.dtype
        )  # <|im_start|>assistant
        matches = (input_ids[0, :-1] == generation_prompt_ids[0]) & (
            input_ids[0, 1:] == generation_prompt_ids[1]
        )
        if matches.any():
            split_pos = torch.nonzero(matches, as_tuple=True)[0][0].item()
            # construct output ids
            gt_output_ids = input_ids[:, split_pos + 3 : prefix_length]
            # remove output part from input
            input_ids = input_ids[:, : split_pos + 3]
            moe_token_types = moe_token_types[:, : split_pos + 3]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : split_pos + 3]
            if labels is not None:
                labels = labels[:, split_pos + 3 : prefix_length]

        batch.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "moe_token_types": moe_token_types,
                "labels": labels,
                "gt_output_ids": gt_output_ids,
                "prefix_length": split_pos + 3,
            }
        )

        return batch

    def generate_ar_action(self, inputs):
        # batch = self._preprocess_ar_batch(batch=inputs)
        action_pred = None
        count = 0
        while action_pred is None:
            if count > 5:
                # raise ValueError("re-generate ar action failed")
                return None
            count += 1
            output = self.model.generate_ar_action(
                # action_dim=args.action_dim,
                action_dim=14,
                action_horizon=self.args.action_horizon,
                unnorm=False,
                **inputs,
            )
            action_pred = output["predict_action"]

        action_pred = action_pred[0]
        return action_pred


class WallxInfer:
    def __init__(self, args: WallxInferArgs):
        self.args = args
        self.model_wrapper = WallxModelWrapper(args)

    def run_infer_robochallenge(
        self, state, views, instruction, valid_action_dim=7, action_predict_mode=None
    ):
        action_pred = self.model_wrapper.predict_action_rtc(
            state=state,
            views=views,
            instruction=instruction,
            valid_action_dim=valid_action_dim,
            action_predict_mode=action_predict_mode,
        )
        return action_pred


if __name__ == "__main__":
    args = WallxInferArgs()
    Infer = WallxInfer(args)
    Infer.run_infer()
