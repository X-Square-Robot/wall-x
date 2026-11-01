"""LeRobot layout: trailing-DOF truncation, relative pose, and prompt labels."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from wall_x.data.backends.lerobot.loader import (
    PreprocessedDataset,
    _truncate_to_active,
)
from wall_x.data.backends.lerobot.rotation_layout import maybe_convert_euler_to_6d
from wall_x.data.backends.lerobot.utils import preprocesser_call
from wall_x._vendor.harrix.infer.model_wrapper import (
    apply_fixed_zero_state,
    camera_prompt_label,
)

ARM_6D = {
    "follow_left_ee_cartesian_pos_relative": 3,
    "follow_left_ee_rotation_6D_relative": 6,
    "follow_left_gripper": 1,
    "follow_right_ee_cartesian_pos_relative": 3,
    "follow_right_ee_rotation_6D_relative": 6,
    "follow_right_gripper": 1,
    "action_padding": 6,
}


def _dataset(dof_config):
    dataset = PreprocessedDataset.__new__(PreprocessedDataset)
    dataset._dof_config = dof_config
    return dataset


def test_truncate_drops_only_the_unconfigured_tail():
    vec = torch.arange(26, dtype=torch.float32)
    out = _truncate_to_active(vec, ARM_6D)
    assert out.shape == (20,)
    assert torch.equal(out, vec[:20])


def test_relative_cartesian_and_absolute_gripper():
    dof = {
        "follow_left_ee_cartesian_pos_relative": 3,
        "follow_left_gripper": 1,
        "action_padding": 6,
    }
    action = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)
    state = np.array([0.0, 1.0, 1.0, 0.2], dtype=np.float32)
    out = _dataset(dof)._to_relative_action(action, state)
    np.testing.assert_allclose(out, [[1.0, 1.0, 2.0, 0.5]])


def test_relative_rotation_of_identical_pose_is_identity():
    layout = {"follow_left_ee_rotation_6D_relative": 6}
    state = maybe_convert_euler_to_6d(np.zeros(3, dtype=np.float64), layout, True)
    action = np.repeat(state[None, :], 1, axis=0)
    out = _dataset(layout)._to_relative_action(action, state)
    np.testing.assert_allclose(out[0], state, atol=1e-6)


def test_euler_vector_expands_to_6d_before_relative_stats():
    raw = np.zeros((4, 14), dtype=np.float64)
    converted = maybe_convert_euler_to_6d(raw, ARM_6D, True)
    assert converted.shape == (4, 20)

    spec = importlib.util.spec_from_file_location(
        "compute_norm_stats",
        Path(__file__).resolve().parents[1] / "scripts" / "compute_norm_stats.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    states, actions = module._prepare_arrays_for_layout(raw, raw, ARM_6D, ARM_6D)
    assert states.shape == (4, 20)
    assert actions.shape == (4, 20)


def test_wider_6d_vector_is_truncated_without_reinterpretation():
    spec = importlib.util.spec_from_file_location(
        "compute_norm_stats_wide",
        Path(__file__).resolve().parents[1] / "scripts" / "compute_norm_stats.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wide = np.arange(8 * 26, dtype=np.float64).reshape(8, 26)
    states, actions = module._prepare_arrays_for_layout(wide, wide, ARM_6D, ARM_6D)
    np.testing.assert_allclose(states, wide[:, :20])
    np.testing.assert_allclose(actions, wide[:, :20])


def test_state_string_replaces_propri_token():
    seen = {}

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def encode(self, text):
            return [3]

        def convert_tokens_to_ids(self, token):
            return 4

        def __call__(self, text, add_special_tokens=True, **kwargs):
            del add_special_tokens
            rows = text if isinstance(text, list) else [text]
            if kwargs.get("return_tensors") == "pt":
                seen["rows"] = list(rows)
            ids = torch.ones((len(rows), 3), dtype=torch.long)

            class _Batch(dict):
                @property
                def input_ids(self):
                    return self["input_ids"]

            return _Batch(input_ids=ids)

    class _Processor:
        tokenizer = _Tokenizer()

    norm_state = torch.zeros((1, 1, 2))
    mask = torch.ones((1, 1, 2))
    preprocesser_call(
        processor=_Processor(),
        text=["Proprioception: <|propri|>\n"],
        images=None,
        padding=True,
        return_tensors="pt",
        norm_state=norm_state,
        agent_pos_mask=mask,
        state_bins=256,
    )
    row = seen["rows"][0]
    assert "<|propri|>" not in row
    numbers = row.removeprefix("Proprioception: ").strip().split()
    assert numbers == ["128", "128"]


def test_camera_label_and_fixed_zero_state_follow_the_train_config():
    assert (
        camera_prompt_label(
            "face_view",
            {"face_view": "face_view"},
            {"face_view": "front view"},
        )
        == "face_view"
    )
    assert (
        camera_prompt_label("face_view", None, {"face_view": "front view"})
        == "front view"
    )
    pos = torch.ones(5)
    mask = torch.zeros(5)
    layout = {"follow_left_gripper": 1, "velocity_decomposed": 3, "height": 1}
    out, out_mask = apply_fixed_zero_state(
        pos.clone(), mask.clone(), layout, ["velocity_decomposed", "height"]
    )
    torch.testing.assert_close(out, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(out_mask, torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0]))
