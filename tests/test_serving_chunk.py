"""Chunk length vs. replan advance: the platform wants the whole chunk.

RoboDojo recipes carry ``action_chunk_length`` (what a response must contain)
and ``replan_steps`` (how far the client advances) as separate fields. The
plain policy keeps returning one replan's worth; keyframe-memory checkpoints
return the full prediction horizon, as the in-house service does.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from wall_x.serving.robodojo.memory_policy import KeyframeMemoryRoboDojoPolicy
from wall_x.serving.robodojo.policy import (
    RoboDojoPolicyConfig,
    WallXRoboDojoPolicy,
    resolve_action_steps,
)


class _Stub(WallXRoboDojoPolicy):
    """Only the slicing half of ``_actions`` is under test."""

    def __init__(self, steps: int, start: int = 0):
        self.config = RoboDojoPolicyConfig(
            checkpoint_path="/nonexistent", action_start_index=start
        )
        self.action_steps = steps

    def _absolute_arm(self, data, side):
        pose = np.zeros((40, 3))
        # Two orthonormal rows: a valid 6D rotation for every step.
        rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (40, 1))
        return pose, rot


def _payload() -> SimpleNamespace:
    return SimpleNamespace(
        data={f"action_{side}_gripper": np.zeros((40, 1)) for side in ("left", "right")}
    )


def test_plain_policy_keeps_one_replan_of_actions():
    policy = WallXRoboDojoPolicy
    assert policy.returns_full_chunk is False
    config = RoboDojoPolicyConfig(checkpoint_path="/nonexistent", replan_steps=16)
    assert resolve_action_steps(config, 32, full_chunk=False) == 16


def test_memory_policy_returns_the_full_horizon():
    assert KeyframeMemoryRoboDojoPolicy.returns_full_chunk is True
    config = RoboDojoPolicyConfig(checkpoint_path="/nonexistent", replan_steps=16)
    assert resolve_action_steps(config, 32, full_chunk=True) == 32
    actions = _Stub(32)._actions(_payload())
    assert len(actions) == 32
    assert set(actions[0]) == {
        "left_ee_pose",
        "right_ee_pose",
        "left_ee_joint_state",
        "right_ee_joint_state",
    }


def test_chunk_starts_at_the_start_index():
    actions = _Stub(16, start=8)._actions(_payload())
    assert len(actions) == 16


def test_explicit_chunk_length_wins():
    config = RoboDojoPolicyConfig(
        checkpoint_path="/nonexistent", replan_steps=16, action_chunk_length=24
    )
    assert resolve_action_steps(config, 32, full_chunk=True) == 24


@pytest.mark.parametrize(
    "config, horizon, message",
    [
        (
            RoboDojoPolicyConfig(
                checkpoint_path="/x", replan_steps=16, action_chunk_length=40
            ),
            32,
            "exceeds",
        ),
        (
            RoboDojoPolicyConfig(
                checkpoint_path="/x", replan_steps=40, action_chunk_length=32
            ),
            64,
            "replan_steps",
        ),
        (
            RoboDojoPolicyConfig(
                checkpoint_path="/x", action_start_index=24, action_chunk_length=16
            ),
            32,
            "exceeds",
        ),
        (
            RoboDojoPolicyConfig(checkpoint_path="/x", action_chunk_length=0),
            32,
            "positive",
        ),
    ],
)
def test_chunk_length_is_validated(config, horizon, message):
    with pytest.raises(ValueError, match=message):
        resolve_action_steps(config, horizon, full_chunk=True)
