import pytest
import torch

from wall_x.model.vga.modeling.wan_dit import sinusoidal_embedding_1d
from wall_x.model.vga.pipeline.scheduler import FlowMatchScheduler


def _make_training_scheduler():
    scheduler = FlowMatchScheduler(
        num_inference_steps=1000,
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def test_flow_match_scheduler_keeps_full_fp32_timestep_resolution():
    scheduler = _make_training_scheduler()

    assert scheduler.timesteps.dtype == torch.float32
    assert torch.unique(scheduler.timesteps).numel() == 1000
    assert torch.unique(scheduler.timesteps.to(torch.bfloat16)).numel() < 1000


def test_flow_match_scheduler_rejects_low_precision_timestep():
    scheduler = _make_training_scheduler()
    timestep = scheduler.timesteps[123].to(torch.bfloat16)

    with pytest.raises(ValueError, match="fp32"):
        scheduler.add_noise(torch.zeros(1), torch.ones(1), timestep)


def test_flow_match_scheduler_training_weight_has_no_exact_zero():
    scheduler = _make_training_scheduler()

    assert torch.all(scheduler.linear_timesteps_weights > 0)


def test_sinusoidal_embedding_accepts_fp32_timestep_and_returns_model_dtype():
    timestep = torch.linspace(0, 1000, 4, dtype=torch.float32)

    embedding = sinusoidal_embedding_1d(16, timestep, output_dtype=torch.bfloat16)

    assert timestep.dtype == torch.float32
    assert embedding.dtype == torch.bfloat16
    assert embedding.shape == (4, 16)
