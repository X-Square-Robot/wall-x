"""Device selection tests. They need no accelerator except the XPU parity test."""

import importlib.util
from unittest import mock

import pytest
import torch

from wall_x.model.qact.qwen2_5.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl import (
    QWEN2_5_VL_VISION_ATTENTION_CLASSES,
)
from wall_x.utils import device as dev


def _host(xpu=False, cuda=False):
    available = {"xpu": xpu, "cuda": cuda, "cpu": True}
    return mock.patch.object(dev, "is_available", lambda name: available[name])


def test_default_device_type_prefers_xpu_then_cuda_then_cpu():
    with _host(xpu=True, cuda=True):
        assert dev.default_device_type() == "xpu"
    with _host(cuda=True):
        assert dev.default_device_type() == "cuda"
    with _host():
        assert dev.default_device_type() == "cpu"


def test_resolve_device_explicit_wins():
    with _host(xpu=True):
        assert dev.resolve_device("cpu") == torch.device("cpu")
        assert dev.resolve_device("cuda:1") == torch.device("cuda", 1)
        assert dev.resolve_device("xpu") == torch.device("xpu", 0)
        assert dev.resolve_device(None) == torch.device("xpu", 0)
    with _host():
        assert dev.resolve_device("auto") == torch.device("cpu")


def test_vision_attention_is_flash_only_on_cuda():
    with _host(cuda=True):
        assert dev.vision_attn_implementation() == "flash_attention_2"
    with _host(xpu=True):
        assert dev.vision_attn_implementation() == "sdpa"
    with _host():
        assert dev.vision_attn_implementation() == "sdpa"


def test_float32_autocast_only_on_cuda():
    with mock.patch.object(torch, "autocast") as autocast:
        dev.float32_autocast("cuda")
        autocast.assert_called_once_with("cuda", dtype=torch.float32)
        autocast.reset_mock()
        with dev.float32_autocast("xpu"):
            pass
        autocast.assert_not_called()


@pytest.mark.skipif(
    importlib.util.find_spec("flash_attn") is not None,
    reason="flash_attn is installed",
)
def test_joint_attention_imports_without_flash_attn():
    from wall_x.model.core.attention import joint

    assert joint.flash_attn_func is None
    assert joint.JOINT_QWEN_ATTENTION_CLASSES["sdpa"] is joint.JointQwen2VLAttention


def _tiny_vision_attention(name, dtype):
    attn = QWEN2_5_VL_VISION_ATTENTION_CLASSES[name](
        Qwen2_5_VLConfig(), dim=32, num_heads=4
    )
    for param in attn.parameters():
        torch.nn.init.normal_(param, std=0.5)
    return attn.to(dtype).eval()


def _vision_inputs(dtype):
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(12, 32, generator=generator).to(dtype)
    angles = torch.randn(12, 4, generator=generator)
    emb = torch.cat((angles, angles), dim=-1)
    return hidden, torch.tensor([0, 5, 12]), (emb.cos(), emb.sin())


def _run_vision(attn, hidden, cu_seqlens, position_embeddings, device):
    with torch.no_grad():
        return (
            attn.to(device)(
                hidden.to(device),
                cu_seqlens=cu_seqlens.to(device),
                max_seqlen=7,
                position_embeddings=tuple(p.to(device) for p in position_embeddings),
            )
            .float()
            .cpu()
        )


def test_vision_sdpa_and_eager_accept_max_seqlen_and_agree():
    hidden, cu_seqlens, pos = _vision_inputs(torch.float32)
    sdpa = _tiny_vision_attention("sdpa", torch.float32)
    eager = _tiny_vision_attention("eager", torch.float32)
    eager.load_state_dict(sdpa.state_dict())
    out_sdpa = _run_vision(sdpa, hidden, cu_seqlens, pos, "cpu")
    out_eager = _run_vision(eager, hidden, cu_seqlens, pos, "cpu")
    assert out_sdpa.std() > 0
    torch.testing.assert_close(out_sdpa, out_eager, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not dev.is_available("xpu"), reason="XPU is not available")
def test_vision_sdpa_cpu_xpu_parity():
    hidden, cu_seqlens, pos = _vision_inputs(torch.bfloat16)
    attn = _tiny_vision_attention("sdpa", torch.bfloat16)
    cpu = _run_vision(attn, hidden, cu_seqlens, pos, "cpu")
    xpu = _run_vision(attn, hidden, cu_seqlens, pos, "xpu")
    assert cpu.std() > 0
    torch.testing.assert_close(xpu, cpu, atol=5e-2, rtol=5e-2)
