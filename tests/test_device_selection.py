"""Device selection and non-flash vision attention tests."""

from unittest import mock

import pytest
import torch

from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionSdpaAttention,
)
from wall_x.utils.device import default_vision_attn_implementation, get_default_device


def test_default_device_order_is_cuda_then_xpu_then_cpu():
    with mock.patch("torch.cuda.is_available", return_value=True):
        assert get_default_device().type == "cuda"
        assert default_vision_attn_implementation() == "flash_attention_2"
    with mock.patch("torch.cuda.is_available", return_value=False):
        with mock.patch("torch.xpu.is_available", return_value=True):
            assert get_default_device().type == "xpu"
        with mock.patch("torch.xpu.is_available", return_value=False):
            assert get_default_device().type == "cpu"
        assert default_vision_attn_implementation() == "sdpa"


def _vision_attention_pair(dim=64, heads=4, seq=10):
    torch.manual_seed(0)
    eager = Qwen2_5_VLVisionAttention(None, dim, num_heads=heads)
    sdpa = Qwen2_5_VLVisionSdpaAttention(None, dim, num_heads=heads)
    sdpa.load_state_dict(eager.state_dict())
    angles = torch.randn(seq, dim // heads // 2)
    emb = torch.cat((angles, angles), dim=-1)
    inputs = {
        "hidden_states": torch.randn(seq, dim),
        "cu_seqlens": torch.tensor([0, 4, seq], dtype=torch.int32),
        "max_seqlen": seq - 4,
        "position_embeddings": (emb.cos(), emb.sin()),
    }
    return eager, sdpa, inputs


def test_vision_sdpa_attention_accepts_max_seqlen_and_matches_eager():
    eager, sdpa, inputs = _vision_attention_pair()
    with torch.no_grad():
        torch.testing.assert_close(sdpa(**inputs), eager(**inputs))


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU is not available",
)
def test_vision_sdpa_attention_on_xpu_matches_cpu():
    eager, sdpa, inputs = _vision_attention_pair()
    on_xpu = {
        k: (
            v.to("xpu")
            if torch.is_tensor(v)
            else tuple(t.to("xpu") for t in v) if isinstance(v, tuple) else v
        )
        for k, v in inputs.items()
    }
    with torch.no_grad():
        ref = eager(**inputs)
        out = sdpa.to("xpu")(**on_xpu).cpu()
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
