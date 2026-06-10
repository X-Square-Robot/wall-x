"""Public fallback tests for operator proxies."""

import pytest
import torch

from wall_x.model.core.ops import rmsnorm


def test_rmsnorm_pytorch_fallback_matches_reference():
    x = torch.randn(2, 8, dtype=torch.float32)
    weight = torch.randn(8, dtype=torch.float32)
    eps = 1e-6

    out = rmsnorm.call_with_backend("pytorch", x, weight, eps)

    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    ref = ref * weight
    torch.testing.assert_close(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rmsnorm_cuda_inline_matches_pytorch_when_available():
    if "cuda_inline" not in rmsnorm.available_backends():
        pytest.skip("CUDA inline backend is not available")

    x = torch.randn(2, 16, device="cuda", dtype=torch.float16)
    weight = torch.randn(16, device="cuda", dtype=torch.float16)
    eps = 1e-6

    out = rmsnorm.call_with_backend("cuda_inline", x, weight, eps)
    ref = rmsnorm.call_with_backend("pytorch", x, weight, eps)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
