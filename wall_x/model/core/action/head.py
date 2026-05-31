"""Action head helpers used by the VLA action processor."""

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding for action flow timesteps."""

    def __init__(self, dim: int, min_period: float = 4e-3, max_period: float = 4.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"embedding_dim ({dim}) must be divisible by 2")
        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period

    def forward(self, x):
        half_dim = self.dim // 2
        exponent = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=x.device, dtype=torch.float32) * -exponent
        )
        emb = x[:, None] * frequencies[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
