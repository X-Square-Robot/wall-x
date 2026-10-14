from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class TwoExpertLayout:
    action_starts: torch.Tensor
    action_ends: torch.Tensor
    llm_offsets: torch.Tensor
    action_offsets: torch.Tensor
    expert_segments: tuple[tuple[int, int], tuple[int, int]]
    token_perm: Optional[torch.Tensor] = None
    token_perm_inv: Optional[torch.Tensor] = None


def prepare_two_expert_layout_packed(
    experts_indices: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
) -> TwoExpertLayout:
    """Build a permutation layout for multiple action spans in a packed row."""
    if experts_indices.dim() != 2:
        raise ValueError("packed two-expert path expects [B, S] expert indices")
    if device is None:
        device = experts_indices.device
    flat = experts_indices.reshape(-1)
    is_action = flat == 1
    action_pos = is_action.nonzero(as_tuple=True)[0]
    llm_pos = (~is_action).nonzero(as_tuple=True)[0]
    token_perm = torch.cat((llm_pos, action_pos)).to(
        device=device, dtype=torch.int64
    )
    token_perm_inv = torch.empty_like(token_perm)
    token_perm_inv[token_perm] = torch.arange(
        token_perm.numel(), device=device, dtype=torch.int64
    )
    total_llm = int(llm_pos.numel())
    total_tokens = int(flat.numel())
    empty = torch.zeros(0, dtype=torch.int64, device=device)
    return TwoExpertLayout(
        action_starts=empty,
        action_ends=empty,
        llm_offsets=empty,
        action_offsets=empty,
        expert_segments=((0, total_llm), (total_llm, total_tokens)),
        token_perm=token_perm,
        token_perm_inv=token_perm_inv,
    )


def fixed_two_expert_action_ranges(
    experts_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if experts_indices.dim() != 2:
        raise ValueError("fixed scatter MoE path expects [B, S] expert indices")

    action_mask = experts_indices == 1
    has_action = action_mask.any(dim=1)
    starts_raw = torch.argmax(action_mask.to(torch.int64), dim=1)
    ends_raw = experts_indices.size(1) - torch.argmax(
        action_mask.flip(dims=[1]).to(torch.int64), dim=1
    )
    zeros = torch.zeros_like(starts_raw)
    starts = torch.where(has_action, starts_raw, zeros)
    ends = torch.where(has_action, ends_raw, starts)
    action_lens = action_mask.sum(dim=1)
    if not bool(((ends - starts) == action_lens).all().item()):
        raise ValueError("fixed scatter MoE path requires contiguous action tokens")
    return starts.contiguous(), ends.contiguous()


def resolve_two_expert_action_ranges(
    experts_indices: Optional[torch.Tensor],
    action_starts: Optional[torch.Tensor] = None,
    action_ends: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if experts_indices is None or experts_indices.dim() != 2:
        raise ValueError("fixed scatter MoE path expects [B, S] expert indices")

    batch_size = experts_indices.size(0)
    device = experts_indices.device
    if action_starts is None and action_ends is None:
        return fixed_two_expert_action_ranges(experts_indices)
    if action_starts is None or action_ends is None:
        raise ValueError("action_starts and action_ends must be provided together")

    action_starts = torch.as_tensor(
        action_starts, dtype=torch.int64, device=device
    ).reshape(-1)
    action_ends = torch.as_tensor(
        action_ends, dtype=torch.int64, device=device
    ).reshape(-1)

    if action_starts.numel() != batch_size or action_ends.numel() != batch_size:
        raise ValueError("fixed scatter MoE path expects [B] action ranges")
    return action_starts.contiguous(), action_ends.contiguous()


def fixed_two_expert_segments(
    batch_size: int,
    seq_length: int,
    action_starts: torch.Tensor,
    action_ends: torch.Tensor,
) -> tuple[tuple[int, int], tuple[int, int]]:
    starts_cpu = torch.as_tensor(action_starts, dtype=torch.int64).reshape(-1).cpu()
    ends_cpu = torch.as_tensor(action_ends, dtype=torch.int64).reshape(-1).cpu()
    if starts_cpu.numel() != int(batch_size) or ends_cpu.numel() != int(batch_size):
        raise ValueError(
            "fixed scatter MoE path expects one action range per batch sample"
        )

    action_lens = ends_cpu - starts_cpu
    total_action = int(action_lens.sum().item())
    total_tokens = int(batch_size) * int(seq_length)
    llm_end = total_tokens - total_action
    return (0, llm_end), (llm_end, total_tokens)


def _cpu_int64_1d(values: torch.Tensor) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, dtype=torch.int64)
    values = values.detach().reshape(-1)
    if values.device.type != "cpu" or values.dtype != torch.int64:
        values = values.to(device="cpu", dtype=torch.int64)
    return values.contiguous()


def _device_int64_1d(
    values: torch.Tensor,
    fallback_cpu: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor) and values.device == device:
        values = values.reshape(-1)
        if values.dtype != torch.int64:
            values = values.to(dtype=torch.int64)
        if not values.is_contiguous():
            values = values.contiguous()
        return values
    return fallback_cpu.to(device=device, non_blocking=True)


def prepare_two_expert_layout(
    experts_indices: Optional[torch.Tensor],
    *,
    batch_size: Optional[int] = None,
    seq_length: Optional[int] = None,
    action_starts: Optional[torch.Tensor] = None,
    action_ends: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> TwoExpertLayout:
    if experts_indices is not None:
        if experts_indices.dim() != 2:
            raise ValueError("fixed scatter MoE path expects [B, S] expert indices")
        batch_size = experts_indices.size(0)
        seq_length = experts_indices.size(1)
        if device is None:
            device = experts_indices.device
    if batch_size is None or seq_length is None:
        raise ValueError("batch_size and seq_length are required")
    if device is None:
        raise ValueError("device is required")

    if action_starts is None or action_ends is None:
        action_starts, action_ends = resolve_two_expert_action_ranges(
            experts_indices,
            action_starts=action_starts,
            action_ends=action_ends,
        )

    starts_cpu = _cpu_int64_1d(action_starts)
    ends_cpu = _cpu_int64_1d(action_ends)
    if starts_cpu.numel() != batch_size or ends_cpu.numel() != batch_size:
        raise ValueError("fixed scatter MoE path expects [B] action ranges")
    if not bool(
        ((starts_cpu >= 0) & (ends_cpu >= starts_cpu) & (ends_cpu <= seq_length))
        .all()
        .item()
    ):
        raise ValueError(
            "fixed scatter MoE action ranges must satisfy "
            "0 <= start <= end <= seq_length"
        )

    action_lens = ends_cpu - starts_cpu
    llm_counts = int(seq_length) - action_lens
    zero = torch.zeros(1, dtype=torch.int64)
    llm_offsets_cpu = torch.cat((zero, torch.cumsum(llm_counts, dim=0))).contiguous()
    action_offsets_cpu = torch.cat(
        (zero, torch.cumsum(action_lens, dim=0))
    ).contiguous()
    total_llm = int(llm_offsets_cpu[-1].item())
    total_tokens = int(batch_size) * int(seq_length)

    device = torch.device(device)
    return TwoExpertLayout(
        action_starts=_device_int64_1d(action_starts, starts_cpu, device=device),
        action_ends=_device_int64_1d(action_ends, ends_cpu, device=device),
        llm_offsets=llm_offsets_cpu.to(device=device, non_blocking=True),
        action_offsets=action_offsets_cpu.to(device=device, non_blocking=True),
        expert_segments=((0, total_llm), (total_llm, total_tokens)),
    )
