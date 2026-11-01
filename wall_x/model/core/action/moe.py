import torch
import torch.nn as nn

from transformers.activations import ACT2FN

from wall_x.model.core.action.layout import (
    TwoExpertLayout,
    fixed_two_expert_action_ranges,
    fixed_two_expert_segments,
    prepare_two_expert_layout,
    resolve_two_expert_action_ranges,
)
from wall_x.model.core.ops import silu_and_mul, two_expert_pack, two_expert_unpack


__all__ = [
    "TwoExpertLayout",
    "fixed_two_expert_action_ranges",
    "fixed_two_expert_segments",
    "prepare_two_expert_layout",
    "resolve_two_expert_action_ranges",
    "TokenTypeRouter",
    "BlockSparseMLP",
    "SparseMoeBlock",
]


class TokenTypeRouter(nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, token_types: torch.Tensor) -> torch.Tensor:
        """
        Route tokens to experts based on token_type.

        Args:
            token_types (torch.Tensor): Tensor of shape (batch_size, seq_length) containing each token type.

        Returns:
            experts_indices (torch.Tensor): Tensor of shape (batch_size, seq_length) containing each assigned expert index.
        """
        # Simple rule: assign by token_type modulo the expert count
        experts_indices = token_types % self.num_experts
        return experts_indices


class BlockSparseMLP(nn.Module):
    def __init__(self, config, use_selective_recompute: bool = False):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.hidden_act = config["hidden_act"]

        self.use_selective_recompute = use_selective_recompute

        self.gate_up_proj = nn.Linear(
            self.hidden_size, 2 * self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = ACT2FN[self.hidden_act]
        self.use_silu_and_mul = self.hidden_act in ("silu", "swish")

    # FIXME: The full MLP is recomputed for now; recomputing only activations can be optimized later.
    def _full_mlp(self, hidden_state):
        gate_up_out = self.gate_up_proj(hidden_state)

        if self.use_silu_and_mul:
            act_out = silu_and_mul(gate_up_out, inplace=True)
        else:
            gate_out, up_out = gate_up_out.split(
                [self.intermediate_size, self.intermediate_size], dim=-1
            )
            act_out = self.act_fn(gate_out) * up_out
        return self.down_proj(act_out)

    def forward(self, hidden_state):
        # Action expert operates on very few action tokens (~192/sample), so its
        # activations are tiny; recomputing the full expert MLP costs compute for
        # negligible memory saving. Always run directly (no checkpoint).
        return self._full_mlp(hidden_state)


class SparseMoeBlock(nn.Module):
    def __init__(self, config, num_experts: int, use_selective_recompute: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.use_selective_recompute = use_selective_recompute

        # Pass use_selective_recompute to each expert
        self.experts = nn.ModuleList(
            [
                BlockSparseMLP(
                    config.experts[i], use_selective_recompute=use_selective_recompute
                )
                for i in range(num_experts)
            ]
        )

        if not hasattr(config, "dim_inputs") or not config.dim_inputs:
            raise ValueError("config.dim_inputs must be set")

        self.dim_inputs = config.dim_inputs
        self.permuted = config.mot_opt

    def _fixed_two_expert_action_ranges(self, experts_indices: torch.Tensor):
        if self.num_experts != 2:
            raise ValueError("fixed scatter MoE path requires num_experts == 2")
        return fixed_two_expert_action_ranges(experts_indices)

    def forward(
        self,
        hidden_states: torch.Tensor,
        experts_indices: torch.Tensor,
        action_starts: torch.Tensor = None,
        action_ends: torch.Tensor = None,
        expert_segments: tuple[tuple[int, int], tuple[int, int]] = None,
        two_expert_layout: TwoExpertLayout = None,
    ) -> torch.Tensor:
        batch_size, seq_length = experts_indices.shape
        if two_expert_layout is None:
            two_expert_layout = prepare_two_expert_layout(
                experts_indices,
                batch_size=batch_size,
                seq_length=seq_length,
                action_starts=action_starts,
                action_ends=action_ends,
                device=hidden_states.device,
            )
        expert_segments = two_expert_layout.expert_segments

        if self.permuted:
            permuted_inputs = hidden_states
        else:
            permuted_inputs = two_expert_pack(hidden_states, two_expert_layout)

        # buffer
        final_output = torch.zeros_like(permuted_inputs)

        # Expert forward, including selective recompute
        for expert_idx, expert in enumerate(self.experts):
            start, end = expert_segments[expert_idx]
            if start == end:
                continue

            dim_input = self.dim_inputs[expert_idx]
            expert_input = permuted_inputs[start:end, :dim_input]

            partial_output = expert(expert_input)
            final_output[start:end, :dim_input] = partial_output[:, :dim_input]

        if self.permuted:
            return final_output
        else:
            return two_expert_unpack(
                final_output, batch_size, seq_length, two_expert_layout
            )
