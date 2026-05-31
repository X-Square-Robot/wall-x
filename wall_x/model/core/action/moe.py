import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from transformers.activations import ACT2FN

from wall_x.model.core.ops import unpermute, permute


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

    # FIXME: The full MLP is recomputed for now; recomputing only activations can be optimized later.
    def _full_mlp(self, hidden_state):
        gate_up_out = self.gate_up_proj(hidden_state)
        gate_out, up_out = gate_up_out.split(
            [self.intermediate_size, self.intermediate_size], dim=-1
        )

        act_out = self.act_fn(gate_out) * up_out
        return self.down_proj(act_out)

    def forward(self, hidden_state):
        if self.use_selective_recompute:
            # Checkpoint-recompute the whole expert MLP
            return cp.checkpoint(
                self._full_mlp,
                hidden_state,
                use_reentrant=False,
            )
        else:
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        experts_indices: torch.Tensor,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
    ) -> torch.Tensor:

        if self.permuted:
            permuted_inputs = hidden_states
        else:
            batch_size, seq_length, hidden_dim = hidden_states.shape

            flat_hidden = hidden_states.reshape(-1, hidden_dim)
            experts_indices = experts_indices.reshape(-1)
            probs = torch.ones_like(experts_indices, dtype=torch.float32).reshape(-1, 1)
            permuted_inputs, row_id_map = permute(flat_hidden, experts_indices)

        # buffer
        final_output = torch.zeros_like(permuted_inputs)

        # Expert forward, including selective recompute
        for expert_idx, expert in enumerate(self.experts):
            start, end = start_indices[expert_idx], end_indices[expert_idx]
            if start == end:
                continue

            dim_input = self.dim_inputs[expert_idx]
            expert_input = permuted_inputs[start:end, :dim_input]

            partial_output = expert(expert_input)
            final_output[start:end, :dim_input] = partial_output[:, :dim_input]

        if self.permuted:
            return final_output
        else:
            final_output = unpermute(final_output, row_id_map, probs)
            return final_output.reshape(batch_size, seq_length, hidden_dim)
