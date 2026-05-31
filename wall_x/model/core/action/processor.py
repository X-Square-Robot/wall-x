import torch
import torch.nn as nn

from wall_x.model.core.action.head import SinusoidalPosEmb
from wall_x.model.core.action.normalizer import print_rank_last


class ActionProcessor(nn.Module):
    """
    Action processor

    Main responsibilities:
    1. Add noise to action sequences
    2. Generate time encodings
    3. Project actions into the model hidden space

    Uses a Beta distribution to control noise scheduling and provide flexible noise injection.
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration object containing:
                - action_dim: action-space dimension
                - hidden_size: model hidden size
                - noise_scheduler: noise scheduler configuration
        """
        super().__init__()
        self.config = config
        self.dof_config = config.dof_config
        self.agent_pos_config = config.agent_pos_config
        self.action_dim = sum([v for k, v in self.dof_config.items()])
        self.propri_dim = sum([v for k, v in self.agent_pos_config.items()])

        print_rank_last(
            f"self.dof_config: {self.dof_config}; action_dim: {self.action_dim}; self.agent_pos_config: {self.agent_pos_config}; propri_dim: {self.propri_dim}"
        )

        self.action_hidden_size = config.action_hidden_size
        self.state_hidden_size = config.state_hidden_size
        self.hidden_size = getattr(config, "hidden_size", config.dim_inputs[0])

        if not self.config.use_state_string_representation:
            if self.config.proj_with_mask:
                self.propri_proj = nn.Linear(
                    self.propri_dim * 2, self.state_hidden_size, bias=False
                )
            else:
                self.propri_proj = nn.Linear(
                    self.propri_dim, self.state_hidden_size, bias=False
                )

        # noise scheduler configing
        if getattr(self.config, "use_flow_action_expert", True):
            noise_scheduler_config = config.noise_scheduler
            self.s = noise_scheduler_config.get("s", 0.999)
            self.time_shift = noise_scheduler_config.get(
                "time_shift", 1.0
            )  # time shift factor
            self.time_embed = SinusoidalPosEmb(self.action_hidden_size)

            # project to hidden space
            if self.config.proj_with_mask:
                self.w1 = nn.Linear(
                    self.action_dim * 2, self.action_hidden_size, bias=False
                )
            else:
                self.w1 = nn.Linear(
                    self.action_dim, self.action_hidden_size, bias=False
                )
            if not self.config.use_adarms:
                self.w2 = nn.Linear(
                    self.action_hidden_size * 2, self.action_hidden_size, bias=False
                )
                self.w3 = nn.Linear(
                    self.action_hidden_size, self.action_hidden_size, bias=False
                )
                self.act_fn = nn.SiLU()
            else:
                self.time_mlp_in = nn.Linear(
                    self.action_hidden_size, self.action_hidden_size
                )
                self.time_mlp_out = nn.Linear(
                    self.action_hidden_size, self.action_hidden_size
                )
                self.act_fn = nn.SiLU()

            # project back to action space
            self.action_proj_back = nn.Linear(
                self.action_hidden_size, self.action_dim, bias=False
            )
            self.mse_loss = nn.MSELoss(reduction="none")

    def set_normalizer(self, normalizer_action, normalizer_propri):
        self.normalizer_action = normalizer_action
        self.normalizer_propri = normalizer_propri

    def get_inference_times(self, num_steps, device, dtype):
        """
        Get inference timesteps

        Apply time shift and scaling

        Args:
            num_steps (int): number of inference steps
            device: Device type
            dtype: dtype

        Returns:
            torch.Tensor: inference timestep sequence
        """
        times = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
        if self.time_shift != 1.0:
            times = (self.time_shift * times) / (1 + (self.time_shift - 1) * times)
        times = times * self.s
        return times

    def proprioception_proj(
        self, proprioception, dataset_names=None, dof_mask=None, use_history=False
    ):
        """
        Args:
            proprioception: [batch_size, 1, action_dim]
            dataset_names: [batch_size]
            dof_mask: [batch_size, action_dim]
        """
        with torch.autocast("cuda", dtype=torch.float32):
            proprioception = proprioception.to(
                device=self.propri_proj.weight.device
            ).to(dtype=self.propri_proj.weight.dtype)
            if dof_mask is not None:
                if self.config.proj_with_mask:
                    proprioception = torch.cat(
                        [proprioception, dof_mask], dim=-1
                    )  # .unsqueeze(1)
            proprioception = proprioception.to(
                device=self.propri_proj.weight.device
            ).to(dtype=self.propri_proj.weight.dtype)
            proprio_embed = self.propri_proj(
                proprioception
            )  # [batch_size, 1, state_hidden_size]
            if self.state_hidden_size < self.hidden_size:
                # padding to hidden size
                padding_size = self.hidden_size - self.state_hidden_size
                padding = torch.zeros(
                    (proprio_embed.shape[0], 1, padding_size),
                    device=proprio_embed.device,
                    dtype=proprio_embed.dtype,
                )
                proprio_embed = torch.cat([proprio_embed, padding], dim=-1)
        return proprio_embed  # [batch_size, 1, hidden_size]

    def forward(self, action_chunk, dataset_names, sample_time, dof_mask=None):
        """
        Args:
            action_chunk (torch.Tensor): action sequence with shape [batch_size, action_chunk_len, action_dim]
            dataset_names: [batch_size]
            dof_mask: [batch_size, action_dim]

        Returns:
            torch.Tensor: processed action representation with shape [batch_size, seq_len, hidden_size]
        """
        with torch.autocast("cuda", dtype=torch.float32):
            action_chunk = action_chunk.to(dtype=torch.float32)

            # 1. add noise to action_chunk
            noise = torch.randn_like(action_chunk)
            time_expanded = sample_time.unsqueeze(-1).unsqueeze(-1)
            noisy_action = (
                1 - time_expanded
            ) * noise + time_expanded * action_chunk  # denoise from 0 to 1; integration does not need a negative sign
            flow = action_chunk - noise  # used to compute loss

            # 2. sinusoidal positional encoding for timesteps
            time_embed = self.time_embed(sample_time).to(torch.float32)

            self.noise = noise
            self.noisy_action = noisy_action  # for new x-pred
            # 3.action_chunk_nosiy + t_pos_emb -> MLP_act_chunk -> action_chunk_nosiy_emb_with_t (dim=trans * chunk)
            if dof_mask is not None:
                noisy_action = torch.cat([noisy_action, dof_mask], dim=-1)

            noisy_action = noisy_action.to(dtype=self.w1.weight.dtype)
            action_embed = self.w1(noisy_action)

            self.time_expanded = time_expanded  # for new x-pred

            if not self.config.use_adarms:
                time_embed = (
                    time_embed.unsqueeze(1)
                    .repeat(1, action_embed.shape[1], 1)
                    .to(dtype=self.w2.weight.dtype)
                )
                concat_embed = torch.cat([action_embed, time_embed], dim=-1)
                concat_embed = self.w2(concat_embed)
                action_time_embed = self.w3(self.act_fn(concat_embed))
                adarms_cond = None
            else:
                time_embed = self.time_mlp_in(time_embed)
                time_embed = self.act_fn(time_embed)
                time_embed = self.time_mlp_out(time_embed)
                time_embed = self.act_fn(time_embed)
                action_time_embed = action_embed
                adarms_cond = time_embed

            if self.action_hidden_size < self.hidden_size:
                # padding to hidden size
                padding_size = self.hidden_size - self.action_hidden_size
                padding = torch.zeros(
                    (
                        action_time_embed.shape[0],
                        action_time_embed.shape[1],
                        padding_size,
                    ),
                    device=action_time_embed.device,
                    dtype=action_time_embed.dtype,
                )
                action_time_embed = torch.cat([action_time_embed, padding], dim=-1)

        return action_time_embed, flow, adarms_cond

    def step(self, timestep, noisy_action, dof_mask=None):
        # noisy_action: bs, pred_horizon, action_dim
        # timestep: bs
        with torch.autocast("cuda", dtype=torch.float32):
            if dof_mask is not None and self.config.proj_with_mask:
                noisy_action = torch.cat([noisy_action, dof_mask], dim=-1)

            noisy_action = noisy_action.to(dtype=self.w1.weight.dtype)
            time_embed = self.time_embed(timestep).to(torch.float32)  # bs,hidden_size
            action_embed = self.w1(noisy_action)

            if not self.config.use_adarms:
                time_embed = time_embed.unsqueeze(1).repeat(1, action_embed.shape[1], 1)
                time_embed = time_embed.to(device=noisy_action.device).to(
                    dtype=noisy_action.dtype
                )
                concat_embed = torch.cat([action_embed, time_embed], dim=-1)
                concat_embed = self.w2(concat_embed)
                embed = self.w3(self.act_fn(concat_embed))  # is this right?
                adarms_cond = None
            else:
                time_embed = time_embed.to(dtype=self.time_mlp_in.weight.dtype)
                time_embed = self.time_mlp_in(time_embed)
                time_embed = self.act_fn(time_embed)
                time_embed = self.time_mlp_out(time_embed)
                time_embed = self.act_fn(time_embed)
                embed = action_embed
                adarms_cond = time_embed

            if self.action_hidden_size < self.hidden_size:
                # padding to hidden size
                padding_size = self.hidden_size - self.action_hidden_size
                padding = torch.zeros(
                    (embed.shape[0], embed.shape[1], padding_size),
                    device=embed.device,
                    dtype=embed.dtype,
                )
                embed = torch.cat([embed, padding], dim=-1)

        return embed, adarms_cond

    def flow_loss(
        self,
        action_hidden_states,
        flow,
        action_chunk,
        dof_mask=None,
        flow_loss_mask=None,
    ):
        with torch.autocast("cuda", dtype=torch.float32):
            action_pred = self.action_proj_back(
                action_hidden_states[:, : self.action_hidden_size]
            )

            if getattr(self.config, "use_x_pred", False):
                noisy_action_flat = self.noisy_action.reshape(
                    -1, self.noisy_action.shape[-1]
                )
                time_expanded_flat = self.time_expanded.expand(
                    -1, self.noisy_action.shape[1], -1
                ).reshape(-1, 1)
                v_pred = (action_pred - noisy_action_flat) / torch.clamp(
                    1 - time_expanded_flat, min=0.05
                )
                x_pred = action_pred
            else:
                v_pred = action_pred
                time_expanded_flat = self.time_expanded.expand(
                    -1, self.noisy_action.shape[1], -1
                ).reshape(-1, 1)
                x_pred = (1 - time_expanded_flat) * v_pred + self.noisy_action.reshape(
                    -1, self.noisy_action.shape[-1]
                )

            if getattr(self.config, "use_x_loss", False):
                loss = self.mse_loss(
                    x_pred,
                    action_chunk.reshape(-1, action_chunk.shape[-1]).to(
                        dtype=x_pred.dtype
                    ),
                )
            else:
                loss = self.mse_loss(v_pred, flow)

            if dof_mask is not None:
                dof_mask = dof_mask.reshape(-1, dof_mask.shape[-1])
                loss = loss * dof_mask

            if flow_loss_mask is not None:
                flow_loss_mask = (
                    flow_loss_mask.unsqueeze(-1)
                    .reshape(-1, 1)
                    .expand(-1, loss.shape[-1])
                )
                loss = loss * flow_loss_mask
        return loss
