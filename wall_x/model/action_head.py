import math
import torch
import torch.nn as nn
from torch.distributions import Beta
from wall_x.utils.constant import action_statistic_dof
import logging


class Normalizer(nn.Module):
    """
    Action data normalizer for multi-robot systems.

    This module handles normalization and denormalization of action data for different robot
    configurations. It maintains per-robot statistics (min values and deltas) and applies
    normalization to map actions to the [-1, 1] range.
    """

    def _pad_to_action_dim(self, xs, action_dim):
        """
        Pad the action data to the action dimension.
        """
        if xs.shape[-1] < action_dim:
            padding_shape = list(xs.shape)
            padding_shape[-1] = action_dim - padding_shape[-1]
            xs = torch.cat([xs, torch.zeros(padding_shape).to(xs.device)], dim=-1)
        return xs

    def __init__(self, action_statistic_dof, dof_config):
        """
        Initialize the normalizer with robot-specific action statistics.

        Args:
            action_statistic_dof (dict): Statistical data for each robot's degrees of freedom
            dof_config (dict): Configuration mapping for degrees of freedom per robot
        """
        super(Normalizer, self).__init__()

        action_statistic = {}
        # hard code the action dimension to 20
        action_dim = 20

        # Process statistics for each robot
        for robot_name in action_statistic_dof.keys():
            action_statistic[robot_name] = {}
            all_dof_min = []
            all_dof_delta = []

            # Collect min and delta values for all DOFs
            for k in dof_config:
                if k in action_statistic_dof[robot_name]:
                    all_dof_min.extend(action_statistic_dof[robot_name][k]["min"])
                    all_dof_delta.extend(action_statistic_dof[robot_name][k]["delta"])
                else:
                    # Use default values if statistics not available
                    # raise ValueError(f"Statistics not available for {k} of {robot_name}")
                    logging.warning(
                        f"Statistics not available for {k} of {robot_name}, using default values"
                    )
                    all_dof_min.extend([0.0] * dof_config[k])
                    all_dof_delta.extend([1.0] * dof_config[k])

            all_dof_min = self._pad_to_action_dim(torch.tensor(all_dof_min), action_dim)
            all_dof_delta = self._pad_to_action_dim(
                torch.tensor(all_dof_delta), action_dim
            )
            action_statistic[robot_name]["min"] = all_dof_min
            action_statistic[robot_name]["delta"] = all_dof_delta

        # Register statistics as non-trainable parameters
        self.min = nn.ParameterDict(
            {
                k: nn.Parameter(action_statistic[k]["min"], requires_grad=False)
                for k in action_statistic.keys()
            }
        )
        self.delta = nn.ParameterDict(
            {
                k: nn.Parameter(action_statistic[k]["delta"], requires_grad=False)
                for k in action_statistic.keys()
            }
        )

    def normalize_data(self, xs, dataset_names, dof_mask=None):
        """
        Normalize action data to [-1, 1] range using robot-specific statistics.

        Args:
            xs: Input action data tensors
            dataset_names: List of dataset/robot names corresponding to each tensor

        Returns:
            torch.Tensor: Normalized action data in [-1, 1] range
        """
        new_xs = []
        # Filter out multimodal dataset entries
        dataset_names = [name for name in dataset_names if name != "x2_multimodal"]
        dof_mask = dof_mask if dof_mask is not None else [None] * len(xs)

        for x, dataset_name, mask in zip(xs, dataset_names, dof_mask):
            # Apply DOF mask if provided
            if mask is not None:
                mask = mask[0].bool()
                action_space_delta = self.delta[dataset_name][mask]
                action_space_min = self.min[dataset_name][mask]
            else:
                action_space_delta = self.delta[dataset_name]
                action_space_min = self.min[dataset_name]
            # Apply min-max normalization
            x = (x - action_space_min) / (action_space_delta)
            # Scale to [-1, 1] range
            x = x * 2 - 1
            # Clamp to ensure bounds
            x = torch.clamp(x, -1, 1)
            new_xs.append(x)

        new_xs = torch.stack(new_xs)
        return new_xs

    def unnormalize_data(self, xs, dataset_names, dof_mask=None):
        """
        Convert normalized data back to original action space.

        Args:
            xs: Normalized action data in [-1, 1] range
            dataset_names: List of dataset/robot names
            dof_mask: Optional mask to select specific degrees of freedom

        Returns:
            torch.Tensor: Denormalized action data in original scale
        """
        new_xs = []
        # Filter out multimodal dataset entries
        dataset_names = [name for name in dataset_names if name != "x2_multimodal"]
        dof_mask = dof_mask if dof_mask is not None else [None] * len(xs)

        for x, dataset_name, mask in zip(xs, dataset_names, dof_mask):
            # Convert from [-1, 1] to [0, 1] range
            x = (x + 1) / 2

            # Apply DOF mask if provided
            if mask is not None:
                mask = mask[0].bool()
                action_space_delta = self.delta[dataset_name][mask]
                action_space_min = self.min[dataset_name][mask]
            else:
                action_space_delta = self.delta[dataset_name]
                action_space_min = self.min[dataset_name]

            # Scale back to original range
            x = x * action_space_delta + action_space_min
            new_xs.append(x)

        new_xs = torch.stack(new_xs)
        return new_xs


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timesteps.

    Generates sinusoidal embeddings commonly used in diffusion models to encode
    timestep information with different frequencies.
    """

    def __init__(self, dim):
        """
        Initialize sinusoidal positional embedding.

        Args:
            dim (int): Embedding dimension (must be even)
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Generate sinusoidal embeddings for input timesteps.

        Args:
            x (torch.Tensor): Input timesteps

        Returns:
            torch.Tensor: Sinusoidal embeddings of shape (..., dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ActionProcessor(nn.Module):
    """
    Action sequence processor for robotic control with flow matching.

    This module handles action sequence processing for robotic systems with the following capabilities:
    1. Adds controlled noise to action sequences using Beta distribution scheduling
    2. Generates temporal embeddings for timestep conditioning
    3. Projects actions to model hidden space for transformer processing
    4. Supports proprioceptive data integration and multi-robot configurations

    The Beta distribution provides more flexible noise injection strategies compared to
    traditional linear schedules, allowing better control over the noise scheduling process.
    """

    def __init__(self, config):
        """
        Initialize the action processor with multi-robot support.

        Args:
            config: Configuration object containing:
                - dof_config (dict): Degrees of freedom configuration per robot type
                - agent_pos_config (dict): Agent position/proprioception configuration
                - hidden_size (int): Model hidden layer dimension
                - noise_scheduler (dict): Noise scheduler configuration with Beta parameters
        """
        super().__init__()

        # Calculate action and proprioception dimensions from configuration
        self.dof_config = config.dof_config
        self.agent_pos_config = config.agent_pos_config
        self.action_dim = sum([v for k, v in self.dof_config.items()])
        self.propri_dim = sum([v for k, v in self.agent_pos_config.items()])

        # Log configuration details for debugging
        print("ActionProcessor Configuration:", flush=True)
        print(f"  Action dimension: {self.action_dim}", flush=True)
        print(f"  Proprioception dimension: {self.propri_dim}", flush=True)
        print("  DOF configuration:", flush=True)
        for key, value in self.dof_config.items():
            print(f"    {key}: {value}", flush=True)
        print("  Agent position configuration:", flush=True)
        for key, value in self.agent_pos_config.items():
            print(f"    {key}: {value}", flush=True)

        self.hidden_size = config.hidden_size

        # Initialize data normalizers for actions and proprioception
        self.normalizer_action = Normalizer(
            action_statistic_dof,
            (
                config.customized_dof_config
                if hasattr(config, "customized_dof_config")
                else config.dof_config
            ),
        )
        self.normalizer_propri = Normalizer(
            action_statistic_dof,
            (
                config.customized_agent_pos_config
                if hasattr(config, "customized_agent_pos_config")
                else config.agent_pos_config
            ),
        )

        # Proprioception projection layer (includes history/current state)
        self.propri_proj = nn.Linear(self.propri_dim * 2, self.hidden_size, bias=False)

        # Beta distribution noise scheduler configuration
        noise_scheduler_config = config.noise_scheduler
        self.beta_alpha = noise_scheduler_config.get(
            "beta_alpha", 1.5
        )  # Beta distribution α parameter
        self.beta_beta = noise_scheduler_config.get(
            "beta_beta", 1.0
        )  # Beta distribution β parameter
        self.s = noise_scheduler_config.get("s", 0.999)  # Scaling factor

        # Initialize Beta distribution for noise scheduling
        alpha_tensor = torch.tensor(self.beta_alpha, dtype=torch.float32).to("cuda")
        beta_tensor = torch.tensor(self.beta_beta, dtype=torch.float32).to("cuda")
        self.beta_dist = Beta(alpha_tensor, beta_tensor)

        # Sinusoidal positional embedding for timesteps
        self.time_embed = SinusoidalPosEmb(config.hidden_size)

        # Action embedding network: project to hidden space
        self.w1 = nn.Linear(
            self.action_dim * 2, self.hidden_size, bias=False
        )  # *2 for action + DOF mask
        self.w2 = nn.Linear(
            self.hidden_size * 2, self.hidden_size, bias=False
        )  # *2 for action + time embeddings
        self.w3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        # Project back to action space for flow matching loss
        self.action_proj_back = nn.Linear(self.hidden_size, self.action_dim, bias=False)
        self.mse_loss = nn.MSELoss(reduction="none")

    def sample_time(self, batch_size, device, dtype):
        """
        Sample timesteps using Beta distribution for noise scheduling.

        Generates random timesteps in [0,1] range using Beta distribution, then scales them.
        This provides more flexible control over the noise injection schedule compared to
        uniform sampling.

        Args:
            batch_size (int): Number of timesteps to sample
            device: Target device for tensors
            dtype: Target data type for tensors

        Returns:
            torch.Tensor: Sampled timesteps of shape [batch_size]
        """
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        time = (1 - sample) * self.s
        return time

    def proprioception_proj(
        self, proprioception, dataset_names=None, dof_mask=None, use_history=False
    ):
        """
        Project proprioceptive data (joint positions, orientations) to hidden space.

        Args:
            proprioception (torch.Tensor): Proprioceptive data of shape [batch_size, seq_len, propri_dim]
            dataset_names (list, optional): Dataset names for normalization. Defaults to None.
            dof_mask (torch.Tensor, optional): DOF mask of shape [batch_size, propri_dim]. Defaults to None.
            use_history (bool, optional): Whether to use historical proprioceptive data. Defaults to False.

        Returns:
            torch.Tensor: Projected proprioceptive features of shape [batch_size, seq_len, hidden_size]
        """
        # Ensure proper device and dtype alignment
        proprioception = proprioception.to(device=self.propri_proj.weight.device).to(
            dtype=self.propri_proj.weight.dtype
        )

        if dof_mask is not None:
            # Concatenate proprioception with DOF mask
            # TODO: Use variable-based dimension checking for better flexibility
            if use_history:
                proprioception = torch.cat([proprioception, dof_mask], dim=-1)
            else:
                proprioception = torch.cat([proprioception, dof_mask], dim=-1)

        proprioception = proprioception.to(device=self.propri_proj.weight.device).to(
            dtype=self.propri_proj.weight.dtype
        )
        return self.propri_proj(proprioception)

    def forward(self, action_chunk, dataset_names, dof_mask=None):
        """
        Process action sequences with noise injection and temporal embedding.

        This method implements the forward pass for flow matching training:
        1. Adds Beta-distributed noise to action sequences
        2. Generates sinusoidal timestep embeddings
        3. Projects noisy actions to hidden space
        4. Combines action and temporal features

        Args:
            action_chunk (torch.Tensor): Action sequences of shape [batch_size, seq_len, action_dim]
            dataset_names (list): Dataset names for normalization
            dof_mask (torch.Tensor, optional): DOF mask of shape [batch_size, seq_len, action_dim].
                                              Defaults to None.

        Returns:
            tuple: (action_embeddings, flow_target) where:
                - action_embeddings: Processed action features of shape [batch_size, seq_len, hidden_size]
                - flow_target: Flow matching target (action_chunk - noise) for loss computation
        """
        batch_size = action_chunk.shape[0]
        device = action_chunk.device
        dtype = action_chunk.dtype

        # 1. Add noise to action sequences using flow matching
        noise = torch.randn_like(action_chunk)
        time = self.sample_time(batch_size, device, dtype)
        t = time.unsqueeze(-1).unsqueeze(-1)  # Broadcast to match action dimensions

        # Linear interpolation between noise and action (flow matching)
        noisy_action = (1 - t) * noise + t * action_chunk
        flow = action_chunk - noise  # Flow target for loss computation

        # 2. Generate sinusoidal positional encoding for timesteps
        time_embed = self.time_embed(time)

        # 3. Project noisy actions with DOF mask to hidden space
        if dof_mask is not None:
            noisy_action = torch.cat([noisy_action, dof_mask], dim=-1)

        noisy_action = noisy_action.to(dtype=self.w1.weight.dtype)
        action_embed = self.w1(noisy_action)

        # Repeat time embedding for each sequence position
        time_embed = (
            time_embed.unsqueeze(1)
            .repeat(1, action_embed.shape[1], 1)
            .to(dtype=self.w2.weight.dtype)
        )

        # Combine action and temporal embeddings
        concat_embed = torch.cat([action_embed, time_embed], dim=-1)
        concat_embed = self.w2(concat_embed)
        embed = self.w3(self.act_fn(concat_embed))

        return embed, flow

    def step(self, timestep, noisy_action, dof_mask=None):
        """
        Single denoising step for diffusion inference.

        Processes noisy actions at a specific timestep for iterative denoising during inference.

        Args:
            timestep (torch.Tensor): Current timesteps of shape [batch_size]
            noisy_action (torch.Tensor): Noisy actions of shape [batch_size, seq_len, action_dim]
            dof_mask (torch.Tensor, optional): DOF mask for action space. Defaults to None.

        Returns:
            torch.Tensor: Processed action embeddings of shape [batch_size, seq_len, hidden_size]
        """
        # Concatenate noisy action with DOF mask if provided
        if dof_mask is not None:
            noisy_action = torch.cat([noisy_action, dof_mask], dim=-1)

        # Generate timestep embeddings
        time_embed = self.time_embed(timestep)  # [batch_size, hidden_size]

        # Project noisy actions
        action_embed = self.w1(noisy_action)

        # Broadcast time embeddings to sequence length
        time_embed = time_embed.unsqueeze(1).repeat(1, action_embed.shape[1], 1)
        time_embed = time_embed.to(device=noisy_action.device).to(
            dtype=noisy_action.dtype
        )

        # Combine embeddings and process through MLP
        concat_embed = torch.cat([action_embed, time_embed], dim=-1)
        concat_embed = self.w2(concat_embed)
        embed = self.w3(self.act_fn(concat_embed))

        return embed

    def flow_loss(self, action_hidden_states, flow, dof_mask=None):
        """
        Compute flow matching loss between predicted and target actions.

        Args:
            action_hidden_states (torch.Tensor): Hidden states from transformer
            flow (torch.Tensor): Target flow (action - noise) for matching
            dof_mask (torch.Tensor, optional): DOF mask to weight loss per dimension. Defaults to None.

        Returns:
            torch.Tensor: Flow matching loss (no reduction for channel loss computation)
        """
        # Project hidden states back to action space
        action_pred = self.action_proj_back(action_hidden_states)

        # Compute MSE loss between predicted and target flow
        loss = self.mse_loss(action_pred, flow)

        # Apply DOF mask if provided
        if dof_mask is not None:
            dof_mask = dof_mask.reshape(-1, dof_mask.shape[-1])
            loss = loss * dof_mask

        # Return loss without reduction for channel-wise loss computation
        return loss
