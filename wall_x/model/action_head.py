import torch

import torch.nn as nn

from typing import Union
import math

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.distributions import Beta


def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1):
            print(message, flush=True)
    else:
        print(message, flush=True)


class Normalizer(nn.Module):
    @classmethod
    def from_ckpt(cls, ckpt_path):
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        
        instance.min = nn.ParameterDict()
        instance.delta = nn.ParameterDict()
        instance.min_key = "min"
        instance.delta_key = "delta"
        
        ckpt = torch.load(ckpt_path, map_location="cpu")
        
        for key, value in ckpt.items():
            # 解析 key: "min.robot_name" -> prefix="min", name="robot_name"
            try:
                prefix, name = key.split('.', 1)
                if hasattr(instance, prefix):
                    getattr(instance, prefix)[name] = nn.Parameter(value, requires_grad=False)
                    print("prefix",prefix)
                    print("name",name)
            except ValueError:
                continue 
                
        return instance

    def __init__(
        self, action_statistic_dof, dof_config, min_key="min", delta_key="delta"
    ):
        super(Normalizer, self).__init__()

        self.min_key = min_key
        self.delta_key = delta_key

        action_statistic = {}
        for robot_name in action_statistic_dof.keys():
            action_statistic[robot_name] = {}
            all_dof_min = []
            all_dof_delta = []
            for k in dof_config:
                if k in action_statistic_dof[robot_name]:
                    if (
                        min_key in action_statistic_dof[robot_name][k]
                        and delta_key in action_statistic_dof[robot_name][k]
                    ):
                        all_dof_min.extend(action_statistic_dof[robot_name][k][min_key])
                        all_dof_delta.extend(
                            action_statistic_dof[robot_name][k][delta_key]
                        )
                    else:                   
                        if robot_name == "x2_normal" or "libero" in robot_name:
                            print_rank_last(f"Normalizer (Warning): min_key {min_key} or delta_key {delta_key} ")
                            print_rank_last(f"not in action_statistic_dof[{robot_name}][{k}], use default min 0.0 and delta 1.0")
                        all_dof_min.extend([0.0] * dof_config[k])
                        all_dof_delta.extend([1.0] * dof_config[k])
                else:
                    if robot_name == "x2_normal" or "libero" in robot_name:
                        print_rank_last(
                            f"Normalizer (Warning): Action {k} not in action_statistic_dof for {robot_name}, use default min 0.0 and delta 1.0"
                        )
                    all_dof_min.extend([0.0] * dof_config[k])
                    all_dof_delta.extend([1.0] * dof_config[k])
            all_dof_min = torch.tensor(all_dof_min)
            all_dof_delta = torch.tensor(all_dof_delta)
            action_statistic[robot_name][min_key] = all_dof_min
            action_statistic[robot_name][delta_key] = all_dof_delta

        self.min = nn.ParameterDict(
            {
                k: nn.Parameter(action_statistic[k][min_key], requires_grad=False)
                for k in action_statistic.keys()
            }
        )
        self.delta = nn.ParameterDict(
            {
                k: nn.Parameter(action_statistic[k][delta_key], requires_grad=False)
                for k in action_statistic.keys()
            }
        )

        for k,v in action_statistic.items():
            print_rank_last(f"Normalizer: {k} min {action_statistic[k][min_key]} delta {action_statistic[k][delta_key]}")

    def normalize_data(self, xs, dataset_names):
        new_xs = []
        dataset_names = [name for name in dataset_names if name != "x2_multimodal"]
        for x, dataset_name in zip(xs, dataset_names):
            x = (x - self.min[dataset_name]) / (self.delta[dataset_name])
            x = x * 2 - 1
            x = torch.clamp(x, -1, 1)
            new_xs.append(x)
        new_xs = torch.stack(new_xs)
        return new_xs

    def unnormalize_data(self, xs, dataset_names, dof_mask=None):
        new_xs = []
        dataset_names = [name for name in dataset_names if name != "x2_multimodal"]
        dof_mask = dof_mask if dof_mask is not None else [None] * len(xs)
        for x, dataset_name, mask in zip(xs, dataset_names, dof_mask):
            x = (x + 1) / 2
            if mask is not None:
                mask = mask[0].bool()
                action_space_delta = self.delta[dataset_name][mask]
                action_space_min = self.min[dataset_name][mask]
            else:
                action_space_delta = self.delta[dataset_name]
                action_space_min = self.min[dataset_name]
            x = x * action_space_delta + action_space_min
            new_xs.append(x)
        new_xs = torch.stack(new_xs)
        return new_xs


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, min_period: float = 4e-3, max_period: float = 4.0):
        super().__init__()
        self.dim = dim
        if dim % 2 != 0:
            raise ValueError(f"embedding_dim ({dim}) must be divisible by 2")
        self.min_period = min_period
        self.max_period = max_period

    def forward(self, x):  # 旧PE区间
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -emb
        )
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Dropout(0.1),  # 实验一下
            nn.Unflatten(-1, (-1, 1)),
        )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        # down_dims=[512, 1024, 2048],
        kernel_size=5,
        n_groups=8,
    ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        # print("number of parameters: {:e}".format(
        #     sum(p.numel() for p in self.parameters()))
        # )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond=None,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)  # bs, 2048, 5

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)  # bs, 2048, 5

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)  # bs, 512, 20

        x = self.final_conv(x)  # bs, 14, 20

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x


class DP_Action_head(nn.Module):
    def __init__(
        self,
        action_dim=14,
        transformer_dim=896,
        global_cond_dim=1806,
        load_pretrained=True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.transformer_dim = transformer_dim
        self.load_pretrained = load_pretrained
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=132,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        if self.load_pretrained:
            self.global_cond_dim = global_cond_dim
            self.condition_proj = nn.Sequential(
                nn.Linear(self.transformer_dim, 2 * self.transformer_dim),
                nn.ReLU(),
                nn.Linear(2 * self.transformer_dim, 2 * self.global_cond_dim),
                nn.ReLU(),
                nn.Linear(2 * self.global_cond_dim, self.global_cond_dim),
            )

            self.noise_pred_net = ConditionalUnet1D(
                input_dim=self.action_dim,
                # down_dims=[256,512,1024],
                down_dims=[512, 1024, 2048],
                global_cond_dim=self.global_cond_dim,
            )

            # load pretrained model
            action_pretrained_path = "/x2robot/liangyuxin/workspace/DiffusionPolicy/big_mix_0718_mn/30_noise_pred_net.pth"
            print("load noise_pred_net from:", action_pretrained_path, flush=True)
            self.noise_pred_net.load_state_dict(torch.load(action_pretrained_path))
        else:
            self.noise_pred_net = ConditionalUnet1D(
                input_dim=self.action_dim,
                down_dims=[256, 512, 1024],
                global_cond_dim=self.transformer_dim,
            )

    def forward(self, naction, condition, sample_times):
        bs = naction.shape[0]
        noise_shape = (
            naction.shape[0] * sample_times,
            naction.shape[1],
            naction.shape[2],
        )
        noise = torch.randn(noise_shape, device=naction.device)
        naction = (
            naction.unsqueeze(1)
            .repeat(1, sample_times, 1, 1)
            .reshape(bs * sample_times, naction.shape[1], naction.shape[2])
        )

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs * sample_times,),
            device=naction.device,
        ).long()
        condition = condition.to(self.condition_proj[0].weight.data.dtype)
        if self.load_pretrained:
            condition = self.condition_proj(condition)

        noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)
        noise_pred = self.noise_pred_net(
            noisy_actions, timesteps, global_cond=condition
        )
        return noise, noise_pred

    @torch.no_grad()
    def predict(self, condition, naction=None):
        bs = condition.shape[0]
        condition = condition.to(self.condition_proj[0].weight.data.dtype)
        if self.load_pretrained:
            condition = self.condition_proj(condition)

        if naction is not None:
            noise_shape = (naction.shape[0], naction.shape[1], naction.shape[2])
        else:
            noise_shape = (bs, 16, self.action_dim)  # tobe parameterized
        noise = torch.randn(noise_shape, device=condition.device)
        naction_pred = noise
        # init scheduler
        self.noise_scheduler.set_timesteps(
            self.noise_scheduler.config.num_train_timesteps
        )

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = self.noise_pred_net(
                sample=naction_pred, timestep=k, global_cond=condition
            )

            # inverse diffusion step (remove noise)
            naction_pred = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction_pred
            ).prev_sample

        return naction, naction_pred


class ActionProcessor(nn.Module):
    def __init__(self, config):
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
        self.hidden_size = config.hidden_size

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
            self.beta_alpha = noise_scheduler_config.get(
                "beta_alpha", 1.5
            ) 
            self.beta_beta = noise_scheduler_config.get(
                "beta_beta", 1.0
            ) 
            self.s = noise_scheduler_config.get("s", 0.999) 
            alpha_tensor = torch.tensor(self.beta_alpha, dtype=torch.float32).to("cuda")
            beta_tensor = torch.tensor(self.beta_beta, dtype=torch.float32).to("cuda")
            self.beta_dist = Beta(alpha_tensor, beta_tensor)
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

        # dataset_name = self.config["data"]["lerobot_config"]["repo_id"]
        # print("normalizer_propri min", self.normalizer_propri.min.__getattr__(dataset_name), flush=True)
        # print("normalizer_propri delta", self.normalizer_propri.delta.__getattr__(dataset_name), flush=True)
        # print("normalizer_action min", self.normalizer_action.min.__getattr__(dataset_name), flush=True)
        # print("normalizer_action delta", self.normalizer_action.delta.__getattr__(dataset_name), flush=True)

    def sample_time(self, batch_size, device, dtype):
        """
        Sampling Time Step
        Generates random numbers in the range [0, 1] using a Beta distribution, and then scales them.

        Parameters:
            batch_size (int): Batch size
            device: Device type
            dtype: Data type

        Returns:
            torch.Tensor: Sampled time steps, with shape [batch_size]
        """
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        time = (1 - sample) * self.s  
        return time

    def proprioception_proj(
        self, proprioception, dataset_names=None, dof_mask=None, use_history=False
    ):
        """
        proprioception: [batch_size, 1, action_dim]
        dataset_names: [batch_size]
        dof_mask: [batch_size, action_dim]
        """
        proprioception = proprioception.to(device=self.propri_proj.weight.device).to(
            dtype=self.propri_proj.weight.dtype
        )
        if dof_mask is not None:
            if self.config.proj_with_mask:
                proprioception = torch.cat(
                    [proprioception, dof_mask], dim=-1
                )  # .unsqueeze(1)
        proprioception = proprioception.to(device=self.propri_proj.weight.device).to(
            dtype=self.propri_proj.weight.dtype
        )
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

    def forward(self, action_chunk, dataset_names, dof_mask=None):
        """
        Parameters:
            action_chunk (torch.Tensor): Action sequence, shape [batch_size, action_chunk_len, action_dim]
            dataset_names: [batch_size]
            dof_mask: [batch_size, action_dim]

        Returns:
            torch.Tensor: Processed action representation, shape [batch_size, seq_len, hidden_size]
        """
        with torch.autocast("cuda", dtype=torch.float32):
            action_chunk = action_chunk.to(dtype=torch.float32)
            batch_size = action_chunk.shape[0]
            device = action_chunk.device
            dtype = action_chunk.dtype

            # 1. add noise to action_chunk
            noise = torch.randn_like(action_chunk)
            time = self.sample_time(batch_size, device, dtype)
            time_expanded = time.unsqueeze(-1).unsqueeze(-1)
            noisy_action = (
                1 - time_expanded
            ) * noise + time_expanded * action_chunk  
            flow = action_chunk - noise

            # 2. sinusoidal positional encoding for timesteps
            time_embed = self.time_embed(time).to(torch.float32)

            self.noise = noise
            self.noisy_action = noisy_action # for new x-pred
            
            # 3.action_chunk_nosiy + t_pos_emb -> MLP_act_chunk -> action_chunk_nosiy_emb_with_t (dim=trans * chunk)
            if dof_mask is not None:
                noisy_action = torch.cat([noisy_action, dof_mask], dim=-1)

            noisy_action = noisy_action.to(dtype=self.w1.weight.dtype)
            action_embed = self.w1(noisy_action)

            self.time_expanded = time_expanded # for new x-pred

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
                if dof_mask.shape[1] == 1:
                    dof_mask = dof_mask.unsqueeze(1).repeat(1,noisy_action.shape[1],1)
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

    def flow_loss(self, action_hidden_states, flow, action_chunk, dof_mask=None, flow_loss_mask=None):
        with torch.autocast("cuda", dtype=torch.float32):
            action_pred = self.action_proj_back(
                action_hidden_states[:, : self.action_hidden_size]
            )
            v_pred = action_pred
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
