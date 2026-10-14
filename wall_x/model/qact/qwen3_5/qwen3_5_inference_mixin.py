import logging
import os

import numpy as np
import torch
from typing import Optional
from torchdiffeq import odeint

logger = logging.getLogger(__name__)


class Qwen3_5VLAInferenceMixin:

    @torch.no_grad()
    def generate_text(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_history: Optional[torch.Tensor] = None,
        image_grid_thw_history: Optional[torch.LongTensor] = None,
        history_image_mask: Optional[torch.Tensor] = None,
        proprioception: Optional[torch.FloatTensor] = None,
        dataset_names: Optional[str] = None,
        dof_mask: Optional[torch.FloatTensor] = None,
        agent_pos_mask: Optional[torch.FloatTensor] = None,
        prefix_length: Optional[int] = None,
        gt_output_ids: Optional[torch.LongTensor] = None,
        re_generate: bool = False,
        **kwargs,
    ):
        """生成文本（VQA），参考 Qwen2.5 的 generate_text 实现。"""
        # 截取 prefix 部分
        if prefix_length is not None:
            input_ids, _gt_output_ids = (
                input_ids[:, :prefix_length],
                input_ids[:, prefix_length:],
            )
            gt_output_ids = (
                gt_output_ids if gt_output_ids is not None else _gt_output_ids
            )
            attention_mask = (
                attention_mask[:, :prefix_length]
                if attention_mask is not None
                else None
            )
        else:
            prefix_length = input_ids.shape[1]

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "pixel_values_history": pixel_values_history,
            "image_grid_thw_history": image_grid_thw_history,
            "history_image_mask": history_image_mask,
            "proprioception": proprioception,
            "dataset_names": dataset_names,
            "dof_mask": dof_mask,
            "agent_pos_mask": agent_pos_mask,
        }
        # 移除 None 值，避免传入不必要的参数
        batch = {k: v for k, v in batch.items() if v is not None}

        predict_output_ids = self.generate(
            **batch,
            max_new_tokens=100,
            eos_token_id=[self.processor.tokenizer.eos_token_id],
            use_cache=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            temperature=1.0 if not re_generate else 0.7,
            do_sample=False if not re_generate else True,
        )

        input_text = self.processor.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if gt_output_ids is not None and gt_output_ids.shape[-1] != 0:
            gt_output_text = self.processor.batch_decode(
                gt_output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
        else:
            gt_output_text = None
        predict_output_text = self.processor.batch_decode(
            predict_output_ids[:, prefix_length:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        return {
            "input_text": input_text,
            "gt_output_text": gt_output_text,
            "predict_output_text": predict_output_text,
            "predict_output_ids": predict_output_ids,
        }

    @torch.no_grad()
    def generate_flow_action(
        self,
        **kwargs,
    ):
        vla_config = self.config.vla_config
        if (
            self.config._attn_implementation != "sdpa"
            or vla_config.action_linear_attention
            or vla_config.causal_action_attention_mask
        ):
            return self.generate_flow_action_no_cache(**kwargs)
        return self.generate_flow_action_with_cache(**kwargs)

    @staticmethod
    def _get_flow_prefix_length(action_mask: torch.Tensor) -> int:
        has_action = action_mask.any(dim=1)
        if not bool(has_action.all().item()):
            raise ValueError("Every sample must contain flow action tokens")

        prefix_lengths = action_mask.to(torch.int64).argmax(dim=1)
        if not bool((prefix_lengths == prefix_lengths[0]).all().item()):
            raise ValueError(
                "prefix_length differs across batch; batch prompts must align"
            )
        return int(prefix_lengths[0].item())

    @staticmethod
    def _truncate_to_full_attention_prefix(cache, prefix_length: int) -> None:
        """Keep only immutable full-attention prefix K/V in a mixed cache."""
        for layer_idx in range(len(cache)):
            if cache.key_cache[layer_idx] is not None:
                cache.key_cache[layer_idx] = cache.key_cache[layer_idx][
                    :, :, :prefix_length, :
                ]
                cache.value_cache[layer_idx] = cache.value_cache[layer_idx][
                    :, :, :prefix_length, :
                ]
            cache.conv_states[layer_idx] = None
            cache.recurrent_states[layer_idx] = None

    @staticmethod
    def _clear_linear_attention_states(cache) -> None:
        for layer_idx in range(len(cache)):
            cache.conv_states[layer_idx] = None
            cache.recurrent_states[layer_idx] = None

    @staticmethod
    def _build_flow_postfix_attention_mask(
        attention_mask: torch.Tensor,
        prefix_length: int,
    ) -> torch.Tensor:
        """Build [B, postfix_Q, full_KV] visibility for bidirectional actions."""
        valid_tokens = attention_mask.to(torch.bool)
        valid_queries = valid_tokens[:, prefix_length:]
        return valid_queries.unsqueeze(2) & valid_tokens.unsqueeze(1)

    @torch.no_grad()
    def generate_flow_action_with_cache(
        self,
        input_ids,
        action_horizon,
        action_dim,
        init_noise: Optional[torch.Tensor] = None,
        # vlm
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        # for history observation
        pixel_values_history: torch.Tensor | None = None,
        image_grid_thw_history: torch.LongTensor | None = None,
        history_image_mask: torch.Tensor | None = None,
        # for vla
        num_inference_timesteps: int = 10,
        dof_mask: Optional[torch.Tensor] = None,
        proprioception: Optional[torch.FloatTensor] = None,
        agent_pos_mask: Optional[torch.Tensor] = None,
        dataset_names: Optional[str] = None,
        action_chunk: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        """Generate flow actions while reusing the immutable VLM prefix KV."""
        if num_inference_timesteps < 1:
            raise ValueError("num_inference_timesteps must be at least 1")

        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids are required for flow KV-cache inference")
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if input_ids is None:
            raise ValueError("input_ids are required to locate flow action tokens")
        batch_size = input_ids.shape[0]

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs = self.model.get_image_features(
                pixel_values,
                image_grid_thw,
                pixel_values_history=pixel_values_history,
                image_grid_thw_history=image_grid_thw_history,
                history_image_mask=history_image_mask,
            )
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs = self.model.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        inputs_embeds = self.model.scatter_proprioception_embeddings(
            input_ids, inputs_embeds, proprioception, dataset_names, agent_pos_mask
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None:
            position_ids = self.model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
        if position_ids is None:
            # Text-only Qwen3.5 lets the language model derive these from
            # cache_position.  Materialize the same four lanes here because
            # the cached path must later slice the postfix positions.
            position_ids = torch.arange(
                input_ids.shape[1], device=inputs_embeds.device
            ).view(1, 1, -1)
            position_ids = position_ids.expand(4, batch_size, -1)

        action_mask = input_ids == self.action_token_id_set["action_token_id"]
        prefix_length = self._get_flow_prefix_length(action_mask)
        postfix_length = input_ids.shape[1] - prefix_length
        if postfix_length != action_horizon or not bool(
            action_mask[:, prefix_length:].all().item()
        ):
            raise ValueError(
                "Flow KV-cache inference requires a contiguous action-only postfix "
                f"of length action_horizon={action_horizon}; got postfix length "
                f"{postfix_length}"
            )

        if dof_mask is None:
            dof_mask = torch.ones(action_dim, device=inputs_embeds.device)
        dof_mask = dof_mask.to(inputs_embeds.device, dtype=torch.float32)

        if init_noise is not None:
            noise = init_noise.to(device=inputs_embeds.device, dtype=torch.float32)
            expected_shape = (batch_size, action_horizon, action_dim)
            if noise.shape != expected_shape:
                raise ValueError(
                    f"init_noise shape {tuple(noise.shape)} != {expected_shape}"
                )
        else:
            noise = torch.randn(
                (batch_size, action_horizon, action_dim),
                dtype=torch.float32,
                device=inputs_embeds.device,
            )
        noisy_action = noise.clone()
        times = self.action_preprocessor.get_inference_times(
            num_inference_timesteps, inputs_embeds.device, torch.float32
        )

        has_padding_dims = not bool(dof_mask.all().item())
        v_padding = None
        if has_padding_dims:
            padding_action = torch.zeros(
                (1, dof_mask.shape[-1]),
                device=dof_mask.device,
                dtype=torch.float32,
            )
            padding_action = self.action_preprocessor.normalizer_action.normalize_data(
                padding_action, dataset_names
            )
            v_padding = padding_action - noisy_action

        def predict_velocity(timestep, current_action, model_output):
            action_embeds = model_output.action_embeds.to(torch.float32)
            action_pred = self.action_preprocessor.action_proj_back(action_embeds)
            if getattr(self.config.vla_config, "use_x_pred", False):
                velocity = (action_pred - current_action) / torch.clamp(
                    1 - timestep.view(batch_size, 1, 1), min=0.05
                )
            else:
                velocity = action_pred
            if has_padding_dims and v_padding is not None:
                velocity = v_padding * (1 - dof_mask) + velocity * dof_mask
            return velocity.reshape(batch_size, action_horizon, action_dim)

        # The first full-sequence evaluation provides both v0 and a cache.  As
        # prefix queries cannot see the future action block, slicing this cache
        # is equivalent to a separate prefix-only prefill.
        time_0 = times[0].unsqueeze(0).repeat(batch_size)
        action_embeds_0, adarms_cond_0 = self.action_preprocessor.step(
            timestep=time_0, noisy_action=noisy_action, dof_mask=dof_mask
        )
        action_embeds_0 = action_embeds_0[
            :, :, : self.config.vla_config.dim_inputs[1]
        ].to(inputs_embeds.dtype)
        prefill_output = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            action_embeds=action_embeds_0,
            action_mask=action_mask,
            adarms_conds=adarms_cond_0,
        )
        prefix_cache = prefill_output.past_key_values
        v_0 = predict_velocity(time_0, noisy_action, prefill_output)
        noisy_action = noisy_action + (times[1] - times[0]) * v_0

        self._truncate_to_full_attention_prefix(prefix_cache, prefix_length)
        postfix_inputs_embeds = inputs_embeds[:, prefix_length:, :]
        postfix_position_ids = position_ids[..., prefix_length:]
        postfix_action_mask = action_mask[:, prefix_length:]
        postfix_attention_mask = self._build_flow_postfix_attention_mask(
            attention_mask, prefix_length
        )
        postfix_cache_position = torch.arange(
            prefix_length,
            input_ids.shape[1],
            device=inputs_embeds.device,
        )

        def step_with_kvcache(timestep, current_action):
            timestep = timestep.unsqueeze(0).repeat(current_action.shape[0])
            action_embeds, adarms_cond = self.action_preprocessor.step(
                timestep=timestep,
                noisy_action=current_action,
                dof_mask=dof_mask,
            )
            action_embeds = action_embeds[
                :, :, : self.config.vla_config.dim_inputs[1]
            ].to(postfix_inputs_embeds.dtype)

            # Linear-attention states are deliberately excluded from this
            # optimization.  Clearing them keeps every ODE evaluation
            # independent while the full-attention prefix K/V stays shared.
            self._clear_linear_attention_states(prefix_cache)
            model_output = self.model.language_model(
                input_ids=None,
                position_ids=postfix_position_ids,
                attention_mask=postfix_attention_mask,
                past_key_values=prefix_cache,
                inputs_embeds=postfix_inputs_embeds,
                use_cache=False,
                cache_position=postfix_cache_position,
                action_embeds=action_embeds,
                action_mask=postfix_action_mask,
                adarms_conds=adarms_cond,
            )
            self._clear_linear_attention_states(prefix_cache)
            return predict_velocity(timestep, current_action, model_output)

        if num_inference_timesteps == 1:
            predict_action_norm = noisy_action
        else:
            action_trajectory = odeint(
                step_with_kvcache,
                noisy_action,
                times[1:],
                method="euler",
            )
            predict_action_norm = action_trajectory[-1]

        predict_action = self.action_preprocessor.normalizer_action.unnormalize_data(
            predict_action_norm, dataset_names
        )
        output = {
            "predict_action_norm": predict_action_norm,
            "predict_action": predict_action,
        }
        if action_chunk is not None:
            output["gt_action_norm"] = action_chunk
            output["gt_action"] = (
                self.action_preprocessor.normalizer_action.unnormalize_data(
                    action_chunk, dataset_names
                )
            )
        return output

    @torch.no_grad()
    def generate_flow_action_no_cache(
        self,
        input_ids,
        action_horizon,
        action_dim,
        init_noise: Optional[torch.Tensor] = None,
        # vlm
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        # for history observation
        pixel_values_history: torch.Tensor | None = None,
        image_grid_thw_history: torch.LongTensor | None = None,
        history_image_mask: torch.Tensor | None = None,
        # for vla
        num_inference_timesteps: int = 10,
        dof_mask: Optional[torch.Tensor] = None,
        proprioception: Optional[torch.FloatTensor] = None,
        agent_pos_mask: Optional[torch.Tensor] = None,
        dataset_names: Optional[str] = None,
        action_chunk: Optional[torch.FloatTensor] = None,
        # RTC (Real-Time Chunking): when inference_delay > 0 the first
        # `inference_delay` action steps are hard-pinned to `prefix_actions`
        # (normalized model space) throughout the ODE — conditioned as clean data
        # (per-token tau=1) and attention-isolated (the pinned prefix action
        # queries cannot attend to the denoising suffix action keys). This eager
        # path replicates, step-for-step, the verified native denoise engine
        # (third_party/harrix .../qwen3_5_based/denoise.py::_run). inference_delay
        # == 0 (default) is bit-identical to the plain-flow path below.
        prefix_actions: Optional[torch.Tensor] = None,
        inference_delay: int = 0,
        **kwargs,
    ):
        batch_size = (
            input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs = self.model.get_image_features(
                pixel_values,
                image_grid_thw,
                pixel_values_history=pixel_values_history,
                image_grid_thw_history=image_grid_thw_history,
                history_image_mask=history_image_mask,
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs = self.model.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )
            video_embeds = video_outputs.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        inputs_embeds = self.model.scatter_proprioception_embeddings(
            input_ids, inputs_embeds, proprioception, dataset_names, agent_pos_mask
        )

        # RTC delay, clamped to a valid window range. use_rtc gates every RTC
        # branch below so inference_delay == 0 leaves the plain-flow path
        # bit-identical (positional_masks stays None, no per-token tau, no pins).
        rtc_delay = int(max(0, min(int(inference_delay or 0), int(action_horizon))))
        use_rtc = rtc_delay > 0 and prefix_actions is not None
        if use_rtc and attention_mask is None:
            # The asymmetric RTC attention is injected via the joint 2D mask,
            # which only runs when an attention_mask is present (see
            # Qwen3_5_VLATextModel.forward). Mirror the native engine, which
            # defaults the mask to all-valid. Done before compute_3d_position_ids
            # so RoPE positions match the native path (which builds the same
            # all-ones mask).
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        if position_ids is None:
            position_ids = self.model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # past_key_values=past_key_values,
                # mm_token_type_ids=mm_token_type_ids,
            )

        if dof_mask is None:
            dof_mask = torch.ones(action_dim, device=inputs_embeds.device)

        # initialize action from noise (caller-provided for reproducible /
        # batch-invariant sampling, else randn)
        if init_noise is not None:
            noise = init_noise.to(device=inputs_embeds.device, dtype=torch.float32)
            assert noise.shape == (batch_size, action_horizon, action_dim), (
                f"init_noise shape {tuple(noise.shape)} != "
                f"{(batch_size, action_horizon, action_dim)}"
            )
        else:
            noise = torch.randn(
                size=(batch_size, action_horizon, action_dim),
                dtype=torch.float32,
                device=inputs_embeds.device,
            )
        noisy_action = noise.clone()
        dof_mask = dof_mask.to(inputs_embeds.device).to(torch.float32)

        times = self.action_preprocessor.get_inference_times(
            num_inference_timesteps, inputs_embeds.device, torch.float32
        )
        action_mask = input_ids == self.action_token_id_set["action_token_id"]

        # ---- RTC setup (only when inference_delay > 0) ----------------------
        # positional_masks carries the training-equivalent asymmetric attention:
        #   flow_block_id  -> all action-action pairs in a block become
        #                     bidirectional (matches the native decode mask),
        #   rtc_pinned     -> pinned action queries lose visibility of suffix
        #                     action keys (the RTC isolation),
        #   rtc_rope_gap=0 -> RoPE positions unchanged (TR-038 / native).
        # See wall_x/model/core/ops/attention_mask.py::
        # _apply_multisample_flow_blocks_2d, applied through the joint 2D mask.
        positional_masks = None
        adarms_pinned = None
        if use_rtc:
            prefix_actions = prefix_actions.to(
                device=inputs_embeds.device, dtype=torch.float32
            )
            act_int = action_mask.to(torch.int64)
            act_cumsum = torch.cumsum(act_int, dim=1)
            # First `rtc_delay` action tokens of each sample are the pinned prefix.
            rtc_pinned = action_mask & (act_cumsum <= rtc_delay)
            flow_block_id = torch.where(
                action_mask,
                torch.zeros_like(act_int),
                torch.full_like(act_int, -1),
            )
            positional_masks = {
                "flow_block_id": flow_block_id,
                "rtc_pinned": rtc_pinned,
                "rtc_rope_gap": 0,
            }
            # Per-token tau=1 (clean-data) condition for the pinned prefix rows.
            # time_embed depends only on the timestep, so this is computed once
            # (identical to native denoise.py's condition_pinned).
            ones_time = torch.ones(
                batch_size, device=inputs_embeds.device, dtype=torch.float32
            )
            _, adarms_pinned = self.action_preprocessor.step(
                timestep=ones_time,
                noisy_action=noisy_action,
                dof_mask=dof_mask,
            )  # [B, C] or None (None only when use_adarms is off)

        # process padding action dims
        has_padding_dims = not bool(dof_mask.all().item())
        v_padding = None
        if has_padding_dims:
            padding_action = (
                torch.zeros((1, dof_mask.shape[-1]))
                .to(dof_mask.device)
                .to(torch.float32)
            )
            padding_action = self.action_preprocessor.normalizer_action.normalize_data(
                padding_action, dataset_names
            )
            v_padding = padding_action - noisy_action

        def step(timestep, noisy_action):
            timestep = timestep.unsqueeze(0).repeat(noisy_action.shape[0])
            noisy_action_embeds, adarms_cond = self.action_preprocessor.step(
                timestep=timestep,
                noisy_action=noisy_action,
                dof_mask=dof_mask,
            )
            noisy_action_embeds = noisy_action_embeds[
                :, :, : self.config.vla_config.dim_inputs[1]
            ].to(inputs_embeds.dtype)

            if use_rtc and adarms_cond is not None:
                # Per-token RTC condition: broadcast this step's condition over the
                # action tokens, then overwrite the pinned prefix rows with the
                # tau=1 (clean-data) condition. Matches native denoise.py::_run.
                adarms_cond = (
                    adarms_cond.unsqueeze(1).expand(-1, action_horizon, -1).clone()
                )
                adarms_cond[:, :rtc_delay] = adarms_pinned.unsqueeze(1)

            # Thread positional_masks only under RTC so the plain-flow path keeps
            # kwargs byte-identical to before (no positional_masks key at all).
            rtc_kwargs = {"positional_masks": positional_masks} if use_rtc else {}
            model_output = self.model.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                # past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                # cache_position=cache_position,
                # for vla
                action_embeds=noisy_action_embeds,
                action_mask=action_mask,
                adarms_conds=adarms_cond,
                **rtc_kwargs,
            )

            action_embeds = model_output.action_embeds
            action_embeds = action_embeds.to(torch.float32)
            action_pred = self.action_preprocessor.action_proj_back(action_embeds)

            if getattr(self.config.vla_config, "use_x_pred", False):
                # v_t = action_pred - noise.reshape(-1, noise.shape[-1])
                # view timestep to (B,1,1) so it broadcasts over (B,H,D) for
                # batch_size > 1 (bs==1 is numerically unchanged).
                v_t = (action_pred - noisy_action) / torch.clamp(
                    1 - timestep.view(batch_size, 1, 1), min=0.05
                )
            else:
                v_t = action_pred

            if has_padding_dims and v_padding is not None:
                v_t = (v_padding) * (1 - dof_mask) + v_t * dof_mask

            return v_t.reshape(batch_size, action_horizon, action_dim)

        if not use_rtc:
            action_trajectory = odeint(step, noisy_action, times, method="euler")
            predict_action_norm = action_trajectory[-1]
        else:
            # Hand-written Euler over `times` with denoise.py::_run's 6-step RTC
            # pin recipe. (1)(2)(4)(5) are inline below; (3) the per-token tau=1
            # clean-data adarms condition and (6) the asymmetric attention mask
            # (positional_masks) are injected via `step` / the language_model
            # call above. deltas match torchdiffeq's fixed-step Euler over `times`.
            # The entry clone protects the caller's `noisy_action`; thereafter
            # `current` is always an exclusive tensor (entry-cloned, or freshly
            # produced by `current + dt*v`), and `velocity` is step()'s fresh
            # output, so the in-place pins below need no extra per-step clone.
            current = noisy_action.clone()
            # (1) ODE-entry pin.
            current[:, :rtc_delay] = prefix_actions[:, :rtc_delay]
            for i in range(num_inference_timesteps):
                # (2) Per-step entry pin (cancels float drift on the prefix).
                current[:, :rtc_delay] = prefix_actions[:, :rtc_delay]
                velocity = step(times[i], current)
                # (4) Per-step exit pin: the Euler update must not move the prefix.
                velocity[:, :rtc_delay] = 0.0
                current = current + (times[i + 1] - times[i]) * velocity
            # (5) Final pin: eliminate any residual float accumulation.
            current[:, :rtc_delay] = prefix_actions[:, :rtc_delay]
            predict_action_norm = current
        predict_action = self.action_preprocessor.normalizer_action.unnormalize_data(
            predict_action_norm, dataset_names
        )

        output = {}
        output["predict_action_norm"] = predict_action_norm
        if action_chunk is not None:
            output["gt_action_norm"] = action_chunk
        output["predict_action"] = predict_action
        # normalize action chunk to get gt_action
        if action_chunk is not None:
            output["gt_action"] = (
                self.action_preprocessor.normalizer_action.unnormalize_data(
                    action_chunk, dataset_names
                )
            )

        return output

    def _ar_token_layer_accuracy(self, pred_ids, gt_ids):
        """Overall + per-RVQ-layer exact-match accuracy for two position-aligned
        1-D token id tensors (``pred_ids``/``gt_ids`` must be the same length and
        already restricted to AR code tokens). Returns
        ``{"action_accuracy": float, "action_accuracy_rvq_q{q}": float, ...}``.

        Layer membership is read from ``action_mapper["token_id_to_rvq"]`` (RVQ
        delta). Tokenizers without it get overall accuracy only.
        """
        result = {}
        n = int(pred_ids.numel())
        if n == 0:
            return {"action_accuracy": float("nan")}
        correct = pred_ids == gt_ids

        token_id_to_rvq = self.action_mapper.get("token_id_to_rvq")
        num_quantizers = self.action_mapper.get("num_quantizers")
        if token_id_to_rvq is None or not num_quantizers:
            # Non-RVQ tokenizer: no separators to exclude, overall over all tokens.
            result["action_accuracy"] = float(correct.float().mean().item())
            return result

        # Bucket by the GT token's quantizer layer (labels are the reference).
        # token_range / ar_mask also captures the <rvq_group> separators (they live
        # inside token_range but are NOT in token_id_to_rvq); q_of=-1 marks them.
        gt_list = gt_ids.tolist()
        q_of = torch.tensor(
            [token_id_to_rvq[t][0] if t in token_id_to_rvq else -1 for t in gt_list],
            device=pred_ids.device,
        )
        code_mask = q_of >= 0
        # Overall = accuracy over ACTION CODE tokens only. The <rvq_group>
        # separators sit at deterministic positions and are predicted ~100%;
        # counting them would inflate the number above the per-layer breakdown.
        # With equal codes/layer this equals the count-weighted mean of q0..q{Q-1}.
        result["action_accuracy"] = (
            float((correct & code_mask).sum().item() / int(code_mask.sum().item()))
            if bool(code_mask.any())
            else float("nan")
        )
        grp_mask = ~code_mask
        if bool(grp_mask.any()):
            # Structural-separator accuracy, surfaced separately for transparency.
            result["action_accuracy_group"] = float(
                (correct & grp_mask).sum().item() / int(grp_mask.sum().item())
            )
        for q in range(int(num_quantizers)):
            m = q_of == q
            denom = int(m.sum().item())
            if denom > 0:
                result[f"action_accuracy_rvq_q{q}"] = float(
                    (correct & m).sum().item() / denom
                )
        return result

    @torch.no_grad()
    def evaluate_ar_action_teacher_forced(
        self,
        input_ids,
        action_horizon,
        action_dim,
        labels=None,
        attention_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        pixel_values_history=None,
        image_grid_thw_history=None,
        history_image_mask=None,
        proprioception=None,
        agent_pos_mask=None,
        dof_mask=None,
        dataset_names=None,
        action_chunk=None,
        robot_type_id=None,
        **kwargs,
    ):
        """Teacher-forced discrete action-token evaluation for qwen3_5.

        Runs ONE forward over the ground-truth token sequence (no autoregressive
        rollout, no flow expert), takes ``argmax`` of the LM head at the
        action-token label positions, and decodes those predicted tokens back to
        a continuous action via ``tokenizer_mixin.decode_action``.

        The flow expert is intentionally NOT engaged (``action_chunk`` is not
        forwarded to the model): the discrete action tokens sit causally left of
        the flow ``<|action|>`` placeholders (``visible_predict_ar_ratio=0``), so
        their hidden states — hence the CE logits argmax'd here — match the
        training forward regardless of the flow branch.

        Returns the same dict contract as ``generate_flow_action`` so the openloop
        evaluator's ``_accumulate`` works unchanged:
            predict_action / gt_action           [B, T, action_dim] (unnormalized)
            predict_action_norm / gt_action_norm [B, T, action_dim] (normalized)
        plus discrete-token diagnostics:
            token_accuracy         list[float]  per-sample exact-match over AR tokens
            gt_action_decoded      [B, T, D]    GT tokens -> action (quantization floor)
            decode_success         list[bool]
        """
        if labels is None:
            raise ValueError(
                "evaluate_ar_action_teacher_forced requires `labels` carrying the "
                "AR action tokens. Enable action_horizon_ar>0 on the eval dataset "
                "(action_mode='ar_tf')."
            )
        if getattr(self, "tokenizer_mixin", None) is None or (
            getattr(self, "action_mapper", None) is None
        ):
            raise RuntimeError(
                "AR teacher-forced eval requires a discrete action tokenizer "
                "(model config action_tokenizer_type)."
            )

        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Forward WITHOUT action_chunk -> flow expert skipped (see docstring).
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            pixel_values_history=pixel_values_history,
            image_grid_thw_history=image_grid_thw_history,
            history_image_mask=history_image_mask,
            proprioception=proprioception,
            agent_pos_mask=agent_pos_mask,
            dataset_names=dataset_names,
            dof_mask=dof_mask,
        )
        hidden_states = outputs.last_hidden_state  # [B, S, H]

        # Standard next-token shift: hidden[:, :-1] predicts labels[:, 1:].
        shift_hidden = hidden_states[:, :-1, :]
        shift_labels = labels[..., 1:].contiguous().to(device)  # [B, S-1]

        # Identify AR action-token label positions.
        token_range = self.action_mapper.get("token_range")
        if token_range is not None:
            ar_mask = (shift_labels >= token_range["min"]) & (
                shift_labels <= token_range["max"]
            )
        else:
            action_token_list = self.action_token_id_set.get("action_token_list", [])
            first = action_token_list[0] if action_token_list else None
            ar_mask = (
                shift_labels >= first
                if first is not None
                else torch.zeros_like(shift_labels, dtype=torch.bool)
            )

        # LM head only on AR positions (avoids materializing [B,S,V]).
        ar_hidden = shift_hidden[ar_mask]  # [N_ar_total, H]
        if ar_hidden.numel() == 0:
            raise RuntimeError(
                "No AR action tokens found in labels — the eval batch carries no "
                "discrete action supervision. Check action_horizon_ar override and "
                "that the ckpt was trained with the discrete head."
            )
        ar_logits = self.lm_head(ar_hidden.to(self.lm_head.weight.dtype)).float()
        pred_ids_flat = ar_logits.argmax(dim=-1)  # [N_ar_total]
        gt_ids_flat = shift_labels[ar_mask]  # [N_ar_total]
        counts = ar_mask.sum(dim=1).tolist()  # per-sample AR token count

        normalizer = self.action_preprocessor.normalizer_action
        uses_dof_mask = bool(
            getattr(self.tokenizer_mixin, "uses_dof_mask_for_unnorm", True)
        )

        predict_actions, gt_actions = [], []
        predict_actions_norm, gt_actions_norm = [], []
        gt_actions_decoded = []
        token_accs, token_acc_layers, decode_successes = [], [], []

        offset = 0
        for b in range(batch_size):
            n = int(counts[b])
            pred_ids_b = pred_ids_flat[offset : offset + n]
            gt_ids_b = gt_ids_flat[offset : offset + n]
            offset += n

            acc_dict = self._ar_token_layer_accuracy(pred_ids_b, gt_ids_b)
            token_accs.append(acc_dict["action_accuracy"])
            token_acc_layers.append(
                {k: v for k, v in acc_dict.items() if k != "action_accuracy"}
            )

            if b == 0 and os.environ.get("WALLX_DEBUG_AR_LAYOUT"):
                t2r = self.action_mapper.get("token_id_to_rvq", {})
                qseq = [t2r[t][0] for t in gt_ids_b.tolist() if t in t2r]
                # layer-major (dllm=False) => [0]*L, [1]*L, ...; time-major => 0,1,2,3,0,1,2,3,...
                logger.warning(
                    "AR layout debug: first-sample GT q_id sequence (n=%d): %s",
                    len(qseq),
                    qseq,
                )

            # Per-sample decode inputs.
            ds_b = (
                dataset_names[b : b + 1]
                if isinstance(dataset_names, (list, tuple))
                else dataset_names
            )
            propri_b = proprioception[b : b + 1] if proprioception is not None else None
            dof_b = dof_mask[b : b + 1] if dof_mask is not None else None

            rti = robot_type_id
            try:
                _rti = self.tokenizer_mixin.get_robot_type_ids(
                    uids=[None],
                    dataset_names=(
                        list(ds_b) if isinstance(ds_b, (list, tuple)) else [ds_b]
                    ),
                )
                if _rti:
                    rti = _rti[0]
            except Exception as exc:  # pragma: no cover - diagnostic fallback
                logger.warning("get_robot_type_ids failed (%s); using %s", exc, rti)

            pred_norm_b, ok = self.tokenizer_mixin.decode_action(
                output_ids=pred_ids_b.unsqueeze(0),
                action_mapper=self.action_mapper,
                action_horizon=action_horizon,
                action_dim=action_dim,
                device=device,
                proprioception=propri_b,
                dof_mask=dof_b,
                robot_type_id=rti,
            )
            gt_norm_b, _ = self.tokenizer_mixin.decode_action(
                output_ids=gt_ids_b.unsqueeze(0),
                action_mapper=self.action_mapper,
                action_horizon=action_horizon,
                action_dim=action_dim,
                device=device,
                proprioception=propri_b,
                dof_mask=dof_b,
                robot_type_id=rti,
            )
            decode_successes.append(bool(ok))

            def _to_t(x):
                if isinstance(x, np.ndarray):
                    return torch.from_numpy(x).to(device=device, dtype=torch.float32)
                return x.to(device=device, dtype=torch.float32)

            pred_norm_b = _to_t(pred_norm_b)  # [T, D]
            gt_dec_b = _to_t(gt_norm_b)
            gt_norm_chunk_b = action_chunk[b].to(
                device=device, dtype=torch.float32
            )  # [T, D]

            def _unnorm(a_2d):
                a = a_2d.unsqueeze(0)  # [1, T, D]
                if uses_dof_mask:
                    return normalizer.unnormalize_data(a, ds_b, dof_b)[0]
                return normalizer.unnormalize_data(a, ds_b, None)[0]

            predict_actions_norm.append(pred_norm_b)
            gt_actions_norm.append(gt_norm_chunk_b)
            predict_actions.append(_unnorm(pred_norm_b))
            gt_actions.append(_unnorm(gt_norm_chunk_b))
            gt_actions_decoded.append(_unnorm(gt_dec_b))

        output = {
            "predict_action": torch.stack(predict_actions, dim=0),
            "gt_action": torch.stack(gt_actions, dim=0),
            "predict_action_norm": torch.stack(predict_actions_norm, dim=0),
            "gt_action_norm": torch.stack(gt_actions_norm, dim=0),
            "gt_action_decoded": torch.stack(gt_actions_decoded, dim=0),
            "token_accuracy": token_accs,
            "token_accuracy_layers": token_acc_layers,
            "decode_success": decode_successes,
        }
        return output

    @torch.no_grad()
    def generate_ar_action(
        self,
        input_ids,
        action_horizon,
        action_dim,
        labels=None,
        attention_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        pixel_values_history=None,
        image_grid_thw_history=None,
        history_image_mask=None,
        proprioception=None,
        agent_pos_mask=None,
        dof_mask=None,
        dataset_names=None,
        action_chunk=None,
        robot_type_id=None,
        prefix_length=None,
        do_sample=False,
        temperature=1.0,
        max_new_tokens=100,
        **kwargs,
    ):
        """Autoregressive discrete action-token evaluation (true open-loop).

        Truncates the sequence at the assistant AR-response start (derived from
        ``labels``, or an explicit ``prefix_length``), autoregressively generates
        the action tokens with HF ``.generate`` (KV-cache accelerated), decodes the
        emitted RVQ codes back to a continuous action, and reports action L1 +
        generated-token accuracy (overall + per-RVQ-layer, aligned over the shorter
        of gen/gt code subsequences). Same output dict contract as ar_tf.

        Sampling: ``do_sample=False`` (default) = deterministic greedy/argmax, in
        which case ``temperature`` is inert (HF only applies it when sampling). Set
        ``do_sample=True`` to draw from the softmax at ``temperature`` (higher =
        more random → lower token accuracy; T→0 approaches greedy).

        Note: real rollout (no teacher forcing) — exercises HF ``.generate`` through
        qwen3_5's KV cache + gated-delta linear-attention state. Validated bs=1/2/4.
        """
        if getattr(self, "tokenizer_mixin", None) is None or (
            getattr(self, "action_mapper", None) is None
        ):
            raise RuntimeError("AR generation requires a discrete action tokenizer.")

        device = input_ids.device
        batch_size = input_ids.shape[0]

        token_range = self.action_mapper.get("token_range")

        def _is_code(t):
            if token_range is not None:
                return (t >= token_range["min"]) & (t <= token_range["max"])
            lst = self.action_token_id_set.get("action_token_list", [])
            return t >= lst[0] if lst else torch.zeros_like(t, dtype=torch.bool)

        # Per-sample assistant AR-response start (= prompt length). These differ
        # across frames because the state string tokenizes to a variable number of
        # BPE tokens, so a single prompt cut is wrong for batch_size>1. Left-pad
        # every prompt to a common length so batched greedy .generate() aligns
        # (standard variable-length batched generation); the 3D-mRoPE positions are
        # derived from the attention_mask, which accounts for the left padding.
        if prefix_length is not None:
            starts = [int(prefix_length)] * batch_size
        else:
            if labels is None:
                raise ValueError(
                    "generate_ar_action needs `labels` or `prefix_length` to locate "
                    "the assistant AR-response start."
                )
            starts = []
            for b in range(batch_size):
                nz = (labels[b] != -100).nonzero().flatten()
                starts.append(
                    int(nz.min().item()) if nz.numel() else input_ids.shape[1]
                )

        pad_id = self.processor.tokenizer.pad_token_id
        eos_id = self.processor.tokenizer.eos_token_id
        max_start = max(starts)
        prompt_ids = input_ids.new_full((batch_size, max_start), pad_id)
        prompt_attn = torch.zeros(
            (batch_size, max_start), dtype=torch.long, device=device
        )
        for b in range(batch_size):
            s = starts[b]
            prompt_ids[b, max_start - s :] = input_ids[b, :s]
            prompt_attn[b, max_start - s :] = 1

        gen_kwargs = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_attn,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "pixel_values_history": pixel_values_history,
            "image_grid_thw_history": image_grid_thw_history,
            "history_image_mask": history_image_mask,
            "proprioception": proprioception,
            "agent_pos_mask": agent_pos_mask,
            "dof_mask": dof_mask,
            "dataset_names": dataset_names,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        gen_out = self.generate(
            **gen_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            use_cache=True,
            eos_token_id=[eos_id],
            pad_token_id=pad_id,
        )
        # Left padding aligns every sample's generation to start at max_start.
        gen_tail = gen_out[:, max_start:]  # [B, gen_len]

        normalizer = self.action_preprocessor.normalizer_action
        uses_dof_mask = bool(
            getattr(self.tokenizer_mixin, "uses_dof_mask_for_unnorm", True)
        )

        predict_actions, gt_actions = [], []
        predict_actions_norm, gt_actions_norm = [], []
        token_accs, token_acc_layers, decode_successes = [], [], []

        for b in range(batch_size):
            gen_ids_b = gen_tail[b]
            gen_codes = gen_ids_b[_is_code(gen_ids_b)]
            gt_codes = (
                labels[b][_is_code(labels[b])]
                if labels is not None
                else gen_codes.new_empty(0)
            )

            ds_b = (
                dataset_names[b : b + 1]
                if isinstance(dataset_names, (list, tuple))
                else dataset_names
            )
            propri_b = proprioception[b : b + 1] if proprioception is not None else None
            dof_b = dof_mask[b : b + 1] if dof_mask is not None else None
            rti = robot_type_id
            try:
                _rti = self.tokenizer_mixin.get_robot_type_ids(
                    uids=[None],
                    dataset_names=(
                        list(ds_b) if isinstance(ds_b, (list, tuple)) else [ds_b]
                    ),
                )
                if _rti:
                    rti = _rti[0]
            except Exception as exc:  # pragma: no cover - diagnostic fallback
                logger.warning("get_robot_type_ids failed (%s); using %s", exc, rti)

            # Generated-token accuracy on the aligned (min-length) code subsequence.
            m = min(int(gen_codes.numel()), int(gt_codes.numel()))
            acc_dict = (
                self._ar_token_layer_accuracy(gen_codes[:m], gt_codes[:m])
                if m > 0
                else {"action_accuracy": float("nan")}
            )
            token_accs.append(acc_dict["action_accuracy"])
            token_acc_layers.append(
                {k: v for k, v in acc_dict.items() if k != "action_accuracy"}
            )

            pred_norm_b, ok = self.tokenizer_mixin.decode_action(
                output_ids=gen_ids_b.unsqueeze(0),
                action_mapper=self.action_mapper,
                action_horizon=action_horizon,
                action_dim=action_dim,
                device=device,
                proprioception=propri_b,
                dof_mask=dof_b,
                robot_type_id=rti,
            )
            decode_successes.append(bool(ok))

            def _to_t(x):
                if isinstance(x, np.ndarray):
                    return torch.from_numpy(x).to(device=device, dtype=torch.float32)
                return x.to(device=device, dtype=torch.float32)

            pred_norm_b = _to_t(pred_norm_b)

            def _unnorm(a2d):
                a = a2d.unsqueeze(0)
                return normalizer.unnormalize_data(
                    a, ds_b, dof_b if uses_dof_mask else None
                )[0]

            predict_actions_norm.append(pred_norm_b)
            predict_actions.append(_unnorm(pred_norm_b))
            # Ground truth is only available in (open-loop) evaluation; on the
            # robot ``action_chunk`` is None and there is nothing to compare
            # against, so skip the GT branch entirely (deployment path).
            if action_chunk is not None:
                gt_norm_chunk_b = action_chunk[b].to(device=device, dtype=torch.float32)
                gt_actions_norm.append(gt_norm_chunk_b)
                gt_actions.append(_unnorm(gt_norm_chunk_b))

        output = {
            "predict_action": torch.stack(predict_actions, dim=0),
            "predict_action_norm": torch.stack(predict_actions_norm, dim=0),
            "token_accuracy": token_accs,
            "token_accuracy_layers": token_acc_layers,
            "decode_success": decode_successes,
            # Number of tokens actually generated (before EOS or the cap); lets a
            # caller see whether generation stopped early or ran to max_new_tokens.
            "num_new_tokens": int(gen_tail.shape[1]),
        }
        # GT tensors are only populated in evaluation (action_chunk supplied).
        if action_chunk is not None:
            output["gt_action"] = torch.stack(gt_actions, dim=0)
            output["gt_action_norm"] = torch.stack(gt_actions_norm, dim=0)
        return output
