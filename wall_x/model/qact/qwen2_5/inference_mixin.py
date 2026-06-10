import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torchdiffeq import odeint

logger = logging.getLogger(__name__)


def topk_right_tie_break_1d(x, k):
    L = x.size(0)
    x_rev = torch.flip(x, [0])
    idx_in_rev = torch.argsort(x_rev, dim=0, descending=True, stable=True)
    orig_idx = (L - 1) - idx_in_rev
    topk_idx = orig_idx[:k]
    topk_vals = x[topk_idx]
    return topk_vals, topk_idx


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


class VLAInferenceMixin:

    # TODO: Integrate with the optimized implementation.
    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        proprioception: Optional[torch.FloatTensor] = None,
        dataset_names: Optional[str] = None,
        agent_pos_mask: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.Tensor]]:
        """
        Prepare model input embeddings, including text, image, video, and proprioception embeddings.

        Args:
            input_ids: Input token IDs.
            inputs_embeds: Precomputed input embeddings, if provided.
            pixel_values: Image pixel values.
            pixel_values_videos: Video pixel values.
            image_grid_thw: Image grid time, height, and width.
            video_grid_thw: Video grid time, height, and width.
            proprioception: Proprioception data.
            dataset_names: Dataset names.
            agent_pos_mask: Agent position mask.
            attention_mask: Attention mask.

        Returns:
            inputs_embeds: Complete input embeddings.
            attention_mask: Processed attention mask.
        """
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            # Process image embeddings.
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # Process video embeddings.
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # Process proprioception embeddings.
            if proprioception is not None and not getattr(
                self.config, "use_state_string_representation", False
            ):
                proprioception = proprioception.to(inputs_embeds.device)
                agent_pos_mask = agent_pos_mask.to(inputs_embeds.device)
                proprio_embed = self.action_preprocessor.proprioception_proj(
                    proprioception,
                    dataset_names,
                    agent_pos_mask,
                    use_history=proprioception.shape[1] > 1,
                )
                proprioception_mask = (
                    input_ids == self.action_token_id_set["propri_token_id"]
                )
                inputs_embeds[proprioception_mask] = proprio_embed.reshape(
                    -1, inputs_embeds.shape[-1]
                ).to(inputs_embeds.dtype)

            # Process the attention mask.
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        return inputs_embeds, attention_mask

    def prepare_position_ids(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
    ) -> torch.LongTensor:
        """
        Prepare position IDs, including RoPE delta calculation and caching.

        Args:
            input_ids: Input token IDs.
            inputs_embeds: Input embeddings.
            image_grid_thw: Image grid time, height, and width.
            video_grid_thw: Video grid time, height, and width.
            second_per_grid_ts: Time step for each grid.
            attention_mask: Attention mask.
            position_ids: Precomputed position IDs, if provided.
            cache_position: Cache position.
            past_key_values: Previous key/value cache.

        Returns:
            position_ids: Calculated position IDs.
        """
        # RoPE deltas cannot be calculated once the attention mask is 4D.
        if position_ids is None and (
            attention_mask is None or attention_mask.ndim == 2
        ):
            # Calculate RoPE indices once per generation in the prefill stage.
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # Reuse the previous RoPE deltas to obtain the correct position IDs.
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        return position_ids

    @torch.no_grad()
    def generate_dllm_action(
        self,
        input_ids,
        action_horizon,
        action_dim,
        ar_action_dim,
        total_ar_step,
        use_ar_action: bool = False,
        num_inference_timesteps: int = 10,
        prefix_length: Optional[int] = None,  # Prefix length.
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        moe_token_types: Optional[torch.LongTensor] = None,
        positional_masks: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        action_chunk: Optional[torch.FloatTensor] = None,
        proprioception: Optional[torch.FloatTensor] = None,
        unnorm_proprioception: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        dataset_names: Optional[str] = None,
        dof_mask: Optional[torch.FloatTensor] = None,
        agent_pos_mask: Optional[torch.FloatTensor] = None,
        robot_type_id: Optional[int] = None,  # Used for v3.1 delta decoding.
        **kwargs,
    ):
        batch_size = (
            input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        )
        inputs_embeds, attention_mask = self.prepare_inputs_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            proprioception=proprioception,
            dataset_names=dataset_names,
            agent_pos_mask=agent_pos_mask,
            attention_mask=attention_mask,
        )
        position_ids = self.prepare_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )

        noisy_action = torch.randn(
            size=(batch_size, action_horizon, action_dim),
            dtype=torch.float32,
            device=inputs_embeds.device,
        )

        times = torch.linspace(
            0.0,
            1.0,
            num_inference_timesteps + 1,
            device=inputs_embeds.device,
            dtype=torch.float32,
        )

        dt = times[1] - times[0]
        time_0 = times[0].unsqueeze(0).repeat(noisy_action.shape[0])
        action_embed, adarms_cond = self.action_preprocessor.step(
            timestep=time_0, noisy_action=noisy_action, dof_mask=dof_mask
        )
        action_embed = action_embed.reshape(-1, inputs_embeds.shape[-1]).to(
            inputs_embeds.dtype
        )
        flow_action_mask = input_ids == self.action_token_id_set["action_token_id"]
        inputs_embeds[flow_action_mask] = action_embed

        # Compute the token span for each expert group after permutation.
        group_size = torch.zeros(
            self.config.num_experts, dtype=torch.long, device="cpu"
        )
        for i in range(self.config.num_experts):
            group_size[i] = (moe_token_types == i).sum()

        # Calculate start and end indices for each expert group
        start_indices = torch.cumsum(group_size, dim=0) - group_size
        end_indices = torch.cumsum(group_size, dim=0)

        prefetch_output = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            moe_token_types=moe_token_types,
            positional_masks=positional_masks,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            adarms_conds=[None, adarms_cond],
            start_indices=start_indices,
            end_indices=end_indices,
        )

        hidden_states = prefetch_output.last_hidden_state
        prefix_kv_cache = prefetch_output.past_key_values

        action_hidden_states = hidden_states[flow_action_mask].to(torch.float32)
        v_0 = self.action_preprocessor.action_proj_back(
            action_hidden_states[:, : self.action_preprocessor.action_hidden_size]
        )
        noisy_action = noisy_action + dt * v_0.reshape(
            batch_size, action_horizon, action_dim
        )

        ar_prefix_length = positional_masks["ar_action_mask"].nonzero(as_tuple=True)[
            -1
        ][
            0
        ]  # Attention, not tested yet with batch
        start_indices[1] -= ar_prefix_length.item()
        end_indices -= ar_prefix_length.item()
        if hasattr(prefix_kv_cache, "key_cache"):
            for layer_i in range(len(prefix_kv_cache.key_cache)):
                prefix_kv_cache.key_cache[layer_i] = prefix_kv_cache.key_cache[layer_i][
                    :, :, :ar_prefix_length, :
                ]
                prefix_kv_cache.value_cache[layer_i] = prefix_kv_cache.value_cache[
                    layer_i
                ][:, :, :ar_prefix_length, :]
        else:
            for layer_i in range(len(prefix_kv_cache.layers)):
                prefix_kv_cache.layers[layer_i].keys = prefix_kv_cache.layers[
                    layer_i
                ].keys[:, :, :ar_prefix_length, :]
                prefix_kv_cache.layers[layer_i].values = prefix_kv_cache.layers[
                    layer_i
                ].values[:, :, :ar_prefix_length, :]

        ar_postfix_position_ids = position_ids[:, :, ar_prefix_length:]
        ar_postfix_inputs_embeds = inputs_embeds[:, ar_prefix_length:, :]
        ar_postfix_attention_mask = attention_mask[:, ar_prefix_length:]
        ar_postfix_moe_token_types = moe_token_types[:, ar_prefix_length:]
        ar_postfix_input_ids = input_ids[:, ar_prefix_length:]

        noisy_steps_per_sample = (
            torch.ceil((1.0 - times[0] - dt) * (total_ar_step + 1)).long() - 1
        )
        noisy_steps_per_sample = noisy_steps_per_sample.clamp(min=0, max=total_ar_step)
        ar_step = total_ar_step - noisy_steps_per_sample
        ready_step = [False] * total_ar_step
        # Always initialize ar_tokens to record generated discrete codes.
        # Use a tensor to support multi-token steps in v3.1 delta mode.
        ar_tokens = torch.full(
            (ar_postfix_inputs_embeds.shape[1],),
            -1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        ar_postfix_inputs_embeds, ready_step, ar_tokens = self.update_ar_action(
            ar_postfix_inputs_embeds,
            hidden_states[:, ar_prefix_length:],
            positional_masks["ar_action_mask"][:, ar_prefix_length:],
            ar_step=ar_step,
            ready_step=ready_step,
            ar_tokens=ar_tokens,
        )

        pad_token_id = self.processor.tokenizer.pad_token_id
        padding_mask = input_ids == pad_token_id

        ar_postfix_length = input_ids.shape[-1] - ar_prefix_length
        _ar_postfix_attention_mask = torch.ones(
            (batch_size, ar_postfix_length, ar_prefix_length + ar_postfix_length),
            dtype=torch.bool,
            device=ar_postfix_attention_mask.device,
        )

        ar_postfix_padding_mask = padding_mask[
            :, ar_prefix_length:
        ]  # [batch_size, postfix_length]
        full_padding_mask = padding_mask  # [batch_size, prefix_length + postfix_length]

        for batch_idx in range(padding_mask.shape[0]):
            # Disable rows for padded query positions.
            _ar_postfix_attention_mask[
                batch_idx, ar_postfix_padding_mask[batch_idx], :
            ] = False
            # Disable columns for padded key positions.
            _ar_postfix_attention_mask[batch_idx, :, full_padding_mask[batch_idx]] = (
                False
            )

        # The 3D attention mask already filters padding, but kv-cache generation
        # differs from training, so rebuild the mask before passing it to the model.
        if not positional_masks.get("ar_visible", True):
            ar_action_mask = positional_masks["ar_action_mask"] != 0
            flow_positions = moe_token_types == 1
            flow_flow_mask = flow_positions[:, :, None] & flow_positions[:, None, :]
            ar_ar_mask = ar_action_mask[:, :, None] & ar_action_mask[:, None, :]
            flow_ar_mask = flow_flow_mask | ar_ar_mask
            kvcache_flow_ar_mask = flow_ar_mask[:, ar_prefix_length:]

            affected = ar_action_mask | flow_positions  # (B, N)
            affected_pair = affected[:, :, None] & affected[:, None, :]
            affected_pair = affected_pair[:, ar_prefix_length:]

            _ar_postfix_attention_mask = torch.where(
                affected_pair, kvcache_flow_ar_mask, _ar_postfix_attention_mask
            )

        def step_with_kvcache(
            timestep,
            noisy_action,
            ar_postfix_inputs_embeds,
            _ar_postfix_attention_mask,
            ar_prefix_length,
            ready_step,
            dt,
            ar_tokens,
        ):
            action_mask = (
                ar_postfix_input_ids == self.action_token_id_set["action_token_id"]
            )
            assert action_mask.any(), "No action token found in input_ids"
            timestep = timestep.unsqueeze(0).repeat(noisy_action.shape[0])
            action_embed, adarms_cond = self.action_preprocessor.step(
                timestep=timestep, noisy_action=noisy_action, dof_mask=dof_mask
            )
            action_embed = action_embed.reshape(-1, ar_postfix_inputs_embeds.shape[-1])

            # Clone inputs_embeds locally to keep this step isolated.
            temp_inputs_embeds = ar_postfix_inputs_embeds.clone()
            temp_inputs_embeds[action_mask] = action_embed.to(temp_inputs_embeds.dtype)

            transformer_outputs = self.model(
                input_ids=None,
                attention_mask=_ar_postfix_attention_mask,
                position_ids=ar_postfix_position_ids,
                past_key_values=prefix_kv_cache,
                inputs_embeds=temp_inputs_embeds,
                moe_token_types=ar_postfix_moe_token_types,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                adarms_conds=[None, adarms_cond],
                start_indices=start_indices,
                end_indices=end_indices,
            )

            hidden_states = transformer_outputs.last_hidden_state

            noisy_steps_per_sample = (
                torch.ceil((1.0 - timestep - dt) * (total_ar_step + 1)).long() - 1
            )
            noisy_steps_per_sample = noisy_steps_per_sample.clamp(
                min=0, max=total_ar_step
            )
            ar_step = total_ar_step - noisy_steps_per_sample
            ar_postfix_inputs_embeds, ready_step, ar_tokens = self.update_ar_action(
                ar_postfix_inputs_embeds,
                hidden_states,
                positional_masks["ar_action_mask"][:, ar_prefix_length:],
                ar_step=ar_step,
                ready_step=ready_step,
                ar_tokens=ar_tokens,
            )

            action_hidden_states = hidden_states[action_mask].to(torch.float32)
            v_t = self.action_preprocessor.action_proj_back(
                action_hidden_states[:, : self.action_preprocessor.action_hidden_size]
            )

            return v_t.reshape(batch_size, action_horizon, action_dim)

        action_trajectory = odeint(
            lambda timestep, noisy_action: step_with_kvcache(
                timestep,
                noisy_action,
                ar_postfix_inputs_embeds,
                _ar_postfix_attention_mask,
                ar_prefix_length,
                ready_step,
                dt,
                ar_tokens,
            ),
            noisy_action,
            times[1:],
            method="euler",
        )
        output = {}
        if use_ar_action:
            # Decode through tokenizer_mixin.decode_action() for both fast and v3.1 delta modes.
            # ar_tokens is already a tensor.
            ar_tokens_tensor = ar_tokens.unsqueeze(0)  # [1, seq_len]

            predict_action, decode_success = self.tokenizer_mixin.decode_action(
                output_ids=ar_tokens_tensor,
                action_mapper=self.action_mapper,
                action_horizon=action_horizon,
                action_dim=ar_action_dim,
                device=inputs_embeds.device,
                proprioception=proprioception,
                dof_mask=dof_mask,
                robot_type_id=robot_type_id,
            )

            if not decode_success:
                logger.warning("Error in DLLM decoding action, predict_action is None")
                output["predict_action"] = None
            else:
                # unnormalize
                if isinstance(predict_action, np.ndarray):
                    predict_action = torch.tensor(
                        predict_action, device=inputs_embeds.device
                    )
                elif predict_action.device != inputs_embeds.device:
                    predict_action = predict_action.to(inputs_embeds.device)

                # Add the batch dimension when needed.
                if predict_action.dim() == 2:
                    predict_action = predict_action.unsqueeze(0)

                # Decide whether dof_mask is needed from the mixin setting.
                uses_dof_mask = (
                    self.tokenizer_mixin.uses_dof_mask_for_unnorm
                    if self.tokenizer_mixin is not None
                    else True
                )

                if uses_dof_mask:
                    predict_action = (
                        self.action_preprocessor.normalizer_action.unnormalize_data(
                            predict_action, dataset_names, dof_mask
                        )
                    )
                else:
                    predict_action = (
                        self.action_preprocessor.normalizer_action.unnormalize_data(
                            predict_action, dataset_names, None
                        )
                    )
                output["predict_action"] = predict_action

        else:
            predict_action = action_trajectory[-1]
            predict_action = (
                self.action_preprocessor.normalizer_action.unnormalize_data(
                    predict_action, dataset_names
                )
            )
            output["predict_action"] = predict_action

        if action_chunk is not None:
            output["gt_action"] = (
                self.action_preprocessor.normalizer_action.unnormalize_data(
                    action_chunk, dataset_names
                )
            )
        return output

    def update_ar_action2(
        self,
        ar_postfix_inputs_embeds,
        ar_hidden_states,
        ar_action_mask,
        ar_step,
        ready_step,
        remask=False,
        ar_tokens=None,
    ):
        ready_num = sum(ready_step)
        if ready_num >= ar_step and not remask:
            return ar_postfix_inputs_embeds, ready_step, ar_tokens
        valid_steps = torch.unique(
            ar_action_mask[(ar_action_mask != 0) & (ar_action_mask != -1)]
        )
        placeholder_seq_embed = self.model.embed_tokens(
            torch.Tensor(self.processor.placeholder_seq)
            .to(self.model.device)
            .to(torch.int)
        )
        step_confidences = []
        step_token_ids = []
        for step in valid_steps:
            if ready_step[step - 1] and not remask:
                step_confidences.append(1.0)
                step_token_ids.append(None)
                continue
            step_mask = ar_action_mask == step
            step_hidden_states = ar_hidden_states[step_mask]
            logits = self.lm_head(step_hidden_states)
            pred_ids = logits.argmax(dim=-1)
            step_confidence = torch.softmax(logits, dim=-1)
            total_step_confidence = step_confidence.max(dim=-1).values.mean()
            step_confidences.append(total_step_confidence)
            step_token_ids.append(pred_ids)
        step_confidences = torch.tensor(step_confidences).to(self.model.device)
        _, top_k_indices = topk_right_tie_break_1d(step_confidences, ar_step)
        for i, step in enumerate(valid_steps):
            if ready_step[step - 1] and not remask:
                continue
            step_mask = ar_action_mask == step
            step_indices = step_mask.nonzero(as_tuple=True)[1]
            if i in top_k_indices:
                # update
                ar_postfix_inputs_embeds[0, step_indices, :] = self.model.embed_tokens(
                    step_token_ids[i]
                )
                if ar_tokens is not None:
                    ar_tokens[step_indices] = step_token_ids[i]
                ready_step[step - 1] = True
            elif remask and ready_step[step - 1]:
                # back to placeholder
                ar_postfix_inputs_embeds[0, step_indices, :] = placeholder_seq_embed[
                    : len(step_indices)
                ]
                ready_step[step - 1] = False

        return ar_postfix_inputs_embeds, ready_step, ar_tokens

    def update_ar_action(
        self,
        ar_postfix_inputs_embeds,
        ar_hidden_states,
        ar_action_mask,
        ar_step,
        ready_step,
        remask=False,
        ar_tokens=None,
    ):
        ready_num = sum(ready_step)
        if ready_num >= ar_step and not remask:
            return ar_postfix_inputs_embeds, ready_step, ar_tokens
        update_step = ar_step - ready_num
        valid_steps = torch.unique(
            ar_action_mask[(ar_action_mask != 0) & (ar_action_mask != -1)]
        )
        placeholder_seq_embed = self.model.embed_tokens(
            torch.Tensor(self.processor.placeholder_seq)
            .to(self.model.device)
            .to(torch.int)
        )
        step_confidences = []
        step_token_ids = []

        logits = self.lm_head(ar_hidden_states)
        confidence = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1)
        pad_mask = pred_ids == 151668  # pad token id
        step_is_ready = []
        step_is_pad = []
        for step in valid_steps:
            step_mask = ar_action_mask == step
            step_pred_ids = pred_ids[step_mask]
            step_token_ids.append(step_pred_ids)

            is_pad = pad_mask[step_mask].all()
            step_is_pad.append(is_pad)

            is_ready = ready_step[step - 1]
            step_is_ready.append(is_ready)

            step_confidence = confidence[step_mask]
            total_step_confidence = step_confidence.max(dim=-1).values.mean()
            step_confidences.append(total_step_confidence)

        step_confidences = torch.tensor(step_confidences).to(self.model.device)
        step_is_ready = torch.tensor(step_is_ready, device=self.model.device)
        step_is_pad = torch.tensor(step_is_pad, device=self.model.device)

        non_ready_mask = ~step_is_ready
        nr_steps = valid_steps[non_ready_mask]  # 1-based step ID
        nr_conf = step_confidences[non_ready_mask]
        nr_pad = step_is_pad[non_ready_mask]

        if nr_steps.numel() == 0:
            return ar_postfix_inputs_embeds, ready_step, ar_tokens

        chosen_steps = []
        used_mask = torch.zeros_like(nr_steps, dtype=torch.bool)

        # Step 1: choose one non-PAD step by confidence when available.
        non_pad_mask = ~nr_pad
        if non_pad_mask.any():
            non_pad_conf = nr_conf[non_pad_mask]
            non_pad_steps = nr_steps[non_pad_mask]

            best_idx = torch.argmax(non_pad_conf)  # choose highest confidence
            chosen_step = non_pad_steps[best_idx]
            chosen_steps.append(chosen_step.item())

            # Mark this step as used.
            used_mask[nr_steps == chosen_step] = True

        # If enough steps were selected, reuse chosen_steps below.
        if len(chosen_steps) >= update_step:
            chosen_steps = torch.tensor(chosen_steps, device=self.model.device)
            # chosen_steps will be used below.
        else:
            # ============================================================
            # Step 2: choose PAD steps by descending index, ignoring confidence.
            # ============================================================
            remaining = update_step - len(chosen_steps)

            pad_mask_only = nr_pad & (~used_mask)
            pad_steps_only = nr_steps[pad_mask_only]

            if pad_steps_only.numel() > 0:
                # Sort from right to left.
                pad_sorted = torch.argsort(pad_steps_only, descending=True)
                pick = pad_steps_only[pad_sorted[:remaining]]

                chosen_steps.extend(pick.tolist())
                used_mask[(nr_steps.unsqueeze(1) == pick).any(dim=-1)] = True

            # Continue to Step 3 if more steps are still needed.
            if len(chosen_steps) < update_step:
                remaining = update_step - len(chosen_steps)

                # ============================================================
                # Step 3: choose remaining non-PAD steps by confidence.
                # ============================================================
                non_pad_left_mask = (~nr_pad) & (~used_mask)

                if non_pad_left_mask.any():
                    left_conf = nr_conf[non_pad_left_mask]
                    left_steps = nr_steps[non_pad_left_mask]

                    sorted_idx = torch.argsort(left_conf, descending=True)
                    pick = left_steps[sorted_idx[:remaining]]

                    chosen_steps.extend(pick.tolist())
                    used_mask[(nr_steps.unsqueeze(1) == pick).any(dim=-1)] = True

        chosen_steps = torch.tensor(chosen_steps, device=self.model.device)
        for i, step in enumerate(valid_steps):
            if ready_step[step - 1] and not remask:
                continue
            step_mask = ar_action_mask == step
            step_indices = step_mask.nonzero(as_tuple=True)[1]
            if step in chosen_steps:
                # update
                ar_postfix_inputs_embeds[0, step_indices, :] = self.model.embed_tokens(
                    step_token_ids[i]
                )
                if ar_tokens is not None:
                    ar_tokens[step_indices] = step_token_ids[i]
                ready_step[step - 1] = True
            elif remask and ready_step[step - 1]:
                # back to placeholder
                ar_postfix_inputs_embeds[0, step_indices, :] = placeholder_seq_embed[
                    : len(step_indices)
                ]
                ready_step[step - 1] = False

        return ar_postfix_inputs_embeds, ready_step, ar_tokens

    def update_infer_dllm_position_mask(self, model_input):
        positional_masks = self.tokenizer_mixin.update_placeholder_mask(
            self.processor,
            model_input["prefix_length"],
            model_input["input_ids"],
        )
        model_input["positional_masks"] = positional_masks
        model_input["positional_masks"]["ar_visible"] = True
        return model_input

    @torch.no_grad()
    def generate_flow_action_no_cache(
        self,
        input_ids,
        action_horizon,
        action_dim,
        num_inference_timesteps: int = 10,
        prefix_length: Optional[int] = None,  # Prefix length.
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        moe_token_types: Optional[torch.LongTensor] = None,
        start_indices: Optional[torch.Tensor] = None,
        end_indices: Optional[torch.Tensor] = None,
        positional_masks: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        action_chunk: Optional[torch.FloatTensor] = None,
        proprioception: Optional[torch.FloatTensor] = None,
        unnorm_proprioception: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        dataset_names: Optional[str] = None,
        dof_mask: Optional[torch.FloatTensor] = None,
        agent_pos_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        # assert self.config._attn_implementation == "sdpa", "generate_flow_action only support sdpa attn implementation"
        batch_size = (
            input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        )
        assert (
            batch_size == 1
        ), "generate_flow_action_no_cache only support batch size 1"

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if proprioception is not None and not getattr(
                self.config, "use_state_string_representation", False
            ):
                proprioception = proprioception.to(inputs_embeds.device)
                agent_pos_mask = agent_pos_mask.to(inputs_embeds.device)
                proprio_embed = self.action_preprocessor.proprioception_proj(
                    proprioception,
                    dataset_names,
                    agent_pos_mask,
                    use_history=proprioception.shape[1] > 1,
                )
                proprioception_mask = (
                    input_ids == self.action_token_id_set["propri_token_id"]
                )
                inputs_embeds[proprioception_mask] = proprio_embed.reshape(
                    -1, inputs_embeds.shape[-1]
                ).to(inputs_embeds.dtype)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (
            attention_mask is None or attention_mask.ndim == 2
        ):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if start_indices is None or end_indices is None:
            # Compute the token span for each expert group after permutation.
            group_size = torch.zeros(
                self.config.num_experts, dtype=torch.long, device="cpu"
            )
            for i in range(self.config.num_experts):
                group_size[i] = (moe_token_types == i).sum()

            # Calculate start and end indices for each expert group
            start_indices = torch.cumsum(group_size, dim=0) - group_size
            end_indices = torch.cumsum(group_size, dim=0)

        if action_chunk is not None:
            action_chunk = action_chunk.to(inputs_embeds.device).to(torch.float32)

        output = {}

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
        flow_action_mask = input_ids == self.action_token_id_set["action_token_id"]

        if not dof_mask.all():
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
            action_embed, adarms_cond = self.action_preprocessor.step(
                timestep=timestep, noisy_action=noisy_action, dof_mask=dof_mask
            )
            action_embed = action_embed.reshape(-1, inputs_embeds.shape[-1]).to(
                inputs_embeds.dtype
            )

            inputs_embeds[flow_action_mask] = action_embed
            model_output = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                moe_token_types=moe_token_types,
                start_indices=start_indices,
                end_indices=end_indices,
                positional_masks=positional_masks,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                adarms_conds=[None, adarms_cond],
            )

            hidden_states = model_output.last_hidden_state
            action_hidden_states = hidden_states[flow_action_mask].to(torch.float32)
            action_pred = self.action_preprocessor.action_proj_back(
                action_hidden_states[:, : self.action_preprocessor.action_hidden_size]
            )
            if getattr(self.config, "use_x_pred", False):
                v_t = (action_pred - noisy_action) / torch.clamp(1 - timestep, min=0.05)
            else:
                v_t = action_pred

            if not dof_mask.all():
                v_t = (v_padding) * (1 - dof_mask) + v_t * dof_mask

            return v_t.reshape(batch_size, action_horizon, action_dim)

        action_trajectory = odeint(step, noisy_action, times, method="euler")

        predict_action = action_trajectory[-1]
        predict_action = self.action_preprocessor.normalizer_action.unnormalize_data(
            predict_action, dataset_names
        )
        output["predict_action"] = predict_action
        # normalize action chunk to get gt_action
        if action_chunk is not None:
            output["gt_action"] = (
                self.action_preprocessor.normalizer_action.unnormalize_data(
                    action_chunk, dataset_names
                )
            )

        return output
