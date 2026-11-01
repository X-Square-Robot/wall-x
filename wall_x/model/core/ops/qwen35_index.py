"""Qwen3.5-specific index computation operators."""

import itertools
from typing import Optional, Tuple

import torch

from wall_x.model.core.ops.base import OpsProxy


class Qwen35GetRopeIndexOp(OpsProxy):
    """Compute Qwen3.5 3D RoPE position indices from modality-type groups.

    Signature:
      qwen35_get_rope_index(input_ids, mm_token_type_ids, image_grid_thw,
                            video_grid_thw, attention_mask, spatial_merge_size)
        -> (position_ids, mrope_position_deltas)
    """

    @property
    def _external_accel_name(self):
        return "qwen35_get_rope_index"

    def _pytorch_fallback(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        spatial_merge_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=torch.int64,
            device=input_ids.device,
        )
        mrope_position_deltas = []
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                valid_mask = attention_mask[batch_idx].bool()
                current_input_ids = current_input_ids[valid_mask]
                input_token_type = input_token_type[valid_mask]

            input_type_group = []
            for key, group in itertools.groupby(
                enumerate(input_token_type.tolist()), lambda x: x[1]
            ):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device)
                        .view(1, -1)
                        .expand(3, -1)
                        + current_pos
                    )
                    current_pos += text_len
                else:
                    grid_iter = grid_iters[modality_type]
                    if grid_iter is None:
                        raise ValueError(
                            f"missing grid_thw for modality_type={modality_type}"
                        )
                    grid_thw = next(grid_iter)
                    llm_grid_t = grid_thw[0].item()
                    llm_grid_h = grid_thw[1].item() // spatial_merge_size
                    llm_grid_w = grid_thw[2].item() // spatial_merge_size
                    image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
                    position_width = torch.arange(
                        current_pos,
                        current_pos + llm_grid_w,
                        device=input_ids.device,
                    ).repeat(llm_grid_h * llm_grid_t)
                    position_height = torch.arange(
                        current_pos,
                        current_pos + llm_grid_h,
                        device=input_ids.device,
                    ).repeat_interleave(llm_grid_w * llm_grid_t)
                    position_temporal = torch.full(
                        (image_seq_length,),
                        current_pos,
                        device=input_ids.device,
                        dtype=torch.long,
                    )
                    llm_pos_ids_list.append(
                        torch.stack(
                            [position_temporal, position_height, position_width],
                            dim=0,
                        )
                    )
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = (
                    llm_positions.to(position_ids.device)
                )
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(current_input_ids)
            )

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas


qwen35_get_rope_index = Qwen35GetRopeIndexOp()
