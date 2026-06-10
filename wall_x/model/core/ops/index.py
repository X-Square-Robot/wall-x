"""Index computation operators for attention."""

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from wall_x.model.core.ops.base import OpsProxy

logger = logging.getLogger(__name__)

# Sentinel value for padding in window index computation.
# Valid indices are always >= 0 (they are positional offsets within flattened
# grid tensors), so any negative value is safe.  -100 is chosen by convention
# (same as HuggingFace's ignore_index for cross-entropy loss).
_PAD_VALUE = -100


def _compute_vision_position_ids(
    input_tokens,
    token_set,
    st,
    image_grid_thw,
    video_grid_thw,
    second_per_grid_ts,
    image_index,
    video_index,
    remain_images,
    remain_videos,
    image_token_id,
    video_token_id,
    spatial_merge_size,
    tokens_per_second,
    device,
    llm_pos_ids_list,
):
    """Compute position IDs for a single vision token (image or video).

    Returns updated (st, image_index, video_index, remain_images, remain_videos).
    """
    if image_token_id in token_set and remain_images > 0:
        ed_image = input_tokens.index(image_token_id, st)
    else:
        ed_image = len(input_tokens) + 1
    if video_token_id in token_set and remain_videos > 0:
        ed_video = input_tokens.index(video_token_id, st)
    else:
        ed_video = len(input_tokens) + 1

    if ed_image < ed_video:
        if image_index >= len(image_grid_thw):
            raise IndexError(
                f"image_index {image_index} out of range (have {len(image_grid_thw)} image grids)"
            )
        t, h, w = (
            image_grid_thw[image_index][0],
            image_grid_thw[image_index][1],
            image_grid_thw[image_index][2],
        )
        # Images are single-frame: no temporal progression, so second_per_grid_t = 0.
        # This makes all image tokens share t_index = 0 (only spatial positions vary).
        second_per_grid_t = 0
        image_index += 1
        remain_images -= 1
        ed = ed_image
    else:
        if video_index >= len(video_grid_thw):
            raise IndexError(
                f"video_index {video_index} out of range (have {len(video_grid_thw)} video grids)"
            )
        t, h, w = (
            video_grid_thw[video_index][0],
            video_grid_thw[video_index][1],
            video_grid_thw[video_index][2],
        )
        if second_per_grid_ts is not None:
            second_per_grid_t = second_per_grid_ts[video_index]
        else:
            second_per_grid_t = 1.0
        video_index += 1
        remain_videos -= 1
        ed = ed_video

    llm_grid_t, llm_grid_h, llm_grid_w = (
        int(t),
        int(h) // spatial_merge_size,
        int(w) // spatial_merge_size,
    )
    text_len = ed - st
    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    llm_pos_ids_list.append(
        torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + st_idx
    )
    range_tensor = torch.arange(llm_grid_t, device=device).view(-1, 1)
    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
    time_tensor = expanded_range * second_per_grid_t * tokens_per_second
    t_index = time_tensor.long().flatten()
    h_index = (
        torch.arange(llm_grid_h, device=device)
        .view(1, -1, 1)
        .expand(llm_grid_t, -1, llm_grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(llm_grid_w, device=device)
        .view(1, 1, -1)
        .expand(llm_grid_t, llm_grid_h, -1)
        .flatten()
    )
    llm_pos_ids_list.append(
        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
    )
    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
    return st, image_index, video_index, remain_images, remain_videos


def _compute_text_only_positions(input_ids, attention_mask):
    """Compute position IDs for text-only inputs (no vision tokens)."""
    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = (
            position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        )
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[
            0
        ]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1)
            .expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
    return position_ids, mrope_position_deltas


class GetRopeIndexOp(OpsProxy):
    """Compute 3D RoPE position indices for multimodal inputs.

    Signature: get_rope_index(input_ids, image_grid_thw, video_grid_thw,
                              second_per_grid_ts, attention_mask, spatial_merge_size,
                              image_token_id, video_token_id, vision_start_token_id,
                              tokens_per_second) -> (position_ids, mrope_position_deltas)
    """

    @property
    def _external_accel_name(self):
        return "get_rope_index"

    def _get_cuda_kernel(self):
        try:
            from wall_x.model.core.ops._cuda_wrappers import GetRopeIndex

            return GetRopeIndex()
        except ImportError:
            return None
        except Exception as e:
            logger.warning("GetRopeIndexOp: CUDA kernel load failed: %s", e)
            return None

    def _pytorch_fallback(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor],
        video_grid_thw: Optional[torch.LongTensor],
        second_per_grid_ts: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        spatial_merge_size: int,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        tokens_per_second: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids is None or (image_grid_thw is None and video_grid_thw is None):
            return _compute_text_only_positions(input_ids, attention_mask)

        mrope_position_deltas = []
        device = input_ids.device
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids_i in enumerate(total_input_ids):
            input_ids_i = input_ids_i[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(
                input_ids_i == vision_start_token_id
            ).squeeze(1)
            # Boundary check: ensure vision_start + 1 doesn't exceed sequence length
            valid_mask = (vision_start_indices + 1) < len(input_ids_i)
            vision_start_indices = vision_start_indices[valid_mask]
            vision_tokens = input_ids_i[vision_start_indices + 1]
            image_nums = int((vision_tokens == image_token_id).sum())
            video_nums = int((vision_tokens == video_token_id).sum())
            input_tokens = input_ids_i.tolist()
            token_set = set(input_tokens)
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                st, image_index, video_index, remain_images, remain_videos = (
                    _compute_vision_position_ids(
                        input_tokens,
                        token_set,
                        st,
                        image_grid_thw,
                        video_grid_thw,
                        second_per_grid_ts,
                        image_index,
                        video_index,
                        remain_images,
                        remain_videos,
                        image_token_id,
                        video_token_id,
                        spatial_merge_size,
                        tokens_per_second,
                        device,
                        llm_pos_ids_list,
                    )
                )
            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=device).view(1, -1).expand(3, -1)
                    + st_idx
                )
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=total_input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas


class GetWindowIndexOp(OpsProxy):
    """Compute window attention indices for ViT.

    Signature: get_window_index(grid_thw, window_size, spatial_merge_size,
                                patch_size, spatial_merge_unit=1) -> (window_index, cu_window_seqlens)
    """

    @property
    def _external_accel_name(self):
        return "get_window_index"

    def _get_cuda_kernel(self):
        try:
            from wall_x.model.core.ops._cuda_wrappers import get_window_index_cuda

            return get_window_index_cuda
        except ImportError:
            return None
        except Exception as e:
            logger.warning("GetWindowIndexOp: CUDA kernel load failed: %s", e)
            return None

    def _pytorch_fallback(
        self,
        grid_thw,
        window_size,
        spatial_merge_size,
        patch_size,
        spatial_merge_unit=1,
    ):
        device = grid_thw.device
        vit_merger_window_size = window_size // spatial_merge_size // patch_size
        window_index_list = []
        cu_window_seqlens = [0]
        window_index_id = 0
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // spatial_merge_size
            llm_grid_w = grid_w // spatial_merge_size
            index = torch.arange(
                grid_t * llm_grid_h * llm_grid_w, device=device
            ).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = (
                vit_merger_window_size - llm_grid_h % vit_merger_window_size
            ) % vit_merger_window_size
            pad_w = (
                vit_merger_window_size - llm_grid_w % vit_merger_window_size
            ) % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", _PAD_VALUE)
            index_padded = (
                index_padded.reshape(
                    grid_t,
                    num_windows_h,
                    vit_merger_window_size,
                    num_windows_w,
                    vit_merger_window_size,
                )
                .permute(0, 1, 3, 2, 4)
                .reshape(
                    grid_t,
                    num_windows_h * num_windows_w,
                    vit_merger_window_size,
                    vit_merger_window_size,
                )
            )
            seqlens = (index_padded != _PAD_VALUE).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != _PAD_VALUE]
            window_index_list.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += int(grid_t) * int(llm_grid_h) * int(llm_grid_w)
        window_index = torch.cat(window_index_list, dim=0)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens, dtype=grid_thw.dtype, device=grid_thw.device
        )
        return window_index, cu_window_seqlens


get_rope_index = GetRopeIndexOp()
get_window_index = GetWindowIndexOp()
