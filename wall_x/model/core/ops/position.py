"""Position embedding operators."""

import torch

from wall_x.model.core.ops.base import OpsProxy


class PosEmbedInterpOp(OpsProxy):
    """Fused Qwen3.5 ViT position embedding interpolation.

    Signature:
      pos_embed_interp(grid_thw, pos_embed_weight, num_grid_per_side,
                       spatial_merge_size, total_output_tokens=None)
    """

    @property
    def _external_accel_name(self):
        return "pos_embed_interp"

    def _pytorch_fallback(
        self,
        grid_thw,
        pos_embed_weight,
        num_grid_per_side: int,
        spatial_merge_size: int,
        total_output_tokens=None,
    ):
        # total_output_tokens lets the external_accel backend skip a GPU->CPU sync;
        # the PyTorch fallback derives sizes from grid_thw directly.
        del total_output_tokens
        out_dtype = pos_embed_weight.dtype
        pos_embed_weight = pos_embed_weight.float()
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = pos_embed_weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for _t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * num_grid_per_side
            base_h_ceil = h_idxs_ceil * num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=torch.float32, device=device)
        pos_embeds = pos_embed_weight[idx_tensor] * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )

        patch_pos_embeds_permute = []
        merge_size = spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(
                    t, h // merge_size, merge_size, w // merge_size, merge_size, -1
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute).to(out_dtype)


pos_embed_interp = PosEmbedInterpOp()
