#include <torch/extension.h>

torch::Tensor fused_rot_pos_emb_cuda(
    torch::Tensor inv_freq,
    torch::Tensor grid_thw,
    int spatial_merge_size);
