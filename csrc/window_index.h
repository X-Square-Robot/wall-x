#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> get_window_index_cuda(
    torch::Tensor grid_thw,
    int spatial_merge_size,
    int vit_merger_window_size,
    int patch_size,
    int spatial_merge_unit);
