#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> get_rope_index(
    const torch::optional<torch::Tensor> &input_ids,
    const torch::optional<torch::Tensor> &image_grid_thw,
    const torch::optional<torch::Tensor> &video_grid_thw,
    const torch::optional<torch::Tensor> &second_per_grid_ts,
    const torch::optional<torch::Tensor> &attention_mask,
    int spatial_merge_size,
    int image_token_id,
    int video_token_id,
    int vision_start_token_id,
    float tokens_per_second);
