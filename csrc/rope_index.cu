#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__

#include <vector>
#include <tuple>
#include <optional>
#include <cstdint> // for int64_t

#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#define MAX_SEQ_LEN 8192
#define MAX_VISION_TOKENS 64
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

struct VisionDescriptor
{
    int64_t start_pos;
    int64_t token_pos;
    int64_t patch_count;
    int64_t grid_t, grid_h, grid_w;
    float time_interval;
    int64_t is_video;
    int64_t position_offset;
};

__device__ __forceinline__ int64_t fast_div(int64_t a, int64_t b)
{
    return __float2ll_rd(__ll2float_rn(a) * __frcp_rn(__ll2float_rn(b)));
}

__device__ __forceinline__ void get_3d_coords(int64_t patch_idx, int64_t H, int64_t W,
                                              int64_t &t, int64_t &h, int64_t &w)
{
    int64_t hw = H * W;
    t = fast_div(patch_idx, hw);
    int64_t remaining = patch_idx - t * hw;
    h = fast_div(remaining, W);
    w = remaining - h * W;
}

__global__ void compute_vision_counts(
    const int64_t *input_ids,              // (batch_size, seq_len)
    const int64_t *attention_mask,         // (batch_size, seq_len)
    int64_t *image_counts,                 // (batch_size,)
    int64_t *video_counts,                 // (batch_size,)
    const int64_t batch_size,
    const int64_t seq_len,
    const int64_t image_token_id,
    const int64_t video_token_id,
    const int64_t vision_start_token_id)
{
    int64_t batch_idx = blockIdx.x;
    int64_t thread_idx = threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    __shared__ int64_t shared_image_counts[MAX_THREADS_PER_BLOCK];
    __shared__ int64_t shared_video_counts[MAX_THREADS_PER_BLOCK];

    int64_t thread_image_count = 0;
    int64_t thread_video_count = 0;

    for (int64_t i = thread_idx; i < seq_len - 1; i += blockDim.x)
    {
        if ((attention_mask != nullptr) && attention_mask[batch_idx * seq_len + i] == 0)
            continue;

        int64_t token_id = input_ids[batch_idx * seq_len + i];

        if (token_id == vision_start_token_id && i + 1 < seq_len)
        {
            int64_t next_token = input_ids[batch_idx * seq_len + i + 1];

            if (next_token == image_token_id)
            {
                thread_image_count++;
            }
            else if (next_token == video_token_id)
            {
                thread_video_count++;
            }
        }
    }

    shared_image_counts[thread_idx] = thread_image_count;
    shared_video_counts[thread_idx] = thread_video_count;
    __syncthreads();

    for (int64_t stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (thread_idx < stride)
        {
            shared_image_counts[thread_idx] += shared_image_counts[thread_idx + stride];
            shared_video_counts[thread_idx] += shared_video_counts[thread_idx + stride];
        }
        __syncthreads();
    }

    if (thread_idx == 0)
    {
        image_counts[batch_idx] = shared_image_counts[0];
        video_counts[batch_idx] = shared_video_counts[0];
    }
}


__global__ void preprocess_vision_tokens(
    const int64_t *input_ids,            // (batch_size, seq_len)
    const int64_t *attention_mask,       // (batch_size, seq_len)
    const int64_t *image_grid_thw,       // (max_images, 3)
    const int64_t *video_grid_thw,       // (max_videos, 3)
    const float *second_per_grid_ts,     // (max_videos,)
    const int64_t *image_counts,         // (batch_size,)
    const int64_t *video_counts,         // (batch_size,)
    VisionDescriptor *vision_desc,       // (batch_size, MAX_VISION_TOKENS)
    int64_t *vision_counts,              // (batch_size,)
    int64_t *text_lengths,               // (batch_size, MAX_VISION_TOKENS+1)
    int64_t *position_offsets,           // (batch_size, MAX_VISION_TOKENS+1)
    const int64_t batch_size,
    const int64_t seq_len,
    const int64_t spatial_merge_size,
    const int64_t image_token_id,
    const int64_t video_token_id,
    const int64_t vision_start_token_id,
    const float tokens_per_second)
{
    int64_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size)
        return;

    int64_t image_idx = 0, video_idx = 0;
    for (int64_t i = 0; i < batch_idx; i++)
    {
        image_idx += image_counts[i];
        video_idx += video_counts[i];
    }

    int64_t vision_count = 0;
    int64_t current_pos = 0;
    int64_t position_offset = 0;

    for (int64_t i = 0; i < seq_len - 1; i++)
    {
        if ((attention_mask != nullptr) && attention_mask[batch_idx * seq_len + i] == 0)
            continue;

        int64_t token_id = input_ids[batch_idx * seq_len + i];

        if (token_id == vision_start_token_id)
        {
            int64_t next_token = input_ids[batch_idx * seq_len + i + 1];

            if (next_token == image_token_id || next_token == video_token_id)
            {
                int64_t vision_pos = i + 1;
                text_lengths[batch_idx * (MAX_VISION_TOKENS + 1) + vision_count] = vision_pos - current_pos;
                position_offsets[batch_idx * (MAX_VISION_TOKENS + 1) + vision_count] = position_offset;
                position_offset += (vision_pos - current_pos);

                int64_t T, H, W;
                float time_interval = 0.0f;
                int64_t is_video = (next_token == video_token_id) ? 1 : 0;

                if (is_video == 0)
                {
                    T = image_grid_thw[image_idx * 3 + 0];
                    H = image_grid_thw[image_idx * 3 + 1];
                    W = image_grid_thw[image_idx * 3 + 2];
                    image_idx++;
                }
                else
                {
                    T = video_grid_thw[video_idx * 3 + 0];
                    H = video_grid_thw[video_idx * 3 + 1];
                    W = video_grid_thw[video_idx * 3 + 2];
                    time_interval = second_per_grid_ts[video_idx];
                    video_idx++;
                }

                int64_t H_merged = H / spatial_merge_size;
                int64_t W_merged = W / spatial_merge_size;
                int64_t patch_count = T * H_merged * W_merged;

                if (vision_count < MAX_VISION_TOKENS)
                {
                    VisionDescriptor &desc = vision_desc[batch_idx * MAX_VISION_TOKENS + vision_count];
                    desc.start_pos = current_pos;
                    desc.token_pos = vision_pos;
                    desc.patch_count = patch_count;
                    desc.grid_t = T;
                    desc.grid_h = H_merged;
                    desc.grid_w = W_merged;
                    desc.time_interval = time_interval;
                    desc.is_video = is_video;
                    desc.position_offset = position_offset;

                    if (is_video)
                    {
                        position_offset += max(static_cast<int64_t>((T - 1) * time_interval * tokens_per_second) + 1,
                                              static_cast<int64_t>(max(H_merged, W_merged)));
                    }
                    else
                    {
                        position_offset += max(H_merged, W_merged);
                    }

                    current_pos = vision_pos + patch_count;
                    vision_count++;
                }

                i = vision_pos + patch_count - 1;
            }
        }
    }

    vision_counts[batch_idx] = vision_count;

    int64_t effective_len = seq_len;
    if (attention_mask != nullptr)
    {
        for (int64_t i = seq_len - 1; i >= 0; i--)
        {
            if (attention_mask[batch_idx * seq_len + i] != 0)
            {
                effective_len = i + 1;
                break;
            }
        }
    }

    text_lengths[batch_idx * (MAX_VISION_TOKENS + 1) + vision_count] = effective_len - current_pos;
    position_offsets[batch_idx * (MAX_VISION_TOKENS + 1) + vision_count] = position_offset;
}


__global__ void compute_3d_positions(
    const int64_t *input_ids,                // (batch_size, seq_len)
    const int64_t *attention_mask,           // (batch_size, seq_len)
    const VisionDescriptor *vision_desc,     // (batch_size, MAX_VISION_TOKENS)
    const int64_t *vision_counts,            // (batch_size,)
    const int64_t *text_lengths,             // (batch_size, MAX_VISION_TOKENS+1)
    const int64_t *position_offsets,         // (batch_size, MAX_VISION_TOKENS+1)
    int64_t *position_ids,                   // (3, batch_size, seq_len)
    int64_t *mrope_deltas,                   // (batch_size,)
    const int64_t batch_size,
    const int64_t seq_len,
    const float tokens_per_second)
{
    int64_t batch_idx = blockIdx.x;
    int64_t thread_idx = threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    __shared__ VisionDescriptor shared_visions[MAX_VISION_TOKENS];
    __shared__ int64_t shared_position_offsets[MAX_VISION_TOKENS + 1];
    __shared__ int64_t shared_max_positions[MAX_THREADS_PER_BLOCK];

    int64_t shared_vision_count = vision_counts[batch_idx];

    for (int64_t i = thread_idx; i < MAX_VISION_TOKENS; i += blockDim.x)
    {
        if (i < shared_vision_count)
        {
            shared_visions[i] = vision_desc[batch_idx * MAX_VISION_TOKENS + i];
        }
    }

    for (int64_t i = thread_idx; i < MAX_VISION_TOKENS + 1; i += blockDim.x)
    {
        shared_position_offsets[i] = position_offsets[batch_idx * (MAX_VISION_TOKENS + 1) + i];
    }

    int64_t thread_max_position = -1;

    __syncthreads();

    for (int64_t token_idx = thread_idx; token_idx < seq_len; token_idx += blockDim.x)
    {
        bool is_valid_token = true;
        if (attention_mask != nullptr)
        {
            is_valid_token = attention_mask[batch_idx * seq_len + token_idx] != 0;
        }

        if (!is_valid_token)
        {
            position_ids[0 * batch_size * seq_len + batch_idx * seq_len + token_idx] = 1;
            position_ids[1 * batch_size * seq_len + batch_idx * seq_len + token_idx] = 1;
            position_ids[2 * batch_size * seq_len + batch_idx * seq_len + token_idx] = 1;
            continue;
        }

        int64_t segment_idx = -9999999;
        int64_t local_pos = token_idx;

        for (int64_t v = 0; v < shared_vision_count; v++)
        {
            if (token_idx < shared_visions[v].token_pos)
            {
                segment_idx = v;
                local_pos = token_idx - (v > 0 ? shared_visions[v - 1].token_pos + shared_visions[v - 1].patch_count : 0);
                break;
            }
            else if (token_idx < shared_visions[v].token_pos + shared_visions[v].patch_count)
            {
                segment_idx = -(v + 1);
                local_pos = token_idx - shared_visions[v].token_pos;
                break;
            }
        }

        if (segment_idx == -9999999)
        {
            segment_idx = shared_vision_count;
            int64_t last_vision_end = 0;
            if (shared_vision_count > 0)
            {
                last_vision_end = shared_visions[shared_vision_count - 1].token_pos +
                                shared_visions[shared_vision_count - 1].patch_count;
            }
            local_pos = token_idx - last_vision_end;
        }

        int64_t pos_t, pos_h, pos_w;

        if (segment_idx >= 0)
        {
            int64_t offset = shared_position_offsets[segment_idx];
            pos_t = pos_h = pos_w = offset + local_pos;
        }
        else
        {
            int64_t vision_idx = -(segment_idx + 1);
            const VisionDescriptor &desc = shared_visions[vision_idx];

            int64_t t, h, w;
            get_3d_coords(local_pos, desc.grid_h, desc.grid_w, t, h, w);

            pos_t = static_cast<int64_t>(t * desc.time_interval * tokens_per_second) + desc.position_offset;
            pos_h = h + desc.position_offset;
            pos_w = w + desc.position_offset;
        }

        position_ids[0 * batch_size * seq_len + batch_idx * seq_len + token_idx] = pos_t;
        position_ids[1 * batch_size * seq_len + batch_idx * seq_len + token_idx] = pos_h;
        position_ids[2 * batch_size * seq_len + batch_idx * seq_len + token_idx] = pos_w;

        int64_t max_pos = max(pos_t, max(pos_h, pos_w));
        thread_max_position = max(thread_max_position, max_pos);
    }

    shared_max_positions[thread_idx] = thread_max_position;
    __syncthreads();

    for (int64_t stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (thread_idx < stride)
        {
            shared_max_positions[thread_idx] = max(shared_max_positions[thread_idx],
                                                 shared_max_positions[thread_idx + stride]);
        }
        __syncthreads();
    }

    if (thread_idx == 0)
    {
        int64_t global_max_position = shared_max_positions[0];
        mrope_deltas[batch_idx] = global_max_position + 1 - seq_len;
    }
}


void launch_optimized_3d_rope_kernel(
    const int64_t *input_ids,
    const int64_t *attention_mask,
    const int64_t *image_grid_thw,
    const int64_t *video_grid_thw,
    const float *second_per_grid_ts,
    int64_t *position_ids,
    int64_t *mrope_deltas,
    int64_t batch_size,
    int64_t seq_len,
    int64_t spatial_merge_size,
    int64_t image_token_id,
    int64_t video_token_id,
    int64_t vision_start_token_id,
    float tokens_per_second)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    torch::Device device(torch::kCUDA, at::cuda::current_device());

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(device);

    auto vision_desc_tensor = torch::empty(
        {batch_size * MAX_VISION_TOKENS * static_cast<int64_t>(sizeof(VisionDescriptor))},
        torch::TensorOptions().dtype(torch::kUInt8).device(device)
    );
    VisionDescriptor *d_vision_desc = reinterpret_cast<VisionDescriptor*>(vision_desc_tensor.data_ptr());

    auto vision_counts_tensor = torch::empty({batch_size}, options);
    auto text_lengths_tensor = torch::empty({batch_size * (MAX_VISION_TOKENS + 1)}, options);
    auto position_offsets_tensor = torch::empty({batch_size * (MAX_VISION_TOKENS + 1)}, options);
    auto image_counts_tensor = torch::empty({batch_size}, options);
    auto video_counts_tensor = torch::empty({batch_size}, options);

    int64_t *d_vision_counts = vision_counts_tensor.data_ptr<int64_t>();
    int64_t *d_text_lengths = text_lengths_tensor.data_ptr<int64_t>();
    int64_t *d_position_offsets = position_offsets_tensor.data_ptr<int64_t>();
    int64_t *d_image_counts = image_counts_tensor.data_ptr<int64_t>();
    int64_t *d_video_counts = video_counts_tensor.data_ptr<int64_t>();

    dim3 index_grid(static_cast<unsigned int>(batch_size));
    dim3 index_block(256);

    compute_vision_counts<<<index_grid, index_block, 0, stream>>>(
        input_ids, attention_mask,
        d_image_counts, d_video_counts,
        batch_size, seq_len, image_token_id, video_token_id, vision_start_token_id);

    int64_t threads_per_block = std::min(batch_size, static_cast<int64_t>(256));
    int64_t num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    dim3 preprocess_grid(static_cast<unsigned int>(num_blocks));
    dim3 preprocess_block(static_cast<unsigned int>(threads_per_block));

    preprocess_vision_tokens<<<preprocess_grid, preprocess_block, 0, stream>>>(
        input_ids, attention_mask, image_grid_thw, video_grid_thw,
        second_per_grid_ts, d_image_counts, d_video_counts,
        d_vision_desc, d_vision_counts, d_text_lengths, d_position_offsets,
        batch_size, seq_len, spatial_merge_size,
        image_token_id, video_token_id, vision_start_token_id, tokens_per_second);

    threads_per_block = std::min(static_cast<int64_t>(seq_len), static_cast<int64_t>(MAX_THREADS_PER_BLOCK));
    if (threads_per_block < 32) threads_per_block = 32;

    int64_t power_of_2 = 1;
    while (power_of_2 < threads_per_block) power_of_2 *= 2;
    if (power_of_2 > MAX_THREADS_PER_BLOCK) power_of_2 = MAX_THREADS_PER_BLOCK;
    threads_per_block = power_of_2;

    dim3 compute_grid(static_cast<unsigned int>(batch_size));
    dim3 compute_block(static_cast<unsigned int>(threads_per_block));

    compute_3d_positions<<<compute_grid, compute_block, 0, stream>>>(
        input_ids, attention_mask, d_vision_desc, d_vision_counts,
        d_text_lengths, d_position_offsets, position_ids, mrope_deltas,
        batch_size, seq_len, tokens_per_second);

    AT_CUDA_CHECK(cudaGetLastError());
}

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
    float tokens_per_second)
{
    TORCH_CHECK(input_ids.has_value(), "input_ids cannot be None");
    TORCH_CHECK(input_ids->dim() == 2, "input_ids must be 2D tensor (batch_size, seq_len)");

    const auto batch_size = input_ids->size(0);
    const auto seq_len = input_ids->size(1);
    const auto device = input_ids->device();

    TORCH_CHECK(device.is_cuda(), "All tensors must be on CUDA device");

    torch::Tensor input_ids_tensor = input_ids->contiguous();
    const int64_t *input_ids_ptr = input_ids_tensor.data_ptr<int64_t>();

    torch::Tensor attention_mask_tensor;
    const int64_t *attention_mask_ptr = nullptr;

    if (attention_mask.has_value()) {
        attention_mask_tensor = attention_mask->contiguous();
        attention_mask_ptr = attention_mask_tensor.data_ptr<int64_t>();
    }

    torch::Tensor image_grid_thw_tensor;
    const int64_t *image_grid_thw_ptr = nullptr;

    if (image_grid_thw.has_value()) {
        TORCH_CHECK(image_grid_thw->dim() == 2 && image_grid_thw->size(1) == 3,
                    "image_grid_thw must be shape (num_images, 3)");
        image_grid_thw_tensor = image_grid_thw->contiguous();
        image_grid_thw_ptr = image_grid_thw_tensor.data_ptr<int64_t>();
    }

    torch::Tensor video_grid_thw_tensor;
    const int64_t *video_grid_thw_ptr = nullptr;

    if (video_grid_thw.has_value()) {
        TORCH_CHECK(video_grid_thw->dim() == 2 && video_grid_thw->size(1) == 3,
                    "video_grid_thw must be shape (num_videos, 3)");
        video_grid_thw_tensor = video_grid_thw->contiguous();
        video_grid_thw_ptr = video_grid_thw_tensor.data_ptr<int64_t>();
    }

    torch::Tensor second_per_grid_ts_tensor;
    const float *second_per_grid_ts_ptr = nullptr;

    if (second_per_grid_ts.has_value()) {
        TORCH_CHECK(second_per_grid_ts->dim() == 1,
                    "second_per_grid_ts must be 1D tensor");
        second_per_grid_ts_tensor = second_per_grid_ts->contiguous();
        second_per_grid_ts_ptr = second_per_grid_ts_tensor.data_ptr<float>();
    }

    if (!image_grid_thw.has_value() && !video_grid_thw.has_value()) {
        torch::Tensor position_ids;
        torch::Tensor mrope_deltas;

        if (attention_mask.has_value()) {
            auto cumsum_result = attention_mask_tensor.to(torch::kInt64).cumsum(-1) - 1;
            position_ids = cumsum_result.masked_fill_(attention_mask_tensor.eq(0), 1);
            position_ids = position_ids.unsqueeze(0).expand({3, -1, -1});

            auto max_position_ids = std::get<0>(std::get<0>(position_ids.max(0)).max(-1, true));
            mrope_deltas = max_position_ids + 1 - attention_mask_tensor.size(-1);
            mrope_deltas = mrope_deltas.view({batch_size, 1});
        } else {
            auto pos_range = torch::arange(seq_len, torch::TensorOptions().dtype(torch::kInt64).device(device));
            position_ids = pos_range.view({1, 1, -1}).expand({3, batch_size, -1});
            mrope_deltas = torch::zeros({batch_size, 1}, torch::TensorOptions().dtype(torch::kInt64).device(device));
        }

        return std::make_tuple(position_ids, mrope_deltas);
    }

    auto position_ids = torch::empty({3, batch_size, seq_len},
                                     torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto mrope_deltas = torch::empty({batch_size, 1},
                                     torch::TensorOptions().dtype(torch::kInt64).device(device));

    launch_optimized_3d_rope_kernel(
        input_ids_ptr,
        attention_mask_ptr,
        image_grid_thw_ptr,
        video_grid_thw_ptr,
        second_per_grid_ts_ptr,
        position_ids.data_ptr<int64_t>(),
        mrope_deltas.data_ptr<int64_t>(),
        static_cast<int64_t>(batch_size),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(spatial_merge_size),
        static_cast<int64_t>(image_token_id),
        static_cast<int64_t>(video_token_id),
        static_cast<int64_t>(vision_start_token_id),
        tokens_per_second);

    return std::make_tuple(position_ids, mrope_deltas);
}
