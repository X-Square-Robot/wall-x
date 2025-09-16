#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__

#include <vector>
#include <tuple>
#include <optional>
#include <type_traits>

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

template <typename T>
struct VisionDescriptor
{
    T start_pos;
    T token_pos;
    T patch_count;
    T grid_t, grid_h, grid_w;
    float time_interval;
    T is_video;
    T position_offset;
};

template <typename T>
__device__ __forceinline__ T fast_div(T a, T b)
{
    if constexpr (std::is_same_v<T, int>)
    {
        return __float2int_rd(__int2float_rn(a) * __frcp_rn(__int2float_rn(b)));
    }
    else
    {
        return a / b;
    }
}

template <typename T>
__device__ __forceinline__ void get_3d_coords(T patch_idx, T H, T W,
                                              T &t, T &h, T &w)
{
    T hw = H * W;
    t = fast_div(patch_idx, hw);
    T remaining = patch_idx - t * hw;
    h = fast_div(remaining, W);
    w = remaining - h * W;
}

// 模板化的统计kernel
template <typename IndexType>
__global__ void compute_vision_counts_template(
    const IndexType *input_ids,
    const IndexType *attention_mask,
    IndexType *image_counts,
    IndexType *video_counts,
    const IndexType batch_size,
    const IndexType seq_len,
    const IndexType image_token_id,
    const IndexType video_token_id,
    const IndexType vision_start_token_id)
{
    IndexType batch_idx = blockIdx.x;
    IndexType thread_idx = threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    __shared__ IndexType shared_image_counts[MAX_THREADS_PER_BLOCK];
    __shared__ IndexType shared_video_counts[MAX_THREADS_PER_BLOCK];

    IndexType thread_image_count = 0;
    IndexType thread_video_count = 0;

    for (IndexType i = thread_idx; i < seq_len - 1; i += blockDim.x)
    {
        if ((attention_mask != nullptr) && attention_mask[batch_idx * seq_len + i] == 0)
            continue;

        IndexType token_id = input_ids[batch_idx * seq_len + i];

        if (token_id == vision_start_token_id && i + 1 < seq_len)
        {
            IndexType next_token = input_ids[batch_idx * seq_len + i + 1];

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

    for (IndexType stride = blockDim.x / 2; stride > 0; stride /= 2)
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

template <typename IndexType>
__global__ void preprocess_vision_tokens_template(
    const IndexType *input_ids,               // (batch_size, seq_len)
    const IndexType *attention_mask,          // (batch_size, seq_len)
    const IndexType *image_grid_thw,          // (max_images, 3)
    const IndexType *video_grid_thw,          // (max_videos, 3)
    const float *second_per_grid_ts,          // (max_videos,)
    const IndexType *image_counts,            // (batch_size,)
    const IndexType *video_counts,            // (batch_size,)
    VisionDescriptor<IndexType> *vision_desc, // (batch_size, MAX_VISION_TOKENS)
    IndexType *vision_counts,                 // (batch_size,)
    IndexType *text_lengths,                  // (batch_size, MAX_VISION_TOKENS+1)
    IndexType *position_offsets,              // (batch_size, MAX_VISION_TOKENS+1)
    const IndexType batch_size,
    const IndexType seq_len,
    const IndexType spatial_merge_size,
    const IndexType image_token_id,
    const IndexType video_token_id,
    const IndexType vision_start_token_id,
    const float tokens_per_second)
{
    IndexType batch_idx = blockIdx.x;
    if (batch_idx >= batch_size)
        return;

    __shared__ IndexType shared_vision_positions[MAX_VISION_TOKENS];
    __shared__ IndexType shared_vision_types[MAX_VISION_TOKENS];
    __shared__ IndexType vision_count;

    if (threadIdx.x == 0)
        vision_count = 0;
    __syncthreads();

    for (IndexType i = threadIdx.x; i < seq_len; i += blockDim.x)
    {
        if ((attention_mask != nullptr) && attention_mask[batch_idx * seq_len + i] == 0)
            continue;

        IndexType token_id = input_ids[batch_idx * seq_len + i];

        if (token_id == vision_start_token_id && i + 1 < seq_len)
        {
            IndexType next_token = input_ids[batch_idx * seq_len + i + 1];
            if (next_token == image_token_id || next_token == video_token_id)
            {
                IndexType pos;
                if constexpr (std::is_same_v<IndexType, int>)
                {
                    pos = atomicAdd(&vision_count, 1);
                }
                else
                {
                    pos = atomicAdd(reinterpret_cast<unsigned long long *>(&vision_count), 1ULL);
                }
                if (pos < MAX_VISION_TOKENS)
                {
                    shared_vision_positions[pos] = i + 1;
                    shared_vision_types[pos] = (next_token == video_token_id) ? 1 : 0;
                }
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        vision_counts[batch_idx] = vision_count;

        for (IndexType i = 0; i < vision_count - 1; i++)
        {
            for (IndexType j = 0; j < vision_count - 1 - i; j++)
            {
                if (shared_vision_positions[j] > shared_vision_positions[j + 1])
                {
                    IndexType temp_pos = shared_vision_positions[j];
                    shared_vision_positions[j] = shared_vision_positions[j + 1];
                    shared_vision_positions[j + 1] = temp_pos;

                    IndexType temp_type = shared_vision_types[j];
                    shared_vision_types[j] = shared_vision_types[j + 1];
                    shared_vision_types[j + 1] = temp_type;
                }
            }
        }

        IndexType current_pos = 0;
        IndexType position_offset = 0;
        IndexType image_idx = 0, video_idx = 0;

        for (IndexType i = 0; i < batch_idx; i++)
        {
            image_idx += image_counts[i];
            video_idx += video_counts[i];
        }

        for (IndexType v = 0; v < vision_count; v++)
        {
            IndexType vision_pos = shared_vision_positions[v];
            IndexType vision_type = shared_vision_types[v];

            text_lengths[batch_idx * (MAX_VISION_TOKENS + 1) + v] = vision_pos - current_pos;
            position_offsets[batch_idx * (MAX_VISION_TOKENS + 1) + v] = position_offset;
            position_offset += (vision_pos - current_pos);

            IndexType T, H, W;
            float time_interval = 0.0f;

            if (vision_type == 0)
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

            IndexType H_merged = H / spatial_merge_size;
            IndexType W_merged = W / spatial_merge_size;
            IndexType patch_count = T * H_merged * W_merged;

            VisionDescriptor<IndexType> &desc = vision_desc[batch_idx * MAX_VISION_TOKENS + v];
            desc.start_pos = current_pos;
            desc.token_pos = vision_pos;
            desc.patch_count = patch_count;
            desc.grid_t = T;
            desc.grid_h = H_merged;
            desc.grid_w = W_merged;
            desc.time_interval = time_interval;
            desc.is_video = vision_type;
            desc.position_offset = position_offset;

            current_pos = vision_pos + patch_count;

            IndexType time_offset = static_cast<IndexType>((T - 1) * desc.time_interval * tokens_per_second) + 1;
            IndexType spatial_offset = static_cast<IndexType>(max(H_merged, W_merged));
            position_offset += max(time_offset, spatial_offset);
        }

        IndexType effective_len = seq_len;
        text_lengths[batch_idx * (MAX_VISION_TOKENS + 1) + vision_count] = effective_len - current_pos;
        position_offsets[batch_idx * (MAX_VISION_TOKENS + 1) + vision_count] = position_offset;
    }
}

template <typename IndexType>
__global__ void compute_3d_positions_template(
    const IndexType *input_ids,                     // (batch_size, seq_len)
    const IndexType *attention_mask,                // (batch_size, seq_len)
    const VisionDescriptor<IndexType> *vision_desc, // (batch_size, MAX_VISION_TOKENS)
    const IndexType *vision_counts,                 // (batch_size,)
    const IndexType *text_lengths,                  // (batch_size, MAX_VISION_TOKENS+1)
    const IndexType *position_offsets,              // (batch_size, MAX_VISION_TOKENS+1)
    IndexType *position_ids,                        // (3, batch_size, seq_len)
    IndexType *mrope_deltas,                        // (batch_size,)
    const IndexType batch_size,
    const IndexType seq_len,
    const float tokens_per_second)
{
    IndexType batch_idx = blockIdx.x;
    IndexType thread_idx = threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    __shared__ VisionDescriptor<IndexType> shared_visions[MAX_VISION_TOKENS];
    __shared__ IndexType shared_position_offsets[MAX_VISION_TOKENS + 1];
    __shared__ IndexType shared_max_positions[MAX_THREADS_PER_BLOCK];

    IndexType shared_vision_count = vision_counts[batch_idx];

    for (IndexType i = thread_idx; i < MAX_VISION_TOKENS; i += blockDim.x)
    {
        if (i < shared_vision_count)
        {
            shared_visions[i] = vision_desc[batch_idx * MAX_VISION_TOKENS + i];
        }
    }

    for (IndexType i = thread_idx; i < MAX_VISION_TOKENS + 1; i += blockDim.x)
    {
        shared_position_offsets[i] = position_offsets[batch_idx * (MAX_VISION_TOKENS + 1) + i];
    }

    IndexType thread_max_position = -1;

    __syncthreads();

    for (IndexType token_idx = thread_idx; token_idx < seq_len; token_idx += blockDim.x)
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

        IndexType segment_idx = static_cast<IndexType>(-9999999);
        IndexType local_pos = token_idx;

        for (IndexType v = 0; v < shared_vision_count; v++)
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

        if (segment_idx == static_cast<IndexType>(-9999999))
        {
            segment_idx = shared_vision_count;
            IndexType last_vision_end = 0;
            if (shared_vision_count > 0)
            {
                last_vision_end = shared_visions[shared_vision_count - 1].token_pos +
                                  shared_visions[shared_vision_count - 1].patch_count;
            }
            local_pos = token_idx - last_vision_end;
        }

        IndexType pos_t, pos_h, pos_w;

        if (segment_idx >= 0)
        {
            IndexType offset = shared_position_offsets[segment_idx];
            pos_t = pos_h = pos_w = offset + local_pos;
        }
        else
        {
            IndexType vision_idx = -(segment_idx + 1);
            const VisionDescriptor<IndexType> &desc = shared_visions[vision_idx];

            IndexType t, h, w;
            get_3d_coords(local_pos, desc.grid_h, desc.grid_w, t, h, w);

            pos_t = static_cast<IndexType>(t * desc.time_interval * tokens_per_second) + desc.position_offset;
            pos_h = h + desc.position_offset;
            pos_w = w + desc.position_offset;
        }

        position_ids[0 * batch_size * seq_len + batch_idx * seq_len + token_idx] = pos_t;
        position_ids[1 * batch_size * seq_len + batch_idx * seq_len + token_idx] = pos_h;
        position_ids[2 * batch_size * seq_len + batch_idx * seq_len + token_idx] = pos_w;

        IndexType max_pos = max(pos_t, max(pos_h, pos_w));
        thread_max_position = max(thread_max_position, max_pos);
    }

    shared_max_positions[thread_idx] = thread_max_position;
    __syncthreads();

    for (IndexType stride = blockDim.x / 2; stride > 0; stride /= 2)
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
        IndexType global_max_position = shared_max_positions[0];
        mrope_deltas[batch_idx] = global_max_position + 1 - seq_len;
    }
}

template <typename IndexType>
void launch_optimized_3d_rope_kernel_template(
    const IndexType *input_ids,
    const IndexType *attention_mask,
    const IndexType *image_grid_thw,
    const IndexType *video_grid_thw,
    const float *second_per_grid_ts,
    IndexType *position_ids,
    IndexType *mrope_deltas,
    IndexType batch_size,
    IndexType seq_len,
    IndexType spatial_merge_size,
    IndexType image_token_id,
    IndexType video_token_id,
    IndexType vision_start_token_id,
    float tokens_per_second)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    VisionDescriptor<IndexType> *d_vision_desc;
    IndexType *d_vision_counts;
    IndexType *d_text_lengths;
    IndexType *d_position_offsets;
    IndexType *d_image_counts;
    IndexType *d_video_counts;

    cudaMalloc(&d_vision_desc, batch_size * MAX_VISION_TOKENS * sizeof(VisionDescriptor<IndexType>));
    cudaMalloc(&d_vision_counts, batch_size * sizeof(IndexType));
    cudaMalloc(&d_text_lengths, batch_size * (MAX_VISION_TOKENS + 1) * sizeof(IndexType));
    cudaMalloc(&d_position_offsets, batch_size * (MAX_VISION_TOKENS + 1) * sizeof(IndexType));
    cudaMalloc(&d_image_counts, batch_size * sizeof(IndexType));
    cudaMalloc(&d_video_counts, batch_size * sizeof(IndexType));

    dim3 index_grid(batch_size);
    dim3 index_block(min(static_cast<IndexType>(512), seq_len));

    compute_vision_counts_template<<<index_grid, index_block, 0, stream>>>(
        input_ids, attention_mask,
        d_image_counts, d_video_counts,
        batch_size, seq_len, image_token_id, video_token_id, vision_start_token_id);

    dim3 preprocess_grid(batch_size);
    dim3 preprocess_block(min(seq_len, static_cast<IndexType>(MAX_THREADS_PER_BLOCK)));

    preprocess_vision_tokens_template<<<preprocess_grid, preprocess_block, 0, stream>>>(
        input_ids, attention_mask, image_grid_thw, video_grid_thw,
        second_per_grid_ts, d_image_counts, d_video_counts,
        d_vision_desc, d_vision_counts, d_text_lengths, d_position_offsets,
        batch_size, seq_len, spatial_merge_size,
        image_token_id, video_token_id, vision_start_token_id, tokens_per_second);

    IndexType threads_per_block = min(seq_len, static_cast<IndexType>(MAX_THREADS_PER_BLOCK));
    if (threads_per_block < 32)
        threads_per_block = 32;

    IndexType power_of_2 = 1;
    while (power_of_2 < threads_per_block)
        power_of_2 *= 2;
    if (power_of_2 > MAX_THREADS_PER_BLOCK)
        power_of_2 = MAX_THREADS_PER_BLOCK;
    threads_per_block = power_of_2;

    dim3 compute_grid(batch_size);
    dim3 compute_block(threads_per_block);

    compute_3d_positions_template<<<compute_grid, compute_block, 0, stream>>>(
        input_ids, attention_mask, d_vision_desc, d_vision_counts,
        d_text_lengths, d_position_offsets, position_ids, mrope_deltas,
        batch_size, seq_len, tokens_per_second);

    cudaFree(d_vision_desc);
    cudaFree(d_vision_counts);
    cudaFree(d_text_lengths);
    cudaFree(d_position_offsets);
    cudaFree(d_image_counts);
    cudaFree(d_video_counts);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
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

    auto input_dtype = input_ids->dtype();

    if (!image_grid_thw.has_value() && !video_grid_thw.has_value())
    {
        torch::Tensor position_ids;
        torch::Tensor mrope_deltas;

        if (attention_mask.has_value())
        {
            auto attention_mask_contiguous = attention_mask->contiguous();
            auto cumsum_result = attention_mask_contiguous.to(torch::kInt64).cumsum(-1) - 1;
            position_ids = cumsum_result.masked_fill_(attention_mask_contiguous.eq(0), 1);
            position_ids = position_ids.unsqueeze(0).expand({3, -1, -1}).to(input_dtype);

            auto max_position_ids = std::get<0>(std::get<0>(position_ids.max(0)).max(-1, true));
            mrope_deltas = max_position_ids + 1 - attention_mask_contiguous.size(-1);
            mrope_deltas = mrope_deltas.view({batch_size, 1});
        }
        else
        {
            auto pos_range = torch::arange(seq_len, torch::TensorOptions().dtype(input_dtype).device(device));
            position_ids = pos_range.view({1, 1, -1}).expand({3, batch_size, -1});
            mrope_deltas = torch::zeros({batch_size, 1}, torch::TensorOptions().dtype(input_dtype).device(device));
        }

        return std::make_tuple(position_ids, mrope_deltas);
    }

    auto input_ids_contiguous = input_ids->contiguous();

    torch::Tensor attention_mask_contiguous;
    const void *attention_mask_ptr = nullptr;
    if (attention_mask.has_value())
    {
        attention_mask_contiguous = attention_mask->to(device).to(input_dtype).contiguous();
        attention_mask_ptr = attention_mask_contiguous.data_ptr();
    }

    torch::Tensor image_grid_thw_contiguous;
    const void *image_grid_thw_ptr = nullptr;
    if (image_grid_thw.has_value())
    {
        TORCH_CHECK(image_grid_thw->dim() == 2 && image_grid_thw->size(1) == 3,
                    "image_grid_thw must be shape (num_images, 3)");
        image_grid_thw_contiguous = image_grid_thw->to(device).to(input_dtype).contiguous();
        image_grid_thw_ptr = image_grid_thw_contiguous.data_ptr();
    }

    torch::Tensor video_grid_thw_contiguous;
    const void *video_grid_thw_ptr = nullptr;
    if (video_grid_thw.has_value())
    {
        TORCH_CHECK(video_grid_thw->dim() == 2 && video_grid_thw->size(1) == 3,
                    "video_grid_thw must be shape (num_videos, 3)");
        video_grid_thw_contiguous = video_grid_thw->to(device).to(input_dtype).contiguous();
        video_grid_thw_ptr = video_grid_thw_contiguous.data_ptr();
    }

    torch::Tensor second_per_grid_ts_contiguous;
    const float *second_per_grid_ts_ptr = nullptr;
    if (second_per_grid_ts.has_value())
    {
        TORCH_CHECK(second_per_grid_ts->dim() == 1,
                    "second_per_grid_ts must be 1D tensor");
        second_per_grid_ts_contiguous = second_per_grid_ts->to(device).to(torch::kFloat32).contiguous();
        second_per_grid_ts_ptr = second_per_grid_ts_contiguous.data_ptr<float>();
    }

    auto position_ids = torch::empty({3, batch_size, seq_len},
                                     torch::TensorOptions().dtype(input_dtype).device(device));
    auto mrope_deltas = torch::empty({batch_size, 1},
                                     torch::TensorOptions().dtype(input_dtype).device(device));

    if (input_dtype == torch::kInt32)
    {
        launch_optimized_3d_rope_kernel_template<int>(
            static_cast<const int *>(input_ids_contiguous.data_ptr()),
            static_cast<const int *>(attention_mask_ptr),
            static_cast<const int *>(image_grid_thw_ptr),
            static_cast<const int *>(video_grid_thw_ptr),
            second_per_grid_ts_ptr,
            static_cast<int *>(position_ids.data_ptr()),
            static_cast<int *>(mrope_deltas.data_ptr()),
            static_cast<int>(batch_size),
            static_cast<int>(seq_len),
            spatial_merge_size,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            tokens_per_second);
    }
    else if (input_dtype == torch::kInt64 || input_dtype == torch::kLong)
    {
        launch_optimized_3d_rope_kernel_template<int64_t>(
            static_cast<const int64_t *>(input_ids_contiguous.data_ptr()),
            static_cast<const int64_t *>(attention_mask_ptr),
            static_cast<const int64_t *>(image_grid_thw_ptr),
            static_cast<const int64_t *>(video_grid_thw_ptr),
            second_per_grid_ts_ptr,
            static_cast<int64_t *>(position_ids.data_ptr()),
            static_cast<int64_t *>(mrope_deltas.data_ptr()),
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(seq_len),
            static_cast<int64_t>(spatial_merge_size),
            static_cast<int64_t>(image_token_id),
            static_cast<int64_t>(video_token_id),
            static_cast<int64_t>(vision_start_token_id),
            tokens_per_second);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported data type for input_ids. Supported types: int32, int64/long");
    }

    return std::make_tuple(position_ids, mrope_deltas);
}

template void launch_optimized_3d_rope_kernel_template<int>(
    const int *, const int *, const int *, const int *, const float *,
    int *, int *, int, int, int, int, int, int, float);

template void launch_optimized_3d_rope_kernel_template<int64_t>(
    const int64_t *, const int64_t *, const int64_t *, const int64_t *, const float *,
    int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float);
