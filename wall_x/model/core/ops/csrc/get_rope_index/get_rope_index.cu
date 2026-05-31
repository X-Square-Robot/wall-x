#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__

#include <vector>
#include <tuple>
#include <cstdint>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/cub.cuh>

#include "../common/cuda_utils.h"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace wallx_cuda_get_rope_index {

// Constants and macros
#define MAX_SEQ_LEN 8192
#define MAX_VISION_TOKENS 64
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

// Vision content descriptor
struct VisionDescriptor
{
    int64_t start_pos;              // Starting position in the sequence
    int64_t token_pos;              // Position of the vision token
    int64_t patch_count;            // Number of patches
    int64_t grid_t, grid_h, grid_w; // Grid dimensions (T, H, W)
    float time_interval;            // Time interval (for video)
    int64_t is_video;               // 0 = image, 1 = video
    int64_t position_offset;        // Positional encoding offset
};

// Device function: fast integer division using reciprocal multiplication
__device__ __forceinline__ int64_t fast_div(int64_t a, int64_t b)
{
    return __float2ll_rd(__ll2float_rn(a) * __frcp_rn(__ll2float_rn(b)));
}

// Device function: map 3D patch index to (t, h, w) coordinates
__device__ __forceinline__ void get_3d_coords(int64_t patch_idx, int64_t H, int64_t W,
                                              int64_t &t, int64_t &h, int64_t &w)
{
    int64_t hw = H * W;
    t = fast_div(patch_idx, hw);
    int64_t remaining = patch_idx - t * hw;
    h = fast_div(remaining, W);
    w = remaining - h * W;
}

// Stage 1: Kernel to count image and video tokens per batch
__global__ void compute_vision_counts(
    const int64_t *input_ids,              // (batch_size, seq_len)
    const int64_t *attention_mask,         // (batch_size, seq_len)
    int64_t *image_counts,                 // (batch_size,) - output: number of images per batch
    int64_t *video_counts,                 // (batch_size,) - output: number of videos per batch
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

    // Initialize per-thread counters
    int64_t thread_image_count = 0;
    int64_t thread_video_count = 0;

    // Parallel scan over sequence tokens (each thread processes multiple tokens)
    for (int64_t i = thread_idx; i < seq_len - 1; i += blockDim.x)
    {
        // Skip if masked out
        if ((attention_mask != nullptr) && attention_mask[batch_idx * seq_len + i] == 0)
            continue;

        int64_t token_id = input_ids[batch_idx * seq_len + i];

        // Check if current token is a vision start token followed by image/video token
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

    // Store per-thread counts in shared memory
    shared_image_counts[thread_idx] = thread_image_count;
    shared_video_counts[thread_idx] = thread_video_count;
    __syncthreads();

    // Parallel reduction to sum counts across threads
    for (int64_t stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (thread_idx < stride)
        {
            shared_image_counts[thread_idx] += shared_image_counts[thread_idx + stride];
            shared_video_counts[thread_idx] += shared_video_counts[thread_idx + stride];
        }
        __syncthreads();
    }

    // Write final result for this batch to global memory
    if (thread_idx == 0)
    {
        image_counts[batch_idx] = shared_image_counts[0];
        video_counts[batch_idx] = shared_video_counts[0];
    }
}


// Stage 2: Preprocessing kernel – parse and analyze all vision content
__global__ void preprocess_vision_tokens(
    const int64_t *input_ids,            // (batch_size, seq_len)
    const int64_t *attention_mask,       // (batch_size, seq_len)
    const int64_t *image_grid_thw,       // (max_images, 3)
    const int64_t *video_grid_thw,       // (max_videos, 3)
    const float *second_per_grid_ts,     // (max_videos,)
    const int64_t *image_counts,         // (batch_size,)
    const int64_t *video_counts,         // (batch_size,)
    VisionDescriptor *vision_desc,       // (batch_size, MAX_VISION_TOKENS)
    int64_t *vision_counts,              // (batch_size,) - total vision tokens per batch
    int64_t *text_lengths,               // (batch_size, MAX_VISION_TOKENS+1) - lengths of text segments
    int64_t *position_offsets,           // (batch_size, MAX_VISION_TOKENS+1) - cumulative position offsets
    const int64_t batch_size,
    const int64_t seq_len,
    const int64_t spatial_merge_size,
    const int64_t image_token_id,
    const int64_t video_token_id,
    const int64_t vision_start_token_id,
    const float tokens_per_second)
{
    // Each thread processes one batch sequence
    int64_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size)
        return;

    // Compute cumulative image/video indices before this batch
    int64_t image_idx = 0, video_idx = 0;
    for (int64_t i = 0; i < batch_idx; i++)
    {
        image_idx += image_counts[i];
        video_idx += video_counts[i];
    }

    // Local state
    int64_t vision_count = 0;
    int64_t current_pos = 0;
    int64_t position_offset = 0;

    // Sequential scan over the sequence (single-threaded per batch)
    for (int64_t i = 0; i < seq_len - 1; i++)
    {
        // Skip masked tokens
        if ((attention_mask != nullptr) && attention_mask[batch_idx * seq_len + i] == 0)
            continue;

        int64_t token_id = input_ids[batch_idx * seq_len + i];

        // Check for vision start token
        if (token_id == vision_start_token_id)
        {
            int64_t next_token = input_ids[batch_idx * seq_len + i + 1];

            if (next_token == image_token_id || next_token == video_token_id)
            {
                // Record length of preceding text segment
                int64_t vision_pos = i + 1;
                text_lengths[batch_idx * (MAX_VISION_TOKENS + 1) + vision_count] = vision_pos - current_pos;
                position_offsets[batch_idx * (MAX_VISION_TOKENS + 1) + vision_count] = position_offset;
                position_offset += (vision_pos - current_pos);

                // Fetch grid dimensions
                int64_t T, H, W;
                float time_interval = 0.0f;
                int64_t is_video = (next_token == video_token_id) ? 1 : 0;

                if (is_video == 0)
                {
                    // Image case
                    T = image_grid_thw[image_idx * 3 + 0];
                    H = image_grid_thw[image_idx * 3 + 1];
                    W = image_grid_thw[image_idx * 3 + 2];
                    image_idx++;
                }
                else
                {
                    // Video case
                    T = video_grid_thw[video_idx * 3 + 0];
                    H = video_grid_thw[video_idx * 3 + 1];
                    W = video_grid_thw[video_idx * 3 + 2];
                    time_interval = second_per_grid_ts[video_idx];
                    video_idx++;
                }

                // Compute number of patches after spatial merging
                int64_t H_merged = H / spatial_merge_size;
                int64_t W_merged = W / spatial_merge_size;
                int64_t patch_count = T * H_merged * W_merged;

                // Populate vision descriptor
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

                    // Update position offset based on content type
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

                // Skip over the vision patch tokens
                i = vision_pos + patch_count - 1;
            }
        }
    }

    // Store total vision token count for this batch
    vision_counts[batch_idx] = vision_count;

    // Handle final text segment after last vision token
    int64_t effective_len = seq_len;
    if (attention_mask != nullptr)
    {
        // Find last valid (unmasked) token
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


// Stage 3: Main kernel – compute 3D position IDs for all tokens in parallel
__global__ void compute_3d_positions(
    const int64_t *input_ids,                // (batch_size, seq_len)
    const int64_t *attention_mask,           // (batch_size, seq_len)
    const VisionDescriptor *vision_desc,     // (batch_size, MAX_VISION_TOKENS)
    const int64_t *vision_counts,            // (batch_size,)
    const int64_t *text_lengths,             // (batch_size, MAX_VISION_TOKENS+1)
    const int64_t *position_offsets,         // (batch_size, MAX_VISION_TOKENS+1)
    int64_t *position_ids,                   // (3, batch_size, seq_len) — output 3D position IDs
    int64_t *mrope_deltas,                   // (batch_size,) — RoPE length adjustment
    const int64_t batch_size,
    const int64_t seq_len,
    const float tokens_per_second)
{
    // Grid: (batch_size), Block: (threads_per_block)
    int64_t batch_idx = blockIdx.x;
    int64_t thread_idx = threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    __shared__ VisionDescriptor shared_visions[MAX_VISION_TOKENS];
    __shared__ int64_t shared_position_offsets[MAX_VISION_TOKENS + 1];
    __shared__ int64_t shared_max_positions[MAX_THREADS_PER_BLOCK]; // For block-wide reduction

    int64_t shared_vision_count = vision_counts[batch_idx];

    // Cooperative loading of vision descriptors into shared memory
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

    // Initialize per-thread max position
    int64_t thread_max_position = -1;

    __syncthreads();

    // Parallel processing: each thread handles multiple tokens
    for (int64_t token_idx = thread_idx; token_idx < seq_len; token_idx += blockDim.x)
    {
        // Check validity via attention mask
        int mask_offset = 0;
        bool is_valid_token = true;
        if (attention_mask != nullptr)
        {
            is_valid_token = attention_mask[batch_idx * seq_len + token_idx] != 0;
            for (int i = 0; i < token_idx; i++) {
                if (attention_mask[batch_idx * seq_len + i] == 0) {
                    mask_offset += 1;
                }
            }
        }

        if (!is_valid_token)
        {
            // Set masked tokens to default position (1)
            position_ids[0 * batch_size * seq_len + batch_idx * seq_len + token_idx] = 1;
            position_ids[1 * batch_size * seq_len + batch_idx * seq_len + token_idx] = 1;
            position_ids[2 * batch_size * seq_len + batch_idx * seq_len + token_idx] = 1;
            continue;
        }

        // Determine which segment this token belongs to
        int64_t segment_idx = -9999999;
        int64_t local_pos = token_idx;

        // Linear search (vision count is small, so this is efficient)
        for (int64_t v = 0; v < shared_vision_count; v++)
        {
            if (token_idx < shared_visions[v].token_pos)
            {
                // Token belongs to text segment before vision v
                segment_idx = v;
                local_pos = token_idx - (v > 0 ? shared_visions[v - 1].token_pos + shared_visions[v - 1].patch_count : 0);
                break;
            }
            else if (token_idx < shared_visions[v].token_pos + shared_visions[v].patch_count)
            {
                // Token belongs to vision patch v
                segment_idx = -(v + 1); // Negative index indicates vision
                local_pos = token_idx - shared_visions[v].token_pos;
                break;
            }
        }

        if (segment_idx == -9999999)
        {
            // Token belongs to final text segment
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
            // Text token: use 1D position encoding
            int64_t offset = shared_position_offsets[segment_idx];
            pos_t = pos_h = pos_w = offset + local_pos;
        }
        else
        {
            // Vision patch: compute 3D position encoding
            int64_t vision_idx = -(segment_idx + 1);
            const VisionDescriptor &desc = shared_visions[vision_idx];

            // Map linear patch index to (t, h, w)
            int64_t t, h, w;
            get_3d_coords(local_pos, desc.grid_h, desc.grid_w, t, h, w);

            // Compute 3D positions with temporal scaling for video
            pos_t = static_cast<int64_t>(t * desc.time_interval * tokens_per_second) + desc.position_offset;
            pos_h = h + desc.position_offset;
            pos_w = w + desc.position_offset;
        }

        // Write 3D position IDs
        position_ids[0 * batch_size * seq_len + batch_idx * seq_len + token_idx] = pos_t - mask_offset;
        position_ids[1 * batch_size * seq_len + batch_idx * seq_len + token_idx] = pos_h - mask_offset;
        position_ids[2 * batch_size * seq_len + batch_idx * seq_len + token_idx] = pos_w - mask_offset;

        // Track per-thread maximum position
        int64_t max_pos = max(pos_t - mask_offset, max(pos_h - mask_offset, pos_w - mask_offset));
        thread_max_position = max(thread_max_position, max_pos);
    }

    // Store per-thread max into shared memory
    shared_max_positions[thread_idx] = thread_max_position;
    __syncthreads();

    // Block-wide reduction to find global max position
    for (int64_t stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (thread_idx < stride)
        {
            shared_max_positions[thread_idx] = max(shared_max_positions[thread_idx],
                                                 shared_max_positions[thread_idx + stride]);
        }
        __syncthreads();
    }

    // Compute mRoPE delta: (max_pos + 1) - seq_len
    if (thread_idx == 0)
    {
        int64_t global_max_position = shared_max_positions[0];
        mrope_deltas[batch_idx] = global_max_position + 1 - seq_len;
    }
}

// Fallback kernel when no vision tokens exist: assign sequential positions based on attention mask
__global__ void compute_3d_positions_mask_text(
    const int64_t* __restrict__ attention_mask,   // (batch_size, seq_len)
    int64_t* __restrict__ position_ids,           // (3, batch_size, seq_len) — flattened row-major
    int64_t* __restrict__ mrope_deltas,           // (batch_size,)
    const int64_t batch_size,
    const int64_t seq_len
) {
    // One block per batch
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    extern __shared__ int64_t shared_mask[];
    int64_t* mask = shared_mask;

    // Load attention mask for current batch into shared memory
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        mask[i] = attention_mask[batch_idx * seq_len + i];
    }
    __syncthreads();

    int tid = threadIdx.x;
    int64_t base_offset = batch_idx * seq_len;
    int64_t total_valid = 0;

    // Compute prefix sum (cumulative count of valid tokens)
    // Simple O(seq_len^2) per-thread loop — acceptable for moderate seq_len
    for (int idx = tid; idx < seq_len; idx += blockDim.x) {
        int64_t sum = 0;
        for (int i = 0; i <= idx; ++i) {
            sum += mask[i];
        }
        int64_t pos_val = (mask[idx] ? (sum - 1) : 1); // Use 0-based if valid, else 1

        // Write same position to all three dimensions
        int64_t flat_idx0 = 0 * batch_size * seq_len + base_offset + idx;
        int64_t flat_idx1 = 1 * batch_size * seq_len + base_offset + idx;
        int64_t flat_idx2 = 2 * batch_size * seq_len + base_offset + idx;

        position_ids[flat_idx0] = pos_val;
        position_ids[flat_idx1] = pos_val;
        position_ids[flat_idx2] = pos_val;

        // Only thread 0 tracks total valid tokens
        if (tid == 0) {
            total_valid = sum; // After last idx, sum = total number of 1s
        }
    }

    // Compute mrope_deltas: (total_valid) - seq_len
    if (tid == 0) {
        // Re-compute total_valid robustly
        total_valid = 0;
        for (int i = 0; i < seq_len; ++i) {
            total_valid += mask[i];
        }
        mrope_deltas[batch_idx] = total_valid - seq_len;
    }
}

// Kernel to fill position_ids with arange(seq_len) in 3D layout when no vision or mask
__global__ void arange_3d(int64_t seq_len,
                          int64_t* out) {
    int i = blockIdx.x;       // dimension index: 0, 1, 2
    int j = blockIdx.y;       // batch index
    int64_t k = blockIdx.z * blockDim.x + threadIdx.x;  // token index

    if (k >= seq_len) {
        return;
    }

    // Flatten index: (dim, batch, token) → linear
    int64_t idx = i * gridDim.y * seq_len + j * seq_len + k;
    out[idx] = k;
}

// Kernel to zero-initialize a 1D tensor
__global__ void fill_zeros_1d(int64_t numel, int64_t* data) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] = 0;
    }
}

// Compute required workspace size (in bytes)
int64_t GetRopeIndexGetWorkSpace(const c10::optional<at::Tensor>& input_ids,
    const c10::optional<at::Tensor>& image_grid_thw,
    const c10::optional<at::Tensor>& video_grid_thw) {

    if (!image_grid_thw.has_value() && !video_grid_thw.has_value()) {
        return 1; // minimal workspace
    } else {
        const at::Tensor& input_ids_tensor = input_ids.value();
        const int batch_size = input_ids_tensor.size(0);
        return (batch_size * MAX_VISION_TOKENS * static_cast<int64_t>(sizeof(VisionDescriptor)) +
            3 * batch_size * sizeof(int64_t) +
            2 * batch_size * (MAX_VISION_TOKENS + 1) * sizeof(int64_t));
    }
}

// Main entry function: compute RoPE position IDs and mRoPE deltas
void GetRopeIndex(const c10::optional<at::Tensor>& input_ids,
    const c10::optional<at::Tensor>& image_grid_thw,
    const c10::optional<at::Tensor>& video_grid_thw,
    const c10::optional<at::Tensor>& second_per_grid_ts,
    const c10::optional<at::Tensor>& attention_mask,
    const at::Tensor& position_ids,
    const at::Tensor& mrope_deltas,
    const at::Tensor& workspace,
    int spatial_merge_size,
    int image_token_id,
    int video_token_id,
    int vision_start_token_id,
    float tokens_per_second) {

    ASSERT_CHECK(input_ids.has_value());

    const at::Tensor& input_ids_tensor = input_ids.value();
    ASSERT_CHECK(input_ids_tensor.dim() == 2);
    ASSERT_CHECK(input_ids_tensor.is_contiguous());

    const auto batch_size = input_ids_tensor.size(0);
    const auto seq_len = input_ids_tensor.size(1);
    const auto device = input_ids_tensor.device();

    const int64_t *input_ids_ptr = static_cast<const int64_t*>(input_ids_tensor.data_ptr());

    const int64_t *attention_mask_ptr = nullptr;

    if (attention_mask.has_value()) {
        const at::Tensor& attention_mask_tensor = attention_mask.value();
        ASSERT_CHECK(attention_mask_tensor.is_contiguous());
        attention_mask_ptr = static_cast<const int64_t*>(attention_mask_tensor.data_ptr());
    }

    const int64_t *image_grid_thw_ptr = nullptr;

    if (image_grid_thw.has_value()) {
        const at::Tensor& image_grid_thw_tensor = image_grid_thw.value();
        ASSERT_CHECK(image_grid_thw_tensor.dim() == 2 && image_grid_thw_tensor.size(1) == 3);
        ASSERT_CHECK(image_grid_thw_tensor.is_contiguous());
        image_grid_thw_ptr = static_cast<const int64_t*>(image_grid_thw_tensor.data_ptr());
    }

    const int64_t *video_grid_thw_ptr = nullptr;

    if (video_grid_thw.has_value()) {
        const at::Tensor& video_grid_thw_tensor = video_grid_thw.value();
        ASSERT_CHECK(video_grid_thw_tensor.is_contiguous());
        ASSERT_CHECK(video_grid_thw_tensor.dim() == 2 && video_grid_thw_tensor.size(1) == 3);
        video_grid_thw_ptr = static_cast<const int64_t*>(video_grid_thw_tensor.data_ptr());
    }

    const float *second_per_grid_ts_ptr = nullptr;

    if (second_per_grid_ts.has_value()) {
        const at::Tensor& second_per_grid_ts_tensor = second_per_grid_ts.value();
        ASSERT_CHECK(second_per_grid_ts_tensor.is_contiguous());
        ASSERT_CHECK(second_per_grid_ts_tensor.dim() == 1);
        second_per_grid_ts_ptr = static_cast<const float*>(second_per_grid_ts_tensor.data_ptr());
    }

    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    // Case: no vision tokens → use simple positional encoding
    if (!image_grid_thw.has_value() && !video_grid_thw.has_value()) {
        if (attention_mask.has_value()) {
            // Use attention mask to compute valid positions
            int block_size = min(1024, (int)seq_len);
            dim3 grid(batch_size);
            dim3 block(block_size);
            size_t shared_mem_size = seq_len * sizeof(int64_t);

            compute_3d_positions_mask_text<<<grid, block, shared_mem_size>>>(
                attention_mask_ptr, static_cast<int64_t*>(position_ids.data_ptr()), static_cast<int64_t*>(mrope_deltas.data_ptr()), batch_size, seq_len
            );
            return;
        } else {
            // No mask, no vision → use simple arange
            dim3 thread_num(256);
            dim3 grid(3, batch_size, (seq_len + 255) / 256);
            arange_3d<<<grid, thread_num, 0, stream>>>(seq_len, static_cast<int64_t*>(position_ids.data_ptr()));

            int block_size = 256;
            int grid_size = (batch_size + block_size - 1) / block_size;
            fill_zeros_1d<<<grid_size, block_size, 0, stream>>>(batch_size, static_cast<int64_t*>(mrope_deltas.data_ptr()));
            return;
        }
    }

    // Allocate workspace buffers
    void* work_ptr = workspace.data_ptr();
    VisionDescriptor *d_vision_desc = reinterpret_cast<VisionDescriptor*>(work_ptr);

    int64_t *d_vision_counts = reinterpret_cast<int64_t*>(reinterpret_cast<VisionDescriptor*>(work_ptr) + batch_size * MAX_VISION_TOKENS);
    int64_t *d_text_lengths = d_vision_counts + batch_size;
    int64_t *d_position_offsets = d_text_lengths + batch_size * (MAX_VISION_TOKENS + 1);
    int64_t *d_image_counts = d_position_offsets + batch_size * (MAX_VISION_TOKENS + 1);
    int64_t *d_video_counts = d_image_counts + batch_size;

    // Stage 1: count vision tokens
    dim3 index_grid(static_cast<unsigned int>(batch_size));
    dim3 index_block(256);

    compute_vision_counts<<<index_grid, index_block, 0, stream>>>(
        input_ids_ptr, attention_mask_ptr,
        d_image_counts, d_video_counts,
        batch_size, seq_len, image_token_id, video_token_id, vision_start_token_id);

    // Stage 2: preprocess vision tokens
    int64_t threads_per_block = std::min(static_cast<int64_t>(batch_size), static_cast<int64_t>(256));
    int64_t num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    dim3 preprocess_grid(static_cast<unsigned int>(num_blocks));
    dim3 preprocess_block(static_cast<unsigned int>(threads_per_block));

    preprocess_vision_tokens<<<preprocess_grid, preprocess_block, 0, stream>>>(
        input_ids_ptr, attention_mask_ptr, image_grid_thw_ptr, video_grid_thw_ptr,
        second_per_grid_ts_ptr, d_image_counts, d_video_counts,
        d_vision_desc, d_vision_counts, d_text_lengths, d_position_offsets,
        batch_size, seq_len, spatial_merge_size,
        image_token_id, video_token_id, vision_start_token_id, tokens_per_second);

    // Stage 3: compute 3D positions
    threads_per_block = std::min(static_cast<int64_t>(seq_len), static_cast<int64_t>(MAX_THREADS_PER_BLOCK));
    if (threads_per_block < 32) threads_per_block = 32;

    int64_t power_of_2 = 1;
    while (power_of_2 < threads_per_block) power_of_2 *= 2;
    if (power_of_2 > MAX_THREADS_PER_BLOCK) power_of_2 = MAX_THREADS_PER_BLOCK;
    threads_per_block = power_of_2;

    dim3 compute_grid(static_cast<unsigned int>(batch_size));
    dim3 compute_block(static_cast<unsigned int>(threads_per_block));

    compute_3d_positions<<<compute_grid, compute_block, 0, stream>>>(
        input_ids_ptr, attention_mask_ptr, d_vision_desc, d_vision_counts,
        d_text_lengths, d_position_offsets,
        static_cast<int64_t*>(position_ids.data_ptr()), static_cast<int64_t*>(mrope_deltas.data_ptr()),
        batch_size, seq_len, tokens_per_second);

    sync_check_cuda_error();
}

} // wallx_cuda_get_rope_index

// TVM FFI exports
// Removed TVM FFI export (see pybind11 registration)
// Removed TVM FFI export (see pybind11 registration)
