#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cassert>
#include "../common/cuda_utils.h"

namespace wallx_cuda_rot_pos {

__global__ void fused_rot_pos_emb_kernel_int32(
    const float *__restrict__ inv_freq,    // [dim/2] - precomputed inverse frequencies
    const int32_t *__restrict__ grid_thw,      // [num_grids, 3] - (t, h, w) for each grid
    float *__restrict__ output,            // [total_tokens, dim] - output rotary embeddings
    const int32_t *__restrict__ cumsum_tokens, // [num_grids+1] - cumulative sum of tokens per grid
    const int dim_half,                    // dim/2 (size of inv_freq)
    const int spatial_merge_size,          // spatial merge size
    const int num_grids                    // number of grids
)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t total_tokens = cumsum_tokens[num_grids];

    extern __shared__ float freq_smem[];
    for (int i = threadIdx.x; i < dim_half; i += blockDim.x) {
        freq_smem[i] = inv_freq[i];
    }
    if (tid >= total_tokens * dim_half)
        return;

    const int32_t token_idx = tid / dim_half;
    const int freq_idx = tid % dim_half;

    // Find which grid this token belongs to
    int grid_idx = 0;
    int32_t local_token_idx = token_idx;
    for (int g = 0; g < num_grids; g++)
    {
        if (token_idx < cumsum_tokens[g + 1])
        {
            grid_idx = g;
            local_token_idx = token_idx - cumsum_tokens[g];
            break;
        }
    }

    // Get grid dimensions
    const int32_t h = grid_thw[grid_idx * 3 + 1];
    const int32_t w = grid_thw[grid_idx * 3 + 2];

    // Calculate spatial dimensions after merging
    const int32_t h_merged = h / spatial_merge_size;
    const int32_t w_merged = w / spatial_merge_size;
    const int32_t spatial_tokens = h_merged * w_merged * spatial_merge_size * spatial_merge_size;

    // Get spatial index
    const int32_t spatial_idx = local_token_idx % spatial_tokens;

    // Decompose spatial index to get merged block and position within block
    const int32_t tokens_per_block = spatial_merge_size * spatial_merge_size;
    const int32_t block_idx = spatial_idx / tokens_per_block;
    const int32_t within_block_idx = spatial_idx % tokens_per_block;

    // Get block coordinates in merged grid
    const int32_t block_h = block_idx / w_merged;
    const int32_t block_w = block_idx % w_merged;

    // Get position within block
    const int32_t within_h = within_block_idx / spatial_merge_size;
    const int32_t within_w = within_block_idx % spatial_merge_size;

    // Calculate actual h and w positions
    const int32_t h_pos = block_h * spatial_merge_size + within_h;
    const int32_t w_pos = block_w * spatial_merge_size + within_w;

    // Compute rotary embedding
    float freq_val = freq_smem[freq_idx];

    // Output has shape [total_tokens, dim] where dim = 2 * dim_half
    int32_t out_idx = token_idx * dim_half * 2 + freq_idx;
    output[out_idx] = h_pos * freq_val;            // h_pos frequencies
    output[out_idx + dim_half] = w_pos * freq_val; // w_pos frequencies
}

__global__ void fused_rot_pos_emb_kernel_int64(
    const float *__restrict__ inv_freq,    // [dim/2] - precomputed inverse frequencies
    const int64_t *__restrict__ grid_thw,      // [num_grids, 3] - (t, h, w) for each grid
    float *__restrict__ output,            // [total_tokens, dim] - output rotary embeddings
    const int64_t *__restrict__ cumsum_tokens, // [num_grids+1] - cumulative sum of tokens per grid
    const int dim_half,                    // dim/2 (size of inv_freq)
    const int spatial_merge_size,          // spatial merge size
    const int num_grids                    // number of grids
)
{
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_tokens = cumsum_tokens[num_grids];

    extern __shared__ float freq_smem[];
    for (int i = threadIdx.x; i < dim_half; i += blockDim.x) {
        freq_smem[i] = inv_freq[i];
    }

    if (tid >= total_tokens * dim_half)
        return;

    const int64_t token_idx = tid / dim_half;
    const int freq_idx = tid % dim_half;

    // Find which grid this token belongs to
    int grid_idx = 0;
    int64_t local_token_idx = token_idx;
    for (int g = 0; g < num_grids; g++)
    {
        if (token_idx < cumsum_tokens[g + 1])
        {
            grid_idx = g;
            local_token_idx = token_idx - cumsum_tokens[g];
            break;
        }
    }

    // Get grid dimensions
    const int64_t h = grid_thw[grid_idx * 3 + 1];
    const int64_t w = grid_thw[grid_idx * 3 + 2];

    // Calculate spatial dimensions after merging
    const int64_t h_merged = h / spatial_merge_size;
    const int64_t w_merged = w / spatial_merge_size;
    const int64_t spatial_tokens = h_merged * w_merged * spatial_merge_size * spatial_merge_size;

    // Get spatial index
    const int64_t spatial_idx = local_token_idx % spatial_tokens;

    // Decompose spatial index to get merged block and position within block
    const int64_t tokens_per_block = spatial_merge_size * spatial_merge_size;
    const int64_t block_idx = spatial_idx / tokens_per_block;
    const int64_t within_block_idx = spatial_idx % tokens_per_block;

    // Get block coordinates in merged grid
    const int64_t block_h = block_idx / w_merged;
    const int64_t block_w = block_idx % w_merged;

    // Get position within block
    const int64_t within_h = within_block_idx / spatial_merge_size;
    const int64_t within_w = within_block_idx % spatial_merge_size;

    // Calculate actual h and w positions
    const int64_t h_pos = block_h * spatial_merge_size + within_h;
    const int64_t w_pos = block_w * spatial_merge_size + within_w;

    // Compute rotary embedding
    float freq_val = freq_smem[freq_idx];

    // Output has shape [total_tokens, dim] where dim = 2 * dim_half
    int64_t out_idx = token_idx * dim_half * 2 + freq_idx;
    output[out_idx] = h_pos * freq_val;            // h_pos frequencies
    output[out_idx + dim_half] = w_pos * freq_val; // w_pos frequencies
}

__global__ void compute_token_counts_kernel_int32(
    const int32_t *__restrict__ grid_thw,
    int32_t *__restrict__ token_counts,
    const int spatial_merge_size,
    const int num_grids)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_grids)
        return;

    int32_t t = grid_thw[idx * 3 + 0];
    int32_t h = grid_thw[idx * 3 + 1];
    int32_t w = grid_thw[idx * 3 + 2];
    int32_t h_merged = h / spatial_merge_size;
    int32_t w_merged = w / spatial_merge_size;
    token_counts[idx] = t * h_merged * w_merged * spatial_merge_size * spatial_merge_size;
}

__global__ void compute_token_counts_kernel_int64(
    const int64_t *__restrict__ grid_thw,
    int64_t *__restrict__ token_counts,
    const int spatial_merge_size,
    const int num_grids)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_grids)
        return;

    int64_t t = grid_thw[idx * 3 + 0];
    int64_t h = grid_thw[idx * 3 + 1];
    int64_t w = grid_thw[idx * 3 + 2];
    int64_t h_merged = h / spatial_merge_size;
    int64_t w_merged = w / spatial_merge_size;
    token_counts[idx] = t * h_merged * w_merged * spatial_merge_size * spatial_merge_size;
}

void GetTokenCounts(
    const at::Tensor& grid_thw, // [num_grids, 3]
    const at::Tensor& token_counts, // [num_grids]
    int spatial_merge_size
) {
    ASSERT_CHECK(grid_thw.dim() == 2);
    ASSERT_CHECK(grid_thw.size(1) == 3);
    ASSERT_CHECK(grid_thw.size(0) == token_counts.size(0));

    ASSERT_CHECK(spatial_merge_size > 0);

    const int num_grids = grid_thw.size(0);

    const int threads = 256;
    const int blocks = (num_grids + threads - 1) / threads;

    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    if (grid_thw.scalar_type() == at::kInt) {
        compute_token_counts_kernel_int32<<<blocks, threads, 0, stream>>>(
            static_cast<int32_t*>(grid_thw.data_ptr()),
            static_cast<int32_t*>(token_counts.data_ptr()),
            spatial_merge_size,
            num_grids);
    } else if (grid_thw.scalar_type() == at::kLong) {
        compute_token_counts_kernel_int64<<<blocks, threads, 0, stream>>>(
            static_cast<int64_t*>(grid_thw.data_ptr()),
            static_cast<int64_t*>(token_counts.data_ptr()),
            spatial_merge_size,
            num_grids);
    } else {
        throw std::runtime_error("Unsupported data type for GetTokenCounts");
    }

    sync_check_cuda_error();
}

void RotPosEmb(
    const at::Tensor& inv_freq, // [dim/2]
    const at::Tensor& grid_thw, // [num_grids, 3]
    const at::Tensor& output,
    const at::Tensor& cumsum_tokens,
    int spatial_merge_size
) {
    ASSERT_CHECK(output.size(0) > 0);
    ASSERT_CHECK(inv_freq.dim() == 1);
    ASSERT_CHECK(inv_freq.scalar_type() == at::kFloat);

    const int dim_half = inv_freq.size(0);
    const int num_grids = grid_thw.size(0);
    const int total_tokens = output.size(0);

    const int64_t num_elements = total_tokens * dim_half;

    const int threads_per_block = 256;
    const int num_blocks = static_cast<int>((num_elements + threads_per_block - 1) / threads_per_block);
    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    if (grid_thw.scalar_type() == at::kInt) {
        fused_rot_pos_emb_kernel_int32<<<num_blocks, threads_per_block, dim_half * sizeof(float), stream>>>(
            static_cast<float*>(inv_freq.data_ptr()),
            static_cast<int32_t*>(grid_thw.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            static_cast<int32_t*>(cumsum_tokens.data_ptr()),
            dim_half,
            spatial_merge_size,
            num_grids);
    } else if (grid_thw.scalar_type() == at::kLong) {
        fused_rot_pos_emb_kernel_int64<<<num_blocks, threads_per_block, dim_half * sizeof(float), stream>>>(
            static_cast<float*>(inv_freq.data_ptr()),
            static_cast<int64_t*>(grid_thw.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            static_cast<int64_t*>(cumsum_tokens.data_ptr()),
            dim_half,
            spatial_merge_size,
            num_grids);
    } else {
        throw std::runtime_error("Unsupported data type for RotPosEmb");
    }

    sync_check_cuda_error();
}

// Removed TVM FFI export (see pybind11 registration)
// Removed TVM FFI export (see pybind11 registration)

} // wallx_cuda_rot_pos
