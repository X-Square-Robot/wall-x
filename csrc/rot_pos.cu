#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for fused rotary position embedding computation - int32 version
__global__ void fused_rot_pos_emb_kernel_int32(
    const float *__restrict__ inv_freq,        // [dim/2] - precomputed inverse frequencies
    const int32_t *__restrict__ grid_thw,      // [num_grids, 3] - (t, h, w) for each grid
    float *__restrict__ output,                // [total_tokens, dim] - output rotary embeddings
    const int32_t *__restrict__ cumsum_tokens, // [num_grids+1] - cumulative sum of tokens per grid
    const int dim_half,                        // dim/2 (size of inv_freq)
    const int spatial_merge_size,              // spatial merge size
    const int num_grids                        // number of grids
)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t total_tokens = cumsum_tokens[num_grids];

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
    float freq_val = inv_freq[freq_idx];

    // Output has shape [total_tokens, dim] where dim = 2 * dim_half
    int32_t out_idx = token_idx * dim_half * 2 + freq_idx;
    output[out_idx] = h_pos * freq_val;            // h_pos frequencies
    output[out_idx + dim_half] = w_pos * freq_val; // w_pos frequencies
}

// CUDA kernel for fused rotary position embedding computation - int64 version
__global__ void fused_rot_pos_emb_kernel_int64(
    const float *__restrict__ inv_freq,        // [dim/2] - precomputed inverse frequencies
    const int64_t *__restrict__ grid_thw,      // [num_grids, 3] - (t, h, w) for each grid
    float *__restrict__ output,                // [total_tokens, dim] - output rotary embeddings
    const int64_t *__restrict__ cumsum_tokens, // [num_grids+1] - cumulative sum of tokens per grid
    const int dim_half,                        // dim/2 (size of inv_freq)
    const int spatial_merge_size,              // spatial merge size
    const int num_grids                        // number of grids
)
{
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_tokens = cumsum_tokens[num_grids];

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
    float freq_val = inv_freq[freq_idx];

    // Output has shape [total_tokens, dim] where dim = 2 * dim_half
    int64_t out_idx = token_idx * dim_half * 2 + freq_idx;
    output[out_idx] = h_pos * freq_val;            // h_pos frequencies
    output[out_idx + dim_half] = w_pos * freq_val; // w_pos frequencies
}

// Parallel computation of token counts per grid - int32 version
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

// Parallel computation of token counts per grid - int64 version
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

// Implementation for int32
torch::Tensor fused_rot_pos_emb_cuda_int32(
    torch::Tensor inv_freq, // [dim/2]
    torch::Tensor grid_thw, // [num_grids, 3]
    int spatial_merge_size)
{
    TORCH_CHECK(inv_freq.dim() == 1, "inv_freq must be 1-dimensional");
    TORCH_CHECK(inv_freq.is_cuda(), "inv_freq must be a CUDA tensor");
    TORCH_CHECK(inv_freq.scalar_type() == torch::kFloat32, "inv_freq must be float32");

    TORCH_CHECK(grid_thw.dim() == 2, "grid_thw must be 2-dimensional");
    TORCH_CHECK(grid_thw.size(1) == 3, "grid_thw must have shape [num_grids, 3]");
    TORCH_CHECK(grid_thw.is_cuda(), "grid_thw must be a CUDA tensor");
    TORCH_CHECK(grid_thw.scalar_type() == torch::kInt32, "grid_thw must be int32");

    TORCH_CHECK(spatial_merge_size > 0, "spatial_merge_size must be positive");

    const int dim_half = inv_freq.size(0);
    const int num_grids = grid_thw.size(0);

    auto token_counts = torch::zeros({num_grids}, torch::TensorOptions().dtype(torch::kInt32).device(grid_thw.device()));
    const int threads = 256;
    const int blocks = (num_grids + threads - 1) / threads;

    compute_token_counts_kernel_int32<<<blocks, threads>>>(
        grid_thw.data_ptr<int32_t>(),
        token_counts.data_ptr<int32_t>(),
        spatial_merge_size,
        num_grids);

    auto cumsum_tokens = torch::cat({torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(grid_thw.device())),
                                     token_counts.cumsum(0).to(torch::kInt32)},
                                    0);

    cudaDeviceSynchronize();

    int64_t total_tokens = cumsum_tokens[-1].item<int64_t>();
    TORCH_CHECK(total_tokens > 0, "total_tokens must be positive");

    auto output = torch::zeros({total_tokens, dim_half * 2},
                               torch::TensorOptions().dtype(torch::kFloat32).device(inv_freq.device()));

    const int threads_per_block = 256;
    const int64_t num_elements = total_tokens * dim_half;
    const int num_blocks = static_cast<int>((num_elements + threads_per_block - 1) / threads_per_block);

    fused_rot_pos_emb_kernel_int32<<<num_blocks, threads_per_block>>>(
        inv_freq.data_ptr<float>(),
        grid_thw.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        cumsum_tokens.data_ptr<int32_t>(),
        dim_half,
        spatial_merge_size,
        num_grids);

    cudaDeviceSynchronize();

    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "Output must be float32");
    TORCH_CHECK(output.size(0) == total_tokens, "Output token count mismatch");
    TORCH_CHECK(output.size(1) == dim_half * 2, "Output dimension mismatch");

    return output;
}

// Implementation for int64
torch::Tensor fused_rot_pos_emb_cuda_int64(
    torch::Tensor inv_freq, // [dim/2]
    torch::Tensor grid_thw, // [num_grids, 3]
    int spatial_merge_size)
{
    TORCH_CHECK(inv_freq.dim() == 1, "inv_freq must be 1-dimensional");
    TORCH_CHECK(inv_freq.is_cuda(), "inv_freq must be a CUDA tensor");
    TORCH_CHECK(inv_freq.scalar_type() == torch::kFloat32, "inv_freq must be float32");

    TORCH_CHECK(grid_thw.dim() == 2, "grid_thw must be 2-dimensional");
    TORCH_CHECK(grid_thw.size(1) == 3, "grid_thw must have shape [num_grids, 3]");
    TORCH_CHECK(grid_thw.is_cuda(), "grid_thw must be a CUDA tensor");
    TORCH_CHECK(grid_thw.scalar_type() == torch::kInt64, "grid_thw must be int64");

    TORCH_CHECK(spatial_merge_size > 0, "spatial_merge_size must be positive");

    const int dim_half = inv_freq.size(0);
    const int num_grids = grid_thw.size(0);

    auto token_counts = torch::zeros({num_grids}, torch::TensorOptions().dtype(torch::kInt64).device(grid_thw.device()));
    const int threads = 256;
    const int blocks = (num_grids + threads - 1) / threads;

    compute_token_counts_kernel_int64<<<blocks, threads>>>(
        grid_thw.data_ptr<int64_t>(),
        token_counts.data_ptr<int64_t>(),
        spatial_merge_size,
        num_grids);

    auto cumsum_tokens = torch::cat({torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64).device(grid_thw.device())),
                                     token_counts.cumsum(0).to(torch::kInt64)},
                                    0);

    cudaDeviceSynchronize();

    int64_t total_tokens = cumsum_tokens[-1].item<int64_t>();
    TORCH_CHECK(total_tokens > 0, "total_tokens must be positive");

    auto output = torch::zeros({total_tokens, dim_half * 2},
                               torch::TensorOptions().dtype(torch::kFloat32).device(inv_freq.device()));

    const int threads_per_block = 256;
    const int64_t num_elements = total_tokens * dim_half;
    const int num_blocks = static_cast<int>((num_elements + threads_per_block - 1) / threads_per_block);

    fused_rot_pos_emb_kernel_int64<<<num_blocks, threads_per_block>>>(
        inv_freq.data_ptr<float>(),
        grid_thw.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        cumsum_tokens.data_ptr<int64_t>(),
        dim_half,
        spatial_merge_size,
        num_grids);

    cudaDeviceSynchronize();

    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "Output must be float32");
    TORCH_CHECK(output.size(0) == total_tokens, "Output token count mismatch");
    TORCH_CHECK(output.size(1) == dim_half * 2, "Output dimension mismatch");

    return output;
}

// Main function that dispatches based on grid_thw scalar type
torch::Tensor fused_rot_pos_emb_cuda(
    torch::Tensor inv_freq,
    torch::Tensor grid_thw,
    int spatial_merge_size)
{
    if (grid_thw.scalar_type() == torch::kInt32)
    {
        return fused_rot_pos_emb_cuda_int32(inv_freq, grid_thw, spatial_merge_size);
    }
    else if (grid_thw.scalar_type() == torch::kInt64)
    {
        return fused_rot_pos_emb_cuda_int64(inv_freq, grid_thw, spatial_merge_size);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported grid_thw scalar type: ", grid_thw.scalar_type());
    }
}
