#include "tensor_view.h"
/**
 * Fused position embedding interpolation kernel for Qwen3.5 ViT.
 *
 * Fuses bilinear interpolation + spatial merge permutation into a single
 * CUDA kernel. Each thread computes one (output_token, hidden_dim_chunk)
 * independently:
 *   1. Find which image this output token belongs to (binary search on cumsum)
 *   2. Reverse spatial merge permutation to get source (row, col) in image grid
 *   3. Bilinear interpolation: map (row, col) to continuous coord in pos_embed table
 *   4. Read 4 corners from pos_embed_weight, weighted sum → write output
 *
 * This replaces ~20 PyTorch kernel launches with 1 fused kernel.
 */


#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cassert>
#include "../common/cuda_utils.h"

namespace wallx_cuda_pos_embed_interp {

template <typename T>
__device__ __forceinline__ float to_float(T x) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half x) {
    return __half2float(x);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x) {
    return static_cast<T>(x);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half_rn(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16_rn(x);
}

// Bit-exact replica of torch.linspace(0, N-1, steps) in fp32 (ATen fill):
// the step is rounded to fp32 first, values fill from both ends toward the
// middle, and each value is one rounding of an exact double product /
// difference. Verified bitwise against CPU torch.linspace for the grid sizes
// used by the vision tower (see tests/test_vision_ops_bitexact.py).
//
// steps == 1 returns 0.0f, matching torch.linspace(0, N-1, 1) == [0.].
// steps == 0 (torch would return an empty tensor) is unreachable here: callers
// derive steps from the interpolation grid h/w, and the kernel launch grid is
// sized by h*w, so no thread ever executes with steps == 0.
__device__ __forceinline__ float linspace_coord(
    int32_t i, int32_t steps, int32_t N
) {
    if (steps <= 1) return 0.0f;
    const float step = (float)(N - 1) / (float)(steps - 1);
    if (i < steps / 2) {
        return (float)((double)i * (double)step);
    }
    return (float)((double)(N - 1) - (double)(steps - 1 - i) * (double)step);
}

// Helper: binary search to find which image a token belongs to
__device__ __forceinline__ int find_image_idx(
    const int32_t* cumsum, int num_images, int32_t token_idx
) {
    int lo = 0, hi = num_images;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cumsum[mid + 1] <= token_idx) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

/**
 * Each thread handles one element of the output: (token_idx, dim_idx).
 * We process VEC_SIZE elements per thread along the hidden dim for coalescing.
 */
template <typename T, int VEC_SIZE>
__global__ void fused_pos_embed_interp_kernel(
    const T* __restrict__ pos_embed_weight,      // [N*N, hidden_dim]
    const int32_t* __restrict__ grid_thw,    // [num_images, 3]
    const int32_t* __restrict__ out_cumsum,  // [num_images+1] cumsum of t*h*w
    const int32_t* __restrict__ hw_cumsum,   // [num_images+1] cumsum of h*w (pre-temporal-repeat)
    T* __restrict__ output,                  // [total_output, hidden_dim]
    const int num_grid_per_side,             // N
    const int spatial_merge_size,            // m
    const int hidden_dim,                    // D
    const int num_images,
    const int total_output
) {
    // Each thread processes VEC_SIZE hidden dims for one token
    const int token_idx = blockIdx.x;
    const int dim_start = (blockIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;

    if (token_idx >= total_output || dim_start >= hidden_dim) return;

    const int N = num_grid_per_side;
    const int m = spatial_merge_size;
    const int m2 = m * m;

    // 1. Find which image this output token belongs to
    int img = find_image_idx(out_cumsum, num_images, token_idx);

    int32_t h = grid_thw[img * 3 + 1];
    int32_t w = grid_thw[img * 3 + 2];
    int32_t hw = h * w;

    // Local index within this image's output
    int32_t local_idx = token_idx - out_cumsum[img];

    // Frame index (temporal) and position within frame
    int32_t pos_in_frame = local_idx % hw;

    // 2. Reverse spatial merge permutation
    // Output layout: (h/m, w/m, m, m) flattened
    // Source layout: (h, w) row-major = (h/m, m, w/m, m) flattened
    int32_t w_div_m = w / m;
    int32_t block_idx = pos_in_frame / m2;
    int32_t inner_idx = pos_in_frame % m2;
    int32_t block_row = block_idx / w_div_m;
    int32_t block_col = block_idx % w_div_m;
    int32_t inner_row = inner_idx / m;
    int32_t inner_col = inner_idx % m;

    int32_t src_row = block_row * m + inner_row;
    int32_t src_col = block_col * m + inner_col;

    // 3. Bilinear interpolation coordinates — bit-exact torch.linspace replica
    const float h_coord = linspace_coord(src_row, h, N);
    const float w_coord = linspace_coord(src_col, w, N);

    int h_floor = (int)h_coord;
    int w_floor = (int)w_coord;
    int h_ceil = min(h_floor + 1, N - 1);
    int w_ceil = min(w_floor + 1, N - 1);

    float dh = h_coord - (float)h_floor;
    float dw = w_coord - (float)w_floor;

    const float w_ff = (1.0f - dh) * (1.0f - dw);
    const float w_fc = (1.0f - dh) * dw;
    const float w_cf = dh * (1.0f - dw);
    const float w_cc = dh * dw;

    // 4-corner flat indices into pos_embed_weight [N*N, D]
    int idx_ff = h_floor * N + w_floor;
    int idx_fc = h_floor * N + w_ceil;
    int idx_cf = h_ceil * N + w_floor;
    int idx_cc = h_ceil * N + w_ceil;

    // 4. Read 4 corners and weighted sum → write output
    // Process VEC_SIZE elements for memory coalescing
    const T* row_ff = pos_embed_weight + idx_ff * hidden_dim;
    const T* row_fc = pos_embed_weight + idx_fc * hidden_dim;
    const T* row_cf = pos_embed_weight + idx_cf * hidden_dim;
    const T* row_cc = pos_embed_weight + idx_cc * hidden_dim;
    T* out_row = output + (int64_t)token_idx * hidden_dim;

    #pragma unroll
    for (int v = 0; v < VEC_SIZE; v++) {
        int d = dim_start + v;
        if (d < hidden_dim) {
            float val = w_ff * to_float(row_ff[d]);
            val = val + w_fc * to_float(row_fc[d]);
            val = val + w_cf * to_float(row_cf[d]);
            val = val + w_cc * to_float(row_cc[d]);
            out_row[d] = from_float<T>(val);
        }
    }
}

/**
 * Backward for pos_embed_weight.
 *
 * Forward computes:
 *   output[token, d] = sum_i weight_i(token) * pos_embed_weight[idx_i(token), d]
 *
 * Therefore:
 *   grad_weight[idx_i(token), d] += grad_output[token, d] * weight_i(token)
 *
 * Different output tokens can map to the same source corner, so this path uses
 * atomicAdd. Accumulation is fp32 by definition.
 */
template <int VEC_SIZE>
__global__ void fused_pos_embed_interp_backward_kernel(
    const float* __restrict__ grad_output,  // [total_output, hidden_dim]
    const int32_t* __restrict__ grid_thw,   // [num_images, 3]
    const int32_t* __restrict__ out_cumsum, // [num_images+1] cumsum of t*h*w
    float* __restrict__ grad_weight,        // [N*N, hidden_dim]
    const int num_grid_per_side,
    const int spatial_merge_size,
    const int hidden_dim,
    const int num_images,
    const int total_output
) {
    const int token_idx = blockIdx.x;
    const int dim_start = (blockIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;

    if (token_idx >= total_output || dim_start >= hidden_dim) return;

    const int N = num_grid_per_side;
    const int m = spatial_merge_size;
    const int m2 = m * m;

    int img = find_image_idx(out_cumsum, num_images, token_idx);

    int32_t h = grid_thw[img * 3 + 1];
    int32_t w = grid_thw[img * 3 + 2];
    int32_t hw = h * w;
    int32_t local_idx = token_idx - out_cumsum[img];
    int32_t pos_in_frame = local_idx % hw;

    int32_t w_div_m = w / m;
    int32_t block_idx = pos_in_frame / m2;
    int32_t inner_idx = pos_in_frame % m2;
    int32_t block_row = block_idx / w_div_m;
    int32_t block_col = block_idx % w_div_m;
    int32_t inner_row = inner_idx / m;
    int32_t inner_col = inner_idx % m;

    int32_t src_row = block_row * m + inner_row;
    int32_t src_col = block_col * m + inner_col;

    const float h_coord = linspace_coord(src_row, h, N);
    const float w_coord = linspace_coord(src_col, w, N);

    int h_floor = (int)h_coord;
    int w_floor = (int)w_coord;
    int h_ceil = min(h_floor + 1, N - 1);
    int w_ceil = min(w_floor + 1, N - 1);

    float dh = h_coord - (float)h_floor;
    float dw = w_coord - (float)w_floor;

    const float w_ff = (1.0f - dh) * (1.0f - dw);
    const float w_fc = (1.0f - dh) * dw;
    const float w_cf = dh * (1.0f - dw);
    const float w_cc = dh * dw;

    int idx_ff = h_floor * N + w_floor;
    int idx_fc = h_floor * N + w_ceil;
    int idx_cf = h_ceil * N + w_floor;
    int idx_cc = h_ceil * N + w_ceil;

    const float* grad_row = grad_output + (int64_t)token_idx * hidden_dim;

    #pragma unroll
    for (int v = 0; v < VEC_SIZE; v++) {
        int d = dim_start + v;
        if (d < hidden_dim) {
            const float grad = grad_row[d];
            atomicAdd(grad_weight + (int64_t)idx_ff * hidden_dim + d, grad * w_ff);
            atomicAdd(grad_weight + (int64_t)idx_fc * hidden_dim + d, grad * w_fc);
            atomicAdd(grad_weight + (int64_t)idx_cf * hidden_dim + d, grad * w_cf);
            atomicAdd(grad_weight + (int64_t)idx_cc * hidden_dim + d, grad * w_cc);
        }
    }
}

void PosEmbedInterp(
    wallx_cuda::TensorView pos_embed_weight,  // [N*N, hidden_dim]
    wallx_cuda::TensorView grid_thw,          // [num_images, 3], int32
    wallx_cuda::TensorView out_cumsum,        // [num_images+1], int32
    wallx_cuda::TensorView hw_cumsum,         // [num_images+1], int32
    wallx_cuda::TensorView output,            // [total_output, hidden_dim]
    int num_grid_per_side,
    int spatial_merge_size
) {
    ASSERT_CHECK(pos_embed_weight.ndim() == 2);
    ASSERT_CHECK(pos_embed_weight.dtype().code == output.dtype().code);
    ASSERT_CHECK(pos_embed_weight.dtype().bits == output.dtype().bits);
    ASSERT_CHECK(grid_thw.ndim() == 2 && grid_thw.size(1) == 3);
    ASSERT_CHECK(output.ndim() == 2);
    ASSERT_CHECK(pos_embed_weight.size(1) == output.size(1));

    const int num_images = grid_thw.size(0);
    const int hidden_dim = output.size(1);
    const int total_output = output.size(0);

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(output.device().device_type, output.device().device_id));

    constexpr int VEC_SIZE = 4;
    const int threads = 256;
    const int dim_blocks = (hidden_dim + threads * VEC_SIZE - 1) / (threads * VEC_SIZE);
    dim3 grid(total_output, dim_blocks);

    const int32_t* grid_ptr = static_cast<int32_t*>(grid_thw.data_ptr());
    const int32_t* out_cum_ptr = static_cast<int32_t*>(out_cumsum.data_ptr());
    const int32_t* hw_cum_ptr = static_cast<int32_t*>(hw_cumsum.data_ptr());

    if (output.dtype().code == kDLFloat && output.dtype().bits == 32) {
        fused_pos_embed_interp_kernel<float, VEC_SIZE><<<grid, threads, 0, stream>>>(
            static_cast<const float*>(pos_embed_weight.data_ptr()),
            grid_ptr, out_cum_ptr, hw_cum_ptr,
            static_cast<float*>(output.data_ptr()),
            num_grid_per_side, spatial_merge_size, hidden_dim, num_images, total_output);
    } else if (output.dtype().code == kDLFloat && output.dtype().bits == 16) {
        fused_pos_embed_interp_kernel<__half, VEC_SIZE><<<grid, threads, 0, stream>>>(
            static_cast<const __half*>(pos_embed_weight.data_ptr()),
            grid_ptr, out_cum_ptr, hw_cum_ptr,
            static_cast<__half*>(output.data_ptr()),
            num_grid_per_side, spatial_merge_size, hidden_dim, num_images, total_output);
    } else if (output.dtype().code == kDLBfloat && output.dtype().bits == 16) {
        fused_pos_embed_interp_kernel<__nv_bfloat16, VEC_SIZE><<<grid, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(pos_embed_weight.data_ptr()),
            grid_ptr, out_cum_ptr, hw_cum_ptr,
            static_cast<__nv_bfloat16*>(output.data_ptr()),
            num_grid_per_side, spatial_merge_size, hidden_dim, num_images, total_output);
    } else {
        ASSERT_CHECK(false);
    }

    sync_check_cuda_error();
}

void PosEmbedInterpBackward(
    wallx_cuda::TensorView grad_output,  // [total_output, hidden_dim]
    wallx_cuda::TensorView grid_thw,     // [num_images, 3], int32
    wallx_cuda::TensorView out_cumsum,   // [num_images+1], int32
    wallx_cuda::TensorView hw_cumsum,    // unused, kept for metadata symmetry
    wallx_cuda::TensorView grad_weight,  // [N*N, hidden_dim]
    int num_grid_per_side,
    int spatial_merge_size
) {
    ASSERT_CHECK(grad_output.ndim() == 2);
    ASSERT_CHECK(grid_thw.ndim() == 2 && grid_thw.size(1) == 3);
    ASSERT_CHECK(out_cumsum.ndim() == 1);
    ASSERT_CHECK(hw_cumsum.ndim() == 1);
    ASSERT_CHECK(grad_weight.ndim() == 2);
    ASSERT_CHECK(grad_output.dtype().code == kDLFloat);
    ASSERT_CHECK(grad_output.dtype().bits == 32);
    ASSERT_CHECK(grad_weight.dtype().code == kDLFloat);
    ASSERT_CHECK(grad_weight.dtype().bits == 32);
    ASSERT_CHECK(grad_output.size(1) == grad_weight.size(1));
    ASSERT_CHECK(grad_weight.size(0) == num_grid_per_side * num_grid_per_side);
    ASSERT_CHECK(grid_thw.IsContiguous());
    ASSERT_CHECK(out_cumsum.IsContiguous());
    ASSERT_CHECK(grad_output.IsContiguous());
    ASSERT_CHECK(grad_weight.IsContiguous());

    (void)hw_cumsum;

    const int num_images = grid_thw.size(0);
    const int hidden_dim = grad_output.size(1);
    const int total_output = grad_output.size(0);

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(grad_output.device().device_type, grad_output.device().device_id));

    constexpr int VEC_SIZE = 4;
    const int threads = 256;
    const int dim_blocks = (hidden_dim + threads * VEC_SIZE - 1) / (threads * VEC_SIZE);
    dim3 grid(total_output, dim_blocks);

    const int32_t* grid_ptr = static_cast<int32_t*>(grid_thw.data_ptr());
    const int32_t* out_cum_ptr = static_cast<int32_t*>(out_cumsum.data_ptr());

    fused_pos_embed_interp_backward_kernel<VEC_SIZE><<<grid, threads, 0, stream>>>(
        static_cast<float*>(grad_output.data_ptr()),
        grid_ptr,
        out_cum_ptr,
        static_cast<float*>(grad_weight.data_ptr()),
        num_grid_per_side,
        spatial_merge_size,
        hidden_dim,
        num_images,
        total_output);

    sync_check_cuda_error();
}




} // namespace wallx_cuda_pos_embed_interp
