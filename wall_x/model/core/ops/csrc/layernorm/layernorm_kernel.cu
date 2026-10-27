#include "tensor_view.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../common/cuda_utils.h"
#include "../common/reduction.cuh"

namespace wallx_cuda_layernorm {

constexpr int WARP_SIZE = 32;
constexpr int NUM_THREADS = 256;
constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

struct RowShape {
  int64_t num_tokens;
  int64_t hidden_size;
};

// LayerNorm normalizes over the LAST dim; every leading dim is a token dim.
// Reading size(0)/size(1) directly would normalize any 3D+ input ([B, S, H])
// over S with B blocks -- silently garbage. Derive the shape
// rank-agnostically and require dense row-major layout.
inline RowShape GetRowShape(const wallx_cuda::TensorView& t) {
  const int ndim = t.ndim();
  ASSERT_CHECK(ndim >= 1);
  const int64_t hidden_size = t.size(ndim - 1);
  ASSERT_CHECK(hidden_size > 0);
  ASSERT_CHECK(hidden_size % 2 == 0);  // T2-vectorized kernel
  int64_t num_tokens = 1;
  for (int i = 0; i < ndim - 1; ++i) num_tokens *= t.size(i);
  ASSERT_CHECK(num_tokens > 0);
  // dense row-major check
  int64_t expect = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    ASSERT_CHECK(t.stride(i) == expect);
    expect *= t.size(i);
  }
  return {num_tokens, hidden_size};
}

template<typename T>
__global__ void layernormKernel(
    T* __restrict__ out,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const float epsilon,
    const int64_t num_tokens,
    const int64_t hidden_size
) {
    typedef std::conditional_t<std::is_same<T, half>::value, half2,
            std::conditional_t<std::is_same<T, __nv_bfloat16>::value, __nv_bfloat162, float2>> T2;

    extern __shared__ float shared_mem[];
    T2* input_buf = (T2*)shared_mem;

    __shared__ float s_mean, s_variance;
    float mean = 0.0, variance = 0.0;

    #pragma unroll 4
    for (int64_t idx = threadIdx.x; idx < hidden_size/2; idx += blockDim.x) {
        T2 elem = ((T2*)input)[blockIdx.x * hidden_size/2 + idx];
        input_buf[idx] = elem;
        const float x = (float)elem.x;
        const float y = (float)elem.y;
        mean += x+y;
        variance += x*x + y*y;
    }

    // Reduce within warp
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        mean += __shfl_down_sync(0xffffffff, mean, offset);
        variance += __shfl_down_sync(0xffffffff, variance, offset);
    }
    static __shared__ float reduction_wksp[2][NUM_WARPS];
    if ((threadIdx.x & 31) == 0) {
        reduction_wksp[0][threadIdx.x >> 5] = mean;
        reduction_wksp[1][threadIdx.x >> 5] = variance;
    }
    __syncthreads();

    // Reduce within block
    if (threadIdx.x < NUM_WARPS) {
        mean = reduction_wksp[0][threadIdx.x];
        variance = reduction_wksp[1][threadIdx.x];
    }
    #pragma unroll
    for (int offset = NUM_WARPS/2; offset > 0; offset /= 2) {
        mean += __shfl_down_sync(0xffffffff, mean, offset);
        variance += __shfl_down_sync(0xffffffff, variance, offset);
    }

    if (threadIdx.x == 0) {
        float hidden_size_fp = (float)hidden_size;
        s_mean = mean / hidden_size_fp;
        s_variance = rsqrtf(variance / hidden_size_fp - s_mean * s_mean + epsilon);
    }
    __syncthreads();

    // Normalize in fp32 and round once at the store. The previous version
    // cast mean/rstd to T (bf16) and did the whole affine in bf16, costing
    // ~4x the rounding error of aten's fp32-internal layer_norm.
    const float final_mean = s_mean;
    const float final_rstd = s_variance;
    #pragma unroll 4
    for (int64_t idx = threadIdx.x; idx < hidden_size/2; idx += blockDim.x) {
        T2 x = input_buf[idx];
        T2 weight_elem = ((T2*)weight)[idx];
        T2 bias_elem = ((T2*)bias)[idx];
        T2 result;
        result.x = (decltype(result.x))(
            ((float)x.x - final_mean) * final_rstd * (float)weight_elem.x
            + (float)bias_elem.x);
        result.y = (decltype(result.y))(
            ((float)x.y - final_mean) * final_rstd * (float)weight_elem.y
            + (float)bias_elem.y);
        ((T2*)out)[blockIdx.x * hidden_size/2 + idx] = result;
    }
}

void Layernorm(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView weight,
    wallx_cuda::TensorView bias,
    double epsilon
) {
    const RowShape shape = GetRowShape(output);
    const RowShape in_shape = GetRowShape(input);
    ASSERT_CHECK(in_shape.num_tokens == shape.num_tokens);
    ASSERT_CHECK(in_shape.hidden_size == shape.hidden_size);
    const int64_t num_tokens = shape.num_tokens;
    const int64_t hidden_size = shape.hidden_size;
    ASSERT_CHECK(weight.size(weight.ndim() - 1) == hidden_size);
    ASSERT_CHECK(bias.size(bias.ndim() - 1) == hidden_size);

    dim3 grid(num_tokens);
    dim3 block(NUM_THREADS);

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(output.device().device_type, output.device().device_id));

    if (output.dtype().code == kDLFloat && output.dtype().bits == 32) {
        layernormKernel<float><<<grid, block, hidden_size*sizeof(float), stream>>>(
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(input.data_ptr()),
            static_cast<const float*>(weight.data_ptr()),
            static_cast<const float*>(bias.data_ptr()),
            (float)epsilon, num_tokens, hidden_size);
    } else if (output.dtype().code == kDLFloat && output.dtype().bits == 16) {
        layernormKernel<half><<<grid, block, hidden_size*sizeof(half), stream>>>(
            static_cast<half*>(output.data_ptr()),
            static_cast<const half*>(input.data_ptr()),
            static_cast<const half*>(weight.data_ptr()),
            static_cast<const half*>(bias.data_ptr()),
            (float)epsilon, num_tokens, hidden_size);
    } else if (output.dtype().code == kDLBfloat && output.dtype().bits == 16) {
        layernormKernel<__nv_bfloat16><<<grid, block, hidden_size*sizeof(__nv_bfloat16), stream>>>(
            static_cast<__nv_bfloat16*>(output.data_ptr()),
            static_cast<const __nv_bfloat16*>(input.data_ptr()),
            static_cast<const __nv_bfloat16*>(weight.data_ptr()),
            static_cast<const __nv_bfloat16*>(bias.data_ptr()),
            (float)epsilon, num_tokens, hidden_size);
    } else {
        throw std::runtime_error("Unsupported data type for layernorm");
    }

    sync_check_cuda_error();
}



}  // namespace wallx_cuda_layernorm
