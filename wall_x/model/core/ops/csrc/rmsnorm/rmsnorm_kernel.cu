// Re-enable CUDA half operators (PyTorch disables them)
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../common/cuda_utils.h"

namespace tvm_ffi_rmsnorm {

struct RMSNormShape {
  int64_t num_tokens;
  int64_t hidden_size;
};

RMSNormShape GetRMSNormShape(const at::Tensor& tensor) {
  ASSERT_CHECK(tensor.dim() >= 1);
  ASSERT_CHECK(tensor.is_contiguous());
  const int ndim = tensor.dim();
  const int64_t hidden_size = tensor.size(ndim - 1);
  ASSERT_CHECK(hidden_size > 0);
  int64_t num_tokens = 1;
  for (int i = 0; i < ndim - 1; ++i) {
    num_tokens *= tensor.size(i);
  }
  ASSERT_CHECK(num_tokens > 0);
  return {num_tokens, hidden_size};
}

void CheckSameShape(const at::Tensor& lhs,
                    const at::Tensor& rhs) {
  ASSERT_CHECK(lhs.dim() == rhs.dim());
  for (int i = 0; i < lhs.dim(); ++i) {
    ASSERT_CHECK(lhs.size(i) == rhs.size(i));
  }
}

int64_t GetVecSize(at::ScalarType dtype) {
  return VecTraits<float>::vec_size;
  throw std::runtime_error("Unsupported data type for RMSNorm");
}

__device__ __forceinline__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;
  val = warpReduceSum(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  int num_warps = (blockDim.x + 31) / 32;
  val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
  if (wid == 0) val = warpReduceSum(val);
  return val;
}

// ======================== Forward ========================

template<typename T>
__global__ void rmsnormKernel(
    T* output, const T* input, const T* weight,
    const float epsilon, const int64_t hidden_size) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;
    float square_sum = 0.0f;
    __shared__ float inv_rms;
    const int64_t tid = threadIdx.x;
    const int64_t bid = blockIdx.x;
    const int64_t tokens_offset = bid * hidden_size;
    const int64_t vec_hidden = hidden_size / VEC_SIZE;

    for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
        Vec x_vec = VecTraits<T>::load(input + tokens_offset + i * VEC_SIZE);
        if constexpr (std::is_same_v<T, float>) {
            square_sum += x_vec.x*x_vec.x + x_vec.y*x_vec.y + x_vec.z*x_vec.z + x_vec.w*x_vec.w;
        } else if constexpr (std::is_same_v<T, __half>) {
            square_sum += __half2float(x_vec.x)*__half2float(x_vec.x) + __half2float(x_vec.y)*__half2float(x_vec.y);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            square_sum += __bfloat162float(x_vec.x)*__bfloat162float(x_vec.x) + __bfloat162float(x_vec.y)*__bfloat162float(x_vec.y);
        }
    }
    square_sum = blockReduceSum(square_sum);
    if (tid == 0) inv_rms = rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
    __syncthreads();

    for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
        Vec in_vec = VecTraits<T>::load(input + tokens_offset + i * VEC_SIZE);
        Vec w_vec = VecTraits<T>::load(weight + i * VEC_SIZE);
        if constexpr (std::is_same_v<T, float>) {
            float4 out;
            out.x = in_vec.x * w_vec.x * inv_rms; out.y = in_vec.y * w_vec.y * inv_rms;
            out.z = in_vec.z * w_vec.z * inv_rms; out.w = in_vec.w * w_vec.w * inv_rms;
            VecTraits<T>::store(output + tokens_offset + i * VEC_SIZE, out);
        } else if constexpr (std::is_same_v<T, __half>) {
            __half2 out;
            out.x = __float2half(__half2float(in_vec.x) * __half2float(w_vec.x) * inv_rms);
            out.y = __float2half(__half2float(in_vec.y) * __half2float(w_vec.y) * inv_rms);
            VecTraits<T>::store(output + tokens_offset + i * VEC_SIZE, out);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            __nv_bfloat162 out;
            out.x = __float2bfloat16(__bfloat162float(in_vec.x) * __bfloat162float(w_vec.x) * inv_rms);
            out.y = __float2bfloat16(__bfloat162float(in_vec.y) * __bfloat162float(w_vec.y) * inv_rms);
            VecTraits<T>::store(output + tokens_offset + i * VEC_SIZE, out);
        }
    }
}

void RMSNorm(const at::Tensor& output, const at::Tensor& input,
             const at::Tensor& weight, double epsilon) {
  ASSERT_CHECK(weight.dim() == 1);
  ASSERT_CHECK(weight.is_contiguous());
  ASSERT_CHECK(output.is_contiguous());
  CheckSameShape(output, input);
  const RMSNormShape shape = GetRMSNormShape(input);
  const int64_t hidden_size = shape.hidden_size;
  const int64_t num_tokens = shape.num_tokens;
  ASSERT_CHECK(hidden_size == weight.size(0));
  ASSERT_CHECK(hidden_size % GetVecSize(input.scalar_type()) == 0);
  cudaStream_t stream = static_cast<cudaStream_t>(
      at::cuda::getCurrentCUDAStream().stream());
  const int64_t block_size = std::min(hidden_size, 1024L);
  if (input.scalar_type() == at::kFloat)
    rmsnormKernel<float><<<num_tokens, block_size, 0, stream>>>(
        static_cast<float*>(output.data_ptr()), static_cast<const float*>(input.data_ptr()),
        static_cast<const float*>(weight.data_ptr()), static_cast<float>(epsilon), hidden_size);
  else if (input.scalar_type() == at::kHalf)
    rmsnormKernel<half><<<num_tokens, block_size, 0, stream>>>(
        static_cast<half*>(output.data_ptr()), static_cast<const half*>(input.data_ptr()),
        static_cast<const half*>(weight.data_ptr()), static_cast<float>(epsilon), hidden_size);
  else if (input.scalar_type() == at::kBFloat16)
    rmsnormKernel<__nv_bfloat16><<<num_tokens, block_size, 0, stream>>>(
        static_cast<__nv_bfloat16*>(output.data_ptr()), static_cast<const __nv_bfloat16*>(input.data_ptr()),
        static_cast<const __nv_bfloat16*>(weight.data_ptr()), static_cast<float>(epsilon), hidden_size);
  else throw std::runtime_error("Unsupported data type for RMSNorm");
  sync_check_cuda_error();
}

// Removed TVM FFI export (see pybind11 registration)

// ======================== Backward: Persistent Kernel + Partial Reduction ========================
//
// Launch num_blocks = min(num_tokens, SM_count) persistent blocks.
// Each block loops over assigned rows computing dx, accumulating dw in registers.
// After the loop, each block writes its partial dw to partial_dw[block_id, hidden_size].
// A lightweight reduction kernel sums partial_dw into final dw.
//
// Advantages:
//   - No atomicAdd contention (lock-free)
//   - Weight W loaded once into L1 cache, reused across all rows
//   - X and dY read exactly once per row, producing both dX and partial dW
//

template<typename T, int MAX_VEC_PER_THREAD>
__global__ void rmsnormBackwardPersistentKernel(
    T* __restrict__ dx,
    float* __restrict__ partial_dw,
    const T* __restrict__ dy,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const float epsilon,
    const int64_t hidden_size,
    const int64_t num_tokens
) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;

    const int64_t tid = threadIdx.x;
    const int64_t num_blocks = gridDim.x;
    const int64_t vec_hidden = hidden_size / VEC_SIZE;
    const float inv_hidden = 1.0f / static_cast<float>(hidden_size);

    __shared__ float s_inv_rms;
    __shared__ float s_c;

    // Register accumulator for partial dw
    float local_dw[MAX_VEC_PER_THREAD * 4];
    #pragma unroll
    for (int j = 0; j < MAX_VEC_PER_THREAD * 4; j++) local_dw[j] = 0.0f;

    // Persistent loop: each block processes multiple rows
    for (int64_t row = blockIdx.x; row < num_tokens; row += num_blocks) {
        const int64_t off = row * hidden_size;
        float sq = 0.0f, dp = 0.0f;

        // Pass 1: reductions
        for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
            Vec xv = VecTraits<T>::load(input + off + i * VEC_SIZE);
            Vec dv = VecTraits<T>::load(dy + off + i * VEC_SIZE);
            Vec wv = VecTraits<T>::load(weight + i * VEC_SIZE);
            if constexpr (std::is_same_v<T, float>) {
                sq += xv.x*xv.x + xv.y*xv.y + xv.z*xv.z + xv.w*xv.w;
                dp += dv.x*wv.x*xv.x + dv.y*wv.y*xv.y + dv.z*wv.z*xv.z + dv.w*wv.w*xv.w;
            } else if constexpr (std::is_same_v<T, __half>) {
                float x0=__half2float(xv.x), x1=__half2float(xv.y);
                float d0=__half2float(dv.x), d1=__half2float(dv.y);
                float w0=__half2float(wv.x), w1=__half2float(wv.y);
                sq += x0*x0 + x1*x1; dp += d0*w0*x0 + d1*w1*x1;
            } else {
                float x0=__bfloat162float(xv.x), x1=__bfloat162float(xv.y);
                float d0=__bfloat162float(dv.x), d1=__bfloat162float(dv.y);
                float w0=__bfloat162float(wv.x), w1=__bfloat162float(wv.y);
                sq += x0*x0 + x1*x1; dp += d0*w0*x0 + d1*w1*x1;
            }
        }
        sq = blockReduceSum(sq); __syncthreads();
        dp = blockReduceSum(dp);
        if (tid == 0) {
            float ve = sq * inv_hidden + epsilon;
            s_inv_rms = rsqrtf(ve);
            s_c = dp / (static_cast<float>(hidden_size) * ve);
        }
        __syncthreads();
        const float irms = s_inv_rms, cv = s_c;

        // Pass 2: compute dx + accumulate dw in registers
        int li = 0;
        for (int64_t i = tid; i < vec_hidden; i += blockDim.x, li++) {
            Vec xv = VecTraits<T>::load(input + off + i * VEC_SIZE);
            Vec dv = VecTraits<T>::load(dy + off + i * VEC_SIZE);
            Vec wv = VecTraits<T>::load(weight + i * VEC_SIZE);
            if constexpr (std::is_same_v<T, float>) {
                float4 o;
                o.x = irms*(dv.x*wv.x - xv.x*cv); o.y = irms*(dv.y*wv.y - xv.y*cv);
                o.z = irms*(dv.z*wv.z - xv.z*cv); o.w = irms*(dv.w*wv.w - xv.w*cv);
                VecTraits<T>::store(dx + off + i*VEC_SIZE, o);
                local_dw[li*4+0] += dv.x*xv.x*irms; local_dw[li*4+1] += dv.y*xv.y*irms;
                local_dw[li*4+2] += dv.z*xv.z*irms; local_dw[li*4+3] += dv.w*xv.w*irms;
            } else if constexpr (std::is_same_v<T, __half>) {
                float x0=__half2float(xv.x),x1=__half2float(xv.y);
                float d0=__half2float(dv.x),d1=__half2float(dv.y);
                float w0=__half2float(wv.x),w1=__half2float(wv.y);
                __half2 o; o.x=__float2half(irms*(d0*w0-x0*cv)); o.y=__float2half(irms*(d1*w1-x1*cv));
                VecTraits<T>::store(dx + off + i*VEC_SIZE, o);
                local_dw[li*2+0] += d0*x0*irms; local_dw[li*2+1] += d1*x1*irms;
            } else {
                float x0=__bfloat162float(xv.x),x1=__bfloat162float(xv.y);
                float d0=__bfloat162float(dv.x),d1=__bfloat162float(dv.y);
                float w0=__bfloat162float(wv.x),w1=__bfloat162float(wv.y);
                __nv_bfloat162 o; o.x=__float2bfloat16(irms*(d0*w0-x0*cv)); o.y=__float2bfloat16(irms*(d1*w1-x1*cv));
                VecTraits<T>::store(dx + off + i*VEC_SIZE, o);
                local_dw[li*2+0] += d0*x0*irms; local_dw[li*2+1] += d1*x1*irms;
            }
        }
    }

    // Write partial dw to global memory
    const int64_t poff = blockIdx.x * hidden_size;
    int li = 0;
    for (int64_t i = tid; i < vec_hidden; i += blockDim.x, li++) {
        if constexpr (std::is_same_v<T, float>) {
            float4 dw_out;
            dw_out.x=local_dw[li*4+0]; dw_out.y=local_dw[li*4+1];
            dw_out.z=local_dw[li*4+2]; dw_out.w=local_dw[li*4+3];
            *reinterpret_cast<float4*>(partial_dw + poff + i*VEC_SIZE) = dw_out;
        } else {
            float2 dw_out;
            dw_out.x=local_dw[li*2+0]; dw_out.y=local_dw[li*2+1];
            *reinterpret_cast<float2*>(partial_dw + poff + i*VEC_SIZE) = dw_out;
        }
    }
}

// Final reduction: partial_dw[num_blocks, hidden_size] -> dw[hidden_size]
__global__ void rmsnormReduceDwKernel(
    float* __restrict__ dw, const float* __restrict__ partial_dw,
    const int64_t hidden_size, const int64_t num_blocks) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_size) return;
    float sum = 0.0f;
    for (int64_t b = 0; b < num_blocks; b++) sum += partial_dw[b * hidden_size + idx];
    dw[idx] = sum;
}

static int get_sm_count() {
    static int sm_count = -1;
    if (sm_count < 0) {
        int device; cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    }
    return sm_count;
}

// partial_dw_view is pre-allocated from Python side to avoid cudaMallocAsync overhead
void RMSNormBackward(
    const at::Tensor& dx,
    const at::Tensor& dw,
    const at::Tensor& partial_dw_view,
    const at::Tensor& dy,
    const at::Tensor& input,
    const at::Tensor& weight,
    double epsilon) {

  ASSERT_CHECK(dw.scalar_type() == at::kFloat);
  ASSERT_CHECK(partial_dw_view.scalar_type() == at::kFloat);
  ASSERT_CHECK(dx.is_contiguous());
  ASSERT_CHECK(dy.is_contiguous());
  ASSERT_CHECK(input.is_contiguous());
  ASSERT_CHECK(weight.is_contiguous());
  ASSERT_CHECK(dw.is_contiguous());
  ASSERT_CHECK(partial_dw_view.is_contiguous());
  const RMSNormShape shape = GetRMSNormShape(input);
  const int64_t hidden_size = shape.hidden_size;
  const int64_t num_tokens = shape.num_tokens;
  ASSERT_CHECK(hidden_size == weight.size(0));
  ASSERT_CHECK(hidden_size % GetVecSize(input.scalar_type()) == 0);
  cudaStream_t stream = static_cast<cudaStream_t>(
      at::cuda::getCurrentCUDAStream().stream());
  const int64_t block_size = std::min(hidden_size, 1024L);
  const int sm_count = get_sm_count();
  const int64_t num_blocks = std::min(static_cast<int64_t>(sm_count), num_tokens);
  ASSERT_CHECK(partial_dw_view.dim() == 2);
  ASSERT_CHECK(partial_dw_view.size(0) == num_blocks);
  ASSERT_CHECK(partial_dw_view.size(1) == hidden_size);
  ASSERT_CHECK(dw.dim() == 1);
  ASSERT_CHECK(dw.size(0) == hidden_size);
  float* partial_dw = static_cast<float*>(partial_dw_view.data_ptr());
  constexpr int MAX_VPT = 8;

  if (input.scalar_type() == at::kFloat)
    rmsnormBackwardPersistentKernel<float, MAX_VPT><<<num_blocks, block_size, 0, stream>>>(
        static_cast<float*>(dx.data_ptr()), partial_dw,
        static_cast<const float*>(dy.data_ptr()), static_cast<const float*>(input.data_ptr()),
        static_cast<const float*>(weight.data_ptr()), static_cast<float>(epsilon), hidden_size, num_tokens);
  else if (input.scalar_type() == at::kHalf)
    rmsnormBackwardPersistentKernel<half, MAX_VPT><<<num_blocks, block_size, 0, stream>>>(
        static_cast<half*>(dx.data_ptr()), partial_dw,
        static_cast<const half*>(dy.data_ptr()), static_cast<const half*>(input.data_ptr()),
        static_cast<const half*>(weight.data_ptr()), static_cast<float>(epsilon), hidden_size, num_tokens);
  else if (input.scalar_type() == at::kBFloat16)
    rmsnormBackwardPersistentKernel<__nv_bfloat16, MAX_VPT><<<num_blocks, block_size, 0, stream>>>(
        static_cast<__nv_bfloat16*>(dx.data_ptr()), partial_dw,
        static_cast<const __nv_bfloat16*>(dy.data_ptr()), static_cast<const __nv_bfloat16*>(input.data_ptr()),
        static_cast<const __nv_bfloat16*>(weight.data_ptr()), static_cast<float>(epsilon), hidden_size, num_tokens);
  else throw std::runtime_error("Unsupported data type for RMSNorm backward");

  const int64_t rt = 256;
  rmsnormReduceDwKernel<<<(hidden_size+rt-1)/rt, rt, 0, stream>>>(
      static_cast<float*>(dw.data_ptr()), partial_dw, hidden_size, num_blocks);
  sync_check_cuda_error();
}

// Removed TVM FFI export (see pybind11 registration)

}  // namespace tvm_ffi_rmsnorm
