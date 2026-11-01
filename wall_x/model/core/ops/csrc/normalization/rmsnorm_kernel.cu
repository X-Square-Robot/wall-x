#include "tensor_view.h"
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


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <type_traits>

#include "../common/cuda_utils.h"

namespace wallx_cuda_rmsnorm {

struct RMSNormShape {
  int64_t num_tokens;
  int64_t hidden_size;
  int64_t row_stride;
};

struct TensorRowLayout {
  int ndim;
  int64_t sizes[8];
  int64_t strides[8];
};

RMSNormShape GetRMSNormShape(const wallx_cuda::TensorView& tensor) {
  ASSERT_CHECK(tensor.ndim() >= 1);
  const int ndim = tensor.ndim();
  const int64_t hidden_size = tensor.size(ndim - 1);
  ASSERT_CHECK(hidden_size > 0);
  ASSERT_CHECK(tensor.stride(ndim - 1) == 1);

  int64_t num_tokens = 1;
  for (int i = 0; i < ndim - 1; ++i) {
    num_tokens *= tensor.size(i);
  }
  ASSERT_CHECK(num_tokens > 0);

  int64_t row_stride = hidden_size;
  if (ndim > 1) {
    row_stride = tensor.stride(ndim - 2);
    ASSERT_CHECK(row_stride >= hidden_size);
    int64_t expected_stride = row_stride;
    for (int i = ndim - 2; i >= 0; --i) {
      ASSERT_CHECK(tensor.stride(i) == expected_stride);
      expected_stride *= tensor.size(i);
    }
  }
  return {num_tokens, hidden_size, row_stride};
}

void CheckSameShape(const wallx_cuda::TensorView& lhs,
                    const wallx_cuda::TensorView& rhs) {
  ASSERT_CHECK(lhs.ndim() == rhs.ndim());
  for (int i = 0; i < lhs.ndim(); ++i) {
    ASSERT_CHECK(lhs.size(i) == rhs.size(i));
  }
}

TensorRowLayout GetTensorRowLayout(const wallx_cuda::TensorView& tensor) {
  ASSERT_CHECK(tensor.ndim() >= 1);
  ASSERT_CHECK(tensor.ndim() <= 8);
  const int ndim = tensor.ndim();
  ASSERT_CHECK(tensor.size(ndim - 1) > 0);
  ASSERT_CHECK(tensor.stride(ndim - 1) == 1);

  TensorRowLayout layout;
  layout.ndim = ndim;
  for (int i = 0; i < 8; ++i) {
    layout.sizes[i] = 1;
    layout.strides[i] = 0;
  }
  for (int i = 0; i < ndim; ++i) {
    layout.sizes[i] = tensor.size(i);
    layout.strides[i] = tensor.stride(i);
  }
  return layout;
}

int64_t GetTensorNumTokens(const wallx_cuda::TensorView& tensor) {
  ASSERT_CHECK(tensor.ndim() >= 1);
  int64_t num_tokens = 1;
  for (int i = 0; i < tensor.ndim() - 1; ++i) {
    num_tokens *= tensor.size(i);
  }
  return num_tokens;
}

void CheckTensorRowLayoutVectorized(const TensorRowLayout& layout, int64_t vec_size) {
  for (int i = 0; i < layout.ndim - 1; ++i) {
    ASSERT_CHECK(layout.strides[i] % vec_size == 0);
  }
}

bool IsFloat32(const DLDataType& dtype) {
  return dtype.code == kDLFloat && dtype.bits == 32;
}

bool IsFloat16(const DLDataType& dtype) {
  return dtype.code == kDLFloat && dtype.bits == 16;
}

bool IsBFloat16(const DLDataType& dtype) {
  return dtype.code == kDLBfloat && dtype.bits == 16;
}

int64_t GetVecSize(const DLDataType& dtype) {
  if (IsFloat32(dtype)) return VecTraits<float>::vec_size;
  if (IsFloat16(dtype)) return VecTraits<__half>::vec_size;
  if (IsBFloat16(dtype)) return VecTraits<__nv_bfloat16>::vec_size;
  throw std::runtime_error("Unsupported data type for RMSNorm");
}

template <typename T>
struct dependent_false : std::false_type {};

template<typename T>
__device__ __forceinline__ float ScalarToFloat(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, __half>) {
    return __half2float(value);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(value);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported RMSNorm scalar type");
  }
}

template<typename T>
__device__ __forceinline__ float RoundFloatToScalarType(float value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, __half>) {
    return __half2float(__float2half(value));
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(__float2bfloat16(value));
  } else {
    static_assert(dependent_false<T>::value, "Unsupported RMSNorm scalar type");
  }
}

template<typename T>
__device__ __forceinline__ void VecToFloats(
    typename VecTraits<T>::Type vec, float* values) {
  if constexpr (std::is_same_v<T, float>) {
    values[0] = vec.x;
    values[1] = vec.y;
    values[2] = vec.z;
    values[3] = vec.w;
  } else if constexpr (std::is_same_v<T, __half>) {
    values[0] = __half2float(vec.x);
    values[1] = __half2float(vec.y);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    values[0] = __bfloat162float(vec.x);
    values[1] = __bfloat162float(vec.y);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported RMSNorm vector type");
  }
}

template<typename T>
__device__ __forceinline__ typename VecTraits<T>::Type FloatsToVec(const float* values) {
  typename VecTraits<T>::Type vec;
  if constexpr (std::is_same_v<T, float>) {
    vec.x = values[0];
    vec.y = values[1];
    vec.z = values[2];
    vec.w = values[3];
  } else if constexpr (std::is_same_v<T, __half>) {
    vec.x = __float2half(values[0]);
    vec.y = __float2half(values[1]);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    vec.x = __float2bfloat16(values[0]);
    vec.y = __float2bfloat16(values[1]);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported RMSNorm vector type");
  }
  return vec;
}

template<typename WeightDType, int VEC_SIZE>
__device__ __forceinline__ void LoadWeightFloats(
    const WeightDType* weight, int64_t offset, float* values) {
  #pragma unroll
  for (int j = 0; j < VEC_SIZE; ++j) {
    values[j] = ScalarToFloat(weight[offset + j]);
  }
}

template<typename T>
__device__ __forceinline__ void Load4Floats(const T* ptr, float* values) {
  if constexpr (std::is_same_v<T, float>) {
    VecToFloats<T>(VecTraits<T>::load(ptr), values);
  } else {
    VecToFloats<T>(VecTraits<T>::load(ptr), values);
    VecToFloats<T>(VecTraits<T>::load(ptr + 2), values + 2);
  }
}

template<typename T>
__device__ __forceinline__ void Store4Scalars(T* ptr, const float* values) {
  if constexpr (std::is_same_v<T, float>) {
    VecTraits<T>::store(ptr, FloatsToVec<T>(values));
  } else {
    VecTraits<T>::store(ptr, FloatsToVec<T>(values));
    VecTraits<T>::store(ptr + 2, FloatsToVec<T>(values + 2));
  }
}

__device__ __forceinline__ void CpAsync8(void* smem_dst, const void* gmem_src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned smem_addr =
      static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, 8;\n" ::
                   "r"(smem_addr), "l"(gmem_src));
#else
  *reinterpret_cast<uint2*>(smem_dst) =
      *reinterpret_cast<const uint2*>(gmem_src);
#endif
}

__device__ __forceinline__ void CpAsyncCommitGroup() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template<int PendingGroups>
__device__ __forceinline__ void CpAsyncWaitGroup() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" :: "n"(PendingGroups));
#endif
}

template<typename T>
__device__ __forceinline__ uint2 Load4Raw(const T* ptr) {
  static_assert(!std::is_same_v<T, float>, "Load4Raw is for 16-bit RMSNorm data");
  return *reinterpret_cast<const uint2*>(ptr);
}

template<typename T>
__device__ __forceinline__ uint32_t PackVec2Bits(typename VecTraits<T>::Type vec) {
  static_assert(!std::is_same_v<T, float>, "PackVec2Bits is for 16-bit RMSNorm data");
  union Bits {
    typename VecTraits<T>::Type vec;
    uint32_t raw;
  };
  Bits bits;
  bits.vec = vec;
  return bits.raw;
}

template<typename T>
__device__ __forceinline__ typename VecTraits<T>::Type UnpackVec2Bits(uint32_t raw) {
  static_assert(!std::is_same_v<T, float>, "UnpackVec2Bits is for 16-bit RMSNorm data");
  union Bits {
    uint32_t raw;
    typename VecTraits<T>::Type vec;
  };
  Bits bits;
  bits.raw = raw;
  return bits.vec;
}

template<typename T>
__device__ __forceinline__ void Raw4ToFloats(uint2 raw, float* values) {
  VecToFloats<T>(UnpackVec2Bits<T>(raw.x), values);
  VecToFloats<T>(UnpackVec2Bits<T>(raw.y), values + 2);
}

template<typename T>
__device__ __forceinline__ void Store4RawFromFloats(T* ptr, const float* values) {
  uint2 raw;
  raw.x = PackVec2Bits<T>(FloatsToVec<T>(values));
  raw.y = PackVec2Bits<T>(FloatsToVec<T>(values + 2));
  *reinterpret_cast<uint2*>(ptr) = raw;
}

template<typename WeightDType>
__device__ __forceinline__ void LoadWeight4Floats(
    const WeightDType* weight, int64_t offset, float* values) {
  if constexpr (std::is_same_v<WeightDType, float>) {
    LoadWeightFloats<WeightDType, 4>(weight, offset, values);
  } else {
    Raw4ToFloats<WeightDType>(*reinterpret_cast<const uint2*>(weight + offset),
                              values);
  }
}

// ---------------- 16-byte wide access helpers (16-bit dtypes) ----------------
// The default VecTraits for half/bf16 are 4-byte (x2) vectors, which cannot
// saturate HBM on A100/H100-class parts. The wide fast paths below move data as
// 16-byte uint4 transactions (8 x 16-bit scalars) instead.

template<typename T>
__device__ __forceinline__ void Raw8ToFloats(uint4 raw, float* values) {
  static_assert(!std::is_same_v<T, float>, "Raw8ToFloats is for 16-bit data");
  VecToFloats<T>(UnpackVec2Bits<T>(raw.x), values);
  VecToFloats<T>(UnpackVec2Bits<T>(raw.y), values + 2);
  VecToFloats<T>(UnpackVec2Bits<T>(raw.z), values + 4);
  VecToFloats<T>(UnpackVec2Bits<T>(raw.w), values + 6);
}

template<typename T>
__device__ __forceinline__ uint4 FloatsToRaw8(const float* values) {
  static_assert(!std::is_same_v<T, float>, "FloatsToRaw8 is for 16-bit data");
  uint4 raw;
  raw.x = PackVec2Bits<T>(FloatsToVec<T>(values));
  raw.y = PackVec2Bits<T>(FloatsToVec<T>(values + 2));
  raw.z = PackVec2Bits<T>(FloatsToVec<T>(values + 4));
  raw.w = PackVec2Bits<T>(FloatsToVec<T>(values + 6));
  return raw;
}

// Load 8 consecutive weight scalars as floats. fp32 weights use two float4
// transactions, 16-bit weights one uint4 transaction.
template<typename WeightDType>
__device__ __forceinline__ void LoadWeight8Floats(
    const WeightDType* weight, int64_t offset, float* values) {
  if constexpr (std::is_same_v<WeightDType, float>) {
    const float4 lo = *reinterpret_cast<const float4*>(weight + offset);
    const float4 hi = *reinterpret_cast<const float4*>(weight + offset + 4);
    values[0] = lo.x; values[1] = lo.y; values[2] = lo.z; values[3] = lo.w;
    values[4] = hi.x; values[5] = hi.y; values[6] = hi.z; values[7] = hi.w;
  } else {
    Raw8ToFloats<WeightDType>(
        *reinterpret_cast<const uint4*>(weight + offset), values);
  }
}

__device__ __forceinline__ bool IsAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}


inline bool HostIsAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

__device__ __forceinline__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

template<int WIDTH>
__device__ __forceinline__ float subwarpReduceSum(float val) {
  for (int offset = WIDTH / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset, WIDTH);
  }
  return val;
}

// Mask-aware variant: for kernels where whole sub-warp row-groups may have exited
// (early-returned) before the reduction, the shuffle mask must name only the active
// lanes of this row-group — 0xffffffff would name exited lanes, which is UB.
template<int WIDTH>
__device__ __forceinline__ float subwarpReduceSum(float val, unsigned mask) {
  for (int offset = WIDTH / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(mask, val, offset, WIDTH);
  }
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

__device__ __forceinline__ float2 warpReduceSum2(float2 val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val.x += __shfl_down_sync(0xffffffff, val.x, offset);
    val.y += __shfl_down_sync(0xffffffff, val.y, offset);
  }
  return val;
}

// Reduce two independent sums in one pass (one shared round-trip / one set of
// syncthreads instead of two sequential blockReduceSum calls). Result valid on
// thread 0. NOTE: when called in a loop, callers must have a __syncthreads()
// between the consumption of the previous result and the next call (the wide
// backward kernel has one after broadcasting the row scales).
__device__ __forceinline__ float2 blockReduceSum2(float2 val) {
  static __shared__ float2 shared2[32];
  const int lane = threadIdx.x % 32;
  const int wid = threadIdx.x / 32;
  val = warpReduceSum2(val);
  if (lane == 0) shared2[wid] = val;
  __syncthreads();
  const int num_warps = (blockDim.x + 31) / 32;
  val = (threadIdx.x < num_warps) ? shared2[lane] : make_float2(0.0f, 0.0f);
  if (wid == 0) val = warpReduceSum2(val);
  return val;
}

// ======================== Forward ========================

template<typename T, typename WeightDType, bool ZeroCentered>
__global__ void rmsnormKernel(
    T* output, const T* input, const WeightDType* weight,
    float* rstd_output,
    const float epsilon, const int64_t hidden_size,
    const int64_t input_row_stride, const int64_t output_row_stride) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;
    float square_sum = 0.0f;
    __shared__ float inv_rms;
    const int64_t tid = threadIdx.x;
    const int64_t bid = blockIdx.x;
    const int64_t input_offset = bid * input_row_stride;
    const int64_t output_offset = bid * output_row_stride;
    const int64_t vec_hidden = hidden_size / VEC_SIZE;

    for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
        Vec x_vec = VecTraits<T>::load(input + input_offset + i * VEC_SIZE);
        float x_vals[VEC_SIZE];
        VecToFloats<T>(x_vec, x_vals);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            square_sum += x_vals[j] * x_vals[j];
        }
    }
    square_sum = blockReduceSum(square_sum);
    if (tid == 0) {
        inv_rms = rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
        if (rstd_output != nullptr) {
            rstd_output[bid] = inv_rms;
        }
    }
    __syncthreads();

    for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
        Vec in_vec = VecTraits<T>::load(input + input_offset + i * VEC_SIZE);
        float x_vals[VEC_SIZE], w_vals[VEC_SIZE], out_vals[VEC_SIZE];
        VecToFloats<T>(in_vec, x_vals);
        LoadWeightFloats<WeightDType, VEC_SIZE>(weight, i * VEC_SIZE, w_vals);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            if constexpr (ZeroCentered) w_vals[j] += 1.0f;
            out_vals[j] = x_vals[j] * w_vals[j] * inv_rms;
        }
        VecTraits<T>::store(output + output_offset + i * VEC_SIZE, FloatsToVec<T>(out_vals));
    }
}

template<typename T, typename WeightDType, bool ZeroCentered>
void LaunchRMSNormKernel(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView weight,
    float* rstd_output,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t input_row_stride,
    int64_t output_row_stride,
    int64_t block_size,
    cudaStream_t stream) {
  rmsnormKernel<T, WeightDType, ZeroCentered><<<num_tokens, block_size, 0, stream>>>(
      static_cast<T*>(output.data_ptr()),
      static_cast<const T*>(input.data_ptr()),
      static_cast<const WeightDType*>(weight.data_ptr()),
      rstd_output,
      epsilon, hidden_size, input_row_stride, output_row_stride);
}

// Wide (16-byte vectorized) forward path for 16-bit dtypes. One block per row;
// each thread owns up to VPT uint4 vectors (8 scalars each), the raw input bits
// are kept in registers between the reduction and the normalize pass, so each
// input element is read from global memory exactly once and written once.
template<typename T, typename WeightDType, int VPT, bool ZeroCentered>
__global__ void __launch_bounds__(1024)
rmsnormForwardWideKernel(
    T* __restrict__ output, const T* __restrict__ input,
    const WeightDType* __restrict__ weight,
    float* __restrict__ rstd_output,
    const float epsilon, const int64_t hidden_size,
    const int64_t input_row_stride, const int64_t output_row_stride) {
  constexpr int ELEMS = 8;
  static_assert(!std::is_same_v<T, float>, "wide path is for 16-bit tensors");
  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x;
  const T* in_row = input + row * input_row_stride;
  T* out_row = output + row * output_row_stride;
  const int64_t vec_hidden = hidden_size / ELEMS;

  // The weight is prefetched alongside x so its (L2) latency hides behind the
  // block reduction instead of stalling the normalize pass.
  uint4 x_raw[VPT];
  float w_cache[VPT][ELEMS];
  float sq = 0.0f;
  #pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int64_t i = tid + k * static_cast<int64_t>(blockDim.x);
    if (i < vec_hidden) {
      x_raw[k] = *reinterpret_cast<const uint4*>(in_row + i * ELEMS);
      LoadWeight8Floats<WeightDType>(weight, i * ELEMS, w_cache[k]);
      float x_vals[ELEMS];
      Raw8ToFloats<T>(x_raw[k], x_vals);
      #pragma unroll
      for (int j = 0; j < ELEMS; ++j) sq += x_vals[j] * x_vals[j];
    }
  }
  sq = blockReduceSum(sq);
  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = rsqrtf(sq / static_cast<float>(hidden_size) + epsilon);
    if (rstd_output != nullptr) {
      rstd_output[row] = s_inv_rms;
    }
  }
  __syncthreads();
  const float irms = s_inv_rms;

  #pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int64_t i = tid + k * static_cast<int64_t>(blockDim.x);
    if (i < vec_hidden) {
      float x_vals[ELEMS], out_vals[ELEMS];
      Raw8ToFloats<T>(x_raw[k], x_vals);
      #pragma unroll
      for (int j = 0; j < ELEMS; ++j) {
        float wv = w_cache[k][j];
        if constexpr (ZeroCentered) wv += 1.0f;
        out_vals[j] = x_vals[j] * wv * irms;
      }
      *reinterpret_cast<uint4*>(out_row + i * ELEMS) = FloatsToRaw8<T>(out_vals);
    }
  }
}

constexpr int kWideElems = 8;          // 16-bit scalars per 16B vector

inline int64_t WideRoundUp(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

// Pick vectors-per-thread for the forward wide kernel: the per-thread register
// cache is 16B * VPT, so with __launch_bounds__(1024) up to VPT=4 fits without
// spilling. Returns 0 (-> generic fallback) for larger hidden sizes.
inline int PickWideVPTFwd(int64_t vec_hidden) {
  if (vec_hidden <= 1024) return 1;
  if (vec_hidden <= 2048) return 2;
  if (vec_hidden <= 4096) return 4;
  return 0;
}

// Backward caches x, dy, w and the dw accumulator per thread (~32 floats per
// VPT), so it is capped at 512 threads (128 regs) and VPT<=2 (H <= 8192).
inline int PickWideVPTBwd(int64_t vec_hidden) {
  if (vec_hidden <= 512) return 1;
  if (vec_hidden <= 1024) return 2;
  return 0;
}

inline int64_t WideBlockSize(int64_t vec_hidden, int vpt) {
  return std::min<int64_t>(1024, WideRoundUp((vec_hidden + vpt - 1) / vpt, 32));
}

// Whether the wide 16B path can serve this layout (dtype/VPT checked separately).
inline bool WideLayoutOK(int64_t hidden_size, std::initializer_list<int64_t> row_strides,
                         std::initializer_list<const void*> ptrs) {
  if (hidden_size % kWideElems != 0) return false;
  for (int64_t s : row_strides) {
    if (s % kWideElems != 0) return false;
  }
  for (const void* p : ptrs) {
    if (!HostIsAligned16(p)) return false;
  }
  return true;
}

template<typename T, typename WeightDType, bool ZeroCentered>
void LaunchRMSNormForwardWideKernel(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView weight,
    float* rstd_output,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t input_row_stride,
    int64_t output_row_stride,
    cudaStream_t stream) {
  const int64_t vec_hidden = hidden_size / kWideElems;
  const int vpt = PickWideVPTFwd(vec_hidden);
  const int64_t block_size = WideBlockSize(vec_hidden, vpt);
  T* out_ptr = static_cast<T*>(output.data_ptr());
  const T* in_ptr = static_cast<const T*>(input.data_ptr());
  const WeightDType* w_ptr = static_cast<const WeightDType*>(weight.data_ptr());
  switch (vpt) {
    case 1:
      rmsnormForwardWideKernel<T, WeightDType, 1, ZeroCentered>
          <<<num_tokens, block_size, 0, stream>>>(
              out_ptr, in_ptr, w_ptr, rstd_output, epsilon, hidden_size,
              input_row_stride, output_row_stride);
      break;
    case 2:
      rmsnormForwardWideKernel<T, WeightDType, 2, ZeroCentered>
          <<<num_tokens, block_size, 0, stream>>>(
              out_ptr, in_ptr, w_ptr, rstd_output, epsilon, hidden_size,
              input_row_stride, output_row_stride);
      break;
    case 4:
      rmsnormForwardWideKernel<T, WeightDType, 4, ZeroCentered>
          <<<num_tokens, block_size, 0, stream>>>(
              out_ptr, in_ptr, w_ptr, rstd_output, epsilon, hidden_size,
              input_row_stride, output_row_stride);
      break;
    default:
      throw std::runtime_error("RMSNorm wide path: unsupported VPT");
  }
}

// Try dispatching the forward wide path; returns false if the dtype/layout
// combination is not covered (caller falls back to the generic kernel).
template<bool ZeroCentered>
bool TryRMSNormForwardWide(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView weight,
    float* rstd_output,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t input_row_stride,
    int64_t output_row_stride,
    cudaStream_t stream) {
  if (!WideLayoutOK(hidden_size, {input_row_stride, output_row_stride},
                    {input.data_ptr(), output.data_ptr(), weight.data_ptr()})) {
    return false;
  }
  if (PickWideVPTFwd(hidden_size / kWideElems) == 0) return false;
  if (IsBFloat16(input.dtype()) && IsBFloat16(weight.dtype())) {
    LaunchRMSNormForwardWideKernel<__nv_bfloat16, __nv_bfloat16, ZeroCentered>(
        output, input, weight, rstd_output, epsilon, hidden_size, num_tokens,
        input_row_stride, output_row_stride, stream);
  } else if (IsBFloat16(input.dtype()) && IsFloat32(weight.dtype())) {
    LaunchRMSNormForwardWideKernel<__nv_bfloat16, float, ZeroCentered>(
        output, input, weight, rstd_output, epsilon, hidden_size, num_tokens,
        input_row_stride, output_row_stride, stream);
  } else if (IsFloat16(input.dtype()) && IsFloat16(weight.dtype())) {
    LaunchRMSNormForwardWideKernel<half, half, ZeroCentered>(
        output, input, weight, rstd_output, epsilon, hidden_size, num_tokens,
        input_row_stride, output_row_stride, stream);
  } else if (IsFloat16(input.dtype()) && IsFloat32(weight.dtype())) {
    LaunchRMSNormForwardWideKernel<half, float, ZeroCentered>(
        output, input, weight, rstd_output, epsilon, hidden_size, num_tokens,
        input_row_stride, output_row_stride, stream);
  } else if (IsFloat16(input.dtype()) && IsBFloat16(weight.dtype())) {
    LaunchRMSNormForwardWideKernel<half, __nv_bfloat16, ZeroCentered>(
        output, input, weight, rstd_output, epsilon, hidden_size, num_tokens,
        input_row_stride, output_row_stride, stream);
  } else {
    return false;
  }
  return true;
}

template<typename T, typename WeightDType, int VPT, bool ZeroCentered>
__global__ void __launch_bounds__(1024)
addRMSNormForwardWideKernel(
    T* __restrict__ sum_output, T* __restrict__ norm_output,
    const T* __restrict__ residual, const T* __restrict__ input,
    const WeightDType* __restrict__ weight,
    float* __restrict__ rstd_output,
    const float epsilon, const int64_t hidden_size,
    const int64_t residual_row_stride, const int64_t input_row_stride,
    const int64_t sum_row_stride, const int64_t norm_row_stride);

template<typename T, typename WeightDType, bool ZeroCentered>
void LaunchAddRMSNormForwardWideKernel(
    wallx_cuda::TensorView sum_output,
    wallx_cuda::TensorView norm_output,
    wallx_cuda::TensorView residual,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView weight,
    float* rstd_output,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t residual_row_stride,
    int64_t input_row_stride,
    int64_t sum_row_stride,
    int64_t norm_row_stride,
    cudaStream_t stream) {
  const int64_t vec_hidden = hidden_size / kWideElems;
  const int vpt = PickWideVPTFwd(vec_hidden);
  const int64_t block_size = WideBlockSize(vec_hidden, vpt);
  T* sum_ptr = static_cast<T*>(sum_output.data_ptr());
  T* norm_ptr = static_cast<T*>(norm_output.data_ptr());
  const T* res_ptr = static_cast<const T*>(residual.data_ptr());
  const T* in_ptr = static_cast<const T*>(input.data_ptr());
  const WeightDType* w_ptr = static_cast<const WeightDType*>(weight.data_ptr());
  switch (vpt) {
    case 1:
      addRMSNormForwardWideKernel<T, WeightDType, 1, ZeroCentered>
          <<<num_tokens, block_size, 0, stream>>>(
              sum_ptr, norm_ptr, res_ptr, in_ptr, w_ptr, rstd_output, epsilon,
              hidden_size, residual_row_stride, input_row_stride,
              sum_row_stride, norm_row_stride);
      break;
    case 2:
      addRMSNormForwardWideKernel<T, WeightDType, 2, ZeroCentered>
          <<<num_tokens, block_size, 0, stream>>>(
              sum_ptr, norm_ptr, res_ptr, in_ptr, w_ptr, rstd_output, epsilon,
              hidden_size, residual_row_stride, input_row_stride,
              sum_row_stride, norm_row_stride);
      break;
    case 4:
      addRMSNormForwardWideKernel<T, WeightDType, 4, ZeroCentered>
          <<<num_tokens, block_size, 0, stream>>>(
              sum_ptr, norm_ptr, res_ptr, in_ptr, w_ptr, rstd_output, epsilon,
              hidden_size, residual_row_stride, input_row_stride,
              sum_row_stride, norm_row_stride);
      break;
    default:
      throw std::runtime_error("AddRMSNorm wide path: unsupported VPT");
  }
}

template<bool ZeroCentered>
bool TryAddRMSNormForwardWide(
    wallx_cuda::TensorView sum_output,
    wallx_cuda::TensorView norm_output,
    wallx_cuda::TensorView residual,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView weight,
    float* rstd_output,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t residual_row_stride,
    int64_t input_row_stride,
    int64_t sum_row_stride,
    int64_t norm_row_stride,
    cudaStream_t stream) {
  if (!WideLayoutOK(hidden_size,
                    {residual_row_stride, input_row_stride, sum_row_stride,
                     norm_row_stride},
                    {residual.data_ptr(), input.data_ptr(),
                     sum_output.data_ptr(), norm_output.data_ptr(),
                     weight.data_ptr()})) {
    return false;
  }
  if (PickWideVPTFwd(hidden_size / kWideElems) == 0) return false;
  if (IsBFloat16(input.dtype()) && IsBFloat16(weight.dtype())) {
    LaunchAddRMSNormForwardWideKernel<__nv_bfloat16, __nv_bfloat16, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_output, epsilon,
        hidden_size, num_tokens, residual_row_stride, input_row_stride,
        sum_row_stride, norm_row_stride, stream);
  } else if (IsBFloat16(input.dtype()) && IsFloat32(weight.dtype())) {
    LaunchAddRMSNormForwardWideKernel<__nv_bfloat16, float, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_output, epsilon,
        hidden_size, num_tokens, residual_row_stride, input_row_stride,
        sum_row_stride, norm_row_stride, stream);
  } else if (IsFloat16(input.dtype()) && IsFloat16(weight.dtype())) {
    LaunchAddRMSNormForwardWideKernel<half, half, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_output, epsilon,
        hidden_size, num_tokens, residual_row_stride, input_row_stride,
        sum_row_stride, norm_row_stride, stream);
  } else if (IsFloat16(input.dtype()) && IsFloat32(weight.dtype())) {
    LaunchAddRMSNormForwardWideKernel<half, float, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_output, epsilon,
        hidden_size, num_tokens, residual_row_stride, input_row_stride,
        sum_row_stride, norm_row_stride, stream);
  } else if (IsFloat16(input.dtype()) && IsBFloat16(weight.dtype())) {
    LaunchAddRMSNormForwardWideKernel<half, __nv_bfloat16, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_output, epsilon,
        hidden_size, num_tokens, residual_row_stride, input_row_stride,
        sum_row_stride, norm_row_stride, stream);
  } else {
    return false;
  }
  return true;
}

// Wide forward for the fused add+rmsnorm. Deliberately mirrors
// rmsnormForwardWideKernel's block configuration and accumulation order so the
// fused op stays BITWISE identical to the materialized path
// (sum = round(residual+input); rmsnorm(sum)) — tests assert this coupling.
template<typename T, typename WeightDType, int VPT, bool ZeroCentered>
__global__ void __launch_bounds__(1024)
addRMSNormForwardWideKernel(
    T* __restrict__ sum_output, T* __restrict__ norm_output,
    const T* __restrict__ residual, const T* __restrict__ input,
    const WeightDType* __restrict__ weight,
    float* __restrict__ rstd_output,
    const float epsilon, const int64_t hidden_size,
    const int64_t residual_row_stride, const int64_t input_row_stride,
    const int64_t sum_row_stride, const int64_t norm_row_stride) {
  constexpr int ELEMS = 8;
  static_assert(!std::is_same_v<T, float>, "wide path is for 16-bit tensors");
  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x;
  const T* res_row = residual + row * residual_row_stride;
  const T* in_row = input + row * input_row_stride;
  T* sum_row = sum_output + row * sum_row_stride;
  T* norm_row = norm_output + row * norm_row_stride;
  const int64_t vec_hidden = hidden_size / ELEMS;

  uint4 s_raw[VPT];
  float w_cache[VPT][ELEMS];
  float sq = 0.0f;
  #pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int64_t i = tid + k * static_cast<int64_t>(blockDim.x);
    if (i < vec_hidden) {
      const uint4 r_raw = *reinterpret_cast<const uint4*>(res_row + i * ELEMS);
      const uint4 x_raw = *reinterpret_cast<const uint4*>(in_row + i * ELEMS);
      LoadWeight8Floats<WeightDType>(weight, i * ELEMS, w_cache[k]);
      float r_vals[ELEMS], x_vals[ELEMS], s_vals[ELEMS];
      Raw8ToFloats<T>(r_raw, r_vals);
      Raw8ToFloats<T>(x_raw, x_vals);
      #pragma unroll
      for (int j = 0; j < ELEMS; ++j) {
        // Match the materialized path: the bf16/fp16 add produces a rounded
        // sum tensor, and rmsnorm consumes that rounded tensor.
        s_vals[j] = RoundFloatToScalarType<T>(r_vals[j] + x_vals[j]);
        sq += s_vals[j] * s_vals[j];
      }
      s_raw[k] = FloatsToRaw8<T>(s_vals);
      *reinterpret_cast<uint4*>(sum_row + i * ELEMS) = s_raw[k];
    }
  }
  sq = blockReduceSum(sq);
  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = rsqrtf(sq / static_cast<float>(hidden_size) + epsilon);
    if (rstd_output != nullptr) {
      rstd_output[row] = s_inv_rms;
    }
  }
  __syncthreads();
  const float irms = s_inv_rms;

  #pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int64_t i = tid + k * static_cast<int64_t>(blockDim.x);
    if (i < vec_hidden) {
      float s_vals[ELEMS], out_vals[ELEMS];
      Raw8ToFloats<T>(s_raw[k], s_vals);
      #pragma unroll
      for (int j = 0; j < ELEMS; ++j) {
        float wv = w_cache[k][j];
        if constexpr (ZeroCentered) wv += 1.0f;
        out_vals[j] = s_vals[j] * wv * irms;
      }
      *reinterpret_cast<uint4*>(norm_row + i * ELEMS) = FloatsToRaw8<T>(out_vals);
    }
  }
}

template<typename T, typename WeightDType, bool ZeroCentered>
__global__ void addRMSNormKernel(
    T* sum_output,
    T* norm_output,
    const T* residual,
    const T* input,
    const WeightDType* weight,
    float* rstd_output,
    const float epsilon,
    const int64_t hidden_size,
    const int64_t residual_row_stride,
    const int64_t input_row_stride,
    const int64_t sum_row_stride,
    const int64_t norm_row_stride) {
  using Vec = typename VecTraits<T>::Type;
  constexpr int VEC_SIZE = VecTraits<T>::vec_size;
  float square_sum = 0.0f;
  __shared__ float inv_rms;
  const int64_t tid = threadIdx.x;
  const int64_t bid = blockIdx.x;
  const int64_t residual_offset = bid * residual_row_stride;
  const int64_t input_offset = bid * input_row_stride;
  const int64_t sum_offset = bid * sum_row_stride;
  const int64_t norm_offset = bid * norm_row_stride;
  const int64_t vec_hidden = hidden_size / VEC_SIZE;

  for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
    Vec residual_vec = VecTraits<T>::load(residual + residual_offset + i * VEC_SIZE);
    Vec input_vec = VecTraits<T>::load(input + input_offset + i * VEC_SIZE);
    float residual_vals[VEC_SIZE], input_vals[VEC_SIZE], sum_vals[VEC_SIZE];
    VecToFloats<T>(residual_vec, residual_vals);
    VecToFloats<T>(input_vec, input_vals);
    #pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      // Match the old path: bf16/fp16 add materializes a rounded sum tensor,
      // and RMSNorm consumes that rounded tensor.
      sum_vals[j] = RoundFloatToScalarType<T>(residual_vals[j] + input_vals[j]);
      square_sum += sum_vals[j] * sum_vals[j];
    }
    VecTraits<T>::store(sum_output + sum_offset + i * VEC_SIZE,
                        FloatsToVec<T>(sum_vals));
  }
  square_sum = blockReduceSum(square_sum);
  if (tid == 0) {
    inv_rms = rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
    if (rstd_output != nullptr) {
      rstd_output[bid] = inv_rms;
    }
  }
  __syncthreads();

  for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
    Vec sum_vec = VecTraits<T>::load(sum_output + sum_offset + i * VEC_SIZE);
    float sum_vals[VEC_SIZE];
    float w_vals[VEC_SIZE], out_vals[VEC_SIZE];
    VecToFloats<T>(sum_vec, sum_vals);
    LoadWeightFloats<WeightDType, VEC_SIZE>(weight, i * VEC_SIZE, w_vals);
    #pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      if constexpr (ZeroCentered) w_vals[j] += 1.0f;
      out_vals[j] = sum_vals[j] * w_vals[j] * inv_rms;
    }
    VecTraits<T>::store(norm_output + norm_offset + i * VEC_SIZE,
                        FloatsToVec<T>(out_vals));
  }
}

template<typename T, typename WeightDType, bool ZeroCentered>
void LaunchAddRMSNormKernel(
    wallx_cuda::TensorView sum_output,
    wallx_cuda::TensorView norm_output,
    wallx_cuda::TensorView residual,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView weight,
    float* rstd_output,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t residual_row_stride,
    int64_t input_row_stride,
    int64_t sum_row_stride,
    int64_t norm_row_stride,
    int64_t block_size,
    cudaStream_t stream) {
  addRMSNormKernel<T, WeightDType, ZeroCentered><<<num_tokens, block_size, 0, stream>>>(
      static_cast<T*>(sum_output.data_ptr()),
      static_cast<T*>(norm_output.data_ptr()),
      static_cast<const T*>(residual.data_ptr()),
      static_cast<const T*>(input.data_ptr()),
      static_cast<const WeightDType*>(weight.data_ptr()),
      rstd_output,
      epsilon, hidden_size, residual_row_stride, input_row_stride,
      sum_row_stride, norm_row_stride);
}

template<bool ZeroCentered>
void RMSNormImpl(wallx_cuda::TensorView output, wallx_cuda::TensorView input,
                 wallx_cuda::TensorView weight, double epsilon,
                 float* rstd_ptr = nullptr) {
  ASSERT_CHECK(weight.ndim() == 1);
  ASSERT_CHECK(weight.IsContiguous());
  CheckSameShape(output, input);
  const RMSNormShape shape = GetRMSNormShape(input);
  const RMSNormShape output_shape = GetRMSNormShape(output);
  const int64_t hidden_size = shape.hidden_size;
  const int64_t num_tokens = shape.num_tokens;
  ASSERT_CHECK(output_shape.hidden_size == hidden_size);
  ASSERT_CHECK(output_shape.num_tokens == num_tokens);
  ASSERT_CHECK(hidden_size == weight.size(0));
  ASSERT_CHECK(hidden_size % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(output_shape.row_stride % GetVecSize(input.dtype()) == 0);
  cudaStream_t stream = static_cast<cudaStream_t>(
      wallx_cuda::CurrentStream(input.device().device_type, input.device().device_id));
  if (TryRMSNormForwardWide<ZeroCentered>(
          output, input, weight, rstd_ptr, static_cast<float>(epsilon),
          hidden_size, num_tokens, shape.row_stride, output_shape.row_stride,
          stream)) {
    sync_check_cuda_error();
    return;
  }
  const int64_t block_size = std::min(hidden_size, 1024L);
  if (IsFloat32(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchRMSNormKernel<float, float, ZeroCentered>(
        output, input, weight, rstd_ptr, static_cast<float>(epsilon), hidden_size, num_tokens,
        shape.row_stride, output_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat16(weight.dtype()))
    LaunchRMSNormKernel<half, half, ZeroCentered>(
        output, input, weight, rstd_ptr, static_cast<float>(epsilon), hidden_size, num_tokens,
        shape.row_stride, output_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchRMSNormKernel<half, float, ZeroCentered>(
        output, input, weight, rstd_ptr, static_cast<float>(epsilon), hidden_size, num_tokens,
        shape.row_stride, output_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchRMSNormKernel<half, __nv_bfloat16, ZeroCentered>(
        output, input, weight, rstd_ptr, static_cast<float>(epsilon), hidden_size, num_tokens,
        shape.row_stride, output_shape.row_stride, block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchRMSNormKernel<__nv_bfloat16, __nv_bfloat16, ZeroCentered>(
        output, input, weight, rstd_ptr, static_cast<float>(epsilon), hidden_size, num_tokens,
        shape.row_stride, output_shape.row_stride, block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchRMSNormKernel<__nv_bfloat16, float, ZeroCentered>(
        output, input, weight, rstd_ptr, static_cast<float>(epsilon), hidden_size, num_tokens,
        shape.row_stride, output_shape.row_stride, block_size, stream);
  else throw std::runtime_error("Unsupported data type for RMSNorm");
  sync_check_cuda_error();
}

template<bool ZeroCentered>
void AddRMSNormImpl(wallx_cuda::TensorView sum_output,
                    wallx_cuda::TensorView norm_output,
                    wallx_cuda::TensorView residual,
                    wallx_cuda::TensorView input,
                    wallx_cuda::TensorView weight,
                    double epsilon) {
  ASSERT_CHECK(weight.ndim() == 1);
  ASSERT_CHECK(weight.IsContiguous());
  CheckSameShape(sum_output, input);
  CheckSameShape(norm_output, input);
  CheckSameShape(residual, input);
  const RMSNormShape input_shape = GetRMSNormShape(input);
  const RMSNormShape residual_shape = GetRMSNormShape(residual);
  const RMSNormShape sum_shape = GetRMSNormShape(sum_output);
  const RMSNormShape norm_shape = GetRMSNormShape(norm_output);
  const int64_t hidden_size = input_shape.hidden_size;
  const int64_t num_tokens = input_shape.num_tokens;
  ASSERT_CHECK(residual_shape.hidden_size == hidden_size);
  ASSERT_CHECK(residual_shape.num_tokens == num_tokens);
  ASSERT_CHECK(sum_shape.hidden_size == hidden_size);
  ASSERT_CHECK(sum_shape.num_tokens == num_tokens);
  ASSERT_CHECK(norm_shape.hidden_size == hidden_size);
  ASSERT_CHECK(norm_shape.num_tokens == num_tokens);
  ASSERT_CHECK(hidden_size == weight.size(0));
  ASSERT_CHECK(input.dtype() == residual.dtype());
  ASSERT_CHECK(input.dtype() == sum_output.dtype());
  ASSERT_CHECK(input.dtype() == norm_output.dtype());
  ASSERT_CHECK(hidden_size % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(input_shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(residual_shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(sum_shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(norm_shape.row_stride % GetVecSize(input.dtype()) == 0);

  cudaStream_t stream = static_cast<cudaStream_t>(
      wallx_cuda::CurrentStream(input.device().device_type, input.device().device_id));
  if (TryAddRMSNormForwardWide<ZeroCentered>(
          sum_output, norm_output, residual, input, weight, nullptr,
          static_cast<float>(epsilon), hidden_size, num_tokens,
          residual_shape.row_stride, input_shape.row_stride,
          sum_shape.row_stride, norm_shape.row_stride, stream)) {
    sync_check_cuda_error();
    return;
  }
  const int64_t block_size = std::min(hidden_size, 1024L);
  if (IsFloat32(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchAddRMSNormKernel<float, float, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, nullptr, static_cast<float>(epsilon),
        hidden_size, num_tokens, residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat16(weight.dtype()))
    LaunchAddRMSNormKernel<half, half, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, nullptr, static_cast<float>(epsilon),
        hidden_size, num_tokens, residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchAddRMSNormKernel<half, float, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, nullptr, static_cast<float>(epsilon),
        hidden_size, num_tokens, residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchAddRMSNormKernel<half, __nv_bfloat16, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, nullptr, static_cast<float>(epsilon),
        hidden_size, num_tokens, residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchAddRMSNormKernel<__nv_bfloat16, __nv_bfloat16, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, nullptr, static_cast<float>(epsilon),
        hidden_size, num_tokens, residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchAddRMSNormKernel<__nv_bfloat16, float, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, nullptr, static_cast<float>(epsilon),
        hidden_size, num_tokens, residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else throw std::runtime_error("Unsupported data type for AddRMSNorm");
  sync_check_cuda_error();
}

void RMSNorm(wallx_cuda::TensorView output, wallx_cuda::TensorView input,
             wallx_cuda::TensorView weight, double epsilon) {
  RMSNormImpl<false>(output, input, weight, epsilon);
}

void RMSNormZeroCentered(wallx_cuda::TensorView output, wallx_cuda::TensorView input,
                         wallx_cuda::TensorView weight, double epsilon) {
  RMSNormImpl<true>(output, input, weight, epsilon);
}

// Forward that additionally materializes per-row rstd (1/rms), so the plain
// backward can skip recomputing the sum(x^2) reduction. Mirrors the
// AddRMSNorm*WithRstd pattern.
template<bool ZeroCentered>
void RMSNormWithRstdImpl(wallx_cuda::TensorView output,
                         wallx_cuda::TensorView rstd_output,
                         wallx_cuda::TensorView input,
                         wallx_cuda::TensorView weight, double epsilon) {
  ASSERT_CHECK(rstd_output.dtype().code == kDLFloat &&
               rstd_output.dtype().bits == 32);
  ASSERT_CHECK(rstd_output.IsContiguous());
  ASSERT_CHECK(rstd_output.ndim() == 1);
  ASSERT_CHECK(rstd_output.size(0) == GetTensorNumTokens(input));
  RMSNormImpl<ZeroCentered>(output, input, weight, epsilon,
                            static_cast<float*>(rstd_output.data_ptr()));
}

void RMSNormWithRstd(wallx_cuda::TensorView output,
                     wallx_cuda::TensorView rstd_output,
                     wallx_cuda::TensorView input,
                     wallx_cuda::TensorView weight, double epsilon) {
  RMSNormWithRstdImpl<false>(output, rstd_output, input, weight, epsilon);
}

void RMSNormZeroCenteredWithRstd(wallx_cuda::TensorView output,
                                 wallx_cuda::TensorView rstd_output,
                                 wallx_cuda::TensorView input,
                                 wallx_cuda::TensorView weight, double epsilon) {
  RMSNormWithRstdImpl<true>(output, rstd_output, input, weight, epsilon);
}

void AddRMSNorm(wallx_cuda::TensorView sum_output,
                wallx_cuda::TensorView norm_output,
                wallx_cuda::TensorView residual,
                wallx_cuda::TensorView input,
                wallx_cuda::TensorView weight,
                double epsilon) {
  AddRMSNormImpl<false>(sum_output, norm_output, residual, input, weight, epsilon);
}

void AddRMSNormZeroCentered(wallx_cuda::TensorView sum_output,
                            wallx_cuda::TensorView norm_output,
                            wallx_cuda::TensorView residual,
                            wallx_cuda::TensorView input,
                            wallx_cuda::TensorView weight,
                            double epsilon) {
  AddRMSNormImpl<true>(sum_output, norm_output, residual, input, weight, epsilon);
}

template<bool ZeroCentered>
void AddRMSNormWithRstdImpl(wallx_cuda::TensorView sum_output,
                            wallx_cuda::TensorView norm_output,
                            wallx_cuda::TensorView rstd_output,
                            wallx_cuda::TensorView residual,
                            wallx_cuda::TensorView input,
                            wallx_cuda::TensorView weight,
                            double epsilon) {
  ASSERT_CHECK(rstd_output.dtype().code == kDLFloat && rstd_output.dtype().bits == 32);
  ASSERT_CHECK(rstd_output.IsContiguous());
  ASSERT_CHECK(weight.ndim() == 1);
  ASSERT_CHECK(weight.IsContiguous());
  CheckSameShape(sum_output, input);
  CheckSameShape(norm_output, input);
  CheckSameShape(residual, input);
  const RMSNormShape input_shape = GetRMSNormShape(input);
  const RMSNormShape residual_shape = GetRMSNormShape(residual);
  const RMSNormShape sum_shape = GetRMSNormShape(sum_output);
  const RMSNormShape norm_shape = GetRMSNormShape(norm_output);
  const int64_t hidden_size = input_shape.hidden_size;
  const int64_t num_tokens = input_shape.num_tokens;
  ASSERT_CHECK(rstd_output.ndim() == 1);
  ASSERT_CHECK(rstd_output.size(0) == num_tokens);
  ASSERT_CHECK(residual_shape.hidden_size == hidden_size);
  ASSERT_CHECK(residual_shape.num_tokens == num_tokens);
  ASSERT_CHECK(sum_shape.hidden_size == hidden_size);
  ASSERT_CHECK(sum_shape.num_tokens == num_tokens);
  ASSERT_CHECK(norm_shape.hidden_size == hidden_size);
  ASSERT_CHECK(norm_shape.num_tokens == num_tokens);
  ASSERT_CHECK(hidden_size == weight.size(0));
  ASSERT_CHECK(input.dtype() == residual.dtype());
  ASSERT_CHECK(input.dtype() == sum_output.dtype());
  ASSERT_CHECK(input.dtype() == norm_output.dtype());
  ASSERT_CHECK(hidden_size % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(input_shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(residual_shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(sum_shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(norm_shape.row_stride % GetVecSize(input.dtype()) == 0);

  cudaStream_t stream = static_cast<cudaStream_t>(
      wallx_cuda::CurrentStream(input.device().device_type, input.device().device_id));
  float* rstd_ptr = static_cast<float*>(rstd_output.data_ptr());
  if (TryAddRMSNormForwardWide<ZeroCentered>(
          sum_output, norm_output, residual, input, weight, rstd_ptr,
          static_cast<float>(epsilon), hidden_size, num_tokens,
          residual_shape.row_stride, input_shape.row_stride,
          sum_shape.row_stride, norm_shape.row_stride, stream)) {
    sync_check_cuda_error();
    return;
  }
  const int64_t block_size = std::min(hidden_size, 1024L);
  if (IsFloat32(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchAddRMSNormKernel<float, float, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_ptr,
        static_cast<float>(epsilon), hidden_size, num_tokens,
        residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat16(weight.dtype()))
    LaunchAddRMSNormKernel<half, half, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_ptr,
        static_cast<float>(epsilon), hidden_size, num_tokens,
        residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchAddRMSNormKernel<half, float, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_ptr,
        static_cast<float>(epsilon), hidden_size, num_tokens,
        residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchAddRMSNormKernel<half, __nv_bfloat16, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_ptr,
        static_cast<float>(epsilon), hidden_size, num_tokens,
        residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchAddRMSNormKernel<__nv_bfloat16, __nv_bfloat16, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_ptr,
        static_cast<float>(epsilon), hidden_size, num_tokens,
        residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchAddRMSNormKernel<__nv_bfloat16, float, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_ptr,
        static_cast<float>(epsilon), hidden_size, num_tokens,
        residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else throw std::runtime_error("Unsupported data type for AddRMSNorm with rstd");
  sync_check_cuda_error();
}

void AddRMSNormWithRstd(wallx_cuda::TensorView sum_output,
                        wallx_cuda::TensorView norm_output,
                        wallx_cuda::TensorView rstd_output,
                        wallx_cuda::TensorView residual,
                        wallx_cuda::TensorView input,
                        wallx_cuda::TensorView weight,
                        double epsilon) {
  AddRMSNormWithRstdImpl<false>(
      sum_output, norm_output, rstd_output, residual, input, weight, epsilon);
}

void AddRMSNormZeroCenteredWithRstd(wallx_cuda::TensorView sum_output,
                                    wallx_cuda::TensorView norm_output,
                                    wallx_cuda::TensorView rstd_output,
                                    wallx_cuda::TensorView residual,
                                    wallx_cuda::TensorView input,
                                    wallx_cuda::TensorView weight,
                                    double epsilon) {
  AddRMSNormWithRstdImpl<true>(
      sum_output, norm_output, rstd_output, residual, input, weight, epsilon);
}










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

template<typename T, typename WeightDType, int MAX_VEC_PER_THREAD, bool ZeroCentered>
__global__ void rmsnormBackwardPersistentKernel(
    T* __restrict__ dx,
    float* __restrict__ partial_dw,
    const T* __restrict__ dy,
    const T* __restrict__ input,
    const WeightDType* __restrict__ weight,
    const float epsilon,
    const int64_t hidden_size,
    const int64_t num_tokens,
    const int64_t dx_row_stride,
    const int64_t dy_row_stride,
    const int64_t input_row_stride
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
    float local_dw[MAX_VEC_PER_THREAD * VEC_SIZE];
    #pragma unroll
    for (int j = 0; j < MAX_VEC_PER_THREAD * VEC_SIZE; j++) local_dw[j] = 0.0f;

    // Persistent loop: each block processes multiple rows
    for (int64_t row = blockIdx.x; row < num_tokens; row += num_blocks) {
        const int64_t dx_off = row * dx_row_stride;
        const int64_t dy_off = row * dy_row_stride;
        const int64_t input_off = row * input_row_stride;
        float sq = 0.0f, dp = 0.0f;

        // Pass 1: reductions
        for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
            Vec xv = VecTraits<T>::load(input + input_off + i * VEC_SIZE);
            Vec dv = VecTraits<T>::load(dy + dy_off + i * VEC_SIZE);
            float x_vals[VEC_SIZE], d_vals[VEC_SIZE], w_vals[VEC_SIZE];
            VecToFloats<T>(xv, x_vals);
            VecToFloats<T>(dv, d_vals);
            LoadWeightFloats<WeightDType, VEC_SIZE>(weight, i * VEC_SIZE, w_vals);
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                if constexpr (ZeroCentered) w_vals[j] += 1.0f;
                sq += x_vals[j] * x_vals[j];
                dp += d_vals[j] * w_vals[j] * x_vals[j];
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
            Vec xv = VecTraits<T>::load(input + input_off + i * VEC_SIZE);
            Vec dv = VecTraits<T>::load(dy + dy_off + i * VEC_SIZE);
            float x_vals[VEC_SIZE], d_vals[VEC_SIZE], w_vals[VEC_SIZE], out_vals[VEC_SIZE];
            VecToFloats<T>(xv, x_vals);
            VecToFloats<T>(dv, d_vals);
            LoadWeightFloats<WeightDType, VEC_SIZE>(weight, i * VEC_SIZE, w_vals);
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                if constexpr (ZeroCentered) w_vals[j] += 1.0f;
                out_vals[j] = irms * (d_vals[j] * w_vals[j] - x_vals[j] * cv);
                local_dw[li * VEC_SIZE + j] += d_vals[j] * x_vals[j] * irms;
            }
            VecTraits<T>::store(dx + dx_off + i * VEC_SIZE, FloatsToVec<T>(out_vals));
        }
    }

    // Write partial dw to global memory
    const int64_t poff = blockIdx.x * hidden_size;
    int li = 0;
    for (int64_t i = tid; i < vec_hidden; i += blockDim.x, li++) {
        if constexpr (VEC_SIZE == 4) {
            float4 dw_out;
            dw_out.x = local_dw[li * VEC_SIZE + 0];
            dw_out.y = local_dw[li * VEC_SIZE + 1];
            dw_out.z = local_dw[li * VEC_SIZE + 2];
            dw_out.w = local_dw[li * VEC_SIZE + 3];
            *reinterpret_cast<float4*>(partial_dw + poff + i * VEC_SIZE) = dw_out;
        } else {
            float2 dw_out;
            dw_out.x = local_dw[li * VEC_SIZE + 0];
            dw_out.y = local_dw[li * VEC_SIZE + 1];
            *reinterpret_cast<float2*>(partial_dw + poff + i * VEC_SIZE) = dw_out;
        }
    }
}

template<typename T>
__device__ __forceinline__ float AddGradSumLikeMaterializedAdd(
    float grad_sum, float rms_dx) {
  const float rounded_rms_dx = RoundFloatToScalarType<T>(rms_dx);
  return RoundFloatToScalarType<T>(grad_sum + rounded_rms_dx);
}

template<typename T, typename WeightDType, int MAX_VEC_PER_THREAD, bool ZeroCentered>
__global__ void addRMSNormBackwardPersistentKernel(
    T* __restrict__ d_add,
    float* __restrict__ partial_dw,
    const T* __restrict__ grad_sum,
    const T* __restrict__ grad_norm,
    const T* __restrict__ sum_input,
    const WeightDType* __restrict__ weight,
    const float epsilon,
    const int64_t hidden_size,
    const int64_t num_tokens,
    const int64_t d_add_row_stride,
    const int64_t grad_sum_row_stride,
    const int64_t grad_norm_row_stride,
    const int64_t sum_row_stride
) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;

    const int64_t tid = threadIdx.x;
    const int64_t num_blocks = gridDim.x;
    const int64_t vec_hidden = hidden_size / VEC_SIZE;
    const float inv_hidden = 1.0f / static_cast<float>(hidden_size);

    __shared__ float s_inv_rms;
    __shared__ float s_c;

    float local_dw[MAX_VEC_PER_THREAD * VEC_SIZE];
    #pragma unroll
    for (int j = 0; j < MAX_VEC_PER_THREAD * VEC_SIZE; ++j) local_dw[j] = 0.0f;

    for (int64_t row = blockIdx.x; row < num_tokens; row += num_blocks) {
        const int64_t d_add_off = row * d_add_row_stride;
        const int64_t grad_sum_off = row * grad_sum_row_stride;
        const int64_t grad_norm_off = row * grad_norm_row_stride;
        const int64_t sum_off = row * sum_row_stride;
        float sq = 0.0f, dp = 0.0f;

        for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
            Vec xv = VecTraits<T>::load(sum_input + sum_off + i * VEC_SIZE);
            Vec dv = VecTraits<T>::load(grad_norm + grad_norm_off + i * VEC_SIZE);
            float x_vals[VEC_SIZE], d_vals[VEC_SIZE], w_vals[VEC_SIZE];
            VecToFloats<T>(xv, x_vals);
            VecToFloats<T>(dv, d_vals);
            LoadWeightFloats<WeightDType, VEC_SIZE>(weight, i * VEC_SIZE, w_vals);
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                if constexpr (ZeroCentered) w_vals[j] += 1.0f;
                sq += x_vals[j] * x_vals[j];
                dp += d_vals[j] * w_vals[j] * x_vals[j];
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

        int li = 0;
        for (int64_t i = tid; i < vec_hidden; i += blockDim.x, ++li) {
            Vec xv = VecTraits<T>::load(sum_input + sum_off + i * VEC_SIZE);
            Vec dv = VecTraits<T>::load(grad_norm + grad_norm_off + i * VEC_SIZE);
            Vec gs = VecTraits<T>::load(grad_sum + grad_sum_off + i * VEC_SIZE);
            float x_vals[VEC_SIZE], d_vals[VEC_SIZE], gs_vals[VEC_SIZE];
            float w_vals[VEC_SIZE], out_vals[VEC_SIZE];
            VecToFloats<T>(xv, x_vals);
            VecToFloats<T>(dv, d_vals);
            VecToFloats<T>(gs, gs_vals);
            LoadWeightFloats<WeightDType, VEC_SIZE>(weight, i * VEC_SIZE, w_vals);
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                if constexpr (ZeroCentered) w_vals[j] += 1.0f;
                const float rms_dx =
                    irms * (d_vals[j] * w_vals[j] - x_vals[j] * cv);
                out_vals[j] =
                    AddGradSumLikeMaterializedAdd<T>(gs_vals[j], rms_dx);
                local_dw[li * VEC_SIZE + j] += d_vals[j] * x_vals[j] * irms;
            }
            VecTraits<T>::store(d_add + d_add_off + i * VEC_SIZE,
                                FloatsToVec<T>(out_vals));
        }
    }

    const int64_t poff = blockIdx.x * hidden_size;
    int li = 0;
    for (int64_t i = tid; i < vec_hidden; i += blockDim.x, ++li) {
        if constexpr (VEC_SIZE == 4) {
            float4 dw_out;
            dw_out.x = local_dw[li * VEC_SIZE + 0];
            dw_out.y = local_dw[li * VEC_SIZE + 1];
            dw_out.z = local_dw[li * VEC_SIZE + 2];
            dw_out.w = local_dw[li * VEC_SIZE + 3];
            *reinterpret_cast<float4*>(partial_dw + poff + i * VEC_SIZE) = dw_out;
        } else {
            float2 dw_out;
            dw_out.x = local_dw[li * VEC_SIZE + 0];
            dw_out.y = local_dw[li * VEC_SIZE + 1];
            *reinterpret_cast<float2*>(partial_dw + poff + i * VEC_SIZE) = dw_out;
        }
    }
}

template<
    typename T,
    typename WeightDType,
    int MAX_VEC_PER_THREAD,
    bool ZeroCentered,
    bool FuseGradSum>
__global__ void rmsnormBackwardCachedKernel(
    T* __restrict__ dx,
    float* __restrict__ partial_dw,
    const T* __restrict__ dy,
    const T* __restrict__ input,
    const WeightDType* __restrict__ weight,
    const T* __restrict__ grad_sum,
    const float epsilon,
    const int64_t hidden_size,
    const int64_t num_tokens,
    const int64_t dx_row_stride,
    const int64_t dy_row_stride,
    const int64_t input_row_stride,
    const int64_t grad_sum_row_stride
) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;

    const int64_t tid = threadIdx.x;
    const int64_t num_blocks = gridDim.x;
    const int64_t vec_hidden = hidden_size / VEC_SIZE;
    const float inv_hidden = 1.0f / static_cast<float>(hidden_size);

    __shared__ float s_inv_rms;
    __shared__ float s_c;

    float local_dw[MAX_VEC_PER_THREAD * VEC_SIZE];
    #pragma unroll
    for (int j = 0; j < MAX_VEC_PER_THREAD * VEC_SIZE; ++j) {
      local_dw[j] = 0.0f;
    }

    for (int64_t row = blockIdx.x; row < num_tokens; row += num_blocks) {
        const int64_t dx_off = row * dx_row_stride;
        const int64_t dy_off = row * dy_row_stride;
        const int64_t input_off = row * input_row_stride;
        const int64_t grad_sum_off = row * grad_sum_row_stride;
        float sq = 0.0f, dp = 0.0f;

        float x_cache[MAX_VEC_PER_THREAD * VEC_SIZE];
        float d_cache[MAX_VEC_PER_THREAD * VEC_SIZE];
        float w_cache[MAX_VEC_PER_THREAD * VEC_SIZE];
        int cache_count = 0;

        for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
            float x_vals[VEC_SIZE], d_vals[VEC_SIZE], w_vals[VEC_SIZE];
            Vec xv = VecTraits<T>::load(input + input_off + i * VEC_SIZE);
            Vec dv = VecTraits<T>::load(dy + dy_off + i * VEC_SIZE);
            VecToFloats<T>(xv, x_vals);
            VecToFloats<T>(dv, d_vals);
            LoadWeightFloats<WeightDType, VEC_SIZE>(weight, i * VEC_SIZE, w_vals);
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                if constexpr (ZeroCentered) w_vals[j] += 1.0f;
                x_cache[cache_count * VEC_SIZE + j] = x_vals[j];
                d_cache[cache_count * VEC_SIZE + j] = d_vals[j];
                w_cache[cache_count * VEC_SIZE + j] = w_vals[j];
                sq += x_vals[j] * x_vals[j];
                dp += d_vals[j] * w_vals[j] * x_vals[j];
            }
            ++cache_count;
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

        int li = 0;
        for (int64_t i = tid; i < vec_hidden; i += blockDim.x, ++li) {
            float out_vals[VEC_SIZE];
            float gs_vals[VEC_SIZE];
            if constexpr (FuseGradSum) {
                Vec gs_vec =
                    VecTraits<T>::load(grad_sum + grad_sum_off + i * VEC_SIZE);
                VecToFloats<T>(gs_vec, gs_vals);
            }
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                const int cache_idx = li * VEC_SIZE + j;
                const float x = x_cache[cache_idx];
                const float d = d_cache[cache_idx];
                const float rms_dx = irms * (d * w_cache[cache_idx] - x * cv);
                if constexpr (FuseGradSum) {
                    out_vals[j] =
                        AddGradSumLikeMaterializedAdd<T>(gs_vals[j], rms_dx);
                } else {
                    out_vals[j] = rms_dx;
                }
                local_dw[cache_idx] += d * x * irms;
            }
            VecTraits<T>::store(dx + dx_off + i * VEC_SIZE,
                                FloatsToVec<T>(out_vals));
        }
        __syncthreads();
    }

    const int64_t poff = blockIdx.x * hidden_size;
    int li = 0;
    for (int64_t i = tid; i < vec_hidden; i += blockDim.x, ++li) {
        if constexpr (VEC_SIZE == 4) {
            float4 dw_out;
            dw_out.x = local_dw[li * VEC_SIZE + 0];
            dw_out.y = local_dw[li * VEC_SIZE + 1];
            dw_out.z = local_dw[li * VEC_SIZE + 2];
            dw_out.w = local_dw[li * VEC_SIZE + 3];
            *reinterpret_cast<float4*>(partial_dw + poff + i * VEC_SIZE) = dw_out;
        } else {
            float2 dw_out;
            dw_out.x = local_dw[li * VEC_SIZE + 0];
            dw_out.y = local_dw[li * VEC_SIZE + 1];
            *reinterpret_cast<float2*>(partial_dw + poff + i * VEC_SIZE) = dw_out;
        }
    }
}

template<typename T, typename WeightDType, int MAX_VPT, bool ZeroCentered>
void LaunchRMSNormBackwardKernel(
    wallx_cuda::TensorView dx,
    float* partial_dw,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView weight,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t dx_row_stride,
    int64_t dy_row_stride,
    int64_t input_row_stride,
    int64_t block_size,
    cudaStream_t stream) {
  const int64_t vec_hidden = hidden_size / VecTraits<T>::vec_size;
  rmsnormBackwardPersistentKernel<T, WeightDType, MAX_VPT, ZeroCentered><<<num_blocks, block_size, 0, stream>>>(
      static_cast<T*>(dx.data_ptr()), partial_dw,
      static_cast<const T*>(dy.data_ptr()), static_cast<const T*>(input.data_ptr()),
      static_cast<const WeightDType*>(weight.data_ptr()), epsilon, hidden_size, num_tokens,
      dx_row_stride, dy_row_stride, input_row_stride);
}

// Wide (16-byte vectorized) persistent backward for 16-bit dtypes.
// Grid = num_blocks persistent blocks; each block owns full rows, ROWS at a
// time. The weight (with the zero-centered +1 folded in) is cached in
// registers once per block, x/dy raw bits are cached in registers between the
// reduction and the dx pass, so x and dy are read from HBM exactly once and dx
// written once. All 2*ROWS row reductions (sum(x*x), sum(dy*w*x) per row) are
// fused into a single one-barrier block all-reduce, so the barrier cost is
// amortized over ROWS rows and 2*ROWS*16B per thread of loads are in flight
// between barriers.
template<typename T, typename WeightDType, int VPT, int ROWS, bool CACHE_ROWS,
         bool HAS_RSTD, bool FUSE_GS, bool ZeroCentered>
__global__ void __launch_bounds__(512)
rmsnormBackwardWideKernel(
    T* __restrict__ dx,
    float* __restrict__ partial_dw,
    const T* __restrict__ dy,
    const T* __restrict__ input,
    const WeightDType* __restrict__ weight,
    const float* __restrict__ rstd,
    const T* __restrict__ grad_sum,
    const float epsilon,
    const int64_t hidden_size,
    const int64_t num_tokens,
    const int64_t dx_row_stride,
    const int64_t dy_row_stride,
    const int64_t input_row_stride,
    const int64_t grad_sum_row_stride) {
  constexpr int ELEMS = 8;
  // (sq, dp) per row without stored rstd; only dp per row with it.
  constexpr int NRED = HAS_RSTD ? ROWS : 2 * ROWS;
  static_assert(!std::is_same_v<T, float>, "wide path is for 16-bit tensors");
  const int tid = threadIdx.x;
  const int64_t num_blocks = gridDim.x;
  const int64_t vec_hidden = hidden_size / ELEMS;
  const float inv_hidden = 1.0f / static_cast<float>(hidden_size);

  // Parity double-buffered all-reduce scratch: one __syncthreads per chunk.
  // Chunk c reads scratch[p] while chunk c+1 writes scratch[1-p]; chunk c+2
  // can only write scratch[p] after passing chunk c+1's barrier, which in turn
  // requires every thread to have finished its chunk-c reads.
  __shared__ float s_red[2][32][NRED];
  const int lane = tid % 32;
  const int wid = tid / 32;
  const int num_warps = (blockDim.x + 31) / 32;

  float w_cache[VPT][ELEMS];
  float local_dw[VPT][ELEMS];
  #pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int64_t i = tid + k * static_cast<int64_t>(blockDim.x);
    #pragma unroll
    for (int j = 0; j < ELEMS; ++j) {
      w_cache[k][j] = 0.0f;
      local_dw[k][j] = 0.0f;
    }
    if (i < vec_hidden) {
      LoadWeight8Floats<WeightDType>(weight, i * ELEMS, w_cache[k]);
      #pragma unroll
      for (int j = 0; j < ELEMS; ++j) {
        if constexpr (ZeroCentered) w_cache[k][j] += 1.0f;
      }
    }
  }

  const int64_t num_chunks = (num_tokens + ROWS - 1) / ROWS;
  int parity = 0;
  for (int64_t chunk = blockIdx.x; chunk < num_chunks; chunk += num_blocks) {
    const int64_t row0 = chunk * ROWS;
    const int rows_here = static_cast<int>(
        num_tokens - row0 < ROWS ? num_tokens - row0 : ROWS);

    // Prefetch the chunk's rstd values first: they are consumed right after
    // the barrier, and issuing them before the bulk loads hides their (cold)
    // latency behind the partial-sum computation.
    float chunk_rstd[ROWS];
    if constexpr (HAS_RSTD) {
      #pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        chunk_rstd[r] = r < rows_here ? rstd[row0 + r] : 1.0f;
      }
    }

    // Issue all of the chunk's loads back-to-back before touching the data so
    // 2*ROWS*16B per thread are in flight at once. With CACHE_ROWS the raw
    // bits are kept in registers for the dx pass (single HBM+L2 read); without
    // it the dx pass re-reads the (L2-hot) rows, freeing registers for more
    // resident blocks.
    uint4 x_raw[ROWS][VPT], d_raw[ROWS][VPT];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      if (r < rows_here) {
        const T* in_row = input + (row0 + r) * input_row_stride;
        const T* dy_row = dy + (row0 + r) * dy_row_stride;
        #pragma unroll
        for (int k = 0; k < VPT; ++k) {
          const int64_t i = tid + k * static_cast<int64_t>(blockDim.x);
          if (i < vec_hidden) {
            x_raw[r][k] = *reinterpret_cast<const uint4*>(in_row + i * ELEMS);
            d_raw[r][k] = *reinterpret_cast<const uint4*>(dy_row + i * ELEMS);
          }
        }
      }
    }

    float red[NRED];
    #pragma unroll
    for (int j = 0; j < NRED; ++j) red[j] = 0.0f;
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      if (r < rows_here) {
        #pragma unroll
        for (int k = 0; k < VPT; ++k) {
          const int64_t i = tid + k * static_cast<int64_t>(blockDim.x);
          if (i < vec_hidden) {
            float x_vals[ELEMS], d_vals[ELEMS];
            Raw8ToFloats<T>(x_raw[r][k], x_vals);
            Raw8ToFloats<T>(d_raw[r][k], d_vals);
            #pragma unroll
            for (int j = 0; j < ELEMS; ++j) {
              if constexpr (HAS_RSTD) {
                red[r] += d_vals[j] * w_cache[k][j] * x_vals[j];
              } else {
                red[2 * r] += x_vals[j] * x_vals[j];
                red[2 * r + 1] += d_vals[j] * w_cache[k][j] * x_vals[j];
              }
            }
          }
        }
      }
    }

    // Fused block all-reduce of all 2*ROWS sums with a single barrier:
    // warp-reduce, lane 0 publishes, one syncthreads, then every thread sums
    // the per-warp partials itself (broadcast reads from shared memory).
    #pragma unroll
    for (int j = 0; j < NRED; ++j) {
      for (int offset = 16; offset > 0; offset /= 2) {
        red[j] += __shfl_down_sync(0xffffffff, red[j], offset);
      }
    }
    if (lane == 0) {
      #pragma unroll
      for (int j = 0; j < NRED; ++j) s_red[parity][wid][j] = red[j];
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < NRED; ++j) red[j] = s_red[parity][0][j];
    for (int wg = 1; wg < num_warps; ++wg) {
      #pragma unroll
      for (int j = 0; j < NRED; ++j) red[j] += s_red[parity][wg][j];
    }
    parity ^= 1;

    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      if (r < rows_here) {
        float irms, cv;
        if constexpr (HAS_RSTD) {
          irms = chunk_rstd[r];
          // 1/ve == irms^2, so cv = dp/(H*ve) = dp*irms^2/H.
          cv = red[r] * irms * irms * inv_hidden;
        } else {
          const float ve = red[2 * r] * inv_hidden + epsilon;
          irms = rsqrtf(ve);
          cv = red[2 * r + 1] / (static_cast<float>(hidden_size) * ve);
        }
        T* dx_row = dx + (row0 + r) * dx_row_stride;
        const T* in_row = input + (row0 + r) * input_row_stride;
        const T* dy_row = dy + (row0 + r) * dy_row_stride;
        const T* gs_row =
            FUSE_GS ? grad_sum + (row0 + r) * grad_sum_row_stride : nullptr;
        #pragma unroll
        for (int k = 0; k < VPT; ++k) {
          const int64_t i = tid + k * static_cast<int64_t>(blockDim.x);
          if (i < vec_hidden) {
            float x_vals[ELEMS], d_vals[ELEMS], out_vals[ELEMS];
            uint4 xr4, dr4;
            if constexpr (CACHE_ROWS) {
              xr4 = x_raw[r][k];
              dr4 = d_raw[r][k];
            } else {
              xr4 = *reinterpret_cast<const uint4*>(in_row + i * ELEMS);
              dr4 = *reinterpret_cast<const uint4*>(dy_row + i * ELEMS);
            }
            Raw8ToFloats<T>(xr4, x_vals);
            Raw8ToFloats<T>(dr4, d_vals);
            float gs_vals[ELEMS];
            if constexpr (FUSE_GS) {
              Raw8ToFloats<T>(
                  *reinterpret_cast<const uint4*>(gs_row + i * ELEMS), gs_vals);
            }
            #pragma unroll
            for (int j = 0; j < ELEMS; ++j) {
              const float rms_dx =
                  irms * (d_vals[j] * w_cache[k][j] - x_vals[j] * cv);
              if constexpr (FUSE_GS) {
                out_vals[j] =
                    AddGradSumLikeMaterializedAdd<T>(gs_vals[j], rms_dx);
              } else {
                out_vals[j] = rms_dx;
              }
              local_dw[k][j] += d_vals[j] * x_vals[j] * irms;
            }
            *reinterpret_cast<uint4*>(dx_row + i * ELEMS) =
                FloatsToRaw8<T>(out_vals);
          }
        }
      }
    }
  }

  const int64_t poff = blockIdx.x * hidden_size;
  #pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int64_t i = tid + k * static_cast<int64_t>(blockDim.x);
    if (i < vec_hidden) {
      *reinterpret_cast<float4*>(partial_dw + poff + i * ELEMS) =
          make_float4(local_dw[k][0], local_dw[k][1],
                      local_dw[k][2], local_dw[k][3]);
      *reinterpret_cast<float4*>(partial_dw + poff + i * ELEMS + 4) =
          make_float4(local_dw[k][4], local_dw[k][5],
                      local_dw[k][6], local_dw[k][7]);
    }
  }
}

template<typename T, typename WeightDType, bool HAS_RSTD, bool ZeroCentered>
void LaunchRMSNormBackwardWideKernel(
    wallx_cuda::TensorView dx,
    float* partial_dw,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView weight,
    const float* rstd,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t dx_row_stride,
    int64_t dy_row_stride,
    int64_t input_row_stride,
    cudaStream_t stream) {
  const int64_t vec_hidden = hidden_size / kWideElems;
  const int vpt = PickWideVPTBwd(vec_hidden);
  const int64_t block_size = WideBlockSize(vec_hidden, vpt);
  T* dx_ptr = static_cast<T*>(dx.data_ptr());
  const T* dy_ptr = static_cast<const T*>(dy.data_ptr());
  const T* in_ptr = static_cast<const T*>(input.data_ptr());
  const WeightDType* w_ptr = static_cast<const WeightDType*>(weight.data_ptr());
  switch (vpt) {
    case 1:
      // VPT=1 has register room to batch two rows per block iteration.
      rmsnormBackwardWideKernel<T, WeightDType, 1, 2, true, HAS_RSTD, false,
                                ZeroCentered>
          <<<num_blocks, block_size, 0, stream>>>(
              dx_ptr, partial_dw, dy_ptr, in_ptr, w_ptr, rstd, nullptr,
              epsilon, hidden_size, num_tokens, dx_row_stride, dy_row_stride,
              input_row_stride, 0);
      break;
    case 2:
      rmsnormBackwardWideKernel<T, WeightDType, 2, 1, true, HAS_RSTD, false,
                                ZeroCentered>
          <<<num_blocks, block_size, 0, stream>>>(
              dx_ptr, partial_dw, dy_ptr, in_ptr, w_ptr, rstd, nullptr,
              epsilon, hidden_size, num_tokens, dx_row_stride, dy_row_stride,
              input_row_stride, 0);
      break;
    default:
      throw std::runtime_error("RMSNorm backward wide path: unsupported VPT");
  }
}

// Fused (add_rmsnorm) rstd backward on the same wide kernel: identical row
// partitioning and reduction order as the plain wide backward, so its dw is
// bit-identical to the materialized (add + rmsnorm) path.
template<typename T, typename WeightDType, bool ZeroCentered>
void LaunchAddRMSNormBackwardWideKernel(
    wallx_cuda::TensorView d_add,
    float* partial_dw,
    wallx_cuda::TensorView grad_sum,
    wallx_cuda::TensorView grad_norm,
    wallx_cuda::TensorView sum_input,
    wallx_cuda::TensorView weight,
    const float* rstd,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t d_add_row_stride,
    int64_t grad_sum_row_stride,
    int64_t grad_norm_row_stride,
    int64_t sum_row_stride,
    cudaStream_t stream) {
  const int64_t vec_hidden = hidden_size / kWideElems;
  const int vpt = PickWideVPTBwd(vec_hidden);
  const int64_t block_size = WideBlockSize(vec_hidden, vpt);
  T* d_add_ptr = static_cast<T*>(d_add.data_ptr());
  const T* gs_ptr = static_cast<const T*>(grad_sum.data_ptr());
  const T* gn_ptr = static_cast<const T*>(grad_norm.data_ptr());
  const T* sum_ptr = static_cast<const T*>(sum_input.data_ptr());
  const WeightDType* w_ptr = static_cast<const WeightDType*>(weight.data_ptr());
  switch (vpt) {
    case 1:
      rmsnormBackwardWideKernel<T, WeightDType, 1, 2, true, true, true,
                                ZeroCentered>
          <<<num_blocks, block_size, 0, stream>>>(
              d_add_ptr, partial_dw, gn_ptr, sum_ptr, w_ptr, rstd, gs_ptr,
              0.0f, hidden_size, num_tokens, d_add_row_stride,
              grad_norm_row_stride, sum_row_stride, grad_sum_row_stride);
      break;
    case 2:
      rmsnormBackwardWideKernel<T, WeightDType, 2, 1, true, true, true,
                                ZeroCentered>
          <<<num_blocks, block_size, 0, stream>>>(
              d_add_ptr, partial_dw, gn_ptr, sum_ptr, w_ptr, rstd, gs_ptr,
              0.0f, hidden_size, num_tokens, d_add_row_stride,
              grad_norm_row_stride, sum_row_stride, grad_sum_row_stride);
      break;
    default:
      throw std::runtime_error("AddRMSNorm backward wide path: unsupported VPT");
  }
}

template<bool ZeroCentered>
bool TryAddRMSNormBackwardWide(
    wallx_cuda::TensorView d_add,
    float* partial_dw,
    wallx_cuda::TensorView grad_sum,
    wallx_cuda::TensorView grad_norm,
    wallx_cuda::TensorView sum_input,
    wallx_cuda::TensorView weight,
    const float* rstd,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t d_add_row_stride,
    int64_t grad_sum_row_stride,
    int64_t grad_norm_row_stride,
    int64_t sum_row_stride,
    cudaStream_t stream) {
  if (!WideLayoutOK(hidden_size,
                    {d_add_row_stride, grad_sum_row_stride,
                     grad_norm_row_stride, sum_row_stride},
                    {d_add.data_ptr(), grad_sum.data_ptr(),
                     grad_norm.data_ptr(), sum_input.data_ptr(),
                     weight.data_ptr()})) {
    return false;
  }
  if (PickWideVPTBwd(hidden_size / kWideElems) == 0) return false;
  const auto& in_dtype = sum_input.dtype();
  if (IsBFloat16(in_dtype) && IsBFloat16(weight.dtype())) {
    LaunchAddRMSNormBackwardWideKernel<__nv_bfloat16, __nv_bfloat16,
                                       ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight, rstd,
        hidden_size, num_tokens, num_blocks, d_add_row_stride,
        grad_sum_row_stride, grad_norm_row_stride, sum_row_stride, stream);
  } else if (IsBFloat16(in_dtype) && IsFloat32(weight.dtype())) {
    LaunchAddRMSNormBackwardWideKernel<__nv_bfloat16, float, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight, rstd,
        hidden_size, num_tokens, num_blocks, d_add_row_stride,
        grad_sum_row_stride, grad_norm_row_stride, sum_row_stride, stream);
  } else if (IsFloat16(in_dtype) && IsFloat16(weight.dtype())) {
    LaunchAddRMSNormBackwardWideKernel<half, half, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight, rstd,
        hidden_size, num_tokens, num_blocks, d_add_row_stride,
        grad_sum_row_stride, grad_norm_row_stride, sum_row_stride, stream);
  } else if (IsFloat16(in_dtype) && IsFloat32(weight.dtype())) {
    LaunchAddRMSNormBackwardWideKernel<half, float, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight, rstd,
        hidden_size, num_tokens, num_blocks, d_add_row_stride,
        grad_sum_row_stride, grad_norm_row_stride, sum_row_stride, stream);
  } else if (IsFloat16(in_dtype) && IsBFloat16(weight.dtype())) {
    LaunchAddRMSNormBackwardWideKernel<half, __nv_bfloat16, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight, rstd,
        hidden_size, num_tokens, num_blocks, d_add_row_stride,
        grad_sum_row_stride, grad_norm_row_stride, sum_row_stride, stream);
  } else {
    return false;
  }
  return true;
}

// Try dispatching the backward wide path; returns false when the dtype/layout
// is not covered (caller falls back to the generic persistent kernel).
template<bool HAS_RSTD, bool ZeroCentered>
bool TryRMSNormBackwardWide(
    wallx_cuda::TensorView dx,
    float* partial_dw,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView weight,
    const float* rstd,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t dx_row_stride,
    int64_t dy_row_stride,
    int64_t input_row_stride,
    cudaStream_t stream) {
  if (!WideLayoutOK(hidden_size,
                    {dx_row_stride, dy_row_stride, input_row_stride},
                    {dx.data_ptr(), dy.data_ptr(), input.data_ptr(),
                     weight.data_ptr()})) {
    return false;
  }
  if (PickWideVPTBwd(hidden_size / kWideElems) == 0) return false;
  if (IsBFloat16(input.dtype()) && IsBFloat16(weight.dtype())) {
    LaunchRMSNormBackwardWideKernel<__nv_bfloat16, __nv_bfloat16, HAS_RSTD,
                                    ZeroCentered>(
        dx, partial_dw, dy, input, weight, rstd, epsilon, hidden_size,
        num_tokens, num_blocks, dx_row_stride, dy_row_stride, input_row_stride,
        stream);
  } else if (IsBFloat16(input.dtype()) && IsFloat32(weight.dtype())) {
    LaunchRMSNormBackwardWideKernel<__nv_bfloat16, float, HAS_RSTD,
                                    ZeroCentered>(
        dx, partial_dw, dy, input, weight, rstd, epsilon, hidden_size,
        num_tokens, num_blocks, dx_row_stride, dy_row_stride, input_row_stride,
        stream);
  } else if (IsFloat16(input.dtype()) && IsFloat16(weight.dtype())) {
    LaunchRMSNormBackwardWideKernel<half, half, HAS_RSTD, ZeroCentered>(
        dx, partial_dw, dy, input, weight, rstd, epsilon, hidden_size,
        num_tokens, num_blocks, dx_row_stride, dy_row_stride, input_row_stride,
        stream);
  } else if (IsFloat16(input.dtype()) && IsFloat32(weight.dtype())) {
    LaunchRMSNormBackwardWideKernel<half, float, HAS_RSTD, ZeroCentered>(
        dx, partial_dw, dy, input, weight, rstd, epsilon, hidden_size,
        num_tokens, num_blocks, dx_row_stride, dy_row_stride, input_row_stride,
        stream);
  } else if (IsFloat16(input.dtype()) && IsBFloat16(weight.dtype())) {
    LaunchRMSNormBackwardWideKernel<half, __nv_bfloat16, HAS_RSTD,
                                    ZeroCentered>(
        dx, partial_dw, dy, input, weight, rstd, epsilon, hidden_size,
        num_tokens, num_blocks, dx_row_stride, dy_row_stride, input_row_stride,
        stream);
  } else {
    return false;
  }
  return true;
}

template<typename T, typename WeightDType, bool ZeroCentered>
void LaunchAddRMSNormBackwardKernel(
    wallx_cuda::TensorView d_add,
    float* partial_dw,
    wallx_cuda::TensorView grad_sum,
    wallx_cuda::TensorView grad_norm,
    wallx_cuda::TensorView sum_input,
    wallx_cuda::TensorView weight,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t d_add_row_stride,
    int64_t grad_sum_row_stride,
    int64_t grad_norm_row_stride,
    int64_t sum_row_stride,
    int64_t block_size,
    cudaStream_t stream) {
  addRMSNormBackwardPersistentKernel<T, WeightDType, 8, ZeroCentered>
      <<<num_blocks, block_size, 0, stream>>>(
          static_cast<T*>(d_add.data_ptr()), partial_dw,
          static_cast<const T*>(grad_sum.data_ptr()),
          static_cast<const T*>(grad_norm.data_ptr()),
          static_cast<const T*>(sum_input.data_ptr()),
          static_cast<const WeightDType*>(weight.data_ptr()),
          epsilon, hidden_size, num_tokens, d_add_row_stride,
          grad_sum_row_stride, grad_norm_row_stride, sum_row_stride);
}

template<typename T, typename WeightDType, int MAX_VEC_PER_THREAD, bool ZeroCentered>
__global__ void addRMSNormBackwardRstdCachedKernel(
    T* __restrict__ d_add,
    float* __restrict__ partial_dw,
    const T* __restrict__ grad_sum,
    const T* __restrict__ grad_norm,
    const T* __restrict__ sum_input,
    const WeightDType* __restrict__ weight,
    const float* __restrict__ rstd,
    const int64_t hidden_size,
    const int64_t num_tokens,
    const int64_t d_add_row_stride,
    const int64_t grad_sum_row_stride,
    const int64_t grad_norm_row_stride,
    const int64_t sum_row_stride
) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;

    const int64_t tid = threadIdx.x;
    const int64_t num_blocks = gridDim.x;
    const int64_t vec_hidden = hidden_size / VEC_SIZE;
    const float inv_hidden = 1.0f / static_cast<float>(hidden_size);

    __shared__ float s_norm_scale;

    float local_dw[MAX_VEC_PER_THREAD * VEC_SIZE];
    #pragma unroll
    for (int j = 0; j < MAX_VEC_PER_THREAD * VEC_SIZE; ++j) local_dw[j] = 0.0f;

    for (int64_t row = blockIdx.x; row < num_tokens; row += num_blocks) {
        const int64_t d_add_off = row * d_add_row_stride;
        const int64_t grad_sum_off = row * grad_sum_row_stride;
        const int64_t grad_norm_off = row * grad_norm_row_stride;
        const int64_t sum_off = row * sum_row_stride;
        const float row_rstd = rstd[row];
        float dp = 0.0f;
        float x_cache[MAX_VEC_PER_THREAD * VEC_SIZE];
        float d_cache[MAX_VEC_PER_THREAD * VEC_SIZE];
        float w_cache[MAX_VEC_PER_THREAD * VEC_SIZE];

        int li = 0;
        for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
            Vec xv = VecTraits<T>::load(sum_input + sum_off + i * VEC_SIZE);
            Vec dv = VecTraits<T>::load(grad_norm + grad_norm_off + i * VEC_SIZE);
            float x_vals[VEC_SIZE], d_vals[VEC_SIZE], w_vals[VEC_SIZE];
            VecToFloats<T>(xv, x_vals);
            VecToFloats<T>(dv, d_vals);
            LoadWeightFloats<WeightDType, VEC_SIZE>(weight, i * VEC_SIZE, w_vals);
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                if constexpr (ZeroCentered) w_vals[j] += 1.0f;
                const int cache_idx = li * VEC_SIZE + j;
                x_cache[cache_idx] = x_vals[j];
                d_cache[cache_idx] = d_vals[j];
                w_cache[cache_idx] = w_vals[j];
                dp += d_vals[j] * w_vals[j] * x_vals[j];
                local_dw[cache_idx] += d_vals[j] * x_vals[j] * row_rstd;
            }
            ++li;
        }
        dp = blockReduceSum(dp);
        if (tid == 0) {
            s_norm_scale = dp * inv_hidden * row_rstd * row_rstd;
        }
        __syncthreads();
        const float norm_scale = s_norm_scale;

        li = 0;
        for (int64_t i = tid; i < vec_hidden; i += blockDim.x, ++li) {
            Vec gs = VecTraits<T>::load(grad_sum + grad_sum_off + i * VEC_SIZE);
            float x_vals[VEC_SIZE], d_vals[VEC_SIZE], gs_vals[VEC_SIZE];
            float w_vals[VEC_SIZE], out_vals[VEC_SIZE];
            VecToFloats<T>(gs, gs_vals);
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                const int cache_idx = li * VEC_SIZE + j;
                x_vals[j] = x_cache[cache_idx];
                d_vals[j] = d_cache[cache_idx];
                w_vals[j] = w_cache[cache_idx];
                const float rms_dx =
                    row_rstd * (d_vals[j] * w_vals[j] - x_vals[j] * norm_scale);
                out_vals[j] =
                    AddGradSumLikeMaterializedAdd<T>(gs_vals[j], rms_dx);
            }
            VecTraits<T>::store(d_add + d_add_off + i * VEC_SIZE,
                                FloatsToVec<T>(out_vals));
        }
    }

    const int64_t poff = blockIdx.x * hidden_size;
    int li = 0;
    for (int64_t i = tid; i < vec_hidden; i += blockDim.x, ++li) {
        if constexpr (VEC_SIZE == 4) {
            float4 dw_out;
            dw_out.x = local_dw[li * VEC_SIZE + 0];
            dw_out.y = local_dw[li * VEC_SIZE + 1];
            dw_out.z = local_dw[li * VEC_SIZE + 2];
            dw_out.w = local_dw[li * VEC_SIZE + 3];
            *reinterpret_cast<float4*>(partial_dw + poff + i * VEC_SIZE) = dw_out;
        } else {
            float2 dw_out;
            dw_out.x = local_dw[li * VEC_SIZE + 0];
            dw_out.y = local_dw[li * VEC_SIZE + 1];
            *reinterpret_cast<float2*>(partial_dw + poff + i * VEC_SIZE) = dw_out;
        }
    }
}

template<typename T, typename WeightDType, int MAX_VEC4_PER_THREAD, bool ZeroCentered>
__global__ void addRMSNormBackwardRstdDirectVec4Kernel(
    T* __restrict__ d_add,
    float* __restrict__ partial_dw,
    const T* __restrict__ grad_sum,
    const T* __restrict__ grad_norm,
    const T* __restrict__ sum_input,
    const WeightDType* __restrict__ weight,
    const float* __restrict__ rstd,
    const int64_t hidden_size,
    const int64_t num_tokens,
    const int64_t d_add_row_stride,
    const int64_t grad_sum_row_stride,
    const int64_t grad_norm_row_stride,
    const int64_t sum_row_stride
) {
    constexpr int VEC4_SIZE = 4;
    static_assert(!std::is_same_v<T, float>,
                  "DirectVec4 fast path is intended for fp16/bf16 tensors");

    const int64_t tid = threadIdx.x;
    const int64_t num_blocks = gridDim.x;
    const int64_t vec4_hidden = hidden_size / VEC4_SIZE;
    const float inv_hidden = 1.0f / static_cast<float>(hidden_size);

    __shared__ float s_norm_scale;

    float w_cache[MAX_VEC4_PER_THREAD * VEC4_SIZE];
    float local_dw[MAX_VEC4_PER_THREAD * VEC4_SIZE];
    #pragma unroll
    for (int j = 0; j < MAX_VEC4_PER_THREAD * VEC4_SIZE; ++j) {
        w_cache[j] = 0.0f;
        local_dw[j] = 0.0f;
    }

    int w_li = 0;
    for (int64_t i = tid; i < vec4_hidden; i += blockDim.x, ++w_li) {
        const int64_t col = i * VEC4_SIZE;
        LoadWeightFloats<WeightDType, VEC4_SIZE>(
            weight, col, w_cache + w_li * VEC4_SIZE);
        #pragma unroll
        for (int j = 0; j < VEC4_SIZE; ++j) {
            if constexpr (ZeroCentered) {
                w_cache[w_li * VEC4_SIZE + j] += 1.0f;
            }
        }
    }

    for (int64_t row = blockIdx.x; row < num_tokens; row += num_blocks) {
        const int64_t d_add_off = row * d_add_row_stride;
        const int64_t grad_sum_off = row * grad_sum_row_stride;
        const int64_t grad_norm_off = row * grad_norm_row_stride;
        const int64_t sum_off = row * sum_row_stride;
        const float row_rstd = rstd[row];
        float dp = 0.0f;

        int li = 0;
        for (int64_t i = tid; i < vec4_hidden; i += blockDim.x, ++li) {
            const int64_t col = i * VEC4_SIZE;
            float x_vals[VEC4_SIZE], d_vals[VEC4_SIZE];
            Load4Floats<T>(sum_input + sum_off + col, x_vals);
            Load4Floats<T>(grad_norm + grad_norm_off + col, d_vals);
            #pragma unroll
            for (int j = 0; j < VEC4_SIZE; ++j) {
                const float w = w_cache[li * VEC4_SIZE + j];
                dp += d_vals[j] * w * x_vals[j];
                local_dw[li * VEC4_SIZE + j] +=
                    d_vals[j] * x_vals[j] * row_rstd;
            }
        }
        dp = blockReduceSum(dp);
        if (tid == 0) {
            s_norm_scale = dp * inv_hidden * row_rstd * row_rstd;
        }
        __syncthreads();
        const float norm_scale = s_norm_scale;

        li = 0;
        for (int64_t i = tid; i < vec4_hidden; i += blockDim.x, ++li) {
            const int64_t col = i * VEC4_SIZE;
            float x_vals[VEC4_SIZE], d_vals[VEC4_SIZE], gs_vals[VEC4_SIZE];
            float out_vals[VEC4_SIZE];
            Load4Floats<T>(sum_input + sum_off + col, x_vals);
            Load4Floats<T>(grad_norm + grad_norm_off + col, d_vals);
            Load4Floats<T>(grad_sum + grad_sum_off + col, gs_vals);
            #pragma unroll
            for (int j = 0; j < VEC4_SIZE; ++j) {
                const float w = w_cache[li * VEC4_SIZE + j];
                const float rms_dx =
                    row_rstd * (d_vals[j] * w - x_vals[j] * norm_scale);
                out_vals[j] =
                    AddGradSumLikeMaterializedAdd<T>(gs_vals[j], rms_dx);
            }
            Store4Scalars<T>(d_add + d_add_off + col, out_vals);
        }
    }

    const int64_t poff = blockIdx.x * hidden_size;
    int li = 0;
    for (int64_t i = tid; i < vec4_hidden; i += blockDim.x, ++li) {
        const int64_t col = i * VEC4_SIZE;
        *reinterpret_cast<float4*>(partial_dw + poff + col) =
            make_float4(local_dw[li * VEC4_SIZE + 0],
                        local_dw[li * VEC4_SIZE + 1],
                        local_dw[li * VEC4_SIZE + 2],
                        local_dw[li * VEC4_SIZE + 3]);
    }
}

template<typename T, typename WeightDType, bool ZeroCentered>
__global__ void addRMSNormBackwardRstdD2560Vec4CpAsyncKernel(
    T* __restrict__ d_add,
    float* __restrict__ partial_dw,
    const T* __restrict__ grad_sum,
    const T* __restrict__ grad_norm,
    const T* __restrict__ sum_input,
    const WeightDType* __restrict__ weight,
    const float* __restrict__ rstd,
    const int64_t num_tokens,
    const int64_t d_add_row_stride,
    const int64_t grad_sum_row_stride,
    const int64_t grad_norm_row_stride,
    const int64_t sum_row_stride
) {
    static_assert(!std::is_same_v<T, float>,
                  "D2560 vec4 cp.async fast path is for fp16/bf16 tensors");
    constexpr int HIDDEN_SIZE = 2560;
    constexpr int THREADS = 128;
    constexpr int VEC_ELEMS = 4;
    constexpr int VEC_STRIDE = THREADS * VEC_ELEMS;
    constexpr int VEC_PER_THREAD = HIDDEN_SIZE / VEC_STRIDE;
    constexpr int VALUES_PER_THREAD = VEC_PER_THREAD * VEC_ELEMS;
    constexpr float INV_HIDDEN = 1.0f / static_cast<float>(HIDDEN_SIZE);
    static_assert(VEC_PER_THREAD == 5);

    extern __shared__ __align__(16) unsigned char smem_bytes[];
    T* smem = reinterpret_cast<T*>(smem_bytes);
    T* sx0 = smem;
    T* sd0 = sx0 + HIDDEN_SIZE;
    T* sx1 = sd0 + HIDDEN_SIZE;
    T* sd1 = sx1 + HIDDEN_SIZE;

    const int tid = threadIdx.x;
    const int64_t num_blocks = gridDim.x;
    __shared__ float s_norm_scale;

    float w_cache[VALUES_PER_THREAD];
    float local_dw[VALUES_PER_THREAD];
    #pragma unroll
    for (int j = 0; j < VALUES_PER_THREAD; ++j) {
        w_cache[j] = 0.0f;
        local_dw[j] = 0.0f;
    }

    #pragma unroll
    for (int k = 0; k < VEC_PER_THREAD; ++k) {
        const int col = tid * VEC_ELEMS + k * VEC_STRIDE;
        LoadWeight4Floats<WeightDType>(
            weight, col, w_cache + k * VEC_ELEMS);
        #pragma unroll
        for (int j = 0; j < VEC_ELEMS; ++j) {
            if constexpr (ZeroCentered) {
                w_cache[k * VEC_ELEMS + j] += 1.0f;
            }
        }
    }

    int stage = 0;
    const int64_t first_row = blockIdx.x;
    if (first_row < num_tokens) {
        T* sx = sx0;
        T* sd = sd0;
        const int64_t sum_off = first_row * sum_row_stride;
        const int64_t grad_norm_off = first_row * grad_norm_row_stride;
        #pragma unroll
        for (int k = 0; k < VEC_PER_THREAD; ++k) {
            const int col = tid * VEC_ELEMS + k * VEC_STRIDE;
            CpAsync8(sx + col, sum_input + sum_off + col);
            CpAsync8(sd + col, grad_norm + grad_norm_off + col);
        }
        CpAsyncCommitGroup();
    }

    for (int64_t row = first_row; row < num_tokens; row += num_blocks) {
        const int next_stage = stage ^ 1;
        const int64_t next_row = row + num_blocks;
        if (next_row < num_tokens) {
            T* sx_next = next_stage == 0 ? sx0 : sx1;
            T* sd_next = next_stage == 0 ? sd0 : sd1;
            const int64_t next_sum_off = next_row * sum_row_stride;
            const int64_t next_grad_norm_off = next_row * grad_norm_row_stride;
            #pragma unroll
            for (int k = 0; k < VEC_PER_THREAD; ++k) {
                const int col = tid * VEC_ELEMS + k * VEC_STRIDE;
                CpAsync8(sx_next + col, sum_input + next_sum_off + col);
                CpAsync8(sd_next + col, grad_norm + next_grad_norm_off + col);
            }
            CpAsyncCommitGroup();
            CpAsyncWaitGroup<1>();
        } else {
            CpAsyncWaitGroup<0>();
        }

        T* sx = stage == 0 ? sx0 : sx1;
        T* sd = stage == 0 ? sd0 : sd1;
        const float row_rstd = rstd[row];
        float dp = 0.0f;
        uint2 x_raw[VEC_PER_THREAD];
        uint2 d_raw[VEC_PER_THREAD];

        #pragma unroll
        for (int k = 0; k < VEC_PER_THREAD; ++k) {
            const int col = tid * VEC_ELEMS + k * VEC_STRIDE;
            float x_vals[VEC_ELEMS], d_vals[VEC_ELEMS];
            const uint2 xr = Load4Raw(sx + col);
            const uint2 dr = Load4Raw(sd + col);
            x_raw[k] = xr;
            d_raw[k] = dr;
            Raw4ToFloats<T>(xr, x_vals);
            Raw4ToFloats<T>(dr, d_vals);
            #pragma unroll
            for (int j = 0; j < VEC_ELEMS; ++j) {
                const int idx = k * VEC_ELEMS + j;
                const float x = x_vals[j];
                const float d = d_vals[j];
                dp += d * w_cache[idx] * x;
                local_dw[idx] += d * x * row_rstd;
            }
        }

        dp = blockReduceSum(dp);
        if (tid == 0) {
            s_norm_scale = dp * INV_HIDDEN * row_rstd * row_rstd;
        }
        __syncthreads();
        const float norm_scale = s_norm_scale;

        const int64_t d_add_off = row * d_add_row_stride;
        const int64_t grad_sum_off = row * grad_sum_row_stride;
        #pragma unroll
        for (int k = 0; k < VEC_PER_THREAD; ++k) {
            const int col = tid * VEC_ELEMS + k * VEC_STRIDE;
            float x_vals[VEC_ELEMS], d_vals[VEC_ELEMS], gs_vals[VEC_ELEMS];
            float out_vals[VEC_ELEMS];
            Raw4ToFloats<T>(x_raw[k], x_vals);
            Raw4ToFloats<T>(d_raw[k], d_vals);
            Raw4ToFloats<T>(Load4Raw(grad_sum + grad_sum_off + col), gs_vals);
            #pragma unroll
            for (int j = 0; j < VEC_ELEMS; ++j) {
                const int idx = k * VEC_ELEMS + j;
                const float rms_dx =
                    row_rstd * (d_vals[j] * w_cache[idx] - x_vals[j] * norm_scale);
                out_vals[j] =
                    AddGradSumLikeMaterializedAdd<T>(gs_vals[j], rms_dx);
            }
            Store4RawFromFloats(d_add + d_add_off + col, out_vals);
        }
        stage = next_stage;
    }

    const int64_t poff = blockIdx.x * HIDDEN_SIZE;
    #pragma unroll
    for (int k = 0; k < VEC_PER_THREAD; ++k) {
        const int col = tid * VEC_ELEMS + k * VEC_STRIDE;
        *reinterpret_cast<float4*>(partial_dw + poff + col) =
            make_float4(local_dw[k * VEC_ELEMS + 0],
                        local_dw[k * VEC_ELEMS + 1],
                        local_dw[k * VEC_ELEMS + 2],
                        local_dw[k * VEC_ELEMS + 3]);
    }
}

template<typename T, typename WeightDType, int MAX_VEC_PER_THREAD, bool ZeroCentered>
__global__ void addRMSNormBackwardRstdDirectKernel(
    T* __restrict__ d_add,
    float* __restrict__ partial_dw,
    const T* __restrict__ grad_sum,
    const T* __restrict__ grad_norm,
    const T* __restrict__ sum_input,
    const WeightDType* __restrict__ weight,
    const float* __restrict__ rstd,
    const int64_t hidden_size,
    const int64_t num_tokens,
    const int64_t d_add_row_stride,
    const int64_t grad_sum_row_stride,
    const int64_t grad_norm_row_stride,
    const int64_t sum_row_stride
) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;

    const int64_t tid = threadIdx.x;
    const int64_t num_blocks = gridDim.x;
    const int64_t vec_hidden = hidden_size / VEC_SIZE;
    const float inv_hidden = 1.0f / static_cast<float>(hidden_size);

    __shared__ float s_norm_scale;

    float local_dw[MAX_VEC_PER_THREAD * VEC_SIZE];
    #pragma unroll
    for (int j = 0; j < MAX_VEC_PER_THREAD * VEC_SIZE; ++j) local_dw[j] = 0.0f;

    for (int64_t row = blockIdx.x; row < num_tokens; row += num_blocks) {
        const int64_t d_add_off = row * d_add_row_stride;
        const int64_t grad_sum_off = row * grad_sum_row_stride;
        const int64_t grad_norm_off = row * grad_norm_row_stride;
        const int64_t sum_off = row * sum_row_stride;
        const float row_rstd = rstd[row];
        float dp = 0.0f;

        int li = 0;
        for (int64_t i = tid; i < vec_hidden; i += blockDim.x, ++li) {
            Vec xv = VecTraits<T>::load(sum_input + sum_off + i * VEC_SIZE);
            Vec dv = VecTraits<T>::load(grad_norm + grad_norm_off + i * VEC_SIZE);
            float x_vals[VEC_SIZE], d_vals[VEC_SIZE], w_vals[VEC_SIZE];
            VecToFloats<T>(xv, x_vals);
            VecToFloats<T>(dv, d_vals);
            LoadWeightFloats<WeightDType, VEC_SIZE>(weight, i * VEC_SIZE, w_vals);
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                if constexpr (ZeroCentered) w_vals[j] += 1.0f;
                dp += d_vals[j] * w_vals[j] * x_vals[j];
                local_dw[li * VEC_SIZE + j] += d_vals[j] * x_vals[j] * row_rstd;
            }
        }
        dp = blockReduceSum(dp);
        if (tid == 0) {
            s_norm_scale = dp * inv_hidden * row_rstd * row_rstd;
        }
        __syncthreads();
        const float norm_scale = s_norm_scale;

        for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
            Vec xv = VecTraits<T>::load(sum_input + sum_off + i * VEC_SIZE);
            Vec dv = VecTraits<T>::load(grad_norm + grad_norm_off + i * VEC_SIZE);
            Vec gs = VecTraits<T>::load(grad_sum + grad_sum_off + i * VEC_SIZE);
            float x_vals[VEC_SIZE], d_vals[VEC_SIZE], gs_vals[VEC_SIZE];
            float w_vals[VEC_SIZE], out_vals[VEC_SIZE];
            VecToFloats<T>(xv, x_vals);
            VecToFloats<T>(dv, d_vals);
            VecToFloats<T>(gs, gs_vals);
            LoadWeightFloats<WeightDType, VEC_SIZE>(weight, i * VEC_SIZE, w_vals);
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                if constexpr (ZeroCentered) w_vals[j] += 1.0f;
                const float rms_dx =
                    row_rstd * (d_vals[j] * w_vals[j] - x_vals[j] * norm_scale);
                // Match the other backward paths (and the unfused reference): round
                // rms_dx to the storage dtype before the add, like a materialized add.
                out_vals[j] =
                    AddGradSumLikeMaterializedAdd<T>(gs_vals[j], rms_dx);
            }
            VecTraits<T>::store(d_add + d_add_off + i * VEC_SIZE,
                                FloatsToVec<T>(out_vals));
        }
    }

    const int64_t poff = blockIdx.x * hidden_size;
    int li = 0;
    for (int64_t i = tid; i < vec_hidden; i += blockDim.x, ++li) {
        if constexpr (VEC_SIZE == 4) {
            float4 dw_out;
            dw_out.x = local_dw[li * VEC_SIZE + 0];
            dw_out.y = local_dw[li * VEC_SIZE + 1];
            dw_out.z = local_dw[li * VEC_SIZE + 2];
            dw_out.w = local_dw[li * VEC_SIZE + 3];
            *reinterpret_cast<float4*>(partial_dw + poff + i * VEC_SIZE) = dw_out;
        } else {
            float2 dw_out;
            dw_out.x = local_dw[li * VEC_SIZE + 0];
            dw_out.y = local_dw[li * VEC_SIZE + 1];
            *reinterpret_cast<float2*>(partial_dw + poff + i * VEC_SIZE) = dw_out;
        }
    }
}

template<typename T, typename WeightDType, bool ZeroCentered>
void LaunchAddRMSNormBackwardWithRstdKernel(
    wallx_cuda::TensorView d_add,
    float* partial_dw,
    wallx_cuda::TensorView grad_sum,
    wallx_cuda::TensorView grad_norm,
    wallx_cuda::TensorView sum_input,
    wallx_cuda::TensorView weight,
    wallx_cuda::TensorView rstd,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t d_add_row_stride,
    int64_t grad_sum_row_stride,
    int64_t grad_norm_row_stride,
    int64_t sum_row_stride,
    int64_t block_size,
    cudaStream_t stream) {
  const int64_t vec_hidden = hidden_size / VecTraits<T>::vec_size;
  if constexpr (!std::is_same_v<T, float>) {
    constexpr int D2560_VEC4_CP_ASYNC_BLOCK_SIZE = 128;
    constexpr int D2560_VEC4_CP_ASYNC_VEC_ELEMS = 4;
    if (hidden_size == 2560 &&
        d_add_row_stride % D2560_VEC4_CP_ASYNC_VEC_ELEMS == 0 &&
        grad_sum_row_stride % D2560_VEC4_CP_ASYNC_VEC_ELEMS == 0 &&
        grad_norm_row_stride % D2560_VEC4_CP_ASYNC_VEC_ELEMS == 0 &&
        sum_row_stride % D2560_VEC4_CP_ASYNC_VEC_ELEMS == 0) {
      constexpr size_t SMEM_BYTES = 2 * 2 * 2560 * sizeof(T);
      addRMSNormBackwardRstdD2560Vec4CpAsyncKernel<T, WeightDType, ZeroCentered>
          <<<num_blocks, D2560_VEC4_CP_ASYNC_BLOCK_SIZE, SMEM_BYTES, stream>>>(
              static_cast<T*>(d_add.data_ptr()), partial_dw,
              static_cast<const T*>(grad_sum.data_ptr()),
              static_cast<const T*>(grad_norm.data_ptr()),
              static_cast<const T*>(sum_input.data_ptr()),
              static_cast<const WeightDType*>(weight.data_ptr()),
              static_cast<const float*>(rstd.data_ptr()),
              num_tokens, d_add_row_stride, grad_sum_row_stride,
              grad_norm_row_stride, sum_row_stride);
      return;
    }

    constexpr int DIRECT_VEC4_SIZE = 4;
    constexpr int DIRECT_VEC4_MAX_VEC_PER_THREAD = 2;
    if (hidden_size % DIRECT_VEC4_SIZE == 0 &&
        hidden_size <= DIRECT_VEC4_MAX_VEC_PER_THREAD * block_size * DIRECT_VEC4_SIZE) {
      addRMSNormBackwardRstdDirectVec4Kernel<
          T, WeightDType, DIRECT_VEC4_MAX_VEC_PER_THREAD, ZeroCentered>
          <<<num_blocks, block_size, 0, stream>>>(
              static_cast<T*>(d_add.data_ptr()), partial_dw,
              static_cast<const T*>(grad_sum.data_ptr()),
              static_cast<const T*>(grad_norm.data_ptr()),
              static_cast<const T*>(sum_input.data_ptr()),
              static_cast<const WeightDType*>(weight.data_ptr()),
              static_cast<const float*>(rstd.data_ptr()),
              hidden_size, num_tokens, d_add_row_stride, grad_sum_row_stride,
              grad_norm_row_stride, sum_row_stride);
      return;
    }

  }
  if (vec_hidden <= block_size * 2) {
    addRMSNormBackwardRstdDirectKernel<T, WeightDType, 2, ZeroCentered>
        <<<num_blocks, block_size, 0, stream>>>(
            static_cast<T*>(d_add.data_ptr()), partial_dw,
            static_cast<const T*>(grad_sum.data_ptr()),
            static_cast<const T*>(grad_norm.data_ptr()),
            static_cast<const T*>(sum_input.data_ptr()),
            static_cast<const WeightDType*>(weight.data_ptr()),
            static_cast<const float*>(rstd.data_ptr()),
            hidden_size, num_tokens, d_add_row_stride, grad_sum_row_stride,
            grad_norm_row_stride, sum_row_stride);
    return;
  }
  if (vec_hidden <= block_size * 3) {
    addRMSNormBackwardRstdDirectKernel<T, WeightDType, 3, ZeroCentered>
        <<<num_blocks, block_size, 0, stream>>>(
            static_cast<T*>(d_add.data_ptr()), partial_dw,
            static_cast<const T*>(grad_sum.data_ptr()),
            static_cast<const T*>(grad_norm.data_ptr()),
            static_cast<const T*>(sum_input.data_ptr()),
            static_cast<const WeightDType*>(weight.data_ptr()),
            static_cast<const float*>(rstd.data_ptr()),
            hidden_size, num_tokens, d_add_row_stride, grad_sum_row_stride,
            grad_norm_row_stride, sum_row_stride);
    return;
  }
  if (vec_hidden <= block_size * 4) {
    addRMSNormBackwardRstdDirectKernel<T, WeightDType, 4, ZeroCentered>
        <<<num_blocks, block_size, 0, stream>>>(
            static_cast<T*>(d_add.data_ptr()), partial_dw,
            static_cast<const T*>(grad_sum.data_ptr()),
            static_cast<const T*>(grad_norm.data_ptr()),
            static_cast<const T*>(sum_input.data_ptr()),
            static_cast<const WeightDType*>(weight.data_ptr()),
            static_cast<const float*>(rstd.data_ptr()),
            hidden_size, num_tokens, d_add_row_stride, grad_sum_row_stride,
            grad_norm_row_stride, sum_row_stride);
    return;
  }
  if (vec_hidden <= block_size * 8) {
    addRMSNormBackwardRstdDirectKernel<T, WeightDType, 8, ZeroCentered>
        <<<num_blocks, block_size, 0, stream>>>(
            static_cast<T*>(d_add.data_ptr()), partial_dw,
            static_cast<const T*>(grad_sum.data_ptr()),
            static_cast<const T*>(grad_norm.data_ptr()),
            static_cast<const T*>(sum_input.data_ptr()),
            static_cast<const WeightDType*>(weight.data_ptr()),
            static_cast<const float*>(rstd.data_ptr()),
            hidden_size, num_tokens, d_add_row_stride, grad_sum_row_stride,
            grad_norm_row_stride, sum_row_stride);
    return;
  }
  if (vec_hidden > block_size * 16) {
    throw std::runtime_error("AddRMSNorm backward rstd path hidden size too large");
  }
  addRMSNormBackwardRstdDirectKernel<T, WeightDType, 16, ZeroCentered>
      <<<num_blocks, block_size, 0, stream>>>(
          static_cast<T*>(d_add.data_ptr()), partial_dw,
          static_cast<const T*>(grad_sum.data_ptr()),
          static_cast<const T*>(grad_norm.data_ptr()),
          static_cast<const T*>(sum_input.data_ptr()),
          static_cast<const WeightDType*>(weight.data_ptr()),
          static_cast<const float*>(rstd.data_ptr()),
          hidden_size, num_tokens, d_add_row_stride, grad_sum_row_stride,
          grad_norm_row_stride, sum_row_stride);
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

__global__ void rmsnormReduceDwD128Kernel(
    float* __restrict__ dw, const float* __restrict__ partial_dw,
    const int64_t num_blocks) {
  constexpr int D = 128;
  constexpr int COLS_PER_BLOCK = 32;
  constexpr int ROW_WARPS = 8;
  const int lane = threadIdx.x;
  const int row_warp = threadIdx.y;
  const int col = static_cast<int>(blockIdx.x) * COLS_PER_BLOCK + lane;

  float sum = 0.0f;
  for (int64_t b = row_warp; b < num_blocks; b += ROW_WARPS) {
    sum += partial_dw[b * D + col];
  }

  __shared__ float partial[ROW_WARPS][COLS_PER_BLOCK];
  partial[row_warp][lane] = sum;
  __syncthreads();

  if (row_warp == 0) {
    float total = partial[0][lane];
    #pragma unroll
    for (int i = 1; i < ROW_WARPS; ++i) {
      total += partial[i][lane];
    }
    dw[col] = total;
  }
}

// First-stage row-split reduction, in place: block (bx, by) folds rows
// {by, by + R_SPLIT, by + 2*R_SPLIT, ...} of its column tile into row by.
// Safe because row by's slice is only touched by this block. 4-way unrolled so
// several row loads are in flight per thread.
// Invariant: ALL num_blocks rows of partial_dw are initialized before this
// runs -- the main backward kernel launches exactly partial_dw.size(0) blocks
// and each block unconditionally writes its row (zeros when it owned no input
// rows). The unguarded first-row read (r0 < R_SPLIT < num_blocks/2 by the
// launch condition) relies on that kernel contract, not on the Python
// grid-sizing helper.
__global__ void rmsnormReduceDwRowsKernel(
    float* __restrict__ partial_dw, const int64_t hidden_size,
    const int64_t num_blocks, const int64_t r_split) {
  const int64_t col = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (col >= hidden_size) return;
  const int64_t r0 = blockIdx.y;
  float s0 = partial_dw[r0 * hidden_size + col];
  float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  int64_t r = r0 + r_split;
  for (; r + 3 * r_split < num_blocks; r += 4 * r_split) {
    s0 += partial_dw[r * hidden_size + col];
    s1 += partial_dw[(r + r_split) * hidden_size + col];
    s2 += partial_dw[(r + 2 * r_split) * hidden_size + col];
    s3 += partial_dw[(r + 3 * r_split) * hidden_size + col];
  }
  for (; r < num_blocks; r += r_split) {
    s0 += partial_dw[r * hidden_size + col];
  }
  partial_dw[r0 * hidden_size + col] = (s0 + s1) + (s2 + s3);
}

__device__ __forceinline__ void AccumFloat4(float4& acc, float4 v) {
  acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
}

// float4-vectorized variants of the two kernels above with 8 independent
// accumulators, so up to 8 row loads are in flight per thread (the row walk is
// latency-bound, not bandwidth-bound).
// Single-kernel float4 reduction: block (COLS4 x ROW_PAR) — ROW_PAR thread
// groups walk interleaved row sets with U accumulators each (so up to
// ROW_PAR*U row loads per column are in flight), then a shared-memory fold
// produces the final dw. One launch, fully parallel in both dims.
__global__ void rmsnormReduceDwFusedVec4Kernel(
    float* __restrict__ dw, const float* __restrict__ partial_dw,
    const int64_t hidden_size, const int64_t num_blocks) {
  constexpr int U = 4;
  constexpr int COLS4 = 4;   // narrow column tiles -> many blocks/SMs busy
  constexpr int ROW_PAR = 128;
  const int64_t vec_hidden = hidden_size / 4;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int64_t col4 = blockIdx.x * static_cast<int64_t>(COLS4) + tx;
  __shared__ float4 s_fold[ROW_PAR][COLS4];
  float4 acc[U];
  #pragma unroll
  for (int u = 0; u < U; ++u) acc[u] = make_float4(0.f, 0.f, 0.f, 0.f);
  if (col4 < vec_hidden) {
    const float4* base = reinterpret_cast<const float4*>(partial_dw) + col4;
    int64_t r = ty;
    for (; r + (U - 1) * ROW_PAR < num_blocks; r += U * ROW_PAR) {
      #pragma unroll
      for (int u = 0; u < U; ++u) {
        AccumFloat4(acc[u], base[(r + u * ROW_PAR) * vec_hidden]);
      }
    }
    for (; r < num_blocks; r += ROW_PAR) AccumFloat4(acc[0], base[r * vec_hidden]);
    #pragma unroll
    for (int u = 1; u < U; ++u) AccumFloat4(acc[0], acc[u]);
  }
  s_fold[ty][tx] = acc[0];
  __syncthreads();
  #pragma unroll
  for (int offset = ROW_PAR / 2; offset > 0; offset /= 2) {
    if (ty < offset) AccumFloat4(s_fold[ty][tx], s_fold[ty + offset][tx]);
    __syncthreads();
  }
  if (ty == 0 && col4 < vec_hidden) {
    reinterpret_cast<float4*>(dw)[col4] = s_fold[0][tx];
  }
}

void LaunchRMSNormReduceDw(
    float* dw, float* partial_dw, int64_t hidden_size,
    int64_t num_blocks, cudaStream_t stream) {
  if (hidden_size == 128) {
    dim3 block(32, 8);
    rmsnormReduceDwD128Kernel<<<4, block, 0, stream>>>(dw, partial_dw, num_blocks);
    return;
  }
  const int64_t rt = 256;
  // The final kernel walks rows serially with only col_blocks blocks; for a
  // large partial buffer that serial, latency-bound walk dominates the whole
  // backward. Fold the rows down to <= R_SPLIT in a parallel first stage.
  constexpr int64_t R_SPLIT = 16;
  if (hidden_size % 4 == 0 && HostIsAligned16(partial_dw) &&
      HostIsAligned16(dw)) {
    const int64_t vec_hidden = hidden_size / 4;
    dim3 block(4, 128);
    rmsnormReduceDwFusedVec4Kernel<<<(vec_hidden + 3) / 4, block, 0, stream>>>(
        dw, partial_dw, hidden_size, num_blocks);
    return;
  }
  const int64_t col_blocks = (hidden_size + rt - 1) / rt;
  if (num_blocks > 2 * R_SPLIT) {
    dim3 grid(col_blocks, R_SPLIT);
    rmsnormReduceDwRowsKernel<<<grid, rt, 0, stream>>>(
        partial_dw, hidden_size, num_blocks, R_SPLIT);
    num_blocks = R_SPLIT;
  }
  rmsnormReduceDwKernel<<<col_blocks, rt, 0, stream>>>(
      dw, partial_dw, hidden_size, num_blocks);
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
template<bool ZeroCentered>
void RMSNormBackwardImpl(
    wallx_cuda::TensorView dx,
    wallx_cuda::TensorView dw,
    wallx_cuda::TensorView partial_dw_view,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView weight,
    double epsilon,
    const float* rstd_ptr = nullptr) {

  ASSERT_CHECK(dw.dtype().code == kDLFloat && dw.dtype().bits == 32);
  ASSERT_CHECK(partial_dw_view.dtype().code == kDLFloat &&
               partial_dw_view.dtype().bits == 32);
  ASSERT_CHECK(weight.IsContiguous());
  ASSERT_CHECK(dw.IsContiguous());
  ASSERT_CHECK(partial_dw_view.IsContiguous());
  CheckSameShape(dx, input);
  CheckSameShape(dy, input);
  const RMSNormShape shape = GetRMSNormShape(input);
  const RMSNormShape dx_shape = GetRMSNormShape(dx);
  const RMSNormShape dy_shape = GetRMSNormShape(dy);
  const int64_t hidden_size = shape.hidden_size;
  const int64_t num_tokens = shape.num_tokens;
  ASSERT_CHECK(dx_shape.hidden_size == hidden_size);
  ASSERT_CHECK(dx_shape.num_tokens == num_tokens);
  ASSERT_CHECK(dy_shape.hidden_size == hidden_size);
  ASSERT_CHECK(dy_shape.num_tokens == num_tokens);
  ASSERT_CHECK(hidden_size == weight.size(0));
  ASSERT_CHECK(hidden_size % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(dx_shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(dy_shape.row_stride % GetVecSize(input.dtype()) == 0);
  cudaStream_t stream = static_cast<cudaStream_t>(
      wallx_cuda::CurrentStream(input.device().device_type, input.device().device_id));
  // The Python side owns the persistent-grid size: num_blocks is simply the
  // number of partial_dw rows it allocated (blocks with no assigned rows write
  // zeroed partials, so any num_blocks >= 1 is valid).
  ASSERT_CHECK(partial_dw_view.ndim() == 2);
  const int64_t num_blocks = partial_dw_view.size(0);
  ASSERT_CHECK(num_blocks >= 1);
  ASSERT_CHECK(partial_dw_view.size(1) == hidden_size);
  ASSERT_CHECK(dw.ndim() == 1);
  ASSERT_CHECK(dw.size(0) == hidden_size);
  float* partial_dw = static_cast<float*>(partial_dw_view.data_ptr());

  const bool wide_ok =
      rstd_ptr != nullptr
          ? TryRMSNormBackwardWide<true, ZeroCentered>(
                dx, partial_dw, dy, input, weight, rstd_ptr,
                static_cast<float>(epsilon), hidden_size, num_tokens,
                num_blocks, dx_shape.row_stride, dy_shape.row_stride,
                shape.row_stride, stream)
          : TryRMSNormBackwardWide<false, ZeroCentered>(
                dx, partial_dw, dy, input, weight, nullptr,
                static_cast<float>(epsilon), hidden_size, num_tokens,
                num_blocks, dx_shape.row_stride, dy_shape.row_stride,
                shape.row_stride, stream);
  if (wide_ok) {
    LaunchRMSNormReduceDw(
        static_cast<float*>(dw.data_ptr()), partial_dw, hidden_size, num_blocks,
        stream);
    sync_check_cuda_error();
    return;
  }

  constexpr int MAX_VPT = 8;
  // Each thread runs ceil(vec_hidden / block_size) grid-stride iterations and writes one
  // local_dw[MAX_VPT * VEC_SIZE] slot per iteration; that count must stay <= MAX_VPT or it
  // overflows the per-thread stack array. Grow block_size (hardware cap 1024 threads) to hold
  // the invariant, and reject inputs too large even at the cap (mirrors the rstd path guard).
  const int64_t vec_hidden = hidden_size / GetVecSize(input.dtype());
  int64_t block_size = std::min(hidden_size, 512L);
  block_size = std::min(std::max(block_size, (vec_hidden + MAX_VPT - 1) / MAX_VPT), 1024L);
  ASSERT_CHECK(vec_hidden <= block_size * MAX_VPT);

  if (IsFloat32(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchRMSNormBackwardKernel<float, float, MAX_VPT, ZeroCentered>(
        dx, partial_dw, dy, input, weight, static_cast<float>(epsilon),
        hidden_size, num_tokens, num_blocks, dx_shape.row_stride,
        dy_shape.row_stride, shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat16(weight.dtype()))
    LaunchRMSNormBackwardKernel<half, half, MAX_VPT, ZeroCentered>(
        dx, partial_dw, dy, input, weight, static_cast<float>(epsilon),
        hidden_size, num_tokens, num_blocks, dx_shape.row_stride,
        dy_shape.row_stride, shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchRMSNormBackwardKernel<half, float, MAX_VPT, ZeroCentered>(
        dx, partial_dw, dy, input, weight, static_cast<float>(epsilon),
        hidden_size, num_tokens, num_blocks, dx_shape.row_stride,
        dy_shape.row_stride, shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchRMSNormBackwardKernel<half, __nv_bfloat16, MAX_VPT, ZeroCentered>(
        dx, partial_dw, dy, input, weight, static_cast<float>(epsilon),
        hidden_size, num_tokens, num_blocks, dx_shape.row_stride,
        dy_shape.row_stride, shape.row_stride, block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchRMSNormBackwardKernel<__nv_bfloat16, __nv_bfloat16, MAX_VPT, ZeroCentered>(
        dx, partial_dw, dy, input, weight, static_cast<float>(epsilon),
        hidden_size, num_tokens, num_blocks, dx_shape.row_stride,
        dy_shape.row_stride, shape.row_stride, block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchRMSNormBackwardKernel<__nv_bfloat16, float, MAX_VPT, ZeroCentered>(
        dx, partial_dw, dy, input, weight, static_cast<float>(epsilon),
        hidden_size, num_tokens, num_blocks, dx_shape.row_stride,
        dy_shape.row_stride, shape.row_stride, block_size, stream);
  else throw std::runtime_error("Unsupported data type for RMSNorm backward");

  LaunchRMSNormReduceDw(
      static_cast<float*>(dw.data_ptr()), partial_dw, hidden_size, num_blocks, stream);
  sync_check_cuda_error();
}

void RMSNormBackward(
    wallx_cuda::TensorView dx,
    wallx_cuda::TensorView dw,
    wallx_cuda::TensorView partial_dw_view,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView weight,
    double epsilon) {
  RMSNormBackwardImpl<false>(dx, dw, partial_dw_view, dy, input, weight, epsilon);
}

void RMSNormZeroCenteredBackward(
    wallx_cuda::TensorView dx,
    wallx_cuda::TensorView dw,
    wallx_cuda::TensorView partial_dw_view,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView weight,
    double epsilon) {
  RMSNormBackwardImpl<true>(dx, dw, partial_dw_view, dy, input, weight, epsilon);
}

// Backward that consumes the rstd tensor produced by rmsnorm*_with_rstd; the
// wide path then skips the sum(x^2) reduction. epsilon is still passed so the
// generic fallback (which recomputes rstd, numerically equivalent) works for
// layouts/dtypes the wide path does not cover.
const float* CheckedRstdPtr(const wallx_cuda::TensorView& rstd,
                            const wallx_cuda::TensorView& input) {
  ASSERT_CHECK(rstd.dtype().code == kDLFloat && rstd.dtype().bits == 32);
  ASSERT_CHECK(rstd.IsContiguous());
  ASSERT_CHECK(rstd.ndim() == 1);
  ASSERT_CHECK(rstd.size(0) == GetTensorNumTokens(input));
  return static_cast<const float*>(rstd.data_ptr());
}

void RMSNormBackwardWithRstd(
    wallx_cuda::TensorView dx,
    wallx_cuda::TensorView dw,
    wallx_cuda::TensorView partial_dw_view,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView weight,
    wallx_cuda::TensorView rstd,
    double epsilon) {
  RMSNormBackwardImpl<false>(dx, dw, partial_dw_view, dy, input, weight,
                             epsilon, CheckedRstdPtr(rstd, input));
}

void RMSNormZeroCenteredBackwardWithRstd(
    wallx_cuda::TensorView dx,
    wallx_cuda::TensorView dw,
    wallx_cuda::TensorView partial_dw_view,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView weight,
    wallx_cuda::TensorView rstd,
    double epsilon) {
  RMSNormBackwardImpl<true>(dx, dw, partial_dw_view, dy, input, weight,
                            epsilon, CheckedRstdPtr(rstd, input));
}

template<bool ZeroCentered>
void AddRMSNormBackwardImpl(
    wallx_cuda::TensorView d_add,
    wallx_cuda::TensorView dw,
    wallx_cuda::TensorView partial_dw_view,
    wallx_cuda::TensorView grad_sum,
    wallx_cuda::TensorView grad_norm,
    wallx_cuda::TensorView sum_input,
    wallx_cuda::TensorView weight,
    double epsilon) {

  ASSERT_CHECK(dw.dtype().code == kDLFloat && dw.dtype().bits == 32);
  ASSERT_CHECK(partial_dw_view.dtype().code == kDLFloat &&
               partial_dw_view.dtype().bits == 32);
  ASSERT_CHECK(weight.IsContiguous());
  ASSERT_CHECK(dw.IsContiguous());
  ASSERT_CHECK(partial_dw_view.IsContiguous());
  CheckSameShape(d_add, sum_input);
  CheckSameShape(grad_sum, sum_input);
  CheckSameShape(grad_norm, sum_input);
  const RMSNormShape shape = GetRMSNormShape(sum_input);
  const RMSNormShape d_add_shape = GetRMSNormShape(d_add);
  const RMSNormShape grad_sum_shape = GetRMSNormShape(grad_sum);
  const RMSNormShape grad_norm_shape = GetRMSNormShape(grad_norm);
  const int64_t hidden_size = shape.hidden_size;
  const int64_t num_tokens = shape.num_tokens;
  ASSERT_CHECK(d_add_shape.hidden_size == hidden_size);
  ASSERT_CHECK(d_add_shape.num_tokens == num_tokens);
  ASSERT_CHECK(grad_sum_shape.hidden_size == hidden_size);
  ASSERT_CHECK(grad_sum_shape.num_tokens == num_tokens);
  ASSERT_CHECK(grad_norm_shape.hidden_size == hidden_size);
  ASSERT_CHECK(grad_norm_shape.num_tokens == num_tokens);
  ASSERT_CHECK(hidden_size == weight.size(0));
  ASSERT_CHECK(sum_input.dtype() == d_add.dtype());
  ASSERT_CHECK(sum_input.dtype() == grad_sum.dtype());
  ASSERT_CHECK(sum_input.dtype() == grad_norm.dtype());
  ASSERT_CHECK(hidden_size % GetVecSize(sum_input.dtype()) == 0);
  ASSERT_CHECK(shape.row_stride % GetVecSize(sum_input.dtype()) == 0);
  ASSERT_CHECK(d_add_shape.row_stride % GetVecSize(sum_input.dtype()) == 0);
  ASSERT_CHECK(grad_sum_shape.row_stride % GetVecSize(sum_input.dtype()) == 0);
  ASSERT_CHECK(grad_norm_shape.row_stride % GetVecSize(sum_input.dtype()) == 0);

  cudaStream_t stream = static_cast<cudaStream_t>(
      wallx_cuda::CurrentStream(sum_input.device().device_type, sum_input.device().device_id));
  // LaunchAddRMSNormBackwardKernel instantiates the persistent kernel with MAX_VPT=8; a thread
  // writes one local_dw[MAX_VPT * VEC_SIZE] slot per grid-stride iteration, so keep
  // ceil(vec_hidden / block_size) <= MAX_VPT or it overflows the per-thread stack array. Grow
  // block_size (thread cap 1024) to hold the invariant and reject inputs too large at the cap.
  constexpr int ADD_BWD_MAX_VPT = 8;
  const int64_t vec_hidden = hidden_size / GetVecSize(sum_input.dtype());
  int64_t block_size = std::min(hidden_size, 512L);
  block_size = std::min(std::max(block_size, (vec_hidden + ADD_BWD_MAX_VPT - 1) / ADD_BWD_MAX_VPT), 1024L);
  ASSERT_CHECK(vec_hidden <= block_size * ADD_BWD_MAX_VPT);
  const int sm_count = get_sm_count();
  const int64_t num_blocks = std::min(static_cast<int64_t>(sm_count) * 2, num_tokens);
  ASSERT_CHECK(partial_dw_view.ndim() == 2);
  ASSERT_CHECK(partial_dw_view.size(0) == num_blocks);
  ASSERT_CHECK(partial_dw_view.size(1) == hidden_size);
  ASSERT_CHECK(dw.ndim() == 1);
  ASSERT_CHECK(dw.size(0) == hidden_size);
  float* partial_dw = static_cast<float*>(partial_dw_view.data_ptr());

  if (IsFloat32(sum_input.dtype()) && IsFloat32(weight.dtype()))
    LaunchAddRMSNormBackwardKernel<float, float, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight,
        static_cast<float>(epsilon), hidden_size, num_tokens, num_blocks,
        d_add_shape.row_stride, grad_sum_shape.row_stride,
        grad_norm_shape.row_stride, shape.row_stride, block_size, stream);
  else if (IsFloat16(sum_input.dtype()) && IsFloat16(weight.dtype()))
    LaunchAddRMSNormBackwardKernel<half, half, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight,
        static_cast<float>(epsilon), hidden_size, num_tokens, num_blocks,
        d_add_shape.row_stride, grad_sum_shape.row_stride,
        grad_norm_shape.row_stride, shape.row_stride, block_size, stream);
  else if (IsFloat16(sum_input.dtype()) && IsFloat32(weight.dtype()))
    LaunchAddRMSNormBackwardKernel<half, float, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight,
        static_cast<float>(epsilon), hidden_size, num_tokens, num_blocks,
        d_add_shape.row_stride, grad_sum_shape.row_stride,
        grad_norm_shape.row_stride, shape.row_stride, block_size, stream);
  else if (IsFloat16(sum_input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchAddRMSNormBackwardKernel<half, __nv_bfloat16, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight,
        static_cast<float>(epsilon), hidden_size, num_tokens, num_blocks,
        d_add_shape.row_stride, grad_sum_shape.row_stride,
        grad_norm_shape.row_stride, shape.row_stride, block_size, stream);
  else if (IsBFloat16(sum_input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchAddRMSNormBackwardKernel<__nv_bfloat16, __nv_bfloat16, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight,
        static_cast<float>(epsilon), hidden_size, num_tokens, num_blocks,
        d_add_shape.row_stride, grad_sum_shape.row_stride,
        grad_norm_shape.row_stride, shape.row_stride, block_size, stream);
  else if (IsBFloat16(sum_input.dtype()) && IsFloat32(weight.dtype()))
    LaunchAddRMSNormBackwardKernel<__nv_bfloat16, float, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight,
        static_cast<float>(epsilon), hidden_size, num_tokens, num_blocks,
        d_add_shape.row_stride, grad_sum_shape.row_stride,
        grad_norm_shape.row_stride, shape.row_stride, block_size, stream);
  else throw std::runtime_error("Unsupported data type for AddRMSNorm backward");

  LaunchRMSNormReduceDw(
      static_cast<float*>(dw.data_ptr()), partial_dw, hidden_size, num_blocks, stream);
  sync_check_cuda_error();
}

void AddRMSNormBackward(
    wallx_cuda::TensorView d_add,
    wallx_cuda::TensorView dw,
    wallx_cuda::TensorView partial_dw_view,
    wallx_cuda::TensorView grad_sum,
    wallx_cuda::TensorView grad_norm,
    wallx_cuda::TensorView sum_input,
    wallx_cuda::TensorView weight,
    double epsilon) {
  AddRMSNormBackwardImpl<false>(
      d_add, dw, partial_dw_view, grad_sum, grad_norm, sum_input, weight, epsilon);
}

void AddRMSNormZeroCenteredBackward(
    wallx_cuda::TensorView d_add,
    wallx_cuda::TensorView dw,
    wallx_cuda::TensorView partial_dw_view,
    wallx_cuda::TensorView grad_sum,
    wallx_cuda::TensorView grad_norm,
    wallx_cuda::TensorView sum_input,
    wallx_cuda::TensorView weight,
    double epsilon) {
  AddRMSNormBackwardImpl<true>(
      d_add, dw, partial_dw_view, grad_sum, grad_norm, sum_input, weight, epsilon);
}

template<bool ZeroCentered>
void AddRMSNormBackwardWithRstdImpl(
    wallx_cuda::TensorView d_add,
    wallx_cuda::TensorView dw,
    wallx_cuda::TensorView partial_dw_view,
    wallx_cuda::TensorView grad_sum,
    wallx_cuda::TensorView grad_norm,
    wallx_cuda::TensorView sum_input,
    wallx_cuda::TensorView weight,
    wallx_cuda::TensorView rstd) {

  ASSERT_CHECK(dw.dtype().code == kDLFloat && dw.dtype().bits == 32);
  ASSERT_CHECK(partial_dw_view.dtype().code == kDLFloat &&
               partial_dw_view.dtype().bits == 32);
  ASSERT_CHECK(rstd.dtype().code == kDLFloat && rstd.dtype().bits == 32);
  ASSERT_CHECK(weight.IsContiguous());
  ASSERT_CHECK(dw.IsContiguous());
  ASSERT_CHECK(partial_dw_view.IsContiguous());
  ASSERT_CHECK(rstd.IsContiguous());
  CheckSameShape(d_add, sum_input);
  CheckSameShape(grad_sum, sum_input);
  CheckSameShape(grad_norm, sum_input);
  const RMSNormShape shape = GetRMSNormShape(sum_input);
  const RMSNormShape d_add_shape = GetRMSNormShape(d_add);
  const RMSNormShape grad_sum_shape = GetRMSNormShape(grad_sum);
  const RMSNormShape grad_norm_shape = GetRMSNormShape(grad_norm);
  const int64_t hidden_size = shape.hidden_size;
  const int64_t num_tokens = shape.num_tokens;
  ASSERT_CHECK(rstd.ndim() == 1);
  ASSERT_CHECK(rstd.size(0) == num_tokens);
  ASSERT_CHECK(d_add_shape.hidden_size == hidden_size);
  ASSERT_CHECK(d_add_shape.num_tokens == num_tokens);
  ASSERT_CHECK(grad_sum_shape.hidden_size == hidden_size);
  ASSERT_CHECK(grad_sum_shape.num_tokens == num_tokens);
  ASSERT_CHECK(grad_norm_shape.hidden_size == hidden_size);
  ASSERT_CHECK(grad_norm_shape.num_tokens == num_tokens);
  ASSERT_CHECK(hidden_size == weight.size(0));
  ASSERT_CHECK(sum_input.dtype() == d_add.dtype());
  ASSERT_CHECK(sum_input.dtype() == grad_sum.dtype());
  ASSERT_CHECK(sum_input.dtype() == grad_norm.dtype());
  ASSERT_CHECK(hidden_size % GetVecSize(sum_input.dtype()) == 0);
  ASSERT_CHECK(shape.row_stride % GetVecSize(sum_input.dtype()) == 0);
  ASSERT_CHECK(d_add_shape.row_stride % GetVecSize(sum_input.dtype()) == 0);
  ASSERT_CHECK(grad_sum_shape.row_stride % GetVecSize(sum_input.dtype()) == 0);
  ASSERT_CHECK(grad_norm_shape.row_stride % GetVecSize(sum_input.dtype()) == 0);

  cudaStream_t stream = static_cast<cudaStream_t>(
      wallx_cuda::CurrentStream(sum_input.device().device_type, sum_input.device().device_id));
  const int64_t block_size = std::min(hidden_size, 512L);
  // Grid size is owned by the Python side via the partial_dw allocation
  // (blocks with no assigned rows write zeroed partials).
  ASSERT_CHECK(partial_dw_view.ndim() == 2);
  const int64_t num_blocks = partial_dw_view.size(0);
  ASSERT_CHECK(num_blocks >= 1);
  ASSERT_CHECK(partial_dw_view.size(1) == hidden_size);
  ASSERT_CHECK(dw.ndim() == 1);
  ASSERT_CHECK(dw.size(0) == hidden_size);
  float* partial_dw = static_cast<float*>(partial_dw_view.data_ptr());

  // Wide fast path: shares the plain wide backward's kernel, row partition and
  // reduction order, keeping fused dw bit-identical to the materialized path.
  if (TryAddRMSNormBackwardWide<ZeroCentered>(
          d_add, partial_dw, grad_sum, grad_norm, sum_input, weight,
          static_cast<const float*>(rstd.data_ptr()), hidden_size, num_tokens,
          num_blocks, d_add_shape.row_stride, grad_sum_shape.row_stride,
          grad_norm_shape.row_stride, shape.row_stride, stream)) {
    LaunchRMSNormReduceDw(
        static_cast<float*>(dw.data_ptr()), partial_dw, hidden_size, num_blocks,
        stream);
    sync_check_cuda_error();
    return;
  }

  if (IsFloat32(sum_input.dtype()) && IsFloat32(weight.dtype()))
    LaunchAddRMSNormBackwardWithRstdKernel<float, float, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight, rstd,
        hidden_size, num_tokens, num_blocks, d_add_shape.row_stride,
        grad_sum_shape.row_stride, grad_norm_shape.row_stride,
        shape.row_stride, block_size, stream);
  else if (IsFloat16(sum_input.dtype()) && IsFloat16(weight.dtype()))
    LaunchAddRMSNormBackwardWithRstdKernel<half, half, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight, rstd,
        hidden_size, num_tokens, num_blocks, d_add_shape.row_stride,
        grad_sum_shape.row_stride, grad_norm_shape.row_stride,
        shape.row_stride, block_size, stream);
  else if (IsFloat16(sum_input.dtype()) && IsFloat32(weight.dtype()))
    LaunchAddRMSNormBackwardWithRstdKernel<half, float, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight, rstd,
        hidden_size, num_tokens, num_blocks, d_add_shape.row_stride,
        grad_sum_shape.row_stride, grad_norm_shape.row_stride,
        shape.row_stride, block_size, stream);
  else if (IsFloat16(sum_input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchAddRMSNormBackwardWithRstdKernel<half, __nv_bfloat16, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight, rstd,
        hidden_size, num_tokens, num_blocks, d_add_shape.row_stride,
        grad_sum_shape.row_stride, grad_norm_shape.row_stride,
        shape.row_stride, block_size, stream);
  else if (IsBFloat16(sum_input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchAddRMSNormBackwardWithRstdKernel<__nv_bfloat16, __nv_bfloat16, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight, rstd,
        hidden_size, num_tokens, num_blocks, d_add_shape.row_stride,
        grad_sum_shape.row_stride, grad_norm_shape.row_stride,
        shape.row_stride, block_size, stream);
  else if (IsBFloat16(sum_input.dtype()) && IsFloat32(weight.dtype()))
    LaunchAddRMSNormBackwardWithRstdKernel<__nv_bfloat16, float, ZeroCentered>(
        d_add, partial_dw, grad_sum, grad_norm, sum_input, weight, rstd,
        hidden_size, num_tokens, num_blocks, d_add_shape.row_stride,
        grad_sum_shape.row_stride, grad_norm_shape.row_stride,
        shape.row_stride, block_size, stream);
  else throw std::runtime_error("Unsupported data type for AddRMSNorm backward rstd");

  LaunchRMSNormReduceDw(
      static_cast<float*>(dw.data_ptr()), partial_dw, hidden_size, num_blocks, stream);
  sync_check_cuda_error();
}

void AddRMSNormBackwardWithRstd(
    wallx_cuda::TensorView d_add,
    wallx_cuda::TensorView dw,
    wallx_cuda::TensorView partial_dw_view,
    wallx_cuda::TensorView grad_sum,
    wallx_cuda::TensorView grad_norm,
    wallx_cuda::TensorView sum_input,
    wallx_cuda::TensorView weight,
    wallx_cuda::TensorView rstd) {
  AddRMSNormBackwardWithRstdImpl<false>(
      d_add, dw, partial_dw_view, grad_sum, grad_norm, sum_input, weight, rstd);
}

void AddRMSNormZeroCenteredBackwardWithRstd(
    wallx_cuda::TensorView d_add,
    wallx_cuda::TensorView dw,
    wallx_cuda::TensorView partial_dw_view,
    wallx_cuda::TensorView grad_sum,
    wallx_cuda::TensorView grad_norm,
    wallx_cuda::TensorView sum_input,
    wallx_cuda::TensorView weight,
    wallx_cuda::TensorView rstd) {
  AddRMSNormBackwardWithRstdImpl<true>(
      d_add, dw, partial_dw_view, grad_sum, grad_norm, sum_input, weight, rstd);
}











// ======================== Gated RMSNorm ========================

__device__ __forceinline__ int64_t TensorRowOffset(
    int64_t row, const TensorRowLayout layout) {
  int64_t offset = 0;
  int64_t remaining = row;
  for (int i = layout.ndim - 2; i >= 0; --i) {
    const int64_t index = remaining % layout.sizes[i];
    remaining /= layout.sizes[i];
    offset += index * layout.strides[i];
  }
  return offset;
}

__device__ __forceinline__ int64_t TensorRowOffsetD128(
    int64_t row, const TensorRowLayout layout) {
  if (layout.ndim == 2 && layout.strides[1] == 1) {
    return row * layout.strides[0];
  }
  if (layout.ndim == 4 && layout.strides[3] == 1 &&
      layout.strides[1] == layout.sizes[2] * layout.strides[2] &&
      layout.strides[0] == layout.sizes[1] * layout.strides[1]) {
    return row * layout.strides[2];
  }
  if (layout.ndim == 4 && layout.sizes[3] == 128 && layout.strides[3] == 1 &&
      layout.strides[2] == 128 &&
      layout.strides[0] == layout.sizes[1] * layout.strides[1]) {
    if (layout.sizes[2] == 32) {
      return (row >> 5) * layout.strides[1] + (row & 31) * 128;
    }
    return (row / layout.sizes[2]) * layout.strides[1] +
           (row % layout.sizes[2]) * layout.strides[2];
  }
  return TensorRowOffset(row, layout);
}

constexpr int kD128LayoutGeneric = 0;
constexpr int kD128LayoutLinear = 1;
constexpr int kD128LayoutH32Padded = 2;

int GetD128LayoutKind(const TensorRowLayout& layout) {
  if (layout.ndim == 2 && layout.strides[1] == 1 && layout.strides[0] == 128) {
    return kD128LayoutLinear;
  }
  if (layout.ndim == 4 && layout.strides[3] == 1 && layout.strides[2] == 128 &&
      layout.strides[1] == layout.sizes[2] * layout.strides[2] &&
      layout.strides[0] == layout.sizes[1] * layout.strides[1]) {
    return kD128LayoutLinear;
  }
  if (layout.ndim == 4 && layout.sizes[2] == 32 && layout.sizes[3] == 128 &&
      layout.strides[3] == 1 && layout.strides[2] == 128 &&
      layout.strides[0] == layout.sizes[1] * layout.strides[1]) {
    return kD128LayoutH32Padded;
  }
  return kD128LayoutGeneric;
}

__device__ __forceinline__ int64_t TensorRowOffsetD128(
    int64_t row, const TensorRowLayout layout, const int layout_kind) {
  if (layout_kind == kD128LayoutLinear) {
    return row * 128;
  }
  if (layout_kind == kD128LayoutH32Padded) {
    return (row >> 5) * layout.strides[1] + (row & 31) * 128;
  }
  return TensorRowOffsetD128(row, layout);
}

template<int LAYOUT_KIND>
__device__ __forceinline__ int64_t TensorRowOffsetD128ByKind(
    int64_t row, const TensorRowLayout layout) {
  if constexpr (LAYOUT_KIND == kD128LayoutLinear) {
    return row * 128;
  } else if constexpr (LAYOUT_KIND == kD128LayoutH32Padded) {
    return (row >> 5) * layout.strides[1] + (row & 31) * 128;
  } else {
    return TensorRowOffsetD128(row, layout);
  }
}

__device__ __forceinline__ float Silu(float x) {
  const float sigmoid = 1.0f / (1.0f + __expf(-x));
  return x * sigmoid;
}

__device__ __forceinline__ void SiluAndGrad(
    float x, float& silu, float& grad) {
  const float sigmoid = 1.0f / (1.0f + __expf(-x));
  silu = x * sigmoid;
  grad = sigmoid * (1.0f + x * (1.0f - sigmoid));
}

template<typename T, typename WeightDType>
__global__ void gatedRMSNormKernel(
    T* output,
    const T* input,
    const T* gate,
    const WeightDType* weight,
    const float epsilon,
    const int64_t hidden_size,
    const int64_t input_row_stride,
    const int64_t output_row_stride,
    const TensorRowLayout gate_layout) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;

    float square_sum = 0.0f;
    __shared__ float inv_rms;
    const int64_t tid = threadIdx.x;
    const int64_t bid = blockIdx.x;
    const int64_t input_offset = bid * input_row_stride;
    const int64_t output_offset = bid * output_row_stride;
    const int64_t gate_offset = TensorRowOffset(bid, gate_layout);
    const int64_t vec_hidden = hidden_size / VEC_SIZE;

    for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
        Vec x_vec = VecTraits<T>::load(input + input_offset + i * VEC_SIZE);
        float x_vals[VEC_SIZE];
        VecToFloats<T>(x_vec, x_vals);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            square_sum += x_vals[j] * x_vals[j];
        }
    }
    square_sum = blockReduceSum(square_sum);
    if (tid == 0) {
        inv_rms = rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
    }
    __syncthreads();

    for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
        Vec x_vec = VecTraits<T>::load(input + input_offset + i * VEC_SIZE);
        Vec gate_vec = VecTraits<T>::load(gate + gate_offset + i * VEC_SIZE);
        float x_vals[VEC_SIZE], gate_vals[VEC_SIZE], w_vals[VEC_SIZE], out_vals[VEC_SIZE];
        VecToFloats<T>(x_vec, x_vals);
        VecToFloats<T>(gate_vec, gate_vals);
        LoadWeightFloats<WeightDType, VEC_SIZE>(weight, i * VEC_SIZE, w_vals);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            out_vals[j] = x_vals[j] * inv_rms * w_vals[j] * Silu(gate_vals[j]);
        }
        VecTraits<T>::store(output + output_offset + i * VEC_SIZE, FloatsToVec<T>(out_vals));
    }
}

template<typename T, typename WeightDType, int WARPS_PER_BLOCK, int GATE_LAYOUT_KIND>
__global__ void gatedRMSNormD128ForwardKernel(
    T* __restrict__ output,
    float* __restrict__ rstd,
    const T* __restrict__ input,
    const T* __restrict__ gate,
    const WeightDType* __restrict__ weight,
    const float epsilon,
    const int64_t num_tokens,
    const int64_t input_row_stride,
    const int64_t output_row_stride,
    const TensorRowLayout gate_layout,
    const int gate_layout_kind) {
  constexpr int D = 128;
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int col_base = lane * 4;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;
  if (row >= num_tokens) return;

  const int64_t input_off = row * input_row_stride;
  const int64_t output_off = row * output_row_stride;
  const int64_t gate_off =
      TensorRowOffsetD128ByKind<GATE_LAYOUT_KIND>(row, gate_layout);

  float x_vals[4];
  float gate_vals[4];
  float w_vals[4];
  Load4Floats<T>(input + input_off + col_base, x_vals);
  Load4Floats<T>(gate + gate_off + col_base, gate_vals);
  Load4Floats<WeightDType>(weight + col_base, w_vals);

  float sq = 0.0f;
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    sq += x_vals[k] * x_vals[k];
  }

  const float row_sq = warpReduceSum(sq);
  float row_rstd = 0.0f;
  if (lane == 0) {
    row_rstd = rsqrtf(row_sq / static_cast<float>(D) + epsilon);
    rstd[row] = row_rstd;
  }
  row_rstd = __shfl_sync(0xffffffff, row_rstd, 0);

  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float out = x_vals[k] * row_rstd * w_vals[k] * Silu(gate_vals[k]);
    x_vals[k] = out;
  }
  Store4Scalars<T>(output + output_off + col_base, x_vals);
}

template<typename T, typename WeightDType, int WARPS_PER_BLOCK, int GATE_LAYOUT_KIND>
__global__ void gatedRMSNormD128ForwardHalfWarpVecKernel(
    T* __restrict__ output,
    float* __restrict__ rstd,
    const T* __restrict__ input,
    const T* __restrict__ gate,
    const WeightDType* __restrict__ weight,
    const float epsilon,
    const int64_t num_tokens,
    const int64_t input_row_stride,
    const int64_t output_row_stride,
    const TensorRowLayout gate_layout,
    const int gate_layout_kind) {
  constexpr int D = 128;
  constexpr int LANES_PER_ROW = 16;
  constexpr int ROWS_PER_WARP = 2;
  constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int sub_lane = lane & (LANES_PER_ROW - 1);
  const int row_group = lane >> 4;
  const int col_base0 = sub_lane * 4;
  const int col_base1 = col_base0 + 64;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK +
      warp_id * ROWS_PER_WARP + row_group;

  __shared__ float shared_weight[D];
  if (threadIdx.x < D) {
    shared_weight[threadIdx.x] = ScalarToFloat(weight[threadIdx.x]);
  }
  __syncthreads();

  if (row >= num_tokens) return;

  const int64_t input_off = row * input_row_stride;
  const int64_t output_off = row * output_row_stride;
  const int64_t gate_off =
      TensorRowOffsetD128ByKind<GATE_LAYOUT_KIND>(row, gate_layout);

  float x0[4], x1[4];
  float g0[4], g1[4];
  float w0[4], w1[4];
  Load4Floats<T>(input + input_off + col_base0, x0);
  Load4Floats<T>(input + input_off + col_base1, x1);
  Load4Floats<T>(gate + gate_off + col_base0, g0);
  Load4Floats<T>(gate + gate_off + col_base1, g1);
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    w0[k] = shared_weight[col_base0 + k];
    w1[k] = shared_weight[col_base1 + k];
  }

  float sq = 0.0f;
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    sq += x0[k] * x0[k] + x1[k] * x1[k];
  }

  // A warp covers ROWS_PER_WARP rows; the other row-group may have exited at the
  // num_tokens boundary, so the shuffle mask must name only this row-group's lanes
  // (a full 0xffffffff mask naming exited lanes is UB).
  const unsigned group_mask = 0xffffu << (row_group * LANES_PER_ROW);
  const float row_sq = subwarpReduceSum<LANES_PER_ROW>(sq, group_mask);
  float row_rstd = 0.0f;
  if (sub_lane == 0) {
    row_rstd = rsqrtf(row_sq / static_cast<float>(D) + epsilon);
    rstd[row] = row_rstd;
  }
  row_rstd = __shfl_sync(group_mask, row_rstd, 0, LANES_PER_ROW);

  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    x0[k] = x0[k] * row_rstd * w0[k] * Silu(g0[k]);
    x1[k] = x1[k] * row_rstd * w1[k] * Silu(g1[k]);
  }
  Store4Scalars<T>(output + output_off + col_base0, x0);
  Store4Scalars<T>(output + output_off + col_base1, x1);
}

template<typename T, typename WeightDType, int WARPS_PER_BLOCK, int GATE_LAYOUT_KIND>
__global__ void gatedRMSNormD128ForwardQuarterWarpVecKernel(
    T* __restrict__ output,
    float* __restrict__ rstd,
    const T* __restrict__ input,
    const T* __restrict__ gate,
    const WeightDType* __restrict__ weight,
    const float epsilon,
    const int64_t num_tokens,
    const int64_t input_row_stride,
    const int64_t output_row_stride,
    const TensorRowLayout gate_layout,
    const int gate_layout_kind) {
  constexpr int D = 128;
  constexpr int LANES_PER_ROW = 8;
  constexpr int ROWS_PER_WARP = 4;
  constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int sub_lane = lane & (LANES_PER_ROW - 1);
  const int row_group = lane >> 3;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK +
      warp_id * ROWS_PER_WARP + row_group;
  if (row >= num_tokens) return;

  const int64_t input_off = row * input_row_stride;
  const int64_t output_off = row * output_row_stride;
  const int64_t gate_off =
      TensorRowOffsetD128ByKind<GATE_LAYOUT_KIND>(row, gate_layout);

  float x_vals[4][4];
  float sq = 0.0f;
  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int col_base = q * 32 + sub_lane * 4;
    Load4Floats<T>(input + input_off + col_base, x_vals[q]);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      sq += x_vals[q][k] * x_vals[q][k];
    }
  }

  // Same partial-warp hazard as the HalfWarp kernel: 4 row-groups share a warp and
  // groups past num_tokens have exited, so restrict shuffles to this group's lanes.
  const unsigned group_mask = 0xffu << (row_group * LANES_PER_ROW);
  const float row_sq = subwarpReduceSum<LANES_PER_ROW>(sq, group_mask);
  float row_rstd = 0.0f;
  if (sub_lane == 0) {
    row_rstd = rsqrtf(row_sq / static_cast<float>(D) + epsilon);
    rstd[row] = row_rstd;
  }
  row_rstd = __shfl_sync(group_mask, row_rstd, 0, LANES_PER_ROW);

  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int col_base = q * 32 + sub_lane * 4;
    float gate_vals[4];
    float w_vals[4];
    Load4Floats<T>(gate + gate_off + col_base, gate_vals);
    Load4Floats<WeightDType>(weight + col_base, w_vals);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      x_vals[q][k] =
          x_vals[q][k] * row_rstd * w_vals[k] * Silu(gate_vals[k]);
    }
    Store4Scalars<T>(output + output_off + col_base, x_vals[q]);
  }
}

template<
    typename T, typename WeightDType, int WARPS_PER_BLOCK,
    int DGATE_LAYOUT_KIND, int GATE_LAYOUT_KIND>
__global__ void gatedRMSNormD128BackwardKernel(
    T* __restrict__ dx,
    T* __restrict__ dgate,
    float* __restrict__ partial_dw,
    const T* __restrict__ dy,
    const T* __restrict__ input,
    const T* __restrict__ gate,
    const float* __restrict__ rstd,
    const WeightDType* __restrict__ weight,
    const int64_t num_tokens,
    const int64_t dx_row_stride,
    const int64_t dy_row_stride,
    const int64_t input_row_stride,
    const TensorRowLayout dgate_layout,
    const TensorRowLayout gate_layout,
    const int dgate_layout_kind,
    const int gate_layout_kind) {
  constexpr int D = 128;
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int col_base = lane * 4;
  const int64_t num_blocks = gridDim.x;
  const int64_t rows_per_block = (num_tokens + num_blocks - 1) / num_blocks;
  const int64_t row_start = static_cast<int64_t>(blockIdx.x) * rows_per_block;
  const int64_t row_end =
      (row_start + rows_per_block < num_tokens) ? row_start + rows_per_block : num_tokens;

  float local_dw[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float w_vals[4];
  Load4Floats<WeightDType>(weight + col_base, w_vals);

  for (int64_t row = row_start + warp_id; row < row_end; row += WARPS_PER_BLOCK) {
    const int64_t dx_off = row * dx_row_stride;
    const int64_t dy_off = row * dy_row_stride;
    const int64_t input_off = row * input_row_stride;
    const int64_t dgate_off =
        TensorRowOffsetD128ByKind<DGATE_LAYOUT_KIND>(row, dgate_layout);
    const int64_t gate_off =
        TensorRowOffsetD128ByKind<GATE_LAYOUT_KIND>(row, gate_layout);
    const float row_rstd = rstd[row];

    float x_vals[4];
    float gate_vals[4];
    float dy_vals[4];
    float xhat_vals[4];
    float grad_vals[4];
    float wdy_vals[4];
    float c1_sum = 0.0f;
    Load4Floats<T>(input + input_off + col_base, x_vals);
    Load4Floats<T>(gate + gate_off + col_base, gate_vals);
    Load4Floats<T>(dy + dy_off + col_base, dy_vals);

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float x_val = x_vals[k];
      const float gate_val = gate_vals[k];
      const float dy_val = dy_vals[k];
      const float w_val = w_vals[k];
      float gate_silu;
      float gate_grad;
      SiluAndGrad(gate_val, gate_silu, gate_grad);

      const float xhat = x_val * row_rstd;
      const float wdy = dy_val * gate_silu * w_val;
      xhat_vals[k] = xhat;
      dy_vals[k] = dy_val;
      w_vals[k] = w_val;
      grad_vals[k] = gate_grad;
      wdy_vals[k] = wdy;
      c1_sum += xhat * wdy;
      local_dw[k] += dy_val * xhat * gate_silu;
    }

    const float c1 = __shfl_sync(0xffffffff, warpReduceSum(c1_sum), 0) /
                     static_cast<float>(D);

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float dx_val = (wdy_vals[k] - xhat_vals[k] * c1) * row_rstd;
      const float dgate_val = dy_vals[k] * xhat_vals[k] * w_vals[k] * grad_vals[k];
      x_vals[k] = dx_val;
      gate_vals[k] = dgate_val;
    }
    Store4Scalars<T>(dx + dx_off + col_base, x_vals);
    Store4Scalars<T>(dgate + dgate_off + col_base, gate_vals);
  }

  __shared__ float shared_dw[WARPS_PER_BLOCK][32][4];
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    shared_dw[warp_id][lane][k] = local_dw[k];
  }
  __syncthreads();

  if (warp_id == 0) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      float sum = 0.0f;
      #pragma unroll
      for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
        sum += shared_dw[w][lane][k];
      }
      const int col = col_base + k;
      partial_dw[static_cast<int64_t>(blockIdx.x) * D + col] = sum;
    }
  }
}

template<typename T, typename WeightDType>
void LaunchGatedRMSNormKernel(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView weight,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t input_row_stride,
    int64_t output_row_stride,
    TensorRowLayout gate_layout,
    int64_t block_size,
    cudaStream_t stream) {
  gatedRMSNormKernel<T, WeightDType><<<num_tokens, block_size, 0, stream>>>(
      static_cast<T*>(output.data_ptr()),
      static_cast<const T*>(input.data_ptr()),
      static_cast<const T*>(gate.data_ptr()),
      static_cast<const WeightDType*>(weight.data_ptr()),
      epsilon, hidden_size, input_row_stride, output_row_stride, gate_layout);
}

template<typename T, typename WeightDType, int WARPS_PER_BLOCK, int GATE_LAYOUT_KIND>
void LaunchGatedRMSNormD128ForwardKernel(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView rstd,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView weight,
    float epsilon,
    int64_t num_tokens,
    int64_t input_row_stride,
    int64_t output_row_stride,
    TensorRowLayout gate_layout,
    int gate_layout_kind,
    cudaStream_t stream) {
  const int64_t blocks = (num_tokens + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  gatedRMSNormD128ForwardKernel<T, WeightDType, WARPS_PER_BLOCK, GATE_LAYOUT_KIND>
      <<<blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
          static_cast<T*>(output.data_ptr()),
          static_cast<float*>(rstd.data_ptr()),
          static_cast<const T*>(input.data_ptr()),
          static_cast<const T*>(gate.data_ptr()),
          static_cast<const WeightDType*>(weight.data_ptr()),
          epsilon, num_tokens, input_row_stride, output_row_stride, gate_layout,
          gate_layout_kind);
}

template<typename T, typename WeightDType, int WARPS_PER_BLOCK, int GATE_LAYOUT_KIND>
void LaunchGatedRMSNormD128ForwardQuarterWarpVecKernel(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView rstd,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView weight,
    float epsilon,
    int64_t num_tokens,
    int64_t input_row_stride,
    int64_t output_row_stride,
    TensorRowLayout gate_layout,
    int gate_layout_kind,
    cudaStream_t stream) {
  constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * 4;
  const int64_t blocks = (num_tokens + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  gatedRMSNormD128ForwardQuarterWarpVecKernel<
      T, WeightDType, WARPS_PER_BLOCK, GATE_LAYOUT_KIND>
      <<<blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
          static_cast<T*>(output.data_ptr()),
          static_cast<float*>(rstd.data_ptr()),
          static_cast<const T*>(input.data_ptr()),
          static_cast<const T*>(gate.data_ptr()),
          static_cast<const WeightDType*>(weight.data_ptr()),
          epsilon, num_tokens, input_row_stride, output_row_stride, gate_layout,
          gate_layout_kind);
}

template<typename T, typename WeightDType, int WARPS_PER_BLOCK, int GATE_LAYOUT_KIND>
void LaunchGatedRMSNormD128ForwardHalfWarpVecKernel(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView rstd,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView weight,
    float epsilon,
    int64_t num_tokens,
    int64_t input_row_stride,
    int64_t output_row_stride,
    TensorRowLayout gate_layout,
    int gate_layout_kind,
    cudaStream_t stream) {
  constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * 2;
  const int64_t blocks = (num_tokens + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  gatedRMSNormD128ForwardHalfWarpVecKernel<
      T, WeightDType, WARPS_PER_BLOCK, GATE_LAYOUT_KIND>
      <<<blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
          static_cast<T*>(output.data_ptr()),
          static_cast<float*>(rstd.data_ptr()),
          static_cast<const T*>(input.data_ptr()),
          static_cast<const T*>(gate.data_ptr()),
          static_cast<const WeightDType*>(weight.data_ptr()),
          epsilon, num_tokens, input_row_stride, output_row_stride, gate_layout,
          gate_layout_kind);
}

template<typename T, typename WeightDType, int GATE_LAYOUT_KIND>
void DispatchGatedRMSNormD128ForwardKernelForLayout(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView rstd,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView weight,
    float epsilon,
    int64_t num_tokens,
    int64_t input_row_stride,
    int64_t output_row_stride,
    TensorRowLayout gate_layout,
    int gate_layout_kind,
    int64_t warps_per_block,
    cudaStream_t stream) {
  if (warps_per_block <= 4) {
    LaunchGatedRMSNormD128ForwardQuarterWarpVecKernel<
        T, WeightDType, 4, GATE_LAYOUT_KIND>(
        output, rstd, input, gate, weight, epsilon, num_tokens,
        input_row_stride, output_row_stride, gate_layout, gate_layout_kind, stream);
  } else if (warps_per_block <= 8) {
    LaunchGatedRMSNormD128ForwardHalfWarpVecKernel<
        T, WeightDType, 8, GATE_LAYOUT_KIND>(
        output, rstd, input, gate, weight, epsilon, num_tokens,
        input_row_stride, output_row_stride, gate_layout, gate_layout_kind, stream);
  } else {
    LaunchGatedRMSNormD128ForwardKernel<T, WeightDType, 16, GATE_LAYOUT_KIND>(
        output, rstd, input, gate, weight, epsilon, num_tokens,
        input_row_stride, output_row_stride, gate_layout, gate_layout_kind, stream);
  }
}

template<typename T, typename WeightDType>
void DispatchGatedRMSNormD128ForwardKernel(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView rstd,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView weight,
    float epsilon,
    int64_t num_tokens,
    int64_t input_row_stride,
    int64_t output_row_stride,
    TensorRowLayout gate_layout,
    int gate_layout_kind,
    int64_t warps_per_block,
    cudaStream_t stream) {
  if (gate_layout_kind == kD128LayoutLinear) {
    DispatchGatedRMSNormD128ForwardKernelForLayout<
        T, WeightDType, kD128LayoutLinear>(
        output, rstd, input, gate, weight, epsilon, num_tokens,
        input_row_stride, output_row_stride, gate_layout, gate_layout_kind,
        warps_per_block, stream);
  } else if (gate_layout_kind == kD128LayoutH32Padded) {
    DispatchGatedRMSNormD128ForwardKernelForLayout<
        T, WeightDType, kD128LayoutH32Padded>(
        output, rstd, input, gate, weight, epsilon, num_tokens,
        input_row_stride, output_row_stride, gate_layout, gate_layout_kind,
        warps_per_block, stream);
  } else {
    DispatchGatedRMSNormD128ForwardKernelForLayout<
        T, WeightDType, kD128LayoutGeneric>(
        output, rstd, input, gate, weight, epsilon, num_tokens,
        input_row_stride, output_row_stride, gate_layout, gate_layout_kind,
        warps_per_block, stream);
  }
}

void GatedRMSNorm(wallx_cuda::TensorView output,
                  wallx_cuda::TensorView input,
                  wallx_cuda::TensorView gate,
                  wallx_cuda::TensorView weight,
                  double epsilon) {
  ASSERT_CHECK(weight.ndim() == 1);
  ASSERT_CHECK(weight.IsContiguous());
  ASSERT_CHECK(input.dtype().code == gate.dtype().code);
  ASSERT_CHECK(input.dtype().bits == gate.dtype().bits);
  CheckSameShape(output, input);
  CheckSameShape(gate, input);
  const RMSNormShape shape = GetRMSNormShape(input);
  const RMSNormShape output_shape = GetRMSNormShape(output);
  const TensorRowLayout gate_layout = GetTensorRowLayout(gate);
  const int64_t hidden_size = shape.hidden_size;
  const int64_t num_tokens = shape.num_tokens;
  ASSERT_CHECK(output_shape.hidden_size == hidden_size);
  ASSERT_CHECK(output_shape.num_tokens == num_tokens);
  ASSERT_CHECK(GetTensorNumTokens(gate) == num_tokens);
  ASSERT_CHECK(gate.size(gate.ndim() - 1) == hidden_size);
  ASSERT_CHECK(hidden_size == weight.size(0));
  const int64_t vec_size = GetVecSize(input.dtype());
  ASSERT_CHECK(hidden_size % vec_size == 0);
  ASSERT_CHECK(shape.row_stride % vec_size == 0);
  ASSERT_CHECK(output_shape.row_stride % vec_size == 0);
  CheckTensorRowLayoutVectorized(gate_layout, vec_size);

  cudaStream_t stream = static_cast<cudaStream_t>(
      wallx_cuda::CurrentStream(input.device().device_type, input.device().device_id));
  const int64_t block_size = std::min(hidden_size / vec_size, 1024L);
  if (IsFloat32(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchGatedRMSNormKernel<float, float>(
        output, input, gate, weight, static_cast<float>(epsilon), hidden_size,
        num_tokens, shape.row_stride, output_shape.row_stride, gate_layout,
        block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat16(weight.dtype()))
    LaunchGatedRMSNormKernel<half, half>(
        output, input, gate, weight, static_cast<float>(epsilon), hidden_size,
        num_tokens, shape.row_stride, output_shape.row_stride, gate_layout,
        block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchGatedRMSNormKernel<half, float>(
        output, input, gate, weight, static_cast<float>(epsilon), hidden_size,
        num_tokens, shape.row_stride, output_shape.row_stride, gate_layout,
        block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchGatedRMSNormKernel<__nv_bfloat16, __nv_bfloat16>(
        output, input, gate, weight, static_cast<float>(epsilon), hidden_size,
        num_tokens, shape.row_stride, output_shape.row_stride, gate_layout,
        block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchGatedRMSNormKernel<__nv_bfloat16, float>(
        output, input, gate, weight, static_cast<float>(epsilon), hidden_size,
        num_tokens, shape.row_stride, output_shape.row_stride, gate_layout,
        block_size, stream);
  else throw std::runtime_error("Unsupported data type for Gated RMSNorm");
  sync_check_cuda_error();
}

void GatedRMSNormWithRstd(wallx_cuda::TensorView output,
                          wallx_cuda::TensorView rstd,
                          wallx_cuda::TensorView input,
                          wallx_cuda::TensorView gate,
                          wallx_cuda::TensorView weight,
                          double epsilon,
                          int64_t warps_per_block) {
  ASSERT_CHECK(weight.ndim() == 1);
  ASSERT_CHECK(weight.IsContiguous());
  ASSERT_CHECK(input.dtype().code == gate.dtype().code);
  ASSERT_CHECK(input.dtype().bits == gate.dtype().bits);
  CheckSameShape(output, input);
  CheckSameShape(gate, input);
  const RMSNormShape shape = GetRMSNormShape(input);
  const RMSNormShape output_shape = GetRMSNormShape(output);
  const TensorRowLayout gate_layout = GetTensorRowLayout(gate);
  const int64_t hidden_size = shape.hidden_size;
  const int64_t num_tokens = shape.num_tokens;
  ASSERT_CHECK(output_shape.hidden_size == hidden_size);
  ASSERT_CHECK(output_shape.num_tokens == num_tokens);
  ASSERT_CHECK(GetTensorNumTokens(gate) == num_tokens);
  ASSERT_CHECK(gate.size(gate.ndim() - 1) == hidden_size);
  ASSERT_CHECK(hidden_size == weight.size(0));
  ASSERT_CHECK(rstd.ndim() == 1);
  ASSERT_CHECK(rstd.size(0) == num_tokens);
  ASSERT_CHECK(rstd.IsContiguous());
  ASSERT_CHECK(rstd.dtype().code == kDLFloat && rstd.dtype().bits == 32);
  const int64_t vec_size = GetVecSize(input.dtype());
  ASSERT_CHECK(hidden_size % vec_size == 0);
  ASSERT_CHECK(shape.row_stride % vec_size == 0);
  ASSERT_CHECK(output_shape.row_stride % vec_size == 0);
  CheckTensorRowLayoutVectorized(gate_layout, vec_size);

  if (hidden_size != 128) {
    GatedRMSNorm(output, input, gate, weight, epsilon);
    return;
  }

  cudaStream_t stream = static_cast<cudaStream_t>(
      wallx_cuda::CurrentStream(input.device().device_type, input.device().device_id));
  const int gate_layout_kind = GetD128LayoutKind(gate_layout);
  if (IsFloat32(input.dtype()) && IsFloat32(weight.dtype()))
    DispatchGatedRMSNormD128ForwardKernel<float, float>(
        output, rstd, input, gate, weight, static_cast<float>(epsilon), num_tokens,
        shape.row_stride, output_shape.row_stride, gate_layout, gate_layout_kind,
        warps_per_block, stream);
  else if (IsFloat16(input.dtype()) && IsFloat16(weight.dtype()))
    DispatchGatedRMSNormD128ForwardKernel<half, half>(
        output, rstd, input, gate, weight, static_cast<float>(epsilon), num_tokens,
        shape.row_stride, output_shape.row_stride, gate_layout, gate_layout_kind,
        warps_per_block, stream);
  else if (IsFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    DispatchGatedRMSNormD128ForwardKernel<half, float>(
        output, rstd, input, gate, weight, static_cast<float>(epsilon), num_tokens,
        shape.row_stride, output_shape.row_stride, gate_layout, gate_layout_kind,
        warps_per_block, stream);
  else if (IsBFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    DispatchGatedRMSNormD128ForwardKernel<__nv_bfloat16, __nv_bfloat16>(
        output, rstd, input, gate, weight, static_cast<float>(epsilon), num_tokens,
        shape.row_stride, output_shape.row_stride, gate_layout, gate_layout_kind,
        warps_per_block, stream);
  else if (IsBFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    DispatchGatedRMSNormD128ForwardKernel<__nv_bfloat16, float>(
        output, rstd, input, gate, weight, static_cast<float>(epsilon), num_tokens,
        shape.row_stride, output_shape.row_stride, gate_layout, gate_layout_kind,
        warps_per_block, stream);
  else throw std::runtime_error("Unsupported data type for Gated RMSNorm");
  sync_check_cuda_error();
}

template<typename T, typename WeightDType, int MAX_VEC_PER_THREAD>
__global__ void gatedRMSNormBackwardPersistentKernel(
    T* __restrict__ dx,
    T* __restrict__ dgate,
    float* __restrict__ partial_dw,
    const T* __restrict__ dy,
    const T* __restrict__ input,
    const T* __restrict__ gate,
    const WeightDType* __restrict__ weight,
    const float epsilon,
    const int64_t hidden_size,
    const int64_t num_tokens,
    const int64_t dx_row_stride,
    const int64_t dy_row_stride,
    const int64_t input_row_stride,
    const TensorRowLayout dgate_layout,
    const TensorRowLayout gate_layout) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;

    const int64_t tid = threadIdx.x;
    const int64_t num_blocks = gridDim.x;
    const int64_t vec_hidden = hidden_size / VEC_SIZE;
    const float inv_hidden = 1.0f / static_cast<float>(hidden_size);

    __shared__ float s_inv_rms;
    __shared__ float s_c;

    float local_dw[MAX_VEC_PER_THREAD * VEC_SIZE];
    #pragma unroll
    for (int j = 0; j < MAX_VEC_PER_THREAD * VEC_SIZE; j++) local_dw[j] = 0.0f;

    for (int64_t row = blockIdx.x; row < num_tokens; row += num_blocks) {
        const int64_t dx_off = row * dx_row_stride;
        const int64_t dy_off = row * dy_row_stride;
        const int64_t input_off = row * input_row_stride;
        const int64_t dgate_off = TensorRowOffset(row, dgate_layout);
        const int64_t gate_off = TensorRowOffset(row, gate_layout);
        float sq = 0.0f;
        float dp = 0.0f;

        for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
            Vec x_vec = VecTraits<T>::load(input + input_off + i * VEC_SIZE);
            Vec dy_vec = VecTraits<T>::load(dy + dy_off + i * VEC_SIZE);
            Vec gate_vec = VecTraits<T>::load(gate + gate_off + i * VEC_SIZE);
            float x_vals[VEC_SIZE], dy_vals[VEC_SIZE], gate_vals[VEC_SIZE], w_vals[VEC_SIZE];
            VecToFloats<T>(x_vec, x_vals);
            VecToFloats<T>(dy_vec, dy_vals);
            VecToFloats<T>(gate_vec, gate_vals);
            LoadWeightFloats<WeightDType, VEC_SIZE>(weight, i * VEC_SIZE, w_vals);
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                const float gate_silu = Silu(gate_vals[j]);
                const float p = dy_vals[j] * w_vals[j] * gate_silu;
                sq += x_vals[j] * x_vals[j];
                dp += p * x_vals[j];
            }
        }
        sq = blockReduceSum(sq);
        __syncthreads();
        dp = blockReduceSum(dp);
        if (tid == 0) {
            const float ve = sq * inv_hidden + epsilon;
            s_inv_rms = rsqrtf(ve);
            s_c = dp / (static_cast<float>(hidden_size) * ve);
        }
        __syncthreads();
        const float irms = s_inv_rms;
        const float cv = s_c;

        int li = 0;
        for (int64_t i = tid; i < vec_hidden; i += blockDim.x, li++) {
            Vec x_vec = VecTraits<T>::load(input + input_off + i * VEC_SIZE);
            Vec dy_vec = VecTraits<T>::load(dy + dy_off + i * VEC_SIZE);
            Vec gate_vec = VecTraits<T>::load(gate + gate_off + i * VEC_SIZE);
            float x_vals[VEC_SIZE], dy_vals[VEC_SIZE], gate_vals[VEC_SIZE], w_vals[VEC_SIZE];
            float dx_vals[VEC_SIZE], dgate_vals[VEC_SIZE];
            VecToFloats<T>(x_vec, x_vals);
            VecToFloats<T>(dy_vec, dy_vals);
            VecToFloats<T>(gate_vec, gate_vals);
            LoadWeightFloats<WeightDType, VEC_SIZE>(weight, i * VEC_SIZE, w_vals);
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                float gate_silu;
                float gate_grad;
                SiluAndGrad(gate_vals[j], gate_silu, gate_grad);
                const float normed = x_vals[j] * irms;
                dx_vals[j] = irms * (dy_vals[j] * w_vals[j] * gate_silu - x_vals[j] * cv);
                dgate_vals[j] = dy_vals[j] * normed * w_vals[j] * gate_grad;
                local_dw[li * VEC_SIZE + j] += dy_vals[j] * normed * gate_silu;
            }
            VecTraits<T>::store(dx + dx_off + i * VEC_SIZE, FloatsToVec<T>(dx_vals));
            VecTraits<T>::store(dgate + dgate_off + i * VEC_SIZE, FloatsToVec<T>(dgate_vals));
        }
    }

    const int64_t poff = blockIdx.x * hidden_size;
    int li = 0;
    for (int64_t i = tid; i < vec_hidden; i += blockDim.x, li++) {
        if constexpr (VEC_SIZE == 4) {
            float4 dw_out;
            dw_out.x = local_dw[li * VEC_SIZE + 0];
            dw_out.y = local_dw[li * VEC_SIZE + 1];
            dw_out.z = local_dw[li * VEC_SIZE + 2];
            dw_out.w = local_dw[li * VEC_SIZE + 3];
            *reinterpret_cast<float4*>(partial_dw + poff + i * VEC_SIZE) = dw_out;
        } else {
            float2 dw_out;
            dw_out.x = local_dw[li * VEC_SIZE + 0];
            dw_out.y = local_dw[li * VEC_SIZE + 1];
            *reinterpret_cast<float2*>(partial_dw + poff + i * VEC_SIZE) = dw_out;
        }
    }
}

template<typename T, typename WeightDType, int MAX_VPT>
void LaunchGatedRMSNormBackwardKernel(
    wallx_cuda::TensorView dx,
    wallx_cuda::TensorView dgate,
    float* partial_dw,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView weight,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t dx_row_stride,
    int64_t dy_row_stride,
    int64_t input_row_stride,
    TensorRowLayout dgate_layout,
    TensorRowLayout gate_layout,
    int64_t block_size,
    cudaStream_t stream) {
  gatedRMSNormBackwardPersistentKernel<T, WeightDType, MAX_VPT>
      <<<num_blocks, block_size, 0, stream>>>(
          static_cast<T*>(dx.data_ptr()),
          static_cast<T*>(dgate.data_ptr()),
          partial_dw,
          static_cast<const T*>(dy.data_ptr()),
          static_cast<const T*>(input.data_ptr()),
          static_cast<const T*>(gate.data_ptr()),
          static_cast<const WeightDType*>(weight.data_ptr()),
          epsilon, hidden_size, num_tokens, dx_row_stride, dy_row_stride,
          input_row_stride, dgate_layout, gate_layout);
}

template<
    typename T, typename WeightDType, int WARPS_PER_BLOCK,
    int DGATE_LAYOUT_KIND, int GATE_LAYOUT_KIND>
void LaunchGatedRMSNormD128BackwardKernel(
    wallx_cuda::TensorView dx,
    wallx_cuda::TensorView dgate,
    float* partial_dw,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView rstd,
    wallx_cuda::TensorView weight,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t dx_row_stride,
    int64_t dy_row_stride,
    int64_t input_row_stride,
    TensorRowLayout dgate_layout,
    TensorRowLayout gate_layout,
    int dgate_layout_kind,
    int gate_layout_kind,
    cudaStream_t stream) {
  gatedRMSNormD128BackwardKernel<
      T, WeightDType, WARPS_PER_BLOCK, DGATE_LAYOUT_KIND, GATE_LAYOUT_KIND>
      <<<num_blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
          static_cast<T*>(dx.data_ptr()),
          static_cast<T*>(dgate.data_ptr()),
          partial_dw,
          static_cast<const T*>(dy.data_ptr()),
          static_cast<const T*>(input.data_ptr()),
          static_cast<const T*>(gate.data_ptr()),
          static_cast<const float*>(rstd.data_ptr()),
          static_cast<const WeightDType*>(weight.data_ptr()),
          num_tokens, dx_row_stride, dy_row_stride, input_row_stride,
          dgate_layout, gate_layout, dgate_layout_kind, gate_layout_kind);
}

template<typename T, typename WeightDType, int DGATE_LAYOUT_KIND, int GATE_LAYOUT_KIND>
void DispatchGatedRMSNormD128BackwardKernelForLayout(
    wallx_cuda::TensorView dx,
    wallx_cuda::TensorView dgate,
    float* partial_dw,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView rstd,
    wallx_cuda::TensorView weight,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t dx_row_stride,
    int64_t dy_row_stride,
    int64_t input_row_stride,
    TensorRowLayout dgate_layout,
    TensorRowLayout gate_layout,
    int dgate_layout_kind,
    int gate_layout_kind,
    int64_t warps_per_block,
    cudaStream_t stream) {
  if (warps_per_block <= 4) {
    LaunchGatedRMSNormD128BackwardKernel<
        T, WeightDType, 4, DGATE_LAYOUT_KIND, GATE_LAYOUT_KIND>(
        dx, dgate, partial_dw, dy, input, gate, rstd, weight, num_tokens,
        num_blocks, dx_row_stride, dy_row_stride, input_row_stride,
        dgate_layout, gate_layout, dgate_layout_kind, gate_layout_kind, stream);
  } else if (warps_per_block <= 8) {
    LaunchGatedRMSNormD128BackwardKernel<
        T, WeightDType, 8, DGATE_LAYOUT_KIND, GATE_LAYOUT_KIND>(
        dx, dgate, partial_dw, dy, input, gate, rstd, weight, num_tokens,
        num_blocks, dx_row_stride, dy_row_stride, input_row_stride,
        dgate_layout, gate_layout, dgate_layout_kind, gate_layout_kind, stream);
  } else {
    LaunchGatedRMSNormD128BackwardKernel<
        T, WeightDType, 16, DGATE_LAYOUT_KIND, GATE_LAYOUT_KIND>(
        dx, dgate, partial_dw, dy, input, gate, rstd, weight, num_tokens,
        num_blocks, dx_row_stride, dy_row_stride, input_row_stride,
        dgate_layout, gate_layout, dgate_layout_kind, gate_layout_kind, stream);
  }
}

template<typename T, typename WeightDType, int DGATE_LAYOUT_KIND>
void DispatchGatedRMSNormD128BackwardKernelForDgateLayout(
    wallx_cuda::TensorView dx,
    wallx_cuda::TensorView dgate,
    float* partial_dw,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView rstd,
    wallx_cuda::TensorView weight,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t dx_row_stride,
    int64_t dy_row_stride,
    int64_t input_row_stride,
    TensorRowLayout dgate_layout,
    TensorRowLayout gate_layout,
    int dgate_layout_kind,
    int gate_layout_kind,
    int64_t warps_per_block,
    cudaStream_t stream) {
  if (gate_layout_kind == kD128LayoutLinear) {
    DispatchGatedRMSNormD128BackwardKernelForLayout<
        T, WeightDType, DGATE_LAYOUT_KIND, kD128LayoutLinear>(
        dx, dgate, partial_dw, dy, input, gate, rstd, weight, num_tokens,
        num_blocks, dx_row_stride, dy_row_stride, input_row_stride,
        dgate_layout, gate_layout, dgate_layout_kind, gate_layout_kind,
        warps_per_block, stream);
  } else if (gate_layout_kind == kD128LayoutH32Padded) {
    DispatchGatedRMSNormD128BackwardKernelForLayout<
        T, WeightDType, DGATE_LAYOUT_KIND, kD128LayoutH32Padded>(
        dx, dgate, partial_dw, dy, input, gate, rstd, weight, num_tokens,
        num_blocks, dx_row_stride, dy_row_stride, input_row_stride,
        dgate_layout, gate_layout, dgate_layout_kind, gate_layout_kind,
        warps_per_block, stream);
  } else {
    DispatchGatedRMSNormD128BackwardKernelForLayout<
        T, WeightDType, DGATE_LAYOUT_KIND, kD128LayoutGeneric>(
        dx, dgate, partial_dw, dy, input, gate, rstd, weight, num_tokens,
        num_blocks, dx_row_stride, dy_row_stride, input_row_stride,
        dgate_layout, gate_layout, dgate_layout_kind, gate_layout_kind,
        warps_per_block, stream);
  }
}

template<typename T, typename WeightDType>
void DispatchGatedRMSNormD128BackwardKernel(
    wallx_cuda::TensorView dx,
    wallx_cuda::TensorView dgate,
    float* partial_dw,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView rstd,
    wallx_cuda::TensorView weight,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t dx_row_stride,
    int64_t dy_row_stride,
    int64_t input_row_stride,
    TensorRowLayout dgate_layout,
    TensorRowLayout gate_layout,
    int dgate_layout_kind,
    int gate_layout_kind,
    int64_t warps_per_block,
    cudaStream_t stream) {
  if (dgate_layout_kind == kD128LayoutLinear) {
    DispatchGatedRMSNormD128BackwardKernelForDgateLayout<
        T, WeightDType, kD128LayoutLinear>(
        dx, dgate, partial_dw, dy, input, gate, rstd, weight, num_tokens,
        num_blocks, dx_row_stride, dy_row_stride, input_row_stride,
        dgate_layout, gate_layout, dgate_layout_kind, gate_layout_kind,
        warps_per_block, stream);
  } else if (dgate_layout_kind == kD128LayoutH32Padded) {
    DispatchGatedRMSNormD128BackwardKernelForDgateLayout<
        T, WeightDType, kD128LayoutH32Padded>(
        dx, dgate, partial_dw, dy, input, gate, rstd, weight, num_tokens,
        num_blocks, dx_row_stride, dy_row_stride, input_row_stride,
        dgate_layout, gate_layout, dgate_layout_kind, gate_layout_kind,
        warps_per_block, stream);
  } else {
    DispatchGatedRMSNormD128BackwardKernelForDgateLayout<
        T, WeightDType, kD128LayoutGeneric>(
        dx, dgate, partial_dw, dy, input, gate, rstd, weight, num_tokens,
        num_blocks, dx_row_stride, dy_row_stride, input_row_stride,
        dgate_layout, gate_layout, dgate_layout_kind, gate_layout_kind,
        warps_per_block, stream);
  }
}

void GatedRMSNormBackward(
    wallx_cuda::TensorView dx,
    wallx_cuda::TensorView dgate,
    wallx_cuda::TensorView dw,
    wallx_cuda::TensorView partial_dw_view,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView weight,
    double epsilon) {
  ASSERT_CHECK(dw.dtype().code == kDLFloat && dw.dtype().bits == 32);
  ASSERT_CHECK(partial_dw_view.dtype().code == kDLFloat &&
               partial_dw_view.dtype().bits == 32);
  ASSERT_CHECK(weight.IsContiguous());
  ASSERT_CHECK(dw.IsContiguous());
  ASSERT_CHECK(partial_dw_view.IsContiguous());
  ASSERT_CHECK(input.dtype().code == dy.dtype().code);
  ASSERT_CHECK(input.dtype().bits == dy.dtype().bits);
  ASSERT_CHECK(input.dtype().code == gate.dtype().code);
  ASSERT_CHECK(input.dtype().bits == gate.dtype().bits);
  ASSERT_CHECK(input.dtype().code == dgate.dtype().code);
  ASSERT_CHECK(input.dtype().bits == dgate.dtype().bits);
  CheckSameShape(dx, input);
  CheckSameShape(dy, input);
  CheckSameShape(gate, input);
  CheckSameShape(dgate, input);
  const RMSNormShape shape = GetRMSNormShape(input);
  const RMSNormShape dx_shape = GetRMSNormShape(dx);
  const RMSNormShape dy_shape = GetRMSNormShape(dy);
  const TensorRowLayout gate_layout = GetTensorRowLayout(gate);
  const TensorRowLayout dgate_layout = GetTensorRowLayout(dgate);
  const int64_t hidden_size = shape.hidden_size;
  const int64_t num_tokens = shape.num_tokens;
  ASSERT_CHECK(dx_shape.hidden_size == hidden_size);
  ASSERT_CHECK(dx_shape.num_tokens == num_tokens);
  ASSERT_CHECK(dy_shape.hidden_size == hidden_size);
  ASSERT_CHECK(dy_shape.num_tokens == num_tokens);
  ASSERT_CHECK(GetTensorNumTokens(gate) == num_tokens);
  ASSERT_CHECK(GetTensorNumTokens(dgate) == num_tokens);
  ASSERT_CHECK(hidden_size == weight.size(0));
  const int64_t vec_size = GetVecSize(input.dtype());
  ASSERT_CHECK(hidden_size % vec_size == 0);
  ASSERT_CHECK(shape.row_stride % vec_size == 0);
  ASSERT_CHECK(dx_shape.row_stride % vec_size == 0);
  ASSERT_CHECK(dy_shape.row_stride % vec_size == 0);
  CheckTensorRowLayoutVectorized(gate_layout, vec_size);
  CheckTensorRowLayoutVectorized(dgate_layout, vec_size);

  cudaStream_t stream = static_cast<cudaStream_t>(
      wallx_cuda::CurrentStream(input.device().device_type, input.device().device_id));
  const int64_t block_size = std::min(hidden_size / vec_size, 1024L);
  ASSERT_CHECK(partial_dw_view.ndim() == 2);
  ASSERT_CHECK(partial_dw_view.size(1) == hidden_size);
  ASSERT_CHECK(dw.ndim() == 1);
  ASSERT_CHECK(dw.size(0) == hidden_size);
  const int64_t num_blocks = partial_dw_view.size(0);
  ASSERT_CHECK(num_blocks > 0);
  ASSERT_CHECK(num_blocks <= num_tokens);
  float* partial_dw = static_cast<float*>(partial_dw_view.data_ptr());
  constexpr int MAX_VPT = 8;
  // Persistent kernel accumulates into local_dw[MAX_VPT * VEC_SIZE]; each thread runs
  // ceil(vec_hidden / block_size) iterations, which must not exceed MAX_VPT.
  ASSERT_CHECK(hidden_size / vec_size <= block_size * MAX_VPT);

  if (IsFloat32(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchGatedRMSNormBackwardKernel<float, float, MAX_VPT>(
        dx, dgate, partial_dw, dy, input, gate, weight, static_cast<float>(epsilon),
        hidden_size, num_tokens, num_blocks, dx_shape.row_stride,
        dy_shape.row_stride, shape.row_stride, dgate_layout, gate_layout,
        block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat16(weight.dtype()))
    LaunchGatedRMSNormBackwardKernel<half, half, MAX_VPT>(
        dx, dgate, partial_dw, dy, input, gate, weight, static_cast<float>(epsilon),
        hidden_size, num_tokens, num_blocks, dx_shape.row_stride,
        dy_shape.row_stride, shape.row_stride, dgate_layout, gate_layout,
        block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchGatedRMSNormBackwardKernel<half, float, MAX_VPT>(
        dx, dgate, partial_dw, dy, input, gate, weight, static_cast<float>(epsilon),
        hidden_size, num_tokens, num_blocks, dx_shape.row_stride,
        dy_shape.row_stride, shape.row_stride, dgate_layout, gate_layout,
        block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchGatedRMSNormBackwardKernel<__nv_bfloat16, __nv_bfloat16, MAX_VPT>(
        dx, dgate, partial_dw, dy, input, gate, weight, static_cast<float>(epsilon),
        hidden_size, num_tokens, num_blocks, dx_shape.row_stride,
        dy_shape.row_stride, shape.row_stride, dgate_layout, gate_layout,
        block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchGatedRMSNormBackwardKernel<__nv_bfloat16, float, MAX_VPT>(
        dx, dgate, partial_dw, dy, input, gate, weight, static_cast<float>(epsilon),
        hidden_size, num_tokens, num_blocks, dx_shape.row_stride,
        dy_shape.row_stride, shape.row_stride, dgate_layout, gate_layout,
        block_size, stream);
  else throw std::runtime_error("Unsupported data type for Gated RMSNorm backward");

  LaunchRMSNormReduceDw(
      static_cast<float*>(dw.data_ptr()), partial_dw, hidden_size, num_blocks, stream);
  sync_check_cuda_error();
}

void GatedRMSNormBackwardWithRstd(
    wallx_cuda::TensorView dx,
    wallx_cuda::TensorView dgate,
    wallx_cuda::TensorView dw,
    wallx_cuda::TensorView partial_dw_view,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView rstd,
    wallx_cuda::TensorView weight,
    double epsilon,
    int64_t warps_per_block) {
  ASSERT_CHECK(dw.dtype().code == kDLFloat && dw.dtype().bits == 32);
  ASSERT_CHECK(partial_dw_view.dtype().code == kDLFloat &&
               partial_dw_view.dtype().bits == 32);
  ASSERT_CHECK(weight.IsContiguous());
  ASSERT_CHECK(dw.IsContiguous());
  ASSERT_CHECK(partial_dw_view.IsContiguous());
  ASSERT_CHECK(input.dtype().code == dy.dtype().code);
  ASSERT_CHECK(input.dtype().bits == dy.dtype().bits);
  ASSERT_CHECK(input.dtype().code == gate.dtype().code);
  ASSERT_CHECK(input.dtype().bits == gate.dtype().bits);
  ASSERT_CHECK(input.dtype().code == dgate.dtype().code);
  ASSERT_CHECK(input.dtype().bits == dgate.dtype().bits);
  CheckSameShape(dx, input);
  CheckSameShape(dy, input);
  CheckSameShape(gate, input);
  CheckSameShape(dgate, input);
  const RMSNormShape shape = GetRMSNormShape(input);
  const RMSNormShape dx_shape = GetRMSNormShape(dx);
  const RMSNormShape dy_shape = GetRMSNormShape(dy);
  const TensorRowLayout gate_layout = GetTensorRowLayout(gate);
  const TensorRowLayout dgate_layout = GetTensorRowLayout(dgate);
  const int64_t hidden_size = shape.hidden_size;
  const int64_t num_tokens = shape.num_tokens;
  ASSERT_CHECK(dx_shape.hidden_size == hidden_size);
  ASSERT_CHECK(dx_shape.num_tokens == num_tokens);
  ASSERT_CHECK(dy_shape.hidden_size == hidden_size);
  ASSERT_CHECK(dy_shape.num_tokens == num_tokens);
  ASSERT_CHECK(GetTensorNumTokens(gate) == num_tokens);
  ASSERT_CHECK(GetTensorNumTokens(dgate) == num_tokens);
  ASSERT_CHECK(hidden_size == weight.size(0));
  ASSERT_CHECK(rstd.ndim() == 1);
  ASSERT_CHECK(rstd.size(0) == num_tokens);
  ASSERT_CHECK(rstd.IsContiguous());
  ASSERT_CHECK(rstd.dtype().code == kDLFloat && rstd.dtype().bits == 32);
  const int64_t vec_size = GetVecSize(input.dtype());
  ASSERT_CHECK(hidden_size % vec_size == 0);
  ASSERT_CHECK(shape.row_stride % vec_size == 0);
  ASSERT_CHECK(dx_shape.row_stride % vec_size == 0);
  ASSERT_CHECK(dy_shape.row_stride % vec_size == 0);
  CheckTensorRowLayoutVectorized(gate_layout, vec_size);
  CheckTensorRowLayoutVectorized(dgate_layout, vec_size);

  if (hidden_size != 128) {
    GatedRMSNormBackward(
        dx, dgate, dw, partial_dw_view, dy, input, gate, weight, epsilon);
    return;
  }

  cudaStream_t stream = static_cast<cudaStream_t>(
      wallx_cuda::CurrentStream(input.device().device_type, input.device().device_id));
  ASSERT_CHECK(partial_dw_view.ndim() == 2);
  ASSERT_CHECK(partial_dw_view.size(1) == hidden_size);
  ASSERT_CHECK(dw.ndim() == 1);
  ASSERT_CHECK(dw.size(0) == hidden_size);
  const int64_t num_blocks = partial_dw_view.size(0);
  ASSERT_CHECK(num_blocks > 0);
  ASSERT_CHECK(num_blocks <= num_tokens);
  float* partial_dw = static_cast<float*>(partial_dw_view.data_ptr());
  const int dgate_layout_kind = GetD128LayoutKind(dgate_layout);
  const int gate_layout_kind = GetD128LayoutKind(gate_layout);

  if (IsFloat32(input.dtype()) && IsFloat32(weight.dtype()))
    DispatchGatedRMSNormD128BackwardKernel<float, float>(
        dx, dgate, partial_dw, dy, input, gate, rstd, weight, num_tokens,
        num_blocks, dx_shape.row_stride, dy_shape.row_stride, shape.row_stride,
        dgate_layout, gate_layout, dgate_layout_kind, gate_layout_kind,
        warps_per_block, stream);
  else if (IsFloat16(input.dtype()) && IsFloat16(weight.dtype()))
    DispatchGatedRMSNormD128BackwardKernel<half, half>(
        dx, dgate, partial_dw, dy, input, gate, rstd, weight, num_tokens,
        num_blocks, dx_shape.row_stride, dy_shape.row_stride, shape.row_stride,
        dgate_layout, gate_layout, dgate_layout_kind, gate_layout_kind,
        warps_per_block, stream);
  else if (IsFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    DispatchGatedRMSNormD128BackwardKernel<half, float>(
        dx, dgate, partial_dw, dy, input, gate, rstd, weight, num_tokens,
        num_blocks, dx_shape.row_stride, dy_shape.row_stride, shape.row_stride,
        dgate_layout, gate_layout, dgate_layout_kind, gate_layout_kind,
        warps_per_block, stream);
  else if (IsBFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    DispatchGatedRMSNormD128BackwardKernel<__nv_bfloat16, __nv_bfloat16>(
        dx, dgate, partial_dw, dy, input, gate, rstd, weight, num_tokens,
        num_blocks, dx_shape.row_stride, dy_shape.row_stride, shape.row_stride,
        dgate_layout, gate_layout, dgate_layout_kind, gate_layout_kind,
        warps_per_block, stream);
  else if (IsBFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    DispatchGatedRMSNormD128BackwardKernel<__nv_bfloat16, float>(
        dx, dgate, partial_dw, dy, input, gate, rstd, weight, num_tokens,
        num_blocks, dx_shape.row_stride, dy_shape.row_stride, shape.row_stride,
        dgate_layout, gate_layout, dgate_layout_kind, gate_layout_kind,
        warps_per_block, stream);
  else throw std::runtime_error("Unsupported data type for Gated RMSNorm backward");

  LaunchRMSNormReduceDw(
      static_cast<float*>(dw.data_ptr()), partial_dw, hidden_size, num_blocks, stream);
  sync_check_cuda_error();
}







// ======================== Adaptive RMSNorm ========================

struct AdaRMSNormShape {
  int64_t num_tokens;
  int64_t hidden_size;
  int64_t num_mod_rows;
  int64_t tokens_per_mod;
};

bool SameDType(const DLDataType& lhs, const DLDataType& rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

AdaRMSNormShape GetAdaRMSNormShape(const wallx_cuda::TensorView& input,
                                   const wallx_cuda::TensorView& modulation) {
  const RMSNormShape input_shape = GetRMSNormShape(input);
  const int64_t hidden_size = input_shape.hidden_size;
  ASSERT_CHECK(modulation.ndim() >= 1);
  ASSERT_CHECK(modulation.IsContiguous());
  const int mod_ndim = modulation.ndim();
  ASSERT_CHECK(modulation.size(mod_ndim - 1) == hidden_size * 3);

  if (mod_ndim == input.ndim()) {
    bool same_prefix = true;
    for (int i = 0; i < input.ndim() - 1; ++i) {
      if (modulation.size(i) != input.size(i)) {
        same_prefix = false;
        break;
      }
    }
    if (same_prefix) {
      return {input_shape.num_tokens, hidden_size, input_shape.num_tokens, 1};
    }

    if (input.ndim() == 3 && modulation.size(0) == input.size(0) &&
        modulation.size(1) == 1) {
      return {input_shape.num_tokens, hidden_size, input.size(0), input.size(1)};
    }
  }

  if (input.ndim() == 3 && mod_ndim == 2 && modulation.size(0) == input.size(0)) {
    return {input_shape.num_tokens, hidden_size, input.size(0), input.size(1)};
  }

  throw std::runtime_error(
      "Unsupported AdaRMSNorm shapes: modulation must match input prefix or be "
      "[batch, 3 * hidden] for a [batch, seq, hidden] input");
}

void CheckAdaGateShape(const wallx_cuda::TensorView& gate,
                       const AdaRMSNormShape& shape) {
  ASSERT_CHECK(gate.ndim() >= 1);
  ASSERT_CHECK(gate.IsContiguous());
  ASSERT_CHECK(gate.size(gate.ndim() - 1) == shape.hidden_size);
  int64_t gate_rows = 1;
  for (int i = 0; i < gate.ndim() - 1; ++i) {
    gate_rows *= gate.size(i);
  }
  ASSERT_CHECK(gate_rows == shape.num_mod_rows);
}

template<typename T, typename ModDType>
__global__ void adarmsnormKernel(
    T* __restrict__ output,
    T* __restrict__ gate,
    const T* __restrict__ input,
    const ModDType* __restrict__ modulation,
    const float epsilon,
    const int64_t hidden_size,
    const int64_t tokens_per_mod,
    const int64_t input_row_stride,
    const int64_t output_row_stride) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;

    float square_sum = 0.0f;
    __shared__ float inv_rms;
    const int64_t tid = threadIdx.x;
    const int64_t row = blockIdx.x;
    const int64_t input_offset = row * input_row_stride;
    const int64_t output_offset = row * output_row_stride;
    const int64_t mod_row = row / tokens_per_mod;
    const int64_t mod_base = mod_row * hidden_size * 3;
    const int64_t vec_hidden = hidden_size / VEC_SIZE;

    for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
        Vec x_vec = VecTraits<T>::load(input + input_offset + i * VEC_SIZE);
        float x_vals[VEC_SIZE];
        VecToFloats<T>(x_vec, x_vals);
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            square_sum += x_vals[j] * x_vals[j];
        }
    }
    square_sum = blockReduceSum(square_sum);
    if (tid == 0) inv_rms = rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
    __syncthreads();

    for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
        Vec in_vec = VecTraits<T>::load(input + input_offset + i * VEC_SIZE);
        float x_vals[VEC_SIZE], scale_vals[VEC_SIZE], shift_vals[VEC_SIZE], out_vals[VEC_SIZE];
        VecToFloats<T>(in_vec, x_vals);
        LoadWeightFloats<ModDType, VEC_SIZE>(modulation, mod_base + i * VEC_SIZE, scale_vals);
        LoadWeightFloats<ModDType, VEC_SIZE>(
            modulation, mod_base + hidden_size + i * VEC_SIZE, shift_vals);
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            out_vals[j] = x_vals[j] * inv_rms * (1.0f + scale_vals[j]) + shift_vals[j];
        }
        VecTraits<T>::store(output + output_offset + i * VEC_SIZE, FloatsToVec<T>(out_vals));
    }

    if ((row % tokens_per_mod) == 0) {
        const int64_t gate_offset = mod_row * hidden_size;
        for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
            float gate_vals[VEC_SIZE];
            LoadWeightFloats<ModDType, VEC_SIZE>(
                modulation, mod_base + hidden_size * 2 + i * VEC_SIZE, gate_vals);
            VecTraits<T>::store(gate + gate_offset + i * VEC_SIZE, FloatsToVec<T>(gate_vals));
        }
    }
}

template<typename T, typename ModDType>
void LaunchAdaRMSNormKernel(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView modulation,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t tokens_per_mod,
    int64_t input_row_stride,
    int64_t output_row_stride,
    int64_t block_size,
    cudaStream_t stream) {
  adarmsnormKernel<T, ModDType><<<num_tokens, block_size, 0, stream>>>(
      static_cast<T*>(output.data_ptr()), static_cast<T*>(gate.data_ptr()),
      static_cast<const T*>(input.data_ptr()),
      static_cast<const ModDType*>(modulation.data_ptr()), epsilon, hidden_size,
      tokens_per_mod, input_row_stride, output_row_stride);
}

template<typename T>
void DispatchAdaRMSNormKernel(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView modulation,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t tokens_per_mod,
    int64_t input_row_stride,
    int64_t output_row_stride,
    int64_t block_size,
    cudaStream_t stream) {
  if (IsFloat32(modulation.dtype()))
    LaunchAdaRMSNormKernel<T, float>(
        output, gate, input, modulation, epsilon, hidden_size, num_tokens,
        tokens_per_mod, input_row_stride, output_row_stride, block_size, stream);
  else if (IsFloat16(modulation.dtype()))
    LaunchAdaRMSNormKernel<T, half>(
        output, gate, input, modulation, epsilon, hidden_size, num_tokens,
        tokens_per_mod, input_row_stride, output_row_stride, block_size, stream);
  else if (IsBFloat16(modulation.dtype()))
    LaunchAdaRMSNormKernel<T, __nv_bfloat16>(
        output, gate, input, modulation, epsilon, hidden_size, num_tokens,
        tokens_per_mod, input_row_stride, output_row_stride, block_size, stream);
  else throw std::runtime_error("Unsupported modulation data type for AdaRMSNorm");
}

void AdaRMSNorm(wallx_cuda::TensorView output, wallx_cuda::TensorView gate,
                wallx_cuda::TensorView input, wallx_cuda::TensorView modulation,
                double epsilon) {
  ASSERT_CHECK(SameDType(output.dtype(), input.dtype()));
  ASSERT_CHECK(SameDType(gate.dtype(), input.dtype()));
  CheckSameShape(output, input);
  const AdaRMSNormShape shape = GetAdaRMSNormShape(input, modulation);
  const RMSNormShape input_shape = GetRMSNormShape(input);
  const RMSNormShape output_shape = GetRMSNormShape(output);
  ASSERT_CHECK(output_shape.hidden_size == input_shape.hidden_size);
  ASSERT_CHECK(output_shape.num_tokens == input_shape.num_tokens);
  CheckAdaGateShape(gate, shape);
  ASSERT_CHECK(shape.hidden_size % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(input_shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(output_shape.row_stride % GetVecSize(input.dtype()) == 0);
  cudaStream_t stream = static_cast<cudaStream_t>(
      wallx_cuda::CurrentStream(input.device().device_type, input.device().device_id));
  const int64_t block_size = std::min(shape.hidden_size, 1024L);

  if (IsFloat32(input.dtype()))
    DispatchAdaRMSNormKernel<float>(
        output, gate, input, modulation, static_cast<float>(epsilon),
        shape.hidden_size, shape.num_tokens, shape.tokens_per_mod,
        input_shape.row_stride, output_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()))
    DispatchAdaRMSNormKernel<half>(
        output, gate, input, modulation, static_cast<float>(epsilon),
        shape.hidden_size, shape.num_tokens, shape.tokens_per_mod,
        input_shape.row_stride, output_shape.row_stride, block_size, stream);
  else if (IsBFloat16(input.dtype()))
    DispatchAdaRMSNormKernel<__nv_bfloat16>(
        output, gate, input, modulation, static_cast<float>(epsilon),
        shape.hidden_size, shape.num_tokens, shape.tokens_per_mod,
        input_shape.row_stride, output_shape.row_stride, block_size, stream);
  else throw std::runtime_error("Unsupported data type for AdaRMSNorm");
  sync_check_cuda_error();
}



template<typename T, typename ModDType>
__global__ void adarmsnormBackwardKernel(
    T* __restrict__ dx,
    float* __restrict__ dmod,
    const T* __restrict__ dy,
    const T* __restrict__ dgate,
    const T* __restrict__ input,
    const ModDType* __restrict__ modulation,
    const float epsilon,
    const int64_t hidden_size,
    const int64_t tokens_per_mod,
    const int64_t dx_row_stride,
    const int64_t dy_row_stride,
    const int64_t input_row_stride) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;

    const int64_t tid = threadIdx.x;
    const int64_t row = blockIdx.x;
    const int64_t dx_offset = row * dx_row_stride;
    const int64_t dy_offset = row * dy_row_stride;
    const int64_t input_offset = row * input_row_stride;
    const int64_t mod_row = row / tokens_per_mod;
    const int64_t mod_base = mod_row * hidden_size * 3;
    const int64_t gate_offset = mod_row * hidden_size;
    const int64_t vec_hidden = hidden_size / VEC_SIZE;
    const float inv_hidden = 1.0f / static_cast<float>(hidden_size);

    __shared__ float s_inv_rms;
    __shared__ float s_c;

    float sq = 0.0f, dp = 0.0f;
    for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
        Vec xv = VecTraits<T>::load(input + input_offset + i * VEC_SIZE);
        Vec dv = VecTraits<T>::load(dy + dy_offset + i * VEC_SIZE);
        float x_vals[VEC_SIZE], d_vals[VEC_SIZE], scale_vals[VEC_SIZE];
        VecToFloats<T>(xv, x_vals);
        VecToFloats<T>(dv, d_vals);
        LoadWeightFloats<ModDType, VEC_SIZE>(modulation, mod_base + i * VEC_SIZE, scale_vals);
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            const float z = d_vals[j] * (1.0f + scale_vals[j]);
            sq += x_vals[j] * x_vals[j];
            dp += z * x_vals[j];
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

    for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
        Vec xv = VecTraits<T>::load(input + input_offset + i * VEC_SIZE);
        Vec dv = VecTraits<T>::load(dy + dy_offset + i * VEC_SIZE);
        float x_vals[VEC_SIZE], d_vals[VEC_SIZE], scale_vals[VEC_SIZE], dx_vals[VEC_SIZE];
        VecToFloats<T>(xv, x_vals);
        VecToFloats<T>(dv, d_vals);
        LoadWeightFloats<ModDType, VEC_SIZE>(modulation, mod_base + i * VEC_SIZE, scale_vals);
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            const int64_t elem = i * VEC_SIZE + j;
            const float normed = x_vals[j] * irms;
            const float z = d_vals[j] * (1.0f + scale_vals[j]);
            dx_vals[j] = irms * (z - x_vals[j] * cv);
            const float dscale = d_vals[j] * normed;
            const float dshift = d_vals[j];
            if (tokens_per_mod == 1) {
                dmod[mod_base + elem] = dscale;
                dmod[mod_base + hidden_size + elem] = dshift;
            } else {
                atomicAdd(dmod + mod_base + elem, dscale);
                atomicAdd(dmod + mod_base + hidden_size + elem, dshift);
            }
        }
        VecTraits<T>::store(dx + dx_offset + i * VEC_SIZE, FloatsToVec<T>(dx_vals));
    }

    if ((row % tokens_per_mod) == 0) {
        for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
            Vec gv = VecTraits<T>::load(dgate + gate_offset + i * VEC_SIZE);
            float gate_vals[VEC_SIZE];
            VecToFloats<T>(gv, gate_vals);
#pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                dmod[mod_base + hidden_size * 2 + i * VEC_SIZE + j] = gate_vals[j];
            }
        }
    }
}

template<typename T, typename ModDType>
void LaunchAdaRMSNormBackwardKernel(
    wallx_cuda::TensorView dx,
    wallx_cuda::TensorView dmod,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView dgate,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView modulation,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t tokens_per_mod,
    int64_t dx_row_stride,
    int64_t dy_row_stride,
    int64_t input_row_stride,
    int64_t block_size,
    cudaStream_t stream) {
  adarmsnormBackwardKernel<T, ModDType><<<num_tokens, block_size, 0, stream>>>(
      static_cast<T*>(dx.data_ptr()), static_cast<float*>(dmod.data_ptr()),
      static_cast<const T*>(dy.data_ptr()), static_cast<const T*>(dgate.data_ptr()),
      static_cast<const T*>(input.data_ptr()),
      static_cast<const ModDType*>(modulation.data_ptr()), epsilon, hidden_size,
      tokens_per_mod, dx_row_stride, dy_row_stride, input_row_stride);
}

template<typename T>
void DispatchAdaRMSNormBackwardKernel(
    wallx_cuda::TensorView dx,
    wallx_cuda::TensorView dmod,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView dgate,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView modulation,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t tokens_per_mod,
    int64_t dx_row_stride,
    int64_t dy_row_stride,
    int64_t input_row_stride,
    int64_t block_size,
    cudaStream_t stream) {
  if (IsFloat32(modulation.dtype()))
    LaunchAdaRMSNormBackwardKernel<T, float>(
        dx, dmod, dy, dgate, input, modulation, epsilon, hidden_size,
        num_tokens, tokens_per_mod, dx_row_stride, dy_row_stride,
        input_row_stride, block_size, stream);
  else if (IsFloat16(modulation.dtype()))
    LaunchAdaRMSNormBackwardKernel<T, half>(
        dx, dmod, dy, dgate, input, modulation, epsilon, hidden_size,
        num_tokens, tokens_per_mod, dx_row_stride, dy_row_stride,
        input_row_stride, block_size, stream);
  else if (IsBFloat16(modulation.dtype()))
    LaunchAdaRMSNormBackwardKernel<T, __nv_bfloat16>(
        dx, dmod, dy, dgate, input, modulation, epsilon, hidden_size,
        num_tokens, tokens_per_mod, dx_row_stride, dy_row_stride,
        input_row_stride, block_size, stream);
  else throw std::runtime_error("Unsupported modulation data type for AdaRMSNorm backward");
}

void AdaRMSNormBackward(
    wallx_cuda::TensorView dx,
    wallx_cuda::TensorView dmod,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView dgate,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView modulation,
    double epsilon) {
  ASSERT_CHECK(dmod.dtype().code == kDLFloat && dmod.dtype().bits == 32);
  ASSERT_CHECK(dmod.IsContiguous());
  ASSERT_CHECK(dgate.IsContiguous());
  ASSERT_CHECK(modulation.IsContiguous());
  ASSERT_CHECK(SameDType(dx.dtype(), input.dtype()));
  ASSERT_CHECK(SameDType(dy.dtype(), input.dtype()));
  ASSERT_CHECK(SameDType(dgate.dtype(), input.dtype()));
  CheckSameShape(dx, input);
  CheckSameShape(dy, input);
  CheckSameShape(dmod, modulation);
  const AdaRMSNormShape shape = GetAdaRMSNormShape(input, modulation);
  const RMSNormShape input_shape = GetRMSNormShape(input);
  const RMSNormShape dx_shape = GetRMSNormShape(dx);
  const RMSNormShape dy_shape = GetRMSNormShape(dy);
  ASSERT_CHECK(dx_shape.hidden_size == input_shape.hidden_size);
  ASSERT_CHECK(dx_shape.num_tokens == input_shape.num_tokens);
  ASSERT_CHECK(dy_shape.hidden_size == input_shape.hidden_size);
  ASSERT_CHECK(dy_shape.num_tokens == input_shape.num_tokens);
  CheckAdaGateShape(dgate, shape);
  ASSERT_CHECK(shape.hidden_size % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(input_shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(dx_shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(dy_shape.row_stride % GetVecSize(input.dtype()) == 0);
  cudaStream_t stream = static_cast<cudaStream_t>(
      wallx_cuda::CurrentStream(input.device().device_type, input.device().device_id));
  const int64_t block_size = std::min(shape.hidden_size, 1024L);

  if (IsFloat32(input.dtype()))
    DispatchAdaRMSNormBackwardKernel<float>(
        dx, dmod, dy, dgate, input, modulation, static_cast<float>(epsilon),
        shape.hidden_size, shape.num_tokens, shape.tokens_per_mod,
        dx_shape.row_stride, dy_shape.row_stride, input_shape.row_stride,
        block_size, stream);
  else if (IsFloat16(input.dtype()))
    DispatchAdaRMSNormBackwardKernel<half>(
        dx, dmod, dy, dgate, input, modulation, static_cast<float>(epsilon),
        shape.hidden_size, shape.num_tokens, shape.tokens_per_mod,
        dx_shape.row_stride, dy_shape.row_stride, input_shape.row_stride,
        block_size, stream);
  else if (IsBFloat16(input.dtype()))
    DispatchAdaRMSNormBackwardKernel<__nv_bfloat16>(
        dx, dmod, dy, dgate, input, modulation, static_cast<float>(epsilon),
        shape.hidden_size, shape.num_tokens, shape.tokens_per_mod,
        dx_shape.row_stride, dy_shape.row_stride, input_shape.row_stride,
        block_size, stream);
  else throw std::runtime_error("Unsupported data type for AdaRMSNorm backward");
  sync_check_cuda_error();
}



}  // namespace wallx_cuda_rmsnorm
