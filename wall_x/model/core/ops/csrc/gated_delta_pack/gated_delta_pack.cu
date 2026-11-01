#include "tensor_view.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "../common/cuda_utils.h"

namespace wallx_cuda_gated_delta_pack {

static int checked_blocks(int64_t count, const char* name) {
    if (count < 0 || count > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string(name) + ": launch size exceeds INT_MAX");
    }
    return static_cast<int>(count);
}

static void check_same_device_and_dtype(
    wallx_cuda::TensorView tensor,
    wallx_cuda::TensorView ref,
    const char* name) {
    const std::string prefix = std::string(name) + ": ";
    if (tensor.dtype().code != ref.dtype().code || tensor.dtype().bits != ref.dtype().bits) {
        throw std::runtime_error(prefix + "dtype mismatch");
    }
    if (tensor.device().device_type != ref.device().device_type ||
        tensor.device().device_id != ref.device().device_id) {
        throw std::runtime_error(prefix + "device mismatch");
    }
}

static void check_same_device(
    wallx_cuda::TensorView tensor,
    wallx_cuda::TensorView ref,
    const char* name) {
    const std::string prefix = std::string(name) + ": ";
    if (tensor.device().device_type != ref.device().device_type ||
        tensor.device().device_id != ref.device().device_id) {
        throw std::runtime_error(prefix + "device mismatch");
    }
}

static void check_float32(wallx_cuda::TensorView tensor, const char* name) {
    if (tensor.dtype().code != kDLFloat || tensor.dtype().bits != 32) {
        throw std::runtime_error(std::string(name) + ": expected float32");
    }
}

static void check_supported_dtype(wallx_cuda::TensorView tensor, const char* name) {
    const bool is_fp32 = tensor.dtype().code == kDLFloat && tensor.dtype().bits == 32;
    const bool is_fp16 = tensor.dtype().code == kDLFloat && tensor.dtype().bits == 16;
    const bool is_bf16 = tensor.dtype().code == kDLBfloat && tensor.dtype().bits == 16;
    if (!is_fp32 && !is_fp16 && !is_bf16) {
        throw std::runtime_error(std::string(name) + ": unsupported dtype");
    }
}

static void validate_dims(
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
    if (num_k_heads <= 0 || num_v_heads <= 0 || head_k_dim <= 0 || head_v_dim <= 0) {
        throw std::runtime_error("gated_delta_post_conv_pack: head counts and dims must be positive");
    }
    if (num_v_heads % num_k_heads != 0) {
        throw std::runtime_error("gated_delta_post_conv_pack: num_v_heads must be divisible by num_k_heads");
    }
}

static bool is_aligned_16(const void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % static_cast<uintptr_t>(sizeof(uint4)) == 0;
}

static bool fits_i32(int64_t value) {
    return value >= 0 && value <= std::numeric_limits<int>::max();
}

static bool accum_scaled_offset_fits_i32(int64_t& acc, int64_t scale, int64_t count_minus_one) {
    if (!fits_i32(acc) || scale < 0 || count_minus_one < 0) {
        return false;
    }
    if (scale == 0 || count_minus_one == 0) {
        return true;
    }
    const int64_t remaining = static_cast<int64_t>(std::numeric_limits<int>::max()) - acc;
    if (scale > remaining / count_minus_one) {
        return false;
    }
    acc += scale * count_minus_one;
    return true;
}

template <typename T>
static bool can_use_vec16_post_conv_pack(
    const T* mixed,
    const T* query,
    const T* key,
    const T* value,
    int64_t mixed_b_stride,
    int64_t mixed_c_stride,
    int64_t mixed_t_stride,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
    constexpr int64_t vec_bytes = static_cast<int64_t>(sizeof(uint4));
    constexpr int64_t elems_per_vec = vec_bytes / static_cast<int64_t>(sizeof(T));
    return mixed_c_stride == 1 &&
           head_k_dim % elems_per_vec == 0 &&
           head_v_dim % elems_per_vec == 0 &&
           (mixed_b_stride * static_cast<int64_t>(sizeof(T))) % vec_bytes == 0 &&
           (mixed_t_stride * static_cast<int64_t>(sizeof(T))) % vec_bytes == 0 &&
           (num_v_heads * head_k_dim * static_cast<int64_t>(sizeof(T))) % vec_bytes == 0 &&
           (num_v_heads * head_v_dim * static_cast<int64_t>(sizeof(T))) % vec_bytes == 0 &&
           is_aligned_16(mixed) &&
           is_aligned_16(query) &&
           is_aligned_16(key) &&
           is_aligned_16(value);
}

template <typename T>
static bool can_use_qwen35_vec16_post_conv_pack(
    const T* mixed,
    const T* query,
    const T* key,
    const T* value,
    int64_t batch_size,
    int64_t seq_length,
    int64_t mixed_b_stride,
    int64_t mixed_c_stride,
    int64_t mixed_t_stride,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
    if (num_k_heads != 16 || num_v_heads != 32 || head_k_dim != 128 || head_v_dim != 128) {
        return false;
    }
    if (!can_use_vec16_post_conv_pack<T>(
            mixed,
            query,
            key,
            value,
            mixed_b_stride,
            mixed_c_stride,
            mixed_t_stride,
            num_v_heads,
            head_k_dim,
            head_v_dim)) {
        return false;
    }
    const int64_t rows = batch_size * seq_length;
    if (!fits_i32(seq_length) || !fits_i32(rows) ||
        rows > std::numeric_limits<int>::max() / 4096) {
        return false;
    }
    int64_t max_mixed_offset = 8191;
    return accum_scaled_offset_fits_i32(max_mixed_offset, mixed_b_stride, batch_size - 1) &&
           accum_scaled_offset_fits_i32(max_mixed_offset, mixed_t_stride, seq_length - 1);
}

template <typename T>
static bool can_use_vec16_post_conv_pack_backward(
    const T* grad_mixed,
    const T* grad_query,
    const T* grad_key,
    const T* grad_value,
    int64_t grad_mixed_b_stride,
    int64_t grad_mixed_c_stride,
    int64_t grad_mixed_t_stride,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
    constexpr int64_t vec_bytes = static_cast<int64_t>(sizeof(uint4));
    constexpr int64_t elems_per_vec = vec_bytes / static_cast<int64_t>(sizeof(T));
    return grad_mixed_c_stride == 1 &&
           head_k_dim % elems_per_vec == 0 &&
           head_v_dim % elems_per_vec == 0 &&
           (grad_mixed_b_stride * static_cast<int64_t>(sizeof(T))) % vec_bytes == 0 &&
           (grad_mixed_t_stride * static_cast<int64_t>(sizeof(T))) % vec_bytes == 0 &&
           (num_v_heads * head_k_dim * static_cast<int64_t>(sizeof(T))) % vec_bytes == 0 &&
           (num_v_heads * head_v_dim * static_cast<int64_t>(sizeof(T))) % vec_bytes == 0 &&
           is_aligned_16(grad_mixed) &&
           is_aligned_16(grad_query) &&
           is_aligned_16(grad_key) &&
           is_aligned_16(grad_value);
}

template <typename T>
static bool can_use_qwen35_vec16_post_conv_pack_backward(
    const T* grad_mixed,
    const T* grad_query,
    const T* grad_key,
    const T* grad_value,
    int64_t batch_size,
    int64_t seq_length,
    int64_t grad_mixed_b_stride,
    int64_t grad_mixed_c_stride,
    int64_t grad_mixed_t_stride,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
    if (num_k_heads != 16 || num_v_heads != 32 || head_k_dim != 128 || head_v_dim != 128) {
        return false;
    }
    if (!can_use_vec16_post_conv_pack_backward<T>(
            grad_mixed,
            grad_query,
            grad_key,
            grad_value,
            grad_mixed_b_stride,
            grad_mixed_c_stride,
            grad_mixed_t_stride,
            num_v_heads,
            head_k_dim,
            head_v_dim)) {
        return false;
    }
    const int64_t rows = batch_size * seq_length;
    if (!fits_i32(seq_length) || !fits_i32(rows) ||
        rows > std::numeric_limits<int>::max() / 4096) {
        return false;
    }
    int64_t max_grad_mixed_offset = 8191;
    return accum_scaled_offset_fits_i32(
               max_grad_mixed_offset,
               grad_mixed_b_stride,
               batch_size - 1) &&
           accum_scaled_offset_fits_i32(
               max_grad_mixed_offset,
               grad_mixed_t_stride,
               seq_length - 1);
}

template <typename T>
__device__ __forceinline__ uint4 add_vec16(uint4 lhs, uint4 rhs) {
    constexpr int64_t elems_per_vec = static_cast<int64_t>(sizeof(uint4)) /
                                      static_cast<int64_t>(sizeof(T));
    uint4 out;
    const T* lhs_items = reinterpret_cast<const T*>(&lhs);
    const T* rhs_items = reinterpret_cast<const T*>(&rhs);
    T* out_items = reinterpret_cast<T*>(&out);
#pragma unroll
    for (int64_t i = 0; i < elems_per_vec; ++i) {
        out_items[i] = lhs_items[i] + rhs_items[i];
    }
    return out;
}

template <typename T>
__device__ __forceinline__ float scalar_to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T float_to_scalar(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half float_to_scalar<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_scalar<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float stable_sigmoid(float value) {
    if (value >= 0.0f) {
        const float z = __expf(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = __expf(value);
    return z / (1.0f + z);
}

__device__ __forceinline__ float softplus_default(float value) {
    return value > 20.0f ? value : log1pf(__expf(value));
}

template <int BlockSize>
__device__ __forceinline__ float block_sum(float value) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    constexpr int kWarps = (BlockSize + 31) / 32;
    value = threadIdx.x < kWarps ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
    }
    if (threadIdx.x == 0) {
        warp_sums[0] = value;
    }
    __syncthreads();
    return warp_sums[0];
}

template <typename T>
__global__ void postConvPackQwen35Vec16Kernel(
    const T* __restrict__ mixed,
    T* __restrict__ query,
    T* __restrict__ key,
    T* __restrict__ value,
    int seq_length,
    int mixed_b_stride,
    int mixed_t_stride) {
    constexpr int kNumKHeads = 16;
    constexpr int kNumVHeads = 32;
    constexpr int kHeadDim = 128;
    constexpr int kKeyDim = kNumKHeads * kHeadDim;
    constexpr int kValueBase = 2 * kKeyDim;
    constexpr int kOutRowItems = kNumVHeads * kHeadDim;
    constexpr int kElemsPerVec = static_cast<int>(sizeof(uint4) / sizeof(T));
    constexpr int kHeadVecs = kHeadDim / kElemsPerVec;
    constexpr int kQkSourceVecs = kNumKHeads * kHeadVecs;
    constexpr int kValueVecs = kNumVHeads * kHeadVecs;

    const int row = static_cast<int>(blockIdx.x);
    const int part = static_cast<int>(blockIdx.y);
    const int t = row % seq_length;
    const int b = row / seq_length;
    const int base_mixed = b * mixed_b_stride + t * mixed_t_stride;
    const int base_out = row * kOutRowItems;

    if (part == 0) {
        for (int idx = threadIdx.x; idx < kQkSourceVecs; idx += blockDim.x) {
            const int hk = idx / kHeadVecs;
            const int vec = idx - hk * kHeadVecs;
            const int d = vec * kElemsPerVec;
            const int q_channel = hk * kHeadDim + d;
            const int k_channel = kKeyDim + q_channel;
            const uint4 q_val = *reinterpret_cast<const uint4*>(
                mixed + base_mixed + q_channel);
            const uint4 k_val = *reinterpret_cast<const uint4*>(
                mixed + base_mixed + k_channel);
            const int out_offset = base_out + hk * 2 * kHeadDim + d;
            *reinterpret_cast<uint4*>(query + out_offset) = q_val;
            *reinterpret_cast<uint4*>(query + out_offset + kHeadDim) = q_val;
            *reinterpret_cast<uint4*>(key + out_offset) = k_val;
            *reinterpret_cast<uint4*>(key + out_offset + kHeadDim) = k_val;
        }
        return;
    }

    for (int idx = threadIdx.x; idx < kValueVecs; idx += blockDim.x) {
        const int hv = idx / kHeadVecs;
        const int vec = idx - hv * kHeadVecs;
        const int d = vec * kElemsPerVec;
        const int v_channel = kValueBase + hv * kHeadDim + d;
        const int out_offset = base_out + hv * kHeadDim + d;
        const uint4 v_val = *reinterpret_cast<const uint4*>(
            mixed + base_mixed + v_channel);
        *reinterpret_cast<uint4*>(value + out_offset) = v_val;
    }
}

template <typename T>
__global__ void postConvPackPreprocessQwen35ForwardKernel(
    const T* __restrict__ mixed,
    const T* __restrict__ b_gate,
    const T* __restrict__ a_gate,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    T* __restrict__ query,
    T* __restrict__ key,
    T* __restrict__ value,
    T* __restrict__ beta,
    float* __restrict__ g,
    float* __restrict__ q_rstd,
    float* __restrict__ k_rstd,
    int seq_length,
    int mixed_b_stride,
    int mixed_t_stride,
    int b_gate_b_stride,
    int b_gate_t_stride,
    int a_gate_b_stride,
    int a_gate_t_stride) {
    constexpr int kNumKHeads = 16;
    constexpr int kNumVHeads = 32;
    constexpr int kHeadDim = 128;
    constexpr int kKeyDim = kNumKHeads * kHeadDim;
    constexpr int kValueBase = 2 * kKeyDim;
    constexpr int kQkItems = 2 * kKeyDim;
    constexpr int kOutRowItems = kNumVHeads * kHeadDim;

    const int row = static_cast<int>(blockIdx.x);
    const int part = static_cast<int>(blockIdx.y);
    const int t = row % seq_length;
    const int b = row / seq_length;
    const int base_mixed = b * mixed_b_stride + t * mixed_t_stride;

    if (part == 0) {
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const int lane = static_cast<int>(threadIdx.x) & 31;
        if (warp >= 2 * kNumKHeads) {
            return;
        }
        const int is_key = warp >= kNumKHeads;
        const int hk = warp - is_key * kNumKHeads;
        const int channel_base = is_key * kKeyDim + hk * kHeadDim;

        float items[4];
        float sum_sq = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = lane + i * 32;
            items[i] = scalar_to_float<T>(mixed[base_mixed + channel_base + d]);
            sum_sq += items[i] * items[i];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        const float head_sum = __shfl_sync(0xffffffff, sum_sq, 0);
        const float rstd = rsqrtf(head_sum + 1.0e-6f);
        if (lane == 0) {
            const int rstd_offset = row * kNumKHeads + hk;
            if (is_key) {
                k_rstd[rstd_offset] = rstd;
            } else {
                q_rstd[rstd_offset] = rstd;
            }
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = lane + i * 32;
            const T normalized = float_to_scalar<T>(items[i] * rstd);
            const int out_offset = row * kOutRowItems + hk * 2 * kHeadDim + d;
            if (is_key) {
                key[out_offset] = normalized;
                key[out_offset + kHeadDim] = normalized;
            } else {
                query[out_offset] = normalized;
                query[out_offset + kHeadDim] = normalized;
            }
        }
        return;
    }

    const int base_value = row * kOutRowItems;
    for (int idx = threadIdx.x; idx < kOutRowItems; idx += blockDim.x) {
        value[base_value + idx] = mixed[base_mixed + kValueBase + idx];
    }

    for (int hv = threadIdx.x; hv < kNumVHeads; hv += blockDim.x) {
        const int gate_offset = row * kNumVHeads + hv;
        const int b_gate_offset = b * b_gate_b_stride + t * b_gate_t_stride + hv;
        const int a_gate_offset = b * a_gate_b_stride + t * a_gate_t_stride + hv;
        const float beta_value =
            stable_sigmoid(scalar_to_float<T>(b_gate[b_gate_offset]));
        beta[gate_offset] = float_to_scalar<T>(beta_value);

        const float u = scalar_to_float<T>(a_gate[a_gate_offset]) + dt_bias[hv];
        g[gate_offset] = -__expf(A_log[hv]) * softplus_default(u);
    }
}

template <typename T>
__global__ void postConvPackKernel(
    const T* __restrict__ mixed,
    T* __restrict__ query,
    T* __restrict__ key,
    T* __restrict__ value,
    int64_t batch_size,
    int64_t seq_length,
    int64_t channels,
    int64_t mixed_b_stride,
    int64_t mixed_c_stride,
    int64_t mixed_t_stride,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    int64_t repeat) {
    const int64_t row = blockIdx.x;
    const int64_t t = row % seq_length;
    const int64_t b = row / seq_length;
    const int64_t key_dim = num_k_heads * head_k_dim;
    const int64_t qk_source_items = num_k_heads * head_k_dim;
    const int64_t value_items = num_v_heads * head_v_dim;
    const int64_t base_mixed = b * mixed_b_stride + t * mixed_t_stride;
    const int64_t base_qk_out = (b * seq_length + t) * num_v_heads * head_k_dim;
    const int64_t base_v_out = (b * seq_length + t) * num_v_heads * head_v_dim;

    for (int64_t idx = threadIdx.x; idx < qk_source_items; idx += blockDim.x) {
        const int64_t hk = idx / head_k_dim;
        const int64_t d = idx - hk * head_k_dim;
        const int64_t q_channel = hk * head_k_dim + d;
        const int64_t k_channel = key_dim + hk * head_k_dim + d;
        const T q_val = mixed[base_mixed + q_channel * mixed_c_stride];
        const T k_val = mixed[base_mixed + k_channel * mixed_c_stride];
        const int64_t out_h_base = hk * repeat;
        for (int64_t r = 0; r < repeat; ++r) {
            const int64_t out_offset = base_qk_out + (out_h_base + r) * head_k_dim + d;
            query[out_offset] = q_val;
            key[out_offset] = k_val;
        }
    }

    for (int64_t idx = threadIdx.x; idx < value_items; idx += blockDim.x) {
        const int64_t hv = idx / head_v_dim;
        const int64_t d = idx - hv * head_v_dim;
        const int64_t v_channel = 2 * key_dim + hv * head_v_dim + d;
        value[base_v_out + idx] = mixed[base_mixed + v_channel * mixed_c_stride];
    }
}

template <typename T>
__global__ void postConvPackVec16Kernel(
    const T* __restrict__ mixed,
    T* __restrict__ query,
    T* __restrict__ key,
    T* __restrict__ value,
    int64_t batch_size,
    int64_t seq_length,
    int64_t mixed_b_stride,
    int64_t mixed_t_stride,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    int64_t repeat) {
    constexpr int64_t elems_per_vec = static_cast<int64_t>(sizeof(uint4)) /
                                      static_cast<int64_t>(sizeof(T));
    const int64_t row = blockIdx.x;
    const int64_t t = row % seq_length;
    const int64_t b = row / seq_length;
    const int64_t key_dim = num_k_heads * head_k_dim;
    const int64_t qk_head_vecs = head_k_dim / elems_per_vec;
    const int64_t qk_source_vecs = num_k_heads * qk_head_vecs;
    const int64_t value_head_vecs = head_v_dim / elems_per_vec;
    const int64_t value_vecs = num_v_heads * value_head_vecs;
    const int64_t base_mixed = b * mixed_b_stride + t * mixed_t_stride;
    const int64_t base_qk_out = (b * seq_length + t) * num_v_heads * head_k_dim;
    const int64_t base_v_out = (b * seq_length + t) * num_v_heads * head_v_dim;

    for (int64_t idx = threadIdx.x; idx < qk_source_vecs; idx += blockDim.x) {
        const int64_t hk = idx / qk_head_vecs;
        const int64_t vec = idx - hk * qk_head_vecs;
        const int64_t d = vec * elems_per_vec;
        const int64_t q_channel = hk * head_k_dim + d;
        const int64_t k_channel = key_dim + hk * head_k_dim + d;
        const uint4 q_val = *reinterpret_cast<const uint4*>(mixed + base_mixed + q_channel);
        const uint4 k_val = *reinterpret_cast<const uint4*>(mixed + base_mixed + k_channel);
        const int64_t out_h_base = hk * repeat;
        for (int64_t r = 0; r < repeat; ++r) {
            const int64_t out_offset = base_qk_out + (out_h_base + r) * head_k_dim + d;
            *reinterpret_cast<uint4*>(query + out_offset) = q_val;
            *reinterpret_cast<uint4*>(key + out_offset) = k_val;
        }
    }

    for (int64_t idx = threadIdx.x; idx < value_vecs; idx += blockDim.x) {
        const int64_t hv = idx / value_head_vecs;
        const int64_t vec = idx - hv * value_head_vecs;
        const int64_t d = vec * elems_per_vec;
        const int64_t v_channel = 2 * key_dim + hv * head_v_dim + d;
        const int64_t out_offset = base_v_out + hv * head_v_dim + d;
        const uint4 v_val = *reinterpret_cast<const uint4*>(mixed + base_mixed + v_channel);
        *reinterpret_cast<uint4*>(value + out_offset) = v_val;
    }
}

template <typename T>
__global__ void postConvPackPreprocessBackwardQwen35Kernel(
    T* __restrict__ grad_mixed,
    T* __restrict__ grad_b,
    T* __restrict__ grad_a,
    float* __restrict__ grad_A_log,
    float* __restrict__ grad_dt_bias,
    const T* __restrict__ grad_query,
    const T* __restrict__ grad_key,
    const T* __restrict__ grad_value,
    const T* __restrict__ grad_beta,
    const float* __restrict__ grad_g,
    const T* __restrict__ query,
    const T* __restrict__ key,
    const float* __restrict__ q_rstd,
    const float* __restrict__ k_rstd,
    const T* __restrict__ b_gate,
    const T* __restrict__ a_gate,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    int seq_length,
    int grad_mixed_b_stride,
    int grad_mixed_t_stride,
    int grad_b_b_stride,
    int grad_b_t_stride,
    int grad_a_b_stride,
    int grad_a_t_stride,
    int b_gate_b_stride,
    int b_gate_t_stride,
    int a_gate_b_stride,
    int a_gate_t_stride) {
    constexpr int kNumKHeads = 16;
    constexpr int kNumVHeads = 32;
    constexpr int kHeadDim = 128;
    constexpr int kKeyDim = kNumKHeads * kHeadDim;
    constexpr int kValueBase = 2 * kKeyDim;
    constexpr int kOutRowItems = kNumVHeads * kHeadDim;
    constexpr int kBlockSize = 128;

    const int row = static_cast<int>(blockIdx.x);
    const int part = static_cast<int>(blockIdx.y);
    const int t = row % seq_length;
    const int b = row / seq_length;
    const int base_grad_mixed = b * grad_mixed_b_stride + t * grad_mixed_t_stride;

    if (part < kNumKHeads) {
        const int hk = part;
        const int d = static_cast<int>(threadIdx.x);
        const int out_offset = row * kOutRowItems + hk * 2 * kHeadDim + d;
        float q_dot_local = 0.0f;
        float k_dot_local = 0.0f;
        float q_y = 0.0f;
        float k_y = 0.0f;
        float q_grad_sum = 0.0f;
        float k_grad_sum = 0.0f;

        if (d < kHeadDim) {
            q_y = scalar_to_float<T>(query[out_offset]);
            k_y = scalar_to_float<T>(key[out_offset]);
            const float q_grad_0 = scalar_to_float<T>(grad_query[out_offset]);
            const float q_grad_1 = scalar_to_float<T>(grad_query[out_offset + kHeadDim]);
            const float k_grad_0 = scalar_to_float<T>(grad_key[out_offset]);
            const float k_grad_1 = scalar_to_float<T>(grad_key[out_offset + kHeadDim]);
            q_grad_sum = q_grad_0 + q_grad_1;
            k_grad_sum = k_grad_0 + k_grad_1;
            q_dot_local = q_grad_sum * q_y;
            k_dot_local = k_grad_sum * k_y;
        }

        const float q_dot = block_sum<kBlockSize>(q_dot_local);
        const float k_dot = block_sum<kBlockSize>(k_dot_local);
        if (d < kHeadDim) {
            const int rstd_offset = row * kNumKHeads + hk;
            const float q_dx = q_rstd[rstd_offset] * (q_grad_sum - q_dot * q_y);
            const float k_dx = k_rstd[rstd_offset] * (k_grad_sum - k_dot * k_y);
            const int q_channel = hk * kHeadDim + d;
            const int k_channel = kKeyDim + q_channel;
            grad_mixed[base_grad_mixed + q_channel] = float_to_scalar<T>(q_dx);
            grad_mixed[base_grad_mixed + k_channel] = float_to_scalar<T>(k_dx);
        }
        return;
    }

    if (part == kNumKHeads) {
        constexpr int kValueItems = kNumVHeads * kHeadDim;
        const int base_value_grad = row * kValueItems;
        for (int idx = threadIdx.x; idx < kValueItems; idx += blockDim.x) {
            grad_mixed[base_grad_mixed + kValueBase + idx] = grad_value[base_value_grad + idx];
        }
        return;
    }

    for (int hv = threadIdx.x; hv < kNumVHeads; hv += blockDim.x) {
        const int gate_offset = row * kNumVHeads + hv;
        const int grad_b_offset = b * grad_b_b_stride + t * grad_b_t_stride + hv;
        const int grad_a_offset = b * grad_a_b_stride + t * grad_a_t_stride + hv;
        const int b_gate_offset = b * b_gate_b_stride + t * b_gate_t_stride + hv;
        const int a_gate_offset = b * a_gate_b_stride + t * a_gate_t_stride + hv;

        const float beta_value =
            stable_sigmoid(scalar_to_float<T>(b_gate[b_gate_offset]));
        const float grad_beta_value = scalar_to_float<T>(grad_beta[gate_offset]);
        grad_b[grad_b_offset] = float_to_scalar<T>(
            grad_beta_value * beta_value * (1.0f - beta_value));

        const float u = scalar_to_float<T>(a_gate[a_gate_offset]) + dt_bias[hv];
        const float exp_A = __expf(A_log[hv]);
        const float softplus_u = softplus_default(u);
        const float g_value = -exp_A * softplus_u;
        const float grad_g_value = grad_g[gate_offset];
        const float grad_u = grad_g_value * (-exp_A) * stable_sigmoid(u);
        grad_a[grad_a_offset] = float_to_scalar<T>(grad_u);
        atomicAdd(grad_dt_bias + hv, grad_u);
        atomicAdd(grad_A_log + hv, grad_g_value * g_value);
    }
}

template <typename T>
__global__ void postConvPackBackwardQwen35Vec16Kernel(
    T* __restrict__ grad_mixed,
    const T* __restrict__ grad_query,
    const T* __restrict__ grad_key,
    const T* __restrict__ grad_value,
    int seq_length,
    int grad_mixed_b_stride,
    int grad_mixed_t_stride) {
    constexpr int kNumKHeads = 16;
    constexpr int kNumVHeads = 32;
    constexpr int kHeadDim = 128;
    constexpr int kKeyDim = kNumKHeads * kHeadDim;
    constexpr int kValueBase = 2 * kKeyDim;
    constexpr int kOutRowItems = kNumVHeads * kHeadDim;
    constexpr int kElemsPerVec = static_cast<int>(sizeof(uint4) / sizeof(T));
    constexpr int kHeadVecs = kHeadDim / kElemsPerVec;
    constexpr int kQkSourceVecs = kNumKHeads * kHeadVecs;
    constexpr int kValueVecs = kNumVHeads * kHeadVecs;

    const int row = static_cast<int>(blockIdx.x);
    const int t = row % seq_length;
    const int b = row / seq_length;
    const int base_grad_mixed = b * grad_mixed_b_stride + t * grad_mixed_t_stride;
    const int base_grad = row * kOutRowItems;

    for (int idx = threadIdx.x; idx < kValueVecs; idx += blockDim.x) {
        if (idx < kQkSourceVecs) {
            const int hk = idx / kHeadVecs;
            const int vec = idx - hk * kHeadVecs;
            const int d = vec * kElemsPerVec;
            const int grad_offset = base_grad + hk * 2 * kHeadDim + d;
            const uint4 q_grad = add_vec16<T>(
                *reinterpret_cast<const uint4*>(grad_query + grad_offset),
                *reinterpret_cast<const uint4*>(grad_query + grad_offset + kHeadDim));
            const uint4 k_grad = add_vec16<T>(
                *reinterpret_cast<const uint4*>(grad_key + grad_offset),
                *reinterpret_cast<const uint4*>(grad_key + grad_offset + kHeadDim));
            const int q_channel = hk * kHeadDim + d;
            const int k_channel = kKeyDim + q_channel;
            *reinterpret_cast<uint4*>(grad_mixed + base_grad_mixed + q_channel) = q_grad;
            *reinterpret_cast<uint4*>(grad_mixed + base_grad_mixed + k_channel) = k_grad;
        }

        const int hv = idx / kHeadVecs;
        const int vec = idx - hv * kHeadVecs;
        const int d = vec * kElemsPerVec;
        const int v_channel = kValueBase + hv * kHeadDim + d;
        const int grad_offset = base_grad + hv * kHeadDim + d;
        const uint4 v_grad = *reinterpret_cast<const uint4*>(grad_value + grad_offset);
        *reinterpret_cast<uint4*>(grad_mixed + base_grad_mixed + v_channel) = v_grad;
    }
}

template <typename T>
__global__ void postConvPackBackwardVec16Kernel(
    T* __restrict__ grad_mixed,
    const T* __restrict__ grad_query,
    const T* __restrict__ grad_key,
    const T* __restrict__ grad_value,
    int64_t batch_size,
    int64_t seq_length,
    int64_t grad_mixed_b_stride,
    int64_t grad_mixed_t_stride,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    int64_t repeat) {
    constexpr int64_t elems_per_vec = static_cast<int64_t>(sizeof(uint4)) /
                                      static_cast<int64_t>(sizeof(T));
    const int64_t row = blockIdx.x;
    const int64_t t = row % seq_length;
    const int64_t b = row / seq_length;
    const int64_t key_dim = num_k_heads * head_k_dim;
    const int64_t qk_head_vecs = head_k_dim / elems_per_vec;
    const int64_t qk_source_vecs = num_k_heads * qk_head_vecs;
    const int64_t value_head_vecs = head_v_dim / elems_per_vec;
    const int64_t value_vecs = num_v_heads * value_head_vecs;
    const int64_t base_grad_mixed = b * grad_mixed_b_stride + t * grad_mixed_t_stride;
    const int64_t base_qk_grad = (b * seq_length + t) * num_v_heads * head_k_dim;
    const int64_t base_v_grad = (b * seq_length + t) * num_v_heads * head_v_dim;

    for (int64_t idx = threadIdx.x; idx < qk_source_vecs; idx += blockDim.x) {
        const int64_t hk = idx / qk_head_vecs;
        const int64_t vec = idx - hk * qk_head_vecs;
        const int64_t d = vec * elems_per_vec;
        const int64_t out_h_base = hk * repeat;

        uint4 q_grad = *reinterpret_cast<const uint4*>(
            grad_query + base_qk_grad + out_h_base * head_k_dim + d);
        uint4 k_grad = *reinterpret_cast<const uint4*>(
            grad_key + base_qk_grad + out_h_base * head_k_dim + d);
        for (int64_t r = 1; r < repeat; ++r) {
            const int64_t grad_offset = base_qk_grad + (out_h_base + r) * head_k_dim + d;
            q_grad = add_vec16<T>(
                q_grad,
                *reinterpret_cast<const uint4*>(grad_query + grad_offset));
            k_grad = add_vec16<T>(
                k_grad,
                *reinterpret_cast<const uint4*>(grad_key + grad_offset));
        }

        const int64_t q_channel = hk * head_k_dim + d;
        const int64_t k_channel = key_dim + hk * head_k_dim + d;
        *reinterpret_cast<uint4*>(grad_mixed + base_grad_mixed + q_channel) = q_grad;
        *reinterpret_cast<uint4*>(grad_mixed + base_grad_mixed + k_channel) = k_grad;
    }

    for (int64_t idx = threadIdx.x; idx < value_vecs; idx += blockDim.x) {
        const int64_t hv = idx / value_head_vecs;
        const int64_t vec = idx - hv * value_head_vecs;
        const int64_t d = vec * elems_per_vec;
        const int64_t v_channel = 2 * key_dim + hv * head_v_dim + d;
        const uint4 v_grad = *reinterpret_cast<const uint4*>(
            grad_value + base_v_grad + hv * head_v_dim + d);
        *reinterpret_cast<uint4*>(grad_mixed + base_grad_mixed + v_channel) = v_grad;
    }
}

template <typename T>
__global__ void postConvPackBackwardKernel(
    T* __restrict__ grad_mixed,
    const T* __restrict__ grad_query,
    const T* __restrict__ grad_key,
    const T* __restrict__ grad_value,
    int64_t batch_size,
    int64_t seq_length,
    int64_t channels,
    int64_t grad_mixed_b_stride,
    int64_t grad_mixed_c_stride,
    int64_t grad_mixed_t_stride,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    int64_t repeat) {
    const int64_t key_dim = num_k_heads * head_k_dim;
    const int64_t row = blockIdx.x;
    const int64_t t = row % seq_length;
    const int64_t b = row / seq_length;
    const int64_t base_grad_mixed = b * grad_mixed_b_stride + t * grad_mixed_t_stride;
    const int64_t base_qk_grad = (b * seq_length + t) * num_v_heads * head_k_dim;
    const int64_t base_v_grad = (b * seq_length + t) * num_v_heads * head_v_dim;

    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {

        T grad;
        if (c < key_dim) {
            const int64_t hk = c / head_k_dim;
            const int64_t d = c - hk * head_k_dim;
            int64_t out_h = hk * repeat;
            grad = grad_query[base_qk_grad + out_h * head_k_dim + d];
            for (int64_t r = 1; r < repeat; ++r) {
                ++out_h;
                grad = grad + grad_query[base_qk_grad + out_h * head_k_dim + d];
            }
        } else if (c < 2 * key_dim) {
            const int64_t local_c = c - key_dim;
            const int64_t hk = local_c / head_k_dim;
            const int64_t d = local_c - hk * head_k_dim;
            int64_t out_h = hk * repeat;
            grad = grad_key[base_qk_grad + out_h * head_k_dim + d];
            for (int64_t r = 1; r < repeat; ++r) {
                ++out_h;
                grad = grad + grad_key[base_qk_grad + out_h * head_k_dim + d];
            }
        } else {
            const int64_t local_c = c - 2 * key_dim;
            const int64_t hv = local_c / head_v_dim;
            const int64_t d = local_c - hv * head_v_dim;
            grad = grad_value[base_v_grad + hv * head_v_dim + d];
        }

        grad_mixed[base_grad_mixed + c * grad_mixed_c_stride] = grad;
    }
}

template <typename T>
static void launch_post_conv_pack_preprocess_qwen35(
    const T* mixed,
    const T* b_gate,
    const T* a_gate,
    const float* A_log,
    const float* dt_bias,
    T* query,
    T* key,
    T* value,
    T* beta,
    float* g,
    float* q_rstd,
    float* k_rstd,
    int64_t batch_size,
    int64_t seq_length,
    int64_t mixed_b_stride,
    int64_t mixed_t_stride,
    int64_t b_gate_b_stride,
    int64_t b_gate_t_stride,
    int64_t a_gate_b_stride,
    int64_t a_gate_t_stride,
    cudaStream_t stream) {
    const int64_t rows = batch_size * seq_length;
    if (rows == 0) {
        return;
    }
    const int blocks = checked_blocks(rows, "gated_delta_post_conv_pack_preprocess");
    const dim3 preprocess_blocks(blocks, 2);
    postConvPackPreprocessQwen35ForwardKernel<T><<<preprocess_blocks, 1024, 0, stream>>>(
        mixed,
        b_gate,
        a_gate,
        A_log,
        dt_bias,
        query,
        key,
        value,
        beta,
        g,
        q_rstd,
        k_rstd,
        static_cast<int>(seq_length),
        static_cast<int>(mixed_b_stride),
        static_cast<int>(mixed_t_stride),
        static_cast<int>(b_gate_b_stride),
        static_cast<int>(b_gate_t_stride),
        static_cast<int>(a_gate_b_stride),
        static_cast<int>(a_gate_t_stride));
}

template <typename T>
static void launch_post_conv_pack_preprocess_backward_qwen35(
    T* grad_mixed,
    T* grad_b,
    T* grad_a,
    float* grad_A_log,
    float* grad_dt_bias,
    const T* grad_query,
    const T* grad_key,
    const T* grad_value,
    const T* grad_beta,
    const float* grad_g,
    const T* query,
    const T* key,
    const float* q_rstd,
    const float* k_rstd,
    const T* b_gate,
    const T* a_gate,
    const float* A_log,
    const float* dt_bias,
    int64_t batch_size,
    int64_t seq_length,
    int64_t grad_mixed_b_stride,
    int64_t grad_mixed_t_stride,
    int64_t grad_b_b_stride,
    int64_t grad_b_t_stride,
    int64_t grad_a_b_stride,
    int64_t grad_a_t_stride,
    int64_t b_gate_b_stride,
    int64_t b_gate_t_stride,
    int64_t a_gate_b_stride,
    int64_t a_gate_t_stride,
    cudaStream_t stream) {
    const int64_t rows = batch_size * seq_length;
    if (rows == 0) {
        return;
    }
    const int blocks = checked_blocks(rows, "gated_delta_post_conv_pack_preprocess_backward");
    const dim3 preprocess_blocks(blocks, 18);
    postConvPackPreprocessBackwardQwen35Kernel<T><<<preprocess_blocks, 128, 0, stream>>>(
        grad_mixed,
        grad_b,
        grad_a,
        grad_A_log,
        grad_dt_bias,
        grad_query,
        grad_key,
        grad_value,
        grad_beta,
        grad_g,
        query,
        key,
        q_rstd,
        k_rstd,
        b_gate,
        a_gate,
        A_log,
        dt_bias,
        static_cast<int>(seq_length),
        static_cast<int>(grad_mixed_b_stride),
        static_cast<int>(grad_mixed_t_stride),
        static_cast<int>(grad_b_b_stride),
        static_cast<int>(grad_b_t_stride),
        static_cast<int>(grad_a_b_stride),
        static_cast<int>(grad_a_t_stride),
        static_cast<int>(b_gate_b_stride),
        static_cast<int>(b_gate_t_stride),
        static_cast<int>(a_gate_b_stride),
        static_cast<int>(a_gate_t_stride));
}

template <typename T>
static void launch_post_conv_pack(
    const T* mixed,
    T* query,
    T* key,
    T* value,
    int64_t batch_size,
    int64_t seq_length,
    int64_t channels,
    int64_t mixed_b_stride,
    int64_t mixed_c_stride,
    int64_t mixed_t_stride,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    cudaStream_t stream) {
    const int64_t rows = batch_size * seq_length;
    if (rows == 0) {
        return;
    }
    const int threads = 256;
    const int qwen35_threads = 512;
    const int blocks = checked_blocks(rows, "gated_delta_post_conv_pack");
    if (can_use_qwen35_vec16_post_conv_pack<T>(
            mixed,
            query,
            key,
            value,
            batch_size,
            seq_length,
            mixed_b_stride,
            mixed_c_stride,
            mixed_t_stride,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim)) {
        const dim3 qwen35_blocks(blocks, 2);
        postConvPackQwen35Vec16Kernel<T><<<qwen35_blocks, qwen35_threads, 0, stream>>>(
            mixed,
            query,
            key,
            value,
            static_cast<int>(seq_length),
            static_cast<int>(mixed_b_stride),
            static_cast<int>(mixed_t_stride));
        return;
    }
    if (can_use_vec16_post_conv_pack<T>(
            mixed,
            query,
            key,
            value,
            mixed_b_stride,
            mixed_c_stride,
            mixed_t_stride,
            num_v_heads,
            head_k_dim,
            head_v_dim)) {
        postConvPackVec16Kernel<T><<<blocks, threads, 0, stream>>>(
            mixed,
            query,
            key,
            value,
            batch_size,
            seq_length,
            mixed_b_stride,
            mixed_t_stride,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            num_v_heads / num_k_heads);
        return;
    }

    postConvPackKernel<T><<<blocks, threads, 0, stream>>>(
        mixed,
        query,
        key,
        value,
        batch_size,
        seq_length,
        channels,
        mixed_b_stride,
        mixed_c_stride,
        mixed_t_stride,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        num_v_heads / num_k_heads);
}

template <typename T>
static void launch_post_conv_pack_backward(
    T* grad_mixed,
    const T* grad_query,
    const T* grad_key,
    const T* grad_value,
    int64_t batch_size,
    int64_t seq_length,
    int64_t channels,
    int64_t grad_mixed_b_stride,
    int64_t grad_mixed_c_stride,
    int64_t grad_mixed_t_stride,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    cudaStream_t stream) {
    const int64_t rows = batch_size * seq_length;
    if (rows == 0) {
        return;
    }
    const int threads = 256;
    const int qwen35_threads = 1024;
    const int blocks = checked_blocks(rows, "gated_delta_post_conv_pack_backward");
    if (can_use_qwen35_vec16_post_conv_pack_backward<T>(
            grad_mixed,
            grad_query,
            grad_key,
            grad_value,
            batch_size,
            seq_length,
            grad_mixed_b_stride,
            grad_mixed_c_stride,
            grad_mixed_t_stride,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim)) {
        postConvPackBackwardQwen35Vec16Kernel<T><<<blocks, qwen35_threads, 0, stream>>>(
            grad_mixed,
            grad_query,
            grad_key,
            grad_value,
            static_cast<int>(seq_length),
            static_cast<int>(grad_mixed_b_stride),
            static_cast<int>(grad_mixed_t_stride));
        return;
    }
    if (can_use_vec16_post_conv_pack_backward<T>(
            grad_mixed,
            grad_query,
            grad_key,
            grad_value,
            grad_mixed_b_stride,
            grad_mixed_c_stride,
            grad_mixed_t_stride,
            num_v_heads,
            head_k_dim,
            head_v_dim)) {
        postConvPackBackwardVec16Kernel<T><<<blocks, threads, 0, stream>>>(
            grad_mixed,
            grad_query,
            grad_key,
            grad_value,
            batch_size,
            seq_length,
            grad_mixed_b_stride,
            grad_mixed_t_stride,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            num_v_heads / num_k_heads);
        return;
    }

    postConvPackBackwardKernel<T><<<blocks, threads, 0, stream>>>(
        grad_mixed,
        grad_query,
        grad_key,
        grad_value,
        batch_size,
        seq_length,
        channels,
        grad_mixed_b_stride,
        grad_mixed_c_stride,
        grad_mixed_t_stride,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        num_v_heads / num_k_heads);
}

void GatedDeltaPostConvPack(
    wallx_cuda::TensorView query,
    wallx_cuda::TensorView key,
    wallx_cuda::TensorView value,
    wallx_cuda::TensorView mixed_qkv,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
    validate_dims(num_k_heads, num_v_heads, head_k_dim, head_v_dim);
    ASSERT_CHECK(mixed_qkv.ndim() == 3);
    ASSERT_CHECK(query.ndim() == 4);
    ASSERT_CHECK(key.ndim() == 4);
    ASSERT_CHECK(value.ndim() == 4);
    ASSERT_CHECK(query.IsContiguous());
    ASSERT_CHECK(key.IsContiguous());
    ASSERT_CHECK(value.IsContiguous());
    check_supported_dtype(mixed_qkv, "mixed_qkv");
    check_same_device_and_dtype(query, mixed_qkv, "query");
    check_same_device_and_dtype(key, mixed_qkv, "key");
    check_same_device_and_dtype(value, mixed_qkv, "value");

    const int64_t batch_size = mixed_qkv.size(0);
    const int64_t channels = mixed_qkv.size(1);
    const int64_t seq_length = mixed_qkv.size(2);
    const int64_t key_dim = num_k_heads * head_k_dim;
    const int64_t value_dim = num_v_heads * head_v_dim;
    ASSERT_CHECK(channels == 2 * key_dim + value_dim);
    ASSERT_CHECK(query.size(0) == batch_size);
    ASSERT_CHECK(query.size(1) == seq_length);
    ASSERT_CHECK(query.size(2) == num_v_heads);
    ASSERT_CHECK(query.size(3) == head_k_dim);
    ASSERT_CHECK(key.size(0) == batch_size);
    ASSERT_CHECK(key.size(1) == seq_length);
    ASSERT_CHECK(key.size(2) == num_v_heads);
    ASSERT_CHECK(key.size(3) == head_k_dim);
    ASSERT_CHECK(value.size(0) == batch_size);
    ASSERT_CHECK(value.size(1) == seq_length);
    ASSERT_CHECK(value.size(2) == num_v_heads);
    ASSERT_CHECK(value.size(3) == head_v_dim);

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(mixed_qkv.device().device_type, mixed_qkv.device().device_id));

    if (mixed_qkv.dtype().code == kDLFloat && mixed_qkv.dtype().bits == 32) {
        launch_post_conv_pack<float>(
            static_cast<const float*>(mixed_qkv.data_ptr()),
            static_cast<float*>(query.data_ptr()),
            static_cast<float*>(key.data_ptr()),
            static_cast<float*>(value.data_ptr()),
            batch_size,
            seq_length,
            channels,
            mixed_qkv.stride(0),
            mixed_qkv.stride(1),
            mixed_qkv.stride(2),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            stream);
    } else if (mixed_qkv.dtype().code == kDLFloat && mixed_qkv.dtype().bits == 16) {
        launch_post_conv_pack<__half>(
            static_cast<const __half*>(mixed_qkv.data_ptr()),
            static_cast<__half*>(query.data_ptr()),
            static_cast<__half*>(key.data_ptr()),
            static_cast<__half*>(value.data_ptr()),
            batch_size,
            seq_length,
            channels,
            mixed_qkv.stride(0),
            mixed_qkv.stride(1),
            mixed_qkv.stride(2),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            stream);
    } else if (mixed_qkv.dtype().code == kDLBfloat && mixed_qkv.dtype().bits == 16) {
        launch_post_conv_pack<__nv_bfloat16>(
            static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()),
            static_cast<__nv_bfloat16*>(query.data_ptr()),
            static_cast<__nv_bfloat16*>(key.data_ptr()),
            static_cast<__nv_bfloat16*>(value.data_ptr()),
            batch_size,
            seq_length,
            channels,
            mixed_qkv.stride(0),
            mixed_qkv.stride(1),
            mixed_qkv.stride(2),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            stream);
    } else {
        throw std::runtime_error("gated_delta_post_conv_pack: unsupported dtype");
    }
    sync_check_cuda_error();
}

void GatedDeltaPostConvPackPreprocess(
    wallx_cuda::TensorView query,
    wallx_cuda::TensorView key,
    wallx_cuda::TensorView value,
    wallx_cuda::TensorView beta,
    wallx_cuda::TensorView g,
    wallx_cuda::TensorView q_rstd,
    wallx_cuda::TensorView k_rstd,
    wallx_cuda::TensorView mixed_qkv,
    wallx_cuda::TensorView b_gate,
    wallx_cuda::TensorView a_gate,
    wallx_cuda::TensorView A_log,
    wallx_cuda::TensorView dt_bias,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
    validate_dims(num_k_heads, num_v_heads, head_k_dim, head_v_dim);
    if (num_k_heads != 16 || num_v_heads != 32 || head_k_dim != 128 || head_v_dim != 128) {
        throw std::runtime_error("gated_delta_post_conv_pack_preprocess: only Qwen3.5 16/32/128 fast path is supported");
    }
    ASSERT_CHECK(mixed_qkv.ndim() == 3);
    ASSERT_CHECK(query.ndim() == 4);
    ASSERT_CHECK(key.ndim() == 4);
    ASSERT_CHECK(value.ndim() == 4);
    ASSERT_CHECK(beta.ndim() == 3);
    ASSERT_CHECK(g.ndim() == 3);
    ASSERT_CHECK(q_rstd.ndim() == 3);
    ASSERT_CHECK(k_rstd.ndim() == 3);
    ASSERT_CHECK(b_gate.ndim() == 3);
    ASSERT_CHECK(a_gate.ndim() == 3);
    ASSERT_CHECK(A_log.ndim() == 1);
    ASSERT_CHECK(dt_bias.ndim() == 1);
    ASSERT_CHECK(query.IsContiguous());
    ASSERT_CHECK(key.IsContiguous());
    ASSERT_CHECK(value.IsContiguous());
    ASSERT_CHECK(beta.IsContiguous());
    ASSERT_CHECK(g.IsContiguous());
    ASSERT_CHECK(q_rstd.IsContiguous());
    ASSERT_CHECK(k_rstd.IsContiguous());
    ASSERT_CHECK(b_gate.stride(2) == 1);
    ASSERT_CHECK(a_gate.stride(2) == 1);
    ASSERT_CHECK(A_log.IsContiguous());
    ASSERT_CHECK(dt_bias.IsContiguous());
    check_supported_dtype(mixed_qkv, "mixed_qkv");
    check_same_device_and_dtype(query, mixed_qkv, "query");
    check_same_device_and_dtype(key, mixed_qkv, "key");
    check_same_device_and_dtype(value, mixed_qkv, "value");
    check_same_device_and_dtype(beta, mixed_qkv, "beta");
    check_same_device_and_dtype(b_gate, mixed_qkv, "b_gate");
    check_same_device_and_dtype(a_gate, mixed_qkv, "a_gate");
    check_same_device(g, mixed_qkv, "g");
    check_same_device(q_rstd, mixed_qkv, "q_rstd");
    check_same_device(k_rstd, mixed_qkv, "k_rstd");
    check_same_device(A_log, mixed_qkv, "A_log");
    check_same_device(dt_bias, mixed_qkv, "dt_bias");
    check_float32(g, "g");
    check_float32(q_rstd, "q_rstd");
    check_float32(k_rstd, "k_rstd");
    check_float32(A_log, "A_log");
    check_float32(dt_bias, "dt_bias");

    const int64_t batch_size = mixed_qkv.size(0);
    const int64_t channels = mixed_qkv.size(1);
    const int64_t seq_length = mixed_qkv.size(2);
    ASSERT_CHECK(channels == 8192);
    ASSERT_CHECK(mixed_qkv.stride(1) == 1);
    ASSERT_CHECK(fits_i32(batch_size * seq_length));
    ASSERT_CHECK(fits_i32(seq_length));
    ASSERT_CHECK(fits_i32(mixed_qkv.stride(0)));
    ASSERT_CHECK(fits_i32(mixed_qkv.stride(2)));
    ASSERT_CHECK(fits_i32(b_gate.stride(0)));
    ASSERT_CHECK(fits_i32(b_gate.stride(1)));
    ASSERT_CHECK(fits_i32(a_gate.stride(0)));
    ASSERT_CHECK(fits_i32(a_gate.stride(1)));
    ASSERT_CHECK(query.size(0) == batch_size);
    ASSERT_CHECK(query.size(1) == seq_length);
    ASSERT_CHECK(query.size(2) == num_v_heads);
    ASSERT_CHECK(query.size(3) == head_k_dim);
    ASSERT_CHECK(key.size(0) == batch_size);
    ASSERT_CHECK(key.size(1) == seq_length);
    ASSERT_CHECK(key.size(2) == num_v_heads);
    ASSERT_CHECK(key.size(3) == head_k_dim);
    ASSERT_CHECK(value.size(0) == batch_size);
    ASSERT_CHECK(value.size(1) == seq_length);
    ASSERT_CHECK(value.size(2) == num_v_heads);
    ASSERT_CHECK(value.size(3) == head_v_dim);
    ASSERT_CHECK(beta.size(0) == batch_size);
    ASSERT_CHECK(beta.size(1) == seq_length);
    ASSERT_CHECK(beta.size(2) == num_v_heads);
    ASSERT_CHECK(g.size(0) == batch_size);
    ASSERT_CHECK(g.size(1) == seq_length);
    ASSERT_CHECK(g.size(2) == num_v_heads);
    ASSERT_CHECK(b_gate.size(0) == batch_size);
    ASSERT_CHECK(b_gate.size(1) == seq_length);
    ASSERT_CHECK(b_gate.size(2) == num_v_heads);
    ASSERT_CHECK(a_gate.size(0) == batch_size);
    ASSERT_CHECK(a_gate.size(1) == seq_length);
    ASSERT_CHECK(a_gate.size(2) == num_v_heads);
    ASSERT_CHECK(q_rstd.size(0) == batch_size);
    ASSERT_CHECK(q_rstd.size(1) == seq_length);
    ASSERT_CHECK(q_rstd.size(2) == num_k_heads);
    ASSERT_CHECK(k_rstd.size(0) == batch_size);
    ASSERT_CHECK(k_rstd.size(1) == seq_length);
    ASSERT_CHECK(k_rstd.size(2) == num_k_heads);
    ASSERT_CHECK(A_log.size(0) == num_v_heads);
    ASSERT_CHECK(dt_bias.size(0) == num_v_heads);

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(mixed_qkv.device().device_type, mixed_qkv.device().device_id));

    if (mixed_qkv.dtype().code == kDLFloat && mixed_qkv.dtype().bits == 32) {
        launch_post_conv_pack_preprocess_qwen35<float>(
            static_cast<const float*>(mixed_qkv.data_ptr()),
            static_cast<const float*>(b_gate.data_ptr()),
            static_cast<const float*>(a_gate.data_ptr()),
            static_cast<const float*>(A_log.data_ptr()),
            static_cast<const float*>(dt_bias.data_ptr()),
            static_cast<float*>(query.data_ptr()),
            static_cast<float*>(key.data_ptr()),
            static_cast<float*>(value.data_ptr()),
            static_cast<float*>(beta.data_ptr()),
            static_cast<float*>(g.data_ptr()),
            static_cast<float*>(q_rstd.data_ptr()),
            static_cast<float*>(k_rstd.data_ptr()),
            batch_size,
            seq_length,
            mixed_qkv.stride(0),
            mixed_qkv.stride(2),
            b_gate.stride(0),
            b_gate.stride(1),
            a_gate.stride(0),
            a_gate.stride(1),
            stream);
    } else if (mixed_qkv.dtype().code == kDLFloat && mixed_qkv.dtype().bits == 16) {
        launch_post_conv_pack_preprocess_qwen35<__half>(
            static_cast<const __half*>(mixed_qkv.data_ptr()),
            static_cast<const __half*>(b_gate.data_ptr()),
            static_cast<const __half*>(a_gate.data_ptr()),
            static_cast<const float*>(A_log.data_ptr()),
            static_cast<const float*>(dt_bias.data_ptr()),
            static_cast<__half*>(query.data_ptr()),
            static_cast<__half*>(key.data_ptr()),
            static_cast<__half*>(value.data_ptr()),
            static_cast<__half*>(beta.data_ptr()),
            static_cast<float*>(g.data_ptr()),
            static_cast<float*>(q_rstd.data_ptr()),
            static_cast<float*>(k_rstd.data_ptr()),
            batch_size,
            seq_length,
            mixed_qkv.stride(0),
            mixed_qkv.stride(2),
            b_gate.stride(0),
            b_gate.stride(1),
            a_gate.stride(0),
            a_gate.stride(1),
            stream);
    } else if (mixed_qkv.dtype().code == kDLBfloat && mixed_qkv.dtype().bits == 16) {
        launch_post_conv_pack_preprocess_qwen35<__nv_bfloat16>(
            static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()),
            static_cast<const __nv_bfloat16*>(b_gate.data_ptr()),
            static_cast<const __nv_bfloat16*>(a_gate.data_ptr()),
            static_cast<const float*>(A_log.data_ptr()),
            static_cast<const float*>(dt_bias.data_ptr()),
            static_cast<__nv_bfloat16*>(query.data_ptr()),
            static_cast<__nv_bfloat16*>(key.data_ptr()),
            static_cast<__nv_bfloat16*>(value.data_ptr()),
            static_cast<__nv_bfloat16*>(beta.data_ptr()),
            static_cast<float*>(g.data_ptr()),
            static_cast<float*>(q_rstd.data_ptr()),
            static_cast<float*>(k_rstd.data_ptr()),
            batch_size,
            seq_length,
            mixed_qkv.stride(0),
            mixed_qkv.stride(2),
            b_gate.stride(0),
            b_gate.stride(1),
            a_gate.stride(0),
            a_gate.stride(1),
            stream);
    } else {
        throw std::runtime_error("gated_delta_post_conv_pack_preprocess: unsupported dtype");
    }
    sync_check_cuda_error();
}

void GatedDeltaPostConvPackBackward(
    wallx_cuda::TensorView grad_mixed_qkv,
    wallx_cuda::TensorView grad_query,
    wallx_cuda::TensorView grad_key,
    wallx_cuda::TensorView grad_value,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
    validate_dims(num_k_heads, num_v_heads, head_k_dim, head_v_dim);
    ASSERT_CHECK(grad_mixed_qkv.ndim() == 3);
    ASSERT_CHECK(grad_query.ndim() == 4);
    ASSERT_CHECK(grad_key.ndim() == 4);
    ASSERT_CHECK(grad_value.ndim() == 4);
    ASSERT_CHECK(grad_query.IsContiguous());
    ASSERT_CHECK(grad_key.IsContiguous());
    ASSERT_CHECK(grad_value.IsContiguous());
    check_supported_dtype(grad_mixed_qkv, "grad_mixed_qkv");
    check_same_device_and_dtype(grad_query, grad_mixed_qkv, "grad_query");
    check_same_device_and_dtype(grad_key, grad_mixed_qkv, "grad_key");
    check_same_device_and_dtype(grad_value, grad_mixed_qkv, "grad_value");

    const int64_t batch_size = grad_mixed_qkv.size(0);
    const int64_t channels = grad_mixed_qkv.size(1);
    const int64_t seq_length = grad_mixed_qkv.size(2);
    const int64_t key_dim = num_k_heads * head_k_dim;
    const int64_t value_dim = num_v_heads * head_v_dim;
    ASSERT_CHECK(channels == 2 * key_dim + value_dim);
    ASSERT_CHECK(grad_query.size(0) == batch_size);
    ASSERT_CHECK(grad_query.size(1) == seq_length);
    ASSERT_CHECK(grad_query.size(2) == num_v_heads);
    ASSERT_CHECK(grad_query.size(3) == head_k_dim);
    ASSERT_CHECK(grad_key.size(0) == batch_size);
    ASSERT_CHECK(grad_key.size(1) == seq_length);
    ASSERT_CHECK(grad_key.size(2) == num_v_heads);
    ASSERT_CHECK(grad_key.size(3) == head_k_dim);
    ASSERT_CHECK(grad_value.size(0) == batch_size);
    ASSERT_CHECK(grad_value.size(1) == seq_length);
    ASSERT_CHECK(grad_value.size(2) == num_v_heads);
    ASSERT_CHECK(grad_value.size(3) == head_v_dim);

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(grad_mixed_qkv.device().device_type, grad_mixed_qkv.device().device_id));

    if (grad_mixed_qkv.dtype().code == kDLFloat && grad_mixed_qkv.dtype().bits == 32) {
        launch_post_conv_pack_backward<float>(
            static_cast<float*>(grad_mixed_qkv.data_ptr()),
            static_cast<const float*>(grad_query.data_ptr()),
            static_cast<const float*>(grad_key.data_ptr()),
            static_cast<const float*>(grad_value.data_ptr()),
            batch_size,
            seq_length,
            channels,
            grad_mixed_qkv.stride(0),
            grad_mixed_qkv.stride(1),
            grad_mixed_qkv.stride(2),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            stream);
    } else if (grad_mixed_qkv.dtype().code == kDLFloat && grad_mixed_qkv.dtype().bits == 16) {
        launch_post_conv_pack_backward<__half>(
            static_cast<__half*>(grad_mixed_qkv.data_ptr()),
            static_cast<const __half*>(grad_query.data_ptr()),
            static_cast<const __half*>(grad_key.data_ptr()),
            static_cast<const __half*>(grad_value.data_ptr()),
            batch_size,
            seq_length,
            channels,
            grad_mixed_qkv.stride(0),
            grad_mixed_qkv.stride(1),
            grad_mixed_qkv.stride(2),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            stream);
    } else if (grad_mixed_qkv.dtype().code == kDLBfloat && grad_mixed_qkv.dtype().bits == 16) {
        launch_post_conv_pack_backward<__nv_bfloat16>(
            static_cast<__nv_bfloat16*>(grad_mixed_qkv.data_ptr()),
            static_cast<const __nv_bfloat16*>(grad_query.data_ptr()),
            static_cast<const __nv_bfloat16*>(grad_key.data_ptr()),
            static_cast<const __nv_bfloat16*>(grad_value.data_ptr()),
            batch_size,
            seq_length,
            channels,
            grad_mixed_qkv.stride(0),
            grad_mixed_qkv.stride(1),
            grad_mixed_qkv.stride(2),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            stream);
    } else {
        throw std::runtime_error("gated_delta_post_conv_pack_backward: unsupported dtype");
    }
    sync_check_cuda_error();
}

void GatedDeltaPostConvPackPreprocessBackward(
    wallx_cuda::TensorView grad_mixed_qkv,
    wallx_cuda::TensorView grad_b,
    wallx_cuda::TensorView grad_a,
    wallx_cuda::TensorView grad_A_log,
    wallx_cuda::TensorView grad_dt_bias,
    wallx_cuda::TensorView grad_query,
    wallx_cuda::TensorView grad_key,
    wallx_cuda::TensorView grad_value,
    wallx_cuda::TensorView grad_beta,
    wallx_cuda::TensorView grad_g,
    wallx_cuda::TensorView query,
    wallx_cuda::TensorView key,
    wallx_cuda::TensorView q_rstd,
    wallx_cuda::TensorView k_rstd,
    wallx_cuda::TensorView b_gate,
    wallx_cuda::TensorView a_gate,
    wallx_cuda::TensorView A_log,
    wallx_cuda::TensorView dt_bias,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
    validate_dims(num_k_heads, num_v_heads, head_k_dim, head_v_dim);
    if (num_k_heads != 16 || num_v_heads != 32 || head_k_dim != 128 || head_v_dim != 128) {
        throw std::runtime_error("gated_delta_post_conv_pack_preprocess_backward: only Qwen3.5 16/32/128 fast path is supported");
    }
    ASSERT_CHECK(grad_mixed_qkv.ndim() == 3);
    ASSERT_CHECK(grad_b.ndim() == 3);
    ASSERT_CHECK(grad_a.ndim() == 3);
    ASSERT_CHECK(grad_A_log.ndim() == 1);
    ASSERT_CHECK(grad_dt_bias.ndim() == 1);
    ASSERT_CHECK(grad_query.ndim() == 4);
    ASSERT_CHECK(grad_key.ndim() == 4);
    ASSERT_CHECK(grad_value.ndim() == 4);
    ASSERT_CHECK(grad_beta.ndim() == 3);
    ASSERT_CHECK(grad_g.ndim() == 3);
    ASSERT_CHECK(query.ndim() == 4);
    ASSERT_CHECK(key.ndim() == 4);
    ASSERT_CHECK(q_rstd.ndim() == 3);
    ASSERT_CHECK(k_rstd.ndim() == 3);
    ASSERT_CHECK(b_gate.ndim() == 3);
    ASSERT_CHECK(a_gate.ndim() == 3);
    ASSERT_CHECK(A_log.ndim() == 1);
    ASSERT_CHECK(dt_bias.ndim() == 1);
    ASSERT_CHECK(grad_query.IsContiguous());
    ASSERT_CHECK(grad_key.IsContiguous());
    ASSERT_CHECK(grad_value.IsContiguous());
    ASSERT_CHECK(grad_beta.IsContiguous());
    ASSERT_CHECK(grad_g.IsContiguous());
    ASSERT_CHECK(query.IsContiguous());
    ASSERT_CHECK(key.IsContiguous());
    ASSERT_CHECK(q_rstd.IsContiguous());
    ASSERT_CHECK(k_rstd.IsContiguous());
    ASSERT_CHECK(grad_b.stride(2) == 1);
    ASSERT_CHECK(grad_a.stride(2) == 1);
    ASSERT_CHECK(grad_A_log.IsContiguous());
    ASSERT_CHECK(grad_dt_bias.IsContiguous());
    ASSERT_CHECK(b_gate.stride(2) == 1);
    ASSERT_CHECK(a_gate.stride(2) == 1);
    ASSERT_CHECK(A_log.IsContiguous());
    ASSERT_CHECK(dt_bias.IsContiguous());
    check_supported_dtype(grad_mixed_qkv, "grad_mixed_qkv");
    check_same_device_and_dtype(grad_query, grad_mixed_qkv, "grad_query");
    check_same_device_and_dtype(grad_key, grad_mixed_qkv, "grad_key");
    check_same_device_and_dtype(grad_value, grad_mixed_qkv, "grad_value");
    check_same_device_and_dtype(grad_beta, grad_mixed_qkv, "grad_beta");
    check_same_device_and_dtype(query, grad_mixed_qkv, "query");
    check_same_device_and_dtype(key, grad_mixed_qkv, "key");
    check_same_device_and_dtype(grad_b, grad_mixed_qkv, "grad_b");
    check_same_device_and_dtype(grad_a, grad_mixed_qkv, "grad_a");
    check_same_device_and_dtype(b_gate, grad_mixed_qkv, "b_gate");
    check_same_device_and_dtype(a_gate, grad_mixed_qkv, "a_gate");
    check_same_device(grad_A_log, grad_mixed_qkv, "grad_A_log");
    check_same_device(grad_dt_bias, grad_mixed_qkv, "grad_dt_bias");
    check_same_device(grad_g, grad_mixed_qkv, "grad_g");
    check_same_device(q_rstd, grad_mixed_qkv, "q_rstd");
    check_same_device(k_rstd, grad_mixed_qkv, "k_rstd");
    check_same_device(A_log, grad_mixed_qkv, "A_log");
    check_same_device(dt_bias, grad_mixed_qkv, "dt_bias");
    check_float32(grad_A_log, "grad_A_log");
    check_float32(grad_dt_bias, "grad_dt_bias");
    check_float32(grad_g, "grad_g");
    check_float32(q_rstd, "q_rstd");
    check_float32(k_rstd, "k_rstd");
    check_float32(A_log, "A_log");
    check_float32(dt_bias, "dt_bias");

    const int64_t batch_size = grad_mixed_qkv.size(0);
    const int64_t channels = grad_mixed_qkv.size(1);
    const int64_t seq_length = grad_mixed_qkv.size(2);
    ASSERT_CHECK(channels == 8192);
    ASSERT_CHECK(grad_mixed_qkv.stride(1) == 1);
    ASSERT_CHECK(fits_i32(batch_size * seq_length));
    ASSERT_CHECK(fits_i32(seq_length));
    ASSERT_CHECK(fits_i32(grad_mixed_qkv.stride(0)));
    ASSERT_CHECK(fits_i32(grad_mixed_qkv.stride(2)));
    ASSERT_CHECK(fits_i32(grad_b.stride(0)));
    ASSERT_CHECK(fits_i32(grad_b.stride(1)));
    ASSERT_CHECK(fits_i32(grad_a.stride(0)));
    ASSERT_CHECK(fits_i32(grad_a.stride(1)));
    ASSERT_CHECK(fits_i32(b_gate.stride(0)));
    ASSERT_CHECK(fits_i32(b_gate.stride(1)));
    ASSERT_CHECK(fits_i32(a_gate.stride(0)));
    ASSERT_CHECK(fits_i32(a_gate.stride(1)));
    ASSERT_CHECK(grad_query.size(0) == batch_size);
    ASSERT_CHECK(grad_query.size(1) == seq_length);
    ASSERT_CHECK(grad_query.size(2) == num_v_heads);
    ASSERT_CHECK(grad_query.size(3) == head_k_dim);
    ASSERT_CHECK(grad_key.size(0) == batch_size);
    ASSERT_CHECK(grad_key.size(1) == seq_length);
    ASSERT_CHECK(grad_key.size(2) == num_v_heads);
    ASSERT_CHECK(grad_key.size(3) == head_k_dim);
    ASSERT_CHECK(grad_value.size(0) == batch_size);
    ASSERT_CHECK(grad_value.size(1) == seq_length);
    ASSERT_CHECK(grad_value.size(2) == num_v_heads);
    ASSERT_CHECK(grad_value.size(3) == head_v_dim);
    ASSERT_CHECK(grad_beta.size(0) == batch_size);
    ASSERT_CHECK(grad_beta.size(1) == seq_length);
    ASSERT_CHECK(grad_beta.size(2) == num_v_heads);
    ASSERT_CHECK(grad_g.size(0) == batch_size);
    ASSERT_CHECK(grad_g.size(1) == seq_length);
    ASSERT_CHECK(grad_g.size(2) == num_v_heads);
    ASSERT_CHECK(query.size(0) == batch_size);
    ASSERT_CHECK(query.size(1) == seq_length);
    ASSERT_CHECK(query.size(2) == num_v_heads);
    ASSERT_CHECK(query.size(3) == head_k_dim);
    ASSERT_CHECK(key.size(0) == batch_size);
    ASSERT_CHECK(key.size(1) == seq_length);
    ASSERT_CHECK(key.size(2) == num_v_heads);
    ASSERT_CHECK(key.size(3) == head_k_dim);
    ASSERT_CHECK(b_gate.size(0) == batch_size);
    ASSERT_CHECK(b_gate.size(1) == seq_length);
    ASSERT_CHECK(b_gate.size(2) == num_v_heads);
    ASSERT_CHECK(a_gate.size(0) == batch_size);
    ASSERT_CHECK(a_gate.size(1) == seq_length);
    ASSERT_CHECK(a_gate.size(2) == num_v_heads);
    ASSERT_CHECK(grad_b.size(0) == batch_size);
    ASSERT_CHECK(grad_b.size(1) == seq_length);
    ASSERT_CHECK(grad_b.size(2) == num_v_heads);
    ASSERT_CHECK(grad_a.size(0) == batch_size);
    ASSERT_CHECK(grad_a.size(1) == seq_length);
    ASSERT_CHECK(grad_a.size(2) == num_v_heads);
    ASSERT_CHECK(q_rstd.size(0) == batch_size);
    ASSERT_CHECK(q_rstd.size(1) == seq_length);
    ASSERT_CHECK(q_rstd.size(2) == num_k_heads);
    ASSERT_CHECK(k_rstd.size(0) == batch_size);
    ASSERT_CHECK(k_rstd.size(1) == seq_length);
    ASSERT_CHECK(k_rstd.size(2) == num_k_heads);
    ASSERT_CHECK(grad_A_log.size(0) == num_v_heads);
    ASSERT_CHECK(grad_dt_bias.size(0) == num_v_heads);
    ASSERT_CHECK(A_log.size(0) == num_v_heads);
    ASSERT_CHECK(dt_bias.size(0) == num_v_heads);

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(grad_mixed_qkv.device().device_type, grad_mixed_qkv.device().device_id));

    if (grad_mixed_qkv.dtype().code == kDLFloat && grad_mixed_qkv.dtype().bits == 32) {
        launch_post_conv_pack_preprocess_backward_qwen35<float>(
            static_cast<float*>(grad_mixed_qkv.data_ptr()),
            static_cast<float*>(grad_b.data_ptr()),
            static_cast<float*>(grad_a.data_ptr()),
            static_cast<float*>(grad_A_log.data_ptr()),
            static_cast<float*>(grad_dt_bias.data_ptr()),
            static_cast<const float*>(grad_query.data_ptr()),
            static_cast<const float*>(grad_key.data_ptr()),
            static_cast<const float*>(grad_value.data_ptr()),
            static_cast<const float*>(grad_beta.data_ptr()),
            static_cast<const float*>(grad_g.data_ptr()),
            static_cast<const float*>(query.data_ptr()),
            static_cast<const float*>(key.data_ptr()),
            static_cast<const float*>(q_rstd.data_ptr()),
            static_cast<const float*>(k_rstd.data_ptr()),
            static_cast<const float*>(b_gate.data_ptr()),
            static_cast<const float*>(a_gate.data_ptr()),
            static_cast<const float*>(A_log.data_ptr()),
            static_cast<const float*>(dt_bias.data_ptr()),
            batch_size,
            seq_length,
            grad_mixed_qkv.stride(0),
            grad_mixed_qkv.stride(2),
            grad_b.stride(0),
            grad_b.stride(1),
            grad_a.stride(0),
            grad_a.stride(1),
            b_gate.stride(0),
            b_gate.stride(1),
            a_gate.stride(0),
            a_gate.stride(1),
            stream);
    } else if (grad_mixed_qkv.dtype().code == kDLFloat && grad_mixed_qkv.dtype().bits == 16) {
        launch_post_conv_pack_preprocess_backward_qwen35<__half>(
            static_cast<__half*>(grad_mixed_qkv.data_ptr()),
            static_cast<__half*>(grad_b.data_ptr()),
            static_cast<__half*>(grad_a.data_ptr()),
            static_cast<float*>(grad_A_log.data_ptr()),
            static_cast<float*>(grad_dt_bias.data_ptr()),
            static_cast<const __half*>(grad_query.data_ptr()),
            static_cast<const __half*>(grad_key.data_ptr()),
            static_cast<const __half*>(grad_value.data_ptr()),
            static_cast<const __half*>(grad_beta.data_ptr()),
            static_cast<const float*>(grad_g.data_ptr()),
            static_cast<const __half*>(query.data_ptr()),
            static_cast<const __half*>(key.data_ptr()),
            static_cast<const float*>(q_rstd.data_ptr()),
            static_cast<const float*>(k_rstd.data_ptr()),
            static_cast<const __half*>(b_gate.data_ptr()),
            static_cast<const __half*>(a_gate.data_ptr()),
            static_cast<const float*>(A_log.data_ptr()),
            static_cast<const float*>(dt_bias.data_ptr()),
            batch_size,
            seq_length,
            grad_mixed_qkv.stride(0),
            grad_mixed_qkv.stride(2),
            grad_b.stride(0),
            grad_b.stride(1),
            grad_a.stride(0),
            grad_a.stride(1),
            b_gate.stride(0),
            b_gate.stride(1),
            a_gate.stride(0),
            a_gate.stride(1),
            stream);
    } else if (grad_mixed_qkv.dtype().code == kDLBfloat && grad_mixed_qkv.dtype().bits == 16) {
        launch_post_conv_pack_preprocess_backward_qwen35<__nv_bfloat16>(
            static_cast<__nv_bfloat16*>(grad_mixed_qkv.data_ptr()),
            static_cast<__nv_bfloat16*>(grad_b.data_ptr()),
            static_cast<__nv_bfloat16*>(grad_a.data_ptr()),
            static_cast<float*>(grad_A_log.data_ptr()),
            static_cast<float*>(grad_dt_bias.data_ptr()),
            static_cast<const __nv_bfloat16*>(grad_query.data_ptr()),
            static_cast<const __nv_bfloat16*>(grad_key.data_ptr()),
            static_cast<const __nv_bfloat16*>(grad_value.data_ptr()),
            static_cast<const __nv_bfloat16*>(grad_beta.data_ptr()),
            static_cast<const float*>(grad_g.data_ptr()),
            static_cast<const __nv_bfloat16*>(query.data_ptr()),
            static_cast<const __nv_bfloat16*>(key.data_ptr()),
            static_cast<const float*>(q_rstd.data_ptr()),
            static_cast<const float*>(k_rstd.data_ptr()),
            static_cast<const __nv_bfloat16*>(b_gate.data_ptr()),
            static_cast<const __nv_bfloat16*>(a_gate.data_ptr()),
            static_cast<const float*>(A_log.data_ptr()),
            static_cast<const float*>(dt_bias.data_ptr()),
            batch_size,
            seq_length,
            grad_mixed_qkv.stride(0),
            grad_mixed_qkv.stride(2),
            grad_b.stride(0),
            grad_b.stride(1),
            grad_a.stride(0),
            grad_a.stride(1),
            b_gate.stride(0),
            b_gate.stride(1),
            a_gate.stride(0),
            a_gate.stride(1),
            stream);
    } else {
        throw std::runtime_error("gated_delta_post_conv_pack_preprocess_backward: unsupported dtype");
    }
    sync_check_cuda_error();
}






}  // namespace wallx_cuda_gated_delta_pack
