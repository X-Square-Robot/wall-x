#include "tensor_view.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../common/cuda_utils.h"

namespace wallx_cuda_qwen35_vla_qkv_postprocess {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kHeadsPerGroupedBlock = kThreads / kWarpSize;
constexpr int kMaxBackwardVecPerThread = 4;
constexpr int kBackwardPersistentBlockMultiplier = 8;

static void check_same_device(
    wallx_cuda::TensorView tensor,
    const char* name,
    int device_type,
    int device_id) {
    if (tensor.device().device_type != device_type || tensor.device().device_id != device_id) {
        throw std::runtime_error(std::string(name) + ": expected tensor on the same device");
    }
}

static void check_i64_1d(
    wallx_cuda::TensorView tensor,
    const char* name,
    int64_t expected_size,
    int device_type,
    int device_id) {
    if (tensor.ndim() != 1 || tensor.size(0) != expected_size) {
        throw std::runtime_error(
            std::string(name) + ": expected shape [" + std::to_string(expected_size) + "]");
    }
    if (!tensor.IsContiguous()) {
        throw std::runtime_error(std::string(name) + ": expected contiguous tensor");
    }
    if (!(tensor.dtype().code == kDLInt && tensor.dtype().bits == 64)) {
        throw std::runtime_error(std::string(name) + ": expected int64 tensor");
    }
    check_same_device(tensor, name, device_type, device_id);
}

static void check_weight(
    wallx_cuda::TensorView tensor,
    const char* name,
    int64_t head_dim,
    int device_type,
    int device_id) {
    if (tensor.ndim() != 1 || tensor.size(0) != head_dim) {
        throw std::runtime_error(
            std::string(name) + ": expected shape [" + std::to_string(head_dim) + "]");
    }
    if (!tensor.IsContiguous()) {
        throw std::runtime_error(std::string(name) + ": expected contiguous tensor");
    }
    const bool supported_dtype =
        (tensor.dtype().code == kDLFloat && tensor.dtype().bits == 32) ||
        (tensor.dtype().code == kDLFloat && tensor.dtype().bits == 16) ||
        (tensor.dtype().code == kDLBfloat && tensor.dtype().bits == 16);
    if (!supported_dtype) {
        throw std::runtime_error(std::string(name) + ": expected float32, float16, or bfloat16 tensor");
    }
    check_same_device(tensor, name, device_type, device_id);
}

static bool same_dtype(wallx_cuda::TensorView lhs, wallx_cuda::TensorView rhs) {
    return lhs.dtype().code == rhs.dtype().code && lhs.dtype().bits == rhs.dtype().bits;
}

template <typename T>
__device__ __forceinline__ float to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();
    const int num_warps = (blockDim.x + 31) >> 5;
    val = threadIdx.x < num_warps ? shared[lane] : 0.0f;
    if (warp == 0) {
        val = warp_reduce_sum(val);
    }
    // Trailing barrier: `shared` is reused across calls, so without it a fast warp
    // entering the next call can overwrite shared[warp] while a slow warp is still
    // reading shared[lane] above (WAR race on back-to-back reductions).
    __syncthreads();
    return val;
}

__device__ __forceinline__ void block_reduce_sum_pair(float& lhs, float& rhs) {
    static __shared__ float shared_lhs[32];
    static __shared__ float shared_rhs[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    lhs = warp_reduce_sum(lhs);
    rhs = warp_reduce_sum(rhs);
    if (lane == 0) {
        shared_lhs[warp] = lhs;
        shared_rhs[warp] = rhs;
    }
    __syncthreads();
    const int num_warps = (blockDim.x + 31) >> 5;
    lhs = threadIdx.x < num_warps ? shared_lhs[lane] : 0.0f;
    rhs = threadIdx.x < num_warps ? shared_rhs[lane] : 0.0f;
    if (warp == 0) {
        lhs = warp_reduce_sum(lhs);
        rhs = warp_reduce_sum(rhs);
    }
}

template <typename T>
__device__ __forceinline__ void vec_to_floats(typename VecTraits<T>::Type vec, float* values) {
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
    }
}

template <typename T>
__device__ __forceinline__ typename VecTraits<T>::Type floats_to_vec(const float* values) {
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
    }
    return vec;
}

template <typename W, int VEC_SIZE>
__device__ __forceinline__ void load_weight_floats(
    const W* __restrict__ weight,
    int64_t offset,
    float* values) {
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        values[i] = to_float<W>(weight[offset + i]);
    }
}

__device__ __forceinline__ bool is_action_token(
    int64_t seq,
    int64_t action_start,
    int64_t action_end) {
    return seq >= action_start && seq < action_end;
}

__device__ __forceinline__ int64_t q_offset(int64_t head, int64_t dim, int64_t head_dim) {
    return head * 2 * head_dim + dim;
}

__device__ __forceinline__ int64_t gate_offset(int64_t head, int64_t dim, int64_t head_dim) {
    return head * 2 * head_dim + head_dim + dim;
}

__device__ __forceinline__ int64_t k_offset(
    int64_t num_heads,
    int64_t head,
    int64_t dim,
    int64_t head_dim) {
    return num_heads * 2 * head_dim + head * head_dim + dim;
}

__device__ __forceinline__ int64_t v_offset(
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head,
    int64_t dim,
    int64_t head_dim) {
    return (num_heads * 2 + num_key_value_heads) * head_dim + head * head_dim + dim;
}

template <typename T, typename W>
__global__ void qkvNormReplaceForwardKernel(
    T* __restrict__ query,
    T* __restrict__ key,
    T* __restrict__ value,
    T* __restrict__ gate,
    const T* __restrict__ text_qkv,
    const T* __restrict__ action_qkv,
    const int64_t* __restrict__ action_starts,
    const int64_t* __restrict__ action_ends,
    const int64_t* __restrict__ action_offsets,
    const W* __restrict__ q_weight,
    const W* __restrict__ k_weight,
    const W* __restrict__ q_action_weight,
    const W* __restrict__ k_action_weight,
    int64_t batch_size,
    int64_t seq_length,
    int64_t action_length,
    int64_t qkv_dim,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    float epsilon) {
    const int64_t row = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int field = blockIdx.z;  // 0: q+gate, 1: k+value
    if (row >= batch_size * seq_length) {
        return;
    }
    if ((field == 0 && head >= num_heads) || (field == 1 && head >= num_key_value_heads)) {
        return;
    }

    const int64_t batch = row / seq_length;
    const int64_t seq = row - batch * seq_length;
    const int64_t action_start = action_starts[batch];
    const int64_t action_end = action_ends[batch];
    const bool use_action = is_action_token(seq, action_start, action_end);
    const int64_t source_row = use_action
        ? (action_offsets[batch] + seq - action_start)
        : row;
    const T* source = use_action
        ? action_qkv + source_row * qkv_dim
        : text_qkv + source_row * qkv_dim;

    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;
    const int64_t vec_hidden = head_dim / VEC_SIZE;
    float sum_sq = 0.0f;
    const int64_t base_offset = (field == 0)
        ? q_offset(head, 0, head_dim)
        : k_offset(num_heads, head, 0, head_dim);

    for (int64_t idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x) {
        const Vec x_vec = VecTraits<T>::load(source + base_offset + idx * VEC_SIZE);
        float x_vals[VEC_SIZE];
        vec_to_floats<T>(x_vec, x_vals);
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            sum_sq += x_vals[i] * x_vals[i];
        }
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sum_sq / static_cast<float>(head_dim) + epsilon);
    }
    __syncthreads();

    const W* weight = nullptr;
    if (field == 0) {
        weight = use_action ? q_action_weight : q_weight;
        const int64_t out_base = (row * num_heads + head) * head_dim;
        const int64_t gate_base = row * num_heads * head_dim + head * head_dim;
        for (int64_t idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x) {
            const int64_t dim = idx * VEC_SIZE;
            const Vec x_vec = VecTraits<T>::load(source + q_offset(head, dim, head_dim));
            float x_vals[VEC_SIZE], w_vals[VEC_SIZE], out_vals[VEC_SIZE];
            vec_to_floats<T>(x_vec, x_vals);
            load_weight_floats<W, VEC_SIZE>(weight, dim, w_vals);
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                out_vals[i] = x_vals[i] * (w_vals[i] + 1.0f) * inv_rms;
            }
            VecTraits<T>::store(query + out_base + dim, floats_to_vec<T>(out_vals));
            const Vec gate_vec = VecTraits<T>::load(source + gate_offset(head, dim, head_dim));
            VecTraits<T>::store(gate + gate_base + dim, gate_vec);
        }
    } else {
        weight = use_action ? k_action_weight : k_weight;
        const int64_t out_base = (row * num_key_value_heads + head) * head_dim;
        for (int64_t idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x) {
            const int64_t dim = idx * VEC_SIZE;
            const Vec x_vec = VecTraits<T>::load(source + k_offset(num_heads, head, dim, head_dim));
            float x_vals[VEC_SIZE], w_vals[VEC_SIZE], out_vals[VEC_SIZE];
            vec_to_floats<T>(x_vec, x_vals);
            load_weight_floats<W, VEC_SIZE>(weight, dim, w_vals);
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                out_vals[i] = x_vals[i] * (w_vals[i] + 1.0f) * inv_rms;
            }
            VecTraits<T>::store(key + out_base + dim, floats_to_vec<T>(out_vals));
            const Vec value_vec = VecTraits<T>::load(
                source + v_offset(num_heads, num_key_value_heads, head, dim, head_dim));
            VecTraits<T>::store(value + out_base + dim, value_vec);
        }
    }
}

template <typename T, typename W>
__global__ void qkvNormReplaceForwardGroupedKernel(
    T* __restrict__ query,
    T* __restrict__ key,
    T* __restrict__ value,
    T* __restrict__ gate,
    const T* __restrict__ text_qkv,
    const T* __restrict__ action_qkv,
    const int64_t* __restrict__ action_starts,
    const int64_t* __restrict__ action_ends,
    const int64_t* __restrict__ action_offsets,
    const W* __restrict__ q_weight,
    const W* __restrict__ k_weight,
    const W* __restrict__ q_action_weight,
    const W* __restrict__ k_action_weight,
    int64_t batch_size,
    int64_t seq_length,
    int64_t action_length,
    int64_t qkv_dim,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    float epsilon) {
    const int64_t row = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t logical_head = blockIdx.y * kHeadsPerGroupedBlock + warp;
    const int64_t total_heads = num_heads + num_key_value_heads;
    if (row >= batch_size * seq_length || logical_head >= total_heads) {
        return;
    }

    const bool is_query = logical_head < num_heads;
    const int64_t head = is_query ? logical_head : logical_head - num_heads;
    const int64_t batch = row / seq_length;
    const int64_t seq = row - batch * seq_length;
    const int64_t action_start = action_starts[batch];
    const int64_t action_end = action_ends[batch];
    const bool use_action = is_action_token(seq, action_start, action_end);
    const int64_t source_row = use_action
        ? (action_offsets[batch] + seq - action_start)
        : row;
    const T* source = use_action
        ? action_qkv + source_row * qkv_dim
        : text_qkv + source_row * qkv_dim;

    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;
    const int64_t vec_hidden = head_dim / VEC_SIZE;
    const int64_t base_offset = is_query
        ? q_offset(head, 0, head_dim)
        : k_offset(num_heads, head, 0, head_dim);

    float sum_sq = 0.0f;
    for (int64_t idx = lane; idx < vec_hidden; idx += kWarpSize) {
        const Vec x_vec = VecTraits<T>::load(source + base_offset + idx * VEC_SIZE);
        float x_vals[VEC_SIZE];
        vec_to_floats<T>(x_vec, x_vals);
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            sum_sq += x_vals[i] * x_vals[i];
        }
    }
    sum_sq = warp_reduce_sum(sum_sq);
    const float inv_rms = rsqrtf(
        __shfl_sync(0xffffffff, sum_sq, 0) / static_cast<float>(head_dim) + epsilon);

    const W* weight = is_query
        ? (use_action ? q_action_weight : q_weight)
        : (use_action ? k_action_weight : k_weight);

    if (is_query) {
        const int64_t out_base = (row * num_heads + head) * head_dim;
        const int64_t gate_base = row * num_heads * head_dim + head * head_dim;
        for (int64_t idx = lane; idx < vec_hidden; idx += kWarpSize) {
            const int64_t dim = idx * VEC_SIZE;
            const Vec x_vec = VecTraits<T>::load(source + q_offset(head, dim, head_dim));
            float x_vals[VEC_SIZE], w_vals[VEC_SIZE], out_vals[VEC_SIZE];
            vec_to_floats<T>(x_vec, x_vals);
            load_weight_floats<W, VEC_SIZE>(weight, dim, w_vals);
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                out_vals[i] = x_vals[i] * (w_vals[i] + 1.0f) * inv_rms;
            }
            VecTraits<T>::store(query + out_base + dim, floats_to_vec<T>(out_vals));
            const Vec gate_vec = VecTraits<T>::load(source + gate_offset(head, dim, head_dim));
            VecTraits<T>::store(gate + gate_base + dim, gate_vec);
        }
    } else {
        const int64_t out_base = (row * num_key_value_heads + head) * head_dim;
        for (int64_t idx = lane; idx < vec_hidden; idx += kWarpSize) {
            const int64_t dim = idx * VEC_SIZE;
            const Vec x_vec = VecTraits<T>::load(source + k_offset(num_heads, head, dim, head_dim));
            float x_vals[VEC_SIZE], w_vals[VEC_SIZE], out_vals[VEC_SIZE];
            vec_to_floats<T>(x_vec, x_vals);
            load_weight_floats<W, VEC_SIZE>(weight, dim, w_vals);
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                out_vals[i] = x_vals[i] * (w_vals[i] + 1.0f) * inv_rms;
            }
            VecTraits<T>::store(key + out_base + dim, floats_to_vec<T>(out_vals));
            const Vec value_vec = VecTraits<T>::load(
                source + v_offset(num_heads, num_key_value_heads, head, dim, head_dim));
            VecTraits<T>::store(value + out_base + dim, value_vec);
        }
    }
}

template <typename T, typename W>
__global__ void qkvNormReplaceBackwardKernel(
    T* __restrict__ grad_text_qkv,
    T* __restrict__ grad_action_qkv,
    float* __restrict__ grad_q_weight,
    float* __restrict__ grad_k_weight,
    float* __restrict__ grad_q_action_weight,
    float* __restrict__ grad_k_action_weight,
    const T* __restrict__ grad_query,
    const T* __restrict__ grad_key,
    const T* __restrict__ grad_value,
    const T* __restrict__ grad_gate,
    const T* __restrict__ text_qkv,
    const T* __restrict__ action_qkv,
    const int64_t* __restrict__ action_starts,
    const int64_t* __restrict__ action_ends,
    const int64_t* __restrict__ action_offsets,
    const W* __restrict__ q_weight,
    const W* __restrict__ k_weight,
    const W* __restrict__ q_action_weight,
    const W* __restrict__ k_action_weight,
    int64_t batch_size,
    int64_t seq_length,
    int64_t action_length,
    int64_t qkv_dim,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    float epsilon) {
    const int64_t row = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int field = blockIdx.z;  // 0: q+gate, 1: k+value
    if (row >= batch_size * seq_length) {
        return;
    }
    if ((field == 0 && head >= num_heads) || (field == 1 && head >= num_key_value_heads)) {
        return;
    }

    const int64_t batch = row / seq_length;
    const int64_t seq = row - batch * seq_length;
    const int64_t action_start = action_starts[batch];
    const int64_t action_end = action_ends[batch];
    const bool use_action = is_action_token(seq, action_start, action_end);
    const int64_t source_row = use_action
        ? (action_offsets[batch] + seq - action_start)
        : row;
    const T* source = use_action
        ? action_qkv + source_row * qkv_dim
        : text_qkv + source_row * qkv_dim;
    T* grad_source = use_action
        ? grad_action_qkv + source_row * qkv_dim
        : grad_text_qkv + source_row * qkv_dim;

    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;
    const int64_t vec_hidden = head_dim / VEC_SIZE;
    float sum_sq = 0.0f;
    float dot = 0.0f;

    const int64_t base_offset = (field == 0)
        ? q_offset(head, 0, head_dim)
        : k_offset(num_heads, head, 0, head_dim);
    const int64_t grad_base = (field == 0)
        ? (row * num_heads + head) * head_dim
        : (row * num_key_value_heads + head) * head_dim;
    const W* weight = (field == 0)
        ? (use_action ? q_action_weight : q_weight)
        : (use_action ? k_action_weight : k_weight);

    for (int64_t idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x) {
        const int64_t dim = idx * VEC_SIZE;
        const Vec x_vec = VecTraits<T>::load(source + base_offset + dim);
        const Vec gy_vec = VecTraits<T>::load(
            (field == 0 ? grad_query : grad_key) + grad_base + dim);
        float x_vals[VEC_SIZE], gy_vals[VEC_SIZE], w_vals[VEC_SIZE];
        vec_to_floats<T>(x_vec, x_vals);
        vec_to_floats<T>(gy_vec, gy_vals);
        load_weight_floats<W, VEC_SIZE>(weight, dim, w_vals);
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            const float gw = gy_vals[i] * (w_vals[i] + 1.0f);
            sum_sq += x_vals[i] * x_vals[i];
            dot += gw * x_vals[i];
        }
    }
    sum_sq = block_reduce_sum(sum_sq);
    dot = block_reduce_sum(dot);
    __shared__ float inv_rms;
    __shared__ float norm_scale;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sum_sq / static_cast<float>(head_dim) + epsilon);
        norm_scale = (dot / static_cast<float>(head_dim)) * inv_rms * inv_rms;
    }
    __syncthreads();

    if (field == 0) {
        float* grad_weight = use_action ? grad_q_action_weight : grad_q_weight;
        const int64_t gate_grad_base = row * num_heads * head_dim + head * head_dim;
        for (int64_t idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x) {
            const int64_t dim = idx * VEC_SIZE;
            const Vec x_vec = VecTraits<T>::load(source + q_offset(head, dim, head_dim));
            const Vec gy_vec = VecTraits<T>::load(grad_query + grad_base + dim);
            float x_vals[VEC_SIZE], gy_vals[VEC_SIZE], w_vals[VEC_SIZE], dx_vals[VEC_SIZE];
            vec_to_floats<T>(x_vec, x_vals);
            vec_to_floats<T>(gy_vec, gy_vals);
            load_weight_floats<W, VEC_SIZE>(weight, dim, w_vals);
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                dx_vals[i] = inv_rms * (gy_vals[i] * (w_vals[i] + 1.0f) - x_vals[i] * norm_scale);
                atomicAdd(grad_weight + dim + i, gy_vals[i] * x_vals[i] * inv_rms);
            }
            VecTraits<T>::store(grad_source + q_offset(head, dim, head_dim), floats_to_vec<T>(dx_vals));
            const Vec gate_grad_vec = VecTraits<T>::load(grad_gate + gate_grad_base + dim);
            VecTraits<T>::store(grad_source + gate_offset(head, dim, head_dim), gate_grad_vec);
        }
    } else {
        float* grad_weight = use_action ? grad_k_action_weight : grad_k_weight;
        for (int64_t idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x) {
            const int64_t dim = idx * VEC_SIZE;
            const Vec x_vec = VecTraits<T>::load(source + k_offset(num_heads, head, dim, head_dim));
            const Vec gy_vec = VecTraits<T>::load(grad_key + grad_base + dim);
            float x_vals[VEC_SIZE], gy_vals[VEC_SIZE], w_vals[VEC_SIZE], dx_vals[VEC_SIZE];
            vec_to_floats<T>(x_vec, x_vals);
            vec_to_floats<T>(gy_vec, gy_vals);
            load_weight_floats<W, VEC_SIZE>(weight, dim, w_vals);
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                dx_vals[i] = inv_rms * (gy_vals[i] * (w_vals[i] + 1.0f) - x_vals[i] * norm_scale);
                atomicAdd(grad_weight + dim + i, gy_vals[i] * x_vals[i] * inv_rms);
            }
            VecTraits<T>::store(grad_source + k_offset(num_heads, head, dim, head_dim), floats_to_vec<T>(dx_vals));
            const Vec value_grad_vec = VecTraits<T>::load(grad_value + grad_base + dim);
            VecTraits<T>::store(
                grad_source + v_offset(num_heads, num_key_value_heads, head, dim, head_dim),
                value_grad_vec);
        }
    }
}

static int get_sm_count() {
    static int sm_count = -1;
    if (sm_count < 0) {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    }
    return sm_count;
}

template <typename T, typename W, int MAX_VEC_PER_THREAD>
__global__ void qkvNormReplaceBackwardPersistentKernel(
    T* __restrict__ grad_text_qkv,
    T* __restrict__ grad_action_qkv,
    float* __restrict__ partial_dw,
    const T* __restrict__ grad_query,
    const T* __restrict__ grad_key,
    const T* __restrict__ grad_value,
    const T* __restrict__ grad_gate,
    const T* __restrict__ text_qkv,
    const T* __restrict__ action_qkv,
    const int64_t* __restrict__ action_starts,
    const int64_t* __restrict__ action_ends,
    const int64_t* __restrict__ action_offsets,
    const W* __restrict__ q_weight,
    const W* __restrict__ k_weight,
    const W* __restrict__ q_action_weight,
    const W* __restrict__ k_action_weight,
    int64_t batch_size,
    int64_t seq_length,
    int64_t action_length,
    int64_t qkv_dim,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    float epsilon) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;

    const bool is_query = blockIdx.y == 0;
    const int64_t heads = is_query ? num_heads : num_key_value_heads;
    const int64_t total_pairs = batch_size * seq_length * heads;
    const int64_t num_blocks = gridDim.x;
    const int64_t vec_hidden = head_dim / VEC_SIZE;
    const int64_t tid = threadIdx.x;
    const float inv_hidden = 1.0f / static_cast<float>(head_dim);

    __shared__ float shared_inv_rms;
    __shared__ float shared_norm_scale;

    float local_dw_text[MAX_VEC_PER_THREAD * VEC_SIZE];
    float local_dw_action[MAX_VEC_PER_THREAD * VEC_SIZE];
    #pragma unroll
    for (int i = 0; i < MAX_VEC_PER_THREAD * VEC_SIZE; ++i) {
        local_dw_text[i] = 0.0f;
        local_dw_action[i] = 0.0f;
    }

    for (int64_t pair = blockIdx.x; pair < total_pairs; pair += num_blocks) {
        const int64_t row = pair / heads;
        const int64_t head = pair - row * heads;
        const int64_t batch = row / seq_length;
        const int64_t seq = row - batch * seq_length;
        const int64_t action_start = action_starts[batch];
        const int64_t action_end = action_ends[batch];
        const bool use_action = is_action_token(seq, action_start, action_end);
        const int64_t source_row = use_action
            ? (action_offsets[batch] + seq - action_start)
            : row;
        const T* source = use_action
            ? action_qkv + source_row * qkv_dim
            : text_qkv + source_row * qkv_dim;
        T* grad_source = use_action
            ? grad_action_qkv + source_row * qkv_dim
            : grad_text_qkv + source_row * qkv_dim;

        const int64_t base_offset = is_query
            ? q_offset(head, 0, head_dim)
            : k_offset(num_heads, head, 0, head_dim);
        const int64_t grad_base = is_query
            ? (row * num_heads + head) * head_dim
            : (row * num_key_value_heads + head) * head_dim;
        const T* grad_norm = is_query ? grad_query : grad_key;
        const W* weight = is_query
            ? (use_action ? q_action_weight : q_weight)
            : (use_action ? k_action_weight : k_weight);

        float sum_sq = 0.0f;
        float dot = 0.0f;
        for (int64_t idx = tid; idx < vec_hidden; idx += blockDim.x) {
            const int64_t dim = idx * VEC_SIZE;
            const Vec x_vec = VecTraits<T>::load(source + base_offset + dim);
            const Vec gy_vec = VecTraits<T>::load(grad_norm + grad_base + dim);
            float x_vals[VEC_SIZE], gy_vals[VEC_SIZE], w_vals[VEC_SIZE];
            vec_to_floats<T>(x_vec, x_vals);
            vec_to_floats<T>(gy_vec, gy_vals);
            load_weight_floats<W, VEC_SIZE>(weight, dim, w_vals);
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                const float gw = gy_vals[i] * (w_vals[i] + 1.0f);
                sum_sq += x_vals[i] * x_vals[i];
                dot += gw * x_vals[i];
            }
        }
        block_reduce_sum_pair(sum_sq, dot);
        if (tid == 0) {
            const float inv_rms = rsqrtf(sum_sq * inv_hidden + epsilon);
            shared_inv_rms = inv_rms;
            shared_norm_scale = dot * inv_hidden * inv_rms * inv_rms;
        }
        __syncthreads();

        const float inv_rms = shared_inv_rms;
        const float norm_scale = shared_norm_scale;
        int local_idx = 0;
        if (is_query) {
            const int64_t gate_grad_base = row * num_heads * head_dim + head * head_dim;
            for (int64_t idx = tid; idx < vec_hidden; idx += blockDim.x, ++local_idx) {
                const int64_t dim = idx * VEC_SIZE;
                const Vec x_vec = VecTraits<T>::load(source + q_offset(head, dim, head_dim));
                const Vec gy_vec = VecTraits<T>::load(grad_query + grad_base + dim);
                float x_vals[VEC_SIZE], gy_vals[VEC_SIZE], w_vals[VEC_SIZE], dx_vals[VEC_SIZE];
                vec_to_floats<T>(x_vec, x_vals);
                vec_to_floats<T>(gy_vec, gy_vals);
                load_weight_floats<W, VEC_SIZE>(weight, dim, w_vals);
                #pragma unroll
                for (int i = 0; i < VEC_SIZE; ++i) {
                    dx_vals[i] =
                        inv_rms * (gy_vals[i] * (w_vals[i] + 1.0f) - x_vals[i] * norm_scale);
                    const float dw = gy_vals[i] * x_vals[i] * inv_rms;
                    if (use_action) {
                        local_dw_action[local_idx * VEC_SIZE + i] += dw;
                    } else {
                        local_dw_text[local_idx * VEC_SIZE + i] += dw;
                    }
                }
                VecTraits<T>::store(
                    grad_source + q_offset(head, dim, head_dim),
                    floats_to_vec<T>(dx_vals));
                const Vec gate_grad_vec = VecTraits<T>::load(grad_gate + gate_grad_base + dim);
                VecTraits<T>::store(grad_source + gate_offset(head, dim, head_dim), gate_grad_vec);
            }
        } else {
            for (int64_t idx = tid; idx < vec_hidden; idx += blockDim.x, ++local_idx) {
                const int64_t dim = idx * VEC_SIZE;
                const Vec x_vec = VecTraits<T>::load(source + k_offset(num_heads, head, dim, head_dim));
                const Vec gy_vec = VecTraits<T>::load(grad_key + grad_base + dim);
                float x_vals[VEC_SIZE], gy_vals[VEC_SIZE], w_vals[VEC_SIZE], dx_vals[VEC_SIZE];
                vec_to_floats<T>(x_vec, x_vals);
                vec_to_floats<T>(gy_vec, gy_vals);
                load_weight_floats<W, VEC_SIZE>(weight, dim, w_vals);
                #pragma unroll
                for (int i = 0; i < VEC_SIZE; ++i) {
                    dx_vals[i] =
                        inv_rms * (gy_vals[i] * (w_vals[i] + 1.0f) - x_vals[i] * norm_scale);
                    const float dw = gy_vals[i] * x_vals[i] * inv_rms;
                    if (use_action) {
                        local_dw_action[local_idx * VEC_SIZE + i] += dw;
                    } else {
                        local_dw_text[local_idx * VEC_SIZE + i] += dw;
                    }
                }
                VecTraits<T>::store(
                    grad_source + k_offset(num_heads, head, dim, head_dim),
                    floats_to_vec<T>(dx_vals));
                const Vec value_grad_vec = VecTraits<T>::load(grad_value + grad_base + dim);
                VecTraits<T>::store(
                    grad_source + v_offset(num_heads, num_key_value_heads, head, dim, head_dim),
                    value_grad_vec);
            }
        }
    }

    const int text_partial_idx = is_query ? 0 : 1;
    const int action_partial_idx = is_query ? 2 : 3;
    int local_idx = 0;
    for (int64_t idx = tid; idx < vec_hidden; idx += blockDim.x, ++local_idx) {
        const int64_t dim = idx * VEC_SIZE;
        float* text_partial =
            partial_dw + (text_partial_idx * num_blocks + blockIdx.x) * head_dim + dim;
        float* action_partial =
            partial_dw + (action_partial_idx * num_blocks + blockIdx.x) * head_dim + dim;
        if constexpr (VEC_SIZE == 4) {
            float4 text_vec;
            text_vec.x = local_dw_text[local_idx * VEC_SIZE + 0];
            text_vec.y = local_dw_text[local_idx * VEC_SIZE + 1];
            text_vec.z = local_dw_text[local_idx * VEC_SIZE + 2];
            text_vec.w = local_dw_text[local_idx * VEC_SIZE + 3];
            *reinterpret_cast<float4*>(text_partial) = text_vec;
            float4 action_vec;
            action_vec.x = local_dw_action[local_idx * VEC_SIZE + 0];
            action_vec.y = local_dw_action[local_idx * VEC_SIZE + 1];
            action_vec.z = local_dw_action[local_idx * VEC_SIZE + 2];
            action_vec.w = local_dw_action[local_idx * VEC_SIZE + 3];
            *reinterpret_cast<float4*>(action_partial) = action_vec;
        } else {
            float2 text_vec;
            text_vec.x = local_dw_text[local_idx * VEC_SIZE + 0];
            text_vec.y = local_dw_text[local_idx * VEC_SIZE + 1];
            *reinterpret_cast<float2*>(text_partial) = text_vec;
            float2 action_vec;
            action_vec.x = local_dw_action[local_idx * VEC_SIZE + 0];
            action_vec.y = local_dw_action[local_idx * VEC_SIZE + 1];
            *reinterpret_cast<float2*>(action_partial) = action_vec;
        }
    }
}

__global__ void qkvNormReplaceReduceDwKernel(
    float* __restrict__ grad_q_weight,
    float* __restrict__ grad_k_weight,
    float* __restrict__ grad_q_action_weight,
    float* __restrict__ grad_k_action_weight,
    const float* __restrict__ partial_dw,
    int64_t num_blocks,
    int64_t head_dim) {
    const int64_t dim = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim >= head_dim) {
        return;
    }
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int kind = 0; kind < 4; ++kind) {
        const float* partial = partial_dw + kind * num_blocks * head_dim + dim;
        float sum = 0.0f;
        for (int64_t block = 0; block < num_blocks; ++block) {
            sum += partial[block * head_dim];
        }
        sums[kind] = sum;
    }
    grad_q_weight[dim] = sums[0];
    grad_k_weight[dim] = sums[1];
    grad_q_action_weight[dim] = sums[2];
    grad_k_action_weight[dim] = sums[3];
}

__global__ void qkvNormReplaceReduceDwD256Kernel(
    float* __restrict__ grad_q_weight,
    float* __restrict__ grad_k_weight,
    float* __restrict__ grad_q_action_weight,
    float* __restrict__ grad_k_action_weight,
    const float* __restrict__ partial_dw,
    int64_t num_blocks) {
    constexpr int D = 256;
    constexpr int ColsPerBlock = 32;
    constexpr int RowWarps = 8;
    const int lane = threadIdx.x;
    const int row_warp = threadIdx.y;
    const int col = static_cast<int>(blockIdx.x) * ColsPerBlock + lane;
    const int kind = static_cast<int>(blockIdx.y);

    float sum = 0.0f;
    const float* partial = partial_dw + kind * num_blocks * D + col;
    for (int64_t block = row_warp; block < num_blocks; block += RowWarps) {
        sum += partial[block * D];
    }

    __shared__ float shared[RowWarps][ColsPerBlock];
    shared[row_warp][lane] = sum;
    __syncthreads();

    if (row_warp == 0) {
        float total = shared[0][lane];
        #pragma unroll
        for (int i = 1; i < RowWarps; ++i) {
            total += shared[i][lane];
        }
        if (kind == 0) {
            grad_q_weight[col] = total;
        } else if (kind == 1) {
            grad_k_weight[col] = total;
        } else if (kind == 2) {
            grad_q_action_weight[col] = total;
        } else {
            grad_k_action_weight[col] = total;
        }
    }
}

template <typename T, typename W>
void launch_forward(
    wallx_cuda::TensorView query,
    wallx_cuda::TensorView key,
    wallx_cuda::TensorView value,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView text_qkv,
    wallx_cuda::TensorView action_qkv,
    wallx_cuda::TensorView action_starts,
    wallx_cuda::TensorView action_ends,
    wallx_cuda::TensorView action_offsets,
    wallx_cuda::TensorView q_weight,
    wallx_cuda::TensorView k_weight,
    wallx_cuda::TensorView q_action_weight,
    wallx_cuda::TensorView k_action_weight,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    float epsilon,
    cudaStream_t stream) {
    const int64_t batch_size = text_qkv.size(0);
    const int64_t seq_length = text_qkv.size(1);
    const int64_t qkv_dim = text_qkv.size(2);
    const int64_t action_length = action_qkv.size(1);
    // Every kernel here strides in VEC_SIZE units (vec_hidden = head_dim / VEC_SIZE
    // truncates): a non-divisible head_dim would leave the tail dims unwritten and
    // compute inv_rms from a partial sum. Reject it explicitly.
    ASSERT_CHECK(head_dim % VecTraits<T>::vec_size == 0);
    if (head_dim % VecTraits<T>::vec_size == 0 && head_dim >= 128) {
        const int64_t total_heads = num_heads + num_key_value_heads;
        const dim3 grid(
            batch_size * seq_length,
            (total_heads + kHeadsPerGroupedBlock - 1) / kHeadsPerGroupedBlock);
        qkvNormReplaceForwardGroupedKernel<T, W><<<grid, kThreads, 0, stream>>>(
            static_cast<T*>(query.data_ptr()),
            static_cast<T*>(key.data_ptr()),
            static_cast<T*>(value.data_ptr()),
            static_cast<T*>(gate.data_ptr()),
            static_cast<const T*>(text_qkv.data_ptr()),
            static_cast<const T*>(action_qkv.data_ptr()),
            static_cast<const int64_t*>(action_starts.data_ptr()),
            static_cast<const int64_t*>(action_ends.data_ptr()),
            static_cast<const int64_t*>(action_offsets.data_ptr()),
            static_cast<const W*>(q_weight.data_ptr()),
            static_cast<const W*>(k_weight.data_ptr()),
            static_cast<const W*>(q_action_weight.data_ptr()),
            static_cast<const W*>(k_action_weight.data_ptr()),
            batch_size,
            seq_length,
            action_length,
            qkv_dim,
            num_heads,
            num_key_value_heads,
            head_dim,
            epsilon);
    } else {
        const int64_t max_heads = num_heads > num_key_value_heads ? num_heads : num_key_value_heads;
        const dim3 grid(batch_size * seq_length, max_heads, 2);
        qkvNormReplaceForwardKernel<T, W><<<grid, kThreads, 0, stream>>>(
            static_cast<T*>(query.data_ptr()),
            static_cast<T*>(key.data_ptr()),
            static_cast<T*>(value.data_ptr()),
            static_cast<T*>(gate.data_ptr()),
            static_cast<const T*>(text_qkv.data_ptr()),
            static_cast<const T*>(action_qkv.data_ptr()),
            static_cast<const int64_t*>(action_starts.data_ptr()),
            static_cast<const int64_t*>(action_ends.data_ptr()),
            static_cast<const int64_t*>(action_offsets.data_ptr()),
            static_cast<const W*>(q_weight.data_ptr()),
            static_cast<const W*>(k_weight.data_ptr()),
            static_cast<const W*>(q_action_weight.data_ptr()),
            static_cast<const W*>(k_action_weight.data_ptr()),
            batch_size,
            seq_length,
            action_length,
            qkv_dim,
            num_heads,
            num_key_value_heads,
            head_dim,
            epsilon);
    }
}

template <typename T, typename W>
void launch_backward(
    wallx_cuda::TensorView grad_text_qkv,
    wallx_cuda::TensorView grad_action_qkv,
    wallx_cuda::TensorView grad_q_weight,
    wallx_cuda::TensorView grad_k_weight,
    wallx_cuda::TensorView grad_q_action_weight,
    wallx_cuda::TensorView grad_k_action_weight,
    wallx_cuda::TensorView partial_dw,
    wallx_cuda::TensorView grad_query,
    wallx_cuda::TensorView grad_key,
    wallx_cuda::TensorView grad_value,
    wallx_cuda::TensorView grad_gate,
    wallx_cuda::TensorView text_qkv,
    wallx_cuda::TensorView action_qkv,
    wallx_cuda::TensorView action_starts,
    wallx_cuda::TensorView action_ends,
    wallx_cuda::TensorView action_offsets,
    wallx_cuda::TensorView q_weight,
    wallx_cuda::TensorView k_weight,
    wallx_cuda::TensorView q_action_weight,
    wallx_cuda::TensorView k_action_weight,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    float epsilon,
    cudaStream_t stream) {
    const int64_t batch_size = text_qkv.size(0);
    const int64_t seq_length = text_qkv.size(1);
    const int64_t qkv_dim = text_qkv.size(2);
    const int64_t action_length = action_qkv.size(1);
    // See launch_forward: vectorized kernels require head_dim % VEC_SIZE == 0.
    ASSERT_CHECK(head_dim % VecTraits<T>::vec_size == 0);
    const int64_t vec_hidden = head_dim / VecTraits<T>::vec_size;
    if (head_dim % VecTraits<T>::vec_size == 0 &&
        head_dim >= 128 &&
        vec_hidden <= kThreads * kMaxBackwardVecPerThread) {
        const int64_t max_pairs = batch_size * seq_length * num_heads;
        const int64_t num_blocks = std::min(
            static_cast<int64_t>(get_sm_count()) * kBackwardPersistentBlockMultiplier,
            max_pairs);
        ASSERT_CHECK(partial_dw.ndim() == 3);
        ASSERT_CHECK(partial_dw.size(0) == 4);
        ASSERT_CHECK(partial_dw.size(1) == num_blocks);
        ASSERT_CHECK(partial_dw.size(2) == head_dim);
        ASSERT_CHECK(partial_dw.IsContiguous());
        ASSERT_CHECK(partial_dw.dtype().code == kDLFloat && partial_dw.dtype().bits == 32);
        const dim3 grid(num_blocks, 2);
        qkvNormReplaceBackwardPersistentKernel<T, W, kMaxBackwardVecPerThread><<<grid, kThreads, 0, stream>>>(
            static_cast<T*>(grad_text_qkv.data_ptr()),
            static_cast<T*>(grad_action_qkv.data_ptr()),
            static_cast<float*>(partial_dw.data_ptr()),
            static_cast<const T*>(grad_query.data_ptr()),
            static_cast<const T*>(grad_key.data_ptr()),
            static_cast<const T*>(grad_value.data_ptr()),
            static_cast<const T*>(grad_gate.data_ptr()),
            static_cast<const T*>(text_qkv.data_ptr()),
            static_cast<const T*>(action_qkv.data_ptr()),
            static_cast<const int64_t*>(action_starts.data_ptr()),
            static_cast<const int64_t*>(action_ends.data_ptr()),
            static_cast<const int64_t*>(action_offsets.data_ptr()),
            static_cast<const W*>(q_weight.data_ptr()),
            static_cast<const W*>(k_weight.data_ptr()),
            static_cast<const W*>(q_action_weight.data_ptr()),
            static_cast<const W*>(k_action_weight.data_ptr()),
            batch_size,
            seq_length,
            action_length,
            qkv_dim,
            num_heads,
            num_key_value_heads,
            head_dim,
            epsilon);
        if (head_dim == 256) {
            const dim3 block(32, 8);
            const dim3 grid(8, 4);
            qkvNormReplaceReduceDwD256Kernel<<<grid, block, 0, stream>>>(
                static_cast<float*>(grad_q_weight.data_ptr()),
                static_cast<float*>(grad_k_weight.data_ptr()),
                static_cast<float*>(grad_q_action_weight.data_ptr()),
                static_cast<float*>(grad_k_action_weight.data_ptr()),
                static_cast<const float*>(partial_dw.data_ptr()),
                num_blocks);
        } else {
            qkvNormReplaceReduceDwKernel<<<(head_dim + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                static_cast<float*>(grad_q_weight.data_ptr()),
                static_cast<float*>(grad_k_weight.data_ptr()),
                static_cast<float*>(grad_q_action_weight.data_ptr()),
                static_cast<float*>(grad_k_action_weight.data_ptr()),
                static_cast<const float*>(partial_dw.data_ptr()),
                num_blocks,
                head_dim);
        }
    } else {
        const int64_t max_heads = num_heads > num_key_value_heads ? num_heads : num_key_value_heads;
        const dim3 grid(batch_size * seq_length, max_heads, 2);
        qkvNormReplaceBackwardKernel<T, W><<<grid, kThreads, 0, stream>>>(
            static_cast<T*>(grad_text_qkv.data_ptr()),
            static_cast<T*>(grad_action_qkv.data_ptr()),
            static_cast<float*>(grad_q_weight.data_ptr()),
            static_cast<float*>(grad_k_weight.data_ptr()),
            static_cast<float*>(grad_q_action_weight.data_ptr()),
            static_cast<float*>(grad_k_action_weight.data_ptr()),
            static_cast<const T*>(grad_query.data_ptr()),
            static_cast<const T*>(grad_key.data_ptr()),
            static_cast<const T*>(grad_value.data_ptr()),
            static_cast<const T*>(grad_gate.data_ptr()),
            static_cast<const T*>(text_qkv.data_ptr()),
            static_cast<const T*>(action_qkv.data_ptr()),
            static_cast<const int64_t*>(action_starts.data_ptr()),
            static_cast<const int64_t*>(action_ends.data_ptr()),
            static_cast<const int64_t*>(action_offsets.data_ptr()),
            static_cast<const W*>(q_weight.data_ptr()),
            static_cast<const W*>(k_weight.data_ptr()),
            static_cast<const W*>(q_action_weight.data_ptr()),
            static_cast<const W*>(k_action_weight.data_ptr()),
            batch_size,
            seq_length,
            action_length,
            qkv_dim,
            num_heads,
            num_key_value_heads,
            head_dim,
            epsilon);
    }
}

template <typename T>
void launch_forward_for_weight_dtype(
    wallx_cuda::TensorView query,
    wallx_cuda::TensorView key,
    wallx_cuda::TensorView value,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView text_qkv,
    wallx_cuda::TensorView action_qkv,
    wallx_cuda::TensorView action_starts,
    wallx_cuda::TensorView action_ends,
    wallx_cuda::TensorView action_offsets,
    wallx_cuda::TensorView q_weight,
    wallx_cuda::TensorView k_weight,
    wallx_cuda::TensorView q_action_weight,
    wallx_cuda::TensorView k_action_weight,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    float epsilon,
    cudaStream_t stream) {
    if (q_weight.dtype().code == kDLFloat && q_weight.dtype().bits == 32) {
        launch_forward<T, float>(
            query,
            key,
            value,
            gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            epsilon,
            stream);
    } else if (q_weight.dtype().code == kDLFloat && q_weight.dtype().bits == 16) {
        launch_forward<T, __half>(
            query,
            key,
            value,
            gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            epsilon,
            stream);
    } else if (q_weight.dtype().code == kDLBfloat && q_weight.dtype().bits == 16) {
        launch_forward<T, __nv_bfloat16>(
            query,
            key,
            value,
            gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            epsilon,
            stream);
    } else {
        throw std::runtime_error("qwen35_vla_qkv_norm_replace_forward: unsupported weight dtype");
    }
}

template <typename T>
void launch_backward_for_weight_dtype(
    wallx_cuda::TensorView grad_text_qkv,
    wallx_cuda::TensorView grad_action_qkv,
    wallx_cuda::TensorView grad_q_weight,
    wallx_cuda::TensorView grad_k_weight,
    wallx_cuda::TensorView grad_q_action_weight,
    wallx_cuda::TensorView grad_k_action_weight,
    wallx_cuda::TensorView partial_dw,
    wallx_cuda::TensorView grad_query,
    wallx_cuda::TensorView grad_key,
    wallx_cuda::TensorView grad_value,
    wallx_cuda::TensorView grad_gate,
    wallx_cuda::TensorView text_qkv,
    wallx_cuda::TensorView action_qkv,
    wallx_cuda::TensorView action_starts,
    wallx_cuda::TensorView action_ends,
    wallx_cuda::TensorView action_offsets,
    wallx_cuda::TensorView q_weight,
    wallx_cuda::TensorView k_weight,
    wallx_cuda::TensorView q_action_weight,
    wallx_cuda::TensorView k_action_weight,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    float epsilon,
    cudaStream_t stream) {
    if (q_weight.dtype().code == kDLFloat && q_weight.dtype().bits == 32) {
        launch_backward<T, float>(
            grad_text_qkv,
            grad_action_qkv,
            grad_q_weight,
            grad_k_weight,
            grad_q_action_weight,
            grad_k_action_weight,
            partial_dw,
            grad_query,
            grad_key,
            grad_value,
            grad_gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            epsilon,
            stream);
    } else if (q_weight.dtype().code == kDLFloat && q_weight.dtype().bits == 16) {
        launch_backward<T, __half>(
            grad_text_qkv,
            grad_action_qkv,
            grad_q_weight,
            grad_k_weight,
            grad_q_action_weight,
            grad_k_action_weight,
            partial_dw,
            grad_query,
            grad_key,
            grad_value,
            grad_gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            epsilon,
            stream);
    } else if (q_weight.dtype().code == kDLBfloat && q_weight.dtype().bits == 16) {
        launch_backward<T, __nv_bfloat16>(
            grad_text_qkv,
            grad_action_qkv,
            grad_q_weight,
            grad_k_weight,
            grad_q_action_weight,
            grad_k_action_weight,
            partial_dw,
            grad_query,
            grad_key,
            grad_value,
            grad_gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            epsilon,
            stream);
    } else {
        throw std::runtime_error("qwen35_vla_qkv_norm_replace_backward: unsupported weight dtype");
    }
}

void validate_common(
    wallx_cuda::TensorView text_qkv,
    wallx_cuda::TensorView action_qkv,
    wallx_cuda::TensorView action_starts,
    wallx_cuda::TensorView action_ends,
    wallx_cuda::TensorView action_offsets,
    wallx_cuda::TensorView q_weight,
    wallx_cuda::TensorView k_weight,
    wallx_cuda::TensorView q_action_weight,
    wallx_cuda::TensorView k_action_weight,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim) {
    ASSERT_CHECK(text_qkv.ndim() == 3);
    ASSERT_CHECK(action_qkv.ndim() == 3);
    ASSERT_CHECK(text_qkv.IsContiguous());
    ASSERT_CHECK(action_qkv.IsContiguous());
    // action_qkv is a packed action buffer indexed via action_offsets (flat
    // source_row = action_offsets[batch] + seq - action_start), NOT by its
    // batch dim, so it may hold only the action-bearing samples (B_a <= B) for
    // mixed action+multimodal batches. Only require it not exceed the batch.
    ASSERT_CHECK(action_qkv.size(0) <= text_qkv.size(0));
    const int64_t expected_qkv = (num_heads * 2 + num_key_value_heads * 2) * head_dim;
    ASSERT_CHECK(text_qkv.size(2) == expected_qkv);
    ASSERT_CHECK(action_qkv.size(2) == expected_qkv);
    ASSERT_CHECK(text_qkv.dtype().code == action_qkv.dtype().code);
    ASSERT_CHECK(text_qkv.dtype().bits == action_qkv.dtype().bits);
    const int device_type = text_qkv.device().device_type;
    const int device_id = text_qkv.device().device_id;
    check_same_device(action_qkv, "action_qkv", device_type, device_id);
    check_i64_1d(action_starts, "action_starts", text_qkv.size(0), device_type, device_id);
    check_i64_1d(action_ends, "action_ends", text_qkv.size(0), device_type, device_id);
    check_i64_1d(action_offsets, "action_offsets", text_qkv.size(0) + 1, device_type, device_id);
    check_weight(q_weight, "q_weight", head_dim, device_type, device_id);
    check_weight(k_weight, "k_weight", head_dim, device_type, device_id);
    check_weight(q_action_weight, "q_action_weight", head_dim, device_type, device_id);
    check_weight(k_action_weight, "k_action_weight", head_dim, device_type, device_id);
    if (!same_dtype(q_weight, k_weight) ||
        !same_dtype(q_weight, q_action_weight) ||
        !same_dtype(q_weight, k_action_weight)) {
        throw std::runtime_error("qwen35_vla_qkv_norm_replace: all norm weights must have the same dtype");
    }
}

void Qwen35VlaQkvNormReplaceForward(
    wallx_cuda::TensorView query,
    wallx_cuda::TensorView key,
    wallx_cuda::TensorView value,
    wallx_cuda::TensorView gate,
    wallx_cuda::TensorView text_qkv,
    wallx_cuda::TensorView action_qkv,
    wallx_cuda::TensorView action_starts,
    wallx_cuda::TensorView action_ends,
    wallx_cuda::TensorView action_offsets,
    wallx_cuda::TensorView q_weight,
    wallx_cuda::TensorView k_weight,
    wallx_cuda::TensorView q_action_weight,
    wallx_cuda::TensorView k_action_weight,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    double epsilon) {
    validate_common(
        text_qkv,
        action_qkv,
        action_starts,
        action_ends,
        action_offsets,
        q_weight,
        k_weight,
        q_action_weight,
        k_action_weight,
        num_heads,
        num_key_value_heads,
        head_dim);
    const int64_t batch_size = text_qkv.size(0);
    const int64_t seq_length = text_qkv.size(1);
    ASSERT_CHECK(query.ndim() == 4 && query.size(0) == batch_size && query.size(1) == seq_length);
    ASSERT_CHECK(query.size(2) == num_heads && query.size(3) == head_dim);
    ASSERT_CHECK(key.ndim() == 4 && key.size(0) == batch_size && key.size(1) == seq_length);
    ASSERT_CHECK(key.size(2) == num_key_value_heads && key.size(3) == head_dim);
    ASSERT_CHECK(value.ndim() == 4 && value.size(0) == batch_size && value.size(1) == seq_length);
    ASSERT_CHECK(value.size(2) == num_key_value_heads && value.size(3) == head_dim);
    ASSERT_CHECK(gate.ndim() == 3 && gate.size(0) == batch_size && gate.size(1) == seq_length);
    ASSERT_CHECK(gate.size(2) == num_heads * head_dim);
    ASSERT_CHECK(query.IsContiguous() && key.IsContiguous() && value.IsContiguous() && gate.IsContiguous());

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(text_qkv.device().device_type, text_qkv.device().device_id));

    if (text_qkv.dtype().code == kDLFloat && text_qkv.dtype().bits == 32) {
        launch_forward_for_weight_dtype<float>(
            query,
            key,
            value,
            gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            static_cast<float>(epsilon),
            stream);
    } else if (text_qkv.dtype().code == kDLFloat && text_qkv.dtype().bits == 16) {
        launch_forward_for_weight_dtype<__half>(
            query,
            key,
            value,
            gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            static_cast<float>(epsilon),
            stream);
    } else if (text_qkv.dtype().code == kDLBfloat && text_qkv.dtype().bits == 16) {
        launch_forward_for_weight_dtype<__nv_bfloat16>(
            query,
            key,
            value,
            gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            static_cast<float>(epsilon),
            stream);
    } else {
        throw std::runtime_error("qwen35_vla_qkv_norm_replace_forward: unsupported dtype");
    }
    sync_check_cuda_error();
}

void Qwen35VlaQkvNormReplaceBackward(
    wallx_cuda::TensorView grad_text_qkv,
    wallx_cuda::TensorView grad_action_qkv,
    wallx_cuda::TensorView grad_q_weight,
    wallx_cuda::TensorView grad_k_weight,
    wallx_cuda::TensorView grad_q_action_weight,
    wallx_cuda::TensorView grad_k_action_weight,
    wallx_cuda::TensorView partial_dw,
    wallx_cuda::TensorView grad_query,
    wallx_cuda::TensorView grad_key,
    wallx_cuda::TensorView grad_value,
    wallx_cuda::TensorView grad_gate,
    wallx_cuda::TensorView text_qkv,
    wallx_cuda::TensorView action_qkv,
    wallx_cuda::TensorView action_starts,
    wallx_cuda::TensorView action_ends,
    wallx_cuda::TensorView action_offsets,
    wallx_cuda::TensorView q_weight,
    wallx_cuda::TensorView k_weight,
    wallx_cuda::TensorView q_action_weight,
    wallx_cuda::TensorView k_action_weight,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    double epsilon) {
    validate_common(
        text_qkv,
        action_qkv,
        action_starts,
        action_ends,
        action_offsets,
        q_weight,
        k_weight,
        q_action_weight,
        k_action_weight,
        num_heads,
        num_key_value_heads,
        head_dim);
    ASSERT_CHECK(grad_text_qkv.IsContiguous() && grad_action_qkv.IsContiguous());
    ASSERT_CHECK(grad_query.IsContiguous() && grad_key.IsContiguous());
    ASSERT_CHECK(grad_value.IsContiguous() && grad_gate.IsContiguous());
    ASSERT_CHECK(grad_q_weight.IsContiguous() && grad_k_weight.IsContiguous());
    ASSERT_CHECK(grad_q_action_weight.IsContiguous() && grad_k_action_weight.IsContiguous());
    ASSERT_CHECK(grad_text_qkv.size(0) == text_qkv.size(0));
    ASSERT_CHECK(grad_text_qkv.size(1) == text_qkv.size(1));
    ASSERT_CHECK(grad_text_qkv.size(2) == text_qkv.size(2));
    ASSERT_CHECK(grad_action_qkv.size(0) == action_qkv.size(0));
    ASSERT_CHECK(grad_action_qkv.size(1) == action_qkv.size(1));
    ASSERT_CHECK(grad_action_qkv.size(2) == action_qkv.size(2));
    ASSERT_CHECK(grad_q_weight.dtype().code == kDLFloat && grad_q_weight.dtype().bits == 32);
    ASSERT_CHECK(grad_k_weight.dtype().code == kDLFloat && grad_k_weight.dtype().bits == 32);
    ASSERT_CHECK(grad_q_action_weight.dtype().code == kDLFloat && grad_q_action_weight.dtype().bits == 32);
    ASSERT_CHECK(grad_k_action_weight.dtype().code == kDLFloat && grad_k_action_weight.dtype().bits == 32);
    check_same_device(
        partial_dw,
        "partial_dw",
        text_qkv.device().device_type,
        text_qkv.device().device_id);

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(text_qkv.device().device_type, text_qkv.device().device_id));

    if (text_qkv.dtype().code == kDLFloat && text_qkv.dtype().bits == 32) {
        launch_backward_for_weight_dtype<float>(
            grad_text_qkv,
            grad_action_qkv,
            grad_q_weight,
            grad_k_weight,
            grad_q_action_weight,
            grad_k_action_weight,
            partial_dw,
            grad_query,
            grad_key,
            grad_value,
            grad_gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            static_cast<float>(epsilon),
            stream);
    } else if (text_qkv.dtype().code == kDLFloat && text_qkv.dtype().bits == 16) {
        launch_backward_for_weight_dtype<__half>(
            grad_text_qkv,
            grad_action_qkv,
            grad_q_weight,
            grad_k_weight,
            grad_q_action_weight,
            grad_k_action_weight,
            partial_dw,
            grad_query,
            grad_key,
            grad_value,
            grad_gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            static_cast<float>(epsilon),
            stream);
    } else if (text_qkv.dtype().code == kDLBfloat && text_qkv.dtype().bits == 16) {
        launch_backward_for_weight_dtype<__nv_bfloat16>(
            grad_text_qkv,
            grad_action_qkv,
            grad_q_weight,
            grad_k_weight,
            grad_q_action_weight,
            grad_k_action_weight,
            partial_dw,
            grad_query,
            grad_key,
            grad_value,
            grad_gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            static_cast<float>(epsilon),
            stream);
    } else {
        throw std::runtime_error("qwen35_vla_qkv_norm_replace_backward: unsupported dtype");
    }
    sync_check_cuda_error();
}




}  // namespace wallx_cuda_qwen35_vla_qkv_postprocess
