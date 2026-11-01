#include "tensor_view.h"
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "../common/cuda_utils.h"

namespace wallx_cuda_qwen35_attention_mask {

constexpr int kThreads = 256;

static void check_same_device(
    wallx_cuda::TensorView tensor,
    const char* name,
    int device_type,
    int device_id) {
    if (tensor.device().device_type != device_type ||
        tensor.device().device_id != device_id) {
        throw std::runtime_error(std::string(name) + ": expected tensor on the same device");
    }
}

static bool is_bool_tensor(wallx_cuda::TensorView tensor) {
    return tensor.dtype().code == kDLBool ||
           (tensor.dtype().code == kDLUInt && tensor.dtype().bits == 8);
}

static void check_optional_mask(
    const wallx_cuda::OptionalTensorView& maybe_tensor,
    const char* name,
    int64_t batch_size,
    int64_t seq_len,
    int device_type,
    int device_id,
    bool require_bool = true) {
    if (!maybe_tensor.has_value()) {
        return;
    }
    const wallx_cuda::TensorView tensor = maybe_tensor.value();
    if (tensor.ndim() != 2 || tensor.size(0) != batch_size || tensor.size(1) != seq_len) {
        throw std::runtime_error(std::string(name) + ": expected shape [B, S]");
    }
    if (!tensor.IsContiguous()) {
        throw std::runtime_error(std::string(name) + ": expected contiguous tensor");
    }
    if (require_bool && !is_bool_tensor(tensor)) {
        throw std::runtime_error(std::string(name) + ": expected bool tensor");
    }
    check_same_device(tensor, name, device_type, device_id);
}

static void check_optional_i64(
    const wallx_cuda::OptionalTensorView& maybe_tensor,
    const char* name,
    int64_t batch_size,
    int64_t seq_len,
    int device_type,
    int device_id) {
    if (!maybe_tensor.has_value()) {
        return;
    }
    const wallx_cuda::TensorView tensor = maybe_tensor.value();
    if (tensor.ndim() != 2 || tensor.size(0) != batch_size || tensor.size(1) != seq_len) {
        throw std::runtime_error(std::string(name) + ": expected shape [B, S]");
    }
    if (!tensor.IsContiguous()) {
        throw std::runtime_error(std::string(name) + ": expected contiguous tensor");
    }
    if (!(tensor.dtype().code == kDLInt && tensor.dtype().bits == 64)) {
        throw std::runtime_error(std::string(name) + ": expected int64 tensor");
    }
    check_same_device(tensor, name, device_type, device_id);
}

__device__ __forceinline__ bool read_mask(const bool* ptr, int64_t idx) {
    return ptr != nullptr && ptr[idx];
}

__device__ __forceinline__ int64_t read_i64_or(const int64_t* ptr, int64_t idx, int64_t fallback) {
    return ptr == nullptr ? fallback : ptr[idx];
}

__device__ __forceinline__ int min_i32(int a, int b) {
    return a < b ? a : b;
}

__device__ __forceinline__ int max_i32(int a, int b) {
    return a > b ? a : b;
}

__device__ void reduce_first_last_mask(
    int batch,
    int seq_len,
    const bool* __restrict__ mask_a,
    const bool* __restrict__ mask_b,
    int& first,
    int& last) {
    int local_first = seq_len;
    int local_last = -1;
    const int64_t base = static_cast<int64_t>(batch) * seq_len;
    for (int s = threadIdx.x; s < seq_len; s += blockDim.x) {
        const int64_t idx = base + s;
        if (read_mask(mask_a, idx) || read_mask(mask_b, idx)) {
            local_first = min_i32(local_first, s);
            local_last = max_i32(local_last, s);
        }
    }

    __shared__ int first_buf[kThreads];
    __shared__ int last_buf[kThreads];
    first_buf[threadIdx.x] = local_first;
    last_buf[threadIdx.x] = local_last;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            first_buf[threadIdx.x] = min_i32(first_buf[threadIdx.x], first_buf[threadIdx.x + stride]);
            last_buf[threadIdx.x] = max_i32(last_buf[threadIdx.x], last_buf[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    __shared__ int sh_first;
    __shared__ int sh_last;
    if (threadIdx.x == 0) {
        sh_first = first_buf[0] == seq_len ? -1 : first_buf[0];
        sh_last = last_buf[0];
    }
    __syncthreads();
    first = sh_first;
    last = sh_last;
}

__global__ void update_flash_mask_kernel(
    int32_t* __restrict__ output,
    const bool* __restrict__ moe_token_types,
    const bool* __restrict__ padding_positions,
    const bool* __restrict__ ar_predict_positions,
    const bool* __restrict__ valid_flow_positions,
    const bool* __restrict__ ar_action_mask,
    const int64_t* __restrict__ flow_block_id,
    const bool* __restrict__ rtc_pinned,
    int batch_size,
    int num_heads,
    int seq_len,
    bool causal_action_attention_mask,
    bool ar_visible) {
    const int batch = blockIdx.x;
    if (batch >= batch_size) {
        return;
    }
    const int64_t base = static_cast<int64_t>(batch) * seq_len;

    int start_ar = -1;
    int end_ar = -1;
    int start_flow = -1;
    int end_flow = -1;
    int start_ar_action = -1;
    int end_ar_action = -1;
    int start_flow_ar = -1;
    int end_flow_ar = -1;

    reduce_first_last_mask(batch, seq_len, ar_predict_positions, nullptr, start_ar, end_ar);
    reduce_first_last_mask(batch, seq_len, moe_token_types, nullptr, start_flow, end_flow);
    reduce_first_last_mask(batch, seq_len, ar_action_mask, nullptr, start_ar_action, end_ar_action);
    reduce_first_last_mask(batch, seq_len, ar_action_mask, moe_token_types, start_flow_ar, end_flow_ar);

    for (int s = threadIdx.x; s < seq_len; s += blockDim.x) {
        const int64_t idx = base + s;
        int32_t lts = seq_len;
        int32_t ute = s;

        if (read_mask(padding_positions, idx)) {
            lts = 0;
            ute = seq_len;
        }

        if (read_mask(ar_predict_positions, idx) && end_ar >= 0) {
            lts = end_ar + 1;
        }

        const bool is_flow = read_mask(moe_token_types, idx);
        if (ar_action_mask == nullptr) {
            if (is_flow && !causal_action_attention_mask && start_flow >= 0) {
                ute = start_flow;
            }
        } else if (ar_visible) {
            const bool in_block = read_mask(ar_action_mask, idx) || is_flow;
            if (in_block && start_flow_ar >= 0) {
                ute = start_flow_ar;
                lts = end_flow_ar + 1;
            }
        } else {
            if (is_flow && start_flow >= 0) {
                ute = start_flow;
                lts = end_flow + 1;
            }
            if (read_mask(ar_action_mask, idx) && start_ar_action >= 0) {
                ute = start_ar_action;
                lts = end_ar_action + 1;
            }
        }

        const int64_t block_id = read_i64_or(flow_block_id, idx, -1);
        if (flow_block_id != nullptr && is_flow && block_id >= 0) {
            int block_start = seq_len;
            int block_end = -1;
            int suffix_start = seq_len;
            int pinned_count = 0;
            int suffix_count = 0;
            for (int j = 0; j < seq_len; ++j) {
                const int64_t jidx = base + j;
                if (!read_mask(moe_token_types, jidx) || read_i64_or(flow_block_id, jidx, -1) != block_id) {
                    continue;
                }
                block_start = min_i32(block_start, j);
                block_end = max_i32(block_end, j);
                const bool pinned = read_mask(rtc_pinned, jidx);
                if (rtc_pinned != nullptr && pinned) {
                    ++pinned_count;
                } else {
                    ++suffix_count;
                    suffix_start = min_i32(suffix_start, j);
                }
            }
            if (block_end >= 0) {
                ute = block_start;
                lts = block_end + 1;
                const bool token_suffix = rtc_pinned != nullptr && !read_mask(rtc_pinned, idx);
                if (token_suffix && pinned_count > 0 && suffix_count > 0 && suffix_start < seq_len) {
                    ute = suffix_start;
                }
            }
        }

        if (valid_flow_positions != nullptr && is_flow && !valid_flow_positions[idx]) {
            lts = 0;
            ute = seq_len;
        }

        for (int head = 0; head < num_heads; ++head) {
            const int64_t out_idx =
                (((static_cast<int64_t>(batch) * num_heads + head) * seq_len + s) * 2);
            output[out_idx] = lts;
            output[out_idx + 1] = ute;
        }
    }
}

__device__ __forceinline__ int64_t position_after_ar(
    const int64_t* __restrict__ position_ids,
    int dim,
    int batch,
    int seq,
    int dims,
    int batch_size,
    int seq_len,
    int start_ar,
    int end_ar,
    int start_flow) {
    const int64_t pos_idx =
        (static_cast<int64_t>(dim) * batch_size + batch) * seq_len + seq;
    if (start_ar < 0 || end_ar < 0 || start_flow < 0 || seq < start_flow) {
        return position_ids[pos_idx];
    }
    const int64_t flow_idx =
        (static_cast<int64_t>(dim) * batch_size + batch) * seq_len + start_flow;
    const int64_t ar_idx =
        (static_cast<int64_t>(dim) * batch_size + batch) * seq_len + start_ar;
    return position_ids[pos_idx] - (position_ids[flow_idx] - position_ids[ar_idx]);
}

__global__ void update_position_ids_kernel(
    int64_t* __restrict__ output,
    const int64_t* __restrict__ position_ids,
    const bool* __restrict__ moe_token_types,
    const bool* __restrict__ ar_predict_positions,
    const int64_t* __restrict__ flow_block_id,
    const bool* __restrict__ rtc_pinned,
    int dims,
    int batch_size,
    int seq_len,
    int rtc_rope_gap) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(dims) * batch_size * seq_len;
    if (linear >= total) {
        return;
    }

    const int seq = static_cast<int>(linear % seq_len);
    const int batch = static_cast<int>((linear / seq_len) % batch_size);
    const int dim = static_cast<int>(linear / (static_cast<int64_t>(batch_size) * seq_len));
    const int64_t base = static_cast<int64_t>(batch) * seq_len;
    const int64_t idx = base + seq;

    int start_ar = -1;
    int end_ar = -1;
    int start_flow = -1;
    if (ar_predict_positions != nullptr) {
        for (int s = 0; s < seq_len; ++s) {
            const int64_t jidx = base + s;
            if (ar_predict_positions[jidx]) {
                if (start_ar < 0) {
                    start_ar = s;
                }
                end_ar = s;
            }
            if (moe_token_types != nullptr && moe_token_types[jidx] && start_flow < 0) {
                start_flow = s;
            }
        }
    }

    int source_seq = seq;
    const int64_t block_id = read_i64_or(flow_block_id, idx, -1);
    if (flow_block_id != nullptr && block_id > 0) {
        int ordinal = 0;
        for (int s = 0; s < seq; ++s) {
            if (read_i64_or(flow_block_id, base + s, -1) == block_id) {
                ++ordinal;
            }
        }
        int seen = 0;
        int matched = -1;
        for (int s = 0; s < seq_len; ++s) {
            if (read_i64_or(flow_block_id, base + s, -1) != 0) {
                continue;
            }
            if (seen == ordinal) {
                matched = s;
                break;
            }
            ++seen;
        }
        if (matched >= 0) {
            source_seq = matched;
        }
    }

    int64_t value = position_after_ar(
        position_ids,
        dim,
        batch,
        source_seq,
        dims,
        batch_size,
        seq_len,
        start_ar,
        end_ar,
        start_flow);

    if (rtc_rope_gap > 0 && rtc_pinned != nullptr && moe_token_types != nullptr &&
        moe_token_types[idx] && !rtc_pinned[idx]) {
        value += static_cast<int64_t>(rtc_rope_gap);
    }
    output[linear] = value;
}

void Qwen35UpdateJointAttentionFlashMask(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView moe_token_types,
    const wallx_cuda::OptionalTensorView padding_positions,
    const wallx_cuda::OptionalTensorView ar_predict_positions,
    const wallx_cuda::OptionalTensorView valid_flow_positions,
    const wallx_cuda::OptionalTensorView ar_action_mask,
    const wallx_cuda::OptionalTensorView flow_block_id,
    const wallx_cuda::OptionalTensorView rtc_pinned,
    bool causal_action_attention_mask,
    bool ar_visible,
    int64_t num_heads) {
    ASSERT_CHECK(moe_token_types.ndim() == 2);
    ASSERT_CHECK(moe_token_types.IsContiguous());
    ASSERT_CHECK(is_bool_tensor(moe_token_types));
    const int64_t batch_size = moe_token_types.size(0);
    const int64_t seq_len = moe_token_types.size(1);
    ASSERT_CHECK(num_heads >= 1);
    ASSERT_CHECK(output.ndim() == 4);
    ASSERT_CHECK(output.size(0) == batch_size);
    ASSERT_CHECK(output.size(1) == num_heads);
    ASSERT_CHECK(output.size(2) == seq_len);
    ASSERT_CHECK(output.size(3) == 2);
    ASSERT_CHECK(output.IsContiguous());
    ASSERT_CHECK(output.dtype().code == kDLInt && output.dtype().bits == 32);

    const int device_type = moe_token_types.device().device_type;
    const int device_id = moe_token_types.device().device_id;
    check_same_device(output, "output", device_type, device_id);
    check_optional_mask(padding_positions, "padding_positions", batch_size, seq_len, device_type, device_id);
    check_optional_mask(ar_predict_positions, "ar_predict_positions", batch_size, seq_len, device_type, device_id);
    check_optional_mask(valid_flow_positions, "valid_flow_positions", batch_size, seq_len, device_type, device_id);
    check_optional_mask(ar_action_mask, "ar_action_mask", batch_size, seq_len, device_type, device_id);
    check_optional_i64(flow_block_id, "flow_block_id", batch_size, seq_len, device_type, device_id);
    check_optional_mask(rtc_pinned, "rtc_pinned", batch_size, seq_len, device_type, device_id);

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(output.device().device_type, output.device().device_id));
    update_flash_mask_kernel<<<static_cast<unsigned int>(batch_size), kThreads, 0, stream>>>(
        static_cast<int32_t*>(output.data_ptr()),
        static_cast<const bool*>(moe_token_types.data_ptr()),
        padding_positions.has_value() ? static_cast<const bool*>(padding_positions.value().data_ptr()) : nullptr,
        ar_predict_positions.has_value() ? static_cast<const bool*>(ar_predict_positions.value().data_ptr()) : nullptr,
        valid_flow_positions.has_value() ? static_cast<const bool*>(valid_flow_positions.value().data_ptr()) : nullptr,
        ar_action_mask.has_value() ? static_cast<const bool*>(ar_action_mask.value().data_ptr()) : nullptr,
        flow_block_id.has_value() ? static_cast<const int64_t*>(flow_block_id.value().data_ptr()) : nullptr,
        rtc_pinned.has_value() ? static_cast<const bool*>(rtc_pinned.value().data_ptr()) : nullptr,
        static_cast<int>(batch_size),
        static_cast<int>(num_heads),
        static_cast<int>(seq_len),
        causal_action_attention_mask,
        ar_visible);
    sync_check_cuda_error();
}

void Qwen35UpdatePositionIds(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView position_ids,
    wallx_cuda::TensorView moe_token_types,
    const wallx_cuda::OptionalTensorView ar_predict_positions,
    const wallx_cuda::OptionalTensorView flow_block_id,
    const wallx_cuda::OptionalTensorView rtc_pinned,
    int rtc_rope_gap) {
    ASSERT_CHECK(position_ids.ndim() == 3);
    ASSERT_CHECK(position_ids.IsContiguous());
    ASSERT_CHECK(position_ids.dtype().code == kDLInt && position_ids.dtype().bits == 64);
    ASSERT_CHECK(moe_token_types.ndim() == 2);
    ASSERT_CHECK(moe_token_types.IsContiguous());
    ASSERT_CHECK(is_bool_tensor(moe_token_types));
    const int64_t dims = position_ids.size(0);
    const int64_t batch_size = position_ids.size(1);
    const int64_t seq_len = position_ids.size(2);
    ASSERT_CHECK(moe_token_types.size(0) == batch_size);
    ASSERT_CHECK(moe_token_types.size(1) == seq_len);
    ASSERT_CHECK(output.ndim() == 3);
    ASSERT_CHECK(output.size(0) == dims);
    ASSERT_CHECK(output.size(1) == batch_size);
    ASSERT_CHECK(output.size(2) == seq_len);
    ASSERT_CHECK(output.IsContiguous());
    ASSERT_CHECK(output.dtype().code == kDLInt && output.dtype().bits == 64);

    const int device_type = position_ids.device().device_type;
    const int device_id = position_ids.device().device_id;
    check_same_device(output, "output", device_type, device_id);
    check_same_device(moe_token_types, "moe_token_types", device_type, device_id);
    check_optional_mask(ar_predict_positions, "ar_predict_positions", batch_size, seq_len, device_type, device_id);
    check_optional_i64(flow_block_id, "flow_block_id", batch_size, seq_len, device_type, device_id);
    check_optional_mask(rtc_pinned, "rtc_pinned", batch_size, seq_len, device_type, device_id);

    const int64_t total = dims * batch_size * seq_len;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(output.device().device_type, output.device().device_id));
    update_position_ids_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<int64_t*>(output.data_ptr()),
        static_cast<const int64_t*>(position_ids.data_ptr()),
        static_cast<const bool*>(moe_token_types.data_ptr()),
        ar_predict_positions.has_value() ? static_cast<const bool*>(ar_predict_positions.value().data_ptr()) : nullptr,
        flow_block_id.has_value() ? static_cast<const int64_t*>(flow_block_id.value().data_ptr()) : nullptr,
        rtc_pinned.has_value() ? static_cast<const bool*>(rtc_pinned.value().data_ptr()) : nullptr,
        static_cast<int>(dims),
        static_cast<int>(batch_size),
        static_cast<int>(seq_len),
        rtc_rope_gap);
    sync_check_cuda_error();
}




}  // namespace wallx_cuda_qwen35_attention_mask
