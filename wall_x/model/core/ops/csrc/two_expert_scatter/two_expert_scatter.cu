#include "tensor_view.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

#include "../common/cuda_utils.h"

namespace wallx_cuda_two_expert_scatter {

static int checked_row_blocks(int64_t batch_size, int64_t seq_length) {
    ASSERT_CHECK(batch_size >= 0);
    ASSERT_CHECK(seq_length >= 0);
    const int64_t max_blocks = std::numeric_limits<int>::max();
    if (seq_length != 0 && batch_size > max_blocks / seq_length) {
        throw std::runtime_error("two_expert_scatter: batch_size * seq_length exceeds INT_MAX");
    }
    return static_cast<int>(batch_size * seq_length);
}

static void check_index_tensor(
    wallx_cuda::TensorView tensor,
    const char* name,
    int64_t expected_size,
    int device_type,
    int device_id) {
    const std::string prefix = std::string(name) + ": ";
    if (tensor.ndim() != 1) {
        throw std::runtime_error(
            prefix + "expected 1D tensor, got " + std::to_string(tensor.ndim()) + "D");
    }
    if (tensor.size(0) != expected_size) {
        throw std::runtime_error(
            prefix + "expected size " + std::to_string(expected_size) + ", got " +
            std::to_string(tensor.size(0)));
    }
    if (!tensor.IsContiguous()) {
        throw std::runtime_error(prefix + "expected contiguous tensor");
    }
    if (!(tensor.dtype().code == kDLInt && tensor.dtype().bits == 64)) {
        throw std::runtime_error(prefix + "expected int64 tensor");
    }
    if (tensor.device().device_type != device_type ||
        tensor.device().device_id != device_id) {
        throw std::runtime_error(prefix + "expected tensor on the same device as input");
    }
}

static bool can_use_vec16(const void* output, const void* input, int64_t row_bytes) {
    return row_bytes % static_cast<int64_t>(sizeof(uint4)) == 0 &&
           (reinterpret_cast<uintptr_t>(output) % alignof(uint4)) == 0 &&
           (reinterpret_cast<uintptr_t>(input) % alignof(uint4)) == 0;
}

static int row_copy_threads(int64_t row_items) {
    if (row_items <= 64) {
        return 64;
    }
    if (row_items <= 128) {
        return 128;
    }
    return 256;
}

__device__ __forceinline__ int64_t find_batch_for_packed_row(
    const int64_t* __restrict__ offsets,
    int64_t batch_size,
    int64_t row) {
    int64_t low = 0;
    int64_t high = batch_size;
    while (low < high) {
        const int64_t mid = (low + high) >> 1;
        if (offsets[mid + 1] <= row) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

__device__ __forceinline__ int64_t source_row_from_packed_row(
    int64_t packed_row,
    int64_t batch_size,
    int64_t seq_length,
    const int64_t* __restrict__ action_starts,
    const int64_t* __restrict__ action_ends,
    const int64_t* __restrict__ llm_offsets,
    const int64_t* __restrict__ action_offsets) {
    const int64_t total_llm = llm_offsets[batch_size];

    if (packed_row < total_llm) {
        const int64_t batch = find_batch_for_packed_row(llm_offsets, batch_size, packed_row);
        const int64_t local_llm = packed_row - llm_offsets[batch];
        const int64_t action_start = action_starts[batch];
        const int64_t action_len = action_ends[batch] - action_start;
        const int64_t seq = (local_llm < action_start) ? local_llm : (local_llm + action_len);
        return batch * seq_length + seq;
    }

    const int64_t action_row = packed_row - total_llm;
    const int64_t batch = find_batch_for_packed_row(action_offsets, batch_size, action_row);
    const int64_t seq = action_starts[batch] + (action_row - action_offsets[batch]);
    return batch * seq_length + seq;
}

__device__ __forceinline__ int64_t packed_row_from_source_row(
    int64_t source_row,
    int64_t batch_size,
    int64_t seq_length,
    const int64_t* __restrict__ action_starts,
    const int64_t* __restrict__ action_ends,
    const int64_t* __restrict__ llm_offsets,
    const int64_t* __restrict__ action_offsets) {
    const int64_t batch = source_row / seq_length;
    const int64_t seq = source_row - batch * seq_length;
    const int64_t action_start = action_starts[batch];
    const int64_t action_end = action_ends[batch];
    const int64_t action_len = action_end - action_start;
    const int64_t total_llm = llm_offsets[batch_size];

    if (seq >= action_start && seq < action_end) {
        return total_llm + action_offsets[batch] + (seq - action_start);
    }

    const int64_t local_llm = (seq < action_start) ? seq : (seq - action_len);
    return llm_offsets[batch] + local_llm;
}

template <typename T>
__global__ void twoExpertPackRowsKernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    int64_t batch_size,
    int64_t seq_length,
    int64_t hidden_dim,
    const int64_t* __restrict__ action_starts,
    const int64_t* __restrict__ action_ends,
    const int64_t* __restrict__ llm_offsets,
    const int64_t* __restrict__ action_offsets) {
    const int64_t packed_row = blockIdx.x;
    const int64_t src_row = source_row_from_packed_row(
        packed_row, batch_size, seq_length, action_starts, action_ends, llm_offsets, action_offsets);

    for (int64_t col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
        output[packed_row * hidden_dim + col] = input[src_row * hidden_dim + col];
    }
}

__global__ void twoExpertPackVec16Kernel(
    uint4* __restrict__ output,
    const uint4* __restrict__ input,
    int64_t batch_size,
    int64_t seq_length,
    int64_t row_vecs,
    const int64_t* __restrict__ action_starts,
    const int64_t* __restrict__ action_ends,
    const int64_t* __restrict__ llm_offsets,
    const int64_t* __restrict__ action_offsets) {
    const int64_t packed_row = blockIdx.x;
    const int64_t src_row = source_row_from_packed_row(
        packed_row, batch_size, seq_length, action_starts, action_ends, llm_offsets, action_offsets);

    for (int64_t col = threadIdx.x; col < row_vecs; col += blockDim.x) {
        output[packed_row * row_vecs + col] = input[src_row * row_vecs + col];
    }
}

template <typename T>
__global__ void twoExpertUnpackRowsKernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    int64_t batch_size,
    int64_t seq_length,
    int64_t hidden_dim,
    const int64_t* __restrict__ action_starts,
    const int64_t* __restrict__ action_ends,
    const int64_t* __restrict__ llm_offsets,
    const int64_t* __restrict__ action_offsets) {
    const int64_t src_row = blockIdx.x;
    const int64_t packed_row = packed_row_from_source_row(
        src_row, batch_size, seq_length, action_starts, action_ends, llm_offsets, action_offsets);

    for (int64_t col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
        output[src_row * hidden_dim + col] = input[packed_row * hidden_dim + col];
    }
}

__global__ void twoExpertUnpackVec16Kernel(
    uint4* __restrict__ output,
    const uint4* __restrict__ input,
    int64_t batch_size,
    int64_t seq_length,
    int64_t row_vecs,
    const int64_t* __restrict__ action_starts,
    const int64_t* __restrict__ action_ends,
    const int64_t* __restrict__ llm_offsets,
    const int64_t* __restrict__ action_offsets) {
    const int64_t src_row = blockIdx.x;
    const int64_t packed_row = packed_row_from_source_row(
        src_row, batch_size, seq_length, action_starts, action_ends, llm_offsets, action_offsets);

    for (int64_t col = threadIdx.x; col < row_vecs; col += blockDim.x) {
        output[src_row * row_vecs + col] = input[packed_row * row_vecs + col];
    }
}

static void launch_pack_vec16(
    void* output,
    const void* input,
    int64_t batch_size,
    int64_t seq_length,
    int64_t row_bytes,
    const int64_t* action_starts,
    const int64_t* action_ends,
    const int64_t* llm_offsets,
    const int64_t* action_offsets,
    cudaStream_t stream) {
    const int64_t row_vecs = row_bytes / static_cast<int64_t>(sizeof(uint4));
    const int blocks = checked_row_blocks(batch_size, seq_length);
    const int threads = row_copy_threads(row_vecs);
    twoExpertPackVec16Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<uint4*>(output),
        static_cast<const uint4*>(input),
        batch_size,
        seq_length,
        row_vecs,
        action_starts,
        action_ends,
        llm_offsets,
        action_offsets);
}

static void launch_unpack_vec16(
    void* output,
    const void* input,
    int64_t batch_size,
    int64_t seq_length,
    int64_t row_bytes,
    const int64_t* action_starts,
    const int64_t* action_ends,
    const int64_t* llm_offsets,
    const int64_t* action_offsets,
    cudaStream_t stream) {
    const int64_t row_vecs = row_bytes / static_cast<int64_t>(sizeof(uint4));
    const int blocks = checked_row_blocks(batch_size, seq_length);
    const int threads = row_copy_threads(row_vecs);
    twoExpertUnpackVec16Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<uint4*>(output),
        static_cast<const uint4*>(input),
        batch_size,
        seq_length,
        row_vecs,
        action_starts,
        action_ends,
        llm_offsets,
        action_offsets);
}

template <typename T>
static void launch_pack(
    T* output,
    const T* input,
    int64_t batch_size,
    int64_t seq_length,
    int64_t hidden_dim,
    const int64_t* action_starts,
    const int64_t* action_ends,
    const int64_t* llm_offsets,
    const int64_t* action_offsets,
    cudaStream_t stream) {
    const int64_t row_bytes = hidden_dim * static_cast<int64_t>(sizeof(T));
    if (can_use_vec16(output, input, row_bytes)) {
        launch_pack_vec16(
            output,
            input,
            batch_size,
            seq_length,
            row_bytes,
            action_starts,
            action_ends,
            llm_offsets,
            action_offsets,
            stream);
        return;
    }

    const int blocks = checked_row_blocks(batch_size, seq_length);
    const int threads = row_copy_threads(hidden_dim);
    twoExpertPackRowsKernel<T><<<blocks, threads, 0, stream>>>(
        output,
        input,
        batch_size,
        seq_length,
        hidden_dim,
        action_starts,
        action_ends,
        llm_offsets,
        action_offsets);
}

template <typename T>
static void launch_unpack(
    T* output,
    const T* input,
    int64_t batch_size,
    int64_t seq_length,
    int64_t hidden_dim,
    const int64_t* action_starts,
    const int64_t* action_ends,
    const int64_t* llm_offsets,
    const int64_t* action_offsets,
    cudaStream_t stream) {
    const int64_t row_bytes = hidden_dim * static_cast<int64_t>(sizeof(T));
    if (can_use_vec16(output, input, row_bytes)) {
        launch_unpack_vec16(
            output,
            input,
            batch_size,
            seq_length,
            row_bytes,
            action_starts,
            action_ends,
            llm_offsets,
            action_offsets,
            stream);
        return;
    }

    const int blocks = checked_row_blocks(batch_size, seq_length);
    const int threads = row_copy_threads(hidden_dim);
    twoExpertUnpackRowsKernel<T><<<blocks, threads, 0, stream>>>(
        output,
        input,
        batch_size,
        seq_length,
        hidden_dim,
        action_starts,
        action_ends,
        llm_offsets,
        action_offsets);
}

void TwoExpertPack(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView input,
    wallx_cuda::TensorView action_starts,
    wallx_cuda::TensorView action_ends,
    wallx_cuda::TensorView llm_offsets,
    wallx_cuda::TensorView action_offsets) {
    ASSERT_CHECK(input.ndim() == 3);
    ASSERT_CHECK(output.ndim() == 2);
    ASSERT_CHECK(input.IsContiguous());
    ASSERT_CHECK(output.IsContiguous());

    const int64_t batch_size = input.size(0);
    const int64_t seq_length = input.size(1);
    const int64_t hidden_dim = input.size(2);
    check_index_tensor(
        action_starts, "action_starts", batch_size, input.device().device_type, input.device().device_id);
    check_index_tensor(
        action_ends, "action_ends", batch_size, input.device().device_type, input.device().device_id);
    check_index_tensor(
        llm_offsets, "llm_offsets", batch_size + 1, input.device().device_type, input.device().device_id);
    check_index_tensor(
        action_offsets, "action_offsets", batch_size + 1, input.device().device_type, input.device().device_id);
    ASSERT_CHECK(output.size(0) == batch_size * seq_length);
    ASSERT_CHECK(output.size(1) == hidden_dim);
    ASSERT_CHECK(output.dtype().code == input.dtype().code);
    ASSERT_CHECK(output.dtype().bits == input.dtype().bits);
    ASSERT_CHECK(output.device().device_type == input.device().device_type);
    ASSERT_CHECK(output.device().device_id == input.device().device_id);

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(input.device().device_type, input.device().device_id));

    const int64_t* action_starts_ptr = static_cast<const int64_t*>(action_starts.data_ptr());
    const int64_t* action_ends_ptr = static_cast<const int64_t*>(action_ends.data_ptr());
    const int64_t* llm_offsets_ptr = static_cast<const int64_t*>(llm_offsets.data_ptr());
    const int64_t* action_offsets_ptr = static_cast<const int64_t*>(action_offsets.data_ptr());

    if (input.dtype().code == kDLFloat && input.dtype().bits == 32) {
        launch_pack<float>(
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(input.data_ptr()),
            batch_size,
            seq_length,
            hidden_dim,
            action_starts_ptr,
            action_ends_ptr,
            llm_offsets_ptr,
            action_offsets_ptr,
            stream);
    } else if (input.dtype().code == kDLFloat && input.dtype().bits == 16) {
        launch_pack<__half>(
            static_cast<__half*>(output.data_ptr()),
            static_cast<const __half*>(input.data_ptr()),
            batch_size,
            seq_length,
            hidden_dim,
            action_starts_ptr,
            action_ends_ptr,
            llm_offsets_ptr,
            action_offsets_ptr,
            stream);
    } else if (input.dtype().code == kDLBfloat && input.dtype().bits == 16) {
        launch_pack<__nv_bfloat16>(
            static_cast<__nv_bfloat16*>(output.data_ptr()),
            static_cast<const __nv_bfloat16*>(input.data_ptr()),
            batch_size,
            seq_length,
            hidden_dim,
            action_starts_ptr,
            action_ends_ptr,
            llm_offsets_ptr,
            action_offsets_ptr,
            stream);
    } else {
        throw std::runtime_error("two_expert_pack: unsupported dtype");
    }
    sync_check_cuda_error();
}

void TwoExpertUnpack(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView input,
    int64_t batch_size,
    int64_t seq_length,
    wallx_cuda::TensorView action_starts,
    wallx_cuda::TensorView action_ends,
    wallx_cuda::TensorView llm_offsets,
    wallx_cuda::TensorView action_offsets) {
    ASSERT_CHECK(input.ndim() == 2);
    ASSERT_CHECK(output.ndim() == 3);
    ASSERT_CHECK(input.IsContiguous());
    ASSERT_CHECK(output.IsContiguous());

    const int64_t hidden_dim = input.size(1);
    check_index_tensor(
        action_starts, "action_starts", batch_size, input.device().device_type, input.device().device_id);
    check_index_tensor(
        action_ends, "action_ends", batch_size, input.device().device_type, input.device().device_id);
    check_index_tensor(
        llm_offsets, "llm_offsets", batch_size + 1, input.device().device_type, input.device().device_id);
    check_index_tensor(
        action_offsets, "action_offsets", batch_size + 1, input.device().device_type, input.device().device_id);
    ASSERT_CHECK(input.size(0) == batch_size * seq_length);
    ASSERT_CHECK(output.size(0) == batch_size);
    ASSERT_CHECK(output.size(1) == seq_length);
    ASSERT_CHECK(output.size(2) == hidden_dim);
    ASSERT_CHECK(output.dtype().code == input.dtype().code);
    ASSERT_CHECK(output.dtype().bits == input.dtype().bits);
    ASSERT_CHECK(output.device().device_type == input.device().device_type);
    ASSERT_CHECK(output.device().device_id == input.device().device_id);

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(input.device().device_type, input.device().device_id));

    const int64_t* action_starts_ptr = static_cast<const int64_t*>(action_starts.data_ptr());
    const int64_t* action_ends_ptr = static_cast<const int64_t*>(action_ends.data_ptr());
    const int64_t* llm_offsets_ptr = static_cast<const int64_t*>(llm_offsets.data_ptr());
    const int64_t* action_offsets_ptr = static_cast<const int64_t*>(action_offsets.data_ptr());

    if (input.dtype().code == kDLFloat && input.dtype().bits == 32) {
        launch_unpack<float>(
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(input.data_ptr()),
            batch_size,
            seq_length,
            hidden_dim,
            action_starts_ptr,
            action_ends_ptr,
            llm_offsets_ptr,
            action_offsets_ptr,
            stream);
    } else if (input.dtype().code == kDLFloat && input.dtype().bits == 16) {
        launch_unpack<__half>(
            static_cast<__half*>(output.data_ptr()),
            static_cast<const __half*>(input.data_ptr()),
            batch_size,
            seq_length,
            hidden_dim,
            action_starts_ptr,
            action_ends_ptr,
            llm_offsets_ptr,
            action_offsets_ptr,
            stream);
    } else if (input.dtype().code == kDLBfloat && input.dtype().bits == 16) {
        launch_unpack<__nv_bfloat16>(
            static_cast<__nv_bfloat16*>(output.data_ptr()),
            static_cast<const __nv_bfloat16*>(input.data_ptr()),
            batch_size,
            seq_length,
            hidden_dim,
            action_starts_ptr,
            action_ends_ptr,
            llm_offsets_ptr,
            action_offsets_ptr,
            stream);
    } else {
        throw std::runtime_error("two_expert_unpack: unsupported dtype");
    }
    sync_check_cuda_error();
}




}  // namespace wallx_cuda_two_expert_scatter
