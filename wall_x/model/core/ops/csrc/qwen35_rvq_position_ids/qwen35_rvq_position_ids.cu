#include "tensor_view.h"
// qwen35_rvq_position_ids: rewrite mrope position ids for serialized RVQ
// action spans.
//
// Every <rvq_start> token anchors a span of num_latents * num_quantizers code
// tokens. For a span slot the temporal route (dim 1) becomes
// anchor + 1 + time and the level routes (dims 2, 3) become
// anchor + 1 + level, where anchor is the pre-rewrite dim-1 position of the
// <rvq_start> token and time / level derive from the slot index by
// token_layout (time_major: time, level = divmod(slot, num_quantizers);
// else level, time = divmod(slot, num_latents)).
//
// Semantics are bit-exact against the wall-x PyTorch fallback
// (wall_x/model/core/ops/seq_rvq.py::qwen35_rvq_position_ids_pytorch): the
// fallback resolves each position's span start with a running cummax over
// start positions and only rewrites offsets 1..num_code_tokens, so a
// backward scan bounded by num_code_tokens positions is exactly equivalent.
// All arithmetic is int64; there is no floating point anywhere.
//
// Error contract (mirrors the fallback's fused validation; the Python
// binding turns any set flag into the legacy ValueError):
//   err_token   — an in-span slot whose token's level_lut level differs from
//                 the slot's expected level (ids outside the lut are -1),
//   err_tail    — a start token whose span would run past the row end,
//   err_overlap — a start token within num_code_tokens of a previous start.

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "../common/cuda_utils.h"

namespace wallx_cuda_qwen35_rvq_position_ids {

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

__global__ void rvq_position_ids_kernel(
    int64_t* __restrict__ output,           // [4, B, L] pre-filled copy of position ids
    const int64_t* __restrict__ input_ids,  // [B, L]
    const int8_t* __restrict__ level_lut,   // [lut_len] token id -> level, -1 otherwise
    int32_t* __restrict__ err,              // [B] zero-initialized error flags
    int64_t lut_len,
    int64_t start_token_id,
    int64_t num_latents,
    int64_t num_quantizers,
    bool time_major,
    int64_t batch_size,
    int64_t seq_len) {
    const int64_t batch = blockIdx.x;
    if (batch >= batch_size) {
        return;
    }
    const int64_t num_code = num_latents * num_quantizers;
    const int64_t row = batch * seq_len;
    const int64_t plane = batch_size * seq_len;
    int local_err = 0;

    for (int64_t i = threadIdx.x; i < seq_len; i += blockDim.x) {
        const int64_t token = input_ids[row + i];
        const int64_t lo = (i - num_code) > 0 ? (i - num_code) : 0;
        if (token == start_token_id) {
            // err_tail: the span would run past the row end.
            if (i >= seq_len - num_code) {
                local_err = 1;
            }
            // err_overlap: a previous start within num_code positions. The
            // fallback checks i - prev_start <= num_code against the latest
            // start before i, so the bounded backward scan is equivalent.
            for (int64_t j = i - 1; j >= lo; --j) {
                if (input_ids[row + j] == start_token_id) {
                    local_err = 1;
                    break;
                }
            }
            continue;  // start positions are never rewritten
        }

        // Nearest start before i within num_code positions (== the fallback's
        // cummax(last_start) restricted to offsets 1..num_code, the only ones
        // that are rewritten).
        int64_t start = -1;
        for (int64_t j = i - 1; j >= lo; --j) {
            if (input_ids[row + j] == start_token_id) {
                start = j;
                break;
            }
        }
        if (start < 0) {
            continue;
        }
        const int64_t slot = i - start - 1;
        int64_t time_idx;
        int64_t level_idx;
        if (time_major) {
            time_idx = slot / num_quantizers;
            level_idx = slot % num_quantizers;
        } else {
            level_idx = slot / num_latents;
            time_idx = slot % num_latents;
        }
        const int64_t token_level = (token >= 0 && token < lut_len)
            ? static_cast<int64_t>(level_lut[token])
            : static_cast<int64_t>(-1);
        if (token_level != level_idx) {
            local_err = 1;  // err_token
        }
        // Anchor: pre-rewrite dim-1 position of the start token. Start
        // positions are never written by any thread, so this read is
        // race-free.
        const int64_t anchor = output[plane + row + start];
        output[plane + row + i] = anchor + 1 + time_idx;
        output[2 * plane + row + i] = anchor + 1 + level_idx;
        output[3 * plane + row + i] = anchor + 1 + level_idx;
    }

    if (local_err) {
        atomicOr(reinterpret_cast<int*>(err) + batch, 1);
    }
}

void Qwen35RvqPositionIds(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView input_ids,
    wallx_cuda::TensorView level_lut,
    wallx_cuda::TensorView err,
    int64_t start_token_id,
    int64_t num_latents,
    int64_t num_quantizers,
    bool time_major) {
    ASSERT_CHECK(input_ids.ndim() == 2);
    ASSERT_CHECK(input_ids.IsContiguous());
    ASSERT_CHECK(input_ids.dtype().code == kDLInt && input_ids.dtype().bits == 64);
    const int64_t batch_size = input_ids.size(0);
    const int64_t seq_len = input_ids.size(1);
    ASSERT_CHECK(output.ndim() == 3);
    ASSERT_CHECK(output.size(0) == 4);
    ASSERT_CHECK(output.size(1) == batch_size);
    ASSERT_CHECK(output.size(2) == seq_len);
    ASSERT_CHECK(output.IsContiguous());
    ASSERT_CHECK(output.dtype().code == kDLInt && output.dtype().bits == 64);
    ASSERT_CHECK(level_lut.ndim() == 1);
    ASSERT_CHECK(level_lut.IsContiguous());
    ASSERT_CHECK(level_lut.dtype().code == kDLInt && level_lut.dtype().bits == 8);
    ASSERT_CHECK(err.ndim() == 1);
    ASSERT_CHECK(err.size(0) == batch_size);
    ASSERT_CHECK(err.IsContiguous());
    ASSERT_CHECK(err.dtype().code == kDLInt && err.dtype().bits == 32);
    ASSERT_CHECK(num_latents >= 0);
    ASSERT_CHECK(num_quantizers >= 0);

    const int device_type = input_ids.device().device_type;
    const int device_id = input_ids.device().device_id;
    check_same_device(output, "output", device_type, device_id);
    check_same_device(level_lut, "level_lut", device_type, device_id);
    check_same_device(err, "err", device_type, device_id);

    if (batch_size == 0 || seq_len == 0) {
        return;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(output.device().device_type, output.device().device_id));
    rvq_position_ids_kernel<<<static_cast<unsigned int>(batch_size), kThreads, 0, stream>>>(
        static_cast<int64_t*>(output.data_ptr()),
        static_cast<const int64_t*>(input_ids.data_ptr()),
        static_cast<const int8_t*>(level_lut.data_ptr()),
        static_cast<int32_t*>(err.data_ptr()),
        level_lut.size(0),
        start_token_id,
        num_latents,
        num_quantizers,
        time_major,
        batch_size,
        seq_len);
    sync_check_cuda_error();
}



}  // namespace wallx_cuda_qwen35_rvq_position_ids
