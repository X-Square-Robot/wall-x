#include "tensor_view.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "../common/cuda_utils.h"

namespace wallx_cuda_qwen35_get_rope_index {

constexpr int kCountThreadsPerBatch = 256;
constexpr int kComputeThreadsPerBatchShort = 512;
constexpr int kComputeThreadsPerBatchLong = 640;

static int compute_threads_per_batch(int64_t seq_len) {
    return seq_len >= 2048 ? kComputeThreadsPerBatchLong
                           : kComputeThreadsPerBatchShort;
}

template <typename T>
__device__ __forceinline__ int64_t read_i64(const T* ptr, int64_t index) {
    return static_cast<int64_t>(ptr[index]);
}

__device__ __forceinline__ int64_t max_i64(int64_t lhs, int64_t rhs) {
    return lhs > rhs ? lhs : rhs;
}

__device__ __forceinline__ int64_t min_i64(int64_t lhs, int64_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

__device__ __forceinline__ int max_i32(int lhs, int rhs) {
    return lhs > rhs ? lhs : rhs;
}

template <typename MaskT, bool HasMask>
__device__ __forceinline__ bool is_valid_token(
    const MaskT* mask,
    int64_t index) {
    if constexpr (!HasMask) {
        return true;
    }
    return static_cast<int64_t>(mask[index]) != 0;
}

template <typename TokenTypeT, typename MaskT, bool HasMask>
__global__ void count_modality_groups_kernel(
    const TokenTypeT* __restrict__ mm_token_type_ids,
    const MaskT* __restrict__ attention_mask,
    int64_t* __restrict__ image_counts,
    int64_t* __restrict__ video_counts,
    int64_t* __restrict__ valid_lens,
    int64_t* __restrict__ mask_holes,
    int64_t batch_size,
    int64_t seq_len) {
    const int64_t batch = blockIdx.x;
    if (batch >= batch_size) {
        return;
    }

    const int64_t base = batch * seq_len;
    int64_t thread_image_count = 0;
    int64_t thread_video_count = 0;
    int64_t thread_last_valid = -1;
    int64_t thread_first_invalid = seq_len;

    for (int64_t seq = threadIdx.x; seq < seq_len; seq += blockDim.x) {
        const int64_t index = base + seq;
        if (!is_valid_token<MaskT, HasMask>(attention_mask, index)) {
            if constexpr (HasMask) {
                thread_first_invalid = min_i64(thread_first_invalid, seq);
            }
            continue;
        }

        if constexpr (HasMask) {
            thread_last_valid = max_i64(thread_last_valid, seq);
        }
        const int64_t token_type = read_i64(mm_token_type_ids, index);
        int64_t prev = seq - 1;
        if constexpr (HasMask) {
            while (prev >= 0 &&
                   !is_valid_token<MaskT, HasMask>(attention_mask, base + prev)) {
                --prev;
            }
        }
        const bool is_group_start =
            prev < 0 || read_i64(mm_token_type_ids, base + prev) != token_type;
        if (is_group_start) {
            if (token_type == 1) {
                ++thread_image_count;
            } else if (token_type == 2) {
                ++thread_video_count;
            }
        }
    }

    __shared__ int64_t image_shared[kCountThreadsPerBatch];
    __shared__ int64_t video_shared[kCountThreadsPerBatch];
    __shared__ int64_t last_valid_shared[kCountThreadsPerBatch];
    __shared__ int64_t first_invalid_shared[kCountThreadsPerBatch];
    image_shared[threadIdx.x] = thread_image_count;
    video_shared[threadIdx.x] = thread_video_count;
    last_valid_shared[threadIdx.x] = thread_last_valid;
    first_invalid_shared[threadIdx.x] = thread_first_invalid;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            image_shared[threadIdx.x] += image_shared[threadIdx.x + stride];
            video_shared[threadIdx.x] += video_shared[threadIdx.x + stride];
            if constexpr (HasMask) {
                last_valid_shared[threadIdx.x] = max_i64(
                    last_valid_shared[threadIdx.x],
                    last_valid_shared[threadIdx.x + stride]);
                first_invalid_shared[threadIdx.x] = min_i64(
                    first_invalid_shared[threadIdx.x],
                    first_invalid_shared[threadIdx.x + stride]);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        image_counts[batch] = image_shared[0];
        video_counts[batch] = video_shared[0];
        if constexpr (HasMask) {
            valid_lens[batch] = last_valid_shared[0] + 1;
            mask_holes[batch] =
                first_invalid_shared[0] <= last_valid_shared[0] ? 1 : 0;
        } else {
            valid_lens[batch] = seq_len;
            mask_holes[batch] = 0;
        }
    }
}

__device__ __forceinline__ void write_position(
    int64_t* __restrict__ position_ids,
    int64_t batch,
    int64_t seq,
    int64_t batch_size,
    int64_t seq_len,
    int64_t temporal,
    int64_t height,
    int64_t width) {
    position_ids[(0 * batch_size + batch) * seq_len + seq] = temporal;
    position_ids[(1 * batch_size + batch) * seq_len + seq] = height;
    position_ids[(2 * batch_size + batch) * seq_len + seq] = width;
}

template <typename MaskT, bool HasMask>
__device__ __forceinline__ int64_t local_valid_index(
    const MaskT* __restrict__ attention_mask,
    int64_t base,
    int64_t group_start,
    int64_t seq,
    bool group_has_mask_gap) {
    if constexpr (!HasMask) {
        return seq - group_start;
    }
    if (!group_has_mask_gap) {
        return seq - group_start;
    }

    int64_t local_index = 0;
    for (int64_t prev = group_start; prev < seq; ++prev) {
        if (is_valid_token<MaskT, HasMask>(attention_mask, base + prev)) {
            ++local_index;
        }
    }
    return local_index;
}

template <typename TokenTypeT, typename MaskT, bool HasMask>
__global__ void compute_qwen35_positions_kernel(
    const TokenTypeT* __restrict__ mm_token_type_ids,
    const MaskT* __restrict__ attention_mask,
    const int64_t* __restrict__ image_grid_thw,
    const int64_t* __restrict__ video_grid_thw,
    const int64_t* __restrict__ image_counts,
    const int64_t* __restrict__ video_counts,
    const int64_t* __restrict__ valid_lens,
    const int64_t* __restrict__ mask_holes,
    int64_t* __restrict__ position_ids,
    int64_t* __restrict__ mrope_deltas,
    int64_t batch_size,
    int64_t seq_len,
    int spatial_merge_size) {
    const int64_t batch = blockIdx.x;
    if (batch >= batch_size) {
        return;
    }

    const int64_t base = batch * seq_len;
    bool compact_mask = false;
    int64_t scan_limit = seq_len;
    if constexpr (HasMask) {
        const int64_t valid_len = valid_lens[batch];
        compact_mask = mask_holes[batch] == 0;
        scan_limit = compact_mask ? valid_len : seq_len;

        if (compact_mask) {
            for (int64_t seq = valid_len + threadIdx.x; seq < seq_len;
                 seq += blockDim.x) {
                write_position(
                    position_ids,
                    batch,
                    seq,
                    batch_size,
                    seq_len,
                    0,
                    0,
                    0);
            }
        } else {
            for (int64_t seq = threadIdx.x; seq < seq_len; seq += blockDim.x) {
                if (!is_valid_token<MaskT, HasMask>(
                        attention_mask, base + seq)) {
                    write_position(
                        position_ids,
                        batch,
                        seq,
                        batch_size,
                        seq_len,
                        0,
                        0,
                        0);
                }
            }
        }
        __syncthreads();
    }

    __shared__ int64_t sh_seq;
    __shared__ int64_t sh_current_pos;
    __shared__ int64_t sh_valid_count;
    __shared__ int64_t sh_max_position;
    __shared__ int64_t sh_image_index;
    __shared__ int64_t sh_video_index;
    __shared__ int64_t sh_modality_type;
    __shared__ int64_t sh_group_start;
    __shared__ int64_t sh_group_end;
    __shared__ int64_t sh_group_len;
    __shared__ int64_t sh_grid_t;
    __shared__ int64_t sh_grid_h;
    __shared__ int64_t sh_grid_w;
    __shared__ int64_t sh_llm_grid_t;
    __shared__ int64_t sh_llm_grid_h;
    __shared__ int64_t sh_llm_grid_w;
    __shared__ int64_t sh_group_pos_advance;
    __shared__ int64_t sh_group_max_position;
    __shared__ int sh_group_kind;
    __shared__ bool sh_group_has_mask_gap;
    __shared__ bool sh_done;

    if (threadIdx.x == 0) {
        int64_t image_index = 0;
        int64_t video_index = 0;
        for (int64_t prev_batch = 0; prev_batch < batch; ++prev_batch) {
            image_index += image_counts[prev_batch];
            video_index += video_counts[prev_batch];
        }

        sh_seq = 0;
        sh_current_pos = 0;
        sh_valid_count = 0;
        sh_max_position = -1;
        sh_image_index = image_index;
        sh_video_index = video_index;
        sh_done = false;
    }
    __syncthreads();

    while (true) {
        if (threadIdx.x == 0) {
            int64_t seq = sh_seq;
            while (seq < scan_limit) {
                if constexpr (HasMask) {
                    if (!compact_mask &&
                        !is_valid_token<MaskT, HasMask>(
                            attention_mask, base + seq)) {
                        ++seq;
                        continue;
                    }
                }
                break;
            }
            if (seq >= scan_limit) {
                sh_done = true;
            } else {
                const int64_t modality_type = read_i64(mm_token_type_ids, base + seq);
                const int64_t group_start = seq;
                int64_t scan = group_start;
                int64_t group_len = 0;
                bool group_has_mask_gap = false;

                while (scan < scan_limit) {
                    const int64_t index = base + scan;
                    if constexpr (HasMask) {
                        if (!compact_mask &&
                            !is_valid_token<MaskT, HasMask>(
                                attention_mask, index)) {
                            group_has_mask_gap = true;
                            ++scan;
                            continue;
                        }
                    }
                    if (read_i64(mm_token_type_ids, index) != modality_type) {
                        break;
                    }
                    ++group_len;
                    ++scan;
                }

                sh_done = false;
                sh_modality_type = modality_type;
                sh_group_start = group_start;
                sh_group_end = scan;
                sh_group_len = group_len;
                sh_group_has_mask_gap = group_has_mask_gap;
                sh_group_kind = 0;
                sh_group_pos_advance = group_len;
                sh_group_max_position = sh_current_pos + group_len - 1;

                if (modality_type == 1 || modality_type == 2) {
                    const int64_t* grid = nullptr;
                    if (modality_type == 1) {
                        grid = image_grid_thw + sh_image_index * 3;
                        ++sh_image_index;
                    } else {
                        grid = video_grid_thw + sh_video_index * 3;
                        ++sh_video_index;
                    }

                    sh_grid_t = grid[0];
                    sh_grid_h = grid[1];
                    sh_grid_w = grid[2];
                    sh_llm_grid_t = sh_grid_t;
                    sh_llm_grid_h = sh_grid_h / spatial_merge_size;
                    sh_llm_grid_w = sh_grid_w / spatial_merge_size;
                    sh_group_kind = 1;
                    sh_group_pos_advance =
                        max_i64(sh_grid_h, sh_grid_w) / spatial_merge_size;
                    sh_group_max_position =
                        sh_current_pos +
                        max_i64(sh_llm_grid_h, sh_llm_grid_w) - 1;
                }

                sh_seq = scan;
            }
        }
        __syncthreads();

        if (sh_done) {
            break;
        }

        for (int64_t scan = sh_group_start + threadIdx.x;
             scan < sh_group_end;
             scan += blockDim.x) {
            const int64_t index = base + scan;
            if constexpr (HasMask) {
                if (!compact_mask &&
                    !is_valid_token<MaskT, HasMask>(attention_mask, index)) {
                    continue;
                }
            }
            if (read_i64(mm_token_type_ids, index) != sh_modality_type) {
                continue;
            }

            int local_index = static_cast<int>(scan - sh_group_start);
            if constexpr (HasMask) {
                if (!compact_mask && sh_group_has_mask_gap) {
                    local_index = static_cast<int>(
                        local_valid_index<MaskT, HasMask>(
                            attention_mask,
                            base,
                            sh_group_start,
                            scan,
                            sh_group_has_mask_gap));
                }
            }
            if (sh_group_kind == 1) {
                const int llm_grid_t =
                    max_i32(static_cast<int>(1), static_cast<int>(sh_llm_grid_t));
                const int llm_grid_h =
                    max_i32(static_cast<int>(1), static_cast<int>(sh_llm_grid_h));
                const int llm_grid_w =
                    max_i32(static_cast<int>(1), static_cast<int>(sh_llm_grid_w));
                const int64_t width = sh_current_pos + (local_index % llm_grid_w);
                const int64_t height =
                    sh_current_pos +
                    ((local_index / (llm_grid_w * llm_grid_t)) % llm_grid_h);
                const int64_t temporal = sh_current_pos;
                write_position(
                    position_ids,
                    batch,
                    scan,
                    batch_size,
                    seq_len,
                    temporal,
                    height,
                    width);
            } else {
                const int64_t pos = sh_current_pos + local_index;
                write_position(
                    position_ids, batch, scan, batch_size, seq_len, pos, pos, pos);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            sh_current_pos += sh_group_pos_advance;
            sh_valid_count += sh_group_len;
            sh_max_position = max_i64(sh_max_position, sh_group_max_position);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        mrope_deltas[batch] = sh_max_position + 1 - sh_valid_count;
    }
}

int64_t Qwen35GetRopeIndexGetWorkspace(wallx_cuda::TensorView input_ids) {
    ASSERT_CHECK(input_ids.ndim() == 2);
    const int64_t batch_size = input_ids.size(0);
    const int64_t bytes = 4 * batch_size * static_cast<int64_t>(sizeof(int64_t));
    return std::max(static_cast<int64_t>(1), bytes);
}

static void check_same_device(
    wallx_cuda::TensorView tensor,
    const char* name,
    int device_type,
    int device_id) {
    if (tensor.device().device_type != device_type ||
        tensor.device().device_id != device_id) {
        throw std::runtime_error(std::string(name) + ": expected tensor on the same device as input_ids");
    }
}

static void check_grid_tensor(
    const wallx_cuda::OptionalTensorView& maybe_tensor,
    const char* name,
    int device_type,
    int device_id) {
    if (!maybe_tensor.has_value()) {
        return;
    }

    const wallx_cuda::TensorView tensor = maybe_tensor.value();
    if (tensor.ndim() != 2 || tensor.size(1) != 3) {
        throw std::runtime_error(std::string(name) + ": expected shape [N, 3]");
    }
    if (!tensor.IsContiguous()) {
        throw std::runtime_error(std::string(name) + ": expected contiguous tensor");
    }
    if (!(tensor.dtype().code == kDLInt && tensor.dtype().bits == 64)) {
        throw std::runtime_error(std::string(name) + ": expected int64 tensor");
    }
    check_same_device(tensor, name, device_type, device_id);
}

template <typename TokenTypeT, typename MaskT, bool HasMask>
static void launch_kernels(
    wallx_cuda::TensorView mm_token_type_ids,
    const wallx_cuda::OptionalTensorView& attention_mask,
    const int64_t* image_grid_thw,
    const int64_t* video_grid_thw,
    int64_t* image_counts,
    int64_t* video_counts,
    int64_t* valid_lens,
    int64_t* mask_holes,
    wallx_cuda::TensorView position_ids,
    wallx_cuda::TensorView mrope_deltas,
    int64_t batch_size,
    int64_t seq_len,
    int spatial_merge_size,
    cudaStream_t stream) {
    const auto* token_types =
        static_cast<const TokenTypeT*>(mm_token_type_ids.data_ptr());
    const MaskT* mask = nullptr;
    if constexpr (HasMask) {
        mask = static_cast<const MaskT*>(attention_mask.value().data_ptr());
    }

    count_modality_groups_kernel<TokenTypeT, MaskT, HasMask>
        <<<static_cast<unsigned int>(batch_size), kCountThreadsPerBatch, 0, stream>>>(
            token_types,
            mask,
            image_counts,
            video_counts,
            valid_lens,
            mask_holes,
            batch_size,
            seq_len);
    compute_qwen35_positions_kernel<TokenTypeT, MaskT, HasMask>
        <<<static_cast<unsigned int>(batch_size),
           compute_threads_per_batch(seq_len),
           0,
           stream>>>(
            token_types,
            mask,
            image_grid_thw,
            video_grid_thw,
            image_counts,
            video_counts,
            valid_lens,
            mask_holes,
            static_cast<int64_t*>(position_ids.data_ptr()),
            static_cast<int64_t*>(mrope_deltas.data_ptr()),
            batch_size,
            seq_len,
            spatial_merge_size);
}

template <typename TokenTypeT>
static void dispatch_mask_dtype(
    wallx_cuda::TensorView mm_token_type_ids,
    const wallx_cuda::OptionalTensorView& attention_mask,
    const int64_t* image_grid_thw,
    const int64_t* video_grid_thw,
    int64_t* image_counts,
    int64_t* video_counts,
    int64_t* valid_lens,
    int64_t* mask_holes,
    wallx_cuda::TensorView position_ids,
    wallx_cuda::TensorView mrope_deltas,
    int64_t batch_size,
    int64_t seq_len,
    int spatial_merge_size,
    cudaStream_t stream) {
    if (!attention_mask.has_value()) {
        launch_kernels<TokenTypeT, int64_t, false>(
            mm_token_type_ids,
            attention_mask,
            image_grid_thw,
            video_grid_thw,
            image_counts,
            video_counts,
            valid_lens,
            mask_holes,
            position_ids,
            mrope_deltas,
            batch_size,
            seq_len,
            spatial_merge_size,
            stream);
        return;
    }

    const wallx_cuda::TensorView mask = attention_mask.value();
    if (mask.dtype().code == kDLInt && mask.dtype().bits == 64) {
        launch_kernels<TokenTypeT, int64_t, true>(
            mm_token_type_ids, attention_mask, image_grid_thw, video_grid_thw,
            image_counts, video_counts, valid_lens, mask_holes, position_ids,
            mrope_deltas, batch_size, seq_len, spatial_merge_size, stream);
    } else if (mask.dtype().code == kDLInt && mask.dtype().bits == 32) {
        launch_kernels<TokenTypeT, int32_t, true>(
            mm_token_type_ids, attention_mask, image_grid_thw, video_grid_thw,
            image_counts, video_counts, valid_lens, mask_holes, position_ids,
            mrope_deltas, batch_size, seq_len, spatial_merge_size, stream);
    } else if (mask.dtype().code == kDLUInt && mask.dtype().bits == 8) {
        launch_kernels<TokenTypeT, uint8_t, true>(
            mm_token_type_ids, attention_mask, image_grid_thw, video_grid_thw,
            image_counts, video_counts, valid_lens, mask_holes, position_ids,
            mrope_deltas, batch_size, seq_len, spatial_merge_size, stream);
    } else if (mask.dtype().code == kDLBool) {
        launch_kernels<TokenTypeT, bool, true>(
            mm_token_type_ids, attention_mask, image_grid_thw, video_grid_thw,
            image_counts, video_counts, valid_lens, mask_holes, position_ids,
            mrope_deltas, batch_size, seq_len, spatial_merge_size, stream);
    } else {
        throw std::runtime_error("qwen35_get_rope_index: unsupported attention_mask dtype");
    }
}

void Qwen35GetRopeIndex(
    wallx_cuda::TensorView input_ids,
    wallx_cuda::TensorView mm_token_type_ids,
    const wallx_cuda::OptionalTensorView image_grid_thw,
    const wallx_cuda::OptionalTensorView video_grid_thw,
    const wallx_cuda::OptionalTensorView attention_mask,
    wallx_cuda::TensorView position_ids,
    wallx_cuda::TensorView mrope_deltas,
    wallx_cuda::TensorView workspace,
    int spatial_merge_size) {
    ASSERT_CHECK(input_ids.ndim() == 2);
    ASSERT_CHECK(input_ids.IsContiguous());
    ASSERT_CHECK(mm_token_type_ids.ndim() == 2);
    ASSERT_CHECK(mm_token_type_ids.IsContiguous());
    ASSERT_CHECK(input_ids.size(0) == mm_token_type_ids.size(0));
    ASSERT_CHECK(input_ids.size(1) == mm_token_type_ids.size(1));
    ASSERT_CHECK(position_ids.ndim() == 3);
    ASSERT_CHECK(position_ids.size(0) == 3);
    ASSERT_CHECK(position_ids.size(1) == input_ids.size(0));
    ASSERT_CHECK(position_ids.size(2) == input_ids.size(1));
    ASSERT_CHECK(position_ids.IsContiguous());
    ASSERT_CHECK(mrope_deltas.ndim() == 2);
    ASSERT_CHECK(mrope_deltas.size(0) == input_ids.size(0));
    ASSERT_CHECK(mrope_deltas.size(1) == 1);
    ASSERT_CHECK(mrope_deltas.IsContiguous());
    ASSERT_CHECK(workspace.IsContiguous());
    ASSERT_CHECK(spatial_merge_size > 0);
    ASSERT_CHECK(position_ids.dtype().code == kDLInt && position_ids.dtype().bits == 64);
    ASSERT_CHECK(mrope_deltas.dtype().code == kDLInt && mrope_deltas.dtype().bits == 64);

    const int device_type = input_ids.device().device_type;
    const int device_id = input_ids.device().device_id;
    check_same_device(mm_token_type_ids, "mm_token_type_ids", device_type, device_id);
    check_same_device(position_ids, "position_ids", device_type, device_id);
    check_same_device(mrope_deltas, "mrope_deltas", device_type, device_id);
    check_same_device(workspace, "workspace", device_type, device_id);
    check_grid_tensor(image_grid_thw, "image_grid_thw", device_type, device_id);
    check_grid_tensor(video_grid_thw, "video_grid_thw", device_type, device_id);
    if (attention_mask.has_value()) {
        const wallx_cuda::TensorView mask = attention_mask.value();
        if (mask.ndim() != 2 || mask.size(0) != input_ids.size(0) ||
            mask.size(1) != input_ids.size(1)) {
            throw std::runtime_error("attention_mask: expected shape [B, S]");
        }
        if (!mask.IsContiguous()) {
            throw std::runtime_error("attention_mask: expected contiguous tensor");
        }
        check_same_device(mask, "attention_mask", device_type, device_id);
    }

    const int64_t batch_size = input_ids.size(0);
    const int64_t seq_len = input_ids.size(1);
    void* work_ptr = workspace.data_ptr();
    auto* image_counts = static_cast<int64_t*>(work_ptr);
    auto* video_counts = image_counts + batch_size;
    auto* valid_lens = video_counts + batch_size;
    auto* mask_holes = valid_lens + batch_size;
    const int64_t* image_grid_ptr = image_grid_thw.has_value()
        ? static_cast<const int64_t*>(image_grid_thw.value().data_ptr())
        : nullptr;
    const int64_t* video_grid_ptr = video_grid_thw.has_value()
        ? static_cast<const int64_t*>(video_grid_thw.value().data_ptr())
        : nullptr;

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(position_ids.device().device_type, position_ids.device().device_id));

    if (mm_token_type_ids.dtype().code == kDLInt && mm_token_type_ids.dtype().bits == 64) {
        dispatch_mask_dtype<int64_t>(
            mm_token_type_ids, attention_mask, image_grid_ptr, video_grid_ptr,
            image_counts, video_counts, valid_lens, mask_holes, position_ids,
            mrope_deltas, batch_size, seq_len, spatial_merge_size, stream);
    } else if (mm_token_type_ids.dtype().code == kDLInt &&
               mm_token_type_ids.dtype().bits == 32) {
        dispatch_mask_dtype<int32_t>(
            mm_token_type_ids, attention_mask, image_grid_ptr, video_grid_ptr,
            image_counts, video_counts, valid_lens, mask_holes, position_ids,
            mrope_deltas, batch_size, seq_len, spatial_merge_size, stream);
    } else {
        throw std::runtime_error("qwen35_get_rope_index: unsupported mm_token_type_ids dtype");
    }

    sync_check_cuda_error();
}

}  // namespace wallx_cuda_qwen35_get_rope_index



