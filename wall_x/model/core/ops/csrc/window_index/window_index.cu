#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cassert>
#include "../common/cuda_utils.h"

namespace wallx_cuda_window_index {

__global__ void compute_metadata(
    const int *grid_thw, // [num_grids, 3]
    int *grid_info,      // [num_grids, 6]: [grid_elements, grid_windows, llm_h, llm_w, num_windows_h, num_windows_w]
    int *global_totals,  // [total_elements, total_windows]
    int num_grids,
    int spatial_merge_size,
    int vit_merger_window_size)
{
    int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_idx >= num_grids)
        return;

    int grid_t = grid_thw[grid_idx * 3 + 0];
    int grid_h = grid_thw[grid_idx * 3 + 1];
    int grid_w = grid_thw[grid_idx * 3 + 2];

    int llm_h = grid_h / spatial_merge_size;
    int llm_w = grid_w / spatial_merge_size;

    int pad_h = (vit_merger_window_size - llm_h % vit_merger_window_size) % vit_merger_window_size;
    int pad_w = (vit_merger_window_size - llm_w % vit_merger_window_size) % vit_merger_window_size;

    int num_windows_h = (llm_h + pad_h) / vit_merger_window_size;
    int num_windows_w = (llm_w + pad_w) / vit_merger_window_size;

    int grid_elements = grid_t * llm_h * llm_w;
    int grid_windows = grid_t * num_windows_h * num_windows_w;

    grid_info[grid_idx * 6 + 0] = grid_elements;
    grid_info[grid_idx * 6 + 1] = grid_windows;
    grid_info[grid_idx * 6 + 2] = llm_h;
    grid_info[grid_idx * 6 + 3] = llm_w;
    grid_info[grid_idx * 6 + 4] = num_windows_h;
    grid_info[grid_idx * 6 + 5] = num_windows_w;

    atomicAdd(&global_totals[0], grid_elements);
    atomicAdd(&global_totals[1], grid_windows);
}

__global__ void compute_window_counts(
    const int *grid_thw,
    const int *grid_info,
    int *window_counts,
    int vit_merger_window_size,
    int spatial_merge_unit,
    int num_grids)
{
    int grid_idx = blockIdx.y;
    int t_idx = blockIdx.x;

    if (grid_idx >= num_grids)
        return;

    int grid_t = grid_thw[grid_idx * 3 + 0];
    if (t_idx >= grid_t)
        return;

    int llm_h = grid_info[grid_idx * 6 + 2];
    int llm_w = grid_info[grid_idx * 6 + 3];
    int num_windows_h = grid_info[grid_idx * 6 + 4];
    int num_windows_w = grid_info[grid_idx * 6 + 5];

    int window_base = 0;
    for (int g = 0; g < grid_idx; g++)
    {
        window_base += grid_info[g * 6 + 1];
    }

    int t_window_base = window_base + t_idx * num_windows_h * num_windows_w;

    int thread_id = threadIdx.x;
    int warp_id = thread_id / 32;
    int lane_id = thread_id % 32;

    int windows_per_t = num_windows_h * num_windows_w;
    int warps_per_block = blockDim.x / 32;

    if (lane_id == 0)
    {
        for (int window_idx = warp_id; window_idx < windows_per_t; window_idx += warps_per_block)
        {
            int win_h = window_idx / num_windows_w;
            int win_w = window_idx % num_windows_w;

            int start_h = win_h * vit_merger_window_size;
            int start_w = win_w * vit_merger_window_size;

            int valid_h = min(vit_merger_window_size, llm_h - start_h);
            int valid_w = min(vit_merger_window_size, llm_w - start_w);

            int valid_count = (valid_h > 0 && valid_w > 0) ? valid_h * valid_w : 0;

            window_counts[t_window_base + window_idx] = valid_count;
        }
    }
}

__global__ void compute_cu_window_seqlens(
    const int *window_counts,
    int *cu_window_seqlens,   // [total_windows + 1]
    int total_windows,
    int spatial_merge_unit
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        cu_window_seqlens[0] = 0;
    }

    if (tid < total_windows)
    {
        cu_window_seqlens[tid + 1] = window_counts[tid] * spatial_merge_unit;
    }

    __syncthreads();

    if (tid == 0)
    {
        for (int i = 1; i <= total_windows; i++)
        {
            cu_window_seqlens[i] += cu_window_seqlens[i - 1];
        }
    }
}

__global__ void generate_window_indices(
    const int *grid_thw,
    const int *grid_info,
    const int *cu_window_seqlens,
    int *window_indices,
    int vit_merger_window_size,
    int spatial_merge_unit,
    int num_grids)
{
    int grid_idx = blockIdx.y;
    int t_idx = blockIdx.x;

    if (grid_idx >= num_grids)
        return;

    int grid_t = grid_thw[grid_idx * 3 + 0];
    if (t_idx >= grid_t)
        return;

    int llm_h = grid_info[grid_idx * 6 + 2];
    int llm_w = grid_info[grid_idx * 6 + 3];
    int num_windows_h = grid_info[grid_idx * 6 + 4];
    int num_windows_w = grid_info[grid_idx * 6 + 5];

    int element_base = 0;
    for (int g = 0; g < grid_idx; g++)
    {
        element_base += grid_info[g * 6 + 0];
    }
    int t_element_base = element_base + t_idx * llm_h * llm_w;

    int window_base = 0;
    for (int g = 0; g < grid_idx; g++)
    {
        window_base += grid_info[g * 6 + 1];
    }
    int t_window_base = window_base + t_idx * num_windows_h * num_windows_w;

    int thread_id = threadIdx.x;
    int warp_id = thread_id / 32;
    int lane_id = thread_id % 32;

    int windows_per_t = num_windows_h * num_windows_w;
    int warps_per_block = blockDim.x / 32;

    for (int window_idx = warp_id; window_idx < windows_per_t; window_idx += warps_per_block)
    {
        int win_h = window_idx / num_windows_w;
        int win_w = window_idx % num_windows_w;

        int global_window_idx = t_window_base + window_idx;
        int output_offset = cu_window_seqlens[global_window_idx] / spatial_merge_unit;

        int start_h = win_h * vit_merger_window_size;
        int start_w = win_w * vit_merger_window_size;

        int valid_h = min(vit_merger_window_size, llm_h - start_h);
        int valid_w = min(vit_merger_window_size, llm_w - start_w);

        for (int elem_idx = lane_id; elem_idx < valid_h * valid_w; elem_idx += 32)
        {
            int local_h = elem_idx / valid_w;
            int local_w = elem_idx % valid_w;

            int abs_h = start_h + local_h;
            int abs_w = start_w + local_w;

            int value = t_element_base + abs_h * llm_w + abs_w;

            int base_offset = output_offset + elem_idx;
            window_indices[base_offset] = value;
        }
    }
}

void GetTotals(
    const at::Tensor& grid_thw,
    const at::Tensor& grid_info_tensor,
    const at::Tensor& global_totals_tensor,
    int spatial_merge_size,
    int vit_merger_window_size
) {
    ASSERT_CHECK(grid_thw.dim() == 2 && grid_thw.size(1) == 3);
    ASSERT_CHECK(grid_thw.scalar_type() == at::kInt);

    int num_grids = grid_thw.size(0);
    const int *d_grid_thw = static_cast<int32_t*>(grid_thw.data_ptr());

    int *d_grid_info = static_cast<int32_t*>(grid_info_tensor.data_ptr());
    int *d_global_totals = static_cast<int32_t*>(global_totals_tensor.data_ptr());

    int threads1 = 256;
    int blocks1 = (num_grids + threads1 - 1) / threads1;

    compute_metadata<<<blocks1, threads1>>>(
        d_grid_thw, d_grid_info, d_global_totals,
        num_grids, spatial_merge_size, vit_merger_window_size);

    sync_check_cuda_error();
}

void GetWindowIndex(
    const at::Tensor& grid_thw,
    const at::Tensor& grid_info_tensor,
    const at::Tensor& window_indices,
    const at::Tensor& cu_window_seqlens,
    const at::Tensor& window_counts_tensor,
    int max_grid_t,
    int spatial_merge_size,
    int vit_merger_window_size,
    int patch_size,
    int spatial_merge_unit) {

    int num_grids = grid_thw.size(0);

    dim3 blocks2(max_grid_t, num_grids);
    dim3 threads2(256);

    const int *d_grid_thw = static_cast<int32_t*>(grid_thw.data_ptr());

    int *d_grid_info = static_cast<int32_t*>(grid_info_tensor.data_ptr());

    int *d_window_indices = static_cast<int32_t*>(window_indices.data_ptr());
    int *d_cu_window_seqlens = static_cast<int32_t*>(cu_window_seqlens.data_ptr());

    int *d_window_counts = static_cast<int32_t*>(window_counts_tensor.data_ptr());

    compute_window_counts<<<blocks2, threads2>>>(
        d_grid_thw, d_grid_info, d_window_counts,
        vit_merger_window_size, spatial_merge_unit, num_grids);

    int total_windows = window_counts_tensor.size(0);

    int threads4 = 256;
    int blocks4 = (total_windows + threads4 - 1) / threads4;
    compute_cu_window_seqlens<<<blocks4, threads4>>>(
        d_window_counts, d_cu_window_seqlens, total_windows, spatial_merge_unit);

    generate_window_indices<<<blocks2, threads2>>>(
        d_grid_thw, d_grid_info, d_cu_window_seqlens, d_window_indices,
        vit_merger_window_size, spatial_merge_unit, num_grids);

    sync_check_cuda_error();
}

// Removed TVM FFI export (see pybind11 registration)
// Removed TVM FFI export (see pybind11 registration)

} // wallx_cuda_window_index
