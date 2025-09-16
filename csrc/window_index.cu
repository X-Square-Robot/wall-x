#include <cuda_runtime.h>
#include <torch/extension.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <algorithm>

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
    int *cu_window_seqlens, // [total_windows + 1]
    int total_windows,
    int spatial_merge_unit)
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

std::tuple<torch::Tensor, torch::Tensor> get_window_index_cuda(
    torch::Tensor grid_thw,
    int spatial_merge_size,
    int vit_merger_window_size,
    int patch_size,
    int spatial_merge_unit)
{
    TORCH_CHECK(grid_thw.is_cuda(), "grid_thw must be a CUDA tensor");
    TORCH_CHECK(grid_thw.dim() == 2 && grid_thw.size(1) == 3);
    TORCH_CHECK(grid_thw.dtype() == torch::kInt32);

    int num_grids = grid_thw.size(0);
    if (num_grids == 0)
    {
        return std::make_tuple(
            torch::empty({0}, grid_thw.options()),
            torch::zeros({1}, grid_thw.options()));
    }

    const int *d_grid_thw = grid_thw.data_ptr<int>();
    auto options = grid_thw.options();

    auto grid_thw_cpu = grid_thw.cpu();
    int max_grid_t = 0;
    for (int i = 0; i < num_grids; i++)
    {
        max_grid_t = std::max(max_grid_t, grid_thw_cpu[i][0].item<int>());
    }

    int *d_grid_info;
    int *d_global_totals;
    int *d_window_counts;

    cudaMalloc(&d_grid_info, num_grids * 6 * sizeof(int));
    cudaMalloc(&d_global_totals, 2 * sizeof(int));
    cudaMemset(d_global_totals, 0, 2 * sizeof(int));

    int threads1 = 256;
    int blocks1 = (num_grids + threads1 - 1) / threads1;
    compute_metadata<<<blocks1, threads1>>>(
        d_grid_thw, d_grid_info, d_global_totals,
        num_grids, spatial_merge_size, vit_merger_window_size);

    int h_totals[2];
    cudaMemcpy(h_totals, d_global_totals, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    int total_elements = h_totals[0];
    int total_windows = h_totals[1];

    if (total_elements == 0 || total_windows == 0)
    {
        cudaFree(d_grid_info);
        cudaFree(d_global_totals);
        return std::make_tuple(
            torch::empty({0}, options),
            torch::zeros({1}, options));
    }

    torch::Tensor window_indices = torch::empty({total_elements}, options);
    torch::Tensor cu_window_seqlens = torch::empty({total_windows + 1}, options);

    int *d_window_indices = window_indices.data_ptr<int>();
    int *d_cu_window_seqlens = cu_window_seqlens.data_ptr<int>();

    cudaMalloc(&d_window_counts, total_windows * sizeof(int));

    dim3 blocks2(max_grid_t, num_grids);
    dim3 threads2(256);

    compute_window_counts<<<blocks2, threads2>>>(
        d_grid_thw, d_grid_info, d_window_counts,
        vit_merger_window_size, spatial_merge_unit, num_grids);

    int threads4 = 256;
    int blocks4 = (total_windows + threads4 - 1) / threads4;
    compute_cu_window_seqlens<<<blocks4, threads4>>>(
        d_window_counts, d_cu_window_seqlens, total_windows, spatial_merge_unit);

    generate_window_indices<<<blocks2, threads2>>>(
        d_grid_thw, d_grid_info, d_cu_window_seqlens, d_window_indices,
        vit_merger_window_size, spatial_merge_unit, num_grids);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA error: ", cudaGetErrorString(error));

    cudaFree(d_grid_info);
    cudaFree(d_global_totals);
    cudaFree(d_window_counts);

    return std::make_tuple(window_indices, cu_window_seqlens);
}
