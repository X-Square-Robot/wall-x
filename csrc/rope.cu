#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

// Type traits for CUDA types
template <typename T>
struct CudaTypeTraits
{
    static constexpr bool is_supported = false;
};

template <>
struct CudaTypeTraits<float>
{
    static constexpr bool is_supported = true;
    using type = float;
    static constexpr const char *name = "float";
};

template <>
struct CudaTypeTraits<half>
{
    static constexpr bool is_supported = true;
    using type = half;
    static constexpr const char *name = "half";
};

template <>
struct CudaTypeTraits<__nv_bfloat16>
{
    static constexpr bool is_supported = true;
    using type = __nv_bfloat16;
    static constexpr const char *name = "bfloat16";
};

// Optimized forward kernel template
template <typename T>
__global__ void multimodal_rope_forward_kernel(
    const T *__restrict__ q,                       // [batch, q_heads, seq_len, head_dim]
    const T *__restrict__ k,                       // [batch, kv_heads, seq_len, head_dim]
    const T *__restrict__ cos,                     // [3, batch, seq_len, head_dim]
    const T *__restrict__ sin,                     // [3, batch, seq_len, head_dim]
    T *__restrict__ q_out,                         // [batch, q_heads, seq_len, head_dim]
    T *__restrict__ k_out,                         // [batch, kv_heads, seq_len, head_dim]
    const int *__restrict__ mrope_section_doubled, // [32, 48, 48]
    int batch_size,
    int q_heads,
    int kv_heads,
    int seq_len,
    int head_dim)
{
    static_assert(CudaTypeTraits<T>::is_supported, "Unsupported data type");

    // Block organization: batch_size * (q_heads + kv_heads) blocks
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_paral_size = gridDim.z;
    int seq_paral_idx = blockIdx.z;

    bool is_q_head = head_idx < q_heads;
    int actual_head_idx = is_q_head ? head_idx : (head_idx - q_heads);
    int total_heads = is_q_head ? q_heads : kv_heads;

    // Each warp processes one token (seq position)
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warps_per_block = blockDim.x / 32;

    // Process tokens in batches across warps

    for (int seq_base = warp_id + seq_paral_idx * warps_per_block; seq_base < seq_len; seq_base += seq_paral_size * warps_per_block)
    {
        int seq_idx = seq_base;
        if (seq_idx >= seq_len)
            break;

        // Each lane processes multiple dimensions (head_dim=128, 32 lanes)
        constexpr int dims_per_lane = 4;
        for (int dim_batch = 0; dim_batch < (head_dim + 31) / 32; ++dim_batch)
        {
            int dim_start = dim_batch * 32 + lane_id;
            if (dim_start >= head_dim)
                break;

#pragma unroll
            for (int dim_offset = 0; dim_offset < dims_per_lane; ++dim_offset)
            {
                int dim_idx = dim_start + dim_offset * 32;
                if (dim_idx >= head_dim)
                    break;

                // Determine which section this dimension belongs to
                int section_idx;
                int cos_sin_d = dim_idx;
                if (dim_idx < mrope_section_doubled[0])
                {
                    section_idx = 0;
                }
                else if (dim_idx < mrope_section_doubled[0] + mrope_section_doubled[1])
                {
                    section_idx = 1;
                }
                else
                {
                    section_idx = 2;
                }

                // Load cos/sin values (coalesced access)
                int cos_sin_idx = section_idx * batch_size * seq_len * head_dim +
                                  batch_idx * seq_len * head_dim +
                                  seq_idx * head_dim + cos_sin_d;

                T cos_val = cos[cos_sin_idx];
                T sin_val = sin[cos_sin_idx];

                // Calculate tensor indices
                int tensor_idx = batch_idx * total_heads * seq_len * head_dim +
                                 actual_head_idx * seq_len * head_dim +
                                 seq_idx * head_dim + dim_idx;

                // Get input value
                T input_val = is_q_head ? q[tensor_idx] : k[tensor_idx];

                // Calculate rotate_half value
                int half_dim = head_dim / 2;
                int rotate_dim = (dim_idx < half_dim) ? (dim_idx + half_dim) : (dim_idx - half_dim);
                int rotate_tensor_idx = batch_idx * total_heads * seq_len * head_dim +
                                        actual_head_idx * seq_len * head_dim +
                                        seq_idx * head_dim + rotate_dim;

                T rotate_val = is_q_head ? q[rotate_tensor_idx] : k[rotate_tensor_idx];
                if (dim_idx < half_dim)
                {
                    if constexpr (std::is_same_v<T, half>)
                    {
                        rotate_val = __hneg(rotate_val);
                    }
                    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
                    {
#if __CUDA_ARCH__ >= 800                                 // BFloat16 support requires Ampere or newer
                        rotate_val = __hneg(rotate_val); // __hneg works for bfloat16 in newer CUDA
#else
                        rotate_val = __float2bfloat16(-__bfloat162float(rotate_val));
#endif
                    }
                    else
                    {
                        rotate_val = -rotate_val;
                    }
                }

                // Apply RoPE: output = input * cos + rotate_half(input) * sin
                T output_val = input_val * cos_val + rotate_val * sin_val;

                // Store result
                if (is_q_head)
                {
                    q_out[tensor_idx] = output_val;
                }
                else
                {
                    k_out[tensor_idx] = output_val;
                }
            }
        }
    }
}

// Optimized backward kernel template
template <typename T>
__global__ void multimodal_rope_backward_kernel(
    const T *__restrict__ grad_q_out, // [batch, q_heads, seq_len, head_dim]
    const T *__restrict__ grad_k_out, // [batch, kv_heads, seq_len, head_dim]
    const T *__restrict__ q,          // [batch, q_heads, seq_len, head_dim]
    const T *__restrict__ k,          // [batch, kv_heads, seq_len, head_dim]
    const T *__restrict__ cos,        // [3, batch, seq_len, head_dim]
    const T *__restrict__ sin,        // [3, batch, seq_len, head_dim]
    T *__restrict__ grad_q,           // [batch, q_heads, seq_len, head_dim]
    T *__restrict__ grad_k,           // [batch, kv_heads, seq_len, head_dim]
    const int *__restrict__ mrope_section_doubled,
    int batch_size,
    int q_heads,
    int kv_heads,
    int seq_len,
    int head_dim)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_paral_size = gridDim.z;
    int seq_paral_idx = blockIdx.z;

    bool is_q_head = head_idx < q_heads;
    int actual_head_idx = is_q_head ? head_idx : (head_idx - q_heads);
    int total_heads = is_q_head ? q_heads : kv_heads;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warps_per_block = blockDim.x / 32;

    // Process tokens
    for (int seq_base = warp_id + seq_paral_idx * warps_per_block; seq_base < seq_len; seq_base += seq_paral_size * warps_per_block)
    {
        int seq_idx = seq_base;
        if (seq_idx >= seq_len)
            break;

        // Process dimensions in chunks - much simpler now!
        constexpr int dims_per_lane = 4;
        for (int dim_batch = 0; dim_batch < (head_dim + 31) / 32; ++dim_batch)
        {
            int dim_start = dim_batch * 32 + lane_id;
            if (dim_start >= head_dim)
                break;

#pragma unroll
            for (int dim_offset = 0; dim_offset < dims_per_lane; ++dim_offset)
            {
                int dim_idx = dim_start + dim_offset * 32;
                if (dim_idx >= head_dim)
                    break;

                // Get section info for cos/sin reconstruction
                int section_idx;
                int cos_sin_d = dim_idx;
                if (dim_idx < mrope_section_doubled[0])
                {
                    section_idx = 0;
                }
                else if (dim_idx < mrope_section_doubled[0] + mrope_section_doubled[1])
                {
                    section_idx = 1;
                }
                else
                {
                    section_idx = 2;
                }

                // Global cos/sin index
                int cos_sin_idx = section_idx * batch_size * seq_len * head_dim +
                                  batch_idx * seq_len * head_dim +
                                  seq_idx * head_dim + cos_sin_d;

                // Tensor index for current position
                int tensor_idx = batch_idx * total_heads * seq_len * head_dim +
                                 actual_head_idx * seq_len * head_dim +
                                 seq_idx * head_dim + dim_idx;

                // Load values
                T cos_val = cos[cos_sin_idx];
                T sin_val = sin[cos_sin_idx];
                T grad_out_val = is_q_head ? grad_q_out[tensor_idx] : grad_k_out[tensor_idx];

                // Calculate paired dimension for rotate_half
                int half_dim = head_dim / 2;
                int rotate_dim = (dim_idx < half_dim) ? (dim_idx + half_dim) : (dim_idx - half_dim);
                int rotate_tensor_idx = batch_idx * total_heads * seq_len * head_dim +
                                        actual_head_idx * seq_len * head_dim +
                                        seq_idx * head_dim + rotate_dim;

                // Get gradient from the paired dimension
                T paired_grad_out = is_q_head ? grad_q_out[rotate_tensor_idx] : grad_k_out[rotate_tensor_idx];

                // Get paired sin value
                int paired_section_idx;
                int paired_cos_sin_d = rotate_dim;
                if (rotate_dim < mrope_section_doubled[0])
                {
                    paired_section_idx = 0;
                }
                else if (rotate_dim < mrope_section_doubled[0] + mrope_section_doubled[1])
                {
                    paired_section_idx = 1;
                }
                else
                {
                    paired_section_idx = 2;
                }

                int paired_cos_sin_idx = paired_section_idx * batch_size * seq_len * head_dim +
                                         batch_idx * seq_len * head_dim +
                                         seq_idx * head_dim + paired_cos_sin_d;
                T paired_sin_val = sin[paired_cos_sin_idx];

                // === Compute input gradients (the only thing we need!) ===

                // Direct term: grad_input = grad_out * cos
                T grad_input_direct = grad_out_val * cos_val;

                // Cross term from rotate_half
                T grad_input_from_rotate;
                if (dim_idx < half_dim)
                {
                    // First half: gets contribution from second half (positive)
                    grad_input_from_rotate = paired_grad_out * paired_sin_val;
                }
                else
                {
                    // Second half: gets contribution from first half (negative due to rotate_half)
                    grad_input_from_rotate = -paired_grad_out * paired_sin_val;
                }

                // Total input gradient
                T total_grad_input = grad_input_direct + grad_input_from_rotate;

                // Store input gradients (no atomic operations needed!)
                if (is_q_head)
                {
                    grad_q[tensor_idx] = total_grad_input;
                }
                else
                {
                    grad_k[tensor_idx] = total_grad_input;
                }

                // No cos/sin gradient computation - they're not learnable parameters!
            }
        }
    }
}

// Template-based host functions
template <typename T>
void launch_multimodal_rope_forward_template(
    const T *q, const T *k, const T *cos, const T *sin,
    T *q_out, T *k_out,
    const int *mrope_section_doubled,
    int batch_size, int q_heads, int kv_heads, int seq_len, int head_dim,
    cudaStream_t stream)
{
    static_assert(CudaTypeTraits<T>::is_supported, "Unsupported data type for multimodal RoPE");

    // Grid: batch_size x (q_heads + kv_heads)
    dim3 grid(batch_size, q_heads + kv_heads, 8);

    // Block: enough threads to handle seq_len with multiple warps
    int threads_per_block = min(512, ((seq_len + 3) / 4) * 32); // 4 warps max
    threads_per_block = ((threads_per_block + 31) / 32) * 32;   // Round to warp size

    multimodal_rope_forward_kernel<T><<<grid, threads_per_block, 0, stream>>>(
        q, k, cos, sin, q_out, k_out, mrope_section_doubled,
        batch_size, q_heads, kv_heads, seq_len, head_dim);
}

template <typename T>
void launch_multimodal_rope_backward_template(
    const T *grad_q_out, const T *grad_k_out,
    const T *q, const T *k, const T *cos, const T *sin,
    T *grad_q, T *grad_k, const int *mrope_section_doubled,
    int batch_size, int q_heads, int kv_heads, int seq_len, int head_dim,
    cudaStream_t stream)
{
    static_assert(CudaTypeTraits<T>::is_supported, "Unsupported data type for multimodal RoPE");

    dim3 grid(batch_size, q_heads + kv_heads, 8);
    int threads_per_block = min(512, ((seq_len + 3) / 4) * 32);
    threads_per_block = ((threads_per_block + 31) / 32) * 32;

    multimodal_rope_backward_kernel<T><<<grid, threads_per_block, 0, stream>>>(
        grad_q_out, grad_k_out, q, k, cos, sin,
        grad_q, grad_k, mrope_section_doubled,
        batch_size, q_heads, kv_heads, seq_len, head_dim);
}

// Enum for data type dispatch
enum class DataType
{
    FLOAT32,
    FLOAT16,
    BFLOAT16
};

// Template dispatcher
template <DataType DT>
struct DataTypeDispatcher;

template <>
struct DataTypeDispatcher<DataType::FLOAT32>
{
    using type = float;
};

template <>
struct DataTypeDispatcher<DataType::FLOAT16>
{
    using type = half;
};

template <>
struct DataTypeDispatcher<DataType::BFLOAT16>
{
    using type = __nv_bfloat16;
};

// Type-safe host interface
template <DataType DT>
void launch_multimodal_rope_forward_typed(
    const void *q, const void *k, const void *cos, const void *sin,
    void *q_out, void *k_out,
    const int *mrope_section_doubled,
    int batch_size, int q_heads, int kv_heads, int seq_len, int head_dim,
    cudaStream_t stream)
{
    using T = typename DataTypeDispatcher<DT>::type;

    launch_multimodal_rope_forward_template<T>(
        static_cast<const T *>(q),
        static_cast<const T *>(k),
        static_cast<const T *>(cos),
        static_cast<const T *>(sin),
        static_cast<T *>(q_out),
        static_cast<T *>(k_out),
        mrope_section_doubled,
        batch_size, q_heads, kv_heads, seq_len, head_dim,
        stream);
}

template <DataType DT>
void launch_multimodal_rope_backward_typed(
    const void *grad_q_out, const void *grad_k_out,
    const void *q, const void *k, const void *cos, const void *sin,
    void *grad_q, void *grad_k,
    const int *mrope_section_doubled,
    int batch_size, int q_heads, int kv_heads, int seq_len, int head_dim,
    cudaStream_t stream)
{
    using T = typename DataTypeDispatcher<DT>::type;

    launch_multimodal_rope_backward_template<T>(
        static_cast<const T *>(grad_q_out),
        static_cast<const T *>(grad_k_out),
        static_cast<const T *>(q),
        static_cast<const T *>(k),
        static_cast<const T *>(cos),
        static_cast<const T *>(sin),
        static_cast<T *>(grad_q),
        static_cast<T *>(grad_k),
        mrope_section_doubled,
        batch_size, q_heads, kv_heads, seq_len, head_dim,
        stream);
}

void launch_multimodal_rope_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin,
    torch::Tensor q_out, torch::Tensor k_out,
    std::vector<int> mrope_section_doubled)
{

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int batch_size = q.size(0);
    int q_heads = q.size(1);
    int seq_len = q.size(2);
    int head_dim = q.size(3);
    int kv_heads = k.size(1);

    int data_type;
    if (q.scalar_type() == torch::kFloat32)
    {
        data_type = 0;
    }
    else if (q.scalar_type() == torch::kFloat16)
    {
        data_type = 1;
    }
    else if (q.scalar_type() == torch::kBFloat16)
    {
        data_type = 2;
    }
    int *d_mrope_section_doubled;
    cudaMalloc(&d_mrope_section_doubled, 3 * sizeof(int));
    cudaMemcpyAsync(d_mrope_section_doubled, mrope_section_doubled.data(), 3 * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    switch (data_type)
    {
    case 0: // float32
        launch_multimodal_rope_forward_typed<DataType::FLOAT32>(
            q.data_ptr(), k.data_ptr(), cos.data_ptr(), sin.data_ptr(),
            q_out.data_ptr(), k_out.data_ptr(), d_mrope_section_doubled,
            batch_size, q_heads, kv_heads, seq_len, head_dim, stream);
        break;
    case 1: // float16
        launch_multimodal_rope_forward_typed<DataType::FLOAT16>(
            q.data_ptr(), k.data_ptr(), cos.data_ptr(), sin.data_ptr(),
            q_out.data_ptr(), k_out.data_ptr(), d_mrope_section_doubled,
            batch_size, q_heads, kv_heads, seq_len, head_dim, stream);
        break;
    case 2: // bfloat16
        launch_multimodal_rope_forward_typed<DataType::BFLOAT16>(
            q.data_ptr(), k.data_ptr(), cos.data_ptr(), sin.data_ptr(),
            q_out.data_ptr(), k_out.data_ptr(), d_mrope_section_doubled,
            batch_size, q_heads, kv_heads, seq_len, head_dim, stream);
        break;
    }
}

void launch_multimodal_rope_backward(
    torch::Tensor grad_q_out, torch::Tensor grad_k_out,
    torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin,
    torch::Tensor grad_q, torch::Tensor grad_k,
    std::vector<int> mrope_section_doubled)
{

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int batch_size = q.size(0);
    int q_heads = q.size(1);
    int seq_len = q.size(2);
    int head_dim = q.size(3);
    int kv_heads = k.size(1);

    int data_type;
    if (q.scalar_type() == torch::kFloat32)
    {
        data_type = 0;
    }
    else if (q.scalar_type() == torch::kFloat16)
    {
        data_type = 1;
    }
    else if (q.scalar_type() == torch::kBFloat16)
    {
        data_type = 2;
    }
    int *d_mrope_section_doubled;
    cudaMalloc(&d_mrope_section_doubled, 3 * sizeof(int));
    cudaMemcpyAsync(d_mrope_section_doubled, mrope_section_doubled.data(), 3 * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    switch (data_type)
    {
    case 0: // float32
        launch_multimodal_rope_backward_typed<DataType::FLOAT32>(
            grad_q_out.data_ptr(), grad_k_out.data_ptr(),
            q.data_ptr(), k.data_ptr(), cos.data_ptr(), sin.data_ptr(),
            grad_q.data_ptr(), grad_k.data_ptr(), d_mrope_section_doubled,
            batch_size, q_heads, kv_heads, seq_len, head_dim, stream);
        break;
    case 1: // float16
        launch_multimodal_rope_backward_typed<DataType::FLOAT16>(
            grad_q_out.data_ptr(), grad_k_out.data_ptr(),
            q.data_ptr(), k.data_ptr(), cos.data_ptr(), sin.data_ptr(),
            grad_q.data_ptr(), grad_k.data_ptr(), d_mrope_section_doubled,
            batch_size, q_heads, kv_heads, seq_len, head_dim, stream);
        break;
    case 2: // bfloat16
        launch_multimodal_rope_backward_typed<DataType::BFLOAT16>(
            grad_q_out.data_ptr(), grad_k_out.data_ptr(),
            q.data_ptr(), k.data_ptr(), cos.data_ptr(), sin.data_ptr(),
            grad_q.data_ptr(), grad_k.data_ptr(), d_mrope_section_doubled,
            batch_size, q_heads, kv_heads, seq_len, head_dim, stream);
        break;
    }
}
