// Re-enable CUDA half operators (PyTorch disables them)
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include "../common/cuda_utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Global FP8 guard: use CUDA_VERSION (defined in both host and device passes),
// not __CUDA_ARCH__ (device-pass only — leaves host dispatchers unable to see
// FP8 typedefs/branches). FP8 storage types arrived in CUDA 11.8.
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 11800)
#include <cuda_fp8.h>
#define XCOMPUTE_HAS_FP8 1
#else
#define XCOMPUTE_HAS_FP8 0
#endif

namespace wallx_cuda_permute {

// ---------------------------------------------------------------------------
// CUTLASS-free helpers: replace cutlass::Array, NumericArrayConverter,
// arch::global_load, and type aliases with standard CUDA equivalents.
// ---------------------------------------------------------------------------

// Vectorized array: replaces cutlass::Array<T, N>
// __align__(16) ensures safe reinterpret_cast<float4*> in vec_load/store.
template <typename T, int N>
struct __align__(16) VecArray {
    T data[N];
    __device__ __forceinline__ T& at(int i) { return data[i]; }
    __device__ __forceinline__ const T& at(int i) const { return data[i]; }
    __device__ __forceinline__ T* raw() { return data; }
    __device__ __forceinline__ const T* raw() const { return data; }
    __device__ __forceinline__ void clear() {
        #pragma unroll
        for (int i = 0; i < N; i++) data[i] = T(0);
    }
};

// Scalar multiply: VecArray * scalar
template <typename T, int N>
__device__ __forceinline__ VecArray<T, N> operator*(const VecArray<T, N>& a, T s) {
    VecArray<T, N> r;
    #pragma unroll
    for (int i = 0; i < N; i++) r.data[i] = a.data[i] * s;
    return r;
}

// Element-wise convert: replaces cutlass::NumericArrayConverter
template <typename To, typename From, int N>
struct ArrayConverter {
    __device__ __forceinline__ VecArray<To, N> operator()(const VecArray<From, N>& src) const {
        VecArray<To, N> dst;
        #pragma unroll
        for (int i = 0; i < N; i++) dst.data[i] = To(src.data[i]);
        return dst;
    }
};

// Vectorized global load: replaces cutlass::arch::global_load
template <typename Fragment>
__device__ __forceinline__ void vec_load(Fragment& frag, const void* ptr) {
    static_assert(sizeof(Fragment) == sizeof(float4) || sizeof(Fragment) == 16,
                  "vec_load expects 16-byte fragment");
    *reinterpret_cast<float4*>(&frag) = *reinterpret_cast<const float4*>(ptr);
}

// Type aliases: replace cutlass types with standard CUDA types
using half_t = __half;
using bfloat16_t = __nv_bfloat16;
#if XCOMPUTE_HAS_FP8
using float_e5m2_t = __nv_fp8_e5m2;
using float_e4m3_t = __nv_fp8_e4m3;  // fn variant; CUDA's __nv_fp8_e4m3 IS the "fn" form
#endif

// ---------------------------------------------------------------------------
// Kernels (unchanged logic, only CUTLASS types/calls replaced)
// ---------------------------------------------------------------------------

static __global__ void moe_permute_topK_row_map(
    const int *sorted_row_id,
    int *row_id_map,
    const int num_rows,
    const int num_topK,
    const int num_out_tokens)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = bid * blockDim.x + tid;

    if (idx >= num_rows * num_topK)
        return;

    int source_row = sorted_row_id[idx];
    int source_token_id = source_row / num_topK;
    int source_topK_id = source_row % num_topK;

    if (idx >= num_out_tokens)
    {
        row_id_map[source_topK_id * num_rows + source_token_id] = -1;
    }
    else
    {
        row_id_map[source_topK_id * num_rows + source_token_id] = idx;
    }
}

template <typename T, typename TCompute, int kElementsPerAccess, bool hasProb>
__global__ void moe_recover_topK_kernel(const T *input,
                                        T *unpermuted_output,
                                        const int *row_id_map,
                                        const float *prob,
                                        const int num_rows,
                                        const int num_topK,
                                        const int num_cols)
{
    extern __shared__ int8_t s_mem[];
    TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

    using FragLS = VecArray<T, kElementsPerAccess>;
    using FragC  = VecArray<TCompute, kElementsPerAccess>;

    ArrayConverter<TCompute, T, kElementsPerAccess> src_converter;
    ArrayConverter<T, TCompute, kElementsPerAccess> dst_converter;

    const int source_token = blockIdx.x;
    const int tid = threadIdx.x;

    if (hasProb)
    {
        for (int i = tid; i < num_topK; i += blockDim.x * blockDim.y)
        {
            s_prob[i] = TCompute(prob[source_token * num_topK + i]);
        }
        __syncthreads();
    }

    for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess)
    {
        FragLS frag_load_store;
        FragC frag_elem;
        FragC frag_sum;

        int source_row = row_id_map[source_token];

        if (source_row != -1)
        {
            const T *source_row_ptr = input + source_row * num_cols;

            vec_load(frag_load_store, source_row_ptr + i);
            frag_sum = src_converter(frag_load_store);

            if (hasProb)
            {
                frag_sum = frag_sum * s_prob[0];
            }
        }
        else
        {
            frag_sum.clear();
        }

        for (int k = 1; k < num_topK; k++)
        {
            source_row = row_id_map[k * num_rows + source_token];

            if (source_row == -1)
                continue;

            const T *source_row_ptr = input + source_row * num_cols;

            vec_load(frag_load_store, source_row_ptr + i);
            frag_elem = src_converter(frag_load_store);

            if (hasProb)
            {
                frag_elem = frag_elem * s_prob[k];
            }

            for (int e = 0; e < kElementsPerAccess; e++)
            {
                frag_sum.at(e) = frag_sum.at(e) + frag_elem.at(e);
            }
        }

        T *dest_row_ptr = unpermuted_output + source_token * num_cols;
        frag_load_store = dst_converter(frag_sum);
        *(float4 *)(dest_row_ptr + i) = *(float4 *)(frag_load_store.raw());
    }
}

template <typename T,
          typename TCompute,
          int kElementsPerAccess,
          int topKTile,
          bool hasProb>
__global__ void moe_permute_topK_kernel(const T *input_bwd,
                                        const T *input_fwd,
                                        T *act_grad,
                                        const float *prob,
                                        float *prob_grad,
                                        const int *row_id_map,
                                        const int num_rows,
                                        const int num_topK,
                                        const int num_cols)
{
    extern __shared__ int8_t s_mem[];
    TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

    using FragLS = VecArray<T, kElementsPerAccess>;
    using FragC  = VecArray<TCompute, kElementsPerAccess>;

    ArrayConverter<TCompute, T, kElementsPerAccess> src_converter;
    ArrayConverter<T, TCompute, kElementsPerAccess> dst_converter;

    const int source_token = blockIdx.x;
    const int tid = threadIdx.x;

    if (hasProb)
    {
        for (int i = tid; i < num_topK; i += blockDim.x)
        {
            s_prob[i] = TCompute(prob[source_token * num_topK + i]);
        }
        __syncthreads();
    }

    float accum[topKTile] = {0.0f};
    FragLS frag_load_store;

    const T *source_row_ptr = input_bwd + source_token * num_cols;
    for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess)
    {
        vec_load(frag_load_store, source_row_ptr + i);
        FragC frag_src = src_converter(frag_load_store);

        int index = source_token;

        for (int k = 0; k < topKTile; k++)
        {
            if (k == num_topK)
                break;

            int dest_row = row_id_map[index];
            index += num_rows;

            if (dest_row == -1)
                continue;

            if (hasProb)
            {
                frag_load_store = dst_converter(frag_src * s_prob[k]);
            }
            else
            {
                frag_load_store = dst_converter(frag_src);
            }

            T *dest_row_ptr = act_grad + dest_row * num_cols;
            *(float4 *)(dest_row_ptr + i) = *(float4 *)(frag_load_store.raw());

            if (hasProb)
            {
                const T *input_fwd_ptr = input_fwd + dest_row * num_cols;
                vec_load(frag_load_store, input_fwd_ptr + i);
                FragC frag_input_fwd = src_converter(frag_load_store);

                for (int e = 0; e < kElementsPerAccess; e++)
                {
                    accum[k] += float(frag_src.at(e) * frag_input_fwd.at(e));
                }
            }
        }
    }

    if (hasProb)
    {
        for (int k = 0; k < topKTile; k++)
        {
            if (k == num_topK)
                break;

            for (int mask = 16; mask > 0; mask /= 2)
            {
                accum[k] = accum[k] + __shfl_xor_sync(0xffffffff, accum[k], mask, 32);
            }
        }

        if (tid == 0)
        {
            for (int k = 0; k < topKTile; k++)
            {
                if (k == num_topK)
                    break;
                prob_grad[source_token * num_topK + k] = accum[k];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Launcher (unchanged)
// ---------------------------------------------------------------------------

template <typename T, typename TCompute, bool FWD, int kElementsPerAccess>
void moe_permute_topK_kernel_launcher(
    const T *input,
    T *output,
    const int *sorted_row_id,
    int *row_id_map,
    const float *prob,
    const int num_rows,
    const int num_topK,
    const int num_cols,
    const int num_out_tokens,
    cudaStream_t stream,
    float *prob_grad = nullptr,
    const T *input_fwd = nullptr)
{
    if (FWD)
    {
        if (prob_grad == nullptr)
        {
            int threads = 64;
            int blocks = (num_rows * num_topK + threads - 1) / threads;
            moe_permute_topK_row_map<<<blocks, threads, 0, stream>>>(
                sorted_row_id, row_id_map, num_rows, num_topK, num_out_tokens);

            blocks = num_rows;
            threads = std::min(num_cols / kElementsPerAccess, 1024);
            moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 128, false><<<blocks, threads, 0, stream>>>(
                input, nullptr, output, nullptr, nullptr, row_id_map,
                num_rows, num_topK, num_cols);
        }
        else
        {
            int blocks = num_rows;
            int threads = 32;
            size_t smem_bytes = num_topK * sizeof(TCompute);

            if (num_topK == 1)
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 1, false><<<blocks, threads, 0, stream>>>(
                    input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, num_topK, num_cols);
            else if (num_topK <= 8)
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 8, true><<<blocks, threads, smem_bytes, stream>>>(
                    input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, num_topK, num_cols);
            else if (num_topK <= 16)
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 16, true><<<blocks, threads, smem_bytes, stream>>>(
                    input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, num_topK, num_cols);
            else if (num_topK <= 32)
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 32, true><<<blocks, threads, smem_bytes, stream>>>(
                    input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, num_topK, num_cols);
            else if (num_topK <= 64)
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 64, true><<<blocks, threads, smem_bytes, stream>>>(
                    input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, num_topK, num_cols);
            else if (num_topK <= 128)
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 128, true><<<blocks, threads, smem_bytes, stream>>>(
                    input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, num_topK, num_cols);
            else
                throw std::runtime_error("num_topK cannot exceed 128.");
        }
    }
    else
    {
        int blocks = num_rows;
        int threads = std::min(num_cols / kElementsPerAccess, 1024);
        size_t smem_bytes = num_topK * sizeof(TCompute);

        if (num_topK == 1)
            moe_recover_topK_kernel<T, TCompute, kElementsPerAccess, false><<<blocks, threads, smem_bytes, stream>>>(
                input, output, row_id_map, prob, num_rows, num_topK, num_cols);
        else if (prob == nullptr)
            moe_recover_topK_kernel<T, TCompute, kElementsPerAccess, false><<<blocks, threads, smem_bytes, stream>>>(
                input, output, row_id_map, prob, num_rows, num_topK, num_cols);
        else
            moe_recover_topK_kernel<T, TCompute, kElementsPerAccess, true><<<blocks, threads, smem_bytes, stream>>>(
                input, output, row_id_map, prob, num_rows, num_topK, num_cols);
    }
}

// ---------------------------------------------------------------------------
// Host-side ops
// ---------------------------------------------------------------------------

void MoePermuteTopKOp(
    const at::Tensor& input,
    const at::Tensor& indices,
    const at::Tensor& sorted_indices,
    const at::Tensor& row_id,
    const at::Tensor& sorted_row_id,
    const at::Tensor& temp_storage,
    const at::Tensor& permuted_output,
    const at::Tensor& row_id_map,
    int64_t num_out_tokens,
    int64_t max_expanded_token_num
) {
    ASSERT_CHECK(input.size(0) == indices.size(0));
    const int num_tokens = input.size(0);
    const int num_cols = input.size(1);
    const int num_topK = indices.size(1);

    int *indices_ptr = static_cast<int*>(indices.data_ptr());
    int *sorted_indices_ptr = static_cast<int*>(sorted_indices.data_ptr());
    int *row_id_ptr = static_cast<int*>(row_id.data_ptr());
    int *sorted_row_id_ptr = static_cast<int*>(sorted_row_id.data_ptr());

    void *d_temp_storage = static_cast<void*>(temp_storage.data_ptr());
    size_t temp_storage_bytes = std::numeric_limits<size_t>::max();

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    indices_ptr, sorted_indices_ptr,
                                    row_id_ptr, sorted_row_id_ptr, num_tokens * num_topK);

    num_out_tokens = (num_out_tokens > 0) ? num_out_tokens : num_tokens * num_topK;
    int *row_id_map_ptr = static_cast<int*>(row_id_map.data_ptr());
    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    if (input.scalar_type() == at::kFloat) {
        float *input_ptr = static_cast<float*>(input.data_ptr());
        float *out_ptr = static_cast<float*>(permuted_output.data_ptr());
        moe_permute_topK_kernel_launcher<float, float, true, 4>(
            input_ptr, out_ptr, sorted_row_id_ptr, row_id_map_ptr,
            nullptr, num_tokens, num_topK, num_cols, num_out_tokens, stream);
    } else if (input.scalar_type() == at::kHalf) {
        half_t *input_ptr = static_cast<half_t*>(input.data_ptr());
        half_t *out_ptr = static_cast<half_t*>(permuted_output.data_ptr());
        moe_permute_topK_kernel_launcher<half_t, half_t, true, 8>(
            input_ptr, out_ptr, sorted_row_id_ptr, row_id_map_ptr,
            nullptr, num_tokens, num_topK, num_cols, num_out_tokens, stream);
    } else if (input.scalar_type() == at::kBFloat16) {
        bfloat16_t *input_ptr = static_cast<bfloat16_t*>(input.data_ptr());
        bfloat16_t *out_ptr = static_cast<bfloat16_t*>(permuted_output.data_ptr());
        moe_permute_topK_kernel_launcher<bfloat16_t, bfloat16_t, true, 8>(
            input_ptr, out_ptr, sorted_row_id_ptr, row_id_map_ptr,
            nullptr, num_tokens, num_topK, num_cols, num_out_tokens, stream);
#if XCOMPUTE_HAS_FP8
    } else if (input.scalar_type() == at::kFloat8_e5m2) {
        float_e5m2_t *input_ptr = static_cast<float_e5m2_t*>(input.data_ptr());
        float_e5m2_t *out_ptr = static_cast<float_e5m2_t*>(permuted_output.data_ptr());
        moe_permute_topK_kernel_launcher<float_e5m2_t, half_t, true, 16>(
            input_ptr, out_ptr, sorted_row_id_ptr, row_id_map_ptr,
            nullptr, num_tokens, num_topK, num_cols, num_out_tokens, stream);
    } else if (input.scalar_type() == at::kFloat8_e4m3fn) {
        float_e4m3_t *input_ptr = static_cast<float_e4m3_t*>(input.data_ptr());
        float_e4m3_t *out_ptr = static_cast<float_e4m3_t*>(permuted_output.data_ptr());
        moe_permute_topK_kernel_launcher<float_e4m3_t, half_t, true, 16>(
            input_ptr, out_ptr, sorted_row_id_ptr, row_id_map_ptr,
            nullptr, num_tokens, num_topK, num_cols, num_out_tokens, stream);
#endif
    } else {
        throw std::runtime_error("Unsupported data type for MoePermuteTopKOp");
    }
}

void MoeRecoverTopKOp(
    const at::Tensor& input,
    const at::Tensor& row_id_map,
    const c10::optional<at::Tensor>& prob,
    const at::Tensor& unpermuted_output,
    int64_t num_tokens,
    int64_t num_topK
) {
    const int num_cols = input.size(1);
    int *row_id_map_ptr = static_cast<int*>(row_id_map.data_ptr());
    float *prob_ptr = (prob.has_value()) ? static_cast<float*>(prob.value().data_ptr()) : nullptr;
    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    if (input.scalar_type() == at::kFloat) {
        float *in_ptr = static_cast<float*>(input.data_ptr());
        float *out_ptr = static_cast<float*>(unpermuted_output.data_ptr());
        moe_permute_topK_kernel_launcher<float, float, false, 4>(
            in_ptr, out_ptr, nullptr, row_id_map_ptr, prob_ptr, num_tokens, num_topK, num_cols, 0, stream);
    } else if (input.scalar_type() == at::kHalf) {
        half_t *in_ptr = static_cast<half_t*>(input.data_ptr());
        half_t *out_ptr = static_cast<half_t*>(unpermuted_output.data_ptr());
        moe_permute_topK_kernel_launcher<half_t, half_t, false, 8>(
            in_ptr, out_ptr, nullptr, row_id_map_ptr, prob_ptr, num_tokens, num_topK, num_cols, 0, stream);
    } else if (input.scalar_type() == at::kBFloat16) {
        bfloat16_t *in_ptr = static_cast<bfloat16_t*>(input.data_ptr());
        bfloat16_t *out_ptr = static_cast<bfloat16_t*>(unpermuted_output.data_ptr());
        moe_permute_topK_kernel_launcher<bfloat16_t, bfloat16_t, false, 8>(
            in_ptr, out_ptr, nullptr, row_id_map_ptr, prob_ptr, num_tokens, num_topK, num_cols, 0, stream);
#if XCOMPUTE_HAS_FP8
    } else if (input.scalar_type() == at::kFloat8_e5m2) {
        float_e5m2_t *in_ptr = static_cast<float_e5m2_t*>(input.data_ptr());
        float_e5m2_t *out_ptr = static_cast<float_e5m2_t*>(unpermuted_output.data_ptr());
        moe_permute_topK_kernel_launcher<float_e5m2_t, half_t, false, 16>(
            in_ptr, out_ptr, nullptr, row_id_map_ptr, prob_ptr, num_tokens, num_topK, num_cols, 0, stream);
    } else if (input.scalar_type() == at::kFloat8_e4m3fn) {
        float_e4m3_t *in_ptr = static_cast<float_e4m3_t*>(input.data_ptr());
        float_e4m3_t *out_ptr = static_cast<float_e4m3_t*>(unpermuted_output.data_ptr());
        moe_permute_topK_kernel_launcher<float_e4m3_t, half_t, false, 16>(
            in_ptr, out_ptr, nullptr, row_id_map_ptr, prob_ptr, num_tokens, num_topK, num_cols, 0, stream);
#endif
    } else {
        throw std::runtime_error("Unsupported data type for MoeRecoverTopKOp");
    }
    sync_check_cuda_error();
}

void MoeRecoverTopKBwdOp(
    const at::Tensor& input_bwd,
    const at::Tensor& input_fwd,
    const at::Tensor& row_id_map,
    const at::Tensor& prob,
    const at::Tensor& act_grad,
    const at::Tensor& prob_grad
) {
    const int num_tokens = prob.size(0);
    const int num_topK = prob.size(1);
    const int num_cols = input_bwd.size(1);
    int *row_id_map_ptr = static_cast<int*>(row_id_map.data_ptr());
    float *prob_ptr = static_cast<float*>(prob.data_ptr());
    float *prob_grad_ptr = static_cast<float*>(prob_grad.data_ptr());
    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    if (input_bwd.scalar_type() == at::kFloat) {
        float *bwd = static_cast<float*>(input_bwd.data_ptr());
        float *fwd = static_cast<float*>(input_fwd.data_ptr());
        float *grad = static_cast<float*>(act_grad.data_ptr());
        moe_permute_topK_kernel_launcher<float, float, true, 4>(
            bwd, grad, nullptr, row_id_map_ptr, prob_ptr, num_tokens, num_topK, num_cols, 0, stream, prob_grad_ptr, fwd);
    } else if (input_bwd.scalar_type() == at::kHalf) {
        half_t *bwd = static_cast<half_t*>(input_bwd.data_ptr());
        half_t *fwd = static_cast<half_t*>(input_fwd.data_ptr());
        half_t *grad = static_cast<half_t*>(act_grad.data_ptr());
        moe_permute_topK_kernel_launcher<half_t, half_t, true, 8>(
            bwd, grad, nullptr, row_id_map_ptr, prob_ptr, num_tokens, num_topK, num_cols, 0, stream, prob_grad_ptr, fwd);
    } else if (input_bwd.scalar_type() == at::kBFloat16) {
        bfloat16_t *bwd = static_cast<bfloat16_t*>(input_bwd.data_ptr());
        bfloat16_t *fwd = static_cast<bfloat16_t*>(input_fwd.data_ptr());
        bfloat16_t *grad = static_cast<bfloat16_t*>(act_grad.data_ptr());
        moe_permute_topK_kernel_launcher<bfloat16_t, bfloat16_t, true, 8>(
            bwd, grad, nullptr, row_id_map_ptr, prob_ptr, num_tokens, num_topK, num_cols, 0, stream, prob_grad_ptr, fwd);
#if XCOMPUTE_HAS_FP8
    } else if (input_bwd.scalar_type() == at::kFloat8_e5m2) {
        float_e5m2_t *bwd = static_cast<float_e5m2_t*>(input_bwd.data_ptr());
        float_e5m2_t *fwd = static_cast<float_e5m2_t*>(input_fwd.data_ptr());
        float_e5m2_t *grad = static_cast<float_e5m2_t*>(act_grad.data_ptr());
        moe_permute_topK_kernel_launcher<float_e5m2_t, half_t, true, 16>(
            bwd, grad, nullptr, row_id_map_ptr, prob_ptr, num_tokens, num_topK, num_cols, 0, stream, prob_grad_ptr, fwd);
    } else if (input_bwd.scalar_type() == at::kFloat8_e4m3fn) {
        float_e4m3_t *bwd = static_cast<float_e4m3_t*>(input_bwd.data_ptr());
        float_e4m3_t *fwd = static_cast<float_e4m3_t*>(input_fwd.data_ptr());
        float_e4m3_t *grad = static_cast<float_e4m3_t*>(act_grad.data_ptr());
        moe_permute_topK_kernel_launcher<float_e4m3_t, half_t, true, 16>(
            bwd, grad, nullptr, row_id_map_ptr, prob_ptr, num_tokens, num_topK, num_cols, 0, stream, prob_grad_ptr, fwd);
#endif
    } else {
        throw std::runtime_error("Unsupported data type for MoeRecoverTopKBwdOp");
    }
    sync_check_cuda_error();
}

size_t CubSortPairGetStorageBytes(int64_t num_items){
    size_t temp_storage_bytes = 0;
    int *temp_ptr = nullptr;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                    temp_ptr, temp_ptr,
                                    temp_ptr, temp_ptr, num_items);
    return temp_storage_bytes;
}

// Removed TVM FFI export (see pybind11 registration)
// Removed TVM FFI export (see pybind11 registration)
// Removed TVM FFI export (see pybind11 registration)
// Removed TVM FFI export (see pybind11 registration)

} // wallx_cuda_permute
