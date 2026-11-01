#include "tensor_view.h"
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */


#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>

#include "../common/cuda_utils.h"

namespace wallx_cuda_silu_and_mul {

// ============================================================
// Tuning notes (H100 sm90 / A800 sm80, bf16, production shapes
// [21504, 2x9216] and [26624, 2x5120]):
//
//  * The dominant win is vector width. The original kernels used
//    VecTraits (bfloat162 = 4 B per access); 16 B accesses
//    (PackVec<bf16,8>) raise achieved HBM bandwidth from ~70-75% to
//    ~90% of peak on H100 and from ~62-68% to ~86% on A800.
//  * Grid shape: a plain non-persistent 2-D grid (grid.x covers the
//    packed column dimension, grid.y = one block-row per tensor row)
//    beats every persistent variant tried (grid.y = k*sm_count for
//    k in {1,2,4,8}) by 2-6% on both architectures. The row loop is
//    kept only for the num_rows > 65535 case.
//  * BLOCK_X = 128 is best-or-tied on both architectures and divides
//    the packed width of both production shapes exactly.
//  * __ldcs/__stcs streaming hints and manual row unrolling were both
//    measured slower; float math (vs half2 intrinsics) is not a
//    bottleneck at ~90% of peak bandwidth. Do not "optimize" these
//    back in without re-measuring.
//
// The pack width is chosen at runtime per call: the widest of
// {16B, 8B, 4B, scalar} compatible with the last-dim length, every
// outer stride and every base pointer. Production tensors always take
// the 16 B path; odd shapes (odd intermediate size, misaligned slice
// bases) degrade gracefully instead of asserting.
// ============================================================

// Returns the stride (in elements) of dimension d.
// When strides is null (fully contiguous, row-major), computes from shape.
// When strides is non-null, reads the stored value directly.
static int64_t safe_stride(const wallx_cuda::TensorView& t, int d) {
    if (t.IsContiguous()) {
        int64_t s = 1;
        for (int i = t.ndim() - 1; i > d; --i) s *= t.size(i);
        return s;
    }
    return t.stride(d);
}

constexpr int MAX_OUTER_DIMS = 7;  // supports tensors up to 8-D
constexpr int BLOCK_X = 128;       // see tuning notes above
constexpr int MAX_GRID_Y = 65535;  // hardware limit on gridDim.y

struct SiluMulInplaceArgs {
    int64_t strides1[MAX_OUTER_DIMS];  // outer-dim strides of inout1,  in elements
    int64_t strides2[MAX_OUTER_DIMS];  // outer-dim strides of input2,  in elements
    int64_t shapes[MAX_OUTER_DIMS];    // outer-dim sizes (same for both tensors)
    int outer_ndim;                    // ndim - 1  (0 for 1-D tensors)
};

struct SiluMulSplitArgs {
    int64_t output_strides[MAX_OUTER_DIMS];   // outer-dim strides of output
    int64_t gate_up_strides[MAX_OUTER_DIMS];  // outer-dim strides of gate_up
    int64_t shapes[MAX_OUTER_DIMS];           // outer-dim sizes
    int outer_ndim;                           // ndim - 1
};

struct SiluMulSplitBackwardArgs {
    int64_t grad_gate_up_strides[MAX_OUTER_DIMS];  // outer-dim strides of grad_gate_up
    int64_t grad_output_strides[MAX_OUTER_DIMS];   // outer-dim strides of grad_output
    int64_t gate_up_strides[MAX_OUTER_DIMS];       // outer-dim strides of original gate_up
    int64_t shapes[MAX_OUTER_DIMS];                // outer-dim sizes
    int outer_ndim;                                // ndim - 1
};

// ============================================================
// Scalar helpers
// ============================================================

template<typename T>
__device__ __forceinline__ float to_float(T x);

template<>
__device__ __forceinline__ float to_float<float>(float x) {
    return x;
}

template<>
__device__ __forceinline__ float to_float<__half>(__half x) {
    return __half2float(x);
}

template<>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template<typename T>
__device__ __forceinline__ T from_float(float x);

template<>
__device__ __forceinline__ float from_float<float>(float x) {
    return x;
}

template<>
__device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half(x);
}

template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

template<typename T>
__device__ __forceinline__ void silu_mul_backward_lane(
    T gate,
    T up,
    T grad_output,
    T& grad_gate,
    T& grad_up)
{
    const float gate_f = to_float<T>(gate);
    const float up_f = to_float<T>(up);
    const float grad_f = to_float<T>(grad_output);
    const float sigmoid = 1.0f / (1.0f + __expf(-gate_f));
    const float silu = gate_f * sigmoid;
    const float silu_grad = sigmoid * (1.0f + gate_f * (1.0f - sigmoid));

    grad_gate = from_float<T>(grad_f * up_f * silu_grad);
    grad_up = from_float<T>(grad_f * silu);
}

// ============================================================
// Packed vector type: PACK elements of T, loaded/stored as one
// naturally-aligned transaction (up to 16 B).
// ============================================================

template<typename T, int PACK>
struct alignas(sizeof(T) * PACK) PackVec {
    T d[PACK];
};

template<typename T>
constexpr int max_pack() { return 16 / (int)sizeof(T); }

// Widest pack (elements) such that
//   * every length in `lens` is divisible by it, and
//   * every pointer in `ptrs` is aligned to pack*sizeof(T).
// Falls back to 1 (scalar) in the worst case.
template<typename T>
static int choose_pack(std::initializer_list<const void*> ptrs,
                       const int64_t* lens, int n_lens) {
    int pack = max_pack<T>();
    auto incompatible = [&](int p) {
        const uintptr_t align = (uintptr_t)p * sizeof(T);
        for (const void* q : ptrs) {
            if (reinterpret_cast<uintptr_t>(q) % align != 0) return true;
        }
        for (int i = 0; i < n_lens; ++i) {
            if (lens[i] % p != 0) return true;
        }
        return false;
    };
    while (pack > 1 && incompatible(pack)) pack >>= 1;
    return pack;
}

// Decompose a flat row index into per-dimension indices and accumulate
// per-tensor element offsets. NUM_TENSORS stride arrays are walked in lockstep.
template<int NUM_TENSORS>
__device__ __forceinline__ void row_offsets(
    int64_t row,
    const int64_t* shapes,
    int outer_ndim,
    const int64_t* const (&strides)[NUM_TENSORS],
    int64_t (&offsets)[NUM_TENSORS])
{
    #pragma unroll
    for (int t = 0; t < NUM_TENSORS; ++t) offsets[t] = 0;

    if (outer_ndim == 1) {
        #pragma unroll
        for (int t = 0; t < NUM_TENSORS; ++t) offsets[t] = row * strides[t][0];
        return;
    }
    if (outer_ndim > 1) {
        int64_t r = row;
        #pragma unroll MAX_OUTER_DIMS
        for (int d = MAX_OUTER_DIMS - 1; d >= 0; --d) {
            if (d < outer_ndim) {
                const int64_t idx = r % shapes[d];
                r /= shapes[d];
                #pragma unroll
                for (int t = 0; t < NUM_TENSORS; ++t) offsets[t] += idx * strides[t][d];
            }
        }
    }
}

static inline dim3 make_grid_2d(int64_t col_packs, int64_t num_rows) {
    const int grid_x = (int)((col_packs + BLOCK_X - 1) / BLOCK_X);
    const int grid_y = (int)std::min<int64_t>(num_rows, MAX_GRID_Y);
    return dim3(grid_x, grid_y);
}

// ============================================================
// Kernel 1: fully-contiguous flat forward
// output[i] = silu(input1[i]) * input2[i]
// Non-persistent 1-D grid; the grid-stride loop only iterates when the
// grid was clamped (never at production sizes).
// ============================================================
template<typename T, int PACK>
__global__ void siluAndMulFlatKernel(
    T* __restrict__ output,
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    int64_t n_packs)
{
    using P = PackVec<T, PACK>;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n_packs; i += stride) {
        P x = reinterpret_cast<const P*>(input1)[i];
        P g = reinterpret_cast<const P*>(input2)[i];
        P out;
        #pragma unroll
        for (int j = 0; j < PACK; ++j) {
            out.d[j] = from_float<T>(silu_f(to_float<T>(x.d[j])) * to_float<T>(g.d[j]));
        }
        reinterpret_cast<P*>(output)[i] = out;
    }
}

// ============================================================
// Kernel 2: fully-contiguous flat backward (two-tensor mode)
// ============================================================
template<typename T, int PACK>
__global__ void siluAndMulFlatBackwardKernel(
    T* __restrict__ grad_input1,
    T* __restrict__ grad_input2,
    const T* __restrict__ grad_output,
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    int64_t n_packs)
{
    using P = PackVec<T, PACK>;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n_packs; i += stride) {
        P gate = reinterpret_cast<const P*>(input1)[i];
        P up = reinterpret_cast<const P*>(input2)[i];
        P grad = reinterpret_cast<const P*>(grad_output)[i];
        P grad_gate, grad_up;
        #pragma unroll
        for (int j = 0; j < PACK; ++j) {
            silu_mul_backward_lane<T>(gate.d[j], up.d[j], grad.d[j],
                                      grad_gate.d[j], grad_up.d[j]);
        }
        reinterpret_cast<P*>(grad_input1)[i] = grad_gate;
        reinterpret_cast<P*>(grad_input2)[i] = grad_up;
    }
}

// ============================================================
// Kernel 3: last-dim-contiguous two-tensor inplace
// inout1[row, col] = silu(inout1[row, col]) * input2[row, col]
// 2-D grid: x covers packed columns, y covers rows (loop only if clamped).
// ============================================================
template<typename T, int PACK>
__global__ void siluAndMulInplaceKernel(
    T* __restrict__ inout1,
    const T* __restrict__ input2,
    SiluMulInplaceArgs args,
    int64_t num_rows,
    int64_t last_packs)   // last_dim / PACK
{
    using P = PackVec<T, PACK>;
    const int64_t col = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= last_packs) return;

    for (int64_t row = (int64_t)blockIdx.y; row < num_rows; row += (int64_t)gridDim.y) {
        const int64_t* strides[2] = {args.strides1, args.strides2};
        int64_t off[2];
        row_offsets<2>(row, args.shapes, args.outer_ndim, strides, off);

        P* p1 = reinterpret_cast<P*>(inout1 + off[0]) + col;
        const P* p2 = reinterpret_cast<const P*>(input2 + off[1]) + col;

        P x = *p1;
        P g = *p2;
        P out;
        #pragma unroll
        for (int j = 0; j < PACK; ++j) {
            out.d[j] = from_float<T>(silu_f(to_float<T>(x.d[j])) * to_float<T>(g.d[j]));
        }
        *p1 = out;
    }
}

// ============================================================
// Kernel 4: fused split + silu + mul forward.
// Serves both the out-of-place path (output is a distinct tensor) and the
// inplace path (output == gate_up base with gate_up strides): every thread
// reads gate/up of its own (row, col) before writing gate, and no thread
// reads another thread's write location, so aliasing is safe.
//
// Input:  gate_up [..., 2*N], last dim contiguous.
// Output: output [..., N] (last dim contiguous) = silu(gate) * up.
// ============================================================
template<typename T, int PACK>
__global__ void siluAndMulSplitKernel(
    T* __restrict__ output,
    const T* __restrict__ gate_up,
    SiluMulSplitArgs args,
    int64_t num_rows,
    int64_t half_packs,        // intermediate_size / PACK
    int64_t intermediate_size)
{
    using P = PackVec<T, PACK>;
    const int64_t col = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= half_packs) return;

    for (int64_t row = (int64_t)blockIdx.y; row < num_rows; row += (int64_t)gridDim.y) {
        const int64_t* strides[2] = {args.output_strides, args.gate_up_strides};
        int64_t off[2];
        row_offsets<2>(row, args.shapes, args.outer_ndim, strides, off);

        const P* gate_p = reinterpret_cast<const P*>(gate_up + off[1]) + col;
        const P* up_p = reinterpret_cast<const P*>(gate_up + off[1] + intermediate_size) + col;
        P* out_p = reinterpret_cast<P*>(output + off[0]) + col;

        P x = *gate_p;
        P g = *up_p;
        P out;
        #pragma unroll
        for (int j = 0; j < PACK; ++j) {
            out.d[j] = from_float<T>(silu_f(to_float<T>(x.d[j])) * to_float<T>(g.d[j]));
        }
        *out_p = out;
    }
}

// ============================================================
// Kernel 5: fused split backward.
// Given original gate_up [..., 2*N] and grad_output [..., N], writes
// grad_gate_up[..., :N] = grad_output * up * d_silu(gate)
// grad_gate_up[..., N:] = grad_output * silu(gate)
// ============================================================
template<typename T, int PACK>
__global__ void siluAndMulSplitBackwardKernel(
    T* __restrict__ grad_gate_up,
    const T* __restrict__ grad_output,
    const T* __restrict__ gate_up,
    SiluMulSplitBackwardArgs args,
    int64_t num_rows,
    int64_t half_packs,
    int64_t intermediate_size)
{
    using P = PackVec<T, PACK>;
    const int64_t col = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= half_packs) return;

    for (int64_t row = (int64_t)blockIdx.y; row < num_rows; row += (int64_t)gridDim.y) {
        const int64_t* strides[3] = {
            args.grad_gate_up_strides, args.grad_output_strides, args.gate_up_strides};
        int64_t off[3];
        row_offsets<3>(row, args.shapes, args.outer_ndim, strides, off);

        const P* gate_p = reinterpret_cast<const P*>(gate_up + off[2]) + col;
        const P* up_p = reinterpret_cast<const P*>(gate_up + off[2] + intermediate_size) + col;
        const P* grad_p = reinterpret_cast<const P*>(grad_output + off[1]) + col;
        P* grad_gate_p = reinterpret_cast<P*>(grad_gate_up + off[0]) + col;
        P* grad_up_p = reinterpret_cast<P*>(grad_gate_up + off[0] + intermediate_size) + col;

        P gate = *gate_p;
        P up = *up_p;
        P grad = *grad_p;
        P grad_gate, grad_up;
        #pragma unroll
        for (int j = 0; j < PACK; ++j) {
            silu_mul_backward_lane<T>(gate.d[j], up.d[j], grad.d[j],
                                      grad_gate.d[j], grad_up.d[j]);
        }
        *grad_gate_p = grad_gate;
        *grad_up_p = grad_up;
    }
}

// ============================================================
// Pack-width runtime dispatch
// ============================================================

// Invokes launcher.template run<PACK>() with the widest supported pack.
template<typename T, typename Launcher>
static void dispatch_pack(int pack, Launcher&& launcher) {
    switch (pack) {
        case 8:
            if constexpr (max_pack<T>() >= 8) { launcher.template run<8>(); return; }
            break;
        case 4:
            if constexpr (max_pack<T>() >= 4) { launcher.template run<4>(); return; }
            break;
        case 2:
            launcher.template run<2>();
            return;
        default:
            break;
    }
    launcher.template run<1>();
}

// Invokes Fn<T>(args...) for the runtime dtype of `dt`.
#define DISPATCH_FLOAT_DTYPE(dt, op_name, ...)                              \
    do {                                                                     \
        if ((dt).code == kDLFloat && (dt).bits == 32) {                      \
            using scalar_t = float;                                          \
            __VA_ARGS__;                                                     \
        } else if ((dt).code == kDLFloat && (dt).bits == 16) {               \
            using scalar_t = __half;                                         \
            __VA_ARGS__;                                                     \
        } else if ((dt).code == kDLBfloat && (dt).bits == 16) {              \
            using scalar_t = __nv_bfloat16;                                  \
            __VA_ARGS__;                                                     \
        } else {                                                             \
            throw std::runtime_error(op_name ": unsupported dtype");         \
        }                                                                    \
    } while (0)

// ============================================================
// Helpers
// ============================================================

static int64_t get_total_elements(const wallx_cuda::TensorView& t) {
    int64_t n = 1;
    for (int i = 0; i < t.ndim(); ++i) n *= t.size(i);
    return n;
}

template<typename T>
struct FlatFwdLauncher {
    T* output; const T* input1; const T* input2;
    int64_t n; cudaStream_t stream;
    template<int PACK> void run() {
        const int64_t n_packs = n / PACK;
        const int64_t want = (n_packs + BLOCK_X - 1) / BLOCK_X;
        const int grid = (int)std::min<int64_t>(want, INT32_MAX);
        siluAndMulFlatKernel<T, PACK><<<grid, BLOCK_X, 0, stream>>>(
            output, input1, input2, n_packs);
    }
};

template<typename T>
struct FlatBwdLauncher {
    T* grad_input1; T* grad_input2;
    const T* grad_output; const T* input1; const T* input2;
    int64_t n; cudaStream_t stream;
    template<int PACK> void run() {
        const int64_t n_packs = n / PACK;
        const int64_t want = (n_packs + BLOCK_X - 1) / BLOCK_X;
        const int grid = (int)std::min<int64_t>(want, INT32_MAX);
        siluAndMulFlatBackwardKernel<T, PACK><<<grid, BLOCK_X, 0, stream>>>(
            grad_input1, grad_input2, grad_output, input1, input2, n_packs);
    }
};

template<typename T>
struct InplaceLauncher {
    T* inout1; const T* input2;
    const SiluMulInplaceArgs* args;
    int64_t num_rows; int64_t last_dim; cudaStream_t stream;
    template<int PACK> void run() {
        const int64_t last_packs = last_dim / PACK;
        siluAndMulInplaceKernel<T, PACK>
            <<<make_grid_2d(last_packs, num_rows), BLOCK_X, 0, stream>>>(
                inout1, input2, *args, num_rows, last_packs);
    }
};

template<typename T>
struct SplitFwdLauncher {
    T* output; const T* gate_up;
    const SiluMulSplitArgs* args;
    int64_t num_rows; int64_t intermediate_size; cudaStream_t stream;
    template<int PACK> void run() {
        const int64_t half_packs = intermediate_size / PACK;
        siluAndMulSplitKernel<T, PACK>
            <<<make_grid_2d(half_packs, num_rows), BLOCK_X, 0, stream>>>(
                output, gate_up, *args, num_rows, half_packs, intermediate_size);
    }
};

template<typename T>
struct SplitBwdLauncher {
    T* grad_gate_up; const T* grad_output; const T* gate_up;
    const SiluMulSplitBackwardArgs* args;
    int64_t num_rows; int64_t intermediate_size; cudaStream_t stream;
    template<int PACK> void run() {
        const int64_t half_packs = intermediate_size / PACK;
        siluAndMulSplitBackwardKernel<T, PACK>
            <<<make_grid_2d(half_packs, num_rows), BLOCK_X, 0, stream>>>(
                grad_gate_up, grad_output, gate_up, *args, num_rows,
                half_packs, intermediate_size);
    }
};

// Collect the strides (of dims with extent > 1) that constrain the pack
// width. Size-1 dims are skipped: their strides are never multiplied by a
// nonzero index, and torch frequently reports arbitrary values for them.
static void collect_outer_strides(
    const int64_t* strides, const int64_t* shapes, int outer_ndim,
    int64_t* out, int& count)
{
    for (int d = 0; d < outer_ndim; ++d) {
        if (shapes[d] > 1) out[count++] = strides[d];
    }
}

// ============================================================
// Public functions
// ============================================================

// Non-inplace: output = silu(input1) * input2  (all tensors must be fully contiguous)
void SiluAndMul(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView input1,
    wallx_cuda::TensorView input2)
{
    ASSERT_CHECK(output.IsContiguous());
    ASSERT_CHECK(input1.IsContiguous());
    ASSERT_CHECK(input2.IsContiguous());
    ASSERT_CHECK(output.ndim() == input1.ndim());
    ASSERT_CHECK(input1.ndim() == input2.ndim());
    for (int i = 0; i < input1.ndim(); ++i) {
        ASSERT_CHECK(output.size(i) == input1.size(i));
        ASSERT_CHECK(input1.size(i) == input2.size(i));
    }

    const int64_t n = get_total_elements(input1);
    if (n == 0) return;
    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(input1.device().device_type, input1.device().device_id));

    DISPATCH_FLOAT_DTYPE(input1.dtype(), "silu_and_mul", {
        FlatFwdLauncher<scalar_t> launcher{
            static_cast<scalar_t*>(output.data_ptr()),
            static_cast<const scalar_t*>(input1.data_ptr()),
            static_cast<const scalar_t*>(input2.data_ptr()),
            n, stream};
        const int pack = choose_pack<scalar_t>(
            {output.data_ptr(), input1.data_ptr(), input2.data_ptr()}, &n, 1);
        dispatch_pack<scalar_t>(pack, launcher);
    });
    sync_check_cuda_error();
}

// Inplace: input1 = silu(input1) * input2
// Only requires the last dimension to be contiguous (stride[-1] == 1).
// Outer dimensions may be non-contiguous; their offsets are computed via strides.
void SiluAndMulInplace(
    wallx_cuda::TensorView input1,
    wallx_cuda::TensorView input2)
{
    const int ndim = input1.ndim();
    ASSERT_CHECK(ndim >= 1);
    ASSERT_CHECK(ndim == input2.ndim());
    ASSERT_CHECK(ndim - 1 <= MAX_OUTER_DIMS);

    // Shape must match on every dimension.
    for (int i = 0; i < ndim; ++i) {
        ASSERT_CHECK(input1.size(i) == input2.size(i));
    }

    // Only the last dimension must be contiguous.
    ASSERT_CHECK(safe_stride(input1, ndim - 1) == 1);
    ASSERT_CHECK(safe_stride(input2, ndim - 1) == 1);

    const int64_t last_dim = input1.size(ndim - 1);
    const int outer_ndim = ndim - 1;

    SiluMulInplaceArgs args{};
    args.outer_ndim = outer_ndim;
    int64_t num_rows = 1;
    for (int d = 0; d < outer_ndim; ++d) {
        args.strides1[d] = safe_stride(input1, d);
        args.strides2[d] = safe_stride(input2, d);
        args.shapes[d]   = input1.size(d);
        num_rows *= input1.size(d);
    }
    if (num_rows == 0 || last_dim == 0) return;

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(input1.device().device_type, input1.device().device_id));

    int64_t stride_lens[2 * MAX_OUTER_DIMS + 1];
    int n_lens = 0;
    stride_lens[n_lens++] = last_dim;
    collect_outer_strides(args.strides1, args.shapes, outer_ndim, stride_lens, n_lens);
    collect_outer_strides(args.strides2, args.shapes, outer_ndim, stride_lens, n_lens);

    DISPATCH_FLOAT_DTYPE(input1.dtype(), "silu_and_mul_inplace", {
        InplaceLauncher<scalar_t> launcher{
            static_cast<scalar_t*>(input1.data_ptr()),
            static_cast<const scalar_t*>(input2.data_ptr()),
            &args, num_rows, last_dim, stream};
        const int pack = choose_pack<scalar_t>(
            {input1.data_ptr(), input2.data_ptr()},
            stride_lens, n_lens);
        dispatch_pack<scalar_t>(pack, launcher);
    });
    sync_check_cuda_error();
}

// SiluAndMulSplit: fused split + silu + mul, out-of-place.
// gate_up has shape [..., 2*N], output has shape [..., N].
// Only the last dimension must be contiguous; outer dimensions may be strided.
void SiluAndMulSplit(
    wallx_cuda::TensorView output,
    wallx_cuda::TensorView gate_up)
{
    const int ndim = gate_up.ndim();
    ASSERT_CHECK(ndim >= 1);
    ASSERT_CHECK(ndim == output.ndim());
    ASSERT_CHECK(ndim - 1 <= MAX_OUTER_DIMS);
    ASSERT_CHECK(safe_stride(gate_up, ndim - 1) == 1);
    ASSERT_CHECK(safe_stride(output, ndim - 1) == 1);

    const int64_t last_dim = gate_up.size(ndim - 1);
    ASSERT_CHECK(last_dim % 2 == 0);
    const int64_t intermediate_size = last_dim / 2;
    ASSERT_CHECK(output.size(ndim - 1) == intermediate_size);

    const int outer_ndim = ndim - 1;
    SiluMulSplitArgs args{};
    args.outer_ndim = outer_ndim;
    int64_t num_rows = 1;
    for (int d = 0; d < outer_ndim; ++d) {
        ASSERT_CHECK(output.size(d) == gate_up.size(d));
        args.output_strides[d] = safe_stride(output, d);
        args.gate_up_strides[d] = safe_stride(gate_up, d);
        args.shapes[d] = gate_up.size(d);
        num_rows *= gate_up.size(d);
    }
    if (num_rows == 0 || intermediate_size == 0) return;

    ASSERT_CHECK(output.dtype().code == gate_up.dtype().code);
    ASSERT_CHECK(output.dtype().bits == gate_up.dtype().bits);

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(gate_up.device().device_type, gate_up.device().device_id));

    int64_t stride_lens[2 * MAX_OUTER_DIMS + 1];
    int n_lens = 0;
    stride_lens[n_lens++] = intermediate_size;
    collect_outer_strides(args.output_strides, args.shapes, outer_ndim, stride_lens, n_lens);
    collect_outer_strides(args.gate_up_strides, args.shapes, outer_ndim, stride_lens, n_lens);

    DISPATCH_FLOAT_DTYPE(gate_up.dtype(), "silu_and_mul_split", {
        SplitFwdLauncher<scalar_t> launcher{
            static_cast<scalar_t*>(output.data_ptr()),
            static_cast<const scalar_t*>(gate_up.data_ptr()),
            &args, num_rows, intermediate_size, stream};
        const int pack = choose_pack<scalar_t>(
            {output.data_ptr(), gate_up.data_ptr()},
            stride_lens, n_lens);
        dispatch_pack<scalar_t>(pack, launcher);
    });
    sync_check_cuda_error();
}

// SiluAndMulSplitInplace: fused split + silu + mul, inplace.
// gate_up has shape [..., 2*N], last dim must be contiguous and even.
// After this call, gate_up[..., :N] = silu(gate_up[..., :N]) * gate_up[..., N:].
// Reuses the split kernel with output == gate_up (same base, same strides).
void SiluAndMulSplitInplace(wallx_cuda::TensorView gate_up)
{
    const int ndim = gate_up.ndim();
    ASSERT_CHECK(ndim >= 1);
    ASSERT_CHECK(ndim - 1 <= MAX_OUTER_DIMS);
    ASSERT_CHECK(safe_stride(gate_up, ndim - 1) == 1);

    const int64_t last_dim = gate_up.size(ndim - 1);
    ASSERT_CHECK(last_dim % 2 == 0);
    const int64_t intermediate_size = last_dim / 2;

    const int outer_ndim = ndim - 1;
    SiluMulSplitArgs args{};
    args.outer_ndim = outer_ndim;
    int64_t num_rows = 1;
    for (int d = 0; d < outer_ndim; ++d) {
        args.output_strides[d]  = safe_stride(gate_up, d);
        args.gate_up_strides[d] = safe_stride(gate_up, d);
        args.shapes[d]          = gate_up.size(d);
        num_rows *= gate_up.size(d);
    }
    if (num_rows == 0 || intermediate_size == 0) return;

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(gate_up.device().device_type, gate_up.device().device_id));

    int64_t stride_lens[MAX_OUTER_DIMS + 1];
    int n_lens = 0;
    stride_lens[n_lens++] = intermediate_size;
    collect_outer_strides(args.gate_up_strides, args.shapes, outer_ndim, stride_lens, n_lens);

    DISPATCH_FLOAT_DTYPE(gate_up.dtype(), "silu_and_mul_split_inplace", {
        SplitFwdLauncher<scalar_t> launcher{
            static_cast<scalar_t*>(gate_up.data_ptr()),
            static_cast<const scalar_t*>(gate_up.data_ptr()),
            &args, num_rows, intermediate_size, stream};
        const int pack = choose_pack<scalar_t>(
            {gate_up.data_ptr()},
            stride_lens, n_lens);
        dispatch_pack<scalar_t>(pack, launcher);
    });
    sync_check_cuda_error();
}

void SiluAndMulSplitInplaceBackward(
    wallx_cuda::TensorView grad_gate_up,
    wallx_cuda::TensorView grad_output,
    wallx_cuda::TensorView gate_up)
{
    const int ndim = gate_up.ndim();
    ASSERT_CHECK(ndim >= 1);
    ASSERT_CHECK(ndim == grad_gate_up.ndim());
    ASSERT_CHECK(ndim == grad_output.ndim());
    ASSERT_CHECK(ndim - 1 <= MAX_OUTER_DIMS);

    ASSERT_CHECK(safe_stride(grad_gate_up, ndim - 1) == 1);
    ASSERT_CHECK(safe_stride(grad_output, ndim - 1) == 1);
    ASSERT_CHECK(safe_stride(gate_up, ndim - 1) == 1);

    const int64_t last_dim = gate_up.size(ndim - 1);
    ASSERT_CHECK(last_dim % 2 == 0);
    const int64_t intermediate_size = last_dim / 2;

    for (int d = 0; d < ndim - 1; ++d) {
        ASSERT_CHECK(grad_gate_up.size(d) == gate_up.size(d));
        ASSERT_CHECK(grad_output.size(d) == gate_up.size(d));
    }
    ASSERT_CHECK(grad_gate_up.size(ndim - 1) == last_dim);
    ASSERT_CHECK(grad_output.size(ndim - 1) == intermediate_size);

    ASSERT_CHECK(grad_gate_up.dtype().code == gate_up.dtype().code);
    ASSERT_CHECK(grad_gate_up.dtype().bits == gate_up.dtype().bits);
    ASSERT_CHECK(grad_output.dtype().code == gate_up.dtype().code);
    ASSERT_CHECK(grad_output.dtype().bits == gate_up.dtype().bits);

    const int outer_ndim = ndim - 1;
    SiluMulSplitBackwardArgs args{};
    args.outer_ndim = outer_ndim;
    int64_t num_rows = 1;
    for (int d = 0; d < outer_ndim; ++d) {
        args.grad_gate_up_strides[d] = safe_stride(grad_gate_up, d);
        args.grad_output_strides[d] = safe_stride(grad_output, d);
        args.gate_up_strides[d] = safe_stride(gate_up, d);
        args.shapes[d] = gate_up.size(d);
        num_rows *= gate_up.size(d);
    }
    if (num_rows == 0 || intermediate_size == 0) return;

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(gate_up.device().device_type, gate_up.device().device_id));

    int64_t stride_lens[3 * MAX_OUTER_DIMS + 1];
    int n_lens = 0;
    stride_lens[n_lens++] = intermediate_size;
    collect_outer_strides(args.grad_gate_up_strides, args.shapes, outer_ndim, stride_lens, n_lens);
    collect_outer_strides(args.grad_output_strides, args.shapes, outer_ndim, stride_lens, n_lens);
    collect_outer_strides(args.gate_up_strides, args.shapes, outer_ndim, stride_lens, n_lens);

    DISPATCH_FLOAT_DTYPE(gate_up.dtype(), "silu_and_mul_split_inplace_backward", {
        SplitBwdLauncher<scalar_t> launcher{
            static_cast<scalar_t*>(grad_gate_up.data_ptr()),
            static_cast<const scalar_t*>(grad_output.data_ptr()),
            static_cast<const scalar_t*>(gate_up.data_ptr()),
            &args, num_rows, intermediate_size, stream};
        const int pack = choose_pack<scalar_t>(
            {grad_gate_up.data_ptr(), grad_output.data_ptr(), gate_up.data_ptr()},
            stride_lens, n_lens);
        dispatch_pack<scalar_t>(pack, launcher);
    });
    sync_check_cuda_error();
}

void SiluAndMulBackward(
    wallx_cuda::TensorView grad_input1,
    wallx_cuda::TensorView grad_input2,
    wallx_cuda::TensorView grad_output,
    wallx_cuda::TensorView input1,
    wallx_cuda::TensorView input2)
{
    ASSERT_CHECK(grad_input1.IsContiguous());
    ASSERT_CHECK(grad_input2.IsContiguous());
    ASSERT_CHECK(grad_output.IsContiguous());
    ASSERT_CHECK(input1.IsContiguous());
    ASSERT_CHECK(input2.IsContiguous());

    ASSERT_CHECK(input1.ndim() == input2.ndim());
    ASSERT_CHECK(input1.ndim() == grad_output.ndim());
    ASSERT_CHECK(input1.ndim() == grad_input1.ndim());
    ASSERT_CHECK(input1.ndim() == grad_input2.ndim());
    for (int i = 0; i < input1.ndim(); ++i) {
        ASSERT_CHECK(input1.size(i) == input2.size(i));
        ASSERT_CHECK(input1.size(i) == grad_output.size(i));
        ASSERT_CHECK(input1.size(i) == grad_input1.size(i));
        ASSERT_CHECK(input1.size(i) == grad_input2.size(i));
    }

    ASSERT_CHECK(grad_input1.dtype().code == input1.dtype().code);
    ASSERT_CHECK(grad_input1.dtype().bits == input1.dtype().bits);
    ASSERT_CHECK(grad_input2.dtype().code == input1.dtype().code);
    ASSERT_CHECK(grad_input2.dtype().bits == input1.dtype().bits);
    ASSERT_CHECK(grad_output.dtype().code == input1.dtype().code);
    ASSERT_CHECK(grad_output.dtype().bits == input1.dtype().bits);
    ASSERT_CHECK(input2.dtype().code == input1.dtype().code);
    ASSERT_CHECK(input2.dtype().bits == input1.dtype().bits);

    const int64_t n = get_total_elements(input1);
    if (n == 0) return;
    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(input1.device().device_type, input1.device().device_id));

    DISPATCH_FLOAT_DTYPE(input1.dtype(), "silu_and_mul_backward", {
        FlatBwdLauncher<scalar_t> launcher{
            static_cast<scalar_t*>(grad_input1.data_ptr()),
            static_cast<scalar_t*>(grad_input2.data_ptr()),
            static_cast<const scalar_t*>(grad_output.data_ptr()),
            static_cast<const scalar_t*>(input1.data_ptr()),
            static_cast<const scalar_t*>(input2.data_ptr()),
            n, stream};
        const int pack = choose_pack<scalar_t>(
            {grad_input1.data_ptr(), grad_input2.data_ptr(), grad_output.data_ptr(),
             input1.data_ptr(), input2.data_ptr()},
            &n, 1);
        dispatch_pack<scalar_t>(pack, launcher);
    });
    sync_check_cuda_error();
}








}  // namespace wallx_cuda_silu_and_mul
