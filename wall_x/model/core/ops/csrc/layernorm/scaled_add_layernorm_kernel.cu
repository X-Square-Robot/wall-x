#include "tensor_view.h"
// Fused layerscale-multiply + residual-add + LayerNorm, with backward.
//
//   sum = residual + branch * lambda        (lambda: per-channel [H])
//   y   = LayerNorm(sum) * gamma + beta
//
// Emits BOTH sum (the next residual stream) and y (the normalized branch
// input), replacing an addcmul kernel + a layer_norm kernel + their backward
// chain with one kernel each way. Motivated by the DINOv3 tower (see wall_x
// DinoFusedLayer); same fusion structure as the decoder's add_rmsnorm.
//
// All math is fp32 with a single rounding at each store; mean/rstd are saved
// in fp32 for the backward.
//
// Code paths (host picks per shape/alignment; see LaunchFwd/LaunchBwd):
//
//  * FAST forward (16B-vectorized, one block per row): taken when H is a
//    multiple of the 16-byte pack (8 bf16 / 4 fp32) and every row pointer is
//    16B-aligned. The row lives in registers; the block is sized to the row
//    (ceil(hvec/COLS) threads, COLS in {1,2,4} chosen so the block is
//    launchable). At the production shape ([26624,1280] bf16) this runs at
//    the measured 2-read/2-write streaming ceiling of both sm80 and sm90.
//
//  * FAST backward, cp.async pipeline (primary): rows_per_block rows per
//    block, one pack column per thread (hvec <= blockDim.x <= 384). The four
//    [N,H] input streams of the NEXT row are staged into shared memory with
//    cp.async while the current row computes and block-reduces, so the memory
//    pipe never drains across the per-row barrier -- without cp.async the
//    barrier caps sustained DRAM bandwidth ~6-8% below the streaming ceiling
//    on both sm80 and sm90. Each thread copies and reads only its own 16B
//    chunks, so stage rotation needs no extra barriers (one syncthreads per
//    row, for the c1/c2 reduction, with parity-double-buffered warp partials
//    that every thread sums itself in identical order). mean/rstd for the
//    block's rows are preloaded to smem once. dlambda/dgamma/dbeta partials
//    accumulate in per-thread registers across rows and are flushed once per
//    block through smem so the global fp32 atomics are warp-coalesced
//    (atomicAdd on the register [VEC] layout makes every lane hit a
//    32B-strided address -- 32 L2 transactions per instruction -- and
//    serializes the grid on the tiny [H] buffers; measured 2.4x slowdown).
//
//  * FAST backward, register variant (fallback for hvec > 384 or COLS > 1):
//    same scheme minus the cp.async staging; phase-2 operands are prefetched
//    into registers behind the reduction instead.
//
//  * GENERIC fwd/bwd (T2-vectorized): previous implementation, kept for odd
//    H / unaligned rows. Backward accumulates parameter grads in shared
//    memory (v1 measured 68 us/call from per-row atomic contention).
//
// Parameter-grad summation order remains nondeterministic across blocks, same
// class as aten's atomic norm backwards.
//
// Tuning knobs (env, read per call): WALLX_SALN_FWD_COLS,
// WALLX_SALN_BWD_COLS, WALLX_SALN_BWD_ROWS, WALLX_SALN_BWD_PIPE=0.
// Defaults: BWD_ROWS = 10 on sm90+, 12 on sm80 (swept on H100/A800).


#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdlib>
#include "../common/cuda_utils.h"
#include "../common/reduction.cuh"

namespace wallx_cuda_scaled_add_layernorm {

constexpr int WARP_SIZE = 32;
constexpr int NUM_THREADS = 256;      // generic-path block size
constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
constexpr int GENERIC_ROWS_PER_BLOCK = 8;
// Register pressure scales with COLS (per-thread arrays are [COLS][VEC]);
// beyond 4 the fast path would spill and the generic path is no worse.
constexpr int MAX_COLS = 4;
// Pipeline path: 2 stages x 4 streams x NT x 16B of dynamic smem must stay
// within the default 48KB block limit, and mean/rstd smem holds <= 64 rows.
constexpr int PIPE_MAX_THREADS = 384;
constexpr int PIPE_MAX_ROWS = 64;

struct RowShape {
  int64_t num_tokens;
  int64_t hidden_size;
};

inline RowShape GetRowShape(const wallx_cuda::TensorView& t) {
  const int ndim = t.ndim();
  ASSERT_CHECK(ndim >= 1);
  const int64_t hidden_size = t.size(ndim - 1);
  ASSERT_CHECK(hidden_size > 0);
  ASSERT_CHECK(hidden_size % 2 == 0);  // T2-vectorized kernels
  int64_t num_tokens = 1;
  for (int i = 0; i < ndim - 1; ++i) num_tokens *= t.size(i);
  ASSERT_CHECK(num_tokens > 0);
  int64_t expect = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    ASSERT_CHECK(t.stride(i) == expect);
    expect *= t.size(i);
  }
  return {num_tokens, hidden_size};
}

inline bool SameDtype(const wallx_cuda::TensorView& a,
                      const wallx_cuda::TensorView& b) {
  return a.dtype().code == b.dtype().code && a.dtype().bits == b.dtype().bits &&
         a.dtype().lanes == b.dtype().lanes;
}

// Per-row stats (mean/rstd) buffers: dense 1D fp32 of exactly num_tokens.
inline void CheckRowStats(const wallx_cuda::TensorView& t, int64_t num_tokens) {
  ASSERT_CHECK(t.dtype().code == kDLFloat && t.dtype().bits == 32);
  ASSERT_CHECK(t.ndim() == 1);
  ASSERT_CHECK(t.size(0) == num_tokens);
  ASSERT_CHECK(t.stride(0) == 1);
}

// Per-channel params (lambda/gamma/beta and their grads): kernels index them
// flat, so require dense layout and exactly hidden_size elements.
inline void CheckDenseParam(const wallx_cuda::TensorView& t, int64_t hidden_size) {
  int64_t numel = 1, expect = 1;
  for (int i = t.ndim() - 1; i >= 0; --i) {
    ASSERT_CHECK(t.stride(i) == expect);
    expect *= t.size(i);
    numel *= t.size(i);
  }
  ASSERT_CHECK(numel == hidden_size);
}

template<typename T>
struct Vec2Of {
  using type = float2;
};
template<> struct Vec2Of<half> { using type = half2; };
template<> struct Vec2Of<__nv_bfloat16> { using type = __nv_bfloat162; };

// 16-byte pack for the fast paths.
template<typename T>
struct alignas(16) Pack16 {
  static constexpr int kElems = 16 / sizeof(T);
  T data[kElems];
};

// Block-wide sum of two accumulators; returns via s_out[0..1] (tid 0 writes).
// Fixed NUM_THREADS-block version used by the generic kernels.
__device__ __forceinline__ void blockReduce2(
    float& a, float& b, float wksp[2][NUM_WARPS]) {
  #pragma unroll
  for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
    a += __shfl_down_sync(0xffffffff, a, offset);
    b += __shfl_down_sync(0xffffffff, b, offset);
  }
  if ((threadIdx.x & 31) == 0) {
    wksp[0][threadIdx.x >> 5] = a;
    wksp[1][threadIdx.x >> 5] = b;
  }
  __syncthreads();
  if (threadIdx.x < NUM_WARPS) {
    a = wksp[0][threadIdx.x];
    b = wksp[1][threadIdx.x];
  } else {
    a = 0.f; b = 0.f;
  }
  #pragma unroll
  for (int offset = NUM_WARPS/2; offset > 0; offset /= 2) {
    a += __shfl_down_sync(0xffffffff, a, offset);
    b += __shfl_down_sync(0xffffffff, b, offset);
  }
}

// Same, for a runtime block size (any multiple of 32, <= 1024). The block-sum
// lands in thread 0's (a, b); other lanes are garbage.
__device__ __forceinline__ void blockReduce2Dyn(
    float& a, float& b, float wksp[2][32]) {
  #pragma unroll
  for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
    a += __shfl_down_sync(0xffffffff, a, offset);
    b += __shfl_down_sync(0xffffffff, b, offset);
  }
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int nwarps = (blockDim.x + 31) >> 5;
  if (lane == 0) {
    wksp[0][warp] = a;
    wksp[1][warp] = b;
  }
  __syncthreads();
  if (warp == 0) {
    a = (lane < nwarps) ? wksp[0][lane] : 0.f;
    b = (lane < nwarps) ? wksp[1][lane] : 0.f;
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
      a += __shfl_down_sync(0xffffffff, a, offset);
      b += __shfl_down_sync(0xffffffff, b, offset);
    }
  }
}

// Per-row block reduction used by the fast backward kernels: warp-reduce,
// lane 0 posts the warp partial to the parity buffer, ONE barrier, then every
// thread sums the partials itself (identical order => identical value on all
// threads, so no broadcast barrier). Parity double-buffering makes the next
// row's posts race-free against this row's reads.
__device__ __forceinline__ void rowReduce2(
    float& c1, float& c2, float (*wk)[32], const float inv_n) {
  #pragma unroll
  for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
    c1 += __shfl_down_sync(0xffffffff, c1, offset);
    c2 += __shfl_down_sync(0xffffffff, c2, offset);
  }
  if ((threadIdx.x & 31) == 0) {
    wk[0][threadIdx.x >> 5] = c1;
    wk[1][threadIdx.x >> 5] = c2;
  }
  __syncthreads();
  const int nwarps = (blockDim.x + 31) >> 5;
  float m1 = 0.f, m2 = 0.f;
  for (int w = 0; w < nwarps; ++w) {
    m1 += wk[0][w];
    m2 += wk[1][w];
  }
  c1 = m1 * inv_n;
  c2 = m2 * inv_n;
}

// Coalesced flush of the per-thread [VEC] parameter-grad accumulators: stage
// through smem so each warp's atomics cover consecutive floats.
template<int VEC>
__device__ __forceinline__ void flushParamGrads(
    float* s_flush, float* d_lambda, float* d_gamma, float* d_beta,
    const float acc_l[VEC], const float acc_g[VEC], const float acc_b[VEC],
    const int span_begin, const int span_elems, const bool valid) {
  float* outs[3] = {d_lambda, d_gamma, d_beta};
  for (int p = 0; p < 3; ++p) {
    __syncthreads();
    if (valid) {
      #pragma unroll
      for (int e = 0; e < VEC; ++e) {
        s_flush[threadIdx.x * VEC + e] =
            p == 0 ? acc_l[e] : (p == 1 ? acc_g[e] : acc_b[e]);
      }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < span_elems; k += blockDim.x) {
      atomicAdd(&outs[p][span_begin + k], s_flush[k]);
    }
  }
}

// ---------------------------------------------------------------------------
// FAST forward: one block per row, 16B loads/stores, row in registers.
// blockDim.x * COLS >= hvec (host guarantees); thread t owns pack columns
// t, t + blockDim.x, ...
// ---------------------------------------------------------------------------
template<typename T, int COLS>
__global__ void scaledAddLayernormFwdFastKernel(
    T* __restrict__ sum_out,        // [N, H]
    T* __restrict__ y_out,          // [N, H]
    float* __restrict__ mean_out,   // [N]
    float* __restrict__ rstd_out,   // [N]
    const T* __restrict__ residual, // [N, H]
    const T* __restrict__ branch,   // [N, H]
    const T* __restrict__ lambda_,  // [H]
    const T* __restrict__ gamma,    // [H]
    const T* __restrict__ beta,     // [H]
    const float epsilon,
    const int hvec,                 // H / kElems
    const float inv_n               // 1 / H
) {
    using P = Pack16<T>;
    constexpr int VEC = P::kElems;
    __shared__ float s_mr[2];
    __shared__ float wksp[2][32];

    const int64_t rowbase = (int64_t)blockIdx.x * hvec;
    const P* res = (const P*)residual + rowbase;
    const P* br  = (const P*)branch + rowbase;
    const P* lam = (const P*)lambda_;
    P* so = (P*)sum_out + rowbase;
    P* yo = (P*)y_out + rowbase;

    float s[COLS][VEC];
    float mean = 0.f, var = 0.f;
    #pragma unroll
    for (int c = 0; c < COLS; ++c) {
        const int i = c * blockDim.x + threadIdx.x;
        if (i < hvec) {
            const P r_ = res[i];
            const P b_ = br[i];
            const P l_ = lam[i];
            P o;
            #pragma unroll
            for (int e = 0; e < VEC; ++e) {
                const float v =
                    (float)r_.data[e] + (float)b_.data[e] * (float)l_.data[e];
                s[c][e] = v;
                o.data[e] = (T)v;
                mean += v;
                var += v * v;
            }
            so[i] = o;
        }
    }

    blockReduce2Dyn(mean, var, wksp);
    if (threadIdx.x == 0) {
        const float m = mean * inv_n;
        const float r = rsqrtf(var * inv_n - m * m + epsilon);
        s_mr[0] = m;
        s_mr[1] = r;
        mean_out[blockIdx.x] = m;
        rstd_out[blockIdx.x] = r;
    }
    __syncthreads();

    const float m = s_mr[0], r = s_mr[1];
    #pragma unroll
    for (int c = 0; c < COLS; ++c) {
        const int i = c * blockDim.x + threadIdx.x;
        if (i < hvec) {
            const P g_ = ((const P*)gamma)[i];
            const P be_ = ((const P*)beta)[i];
            P o;
            #pragma unroll
            for (int e = 0; e < VEC; ++e) {
                o.data[e] = (T)(
                    (s[c][e] - m) * r * (float)g_.data[e] + (float)be_.data[e]);
            }
            yo[i] = o;
        }
    }
}

// ---------------------------------------------------------------------------
// FAST backward, cp.async pipeline (primary path; see file header).
// Backward math:
//   xhat  = (sum - mean) * rstd
//   dyhat = dy * gamma
//   dsum_ln = rstd * (dyhat - mean_H(dyhat) - xhat * mean_H(dyhat * xhat))
//   dtotal  = dsum_ln + d_sum_ext
//   d_residual = dtotal ; d_branch = dtotal * lambda
//   dlambda += dtotal * branch ; dgamma += dy * xhat ; dbeta += dy
// ---------------------------------------------------------------------------
template<typename T>
__global__ void scaledAddLayernormBwdFastPipeKernel(
    T* __restrict__ d_residual,      // [N, H]
    T* __restrict__ d_branch,        // [N, H]
    float* __restrict__ d_lambda,    // [H] pre-zeroed
    float* __restrict__ d_gamma,     // [H] pre-zeroed
    float* __restrict__ d_beta,      // [H] pre-zeroed
    const T* __restrict__ dy,        // [N, H]
    const T* __restrict__ d_sum_ext, // [N, H]
    const T* __restrict__ sum_in,    // [N, H] saved forward output
    const T* __restrict__ branch,    // [N, H] saved forward input
    const T* __restrict__ lambda_,   // [H]
    const T* __restrict__ gamma,     // [H]
    const float* __restrict__ mean_in, // [N]
    const float* __restrict__ rstd_in, // [N]
    const int64_t num_tokens,
    const int hvec,                  // H / kElems
    const float inv_n,               // 1 / H
    const int rows_per_block         // <= PIPE_MAX_ROWS, <= blockDim.x
) {
    using P = Pack16<T>;
    constexpr int VEC = P::kElems;
    __shared__ float wksp[2][2][32];        // [row parity][c1,c2][warp]
    __shared__ float s_mr[2][PIPE_MAX_ROWS];
    extern __shared__ char smraw[];         // [2 stages][4 streams][NT] packs
    P* stages = (P*)smraw;

    const int nt = blockDim.x;
    const int i = threadIdx.x;
    const bool valid = i < hvec;
    float acc_l[VEC], acc_g[VEC], acc_b[VEC];
    #pragma unroll
    for (int e = 0; e < VEC; ++e) {
        acc_l[e] = 0.f; acc_g[e] = 0.f; acc_b[e] = 0.f;
    }

    const int64_t row_begin = (int64_t)blockIdx.x * rows_per_block;
    const int64_t row_end =
        min(row_begin + (int64_t)rows_per_block, num_tokens);
    const int nrows = (int)(row_end - row_begin);
    if (i < nrows) {
        s_mr[0][i] = mean_in[row_begin + i];
        s_mr[1][i] = rstd_in[row_begin + i];
    }

    // Thread t copies and later reads only its own chunks, so stage rotation
    // needs no barrier; per-thread cp.async group ordering is enough.
    auto issue = [&](int64_t row, int st) {
        if (row < row_end && valid) {
            const int64_t b0 = row * hvec + i;
            P* dst = stages + (size_t)st * 4 * nt;
            __pipeline_memcpy_async(&dst[i],        &((const P*)sum_in)[b0], 16);
            __pipeline_memcpy_async(&dst[nt + i],   &((const P*)dy)[b0], 16);
            __pipeline_memcpy_async(&dst[2*nt + i], &((const P*)d_sum_ext)[b0], 16);
            __pipeline_memcpy_async(&dst[3*nt + i], &((const P*)branch)[b0], 16);
        }
        __pipeline_commit();  // uniform per-thread group count either way
    };
    issue(row_begin, 0);
    __syncthreads();  // s_mr visibility

    int st = 0;
    for (int64_t row = row_begin; row < row_end; ++row, st ^= 1) {
        issue(row + 1, st ^ 1);       // next row streams behind the barrier
        __pipeline_wait_prior(1);
        const int ri = (int)(row - row_begin);
        const float m = s_mr[0][ri];
        const float r = s_mr[1][ri];
        const P* cur = stages + (size_t)st * 4 * nt;

        // Phase 1: gamma is not carried across the barrier and xhat is not
        // kept in registers; both are re-derived in phase 2 (gamma from L1,
        // xhat from the smem stage) -- the ~12 freed registers buy an extra
        // resident block per SM, which matters more than the recompute.
        P dyp;
        float c1 = 0.f, c2 = 0.f;
        if (valid) {
            const P g_ = ((const P*)gamma)[i];
            const P s_ = cur[i];
            dyp = cur[nt + i];
            #pragma unroll
            for (int e = 0; e < VEC; ++e) {
                const float x = ((float)s_.data[e] - m) * r;
                const float dh = (float)dyp.data[e] * (float)g_.data[e];
                c1 += dh;
                c2 += dh * x;
            }
        }

        rowReduce2(c1, c2, wksp[row & 1], inv_n);
        const float mc1 = c1, mc2 = c2;

        if (valid) {
            const P g_ = ((const P*)gamma)[i];
            const P l_ = ((const P*)lambda_)[i];
            const P s_ = cur[i];
            const P dep = cur[2*nt + i];
            const P brp = cur[3*nt + i];
            const int64_t b0 = row * hvec + i;
            P odr, odb;
            #pragma unroll
            for (int e = 0; e < VEC; ++e) {
                const float xh = ((float)s_.data[e] - m) * r;
                const float dyf = (float)dyp.data[e];
                const float dh = dyf * (float)g_.data[e];
                const float dt =
                    r * (dh - mc1 - xh * mc2) + (float)dep.data[e];
                odr.data[e] = (T)dt;
                odb.data[e] = (T)(dt * (float)l_.data[e]);
                acc_l[e] += dt * (float)brp.data[e];
                acc_g[e] += dyf * xh;
                acc_b[e] += dyf;
            }
            ((P*)d_residual)[b0] = odr;
            ((P*)d_branch)[b0] = odb;
        }
    }

    flushParamGrads<VEC>((float*)smraw, d_lambda, d_gamma, d_beta,
                         acc_l, acc_g, acc_b,
                         0, min(nt * VEC, hvec * VEC), valid);
}

// ---------------------------------------------------------------------------
// FAST backward, register variant (fallback: COLS > 1 or hvec > NT limit).
// Thread t owns fixed pack columns t, t + blockDim.x, ... for every row, so
// dlambda/dgamma/dbeta accumulate in registers; phase-2 operands are
// prefetched into registers so their latency hides behind the reduction.
// lambda/gamma are NOT kept in registers across rows: every block reads the
// same [H] vectors, so per-row reloads are L1 hits, and the freed registers
// buy an extra resident block per SM.
// ---------------------------------------------------------------------------
template<typename T, int COLS>
__global__ void scaledAddLayernormBwdFastKernel(
    T* __restrict__ d_residual,      // [N, H]
    T* __restrict__ d_branch,        // [N, H]
    float* __restrict__ d_lambda,    // [H] pre-zeroed
    float* __restrict__ d_gamma,     // [H] pre-zeroed
    float* __restrict__ d_beta,      // [H] pre-zeroed
    const T* __restrict__ dy,        // [N, H]
    const T* __restrict__ d_sum_ext, // [N, H]
    const T* __restrict__ sum_in,    // [N, H] saved forward output
    const T* __restrict__ branch,    // [N, H] saved forward input
    const T* __restrict__ lambda_,   // [H]
    const T* __restrict__ gamma,     // [H]
    const float* __restrict__ mean_in, // [N]
    const float* __restrict__ rstd_in, // [N]
    const int64_t num_tokens,
    const int hvec,                  // H / kElems
    const float inv_n,               // 1 / H
    const int rows_per_block
) {
    using P = Pack16<T>;
    constexpr int VEC = P::kElems;
    __shared__ float wksp[2][2][32];

    int idx[COLS];
    bool valid[COLS];
    float acc_l[COLS][VEC], acc_g[COLS][VEC], acc_b[COLS][VEC];
    #pragma unroll
    for (int c = 0; c < COLS; ++c) {
        idx[c] = c * blockDim.x + threadIdx.x;
        valid[c] = idx[c] < hvec;
        #pragma unroll
        for (int e = 0; e < VEC; ++e) {
            acc_l[c][e] = 0.f; acc_g[c][e] = 0.f; acc_b[c][e] = 0.f;
        }
    }

    const int64_t row_begin = (int64_t)blockIdx.x * rows_per_block;
    const int64_t row_end =
        min(row_begin + (int64_t)rows_per_block, num_tokens);

    for (int64_t row = row_begin; row < row_end; ++row) {
        const int64_t rowbase = row * hvec;
        const float m = mean_in[row];
        const float r = rstd_in[row];

        float xhat[COLS][VEC];
        P dyp[COLS], dep[COLS], brp[COLS];
        float c1 = 0.f, c2 = 0.f;
        #pragma unroll
        for (int c = 0; c < COLS; ++c) {
            if (valid[c]) {
                const P s_ = ((const P*)sum_in)[rowbase + idx[c]];
                dyp[c] = ((const P*)dy)[rowbase + idx[c]];
                dep[c] = ((const P*)d_sum_ext)[rowbase + idx[c]];
                brp[c] = ((const P*)branch)[rowbase + idx[c]];
                const P g_ = ((const P*)gamma)[idx[c]];
                #pragma unroll
                for (int e = 0; e < VEC; ++e) {
                    const float xh = ((float)s_.data[e] - m) * r;
                    const float dh = (float)dyp[c].data[e] * (float)g_.data[e];
                    xhat[c][e] = xh;
                    c1 += dh;
                    c2 += dh * xh;
                }
            }
        }

        rowReduce2(c1, c2, wksp[row & 1], inv_n);
        const float mc1 = c1, mc2 = c2;

        #pragma unroll
        for (int c = 0; c < COLS; ++c) {
            if (valid[c]) {
                const P g_ = ((const P*)gamma)[idx[c]];
                const P l_ = ((const P*)lambda_)[idx[c]];
                P odr, odb;
                #pragma unroll
                for (int e = 0; e < VEC; ++e) {
                    const float dyf = (float)dyp[c].data[e];
                    const float dh = dyf * (float)g_.data[e];
                    const float dt =
                        r * (dh - mc1 - xhat[c][e] * mc2) + (float)dep[c].data[e];
                    odr.data[e] = (T)dt;
                    odb.data[e] = (T)(dt * (float)l_.data[e]);
                    acc_l[c][e] += dt * (float)brp[c].data[e];
                    acc_g[c][e] += dyf * xhat[c][e];
                    acc_b[c][e] += dyf;
                }
                ((P*)d_residual)[rowbase + idx[c]] = odr;
                ((P*)d_branch)[rowbase + idx[c]] = odb;
            }
        }
    }

    extern __shared__ float s_flush[];  // [blockDim.x * VEC]
    #pragma unroll
    for (int c = 0; c < COLS; ++c) {
        const int span_begin = c * blockDim.x * VEC;
        const int span_elems =
            min((int)blockDim.x * VEC, hvec * VEC - span_begin);
        if (span_elems <= 0) break;
        flushParamGrads<VEC>(s_flush, d_lambda, d_gamma, d_beta,
                             acc_l[c], acc_g[c], acc_b[c],
                             span_begin, span_elems, valid[c]);
    }
}

// ---------------------------------------------------------------------------
// GENERIC forward: one block per row; the fp32 sum row is staged in shared
// memory so the normalize pass rereads it at full precision.
// ---------------------------------------------------------------------------
template<typename T>
__global__ void scaledAddLayernormFwdKernel(
    T* __restrict__ sum_out,        // [N, H]
    T* __restrict__ y_out,          // [N, H]
    float* __restrict__ mean_out,   // [N]
    float* __restrict__ rstd_out,   // [N]
    const T* __restrict__ residual, // [N, H]
    const T* __restrict__ branch,   // [N, H]
    const T* __restrict__ lambda_,  // [H]
    const T* __restrict__ gamma,    // [H]
    const T* __restrict__ beta,     // [H]
    const float epsilon,
    const int64_t hidden_size
) {
    using T2 = typename Vec2Of<T>::type;
    extern __shared__ float s_sum_row[];  // [H] fp32
    __shared__ float s_mean, s_rstd;
    static __shared__ float wksp[2][NUM_WARPS];

    const int64_t h2 = hidden_size / 2;
    const int64_t base2 = blockIdx.x * h2;
    const T2* res2 = (const T2*)residual;
    const T2* br2  = (const T2*)branch;
    const T2* lam2 = (const T2*)lambda_;
    const T2* gam2 = (const T2*)gamma;
    const T2* bet2 = (const T2*)beta;
    T2* sum2 = (T2*)sum_out;
    T2* y2   = (T2*)y_out;

    float mean = 0.f, var = 0.f;
    for (int64_t i = threadIdx.x; i < h2; i += blockDim.x) {
        const T2 r_ = res2[base2 + i];
        const T2 b_ = br2[base2 + i];
        const T2 l_ = lam2[i];
        const float s0 = (float)r_.x + (float)b_.x * (float)l_.x;
        const float s1 = (float)r_.y + (float)b_.y * (float)l_.y;
        s_sum_row[2*i]   = s0;
        s_sum_row[2*i+1] = s1;
        T2 out;
        out.x = (decltype(out.x))s0;
        out.y = (decltype(out.y))s1;
        sum2[base2 + i] = out;
        mean += s0 + s1;
        var  += s0*s0 + s1*s1;
    }

    blockReduce2(mean, var, wksp);
    if (threadIdx.x == 0) {
        const float n = (float)hidden_size;
        const float m = mean / n;
        s_mean = m;
        s_rstd = rsqrtf(var / n - m * m + epsilon);
        mean_out[blockIdx.x] = s_mean;
        rstd_out[blockIdx.x] = s_rstd;
    }
    __syncthreads();

    const float m = s_mean, r = s_rstd;
    for (int64_t i = threadIdx.x; i < h2; i += blockDim.x) {
        const T2 g_ = gam2[i];
        const T2 be_ = bet2[i];
        T2 out;
        out.x = (decltype(out.x))(
            (s_sum_row[2*i] - m) * r * (float)g_.x + (float)be_.x);
        out.y = (decltype(out.y))(
            (s_sum_row[2*i+1] - m) * r * (float)g_.y + (float)be_.y);
        y2[base2 + i] = out;
    }
}

// ---------------------------------------------------------------------------
// GENERIC backward: GENERIC_ROWS_PER_BLOCK rows per block, parameter-grad
// partials in shared memory (same math as the pipeline kernel).
// ---------------------------------------------------------------------------
template<typename T>
__global__ void scaledAddLayernormBwdKernel(
    T* __restrict__ d_residual,      // [N, H]
    T* __restrict__ d_branch,        // [N, H]
    float* __restrict__ d_lambda,    // [H] pre-zeroed
    float* __restrict__ d_gamma,     // [H] pre-zeroed
    float* __restrict__ d_beta,      // [H] pre-zeroed
    const T* __restrict__ dy,        // [N, H]
    const T* __restrict__ d_sum_ext, // [N, H]
    const T* __restrict__ sum_in,    // [N, H] saved forward output
    const T* __restrict__ branch,    // [N, H] saved forward input
    const T* __restrict__ lambda_,   // [H]
    const T* __restrict__ gamma,     // [H]
    const float* __restrict__ mean_in, // [N]
    const float* __restrict__ rstd_in, // [N]
    const int64_t num_tokens,
    const int64_t hidden_size
) {
    using T2 = typename Vec2Of<T>::type;
    // layout: xhat[H] | dyhat[H] | acc_dlambda[H] | acc_dgamma[H] | acc_dbeta[H]
    extern __shared__ float smem[];
    float* s_xhat  = smem;
    float* s_dyhat = smem + hidden_size;
    float* s_dlam  = smem + 2 * hidden_size;
    float* s_dgam  = smem + 3 * hidden_size;
    float* s_dbet  = smem + 4 * hidden_size;
    __shared__ float s_c1, s_c2;
    static __shared__ float wksp[2][NUM_WARPS];

    for (int64_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        s_dlam[i] = 0.f; s_dgam[i] = 0.f; s_dbet[i] = 0.f;
    }

    const int64_t h2 = hidden_size / 2;
    const T2* dy2v   = (const T2*)dy;
    const T2* dext2v = (const T2*)d_sum_ext;
    const T2* sum2v  = (const T2*)sum_in;
    const T2* br2v   = (const T2*)branch;
    const T2* lam2v  = (const T2*)lambda_;
    const T2* gam2v  = (const T2*)gamma;
    T2* dres2v = (T2*)d_residual;
    T2* dbr2v  = (T2*)d_branch;

    const int64_t row_begin = (int64_t)blockIdx.x * GENERIC_ROWS_PER_BLOCK;
    const int64_t row_end =
        min(row_begin + (int64_t)GENERIC_ROWS_PER_BLOCK, num_tokens);

    for (int64_t row = row_begin; row < row_end; ++row) {
        const int64_t base2 = row * h2;
        const float m = mean_in[row];
        const float r = rstd_in[row];
        __syncthreads();  // protect s_xhat/s_dyhat/wksp reuse across rows

        float c1 = 0.f, c2 = 0.f;
        for (int64_t i = threadIdx.x; i < h2; i += blockDim.x) {
            const T2 s_ = sum2v[base2 + i];
            const T2 d_ = dy2v[base2 + i];
            const T2 g_ = gam2v[i];
            const float xh0 = ((float)s_.x - m) * r;
            const float xh1 = ((float)s_.y - m) * r;
            const float dh0 = (float)d_.x * (float)g_.x;
            const float dh1 = (float)d_.y * (float)g_.y;
            s_xhat[2*i]    = xh0;
            s_xhat[2*i+1]  = xh1;
            s_dyhat[2*i]   = dh0;
            s_dyhat[2*i+1] = dh1;
            c1 += dh0 + dh1;
            c2 += dh0 * xh0 + dh1 * xh1;
        }

        blockReduce2(c1, c2, wksp);
        if (threadIdx.x == 0) {
            const float n = (float)hidden_size;
            s_c1 = c1 / n;
            s_c2 = c2 / n;
        }
        __syncthreads();

        const float mc1 = s_c1, mc2 = s_c2;
        for (int64_t i = threadIdx.x; i < h2; i += blockDim.x) {
            const T2 de_ = dext2v[base2 + i];
            const T2 l_  = lam2v[i];
            const T2 b_  = br2v[base2 + i];
            const T2 d_  = dy2v[base2 + i];
            const float dt0 =
                r * (s_dyhat[2*i] - mc1 - s_xhat[2*i] * mc2) + (float)de_.x;
            const float dt1 =
                r * (s_dyhat[2*i+1] - mc1 - s_xhat[2*i+1] * mc2) + (float)de_.y;
            T2 o;
            o.x = (decltype(o.x))dt0;
            o.y = (decltype(o.y))dt1;
            dres2v[base2 + i] = o;
            o.x = (decltype(o.x))(dt0 * (float)l_.x);
            o.y = (decltype(o.y))(dt1 * (float)l_.y);
            dbr2v[base2 + i] = o;
            s_dlam[2*i]   += dt0 * (float)b_.x;
            s_dlam[2*i+1] += dt1 * (float)b_.y;
            s_dgam[2*i]   += (float)d_.x * s_xhat[2*i];
            s_dgam[2*i+1] += (float)d_.y * s_xhat[2*i+1];
            s_dbet[2*i]   += (float)d_.x;
            s_dbet[2*i+1] += (float)d_.y;
        }
    }
    __syncthreads();

    for (int64_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        atomicAdd(&d_lambda[i], s_dlam[i]);
        atomicAdd(&d_gamma[i], s_dgam[i]);
        atomicAdd(&d_beta[i], s_dbet[i]);
    }
}

// ---------------------------------------------------------------------------
// host launchers
// ---------------------------------------------------------------------------

inline int SmMajor() {
    static thread_local int cached_dev = -1;
    static thread_local int cached_major = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev != cached_dev) {
        cudaDeviceGetAttribute(&cached_major, cudaDevAttrComputeCapabilityMajor, dev);
        cached_dev = dev;
    }
    return cached_major;
}

inline int EnvInt(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return v ? std::atoi(v) : fallback;
}

inline bool Aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

// Per-instantiation launchable block-size limit (register-file bound; e.g.
// the bf16 COLS=2 backward cannot launch 1024 threads). Queried once.
inline int FuncMaxThreads(const void* func) {
    cudaFuncAttributes attr;
    if (cudaFuncGetAttributes(&attr, func) != cudaSuccess) return 0;
    return attr.maxThreadsPerBlock;
}

// Smallest COLS in {1,2,4} whose block (a 32-multiple covering hvec) is
// launchable for that instantiation. max_threads[c] is indexed by log2(COLS).
inline bool PickFastCfg(int hvec, int cols_override, const int max_threads[3],
                        int* cols, int* nthreads) {
    for (int c = 1, lg = 0; c <= MAX_COLS; c *= 2, ++lg) {
        if (cols_override > 0 && c != cols_override) continue;
        const int per_col = (hvec + c - 1) / c;
        const int nt = ((per_col + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        if (nt <= max_threads[lg] && nt <= 1024) {
            *cols = c;
            *nthreads = nt;
            return true;
        }
    }
    return false;
}

template<typename T>
void LaunchFwd(
    wallx_cuda::TensorView sum_out, wallx_cuda::TensorView y_out,
    wallx_cuda::TensorView mean_out, wallx_cuda::TensorView rstd_out,
    wallx_cuda::TensorView residual, wallx_cuda::TensorView branch,
    wallx_cuda::TensorView lambda_, wallx_cuda::TensorView gamma,
    wallx_cuda::TensorView beta, float epsilon,
    int64_t num_tokens, int64_t hidden_size, cudaStream_t stream
) {
    constexpr int kElems = Pack16<T>::kElems;
    const bool fast_ok =
        hidden_size % kElems == 0 &&
        Aligned16(residual.data_ptr()) && Aligned16(branch.data_ptr()) &&
        Aligned16(lambda_.data_ptr()) && Aligned16(gamma.data_ptr()) &&
        Aligned16(beta.data_ptr()) &&
        Aligned16(sum_out.data_ptr()) && Aligned16(y_out.data_ptr());
    static const int kMaxThreads[3] = {
        FuncMaxThreads((const void*)scaledAddLayernormFwdFastKernel<T, 1>),
        FuncMaxThreads((const void*)scaledAddLayernormFwdFastKernel<T, 2>),
        FuncMaxThreads((const void*)scaledAddLayernormFwdFastKernel<T, 4>),
    };
    int cols = 0, nthreads = 0;
    if (fast_ok &&
        PickFastCfg((int)(hidden_size / kElems),
                    EnvInt("WALLX_SALN_FWD_COLS", 0), kMaxThreads,
                    &cols, &nthreads)) {
        const int hvec = (int)(hidden_size / kElems);
        const float inv_n = 1.f / (float)hidden_size;
        #define XC_SALN_FWD_CASE(C)                                          \
            scaledAddLayernormFwdFastKernel<T, C>                            \
                <<<dim3(num_tokens), dim3(nthreads), 0, stream>>>(           \
                static_cast<T*>(sum_out.data_ptr()),                         \
                static_cast<T*>(y_out.data_ptr()),                           \
                static_cast<float*>(mean_out.data_ptr()),                    \
                static_cast<float*>(rstd_out.data_ptr()),                    \
                static_cast<const T*>(residual.data_ptr()),                  \
                static_cast<const T*>(branch.data_ptr()),                    \
                static_cast<const T*>(lambda_.data_ptr()),                   \
                static_cast<const T*>(gamma.data_ptr()),                     \
                static_cast<const T*>(beta.data_ptr()),                      \
                epsilon, hvec, inv_n)
        switch (cols) {
            case 1: XC_SALN_FWD_CASE(1); break;
            case 2: XC_SALN_FWD_CASE(2); break;
            case 4: XC_SALN_FWD_CASE(4); break;
        }
        #undef XC_SALN_FWD_CASE
        return;
    }
    scaledAddLayernormFwdKernel<T>
        <<<dim3(num_tokens), dim3(NUM_THREADS), hidden_size * sizeof(float), stream>>>(
        static_cast<T*>(sum_out.data_ptr()),
        static_cast<T*>(y_out.data_ptr()),
        static_cast<float*>(mean_out.data_ptr()),
        static_cast<float*>(rstd_out.data_ptr()),
        static_cast<const T*>(residual.data_ptr()),
        static_cast<const T*>(branch.data_ptr()),
        static_cast<const T*>(lambda_.data_ptr()),
        static_cast<const T*>(gamma.data_ptr()),
        static_cast<const T*>(beta.data_ptr()),
        epsilon, hidden_size);
}

void ScaledAddLayernorm(
    wallx_cuda::TensorView sum_out,
    wallx_cuda::TensorView y_out,
    wallx_cuda::TensorView mean_out,
    wallx_cuda::TensorView rstd_out,
    wallx_cuda::TensorView residual,
    wallx_cuda::TensorView branch,
    wallx_cuda::TensorView lambda_,
    wallx_cuda::TensorView gamma,
    wallx_cuda::TensorView beta,
    double epsilon
) {
    const RowShape shape = GetRowShape(residual);
    const RowShape b_shape = GetRowShape(branch);
    const RowShape s_shape = GetRowShape(sum_out);
    const RowShape y_shape = GetRowShape(y_out);
    ASSERT_CHECK(b_shape.num_tokens == shape.num_tokens &&
                 b_shape.hidden_size == shape.hidden_size);
    ASSERT_CHECK(s_shape.num_tokens == shape.num_tokens &&
                 s_shape.hidden_size == shape.hidden_size);
    ASSERT_CHECK(y_shape.num_tokens == shape.num_tokens &&
                 y_shape.hidden_size == shape.hidden_size);
    // residual.dtype() drives every static_cast<T*> below; all activations
    // and all per-channel params must share it or the kernel would silently
    // reinterpret memory at the wrong element width.
    ASSERT_CHECK(SameDtype(branch, residual) && SameDtype(sum_out, residual) &&
                 SameDtype(y_out, residual));
    ASSERT_CHECK(SameDtype(lambda_, residual) && SameDtype(gamma, residual) &&
                 SameDtype(beta, residual));
    CheckDenseParam(lambda_, shape.hidden_size);
    CheckDenseParam(gamma, shape.hidden_size);
    CheckDenseParam(beta, shape.hidden_size);
    CheckRowStats(mean_out, shape.num_tokens);
    CheckRowStats(rstd_out, shape.num_tokens);

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(residual.device().device_type, residual.device().device_id));

    const auto dt = residual.dtype();
    if (dt.code == kDLFloat && dt.bits == 32) {
        LaunchFwd<float>(sum_out, y_out, mean_out, rstd_out, residual, branch,
                         lambda_, gamma, beta, (float)epsilon,
                         shape.num_tokens, shape.hidden_size, stream);
    } else if (dt.code == kDLBfloat && dt.bits == 16) {
        LaunchFwd<__nv_bfloat16>(sum_out, y_out, mean_out, rstd_out, residual,
                                 branch, lambda_, gamma, beta, (float)epsilon,
                                 shape.num_tokens, shape.hidden_size, stream);
    } else {
        throw std::runtime_error("scaled_add_layernorm: only fp32/bf16 supported");
    }
    sync_check_cuda_error();
}

template<typename T>
void LaunchBwd(
    wallx_cuda::TensorView d_residual, wallx_cuda::TensorView d_branch,
    wallx_cuda::TensorView d_lambda, wallx_cuda::TensorView d_gamma,
    wallx_cuda::TensorView d_beta, wallx_cuda::TensorView dy,
    wallx_cuda::TensorView d_sum_ext, wallx_cuda::TensorView sum_in,
    wallx_cuda::TensorView branch, wallx_cuda::TensorView lambda_,
    wallx_cuda::TensorView gamma, wallx_cuda::TensorView mean_in,
    wallx_cuda::TensorView rstd_in,
    int64_t num_tokens, int64_t hidden_size, cudaStream_t stream
) {
    constexpr int kElems = Pack16<T>::kElems;
    const bool fast_ok =
        hidden_size % kElems == 0 &&
        Aligned16(d_residual.data_ptr()) && Aligned16(d_branch.data_ptr()) &&
        Aligned16(dy.data_ptr()) && Aligned16(d_sum_ext.data_ptr()) &&
        Aligned16(sum_in.data_ptr()) && Aligned16(branch.data_ptr()) &&
        Aligned16(lambda_.data_ptr()) && Aligned16(gamma.data_ptr());
    static const int kMaxThreads[3] = {
        FuncMaxThreads((const void*)scaledAddLayernormBwdFastKernel<T, 1>),
        FuncMaxThreads((const void*)scaledAddLayernormBwdFastKernel<T, 2>),
        FuncMaxThreads((const void*)scaledAddLayernormBwdFastKernel<T, 4>),
    };
    int cols = 0, nthreads = 0;
    if (fast_ok &&
        PickFastCfg((int)(hidden_size / kElems),
                    EnvInt("WALLX_SALN_BWD_COLS", 0), kMaxThreads,
                    &cols, &nthreads)) {
        // Swept on H100 (sm90) / A800 (sm80) at the production shape.
        const int default_rows = SmMajor() >= 9 ? 10 : 12;
        // atoi garbage/"0" would divide-by-zero in the row partitioning.
        const int rows_env = EnvInt("WALLX_SALN_BWD_ROWS", default_rows);
        const int rows = rows_env < 1 ? default_rows : rows_env;
        const int hvec = (int)(hidden_size / kElems);
        const float inv_n = 1.f / (float)hidden_size;

        static const int kPipeMaxThreads = FuncMaxThreads(
            (const void*)scaledAddLayernormBwdFastPipeKernel<T>);
        if (cols == 1 && nthreads <= PIPE_MAX_THREADS &&
            nthreads <= kPipeMaxThreads &&
            EnvInt("WALLX_SALN_BWD_PIPE", 1) != 0) {
            const int prows = min(min(rows, PIPE_MAX_ROWS), nthreads);
            const int64_t num_blocks = (num_tokens + prows - 1) / prows;
            const size_t pipe_smem = 2 * 4 * (size_t)nthreads * sizeof(Pack16<T>);
            scaledAddLayernormBwdFastPipeKernel<T>
                <<<dim3(num_blocks), dim3(nthreads), pipe_smem, stream>>>(
                static_cast<T*>(d_residual.data_ptr()),
                static_cast<T*>(d_branch.data_ptr()),
                static_cast<float*>(d_lambda.data_ptr()),
                static_cast<float*>(d_gamma.data_ptr()),
                static_cast<float*>(d_beta.data_ptr()),
                static_cast<const T*>(dy.data_ptr()),
                static_cast<const T*>(d_sum_ext.data_ptr()),
                static_cast<const T*>(sum_in.data_ptr()),
                static_cast<const T*>(branch.data_ptr()),
                static_cast<const T*>(lambda_.data_ptr()),
                static_cast<const T*>(gamma.data_ptr()),
                static_cast<const float*>(mean_in.data_ptr()),
                static_cast<const float*>(rstd_in.data_ptr()),
                num_tokens, hvec, inv_n, prows);
            return;
        }

        const int64_t num_blocks = (num_tokens + rows - 1) / rows;
        const size_t flush_smem = (size_t)nthreads * kElems * sizeof(float);
        #define XC_SALN_BWD_CASE(C)                                          \
            scaledAddLayernormBwdFastKernel<T, C>                            \
                <<<dim3(num_blocks), dim3(nthreads), flush_smem, stream>>>(  \
                static_cast<T*>(d_residual.data_ptr()),                      \
                static_cast<T*>(d_branch.data_ptr()),                        \
                static_cast<float*>(d_lambda.data_ptr()),                    \
                static_cast<float*>(d_gamma.data_ptr()),                     \
                static_cast<float*>(d_beta.data_ptr()),                      \
                static_cast<const T*>(dy.data_ptr()),                        \
                static_cast<const T*>(d_sum_ext.data_ptr()),                 \
                static_cast<const T*>(sum_in.data_ptr()),                    \
                static_cast<const T*>(branch.data_ptr()),                    \
                static_cast<const T*>(lambda_.data_ptr()),                   \
                static_cast<const T*>(gamma.data_ptr()),                     \
                static_cast<const float*>(mean_in.data_ptr()),               \
                static_cast<const float*>(rstd_in.data_ptr()),               \
                num_tokens, hvec, inv_n, rows)
        switch (cols) {
            case 1: XC_SALN_BWD_CASE(1); break;
            case 2: XC_SALN_BWD_CASE(2); break;
            case 4: XC_SALN_BWD_CASE(4); break;
        }
        #undef XC_SALN_BWD_CASE
        return;
    }
    const int64_t num_blocks =
        (num_tokens + GENERIC_ROWS_PER_BLOCK - 1) / GENERIC_ROWS_PER_BLOCK;
    scaledAddLayernormBwdKernel<T>
        <<<dim3(num_blocks), dim3(NUM_THREADS),
           5 * hidden_size * sizeof(float), stream>>>(
        static_cast<T*>(d_residual.data_ptr()),
        static_cast<T*>(d_branch.data_ptr()),
        static_cast<float*>(d_lambda.data_ptr()),
        static_cast<float*>(d_gamma.data_ptr()),
        static_cast<float*>(d_beta.data_ptr()),
        static_cast<const T*>(dy.data_ptr()),
        static_cast<const T*>(d_sum_ext.data_ptr()),
        static_cast<const T*>(sum_in.data_ptr()),
        static_cast<const T*>(branch.data_ptr()),
        static_cast<const T*>(lambda_.data_ptr()),
        static_cast<const T*>(gamma.data_ptr()),
        static_cast<const float*>(mean_in.data_ptr()),
        static_cast<const float*>(rstd_in.data_ptr()),
        num_tokens, hidden_size);
}

void ScaledAddLayernormBackward(
    wallx_cuda::TensorView d_residual,
    wallx_cuda::TensorView d_branch,
    wallx_cuda::TensorView d_lambda,
    wallx_cuda::TensorView d_gamma,
    wallx_cuda::TensorView d_beta,
    wallx_cuda::TensorView dy,
    wallx_cuda::TensorView d_sum_ext,
    wallx_cuda::TensorView sum_in,
    wallx_cuda::TensorView branch,
    wallx_cuda::TensorView lambda_,
    wallx_cuda::TensorView gamma,
    wallx_cuda::TensorView mean_in,
    wallx_cuda::TensorView rstd_in
) {
    const RowShape shape = GetRowShape(dy);
    // Every activation-sized operand must agree with dy in shape and dtype
    // (dy.dtype() drives the static_cast<T*> dispatch).
    for (const wallx_cuda::TensorView* t :
         {&d_residual, &d_branch, &d_sum_ext, &sum_in, &branch}) {
        const RowShape ts = GetRowShape(*t);
        ASSERT_CHECK(ts.num_tokens == shape.num_tokens &&
                     ts.hidden_size == shape.hidden_size);
        ASSERT_CHECK(SameDtype(*t, dy));
    }
    ASSERT_CHECK(SameDtype(lambda_, dy) && SameDtype(gamma, dy));
    CheckDenseParam(lambda_, shape.hidden_size);
    CheckDenseParam(gamma, shape.hidden_size);
    ASSERT_CHECK(d_lambda.dtype().code == kDLFloat && d_lambda.dtype().bits == 32);
    ASSERT_CHECK(d_gamma.dtype().code == kDLFloat && d_gamma.dtype().bits == 32);
    ASSERT_CHECK(d_beta.dtype().code == kDLFloat && d_beta.dtype().bits == 32);
    CheckDenseParam(d_lambda, shape.hidden_size);
    CheckDenseParam(d_gamma, shape.hidden_size);
    CheckDenseParam(d_beta, shape.hidden_size);
    CheckRowStats(mean_in, shape.num_tokens);
    CheckRowStats(rstd_in, shape.num_tokens);

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(dy.device().device_type, dy.device().device_id));

    const auto dt = dy.dtype();
    if (dt.code == kDLFloat && dt.bits == 32) {
        LaunchBwd<float>(d_residual, d_branch, d_lambda, d_gamma, d_beta, dy,
                         d_sum_ext, sum_in, branch, lambda_, gamma, mean_in,
                         rstd_in, shape.num_tokens, shape.hidden_size, stream);
    } else if (dt.code == kDLBfloat && dt.bits == 16) {
        LaunchBwd<__nv_bfloat16>(d_residual, d_branch, d_lambda, d_gamma,
                                 d_beta, dy, d_sum_ext, sum_in, branch,
                                 lambda_, gamma, mean_in, rstd_in,
                                 shape.num_tokens, shape.hidden_size, stream);
    } else {
        throw std::runtime_error(
            "scaled_add_layernorm_backward: only fp32/bf16 supported");
    }
    sync_check_cuda_error();
}




}  // namespace wallx_cuda_scaled_add_layernorm
