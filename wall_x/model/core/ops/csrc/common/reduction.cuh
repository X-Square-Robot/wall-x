#pragma once

namespace wallx_cuda {

static const float HALF_FLT_MAX = 65504.F;
#define FINAL_MASK 0xffffffffu

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffffu, val, mask, 32);
    return val;
}

template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];
    int64_t lane = threadIdx.x & 0x1fu;
    int64_t wid = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);
    return val;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

template<typename T>
__inline__ __device__ T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    int64_t lane = threadIdx.x & 0x1fu;
    int64_t wid  = threadIdx.x >> 5;

    val = warpReduceMax(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}

}  // namespace wallx_cuda
