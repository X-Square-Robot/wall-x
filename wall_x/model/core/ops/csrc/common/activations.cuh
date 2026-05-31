#pragma once

#include <cassert>
#include <cuda_fp16.h>
#include "activation_types.h"

namespace wallx_cuda {

template<typename T, ActivationType activation_type>
__forceinline__ __device__ T applyActivation(const T &x) {
    if constexpr (activation_type == ActivationType::RELU) {
        return x > (T)0.0f ? x : (T)0.0f;
    }
    else if constexpr (activation_type == ActivationType::SILU) {
        return (T)((float)x / (1.0f + __expf((float)-x)));
    }
    else if constexpr (activation_type == ActivationType::GELU) {
        // GELU implementation from vllm (gelu_new_kernel)
        const float x_f = (float)x;
        const float x3 = x_f * x_f * x_f;
        const float t = tanhf(0.79788456f * (x_f + 0.044715f * x3));
        return (T)(0.5f * x_f * (1.0f + t));
    }
    else {
        // No activation matches
        assert(false);
    }
}

}  // namespace wallx_cuda
