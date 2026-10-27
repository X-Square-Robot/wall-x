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

// Forward activation and its derivative computed together so the costly
// transcendental (tanhf for GELU, __expf for SILU) is evaluated once per
// element and shared. Returns both as fp32 — caller does the chain-rule
// multiply in high precision before casting back to T.
//
// Used by activ_and_mul backward kernel:
//     dL/d(gate) = grad * up * dact(gate)
//     dL/d(up)   = grad * act(gate)
// — both consume act / dact evaluated at the same gate element.
//
// Numerics: GELU forward formula matches applyActivation byte-for-byte
// (same x³-first ordering for the tanh argument). GELU' uses the standard
// chain-rule expansion of the tanh approximation:
//     dGELU/dx = 0.5·(1+t) + 0.5·x·(1-t²)·(√(2/π) + √(2/π)·3·0.044715·x²)
// where 0.1070322243 = √(2/π) · 3 · 0.044715 (pre-multiplied).
struct ActAndDeriv {
    float act;
    float dact;
};

template<typename T, ActivationType activation_type>
__forceinline__ __device__ ActAndDeriv applyActivationAndDerivative(const T &x) {
    const float x_f = (float)x;
    ActAndDeriv r;
    if constexpr (activation_type == ActivationType::RELU) {
        r.act  = x_f > 0.0f ? x_f : 0.0f;
        r.dact = x_f > 0.0f ? 1.0f : 0.0f;
    }
    else if constexpr (activation_type == ActivationType::SILU) {
        const float s = 1.0f / (1.0f + __expf(-x_f));
        r.act  = x_f * s;
        r.dact = s * (1.0f + x_f * (1.0f - s));
    }
    else if constexpr (activation_type == ActivationType::GELU) {
        const float x3 = x_f * x_f * x_f;
        const float t  = tanhf(0.79788456f * (x_f + 0.044715f * x3));
        const float x2 = x_f * x_f;
        r.act  = 0.5f * x_f * (1.0f + t);
        r.dact = 0.5f * x_f * ((1.0f - t * t) * (0.79788456f + 0.1070322243f * x2))
               + 0.5f * (1.0f + t);
    }
    else {
        assert(false);
    }
    return r;
}

}  // namespace wallx_cuda
