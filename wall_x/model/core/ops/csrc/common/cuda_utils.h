// Re-enable CUDA half operators (PyTorch disables them)
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__

#pragma once

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CUDA_CHECK(cmd) do { \
    cudaError_t result = cmd; \
    if (result != cudaSuccess) { \
        printf("[ERROR] CUDA error %s:%d '%s': (%d) %s\n", __FILE__, __LINE__, #cmd, (int)result, cudaGetErrorString(result)); \
        exit(-1); \
    } \
} while(0)

inline void syncAndCheck(const char* const file, int const line, bool force_check = false) {
#ifdef DEBUG
    force_check = true;
#endif
    if (force_check) {
        cudaDeviceSynchronize();
        cudaError_t result = cudaGetLastError();
        if (result) {
            throw std::runtime_error(std::string("[ST] CUDA runtime error: ") + cudaGetErrorString(result) + " "
                                    + file + ":" + std::to_string(line) + " \n");
        }
    }
}

#define sync_check_cuda_error() syncAndCheck(__FILE__, __LINE__, false)
#define sync_check_cuda_error_force() syncAndCheck(__FILE__, __LINE__, true)

// #ifdef DEBUG
#define ASSERT_CHECK(__cond)                         \
    do {                                             \
        const bool __cond_var = (__cond);            \
        if (!__cond_var) {                           \
            ::std::string __err_msg =                \
                ::std::string("`") + #__cond +       \
                "` check failed at " +               \
                __FILE__ + ":" +                     \
                ::std::to_string(__LINE__);          \
            throw std::runtime_error(__err_msg);     \
        }                                            \
    } while (0)
// #else
//     #define ASSERT_CHECK(__cond) do { } while (0)
// #endif

// Some stuff for indexing into an 1-D array
#define INDEX_2D(dim1, dim2, index1, index2) \
    (((int64_t)index1) * (dim2) + (index2))
#define INDEX_3D(dim1, dim2, dim3, index1, index2, index3) \
    (((int64_t)index1) * (dim2) * (dim3) + ((int64_t)index2) * (dim3) + (index3))
#define INDEX_4D(dim1, dim2, dim3, dim4, index1, index2, index3, index4) \
    (((int64_t)index1) * (dim2) * (dim3) * (dim4) + ((int64_t)index2) * (dim3) * (dim4) + ((int64_t)index3) * (dim4) + (index4))
#define INDEX_5D(dim1, dim2, dim3, dim4, dim5, index1, index2, index3, index4, index5) \
    (((int64_t)index1) * (dim2) * (dim3) * (dim4) * (dim5) + ((int64_t)index2) * (dim3) * (dim4) * (dim5) + ((int64_t)index3) * (dim4) * (dim5) + (index4) * (dim5) + (index5))

template<typename T>
struct VecTraits;

template<>
struct VecTraits<float> {
    using Type = float4;
    static constexpr int vec_size = 4;
    __device__ static inline Type load(const float* p) { return *reinterpret_cast<const float4*>(p); }
    __device__ static inline void store(float* p, const Type& v) { *reinterpret_cast<float4*>(p) = v; }
};

template<>
struct VecTraits<__half> {
    using Type = __half2;
    static constexpr int vec_size = 2;
    __device__ static inline Type load(const __half* p) { return *reinterpret_cast<const __half2*>(p); }
    __device__ static inline void store(__half* p, const Type& v) { *reinterpret_cast<__half2*>(p) = v; }
};

template<>
struct VecTraits<__nv_bfloat16> {
    using Type = __nv_bfloat162;
    static constexpr int vec_size = 2;
    __device__ static inline Type load(const __nv_bfloat16* p) { return *reinterpret_cast<const __nv_bfloat162*>(p); }
    __device__ static inline void store(__nv_bfloat16* p, const Type& v) { *reinterpret_cast<__nv_bfloat162*>(p) = v; }
};

template<typename T>
cudaDataType_t getCudaDataType() {
    if (std::is_same<T, half>::value) {
        return CUDA_R_16F;
    }
    else if (std::is_same<T, __nv_bfloat16>::value) {
        return CUDA_R_16BF;
    }
    else if (std::is_same<T, float>::value) {
        return CUDA_R_32F;
    }
    else {
        throw std::runtime_error("Cuda data type: Unsupported type");
    }
}
