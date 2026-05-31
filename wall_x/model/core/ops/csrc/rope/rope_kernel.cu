// Re-enable CUDA half operators (PyTorch disables them)
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_fp16.h>
 #include <cassert>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include "../common/cuda_utils.h"

namespace wallx_cuda_rope {

// Helper: convert vector of T to float4
template<typename T>
__device__ __forceinline__ float4 to_float4(const T* ptr) {
    if constexpr (std::is_same_v<T, half>) {
        half2 h2_0 = reinterpret_cast<const half2*>(ptr)[0];
        half2 h2_1 = reinterpret_cast<const half2*>(ptr)[1];
        float2 f2_0 = __half22float2(h2_0);
        float2 f2_1 = __half22float2(h2_1);
        return make_float4(f2_0.x, f2_0.y, f2_1.x, f2_1.y);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        __nv_bfloat162 b2_0 = reinterpret_cast<const __nv_bfloat162*>(ptr)[0];
        __nv_bfloat162 b2_1 = reinterpret_cast<const __nv_bfloat162*>(ptr)[1];
#if __CUDA_ARCH__ >= 800
        float2 f2_0 = __bfloat1622float2(b2_0);
        float2 f2_1 = __bfloat1622float2(b2_1);
        return make_float4(f2_0.x, f2_0.y, f2_1.x, f2_1.y);
#else
        assert(false && "Unsupported on arch < 800");
#endif
    } else if constexpr (std::is_same_v<T, float>) {
        return reinterpret_cast<const float4*>(ptr)[0];
    } else {
        static_assert(sizeof(T) == 0, "Unsupported type");
    }
}

// Helper: convert float4 back to vector of T
template<typename T>
__device__ __forceinline__ void from_float4(T* ptr, const float4& f4) {
    if constexpr (std::is_same_v<T, half>) {
        half2 h2_0 = __float22half2_rn(make_float2(f4.x, f4.y));
        half2 h2_1 = __float22half2_rn(make_float2(f4.z, f4.w));
        reinterpret_cast<half2*>(ptr)[0] = h2_0;
        reinterpret_cast<half2*>(ptr)[1] = h2_1;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
#if __CUDA_ARCH__ >= 800
        __nv_bfloat162 b2_0 = __float22bfloat162_rn(make_float2(f4.x, f4.y));
        __nv_bfloat162 b2_1 = __float22bfloat162_rn(make_float2(f4.z, f4.w));
        reinterpret_cast<__nv_bfloat162*>(ptr)[0] = b2_0;
        reinterpret_cast<__nv_bfloat162*>(ptr)[1] = b2_1;
#else
        assert(false && "Unsupported on arch < 800");
#endif
    } else if constexpr (std::is_same_v<T, float>) {
        reinterpret_cast<float4*>(ptr)[0] = f4;
    } else {
        static_assert(sizeof(T) == 0, "Unsupported type");
    }
}

template<class T, bool interleave>
__global__ void RopeKernel(const float* cos,
                           const float* sin,
                           const T* q,
                           const int q_h,
                           const T* k,
                           const int k_h,
                           T* q_embed,
                           T* k_embed,
                           const int d,
                           const int qb_stride,
                           const int qs_stride,
                           const int qh_stride,
                           const int kb_stride,
                           const int ks_stride,
                           const int kh_stride,
                           const int qeb_stride,
                           const int qes_stride,
                           const int qeh_stride,
                           const int keb_stride,
                           const int kes_stride,
                           const int keh_stride) {
    extern __shared__ char cos_sin[];
    const int half_dim = d / 2;
    float* cos_smem = reinterpret_cast<float*>(cos_sin);
    float* sin_smem = cos_smem + half_dim;
    int b = blockIdx.x;
    int s = blockIdx.y;

    int64_t cos_sin_b_stride = gridDim.y * half_dim;
    int64_t cos_sin_s_stride = half_dim;
#define SIN_GMEM(b, c, d) sin[(b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]
#define COS_GMEM(b, c, d) cos[(b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        cos_smem[i] = COS_GMEM(b, s, i);
        sin_smem[i] = SIN_GMEM(b, s, i);
    }
    __syncthreads();

#define Q_GMEM(a, b, c, d) q[(a) * qb_stride + (b) * qs_stride + (c) * qh_stride + d]
#define K_GMEM(a, b, c, d) k[(a) * kb_stride + (b) * ks_stride + (c) * kh_stride + d]
#define Q_EMBED_GMEM(a, b, c, d) q_embed[(a) * qeb_stride + (b) * qes_stride + (c) * qeh_stride + d]
#define K_EMBED_GMEM(a, b, c, d) k_embed[(a) * keb_stride + (b) * kes_stride + (c) * keh_stride + d]

    for (int j = threadIdx.x * 4; j < q_h * half_dim; j += blockDim.x * 4) {
        int h_idx = j / half_dim;
        int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;

        float4 cos_vec = to_float4<float>(&cos_smem[base_d]);
        float4 sin_vec = to_float4<float>(&sin_smem[base_d]);

        if constexpr (interleave) {
            float4 q_x0 = to_float4<T>(&Q_GMEM(b, s, h_idx, 2 * base_d));
            float4 q_x1 = to_float4<T>(&Q_GMEM(b, s, h_idx, 2 * base_d + 4));
            float4 q0_rot = make_float4(
                q_x0.x * cos_vec.x - q_x0.y * sin_vec.x,
                q_x0.y * cos_vec.x + q_x0.x * sin_vec.x,
                q_x0.z * cos_vec.y - q_x0.w * sin_vec.y,
                q_x0.w * cos_vec.y + q_x0.z * sin_vec.y
            );
            float4 q1_rot = make_float4(
                q_x1.x * cos_vec.z - q_x1.y * sin_vec.z,
                q_x1.y * cos_vec.z + q_x1.x * sin_vec.z,
                q_x1.z * cos_vec.w - q_x1.w * sin_vec.w,
                q_x1.w * cos_vec.w + q_x1.z * sin_vec.w
            );

            from_float4<T>(&Q_EMBED_GMEM(b, s, h_idx, 2 * base_d), q0_rot);
            from_float4<T>(&Q_EMBED_GMEM(b, s, h_idx, 2 * base_d + 4), q1_rot);
        } else {
            float4 q_x0 = to_float4<T>(&Q_GMEM(b, s, h_idx, base_d));
            float4 q_x1 = to_float4<T>(&Q_GMEM(b, s, h_idx, base_d + half_dim));
            float4 q0_rot = make_float4(
                q_x0.x * cos_vec.x - q_x1.x * sin_vec.x,
                q_x0.y * cos_vec.y - q_x1.y * sin_vec.y,
                q_x0.z * cos_vec.z - q_x1.z * sin_vec.z,
                q_x0.w * cos_vec.w - q_x1.w * sin_vec.w
            );

            float4 q1_rot = make_float4(
                q_x1.x * cos_vec.x + q_x0.x * sin_vec.x,
                q_x1.y * cos_vec.y + q_x0.y * sin_vec.y,
                q_x1.z * cos_vec.z + q_x0.z * sin_vec.z,
                q_x1.w * cos_vec.w + q_x0.w * sin_vec.w
            );

            from_float4<T>(&Q_EMBED_GMEM(b, s, h_idx, base_d), q0_rot);
            from_float4<T>(&Q_EMBED_GMEM(b, s, h_idx, base_d + half_dim), q1_rot);
        }
    }

    for (int j = threadIdx.x * 4; j < k_h * half_dim; j += blockDim.x * 4) {
        int h_idx = j / half_dim;
        int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;

        float4 cos_vec = to_float4<float>(&cos_smem[base_d]);
        float4 sin_vec = to_float4<float>(&sin_smem[base_d]);
        if constexpr (interleave) {
            float4 k_x0 = to_float4<T>(&K_GMEM(b, s, h_idx, 2 * base_d));
            float4 k_x1 = to_float4<T>(&K_GMEM(b, s, h_idx, 2 * base_d + 4));

            float4 k0_rot = make_float4(
                k_x0.x * cos_vec.x - k_x0.y * sin_vec.x,
                k_x0.y * cos_vec.x + k_x0.x * sin_vec.x,
                k_x0.z * cos_vec.y - k_x0.w * sin_vec.y,
                k_x0.w * cos_vec.y + k_x0.z * sin_vec.y
            );

            float4 k1_rot = make_float4(
                k_x1.x * cos_vec.z - k_x1.y * sin_vec.z,
                k_x1.y * cos_vec.z + k_x1.x * sin_vec.z,
                k_x1.z * cos_vec.w - k_x1.w * sin_vec.w,
                k_x1.w * cos_vec.w + k_x1.z * sin_vec.w
            );

            from_float4<T>(&K_EMBED_GMEM(b, s, h_idx, 2 * base_d), k0_rot);
            from_float4<T>(&K_EMBED_GMEM(b, s, h_idx, 2 * base_d + 4), k1_rot);
        } else {
            float4 k_x0 = to_float4<T>(&K_GMEM(b, s, h_idx, base_d));
            float4 k_x1 = to_float4<T>(&K_GMEM(b, s, h_idx, base_d + half_dim));

            float4 k0_rot = make_float4(
                k_x0.x * cos_vec.x - k_x1.x * sin_vec.x,
                k_x0.y * cos_vec.y - k_x1.y * sin_vec.y,
                k_x0.z * cos_vec.z - k_x1.z * sin_vec.z,
                k_x0.w * cos_vec.w - k_x1.w * sin_vec.w
            );

            float4 k1_rot = make_float4(
                k_x1.x * cos_vec.x + k_x0.x * sin_vec.x,
                k_x1.y * cos_vec.y + k_x0.y * sin_vec.y,
                k_x1.z * cos_vec.z + k_x0.z * sin_vec.z,
                k_x1.w * cos_vec.w + k_x0.w * sin_vec.w
            );

            from_float4<T>(&K_EMBED_GMEM(b, s, h_idx, base_d), k0_rot);
            from_float4<T>(&K_EMBED_GMEM(b, s, h_idx, base_d + half_dim), k1_rot);
        }
    }
}

template<class T, bool interleave>
__global__ void RopeInplaceKernel(float* cos, float* sin,
                                  T* q, const int q_h, T* k, const int k_h,
                                  const int d,
                                  const int qb_stride, const int qs_stride, const int qh_stride,
                                  const int kb_stride, const int ks_stride, const int kh_stride) {
    extern __shared__ char cos_sin[];
    const int half_dim = d / 2;
    float* cos_smem = reinterpret_cast<float*>(cos_sin);
    float* sin_smem = cos_smem + half_dim;
    int b = blockIdx.x;
    int s = blockIdx.y;

    int64_t cos_sin_b_stride = gridDim.y * half_dim;
    int64_t cos_sin_s_stride = half_dim;
#define SIN_GMEM(b, c, d) sin[(b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]
#define COS_GMEM(b, c, d) cos[(b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        cos_smem[i] = COS_GMEM(b, s, i);
        sin_smem[i] = SIN_GMEM(b, s, i);
    }
    __syncthreads();

#define Q_GMEM(a, b, c, d) q[(a) * qb_stride + (b) * qs_stride + (c) * qh_stride + d]
#define K_GMEM(a, b, c, d) k[(a) * kb_stride + (b) * ks_stride + (c) * kh_stride + d]

    for (int j = threadIdx.x * 4; j < q_h * half_dim; j += blockDim.x * 4) {
        int h_idx = j / half_dim;
        int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;

        float4 cos_vec = to_float4<float>(&cos_smem[base_d]);
        float4 sin_vec = to_float4<float>(&sin_smem[base_d]);

        if constexpr (interleave) {
            float4 q_x0 = to_float4<T>(&Q_GMEM(b, s, h_idx, 2 * base_d));
            float4 q_x1 = to_float4<T>(&Q_GMEM(b, s, h_idx, 2 * base_d + 4));
            float4 q0_rot = make_float4(
                q_x0.x * cos_vec.x - q_x0.y * sin_vec.x,
                q_x0.y * cos_vec.x + q_x0.x * sin_vec.x,
                q_x0.z * cos_vec.y - q_x0.w * sin_vec.y,
                q_x0.w * cos_vec.y + q_x0.z * sin_vec.y
            );
            float4 q1_rot = make_float4(
                q_x1.x * cos_vec.z - q_x1.y * sin_vec.z,
                q_x1.y * cos_vec.z + q_x1.x * sin_vec.z,
                q_x1.z * cos_vec.w - q_x1.w * sin_vec.w,
                q_x1.w * cos_vec.w + q_x1.z * sin_vec.w
            );

            from_float4<T>(&Q_GMEM(b, s, h_idx, 2 * base_d), q0_rot);
            from_float4<T>(&Q_GMEM(b, s, h_idx, 2 * base_d + 4), q1_rot);
        } else {
            float4 q_x0 = to_float4<T>(&Q_GMEM(b, s, h_idx, base_d));
            float4 q_x1 = to_float4<T>(&Q_GMEM(b, s, h_idx, base_d + half_dim));
            float4 q0_rot = make_float4(
                q_x0.x * cos_vec.x - q_x1.x * sin_vec.x,
                q_x0.y * cos_vec.y - q_x1.y * sin_vec.y,
                q_x0.z * cos_vec.z - q_x1.z * sin_vec.z,
                q_x0.w * cos_vec.w - q_x1.w * sin_vec.w
            );

            float4 q1_rot = make_float4(
                q_x1.x * cos_vec.x + q_x0.x * sin_vec.x,
                q_x1.y * cos_vec.y + q_x0.y * sin_vec.y,
                q_x1.z * cos_vec.z + q_x0.z * sin_vec.z,
                q_x1.w * cos_vec.w + q_x0.w * sin_vec.w
            );

            from_float4<T>(&Q_GMEM(b, s, h_idx, base_d), q0_rot);
            from_float4<T>(&Q_GMEM(b, s, h_idx, base_d + half_dim), q1_rot);
        }
    }

    for (int j = threadIdx.x * 4; j < k_h * half_dim; j += blockDim.x * 4) {
        int h_idx = j / half_dim;
        int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;

        float4 cos_vec = to_float4<float>(&cos_smem[base_d]);
        float4 sin_vec = to_float4<float>(&sin_smem[base_d]);
        if constexpr (interleave) {
            float4 k_x0 = to_float4<T>(&K_GMEM(b, s, h_idx, 2 * base_d));
            float4 k_x1 = to_float4<T>(&K_GMEM(b, s, h_idx, 2 * base_d + 4));

            float4 k0_rot = make_float4(
                k_x0.x * cos_vec.x - k_x0.y * sin_vec.x,
                k_x0.y * cos_vec.x + k_x0.x * sin_vec.x,
                k_x0.z * cos_vec.y - k_x0.w * sin_vec.y,
                k_x0.w * cos_vec.y + k_x0.z * sin_vec.y
            );

            float4 k1_rot = make_float4(
                k_x1.x * cos_vec.z - k_x1.y * sin_vec.z,
                k_x1.y * cos_vec.z + k_x1.x * sin_vec.z,
                k_x1.z * cos_vec.w - k_x1.w * sin_vec.w,
                k_x1.w * cos_vec.w + k_x1.z * sin_vec.w
            );

            from_float4<T>(&K_GMEM(b, s, h_idx, 2 * base_d), k0_rot);
            from_float4<T>(&K_GMEM(b, s, h_idx, 2 * base_d + 4), k1_rot);
        } else {
            float4 k_x0 = to_float4<T>(&K_GMEM(b, s, h_idx, base_d));
            float4 k_x1 = to_float4<T>(&K_GMEM(b, s, h_idx, base_d + half_dim));

            float4 k0_rot = make_float4(
                k_x0.x * cos_vec.x - k_x1.x * sin_vec.x,
                k_x0.y * cos_vec.y - k_x1.y * sin_vec.y,
                k_x0.z * cos_vec.z - k_x1.z * sin_vec.z,
                k_x0.w * cos_vec.w - k_x1.w * sin_vec.w
            );

            float4 k1_rot = make_float4(
                k_x1.x * cos_vec.x + k_x0.x * sin_vec.x,
                k_x1.y * cos_vec.y + k_x0.y * sin_vec.y,
                k_x1.z * cos_vec.z + k_x0.z * sin_vec.z,
                k_x1.w * cos_vec.w + k_x0.w * sin_vec.w
            );

            from_float4<T>(&K_GMEM(b, s, h_idx, base_d), k0_rot);
            from_float4<T>(&K_GMEM(b, s, h_idx, base_d + half_dim), k1_rot);
        }
    }
}

template<class T, bool interleave>
__global__ void RopeKernelBackward(
    const float* cos,
    const float* sin,
    const T* grad_q_embed,
    const T* grad_k_embed,
    T* grad_q,
    T* grad_k,
    const int q_h,
    const int k_h,
    const int d,
    const int qb_stride, const int qs_stride, const int qh_stride,
    const int kb_stride, const int ks_stride, const int kh_stride
) {
    extern __shared__ char cos_sin[];
    const int half_dim = d / 2;
    float* cos_smem = reinterpret_cast<float*>(cos_sin);
    float* sin_smem = cos_smem + half_dim;

    int b = blockIdx.x;
    int s = blockIdx.y;

    int64_t cos_sin_b_stride = gridDim.y * half_dim;
    int64_t cos_sin_s_stride = half_dim;
#define SIN_GMEM(b, c, d) sin[(b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]
#define COS_GMEM(b, c, d) cos[(b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        cos_smem[i] = COS_GMEM(b, s, i);
        sin_smem[i] = SIN_GMEM(b, s, i);
    }
    __syncthreads();

#define GQ_EMBED(a, b, c, d) grad_q_embed[(a) * qb_stride + (b) * qs_stride + (c) * qh_stride + (d)]
#define GK_EMBED(a, b, c, d) grad_k_embed[(a) * kb_stride + (b) * ks_stride + (c) * kh_stride + (d)]
#define GQ(a, b, c, d) grad_q[(a) * qb_stride + (b) * qs_stride + (c) * qh_stride + (d)]
#define GK(a, b, c, d) grad_k[(a) * kb_stride + (b) * ks_stride + (c) * kh_stride + (d)]

    // Process grad_q
    for (int j = threadIdx.x * 4; j < q_h * half_dim; j += blockDim.x * 4) {
        int h_idx = j / half_dim;
        int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;

        float4 cos_vec = to_float4<float>(&cos_smem[base_d]);
        float4 sin_vec = to_float4<float>(&sin_smem[base_d]);
        if constexpr (interleave) {
            float4 gq0_embed = to_float4<T>(&GQ_EMBED(b, s, h_idx, 2 * base_d));
            float4 gq1_embed = to_float4<T>(&GQ_EMBED(b, s, h_idx, 2 * base_d + 4));

            float4 gq0 = make_float4(
                gq0_embed.x * cos_vec.x + gq0_embed.y * sin_vec.x,
                gq0_embed.y * cos_vec.x - gq0_embed.x * sin_vec.x,
                gq0_embed.z * cos_vec.y + gq0_embed.w * sin_vec.y,
                gq0_embed.w * cos_vec.y - gq0_embed.z * sin_vec.y
            );

            float4 gq1 = make_float4(
                gq1_embed.x * cos_vec.z + gq1_embed.y * sin_vec.z,
                gq1_embed.y * cos_vec.z - gq1_embed.x * sin_vec.z,
                gq1_embed.z * cos_vec.w + gq1_embed.w * sin_vec.w,
                gq1_embed.w * cos_vec.w - gq1_embed.z * sin_vec.w
            );

            from_float4<T>(&GQ(b, s, h_idx, 2 * base_d), gq0);
            from_float4<T>(&GQ(b, s, h_idx, 2 * base_d + 4), gq1);
        } else {
            float4 gq0_rot = to_float4<T>(&GQ_EMBED(b, s, h_idx, base_d));
            float4 gq1_rot = to_float4<T>(&GQ_EMBED(b, s, h_idx, base_d + half_dim));

            float4 gq0 = make_float4(
                gq0_rot.x * cos_vec.x + gq1_rot.x * sin_vec.x,
                gq0_rot.y * cos_vec.y + gq1_rot.y * sin_vec.y,
                gq0_rot.z * cos_vec.z + gq1_rot.z * sin_vec.z,
                gq0_rot.w * cos_vec.w + gq1_rot.w * sin_vec.w
            );
            float4 gq1 = make_float4(
                gq1_rot.x * cos_vec.x - gq0_rot.x * sin_vec.x,
                gq1_rot.y * cos_vec.y - gq0_rot.y * sin_vec.y,
                gq1_rot.z * cos_vec.z - gq0_rot.z * sin_vec.z,
                gq1_rot.w * cos_vec.w - gq0_rot.w * sin_vec.w
            );

            from_float4<T>(&GQ(b, s, h_idx, base_d), gq0);
            from_float4<T>(&GQ(b, s, h_idx, base_d + half_dim), gq1);
        }
    }

    // Process grad_k
    for (int j = threadIdx.x * 4; j < k_h * half_dim; j += blockDim.x * 4) {
        int h_idx = j / half_dim;
        int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;
        float4 cos_vec = to_float4<float>(&cos_smem[base_d]);
        float4 sin_vec = to_float4<float>(&sin_smem[base_d]);
        if constexpr (interleave) {
            float4 gk0_embed = to_float4<T>(&GK_EMBED(b, s, h_idx, 2 * base_d));
            float4 gk1_embed = to_float4<T>(&GK_EMBED(b, s, h_idx, 2 * base_d + 4));

            float4 gk0 = make_float4(
                gk0_embed.x * cos_vec.x + gk0_embed.y * sin_vec.x,
                gk0_embed.y * cos_vec.x - gk0_embed.x * sin_vec.x,
                gk0_embed.z * cos_vec.y + gk0_embed.w * sin_vec.y,
                gk0_embed.w * cos_vec.y - gk0_embed.z * sin_vec.y
            );

            float4 gk1 = make_float4(
                gk1_embed.x * cos_vec.z + gk1_embed.y * sin_vec.z,
                gk1_embed.y * cos_vec.z - gk1_embed.x * sin_vec.z,
                gk1_embed.z * cos_vec.w + gk1_embed.w * sin_vec.w,
                gk1_embed.w * cos_vec.w - gk1_embed.z * sin_vec.w
            );

            from_float4<T>(&GK(b, s, h_idx, 2 * base_d), gk0);
            from_float4<T>(&GK(b, s, h_idx, 2 * base_d + 4), gk1);
        } else {
            float4 gk0_rot = to_float4<T>(&GK_EMBED(b, s, h_idx, base_d));
            float4 gk1_rot = to_float4<T>(&GK_EMBED(b, s, h_idx, base_d + half_dim));

            float4 gk0 = make_float4(
                gk0_rot.x * cos_vec.x + gk1_rot.x * sin_vec.x,
                gk0_rot.y * cos_vec.y + gk1_rot.y * sin_vec.y,
                gk0_rot.z * cos_vec.z + gk1_rot.z * sin_vec.z,
                gk0_rot.w * cos_vec.w + gk1_rot.w * sin_vec.w
            );
            float4 gk1 = make_float4(
                gk1_rot.x * cos_vec.x - gk0_rot.x * sin_vec.x,
                gk1_rot.y * cos_vec.y - gk0_rot.y * sin_vec.y,
                gk1_rot.z * cos_vec.z - gk0_rot.z * sin_vec.z,
                gk1_rot.w * cos_vec.w - gk0_rot.w * sin_vec.w
            );

            from_float4<T>(&GK(b, s, h_idx, base_d), gk0);
            from_float4<T>(&GK(b, s, h_idx, base_d + half_dim), gk1);
        }
    }
}

void Rope(const at::Tensor& q, // [b, s, h, d] or [s, h, d]
          const at::Tensor& k, // [b, s, h_k, d] or [s, h_k, d]
          const at::Tensor& q_embed, // [b, s, h, d] or [s, h, d]
          const at::Tensor& k_embed, // [b, s, h_k, d] or [s, h_k, d]
          const at::Tensor& cos, // [b, s, d / 2] or [s, d / 2]
          const at::Tensor& sin, // [b, s, d / 2] or [s, d / 2]
          bool interleave
          ) {
    int Nthreads = 256;
    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    bool q_3d = (q.dim() == 3);

    ASSERT_CHECK(q.scalar_type() == k.scalar_type());
    ASSERT_CHECK(q.scalar_type() == q_embed.scalar_type());
    ASSERT_CHECK(k.scalar_type() == k_embed.scalar_type());
    ASSERT_CHECK(cos.scalar_type() == at::kFloat);
    ASSERT_CHECK(sin.scalar_type() == at::kFloat);
    ASSERT_CHECK(cos.is_contiguous());
    ASSERT_CHECK(sin.is_contiguous());

    int64_t batch, q_head_num, k_head_num, seq_len, dim;
    int64_t qb_stride, kb_stride, qs_stride, ks_stride, qh_stride, kh_stride;
    int64_t qeb_stride, keb_stride, qes_stride, kes_stride, qeh_stride, keh_stride;

    if (q_3d) {
        ASSERT_CHECK(q_embed.size(0) == q.size(0));
        ASSERT_CHECK(q_embed.size(1) == q.size(1));
        ASSERT_CHECK(q_embed.size(2) == q.size(2));
        ASSERT_CHECK(k_embed.size(0) == k.size(0));
        ASSERT_CHECK(k_embed.size(1) == k.size(1));
        ASSERT_CHECK(k_embed.size(2) == k.size(2));
        ASSERT_CHECK(q.size(2) % 8 == 0);
        ASSERT_CHECK(cos.size(2) == q.size(2) / 2);
        ASSERT_CHECK(q.stride(2) == 1);
        ASSERT_CHECK(k.stride(2) == 1);
        ASSERT_CHECK(q_embed.stride(2) == 1);
        ASSERT_CHECK(k_embed.stride(2) == 1);
        batch = 1;
        seq_len = q.size(0);
        q_head_num = q.size(1);
        k_head_num = k.size(1);
        dim = q.size(2);
        qb_stride = 0; qs_stride = q.stride(0); qh_stride = q.stride(1);
        kb_stride = 0; ks_stride = k.stride(0); kh_stride = k.stride(1);
        qeb_stride = 0; qes_stride = q_embed.stride(0); qeh_stride = q_embed.stride(1);
        keb_stride = 0; kes_stride = k_embed.stride(0); keh_stride = k_embed.stride(1);
    } else {
        ASSERT_CHECK(q_embed.size(0) == q.size(0));
        ASSERT_CHECK(q_embed.size(1) == q.size(1));
        ASSERT_CHECK(q_embed.size(2) == q.size(2));
        ASSERT_CHECK(q_embed.size(3) == q.size(3));
        ASSERT_CHECK(k_embed.size(0) == k.size(0));
        ASSERT_CHECK(k_embed.size(1) == k.size(1));
        ASSERT_CHECK(k_embed.size(2) == k.size(2));
        ASSERT_CHECK(k_embed.size(3) == k.size(3));
        ASSERT_CHECK(q.size(3) % 8 == 0);
        ASSERT_CHECK(cos.size(2) == q.size(3) / 2);
        ASSERT_CHECK(q.stride(3) == 1);
        ASSERT_CHECK(k.stride(3) == 1);
        ASSERT_CHECK(q_embed.stride(3) == 1);
        ASSERT_CHECK(k_embed.stride(3) == 1);
        batch = q.size(0);
        seq_len = q.size(1);
        q_head_num = q.size(2);
        k_head_num = k.size(2);
        dim = q.size(3);
        qb_stride = q.stride(0); qs_stride = q.stride(1); qh_stride = q.stride(2);
        kb_stride = k.stride(0); ks_stride = k.stride(1); kh_stride = k.stride(2);
        qeb_stride = q_embed.stride(0); qes_stride = q_embed.stride(1); qeh_stride = q_embed.stride(2);
        keb_stride = k_embed.stride(0); kes_stride = k_embed.stride(1); keh_stride = k_embed.stride(2);
    }

    dim3 grid(batch, seq_len);

    if (q.scalar_type() == at::kHalf) {
        const half* q_data = static_cast<const half*>(q.data_ptr());
        const half* k_data = static_cast<const half*>(k.data_ptr());
        half* q_embed_data = static_cast<half*>(q_embed.data_ptr());
        half* k_embed_data = static_cast<half*>(k_embed.data_ptr());
        const float* cos_data = static_cast<const float*>(cos.data_ptr());
        const float* sin_data = static_cast<const float*>(sin.data_ptr());
        if (interleave) {
            RopeKernel<half, true><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_head_num,
                k_data, k_head_num,
                q_embed_data, k_embed_data,
                dim,
                qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride,
                qeb_stride, qes_stride, qeh_stride, keb_stride, kes_stride, keh_stride);
        } else {
            RopeKernel<half, false><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_head_num,
                k_data, k_head_num,
                q_embed_data, k_embed_data,
                dim,
                qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride,
                qeb_stride, qes_stride, qeh_stride, keb_stride, kes_stride, keh_stride);
        }
    } else if (q.scalar_type() == at::kBFloat16) {
        const __nv_bfloat16* q_data = static_cast<const __nv_bfloat16*>(q.data_ptr());
        const __nv_bfloat16* k_data = static_cast<const __nv_bfloat16*>(k.data_ptr());
        __nv_bfloat16* q_embed_data = static_cast<__nv_bfloat16*>(q_embed.data_ptr());
        __nv_bfloat16* k_embed_data = static_cast<__nv_bfloat16*>(k_embed.data_ptr());
        const float* cos_data = static_cast<const float*>(cos.data_ptr());
        const float* sin_data = static_cast<const float*>(sin.data_ptr());
        if (interleave) {
            RopeKernel<__nv_bfloat16, true><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_head_num,
                k_data, k_head_num,
                q_embed_data, k_embed_data,
                dim,
                qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride,
                qeb_stride, qes_stride, qeh_stride, keb_stride, kes_stride, keh_stride);
        } else {
            RopeKernel<__nv_bfloat16, false><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_head_num,
                k_data, k_head_num,
                q_embed_data, k_embed_data,
                dim,
                qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride,
                qeb_stride, qes_stride, qeh_stride, keb_stride, kes_stride, keh_stride);
        }
    } else if (q.scalar_type() == at::kFloat) {
        const float* q_data = static_cast<const float*>(q.data_ptr());
        const float* k_data = static_cast<const float*>(k.data_ptr());
        float* q_embed_data = static_cast<float*>(q_embed.data_ptr());
        float* k_embed_data = static_cast<float*>(k_embed.data_ptr());
        const float* cos_data = static_cast<const float*>(cos.data_ptr());
        const float* sin_data = static_cast<const float*>(sin.data_ptr());
        if (interleave) {
            RopeKernel<float, true><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_head_num,
                k_data, k_head_num,
                q_embed_data, k_embed_data,
                dim, qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride,
                qeb_stride, qes_stride, qeh_stride, keb_stride, kes_stride, keh_stride
            );
        } else {
            RopeKernel<float, false><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_head_num,
                k_data, k_head_num,
                q_embed_data, k_embed_data,
                dim, qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride,
                qeb_stride, qes_stride, qeh_stride, keb_stride, kes_stride, keh_stride
            );
        }
    } else {
        throw std::runtime_error("Unsupported data type for rope");
    }

    sync_check_cuda_error();
}

void RopeInplace(const at::Tensor& q, // [b, s, h, d] or [s, h, d]
                 const at::Tensor& k, // [b, s, h_k, d] or [s, h_k, d]
                 const at::Tensor& cos, // [b, s, d / 2] or [s, d / 2]
                 const at::Tensor& sin, // [b, s, d / 2] or [s, d / 2]
                 bool interleave
                 ) {
    int Nthreads = 256;
    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    bool q_3d = (q.dim() == 3);

    ASSERT_CHECK(q.scalar_type() == k.scalar_type());
    ASSERT_CHECK(cos.scalar_type() == at::kFloat);
    ASSERT_CHECK(sin.scalar_type() == at::kFloat);
    ASSERT_CHECK(cos.is_contiguous());
    ASSERT_CHECK(sin.is_contiguous());

    int64_t batch, q_head_num, k_head_num, seq_len, dim;
    int64_t qb_stride, kb_stride, qs_stride, ks_stride, qh_stride, kh_stride;

    if (q_3d) {
        ASSERT_CHECK(q.size(2) % 8 == 0);
        ASSERT_CHECK(cos.size(2) == q.size(2) / 2);
        ASSERT_CHECK(q.stride(2) == 1);
        ASSERT_CHECK(k.stride(2) == 1);
        batch = 1;
        seq_len = q.size(0);
        q_head_num = q.size(1);
        k_head_num = k.size(1);
        dim = q.size(2);
        qb_stride = 0; qs_stride = q.stride(0); qh_stride = q.stride(1);
        kb_stride = 0; ks_stride = k.stride(0); kh_stride = k.stride(1);
    } else {
        ASSERT_CHECK(q.size(3) % 8 == 0);
        ASSERT_CHECK(cos.size(2) == q.size(3) / 2);
        ASSERT_CHECK(q.stride(3) == 1);
        ASSERT_CHECK(k.stride(3) == 1);
        batch = q.size(0);
        seq_len = q.size(1);
        q_head_num = q.size(2);
        k_head_num = k.size(2);
        dim = q.size(3);
        qb_stride = q.stride(0); qs_stride = q.stride(1); qh_stride = q.stride(2);
        kb_stride = k.stride(0); ks_stride = k.stride(1); kh_stride = k.stride(2);
    }

    dim3 grid(batch, seq_len);

    if (q.scalar_type() == at::kHalf) {
        half* q_data = static_cast<half*>(q.data_ptr());
        half* k_data = static_cast<half*>(k.data_ptr());
        float* cos_data = static_cast<float*>(cos.data_ptr());
        float* sin_data = static_cast<float*>(sin.data_ptr());
        if (interleave) {
            RopeInplaceKernel<half, true><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_head_num,
                k_data, k_head_num,
                dim,
                qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride);
        } else {
            RopeInplaceKernel<half, false><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_head_num,
                k_data, k_head_num,
                dim,
                qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride);
        }
    } else if (q.scalar_type() == at::kBFloat16) {
        __nv_bfloat16* q_data = static_cast<__nv_bfloat16*>(q.data_ptr());
        __nv_bfloat16* k_data = static_cast<__nv_bfloat16*>(k.data_ptr());
        float* cos_data = static_cast<float*>(cos.data_ptr());
        float* sin_data = static_cast<float*>(sin.data_ptr());
        if (interleave) {
            RopeInplaceKernel<__nv_bfloat16, true><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_head_num,
                k_data, k_head_num,
                dim,
                qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride);
        } else {
            RopeInplaceKernel<__nv_bfloat16, false><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_head_num,
                k_data, k_head_num,
                dim,
                qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride);
        }
    } else if (q.scalar_type() == at::kFloat) {
        float* q_data = static_cast<float*>(q.data_ptr());
        float* k_data = static_cast<float*>(k.data_ptr());
        float* cos_data = static_cast<float*>(cos.data_ptr());
        float* sin_data = static_cast<float*>(sin.data_ptr());
        if (interleave) {
            RopeInplaceKernel<float, true><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_head_num,
                k_data, k_head_num,
                dim, qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride
            );
        } else {
            RopeInplaceKernel<float, false><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_head_num,
                k_data, k_head_num,
                dim, qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride
            );
        }
    } else {
        throw std::runtime_error("Unsupported data type for rope");
    }

    sync_check_cuda_error();
}

void RopeBackward(
    const at::Tensor& grad_q_embed,
    const at::Tensor& grad_k_embed,
    const at::Tensor& grad_q,    // output: [b, s, q_h, d] or [s, q_h, d]
    const at::Tensor& grad_k,    // output: [b, s, k_h, d] or [s, k_h, d]
    const at::Tensor& cos,       // [b, s, d/2] or [s, d/2]
    const at::Tensor& sin,       // [b, s, d/2] or [s, d/2]
    bool interleave
) {
    int Nthreads = 256;

    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    bool q_3d = (grad_q_embed.dim() == 3);

    ASSERT_CHECK(grad_q_embed.scalar_type() == grad_k_embed.scalar_type());

    int64_t batch, seq_len, q_head_num, k_head_num, dim;
    int64_t qb_stride, kb_stride, qs_stride, ks_stride, qh_stride, kh_stride;

    if (q_3d) {
        ASSERT_CHECK(grad_q_embed.size(2) == grad_q.size(2));
        ASSERT_CHECK(grad_k_embed.size(2) == grad_k.size(2));
        ASSERT_CHECK(grad_q_embed.size(2) % 8 == 0);
        ASSERT_CHECK(grad_q_embed.stride(2) == 1);
        ASSERT_CHECK(grad_k_embed.stride(2) == 1);
        ASSERT_CHECK(grad_q.stride(0) == grad_q_embed.stride(0));
        ASSERT_CHECK(grad_q.stride(1) == grad_q_embed.stride(1));
        ASSERT_CHECK(grad_q.stride(2) == grad_q_embed.stride(2));
        ASSERT_CHECK(grad_k.stride(0) == grad_k_embed.stride(0));
        ASSERT_CHECK(grad_k.stride(1) == grad_k_embed.stride(1));
        ASSERT_CHECK(grad_k.stride(2) == grad_k_embed.stride(2));
        batch = 1;
        seq_len = grad_q_embed.size(0);
        q_head_num = grad_q.size(1);
        k_head_num = grad_k.size(1);
        dim = grad_q_embed.size(2);
        qb_stride = 0; qs_stride = grad_q_embed.stride(0); qh_stride = grad_q_embed.stride(1);
        kb_stride = 0; ks_stride = grad_k_embed.stride(0); kh_stride = grad_k_embed.stride(1);
    } else {
        ASSERT_CHECK(grad_q_embed.size(3) == grad_q.size(3));
        ASSERT_CHECK(grad_k_embed.size(3) == grad_k.size(3));
        ASSERT_CHECK(grad_q_embed.size(3) % 8 == 0);
        ASSERT_CHECK(grad_q_embed.stride(3) == 1);
        ASSERT_CHECK(grad_k_embed.stride(3) == 1);
        ASSERT_CHECK(grad_q.stride(0) == grad_q_embed.stride(0));
        ASSERT_CHECK(grad_q.stride(1) == grad_q_embed.stride(1));
        ASSERT_CHECK(grad_q.stride(2) == grad_q_embed.stride(2));
        ASSERT_CHECK(grad_q.stride(3) == grad_q_embed.stride(3));
        ASSERT_CHECK(grad_k.stride(0) == grad_k_embed.stride(0));
        ASSERT_CHECK(grad_k.stride(1) == grad_k_embed.stride(1));
        ASSERT_CHECK(grad_k.stride(2) == grad_k_embed.stride(2));
        ASSERT_CHECK(grad_k.stride(3) == grad_k_embed.stride(3));
        batch = grad_q_embed.size(0);
        seq_len = grad_q_embed.size(1);
        q_head_num = grad_q.size(2);
        k_head_num = grad_k.size(2);
        dim = grad_q_embed.size(3);
        qb_stride = grad_q_embed.stride(0); qs_stride = grad_q_embed.stride(1); qh_stride = grad_q_embed.stride(2);
        kb_stride = grad_k_embed.stride(0); ks_stride = grad_k_embed.stride(1); kh_stride = grad_k_embed.stride(2);
    }

    dim3 grid(batch, seq_len);

    if (grad_q_embed.scalar_type() == at::kHalf) {
        const half* gq_embed = static_cast<const half*>(grad_q_embed.data_ptr());
        const half* gk_embed = static_cast<const half*>(grad_k_embed.data_ptr());
        half* gq = static_cast<half*>(grad_q.data_ptr());
        half* gk = static_cast<half*>(grad_k.data_ptr());
        const float* cos_data = static_cast<const float*>(cos.data_ptr());
        const float* sin_data = static_cast<const float*>(sin.data_ptr());
        if (interleave) {
            RopeKernelBackward<half, true><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                gq_embed, gk_embed,
                gq, gk,
                q_head_num, k_head_num, dim,
                qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride
            );
        } else {
            RopeKernelBackward<half, false><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                gq_embed, gk_embed,
                gq, gk,
                q_head_num, k_head_num, dim,
                qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride
            );
        }
    } else if (grad_q_embed.scalar_type() == at::kBFloat16) {
        const __nv_bfloat16* gq_embed = static_cast<const __nv_bfloat16*>(grad_q_embed.data_ptr());
        const __nv_bfloat16* gk_embed = static_cast<const __nv_bfloat16*>(grad_k_embed.data_ptr());
        __nv_bfloat16* gq = static_cast<__nv_bfloat16*>(grad_q.data_ptr());
        __nv_bfloat16* gk = static_cast<__nv_bfloat16*>(grad_k.data_ptr());
        const float* cos_data = static_cast<const float*>(cos.data_ptr());
        const float* sin_data = static_cast<const float*>(sin.data_ptr());
        if (interleave) {
            RopeKernelBackward<__nv_bfloat16, true><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                gq_embed, gk_embed,
                gq, gk,
                q_head_num, k_head_num, dim,
                qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride
            );
        } else {
            RopeKernelBackward<__nv_bfloat16, false><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                gq_embed, gk_embed,
                gq, gk,
                q_head_num, k_head_num, dim,
                qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride
            );
        }
    } else if (grad_q_embed.scalar_type() == at::kFloat) {
        const float* gq_embed = static_cast<const float*>(grad_q_embed.data_ptr());
        const float* gk_embed = static_cast<const float*>(grad_k_embed.data_ptr());
        float* gq = static_cast<float*>(grad_q.data_ptr());
        float* gk = static_cast<float*>(grad_k.data_ptr());
        const float* cos_data = static_cast<const float*>(cos.data_ptr());
        const float* sin_data = static_cast<const float*>(sin.data_ptr());
        if (interleave) {
            RopeKernelBackward<float, true><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                gq_embed, gk_embed,
                gq, gk,
                q_head_num, k_head_num, dim,
                qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride
            );
        } else {
            RopeKernelBackward<float, false><<<grid, Nthreads, dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                gq_embed, gk_embed,
                gq, gk,
                q_head_num, k_head_num, dim,
                qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride
            );
        }
    } else {
        throw std::runtime_error("Unsupported data type for rope backward");
    }

    sync_check_cuda_error();
}

// Pack interface: accepts qkv [seq_len, q_dim + 2*kv_dim], applies rope inplace
// on the q and k slices. Supports GQA (q_num_heads != kv_num_heads).
// CUDA kernels are not modified; only host-side pointer arithmetic is added.
void RopeInplacePack(const at::Tensor& qkv, // [seq_len, q_dim + 2*kv_dim]
                     const at::Tensor& cos,  // [1, seq_len, head_dim/2]
                     const at::Tensor& sin,  // [1, seq_len, head_dim/2]
                     int64_t q_num_heads,
                     int64_t kv_num_heads,
                     bool interleave) {
    int Nthreads = 256;
    cudaStream_t stream = static_cast<cudaStream_t>(
        at::cuda::getCurrentCUDAStream().stream());

    ASSERT_CHECK(qkv.dim() == 2);
    ASSERT_CHECK(qkv.stride(1) == 1);
    ASSERT_CHECK(cos.scalar_type() == at::kFloat);
    ASSERT_CHECK(sin.scalar_type() == at::kFloat);
    ASSERT_CHECK(cos.is_contiguous());
    ASSERT_CHECK(sin.is_contiguous());

    int64_t seq_len  = qkv.size(0);
    int64_t head_dim = cos.size(2) * 2;       // cos: [1, seq, half_dim]
    int64_t q_dim    = q_num_heads  * head_dim;
    int64_t kv_dim   = kv_num_heads * head_dim;
    ASSERT_CHECK(qkv.size(1) == q_dim + 2 * kv_dim);
    ASSERT_CHECK(head_dim % 8 == 0);

    // q at offset 0, k at offset q_dim; logical [seq, *_num_heads, head_dim]
    // with strides [qkv_row_stride, head_dim, 1]
    int64_t qkv_row_stride = qkv.stride(0);
    int64_t qs_stride = qkv_row_stride;
    int64_t qh_stride = head_dim;
    int64_t ks_stride = qkv_row_stride;
    int64_t kh_stride = head_dim;

    dim3 grid(1, seq_len);

    if (qkv.scalar_type() == at::kHalf) {
        half* q_data    = static_cast<half*>(qkv.data_ptr());
        half* k_data    = q_data + q_dim;
        float* cos_data = static_cast<float*>(cos.data_ptr());
        float* sin_data = static_cast<float*>(sin.data_ptr());
        if (interleave) {
            RopeInplaceKernel<half, true><<<grid, Nthreads, head_dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_num_heads, k_data, kv_num_heads, head_dim,
                0, qs_stride, qh_stride, 0, ks_stride, kh_stride);
        } else {
            RopeInplaceKernel<half, false><<<grid, Nthreads, head_dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_num_heads, k_data, kv_num_heads, head_dim,
                0, qs_stride, qh_stride, 0, ks_stride, kh_stride);
        }
    } else if (qkv.scalar_type() == at::kBFloat16) {
        __nv_bfloat16* q_data = static_cast<__nv_bfloat16*>(qkv.data_ptr());
        __nv_bfloat16* k_data = q_data + q_dim;
        float* cos_data = static_cast<float*>(cos.data_ptr());
        float* sin_data = static_cast<float*>(sin.data_ptr());
        if (interleave) {
            RopeInplaceKernel<__nv_bfloat16, true><<<grid, Nthreads, head_dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_num_heads, k_data, kv_num_heads, head_dim,
                0, qs_stride, qh_stride, 0, ks_stride, kh_stride);
        } else {
            RopeInplaceKernel<__nv_bfloat16, false><<<grid, Nthreads, head_dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_num_heads, k_data, kv_num_heads, head_dim,
                0, qs_stride, qh_stride, 0, ks_stride, kh_stride);
        }
    } else if (qkv.scalar_type() == at::kFloat) {
        float* q_data   = static_cast<float*>(qkv.data_ptr());
        float* k_data   = q_data + q_dim;
        float* cos_data = static_cast<float*>(cos.data_ptr());
        float* sin_data = static_cast<float*>(sin.data_ptr());
        if (interleave) {
            RopeInplaceKernel<float, true><<<grid, Nthreads, head_dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_num_heads, k_data, kv_num_heads, head_dim,
                0, qs_stride, qh_stride, 0, ks_stride, kh_stride);
        } else {
            RopeInplaceKernel<float, false><<<grid, Nthreads, head_dim * sizeof(float), stream>>>(
                cos_data, sin_data,
                q_data, q_num_heads, k_data, kv_num_heads, head_dim,
                0, qs_stride, qh_stride, 0, ks_stride, kh_stride);
        }
    } else {
        throw std::runtime_error("Unsupported data type for rope_inplace_pack");
    }

    sync_check_cuda_error();
}

// Removed TVM FFI export (see pybind11 registration)
// Removed TVM FFI export (see pybind11 registration)
// Removed TVM FFI export (see pybind11 registration)
// Removed TVM FFI export (see pybind11 registration)

} // wallx_cuda_rope
