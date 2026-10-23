#include "tensor_view.h"

#include <cuda_fp16.h>
 #include <cassert>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include "../common/cuda_utils.h"

namespace wallx_cuda_m_rope {

constexpr int64_t kMaxMRopeGridDimYZ = 65535;

inline dim3 MakeMRopeGrid(int64_t batch, int64_t seq_len) {
    ASSERT_CHECK(batch > 0);
    ASSERT_CHECK(seq_len > 0);
    ASSERT_CHECK(batch <= 2147483647LL);

    const int64_t grid_z = (seq_len + kMaxMRopeGridDimYZ - 1) / kMaxMRopeGridDimYZ;
    ASSERT_CHECK(grid_z <= kMaxMRopeGridDimYZ);
    const int64_t grid_y = (seq_len + grid_z - 1) / grid_z;
    return dim3(static_cast<unsigned int>(batch),
                static_cast<unsigned int>(grid_y),
                static_cast<unsigned int>(grid_z));
}

__device__ __forceinline__ int64_t MRopeSeqBlockIdx() {
    return static_cast<int64_t>(blockIdx.y) + static_cast<int64_t>(blockIdx.z) * gridDim.y;
}

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

template<typename T>
__device__ __forceinline__ void copy4(T* dst, const T* src) {
    if constexpr (std::is_same_v<T, half>) {
        reinterpret_cast<half2*>(dst)[0] = reinterpret_cast<const half2*>(src)[0];
        reinterpret_cast<half2*>(dst)[1] = reinterpret_cast<const half2*>(src)[1];
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        reinterpret_cast<__nv_bfloat162*>(dst)[0] = reinterpret_cast<const __nv_bfloat162*>(src)[0];
        reinterpret_cast<__nv_bfloat162*>(dst)[1] = reinterpret_cast<const __nv_bfloat162*>(src)[1];
    } else if constexpr (std::is_same_v<T, float>) {
        reinterpret_cast<float4*>(dst)[0] = reinterpret_cast<const float4*>(src)[0];
    } else {
        static_assert(sizeof(T) == 0, "Unsupported type");
    }
}

template<class T>
__global__ void MRopeKernel(const float* cos,
                            const float* sin,
                            const T* q,
                            const int q_h,
                            const T* k,
                            const int k_h,
                            T* q_embed,
                            T* k_embed,
                            const int first,
                            const int second,
                            const int d,
                            const int64_t seq_len,
                            // int64 strides: (batch)*(b_stride) overflows int32 for
                            // tensors past 2^31 elements (e.g. b>=128 at s=4096,h=32,d=128).
                            const int64_t qb_stride,
                            const int64_t qs_stride,
                            const int64_t qh_stride,
                            const int64_t kb_stride,
                            const int64_t ks_stride,
                            const int64_t kh_stride,
                            const int64_t qeb_stride,
                            const int64_t qes_stride,
                            const int64_t qeh_stride,
                            const int64_t keb_stride,
                            const int64_t kes_stride,
                            const int64_t keh_stride) {
    extern __shared__ char cos_sin[];
    const int half_dim = d / 2;
    float* cos_smem = reinterpret_cast<float*>(cos_sin);
    float* sin_smem = cos_smem + half_dim;
    int b = blockIdx.x;
    int64_t s = MRopeSeqBlockIdx();
    if (s >= seq_len) return;

    int64_t offset = static_cast<int64_t>(gridDim.x) * seq_len * half_dim;
    int64_t cos_sin_b_stride = seq_len * half_dim;
    int64_t cos_sin_s_stride = half_dim;
#define SIN_GMEM(a, b, c, d) sin[(a) * offset + (b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]
#define COS_GMEM(a, b, c, d) cos[(a) * offset + (b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        if (i < first) {
            cos_smem[i] = COS_GMEM(0, b, s, i);
            sin_smem[i] = SIN_GMEM(0, b, s, i);
        } else if (i < (second + first)) {
            cos_smem[i] = COS_GMEM(1, b, s, i);
            sin_smem[i] = SIN_GMEM(1, b, s, i);
        } else {
            cos_smem[i] = COS_GMEM(2, b, s, i);
            sin_smem[i] = SIN_GMEM(2, b, s, i);
        }
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

        float4 q_x0 = to_float4<T>(&Q_GMEM(b, s, h_idx, base_d));
        float4 q_x1 = to_float4<T>(&Q_GMEM(b, s, h_idx, base_d + half_dim));

        float4 cos_vec = to_float4<float>(&cos_smem[base_d]);
        float4 sin_vec = to_float4<float>(&sin_smem[base_d]);

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

    for (int j = threadIdx.x * 4; j < k_h * half_dim; j += blockDim.x * 4) {
        int h_idx = j / half_dim;
        int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;

        float4 k_x0 = to_float4<T>(&K_GMEM(b, s, h_idx, base_d));
        float4 k_x1 = to_float4<T>(&K_GMEM(b, s, h_idx, base_d + half_dim));

        float4 cos_vec = to_float4<float>(&cos_smem[base_d]);
        float4 sin_vec = to_float4<float>(&sin_smem[base_d]);

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

template<class T>
__global__ void MRopeInplaceKernel(float* cos, float* sin,
                                   T* q, const int q_h, T* k, const int k_h,
                                   const int first, const int second,
                                   const int d, const int64_t seq_len,
                                   const int64_t qb_stride, const int64_t qs_stride, const int64_t qh_stride,
                                   const int64_t kb_stride, const int64_t ks_stride, const int64_t kh_stride) {
    extern __shared__ char cos_sin[];
    const int half_dim = d / 2;
    float* cos_smem = reinterpret_cast<float*>(cos_sin);
    float* sin_smem = cos_smem + half_dim;
    int b = blockIdx.x;
    int64_t s = MRopeSeqBlockIdx();
    if (s >= seq_len) return;

    int64_t offset = static_cast<int64_t>(gridDim.x) * seq_len * half_dim;
    int64_t cos_sin_b_stride = seq_len * half_dim;
    int64_t cos_sin_s_stride = half_dim;
#define SIN_GMEM(a, b, c, d) sin[(a) * offset + (b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]
#define COS_GMEM(a, b, c, d) cos[(a) * offset + (b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        if (i < first) {
            cos_smem[i] = COS_GMEM(0, b, s, i);
            sin_smem[i] = SIN_GMEM(0, b, s, i);
        } else if (i < (second + first)) {
            cos_smem[i] = COS_GMEM(1, b, s, i);
            sin_smem[i] = SIN_GMEM(1, b, s, i);
        } else {
            cos_smem[i] = COS_GMEM(2, b, s, i);
            sin_smem[i] = SIN_GMEM(2, b, s, i);
        }
    }
    __syncthreads();

#define Q_GMEM(a, b, c, d) q[(a) * qb_stride + (b) * qs_stride + (c) * qh_stride + d]
#define K_GMEM(a, b, c, d) k[(a) * kb_stride + (b) * ks_stride + (c) * kh_stride + d]

    for (int j = threadIdx.x * 4; j < q_h * half_dim; j += blockDim.x * 4) {
        int h_idx = j / half_dim;
        int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;

        float4 q_x0 = to_float4<T>(&Q_GMEM(b, s, h_idx, base_d));
        float4 q_x1 = to_float4<T>(&Q_GMEM(b, s, h_idx, base_d + half_dim));

        float4 cos_vec = to_float4<float>(&cos_smem[base_d]);
        float4 sin_vec = to_float4<float>(&sin_smem[base_d]);

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

    for (int j = threadIdx.x * 4; j < k_h * half_dim; j += blockDim.x * 4) {
        int h_idx = j / half_dim;
        int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;

        float4 k_x0 = to_float4<T>(&K_GMEM(b, s, h_idx, base_d));
        float4 k_x1 = to_float4<T>(&K_GMEM(b, s, h_idx, base_d + half_dim));

        float4 cos_vec = to_float4<float>(&cos_smem[base_d]);
        float4 sin_vec = to_float4<float>(&sin_smem[base_d]);

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

template<class T>
__global__ void MRopeInplaceCfgCacheWriteKernel(
    const float* cos,
    const float* sin,
    T* q,
    const int q_h,
    T* k,
    const int k_h,
    const T* v,
    T* kvcache,
    const int64_t* valid0,
    const int64_t* valid1,
    const int common_len,
    const int cond_tail_len,
    const int first,
    const int second,
    const int d,
    const int64_t qb_stride,
    const int64_t qs_stride,
    const int64_t qh_stride,
    const int64_t kb_stride,
    const int64_t ks_stride,
    const int64_t kh_stride,
    const int64_t vb_stride,
    const int64_t vs_stride,
    const int64_t vh_stride,
    const int64_t cache_kv_stride,
    const int64_t cache_batch_stride,
    const int64_t cache_seq_stride,
    const int64_t cache_head_stride) {
    extern __shared__ char cos_sin[];
    const int half_dim = d / 2;
    float* cos_smem = reinterpret_cast<float*>(cos_sin);
    float* sin_smem = cos_smem + half_dim;
    const int b = blockIdx.x;
    const int s = blockIdx.y;

    const int64_t offset = static_cast<int64_t>(gridDim.x) * gridDim.y * half_dim;
    const int64_t cos_sin_b_stride = static_cast<int64_t>(gridDim.y) * half_dim;
    const int64_t cos_sin_s_stride = half_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        const int rope_axis = (i < first) ? 0 : ((i < first + second) ? 1 : 2);
        const int64_t rope_offset = static_cast<int64_t>(rope_axis) * offset +
                                    static_cast<int64_t>(b) * cos_sin_b_stride +
                                    static_cast<int64_t>(s) * cos_sin_s_stride + i;
        cos_smem[i] = cos[rope_offset];
        sin_smem[i] = sin[rope_offset];
    }
    __syncthreads();

    const int cond_end = common_len + cond_tail_len;
    const bool is_common = s < common_len;
    const bool is_cond_tail = (s >= common_len) && (s < cond_end);
    int64_t cache_pos0 = -1;
    int64_t cache_pos1 = -1;
    if (is_common) {
        cache_pos0 = valid0[s];
        cache_pos1 = valid1[s];
    } else if (is_cond_tail) {
        cache_pos0 = valid0[s];
    } else {
        const int valid1_idx = common_len + (s - cond_end);
        cache_pos1 = valid1[valid1_idx];
    }

    for (int j = threadIdx.x * 4; j < q_h * half_dim; j += blockDim.x * 4) {
        const int h_idx = j / half_dim;
        const int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;

        T* q0_ptr = q + static_cast<int64_t>(b) * qb_stride + static_cast<int64_t>(s) * qs_stride +
                    static_cast<int64_t>(h_idx) * qh_stride + base_d;
        T* q1_ptr = q0_ptr + half_dim;
        const float4 q_x0 = to_float4<T>(q0_ptr);
        const float4 q_x1 = to_float4<T>(q1_ptr);
        const float4 cos_vec = to_float4<float>(cos_smem + base_d);
        const float4 sin_vec = to_float4<float>(sin_smem + base_d);

        const float4 q0_rot = make_float4(
            q_x0.x * cos_vec.x - q_x1.x * sin_vec.x,
            q_x0.y * cos_vec.y - q_x1.y * sin_vec.y,
            q_x0.z * cos_vec.z - q_x1.z * sin_vec.z,
            q_x0.w * cos_vec.w - q_x1.w * sin_vec.w);
        const float4 q1_rot = make_float4(
            q_x1.x * cos_vec.x + q_x0.x * sin_vec.x,
            q_x1.y * cos_vec.y + q_x0.y * sin_vec.y,
            q_x1.z * cos_vec.z + q_x0.z * sin_vec.z,
            q_x1.w * cos_vec.w + q_x0.w * sin_vec.w);

        from_float4<T>(q0_ptr, q0_rot);
        from_float4<T>(q1_ptr, q1_rot);
    }

    T* key_cache = kvcache;
    T* value_cache = kvcache + cache_kv_stride;
    for (int j = threadIdx.x * 4; j < k_h * half_dim; j += blockDim.x * 4) {
        const int h_idx = j / half_dim;
        const int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;

        T* k0_ptr = k + static_cast<int64_t>(b) * kb_stride + static_cast<int64_t>(s) * ks_stride +
                    static_cast<int64_t>(h_idx) * kh_stride + base_d;
        T* k1_ptr = k0_ptr + half_dim;
        const T* v0_ptr = v + static_cast<int64_t>(b) * vb_stride + static_cast<int64_t>(s) * vs_stride +
                          static_cast<int64_t>(h_idx) * vh_stride + base_d;
        const T* v1_ptr = v0_ptr + half_dim;

        const float4 k_x0 = to_float4<T>(k0_ptr);
        const float4 k_x1 = to_float4<T>(k1_ptr);
        const float4 cos_vec = to_float4<float>(cos_smem + base_d);
        const float4 sin_vec = to_float4<float>(sin_smem + base_d);

        const float4 k0_rot = make_float4(
            k_x0.x * cos_vec.x - k_x1.x * sin_vec.x,
            k_x0.y * cos_vec.y - k_x1.y * sin_vec.y,
            k_x0.z * cos_vec.z - k_x1.z * sin_vec.z,
            k_x0.w * cos_vec.w - k_x1.w * sin_vec.w);
        const float4 k1_rot = make_float4(
            k_x1.x * cos_vec.x + k_x0.x * sin_vec.x,
            k_x1.y * cos_vec.y + k_x0.y * sin_vec.y,
            k_x1.z * cos_vec.z + k_x0.z * sin_vec.z,
            k_x1.w * cos_vec.w + k_x0.w * sin_vec.w);

        from_float4<T>(k0_ptr, k0_rot);
        from_float4<T>(k1_ptr, k1_rot);

        const int64_t cache_head_offset = static_cast<int64_t>(h_idx) * cache_head_stride + base_d;
        if (is_common || is_cond_tail) {
            T* kc0 = key_cache + 0 * cache_batch_stride + cache_pos0 * cache_seq_stride + cache_head_offset;
            T* vc0 = value_cache + 0 * cache_batch_stride + cache_pos0 * cache_seq_stride + cache_head_offset;
            from_float4<T>(kc0, k0_rot);
            from_float4<T>(kc0 + half_dim, k1_rot);
            copy4<T>(vc0, v0_ptr);
            copy4<T>(vc0 + half_dim, v1_ptr);
        }
        if (is_common || !is_cond_tail) {
            T* kc1 = key_cache + 1 * cache_batch_stride + cache_pos1 * cache_seq_stride + cache_head_offset;
            T* vc1 = value_cache + 1 * cache_batch_stride + cache_pos1 * cache_seq_stride + cache_head_offset;
            from_float4<T>(kc1, k0_rot);
            from_float4<T>(kc1 + half_dim, k1_rot);
            copy4<T>(vc1, v0_ptr);
            copy4<T>(vc1 + half_dim, v1_ptr);
        }
    }
}

template<class T>
__global__ void MRopeKernelBackward(
    const float* cos,
    const float* sin,
    const T* grad_q_embed,
    const T* grad_k_embed,
    T* grad_q,
    T* grad_k,
    const int first, const int second,
    const int q_h,
    const int k_h,
    const int d,
    const int64_t seq_len,
    const int64_t qb_stride, const int64_t qs_stride, const int64_t qh_stride,
    const int64_t kb_stride, const int64_t ks_stride, const int64_t kh_stride) {

    extern __shared__ char cos_sin[];
    const int half_dim = d / 2;
    float* cos_smem = reinterpret_cast<float*>(cos_sin);
    float* sin_smem = cos_smem + half_dim;

    int b = blockIdx.x;
    int64_t s = MRopeSeqBlockIdx();
    if (s >= seq_len) return;

    int64_t offset = static_cast<int64_t>(gridDim.x) * seq_len * half_dim;
    int64_t cos_sin_b_stride = seq_len * half_dim;
    int64_t cos_sin_s_stride = half_dim;

#define SIN_GMEM(a, b, c, d) sin[(a) * offset + (b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]
#define COS_GMEM(a, b, c, d) cos[(a) * offset + (b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        if (i < first) {
            cos_smem[i] = COS_GMEM(0, b, s, i);
            sin_smem[i] = SIN_GMEM(0, b, s, i);
        } else if (i < (second + first)) {
            cos_smem[i] = COS_GMEM(1, b, s, i);
            sin_smem[i] = SIN_GMEM(1, b, s, i);
        } else {
            cos_smem[i] = COS_GMEM(2, b, s, i);
            sin_smem[i] = SIN_GMEM(2, b, s, i);
        }
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

        // float path
        float4 gq0_rot = to_float4<T>(&GQ_EMBED(b, s, h_idx, base_d));
        float4 gq1_rot = to_float4<T>(&GQ_EMBED(b, s, h_idx, base_d + half_dim));

        float4 cos_vec = to_float4<float>(&cos_smem[base_d]);
        float4 sin_vec = to_float4<float>(&sin_smem[base_d]);

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

    // Process grad_k
    for (int j = threadIdx.x * 4; j < k_h * half_dim; j += blockDim.x * 4) {
        int h_idx = j / half_dim;
        int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;

        // float path
        float4 gk0_rot = to_float4<T>(&GK_EMBED(b, s, h_idx, base_d));
        float4 gk1_rot = to_float4<T>(&GK_EMBED(b, s, h_idx, base_d + half_dim));

        float4 cos_vec = to_float4<float>(&cos_smem[base_d]);
        float4 sin_vec = to_float4<float>(&sin_smem[base_d]);

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

void MRope(wallx_cuda::TensorView q, // [b, s, h, d]
           wallx_cuda::TensorView k, // [b, s, h_k, d]
           wallx_cuda::TensorView q_embed, // [b, s, h, d]
           wallx_cuda::TensorView k_embed, // [b, s, h_k, d]
           wallx_cuda::TensorView cos, // [3, b, s, d / 2]
           wallx_cuda::TensorView sin, // [3, b, s, d / 2]
           const int first,
           const int second
           ) {
    int Nthreads = 256;

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(q.device().device_type, q.device().device_id));

    ASSERT_CHECK(q.dtype() == k.dtype());
    ASSERT_CHECK(q.dtype() == q_embed.dtype());
    ASSERT_CHECK(k.dtype() == k_embed.dtype());
    ASSERT_CHECK(q_embed.size(0) == q.size(0));
    ASSERT_CHECK(q_embed.size(1) == q.size(1));
    ASSERT_CHECK(q_embed.size(2) == q.size(2));
    ASSERT_CHECK(q_embed.size(3) == q.size(3));
    ASSERT_CHECK(k_embed.size(0) == k.size(0));
    ASSERT_CHECK(k_embed.size(1) == k.size(1));
    ASSERT_CHECK(k_embed.size(2) == k.size(2));
    ASSERT_CHECK(k_embed.size(3) == k.size(3));
    ASSERT_CHECK(q.size(3) % 8 == 0);
    ASSERT_CHECK(cos.size(0) == 3);
    ASSERT_CHECK(cos.size(3) == q.size(3) / 2);
    ASSERT_CHECK(cos.dtype().code == kDLFloat && cos.dtype().bits == 32);
    ASSERT_CHECK(sin.dtype().code == kDLFloat && sin.dtype().bits == 32);
    ASSERT_CHECK(sin.size(0) == 3);
    ASSERT_CHECK(q.stride(3) == 1);
    ASSERT_CHECK(k.stride(3) == 1);
    ASSERT_CHECK(q_embed.stride(3) == 1);
    ASSERT_CHECK(k_embed.stride(3) == 1);
    ASSERT_CHECK(cos.stride(3) == 1);
    ASSERT_CHECK(sin.stride(3) == 1);
    ASSERT_CHECK(cos.IsContiguous());
    ASSERT_CHECK(sin.IsContiguous());

    int64_t batch = q.size(0);
    int64_t q_head_num = q.size(2);
    int64_t k_head_num = k.size(2);
    int64_t seq_len = q.size(1);
    int64_t dim = q.size(3);
    int64_t qb_stride = q.stride(0);
    int64_t kb_stride = k.stride(0);
    int64_t qs_stride = q.stride(1);
    int64_t ks_stride = k.stride(1);
    int64_t qh_stride = q.stride(2);
    int64_t kh_stride = k.stride(2);
    int64_t qeb_stride = q_embed.stride(0);
    int64_t keb_stride = k_embed.stride(0);
    int64_t qes_stride = q_embed.stride(1);
    int64_t kes_stride = k_embed.stride(1);
    int64_t qeh_stride = q_embed.stride(2);
    int64_t keh_stride = k_embed.stride(2);

    dim3 grid = MakeMRopeGrid(batch, seq_len);

    if (q.dtype().code == kDLFloat && q.dtype().bits == 16) {
        const half* q_data = static_cast<const half*>(q.data_ptr());
        const half* k_data = static_cast<const half*>(k.data_ptr());
        half* q_embed_data = static_cast<half*>(q_embed.data_ptr());
        half* k_embed_data = static_cast<half*>(k_embed.data_ptr());
        const float* cos_data = static_cast<const float*>(cos.data_ptr());
        const float* sin_data = static_cast<const float*>(sin.data_ptr());
        MRopeKernel<half><<<grid, Nthreads, dim * sizeof(float), stream>>>(
            cos_data, sin_data,
            q_data, q_head_num,
            k_data, k_head_num,
            q_embed_data, k_embed_data,
            first, second, dim, seq_len,
            qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride,
            qeb_stride, qes_stride, qeh_stride, keb_stride, kes_stride, keh_stride);
    } else if (q.dtype().code == kDLBfloat && q.dtype().bits == 16) {
        const __nv_bfloat16* q_data = static_cast<const __nv_bfloat16*>(q.data_ptr());
        const __nv_bfloat16* k_data = static_cast<const __nv_bfloat16*>(k.data_ptr());
        __nv_bfloat16* q_embed_data = static_cast<__nv_bfloat16*>(q_embed.data_ptr());
        __nv_bfloat16* k_embed_data = static_cast<__nv_bfloat16*>(k_embed.data_ptr());
        const float* cos_data = static_cast<const float*>(cos.data_ptr());
        const float* sin_data = static_cast<const float*>(sin.data_ptr());
        MRopeKernel<__nv_bfloat16><<<grid, Nthreads, dim * sizeof(float), stream>>>(
            cos_data, sin_data,
            q_data, q_head_num,
            k_data, k_head_num,
            q_embed_data, k_embed_data,
            first, second, dim, seq_len,
            qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride,
            qeb_stride, qes_stride, qeh_stride, keb_stride, kes_stride, keh_stride);
    } else if (q.dtype().code == kDLFloat && q.dtype().bits == 32) {
        const float* q_data = static_cast<const float*>(q.data_ptr());
        const float* k_data = static_cast<const float*>(k.data_ptr());
        float* q_embed_data = static_cast<float*>(q_embed.data_ptr());
        float* k_embed_data = static_cast<float*>(k_embed.data_ptr());
        const float* cos_data = static_cast<const float*>(cos.data_ptr());
        const float* sin_data = static_cast<const float*>(sin.data_ptr());
        MRopeKernel<float><<<grid, Nthreads, dim * sizeof(float), stream>>>(
            cos_data, sin_data,
            q_data, q_head_num,
            k_data, k_head_num,
            q_embed_data, k_embed_data,
            first, second, dim, seq_len,
            qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride,
            qeb_stride, qes_stride, qeh_stride, keb_stride, kes_stride, keh_stride
        );
    } else {
        throw std::runtime_error("Unsupported data type for m_rope");
    }

    sync_check_cuda_error();
}

void MRopeInplace(wallx_cuda::TensorView q, // [b, s, h, d]
                  wallx_cuda::TensorView k, // [b, s, h_k, d]
                  wallx_cuda::TensorView cos, // [3, b, s, d / 2]
                  wallx_cuda::TensorView sin, // [3, b, s, d / 2]
                  const int first,
                  const int second
                  ) {
    int Nthreads = 256;

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(q.device().device_type, q.device().device_id));

    ASSERT_CHECK(q.dtype() == k.dtype());
    ASSERT_CHECK(q.size(3) % 8 == 0);
    ASSERT_CHECK(cos.size(0) == 3);
    ASSERT_CHECK(cos.size(3) == q.size(3) / 2);
    ASSERT_CHECK(cos.dtype().code == kDLFloat && cos.dtype().bits == 32);
    ASSERT_CHECK(sin.dtype().code == kDLFloat && sin.dtype().bits == 32);
    ASSERT_CHECK(sin.size(0) == 3);
    ASSERT_CHECK(q.stride(3) == 1);
    ASSERT_CHECK(k.stride(3) == 1);
    ASSERT_CHECK(cos.stride(3) == 1);
    ASSERT_CHECK(sin.stride(3) == 1);
    ASSERT_CHECK(cos.IsContiguous());
    ASSERT_CHECK(sin.IsContiguous());

    int64_t batch = q.size(0);
    int64_t q_head_num = q.size(2);
    int64_t k_head_num = k.size(2);
    int64_t seq_len = q.size(1);
    int64_t dim = q.size(3);
    int64_t qb_stride = q.stride(0);
    int64_t kb_stride = k.stride(0);
    int64_t qs_stride = q.stride(1);
    int64_t ks_stride = k.stride(1);
    int64_t qh_stride = q.stride(2);
    int64_t kh_stride = k.stride(2);

    dim3 grid = MakeMRopeGrid(batch, seq_len);

    if (q.dtype().code == kDLFloat && q.dtype().bits == 16) {
        half* q_data = static_cast<half*>(q.data_ptr());
        half* k_data = static_cast<half*>(k.data_ptr());
        float* cos_data = static_cast<float*>(cos.data_ptr());
        float* sin_data = static_cast<float*>(sin.data_ptr());
        MRopeInplaceKernel<half><<<grid, Nthreads, dim * sizeof(float), stream>>>(
            cos_data, sin_data,
            q_data, q_head_num,
            k_data, k_head_num,
            first, second, dim, seq_len, qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride);
    } else if (q.dtype().code == kDLBfloat && q.dtype().bits == 16) {
        __nv_bfloat16* q_data = static_cast<__nv_bfloat16*>(q.data_ptr());
        __nv_bfloat16* k_data = static_cast<__nv_bfloat16*>(k.data_ptr());
        float* cos_data = static_cast<float*>(cos.data_ptr());
        float* sin_data = static_cast<float*>(sin.data_ptr());
        MRopeInplaceKernel<__nv_bfloat16><<<grid, Nthreads, dim * sizeof(float), stream>>>(
            cos_data, sin_data,
            q_data, q_head_num,
            k_data, k_head_num,
            first, second, dim, seq_len, qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride);
    } else if (q.dtype().code == kDLFloat && q.dtype().bits == 32) {
        float* q_data = static_cast<float*>(q.data_ptr());
        float* k_data = static_cast<float*>(k.data_ptr());
        float* cos_data = static_cast<float*>(cos.data_ptr());
        float* sin_data = static_cast<float*>(sin.data_ptr());
        MRopeInplaceKernel<float><<<grid, Nthreads, dim * sizeof(float), stream>>>(
            cos_data, sin_data,
            q_data, q_head_num,
            k_data, k_head_num,
            first, second, dim, seq_len, qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride
        );
    } else {
        throw std::runtime_error("Unsupported data type for m_rope");
    }

    sync_check_cuda_error();
}

void MRopeInplaceCfgCacheWrite(wallx_cuda::TensorView q,        // [1, packed_seq, q_h, d]
                                wallx_cuda::TensorView k,        // [1, packed_seq, kv_h, d]
                                wallx_cuda::TensorView v,        // [1, packed_seq, kv_h, d]
                                wallx_cuda::TensorView cos,      // [3, 1, packed_seq, d / 2]
                                wallx_cuda::TensorView sin,      // [3, 1, packed_seq, d / 2]
                                wallx_cuda::TensorView kvcache,  // [2, max_batch, max_len, kv_h, d]
                                wallx_cuda::TensorView valid0,   // int64 original prefix indices for cond row
                                wallx_cuda::TensorView valid1,   // int64 original prefix indices for uncond row
                                int64_t common_len,
                                int64_t cond_tail_len,
                                const int first,
                                const int second) {
    int Nthreads = 256;
    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(q.device().device_type, q.device().device_id));

    ASSERT_CHECK(q.ndim() == 4);
    ASSERT_CHECK(k.ndim() == 4);
    ASSERT_CHECK(v.ndim() == 4);
    ASSERT_CHECK(kvcache.ndim() == 5);
    ASSERT_CHECK(q.dtype() == k.dtype());
    ASSERT_CHECK(q.dtype() == v.dtype());
    ASSERT_CHECK(q.dtype() == kvcache.dtype());
    ASSERT_CHECK(q.size(0) == 1);
    ASSERT_CHECK(k.size(0) == q.size(0));
    ASSERT_CHECK(v.size(0) == q.size(0));
    ASSERT_CHECK(k.size(1) == q.size(1));
    ASSERT_CHECK(v.size(1) == q.size(1));
    ASSERT_CHECK(v.size(2) == k.size(2));
    ASSERT_CHECK(k.size(3) == q.size(3));
    ASSERT_CHECK(v.size(3) == q.size(3));
    ASSERT_CHECK(q.size(3) % 8 == 0);
    ASSERT_CHECK(q.stride(3) == 1);
    ASSERT_CHECK(k.stride(3) == 1);
    ASSERT_CHECK(v.stride(3) == 1);
    ASSERT_CHECK(kvcache.stride(4) == 1);

    ASSERT_CHECK(cos.size(0) == 3);
    ASSERT_CHECK(sin.size(0) == 3);
    ASSERT_CHECK(cos.size(1) == q.size(0));
    ASSERT_CHECK(sin.size(1) == q.size(0));
    ASSERT_CHECK(cos.size(2) == q.size(1));
    ASSERT_CHECK(sin.size(2) == q.size(1));
    ASSERT_CHECK(cos.size(3) == q.size(3) / 2);
    ASSERT_CHECK(sin.size(3) == q.size(3) / 2);
    ASSERT_CHECK(cos.dtype().code == kDLFloat && cos.dtype().bits == 32);
    ASSERT_CHECK(sin.dtype().code == kDLFloat && sin.dtype().bits == 32);
    ASSERT_CHECK(cos.IsContiguous());
    ASSERT_CHECK(sin.IsContiguous());

    ASSERT_CHECK(kvcache.size(0) == 2);
    ASSERT_CHECK(kvcache.size(1) >= 2);
    ASSERT_CHECK(kvcache.size(3) == k.size(2));
    ASSERT_CHECK(kvcache.size(4) == k.size(3));
    ASSERT_CHECK(valid0.ndim() == 1);
    ASSERT_CHECK(valid1.ndim() == 1);
    ASSERT_CHECK(valid0.dtype().code == kDLInt && valid0.dtype().bits == 64);
    ASSERT_CHECK(valid1.dtype().code == kDLInt && valid1.dtype().bits == 64);

    int64_t seq_len = q.size(1);
    int64_t uncond_tail_len = seq_len - common_len - cond_tail_len;
    ASSERT_CHECK(common_len >= 0);
    ASSERT_CHECK(cond_tail_len >= 0);
    ASSERT_CHECK(uncond_tail_len >= 0);
    ASSERT_CHECK(valid0.size(0) >= common_len + cond_tail_len);
    ASSERT_CHECK(valid1.size(0) >= common_len + uncond_tail_len);

    int64_t q_head_num = q.size(2);
    int64_t k_head_num = k.size(2);
    int64_t dim = q.size(3);
    dim3 grid(q.size(0), seq_len);

    if (q.dtype().code == kDLFloat && q.dtype().bits == 16) {
        MRopeInplaceCfgCacheWriteKernel<half><<<grid, Nthreads, dim * sizeof(float), stream>>>(
            static_cast<const float*>(cos.data_ptr()),
            static_cast<const float*>(sin.data_ptr()),
            static_cast<half*>(q.data_ptr()), q_head_num,
            static_cast<half*>(k.data_ptr()), k_head_num,
            static_cast<const half*>(v.data_ptr()),
            static_cast<half*>(kvcache.data_ptr()),
            static_cast<const int64_t*>(valid0.data_ptr()),
            static_cast<const int64_t*>(valid1.data_ptr()),
            common_len, cond_tail_len, first, second, dim,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            kvcache.stride(0), kvcache.stride(1), kvcache.stride(2), kvcache.stride(3));
    } else if (q.dtype().code == kDLBfloat && q.dtype().bits == 16) {
        MRopeInplaceCfgCacheWriteKernel<__nv_bfloat16><<<grid, Nthreads, dim * sizeof(float), stream>>>(
            static_cast<const float*>(cos.data_ptr()),
            static_cast<const float*>(sin.data_ptr()),
            static_cast<__nv_bfloat16*>(q.data_ptr()), q_head_num,
            static_cast<__nv_bfloat16*>(k.data_ptr()), k_head_num,
            static_cast<const __nv_bfloat16*>(v.data_ptr()),
            static_cast<__nv_bfloat16*>(kvcache.data_ptr()),
            static_cast<const int64_t*>(valid0.data_ptr()),
            static_cast<const int64_t*>(valid1.data_ptr()),
            common_len, cond_tail_len, first, second, dim,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            kvcache.stride(0), kvcache.stride(1), kvcache.stride(2), kvcache.stride(3));
    } else if (q.dtype().code == kDLFloat && q.dtype().bits == 32) {
        MRopeInplaceCfgCacheWriteKernel<float><<<grid, Nthreads, dim * sizeof(float), stream>>>(
            static_cast<const float*>(cos.data_ptr()),
            static_cast<const float*>(sin.data_ptr()),
            static_cast<float*>(q.data_ptr()), q_head_num,
            static_cast<float*>(k.data_ptr()), k_head_num,
            static_cast<const float*>(v.data_ptr()),
            static_cast<float*>(kvcache.data_ptr()),
            static_cast<const int64_t*>(valid0.data_ptr()),
            static_cast<const int64_t*>(valid1.data_ptr()),
            common_len, cond_tail_len, first, second, dim,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            kvcache.stride(0), kvcache.stride(1), kvcache.stride(2), kvcache.stride(3));
    } else {
        throw std::runtime_error("Unsupported data type for m_rope_inplace_cfg_cache_write");
    }

    sync_check_cuda_error();
}

void MRopeBackward(
    wallx_cuda::TensorView grad_q_embed,
    wallx_cuda::TensorView grad_k_embed,
    wallx_cuda::TensorView grad_q,    // output: [b, s, q_h, d]
    wallx_cuda::TensorView grad_k,    // output: [b, s, k_h, d]
    wallx_cuda::TensorView cos,       // [3, b, s, d/2]
    wallx_cuda::TensorView sin,       // [3, b, s, d/2]
    const int first,
    const int second
) {
    int Nthreads = 256;

    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(grad_q_embed.device().device_type, grad_q_embed.device().device_id));

    ASSERT_CHECK(grad_q_embed.dtype() == grad_k_embed.dtype());
    ASSERT_CHECK(grad_q.dtype() == grad_k.dtype());
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

    int64_t batch = grad_q_embed.size(0);
    int64_t seq_len = grad_q_embed.size(1);
    int64_t q_head_num = grad_q.size(2);
    int64_t k_head_num = grad_k.size(2);
    int64_t dim = grad_q_embed.size(3);
    int64_t qb_stride = grad_q_embed.stride(0);
    int64_t kb_stride = grad_k_embed.stride(0);
    int64_t qs_stride = grad_q_embed.stride(1);
    int64_t ks_stride = grad_k_embed.stride(1);
    int64_t qh_stride = grad_q_embed.stride(2);
    int64_t kh_stride = grad_k_embed.stride(2);

    dim3 grid = MakeMRopeGrid(batch, seq_len);

    if (grad_q_embed.dtype().code == kDLFloat && grad_q_embed.dtype().bits == 16) {
        const half* gq_embed = static_cast<const half*>(grad_q_embed.data_ptr());
        const half* gk_embed = static_cast<const half*>(grad_k_embed.data_ptr());
        half* gq = static_cast<half*>(grad_q.data_ptr());
        half* gk = static_cast<half*>(grad_k.data_ptr());
        const float* cos_data = static_cast<const float*>(cos.data_ptr());
        const float* sin_data = static_cast<const float*>(sin.data_ptr());
        MRopeKernelBackward<half><<<grid, Nthreads, dim * sizeof(float), stream>>>(
            cos_data, sin_data,
            gq_embed, gk_embed,
            gq, gk,
            first, second,
            q_head_num, k_head_num, dim, seq_len, qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride
        );
    } else if (grad_q_embed.dtype().code == kDLBfloat && grad_q_embed.dtype().bits == 16) {
        const __nv_bfloat16* gq_embed = static_cast<const __nv_bfloat16*>(grad_q_embed.data_ptr());
        const __nv_bfloat16* gk_embed = static_cast<const __nv_bfloat16*>(grad_k_embed.data_ptr());
        __nv_bfloat16* gq = static_cast<__nv_bfloat16*>(grad_q.data_ptr());
        __nv_bfloat16* gk = static_cast<__nv_bfloat16*>(grad_k.data_ptr());
        const float* cos_data = static_cast<const float*>(cos.data_ptr());
        const float* sin_data = static_cast<const float*>(sin.data_ptr());
        MRopeKernelBackward<__nv_bfloat16><<<grid, Nthreads, dim * sizeof(float), stream>>>(
            cos_data, sin_data,
            gq_embed, gk_embed,
            gq, gk,
            first, second,
            q_head_num, k_head_num, dim, seq_len, qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride
        );
    } else if (grad_q_embed.dtype().code == kDLFloat && grad_q_embed.dtype().bits == 32) {
        const float* gq_embed = static_cast<const float*>(grad_q_embed.data_ptr());
        const float* gk_embed = static_cast<const float*>(grad_k_embed.data_ptr());
        float* gq = static_cast<float*>(grad_q.data_ptr());
        float* gk = static_cast<float*>(grad_k.data_ptr());
        const float* cos_data = static_cast<const float*>(cos.data_ptr());
        const float* sin_data = static_cast<const float*>(sin.data_ptr());
        MRopeKernelBackward<float><<<grid, Nthreads, dim * sizeof(float), stream>>>(
            cos_data, sin_data,
            gq_embed, gk_embed,
            gq, gk,
            first, second,
            q_head_num, k_head_num, dim, seq_len, qb_stride, qs_stride, qh_stride, kb_stride, ks_stride, kh_stride
        );
    } else {
        throw std::runtime_error("Unsupported data type for m_rope backward");
    }

    sync_check_cuda_error();
}

// Pack interface: accepts qkv [bz, seq_len, q_dim + 2*kv_dim], applies mrope
// inplace on q and k without Python-side split/view overhead.
// Supports GQA (q_num_heads != kv_num_heads). CUDA kernels are not modified.
void MRopeInplacePack(wallx_cuda::TensorView qkv, // [bz, seq_len, q_dim + 2*kv_dim]
                      wallx_cuda::TensorView cos,  // [3, bz, seq_len, head_dim/2]
                      wallx_cuda::TensorView sin,  // [3, bz, seq_len, head_dim/2]
                      int64_t q_num_heads,
                      int64_t kv_num_heads,
                      const int first,
                      const int second) {
    int Nthreads = 256;
    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(qkv.device().device_type, qkv.device().device_id));

    ASSERT_CHECK(qkv.ndim() == 3);
    ASSERT_CHECK(qkv.stride(2) == 1);
    ASSERT_CHECK(cos.dtype().code == kDLFloat && cos.dtype().bits == 32);
    ASSERT_CHECK(sin.dtype().code == kDLFloat && sin.dtype().bits == 32);
    ASSERT_CHECK(cos.size(0) == 3);
    ASSERT_CHECK(sin.size(0) == 3);
    ASSERT_CHECK(cos.IsContiguous());
    ASSERT_CHECK(sin.IsContiguous());

    int64_t bz       = qkv.size(0);
    int64_t seq_len  = qkv.size(1);
    int64_t head_dim = cos.size(3) * 2;        // cos: [3, bz, seq, half_dim]
    int64_t q_dim    = q_num_heads  * head_dim;
    int64_t kv_dim   = kv_num_heads * head_dim;
    ASSERT_CHECK(qkv.size(2) == q_dim + 2 * kv_dim);
    ASSERT_CHECK(head_dim % 8 == 0);

    // q at offset 0, k at offset q_dim; logical [bz, seq, *_num_heads, head_dim]
    // with strides [qkv.stride(0), qkv.stride(1), head_dim, 1]
    int64_t qkv_b_stride = qkv.stride(0);
    int64_t qkv_s_stride = qkv.stride(1);

    dim3 grid = MakeMRopeGrid(bz, seq_len);

    if (qkv.dtype().code == kDLFloat && qkv.dtype().bits == 16) {
        half* q_data    = static_cast<half*>(qkv.data_ptr());
        half* k_data    = q_data + q_dim;
        float* cos_data = static_cast<float*>(cos.data_ptr());
        float* sin_data = static_cast<float*>(sin.data_ptr());
        MRopeInplaceKernel<half><<<grid, Nthreads, head_dim * sizeof(float), stream>>>(
            cos_data, sin_data,
            q_data, q_num_heads, k_data, kv_num_heads,
            first, second, head_dim, seq_len,
            qkv_b_stride, qkv_s_stride, head_dim,
            qkv_b_stride, qkv_s_stride, head_dim);
    } else if (qkv.dtype().code == kDLBfloat && qkv.dtype().bits == 16) {
        __nv_bfloat16* q_data = static_cast<__nv_bfloat16*>(qkv.data_ptr());
        __nv_bfloat16* k_data = q_data + q_dim;
        float* cos_data = static_cast<float*>(cos.data_ptr());
        float* sin_data = static_cast<float*>(sin.data_ptr());
        MRopeInplaceKernel<__nv_bfloat16><<<grid, Nthreads, head_dim * sizeof(float), stream>>>(
            cos_data, sin_data,
            q_data, q_num_heads, k_data, kv_num_heads,
            first, second, head_dim, seq_len,
            qkv_b_stride, qkv_s_stride, head_dim,
            qkv_b_stride, qkv_s_stride, head_dim);
    } else if (qkv.dtype().code == kDLFloat && qkv.dtype().bits == 32) {
        float* q_data   = static_cast<float*>(qkv.data_ptr());
        float* k_data   = q_data + q_dim;
        float* cos_data = static_cast<float*>(cos.data_ptr());
        float* sin_data = static_cast<float*>(sin.data_ptr());
        MRopeInplaceKernel<float><<<grid, Nthreads, head_dim * sizeof(float), stream>>>(
            cos_data, sin_data,
            q_data, q_num_heads, k_data, kv_num_heads,
            first, second, head_dim, seq_len,
            qkv_b_stride, qkv_s_stride, head_dim,
            qkv_b_stride, qkv_s_stride, head_dim);
    } else {
        throw std::runtime_error("Unsupported data type for m_rope_inplace_pack");
    }

    sync_check_cuda_error();
}

// Pack backward interface: accepts dqkv [bz, seq_len, q_dim + 2*kv_dim] and
// applies the inverse mrope transform inplace on the dq and dk slices. The dv
// slice is left unchanged.
void MRopeInplacePackBackward(wallx_cuda::TensorView dqkv, // [bz, seq_len, q_dim + 2*kv_dim]
                              wallx_cuda::TensorView cos,  // [3, bz, seq_len, head_dim/2]
                              wallx_cuda::TensorView sin,  // [3, bz, seq_len, head_dim/2]
                              int64_t q_num_heads,
                              int64_t kv_num_heads,
                              const int first,
                              const int second) {
    int Nthreads = 256;
    cudaStream_t stream = static_cast<cudaStream_t>(
        wallx_cuda::CurrentStream(dqkv.device().device_type, dqkv.device().device_id));

    ASSERT_CHECK(dqkv.ndim() == 3);
    ASSERT_CHECK(dqkv.stride(2) == 1);
    ASSERT_CHECK(cos.dtype().code == kDLFloat && cos.dtype().bits == 32);
    ASSERT_CHECK(sin.dtype().code == kDLFloat && sin.dtype().bits == 32);
    ASSERT_CHECK(cos.size(0) == 3);
    ASSERT_CHECK(sin.size(0) == 3);
    ASSERT_CHECK(cos.IsContiguous());
    ASSERT_CHECK(sin.IsContiguous());

    int64_t bz       = dqkv.size(0);
    int64_t seq_len  = dqkv.size(1);
    int64_t head_dim = cos.size(3) * 2;        // cos: [3, bz, seq, half_dim]
    int64_t q_dim    = q_num_heads  * head_dim;
    int64_t kv_dim   = kv_num_heads * head_dim;
    ASSERT_CHECK(dqkv.size(2) == q_dim + 2 * kv_dim);
    ASSERT_CHECK(head_dim % 8 == 0);

    int64_t dqkv_b_stride = dqkv.stride(0);
    int64_t dqkv_s_stride = dqkv.stride(1);

    dim3 grid = MakeMRopeGrid(bz, seq_len);

    if (dqkv.dtype().code == kDLFloat && dqkv.dtype().bits == 16) {
        half* dq_data = static_cast<half*>(dqkv.data_ptr());
        half* dk_data = dq_data + q_dim;
        const float* cos_data = static_cast<const float*>(cos.data_ptr());
        const float* sin_data = static_cast<const float*>(sin.data_ptr());
        MRopeKernelBackward<half><<<grid, Nthreads, head_dim * sizeof(float), stream>>>(
            cos_data, sin_data,
            dq_data, dk_data,
            dq_data, dk_data,
            first, second,
            q_num_heads, kv_num_heads, head_dim, seq_len,
            dqkv_b_stride, dqkv_s_stride, head_dim,
            dqkv_b_stride, dqkv_s_stride, head_dim);
    } else if (dqkv.dtype().code == kDLBfloat && dqkv.dtype().bits == 16) {
        __nv_bfloat16* dq_data = static_cast<__nv_bfloat16*>(dqkv.data_ptr());
        __nv_bfloat16* dk_data = dq_data + q_dim;
        const float* cos_data = static_cast<const float*>(cos.data_ptr());
        const float* sin_data = static_cast<const float*>(sin.data_ptr());
        MRopeKernelBackward<__nv_bfloat16><<<grid, Nthreads, head_dim * sizeof(float), stream>>>(
            cos_data, sin_data,
            dq_data, dk_data,
            dq_data, dk_data,
            first, second,
            q_num_heads, kv_num_heads, head_dim, seq_len,
            dqkv_b_stride, dqkv_s_stride, head_dim,
            dqkv_b_stride, dqkv_s_stride, head_dim);
    } else if (dqkv.dtype().code == kDLFloat && dqkv.dtype().bits == 32) {
        float* dq_data = static_cast<float*>(dqkv.data_ptr());
        float* dk_data = dq_data + q_dim;
        const float* cos_data = static_cast<const float*>(cos.data_ptr());
        const float* sin_data = static_cast<const float*>(sin.data_ptr());
        MRopeKernelBackward<float><<<grid, Nthreads, head_dim * sizeof(float), stream>>>(
            cos_data, sin_data,
            dq_data, dk_data,
            dq_data, dk_data,
            first, second,
            q_num_heads, kv_num_heads, head_dim, seq_len,
            dqkv_b_stride, dqkv_s_stride, head_dim,
            dqkv_b_stride, dqkv_s_stride, head_dim);
    } else {
        throw std::runtime_error("Unsupported data type for m_rope_inplace_pack_bwd");
    }

    sync_check_cuda_error();
}








} // wallx_cuda_m_rope
