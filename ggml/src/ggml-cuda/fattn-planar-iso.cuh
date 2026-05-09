#pragma once

// PlanarQuant/IsoQuant vec_dot_KQ and dequantize_V functions for the VEC
// flash attention path. Included by fattn-common.cuh after planar-iso-constants.cuh.
//
// Planar3/Iso3: 2-bit + 1-bit sign packed (same block layout, different rotation).
// Planar4/Iso4: 4-bit nibble-packed (reuses block_tbq4_0 layout, different rotation).

// ── Planar3 KQ dot product: Givens inverse rotation + centroid lookup ──
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_planar3_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_planar3_0 * K = (const block_planar3_0 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int k_KQ = k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne + k_KQ_1;
            const int elem0 = k_KQ * 2;
            const int ib = elem0 / QK_PLANAR3;
            const int j0 = elem0 % QK_PLANAR3;

            const float norm = __half2float(K[ib].d);
            const uint8_t qs_byte = K[ib].qs[j0/4];
            const uint8_t sgn_byte = K[ib].signs[j0/8];
            const int shift = (j0%4)*2;
            const uint8_t idx0 = ((qs_byte >> shift) & 0x3) | (((sgn_byte >> (j0%8)) & 0x1) << 2);
            const uint8_t idx1 = ((qs_byte >> (shift+2)) & 0x3) | (((sgn_byte >> (j0%8+1)) & 0x1) << 2);

            float q0 = PI_CENTROIDS_3BIT[idx0];
            float q1 = PI_CENTROIDS_3BIT[idx1];

            // Inverse Givens rotation
            int p = (ib * QK_PLANAR3 + j0) / 2;
            float c = PI_COS[p % 64];
            float s = PI_SIN[p % 64];
            float2 kv;
            kv.x = ( c * q0 + s * q1) * norm;
            kv.y = (-s * q0 + c * q1) * norm;

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 qv = ((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            ggml_cuda_mad(sum, make_float2(kv.x, kv.y), __half22float2(qv));
#else
            const float2 qv = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            sum += kv.x * qv.x + kv.y * qv.y;
#endif
        }
    }
    return sum;
}

// ── Iso3 KQ dot product: quaternion inverse rotation ──
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_iso3_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_iso3_0 * K = (const block_iso3_0 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int k_KQ = k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne + k_KQ_1;
            const int elem0 = k_KQ * 2;
            const int ib = elem0 / QK_ISO3;
            const int j0 = elem0 % QK_ISO3;

            const float norm = __half2float(K[ib].d);

            // Unpack all 4 elements of the quaternion group
            int g = (ib * QK_ISO3 + j0) / 4;
            int offset = j0 % 4;
            int base = (g % 32) * 4;
            float qvals[4];
            for (int c = 0; c < 4; c++) {
                int jj = base + c;
                uint8_t low = (K[ib].qs[jj/4] >> ((jj%4)*2)) & 0x3;
                uint8_t hi = (K[ib].signs[jj/8] >> (jj%8)) & 0x1;
                qvals[c] = PI_CENTROIDS_3BIT[low | (hi << 2)];
            }

            // Inverse quaternion
            int qg = g % 32;
            float qw = PI_QW[qg], qx = -PI_QX[qg], qy = -PI_QY[qg], qz = -PI_QZ[qg];
            float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
            float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
            float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
            float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];

            float results[4] = {rw, rx, ry, rz};
            float2 kv;
            kv.x = results[offset] * norm;
            kv.y = results[offset + 1] * norm;

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 qv = ((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            ggml_cuda_mad(sum, make_float2(kv.x, kv.y), __half22float2(qv));
#else
            const float2 qv = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            sum += kv.x * qv.x + kv.y * qv.y;
#endif
        }
    }
    return sum;
}

// ── Planar4 KQ dot product: Givens inverse + 4-bit centroids ──
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_planar4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_planar4_0 * K = (const block_planar4_0 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int k_KQ = k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne + k_KQ_1;
            const int elem0 = k_KQ * 2;
            const int ib = elem0 / QK_PLANAR4;
            const int j0 = elem0 % QK_PLANAR4;

            const float norm = __half2float(K[ib].d);
            const uint8_t qs_byte = K[ib].qs[j0 / 2];
            const uint8_t idx0 = (qs_byte >> 0) & 0xF;
            const uint8_t idx1 = (qs_byte >> 4) & 0xF;

            float q0 = PI_CENTROIDS_4BIT[idx0];
            float q1 = PI_CENTROIDS_4BIT[idx1];

            int p = (ib * QK_PLANAR4 + j0) / 2;
            float c = PI_COS[p % 64];
            float s = PI_SIN[p % 64];
            float2 kv;
            kv.x = ( c * q0 + s * q1) * norm;
            kv.y = (-s * q0 + c * q1) * norm;

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 qv = ((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            ggml_cuda_mad(sum, make_float2(kv.x, kv.y), __half22float2(qv));
#else
            const float2 qv = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            sum += kv.x * qv.x + kv.y * qv.y;
#endif
        }
    }
    return sum;
}

// ── Iso4 KQ dot product: quaternion inverse + 4-bit ──
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_iso4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_iso4_0 * K = (const block_iso4_0 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int k_KQ = k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne + k_KQ_1;
            const int elem0 = k_KQ * 2;
            const int ib = elem0 / QK_ISO4;
            const int j0 = elem0 % QK_ISO4;

            const float norm = __half2float(K[ib].d);

            int g = (ib * QK_ISO4 + j0) / 4;
            int offset = j0 % 4;
            int base = (g % 32) * 4;
            float qvals[4];
            for (int c = 0; c < 4; c++) {
                int jj = base + c;
                uint8_t idx = (K[ib].qs[jj/2] >> ((jj%2)*4)) & 0xF;
                qvals[c] = PI_CENTROIDS_4BIT[idx];
            }

            int qg = g % 32;
            float qw = PI_QW[qg], qx = -PI_QX[qg], qy = -PI_QY[qg], qz = -PI_QZ[qg];
            float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
            float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
            float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
            float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];

            float results[4] = {rw, rx, ry, rz};
            float2 kv;
            kv.x = results[offset] * norm;
            kv.y = results[offset + 1] * norm;

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 qv = ((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            ggml_cuda_mad(sum, make_float2(kv.x, kv.y), __half22float2(qv));
#else
            const float2 qv = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            sum += kv.x * qv.x + kv.y * qv.y;
#endif
        }
    }
    return sum;
}

// ── Planar3 V dequantize: inverse Givens rotation ──
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_planar3_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_planar3_0 * x = (const block_planar3_0 *) vx;
    const int64_t ib = i0 / QK_PLANAR3;
    const int     j0 = i0 % QK_PLANAR3;
    const float   norm = __half2float(x[ib].d);

    static_assert(ne == 2 || ne == 4, "bad ne");

    auto unpack3 = [&](int j) -> uint8_t {
        uint8_t low = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
        uint8_t hi  = (x[ib].signs[j/8] >> (j%8)) & 0x1;
        return low | (hi << 2);
    };

    if constexpr (ne == 4) {
        float q0 = PI_CENTROIDS_3BIT[unpack3(j0)];
        float q1 = PI_CENTROIDS_3BIT[unpack3(j0+1)];
        float q2 = PI_CENTROIDS_3BIT[unpack3(j0+2)];
        float q3 = PI_CENTROIDS_3BIT[unpack3(j0+3)];

        int p0 = j0 / 2;
        float c0 = PI_COS[p0], s0 = PI_SIN[p0];
        float r0 = ( c0 * q0 + s0 * q1) * norm;
        float r1 = (-s0 * q0 + c0 * q1) * norm;

        int p1 = (j0 + 2) / 2;
        float c1 = PI_COS[p1], s1 = PI_SIN[p1];
        float r2 = ( c1 * q2 + s1 * q3) * norm;
        float r3 = (-s1 * q2 + c1 * q3) * norm;

#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            ((half2 *)dst)[0] = make_half2(__float2half(r0), __float2half(r1));
            ((half2 *)dst)[1] = make_half2(__float2half(r2), __float2half(r3));
        } else
#endif
        if constexpr (std::is_same_v<T, float>) {
            ((float2 *)dst)[0] = make_float2(r0, r1);
            ((float2 *)dst)[1] = make_float2(r2, r3);
        }
    } else { // ne == 2
        float q0 = PI_CENTROIDS_3BIT[unpack3(j0)];
        float q1 = PI_CENTROIDS_3BIT[unpack3(j0+1)];

        int p = j0 / 2;
        float c = PI_COS[p], s = PI_SIN[p];
        float r0 = ( c * q0 + s * q1) * norm;
        float r1 = (-s * q0 + c * q1) * norm;

#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            ((half2 *)dst)[0] = make_half2(__float2half(r0), __float2half(r1));
        } else
#endif
        if constexpr (std::is_same_v<T, float>) {
            ((float *)dst)[0] = r0; ((float *)dst)[1] = r1;
        }
    }
}

// ── Iso3 V dequantize: inverse quaternion rotation ──
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_iso3_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_iso3_0 * x = (const block_iso3_0 *) vx;
    const int64_t ib = i0 / QK_ISO3;
    const int     j0 = i0 % QK_ISO3;
    const float   norm = __half2float(x[ib].d);

    static_assert(ne == 2 || ne == 4, "bad ne");

    auto unpack3 = [&](int j) -> uint8_t {
        uint8_t low = (((const block_planar3_0 *)&x[ib])->qs[j/4] >> ((j%4)*2)) & 0x3;
        uint8_t hi  = (((const block_planar3_0 *)&x[ib])->signs[j/8] >> (j%8)) & 0x1;
        return low | (hi << 2);
    };

    int g = j0 / 4;
    int offset = j0 % 4;

    float qvals[4];
    for (int c = 0; c < 4; c++) {
        qvals[c] = PI_CENTROIDS_3BIT[unpack3(g*4 + c)];
    }

    float qw = PI_QW[g], qx = -PI_QX[g], qy = -PI_QY[g], qz = -PI_QZ[g];
    float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
    float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
    float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
    float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];

    float results[4] = {rw * norm, rx * norm, ry * norm, rz * norm};

    if constexpr (ne == 4) {
#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            ((half2 *)dst)[0] = make_half2(__float2half(results[0]), __float2half(results[1]));
            ((half2 *)dst)[1] = make_half2(__float2half(results[2]), __float2half(results[3]));
        } else
#endif
        if constexpr (std::is_same_v<T, float>) {
            ((float2 *)dst)[0] = make_float2(results[0], results[1]);
            ((float2 *)dst)[1] = make_float2(results[2], results[3]);
        }
    } else {
#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            ((half2 *)dst)[0] = make_half2(__float2half(results[offset]), __float2half(results[offset+1]));
        } else
#endif
        if constexpr (std::is_same_v<T, float>) {
            ((float *)dst)[0] = results[offset]; ((float *)dst)[1] = results[offset+1];
        }
    }
}

// ── Planar4 V dequantize: inverse Givens + 4-bit ──
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_planar4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_planar4_0 * x = (const block_planar4_0 *) vx;
    const int64_t ib = i0 / QK_PLANAR4;
    const int     j0 = i0 % QK_PLANAR4;
    const float   norm = __half2float(x[ib].d);

    static_assert(ne == 2 || ne == 4, "bad ne");

    auto unpack4 = [&](int j) -> uint8_t {
        return (x[ib].qs[j/2] >> ((j%2)*4)) & 0xF;
    };

    if constexpr (ne == 4) {
        float q0 = PI_CENTROIDS_4BIT[unpack4(j0)];
        float q1 = PI_CENTROIDS_4BIT[unpack4(j0+1)];
        float q2 = PI_CENTROIDS_4BIT[unpack4(j0+2)];
        float q3 = PI_CENTROIDS_4BIT[unpack4(j0+3)];

        int p0 = j0 / 2;
        float c0 = PI_COS[p0], s0 = PI_SIN[p0];
        float r0 = ( c0 * q0 + s0 * q1) * norm;
        float r1 = (-s0 * q0 + c0 * q1) * norm;

        int p1 = (j0 + 2) / 2;
        float c1 = PI_COS[p1], s1 = PI_SIN[p1];
        float r2 = ( c1 * q2 + s1 * q3) * norm;
        float r3 = (-s1 * q2 + c1 * q3) * norm;

#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            ((half2 *)dst)[0] = make_half2(__float2half(r0), __float2half(r1));
            ((half2 *)dst)[1] = make_half2(__float2half(r2), __float2half(r3));
        } else
#endif
        if constexpr (std::is_same_v<T, float>) {
            ((float2 *)dst)[0] = make_float2(r0, r1);
            ((float2 *)dst)[1] = make_float2(r2, r3);
        }
    } else {
        float q0 = PI_CENTROIDS_4BIT[unpack4(j0)];
        float q1 = PI_CENTROIDS_4BIT[unpack4(j0+1)];

        int p = j0 / 2;
        float c = PI_COS[p], s = PI_SIN[p];
        float r0 = ( c * q0 + s * q1) * norm;
        float r1 = (-s * q0 + c * q1) * norm;

#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            ((half2 *)dst)[0] = make_half2(__float2half(r0), __float2half(r1));
        } else
#endif
        if constexpr (std::is_same_v<T, float>) {
            ((float *)dst)[0] = r0; ((float *)dst)[1] = r1;
        }
    }
}

// ── Iso4 V dequantize: inverse quaternion + 4-bit ──
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_iso4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_iso4_0 * x = (const block_iso4_0 *) vx;
    const int64_t ib = i0 / QK_ISO4;
    const int     j0 = i0 % QK_ISO4;
    const float   norm = __half2float(x[ib].d);

    static_assert(ne == 2 || ne == 4, "bad ne");

    auto unpack4 = [&](int j) -> uint8_t {
        return (x[ib].qs[j/2] >> ((j%2)*4)) & 0xF;
    };

    int g = j0 / 4;
    int offset = j0 % 4;

    float qvals[4];
    for (int c = 0; c < 4; c++) {
        qvals[c] = PI_CENTROIDS_4BIT[unpack4(g*4 + c)];
    }

    float qw = PI_QW[g], qx = -PI_QX[g], qy = -PI_QY[g], qz = -PI_QZ[g];
    float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
    float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
    float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
    float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];

    float results[4] = {rw * norm, rx * norm, ry * norm, rz * norm};

    if constexpr (ne == 4) {
#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            ((half2 *)dst)[0] = make_half2(__float2half(results[0]), __float2half(results[1]));
            ((half2 *)dst)[1] = make_half2(__float2half(results[2]), __float2half(results[3]));
        } else
#endif
        if constexpr (std::is_same_v<T, float>) {
            ((float2 *)dst)[0] = make_float2(results[0], results[1]);
            ((float2 *)dst)[1] = make_float2(results[2], results[3]);
        }
    } else {
#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            ((half2 *)dst)[0] = make_half2(__float2half(results[offset]), __float2half(results[offset+1]));
        } else
#endif
        if constexpr (std::is_same_v<T, float>) {
            ((float *)dst)[0] = results[offset]; ((float *)dst)[1] = results[offset+1];
        }
    }
}
