// TurboQuant reference helpers for the CPU path.

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-turboq.h"
#include "ggml-turboq-tables.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TURBOQ_TLS __thread
#elif defined(_MSC_VER)
#define TURBOQ_TLS __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_THREADS__)
#define TURBOQ_TLS _Thread_local
#else
#define TURBOQ_TLS
#endif

static inline uint64_t splitmix64_next(uint64_t * state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void turboq_generate_gaussian(float * out, int64_t n, uint64_t seed) {
    uint64_t state = seed;
    int64_t i = 0;
    for (; i + 1 < n; i += 2) {
        // Generate two uniform (0,1) variates
        double u1 = ((double)(splitmix64_next(&state) >> 11) + 0.5) / (double)(1ULL << 53);
        double u2 = ((double)(splitmix64_next(&state) >> 11) + 0.5) / (double)(1ULL << 53);
        double r  = sqrt(-2.0 * log(u1));
        double th = 2.0 * 3.14159265358979323846 * u2;
        out[i]     = (float)(r * cos(th));
        out[i + 1] = (float)(r * sin(th));
    }
    if (i < n) {
        double u1 = ((double)(splitmix64_next(&state) >> 11) + 0.5) / (double)(1ULL << 53);
        double u2 = ((double)(splitmix64_next(&state) >> 11) + 0.5) / (double)(1ULL << 53);
        double r  = sqrt(-2.0 * log(u1));
        double th = 2.0 * 3.14159265358979323846 * u2;
        out[i] = (float)(r * cos(th));
    }
}

// ---------------------------------------------------------------------------
// Householder QR decomposition (in-place, no LAPACK dependency)
//
// Input:  A[d*d] stored column-major (A[i + j*d] = A_{i,j})
// Output: Q[d*d] column-major orthogonal matrix, with Haar sign correction
//
// Uses Householder reflections: Q = H_1 * H_2 * ... * H_d where
// H_k = I - 2 * v_k * v_k^T / (v_k^T * v_k)
// ---------------------------------------------------------------------------

// Compute Q from Householder QR of column-major matrix A[d×d].
// A is modified in-place (becomes R on upper triangle, v below diagonal).
// Q is written to Q_out[d×d] column-major.
// Applies Haar sign correction: Q[:,j] *= sign(R[j,j]) so that Q is
// uniformly distributed on O(d) (Haar measure).
static void turboq_householder_qr(float * A, float * Q_out, int64_t d) {
    float * tau = (float *)malloc(d * sizeof(float));
    // Store sign(R[k,k]) = -sign(alpha_k) for Haar correction
    float * r_sign = (float *)malloc(d * sizeof(float));

    for (int64_t k = 0; k < d; k++) {
        // Compute norm of A[k:d, k]
        float norm_sq = 0.0f;
        for (int64_t i = k; i < d; i++) {
            float val = A[i + k * d];
            norm_sq += val * val;
        }
        float norm = sqrtf(norm_sq);

        // Choose sign to avoid cancellation
        float alpha = A[k + k * d];
        float sign_alpha = (alpha >= 0.0f) ? 1.0f : -1.0f;
        float u1 = alpha + sign_alpha * norm;

        // R[k,k] = -sign(alpha) * norm, so sign(R[k,k]) = -sign(alpha)
        r_sign[k] = -sign_alpha;

        // Compute tau = 2 / (v^T v)
        float vtv = u1 * u1 + (norm_sq - alpha * alpha);
        if (vtv < 1e-30f) {
            tau[k] = 0.0f;
            continue;
        }
        tau[k] = 2.0f / vtv;

        // Store v in A[k:d, k]
        A[k + k * d] = u1;

        // Apply H_k to remaining columns A[k:d, k+1:d]
        for (int64_t j = k + 1; j < d; j++) {
            float dot = 0.0f;
            dot += u1 * A[k + j * d];
            for (int64_t i = k + 1; i < d; i++) {
                dot += A[i + k * d] * A[i + j * d];
            }
            dot *= tau[k];
            A[k + j * d] -= dot * u1;
            for (int64_t i = k + 1; i < d; i++) {
                A[i + j * d] -= dot * A[i + k * d];
            }
        }
    }

    // Build Q by back-accumulation: Q = H_1 * H_2 * ... * H_{d-1}
    memset(Q_out, 0, d * d * sizeof(float));
    for (int64_t i = 0; i < d; i++) {
        Q_out[i + i * d] = 1.0f;
    }

    for (int64_t k = d - 1; k >= 0; k--) {
        if (tau[k] == 0.0f) continue;
        float u1 = A[k + k * d];
        for (int64_t j = 0; j < d; j++) {
            float dot = 0.0f;
            dot += u1 * Q_out[k + j * d];
            for (int64_t i = k + 1; i < d; i++) {
                dot += A[i + k * d] * Q_out[i + j * d];
            }
            dot *= tau[k];
            Q_out[k + j * d] -= dot * u1;
            for (int64_t i = k + 1; i < d; i++) {
                Q_out[i + j * d] -= dot * A[i + k * d];
            }
        }
    }

    // Haar sign correction: Q[:,j] *= sign(R[j,j])
    // This ensures Q is uniformly distributed on O(d), not just SO(d).
    // Reference: Mezzadri (2007), "How to Generate Random Matrices from the Classical Compact Groups"
    for (int64_t j = 0; j < d; j++) {
        if (r_sign[j] < 0.0f) {
            for (int64_t i = 0; i < d; i++) {
                Q_out[i + j * d] = -Q_out[i + j * d];
            }
        }
    }

    free(tau);
    free(r_sign);
}

// ---------------------------------------------------------------------------
// Rotation matrix cache
//
// For a given (dimension, seed) pair, generate and cache the d×d orthogonal Q.
// The cache is thread-local to avoid locks. In practice, all rows of a weight
// matrix share the same dimension, so the cache hit rate is ~100%.
// ---------------------------------------------------------------------------

static TURBOQ_TLS float * tl_Q = NULL;
static TURBOQ_TLS float * tl_Q_row = NULL;
static TURBOQ_TLS int64_t tl_Q_dim = 0;
static TURBOQ_TLS uint64_t tl_Q_seed = 0;

static const float * turboq_get_rotation(int64_t d, uint64_t seed) {
    if (tl_Q != NULL && tl_Q_dim == d && tl_Q_seed == seed) {
        return tl_Q;
    }
    // Regenerate
    free(tl_Q);
    free(tl_Q_row);
    tl_Q = (float *)malloc(d * d * sizeof(float));
    tl_Q_row = (float *)malloc(d * d * sizeof(float));
    tl_Q_dim = d;
    tl_Q_seed = seed;

    // Generate d×d Gaussian random matrix (column-major)
    float * A = (float *)malloc(d * d * sizeof(float));
    turboq_generate_gaussian(A, d * d, seed);

    // Compute QR, store Q in tl_Q
    turboq_householder_qr(A, tl_Q, d);

    for (int64_t i = 0; i < d; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            tl_Q_row[i * d + j] = tl_Q[i + j * d];
        }
    }

    free(A);
    return tl_Q;
}

static const float * turboq_get_rotation_row(int64_t d, uint64_t seed) {
    turboq_get_rotation(d, seed);
    return tl_Q_row;
}

// ---------------------------------------------------------------------------
// Projection matrix cache (for Q_prod QJL stage)
//
// S is a d×d random Gaussian matrix (NOT orthogonalized), used for QJL:
//   qjl_signs = sign(S · residual)
//   dequant:    sqrt(pi/2)/d · gamma · S^T · signs
// Uses a different seed stream from the rotation matrix Q.
// ---------------------------------------------------------------------------

static TURBOQ_TLS float * tl_S = NULL;
static TURBOQ_TLS float * tl_S_row = NULL;
static TURBOQ_TLS int64_t tl_S_dim = 0;
static TURBOQ_TLS uint64_t tl_S_seed = 0;

static const float * turboq_get_projection(int64_t d, uint64_t seed) {
    // Use a different seed stream for S vs Q
    uint64_t s_seed = seed ^ 0x1234567890abcdefULL;
    if (tl_S != NULL && tl_S_dim == d && tl_S_seed == s_seed) {
        return tl_S;
    }
    free(tl_S);
    free(tl_S_row);
    tl_S = (float *)malloc(d * d * sizeof(float));
    tl_S_row = (float *)malloc(d * d * sizeof(float));
    tl_S_dim = d;
    tl_S_seed = s_seed;

    // Generate d×d Gaussian random matrix (column-major), no QR
    turboq_generate_gaussian(tl_S, d * d, s_seed);

    for (int64_t i = 0; i < d; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            tl_S_row[i * d + j] = tl_S[i + j * d];
        }
    }

    return tl_S;
}

static const float * turboq_get_projection_row(int64_t d, uint64_t seed) {
    turboq_get_projection(d, seed);
    return tl_S_row;
}

// ---------------------------------------------------------------------------
// Dense matrix-vector multiply: y = M * x  (M is d×d column-major)
// ---------------------------------------------------------------------------

static void matvec(float * y, const float * M, const float * x, int64_t d) {
    for (int64_t i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int64_t j = 0; j < d; j++) {
            sum += M[i + j * d] * x[j]; // M[i,j] = M[i + j*d] (column-major)
        }
        y[i] = sum;
    }
}

#if defined(__AVX2__)
static inline float turboq_hsum_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif

static void matvec_row(float * y, const float * M, const float * x, int64_t d) {
    for (int64_t i = 0; i < d; ++i) {
        const float * row = M + i * d;
        float sum = 0.0f;
        int64_t j = 0;
#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; j + 7 < d; j += 8) {
            const __m256 mv = _mm256_loadu_ps(row + j);
            const __m256 xv = _mm256_loadu_ps(x + j);
#if defined(__FMA__)
            acc = _mm256_fmadd_ps(mv, xv, acc);
#else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(mv, xv));
#endif
        }
        sum += turboq_hsum_avx(acc);
#endif
        for (; j < d; ++j) {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

// ---------------------------------------------------------------------------
// Dense matrix-transpose-vector multiply: y = M^T * x  (M is d×d column-major)
// ---------------------------------------------------------------------------

static void matvec_t(float * y, const float * M, const float * x, int64_t d) {
    for (int64_t j = 0; j < d; j++) {
        const float * col = M + j * d;
        float sum = 0.0f;
        int64_t i = 0;
#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; i + 7 < d; i += 8) {
            const __m256 mv = _mm256_loadu_ps(col + i);
            const __m256 xv = _mm256_loadu_ps(x + i);
#if defined(__FMA__)
            acc = _mm256_fmadd_ps(mv, xv, acc);
#else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(mv, xv));
#endif
        }
        sum += turboq_hsum_avx(acc);
#endif
        for (; i < d; ++i) {
            sum += col[i] * x[i]; // M^T[j,i] = M[i,j] = M[i + j*d]
        }
        y[j] = sum;
    }
}

// ---------------------------------------------------------------------------
// Public API (kept for compatibility, now wraps dense rotation)
// ---------------------------------------------------------------------------

// The rotation matrix is a global parameter (same for all vectors), per the paper.
// This seed is used to deterministically generate both Q and S matrices.
uint64_t turboq_seed_from_row(int64_t row_idx) {
    (void)row_idx;
    return 0x517cc1b727220a95ULL;
}

// Forward rotation: y = Q · x  (paper Algorithm 1, line 5: y <- Pi . x)
void turboq_rotate_forward(float * y, const float * x, int64_t d, uint64_t seed) {
    const float * Q = turboq_get_rotation_row(d, seed);
    matvec_row(y, Q, x, d);
}

// Inverse rotation: x = Q^T · y  (paper Algorithm 1, line 10: x_tilde <- Pi^T . y_tilde)
void turboq_rotate_inverse(float * x, const float * y, int64_t d, uint64_t seed) {
    const float * Q = turboq_get_rotation(d, seed);
    matvec_t(x, Q, y, d);
}

// ---------------------------------------------------------------------------
// Scratch buffer (thread-local, for temporary vectors)
// ---------------------------------------------------------------------------

static TURBOQ_TLS float * tl_buf = NULL;
static TURBOQ_TLS int64_t tl_buf_size = 0;

static float * turboq_get_scratch(int64_t n) {
    if (n > tl_buf_size) {
        free(tl_buf);
        tl_buf = (float *)malloc(n * sizeof(float));
        tl_buf_size = n;
    }
    return tl_buf;
}

// Second scratch buffer (needed when two temp vectors are required simultaneously,
// e.g. rotated-domain values + original-domain result in dequant)
static TURBOQ_TLS float * tl_buf2 = NULL;
static TURBOQ_TLS int64_t tl_buf2_size = 0;

static float * turboq_get_scratch2(int64_t n) {
    if (n > tl_buf2_size) {
        free(tl_buf2);
        tl_buf2 = (float *)malloc(n * sizeof(float));
        tl_buf2_size = n;
    }
    return tl_buf2;
}

// Third scratch buffer (needed by Q_prod dequant which requires three simultaneous vectors:
// mse_rot, signs_f, and mse_unit)
static TURBOQ_TLS float * tl_buf3 = NULL;
static TURBOQ_TLS int64_t tl_buf3_size = 0;

static float * turboq_get_scratch3(int64_t n) {
    if (n > tl_buf3_size) {
        free(tl_buf3);
        tl_buf3 = (float *)malloc(n * sizeof(float));
        tl_buf3_size = n;
    }
    return tl_buf3;
}

#define TURBOQ_KV_DIM 128

static inline float turboq_block_scale_up(void) {
    return sqrtf((float) QK_K);
}

static inline float turboq_block_scale_down(void) {
    return 1.0f / turboq_block_scale_up();
}

static void turboq_rotate_block_forward(float * y, const float * x, uint64_t seed) {
    const float * Q = turboq_get_rotation_row(TURBOQ_KV_DIM, seed);

    for (int64_t i = 0; i < QK_K; i += TURBOQ_KV_DIM) {
        matvec_row(y + i, Q, x + i, TURBOQ_KV_DIM);
    }
}

static void turboq_rotate_block_inverse(float * x, const float * y, uint64_t seed) {
    const float * Q = turboq_get_rotation(TURBOQ_KV_DIM, seed);

    for (int64_t i = 0; i < QK_K; i += TURBOQ_KV_DIM) {
        matvec_t(x + i, Q, y + i, TURBOQ_KV_DIM);
    }
}

static void turboq_project_block(float * y, const float * x, uint64_t seed) {
    const float * S = turboq_get_projection_row(TURBOQ_KV_DIM, seed);

    for (int64_t i = 0; i < QK_K; i += TURBOQ_KV_DIM) {
        matvec_row(y + i, S, x + i, TURBOQ_KV_DIM);
    }
}

static void turboq_project_block_inverse(float * x, const float * y, uint64_t seed) {
    const float * S = turboq_get_projection(TURBOQ_KV_DIM, seed);

    for (int64_t i = 0; i < QK_K; i += TURBOQ_KV_DIM) {
        matvec_t(x + i, S, y + i, TURBOQ_KV_DIM);
    }
}

static void turboq_rotate_qk_forward(float * y, const float * x, uint64_t seed) {
    const float * Q = turboq_get_rotation_row(QK_K, seed);
    matvec_row(y, Q, x, QK_K);
}

static void turboq_rotate_qk_inverse(float * x, const float * y, uint64_t seed) {
    const float * Q = turboq_get_rotation(QK_K, seed);
    matvec_t(x, Q, y, QK_K);
}

static void turboq_project_qk(float * y, const float * x, uint64_t seed) {
    const float * S = turboq_get_projection_row(QK_K, seed);
    matvec_row(y, S, x, QK_K);
}

static void turboq_project_qk_inverse(float * x, const float * y, uint64_t seed) {
    const float * S = turboq_get_projection(QK_K, seed);
    matvec_t(x, S, y, QK_K);
}

// ---------------------------------------------------------------------------
// Scalar codebook quantization
// ---------------------------------------------------------------------------

static inline uint8_t quantize_scalar(float val, const float * boundaries, int n_boundaries) {
    for (int i = 0; i < n_boundaries; i++) {
        if (val < boundaries[i]) {
            return (uint8_t)i;
        }
    }
    return (uint8_t)n_boundaries;
}

static inline uint8_t quantize_scalar_3bit(float val) {
    return quantize_scalar(val, turboq_boundaries_3bit, 7);
}

static inline uint8_t quantize_scalar_2bit(float val) {
    return quantize_scalar(val, turboq_boundaries_2bit, 3);
}

static inline uint8_t quantize_scalar_4bit(float val) {
    return quantize_scalar(val, turboq_boundaries_4bit, 15);
}

// ---------------------------------------------------------------------------
// 3-bit packing/unpacking
// ---------------------------------------------------------------------------

static void pack_3bit(uint8_t * dst, const uint8_t * indices, int64_t n) {
    int64_t full_groups = n / 8;
    for (int64_t g = 0; g < full_groups; g++) {
        const uint8_t * idx = indices + g * 8;
        uint32_t bits = 0;
        for (int j = 0; j < 8; j++) {
            bits |= ((uint32_t)(idx[j] & 0x7)) << (j * 3);
        }
        dst[g * 3 + 0] = (uint8_t)(bits & 0xFF);
        dst[g * 3 + 1] = (uint8_t)((bits >> 8) & 0xFF);
        dst[g * 3 + 2] = (uint8_t)((bits >> 16) & 0xFF);
    }
}

static void unpack_3bit(uint8_t * indices, const uint8_t * src, int64_t n) {
    int64_t full_groups = n / 8;
    for (int64_t g = 0; g < full_groups; g++) {
        uint32_t bits = (uint32_t)src[g * 3 + 0]
                     | ((uint32_t)src[g * 3 + 1] << 8)
                     | ((uint32_t)src[g * 3 + 2] << 16);
        for (int j = 0; j < 8; j++) {
            indices[g * 8 + j] = (uint8_t)((bits >> (j * 3)) & 0x7);
        }
    }
}

// ---------------------------------------------------------------------------
// TBQ3_0: TurboQuant 3-bit
// ---------------------------------------------------------------------------

static void tbq4_fwht_128(float * x); // forward decl — defined below in TBQ4_0 section

void quantize_row_tbq3_0_ref(const float * GGML_RESTRICT x, block_tbq3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TBQ3 == 0);
    const int64_t nb = k / QK_TBQ3;

    for (int64_t b = 0; b < nb; b++) {
        const float * xb = x + b * QK_TBQ3;

        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TBQ3; ++j) norm_sq += xb[j] * xb[j];
        float norm = sqrtf(norm_sq);
        if (norm < 1e-10f) norm = 1e-10f;

        // Normalize + FWHT rotation (s1 + FWHT + s2, seed=42)
        float rotated[QK_TBQ3];
        for (int j = 0; j < QK_TBQ3; ++j) rotated[j] = xb[j] / norm * turboq_wht_signs1[j];
        tbq4_fwht_128(rotated);
        for (int j = 0; j < QK_TBQ3; ++j) rotated[j] *= turboq_wht_signs2[j];

        // 3-bit quantize via binary search over FWHT centroids
        uint8_t indices[QK_TBQ3];
        for (int j = 0; j < QK_TBQ3; j++) {
            float v = rotated[j];
            int idx = 3; // start at middle
            if      (v < turboq_fwht_midpoints_3bit[0]) idx = 0;
            else if (v < turboq_fwht_midpoints_3bit[1]) idx = 1;
            else if (v < turboq_fwht_midpoints_3bit[2]) idx = 2;
            else if (v < turboq_fwht_midpoints_3bit[3]) idx = 3;
            else if (v < turboq_fwht_midpoints_3bit[4]) idx = 4;
            else if (v < turboq_fwht_midpoints_3bit[5]) idx = 5;
            else if (v < turboq_fwht_midpoints_3bit[6]) idx = 6;
            else                                         idx = 7;
            indices[j] = idx;
        }

        // Pack 8 × 3-bit values into 3 bytes
        memset(y[b].qs, 0, QK_TBQ3 * 3 / 8);
        for (int j = 0; j < QK_TBQ3; j++) {
            int byte_off = (j / 8) * 3 + (j % 8) / 8 * 0; // simplified: j/8*3
            // Repack: 8 values per 3 bytes, little-endian bit packing
            int block = j / 8;
            int bit = (j % 8) * 3;
            y[b].qs[block * 3 + 0] |= (indices[j] & 0x7) << (bit < 8 ? bit : 0);
            if (bit >= 8) y[b].qs[block * 3 + 1] |= ((indices[j] >> (8 - bit)) & 0x7);
        }

        // Norm correction: corrected = original / reconstruction norm
        float recon_sq = 0.0f;
        for (int j = 0; j < QK_TBQ3; j++) recon_sq += turboq_fwht_centroids_3bit[indices[j]] * turboq_fwht_centroids_3bit[indices[j]];
        float recon_norm = sqrtf(recon_sq);
        if (recon_norm < 1e-10f) recon_norm = 1e-10f;
        y[b].d = GGML_FP32_TO_FP16(norm / recon_norm);
    }
}

void dequantize_row_tbq3_0(const block_tbq3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TBQ3 == 0);
    const int64_t nb = k / QK_TBQ3;

    for (int64_t b = 0; b < nb; b++) {
        const float norm_corrected = GGML_FP16_TO_FP32(x[b].d);

        // Unpack 3-bit values + centroid lookup
        float rotated[QK_TBQ3];
        for (int j = 0; j < QK_TBQ3; j++) {
            int block = j / 8;
            int bit = (j % 8) * 3;
            uint8_t idx;
            if (bit < 6) {
                idx = (x[b].qs[block * 3 + 0] >> bit) & 0x7;
            } else if (bit < 14) {
                // Crosses byte boundary
                uint32_t val = x[b].qs[block * 3 + 0] | (x[b].qs[block * 3 + 1] << 8);
                idx = (val >> bit) & 0x7;
            } else {
                uint32_t val = x[b].qs[block * 3 + 1] | (x[b].qs[block * 3 + 2] << 8);
                idx = (val >> (bit - 8)) & 0x7;
            }
            rotated[j] = turboq_fwht_centroids_3bit[idx];
        }

        // Inverse FWHT
        for (int j = 0; j < QK_TBQ3; ++j) rotated[j] *= turboq_wht_signs2[j];
        tbq4_fwht_128(rotated);
        for (int j = 0; j < QK_TBQ3; ++j) rotated[j] *= turboq_wht_signs1[j];

        // Scale by corrected norm
        for (int j = 0; j < QK_TBQ3; ++j) y[b * QK_TBQ3 + j] = rotated[j] * norm_corrected;
    }
}

size_t quantize_tbq3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_TBQ3 == 0);
    const int64_t nb_per_row = n_per_row / QK_TBQ3;
    const size_t row_size = nb_per_row * sizeof(block_tbq3_0);
    for (int64_t row = 0; row < nrows; row++) {
        const float * row_src = src + row * n_per_row;
        block_tbq3_0 * row_dst = (block_tbq3_0 *)((char *)dst + row * row_size);
        quantize_row_tbq3_0_ref(row_src, row_dst, n_per_row);
    }
    return nrows * row_size;
}

// ---------------------------------------------------------------------------
// TBQ4_0: TurboQuant 4-bit (FWHT rotation, 128-element blocks)
//
// Matches the Turbo4 CUDA algorithm: FWHT butterfly rotation → 4-bit
// PolarQuant codebook → norm correction. CPU and CUDA produce identical output.
// ---------------------------------------------------------------------------

static void tbq4_fwht_128(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128;
}

static void tbq4_rotate_forward(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= turboq_wht_signs1[i];
    tbq4_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= turboq_wht_signs2[i];
}

static void tbq4_rotate_inverse(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= turboq_wht_signs2[i];
    tbq4_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= turboq_wht_signs1[i];
}

static inline uint8_t tbq4_find_nearest(float val) {
    return quantize_scalar(val, turboq_fwht_midpoints_4bit, 15);
}

void quantize_row_tbq4_0_ref(const float * GGML_RESTRICT x, block_tbq4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TBQ4 == 0);
    const int64_t nb = k / QK_TBQ4;

    for (int64_t b = 0; b < nb; b++) {
        const float * xb = x + b * QK_TBQ4;

        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TBQ4; j++) norm_sq += xb[j] * xb[j];
        float norm = sqrtf(norm_sq);
        float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;

        float rot[128];
        for (int j = 0; j < 128; j++) rot[j] = xb[j] * inv_norm;
        tbq4_rotate_forward(rot);

        for (int j = 0; j < 128; j += 2) {
            uint8_t idx0 = tbq4_find_nearest(rot[j]);
            uint8_t idx1 = tbq4_find_nearest(rot[j + 1]);
            y[b].qs[j / 2] = (idx1 << 4) | idx0;
        }

        float recon_sq = 0.0f;
        for (int j = 0; j < 128; j++) {
            uint8_t idx = (j & 1) ? (y[b].qs[j / 2] >> 4) : (y[b].qs[j / 2] & 0xF);
            float r = turboq_fwht_centroids_4bit[idx];
            recon_sq += r * r;
        }
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
        y[b].d = GGML_FP32_TO_FP16(corrected);
    }
}

void dequantize_row_tbq4_0(const block_tbq4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TBQ4 == 0);
    const int64_t nb = k / QK_TBQ4;

    for (int64_t b = 0; b < nb; b++) {
        const float norm = GGML_FP16_TO_FP32(x[b].d);
        float rot[128];

        for (int j = 0; j < 128; j++) {
            uint8_t idx = (j & 1) ? (x[b].qs[j / 2] >> 4) : (x[b].qs[j / 2] & 0xF);
            rot[j] = turboq_fwht_centroids_4bit[idx];
        }

        tbq4_rotate_inverse(rot);

        for (int j = 0; j < 128; j++) {
            y[b * QK_TBQ4 + j] = rot[j] * norm;
        }
    }
}

// Planar3/Iso3 CPU reference dequant/quant — provided by ggml-planar-quant.c and ggml-iso-quant.c
// (ported from rotorquant fork with Givens/quaternion inverse rotation)

size_t quantize_tbq4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_TBQ4 == 0);

    const int64_t nb_per_row = n_per_row / QK_TBQ4;
    const size_t row_size = nb_per_row * sizeof(block_tbq4_0);

    for (int64_t row = 0; row < nrows; row++) {
        const float * row_src = src + row * n_per_row;
        block_tbq4_0 * row_dst = (block_tbq4_0 *)((char *)dst + row * row_size);
        quantize_row_tbq4_0_ref(row_src, row_dst, n_per_row);
    }
    return nrows * row_size;
}
