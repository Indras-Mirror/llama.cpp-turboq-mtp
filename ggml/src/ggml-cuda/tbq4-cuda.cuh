#pragma once

// Turbo4 (TBQ4_0) CUDA quantize/dequant functions.
// Ported from dflash's turbo-quant-cuda.cuh — FWHT rotation + 4-bit PolarQuant.

#include "common.cuh"
#include "ggml-common.h"

// Lloyd-Max centroids for N(0, 1/sqrt(128))
static __constant__ float d_tbq4_centroids[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};

static __constant__ float d_tbq4_midpoints[15] = {
    -0.212232f, -0.162977f, -0.127056f, -0.097191f, -0.070693f,
    -0.046190f, -0.022832f,  0.000000f,  0.022832f,  0.046190f,
     0.070693f,  0.097191f,  0.127056f,  0.162977f,  0.212232f,
};

// FWHT sign arrays (seed=42)
static __constant__ float d_tbq4_wht_s1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1};
static __constant__ float d_tbq4_wht_s2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1};

// In-register FWHT butterfly (called per-thread, operates on 128-element array)
static __device__ __forceinline__
void tbq4_fwht_128(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
        }
    }
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128;
}

static __device__ __forceinline__
void tbq4_rotate_forward(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= d_tbq4_wht_s1[i];
    tbq4_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= d_tbq4_wht_s2[i];
}

static __device__ __forceinline__
void tbq4_rotate_inverse(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= d_tbq4_wht_s2[i];
    tbq4_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= d_tbq4_wht_s1[i];
}

static __device__ __forceinline__
uint8_t tbq4_find_nearest(float val) {
    if (val < d_tbq4_midpoints[7]) {
        if (val < d_tbq4_midpoints[3]) {
            if (val < d_tbq4_midpoints[1]) {
                return val < d_tbq4_midpoints[0] ? 0 : 1;
            } else {
                return val < d_tbq4_midpoints[2] ? 2 : 3;
            }
        } else {
            if (val < d_tbq4_midpoints[5]) {
                return val < d_tbq4_midpoints[4] ? 4 : 5;
            } else {
                return val < d_tbq4_midpoints[6] ? 6 : 7;
            }
        }
    } else {
        if (val < d_tbq4_midpoints[11]) {
            if (val < d_tbq4_midpoints[9]) {
                return val < d_tbq4_midpoints[8] ? 8 : 9;
            } else {
                return val < d_tbq4_midpoints[10] ? 10 : 11;
            }
        } else {
            if (val < d_tbq4_midpoints[13]) {
                return val < d_tbq4_midpoints[12] ? 12 : 13;
            } else {
                return val < d_tbq4_midpoints[14] ? 14 : 15;
            }
        }
    }
}

// SET_ROWS quantize: F32[128] → block_tbq4_0 (per-thread, in registers)
static __device__ __forceinline__
void quantize_f32_tbq4_0_block(const float * src, block_tbq4_0 * dst) {
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    float norm = sqrtf(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;

    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    tbq4_rotate_forward(x);

    for (int j = 0; j < 128; j += 2) {
        uint8_t idx0 = tbq4_find_nearest(x[j]);
        uint8_t idx1 = tbq4_find_nearest(x[j + 1]);
        dst->qs[j / 2] = (idx1 << 4) | idx0;
    }

    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        uint8_t idx = (j & 1) ? (dst->qs[j / 2] >> 4) : (dst->qs[j / 2] & 0xF);
        float r = d_tbq4_centroids[idx];
        recon_sq += r * r;
    }
    float recon_norm = sqrtf(recon_sq);
    float corrected = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
    dst->d = __float2half(corrected);
}

// Per-element dequant (NO inverse rotation) — for get_rows template
#define QR_TBQ4_0 2
static __device__ __forceinline__
void dequantize_tbq4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_tbq4_0 * x = (const block_tbq4_0 *)vx;
    const float norm = __half2float(x[ib].d);
    { const int j = iqs;
      uint8_t idx = (j & 1) ? (x[ib].qs[j / 2] >> 4) : (x[ib].qs[j / 2] & 0xF);
      v.x = d_tbq4_centroids[idx] * norm; }
    { const int j = iqs + 64;
      uint8_t idx = (j & 1) ? (x[ib].qs[j / 2] >> 4) : (x[ib].qs[j / 2] & 0xF);
      v.y = d_tbq4_centroids[idx] * norm; }
}

// Full-block dequant WITH inverse FWHT — for the ggml_cast / attention path.
// One thread block (128 threads) per quantized block.
//
// Optimized: warp-shuffle butterfly for within-warp stages (h=1..16),
// shared memory only for cross-warp stages (h=32,64). 7 barriers → 3 barriers.
// All 128 threads active through all stages.
static __global__ void k_tbq4_dequant_full(
        const block_tbq4_0 * __restrict__ src,
        float * __restrict__ dst,
        const int64_t n_blocks) {

    const int64_t bid = blockIdx.x;
    if (bid >= n_blocks) return;

    const block_tbq4_0 * b = src + bid;
    const float norm = __half2float(b->d);
    const int tid = threadIdx.x;

    // Dequant + s2 sign (fused, per-element — no sync needed)
    const uint8_t byte = b->qs[tid / 2];
    const uint8_t idx = (tid & 1) ? (byte >> 4) : (byte & 0xF);
    float val = d_tbq4_centroids[idx] * d_tbq4_wht_s2[tid];

    // Stages 0-4 (h = 1, 2, 4, 8, 16): within-warp butterfly via warp shuffle.
    // All 128 threads compute independently — no barriers, no shared memory.
    // Thread t pairs with thread t^h via __shfl_xor_sync.
    #pragma unroll
    for (int h = 1; h <= 16; h *= 2) {
        float partner = __shfl_xor_sync(0xFFFFFFFF, val, h, 32);
        if ((tid & h) == 0) {
            val = val + partner;   // lower partner
        } else {
            val = partner - val;   // upper partner
        }
    }

    // Stages 5-6 (h = 32, 64): cross-warp — need shared memory.
    // 3 barriers total (write initial, sync after stage 5, write stage 5 result).
    __shared__ float buf[128];
    buf[tid] = val;
    __syncthreads();

    // Stage 5: h=32
    {
        float partner = buf[tid ^ 32];
        if ((tid & 32) == 0) {
            val = val + partner;
        } else {
            val = partner - val;
        }
    }
    __syncthreads();
    buf[tid] = val;
    __syncthreads();

    // Stage 6: h=64
    {
        float partner = buf[tid ^ 64];
        if ((tid & 64) == 0) {
            val = val + partner;
        } else {
            val = partner - val;
        }
    }

    // Final scaling (per-element, no sync needed)
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    val *= inv_sqrt_128 * d_tbq4_wht_s1[tid];

    dst[bid * 128 + tid] = val * norm;
}

// Host launcher for full-block dequant
static void tbq4_dequant_full_cuda(
        const block_tbq4_0 * src, float * dst,
        int64_t n_blocks, cudaStream_t stream) {
    if (n_blocks <= 0) return;
    k_tbq4_dequant_full<<<(int)n_blocks, 128, 0, stream>>>(src, dst, n_blocks);
}

// ============================================================================
// Block-Diagonal Hadamard (4×32) — alternative mode with ZERO barriers.
//
// Replaces the full 128-point FWHT with 4 independent 32-point transforms.
// Each 32-point block fits entirely within a single warp (32 threads), so
// all butterfly stages use __shfl_xor_sync — no __syncthreads() needed.
//
// Quality: SAW-INT4 (Together AI + Tri Dao, arXiv:2604.19157) shows
// block-diagonal Hadamard achieves nearly identical quality to full FWHT
// for KV cache quantization.
//
// IMPORTANT: This creates an incompatible block format. GGUFs quantized
// with full 128-point FWHT CANNOT be dequantized with 32-point block-diagonal.
// Enable with env var TBQ4_BD32=1 or compile-time #define.
// ============================================================================

// 32-point FWHT butterfly (in-register, operates on 32-element array)
static __device__ __forceinline__
void tbq4_fwht_32(float * x) {
    for (int h = 1; h < 32; h *= 2) {
        for (int i = 0; i < 32; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
        }
    }
    constexpr float inv_sqrt_32 = 0.1767766952966369f;
    for (int i = 0; i < 32; i++) x[i] *= inv_sqrt_32;
}

// Block-diagonal rotation: 4 independent 32-point transforms.
// Each 32-element block gets its own s1/s2 signs from the corresponding
// slice of the 128-element sign arrays.

static __device__ __forceinline__
void tbq4_bd32_rotate_forward(float * x) {
    for (int block = 0; block < 4; block++) {
        float * bx = x + block * 32;
        for (int i = 0; i < 32; i++) bx[i] *= d_tbq4_wht_s1[block * 32 + i];
        tbq4_fwht_32(bx);
        for (int i = 0; i < 32; i++) bx[i] *= d_tbq4_wht_s2[block * 32 + i];
    }
}

static __device__ __forceinline__
void tbq4_bd32_rotate_inverse(float * x) {
    for (int block = 0; block < 4; block++) {
        float * bx = x + block * 32;
        for (int i = 0; i < 32; i++) bx[i] *= d_tbq4_wht_s2[block * 32 + i];
        tbq4_fwht_32(bx);
        for (int i = 0; i < 32; i++) bx[i] *= d_tbq4_wht_s1[block * 32 + i];
    }
}

// Quantize with block-diagonal forward rotation (SET_ROWS path).
// Produces identical block_tbq4_0 format but with 4×32 FWHT instead of 1×128.
static __device__ __forceinline__
void quantize_f32_tbq4_bd32_block(const float * src, block_tbq4_0 * dst) {
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    float norm = sqrtf(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;

    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    tbq4_bd32_rotate_forward(x);

    for (int j = 0; j < 128; j += 2) {
        uint8_t idx0 = tbq4_find_nearest(x[j]);
        uint8_t idx1 = tbq4_find_nearest(x[j + 1]);
        dst->qs[j / 2] = (idx1 << 4) | idx0;
    }

    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        uint8_t idx = (j & 1) ? (dst->qs[j / 2] >> 4) : (dst->qs[j / 2] & 0xF);
        float r = d_tbq4_centroids[idx];
        recon_sq += r * r;
    }
    float recon_norm = sqrtf(recon_sq);
    float corrected = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
    dst->d = __float2half(corrected);
}

// Dequant kernel: 32 threads (ONE WARP), ZERO barriers.
// Each thread owns 4 elements (same index within each 32-point block).
// All 4 blocks process simultaneously via 4 parallel shuffle exchanges per stage.
// 5 butterfly stages × 4 shuffle exchanges = 20 shuffle ops, 0 syncs.
static __global__ void k_tbq4_bd32_dequant_full(
        const block_tbq4_0 * __restrict__ src,
        float * __restrict__ dst,
        const int64_t n_blocks) {

    const int64_t bid = blockIdx.x;
    if (bid >= n_blocks) return;

    const block_tbq4_0 * b = src + bid;
    const float norm = __half2float(b->d);
    const int tid = threadIdx.x;  // 0..31

    // Load all 4 elements (one per block) + s2 sign
    float val0, val1, val2, val3;
    {
        // Block 0: element tid + 0
        int off = tid;
        uint8_t byte = b->qs[off / 2];
        uint8_t idx = (off & 1) ? (byte >> 4) : (byte & 0xF);
        val0 = d_tbq4_centroids[idx] * d_tbq4_wht_s2[off];

        // Block 1: element tid + 32
        off = tid + 32;
        byte = b->qs[off / 2];
        idx = (off & 1) ? (byte >> 4) : (byte & 0xF);
        val1 = d_tbq4_centroids[idx] * d_tbq4_wht_s2[off];

        // Block 2: element tid + 64
        off = tid + 64;
        byte = b->qs[off / 2];
        idx = (off & 1) ? (byte >> 4) : (byte & 0xF);
        val2 = d_tbq4_centroids[idx] * d_tbq4_wht_s2[off];

        // Block 3: element tid + 96
        off = tid + 96;
        byte = b->qs[off / 2];
        idx = (off & 1) ? (byte >> 4) : (byte & 0xF);
        val3 = d_tbq4_centroids[idx] * d_tbq4_wht_s2[off];
    }

    // Butterfly stages: h=1,2,4,8,16 — all within 32-thread warp.
    // All 4 blocks processed simultaneously via independent shuffle exchanges.
    #pragma unroll
    for (int h = 1; h < 32; h *= 2) {
        float p0 = __shfl_xor_sync(0xFFFFFFFF, val0, h, 32);
        float p1 = __shfl_xor_sync(0xFFFFFFFF, val1, h, 32);
        float p2 = __shfl_xor_sync(0xFFFFFFFF, val2, h, 32);
        float p3 = __shfl_xor_sync(0xFFFFFFFF, val3, h, 32);
        if ((tid & h) == 0) {
            val0 += p0; val1 += p1; val2 += p2; val3 += p3;
        } else {
            val0 = p0 - val0; val1 = p1 - val1;
            val2 = p2 - val2; val3 = p3 - val3;
        }
    }

    // Final scaling + s1 sign
    constexpr float inv_sqrt_32 = 0.1767766952966369f;
    dst[bid * 128 + tid + 0]  = val0 * inv_sqrt_32 * d_tbq4_wht_s1[tid + 0]  * norm;
    dst[bid * 128 + tid + 32] = val1 * inv_sqrt_32 * d_tbq4_wht_s1[tid + 32] * norm;
    dst[bid * 128 + tid + 64] = val2 * inv_sqrt_32 * d_tbq4_wht_s1[tid + 64] * norm;
    dst[bid * 128 + tid + 96] = val3 * inv_sqrt_32 * d_tbq4_wht_s1[tid + 96] * norm;
}

// Host launcher for block-diagonal dequant (32 threads/block, zero barriers)
static void tbq4_bd32_dequant_full_cuda(
        const block_tbq4_0 * src, float * dst,
        int64_t n_blocks, cudaStream_t stream) {
    if (n_blocks <= 0) return;
    k_tbq4_bd32_dequant_full<<<(int)n_blocks, 32, 0, stream>>>(src, dst, n_blocks);
}
