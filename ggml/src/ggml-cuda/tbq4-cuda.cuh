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
// One thread block (128 threads) per quantized block. Uses shared memory for butterfly.
static __global__ void k_tbq4_dequant_full(
        const block_tbq4_0 * __restrict__ src,
        float * __restrict__ dst,
        const int64_t n_blocks) {

    const int64_t bid = blockIdx.x;
    if (bid >= n_blocks) return;

    const block_tbq4_0 * b = src + bid;
    const float norm = __half2float(b->d);
    const int tid = threadIdx.x;

    __shared__ float buf[128];

    // Dequant to rotated domain
    uint8_t byte = b->qs[tid / 2];
    uint8_t idx = (tid & 1) ? (byte >> 4) : (byte & 0xF);
    buf[tid] = d_tbq4_centroids[idx];
    __syncthreads();

    // Inverse FWHT: s2 → butterfly → s1
    buf[tid] *= d_tbq4_wht_s2[tid];
    __syncthreads();

    for (int h = 1; h < 128; h *= 2) {
        if (tid < 64) {
            int j = (tid / h) * (2 * h) + (tid % h);
            float a = buf[j], bv = buf[j + h];
            buf[j] = a + bv; buf[j + h] = a - bv;
        }
        __syncthreads();
    }

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    buf[tid] *= inv_sqrt_128 * d_tbq4_wht_s1[tid];
    __syncthreads();

    dst[bid * 128 + tid] = buf[tid] * norm;
}

// Host launcher for full-block dequant
static void tbq4_dequant_full_cuda(
        const block_tbq4_0 * src, float * dst,
        int64_t n_blocks, cudaStream_t stream) {
    if (n_blocks <= 0) return;
    k_tbq4_dequant_full<<<(int)n_blocks, 128, 0, stream>>>(src, dst, n_blocks);
}
