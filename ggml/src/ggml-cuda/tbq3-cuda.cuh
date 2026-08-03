#pragma once
// TBQ3_0 CUDA kernels — 3-bit TurboQuant with FWHT rotation
// Block: 128 elements, fp16 norm + 48 bytes packed 3-bit = 50 bytes
// 3.0625 bpw, ~5.2x compression vs fp16

#include "common.cuh"

#define QK_TBQ3 128

// ── FWHT sign arrays (seed=42) ─────────────────────────────────────────────

__constant__ float wht_signs1_tbq3[QK_TBQ3] = {
    -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1,
    1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1,
    1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1,
    1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1,
    -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1,
    1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1,
    -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1,
    -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1,
};

__constant__ float wht_signs2_tbq3[QK_TBQ3] = {
    1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1,
    1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1,
    1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1,
    -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1,
    -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1,
    1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1,
    -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1,
    1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1,
};

// ── 3-bit FWHT centroids — Lloyd-Max for N(0, 1/sqrt(128)), 8 levels ──────

__constant__ float tbq3_centroids_3bit[8] = {
    -1.646828f, -0.895384f, -0.491349f, -0.157976f,
     0.157976f,  0.491349f,  0.895384f,  1.646828f
};

__constant__ float tbq3_midpoints_3bit[7] = {
    -1.271106f, -0.693367f, -0.324663f, 0.0f,
     0.324663f,  0.693367f,  1.271106f
};

// ── FWHT butterfly (identical to tbq4 version) ─────────────────────────────

static __device__ __forceinline__ void tbq3_fwht_128(float * x) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int i = 0; i < 128; i += 2 * len) {
            for (int j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j] = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
}

// ── FWHT rotation (forward / inverse) ──────────────────────────────────────

static __device__ __forceinline__ void tbq3_rotate_forward(
    float * rotated, const float * input, const float * s1, const float * s2
) {
    for (int j = 0; j < QK_TBQ3; j++) {
        rotated[j] = input[j] * s1[j];
    }
    tbq3_fwht_128(rotated);
    for (int j = 0; j < QK_TBQ3; j++) {
        rotated[j] *= s2[j];
    }
}

static __device__ __forceinline__ void tbq3_rotate_inverse(
    float * output, const float * rotated, const float * s1, const float * s2
) {
    for (int j = 0; j < QK_TBQ3; j++) {
        output[j] = rotated[j] * s2[j];
    }
    tbq3_fwht_128(output);
    for (int j = 0; j < QK_TBQ3; j++) {
        output[j] *= s1[j];
    }
}

// ── 3-bit quantizer (binary search over 7 midpoints) ───────────────────────

static __device__ __forceinline__ uint8_t tbq3_find_nearest(float x) {
    // Binary search over 7 midpoints to find quantized index 0-7
    if (x < tbq3_midpoints_3bit[3]) { // < 0.0
        if (x < tbq3_midpoints_3bit[1]) { // < -0.693
            if (x < tbq3_midpoints_3bit[0]) return 0; // < -1.271
            return 1; // -1.271 to -0.693
        } else {
            if (x < tbq3_midpoints_3bit[2]) return 2; // -0.693 to -0.325
            return 3; // -0.325 to 0.0
        }
    } else {
        if (x < tbq3_midpoints_3bit[5]) { // < 0.693
            if (x < tbq3_midpoints_3bit[4]) return 4; // 0.0 to 0.325
            return 5; // 0.325 to 0.693
        } else {
            if (x < tbq3_midpoints_3bit[6]) return 6; // 0.693 to 1.271
            return 7; // >= 1.271
        }
    }
}

// ── Per-block quantize (for SET_ROWS template) ─────────────────────────────

static __device__ __forceinline__
void quantize_f32_tbq3_0_block(const float * __restrict__ x, block_tbq3_0 * __restrict__ y) {
    ggml_half * d = &y->d;
    uint8_t * qs = y->qs;

    // Compute L2 norm
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TBQ3; j++) norm_sq += x[j] * x[j];
    float norm = sqrtf(norm_sq);
    if (norm < 1e-10f) norm = 1e-10f;

    // Normalize + FWHT rotation
    float unit[QK_TBQ3];
    for (int j = 0; j < QK_TBQ3; j++) unit[j] = x[j] / norm;
    tbq3_rotate_forward(unit, unit, wht_signs1_tbq3, wht_signs2_tbq3);

    // 3-bit quantize
    uint8_t indices[QK_TBQ3];
    for (int j = 0; j < QK_TBQ3; j++) indices[j] = tbq3_find_nearest(unit[j]);

    // Pack 8 × 3-bit values into 3 bytes
    for (int j = 0; j < QK_TBQ3 / 8; j++) {
        int base = j * 8;
        uint32_t packed = (uint32_t)indices[base + 0]
                        | ((uint32_t)indices[base + 1] << 3)
                        | ((uint32_t)indices[base + 2] << 6)
                        | ((uint32_t)indices[base + 3] << 9)
                        | ((uint32_t)indices[base + 4] << 12)
                        | ((uint32_t)indices[base + 5] << 15)
                        | ((uint32_t)indices[base + 6] << 18)
                        | ((uint32_t)indices[base + 7] << 21);
        qs[j * 3 + 0] = (uint8_t)(packed & 0xFF);
        qs[j * 3 + 1] = (uint8_t)((packed >> 8) & 0xFF);
        qs[j * 3 + 2] = (uint8_t)((packed >> 16) & 0xFF);
    }

    // Norm correction
    float recon_sq = 0.0f;
    for (int j = 0; j < QK_TBQ3; j++) recon_sq += tbq3_centroids_3bit[indices[j]] * tbq3_centroids_3bit[indices[j]];
    float recon_norm = sqrtf(recon_sq);
    if (recon_norm < 1e-10f) recon_norm = 1e-10f;
    *d = __float2half(norm / recon_norm);
}

// ── Per-element dequant (for get_rows — NO inverse rotation) ───────────────

static __device__ __forceinline__ float dequantize_tbq3_0(
    const uint8_t * qs, int j, const float * centroids
) {
    // Unpack 3-bit value at position j (8 values per 3 bytes)
    int byte_offset = (j / 8) * 3;
    int bit_offset  = (j % 8) * 3;
    uint32_t packed = (uint32_t)qs[byte_offset]
                    | ((uint32_t)qs[byte_offset + 1] << 8)
                    | ((uint32_t)qs[byte_offset + 2] << 16);
    uint8_t idx = (packed >> bit_offset) & 0x7;
    return centroids[idx];
}

// ── Full-block dequant with inverse FWHT (for CPY/attention) ───────────────
// Inverts quantize_f32_tbq3_0_block: centroid lookup -> s2 sign -> FWHT -> s1 sign -> x inv_sqrt_128 -> x norm.
// block_tbq3_0 layout: {ggml_half d, uint8_t qs[48]} = 50 bytes total.

static __global__ void k_tbq3_dequant_full(
    const uint8_t * __restrict__ src, float * __restrict__ y, int64_t k
) {
    const int tid = threadIdx.x;
    const int block_idx = blockIdx.x;

    __shared__ float shared[QK_TBQ3];

    constexpr int block_size = sizeof(ggml_half) + QK_TBQ3 * 3 / 8; // 50
    const uint8_t * block_start = src + block_idx * block_size;
    const ggml_half * block_d = (const ggml_half *)block_start;
    const uint8_t * block_qs = block_start + sizeof(ggml_half);
    const float norm_corrected = __half2float(*block_d);

    int base_byte = (tid / 8) * 3;
    int bit_offset = (tid % 8) * 3;
    uint32_t packed = (uint32_t)block_qs[base_byte]
                    | ((uint32_t)block_qs[base_byte + 1] << 8)
                    | ((uint32_t)block_qs[base_byte + 2] << 16);
    uint8_t idx = (packed >> bit_offset) & 0x7;

    // inverse rotation: s2 sign, then FWHT, then s1 sign, then ×inv_sqrt_128
    // (d = norm/recon_norm assumes norm-preserving rotation; unnormalized FWHT scales by sqrt(128))
    // Race-free butterfly (same pattern as k_tbq4_dequant_full): stages h=1..16 via
    // warp shuffle, h=32/64 via per-thread shared slots with barriers. The previous
    // all-threads-in-place shared FWHT raced across warps and produced garbage.
    float val = tbq3_centroids_3bit[idx] * wht_signs2_tbq3[tid];

    #pragma unroll
    for (int h = 1; h <= 16; h *= 2) {
        float partner = __shfl_xor_sync(0xFFFFFFFF, val, h, 32);
        if ((tid & h) == 0) {
            val = val + partner;   // lower partner
        } else {
            val = partner - val;   // upper partner
        }
    }

    // Stage 5 (h=32): cross-warp — shared memory
    shared[tid] = val;
    __syncthreads();
    {
        float partner = shared[tid ^ 32];
        if ((tid & 32) == 0) {
            val = val + partner;
        } else {
            val = partner - val;
        }
    }
    __syncthreads();

    // Stage 6 (h=64): cross-warp — shared memory
    shared[tid] = val;
    __syncthreads();
    {
        float partner = shared[tid ^ 64];
        if ((tid & 64) == 0) {
            val = val + partner;
        } else {
            val = partner - val;
        }
    }

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    y[block_idx * QK_TBQ3 + tid] = val * wht_signs1_tbq3[tid] * inv_sqrt_128 * norm_corrected;
}

static void tbq3_dequant_full_cuda(
    const void * src, float * dst, int64_t n_blocks, cudaStream_t stream
) {
    k_tbq3_dequant_full<<<n_blocks, QK_TBQ3, 0, stream>>>(
        (const uint8_t *)src, dst, n_blocks * QK_TBQ3);
}
