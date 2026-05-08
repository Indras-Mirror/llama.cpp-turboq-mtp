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

static __device__ __forceinline__ void tbq3_fwht_128_shared(float * x) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int i = 0; i < 128; i += 2 * len) {
            for (int j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j] = u + v;
                x[i + j + len] = u - v;
            }
        }
        __syncthreads();
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

static __global__ void k_tbq3_dequant_full(
    const uint8_t * __restrict__ qs, float * __restrict__ y, int64_t k
) {
    const int tid = threadIdx.x;
    const int block_idx = blockIdx.x;

    __shared__ float shared[QK_TBQ3];

    // Read fp16 norm from end of block
    const ggml_half * d = (const ggml_half *)(qs + block_idx * QK_TBQ3 / 2); // wrong offset
    // Actually: block size = QK_TBQ3 * 3 / 8 = 48 bytes + 2 bytes norm = 50 bytes
    const uint8_t * block_qs = qs + block_idx * (QK_TBQ3 * 3 / 8 + sizeof(ggml_half));
    const ggml_half * block_d = (const ggml_half *)(block_qs + QK_TBQ3 * 3 / 8);
    const float norm_corrected = __half2float(*block_d);

    // Unpack 3-bit values
    float scale_down = 1.0f / sqrtf((float)QK_TBQ3);
    int base_byte = (tid / 8) * 3;
    int bit_offset = (tid % 8) * 3;
    uint32_t packed = (uint32_t)block_qs[base_byte]
                    | ((uint32_t)block_qs[base_byte + 1] << 8)
                    | ((uint32_t)block_qs[base_byte + 2] << 16);
    uint8_t idx = (packed >> bit_offset) & 0x7;
    shared[tid] = tbq3_centroids_3bit[idx];

    __syncthreads();

    // Inverse FWHT via shared memory butterfly
    tbq3_fwht_128_shared(shared);

    // Normalize result
    y[block_idx * QK_TBQ3 + tid] = shared[tid] * norm_corrected;
}

static void tbq3_dequant_full_cuda(
    const void * src, float * dst, int64_t n_blocks, cudaStream_t stream
) {
    k_tbq3_dequant_full<<<n_blocks, QK_TBQ3, 0, stream>>>(
        (const uint8_t *)src, dst, n_blocks * QK_TBQ3);
}
