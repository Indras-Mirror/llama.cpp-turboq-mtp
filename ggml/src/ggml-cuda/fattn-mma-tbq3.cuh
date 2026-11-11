#pragma once

#include "tbq3-cuda.cuh"

// Fused MMA-native TBQ3 flash attention: reads raw TBQ3_0 K/V directly,
// dequants (centroid*norm) into half2 shmem tiles inside the attention loop.
// Q is pre-rotated (rotate_forward) so K attention works in the rotated domain.
// V accumulates in the rotated domain; output is inverse-rotated after the kernel.
//
// Mechanical port of the TBQ4 fused kernel (fattn-mma-tbq4.cuh). The MMA math is
// type-agnostic; only the tile load/dequant and the FWHT sign arrays differ.
// TBQ3 uses wht_signs1/2_tbq3 (matching quantize_f32_tbq3_0_block, the DSV4 cache
// quantizer) — NOT the TBQ4 d_tbq4_wht_s1/s2 arrays.

// TBQ3 tile loader: reads block_tbq3_0 from GMEM, centroid lookup * norm -> half2 shmem tile.
// No FWHT needed in the loader — we operate entirely in the rotated domain.
// Each row has D/128 TBQ3 blocks. Each block is 50 bytes (2-byte norm + 48-byte qs).
// 3-bit packing: 8 values per 3 bytes, cross-byte bit fields (little-endian).
// A half2 pair (values 2b, 2b+1) always lies within one 8-value group, so a single
// 24-bit load covers both values: group = b/4, shift = (b%4)*6.
//
// Same thread mapping as the TBQ4 loader: one thread per "column group" (elem_idx)
// across all rows — 100% thread utilization.
template<int D, int stride_tile, int nbatch_fa, int nthreads, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_tbq3_load_tile(
        const char * __restrict__ data_raw,
        half2      * __restrict__ tile,
        const int stride_bytes,
        const int i_sup) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_row = D / 128;
    constexpr int elems_per_block = 64; // 128 3-bit values → 64 half2 pairs
    constexpr int elems_per_row = blocks_per_row * elems_per_block; // = D/2
    const int tid = threadIdx.y * warp_size + threadIdx.x;

    for (int base_row = 0; base_row < nbatch_fa; base_row += (nthreads + elems_per_row - 1) / elems_per_row) {
        const int idx = tid;
        const int elem_idx = idx % elems_per_row;
        const int row_offset = idx / elems_per_row;

        if (row_offset + base_row >= nbatch_fa) continue;

        const int blk_idx = elem_idx / elems_per_block;
        const int b       = elem_idx % elems_per_block; // pair index within block (0..63)

        const char * row_ptr = data_raw + (int64_t)(base_row + row_offset) * stride_bytes;
        const block_tbq3_0 * blk = (const block_tbq3_0 *)(row_ptr) + blk_idx;
        const float norm = __half2float(__ldg(&blk->d));

        const int group = b / 4;            // 8-value byte group within block (0..15)
        const int shift = (b % 4) * 6;      // bit offset of value 2b within the 24-bit group
        const uint32_t packed = (uint32_t)__ldg(&blk->qs[group*3 + 0])
                              | ((uint32_t)__ldg(&blk->qs[group*3 + 1]) << 8)
                              | ((uint32_t)__ldg(&blk->qs[group*3 + 2]) << 16);
        const uint8_t idx0 = (packed >> shift) & 0x7;
        const uint8_t idx1 = (packed >> (shift + 3)) & 0x7;
        const half lo = __float2half(tbq3_centroids_3bit[idx0] * norm);
        const half hi = __float2half(tbq3_centroids_3bit[idx1] * norm);
        tile[(base_row + row_offset) * stride_tile + elem_idx] = __halves2half2(lo, hi);
    }

    // Zero-fill OOB rows — fallback to strided pattern for simplicity
    if constexpr (oob_check) {
        for (int r = tid; r < nbatch_fa; r += nthreads) {
            if (r >= i_sup) {
                for (int e = 0; e < elems_per_row; ++e) {
                    tile[r * stride_tile + e] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }
}

// Pre-attention kernel: apply rotate_forward to Q input.
// Each row has DKQ elements, processed as DKQ/128 independent 128-point FWHT blocks.
// rotate_forward(x) = s2 * (1/sqrt(128)) * FWHT(s1 * x) — TBQ3 sign arrays.
// Supports DKQ=128 (1 block) and DKQ=256 (2 blocks).
static __global__ void k_tbq3_rotate_input(
        float * __restrict__ data,
        const int64_t nrows,
        const int DKQ) {
    const int64_t row = blockIdx.x;
    if (row >= nrows) return;

    const int tid = threadIdx.x;
    float * row_data = data + row * DKQ;
    const int n_blocks = DKQ / 128;

    __shared__ float buf[128];

    for (int blk = 0; blk < n_blocks; blk++) {
        const int offset = blk * 128;
        float val = row_data[offset + tid];

        val *= wht_signs1_tbq3[tid];

        // FWHT stages 0-4: warp shuffles
#pragma unroll
        for (int h = 1; h <= 16; h *= 2) {
            float partner = __shfl_xor_sync(0xFFFFFFFF, val, h, 32);
            val = (tid & h) ? (partner - val) : (val + partner);
        }

        // Stages 5-6: shared memory
        buf[tid] = val;
        __syncthreads();
        {
            float p = buf[tid ^ 32];
            val = (tid & 32) ? (p - val) : (val + p);
        }
        buf[tid] = val;
        __syncthreads();
        {
            float p = buf[tid ^ 64];
            val = (tid & 64) ? (p - val) : (val + p);
        }

        constexpr float inv_sqrt_128 = 0.08838834764831845f;
        row_data[offset + tid] = val * inv_sqrt_128 * wht_signs2_tbq3[tid];
    }
}

// Host function to apply rotate_forward to Q input (before FA kernel).
static void tbq3_rotate_input_cuda(
        float * data, int64_t nrows, int DKQ, cudaStream_t stream) {
    if (nrows <= 0) return;
    if (DKQ != 128 && DKQ != 256) return;
    k_tbq3_rotate_input<<<(int)nrows, 128, 0, stream>>>(data, nrows, DKQ);
}

// Post-attention kernel: apply rotate_inverse to VKQ output.
// Each row has DV elements, processed as DV/128 independent 128-point FWHT blocks.
// rotate_inverse(y) = s1 * (1/sqrt(128)) * FWHT(s2 * y) — TBQ3 sign arrays.
// Supports DV=128 (1 block) and DV=256 (2 blocks).
static __global__ void k_tbq3_rotate_output(
        float * __restrict__ data,
        const int64_t nrows,
        const int DV) {
    const int64_t row = blockIdx.x;
    if (row >= nrows) return;

    const int tid = threadIdx.x;
    float * row_data = data + row * DV;
    const int n_blocks = DV / 128;

    __shared__ float buf[128];

    for (int blk = 0; blk < n_blocks; blk++) {
        const int offset = blk * 128;
        float val = row_data[offset + tid];

        val *= wht_signs2_tbq3[tid];

        // FWHT stages 0-4: warp shuffles
#pragma unroll
        for (int h = 1; h <= 16; h *= 2) {
            float partner = __shfl_xor_sync(0xFFFFFFFF, val, h, 32);
            val = (tid & h) ? (partner - val) : (val + partner);
        }

        // Stages 5-6: shared memory
        buf[tid] = val;
        __syncthreads();
        {
            float p = buf[tid ^ 32];
            val = (tid & 32) ? (p - val) : (val + p);
        }
        buf[tid] = val;
        __syncthreads();
        {
            float p = buf[tid ^ 64];
            val = (tid & 64) ? (p - val) : (val + p);
        }

        constexpr float inv_sqrt_128 = 0.08838834764831845f;
        row_data[offset + tid] = val * inv_sqrt_128 * wht_signs1_tbq3[tid];
    }
}

// Host function to apply rotate_inverse to flash attention output.
static void tbq3_rotate_output_cuda(
        float * data, int64_t nrows, int DV, cudaStream_t stream) {
    if (nrows <= 0) return;
    if (DV != 128 && DV != 256) return;
    k_tbq3_rotate_output<<<(int)nrows, 128, 0, stream>>>(data, nrows, DV);
}

// NOTE: The nstages=2 cp_async staging pipeline (raw byte copy + collaborative
// dequant, see fattn-mma-tbq4.cuh) is intentionally NOT ported for TBQ3. TBQ3 rows
// are 50 B (D=128) / 100 B (D=256); 50 % 4 = 2 breaks the int-granular copy, and the
// shipped build config (TBQ4_KV_FUSED) forces nstages=0 for the TBQ path anyway —
// the synchronous flash_attn_ext_tbq3_load_tile above is the active path. nstages is
// forced to 0 for TBQ3 in fattn-mma-f16.cuh so the staging branches never run on
// TBQ3 data in any config.
