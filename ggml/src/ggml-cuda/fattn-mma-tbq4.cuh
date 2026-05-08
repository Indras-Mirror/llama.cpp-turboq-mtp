#pragma once

#include "tbq4-cuda.cuh"

// Fused MMA-native TBQ4 flash attention: reads raw TBQ4_0 K/V directly,
// dequants (centroid*norm) into half2 shmem tiles inside the attention loop.
// Q is pre-rotated (rotate_forward) so K attention works in the rotated domain.
// V accumulates in the rotated domain; output is inverse-rotated after the kernel.

// TBQ4 tile loader: reads block_tbq4_0 from GMEM, centroid lookup * norm -> half2 shmem tile.
// No FWHT needed in the loader — we operate entirely in the rotated domain.
// Each row has D/128 TBQ4 blocks. Each block is 66 bytes (2-byte norm + 64-byte qs).
template<int D, int stride_tile, int nbatch_fa, int nthreads, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_tbq4_load_tile(
        const char * __restrict__ data_raw,
        half2      * __restrict__ tile,
        const int stride_bytes,
        const int i_sup) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_row = D / 128;
    const int tid = threadIdx.y * warp_size + threadIdx.x;

    for (int row = tid; row < nbatch_fa; row += nthreads) {
        if (oob_check && row >= i_sup) {
#pragma unroll
            for (int b = 0; b < D/2; ++b) {
                tile[row * stride_tile + b] = make_half2(0.0f, 0.0f);
            }
            continue;
        }

        const char * row_ptr = data_raw + (int64_t)row * stride_bytes;

#pragma unroll
        for (int blk_idx = 0; blk_idx < blocks_per_row; ++blk_idx) {
            const block_tbq4_0 * blk = (const block_tbq4_0 *)(row_ptr) + blk_idx;
            const float norm = __half2float(blk->d);

            half cn_h[16];
#pragma unroll
            for (int c = 0; c < 16; c++) {
                cn_h[c] = __float2half(d_tbq4_centroids[c] * norm);
            }

#pragma unroll
            for (int b = 0; b < 64; ++b) {
                const uint8_t byte = blk->qs[b];
                tile[row * stride_tile + blk_idx * 64 + b] = __halves2half2(cn_h[byte & 0xF], cn_h[byte >> 4]);
            }
        }
    }
}

// Apply rotate_forward to Q values in shared memory tile.
// rotate_forward(x) = s2 * (1/sqrt(128)) * FWHT(s1 * x)
// Uses warp shuffles for FWHT stages 0-4, shared memory for stages 5-6.
// Each row of Q is DKQ elements, processed as DKQ/128 independent 128-point blocks.
// Called once per kernel invocation (amortized over all K/V iterations).
template<int DKQ, int ncols, int nwarps>
static __device__ __forceinline__ void tbq4_rotate_Q_tile(
        half2 * __restrict__ tile_Q,
        const int stride_tile_Q) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_sub_blocks = DKQ / 128;
    static_assert(DKQ == 128 || DKQ == 256, "TBQ4 Q rotation only supports DKQ=128 or DKQ=256");
    const int tid = threadIdx.y * warp_size + threadIdx.x;

    __shared__ float q_fwht_buf[128];

    for (int jc = 0; jc < ncols; ++jc) {
        if (tid < 128) {
            for (int blk = 0; blk < n_sub_blocks; blk++) {
                const int half2_offset = jc * stride_tile_Q + blk * 64 + tid / 2;
                half2 pair = tile_Q[half2_offset];
                float val = (tid & 1) ? __high2float(pair) : __low2float(pair);

                val *= d_tbq4_wht_s1[tid];

                // FWHT stages 0-4: intra-warp via shuffles
#pragma unroll
                for (int h = 1; h <= 16; h *= 2) {
                    float partner = __shfl_xor_sync(0xFFFFFFFF, val, h, 32);
                    val = (tid & h) ? (partner - val) : (val + partner);
                }

                // Stages 5-6: cross-warp via shared memory
                q_fwht_buf[tid] = val;
                __syncthreads();
                {
                    float p = q_fwht_buf[tid ^ 32];
                    val = (tid & 32) ? (p - val) : (val + p);
                }
                q_fwht_buf[tid] = val;
                __syncthreads();
                {
                    float p = q_fwht_buf[tid ^ 64];
                    val = (tid & 64) ? (p - val) : (val + p);
                }

                constexpr float inv_sqrt_128 = 0.08838834764831845f;
                val *= inv_sqrt_128 * d_tbq4_wht_s2[tid];

                // Write back as half2 — pair adjacent elements
                float partner_val = __shfl_xor_sync(0xFFFFFFFF, val, 1, 32);
                if ((tid & 1) == 0) {
                    tile_Q[half2_offset] = __halves2half2(__float2half(val), __float2half(partner_val));
                }
            }
        }
        __syncthreads();
    }
}

// Post-attention kernel: apply rotate_inverse to VKQ output.
// Each row has DV elements, processed as DV/128 independent 128-point FWHT blocks.
// rotate_inverse(y) = s1 * (1/sqrt(128)) * FWHT(s2 * y)
// Supports DV=128 (1 block) and DV=256 (2 blocks).
static __global__ void k_tbq4_rotate_output(
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

        val *= d_tbq4_wht_s2[tid];

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
        row_data[offset + tid] = val * inv_sqrt_128 * d_tbq4_wht_s1[tid];
    }
}

// Host function to apply rotate_inverse to flash attention output.
static void tbq4_rotate_output_cuda(
        float * data, int64_t nrows, int DV, cudaStream_t stream) {
    if (nrows <= 0) return;
    if (DV != 128 && DV != 256) return;
    k_tbq4_rotate_output<<<(int)nrows, 128, 0, stream>>>(data, nrows, DV);
}
