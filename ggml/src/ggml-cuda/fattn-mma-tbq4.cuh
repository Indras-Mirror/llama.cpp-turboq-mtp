#pragma once

#include "tbq4-cuda.cuh"

// Fused MMA-native TBQ4 flash attention: reads raw TBQ4_0 K/V directly,
// dequants (centroid*norm) into half2 shmem tiles inside the attention loop.
// Q is pre-rotated (rotate_forward) so K attention works in the rotated domain.
// V accumulates in the rotated domain; output is inverse-rotated after the kernel.

// TBQ4 tile loader: reads block_tbq4_0 from GMEM, centroid lookup * norm -> half2 shmem tile.
// No FWHT needed in the loader — we operate entirely in the rotated domain.
// Each row has D/128 TBQ4 blocks. Each block is 66 bytes (2-byte norm + 64-byte qs).
//
// OPTIMIZED: Uses all nthreads threads for full utilization. Each thread handles one
// "column group" (element index within a row) across all rows, rather than one row
// across all columns. This achieves 100% thread utilization vs 25% (nbatch_fa/nthreads).
template<int D, int stride_tile, int nbatch_fa, int nthreads, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_tbq4_load_tile(
        const char * __restrict__ data_raw,
        half2      * __restrict__ tile,
        const int stride_bytes,
        const int i_sup) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_row = D / 128;
    constexpr int elems_per_block = 64; // pairs of 4-bit values → half2
    constexpr int elems_per_row = blocks_per_row * elems_per_block; // = D/2
    const int tid = threadIdx.y * warp_size + threadIdx.x;

    // Each thread handles one column group (elem_idx) across all rows.
    // elem_idx = tid % elems_per_row, row = tid / elems_per_row
    // With 128 threads and 128 elems/row for D=256: thread N handles elem N, row 0 only.
    // For D=128 (64 elems/row): threads 0-63 handle elem 0-63, threads 64-127 handle elem 0-63 row 1.
    constexpr int elems_per_pass = nthreads <= elems_per_row ? nthreads : nthreads / 2;

    for (int base_row = 0; base_row < nbatch_fa; base_row += (nthreads + elems_per_row - 1) / elems_per_row) {
        const int idx = tid;
        const int elem_idx = idx % elems_per_row;
        const int row_offset = idx / elems_per_row;

        if (row_offset + base_row >= nbatch_fa) continue;

        const int blk_idx = elem_idx / elems_per_block;
        const int b       = elem_idx % elems_per_block;

        const char * row_ptr = data_raw + (int64_t)(base_row + row_offset) * stride_bytes;
        const block_tbq4_0 * blk = (const block_tbq4_0 *)(row_ptr) + blk_idx;
        const float norm = __half2float(__ldg(&blk->d));
        const uint8_t byte = __ldg(&blk->qs[b]);
        const half lo = __float2half(d_tbq4_centroids[byte & 0xF] * norm);
        const half hi = __float2half(d_tbq4_centroids[byte >> 4] * norm);
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

// Apply rotate_forward to Q values in shared memory tile.
// rotate_forward(x) = s2 * (1/sqrt(128)) * FWHT(s1 * x)
// Uses warp shuffles for FWHT stages 0-4, shared memory for stages 5-6.
// Each row of Q is DKQ elements, processed as DKQ/128 independent 128-point blocks.
// Called once per kernel invocation (amortized over all K/V iterations).
// q_fwht_buf: 128 floats in DYNAMIC shared memory (NOT static __shared__,
// which causes nvcc codegen issues with extern __shared__ in the same kernel).
template<int DKQ, int ncols, int nwarps>
static __device__ __noinline__ void tbq4_rotate_Q_tile(
        half2 * __restrict__ tile_Q,
        const int stride_tile_Q,
        float * __restrict__ q_fwht_buf) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_sub_blocks = DKQ / 128;
    static_assert(DKQ == 128 || DKQ == 256, "TBQ4 Q rotation only supports DKQ=128 or DKQ=256");
    const int tid = threadIdx.y * warp_size + threadIdx.x;

    for (int jc = 0; jc < ncols; ++jc) {
        if (tid < 128) {
            for (int blk = 0; blk < n_sub_blocks; blk++) {
                const int half2_offset = jc * stride_tile_Q + blk * 64 + tid / 2;
                half2 pair = tile_Q[half2_offset];
                float val = (tid & 1) ? __high2float(pair) : __low2float(pair);

                val *= d_tbq4_wht_s1[tid];

#pragma unroll
                for (int h = 1; h <= 16; h *= 2) {
                    float partner = __shfl_xor_sync(0xFFFFFFFF, val, h, 32);
                    val = (tid & h) ? (partner - val) : (val + partner);
                }

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

                float partner_val = __shfl_xor_sync(0xFFFFFFFF, val, 1, 32);
                if ((tid & 1) == 0) {
                    tile_Q[half2_offset] = __halves2half2(__float2half(val), __float2half(partner_val));
                }
            }
        }
        __syncthreads();
    }
}

// Pre-attention kernel: apply rotate_forward to Q input.
// Each row has DKQ elements, processed as DKQ/128 independent 128-point FWHT blocks.
// rotate_forward(x) = s2 * (1/sqrt(128)) * FWHT(s1 * x)
// Supports DKQ=128 (1 block) and DKQ=256 (2 blocks).
static __global__ void k_tbq4_rotate_input(
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

        val *= d_tbq4_wht_s1[tid];

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
        row_data[offset + tid] = val * inv_sqrt_128 * d_tbq4_wht_s2[tid];
    }
}

// Host function to apply rotate_forward to Q input (before FA kernel).
static void tbq4_rotate_input_cuda(
        float * data, int64_t nrows, int DKQ, cudaStream_t stream) {
    if (nrows <= 0) return;
    if (DKQ != 128 && DKQ != 256) return;
    k_tbq4_rotate_input<<<(int)nrows, 128, 0, stream>>>(data, nrows, DKQ);
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

// ============================================================================
// nstages=2 cp_async pipeline: raw TBQ4 byte copy + collaborative dequant
// ============================================================================

// Compute padded row stride for staging buffer (16-byte aligned).
// block_tbq4_0 is 66 bytes (QK_TBQ4=128). Row = blocks_per_row * 66.
// Padded to next 16-byte boundary for cp_async alignment.
template<int D>
static constexpr __host__ __device__ int tbq4_staging_row_bytes() {
    constexpr int blocks_per_row = D / 128;
    constexpr int raw_bytes      = blocks_per_row * (int)sizeof(block_tbq4_0);
    constexpr int padded         = (raw_bytes + 15) & ~15;
    return padded;
}

// Total staging buffer size for all rows in a batch.
template<int D, int nbatch_fa>
static constexpr __host__ __device__ size_t tbq4_staging_bytes() {
    return (size_t)nbatch_fa * (size_t)tbq4_staging_row_bytes<D>();
}

// cp_async raw TBQ4 block data from GMEM into staging buffer in shared memory.
// Copies entire rows including padding bytes to maintain 16-byte alignment.
// No dequant — just raw byte copy via cp.async.
//
// Template params:
//   D          - head dimension (128 or 256)
//   nbatch_fa  - number of KV rows to copy
//   nwarps     - number of warps
// Load raw TBQ4 block data from GMEM into staging buffer using regular int loads.
// Uses 4-byte int copies (33 per row for D=256) which handle misaligned addresses.
// TBQ4 rows are 132 bytes (= 4×33) — not 16-byte aligned for cp_async.
// Regular loads still benefit from warp-scheduled overlap (load+compute pipelining).
template<int D, int nbatch_fa, int nwarps>
static __device__ __forceinline__ void flash_attn_ext_tbq4_load_raw_async(
        const char * __restrict__ data_raw,
        char       * __restrict__ staging,
        const int stride_bytes,
        const int i_sup) {

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_row = D / 128;
    constexpr int raw_row_bytes  = blocks_per_row * (int)sizeof(block_tbq4_0); // 132 for D=256
    constexpr int ints_per_row   = raw_row_bytes / 4;                           // 33 for D=256
    constexpr int total_ints     = nbatch_fa * ints_per_row;
    constexpr int nthreads       = nwarps * warp_size;

    // Distribute int-sized chunks across threads for coalesced access.
    for (int idx = threadIdx.y * warp_size + threadIdx.x; idx < total_ints; idx += nthreads) {
        const int row   = idx / ints_per_row;
        const int off   = (idx % ints_per_row) * 4;

        if (row < i_sup) {
            const int src = __ldg((const int *)(data_raw + (int64_t)row * stride_bytes + off));
            *(int *)(staging + (int64_t)row * tbq4_staging_row_bytes<D>() + off) = src;
        }
        // OOB rows: staging left uninitialized. Dequant zero-fill handles cleanup.
    }
}

// Collaborative dequant: all threads convert raw TBQ4 blocks in staging
// buffer to half2 tile. Each thread handles one column group across all rows,
// same pattern as the synchronous tile loader.
//
// Template params match the synchronous loader for consistency.
template<int D, int stride_tile, int nbatch_fa, int nthreads, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_tbq4_dequant_staging(
        const char * __restrict__ staging,
        half2      * __restrict__ tile,
        const int i_sup) {

    constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_row  = D / 128;
    constexpr int elems_per_block = 64;  // 128 floats → 64 half2 pairs
    constexpr int elems_per_row   = blocks_per_row * elems_per_block;
    constexpr int row_bytes       = tbq4_staging_row_bytes<D>();
    const int tid = threadIdx.y * warp_size + threadIdx.x;

    for (int base_row = 0; base_row < nbatch_fa; base_row += (nthreads + elems_per_row - 1) / elems_per_row) {
        const int idx        = tid;
        const int elem_idx   = idx % elems_per_row;
        const int row_offset = idx / elems_per_row;

        if (row_offset + base_row >= nbatch_fa) continue;

        const int blk_idx = elem_idx / elems_per_block;
        const int b       = elem_idx % elems_per_block;

        const char * row_ptr = staging + (int64_t)(base_row + row_offset) * row_bytes;
        const block_tbq4_0 * blk = (const block_tbq4_0 *)(row_ptr) + blk_idx;
        const float norm = __half2float(blk->d);
        const uint8_t byte = blk->qs[b];
        const half lo = __float2half(d_tbq4_centroids[byte & 0xF] * norm);
        const half hi = __float2half(d_tbq4_centroids[byte >> 4] * norm);
        tile[(base_row + row_offset) * stride_tile + elem_idx] = __halves2half2(lo, hi);
    }

    // Zero-fill OOB rows
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
