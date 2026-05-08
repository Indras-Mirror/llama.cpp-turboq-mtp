#pragma once

// fattn-mma-f16.cuh must be included before this header (provides flash_attn_ext_f16,
// launch_fattn, fattn_kernel_t, and the MMA helper functions).
// fattn-mma-tbq4.cuh must also be included (provides TBQ4 tile loader, Q rotation,
// output rotation).

// TBQ4 fused MMA flash attention launcher.
// Mirrors the turbo launcher pattern: forces nstages=0, V_is_K_view=false,
// need_f16_K=false, need_f16_V=false. Raw TBQ4 data passes through to kernel.

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_tbq4_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);

#ifdef TBQ4_NSTAGES_2
    const int nstages = 2;
#else
    const int nstages = 0;
#endif

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    constexpr bool V_is_K_view = false;

    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(nbatch_K2 + 4,  nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_KV_2stage = nbatch_fa            *         (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nstages <= 1 ? nbytes_shared_KV_1stage : nbytes_shared_KV_2stage;

    // Layout: tile_Q | tile_K (if !Q_in_reg) | tile_V (if nstages>1) | tile_mask | tbq4_staging
    const size_t nbytes_shared_base = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    // nbatch_fa is runtime; compute staging size explicitly.
    const size_t nbytes_shared_staging_actual = nstages > 1
        ? (size_t)nbatch_fa * (size_t)(((DKQ/128) * sizeof(block_tbq4_0) + 15) & ~15)
        : 0;

    const size_t nbytes_shared_total = nbytes_shared_base + nbytes_shared_staging_actual;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    constexpr ggml_type tK = GGML_TYPE_TBQ4_0;
    constexpr ggml_type tV = GGML_TYPE_TBQ4_0;

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, tK, tV>;

#if !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, tK, tV>;

#if !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif
    }

    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false, true, warp_size_host);
}


#define DECL_FATTN_MMA_TBQ4_CASE(DKQ, DV, ncols1, ncols2)                              \
    template void ggml_cuda_flash_attn_ext_mma_tbq4_case                                \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)       \

#define DECL_FATTN_MMA_TBQ4_CASE_ALL_NCOLS2(DKQ, DV, ncols)    \
    extern DECL_FATTN_MMA_TBQ4_CASE(DKQ, DV, (ncols)/ 1,  1); \
    extern DECL_FATTN_MMA_TBQ4_CASE(DKQ, DV, (ncols)/ 2,  2); \
    extern DECL_FATTN_MMA_TBQ4_CASE(DKQ, DV, (ncols)/ 4,  4); \
    extern DECL_FATTN_MMA_TBQ4_CASE(DKQ, DV, (ncols)/ 8,  8); \

DECL_FATTN_MMA_TBQ4_CASE_ALL_NCOLS2(128, 128,  8)
DECL_FATTN_MMA_TBQ4_CASE_ALL_NCOLS2(128, 128, 16)
DECL_FATTN_MMA_TBQ4_CASE_ALL_NCOLS2(128, 128, 32)
DECL_FATTN_MMA_TBQ4_CASE_ALL_NCOLS2(128, 128, 64)

DECL_FATTN_MMA_TBQ4_CASE_ALL_NCOLS2(256, 256,  8)
DECL_FATTN_MMA_TBQ4_CASE_ALL_NCOLS2(256, 256, 16)
DECL_FATTN_MMA_TBQ4_CASE_ALL_NCOLS2(256, 256, 32)
DECL_FATTN_MMA_TBQ4_CASE_ALL_NCOLS2(256, 256, 64)
