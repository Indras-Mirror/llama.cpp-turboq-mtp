#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// Initialize rotation constants for planar/iso CUDA kernels.
// Must be called before any planar/iso FA or cpy kernel invocation.
void ggml_cuda_init_planar_iso_constants();

void ggml_cuda_cpy_f16_planar3(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_planar4(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_iso3(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_iso4(const char * src, char * dst, int64_t ne, cudaStream_t stream);

// Dequant: planar3/iso3 → F32 (applies inverse rotation)
void ggml_cuda_cpy_planar3_f32(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_iso3_f32(const char * src, char * dst, int64_t ne, cudaStream_t stream);

// Dequant: planar4/iso4 → F32 (applies inverse rotation)
void ggml_cuda_cpy_planar4_f32(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_iso4_f32(const char * src, char * dst, int64_t ne, cudaStream_t stream);
