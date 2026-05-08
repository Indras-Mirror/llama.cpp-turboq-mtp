#pragma once

// Sparse V dequant: attention-aware skip for TBQ4_0 KV cache.
//
// At long context (>8K), 90%+ of post-softmax attention weights are near-zero.
// Skipping V dequant for those positions saves compute with negligible quality loss.
//
// Reference: TheTom's implementation — +22.8% decode speed at 32K context, PPL unchanged.
// Reddit: "Skipping 90% of KV dequant work" (r/LocalLLaMA)

// Threshold for sparse V dequant.
// Positions with attention weight below this get zero V contribution.
// Only activates at sequence lengths > 8192 (no benefit at short context).
static __device__ __forceinline__
float tbq4_sparse_v_threshold(const int seq_len) {
    if (seq_len <= 8192) return 0.0f;
    // 0.25x the average attention weight (= 1/seq_len)
    return 1.0f / (4.0f * (float)seq_len);
}

// Returns true if a V position should be SKIPPED (not dequantized).
// Call this in the flash attention inner loop before loading V.
// When threshold is 0.0 (feature disabled / short context), never skips.
static __device__ __forceinline__
bool tbq4_sparse_v_skip(const float attn_weight, const float threshold) {
    return (threshold > 0.0f) & (attn_weight < threshold);
}

// Convenience: compute threshold once and check in one call.
// Returns true if the position should be skipped.
static __device__ __forceinline__
bool tbq4_sparse_v_check(const float attn_weight, const int seq_len) {
    const float thresh = tbq4_sparse_v_threshold(seq_len);
    return tbq4_sparse_v_skip(attn_weight, thresh);
}
