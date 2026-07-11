#include "gated_delta_net_pipe.cuh"
#include "common.cuh"

// Chunk pipeline kernel.
// Grid:  (H_v, n_seqs)  — each block handles one (v_head, seq)
// Block: S_v threads     — each lane handles one state column
//
// Processes all chunks sequentially within each block.
// The 4 matmuls per chunk are done as simple sequential loops
// (no shared memory for v_t_new exchange — correct, can be optimized later).

template <int S_v, int CS, bool KDA>
__global__ void __launch_bounds__(S_v, 2)
gated_delta_net_pipeline_cuda(
        const float * k_cd,
        const float * v_t,
        const float * kq,
        const float * q_g_exp,
        const float * kg_t_g_last,
        const float * init_state,
        float       * dst,
        int64_t       H_v,
        int64_t       H_k,
        int64_t       n_tokens,
        int64_t       n_chunks,
        int64_t       n_seqs) {
    const int64_t v_head = blockIdx.x;
    const int64_t seq    = blockIdx.y;
    const int lane       = threadIdx.x;  // state column index
    const int64_t k_head = KDA ? v_head : (v_head % H_k);

    if (v_head >= H_v) return;

    // State: each lane holds the full column vector for its column
    float s_col[S_v];
    const int64_t s_base = (seq * H_v + v_head) * S_v * S_v;
    for (int r = 0; r < S_v; r++) {
        s_col[r] = init_state[s_base + lane * S_v + r];
    }

    // Precompute strides
    const int64_t cs_s  = S_v;
    const int64_t ch_s  = S_v * CS;
    const int64_t head_k_s = CS * S_v * n_chunks;
    const int64_t head_v_s = CS * S_v * n_chunks;

    for (int ch = 0; ch < n_chunks; ch++) {
        // ── Matmul 1: v_prime[cs] = sum_i k_cd[i][cs] * state[i][col] ──
        const int64_t kcd_off = v_head * head_k_s + ch * ch_s;
        float v_prime[CS] = {0.0f};
        for (int i = 0; i < S_v; i++) {
            float si = s_col[i];
            for (int cs = 0; cs < CS; cs++) {
                v_prime[cs] += k_cd[kcd_off + cs * cs_s + i] * si;
            }
        }

        // ── Step 2: v_t_new[cs] = v_t[cs][col] - v_prime[cs] ──
        // v_t: [CS, S_v, n_chunks, H_v * n_seqs], ne[0]=CS, ne[1]=S_v, ne[2]=n_chunks, ne[3]=H_v*n_seqs
        // v_t[cs][col][ch][vh] = v_t[vh*ch_s*S_v + ch*ch_s + col*CS + cs]
        const int64_t vt_off = v_head * head_v_s + ch * ch_s + lane * CS; // col * CS

        float v_t_new[CS];
        for (int cs = 0; cs < CS; cs++) {
            v_t_new[cs] = v_t[vt_off + cs] - v_prime[cs];
        }

        // ── Matmul 2: v_attn[col][cs] = sum_j v_t_new[j] * kq[j][cs] ──
        // kq: [CS, CS, n_chunks, H_k * n_seqs], ne[0]=CS, ne[1]=CS, ne[2]=n_chunks
        // kq[j][cs][ch][kh] = kq[kh*hk_s + ch*CS*CS + j*CS + cs]
        const int64_t kq_off = k_head * (CS * CS * n_chunks) + ch * CS * CS;

        float v_attn[CS] = {0.0f};
        for (int j = 0; j < CS; j++) {
            float vtj = v_t_new[j];
            for (int cs = 0; cs < CS; cs++) {
                v_attn[cs] += vtj * kq[kq_off + j * CS + cs];
            }
        }

        // ── Matmul 3: attn_inter[col][cs] = sum_i state[i][col] * q_g_exp[i][cs] ──
        // q_g_exp: [S_v, CS, n_chunks, H_k * n_seqs], ne[0]=S_v, ne[1]=CS
        // q_g_exp[i][cs][ch][kh] = q_g_exp[kh*ch_s + ch*ch_s + cs*S_v + i]

        const int64_t qg_off = k_head * head_k_s + ch * ch_s;
        float attn_inter[CS] = {0.0f};
        for (int i = 0; i < S_v; i++) {
            float si = s_col[i];
            for (int cs = 0; cs < CS; cs++) {
                attn_inter[cs] += si * q_g_exp[qg_off + cs * cs_s + i];
            }
        }

        // ── Write outputs for this chunk: output[col][cs] = attn_inter + v_attn ──
        const int64_t tok_start = ch * CS;
        const int64_t n_out = (ch == n_chunks - 1) ? (n_tokens % CS ? n_tokens % CS : CS) : CS;
        const int64_t out_base = seq * n_tokens * S_v * H_v + v_head * S_v;
        for (int cs = 0; cs < n_out; cs++) {
            int64_t tok_idx = tok_start + cs;
            if (tok_idx < n_tokens) {
                dst[out_base + lane + tok_idx * S_v * H_v] = attn_inter[cs] + v_attn[cs];
            }
        }

        // ── Matmul 4: state update
        // kgv[i][col] = sum_j kg_t[j][i] * v_t_new[j] for all i in S_v
        // Then: s_col[i] = s_col[i] * g_last + kgv[i]
        // kg_t_g_last: [CS + g_extra, S_v, n_chunks, H_v * n_seqs]
        //   first CS entries along dim 0: kg_t[j][i]
        //   last entry if GDA (g_extra=1): g_last (scalar)
        //   last S_v entries if KDA (g_extra=S_v): g_last[i] (vector)
        const int64_t kgt_off = v_head * head_v_s + ch * ch_s + lane * CS; // col * CS
        // Wait, kg_t layout is different: ne[0]=CS+g_extra, ne[1]=S_v
        // kg_t[j][i][ch][vh] at offset: vh * (CS+g_extra)*S_v*n_chunks + ch * (CS+g_extra)*S_v + i * (CS+g_extra) + j

        const int64_t s_kgt_ch = (CS + (KDA ? S_v : 1)) * S_v;
        const int64_t s_kgt_hv = s_kgt_ch * n_chunks;
        const int64_t kgt_hv_off = v_head * s_kgt_hv + ch * s_kgt_ch + lane * (CS + (KDA ? S_v : 1));

        float s_new[S_v];
        for (int i = 0; i < S_v; i++) {
            float acc = 0.0f;
            for (int j = 0; j < CS; j++) {
                acc += kg_t_g_last[kgt_hv_off + i * (CS + (KDA ? S_v : 1)) + j] * v_t_new[j];
            }

            float g_val;
            if constexpr (KDA) {
                g_val = kg_t_g_last[kgt_hv_off + i * (CS + S_v) + CS];
            } else {
                g_val = kg_t_g_last[kgt_hv_off + i * (CS + 1) + CS];
            }
            s_new[i] = s_col[i] * g_val + acc;
        }

        for (int i = 0; i < S_v; i++) {
            s_col[i] = s_new[i];
        }
    }

    // ── Write final state ──
    // state appended after attn_scores
    const int64_t state_base = n_tokens * S_v * H_v * n_seqs;
    for (int r = 0; r < S_v; r++) {
        dst[state_base + seq * H_v * S_v * S_v + v_head * S_v * S_v + lane * S_v + r] = s_col[r];
    }
}

template <int S_v>
static void launch_gated_delta_net_pipeline_impl(
        const float * k_cd, const float * v_t, const float * kq,
        const float * q_g_exp, const float * kg_t_g_last,
        const float * init_state, float * dst,
        int64_t CS, int64_t H_v, int64_t H_k,
        int64_t n_tokens, int64_t n_chunks, int64_t n_seqs,
        bool kda, cudaStream_t stream) {
    dim3 grid_dims(H_v, n_seqs, 1);
    dim3 block_dims(S_v, 1, 1);

    switch (CS) {
        case 16: {
            if (kda) {
                gated_delta_net_pipeline_cuda<S_v, 16, true><<<grid_dims, block_dims, 0, stream>>>(
                    k_cd, v_t, kq, q_g_exp, kg_t_g_last, init_state, dst,
                    H_v, H_k, n_tokens, n_chunks, n_seqs);
            } else {
                gated_delta_net_pipeline_cuda<S_v, 16, false><<<grid_dims, block_dims, 0, stream>>>(
                    k_cd, v_t, kq, q_g_exp, kg_t_g_last, init_state, dst,
                    H_v, H_k, n_tokens, n_chunks, n_seqs);
            }
            break;
        }
        case 64: {
            if (kda) {
                gated_delta_net_pipeline_cuda<S_v, 64, true><<<grid_dims, block_dims, 0, stream>>>(
                    k_cd, v_t, kq, q_g_exp, kg_t_g_last, init_state, dst,
                    H_v, H_k, n_tokens, n_chunks, n_seqs);
            } else {
                gated_delta_net_pipeline_cuda<S_v, 64, false><<<grid_dims, block_dims, 0, stream>>>(
                    k_cd, v_t, kq, q_g_exp, kg_t_g_last, init_state, dst,
                    H_v, H_k, n_tokens, n_chunks, n_seqs);
            }
            break;
        }
        default: GGML_ABORT("unsupported CS for gated_delta_net_pipe");
    }
}

static void launch_gated_delta_net_pipeline(
        const float * k_cd, const float * v_t, const float * kq,
        const float * q_g_exp, const float * kg_t_g_last,
        const float * init_state, float * dst,
        int64_t S_v, int64_t CS, int64_t H_v, int64_t H_k,
        int64_t n_tokens, int64_t n_chunks, int64_t n_seqs,
        bool kda, cudaStream_t stream) {
    switch (S_v) {
        case 16:  launch_gated_delta_net_pipeline_impl<16>(k_cd, v_t, kq, q_g_exp, kg_t_g_last, init_state, dst, CS, H_v, H_k, n_tokens, n_chunks, n_seqs, kda, stream); break;
        case 32:  launch_gated_delta_net_pipeline_impl<32>(k_cd, v_t, kq, q_g_exp, kg_t_g_last, init_state, dst, CS, H_v, H_k, n_tokens, n_chunks, n_seqs, kda, stream); break;
        case 64:  launch_gated_delta_net_pipeline_impl<64>(k_cd, v_t, kq, q_g_exp, kg_t_g_last, init_state, dst, CS, H_v, H_k, n_tokens, n_chunks, n_seqs, kda, stream); break;
        case 128: launch_gated_delta_net_pipeline_impl<128>(k_cd, v_t, kq, q_g_exp, kg_t_g_last, init_state, dst, CS, H_v, H_k, n_tokens, n_chunks, n_seqs, kda, stream); break;
        default: GGML_ABORT("unsupported S_v for gated_delta_net_pipe");
    }
}

void ggml_cuda_op_gated_delta_net_pipe(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_state  = dst->src[0];
    ggml_tensor * src_k_cd   = dst->src[1];
    ggml_tensor * src_v_t    = dst->src[2];
    ggml_tensor * src_kq     = dst->src[3];
    ggml_tensor * src_q_g_exp= dst->src[4];
    ggml_tensor * src_kg_t   = dst->src[5];

    const int64_t S_v      = src_state->ne[0];
    const int64_t H_v      = src_state->ne[2];
    const int64_t n_seqs   = src_state->ne[3];
    const int64_t n_chunks = src_k_cd->ne[2];

    const int32_t * params  = (const int32_t *) dst->op_params;
    const int64_t  n_tokens = params[0];
    const int64_t  CS       = params[1];
    const bool     kda      = params[2] != 0;

    int64_t H_k = 1;
    if (kda) {
        H_k = H_v;
    } else {
        H_k = src_kq->ne[3] / n_seqs;  // H_k from the kq tensor's head dimension
    }

    cudaStream_t stream = ctx.stream();

    launch_gated_delta_net_pipeline(
        (const float *) src_k_cd->data,
        (const float *) src_v_t->data,
        (const float *) src_kq->data,
        (const float *) src_q_g_exp->data,
        (const float *) src_kg_t->data,
        (const float *) src_state->data,
        (float *) dst->data,
        S_v, CS, H_v, H_k, n_tokens, n_chunks, n_seqs, kda, stream);
}
