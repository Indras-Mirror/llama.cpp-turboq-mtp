#include "models.h"

#include "llama-impl.h"

// utility to get one slice from the third dimension
// input dim:  [x, y, c, b]
// output dim: [x, y, 1, b]
static ggml_tensor * get_slice_2d(ggml_context * ctx0, ggml_tensor * t, int64_t c) {
    return ggml_view_4d(ctx0, t, t->ne[0], t->ne[1], 1, t->ne[3],
        t->nb[1], t->nb[2], t->nb[3], t->nb[2] * c);
}

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params & params) : llm_graph_context(params) {}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net_chunking(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s,
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];
    const bool kda = (g->ne[0] == S_k && g->ne[1] == H_k);

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    const float scale = 1.0f / sqrtf(S_k);

    q = ggml_scale(ctx0, q, scale);

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(b, "b_in", il);
    cb(g, "g_in", il);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = ggml_permute(ctx0, k, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    v = ggml_permute(ctx0, v, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]
    g = ggml_permute(ctx0, g, 0, 2, 1, 3); // [g_0, n_tokens, H_v, n_seqs]
    b = ggml_permute(ctx0, b, 0, 2, 1, 3); // [  1, n_tokens, H_v, n_seqs]

    const int CS = kda ? 16 : 64; // chunk size

    const int pad = (CS - n_tokens % CS) % CS;
    const int n_chunks = (n_tokens + pad) / CS;

    q = ggml_pad(ctx0, q, 0, pad, 0, 0);
    k = ggml_pad(ctx0, k, 0, pad, 0, 0);
    v = ggml_pad(ctx0, v, 0, pad, 0, 0);
    g = ggml_pad(ctx0, g, 0, pad, 0, 0);
    b = ggml_pad(ctx0, b, 0, pad, 0, 0);

    ggml_tensor * v_b = ggml_mul(ctx0, v, b);
    ggml_tensor * k_b = ggml_mul(ctx0, k, b);

    cb(v_b, "v_b", il);
    cb(k_b, "k_b", il);

    q   = ggml_reshape_4d(ctx0, q,   S_k, CS, n_chunks, H_k * n_seqs);
    k   = ggml_reshape_4d(ctx0, k,   S_k, CS, n_chunks, H_k * n_seqs);
    k_b = ggml_reshape_4d(ctx0, k_b, S_k, CS, n_chunks, H_v * n_seqs);
    v   = ggml_reshape_4d(ctx0, v,   S_v, CS, n_chunks, H_v * n_seqs);
    v_b = ggml_reshape_4d(ctx0, v_b, S_v, CS, n_chunks, H_v * n_seqs);

    g = ggml_reshape_4d(ctx0, g, g->ne[0], CS, n_chunks, H_v * n_seqs);
    b = ggml_reshape_4d(ctx0, b, 1,        CS, n_chunks, H_v * n_seqs);

    // [CS, g_0, n_chunks, H_v * n_seqs]
    // TODO: extend ggml_cumsum with axis parameter to avoid transpose
    ggml_tensor * g_cs = ggml_cumsum(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, g)));
    cb(g_cs, "g_cs", il);

    ggml_tensor * kb = nullptr;
    ggml_tensor * kq = nullptr;
    if (kda) {
        const int64_t CHB = n_chunks * H_k * n_seqs;

        ggml_tensor * g_cs_i = ggml_reshape_4d(ctx0, g_cs, CS, 1, S_k, CHB);  // [chunk_size, 1, S_k, CHB]
        ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, S_k, CHB);  // [1, chunk_size, S_k, CHB]

        g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, S_k, CHB);  // [1, chunk_size, S_k, CHB] -> [chunk_size, chunk_size, S_k, CHB]

        // decay_mask [chunk_size,chunk_size,S_k,CHB]
        ggml_tensor * decay_mask;
        decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = ggml_exp(ctx0, decay_mask);
        cb(decay_mask, "decay_mask", il);

        // decay_mask [S_k,BT_j,BT_i,CHB] *Note* second and third chunk_sizes are switched
        decay_mask = ggml_cont_4d(ctx0, ggml_permute(ctx0, decay_mask, 2, 1, 0, 3), S_k, CS, CS, CHB);

        ggml_tensor * k_b_i = ggml_reshape_4d(ctx0, k_b, S_k, CS,  1, CHB);
        ggml_tensor * k_j   = ggml_reshape_4d(ctx0, k,   S_k,  1, CS, CHB);
        ggml_tensor * q_i   = ggml_reshape_4d(ctx0, q,   S_k, CS,  1, CHB);

        ggml_tensor * decay_k_b_i = ggml_mul(ctx0, decay_mask, k_b_i);
        ggml_tensor * decay_q_i   = ggml_mul(ctx0, decay_mask, q_i);

        // decay_k_b_i [S,BT,BT,CHB] @ k_j [S,1,BT,CHB] = Akk [BT,1,BT,CHB]
        kb = ggml_mul_mat(ctx0, decay_k_b_i, k_j);
        kq = ggml_mul_mat(ctx0, decay_q_i,   k_j);

        kb = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kb, CS, CS, n_chunks, H_v * n_seqs)));
        kq = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kq, CS, CS, n_chunks, H_v * n_seqs)));
    } else {
        ggml_tensor * g_cs_i = g_cs;
        ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, n_chunks, H_v * n_seqs);

        g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, n_chunks, H_v * n_seqs);

        // [CS, CS, n_chunks, H_v * n_seqs]
        ggml_tensor * decay_mask;
        decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = ggml_exp(ctx0, decay_mask);
        cb(decay_mask, "decay_mask", il);

        // [CS, CS, n_chunks, H_k * n_seqs]
        kb = ggml_mul_mat(ctx0, k,  k_b);
        kb = ggml_mul    (ctx0, kb, decay_mask);

        // [CS, CS, n_chunks, H_k * n_seqs]
        kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_mul(ctx0, kq, decay_mask);
    }

    kq = ggml_tri(ctx0, kq, GGML_TRI_TYPE_LOWER_DIAG);
    cb(kq, "kq", il);

    // [CS, CS, n_chunks, H_k * n_seqs]
    ggml_tensor * attn;
    attn = ggml_tri(ctx0, kb, GGML_TRI_TYPE_LOWER);
    cb(attn, "attn", il);

    ggml_tensor * identity;
    identity = ggml_view_1d(ctx0, attn, CS, 0);
    identity = ggml_fill   (ctx0, identity, 1.0f);
    identity = ggml_diag   (ctx0, identity);

    ggml_tensor * lhs = ggml_add(ctx0, attn, identity);
    cb(lhs, "dnet_add_ch_lhs", il);

    attn = ggml_neg(ctx0, attn);
    cb(attn, "attn_pre_solve", il);

    ggml_tensor * lin_solve = ggml_solve_tri(ctx0, lhs, attn, true, true, false);
    attn = ggml_add(ctx0, lin_solve, identity);
    cb(attn, "dnet_add_ch_attn_solved", il); // [CS, CS, n_chunks, H_k * n_seqs]

    // [S_v, CS, n_chunks, H_v * n_seqs]
    v = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v_b)), attn);

    // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * g_exp = ggml_exp(ctx0, g_cs);

    k_b = ggml_cont(ctx0, ggml_transpose(ctx0, k_b));

    // [CS, S_k, n_chunks, H_k * n_seqs]
    ggml_tensor * kbg = ggml_mul(ctx0, k_b, g_exp);
    cb(kbg, "k_beta_g_exp", il);

    // [S_k, CS, n_chunks, H_k * n_seqs]
    ggml_tensor * k_cd = ggml_mul_mat(ctx0, kbg, attn);
    cb(k_cd, "k_cumdecay", il);

    // [1, CS, n_chunks, H_k * n_seqs] KDA: [S_k, CS, n_chunks, H_k * n_seqs]
    ggml_tensor * g_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, g_exp));
    ggml_tensor * q_g_exp = ggml_mul(ctx0, q, g_exp_t);

    // vectorized calculation of key_gdiff
    // improved from the chunked version:
    //   g_last = torch.clamp(g_cum[:, :, -1], max=50.0).exp().unsqueeze(-1).unsqueeze(-1)
    //   g_diff = torch.clamp(g_cum[:, :, -1:] - g_cum, max=50.0).exp()
    //   key_gdiff = key * g_diff.unsqueeze(-1)
    //   kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
    //   last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew

    // get last element in g_cumsum along CS dimension (ne0)
    // example: [[x, y, z, ..., last], ...] -> [[last], ...]
    // [1, 1, n_chunks, H_v * n_seqs] KDA: [1, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * g_last = ggml_view_4d(ctx0, g_cs, 1, g_cs->ne[1], g_cs->ne[2], g_cs->ne[3],
            g_cs->nb[1],
            g_cs->nb[2],
            g_cs->nb[3],
            ggml_row_size(g_cs->type, g_cs->ne[0] - 1));
    cb(g_last, "g_last", il);

    // TODO: remove this cont when CUDA supports non-cont unary ops
    g_last = ggml_cont(ctx0, g_last);

    // [1, 1, n_chunks, H_v * n_seqs] KDA: [S_k, 1, n_chunks, H_v * n_seqs]
    ggml_tensor * g_last_exp_t = ggml_transpose(ctx0, ggml_exp(ctx0, g_last));
    cb(g_last_exp_t, "g_last_exp_t", il);

    // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * g_diff = ggml_neg(ctx0, ggml_sub(ctx0, g_cs, g_last));
    cb(g_diff, "g_diff", il);

    ggml_tensor * g_diff_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_exp(ctx0, g_diff)));

    // [S_k, CS, n_chunks, H_v * n_seqs]
    ggml_tensor * kg = ggml_mul(ctx0, k, g_diff_exp_t);
    cb(kg, "key_gdiff", il);

    // [CS, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * kg_t = ggml_cont(ctx0, ggml_transpose(ctx0, kg));
    cb(kg_t, "key_gdiff_t", il);

    // [CS, S_v, n_chunks, H_v * n_seqs]
    ggml_tensor * v_t = ggml_cont(ctx0, ggml_transpose(ctx0, v));
    cb(v_t, "v_t", il);

    // Pack kg_t and g_last_exp_t into one tensor for the pipeline op.
    // kg_t:         [CS,          S_k, n_chunks, H_v * n_seqs]
    // g_last_exp_t: [1 or S_k, 1,      n_chunks, H_v * n_seqs]
    // Combined:     [CS + g_extra, S_k, n_chunks, H_v * n_seqs]
    //   first CS entries along ne[0] = kg_t[j][i]
    //   last entry (GDA) or last S_k entries (KDA) = g_last_exp_t[i]
    {
        const int g_extra = kda ? (int)S_v : 1;
        const int64_t ne_kgt[] = {CS + g_extra, S_k, n_chunks, H_v * n_seqs};
        struct ggml_tensor * kg_t_g_last = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,
            ne_kgt[0], ne_kgt[1], ne_kgt[2], ne_kgt[3]);

        // Copy kg_t into the first CS entries
        struct ggml_tensor * v_kg = ggml_view_4d(ctx0, kg_t_g_last,
            CS, S_k, n_chunks, H_v * n_seqs,
            ggml_row_size(GGML_TYPE_F32, ne_kgt[0]),
            ggml_row_size(GGML_TYPE_F32, ne_kgt[0]) * S_k,
            ggml_row_size(GGML_TYPE_F32, ne_kgt[0]) * S_k * n_chunks,
            0);
        ggml_cpy(ctx0, kg_t, v_kg);

        // Copy g_last into the extra entries
        struct ggml_tensor * v_g = ggml_view_4d(ctx0, kg_t_g_last,
            g_extra, S_k, n_chunks, H_v * n_seqs,
            ggml_row_size(GGML_TYPE_F32, ne_kgt[0]),
            ggml_row_size(GGML_TYPE_F32, ne_kgt[0]) * S_k,
            ggml_row_size(GGML_TYPE_F32, ne_kgt[0]) * S_k * n_chunks,
            ggml_row_size(GGML_TYPE_F32, CS));

        struct ggml_tensor * g_bc = ggml_repeat_4d(ctx0, g_last_exp_t, g_extra, S_k, n_chunks, H_v * n_seqs);
        ggml_cpy(ctx0, g_bc, v_g);

        cb(kg_t_g_last, "kg_t_g_last", il);

        // Call the fused pipeline op (replaces the entire sequential chunk loop)
        ggml_tensor * result = ggml_gated_delta_net_pipe(ctx0,
            s, k_cd, v_t, kq, q_g_exp, kg_t_g_last,
            (int) n_tokens, CS, kda);
        cb(result, "dnet_pipe_result", il);

        // Split result into attention output and final state
        // result: [S_v * H_v, n_tokens * n_seqs + S_v * n_seqs, 1, 1]
        //   front: attn_scores [S_v, H_v, n_tokens, n_seqs] row-major
        //   back:  new_state   [S_v, S_v, H_v, n_seqs] row-major
        const size_t ne_attn = (size_t) S_v * H_v * n_tokens * n_seqs;

        struct ggml_tensor * o = ggml_view_4d(ctx0, result,
            S_v, H_v, n_tokens, n_seqs,
            sizeof(float) * S_v,
            sizeof(float) * S_v * H_v,
            sizeof(float) * S_v * H_v * n_tokens,
            0);
        cb(o, "output", il);

        s = ggml_view_4d(ctx0, result,
            S_v, S_v, H_v, n_seqs,
            sizeof(float) * S_v,
            sizeof(float) * S_v * S_v,
            sizeof(float) * S_v * S_v * H_v,
            ne_attn * sizeof(float));
        cb(s, "output_state", il);

        return {o, s};
    }
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net_autoregressive(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b, // beta
        ggml_tensor * s, // state
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(n_tokens == 1);

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    const float scale = 1.0f / sqrtf(S_k);

    q = ggml_scale(ctx0, q, scale);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = ggml_permute(ctx0, k, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    v = ggml_permute(ctx0, v, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(b, "b_in", il);
    cb(g, "g_in", il);

    // GDA: [1,  1,  H_v, n_seqs]
    // KDA: [1, S_k, H_v, n_seqs]
    g = ggml_reshape_4d(ctx0, g, 1, g->ne[0], H_v, n_seqs);
    b = ggml_reshape_4d(ctx0, b, 1,        1, H_v, n_seqs);

    // [S_v, S_v, H_v, n_seqs]
    g = ggml_exp(ctx0, g);
    s = ggml_mul(ctx0, s, g);

    // [1, S_v, H_v, n_seqs]
    ggml_tensor * sk;
    sk = ggml_mul     (ctx0, s, k);
    sk = ggml_sum_rows(ctx0, sk);

    // [S_v, 1, H_v, n_seqs]
    ggml_tensor * d;
    d = ggml_sub(ctx0, v, ggml_transpose(ctx0, sk));
    d = ggml_mul(ctx0, d, b);

    // [1, S_v, H_v, n_seqs]
    ggml_tensor * d_t;
    d_t = ggml_transpose(ctx0, d);

    // [S_v, S_v, H_v, n_seqs]
    ggml_tensor * kd;
    k  = ggml_repeat(ctx0, k, s);
    kd = ggml_mul   (ctx0, k, d_t);

    s = ggml_add(ctx0, s, kd);

    cb(s, "dnet_add_ar_state", il);

    ggml_tensor * s_q = ggml_mul     (ctx0, s, q);
    ggml_tensor * o   = ggml_sum_rows(ctx0, s_q);

    o = ggml_permute  (ctx0, o, 2, 0, 1, 3); // [S_v, H_v, n_tokens, n_seqs]

    return {o, s};
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net_fused(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s,
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, /*keep_intermediates=*/false);
    if (n_tokens == 1) {
        cb(result, LLAMA_TENSOR_NAME_FGDN_AR, il);
    } else {
        cb(result, LLAMA_TENSOR_NAME_FGDN_CH, il);
    }

    ggml_tensor * output = ggml_view_4d(ctx0, result,
            S_v, H_v, n_tokens, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens), 0);

    ggml_tensor * new_state = ggml_view_4d(ctx0, result,
            S_v, S_v, H_v, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * S_v),
            ggml_row_size(result->type, S_v * S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens * n_seqs));

    return {output, new_state};
}

ggml_tensor * llm_build_delta_net_base::build_delta_net_fused_keep_intermediates(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s,
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, /*keep_intermediates=*/true);
    // Always n_tokens > 1 here (gated by n_seq_tokens > 1 at the call site),
    // so this is the chunked variant by the same naming convention used in
    // build_delta_net_fused.
    cb(result, LLAMA_TENSOR_NAME_FGDN_CH, il);
    return result;
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s,
        int           il) {
    const int64_t n_seq_tokens = q->ne[2];

    if (n_seq_tokens == 1) {
        if (cparams.fused_gdn_ar) {
            return build_delta_net_fused(q, k, v, g, b, s, il);
        }
        return build_delta_net_autoregressive(q, k, v, g, b, s, il);
    }

    if (cparams.fused_gdn_ch) {
        // Use build_delta_net_fused for the built-in fused kernel.
        // To test the pipeline op: set --no-fused-gdn-ch
        return build_delta_net_fused(q, k, v, g, b, s, il);
    }

    return build_delta_net_chunking(q, k, v, g, b, s, il);
}
