#pragma once

// /decide: "System-1" parallel multiple-choice scoring (a.k.a. Jev mode).
//
// Instead of generating JSON token by token, every field of a schema is asked as a
// multiple-choice question: each candidate line (e.g. `  "department": "billing",\n`)
// is appended to a copy of the context and its tokens are scored exactly. All
// (context x field x option) branches share the KV cells of the static prefix and of
// their context, and are decoded together in as few batches as possible.
//
// Runs on the target context between update_slots() iterations, using only the
// sequence ids [seq0, seq0 + n_seqs) reserved via --decide-seqs. Slots are never touched.
//
// Request:  {"prefix": "...", "contexts": ["..."], "fields": [{"lines": ["...", "..."]}]}
// Response: {"results": [[{"logprobs": [...], "n_tokens": [...]}, ...]], "timing": {...}, "stats": {...}}

#include "llama.h"

#include <nlohmann/json.hpp>

#include <vector>

struct server_decide {
    llama_context     * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;

    int  seq0           = 0; // first reserved sequence id
    int  n_seqs         = 0; // number of reserved sequence ids (0 = disabled)
    int  n_threads      = 4;
    int  n_outputs_max  = 256; // output rows allowed per llama_decode (context limit)
    bool recurrent_like = false;

    void init(llama_context * ctx, const llama_model * model, int seq0, int n_seqs, int n_threads, int n_outputs_max);

    bool enabled() const { return n_seqs >= 3; }

    // throws std::exception on invalid input or decode failure (all reserved seqs are cleaned up)
    nlohmann::ordered_json run(const nlohmann::ordered_json & req);

private:
    std::vector<llama_token> cached_prefix; // resident in seq0

    void clear_work_seqs();
};
