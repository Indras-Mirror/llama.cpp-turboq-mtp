#include "server-decide.h"

#include "common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>

using json = nlohmann::ordered_json;

static double decide_now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// log-softmax of `tok` against the full logit row
static double decide_logprob(const float * row, int n_vocab, llama_token tok) {
    float mx = row[0];
    for (int i = 1; i < n_vocab; i++) {
        mx = std::max(mx, row[i]);
    }
    double sum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        sum += std::exp((double) (row[i] - mx));
    }
    return (double) (row[tok] - mx) - std::log(sum);
}

namespace {

struct decide_run {
    llama_seq_id                     seq;
    const std::vector<llama_token> * toks;
    llama_pos                        pos0;
    const std::vector<char>        * want; // may be null (no outputs)
};

struct decide_branch {
    int c, f, o;
    std::vector<llama_token> toks;
    std::vector<char>        want;
    double lp       = 0.0;
    int    n_scored = 0;
};

// decode runs chunked to n_batch; on_chunk(map) is called after each llama_decode,
// with map[i] = (run index, token index) for batch row i
// atomic = n_ubatch for recurrent/hybrid models: every llama_decode call then holds only whole
// runs totalling <= n_ubatch tokens (a longer run is decoded alone), so llama.cpp never splits a
// sequence across ubatches next to other sequences -- that corrupts recurrent state for sequences
// copied from a shared parent. 0 = pack freely up to n_batch.
template <typename F>
void decode_runs(llama_context * ctx, int n_outputs_max, int atomic, const std::vector<decide_run> & runs, F && on_chunk) {
    const int n_batch = (int) llama_n_batch(ctx);
    int n_out = 0; // output rows in the current batch
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<std::pair<int, int>> map;
    auto flush = [&]() {
        if (batch.n_tokens == 0) {
            return;
        }
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("llama_decode failed (rc = " + std::to_string(rc) + "), KV cache may be full");
        }
        on_chunk(map);
        batch.n_tokens = 0;
        n_out = 0;
        map.clear();
    };
    for (int r = 0; r < (int) runs.size(); r++) {
        const auto & run = runs[r];
        const int len = (int) run.toks->size();
        if (atomic > 0 && batch.n_tokens > 0 && (len > atomic || batch.n_tokens + len > atomic)) {
            flush(); // start this run in a fresh decode call
        }
        for (int j = 0; j < len; j++) {
            if (atomic > 0 && len > atomic && batch.n_tokens == atomic) {
                flush(); // an over-long run is decoded alone, one ubatch at a time
            }
            const bool want = run.want && (*run.want)[j];
            if (batch.n_tokens == n_batch || (want && n_out == n_outputs_max)) {
                flush();
            }
            n_out += want;
            const int i = batch.n_tokens++;
            batch.token[i]     = (*run.toks)[j];
            batch.pos[i]       = run.pos0 + j;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = run.seq;
            batch.logits[i]    = run.want ? (*run.want)[j] : 0;
            map.push_back({r, j});
        }
    }
    flush();
    llama_batch_free(batch);
}

} // namespace

void server_decide::init(llama_context * ctx, const llama_model * model, int seq0, int n_seqs, int n_threads, int n_outputs_max) {
    this->ctx       = ctx;
    this->vocab     = llama_model_get_vocab(model);
    this->seq0      = seq0;
    this->n_seqs    = n_seqs;
    this->n_threads = std::max(1, n_threads);
    this->n_outputs_max = std::max(1, n_outputs_max);
    // llama.cpp mis-tracks recurrent state when branches copied from one sequence have
    // different lengths in the same batch, so hybrid/recurrent models only batch equal lengths
    this->recurrent_like = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
}

void server_decide::clear_work_seqs() {
    llama_memory_t mem = llama_get_memory(ctx);
    for (int s = 1; s < n_seqs; s++) {
        llama_memory_seq_rm(mem, seq0 + s, -1, -1);
    }
}

json server_decide::run(const json & req) {
    if (!enabled()) {
        throw std::invalid_argument("/decide is disabled; start the server with --decide-seqs N (N >= 3)");
    }
    // more inputs than sequences allow: split into chunks (half the sequences for inputs, half for branches)
    {
        const int max_c = std::max(1, (n_seqs - 1) / 2);
        const json & all = req.at("contexts");
        if ((int) all.size() > max_c) {
            json merged;
            for (size_t i = 0; i < all.size(); i += max_c) {
                json sub = req;
                sub["contexts"] = json(std::vector<json>(all.begin() + i, all.begin() + std::min(all.size(), i + max_c)));
                json part = run(sub);
                if (merged.is_null()) {
                    merged = std::move(part);
                    continue;
                }
                for (auto & r : part["results"]) {
                    merged["results"].push_back(std::move(r));
                }
                for (const char * sect : {"timing", "stats"}) {
                    for (auto it = part[sect].begin(); it != part[sect].end(); ++it) {
                        if (it.value().is_number() && merged[sect].contains(it.key())) {
                            merged[sect][it.key()] = merged[sect][it.key()].get<double>() + it.value().get<double>();
                        }
                    }
                }
            }
            merged["stats"]["chunks"] = (all.size() + max_c - 1) / max_c;
            return merged;
        }
    }

    const double t_start = decide_now_ms();
    llama_memory_t mem = llama_get_memory(ctx);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_ubatch = (int) llama_n_ubatch(ctx);

    try {
        // --- static prefix, cached in seq0 across calls
        const auto prefix = common_tokenize(vocab, req.at("prefix").get<std::string>(), true, true);
        if (prefix.empty()) {
            throw std::invalid_argument("empty prefix");
        }
        // the server may have wiped the cache (e.g. a full clear), so verify residency
        bool hit = prefix == cached_prefix &&
                   llama_memory_seq_pos_max(mem, seq0) == (llama_pos) prefix.size() - 1;
        double t_prefix = 0.0;
        if (!hit) {
            const double t0 = decide_now_ms();
            llama_memory_seq_rm(mem, seq0, -1, -1);
            cached_prefix.clear();
            std::vector<decide_run> runs = {{(llama_seq_id) seq0, &prefix, 0, nullptr}};
            decode_runs(ctx, n_outputs_max, 0, runs, [](auto &) {});
            cached_prefix = prefix;
            t_prefix = decide_now_ms() - t0;
        }
        clear_work_seqs();
        const llama_pos P = (llama_pos) prefix.size();

        // --- contexts: one sequence each, decoded together
        const auto ctx_texts = req.at("contexts").get<std::vector<std::string>>();
        const int n_c = (int) ctx_texts.size();
        if (n_c == 0) {
            throw std::invalid_argument("no contexts");
        }
        if (1 + n_c >= n_seqs) {
            throw std::invalid_argument("too many contexts (" + std::to_string(n_c) + ") for --decide-seqs " +
                                        std::to_string(n_seqs) + "; send fewer per request");
        }
        std::vector<std::vector<llama_token>> ctx_toks(n_c);
        int ctx_total = 0;
        for (int c = 0; c < n_c; c++) {
            ctx_toks[c] = common_tokenize(vocab, ctx_texts[c], false, true);
            ctx_total += (int) ctx_toks[c].size();
        }
        double t0 = decide_now_ms();
        {
            std::vector<decide_run> runs;
            for (int c = 0; c < n_c; c++) {
                llama_memory_seq_cp(mem, seq0, seq0 + 1 + c, -1, -1);
                runs.push_back({(llama_seq_id) (seq0 + 1 + c), &ctx_toks[c], P, nullptr});
            }
            if (recurrent_like) {
                for (const auto & run : runs) { // one input per decode call
                    decode_runs(ctx, n_outputs_max, n_ubatch, {run}, [](auto &) {});
                }
            } else {
                decode_runs(ctx, n_outputs_max, 0, runs, [](auto &) {});
            }
        }
        const double t_ctx = decide_now_ms() - t0;

        // --- branches (context, field, option); score tokens after the part shared by all options
        const json & fields = req.at("fields");
        const int n_f = (int) fields.size();
        std::vector<std::vector<std::vector<llama_token>>> line_toks(n_f);
        std::vector<int> common(n_f);
        for (int f = 0; f < n_f; f++) {
            for (const auto & line : fields[f].at("lines")) {
                line_toks[f].push_back(common_tokenize(vocab, line.get<std::string>(), false, true));
            }
            if (line_toks[f].size() < 2) {
                throw std::invalid_argument("field " + std::to_string(f) + " needs at least 2 options");
            }
            size_t k = 0;
            for (;; k++) {
                bool same = true;
                for (const auto & t : line_toks[f]) {
                    if (k >= t.size() || t[k] != line_toks[f][0][k]) {
                        same = false;
                        break;
                    }
                }
                if (!same) {
                    break;
                }
            }
            if (k == 0) {
                throw std::invalid_argument("field " + std::to_string(f) + ": option lines must share a leading token (the field name)");
            }
            common[f] = (int) k;
        }
        std::vector<decide_branch> br;
        for (int c = 0; c < n_c; c++) {
            for (int f = 0; f < n_f; f++) {
                for (int o = 0; o < (int) line_toks[f].size(); o++) {
                    decide_branch b{c, f, o, line_toks[f][o], {}};
                    b.want.assign(b.toks.size(), 0);
                    // logits at token j predict token j+1
                    for (int j = common[f] - 1; j + 1 < (int) b.toks.size(); j++) {
                        b.want[j] = 1;
                    }
                    br.push_back(std::move(b));
                }
            }
        }
        const bool group_len = req.value("group_len", recurrent_like);
        if (group_len) {
            std::stable_sort(br.begin(), br.end(), [&](const decide_branch & a, const decide_branch & b) {
                return std::make_pair(ctx_toks[a.c].size(), a.toks.size()) <
                       std::make_pair(ctx_toks[b.c].size(), b.toks.size());
            });
        }

        t0 = decide_now_ms();
        const int n_batch    = (int) llama_n_batch(ctx);
        const int first_b    = 1 + n_c;
        const int max_par    = n_seqs - first_b;
        size_t next = 0;
        int n_waves = 0, n_rows = 0, branch_tokens = 0;
        while (next < br.size()) {
            std::vector<int> wave;
            int toks = 0;
            while (next < br.size() && (int) wave.size() < max_par &&
                   (wave.empty() || toks + (int) br[next].toks.size() <= n_batch * 4) &&
                   (!group_len || wave.empty() ||
                    (br[next].toks.size() == br[wave[0]].toks.size() &&
                     ctx_toks[br[next].c].size() == ctx_toks[br[wave[0]].c].size()))) {
                toks += (int) br[next].toks.size();
                wave.push_back((int) next++);
            }
            std::vector<decide_run> runs;
            for (int w = 0; w < (int) wave.size(); w++) {
                const decide_branch & b = br[wave[w]];
                const llama_seq_id s = seq0 + first_b + w;
                llama_memory_seq_cp(mem, seq0 + 1 + b.c, s, -1, -1);
                runs.push_back({s, &b.toks, P + (llama_pos) ctx_toks[b.c].size(), &b.want});
            }
            decode_runs(ctx, n_outputs_max, recurrent_like ? n_ubatch : 0, runs, [&](const std::vector<std::pair<int, int>> & map) {
                std::vector<int> jobs; // batch rows that carry an output we need
                for (int i = 0; i < (int) map.size(); i++) {
                    if (br[wave[map[i].first]].want[map[i].second]) {
                        jobs.push_back(i);
                    }
                }
                std::vector<double> lps(jobs.size());
                const int nt = std::min<int>(n_threads, std::max<int>(1, (int) jobs.size()));
                std::vector<std::thread> th;
                for (int t = 0; t < nt; t++) {
                    th.emplace_back([&, t]() {
                        for (size_t k = t; k < jobs.size(); k += nt) {
                            const auto [r, j] = map[jobs[k]];
                            const decide_branch & b = br[wave[r]];
                            lps[k] = decide_logprob(llama_get_logits_ith(ctx, jobs[k]), n_vocab, b.toks[j + 1]);
                        }
                    });
                }
                for (auto & x : th) {
                    x.join();
                }
                for (size_t k = 0; k < jobs.size(); k++) {
                    const auto [r, j] = map[jobs[k]];
                    br[wave[r]].lp += lps[k];
                    br[wave[r]].n_scored++;
                }
                n_rows += (int) jobs.size();
            });
            for (int w = 0; w < (int) wave.size(); w++) {
                llama_memory_seq_rm(mem, seq0 + first_b + w, -1, -1);
                branch_tokens += (int) br[wave[w]].toks.size();
            }
            n_waves++;
        }
        const double t_branch = decide_now_ms() - t0;
        clear_work_seqs();

        json results = json::array();
        for (int c = 0; c < n_c; c++) {
            json row = json::array();
            for (int f = 0; f < n_f; f++) {
                const size_t n_o = line_toks[f].size();
                row.push_back({{"logprobs", std::vector<double>(n_o, 0.0)}, {"n_tokens", std::vector<int>(n_o, 0)}});
            }
            results.push_back(row);
        }
        for (const auto & b : br) {
            results[b.c][b.f]["logprobs"][b.o] = b.lp;
            results[b.c][b.f]["n_tokens"][b.o] = b.n_scored;
        }
        return {
            {"results", results},
            {"timing", {{"prefix_ms", t_prefix}, {"prefix_cached", hit}, {"context_ms", t_ctx},
                        {"branch_ms", t_branch}, {"total_ms", decide_now_ms() - t_start}}},
            {"stats", {{"prefix_tokens", prefix.size()}, {"context_tokens", ctx_total},
                       {"branches", br.size()}, {"branch_tokens", branch_tokens},
                       {"scored_rows", n_rows}, {"waves", n_waves}, {"group_len", group_len}}},
        };
    } catch (...) {
        clear_work_seqs();
        throw;
    }
}
