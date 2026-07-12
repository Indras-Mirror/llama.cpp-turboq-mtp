# Proposal: Closing the MTP Acceptance Gap (Decode-Throughput Levers)

**Target:** `Qwopus3.6-27B-v2-MTP-Q4_K_M.gguf` on 1× RTX 4090 (TurboQuant TBQ4 fused FA + MTP)
**Goal:** Raise effective decode throughput by raising MTP draft **acceptance** at
`temp 1.0` (current ~50%), *without* lowering quants and without changing the
GPU execution / KV path.

---

## 1. Executive Summary

The ~50% acceptance at `temp 1.0` is **not** a quantization, KV-quant, or runtime
bug. It is the **trained quality of the linear MTP head** at temperature 1.0.
We tested the obvious "runtime" fix (sampling the draft from the target's
temperature) and it **regressed** acceptance 0.534 → 0.437, so it was reverted.
Greedy drafting is already acceptance-optimal (see §2).

Levers, ranked by effort vs. impact:

| # | Lever | Where | Effort | Expected impact |
|---|-------|-------|--------|-----------------|
| **A** | Temperature / "fast mode" preset | Product / CLI | None (config) | temp 0 → **68% acc / 92.7 tok/s** (+49% decode); quality cost |
| **B** | Adaptive draft depth (`n_max` back-pressure) | Inference (`speculative.cpp`) | Low | Trims spikes & wasted draft compute; ~0–3 pts, smoother tail |
| **C** | Tree drafting via **EAGLE3 head** | Model export + flag | Med-High | Big acc gain (tree hits more candidates); fork already supports `draft-eagle3` |
| **D** | **FastMTP** retrain of the MTP head | Model owner / training | High | **+82%** over vanilla MTP (Tencent); the decisive lever |

Recommendation: ship **B** immediately (cheap, safe), offer **A** as a preset,
and route **C/D** to the model owner as the path to the 95% regime.

---

## 2. Root-Cause Analysis (corrected)

### 2.1 Acceptance is an inner product
For speculative decoding the expected acceptance-at-a-position is

```
  E[accept] = Σ_x  p_T(x) · q_D(x)
```

where `p_T` = target (full model) distribution and `q_D` = draft distribution.
This is maximized, for a *fixed* `p_T`, by making `q_D` a **delta on the
argmax of `p_T`** — i.e. a **greedy** draft. Any attempt to "match" the draft
distribution to the tempered `p_T` (top-p, etc.) *lowers* the inner product.

### 2.2 Measurement (this host, temp 1.0, 256-token gen)

| Draft sampler | Acceptance (acc/gen) | Verdict |
|---------------|----------------------|---------|
| **Greedy (original, top_k=1)** | **0.534** (156/292) | baseline / optimal |
| Temp-matched (target temp 1.0 sampling) | 0.437 (169/387) | **regressed → reverted** |

The original `common_speculative_state_mtp` ctor hardcodes the draft sampler to
greedy (`top_k=1`, `samplers={TOP_K}`) — this is **correct**, not a bug.

### 2.3 Why it is NOT a runtime / quant problem
- The MTP head is part of the same Q4_K_M GGUF and runs on the **same GPU**
  (`-ngl 99`) behind the same **active CUDA graphs**. Draft step ≈ 6 ms.
- TBQ4 KV is consumed identically by target and draft; it does not
  differentially hurt the draft.
- Acceptance is a model-quality property. The "95%" figures in the wild are
  EAGLE-family at `temp 0`/greedy, or differently-structured (tree) drafters.

---

## 3. Proposal A — Temperature / "fast mode" preset  *(no code)*

**Goal:** expose the already-available throughput lever as a first-class mode.

- Run the target **greedy** (`-temp 0`) when speed > diversity. Measured:
  `temp 0 → 68% acc / 92.7 tok/s` vs `temp 1.0 → 53% acc / 62 tok/s`.
- Ship a documented **"fast mode"** preset in `start-qwopus-turboq.sh`
  (e.g. `--temp 0 --min-p 0`) for batch / latency-critical workloads.
- **Trade-off:** deterministic, less diverse output. This is a product decision,
  not an optimization — included for completeness.

---

## 4. Proposal B — Adaptive draft depth (`n_max` back-pressure)  *[INFERENCE, low effort]*

**Goal:** stop spending GPU draft compute (and creating inter-token spikes)
when acceptance is low; restore depth when it recovers. Does **not** raise the
per-step acceptance ceiling, but removes waste and smooths the tail.

### 4.1 Architecture
```
            per-step accept depth (from common_sampler_sample_and_accept_n)
                                │
                                ▼
                    EMA(acceptance)  ──── > hi_thr (e.g. 0.6) ──► n_draft = n_max
                                │
                                └── < lo_thr (e.g. 0.45) ──► n_draft = n_min (1)
   n_draft ──► can_speculate / spec_draft  (common/speculative.cpp)
```
A lightweight controller holds an EMA of the last `N` steps' accepted/depth
ratio and selects the effective draft depth between `--spec-draft-n-min` and
`--spec-draft-n-max`. Hysteresis avoids flapping.

### 4.2 Code pointers
- Flags already exist: `--spec-draft-n-max` / `--spec-draft-n-min`
  (`common/arg.cpp:3497`, `:3504`).
- Hook the controller in `common/speculative.cpp` near `can_speculate` /
  `spec_draft` (the `common_speculative_state_mtp` instance). Read the accepted
  count returned by `common_sampler_sample_and_accept_n` (`common/sampling.cpp:621`).
- Add `--spec-adapt-thr-hi/lo` (or reuse defaults) — optional.

### 4.3 Verification
- `n_max=3, n_min=1`, threshold 0.5, over a mixed prompt set.
- Expect: fewer rejected deep drafts when acceptance dips; **p99 inter-token
  latency down**; mean tok/s roughly flat (GPU-bound) or slightly up.
- No change to output distribution (depth only, sampling unchanged) → lossless.

### 4.4 Risks / rollback
Purely a scheduling knob; default OFF (fixed `n_max`). Revert by flag.

---

## 5. Proposal C — Tree drafting via EAGLE3 head  *[MODEL-SIDE, high impact]*

**Goal:** raise acceptance by presenting **multiple candidates per position**
(tree), so the target only needs to match one of several — the standard EAGLE
gain. Our model currently uses the **linear** `draft-mtp` path (1 candidate/depth).

### 5.1 Key finding
This fork **already implements** EAGLE3 tree drafting:
- `COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3` (`common/speculative.cpp:25`)
- `common_speculative_impl_draft_eagle3` (`common/speculative.cpp:344`)
- tree-based sampling in `examples/speculative/speculative.cpp:493`

So the inference code path exists; what's missing is a **model with an EAGLE3
head** and selection via `--speculative-type draft-eagle3`.

### 5.2 Architecture
```
Target (27B, TBQ4)  ─┐
                     ├─► EAGLE3 head (small, on GPU) ─► tree of K candidates/depth
                     │        ▲
                     │        └── trained to predict target's top branches
                     ▼
        tree-verify sampler (lossless) → accepted length ≫ 1
```
A tree drafter with K branches/depth raises the per-position hit probability
roughly to `1 - (1 - p)^K`, so even a weak head gains. Losslessness holds via
the standard EAGLE tree-verify (rejection with residual resampling).

### 5.3 What's needed (model owner)
- Export / train an **EAGLE3 head** for Qwopus3.6-27B (small adapter; same
  recipe as EAGLE / Medusa).
- Bundle it in the GGUF and launch with
  `--speculative-type draft-eagle3 --speculative-...-model <eagle3.gguf>`.
- No changes to the GPU/FA/KV path; reuses `draft-eagle3` impl already present.

### 5.4 Risks
- Requires model retraining/export (out of inference-operator scope).
- Tree drafting increases peak VRAM for the draft graph; on a 24 GiB 4090 with
  a 27B model this must be validated (n_seq / branch budget).

---

## 6. Proposal D — FastMTP retraining of the MTP head  *[MODEL-SIDE, decisive]*

**Goal:** directly maximize acceptance by retraining the existing MTP head with
the **FastMTP** objective (Tencent, 2025), reported **+82% accepted tokens over
vanilla MTP**, plus LK losses (**+8–10%** acceptance length).

- FastMTP adds a lightweight aux head / modified objective during MTP training
  that optimizes the *accepted length* under rejection sampling, not just the
  next-token CE loss.
- Compatible with our existing `draft-mtp` serving path (no tree machinery).
- This is the lever that closes the 50% → 90%+ gap at `temp 1.0` **without**
  dropping temperature.

**Effort:** high (needs base-model access + training pipeline). **Owner:** model
trainer. **Output:** a new MTP head baked into the GGUF; drop-in for the
current server config.

---

## 7. Recommendation / Roadmap

1. **Now (inference, zero risk):** ship **B** (adaptive `n_max`) + document
   **A** ("fast mode" = temp 0) preset in `start-qwopus-turboq.sh`.
2. **Next (model owner):** request **D** (FastMTP head) — the decisive win.
   As a parallel/alternative, evaluate **C** (EAGLE3 head) since the fork
   already supports it and it needs no new inference code.
3. **Do not pursue:** "temp-match the draft" (tested, regresses), lowering
   quants, or runtime/KV-quant changes to the head (no effect on acceptance).

---

## 8. Non-Starters (explicitly rejected)

- **Sampling the draft from target temperature** — implemented and measured;
  acceptance 0.534 → 0.437; **reverted**. Greedy draft is provably optimal.
- **Lowering quants** — rejected by user; also irrelevant to acceptance.
- **Runtime / KV-quant "optimization" of the MTP head** — head already runs
  GPU-side with active CUDA graphs; acceptance is a trained-quality property.
- **Multi-core ggml graph build** — see `ARCHITECTURE-pcore-binding.md`; not
  feasible via config, and orthogonal to acceptance.

---

## 9. Out of Scope

- True multi-core parallelization of ggml graph construction (covered
  separately in `ARCHITECTURE-pcore-binding.md`).
- Any change to the GPU Flash-Attention / TBQ4 KV execution path.
- Model architecture changes beyond the draft head (target stays Q4_K_M 27B).
