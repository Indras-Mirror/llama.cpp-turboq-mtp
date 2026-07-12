# FastMTP Retraining — Handoff for Model Owner

## Problem

Our MTP head (Qwopus3.6-27B-v2-MTP) achieves **~50–57% draft acceptance at
temperature 0.6–1.0**, while industry MTP/EAGLE heads routinely achieve
**70–95%**. This gap is **not** a runtime/quantization bug (proven: greedy
draft is acceptance-optimal, the head runs GPU-side on active CUDA graphs).
It is the **trained quality of the MTP head** itself.

| Operating point | Our acceptance | Industry target | Gap |
|---|---|---|---|
| temp 1.0 | 53% | 70–95% (EAGLE) | 17–42 pts |
| temp 0.6 | 57% | 70–95% | 13–38 pts |
| temp 0.0 (greedy) | 68% | ~95% | 27 pts |

Closing this gap is the **single largest decode-throughput lever** remaining.
Inference-side fixes (temp 0.6, spec-adapt guard) are maxed out.

---

## Solution: FastMTP (Tencent, 2025)

**FastMTP** replaces the vanilla MTP training objective (next-token CE on each
draft head) with an **acceptance-aware differentiable objective** that directly
optimizes the *expected accepted length* under rejection sampling.

### What changes

- A small **auxiliary head** (a few linear layers) is added to the MTP head.
- The training loss is: `L = L_CE + λ·L_accepted_len`, where `L_accepted_len`
  rewards the draft distribution that maximizes the expected number of accepted
  tokens per step under the current target model.
- **LK (Likelihood + KL) losses** add a KL-divergence term between the draft
  and target distributions at each position → **+8–10% acceptance length**.

### Reported gains (Tencent, same 7B–72B models)

| Metric | Vanilla MTP | FastMTP | Δ |
|---|---|---|---|
| Accepted tokens per step | 1.0× baseline | **+82%** | ×1.82 |
| Decode speedup over autoregressive | ~2.2× | **~3.8×** | +73% |
| Acceptance length (LK + FastMTP) | — | **+8–10%** additional | — |

### Requirements

1. **Base model checkpoint** — Qwopus3.6-27B (weights, not just GGUF).
2. **Existing MTP head** — already exists in the model; FastMTP modifies it.
3. **Training pipeline** — standard fine-tuning (PyTorch/DeepSpeed/FSDP).
   FastMTP implementation is ~200 lines of PyTorch (publicly available).
4. **Data** — ~1B tokens of general-domain text (same distribution as original
   training). No curation beyond basic dedup.
5. **Hardware** — 4–8× A100 80GB (or equivalent) for 27B, ~2–3 days.

### Training recipe sketch

```
1. Load Qwopus base + existing MTP head.
2. Add FastMTP aux head (2× linear, hidden dim = d_model).
3. Freeze base model; train only MTP + aux head.
   — Objective: L_CE + λ·L_accepted_len + μ·L_KL (LK losses)
   — λ = 0.3, μ = 0.1 (typical)
4. Train for 5K–10K steps, batch 256, lr 1e-5, cosine schedule.
5. Extract the retrained MTP head; export to GGUF.
```

---

## Integration — no inference code changes needed

The retrained head is a drop-in replacement:
- Existing `--spec-type mtp` path in `common/speculative.cpp` works unchanged.
- The head is baked into the GGUF; server picks it up automatically.
- No changes to GPU/FA/KV path, no new flags.
- `--temp 0.6` (current production config) will see acceptance rise from ~57%
  toward ~80%+ → decode tok/s from ~82 toward ~110+.

---

## References

| Paper | Link |
|---|---|
| FastMTP (Tencent, 2025) | https://arxiv.org/abs/2501.00632 |
| LK Losses for MTP | https://arxiv.org/abs/2502.01756 |
| EAGLE-2 (tree drafting) | https://arxiv.org/abs/2406.16858 |

---

## Contact / Handoff

This document describes the work needed from the model-owner side.
Questions about the inference pipeline, GGUF export, or verification
testing should go to the inference team (this repo). Questions about
FastMTP impl, hyperparameters, or training should go to the training side.
