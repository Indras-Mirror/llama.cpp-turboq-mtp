# KV cache quant A/B — Qwen3.8-27B-MTP on RTX 4090

Build: `sync/indras-2026-09` (merged + fixes). Model: `Qwen3.8-27B-MTP-Q4_K_M`.
Quality: `llama-perplexity` on wikitext-2, ctx 4096 (16 chunks) / 32768 (3 chunks), FA on.
Perf: `llama-bench -p 512 -n 128 -d 0,32768`, FA on, ngl 99. TBQ3 forced with
`LLAMA_ALLOW_TBQ3_KV=1` (V cache is guarded as quality-unsafe by default).

## Quality (PPL, lower is better)

| KV type | ctx 4096 | Δ vs q8_0 | ctx 32768 | Δ vs q8_0 |
|---|---|---|---|---|
| q8_0 | **6.269** | — | **6.574** | — |
| q4_0 | 6.300 | +0.5% | 6.648 | +1.1% |
| tbq4_0 | 7.093 | **+13.1%** | 7.421 | **+12.9%** |
| tbq3_0 | 7.290 | **+16.3%** | 7.803 | **+18.7%** |

## Performance (t/s)

| KV type | pp512 | tg128 | pp512 @ d32768 | tg128 @ d32768 |
|---|---|---|---|---|
| q8_0 | 2924 | 48.7 | **2242** | **44.2** |
| q4_0 | 2894 | 48.5 | 2217 | 43.6 |
| tbq4_0 | 2808 | 46.4 | **872** | 36.1 |
| tbq3_0 | 2846 | 47.0 | **945** | 36.8 |

KV size: `tbq3_0` 3.125 bpv < `tbq4_0` 4.125 bpv ≈ `q4_0` 4.5 bpv << `q8_0` 8.5 bpv.

## Findings

1. **TBQ is not near-lossless on this model.** tbq4 costs **+13% PPL** and tbq3
   **+16–19%** vs q8_0/q4_0 (+0.5–1.1%). (Consistent pre- and post-merge, so it is
   inherent to the codec, not a sync regression.)
2. **tbq3 vs tbq4:** the bit reduction adds a further **+3% (4k) → +5% (32k)**
   PPL, i.e. it does cost quality and the gap grows with context. The dominant
   loss is however *shared* by both (the rotation/codec), not the bit budget.
3. **Performance has a depth cliff for TBQ:** at short context TBQ is only ~3%
   slower, but at 32k depth the fused-TBQ prefill collapses to **872–945 t/s vs
   2217–2242** (~2.4–2.5x slower) and decode drops ~17%. TBQ3 is ~8% faster than
   TBQ4 at depth (smaller blocks).
4. Long-context retrieval at ~238k: tbq4 PASS (315 t/s prefill).

## Recommendation

For Qwen3.8-27B on the 4090, **`q4_0` KV dominates `tbq4_0`/`tbq3_0`**:

- quality: +0.5% PPL vs +13% (tbq4)
- size: 4.5 vs 4.125 bpv (~9% larger — negligible at 262k on 24 GB)
- long-context prefill: ~2.5x faster (2217 vs 872 t/s @ 32k)

Use `-ctk q4_0 -ctv q4_0` for the daily driver (or `q8_0` if VRAM allows; it is the
best quality). Keep tbq4 only if the extra ~9% KV-size saving is needed to fit the
target context. TBQ3 additionally requires `LLAMA_ALLOW_TBQ3_KV=1` and is the
worst of the four on quality.

RotorQuant types (`planar3_0`/`iso3_0`) sit between: PPL 6.36/6.40 at ctx 4096
(~+1.5%/+2.1%) — better than TBQ at 3-bit size, worth a perf pass before discarding.
