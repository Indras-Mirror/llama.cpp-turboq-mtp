# HANDOFF: TBQ4_0 Fused Flash Attention

**Date:** 2026-05-08  
**Repo:** `/home/mal/AI/llama.cpp-mtp` (fork: `github.com/Indras-Mirror/llama.cpp-mtp`, branch `master`)  
**Session:** Gemma + Claude + DeepSeek collaboration  
**Goal:** Fuse TBQ4_0 dequant into flash attention — target 90+ tok/s at 200K on Qwen3.6-27B

---

## TL;DR — What's Working

| Component | Status |
|-----------|--------|
| GPU-side TBQ4 dequant on 27B at 200K | ✅ 85 t/s @ 256, 66 t/s @ 512 |
| llama-graph.cpp cast fix | ✅ Conditional pass-through for D=128/256 |
| Stride unit fix (half2→bytes) | ✅ Applied to K (:572) and V (:910) tile loads |
| D=256 kernel extension | ✅ Rotation, templates, dispatch — all compiled |
| Fused kernel TEST on 27B | ❓ INTERRUPTED — server was starting when session ended |
| 7B TBQ4 support | ❌ nb1=264 alignment issue (deferred, not target) |

## Three Bugs Found

1. **llama-graph.cpp:1952** — Cast TBQ4→F16 before FA → fused kernel never active, CPU dequant
2. **fattn-mma-f16.cuh:572/910** — `stride_K` (half2 units) passed to TBQ4 loader expecting bytes → 128 vs 512
3. **7B nb1=264** — 4-head models produce 8-byte aligned stride → VEC dispatch fails

## Next: Test Fused Kernel on 27B

```bash
fuser -k 8096/tcp
cd /home/mal/AI/llama.cpp-mtp
build/bin/llama-server \
  -m /media/Crucial1TB/models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P-MTP-slim.gguf \
  --port 8096 -c 200000 --flash-attn on --mlock -t 8 --poll 0 -ngl 99 \
  --parallel 1 --spec-type mtp --spec-draft-n-max 3 -b 2048 -ub 32 \
  -ctk tbq4_0 -ctv tbq4_0 --jinja --temp 0.6 --seed 3407
```

Expected: server loads without crash. If successful, benchmark vs 66 t/s baseline.

## Key Files Changed

| File | Change | Author |
|------|--------|--------|
| `src/llama-graph.cpp:1957-1991` | Conditional TBQ4 pass-through | Gemma |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh:570-574,907-913` | stride_K/V * sizeof(half2) fix | Gemma |
| `ggml/src/ggml-cuda/fattn-mma-tbq4.cuh` | Q rotation + output rotation D=256 | DeepSeek |
| `ggml/src/ggml-cuda/fattn.cu:422-432,547-616` | Dispatch D=128+256 | DeepSeek |
| `ggml/src/ggml-cuda/fattn-mma-tbq4-launch.cuh` | TBQ4 MMA launcher | Claude |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh:520-521,562-574` | TBQ4 type hooks | Claude |
| `ggml/src/ggml-cuda/tbq4-cuda.cuh` | Warp-shuffle FWHT + BD | DeepSeek |
| `ggml/src/ggml-cuda/tbq4-sparse-v.cuh` | Sparse V utility | DeepSeek |
| `ggml/src/ggml-cuda/cpy.cu:554-559` | TBQ4→F32 CUDA dequant | Claude |
| `ggml/src/ggml-cuda/set-rows.cu:324-326` | TBQ4 CUDA quantize | Claude |
| 4× template instance files | D=128 + D=256 MMA kernels | Claude + DeepSeek |

See `COLLAB_NOTES.md` for detailed session log.
