# HANDOFF PROMPT — New Context Window Quick-Start

You are taking over a TBQ4_0 fused flash attention project in `/home/mal/AI/llama.cpp-mtp`. Here's everything you need to know in one shot:

## What We're Doing
Fusing TBQ4_0 KV cache dequant into CUDA flash attention to eliminate ~40% overhead. Target: 90+ tok/s at 200K context on Qwen3.6-27B with MTP on RTX 4090 24GB.

## Current Status (CRITICAL)
- **Build:** CLEAN — zero errors, zero warnings
- **GPU dequant path:** WORKING at 200K (85 t/s @ 256, 66 t/s @ 512) on 27B
- **Fused kernel:** Compiled and ready. **NEEDS TESTING** — server was about to start when session ended
- **7B model:** TBQ4 crashes (alignment issue, not target — ignore for now)

## Immediate Task
Start the 27B server with fused kernel and benchmark:
```bash
fuser -k 8096/tcp
cd /home/mal/AI/llama.cpp-mtp
build/bin/llama-server \
  -m /media/Crucial1TB/models/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P-MTP-slim.gguf \
  --port 8096 -c 200000 --flash-attn on --mlock -t 8 --poll 0 -ngl 99 \
  --parallel 1 --spec-type mtp --spec-draft-n-max 3 -b 2048 -ub 32 \
  -ctk tbq4_0 -ctv tbq4_0 --jinja --temp 0.6 --seed 3407
```

**If loads clean:** Benchmark 256/512/1024 tokens. Compare to baseline (66 t/s @ 512, GPU dequant). Fused kernel should be faster.

**If crashes:** Check for "misaligned address" error. Debug steps:
1. Add fprintf to `fattn-mma-tbq4-launch.cuh` to show nb values
2. Test with `-fit off` to skip warmup
3. Try Q4_0 KV first to verify model loads: `-ctk q4_0 -ctv q4_0`
4. Fused kernel only activates when `per_head_dim == 128 || per_head_dim == 256` — verify in `src/llama-graph.cpp:1968/1988`

## Three Bugs Fixed (Don't Revert)
1. **llama-graph.cpp** — Conditional TBQ4 pass-through (was casting TBQ4→F16, killing fused path)
2. **fattn-mma-f16.cuh:570-574** — K stride in half2 vs bytes (was 128, needed 512)
3. **fattn-mma-f16.cuh:907-913** — V stride same fix

## Architecture
Rotated-domain attention: K/V tile loaders do centroid*norm lookup (no FWHT). FWHT runs twice: once on Q (rotate_forward), once on output (rotate_inverse). Eliminates FWHT from inner attention loop.

## Peer: DeepSeek (quetzacodetl-2)
Communicate via: `mcp__quetza-relay__relay_ask` → target `quetzacodetl-2`
Use relay_ask for NEW messages only (relay_reply ask_ids expire fast).
DeepSeek built: warp-shuffle FWHT, sparse V, block-diagonal Hadamard, D=256 rotation/template extension.

## Full Docs
- `COLLAB_NOTES.md` — Complete session log with all findings
- `HANDOFF_TBQ4.md` — File-level reference
- `DEEPSEEK_HANDOFF.md` — DeepSeek's task details

## Build
```bash
cd /home/mal/AI/llama.cpp-mtp
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc) --target llama-server
# Force recompile after header changes:
touch ggml/src/ggml-cuda/fattn-mma-f16.cuh && cmake --build build -j$(nproc) --target llama-server
```
