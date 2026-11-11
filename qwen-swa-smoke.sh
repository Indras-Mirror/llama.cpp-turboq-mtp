#!/bin/bash
# SWA smoke test for qwen35 (Qwen3.8-27B-agg). Run AFTER the -j4 build lands.
# Verifies: qwen loads with SWA override + MTP, SWA KV cache created, no assert,
# one valid draft, one short generation.
set -e
SWA_ROOT=/home/mal/AI/llama.cpp-swa
BIN="$SWA_ROOT/build/bin/llama-server"
MODEL=/media/mal/NVME1TB/Models/Qwen3.8/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
PORT=8095
LOG=/tmp/qwen-swa-smoke.log
URL="http://localhost:$PORT"

echo ">>> binary present?"; test -x "$BIN" || { echo "BUILD NOT READY - $BIN missing"; exit 2; }

echo ">>> GPU free?"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

"$BIN" -m "$MODEL" --port "$PORT" -c 8192 \
  --override-kv "qwen35.attention.sliding_window=int:4096" \
  --flash-attn on -t 8 -ngl 99 --parallel 1 -b 1024 -ub 32 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.82 \
  --jinja --no-warmup --seed 3407 --temp 0.7 --top-p 0.8 -n 128 \
  > "$LOG" 2>&1 &
SPID=$!
echo ">>> server pid $SPID"

for i in $(seq 1 60); do
  if curl -s --max-time 2 "$URL/v1/models" 2>/dev/null | grep -q '"id"'; then echo ">>> ready after ${i}0s"; break; fi
  sleep 5
done

echo "=== SWA KV cache line ==="; grep -Ei "SWA KV cache|swa_type|iswa" "$LOG" | head -5 || echo "no SWA line"
echo "=== errors/asserts ==="; grep -Ei "error|assert|segfault|wrong-type" "$LOG" | grep -viE "tool|repeat" | head -5 || echo "none"

echo "=== one completion ==="
curl -s --max-time 60 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":32}' \
  | head -c 500; echo

echo "=== spec/draft lines ==="; grep -Ei "draft|spec|acceptance" "$LOG" | tail -5 || echo "n/a"

kill "$SPID" 2>/dev/null; echo ">>> smoke test done"
