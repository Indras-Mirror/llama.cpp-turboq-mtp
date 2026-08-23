#!/bin/bash
# Deep-context SWA stress test. Fills ~120K-token context, generates a longer
# response, measures decode t/s + VRAM. Runs against the shrink-base SWA build.
BIN=/home/mal/AI/llama.cpp-swa/build/bin/llama-server
MODEL=/media/mal/NVME1TB/Models/Qwen3.8/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
PORT=8095
LOG=/tmp/qwen-swa-stress.log
URL="http://localhost:$PORT"
PROMPT_FILE=/tmp/qwen-swa-bigprompt.txt

# Build a ~120K-token prompt (~480K chars of repetitive prose)
python3 -c "
p=('The quick brown fox jumps over the lazy dog and carefully considers the meaning of life, the universe, and everything. ')*4000
import sys; sys.stdout.write(p)
" > "$PROMPT_FILE"
echo "prompt chars: $(wc -c < "$PROMPT_FILE")"

"$BIN" -m "$MODEL" --port "$PORT" -c 131072 -ub 32 \
  --override-kv "qwen35.attention.sliding_window=int:4096" \
  -ctk tbq4_0 -ctv tbq4_0 \
  --flash-attn on -t 8 -ngl 99 --parallel 1 -b 512 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.82 \
  --jinja --no-warmup --seed 3407 \
  --chat-template-kwargs '{"enable_thinking":false}' \
  > "$LOG" 2>&1 &
SPID=$!
echo "server pid $SPID"

for i in $(seq 1 60); do
  curl -s --max-time 2 "$URL/v1/models" 2>/dev/null | grep -q '"id"' && { echo "ready after ${i}0s"; break; }
  sleep 5
done

echo "=== KV cache sizes (shrink proof) ==="
grep -niE "creating non-SWA KV cache|creating +SWA KV cache" "$LOG" | head -6

echo "=== VRAM before gen ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo "=== launch big completion ==="
python3 - "$URL" "$PROMPT_FILE" << 'PYEOF'
import json,sys,urllib.request,time
url, pf = sys.argv[1], sys.argv[2]
body = {"messages":[{"role":"user","content":open(pf).read()}],"max_tokens":400,"temperature":0.7,"top_p":0.8}
req = urllib.request.Request(url+"/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"})
t0=time.time()
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        d=json.load(r)
    dt=time.time()-t0
    print("HTTP ok, wall %.1fs" % dt)
    u=d.get("usage",{})
    print("prompt_tokens=%s completion_tokens=%s total=%s" % (u.get("prompt_tokens"),u.get("completion_tokens"),u.get("total_tokens")))
except Exception as e:
    print("ERROR:", e)
PYEOF

echo "=== VRAM after gen ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo "=== decode tg from log ==="
grep -nE "n_decoded =.*tg =|tg =|prompt processing.*tokens per second|draft acceptance|stop processing|error|Error|assert" "$LOG" | tail -12

kill "$SPID" 2>/dev/null
echo "=== done ==="
