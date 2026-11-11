#!/bin/bash
# Swap the in-process model of the llama-turboq server (in-process, no restart).
#
# Usage:
#   swap-model.sh <model-name> [port]
#
# Model names are the sections in /home/jack/llm/llama-turboq/models.ini,
# e.g. qwopus-27b, qwen35-dense, qwen35-pi-reasoning, qwen35-moe,
# gemma4-12b-q4, gemma4-12b-q8.
#
# The POST /models/load call is synchronous: it returns once the new model is
# loaded (or fails). This script then polls GET /models until the requested
# model reports status=loaded.

set -euo pipefail

NAME="${1:?usage: swap-model.sh <model-name> [port]}"
PORT="${2:-8081}"
BASE="http://127.0.0.1:$PORT"

RESP=$(curl -s -m 600 -X POST "$BASE/models/load" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$NAME\"}")

STATUS=$(printf '%s' "$RESP" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("parse-error")
    sys.exit(0)
# success payload: {"status": "loaded", "error": ""}
# failure payload: {"error": {"status": "error", ...}}
if d.get("status") == "loaded":
    print("loaded")
else:
    print("error")
')

if [ "$STATUS" = "error" ]; then
    echo "model swap FAILED:"
    printf '%s\n' "$RESP" | python3 -m json.tool
    exit 1
fi

# poll until the requested model reports loaded (should be immediate since the
# POST is synchronous; kept for robustness / use with the web UI)
for _ in $(seq 1 30); do
    LOADED=$(curl -s -m 10 "$BASE/models" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('no')
    sys.exit(0)
for m in d.get('data', []):
    if m.get('id') == '$NAME' and m.get('status') == 'loaded':
        print('yes')
        sys.exit(0)
print('no')
")
    if [ "$LOADED" = "yes" ]; then
        echo "model '$NAME' is now loaded"
        exit 0
    fi
    sleep 1
done

echo "model '$NAME': loaded but status not confirmed via /models after 30s"
exit 1
