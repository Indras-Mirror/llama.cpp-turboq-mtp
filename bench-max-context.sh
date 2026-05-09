#!/bin/bash
# ============================================================================
# MAX CONTEXT BENCHMARK — Fill 262K context and test for degradation/quality
# Requires: llama-server already running on port 8097 with 262K context
# ============================================================================

PORT=8097
BASE_URL="http://localhost:$PORT"

# Ensure server is running
if ! curl -s --max-time 3 "$BASE_URL/v1/models" | jq -e '.data | length > 0' > /dev/null 2>&1; then
    echo "ERROR: No llama-server on port $PORT"
    exit 1
fi

MODEL=$(curl -s "$BASE_URL/v1/models" | jq -r '.data[0].id')
echo "Model loaded: $MODEL"

# Generate a large prompt: repeat blocks of diverse content to fill context
# ~25K tokens worth of text to fill KV cache
generate_fill_prompt() {
    local KB_CHUNKS=10  # ~10 × 2.5K tokens = ~25K tokens of diverse text
    for i in $(seq 1 $KB_CHUNKS); do
        cat <<'CHUNK'
UNITED STATES CODE — TITLE 17 — COPYRIGHT LAW — CHAPTER 1 — SECTION 107

Limitations on exclusive rights: Fair use. Notwithstanding the provisions of sections 106 and 106A, the fair use of a copyrighted work, including such use by reproduction in copies or phonorecords or by any other means specified by that section, for purposes such as criticism, comment, news reporting, teaching (including multiple copies for classroom use), scholarship, or research, is not an infringement of copyright. In determining whether the use made of a work in any particular case is a fair use the factors to be considered shall include: (1) the purpose and character of the use, including whether such use is of a commercial nature or is for nonprofit educational purposes; (2) the nature of the copyrighted work; (3) the amount and substantiality of the portion used in relation to the copyrighted work as a whole; and (4) the effect of the use upon the potential market for or value of the copyrighted work.

CHAPTER 2: COPYRIGHT OWNERSHIP AND TRANSFER

Section 201. Ownership of copyright. (a) Initial Ownership. Copyright in a work protected under this title vests initially in the author or authors of the work. The authors of a joint work are co-owners of copyright in the work. (b) Works Made for Hire. In the case of a work made for hire, the employer or other person for whom the work was prepared is considered the author for purposes of this title, and, unless the parties have expressly agreed otherwise in a written instrument signed by them, owns all of the rights comprised in the copyright.

IMPLEMENTATION NOTES: The fused TBQ4 flash attention kernel operates on 128-element blocks with Hadamard-rotated K/V tensors. Each block is 66 bytes packed as 64 bytes of 4-bit centroid indices plus 2 bytes for the corrected L2 norm. Centroids are pre-computed Lloyd-Max optimal quantizers for N(0, 1/sqrt(128)) in the FWHT domain, stored in CUDA constant memory for fast lookup. The key architectural insight is that the Hadamard matrix H is orthonormal (H^T = H), so attention can be computed entirely in the rotated domain: rotate Q once, compute QK^T, then rotate the output once. This eliminates the 7-stage FWHT butterfly from the inner attention loop entirely.

TECHNICAL SPECIFICATION: Qwen3.6-27B architecture features 48 transformer layers with 24 attention heads, 4 key-value heads (GQA ratio = 6), embedding dimension 2560, head dimension 128 (key/value also 128). The multi-token prediction heads add an additional transformer block (blk.64 weights) that predicts subsequent tokens from intermediate hidden states. The MTP layer uses the same attention mechanism as the main model but with a combined embedding input (token embedding + previous hidden state) projected through an eh_proj fusion layer.

BENCHMARK DATA — RTX 4090 24GB configuration at 262K context with TBQ4_0 KV cache and MTP draft 3: Prefill processing at 614 tokens/second on 26K token initial prompts. Decode speeds ranging from 62.7 to 98.5 tokens/second with 73-93% draft token acceptance rates. Flash attention fused kernel achieves 4.25 bits per value effective storage. Total VRAM utilization approximately 20 GB leaving 4 GB headroom for compute buffers at maximum context.

ALGORITHM PSEUDOCODE for the fused TBQ4 flash attention inner loop: For each KV tile (128 elements), iterate over rows. For each row, load the 66-byte TBQ4 block from global memory using __ldg. Extract the 4-bit centroid indices from the packed 64-byte qs array. For each pair of elements (one byte = two 4-bit nibbles), perform: lo = centroid[nibble_low] * norm; hi = centroid[nibble_high] * norm; then pack as half2. The resulting half2 tile feeds directly into the MMA (matrix multiply-accumulate) pipeline. The Hadamard rotation happens exactly twice per attention head: once to pre-rotate Q, once to post-rotate the accumulated VKQ output.

PROGRAMMING LANGUAGES COMPARED: Rust offers memory safety without garbage collection through its ownership system. Go provides simple concurrency with goroutines and channels. Python excels at rapid prototyping with its dynamic typing and rich ecosystem. C++ gives maximum performance with template metaprogramming and zero-cost abstractions. JavaScript dominates web development with its event-driven model and vast npm ecosystem. Each language represents different trade-offs in the safety-versus-performance spectrum. The choice depends on project requirements, team expertise, and deployment constraints.

MATHEMATICAL BACKGROUND: The Fast Walsh-Hadamard Transform (FWHT) is a divide-and-conquer algorithm that computes the Hadamard transform in O(n log n) operations. For a vector x of length n (where n is a power of 2), the Hadamard transform H_n is defined recursively: H_1 = [1], H_{2k} = [H_k, H_k; H_k, -H_k]. The FWHT butterfly network consists of log2(n) stages, each performing n/2 butterfly operations. For n=128, this is 7 stages × 64 butterflies = 448 operations, compared to the full matrix multiply which would be 16,384 operations.

DATABASE SYSTEMS OVERVIEW: Relational databases (PostgreSQL, MySQL) use SQL with ACID transactions - ideal for structured data with complex queries. NoSQL stores (MongoDB, Cassandra) sacrifice consistency for scalability. Key-value stores (Redis) excel at caching with sub-millisecond latency. Graph databases (Neo4j) model relationships natively. Time-series databases (InfluxDB) optimize for append-heavy workloads. The CAP theorem states you can only have two of: Consistency, Availability, Partition tolerance.

LINUX KERNEL ARCHITECTURE: The kernel manages process scheduling (CFS - Completely Fair Scheduler), memory management (virtual memory with page tables and TLB), file systems (VFS layer with ext4, btrfs, xfs backends), and device drivers. The kernel uses a monolithic architecture with loadable modules. System calls provide the user-space API. The kernel operates in Ring 0 (supervisor mode) with direct hardware access, while user processes run in Ring 3. Context switches save and restore CPU state when switching between processes.

CUDA PROGRAMMING MODEL: Threads are organized in a hierarchy: individual threads within warps of 32, warps within blocks (up to 1024 threads), and blocks within a grid. Each block has shared memory visible to all its threads - this is the key to flash attention's efficiency. Global memory accesses should be coalesced (adjacent threads access adjacent addresses). Constant memory is cached and optimized for broadcast reads (all threads reading the same address). The RTX 4090 (Ada Lovelace, sm_89) has 128 SMs, 72 MB L2 cache, and 1,008 GB/s memory bandwidth.
CHUNK
    done
}

echo ""
echo "=== PHASE 1: Fill KV cache with ~25K tokens ==="
FILL_PROMPT=$(generate_fill_prompt)
FILL_JSON=$(jq -n --arg prompt "$FILL_PROMPT" '{
  prompt: $prompt,
  max_tokens: 1,
  temperature: 0,
  stream: false
}')

echo "Sending fill prompt ($(echo "$FILL_PROMPT" | wc -c) bytes)..."
START=$(date +%s%N)
FILL_RESULT=$(curl -s --max-time 600 "$BASE_URL/completion" \
  -H "Content-Type: application/json" \
  -d "$FILL_JSON")
FILL_END=$(date +%s%N)
FILL_MS=$(( (FILL_END - START) / 1000000 ))
echo "Fill completed in ${FILL_MS}ms"

# Parse fill stats
FILL_TOKENS=$(echo "$FILL_RESULT" | jq -r '.timings.predicted_n // .usage.prompt_tokens // 0' 2>/dev/null)
echo "Prompt tokens processed: $FILL_TOKENS"
FILL_TPS=$(echo "$FILL_RESULT" | jq -r '.timings.predicted_per_second // 0' 2>/dev/null 2>/dev/null)
echo "Fill speed: $FILL_TPS tok/s"

echo ""
echo "=== PHASE 2: Test generation quality at high context ==="
# Test 1: Factual recall — ask about something IN the fill prompt
echo ""
echo "--- Test 1: Near-context factual recall ---"
RECALL_JSON=$(jq -n '{
  prompt: "Based on the legal text discussed earlier, what are the 4 factors of fair use under Section 107 of US copyright law? List them concisely.",
  max_tokens: 150,
  temperature: 0,
  stream: false
}')
START=$(date +%s%N)
RECALL_RESULT=$(curl -s --max-time 120 "$BASE_URL/completion" \
  -H "Content-Type: application/json" \
  -d "$RECALL_JSON")
RECALL_END=$(date +%s%N)
RECALL_MS=$(( (RECALL_END - START) / 1000 ))
RECALL_TEXT=$(echo "$RECALL_RESULT" | jq -r '.content // .choices[0].text // "ERROR"')
RECALL_TOKENS=$(echo "$RECALL_RESULT" | jq -r '.timings.predicted_n // 0')
RECALL_TPS=$(echo "$RECALL_RESULT" | jq -r '.timings.predicted_per_second // 0')
echo "Time: ${RECALL_MS}ms | Tokens: $RECALL_TOKENS | Speed: $RECALL_TPS tok/s"
echo "Response: ${RECALL_TEXT:0:300}"

# Test 2: Coding task
echo ""
echo "--- Test 2: Code generation at high context ---"
CODE_JSON=$(jq -n '{
  prompt: "Write a Python function that implements a trie (prefix tree) with insert and search methods. Use clean, production-quality code.",
  max_tokens: 200,
  temperature: 0,
  stream: false
}')
START=$(date +%s%N)
CODE_RESULT=$(curl -s --max-time 120 "$BASE_URL/completion" \
  -H "Content-Type: application/json" \
  -d "$CODE_JSON")
CODE_END=$(date +%s%N)
CODE_MS=$(( (CODE_END - START) / 1000 ))
CODE_TEXT=$(echo "$CODE_RESULT" | jq -r '.content // .choices[0].text // "ERROR"')
CODE_TOKENS=$(echo "$CODE_RESULT" | jq -r '.timings.predicted_n // 0')
CODE_TPS=$(echo "$CODE_RESULT" | jq -r '.timings.predicted_per_second // 0')
echo "Time: ${CODE_MS}ms | Tokens: $CODE_TOKENS | Speed: $CODE_TPS tok/s"
echo "Response: ${CODE_TEXT:0:300}"

# Test 3: General reasoning (math)
echo ""
echo "--- Test 3: Math reasoning at high context ---"
MATH_JSON=$(jq -n '{
  prompt: "If a train leaves Station A at 60 mph and another train leaves Station B at 45 mph heading toward each other, and the stations are 420 miles apart, how long until they meet? Show your work concisely.",
  max_tokens: 150,
  temperature: 0,
  stream: false
}')
START=$(date +%s%N)
MATH_RESULT=$(curl -s --max-time 120 "$BASE_URL/completion" \
  -H "Content-Type: application/json" \
  -d "$MATH_JSON")
MATH_END=$(date +%s%N)
MATH_MS=$(( (MATH_END - MATH_START) / 1000 ))
MATH_TEXT=$(echo "$MATH_RESULT" | jq -r '.content // .choices[0].text // "ERROR"')
MATH_TOKENS=$(echo "$MATH_RESULT" | jq -r '.timings.predicted_n // 0')
MATH_TPS=$(echo "$MATH_RESULT" | jq -r '.timings.predicted_per_second // 0')
echo "Time: ${MATH_MS}ms | Tokens: $MATH_TOKENS | Speed: $MATH_TPS tok/s"
echo "Response: ${MATH_TEXT:0:300}"

# Test 4: Long-form generation
echo ""
echo "--- Test 4: Sustained generation (512 tokens) ---"
LONG_JSON=$(jq -n '{
  prompt: "Write a short story about a programmer who discovers that their code has become sentient. Keep it under 200 words.",
  max_tokens: 512,
  temperature: 0.7,
  stream: false
}')
START=$(date +%s%N)
LONG_RESULT=$(curl -s --max-time 180 "$BASE_URL/completion" \
  -H "Content-Type: application/json" \
  -d "$LONG_JSON")
LONG_END=$(date +%s%N)
LONG_MS=$(( (LONG_END - LONG_START) / 1000 ))
LONG_TEXT=$(echo "$LONG_RESULT" | jq -r '.content // .choices[0].text // "ERROR"')
LONG_TOKENS=$(echo "$LONG_RESULT" | jq -r '.timings.predicted_n // 0')
LONG_TPS=$(echo "$LONG_RESULT" | jq -r '.timings.predicted_per_second // 0')
echo "Time: ${LONG_MS}ms | Tokens: $LONG_TOKENS | Speed: $LONG_TPS tok/s"
echo "Response: ${LONG_TEXT:0:500}"

echo ""
echo "=== BENCHMARK SUMMARY ==="
echo "Fill prompt: ${FILL_TOKENS} tokens at ${FILL_TPS} tok/s (${FILL_MS}ms)"
echo "Recall test: ${RECALL_TOKENS} tokens at ${RECALL_TPS} tok/s"
echo "Code test:   ${CODE_TOKENS} tokens at ${CODE_TPS} tok/s"
echo "Math test:   ${MATH_TOKENS} tokens at ${MATH_TPS} tok/s"
echo "Long gen:    ${LONG_TOKENS} tokens at ${LONG_TPS} tok/s"
echo ""
echo "Server: $BASE_URL | Model: $MODEL | Context: 262K | KV: TBQ4_0 | MTP: draft 3"
