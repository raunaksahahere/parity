#!/usr/bin/env bash
# Terminal demo: one honest deployment and one quantized to Q2_K, compared with the reference.
# Needs two llama.cpp servers already running (see README quickstart):
#   GOOD_URL  serving the BF16 GGUF      (default http://127.0.0.1:8081)
#   BAD_URL   serving the Q2_K GGUF      (default http://127.0.0.1:8082)
# and MODEL pointing at the reference (HF id or local directory).
set -u
GOOD_URL=${GOOD_URL:-http://127.0.0.1:8081}
BAD_URL=${BAD_URL:-http://127.0.0.1:8082}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
REV=${REV:-c1899de289a04d12100db370d81485cdf75e47ca}
PROBES=(--probes template.token_counts --probes distribution.teacher_forced --probes greedy.agreement)

say() { printf '\n\033[1m$ %s\033[0m\n' "$*"; }

say parity list-probes
parity list-probes

say parity run --target "$GOOD_URL" "${PROBES[@]}"
parity run --model "$MODEL" --revision "$REV" --target "$GOOD_URL" --served-model qwen3 \
  "${PROBES[@]}" --out demo-runs
echo "exit code: $?"

say parity run --target "$BAD_URL" "${PROBES[@]}"
parity run --model "$MODEL" --revision "$REV" --target "$BAD_URL" --served-model qwen3 \
  "${PROBES[@]}" --out demo-runs
echo "exit code: $?"

say parity history --db demo-runs/history.duckdb
parity history --db demo-runs/history.duckdb
