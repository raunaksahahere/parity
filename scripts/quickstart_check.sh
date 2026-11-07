#!/usr/bin/env bash
# Runs the README quickstart end to end and checks every expected outcome.
#
#   LLAMA_SERVER=/path/to/llama-server [LLAMA_SERVER_CPU=/path/to/cpu/llama-server] \
#   PARITY=/path/to/venv/bin/parity GGUF_DIR=models/gguf bash scripts/quickstart_check.sh OUT_DIR
#
# LLAMA_GPU_ARGS adds arguments to the GPU servers (e.g. "--device Vulkan1" on a machine whose
# first Vulkan device is an integrated GPU).
#
# Expected (asserted below):
#   1. BF16 GGUF, uncalibrated                 -> PASS       (exit 0)
#   2. parity calibrate on BF16 GPU + BF16 CPU -> calibration.json written
#   3. BF16 GGUF, calibrated                   -> PASS       (exit 0)
#   4. Q4_K_M GGUF, calibrated                 -> DEVIATION  (exit 1)
#   5. Q8_0 GGUF, calibrated                   -> DEVIATION  (exit 1, distribution probe)
# The Q8_0 and Q4_K_M files are quantized from the same BF16 file with llama-quantize.
set -uo pipefail
out=${1:?usage: quickstart_check.sh OUT_DIR}
mkdir -p "$out"
llama=${LLAMA_SERVER:?set LLAMA_SERVER to a llama-server binary}
llama_cpu=${LLAMA_SERVER_CPU:-$llama}
parity=${PARITY:-parity}
gguf=${GGUF_DIR:-models/gguf}
model=Qwen/Qwen3-0.6B
rev=c1899de289a04d12100db370d81485cdf75e47ca
read -r -a gpu_args <<<"${LLAMA_GPU_ARGS:-}"
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT

serve() {  # serve BINARY GGUF PORT [extra args...]
  local bin=$1 file=$2 port=$3; shift 3
  (cd "$(dirname "$bin")" && exec ./llama-server -m "$(realpath "$file")" --port "$port" \
     --jinja -c 16384 --parallel 4 --cache-ram 0 "$@") >"$out/server-$port.log" 2>&1 &
  pids+=($!)
  for _ in $(seq 120); do curl -sf "http://127.0.0.1:$port/health" >/dev/null && return 0; sleep 1; done
  echo "server on $port did not start"; tail "$out/server-$port.log"; exit 3
}
stop_all() { cleanup; pids=(); sleep 2; }

check() {  # check NAME EXPECTED_EXIT -- parity args...
  local name=$1 want=$2; shift 3
  local t0=$SECONDS
  "$parity" "$@" --out "$out/runs" >"$out/$name.log" 2>&1
  local got=$?
  local verdict
  verdict=$(grep -o "^Verdict: [A-Z]*" "$out/$name.log" | head -1)
  printf '%-28s expected exit %s, got %s (%s, %ss)\n' "$name" "$want" "$got" "$verdict" $((SECONDS - t0))
  [ "$got" = "$want" ] || { echo "MISMATCH, see $out/$name.log"; failures=$((failures + 1)); }
}
failures=0
common=(--model "$model" --revision "$rev" --served-model qwen3 --claimed-context 4096)

serve "$llama" "$gguf/Qwen3-0.6B-BF16.gguf" 8081 "${gpu_args[@]}"
check bf16-uncalibrated 0 -- run "${common[@]}" --target http://localhost:8081

serve "$llama_cpu" "$gguf/Qwen3-0.6B-BF16.gguf" 8082 --device none
"$parity" calibrate --model "$model" --revision "$rev" --served-model qwen3 \
  --benign gpu=http://localhost:8081 --benign cpu=http://localhost:8082 \
  --out "$out/calibration.json" >"$out/calibrate.log" 2>&1 \
  && echo "calibrate                    wrote $out/calibration.json" \
  || { echo "calibrate FAILED"; failures=$((failures + 1)); }
check bf16-calibrated 0 -- run "${common[@]}" --target http://localhost:8081 \
  --calibration "$out/calibration.json"
stop_all

serve "$llama" "$gguf/Qwen3-0.6B-Q4_K_M.gguf" 8083 "${gpu_args[@]}"
check q4_k_m-calibrated 1 -- run "${common[@]}" --target http://localhost:8083 \
  --calibration "$out/calibration.json"
stop_all

serve "$llama" "$gguf/Qwen3-0.6B-Q8_0.gguf" 8084 "${gpu_args[@]}"
check q8_0-calibrated 1 -- run "${common[@]}" --target http://localhost:8084 \
  --calibration "$out/calibration.json"
stop_all

echo "failures: $failures"
exit "$failures"
