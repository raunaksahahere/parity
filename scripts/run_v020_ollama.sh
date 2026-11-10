#!/usr/bin/env bash
# Live validation of the Ollama adapter (v0.2.0). Needs Ollama running on 127.0.0.1:11434 and
# the models from scripts/ollama_models.sh. Writes runs/v020-ollama/results.jsonl.
#
# SmolLM2-360M-Instruct (library template matches the official one):
#   calibrate on two benign variants: libtpl (GPU) and libtpl-cpu (num_gpu 0)
#   controls (expect PASS):       libtpl-ctx8k (held out), libtpl
#   faults (expect DEVIATION):    q4km-libtpl (Q4_K_M weights), f16-notpl (no chat template),
#                                 libtpl with 8192 claimed while Ollama gives 4096 (real default)
# Qwen3-0.6B (library template differs from the official one; Ollama ignores tool_choice):
#   libtpl, uncalibrated: reported as found, a real difference of the serving stack
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
out=${1:-runs/v020-ollama}
mkdir -p "$out"
url=http://127.0.0.1:11434
smol=(--model models/hf/SmolLM2-360M-Instruct --revision a10cc1512eabd3dde888204e902eca88bddb4951)
qwen=(--model models/hf/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca)
parity=${PARITY:-uv run parity}

if [ ! -f "$out/calibration-smollm2.json" ]; then
  $parity calibrate "${smol[@]}" --benign gpu=$url --benign cpu=$url \
    --benign-model gpu=parity-smollm2-f16-libtpl --benign-model cpu=parity-smollm2-f16-libtpl-cpu \
    --out "$out/calibration-smollm2.json" | tee "$out/calibrate-smollm2.log"
fi

run() {  # run LABEL KIND EXPECT CLAIMED SEED MODEL REFERENCE-ARGS... [--calibration FILE]
  local label=$1 kind=$2 expect=$3 claimed=$4 seed=$5 model=$6; shift 6
  local dir="$out/$label/seed$seed"
  if [ -f "$dir/done" ]; then return; fi
  mkdir -p "$dir"
  $parity run "$@" --target $url --served-model "$model" --claimed-context "$claimed" \
    --seed "$seed" --out "$dir" --history "$dir/history.duckdb" > "$dir/console.log" 2>&1
  local code=$?
  python3 - "$dir" "$label" "$kind" "$expect" "$code" "$seed" >> "$out/results.jsonl" <<'EOF'
import json, sys
from pathlib import Path
d, label, kind, expect, code, seed = sys.argv[1:]
rep = json.loads(next(Path(d).glob("*/report.json")).read_text())
print(json.dumps({
    "scenario": label, "kind": kind, "expects": expect, "rep": int(seed), "seed": int(seed),
    "backend": "ollama", "simulation": "real", "fresh": True, "exit_code": int(code),
    "verdict": rep["verdict"], "requests": rep["requests_total"], "seconds": rep["duration_s"],
    "run_id": rep["run_id"], "run_fingerprint": rep.get("run_fingerprint"),
    "adapter": rep["target"]["adapter"],
    "coverage": rep.get("coverage"),
    "probe_status": {p["result"]["probe"]: p["final_status"] for p in rep["probes"]},
    "probe_requests": {p["result"]["probe"]: p["result"]["n_requests"] for p in rep["probes"]},
    "probe_effect": {p["result"]["probe"]: p["result"]["effect"] for p in rep["probes"]},
    "probe_seconds": {p["result"]["probe"]: p["result"].get("duration_s") for p in rep["probes"]},
    "summary": rep["summary"],
}))
EOF
  touch "$dir/done"
  echo "$label seed$seed: exit $code ($(grep -o '^Verdict: [A-Z]*' "$dir/console.log"))"
}

cal=(--calibration "$out/calibration-smollm2.json")
for seed in 1000 1001 1002; do
  run smollm2-ctx8k control PASS 4096 $seed parity-smollm2-f16-libtpl-ctx8k "${smol[@]}" "${cal[@]}"
  run smollm2-libtpl control PASS 4096 $seed parity-smollm2-f16-libtpl "${smol[@]}" "${cal[@]}"
  run smollm2-q4km fault DEVIATION 4096 $seed parity-smollm2-q4km-libtpl "${smol[@]}" "${cal[@]}"
  run smollm2-notpl fault DEVIATION 4096 $seed parity-smollm2-f16-notpl "${smol[@]}" "${cal[@]}"
  run smollm2-claims-8k fault DEVIATION 8192 $seed parity-smollm2-f16-libtpl "${smol[@]}" "${cal[@]}"
done
run qwen3-libtpl finding DEVIATION 4096 1000 parity-qwen3-bf16-libtpl "${qwen[@]}"
echo "done: $out/results.jsonl"
