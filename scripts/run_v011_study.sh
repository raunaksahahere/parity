#!/usr/bin/env bash
# The v0.1.1 evaluation study (docs/validation.md). As run: phase 1 only, with 4 and later 2
# concurrent requests and a 66 C start gate; phases 2 and 3 (CPU) were dropped (decision D18). Run from a checkout of the
# frozen tag v0.1.1-rc2 with bin/, models/ and .parity-cache/ available:
#
#   bash scripts/run_v011_study.sh /path/to/checkout
#
# The first invocation calibrates on the benign deployments (runs/v011-study/calibration.json);
# later phases reuse that calibration after checking its compatibility. CPU repetitions are
# reduced to keep the study within about a day (docs/decisions.md D17); repetition counts are
# not part of the study fingerprint, so an interrupted run resumes with this same script.
set -uo pipefail
cd "${1:-.}"
run() {
  .venv/bin/parity inject-faults run --out runs/v011-study --seed-base 1000 --concurrency 4 \
    --cool-to 70 "$@"
}
gpu_controls=(control-gpu-p2 control-gpu-p1-smallbatch heldout-gpu-fa-off heldout-gpu-p3-b1024
              heldout-gpu-p1-nocb)
faults=(fault-template-llama3 fault-template-nosystem fault-quant-q8_0 fault-quant-q4_k_m
        fault-quant-q2_k fault-topk-ignored fault-topp-ignored fault-temperature-ignored
        fault-temperature-x1.5 fault-ctx-reduced fault-silent-truncation)
tool_faults=(fault-tools-dropped fault-tool-choice-ignored fault-tool-name-invalid
             fault-tool-args-malformed fault-tool-args-schema fault-json-malformed)
cpu_faults=("${faults[@]/%/-cpu}")

# 1. GPU: every control and fault at its catalogue repetition count
run --scenarios "${gpu_controls[@]}" "${faults[@]}" "${tool_faults[@]}" || exit 1
# 2. CPU controls: 2 repetitions each
run --reps 2 --scenarios control-cpu-p1-t4 heldout-cpu-p2-t8 || exit 1
# 3. CPU twins of the eleven original faults: 1 repetition each
run --reps 1 --scenarios "${cpu_faults[@]}" || exit 1
echo "study complete"
