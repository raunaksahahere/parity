#!/usr/bin/env bash
# Confirmation that the v0.1.1 study carries over to the v0.3.0 code (docs/validation.md).
# Run from a checkout of the exact v0.3.0 release commit, with bin/, models/ and .parity-cache/
# linked, after the v0.1.1 study:  bash scripts/run_v030_confirm.sh /path/to/checkout
#
# v0.3.0 changed the template (v2) and tools (v3) probes, so the v0.1.1 calibration is
# (correctly) refused and this run calibrates afresh on the same benign deployments. New seeds
# (3000+); a subset of the study's controls and faults.
set -uo pipefail
cd "${1:-.}"
.venv/bin/parity inject-faults run --out runs/v030-confirm --seed-base 3000 \
  --concurrency "${CONC:-2}" --cool-to "${COOL_TO:-66}" --reps 2 --scenarios \
  control-gpu-p2 heldout-gpu-fa-off heldout-gpu-p1-nocb \
  fault-template-llama3 fault-quant-q8_0 fault-quant-q4_k_m fault-topk-ignored \
  fault-silent-truncation fault-ctx-reduced fault-tools-dropped fault-tool-choice-ignored \
  fault-json-malformed
