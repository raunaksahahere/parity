#!/usr/bin/env bash
# Live validation of the vLLM adapter. Needs hardware vLLM supports: an NVIDIA GPU (default
# image) or an x86 CPU with AVX-512 (VLLM_IMAGE=<vllm CPU release image>). It could not be run
# on the development laptop (AMD RX 6500M, not supported by vLLM/ROCm; Ryzen 5 5600H has no
# AVX-512), so the adapter stays labelled "fixture-tested" until this script has passed.
#
#   bash scripts/vllm_live.sh OUT_DIR
#
# Steps: start vLLM with Qwen3-0.6B at the pinned revision, run the adapter contract tests
# (tests/live), calibrate on two benign vLLM configurations, then run the core suite against a
# third, held-out configuration and expect PASS (exit 0).
set -euo pipefail
out=${1:?usage: vllm_live.sh OUT_DIR}
mkdir -p "$out"
image=${VLLM_IMAGE:-vllm/vllm-openai:v0.11.0}
model=Qwen/Qwen3-0.6B
rev=c1899de289a04d12100db370d81485cdf75e47ca
gpu_flags=${VLLM_DOCKER_FLAGS:---gpus all}
parity=${PARITY:-uv run parity}

start() {  # start NAME PORT [vllm args...]
  local name=$1 port=$2; shift 2
  # shellcheck disable=SC2086
  docker run -d --rm --name "$name" $gpu_flags -p "$port:8000" \
    -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
    "$image" --model "$model" --revision "$rev" --served-model-name qwen3 \
    --max-model-len 8192 --max-logprobs 100 --seed 0 "$@" >/dev/null
  for _ in $(seq 300); do curl -sf "http://127.0.0.1:$port/health" >/dev/null && return 0; sleep 2; done
  docker logs "$name" | tail -30; exit 3
}
cleanup() { docker rm -f parity-vllm-a parity-vllm-b parity-vllm-c >/dev/null 2>&1 || true; }
trap cleanup EXIT

start parity-vllm-a 8100
PARITY_LIVE_URL=http://127.0.0.1:8100 PARITY_LIVE_ADAPTER=vllm PARITY_LIVE_MODEL=$model \
  PARITY_LIVE_SERVED=qwen3 PARITY_LIVE_REVISION=$rev \
  uv run pytest -m live tests/live -v | tee "$out/contract.log"

start parity-vllm-b 8101 --max-num-seqs 2
$parity calibrate --model "$model" --revision "$rev" --served-model qwen3 \
  --benign seqs256=http://127.0.0.1:8100 --benign seqs2=http://127.0.0.1:8101 \
  --out "$out/calibration.json" | tee "$out/calibrate.log"
docker rm -f parity-vllm-a parity-vllm-b >/dev/null

start parity-vllm-c 8102 --max-num-seqs 8 --enforce-eager
set +e
$parity run --model "$model" --revision "$rev" --served-model qwen3 --adapter vllm \
  --target http://127.0.0.1:8102 --calibration "$out/calibration.json" \
  --require-probe all --out "$out/runs" | tee "$out/run.log"
code=${PIPESTATUS[0]}
echo "parity exit code: $code (expected 0)"
exit "$code"
