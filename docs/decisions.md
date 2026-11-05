# Decisions

Non-obvious choices, recorded as they are made.

## D1. Repo identity and licensing
- Local git identity `raunaksahahere <raunaksahahere@users.noreply.github.com>`, set repo-locally only.
- License: Apache-2.0. Commits stay local; nothing is pushed.
- No third-party provider API keys; validation is local only. Secrets are read
  from environment variables only if a user supplies them.

## D2. Test model and hardware
- Hardware: 12 CPU threads, ~7 GB RAM, AMD RX 6500M (4 GB VRAM, RDNA2) plus an
  integrated Vega GPU. No CUDA; ROCm does not officially support gfx1034.
- Primary study model: Qwen3-0.6B (reference Q8_0; Q4_K_M and Q2_K as
  quantization faults). Optional Qwen3-1.7B smoke check if memory allows.
- Backend: the user chose to run on the GPU. llama.cpp's Vulkan backend is the
  primary backend, with reference and targets always on the same backend so
  backend numerics are never mistaken for a fault. CPU is the fallback if Vulkan
  is unusable, and a CPU-reference vs GPU-target run is kept as an explicit control.

## D3. llama.cpp must be launched from its own directory
Launching the prebuilt Vulkan `llama-server` from another working directory
loaded no GPU backend and silently ran on CPU (observed: 654 MB RSS, 6 threads,
15 MiB VRAM used). `parity.local.llamacpp` launches with the binary's directory
as cwd and asserts "offloaded N/M layers to GPU" from the log, failing loudly
otherwise. Pinned build: llama.cpp b12128. GPU for study runs: `Vulkan1`
(RX 6500M); `Vulkan0` is the integrated Vega and is never used.

## D4. Model provenance
- Reference weights: `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`
  (HF transformers, fp32, CPU).
- Targets: `unsloth/Qwen3-0.6B-GGUF@50968a4` BF16 GGUF quantized locally with the
  pinned `llama-quantize` to Q8_0, Q4_K_M and Q2_K (checksums in `models/gguf/SHA256SUMS`,
  not committed; models are gitignored).
- Sanity check (prompt "The capital of France is", log-prob of " Paris"): HF fp32 -0.42017,
  llama.cpp BF16 CPU -0.42026, Q8_0 -0.37196, Q4_K_M -0.92367, Q2_K flips top-1 to " the".
  So the pipeline agrees with the independent reference at full precision, and even
  Q8_0 is a real (small) numerical deviation for this model, not a harness bug.

## D5. llama.cpp API facts that shape the adapters (observed on b12128)
- `/v1/completions` returns chat-style `logprobs.content[]` with token ids, not the
  classic OpenAI `tokens/token_logprobs` arrays.
- `echo` is ignored: no prompt log-probs. Teacher forcing is done one position at a
  time with token-id prompts (the prompt cache keeps this cheap). Consequence: the
  log-prob of the forced token is only known when it is inside the returned top-k.
- Server-side sampler defaults (`top_k=40`, `top_p=0.95`, `min_p=0.05`) apply unless
  overridden; probes always send every sampling parameter explicitly.
- Qwen3 opens with `<think>` by default; chat probes pass `enable_thinking=false`
  through the template (or account for it) so behaviour is comparable.
- Repeats with `cache_prompt=false` on CPU are bit-identical.
