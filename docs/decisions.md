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
loaded no GPU backend and silently ran on CPU (observed: 654 MB RSS, 6 threads, no Vulkan
lines in the log). Note: the sysfs `mem_info_vram_used` counter is NOT a reliable indicator
here (it reads ~16 MiB even with the model on the GPU); the server log line is. `parity.local.llamacpp` launches with the binary's directory
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

## D6. Reference vs. benign variation
The HF fp32 reference and a correctly deployed llama.cpp BF16 build differ by about 0.001 nats
(mean |Δ| of the forced-token log-prob), CPU BF16 by about 0.007, Q8_0 by about 0.034
(docs/baseline-noise.md). Thresholds therefore cannot be set from same-configuration repeats
(zero on GPU); `parity calibrate` measures variation across benign deployment changes
(parallel slots, backend, concurrency) against the reference.

## D7. Sampler probe takes its expected distribution from the target itself
The expected next-token distribution for the sampler probe comes from the target's own
temperature-1 log-probs, so quantization differences are not blamed on the sampler. Only
order-independent combinations are tested (`top_k` with any temperature; `top_p`/`min_p` at
temperature 1) because llama.cpp truncates before applying temperature while HF/vLLM apply
temperature first.

## D8. Distribution probe: truncated KL as a lower bound, equivalence-style verdicts
- KL(reference || target) is estimated from the top-k lists after merging every id outside
  the *shared* reported set into one bin on both sides. This is a coarse-graining, so the
  estimate is a rigorous lower bound (property-tested against the true KL); the bias is
  downward only. Ids missing from one side lose their divergence rather than being floored.
- Uncertainty is a document-level cluster bootstrap (positions in a document are correlated).
- PASS requires the one-sided upper bound to be below the threshold (equivalence shown),
  DEVIATION requires the lower bound to be above it, otherwise INCONCLUSIVE.
- Measured on Qwen3-0.6B (docs/data/distribution-live.json): BF16 GPU 1e-5, BF16 CPU 5e-5,
  Q8_0 1.25e-3, Q4_K_M 0.083, Q2_K 1.42 nats. The uncalibrated default threshold (2e-3) is
  deliberately lenient; calibrated thresholds sit just above benign variation.
- Threshold calibration uses benign deployment variants (backend, slots, concurrency), not
  same-configuration repeats (zero on GPU): threshold = max(benign upper bounds) x factor.

## D9. Greedy agreement is a divergence-hazard SPRT
Each compared position is a Bernoulli trial for "first divergence here" (a prompt stops
contributing at its first divergence). Measured per-token hazards vs the HF reference
(docs/data/greedy-live.json): BF16 GPU 0.0008, BF16 CPU 0.0047, Q8_0 0.028, Q4_K_M 0.169,
Q2_K 0.676. Defaults h0=0.005, h1=0.025 until calibrated. The SPRT reports
p = min(1, exp(-LLR)), a valid sequential p-value (the likelihood ratio is a nonnegative
martingale under H0), so Holm-Bonferroni can span all probes including the sequential ones.

## D10. Long-context and tool probes lean on deterministic evidence
Long-context: token accounting (usage.prompt_tokens vs the locally rendered prompt) and
rejection inside the claimed window are hard evidence; needle retrieval alone never yields
DEVIATION for a 0.6B model (it yields INCONCLUSIVE). Tools: paired comparison with the
reference on identical greedy prompts (exact binomial tests); the probe reports its own
minimum detectable disagreement because 24 trigger prompts give limited power. Tool parsing
of the reference supports Hermes/Qwen `<tool_call>` markup only in v0.1.

## D11. Sampler probe v2: top_p placed per context (found by the validation study)
In the first validation run the sampler probe flagged a Q2_K deployment (1 of 3 repetitions):
2.4% of samples fell outside the nucleus for top_p=0.6. The sampler was correct; the
server's own reported log-probs jitter slightly between requests (batch composition with
parallel slots), and the context's cumulative mass sat within a hair of 0.6, so the boundary
token moved in and out of the nucleus. v1 only guarded *log-prob gaps* at truncation boundaries,
which is the right check for top_k but not for top_p. v2 places top_p per context at the
midpoint between two consecutive cumulative masses (closest to the nominal 0.6, boundary token
>= 4% probability, >= 5% of mass excluded), so jitter below 0.02 cannot move the boundary;
min_p cutoffs get the analogous margin check. The study was re-run in full with v2; the v1
results are kept in docs/data for transparency.

## D12. Study servers run with llama.cpp's host prompt cache disabled
llama.cpp b12128 keeps KV states of past prompts in host RAM (`--cache-ram`, default
8192 MiB). The study sends thousands of distinct prompts, so the cache filled toward its
limit, the server reached 4.3 GB of anonymous memory on this 7 GB machine, and the system
swapped (repetitions slowed from ~143 s to ~220 s). From 2026-11-07 00:55 IST the harness starts
every server with `--cache-ram 0`. In the second validation run, the eight scenarios that had
already completed (CPU control and eight faults) ran with the default cache; the partially
finished `control-gpu-p1-smallbatch` was discarded (raw rows kept as
`aborted-smallbatch-with-prompt-cache.jsonl`) and re-run with the cache disabled, as were
`control-gpu-p2` and the three quantizations. Each scenario is internally consistent. Disabling
a cache is a benign deployment difference, which is the kind of variation the held-out controls
exist to test; `LlamaServerConfig.cache_ram_mib` exposes the setting.
