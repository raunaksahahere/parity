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
otherwise. Pinned build: llama.cpp b12128. GPU for study runs: the RX 6500M, which the
Vulkan loader lists as `Vulkan1` (the integrated Radeon that drives the display is `Vulkan0`).
The index is not guaranteed across driver updates or reboots, so the launcher selects the GPU
by name (`device="RX 6500"`, resolved from `llama-server --list-devices`, exactly one match
required) instead of trusting the index.

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

## D13. Sampler probe v3: every sample judged against its own response (found by run 2)
The second validation run showed sampler v2 still flagging Q2_K in 1 of 3 repetitions, now
for `top_k=5` (2.0% of samples ranked 6th or worse). Measured directly: Q2_K's reported
log-probs differ by up to 0.058 nats between *identical* requests (BF16: 0.0016), while that
context's 5th/6th gap was 0.077 nats, so no fixed guard threshold is safe. llama.cpp reports the
raw temperature-1 distribution in every response regardless of the sampler settings (verified),
so v3 checks each sample's support membership against its own response and runs the G-test
against the mixture of the per-sample distributions. Expected distributions no longer depend on
a separate fetch. Re-validated on the GPU with a sampler-only run (subset runs keep the
probe's full-suite alpha share): see docs/validation.md.
Two re-validation attempts were discarded before the published one: the harness CLI parsed
`--probes` but did not pass it on, so the first attempt ran the whole suite (fixed, with a
regression test); the second still carried v2's near-tie and "no stable top_p" skips, which
skipped the `top_k=5` case on Q2_K in 3 of 4 repetitions and made those passes uninformative.
Under v3 those skips only cost power (validity no longer depends on the fetched distribution),
so they were removed; `top_p` placement is kept as a preference. Raw rows of the second attempt:
`docs/data/validation-sampler-v3-aborted-near-tie-skips.jsonl`.
The re-validation was run on a laptop that reaches 90 C (chassis sensor) under sustained GPU load.
After two thermal pauses, the remaining repetitions (from `fault-temperature-ignored` on) were run
with a 120 s idle cool-down after each repetition and 2 concurrent requests instead of 4, under a
supervisor that pauses at 90 C and resumes below 60 C. Request concurrency only changes how
requests are batched on the server; sampler v3's verdicts are per response and do not depend on
it. `parity inject-faults run` gained `--cooldown` and `--concurrency` for this.

## D14. v0.1.1: calibrations, saved runs and study rows must prove where they came from
In v0.1.0 nothing stopped a calibration made for another model revision, tokenizer, suite or
probe version from setting a run's thresholds, and `--resume` reused any `probes/*.json` it
found, whatever configuration produced it. Either could change a verdict silently.
* Calibration schema 2 records a context: weights fingerprint (sha256 over config and weight
  files; the Hugging Face cache's blob names are reused instead of rehashing), tokenizer
  fingerprint (tokenizer files plus observed behaviour: ids of a fixed text and renders of a
  fixed chat with and without tools, so a library upgrade that changes rendering also counts),
  reference kind and dtype, suite data name and fingerprint, versions of the calibrated probes
  and the probe options that change calibrated metrics. `parity run` checks it before the first
  request and refuses with every mismatch listed. When both sides have a weights fingerprint it
  decides model identity (a hub id and a local copy of the same files match); otherwise the
  model id and revision must match. Schema-1 files are refused (they cannot be checked).
* Every run directory gets `manifest.json`: an identity (code fingerprint, library versions,
  target, reference fingerprints, probes and versions, suite data, alpha, budget, seed,
  fail-fast, options, calibration fingerprint) that must match exactly to resume, plus one
  checkpoint per saved probe with the sha256 of its file. A checkpoint is written before its
  file appears, so a crash leaves at most a checkpoint without a file (the probe re-runs), never
  an unvouched file. Copied, edited or stale files, and directories without a manifest, are
  refused. Concurrency and rate limits are recorded but not enforced.
* Budgets: the manifest keeps a ledger of every request sent (retries and interrupted probes
  included), saved at most once a second and on exit. A resumed run may only send
  `budget - consumed`; the client enforces it as a hard cap, and probes that cannot run are
  INCONCLUSIVE, never silently PASS.
* The study harness stamps every result row with a study fingerprint (code, reference,
  calibration, probes, seed base, fail-fast, llama-server binaries) and a scenario fingerprint
  (server settings, mutation label, GGUF sha256, template file) and refuses a directory whose
  rows do not match.
* Reports carry a coverage block (evaluated, decided, skipped with reasons). `--require-probe`
  turns a PASS into INCONCLUSIVE unless every required probe reached PASS or DEVIATION; a
  DEVIATION stands on its own evidence.

## D15. Tools probe v2: paired validity, deterministic tool_choice and schema checks, 96 prompts
Measured on llama.cpp b12128 serving Qwen3 (development deployment `dev-gpu-p4-b256`):
`tool_choice="required"` is accepted but not enforced (the model answered a haiku prompt in
text); the named-function object form is rejected with "Wrong type supplied for parameter
'tool_choice', using default value"; `tool_choice="none"` is honoured (no structured
tool_calls; the model's own `<tool_call>` markup stays in the content). Strict
`response_format=json_schema` is enforced by grammar. Consequences:
* Deterministic evidence only for what a conforming server must do whatever the model does:
  no structured tool_calls under `none`; strict JSON-schema replies that finished normally must
  be valid (a reply cut off by `max_tokens` is not counted); auto requests must not leave raw
  `<tool_call>` markup. `required` is reported as a capability, never as evidence.
* v0.1 treated an invalid tool name or argument as deterministic evidence. On a server without
  constrained decoding the *model* can make such calls, and so would the reference. v2 compares
  them pairwise with the reference (one-sided sign test on prompts where both call a tool).
* Prompt-set size: with 24 auto prompts and the calibrated benign rate (2 x Wilson upper bound
  of 0/24 = 0.276) the trigger test had 27% power even at a 50% disagreement rate.
  Target: power >= 0.9 to detect a 0.25 disagreement rate at the per-test alpha
  0.05 / 4 / 3 = 0.0042 (`scripts/tools_power.py`). 96 prompts meet it when the benign
  deployments disagree on no prompt (0.98), 120 would be needed with one disagreement (0.98 vs
  0.79). The development deployment disagreed on 0 of 96, so 96 were kept (48 trigger, 48
  plain, interleaved across four tools), plus 12 `none` prompts, 8 `required` prompts and 24
  JSON prompts (140 requests; reference outputs cached per revision).
* Development check (`docs/data/v0.1.1-dev-tools-probe.jsonl`, never evaluation data): the honest
  deployment PASSed with 0 disagreements; each of the six simulated tool faults was detected.
  These rows informed the design and are excluded from every evaluation table.
* Suite data moved to `core-v2` (only `tools.json` changed); `core-v1` is kept verbatim so v0.1.0
  results stay reproducible with `options={"suite": "core-v1"}`.

## D16. v0.1.1-rc2: concurrent tool and long-context requests (study restarted)
The first evaluation attempt with `v0.1.1-rc1` was stopped after three rows: on the CPU one full
suite took 1865 s, 1069 s of it in the tools probe, because its 140 independent requests were
sent one at a time while the server had four slots (GPU: 143 s of 530 s). The long-context
probe's nine requests were sequential too (CPU 305 s). rc2 sends each section's independent
requests concurrently, bounded by the client's existing concurrency limit (a regression test
checks requests overlap and never exceed it); requests refused by the budget cap are no longer
counted as sent. Measured on the development deployment: tools 34 s instead of 74-143 s,
long-context 22 s instead of 72 s, identical honest result (0 of 96 trigger disagreements).
Statistics, prompts and probe versions are unchanged; greedy replies can differ from sequential
ones only through batch-composition numerics, which is the benign variation calibration and
the controls measure. Because the code changed, the study was restarted from calibration with
`v0.1.1-rc2`; the three rc1 rows and their calibration are kept as
`docs/data/v0.1.1-rc1-aborted-*`. The study also lost two runs to hard power-offs of the laptop
(battery at 0% and not charging; the hottest sensor read 94 C after one of them); results are
appended per run, so it resumed without losing completed rows.
To keep the study within about a day, CPU repetitions were reduced (CPU controls 2 runs each,
CPU fault twins 1 run each; `scripts/run_v011_study.sh`). Repetition counts are not part of
the study fingerprint.

## D17. Ollama 0.33.2: what its OpenAI-compatible API offers (v0.2.0)
Measured with `scripts/ollama_capabilities.py` (raw answers in
`docs/data/v0.2.0-ollama-capabilities-*.json`) on models created from the same verified GGUF
files used with llama.cpp (Qwen3-0.6B BF16, SmolLM2-360M-Instruct F16), with the chat template,
parameters and system prompt copied from the Ollama library (`qwen3:0.6b`, `smollm2:360m`):
* `/v1/completions` rejects token-id prompts (HTTP 400), ignores `echo` and returns no log-probs,
  so the distribution, greedy and sampler probes cannot run; reports say so in their coverage
  block and are marked lower power. `/v1/chat/completions` reports `usage.prompt_tokens` and the
  log-probs of generated tokens.
* `chat_template_kwargs` is ignored; `reasoning_effort: "none"` switches Qwen3's thinking off
  (`think: false` does not). The Ollama adapter translates a probe's `enable_thinking: false`.
* `tool_choice` is ignored: `"none"` still returned a tool call. Parity's deterministic check
  reports this as a deviation, which it is (the endpoint does not honour the parameter).
* A model created from a GGUF without a TEMPLATE gets `{{ .Prompt }}`; chat then goes through an
  embedded llama.cpp without `--jinja`, and requests with `tools` fail with HTTP 500 ("tools
  param requires --jinja flag").
* Models whose library template has no tool format reject `tools` with HTTP 400 ("does not
  support tools"). The tools and template probes now skip their tool sections in that case, and
  when the model's official template has no tool format, instead of reporting an error or a
  false deviation (tools probe v3, template probe v2).
* Token counts for a system + user chat: official Qwen3 template 18 (22 with thinking off),
  Ollama library template 20 (26); official SmolLM2 template 19, Ollama library template 19.
  The Qwen3 library template is therefore not the official template, and the template probe is
  expected to say so.

## D18. v0.1.1 study: operational changes while it ran
The study ran on a laptop that throttles at about 95 C and was switched off several times.
Operational changes, none of which enter the study fingerprint and all recorded per row:
* runs 1-42 used 4 concurrent requests, later runs 2 (to reduce heat);
* the start gate (cool-down before each run) was 70 C, then 66 C;
* after the GPU phase, the two CPU phases (CPU controls, CPU twins of the eleven original faults)
  ran from `runs/cpu_phases_after.sh` in the study checkout with the same settings, because the
  phase script that would have started them had been stopped. Repetition counts as in
  `scripts/run_v011_study.sh` (CPU controls 2 runs, CPU fault twins 1 run).
