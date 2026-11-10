# Changelog

## 0.2.0 (unreleased, branch dev/v0.2)

* **No PASS without identity evidence.** A run in which no model-identity probe (distribution,
  greedy) reached a decision is INCONCLUSIVE; `--allow-unverified-pass` restores the old
  behaviour. Template, long-context, sampler and tool probes describe the serving stack only.
* **Greedy probe, chat mode.** Endpoints without token-id prompts (Ollama, hosted APIs) get an
  identity check: temperature-0 chat replies are compared with the reference as text, with a
  separate calibrated threshold (`greedy_chat_hazard`).
* **`parity doctor`**: a preflight of about 7 requests that reports an endpoint's capabilities
  and the probe coverage to expect; exit code 2 if no identity probe can run.
* **Ollama adapter** (detected from `/api/version`), verified against Ollama 0.33.2.
* Template probe v2 and tools probe v3 skip tool checks, with the reason, when the official
  template has no tool format or the endpoint rejects tools.
* Validation harness: model families (Qwen3-0.6B, SmolLM2-360M-Instruct).
* [docs/threat-model.md](docs/threat-model.md): what a PASS rests on and what can still pass.
* Live adapter contract tests and a vLLM harness for hardware that can run vLLM.

## 0.1.1 (unreleased)

Reliability release: results can no longer be influenced by an incompatible calibration or by
stale saved results, and the tool probe has real power.

* **Calibration compatibility.** Calibration files (schema 2) record the weights, tokenizer and
  chat-template fingerprints, reference kind and dtype, suite data fingerprint, calibrated probe
  versions and metric-relevant settings. `parity run` and `parity canary` refuse a mismatching
  or a 0.1.0 (schema 1) calibration before sending any request, listing every mismatch.
* **Run manifests.** Each run directory has `manifest.json` with the configuration and
  fingerprints the results depend on, a sha256 per saved probe result and a ledger of requests
  consumed. `--resume` reuses only results that match; changed configurations, edited, copied or
  stale files and 0.1.0 run directories are refused with the reasons.
* **Budgets across sessions.** A resumed run may only spend what is left of `--budget` after
  every earlier request (retries and interrupted probes included); the client enforces the cap.
  Probes that cannot run within the budget are INCONCLUSIVE. Reports add `requests_consumed`.
* **Study integrity.** Every validation-study row carries study and scenario fingerprints; a
  study directory holding rows from another configuration is refused. New harness options:
  `--seed-base` (fresh evaluation seeds), `--fail-fast` (real fail-fast runs), `--cool-to`
  (wait for the hardware to cool between repetitions); rows record peak RSS of parity and the
  server.
* **Tools probe v2** (suite data `core-v2`; `core-v1` is kept for reproducing 0.1.0): 96 auto
  prompts instead of 24, a paired call-validity test that does not blame the model's own
  invalid calls on the server, deterministic checks for `tool_choice="none"` violations and for
  strict JSON-schema replies that finished but are invalid; `tool_choice="required"` is reported
  as a capability (llama.cpp b12128 does not enforce it).
* **Fault harness.** Proxy mutations are labelled request rewriting or response corruption; new
  faults: dropped tool definitions, ignored `tool_choice`, invalid tool names, malformed tool
  arguments, schema-violating arguments, malformed structured output. CPU twins of the eleven
  original faults; fresh held-out control deployments.
* **Coverage.** Reports show evaluated and skipped probes prominently; `--require-probe`
  (CLI, canary, GitHub Action `require-probes`) turns a PASS into INCONCLUSIVE unless the named
  probes reached a decision.
* Fixed: Markdown reports broke the probe table when an effect unit contained `|`.
* Quickstart uses the BF16 GGUF as the known-good baseline (Q8_0 legitimately deviates from the
  BF16 reference) and states the expected outcome of each step; `scripts/quickstart_check.sh`
  verifies them.
* Release automation (`.github/workflows/release.yml`): build, wheel and Docker smoke tests on
  tags; publishing to PyPI and GHCR only on manual dispatch behind the `release` environment.

## 0.1.0 (2026-11-07)

First release.

* Six probe families: template and tokenizer conformance, distributional fidelity (truncated
  KL lower bound with cluster bootstrap), greedy agreement (divergence-hazard SPRT), sampler
  conformance (v3: every sample judged against the distribution reported in its own response;
  G-test with bias-corrected effect size), long-context integrity (token accounting that tells
  template differences from truncation), and tool and structured-output conformance.
* Statistics engine: threshold calibration from benign deployments, Bernoulli/Gaussian SPRT,
  cluster bootstrap, Holm-Bonferroni across probes, alpha-spending for sequential looks.
* Verdicts PASS / DEVIATION / INCONCLUSIVE with CI exit codes 0 / 1 / 2.
* Async client with bounded concurrency, token-bucket rate limiting and jittered retries;
  adapters for llama.cpp (validated live), vLLM and generic OpenAI-style endpoints (fixture
  tested only).
* Reports in HTML, Markdown and JSON (stable schema `docs/report-schema.json`); DuckDB run
  history with Parquet export; resumable runs; reference results cached per pinned revision.
* CLI: `run` (with `--fail-fast`), `report`, `calibrate`, `list-probes`, `history`, `canary`
  (Prometheus exporter), `inject-faults` (validation harness with `--probes`, `--cooldown`,
  `--concurrency`). Probes run cheapest first so fail-fast verdicts need few requests.
* Local llama.cpp launcher that selects the GPU by name, asserts GPU offload from the server log,
  and can disable the host prompt cache.
* Validation study on Qwen3-0.6B (docs/validation.md): 0/23 false positives on held-out correct
  deployments, all 11 injected faults detected, raw data in docs/data.
* GitHub Action for upgrade gates and a Dockerfile.
