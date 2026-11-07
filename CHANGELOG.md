# Changelog

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
