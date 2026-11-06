# Changelog

## 0.1.0 (unreleased)

First release.

* Six probe families: template and tokenizer conformance, distributional fidelity (truncated
  KL lower bound with cluster bootstrap), greedy agreement (divergence-hazard SPRT), sampler
  conformance (G-test with bias-corrected effect size), long-context integrity, and tool and
  structured-output conformance.
* Statistics engine: threshold calibration from benign deployments, Bernoulli/Gaussian SPRT,
  cluster bootstrap, Holm-Bonferroni across probes, alpha-spending for sequential looks.
* Verdicts PASS / DEVIATION / INCONCLUSIVE with CI exit codes 0 / 1 / 2.
* Async client with bounded concurrency, token-bucket rate limiting and jittered retries;
  adapters for llama.cpp (validated live), vLLM and generic OpenAI-style endpoints (fixture
  tested only).
* Reports in HTML, Markdown and JSON (stable schema `docs/report-schema.json`); DuckDB run
  history with Parquet export; resumable runs; reference results cached per pinned revision.
* CLI: `run`, `report`, `calibrate`, `list-probes`, `history`, `canary` (Prometheus exporter),
  `inject-faults` (validation harness).
* GitHub Action for upgrade gates and a Dockerfile.
