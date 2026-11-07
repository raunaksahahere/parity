# Progress

- [x] 1. Scaffold
- [x] 2. Local targets and reference runner; baseline noise
- [x] 3. Template/tokenizer probe
- [x] 4. Sampler probe and G-test
- [x] 5. Teacher-forced log-prob comparison, KL, adapters
- [x] 6. Calibration, SPRT, Holm-Bonferroni, bootstrap
- [x] 7. Greedy agreement, long-context, tool/JSON probes
- [x] 8. Verdict logic, DuckDB history, HTML/Markdown/JSON reports, CLI polish
- [x] 9. Fault-injection harness and validation study (two full runs + sampler v3 re-validation)
- [x] 10. GitHub Action, Dockerfile, Prometheus exporter, packaging (Docker image builds and runs;
      wheel pip-installed into a fresh venv runs the full core suite: all PASS, exit 0, 325 s,
      peak RSS 4.4 GB with a cold reference cache)
- [x] 11. Documentation (README with quickstart, results and limits; methodology; validation;
      adding-probes; CONTRIBUTING; CHANGELOG; demo cast recorded from real runs; sample reports)
- [x] 12. Release prep (v0.1.0)

## Reproducing

* Validation study: `parity inject-faults run --out runs/validation` (resumable; `--probes`,
  `--cooldown`, `--concurrency` for subset runs and hot laptops), then
  `uv run python scripts/update_result_docs.py --main runs/validation-v2 --v1 runs/validation
  --sampler runs/validation-sampler-v3` to regenerate the tables.
* Demo: start BF16 (port 8081) and Q2_K (port 8082) llama.cpp servers, then
  `python scripts/record_cast.py docs/demo.cast -- bash docs/demo.sh`.
* Local assets (gitignored): `bin/` (llama.cpp b12128 Vulkan + CPU), `models/` (HF weights,
  GGUFs, see docs/decisions.md D4). On this machine pip is configured with `user=true`; use
  `PIP_USER=0` inside venvs.
