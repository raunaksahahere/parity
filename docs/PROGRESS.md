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
- [ ] 10. GitHub Action, Dockerfile, Prometheus exporter, packaging (all written; wheel verified
      with pip in a clean venv; remaining: live `parity run` from the clean venv, docker build)
- [ ] 11. Documentation (methodology, adding-probes, CONTRIBUTING, CHANGELOG, README draft done;
      remaining: validation results + interpretation, demo cast, sample HTML report)
- [ ] 12. Release prep (v0.1.0)

## Resuming

* Validation study (resumable, skips finished repetitions):
  `PYTHONUNBUFFERED=1 setsid nohup uv run parity inject-faults run --out runs/validation >> runs/validation/study.log 2>&1 &`
  Calibration is in `runs/validation/calibration.json` (copied to `docs/data/`); results append
  to `runs/validation/results.jsonl`.
* After the study: `cp runs/validation/results.jsonl docs/data/validation-results.jsonl`,
  `uv run python scripts/update_result_docs.py runs/validation`, then write the interpretation
  section of `docs/validation.md` from the generated tables.
* Demo: start BF16 (port 8081) and Q2_K (port 8082) servers, then
  `uv run python scripts/record_cast.py docs/demo.cast -- bash docs/demo.sh`.
* Local assets (gitignored): `bin/` (llama.cpp b12128 Vulkan + CPU), `models/` (HF weights,
  GGUFs). pip on this machine is configured with `user=true`; use `PIP_USER=0` inside venvs.
