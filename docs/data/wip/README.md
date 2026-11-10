# v0.1.1 study: work in progress (snapshot)

Snapshot taken 2026-11-10 23:43 local time while the study was still running.
These files are **partial** and are not published results; the final tables will be generated
from the complete `results.jsonl` by `scripts/update_result_docs.py`.

* Code: tag `v0.1.1-rc2` (study run from a separate checkout of it), `scripts/run_v011_study.sh`.
* Calibration thresholds: {"kl_mean": 0.000133, "greedy_hazard": 0.024751, "tool_disagreement": 0.076951}.
* 43 runs so far, 0 with an unexpected verdict. Rows 1-42 ran with 4 concurrent
  requests, later rows with 2 (to reduce heat); each row records its `concurrency`.

| Scenario | Runs |
|---|---:|
| heldout-gpu-p3-b1024 | 6 |
| fault-template-nosystem | 3 |
| fault-template-llama3 | 3 |
| heldout-gpu-fa-off | 6 |
| fault-topk-ignored | 3 |
| fault-topp-ignored | 3 |
| fault-temperature-ignored | 3 |
| fault-temperature-x1.5 | 3 |
| fault-silent-truncation | 3 |
| fault-tools-dropped | 3 |
| fault-tool-choice-ignored | 3 |
| fault-tool-name-invalid | 3 |
| fault-tool-args-malformed | 1 |

The aborted rc1 attempt (three rows, decision D16) is in `../v0.1.1-rc1-aborted-*`.
