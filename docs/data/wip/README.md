# v0.1.1 study: work in progress (snapshot)

Snapshot taken 2026-11-08 09:27 local time while the study was still running.
These files are **partial** and are not published results; the final tables will be generated
from the complete `results.jsonl` by `scripts/update_result_docs.py`.

* Code: tag `v0.1.1-rc2` (study run from a separate checkout of it), `scripts/run_v011_study.sh`.
* Calibration (benign-gpu-p4, benign-cpu-p4): thresholds
  {"kl_mean": 0.000133, "greedy_hazard": 0.024751, "tool_disagreement": 0.076951}.
* Rows so far (5), all with the expected verdict:

| Scenario | Rep | Seed | Verdict | Requests | Seconds |
|---|---:|---:|---|---:|---:|
| heldout-gpu-p3-b1024 | 0 | 1000 | PASS | 7467 | 143 |
| heldout-gpu-p3-b1024 | 1 | 1001 | PASS | 7467 | 137 |
| heldout-gpu-p3-b1024 | 2 | 1002 | PASS | 7467 | 142 |
| heldout-gpu-p3-b1024 | 3 | 1003 | PASS | 7467 | 136 |
| heldout-gpu-p3-b1024 | 4 | 1004 | PASS | 7467 | 138 |

Resume (it continues from the saved rows):

```bash
cd ../parity-v011-rc2   # checkout of v0.1.1-rc2 with bin/, models/, .parity-cache/ linked
PYTHONUNBUFFERED=1 setsid nohup bash scripts/run_v011_study.sh $PWD >> runs/v011-study/study.log 2>&1 &
```

The aborted rc1 attempt (three rows, decision D16) is in `../v0.1.1-rc1-aborted-*`.
