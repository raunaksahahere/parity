"""Insert generated validation tables into README.md and docs/validation.md.

Usage:
  uv run python scripts/update_result_docs.py --main runs/validation-v2 \
      [--v1 runs/validation] [--sampler runs/validation-sampler-v3]

--main is the published full study, --v1 the first run (appendix), --sampler the sampler-only
re-validation. Numbers are never typed by hand: everything comes from results.jsonl files.
"""

import argparse
from pathlib import Path

from parity.faults.study import load_results
from parity.faults.summary import (
    SAMPLER_FAULTS,
    splice,
    summarize,
    to_markdown,
    to_readme_table,
    to_sampler_table,
)

ap = argparse.ArgumentParser()
ap.add_argument("--main", type=Path, required=True)
ap.add_argument("--v1", type=Path)
ap.add_argument("--sampler", type=Path)
ns = ap.parse_args()

summary = summarize(load_results(ns.main / "results.jsonl"))
sampler_note = ""
if ns.sampler:
    srows = load_results(ns.sampler / "results.jsonl")
    flagged = [r["probe_status"].get("sampler.conformance") == "DEVIATION" for r in srows]
    is_fault = [r["scenario"] in SAMPLER_FAULTS for r in srows]
    tp = sum(f and x for f, x in zip(flagged, is_fault, strict=True))
    fp = sum(f and not x for f, x in zip(flagged, is_fault, strict=True))
    q2 = [f for f, r in zip(flagged, srows, strict=True) if r["scenario"] == "fault-quant-q2_k"]
    sampler_note = (
        f"\nThe `sampler (1/3)` on Q2_K was a misattribution by sampler v2 (numerical jitter "
        f"between requests). Sampler v3, re-validated alone on the GPU deployments, reported "
        f"DEVIATION on {tp}/{sum(is_fault)} sampler-fault repetitions and on {fp}/"
        f"{len(srows) - sum(is_fault)} others, including {sum(q2)}/{len(q2)} on Q2_K.\n"
    )
readme = Path("README.md")
readme.write_text(
    splice(
        readme.read_text(),
        "RESULTS",
        "## Measured results\n\n"
        "Qwen3-0.6B on a laptop (CPU + RX 6500M), 23 correct-deployment controls and 11 injected "
        "faults x 3 repetitions. Method, controls and caveats: "
        "[docs/validation.md](docs/validation.md).\n\n" + to_readme_table(summary) + sampler_note,
    )
)
val = Path("docs/validation.md")
text = splice(val.read_text(), "TABLES", to_markdown(summary))
if ns.v1:
    text = splice(text, "TABLES_V1", to_markdown(summarize(load_results(ns.v1 / "results.jsonl"))))
if ns.sampler:
    text = splice(
        text, "TABLES_SAMPLER_V3", to_sampler_table(load_results(ns.sampler / "results.jsonl"))
    )
val.write_text(text)
print("updated README.md and docs/validation.md")
