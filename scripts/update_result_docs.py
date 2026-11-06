"""Insert generated validation tables into README.md and docs/validation.md.

Usage: uv run python scripts/update_result_docs.py runs/validation-v2 [runs/validation]
The first directory is the published run; the optional second is the earlier run (sampler v1)
shown as an appendix. Numbers are never typed by hand: everything comes from results.jsonl.
"""

import sys
from pathlib import Path

from parity.faults.study import load_results
from parity.faults.summary import splice, summarize, to_markdown, to_readme_table

main_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/validation-v2")
summary = summarize(load_results(main_dir / "results.jsonl"))

readme = Path("README.md")
readme.write_text(
    splice(
        readme.read_text(),
        "RESULTS",
        "## Measured results\n\n"
        "Qwen3-0.6B on a laptop (CPU + RX 6500M), 23 correct-deployment controls and 11 injected "
        "faults x 3 repetitions. Method, controls and caveats: "
        "[docs/validation.md](docs/validation.md).\n\n" + to_readme_table(summary),
    )
)
val = Path("docs/validation.md")
text = splice(val.read_text(), "TABLES", to_markdown(summary))
if len(sys.argv) > 2:
    v1 = summarize(load_results(Path(sys.argv[2]) / "results.jsonl"))
    text = splice(text, "TABLES_V1", to_markdown(v1))
val.write_text(text)
print("updated README.md and docs/validation.md")
