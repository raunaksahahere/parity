"""Insert generated validation tables into README.md and docs/validation.md.

Usage: uv run python scripts/update_result_docs.py runs/validation
Numbers are never typed by hand: everything comes from results.jsonl.
"""

import sys
from pathlib import Path

from parity.faults.study import load_results
from parity.faults.summary import splice, summarize, to_markdown, to_readme_table

out = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/validation")
summary = summarize(load_results(out / "results.jsonl"))

readme = Path("README.md")
readme.write_text(
    splice(
        readme.read_text(),
        "RESULTS",
        "## Measured results\n\n"
        "Qwen3-0.6B on a laptop (CPU + RX 6500M). Full method, controls and caveats: "
        "[docs/validation.md](docs/validation.md).\n\n" + to_readme_table(summary),
    )
)
val = Path("docs/validation.md")
val.write_text(splice(val.read_text(), "TABLES", to_markdown(summary)))
print("updated README.md and docs/validation.md")
