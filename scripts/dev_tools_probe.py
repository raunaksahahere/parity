"""Development measurement for tools probe v2 (not evaluation data).

Runs the tools probe alone against the development-only deployment `dev-gpu-p4-b256`, honest
and through each simulated tool fault, and appends one JSON row per run to OUT. These rows
informed the prompt-set size (docs/decisions.md D15); they never enter the evaluation tables.

  uv run python scripts/dev_tools_probe.py docs/data/dev-tools-probe.jsonl
"""

import json
import sys
from pathlib import Path

from parity.faults.proxy import FaultProxy
from parity.faults.scenarios import TOOL_FAULTS, Paths, by_name
from parity.local.llamacpp import LlamaServer
from parity.suite import Suite, Target, hf_reference

out = Path(sys.argv[1])
dev = by_name(["dev-gpu-p4-b256"])[0]
ref = hf_reference(
    "models/hf/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
    model_id="Qwen/Qwen3-0.6B",
)
runs: list[tuple[str, object]] = [("honest", None)] + [(f.name, f.mutation) for f in TOOL_FAULTS]
with LlamaServer(dev.server.config(Paths(), Path("runs/dev-tools/server.log"))) as srv:
    for name, mutation in runs:
        proxy = FaultProxy(srv.base_url, mutation).start() if mutation else None  # type: ignore[arg-type]
        url = proxy.url if proxy else srv.base_url
        try:
            # alpha/4: the probe's share in the full core suite
            rep = Suite(probes=["tools.structured"], alpha=0.05 / 4).run(Target(url, "qwen3"), ref)
        finally:
            if proxy:
                proxy.stop()
        r = rep.probes[0].result
        row = {
            "deployment": dev.name,
            "fault": name,
            "status": r.status.value,
            "hard_evidence": r.hard_evidence,
            "p_value": r.p_value,
            "requests": r.n_requests,
            "seconds": r.duration_s,
            "metrics": r.metrics,
            "findings": [f.message for f in r.findings],
        }
        print(
            name,
            r.status.value,
            r.metrics.get("trigger_disagreements"),
            r.findings[0].message[:120],
        )
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
