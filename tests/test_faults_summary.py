from __future__ import annotations

from parity.faults.scenarios import SCENARIOS, by_name
from parity.faults.summary import PROBES, clopper_pearson, summarize, to_markdown


def row(sc: str, kind: str, rep: int, verdict: str, flagged: list[str]) -> dict:  # type: ignore[type-arg]
    return {
        "scenario": sc, "kind": kind, "rep": rep, "verdict": verdict, "requests": 100 + rep,
        "seconds": 10.0 + rep,
        "probe_status": {p: ("DEVIATION" if p in flagged else "PASS") for p in PROBES},
    }  # fmt: skip


def test_clopper_pearson_known_values() -> None:
    lo, hi = clopper_pearson(0, 30)
    assert lo == 0.0 and abs(hi - 0.1157) < 1e-3  # exact one-sided bound is 9.5%, two-sided 11.6%
    lo, hi = clopper_pearson(30, 30)
    assert hi == 1.0 and lo > 0.88
    lo, hi = clopper_pearson(5, 10)
    assert lo < 0.5 < hi


def test_summary_counts_and_markdown() -> None:
    rows = [row("control-a", "control", i, "PASS", []) for i in range(10)]
    rows += [row("fault-x", "fault", i, "DEVIATION", ["template.token_counts"]) for i in range(3)]
    rows += [row("fault-y", "fault", i, "PASS", []) for i in range(3)]
    s = summarize(rows)
    assert s["control_runs"] == 10 and s["control_false_positives"] == 0
    assert s["faults"]["fault-x"]["deviation"] == 3 and s["faults"]["fault-y"]["deviation"] == 0
    md = to_markdown(s)
    assert "0/10" in md and "fault-x" in md and "3/3" in md
    assert "fault-y" in md and "0/3" in md  # an undetected fault is reported, not hidden


def test_scenario_catalogue_is_consistent() -> None:
    names = [s.name for s in SCENARIOS]
    assert len(names) == len(set(names))
    kinds = {s.kind for s in SCENARIOS}
    assert kinds == {"benign", "control", "fault"}
    assert all(s.expects == "PASS" for s in SCENARIOS if s.kind != "fault")
    # the faults requested in the spec are all present
    required = [
        "template",
        "quant-q8_0",
        "quant-q4_k_m",
        "quant-q2_k",
        "topk",
        "ctx",
        "temperature",
    ]
    for r in required:
        assert any(r in n for n in names if n.startswith("fault-")), r
    # controls are configured differently from the benign calibration variants
    benign = {s.server.key for s in SCENARIOS if s.kind == "benign"}
    assert all(s.server.key not in benign for s in SCENARIOS if s.kind == "control")
    assert by_name(["fault-topk-ignored"])[0].mutation is not None
