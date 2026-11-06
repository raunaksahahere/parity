"""Threshold calibration from benign deployment variation.

Thresholds must not be set from same-configuration repeats: on a deterministic GPU build
those are exactly zero (docs/baseline-noise.md). Instead, the suite is run against known-good
deployments that differ in benign ways (backend, concurrency, slots, batch settings) and each
metric's threshold is placed above the worst benign value, with a safety factor and a floor.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field
from scipy import stats

from parity.probes.base import ProbeResult


class MetricCalibration(BaseModel):
    benign_values: dict[str, float]  # deployment label -> observed upper-bound value
    threshold: float
    method: str


class Calibration(BaseModel):
    schema_version: int = 1
    model_id: str
    revision: str
    suite: str
    created: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    metrics: dict[str, MetricCalibration] = Field(default_factory=dict)

    def thresholds(self) -> dict[str, float]:
        return {k: v.threshold for k, v in self.metrics.items()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=1))

    @classmethod
    def load(cls, path: Path) -> Calibration:
        return cls.model_validate(json.loads(path.read_text()))


def calibrate_threshold(
    benign_values: dict[str, float], *, safety_factor: float = 2.0, floor: float = 0.0
) -> MetricCalibration:
    """Place the threshold at ``max(benign) * safety_factor`` (never below ``floor``).

    ``benign_values`` should be conservative per-deployment values (e.g. the upper confidence
    bound of the metric), so that a correct deployment of the same kind lands below it.
    """
    if not benign_values:
        raise ValueError("need at least one benign deployment")
    if safety_factor < 1.0:
        raise ValueError("safety_factor must be >= 1")
    worst = max(benign_values.values())
    return MetricCalibration(
        benign_values=dict(benign_values),
        threshold=max(worst * safety_factor, floor),
        method=f"max(benign upper bounds) x {safety_factor:g}, floor {floor:g}",
    )


FLOORS = {"kl_mean": 1e-5, "greedy_hazard": 0.002, "tool_disagreement": 0.05}


def wilson_upper(events: float, trials: float, confidence: float = 0.95) -> float:
    """Upper Wilson score bound for a binomial proportion."""
    if trials <= 0:
        return 1.0
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    p = events / trials
    denom = 1 + z**2 / trials
    centre = p + z**2 / (2 * trials)
    margin = z * ((p * (1 - p) + z**2 / (4 * trials)) / trials) ** 0.5
    return float(min(1.0, (centre + margin) / denom))


def benign_values(result: ProbeResult) -> dict[str, float]:
    """Conservative (upper-bound) benign value of each calibratable metric in one result."""
    m = {k: v for k, v in result.metrics.items() if v is not None}
    out: dict[str, float] = {}
    if result.probe == "distribution.teacher_forced" and "kl_ci_hi" in m:
        out["kl_mean"] = m["kl_ci_hi"]
    elif result.probe == "greedy.agreement" and "trials" in m:
        out["greedy_hazard"] = wilson_upper(m["divergences"], m["trials"])
    elif result.probe == "tools.structured" and "trigger_disagreements" in m:
        out["tool_disagreement"] = wilson_upper(m["trigger_disagreements"], m["trigger_prompts"])
    return out
