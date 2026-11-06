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
