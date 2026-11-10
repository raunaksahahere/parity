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
from typing import Any

from pydantic import BaseModel, Field
from scipy import stats

from parity.probes.base import ProbeResult

CALIBRATION_SCHEMA = 2
# probe options that change a calibrated metric's value (None in a record means "default")
CALIBRATION_SETTINGS = (
    "template_kwargs",
    "distribution_top_k",
    "distribution_max_tokens",
    "greedy_tokens",
    "greedy_prompt_tokens",
)
METRIC_PROBES = {
    "kl_mean": "distribution.teacher_forced",
    "kl_chat_mean": "distribution.teacher_forced",
    "greedy_hazard": "greedy.agreement",
    "greedy_chat_hazard": "greedy.agreement",
    "tool_disagreement": "tools.structured",
}


class CalibrationError(ValueError):
    """A calibration file cannot be used for this run."""


class CalibrationContext(BaseModel):
    """What the calibrated thresholds depend on. A run may only use thresholds whose context
    matches its own: same weights, tokenizer and template, suite data, probe versions,
    reference implementation and metric-relevant settings."""

    model_id: str
    revision: str
    model_fingerprint: str | None = None  # None for URL references (no local weights)
    tokenizer_fingerprint: str
    reference_kind: str
    reference_dtype: str | None = None
    suite_data: str
    suite_fingerprint: str
    probe_versions: dict[str, str]
    settings: dict[str, Any] = Field(default_factory=dict)
    parity_version: str = ""


class MetricCalibration(BaseModel):
    benign_values: dict[str, float]  # deployment label -> observed upper-bound value
    threshold: float
    method: str


class Calibration(BaseModel):
    schema_version: int = CALIBRATION_SCHEMA
    model_id: str
    revision: str
    suite: str
    created: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    metrics: dict[str, MetricCalibration] = Field(default_factory=dict)
    context: CalibrationContext | None = None  # None only in schema-1 files (parity 0.1.0)
    benign_deployments: dict[str, str] = Field(default_factory=dict)  # label -> sanitized URL
    safety_factor: float | None = None

    def thresholds(self) -> dict[str, float]:
        return {k: v.threshold for k, v in self.metrics.items()}

    def fingerprint(self) -> str:
        from parity.fingerprint import digest

        return digest("calibration", self.model_dump(mode="json"))

    def mismatches(self, run: CalibrationContext, probes: list[str]) -> list[str]:
        """Human-readable reasons why these thresholds must not be used by a run with context
        ``run`` that executes ``probes`` (empty when compatible)."""
        from parity.fingerprint import short

        c = self.context
        if self.schema_version < CALIBRATION_SCHEMA:
            return [
                f"the file uses calibration schema {self.schema_version} (parity 0.1.0), which "
                "records no tokenizer, suite or probe fingerprints, so its compatibility cannot "
                "be checked"
            ]
        if c is None:
            return [
                "the calibration records no context (model, tokenizer, suite and probe "
                "fingerprints), so its compatibility cannot be checked"
            ]
        out: list[str] = []
        if c.model_fingerprint and run.model_fingerprint:
            if c.model_fingerprint != run.model_fingerprint:
                out.append(
                    f"model weights: calibrated on {short(c.model_fingerprint)} "
                    f"({c.model_id} @ {c.revision}), this run uses "
                    f"{short(run.model_fingerprint)} ({run.model_id} @ {run.revision})"
                )
        else:
            if c.model_id != run.model_id:
                out.append(f"model: calibrated on {c.model_id!r}, this run uses {run.model_id!r}")
            if c.revision != run.revision:
                out.append(
                    f"revision: calibrated at {c.revision!r}, this run uses {run.revision!r}"
                )
        if c.tokenizer_fingerprint != run.tokenizer_fingerprint:
            out.append(
                f"tokenizer or chat template: calibrated with {short(c.tokenizer_fingerprint)}, "
                f"this run uses {short(run.tokenizer_fingerprint)}"
            )
        if c.reference_kind != run.reference_kind:
            out.append(
                f"reference: calibrated against {c.reference_kind!r}, this run uses "
                f"{run.reference_kind!r}"
            )
        if c.reference_dtype != run.reference_dtype:
            out.append(
                f"reference dtype: calibrated with {c.reference_dtype!r}, this run uses "
                f"{run.reference_dtype!r}"
            )
        if c.suite_data != run.suite_data or c.suite_fingerprint != run.suite_fingerprint:
            out.append(
                f"suite data: calibrated with {c.suite_data} ({short(c.suite_fingerprint)}), "
                f"this run uses {run.suite_data} ({short(run.suite_fingerprint)})"
            )
        for pid in probes:
            if pid in c.probe_versions and c.probe_versions[pid] != run.probe_versions.get(pid):
                out.append(
                    f"probe {pid}: calibrated with version {c.probe_versions[pid]}, this run "
                    f"uses version {run.probe_versions.get(pid)}"
                )
        for metric in self.metrics:
            owner = METRIC_PROBES.get(metric)
            if owner is not None and owner in probes and owner not in c.probe_versions:
                out.append(f"metric {metric}: no version recorded for probe {owner}")
        for key in sorted(set(c.settings) | set(run.settings)):
            if c.settings.get(key) != run.settings.get(key):
                out.append(
                    f"setting {key}: calibrated with {c.settings.get(key)!r}, this run uses "
                    f"{run.settings.get(key)!r}"
                )
        return out

    def check(self, run: CalibrationContext, probes: list[str], source: str = "") -> None:
        problems = self.mismatches(run, probes)
        if problems:
            where = f" {source}" if source else ""
            raise CalibrationError(
                f"calibration{where} does not match this run, so its thresholds cannot be "
                "used:\n" + "\n".join(f"  - {p}" for p in problems) + "\nRe-run `parity "
                "calibrate` against known-good deployments of this model with this version of "
                "parity, or run without --calibration (uncalibrated defaults)."
            )

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


FLOORS = {
    "kl_mean": 1e-5,
    "kl_chat_mean": 1e-5,
    "greedy_hazard": 0.002,
    "greedy_chat_hazard": 0.002,
    "tool_disagreement": 0.05,
}


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
        out["kl_chat_mean" if m.get("chat_mode") else "kl_mean"] = m["kl_ci_hi"]
    elif result.probe == "greedy.agreement" and "trials" in m:
        key = "greedy_chat_hazard" if m.get("chat_mode") else "greedy_hazard"
        out[key] = wilson_upper(m["divergences"], m["trials"])
    elif result.probe == "tools.structured" and "trigger_disagreements" in m:
        out["tool_disagreement"] = wilson_upper(m["trigger_disagreements"], m["trigger_prompts"])
    return out
