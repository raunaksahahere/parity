"""Stable report schema. Changing field names or meanings requires bumping SCHEMA_VERSION."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from parity.probes.base import ProbeResult, Status

SCHEMA_VERSION = 1

LIMITS = [
    "A PASS does not prove the endpoint serves exactly the reference weights: it shows no "
    "difference was detected by these probes at this sample size. Differences smaller than the "
    "detection threshold, or in behaviour the probes do not exercise, remain possible.",
    "A DEVIATION says the observed behaviour differs from the reference beyond the calibrated "
    "benign variation. It does not say why (quantization, a different checkpoint, a serving bug, "
    "a changed template) and it does not establish intent or misrepresentation by a provider.",
    "KL is estimated from truncated top-k log-probs and is a lower bound on the true divergence.",
    "Thresholds come from the benign deployment variants that were calibrated; a legitimate "
    "deployment that varies in other ways (different hardware, kernels, batching) may differ more.",
    "The reference is an implementation of the same open weights, not ground truth: numerics "
    "differ slightly between any two correct implementations.",
    "Probes that need token-id prompts or log-probs are skipped on endpoints without them; "
    "the remaining behavioural probes have lower power.",
]


class TargetInfo(BaseModel):
    base_url: str  # credentials and query strings removed
    model: str
    adapter: str
    capabilities: dict[str, Any] = Field(default_factory=dict)


class ReferenceInfo(BaseModel):
    kind: str  # "hf:local" or "url"
    model_id: str
    revision: str
    dtype: str | None = None
    model_fingerprint: str | None = None  # sha256 over config and weight files (hf:local)
    tokenizer_fingerprint: str | None = None  # tokenizer files and observed behaviour


class ProbeOutcome(BaseModel):
    result: ProbeResult
    adjusted_p: float | None = None  # Holm-adjusted across probes
    final_status: Status
    reason: str = ""


class CalibrationInfo(BaseModel):
    path: str
    created: str
    thresholds: dict[str, float]
    fingerprint: str | None = None  # sha256 of the calibration's content


class Coverage(BaseModel):
    """Which probes produced evidence. A verdict only speaks for the probes it evaluated."""

    total: int
    evaluated: list[str]  # ran and returned PASS, DEVIATION or INCONCLUSIVE
    decided: list[str]  # returned PASS or DEVIATION
    skipped: dict[str, str] = Field(default_factory=dict)  # probe -> reason
    required: list[str] = Field(default_factory=list)  # probes the caller required
    unmet_required: list[str] = Field(default_factory=list)  # required but not decided


class RunReport(BaseModel):
    schema_version: int = SCHEMA_VERSION
    parity_version: str
    run_id: str
    started_at: str
    finished_at: str
    duration_s: float
    suite: str
    alpha: float
    budget: int | None = None
    requests_total: int
    # every request sent to the target by this run, in all sessions of a resumed run, including
    # retries and probes that were interrupted (None in reports written by parity 0.1.0)
    requests_consumed: int | None = None
    resumed: bool = False
    run_fingerprint: str | None = None  # sha256 of the run manifest's identity
    coverage: Coverage | None = None
    target: TargetInfo
    reference: ReferenceInfo
    calibration: CalibrationInfo | None = None
    probes: list[ProbeOutcome]
    verdict: Status
    exit_code: int
    lower_power: bool
    summary: str
    limits: list[str] = Field(default_factory=lambda: list(LIMITS))
