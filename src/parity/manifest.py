"""Run manifests: the configuration and fingerprints a run's results depend on.

A run directory holds ``manifest.json`` next to ``probes/<id>.json``. The manifest's
``identity`` must be identical for a run to be resumed; every saved probe result is recorded
with the SHA-256 of its file, so results from another run, an older version of a probe, or an
edited file are rejected instead of silently entering a verdict. ``requests_consumed`` is a
ledger of every request sent to the target in all sessions of the run (retries and probes
that were interrupted included); a resumed run charges it against its budget.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from parity.fingerprint import short

MANIFEST_VERSION = 1


class ResumeError(RuntimeError):
    """Saved results cannot be reused by this run."""


class ProbeCheckpoint(BaseModel):
    version: str
    status: str
    sha256: str  # of the saved probes/<id>.json file
    requests_consumed: int  # requests sent while this probe ran (retries included)


class RunManifest(BaseModel):
    manifest_version: int = MANIFEST_VERSION
    run_id: str
    created: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    identity: dict[str, Any]  # must match exactly to resume
    operational: dict[str, Any] = Field(default_factory=dict)  # recorded, not enforced
    sessions: int = 1
    requests_consumed: int = 0
    checkpoints: dict[str, ProbeCheckpoint] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        from parity.fingerprint import digest

        return digest("run-identity", self.identity)

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=1))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        return cls.model_validate_json(path.read_text())


def canonical(value: Any) -> Any:
    """JSON round trip, so tuples/paths compare equal to what a saved manifest holds."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _show(v: Any) -> str:
    if isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v):
        return short(v)
    text = json.dumps(v, sort_keys=True, default=str)
    return text if len(text) <= 80 else text[:77] + "..."


def identity_diff(saved: dict[str, Any], now: dict[str, Any]) -> list[str]:
    a, b = flatten(canonical(saved)), flatten(canonical(now))
    return [
        f"{k}: saved run has {_show(a.get(k, '<absent>'))}, this run has "
        f"{_show(b.get(k, '<absent>'))}"
        for k in sorted(set(a) | set(b))
        if a.get(k, "<absent>") != b.get(k, "<absent>")
    ]


def file_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def verify_resume(
    run_dir: Path, identity: dict[str, Any], probe_versions: dict[str, str]
) -> RunManifest:
    """Load and check a run directory for resumption; raise ResumeError with every reason it
    cannot be reused. Returns the saved manifest (with checkpoints whose files are missing
    dropped: those probes simply run again)."""
    path = run_dir / "manifest.json"
    if not run_dir.exists():
        raise ResumeError(f"nothing to resume: {run_dir} does not exist")
    if not path.exists():
        raise ResumeError(
            f"{run_dir} has no manifest.json (it was written by parity 0.1.0 or was not "
            "started by `parity run`), so its saved results cannot be verified; start a new run"
        )
    try:
        saved = RunManifest.load(path)
    except Exception as exc:
        raise ResumeError(f"cannot read {path}: {exc}") from exc
    problems = identity_diff(saved.identity, identity)
    probes_dir = run_dir / "probes"
    for f in sorted(probes_dir.glob("*.json")) if probes_dir.exists() else []:
        pid = f.stem
        cp = saved.checkpoints.get(pid)
        if cp is None:
            problems.append(
                f"probes/{f.name} is not recorded in the manifest (a stale or copied result)"
            )
            continue
        text = f.read_text()
        if file_digest(text) != cp.sha256:
            problems.append(f"probes/{f.name} was modified after it was saved (checksum differs)")
            continue
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            problems.append(f"probes/{f.name} is not valid JSON")
            continue
        if result.get("probe") != pid:
            problems.append(f"probes/{f.name} holds a result for {result.get('probe')!r}")
        if pid in probe_versions and result.get("version") != probe_versions[pid]:
            problems.append(
                f"probes/{f.name} was produced by version {result.get('version')} of the probe, "
                f"this run uses version {probe_versions[pid]}"
            )
    if problems:
        raise ResumeError(
            f"cannot resume {run_dir.name}: the saved results do not match this run:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\nStart a new run (omit --resume), or resume with the original configuration."
        )
    saved.checkpoints = {
        pid: cp for pid, cp in saved.checkpoints.items() if (probes_dir / f"{pid}.json").exists()
    }
    return saved
