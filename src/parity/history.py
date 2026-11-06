"""Run history in DuckDB, exportable to Parquet, with trend queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from parity.report_model import RunReport

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS runs (
        run_id VARCHAR PRIMARY KEY, started_at TIMESTAMP, finished_at TIMESTAMP,
        duration_s DOUBLE, suite VARCHAR, alpha DOUBLE, requests_total INTEGER,
        target_url VARCHAR, target_model VARCHAR, adapter VARCHAR,
        ref_model VARCHAR, ref_revision VARCHAR, verdict VARCHAR, exit_code INTEGER,
        lower_power BOOLEAN, parity_version VARCHAR)""",
    """CREATE TABLE IF NOT EXISTS probe_results (
        run_id VARCHAR, probe VARCHAR, family VARCHAR, status VARCHAR, final_status VARCHAR,
        p_value DOUBLE, adjusted_p DOUBLE, effect DOUBLE, effect_unit VARCHAR,
        threshold DOUBLE, n_requests INTEGER, power VARCHAR, hard_evidence BOOLEAN)""",
    """CREATE TABLE IF NOT EXISTS metrics (
        run_id VARCHAR, probe VARCHAR, name VARCHAR, value DOUBLE)""",
]


class History:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.con = duckdb.connect(str(path))
        for stmt in SCHEMA:
            self.con.execute(stmt)

    def close(self) -> None:
        self.con.close()

    def record(self, report: RunReport) -> None:
        """Store a run; re-recording the same run id replaces it (resumed runs)."""
        rid = report.run_id
        for table in ("probe_results", "metrics", "runs"):
            self.con.execute(f"DELETE FROM {table} WHERE run_id = ?", [rid])
        self.con.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                rid, report.started_at, report.finished_at, report.duration_s, report.suite,
                report.alpha, report.requests_total, report.target.base_url, report.target.model,
                report.target.adapter, report.reference.model_id, report.reference.revision,
                report.verdict.value, report.exit_code, report.lower_power, report.parity_version,
            ],
        )  # fmt: skip
        for o in report.probes:
            r = o.result
            self.con.execute(
                "INSERT INTO probe_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    rid, r.probe, r.family, r.status.value, o.final_status.value, r.p_value,
                    o.adjusted_p, r.effect, r.effect_unit, r.threshold, r.n_requests, r.power,
                    r.hard_evidence,
                ],
            )  # fmt: skip
            for name, value in r.metrics.items():
                if value is None:
                    continue
                self.con.execute(
                    "INSERT INTO metrics VALUES (?,?,?,?)", [rid, r.probe, name, float(value)]
                )

    def runs(self, *, model: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        q = "SELECT * FROM runs"
        args: list[Any] = []
        if model:
            q += " WHERE target_model = ?"
            args.append(model)
        q += " ORDER BY started_at DESC LIMIT ?"
        args.append(limit)
        cur = self.con.execute(q, args)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def trend(
        self, probe: str, metric: str, *, target_url: str | None = None, model: str | None = None
    ) -> list[tuple[str, float]]:
        """Time series of one probe metric, oldest first."""
        q = (
            "SELECT CAST(r.started_at AS VARCHAR), m.value FROM metrics m "
            "JOIN runs r USING (run_id) WHERE m.probe = ? AND m.name = ?"
        )
        args: list[Any] = [probe, metric]
        if target_url:
            q += " AND r.target_url = ?"
            args.append(target_url)
        if model:
            q += " AND r.target_model = ?"
            args.append(model)
        q += " ORDER BY r.started_at"
        return [(str(t), float(v)) for t, v in self.con.execute(q, args).fetchall()]

    def export_parquet(self, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for table in ("runs", "probe_results", "metrics"):
            p = out_dir / f"{table}.parquet"
            self.con.execute(f"COPY {table} TO '{p}' (FORMAT PARQUET)")
            paths.append(p)
        return paths
