"""Prometheus exposition for scheduled canary runs.

Implemented without ``prometheus_client``: the text format is small, and the exporter only
ever serves the latest finished run.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from parity.probes.base import Status
from parity.report_model import RunReport

VERDICT_VALUE = {Status.PASS: 0, Status.DEVIATION: 1, Status.INCONCLUSIVE: 2, Status.SKIPPED: 3}


def _esc(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(**kv: str) -> str:
    return "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in kv.items()) + "}"


def render_metrics(report: RunReport | None, runs_total: int, errors_total: int) -> str:
    """Prometheus text exposition (version 0.0.4) of the latest report."""
    lines = [
        "# HELP parity_runs_total Canary runs completed since the exporter started.",
        "# TYPE parity_runs_total counter",
        f"parity_runs_total {runs_total}",
        "# HELP parity_run_errors_total Canary runs that failed with an error.",
        "# TYPE parity_run_errors_total counter",
        f"parity_run_errors_total {errors_total}",
    ]
    if report is None:
        return "\n".join(lines) + "\n"
    base = {"model": report.target.model, "target": report.target.base_url}
    lines += [
        "# HELP parity_verdict Latest overall verdict: 0 pass, 1 deviation, 2 inconclusive.",
        "# TYPE parity_verdict gauge",
        f"parity_verdict{_labels(**base)} {VERDICT_VALUE[report.verdict]}",
        "# HELP parity_lower_power 1 if core probes could not run on this endpoint.",
        "# TYPE parity_lower_power gauge",
        f"parity_lower_power{_labels(**base)} {int(report.lower_power)}",
        "# HELP parity_requests Requests used by the latest run.",
        "# TYPE parity_requests gauge",
        f"parity_requests{_labels(**base)} {report.requests_total}",
        "# HELP parity_run_duration_seconds Duration of the latest run.",
        "# TYPE parity_run_duration_seconds gauge",
        f"parity_run_duration_seconds{_labels(**base)} {report.duration_s}",
        "# HELP parity_probe_status Probe verdict: 0 pass, 1 dev, 2 inconclusive, 3 skipped.",
        "# TYPE parity_probe_status gauge",
    ]
    for o in report.probes:
        lines.append(
            f"parity_probe_status{_labels(**base, probe=o.result.probe)} "
            f"{VERDICT_VALUE[o.final_status]}"
        )
    lines += [
        "# HELP parity_probe_effect Primary effect size of a probe (units differ).",
        "# TYPE parity_probe_effect gauge",
    ]
    for o in report.probes:
        if o.result.effect is not None:
            lines.append(
                f"parity_probe_effect{_labels(**base, probe=o.result.probe)} {o.result.effect}"
            )
    return "\n".join(lines) + "\n"


class CanaryState:
    def __init__(self) -> None:
        self.report: RunReport | None = None
        self.runs = 0
        self.errors = 0
        self.lock = threading.Lock()

    def update(self, report: RunReport | None, error: bool = False) -> None:
        with self.lock:
            if error:
                self.errors += 1
            else:
                self.report = report
                self.runs += 1

    def text(self) -> str:
        with self.lock:
            return render_metrics(self.report, self.runs, self.errors)


def serve(state: CanaryState, port: int, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return None

        def do_GET(self) -> None:
            if self.path.rstrip("/") not in ("/metrics", ""):
                self.send_error(404)
                return
            body = state.text().encode()
            self.send_response(200)
            self.send_header("content-type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
