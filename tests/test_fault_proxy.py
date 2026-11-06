from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import httpx
import pytest

from parity.faults.proxy import FaultProxy, ignore_param, scale_temperature, truncate_chat_prompt


class Echo(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        return None

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["content-length"]))
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        data = b'{"ok": true}'
        self.send_response(200)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def with_upstream() -> tuple[HTTPServer, str]:
    srv = HTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = httpx.post(url + "/v1/completions", json=body).json()
    return out


def test_ignore_param_rewrites_only_that_field() -> None:
    srv, up = with_upstream()
    with FaultProxy(up, ignore_param("top_k", 0)) as p:
        got = post(p.url, {"top_k": 5, "top_p": 0.5, "temperature": 0.7})
    srv.shutdown()
    assert got == {"top_k": 0, "top_p": 0.5, "temperature": 0.7}


def test_temperature_scaling_leaves_greedy_alone() -> None:
    srv, up = with_upstream()
    with FaultProxy(up, scale_temperature(1.5)) as p:
        assert post(p.url, {"temperature": 0.8})["temperature"] == pytest.approx(1.2)
        assert post(p.url, {"temperature": 0.0})["temperature"] == 0.0
    srv.shutdown()


def test_truncation_keeps_the_tail_of_the_last_message() -> None:
    srv, up = with_upstream()
    msgs = [{"role": "user", "content": "A" * 50 + "TAIL"}]
    with FaultProxy(up, truncate_chat_prompt(10)) as p:
        got = post(p.url, {"messages": msgs})
    srv.shutdown()
    assert got["messages"][-1]["content"] == "A" * 6 + "TAIL"


def test_get_requests_pass_through_and_are_counted() -> None:
    srv, up = with_upstream()
    with FaultProxy(up, ignore_param("x", 0)) as p:
        assert httpx.get(p.url + "/props").json() == {"ok": True}
        assert p.requests == 1
    srv.shutdown()
    assert json.dumps({}) == "{}"
