"""A tiny reverse proxy that mutates request bodies to simulate serving bugs.

llama.cpp has no switch to ignore ``top_k`` or to scale the temperature, so those faults
are simulated by rewriting the request before it reaches an otherwise correct server.
Responses pass through untouched (so reported log-probs and usage stay genuine).
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any, Self

import httpx

Mutation = Callable[[str, dict[str, Any]], dict[str, Any]]


def ignore_param(name: str, neutral: float) -> Mutation:
    """The server accepts but ignores ``name`` (always behaves as if it were ``neutral``)."""

    def mutate(path: str, body: dict[str, Any]) -> dict[str, Any]:
        if name in body:
            body[name] = neutral
        return body

    return mutate


def scale_temperature(factor: float) -> Mutation:
    def mutate(path: str, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("temperature", 0) > 0:
            body["temperature"] = body["temperature"] * factor
        return body

    return mutate


def truncate_chat_prompt(max_chars: int) -> Mutation:
    """Silently keep only the last ``max_chars`` characters of the final user message."""

    def mutate(path: str, body: dict[str, Any]) -> dict[str, Any]:
        msgs = body.get("messages")
        if (
            msgs
            and isinstance(msgs[-1].get("content"), str)
            and len(msgs[-1]["content"]) > max_chars
        ):
            msgs[-1]["content"] = msgs[-1]["content"][-max_chars:]
        return body

    return mutate


class FaultProxy:
    def __init__(self, upstream: str, mutate: Mutation) -> None:
        self.upstream = upstream.rstrip("/")
        self.mutate = mutate
        self.requests = 0
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def _forward(self, method: str) -> None:
                proxy.requests += 1
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length) if length else b""
                if method == "POST" and raw:
                    with contextlib.suppress(json.JSONDecodeError):
                        raw = json.dumps(proxy.mutate(self.path, json.loads(raw))).encode()
                with httpx.Client(timeout=600) as client:
                    resp = client.request(
                        method,
                        proxy.upstream + self.path,
                        content=raw,
                        headers={"content-type": "application/json"},
                    )
                self.send_response(resp.status_code)
                self.send_header(
                    "content-type", resp.headers.get("content-type", "application/json")
                )
                self.send_header("content-length", str(len(resp.content)))
                self.end_headers()
                self.wfile.write(resp.content)

            def do_POST(self) -> None:
                self._forward("POST")

            def do_GET(self) -> None:
                self._forward("GET")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> Self:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
