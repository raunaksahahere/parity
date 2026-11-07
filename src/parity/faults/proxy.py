"""A tiny reverse proxy that simulates serving bugs between a client and a correct server.

Two kinds of simulation, always labelled as such in the study documentation:

* **request rewriting**: the request body is changed before it reaches the server (llama.cpp
  has no switch to ignore ``top_k`` or drop tool definitions). The response is genuine.
* **response corruption**: the server's JSON response is changed before it reaches the client
  (an invalid tool name, malformed JSON). The request is genuine.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any, Literal, Self

import httpx

RequestFn = Callable[[str, dict[str, Any]], dict[str, Any]]
ResponseFn = Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class Mutation:
    label: str  # stable description; part of a scenario's fingerprint
    kind: Literal["request", "response"]
    request: RequestFn | None = None
    response: ResponseFn | None = None  # (path, request body, response body) -> response body

    def __call__(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Apply the request rewrite (identity for response corruptions)."""
        return self.request(path, body) if self.request else body


# -- request rewriting ------------------------------------------------------------------------


def ignore_param(name: str, neutral: float) -> Mutation:
    """The server accepts but ignores ``name`` (always behaves as if it were ``neutral``)."""

    def mutate(path: str, body: dict[str, Any]) -> dict[str, Any]:
        if name in body:
            body[name] = neutral
        return body

    return Mutation(f"ignore_param({name}={neutral})", "request", request=mutate)


def scale_temperature(factor: float) -> Mutation:
    def mutate(path: str, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("temperature", 0) > 0:
            body["temperature"] = body["temperature"] * factor
        return body

    return Mutation(f"scale_temperature({factor})", "request", request=mutate)


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

    return Mutation(f"truncate_chat_prompt({max_chars})", "request", request=mutate)


def drop_tools() -> Mutation:
    """The server drops tool definitions (and with them any tool_choice)."""

    def mutate(path: str, body: dict[str, Any]) -> dict[str, Any]:
        body.pop("tools", None)
        body.pop("tool_choice", None)
        return body

    return Mutation("drop_tools()", "request", request=mutate)


def ignore_tool_choice() -> Mutation:
    """The server ignores tool_choice and always behaves as "auto"."""

    def mutate(path: str, body: dict[str, Any]) -> dict[str, Any]:
        body.pop("tool_choice", None)
        return body

    return Mutation("ignore_tool_choice()", "request", request=mutate)


# -- response corruption ----------------------------------------------------------------------


def _tool_calls(resp: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for choice in resp.get("choices") or []:
        out += (choice.get("message") or {}).get("tool_calls") or []
    return out


def rename_tool_calls(suffix: str = "_v2") -> Mutation:
    """Tool calls come back naming a tool that was never provided."""

    def mutate(path: str, req: dict[str, Any], resp: dict[str, Any]) -> dict[str, Any]:
        for tc in _tool_calls(resp):
            fn = tc.get("function") or {}
            if "name" in fn:
                fn["name"] = f"{fn['name']}{suffix}"
        return resp

    return Mutation(f"rename_tool_calls({suffix})", "response", response=mutate)


def malform_tool_arguments() -> Mutation:
    """Tool-call arguments come back as malformed JSON (the closing brace is lost)."""

    def mutate(path: str, req: dict[str, Any], resp: dict[str, Any]) -> dict[str, Any]:
        for tc in _tool_calls(resp):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str) and args.rstrip().endswith("}"):
                fn["arguments"] = args.rstrip()[:-1]
        return resp

    return Mutation("malform_tool_arguments()", "response", response=mutate)


def break_argument_schema() -> Mutation:
    """Tool-call arguments stay valid JSON but no longer match the declared schema: every key
    is renamed, so required properties are missing and unknown ones appear."""

    def mutate(path: str, req: dict[str, Any], resp: dict[str, Any]) -> dict[str, Any]:
        for tc in _tool_calls(resp):
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "")
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(args, dict) and args:
                fn["arguments"] = json.dumps({f"{k}_value": v for k, v in args.items()})
        return resp

    return Mutation("break_argument_schema()", "response", response=mutate)


def malform_json_content() -> Mutation:
    """Structured-output (response_format) replies come back as malformed JSON."""

    def mutate(path: str, req: dict[str, Any], resp: dict[str, Any]) -> dict[str, Any]:
        if "response_format" not in req:
            return resp
        for choice in resp.get("choices") or []:
            msg = choice.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.rstrip().endswith("}"):
                msg["content"] = content.rstrip()[:-1]
        return resp

    return Mutation("malform_json_content()", "response", response=mutate)


class FaultProxy:
    def __init__(self, upstream: str, mutation: Mutation) -> None:
        self.upstream = upstream.rstrip("/")
        self.mutation = mutation
        self.requests = 0
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def _forward(self, method: str) -> None:
                proxy.requests += 1
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length) if length else b""
                body: dict[str, Any] = {}
                if method == "POST" and raw:
                    with contextlib.suppress(json.JSONDecodeError):
                        body = json.loads(raw)
                        if proxy.mutation.request is not None:
                            body = proxy.mutation.request(self.path, body)
                            raw = json.dumps(body).encode()
                with httpx.Client(timeout=600) as client:
                    resp = client.request(
                        method,
                        proxy.upstream + self.path,
                        content=raw,
                        headers={"content-type": "application/json"},
                    )
                content = resp.content
                if proxy.mutation.response is not None and method == "POST" and resp.is_success:
                    with contextlib.suppress(json.JSONDecodeError, ValueError):
                        data = proxy.mutation.response(self.path, body, resp.json())
                        content = json.dumps(data).encode()
                self.send_response(resp.status_code)
                self.send_header(
                    "content-type", resp.headers.get("content-type", "application/json")
                )
                self.send_header("content-length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

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
