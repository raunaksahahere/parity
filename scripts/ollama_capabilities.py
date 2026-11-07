"""Record what an Ollama server's OpenAI-compatible API supports (v0.2.0 adapter evidence).

  uv run python scripts/ollama_capabilities.py MODEL OUT.json [BASE_URL]

MODEL is an Ollama model name (e.g. one created from a local GGUF with `ollama create`).
Each check stores the request, the HTTP status and the relevant part of the response, so the
adapter's declared capabilities can be traced to observed behaviour.
"""

import json
import sys
from typing import Any

import httpx

model, out = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:11434"
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}
results: dict[str, Any] = {}


def post(path: str, body: dict[str, Any]) -> tuple[int, Any]:
    r = httpx.post(base + path, json=body, timeout=600)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text[:300]


def keep(name: str, status: int, data: Any, pick: Any = None) -> None:
    try:
        shown = pick(data) if pick and status < 400 else data
    except Exception as exc:  # record what was missing instead of failing the whole survey
        shown = {"unexpected_shape": str(exc), "raw": str(data)[:400]}
    results[name] = {"status": status, "result": shown}
    print(name, status, json.dumps(shown)[:200])


results["version"] = httpx.get(base + "/api/version", timeout=10).json()
greedy = {"temperature": 0.0, "max_tokens": 8, "seed": 0}
msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi?"}]

s, d = post("/v1/chat/completions", {"model": model, "messages": msgs, **greedy})
keep("chat_usage", s, d, lambda d: {"usage": d.get("usage"), "msg": d["choices"][0]["message"]})

s, d = post(
    "/v1/chat/completions",
    {"model": model, "messages": msgs, **greedy, "logprobs": True, "top_logprobs": 5},
)
keep("chat_logprobs", s, d, lambda d: d["choices"][0].get("logprobs"))

s, d = post(
    "/v1/completions",
    {"model": model, "prompt": "The capital of France is", **greedy, "logprobs": 5},
)
keep(
    "completions_text_logprobs",
    s,
    d,
    lambda d: {
        "text": d["choices"][0]["text"],
        "logprobs": d["choices"][0].get("logprobs"),
        "usage": d.get("usage"),
    },
)

s, d = post("/v1/completions", {"model": model, "prompt": [785, 6722, 315], **greedy})
keep(
    "completions_token_ids",
    s,
    d,
    lambda d: {"text": d["choices"][0]["text"], "usage": d.get("usage")},
)

s, d = post(
    "/v1/completions",
    {"model": model, "prompt": "The capital of France is", **greedy, "echo": True, "logprobs": 5},
)
keep(
    "completions_echo",
    s,
    d,
    lambda d: {"text": d["choices"][0]["text"], "logprobs": d["choices"][0].get("logprobs")},
)

trigger = [{"role": "user", "content": "What's the weather like in Tokyo right now?"}]
for choice in (None, "none", "required"):
    body: dict[str, Any] = {
        "model": model,
        "messages": trigger,
        "tools": [TOOL],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    if choice:
        body["tool_choice"] = choice
    s, d = post("/v1/chat/completions", body)
    keep(
        f"tools_{choice or 'auto'}",
        s,
        d,
        lambda d: {
            "finish": d["choices"][0].get("finish_reason"),
            "tool_calls": d["choices"][0]["message"].get("tool_calls"),
            "content": (d["choices"][0]["message"].get("content") or "")[:120],
        },
    )

s, d = post(
    "/v1/chat/completions",
    {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 80,
        "messages": [{"role": "user", "content": "Extract the person as JSON: Ann is 30."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "p", "schema": SCHEMA, "strict": True},
        },
    },
)
keep(
    "json_schema",
    s,
    d,
    lambda d: {
        "finish": d["choices"][0].get("finish_reason"),
        "content": d["choices"][0]["message"].get("content"),
    },
)

for top_k in (1, 0):
    s, d = post(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": msgs,
            "temperature": 1.0,
            "max_tokens": 4,
            "top_k": top_k,
            "top_p": 1.0,
        },
    )
    keep(f"sampling_top_k_{top_k}", s, d, lambda d: d["choices"][0]["message"].get("content"))

s, d = post("/api/show", {"model": model})
keep(
    "show",
    s,
    d,
    lambda d: {
        "template": (d.get("template") or "")[:600],
        "parameters": d.get("parameters"),
        "context_length": {
            k: v for k, v in (d.get("model_info") or {}).items() if k.endswith("context_length")
        },
    },
)
with open(out, "w") as f:
    json.dump(results, f, indent=1)
