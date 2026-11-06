"""Best-effort detection of the serving software behind an OpenAI-compatible endpoint."""

from __future__ import annotations

from parity.client import EndpointClient, EndpointError


async def detect_adapter(client: EndpointClient) -> str:
    """Returns an adapter name; falls back to the generic ``openai`` adapter."""
    try:
        props = await client.get_json("/props")
        if "default_generation_settings" in props or "build_info" in props:
            return "llamacpp"
    except (EndpointError, ValueError, OSError):
        pass
    try:
        ver = await client.get_json("/version")
        if "version" in ver:
            return "vllm"
    except (EndpointError, ValueError, OSError):
        pass
    return "openai"
