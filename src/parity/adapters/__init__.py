from parity.adapters.base import Adapter, Capabilities
from parity.adapters.llamacpp import LlamaCppAdapter
from parity.adapters.ollama import OllamaAdapter
from parity.adapters.openai import OpenAIAdapter
from parity.adapters.vllm import VLLMAdapter

ADAPTERS: dict[str, type[Adapter]] = {
    "llamacpp": LlamaCppAdapter,
    "vllm": VLLMAdapter,
    "ollama": OllamaAdapter,
    "openai": OpenAIAdapter,
}


def get_adapter(name: str) -> Adapter:
    try:
        return ADAPTERS[name]()
    except KeyError:
        raise ValueError(f"unknown adapter {name!r}; choose from {sorted(ADAPTERS)}") from None


__all__ = ["ADAPTERS", "Adapter", "Capabilities", "get_adapter"]
