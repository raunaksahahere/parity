"""Raw greedy-divergence hazard of each local deployment vs the HF reference."""

import asyncio
import json
import sys
import time
from pathlib import Path

from parity.adapters import get_adapter
from parity.client import EndpointClient
from parity.local.llamacpp import LlamaServer, LlamaServerConfig
from parity.probes.greedy import first_divergence
from parity.reference import HFReference
from parity.suites import load_corpus

REV = "c1899de289a04d12100db370d81485cdf75e47ca"
VK, CPU = Path("bin/vulkan/llama-b12128/llama-server"), Path("bin/cpu/llama-b12128/llama-server")
DEPLOYMENTS = {
    "bf16-gpu": (VK, "BF16", "Vulkan1"),
    "bf16-cpu": (CPU, "BF16", None),
    "q8_0-gpu": (VK, "Q8_0", "Vulkan1"),
    "q4_k_m-gpu": (VK, "Q4_K_M", "Vulkan1"),
    "q2_k-gpu": (VK, "Q2_K", "Vulkan1"),
}
N_TOK, P_LEN = 48, 16


async def main(names: list[str]) -> None:
    ref = HFReference("models/hf/Qwen3-0.6B", revision=REV)
    prompts = [ids for d in load_corpus() if len(ids := ref.encode(d.text)[:P_LEN]) >= 4]
    t = time.time()
    refs = [ref.greedy(p, N_TOK) for p in prompts]
    print(f"{len(prompts)} prompts, reference greedy done in {time.time() - t:.0f}s", flush=True)
    adapter = get_adapter("llamacpp")
    out = {}
    for name in names:
        binary, quant, device = DEPLOYMENTS[name]
        cfg = LlamaServerConfig(
            binary=binary, model=Path(f"models/gguf/Qwen3-0.6B-{quant}.gguf"), device=device
        )
        with LlamaServer(cfg) as srv:
            async with EndpointClient(srv.base_url) as c:
                comps = [
                    await adapter.generate(c, "m", p, n_tokens=N_TOK, sampling={"temperature": 0.0})
                    for p in prompts
                ]
        divs = [
            first_divergence([x.token_id for x in comp.positions if x.token_id is not None], r)
            for comp, r in zip(comps, refs, strict=True)
        ]
        trials = sum(len(r) if d is None else d + 1 for d, r in zip(divs, refs, strict=True))
        events = sum(d is not None for d in divs)
        out[name] = {
            "hazard": events / trials,
            "events": events,
            "trials": trials,
            "divergence_positions": divs,
        }
        print(
            f"{name:12s} hazard {events / trials:.4f} ({events}/{trials}) "
            f"diverged prompts {events}/{len(prompts)} positions {sorted(d for d in divs if d is not None)}",
            flush=True,
        )
    Path("runs/live-greedy.json").write_text(json.dumps(out, indent=1))


asyncio.run(main(sys.argv[1:] or list(DEPLOYMENTS)))
