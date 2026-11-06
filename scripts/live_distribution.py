"""Run the distribution probe on every local deployment and print raw effect sizes."""

import asyncio
import json
import sys
import time
from pathlib import Path

from parity.adapters import get_adapter
from parity.client import EndpointClient
from parity.local.llamacpp import LlamaServer, LlamaServerConfig
from parity.probes.base import ProbeContext
from parity.probes.distribution import DistributionProbe
from parity.reference import HFReference

REV = "c1899de289a04d12100db370d81485cdf75e47ca"
VK, CPU = Path("bin/vulkan/llama-b12128/llama-server"), Path("bin/cpu/llama-b12128/llama-server")
DEPLOYMENTS = {
    "bf16-gpu": (VK, "BF16", "Vulkan1"),
    "bf16-cpu": (CPU, "BF16", None),
    "q8_0-gpu": (VK, "Q8_0", "Vulkan1"),
    "q4_k_m-gpu": (VK, "Q4_K_M", "Vulkan1"),
    "q2_k-gpu": (VK, "Q2_K", "Vulkan1"),
}


async def main(names: list[str]) -> None:
    ref = HFReference("models/hf/Qwen3-0.6B", revision=REV)
    out = {}
    for name in names:
        binary, quant, device = DEPLOYMENTS[name]
        cfg = LlamaServerConfig(
            binary=binary, model=Path(f"models/gguf/Qwen3-0.6B-{quant}.gguf"), device=device
        )
        with LlamaServer(cfg) as srv:
            async with EndpointClient(srv.base_url, max_concurrency=1) as c:
                ctx = ProbeContext(
                    client=c,
                    adapter=get_adapter("llamacpp"),
                    model="m",
                    kit=ref.kit,
                    reference=ref,
                    options={"thresholds": {"kl_mean": 1e9}, "distribution_chunk": 25},
                )
                t = time.time()
                r = await DistributionProbe().run(ctx)
        m = r.metrics
        out[name] = {**m, "seconds": time.time() - t, "requests": r.n_requests}
        print(
            f"{name:12s} KL>= {m['kl_lower_bound_mean']:.5f} "
            f"[{m['kl_ci_lo']:.5f},{m['kl_ci_hi']:.5f}] "
            f"top1 {m['top1_agreement']:.3f} mean|d| {m['mean_abs_delta']:.4f} "
            f"deltaKnown {m['delta_known_fraction']:.3f} refMass {m['mean_ref_mass_in_bound']:.3f} "
            f"{time.time() - t:.0f}s",
            flush=True,
        )
    Path("runs").mkdir(exist_ok=True)
    Path("runs/live-distribution.json").write_text(json.dumps(out, indent=1))


asyncio.run(main(sys.argv[1:] or list(DEPLOYMENTS)))
