"""Measure benign log-prob variation of local llama.cpp deployments vs the HF fp32 reference.

Writes a JSON results file; docs/baseline-noise.md is generated from it.
Usage: uv run python scripts/baseline_noise.py [--max-tokens 64] [--out runs/baseline-noise.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from parity.adapters import get_adapter
from parity.client import EndpointClient
from parity.local.llamacpp import LlamaServer, LlamaServerConfig
from parity.reference import HFReference
from parity.suites import load_corpus

REV = "c1899de289a04d12100db370d81485cdf75e47ca"
VK = Path("bin/vulkan/llama-b12128/llama-server")
CPU = Path("bin/cpu/llama-b12128/llama-server")
GG = Path("models/gguf")
TOP_K = 100

CONFIGS: dict[str, dict[str, Any]] = {
    "bf16-gpu-default": dict(binary=VK, model="BF16", device="Vulkan1"),
    "bf16-gpu-default-repeat": dict(binary=VK, model="BF16", device="Vulkan1"),
    "bf16-gpu-small-batch": dict(
        binary=VK, model="BF16", device="Vulkan1", batch_size=64, ubatch_size=32
    ),
    "bf16-gpu-parallel4": dict(
        binary=VK, model="BF16", device="Vulkan1", parallel=4, concurrency=4
    ),
    "bf16-cpu": dict(binary=CPU, model="BF16", device=None),
    "q8_0-gpu": dict(binary=VK, model="Q8_0", device="Vulkan1"),
}


async def run_config(name: str, spec: dict[str, Any], docs_ids: list[list[int]]) -> list[Any]:
    cfg = LlamaServerConfig(
        binary=spec["binary"],
        model=GG / f"Qwen3-0.6B-{spec['model']}.gguf",
        device=spec["device"],
        batch_size=spec.get("batch_size"),
        ubatch_size=spec.get("ubatch_size"),
        parallel=spec.get("parallel", 1),
        ctx_size=4096 * spec.get("parallel", 1),
    )
    adapter = get_adapter("llamacpp")
    with LlamaServer(cfg) as srv:
        async with EndpointClient(srv.base_url, max_concurrency=spec.get("concurrency", 1)) as c:
            t = time.time()
            res = await asyncio.gather(
                *[adapter.teacher_force(c, "m", ids, top_k=TOP_K) for ids in docs_ids]
            )
            print(f"{name}: {time.time() - t:.0f}s, {c.requests_sent} requests", flush=True)
            return list(res)


def lp_vector(positions: list[Any]) -> np.ndarray:
    return np.array([np.nan if p.logprob is None else p.logprob for p in positions])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("runs/baseline-noise.json"))
    args = ap.parse_args()

    ref = HFReference("models/hf/Qwen3-0.6B", revision=REV)
    docs = [ref.encode(d.text)[: args.max_tokens] for d in load_corpus()]
    ref_lp = [ref.teacher_force(ids, TOP_K)["lp"] for ids in docs]
    n_pos = sum(len(x) for x in ref_lp)
    print(f"{len(docs)} docs, {n_pos} teacher-forced positions", flush=True)

    got: dict[str, list[np.ndarray]] = {}
    for name, spec in CONFIGS.items():
        got[name] = [lp_vector(p) for p in asyncio.run(run_config(name, spec, docs))]

    def stats(a: list[np.ndarray], b: list[np.ndarray]) -> dict[str, float]:
        d = np.concatenate([x - y for x, y in zip(a, b, strict=True)])
        known = d[~np.isnan(d)]
        ad = np.abs(known)
        return dict(
            n=int(known.size),
            n_missing=int(np.isnan(d).sum()),
            mean_delta=float(known.mean()),
            mean_abs=float(ad.mean()),
            p99_abs=float(np.quantile(ad, 0.99)),
            max_abs=float(ad.max()),
        )

    out: dict[str, Any] = {"positions": n_pos, "top_k": TOP_K, "vs_reference": {}, "vs_config": {}}
    for name, v in got.items():
        out["vs_reference"][name] = stats(v, ref_lp)
    base = got["bf16-gpu-default"]
    for name, v in got.items():
        if name != "bf16-gpu-default":
            out["vs_config"][name] = stats(v, base)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
