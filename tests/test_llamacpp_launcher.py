from __future__ import annotations

from pathlib import Path

from parity.faults.scenarios import SCENARIOS, Paths
from parity.local.llamacpp import LlamaServer, LlamaServerConfig


def command(cfg: LlamaServerConfig) -> list[str]:
    s = LlamaServer(cfg)
    s.port = 1234
    return s._command()


def test_gpu_and_cpu_flags() -> None:
    gpu = command(LlamaServerConfig(binary=Path("b"), model=Path("m"), device="Vulkan1"))
    assert gpu[gpu.index("--device") + 1] == "Vulkan1" and "-ngl" in gpu
    cpu = command(LlamaServerConfig(binary=Path("b"), model=Path("m")))
    assert cpu[cpu.index("--device") + 1] == "none" and cpu[cpu.index("-ngl") + 1] == "0"


def test_prompt_cache_flag_only_when_set() -> None:
    off = command(LlamaServerConfig(binary=Path("b"), model=Path("m"), cache_ram_mib=0))
    assert off[off.index("--cache-ram") + 1] == "0"
    assert "--cache-ram" not in command(LlamaServerConfig(binary=Path("b"), model=Path("m")))


def test_study_servers_disable_the_host_prompt_cache() -> None:
    for sc in SCENARIOS:
        assert "--cache-ram" in command(sc.server.config(Paths(), None)), sc.name
