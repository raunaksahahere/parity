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


LISTING = """Available devices:
  Vulkan0: AMD Radeon Graphics (RADV RENOIR) (4143 MiB, 3816 MiB free)
  Vulkan1: AMD Radeon RX 6500 (RADV NAVI24) (4080 MiB, 4064 MiB free)
"""


def test_devices_are_picked_by_name_whatever_their_index() -> None:
    import pytest

    from parity.local.llamacpp import ServerStartError, parse_devices, pick_device

    devs = parse_devices(LISTING)
    assert pick_device(devs, "RX 6500") == "Vulkan1"
    swapped = {"Vulkan0": devs["Vulkan1"], "Vulkan1": devs["Vulkan0"]}
    assert pick_device(swapped, "rx 6500") == "Vulkan0"  # follows the GPU, not the index
    assert pick_device(devs, "Vulkan0") == "Vulkan0"
    with pytest.raises(ServerStartError, match="exactly one"):
        pick_device(devs, "AMD Radeon")  # ambiguous: matches both GPUs
    with pytest.raises(ServerStartError, match="not found"):
        pick_device(devs, "Vulkan7")
