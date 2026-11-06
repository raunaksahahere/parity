"""The validation study's deployments: benign (calibration), controls and injected faults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from parity.faults.proxy import (
    Mutation,
    ignore_param,
    scale_temperature,
    truncate_chat_prompt,
)
from parity.local.llamacpp import LlamaServerConfig

GPU_DEVICE = "Vulkan1"  # RX 6500M; Vulkan0 is the integrated GPU and is never used
CLAIMED_CONTEXT = 4096  # context the deployments are supposed to offer per request


@dataclass(frozen=True)
class Paths:
    bin_dir: Path = Path("bin")
    models_dir: Path = Path("models/gguf")
    templates_dir: Path = Path("faults/templates")
    llama_build: str = "llama-b12128"

    def binary(self, gpu: bool) -> Path:
        return self.bin_dir / ("vulkan" if gpu else "cpu") / self.llama_build / "llama-server"

    def model(self, quant: str) -> Path:
        return self.models_dir / f"Qwen3-0.6B-{quant}.gguf"


@dataclass(frozen=True)
class ServerSpec:
    quant: str = "BF16"
    gpu: bool = True
    ctx_size: int = 16384
    parallel: int = 4
    batch_size: int | None = None
    ubatch_size: int | None = None
    threads: int | None = None
    template_file: str | None = None
    extra_args: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return repr(self)

    def config(self, paths: Paths, log_path: Path | None) -> LlamaServerConfig:
        return LlamaServerConfig(
            binary=paths.binary(self.gpu),
            model=paths.model(self.quant),
            ctx_size=self.ctx_size,
            device=GPU_DEVICE if self.gpu else None,
            parallel=self.parallel,
            batch_size=self.batch_size,
            ubatch_size=self.ubatch_size,
            threads=self.threads,
            chat_template_file=paths.templates_dir / self.template_file
            if self.template_file
            else None,
            extra_args=self.extra_args,
            log_path=log_path,
        )


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str  # benign | control | fault
    description: str
    server: ServerSpec
    mutation: Mutation | None = None
    reps: int = 3
    expects: str = "DEVIATION"  # what a correct tool should report


BASE = ServerSpec()

SCENARIOS: tuple[Scenario, ...] = (
    # benign variants: used only to calibrate thresholds
    Scenario("benign-gpu-p4", "benign", "BF16, GPU, 4 slots", BASE, reps=1, expects="PASS"),
    Scenario(
        "benign-cpu-p4",
        "benign",
        "BF16, CPU, 4 slots",
        ServerSpec(gpu=False),
        reps=1,
        expects="PASS",
    ),
    # held-out controls: correct deployments configured differently from the benign ones
    Scenario(
        "control-gpu-p2",
        "control",
        "BF16, GPU, 2 slots, ctx 8192",
        ServerSpec(ctx_size=8192, parallel=2),
        reps=10,
        expects="PASS",
    ),
    Scenario(
        "control-gpu-p1-smallbatch",
        "control",
        "BF16, GPU, 1 slot, -b 128 -ub 64",
        ServerSpec(ctx_size=4096, parallel=1, batch_size=128, ubatch_size=64),
        reps=10,
        expects="PASS",
    ),
    Scenario(
        "control-cpu-p1-t4",
        "control",
        "BF16, CPU, 1 slot, 4 threads",
        ServerSpec(gpu=False, ctx_size=4096, parallel=1, threads=4),
        reps=3,
        expects="PASS",
    ),
    # injected faults
    Scenario(
        "fault-template-llama3",
        "fault",
        "wrong chat template (llama3 built-in)",
        ServerSpec(extra_args=("--chat-template", "llama3")),
    ),
    Scenario(
        "fault-template-nosystem",
        "fault",
        "chat template without a system role",
        ServerSpec(template_file="chatml-no-system.jinja"),
    ),
    Scenario("fault-quant-q8_0", "fault", "Q8_0 quantization", ServerSpec(quant="Q8_0")),
    Scenario("fault-quant-q4_k_m", "fault", "Q4_K_M quantization", ServerSpec(quant="Q4_K_M")),
    Scenario("fault-quant-q2_k", "fault", "Q2_K quantization", ServerSpec(quant="Q2_K")),
    Scenario(
        "fault-topk-ignored", "fault", "top_k ignored (proxy)", BASE, ignore_param("top_k", 0)
    ),
    Scenario(
        "fault-topp-ignored", "fault", "top_p ignored (proxy)", BASE, ignore_param("top_p", 1.0)
    ),
    Scenario(
        "fault-temperature-ignored",
        "fault",
        "temperature ignored, always 1.0 (proxy)",
        BASE,
        ignore_param("temperature", 1.0),
    ),
    Scenario(
        "fault-temperature-x1.5",
        "fault",
        "temperature scaled by 1.5 (proxy)",
        BASE,
        scale_temperature(1.5),
    ),
    Scenario(
        "fault-ctx-reduced",
        "fault",
        "context reduced to 2048 (--ctx-size) but 4096 claimed",
        ServerSpec(ctx_size=2048, parallel=1),
    ),
    Scenario(
        "fault-silent-truncation",
        "fault",
        "long prompts silently truncated (proxy)",
        BASE,
        truncate_chat_prompt(6000),
    ),
)


def by_name(names: list[str] | None) -> list[Scenario]:
    if not names:
        return list(SCENARIOS)
    index = {s.name: s for s in SCENARIOS}
    missing = [n for n in names if n not in index]
    if missing:
        raise ValueError(f"unknown scenarios {missing}; available: {sorted(index)}")
    return [index[n] for n in names]
