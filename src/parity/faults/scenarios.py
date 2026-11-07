"""The validation study's deployments: benign (calibration), controls and injected faults."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from parity.faults.proxy import (
    Mutation,
    break_argument_schema,
    drop_tools,
    ignore_param,
    ignore_tool_choice,
    malform_json_content,
    malform_tool_arguments,
    rename_tool_calls,
    scale_temperature,
    truncate_chat_prompt,
)
from parity.local.llamacpp import LlamaServerConfig

# selected by name, not index: Vulkan enumeration order (iGPU first here) is not guaranteed
GPU_DEVICE = "RX 6500"
CLAIMED_CONTEXT = 4096  # context the deployments are supposed to offer per request


@dataclass(frozen=True)
class Paths:
    bin_dir: Path = Path("bin")
    models_dir: Path = Path("models/gguf")
    templates_dir: Path = Path("faults/templates")
    llama_build: str = "llama-b12128"
    model_pattern: str = "Qwen3-0.6B-{quant}.gguf"

    def binary(self, gpu: bool) -> Path:
        return self.bin_dir / ("vulkan" if gpu else "cpu") / self.llama_build / "llama-server"

    def model(self, quant: str) -> Path:
        return self.models_dir / self.model_pattern.format(quant=quant)


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
    # 0 disables llama.cpp's host-RAM prompt cache: with its 8 GiB default the study drove this
    # 7 GB machine into swap (see docs/decisions.md D12)
    cache_ram_mib: int | None = 0

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
            cache_ram_mib=self.cache_ram_mib,
            log_path=log_path,
        )


@dataclass(frozen=True)
class Scenario:
    name: str
    # benign: calibration only · dev: used while developing fixes, never evaluated ·
    # control: correct deployment (evaluation) · fault: injected fault (evaluation)
    kind: str
    description: str
    server: ServerSpec
    mutation: Mutation | None = None
    reps: int = 3
    expects: str = "DEVIATION"  # what a correct tool should report
    # True for configurations introduced for the v0.1.1 evaluation and never used while the
    # v0.1.1 fixes were developed (fresh held-out data)
    fresh: bool = False

    @property
    def backend(self) -> str:
        return "gpu" if self.server.gpu else "cpu"

    @property
    def simulation(self) -> str:
        """real (a real server setting), request (request rewriting) or response (response
        corruption)."""
        return "real" if self.mutation is None else self.mutation.kind

    @property
    def key(self) -> str:
        """Stable identity: server spec plus the mutation's label (closures have no stable
        repr), used to fingerprint results."""
        mut = self.mutation.label if self.mutation else None
        return repr((self.kind, self.server, mut, self.expects))


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
        reps=8,
        expects="PASS",
    ),
    Scenario(
        "control-gpu-p1-smallbatch",
        "control",
        "BF16, GPU, 1 slot, -b 128 -ub 64",
        ServerSpec(ctx_size=4096, parallel=1, batch_size=128, ubatch_size=64),
        reps=8,
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


HELD_OUT: tuple[Scenario, ...] = (
    # fresh correct deployments for the v0.1.1 evaluation (never used during development)
    Scenario(
        "heldout-gpu-fa-off",
        "control",
        "BF16, GPU, 4 slots, flash attention off",
        ServerSpec(extra_args=("--flash-attn", "off")),
        reps=6,
        expects="PASS",
        fresh=True,
    ),
    Scenario(
        "heldout-gpu-p3-b1024",
        "control",
        "BF16, GPU, 3 slots, ctx 12288, -b 1024 -ub 256",
        ServerSpec(ctx_size=12288, parallel=3, batch_size=1024, ubatch_size=256),
        reps=6,
        expects="PASS",
        fresh=True,
    ),
    Scenario(
        "heldout-gpu-p1-nocb",
        "control",
        "BF16, GPU, 1 slot, continuous batching off",
        ServerSpec(ctx_size=4096, parallel=1, extra_args=("--no-cont-batching",)),
        reps=5,
        expects="PASS",
        fresh=True,
    ),
    Scenario(
        "heldout-cpu-p2-t8",
        "control",
        "BF16, CPU, 2 slots, 8 threads, ctx 8192",
        ServerSpec(gpu=False, ctx_size=8192, parallel=2, threads=8),
        reps=3,
        expects="PASS",
        fresh=True,
    ),
)

# The development deployment used while building and tuning the v0.1.1 tool probe. It is never
# part of an evaluation table.
DEV: tuple[Scenario, ...] = (
    Scenario(
        "dev-gpu-p4-b256",
        "dev",
        "BF16, GPU, 4 slots, -b 256 (development only)",
        ServerSpec(batch_size=256),
        reps=2,
        expects="PASS",
    ),
)

TOOL_FAULTS: tuple[Scenario, ...] = (
    Scenario(
        "fault-tools-dropped", "fault", "tool definitions dropped (proxy)", BASE, drop_tools()
    ),
    Scenario(
        "fault-tool-choice-ignored",
        "fault",
        "tool_choice ignored, always auto (proxy)",
        BASE,
        ignore_tool_choice(),
    ),
    Scenario(
        "fault-tool-name-invalid",
        "fault",
        "tool calls name a tool that was not provided (proxy)",
        BASE,
        rename_tool_calls(),
    ),
    Scenario(
        "fault-tool-args-malformed",
        "fault",
        "tool-call arguments are malformed JSON (proxy)",
        BASE,
        malform_tool_arguments(),
    ),
    Scenario(
        "fault-tool-args-schema",
        "fault",
        "tool-call arguments violate the declared schema (proxy)",
        BASE,
        break_argument_schema(),
    ),
    Scenario(
        "fault-json-malformed",
        "fault",
        "structured output (response_format) is malformed JSON (proxy)",
        BASE,
        malform_json_content(),
    ),
)

ORIGINAL_FAULTS: tuple[Scenario, ...] = tuple(s for s in SCENARIOS if s.kind == "fault")


def _cpu_twin(sc: Scenario) -> Scenario:
    return replace(
        sc,
        name=f"{sc.name}-cpu",
        description=f"{sc.description}; CPU",
        server=replace(sc.server, gpu=False),
        reps=2,
    )


CPU_FAULTS: tuple[Scenario, ...] = tuple(_cpu_twin(s) for s in ORIGINAL_FAULTS)

SCENARIOS = SCENARIOS + HELD_OUT + DEV + TOOL_FAULTS + CPU_FAULTS


def by_name(names: list[str] | None, family: str = "qwen3-0.6b") -> list[Scenario]:
    scenarios = catalogue(family)
    if not names:
        return list(scenarios)
    index = {s.name: s for s in scenarios}
    missing = [n for n in names if n not in index]
    if missing:
        raise ValueError(f"unknown scenarios {missing}; available: {sorted(index)}")
    return [index[n] for n in names]


# -- a second model with a different architecture and tokenizer (v0.2.0) ------------------------
# SmolLM2-360M-Instruct: Llama architecture, 49k-token GPT-2-style BPE vocabulary, ChatML template
# without tool support (Qwen3: QK-norm, 151k vocabulary, tool-calling template). GGUFs from
# unsloth/SmolLM2-360M-Instruct-GGUF@391ed11 (F16 is the correct deployment: the published
# weights are BF16, which F16 represents exactly except for values below F16's normal range).

SMOL = ServerSpec(quant="F16")

SMOLLM2_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("benign-gpu-p4", "benign", "F16, GPU, 4 slots", SMOL, reps=1, expects="PASS"),
    Scenario(
        "benign-cpu-p4", "benign", "F16, CPU, 4 slots", replace(SMOL, gpu=False), reps=1,
        expects="PASS",
    ),
    Scenario(
        "control-gpu-p2", "control", "F16, GPU, 2 slots, ctx 8192",
        replace(SMOL, ctx_size=8192, parallel=2), reps=5, expects="PASS", fresh=True,
    ),
    Scenario(
        "control-gpu-fa-off", "control", "F16, GPU, 4 slots, flash attention off",
        replace(SMOL, extra_args=("--flash-attn", "off")), reps=5, expects="PASS", fresh=True,
    ),
    Scenario(
        "control-cpu-p1-t4", "control", "F16, CPU, 1 slot, 4 threads",
        replace(SMOL, gpu=False, ctx_size=4096, parallel=1, threads=4), reps=2, expects="PASS",
        fresh=True,
    ),
    Scenario("fault-quant-q8_0", "fault", "Q8_0 quantization", replace(SMOL, quant="Q8_0")),
    Scenario("fault-quant-q4_k_m", "fault", "Q4_K_M quantization", replace(SMOL, quant="Q4_K_M")),
    Scenario(
        "fault-template-llama3", "fault", "wrong chat template (llama3 built-in)",
        replace(SMOL, extra_args=("--chat-template", "llama3")),
    ),
    Scenario(
        "fault-template-nosystem", "fault", "chat template without a system role",
        replace(SMOL, template_file="chatml-no-system.jinja"),
    ),
    Scenario(
        "fault-topk-ignored", "fault", "top_k ignored (proxy)", SMOL, ignore_param("top_k", 0)
    ),
    Scenario(
        "fault-temperature-ignored", "fault", "temperature ignored, always 1.0 (proxy)", SMOL,
        ignore_param("temperature", 1.0),
    ),
    Scenario(
        "fault-ctx-reduced", "fault", "context reduced to 2048 (--ctx-size) but 4096 claimed",
        replace(SMOL, ctx_size=2048, parallel=1),
    ),
    Scenario(
        "fault-silent-truncation", "fault", "long prompts silently truncated (proxy)", SMOL,
        truncate_chat_prompt(6000),
    ),
    Scenario(
        "fault-json-malformed", "fault",
        "structured output (response_format) is malformed JSON (proxy)", SMOL,
        malform_json_content(),
    ),
    Scenario(
        "fault-tools-dropped", "fault", "tool definitions dropped (proxy)", SMOL, drop_tools()
    ),
)  # fmt: skip


@dataclass(frozen=True)
class ModelFamily:
    """A model with its reference weights, GGUF files and study scenarios."""

    key: str
    hf_dir: Path
    model_id: str
    revision: str
    gguf_dir: Path
    gguf_pattern: str
    scenarios: tuple[Scenario, ...]

    def paths(self) -> Paths:
        return Paths(models_dir=self.gguf_dir, model_pattern=self.gguf_pattern)


FAMILIES: dict[str, ModelFamily] = {
    f.key: f
    for f in (
        ModelFamily(
            "qwen3-0.6b",
            Path("models/hf/Qwen3-0.6B"),
            "Qwen/Qwen3-0.6B",
            "c1899de289a04d12100db370d81485cdf75e47ca",
            Path("models/gguf"),
            "Qwen3-0.6B-{quant}.gguf",
            SCENARIOS,
        ),
        ModelFamily(
            "smollm2-360m",
            Path("models/hf/SmolLM2-360M-Instruct"),
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            "a10cc1512eabd3dde888204e902eca88bddb4951",
            Path("models/gguf-smollm2"),
            "SmolLM2-360M-Instruct-{quant}.gguf",
            SMOLLM2_SCENARIOS,
        ),
    )
}


def catalogue(family: str = "qwen3-0.6b") -> tuple[Scenario, ...]:
    try:
        return FAMILIES[family].scenarios
    except KeyError:
        raise ValueError(
            f"unknown model family {family!r}; choose from {sorted(FAMILIES)}"
        ) from None
