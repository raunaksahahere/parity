# Parity

**Is this endpoint really serving the model I think it is?**

Parity compares an OpenAI-compatible endpoint (llama.cpp, vLLM, SGLang, Ollama, hosted
providers) with a reference implementation of the same open-weight model, using cheap,
statistically rigorous, model-agnostic probes. v0.1 is validated live against llama.cpp; the
vLLM and generic OpenAI adapters are implemented from documented formats and tested with
fixtures only. It answers with **PASS**, **DEVIATION** or
**INCONCLUSIVE** and never accuses without evidence: every report states what the result does
*not* prove.

A real run (llama.cpp b12128 serving the BF16 GGUF on an RX 6500M, reference = HF fp32):

```
$ parity run --model models/hf/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
             --target http://127.0.0.1:8081 --served-model qwen3 --claimed-context 4096
Running suite 'core' against http://127.0.0.1:8081 (reference hf:local)
  template.token_counts          PASS         requests=7 effect=0
  distribution.teacher_forced    PASS         requests=303 effect=7.338e-06
  greedy.agreement               PASS         requests=4 effect=0
  sampler.conformance            PASS         requests=6504 effect=0
  longctx.needle                 PASS         requests=9 effect=0
  tools.structured               PASS         requests=32 effect=0

Verdict: PASS. No deviation from the reference was detected by the evaluated probes.
```

## 60-second quickstart

```bash
# 1. install from a checkout (not yet on PyPI); the reference extra adds CPU PyTorch for the
#    local Hugging Face reference
pip install ".[reference]" --extra-index-url https://download.pytorch.org/whl/cpu

# 2. have an endpoint to test, for example llama.cpp (validated with build b12128):
llama-server -m Qwen3-0.6B-Q8_0.gguf --port 8081 --jinja -c 16384 --parallel 4

# 3. compare it with the reference model (downloads Qwen/Qwen3-0.6B once, pinned by revision)
parity run --model Qwen/Qwen3-0.6B --target http://localhost:8081 --served-model qwen3 \
           --revision c1899de289a04d12100db370d81485cdf75e47ca --claimed-context 4096
echo $?     # 0 pass, 1 deviation detected, 2 inconclusive or error
```

Open `parity-runs/<run-id>/report.html`. Python API:

```python
import parity

reference = parity.hf_reference(
    "Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca"
)
target = parity.Target("http://localhost:8081", "qwen3")
report = parity.Suite(name="core").run(target, reference)
print(report.verdict, report.exit_code)
```

For meaningful verdicts, **calibrate** first against known-good deployments that differ in
benign ways (see [methodology](docs/methodology.md#3-calibration)):

```bash
parity calibrate --model Qwen/Qwen3-0.6B --benign gpu=http://localhost:8081 --benign cpu=http://localhost:8082
parity run ... --calibration calibration.json
```

## What it checks

| Family | Probe | Evidence |
|---|---|---|
| 1. Template and tokenizer | `template.token_counts` | Official chat template rendered locally vs the server's `usage.prompt_tokens`; names the cause (dropped system prompt, duplicated BOS, tool formatting, ...) |
| 2. Distributional fidelity | `distribution.teacher_forced` | Teacher-forced corpus; truncated-KL lower bound with document-cluster bootstrap and an equivalence-style decision |
| 3. Greedy agreement | `greedy.agreement` | Temperature-0 continuations; per-token divergence hazard tested sequentially |
| 4. Sampler conformance | `sampler.conformance` | Samples at set temperature/top_k/top_p vs the expected truncated, renormalized distribution (G-test) |
| 5. Long-context integrity | `longctx.needle` | Token accounting and needle retrieval at increasing lengths |
| 6. Tools and structured output | `tools.structured` | Valid JSON and schema rate, tool-name validity, trigger F1 vs the reference |

Statistics: thresholds calibrated from benign variation, SPRT early stopping under a request
budget, bootstrap confidence intervals, Holm-Bonferroni across probes. Details and the places
where each procedure can mislead: [docs/methodology.md](docs/methodology.md).

<!-- RESULTS:BEGIN -->
## Measured results

Qwen3-0.6B on a laptop (CPU + RX 6500M), 23 correct-deployment controls and 11 injected faults x 3 repetitions. Method, controls and caveats: [docs/validation.md](docs/validation.md).

False positives on correct deployments (held-out controls): **0/23** runs (95% CI 0.0-14.8%), nominal alpha 0.05.

| Injected fault | Detected | Flagged by | Requests to detection (fail-fast) | Full-suite requests | Full-suite time |
|---|---|---|---:|---:|---:|
| template-nosystem | 3/3 | template, longctx | 7 | 6859 | 2.3 min |
| template-llama3 | 3/3 | template, longctx | 7 | 7359 | 118 s |
| topk-ignored | 3/3 | sampler | 5259 | 5259 | 2.0 min |
| topp-ignored | 3/3 | sampler | 6559 | 6559 | 2.2 min |
| temperature-ignored | 3/3 | greedy, sampler | 52 | 4959 | 108 s |
| temperature-x1.5 | 3/3 | sampler | 2759 | 2759 | 74 s |
| silent-truncation | 3/3 | longctx | 16 | 7359 | 2.5 min |
| ctx-reduced | 3/3 | longctx | 16 | 6859 | 118 s |
| quant-q2_k | 3/3 | greedy, distribution, sampler (1/3) | 52 | 3359 | 80 s |
| quant-q4_k_m | 3/3 | greedy, distribution | 52 | 6359 | 81 s |
| quant-q8_0 | 3/3 | distribution | 355 | 6859 | 88 s |
<!-- RESULTS:END -->

## Limits of inference

A report is evidence about *behaviour on these probes*, not a certificate.

* **PASS is not proof of identity.** It means no difference beyond the calibrated benign
  variation was found at this sample size. Smaller differences, and behaviour the probes do not
  exercise (other languages, long documents, safety behaviour), can remain.
* **DEVIATION is not an accusation.** It says the behaviour differs from the reference beyond
  benign variation. It cannot tell a different checkpoint from a quantization, a template bug or
  a serving bug, and says nothing about intent.
* **The reference is an implementation, not ground truth**, and two correct implementations
  differ numerically. Thresholds are calibrated on the deployment variants you give it; a legitimate
  variant outside that family (other hardware or kernels) can look like a deviation.
* **KL is a lower bound** (truncated top-k), so true divergence can be larger.
* **Capabilities gate power.** Endpoints without token-id prompts or log-probs skip the
  distributional, greedy and sampler probes; the remaining behavioural probes have lower power and
  the report says so.
* The shipped corpus and prompts are small and mostly English. See the full list in
  [docs/methodology.md](docs/methodology.md#7-known-limitations).

## More

* [Methodology](docs/methodology.md), [validation study](docs/validation.md),
  [baseline noise](docs/baseline-noise.md), [decisions](docs/decisions.md)
* [Adding a probe](docs/adding-probes.md), [contributing](CONTRIBUTING.md), [changelog](CHANGELOG.md)
* CLI: `parity run | report | calibrate | list-probes | history | canary | inject-faults`
* **Upgrade gates:** the repository is a GitHub Action ([action.yml](action.yml),
  [example](docs/examples/upgrade-gate.yml)); a [Dockerfile](Dockerfile) is included.
* **Canary mode:** `parity canary --interval 3600 --port 9108 ...` runs the suite on a schedule
  and serves Prometheus metrics (`parity_verdict`, `parity_probe_status`, ...).
* Secrets are read from environment variables only (`--api-key-env NAME`) and are never written
  to reports; target URLs are stored without credentials or query strings.

## License

Apache-2.0
