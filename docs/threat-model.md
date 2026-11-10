# What Parity can and cannot establish

Parity is a black-box test. It sends requests and compares answers with a reference. No such
test can prove that an endpoint serves particular weights: it can only fail to find a
difference. This page states what each verdict rests on, which mistakes and manipulations are
caught, and which are out of reach, so that nobody reads more into a PASS than it holds.

## What a PASS rests on

A PASS requires all of the following. If one is missing, the run is INCONCLUSIVE instead.

| Requirement | Enforced by |
|---|---|
| At least one **model-identity probe** (distribution or greedy) reached a decision | verdict rule; override only with `--allow-unverified-pass` |
| No probe returned DEVIATION, none was inconclusive | verdict logic |
| Every probe named with `--require-probe` reached a decision | verdict rule |
| The calibration matches the model, tokenizer, suite data and probe versions of this run | checked before the first request |
| Saved results that were reused belong to this exact run configuration | run manifest with per-file checksums |
| The request budget was not exhausted before a decision | probes return INCONCLUSIVE, never PASS, on partial evidence |

Template, long-context, sampler and tool probes describe the *serving stack*. They pass for any
model behind a correct stack, which is why they cannot carry a PASS on their own.

## Identity evidence by endpoint type

| Endpoint offers | Identity probes that run | Strength |
|---|---|---|
| Token-id prompts and top-k log-probs (llama.cpp, vLLM) | distribution (KL on next-token probabilities) and greedy (token level) | Strongest: detected Q8_0 quantization, a KL of about 1e-3 nats |
| Chat only (Ollama, most hosted APIs) | greedy in chat mode: replies at temperature 0 compared as text | Weaker: catches different checkpoints and coarse quantization; near-lossless changes can pass. The chat template is part of the comparison |
| No reference available | none | No PASS is possible |

Run `parity doctor --target URL --served-model NAME` to see which row applies before spending a
budget.

## Faults that are caught (measured)

Wrong or broken chat template, Q8_0 / Q4_K_M / Q2_K quantization, ignored or mis-scaled
sampling parameters, reduced context window, silent prompt truncation, dropped tool
definitions, ignored `tool_choice`, invalid tool names, malformed or schema-violating tool
arguments, malformed structured output. Detection rates, confidence intervals and the
deployments they were measured on are in [validation.md](validation.md). Faults marked
*request* or *response* there are simulated by a proxy; a real server with the same bug could
differ in detail.

## What can still pass

* **Differences below the calibrated threshold.** Thresholds sit just above the variation
  between known-good deployments. A change smaller than that variation is, by construction,
  not distinguishable from a correct deployment.
* **Behaviour the probes do not exercise.** The corpus is small and mostly English. A fine-tune
  that differs only on other languages, long documents or a narrow topic may not move any
  probe. Safety behaviour, latency and throughput are not tested.
* **A minority of bad replicas.** Behind a load balancer, probes see a mixture. A replica that
  answers a small share of requests moves averages by that share only, and can stay under a
  threshold. Repeating runs (canary mode) raises the chance of catching it, without a guarantee.
* **Changes after the run.** A verdict describes the endpoint at the time of the run. Use
  `parity canary` to re-check on a schedule.
* **A provider that recognises the probes.** The shipped prompts are public and fixed. An
  endpoint could route them to the genuine model and everything else elsewhere. Parity does
  not defend against this today; seeds vary the sampler and long-context probes only. Treat a
  PASS from an untrusted party accordingly.
* **Forged log-probs.** A server could report log-probs from the genuine model while sampling
  from another. The sampler probe compares samples with the reported distribution, which
  catches inconsistent forgeries on the tested contexts, not all of them.
* **Text-only endpoints.** In chat mode there are no log-probs to compare, so only the greedy
  path is tested; a model that agrees on the most likely tokens but differs in probabilities
  passes.

## What can be flagged although nothing is wrong

* A correct deployment that differs from the calibrated family in a way nobody calibrated for
  (other hardware, kernels, precision such as fp16 on a GPU without bf16).
* A serving stack that uses its own chat template on purpose (Ollama's library templates differ
  from the official ones for some models). Parity reports the difference; whether it matters is
  the operator's decision.
* A DEVIATION never says why, and never says anything about intent.

## Using it responsibly

1. Calibrate on at least two known-good deployments that differ in benign ways.
2. Run `parity doctor` and read the coverage block of every report.
3. In CI, name the probes a PASS must include: `--require-probe distribution --require-probe greedy`.
4. Re-run on a schedule; a single PASS ages.
5. Treat the verdict as evidence to weigh with what else you know about the provider.
