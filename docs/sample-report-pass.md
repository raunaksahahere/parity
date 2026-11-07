# Parity conformance report: qwen3

**Verdict: PASS**. No deviation from the reference was detected by the evaluated probes.

- Target: `http://127.0.0.1:8081` (model `qwen3`, adapter `llamacpp`)
- Reference: hf:local `/home/raunak/Desktop/Parity/models/hf/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`
- Run `20261107T100530-48f12c`, suite `core`, alpha 0.05, 7359 requests, 317.9 s, parity 0.1.0.dev0
- Calibration (2026-11-06T03:54:25+00:00): kl_mean=0.000136, greedy_hazard=0.0158, tool_disagreement=0.276

## Probes

| Probe | Verdict | Effect | Threshold | p / Holm p | Requests |
|---|---|---|---|---|---|
| `template.token_counts` | PASS | 0 max |token count difference| | n/a | n/a / n/a | 7 |
| `longctx.needle` | PASS | 0 lengths/depths with accounting problems | n/a | n/a / n/a | 9 |
| `tools.structured` | PASS | 0 tool-trigger disagreement rate | n/a | 1 / 1 | 32 |
| `greedy.agreement` | PASS | 0 per-token divergence hazard | 0.0158 | 1 / 1 | 4 |
| `distribution.teacher_forced` | PASS | 3.583e-05 nats (mean KL lower bound) | 0.000136 | 1 / 1 | 303 |
| `sampler.conformance` | PASS | 0 max(violation rate, TV distance) | n/a | 1 / 1 | 7004 |

### `template.token_counts`
- **info**: all token counts match the local render

### `longctx.needle`
- **info**: token accounting consistent up to 4032 tokens; needle retrieval rate 1.00
- note: retrieval ability of the model itself is not compared with the reference; only accounting and start-vs-end consistency are used as evidence

### `tools.structured`
- **info**: tool and structured-output behaviour matches
- note: 24 trigger prompts and 8 JSON prompts give limited power; see min_detectable_disagreements

### `greedy.agreement`
- **info**: per-token divergence hazard 0.0000 (0 divergences in 35 compared positions); benign h0=0.0157613, deviation h1=0.0788066
- note: p-value is exp(-LLR), a valid sequential p-value (Ville's inequality); the test stops for H1 at p <= 0.0125 and for H0 at the Wald boundary (beta=0.1)

### `distribution.teacher_forced`
- **info**: mean KL lower bound 0.00004 nats (one-sided CI 0.00003 to 0.00004) vs threshold 0.000135677; top-1 agreement 0.997
- note: KL is a lower bound from truncated top-k lists; true divergence may be larger

### `sampler.conformance`
- **info**: 7 sampler cases match the expected distributions
- note: context 'A random animal that lives in the forest:' skipped: reported log-probs cover only 0.76
- note: context 'The most surprising fact I learned this week was that' skipped: reported log-probs cover only 0.83
- note: top-p-0.6: context 'My favorite thing about summer is' uses top_p=0.6425
- note: top-p-0.6: context 'Once upon a time, in a village by the sea, there lived' uses top_p=0.8712

## What this does not prove

- A PASS does not prove the endpoint serves exactly the reference weights: it shows no difference was detected by these probes at this sample size. Differences smaller than the detection threshold, or in behaviour the probes do not exercise, remain possible.
- A DEVIATION says the observed behaviour differs from the reference beyond the calibrated benign variation. It does not say why (quantization, a different checkpoint, a serving bug, a changed template) and it does not establish intent or misrepresentation by a provider.
- KL is estimated from truncated top-k log-probs and is a lower bound on the true divergence.
- Thresholds come from the benign deployment variants that were calibrated; a legitimate deployment that varies in other ways (different hardware, kernels, batching) may differ more.
- The reference is an implementation of the same open weights, not ground truth: numerics differ slightly between any two correct implementations.
- Probes that need token-id prompts or log-probs are skipped on endpoints without them; the remaining behavioural probes have lower power.
