# Adding a probe

A probe is one class that takes a `ProbeContext` and returns a `ProbeResult`. The suite runner
handles clients, adapters, budgets, checkpoints, multiple testing and reporting; a probe only
decides what to measure and how to turn it into evidence.

## 1. The contract

```python
from parity.probes.base import Finding, Probe, ProbeContext, ProbeResult, Status


class MyProbe(Probe):
    id = "family.name"  # stable id, shown in reports and used for --probes
    family = "family"
    version = "1"  # bump when the suite data or the statistics change

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        ...
        return self.result(Status.PASS, ...)
```

`ProbeContext` gives you:

| field | use |
|---|---|
| `ctx.client` | async client with bounded concurrency, rate limiting and retries (`ctx.client.chat`, `ctx.client.completions`) |
| `ctx.adapter` | provider quirks: `teacher_force`, `next_token`, `generate`, and `capabilities` |
| `ctx.kit` | tokenizer and official chat template rendering (`encode`, `chat_token_ids`) |
| `ctx.reference` | the reference implementation (`teacher_force`, `greedy`), or `None` |
| `ctx.alpha` | the error budget **for this probe** (already divided among the statistical probes) |
| `ctx.options` | suite data location, calibrated thresholds (`ctx.options["thresholds"]`), tuning knobs |
| `ctx.can_afford(n)` | request-budget check; call it before each batch |

## 2. Rules that keep results honest

1. **Say SKIPPED, not PASS, when you cannot test.** Check `ctx.adapter.capabilities` and the
   presence of `ctx.reference` first, and explain in `notes`.
2. **A PASS needs evidence.** Prefer an equivalence-style rule (the confidence bound is below
   the threshold) over "not significant". If the data are too thin, return `INCONCLUSIVE`.
3. **Pick the right kind of evidence.**
   * Deterministic facts (token counts, schema violations that cannot be noise) set
     `hard_evidence=True` and no `p_value`.
   * Statistical results set `p_value` for H0 = "the endpoint conforms". If you test
     sequentially, spend alpha (Bonferroni across looks, or an SPRT whose `exp(-LLR)` is the
     p-value).
4. **Pair with the reference whenever possible.** Same prompts, same greedy settings.
5. **Report an effect size with units** (`effect`, `effect_unit`, `ci`, `threshold`) so a result
   can be compared across runs and stored in the history database.
6. **Never write secrets.** Findings, notes and metrics go into reports verbatim.
7. **Handle budget and errors.** Use `ctx.can_afford`, catch `EndpointError` and return
   `INCONCLUSIVE` with a finding rather than raising.
8. **Metrics are `dict[str, float | None]`.** Non-finite values become `null` automatically.

## 3. Suite data

Prompts and corpora are versioned JSON under `src/parity/suites/<suite>/`, loaded with
`parity.suites.load_json(suite, name)`. A change to the data that can change results needs a new
suite directory (`core-v2`) or at least a probe `version` bump; old reports must stay
interpretable.

## 4. Register it

Add the class to `PROBE_CLASSES` in `src/parity/suite.py`, append the id to the right entry of
`SUITES`, and, if it computes a p-value or runs sequentially, to `STATISTICAL_PROBES` (this is
what divides alpha). If it has a calibratable metric, teach `benign_values` in
`stats/calibration.py` about it and read the threshold from `ctx.options["thresholds"]`.
Add a line to `parity list-probes` in `cli.py`.

## 5. Test it without a model server

Every probe is tested against fakes, and CI never needs a model.

* `tests/fake_endpoint.py`: a complete fake OpenAI-compatible endpoint with switchable faults.
* `tests/fake_lm.py`: a synthetic language model usable as reference and as target.
* `toy_kit` fixture (`tests/conftest.py`): a tiny byte-level BPE tokenizer with a chat template,
  trained in-process.

Minimum bar for a new probe: an honest target passes; each fault the probe claims to detect
is detected **and named**; a target without the needed capability is SKIPPED; a tiny budget gives
INCONCLUSIVE; and, if it is statistical, a simulation checks the false-positive rate. See
`tests/test_sampler_probe.py` for a worked example, including a regression test written after a
real false positive was found on a live server.

Then validate against a real server (`scripts/` has examples) and add the scenario that exercises
it to `src/parity/faults/scenarios.py` so `parity inject-faults` measures its power.
