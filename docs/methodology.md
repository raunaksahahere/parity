# Methodology

This document explains what each probe measures, which statistical procedure turns the
measurement into a verdict, and where the procedure can mislead. Everything here describes the
code in `src/parity`; the numbers that show how well it works are in [validation.md](validation.md).

## 1. What a verdict means

Parity compares an endpoint (the *target*) with a *reference*: an implementation of the same
open-weight model that you trust, normally Hugging Face `transformers` in fp32 at a pinned
commit. Each probe returns one of

* **PASS**: the probe found no difference beyond the calibrated benign variation, *and* had
  enough data to say so (for the distributional probe: the upper confidence bound is below the
  threshold, which is equivalence evidence, not just "failed to reject");
* **DEVIATION**: the behaviour differs from the reference beyond benign variation;
* **INCONCLUSIVE**: the data do not decide (interval straddles the threshold, budget ran out,
  a request failed, or a multiplicity correction removed the evidence);
* **SKIPPED**: the endpoint lacks a capability the probe needs.

The suite verdict is DEVIATION if any probe is, otherwise INCONCLUSIVE if any probe is or if
nothing was evaluated, otherwise PASS. A DEVIATION says *how* behaviour differs. It never says
why, and it does not claim intent. Every report ends with a "what this does not prove" section.

Exit codes: `0` PASS, `1` DEVIATION, `2` INCONCLUSIVE or error.

## 2. Probes

### 2.1 Template and tokenizer conformance (`template.token_counts`)

Seven crafted conversations (plain, system prompt, multi-turn, Unicode, long system prompt,
tools with and without a system prompt) are rendered locally with the model's official chat
template and tokenized. The server is asked for one token and its `usage.prompt_tokens` is
compared with the local count.

Token counts are deterministic, so any mismatch is **hard evidence**: it is exempt from the
multiple-testing correction. To say *why*, the probe computes, with the local template, the count
change that each hypothesized corruption would cause (system prompt dropped, tools not rendered,
history dropped, system text merged into the user turn) and reports the hypothesis whose
predicted change equals the observed one. A constant offset across all conversations is reported
as such (+1 is consistent with a duplicated BOS). Providers that add accounting overhead can
set `tolerate_constant_offset`.

*Limit:* equal counts do not prove identical prompts; a template that differs in content but not
in length passes this probe. The distributional probe covers that case when log-probs exist.

### 2.2 Distributional fidelity (`distribution.teacher_forced`)

Corpus documents (25 short texts, 64 tokens each, versioned as `core-v1`) are teacher-forced:
the *document's* tokens are the context, so target and reference see identical prefixes. The
target is asked for the top-k log-probs (k=100) at every position.

**Truncated KL is a lower bound.** Providers report only top-k lists, so KL(P‖Q) is not
observable. Let S be the ids reported by both sides. Merging every other id into one "rest" bin
on both sides is a coarse-graining of the vocabulary, and by the data-processing inequality

    KL(P‖Q) ≥ Σ_{i∈S} p_i log(p_i/q_i) + p_r log(p_r/q_r),   p_r = 1 − Σ_S p_i,  q_r = 1 − Σ_S q_i.

The estimate therefore never overstates the divergence (this is property-tested against the true
KL on random distribution pairs, and it tightens as k grows). The bias is one-directional: ids
reported by only one side lose their divergence rather than being floored, and a small k
hides tail divergence. The same partition gives a lower bound on total variation. We also report
the log-prob delta of the actual next token and top-1 agreement; the delta is only defined where
the actual token is inside the target's top-k, which selects toward high-probability tokens.

**Uncertainty.** Positions within a document are correlated, so the confidence interval comes
from a *cluster bootstrap* resampling whole documents. Its coverage is checked by simulation.

**Decision (equivalence test).** With threshold τ (largest mean KL still considered equivalent):
PASS if the one-sided upper bound < τ; DEVIATION if the one-sided lower bound > τ; INCONCLUSIVE
otherwise. Documents are processed in chunks of five with Bonferroni alpha-spending over the
looks, so clear outcomes stop early (a correct deployment typically needs 5 documents) without
inflating the error rate.

**Alignment.** When the target accepts token-id prompts, alignment is exact by construction.
Where it does not (generic hosted APIs), v0.1 *skips* the probe: byte-offset alignment of
differently-tokenized positions is not implemented. See §6.

### 2.3 Greedy agreement (`greedy.agreement`)

Both sides continue the same 16-token prompts greedily for 48 tokens. Even a correct deployment
diverges occasionally (near-tied tokens flip with numerical noise), so exact agreement is the
wrong statistic. Instead, every compared position is a Bernoulli trial for "the continuations
first diverge here", and a prompt stops contributing at its first divergence (later tokens are
conditioned on different contexts). The per-token *divergence hazard* is tested with a Bernoulli
SPRT:  H0: hazard = h0 (benign, from calibration) vs H1: hazard = h1 (default max(0.025, 5·h0)).

The likelihood ratio is a non-negative martingale under H0, so by Ville's inequality
P(sup LR ≥ 1/α) ≤ α. The test therefore stops for H1 when exp(−LLR) ≤ α, and p = min(1, exp(−LLR))
is a valid sequential p-value, which lets this probe take part in the Holm step. (Wald's
(1−β)/α boundary would leave p slightly above α at the stopping time.) The lower boundary
log(β/(1−α)) bounds the miss rate approximately, since the walk can overshoot.

### 2.4 Sampler conformance (`sampler.conformance`)

Question: do temperature, `top_k` and `top_p` do what the API says?

**Expected distribution.** For each of up to two high-entropy contexts, the target is asked for
its own temperature-1, untruncated top-1000 log-probs. The expected distribution under a sampler
setting is derived *from that*, not from the reference: a quantized model's slightly different
logits therefore cannot be blamed on the sampler. Contexts whose reported log-probs cover less
than 90% of the mass are skipped.

**Order independence.** llama.cpp truncates (`top_k`, `top_p`) *before* applying temperature,
while Hugging Face and vLLM apply temperature first. For `top_p` with T≠1 the two orders give
different distributions, so such combinations are refused. The tested cases (neutral T=1,
T=0.7, T=1.4, `top_k`=5, `top_k`=1, `top_p`=0.6 at T=1, T=0.7 with `top_k`=10) give the same
expected distribution under both orders.

**Tests (per case, after pooling both contexts).**

* *Violations*: samples outside the expected support (for example a token ranked below `top_k`).
  Under H0 the rate is ≈ 0; a one-sided binomial test against a small numerical tolerance
  (1e-3) gives p. Run only for truncating cases: without truncation, mass legitimately moves
  beyond the provider's reported top-k at T>1 (an earlier version flagged exactly this on a
  real server; see `tests/test_sampler_probe.py`).
* *G-test* (log-likelihood ratio, Williams-corrected, small cells pooled to expected ≥ 5) of the
  in-support samples against the expected conditional distribution. G statistics from
  independent contexts add. This conditional test is exact for any T because the normalizer
  cancels within the reported set.
* *Effect size*: plug-in total variation is biased upward by roughly √(cells/n), so the effect
  is the TV *minus its expectation under H0* (simulated). A deviation also needs the excess TV
  ≥ 0.03 or the violation rate ≥ 1%, i.e. practical as well as statistical significance.

Sampling is sequential at 100/250/500 draws per context with Bonferroni spending over looks and
tests, so a gross bug (ignored `top_k`) stops at the first look. Cases where the truncation
boundary sits on near-tied tokens, or where little mass lies beyond the truncation, are skipped
or flagged with their reduced detection power.

### 2.5 Long-context integrity (`longctx.needle`)

A unique "needle" sentence is buried at the start, middle and end of a filler haystack of
increasing length (512, 2048, ... up to the claimed window minus headroom for the reply).

1. **Token accounting (hard evidence).** The rendered prompt has an exactly known token count.
   Fewer `usage.prompt_tokens` than expected, or a rejection inside the claimed window, means
   silent truncation or a wrong claimed window. The same difference at *every* length
   (including the shortest) is instead reported as a chat-template or tokenization difference;
   it is still a deviation, but not truncation.
2. **Retrieval (behavioural).** A start needle lost while the end needle survives, at a length
   where the start needle was earlier found, suggests left-truncation that accounting missed.
   Retrieval alone *never* yields DEVIATION (a 0.6B model can simply fail to retrieve); it
   yields INCONCLUSIVE. Retrieval ability is not compared with the reference.

### 2.6 Tool-call and structured-output conformance (`tools.structured`)

The design is paired: identical greedy prompts go to the target and through the reference.

* *Trigger agreement.* 12 prompts that should call a tool and 12 that should not. The number of
  target/reference disagreements is tested against a benign disagreement rate with an exact
  binomial test; F1 of the target against the reference decisions is reported.
* *Validity.* Called tool names must be among those offered, arguments must be valid JSON and
  satisfy the declared schema; raw `<tool_call>` markup in the content (parsing not enabled) is
  reported.
* *Structured output.* Eight extraction prompts with `response_format=json_schema`. The reference
  generates unconstrained, so a target that is invalid where the reference is valid is tested
  with a one-sided sign test on the discordant pairs.

With 24 trigger prompts the probe has limited power; it reports its own
`min_detectable_disagreements`. Reference tool calls are parsed from Hermes/Qwen `<tool_call>`
markup only in v0.1.

## 3. Calibration

Thresholds must reflect *benign* variation: the amount two correct deployments differ. Same-
configuration repeats are the wrong source: on this repository's GPU build they are bit-
identical, so a noise floor built from repeats would be zero and every honest deployment
that differs in backend, slot count or hardware would be flagged
([baseline-noise.md](baseline-noise.md) measures this: CPU BF16 sits ~6× further from the
reference than GPU BF16).

`parity calibrate` runs the distributional, greedy and tool probes without early stopping on
one or more known-good deployments that differ in benign ways (different backend, concurrency,
hardware). For each metric it takes a conservative per-deployment value (an upper confidence
bound), and sets

    threshold = max(benign values) × safety_factor (default 2),  never below a floor.

Without a calibration file, lenient defaults are used and the report says so. Calibration is
specific to a model, a reference and a family of deployments: a correct deployment that varies in
a way the calibration did not see (different kernels, other hardware) may land above the threshold.

## 4. Multiple testing and error budget

* Each *statistical* probe (distribution, greedy, sampler, tools) runs at α/m, where m is the
  number of such probes in the suite (Bonferroni share), so a probe-level DEVIATION always
  survives the Holm step: adjusted p ≤ m·p ≤ α.
* Holm–Bonferroni is then applied across all probes that report a p-value and is reported as
  `adjusted_p`. A probe that is significant on its own but not after correction is downgraded to
  INCONCLUSIVE (never silently kept as DEVIATION).
* Hard-evidence deviations (token-count mismatch, silent truncation) are not statistical tests
  and are exempt.
* Within a probe, sequential looks spend alpha by Bonferroni (distribution, sampler) or Ville
  (greedy), so peeking does not inflate the error rate.

## 5. Budget and early stopping

`--budget N` caps requests sent to the target. Probes check the budget before each batch; a
probe that cannot reach a decision within the budget returns INCONCLUSIVE rather than passing on
partial evidence. Requests per verdict for each scenario are reported in
[validation.md](validation.md).

`--fail-fast` stops the suite after the first probe that returns DEVIATION; later probes are
recorded as SKIPPED ("not run"). This cannot weaken a deviation: each statistical probe runs
at α/m, so its DEVIATION survives the Holm step whatever the other probes would have returned,
and hard evidence is exempt from correction. Probes run cheapest first (template,
long-context, tools, greedy, distribution, sampler), and since they do not influence each
other, the validation study reports the requests a fail-fast run *would* have used, computed
from the full runs.

## 6. Capability fallbacks and where alignment is impossible

| Endpoint capability | Effect |
|---|---|
| No token-id prompts (generic hosted API) | distribution, greedy, sampler **skipped**; the result is flagged lower power |
| No `usage.prompt_tokens` | template probe INCONCLUSIVE |
| No log-probs | distribution and sampler skipped |
| Top-k < 20 (e.g. vLLM default cap) | KL bound is weaker; the sampler skips contexts with < 90% coverage |
| No `response_format` / tools support | tools probe reports the absolute failures; lower power |

Cases where token alignment between target and reference is impossible or unreliable, so the
corresponding probes do not claim a result:

* The target tokenizes the *text* itself and cannot be sent token ids: positions need not
  coincide with the reference's. v0.1 does not align by byte offsets; it skips the probe.
* The endpoint returns log-probs only for *generated* tokens (typical chat APIs), so teacher
  forcing is impossible.
* The reported log-probs are post-processed (temperature or truncation already applied), which
  invalidates both KL and the sampler's expected distribution.
* Different handling of special tokens or BOS between server and reference tokenizer; whitespace
  normalization in the server.
* The model's tokenizer in the server differs from the reference's (a different checkpoint).

## 7. Known limitations

* Truncated-KL is a lower bound; the true divergence can be larger.
* The corpus is small (25 documents) and mostly English; divergences confined to other
  languages, long documents, or rare content can be missed.
* The equivalence threshold is a *practical* choice: Parity detects deviations above the
  benign floor, not any difference at all. A near-lossless change is detectable only if it
  exceeds that floor.
* The reference is itself an implementation, not ground truth.
* Throughput, latency, safety behaviour and sampling seeds are out of scope.
