"""Family 2: distributional fidelity by teacher forcing a fixed corpus.

For each corpus document the target and the reference are asked for the next-token
distribution at every position (teacher forcing: the *document's* tokens are the context,
so both sides see identical prefixes and tokenization alignment is exact when the target
accepts token-id prompts). Per position we compute a lower bound on KL(reference || target)
from the reported top-k lists (``stats.kl``), the log-prob delta of the actual next token,
and top-1 agreement. Positions are correlated within a document, so uncertainty comes from
a document-level cluster bootstrap.

Decision (``threshold`` is the largest mean KL, in nats, still considered equivalent):
* PASS: the one-sided upper confidence bound is below the threshold (equivalence shown);
* DEVIATION: the one-sided lower bound is above it;
* INCONCLUSIVE: otherwise (more data would be needed).

Documents are processed in chunks with Bonferroni alpha-spending over the looks, so a clear
outcome stops early without inflating the error rate.

Known biases (see docs/methodology.md): the KL estimate is a lower bound (truncated top-k);
the delta statistics only cover positions whose actual token is inside the target's top-k.

**Chat mode.** Endpoints without token-id prompts but with chat log-probs (Ollama, OpenAI-style
APIs) are compared on their own temperature-0 replies to plain chat prompts: each reported
token is mapped back to a reference id by its bytes, the reference is teacher-forced on the
officially rendered prompt plus those tokens, and the per-position top-k distributions are
compared as above. Prompts whose server-side prompt-token count differs from the official
render are skipped (their contexts provably differ). Threshold: ``kl_chat_mean``.
"""

from __future__ import annotations

import asyncio
import math

import numpy as np

from parity.adapters.parsing import parse_content_style
from parity.client import EndpointError
from parity.probes.base import Finding, Probe, ProbeContext, ProbeResult, Status
from parity.schema import PositionLogprobs
from parity.stats.bootstrap import cluster_bootstrap_mean
from parity.stats.kl import truncated_kl
from parity.suites import DEFAULT_SUITE, load_corpus, load_json

DEFAULT_KL_THRESHOLD = 2e-3  # nats; replaced by a calibrated value when available


def position_stats(
    ref: dict[str, np.ndarray], tgt: list[PositionLogprobs]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-position (kl_lower_bound, delta, top1_match, shared_mass) for one document.

    ``delta`` is NaN where the target did not report the actual next token's log-prob.
    """
    n = len(tgt)
    kl = np.zeros(n)
    delta = np.full(n, np.nan)
    top1 = np.zeros(n)
    shared = np.zeros(n)
    for j, pos in enumerate(tgt):
        t_ids = np.array([t.token_id for t in pos.top], dtype=np.int64)
        t_lp = np.array([t.logprob for t in pos.top])
        res = truncated_kl(ref["top_ids"][j], ref["top_lp"][j], t_ids, t_lp)
        kl[j] = res.kl
        top1[j] = float(res.top1_match)
        shared[j] = res.ref_mass_shared
        if pos.logprob is not None:
            delta[j] = pos.logprob - ref["lp"][j]
    return kl, delta, top1, shared


def _token_bytes(entry: object) -> bytes:
    raw = getattr(entry, "token_bytes", None)
    return bytes(raw) if raw else str(getattr(entry, "token", "")).encode("utf-8")


async def chat_documents(
    ctx: ProbeContext, top_k: int, max_tokens: int
) -> tuple[list[tuple[list[int], int, list[PositionLogprobs]]], int, list[str]]:
    """Run the chat prompts and align replies with reference ids.

    Returns (documents, requests, notes); each document is (full ids, first predicted index,
    aligned target positions with reference token ids filled in)."""
    opts = ctx.options
    tkw = dict(opts.get("template_kwargs", {"enable_thinking": False}))
    spec = load_json(opts.get("suite", DEFAULT_SUITE), "tools.json")
    vocab = ctx.kit.bytes_to_id()
    msgs_list = [[{"role": "user", "content": t}] for t in spec["plain_prompts"]]

    async def one(msgs: list[dict[str, str]]) -> object:
        try:
            return await ctx.chat(
                msgs,
                max_tokens=max_tokens,
                temperature=0.0,
                logprobs=True,
                top_logprobs=top_k,
                chat_template_kwargs=tkw,
            )
        except EndpointError as exc:
            return exc

    replies = await asyncio.gather(*(one(m) for m in msgs_list))
    docs: list[tuple[list[int], int, list[PositionLogprobs]]] = []
    template_skips = no_logprobs = 0
    for msgs, data in zip(msgs_list, replies, strict=True):
        if isinstance(data, EndpointError):
            raise data
        assert isinstance(data, dict)
        prompt_ids = ctx.kit.chat_token_ids(msgs, **tkw)
        served = (data.get("usage") or {}).get("prompt_tokens")
        if served is not None and served != len(prompt_ids):
            template_skips += 1
            continue
        choice = (data.get("choices") or [{}])[0]
        positions = parse_content_style(choice.get("logprobs") or {})
        if not positions or not any(p.top for p in positions):
            no_logprobs += 1
            continue
        gen: list[int] = []
        aligned: list[PositionLogprobs] = []
        for pos in positions:
            tid = vocab.get(_token_bytes(pos))
            if tid is None:
                break  # alignment lost from here on
            top = [t.model_copy(update={"token_id": vocab[b]}) for t in pos.top
                   if (b := _token_bytes(t)) in vocab]  # fmt: skip
            aligned.append(pos.model_copy(update={"token_id": tid, "top": top}))
            gen.append(tid)
        if aligned:
            docs.append((prompt_ids + gen, len(prompt_ids) - 1, aligned))
    notes = []
    if template_skips:
        notes.append(
            f"{template_skips} prompts skipped: the server counted a different number of prompt "
            "tokens than the official template renders (see the template probe)"
        )
    if no_logprobs:
        notes.append(f"{no_logprobs} replies came without log-probs")
    return docs, len(msgs_list), notes


class DistributionProbe(Probe):
    id = "distribution.teacher_forced"
    family = "distribution"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        caps = ctx.adapter.capabilities
        if ctx.reference is None:
            return self.result(Status.SKIPPED, notes=["no reference implementation configured"])
        if not (caps.token_id_prompts and caps.top_logprobs):
            if caps.chat_logprobs:
                return await self._chat_mode(ctx)
            return self.result(
                Status.SKIPPED,
                notes=[
                    "target cannot take token-id prompts and return top-k log-probs, so exact "
                    "teacher-forced alignment is impossible; behavioural probes run at lower power"
                ],
            )
        opts = ctx.options
        top_k = min(int(opts.get("distribution_top_k", 100)), caps.max_top_k)
        max_tokens = int(opts.get("distribution_max_tokens", 64))
        chunk = int(opts.get("distribution_chunk", 5))
        if opts.get("calibrating"):
            chunk = 10**6  # one look over the whole corpus: no early stopping
        threshold = float(opts.get("thresholds", {}).get("kl_mean", DEFAULT_KL_THRESHOLD))

        docs = [
            ctx.kit.encode(d.text)[:max_tokens]
            for d in load_corpus(opts.get("suite", DEFAULT_SUITE))
        ]
        docs = [d for d in docs if len(d) >= 2]
        n_looks = math.ceil(len(docs) / chunk)
        alpha_look = ctx.alpha / n_looks

        per_doc_kl: list[np.ndarray] = []
        per_doc_abs: list[np.ndarray] = []
        per_doc_delta: list[np.ndarray] = []
        per_doc_top1: list[np.ndarray] = []
        per_doc_shared: list[np.ndarray] = []
        n_requests = 0
        findings: list[Finding] = []
        status = Status.INCONCLUSIVE
        res = None
        notes = ["KL is a lower bound from truncated top-k lists; true divergence may be larger"]

        for look in range(n_looks):
            batch = docs[look * chunk : (look + 1) * chunk]
            if not ctx.can_afford(sum(len(ids) - 1 for ids in batch)):
                if res is None:
                    return self.result(
                        Status.INCONCLUSIVE,
                        n_requests=n_requests,
                        notes=["request budget too small for even one chunk of documents"],
                    )
                notes.append("request budget reached; decision made on fewer documents")
                break
            try:
                outs = await asyncio.gather(
                    *(
                        ctx.adapter.teacher_force(ctx.client, ctx.model, ids, top_k=top_k)
                        for ids in batch
                    )
                )
            except EndpointError as exc:
                return self.result(
                    Status.INCONCLUSIVE,
                    n_requests=n_requests,
                    findings=[Finding(severity="warn", message=f"target request failed: {exc}")],
                )
            n_requests += sum(len(ids) - 1 for ids in batch)
            for ids, tgt in zip(batch, outs, strict=True):
                if any(t.token_id is None for p in tgt for t in p.top):
                    return self.result(
                        Status.SKIPPED,
                        n_requests=n_requests,
                        notes=["target did not report token ids for its top-k list"],
                    )
                ref = ctx.reference.teacher_force(ids, top_k)
                kl, delta, top1, shared = position_stats(ref, tgt)
                per_doc_kl.append(kl)
                per_doc_delta.append(delta[~np.isnan(delta)])
                per_doc_abs.append(np.abs(delta[~np.isnan(delta)]))
                per_doc_top1.append(top1)
                per_doc_shared.append(shared)

            boot = cluster_bootstrap_mean(per_doc_kl, seed=ctx.seed)
            hi = float(np.quantile(boot.boot, 1 - alpha_look))
            lo = float(np.quantile(boot.boot, alpha_look))
            res = boot
            if lo > threshold:
                status = Status.DEVIATION
                break
            if hi < threshold:
                status = Status.PASS
                break
            status = Status.INCONCLUSIVE

        assert res is not None
        alpha_final = alpha_look
        ci = (
            float(np.quantile(res.boot, alpha_final)),
            float(np.quantile(res.boot, 1 - alpha_final)),
        )
        all_top1 = np.concatenate(per_doc_top1)
        all_delta = np.concatenate(per_doc_delta) if per_doc_delta else np.array([])
        all_abs = np.concatenate(per_doc_abs) if per_doc_abs else np.array([])
        n_pos = int(sum(len(x) for x in per_doc_kl))
        metrics = {
            "kl_lower_bound_mean": res.estimate,
            "kl_ci_lo": ci[0],
            "kl_ci_hi": ci[1],
            "top1_agreement": float(all_top1.mean()),
            "positions": float(n_pos),
            "documents": float(len(per_doc_kl)),
            "mean_ref_mass_in_bound": float(np.concatenate(per_doc_shared).mean()),
            "delta_known_fraction": float(len(all_delta) / n_pos),
            "mean_delta": float(all_delta.mean()) if len(all_delta) else float("nan"),
            "mean_abs_delta": float(all_abs.mean()) if len(all_abs) else float("nan"),
            "threshold": threshold,
        }
        msg = (
            f"mean KL lower bound {res.estimate:.5f} nats "
            f"(one-sided CI {ci[0]:.5f} to {ci[1]:.5f}) vs threshold {threshold:g}; "
            f"top-1 agreement {metrics['top1_agreement']:.3f}"
        )
        sev = {"DEVIATION": "deviation", "PASS": "info", "INCONCLUSIVE": "warn"}[status.value]
        findings.append(Finding(severity=sev, message=msg))
        return self.result(
            status,
            p_value=res.p_value_le(threshold),
            effect=res.estimate,
            effect_unit="nats (mean KL lower bound)",
            ci=ci,
            threshold=threshold,
            n_requests=n_requests,
            metrics=metrics,
            findings=findings,
            notes=["KL is a lower bound from truncated top-k lists; true divergence may be larger"],
        )

    async def _chat_mode(self, ctx: ProbeContext) -> ProbeResult:
        assert ctx.reference is not None
        opts = ctx.options
        top_k = min(int(opts.get("distribution_top_k", 100)), ctx.adapter.capabilities.max_top_k)
        threshold = float(opts.get("thresholds", {}).get("kl_chat_mean", DEFAULT_KL_THRESHOLD))
        try:
            docs, n_requests, notes = await chat_documents(
                ctx, top_k, int(opts.get("distribution_chat_tokens", 32))
            )
        except EndpointError as exc:
            return self.result(
                Status.INCONCLUSIVE,
                findings=[Finding(severity="warn", message=f"chat request failed: {exc}")],
            )
        per_doc_kl: list[np.ndarray] = []
        per_doc_top1: list[np.ndarray] = []
        for full, start, tgt in docs:
            ref = ctx.reference.teacher_force(full, top_k)
            sl = {k: v[start : start + len(tgt)] for k, v in ref.items()}
            kl, _, top1, _ = position_stats(sl, tgt)
            per_doc_kl.append(kl)
            per_doc_top1.append(top1)
        if len(per_doc_kl) < 5:
            return self.result(
                Status.SKIPPED if not per_doc_kl else Status.INCONCLUSIVE,
                n_requests=n_requests,
                notes=[*notes, f"only {len(per_doc_kl)} chat replies could be aligned"],
            )
        boot = cluster_bootstrap_mean(per_doc_kl, seed=ctx.seed)
        lo = float(np.quantile(boot.boot, ctx.alpha))
        hi = float(np.quantile(boot.boot, 1 - ctx.alpha))
        status = (
            Status.DEVIATION if lo > threshold else Status.PASS if hi < threshold
            else Status.INCONCLUSIVE
        )  # fmt: skip
        top1_rate = float(np.concatenate(per_doc_top1).mean())
        metrics = {
            "chat_mode": 1.0,
            "kl_lower_bound_mean": boot.estimate,
            "kl_ci_lo": lo,
            "kl_ci_hi": hi,
            "top1_agreement": top1_rate,
            "positions": float(sum(len(x) for x in per_doc_kl)),
            "documents": float(len(per_doc_kl)),
            "threshold": threshold,
        }
        sev = {"DEVIATION": "deviation", "PASS": "info", "INCONCLUSIVE": "warn"}[status.value]
        msg = (
            f"chat mode: mean KL lower bound {boot.estimate:.5f} nats over "
            f"{len(per_doc_kl)} replies (one-sided CI {lo:.5f} to {hi:.5f}) vs threshold "
            f"{threshold:g}; top-1 agreement {top1_rate:.3f}"
        )
        return self.result(
            status,
            p_value=boot.p_value_le(threshold),
            effect=boot.estimate,
            effect_unit="nats (mean KL lower bound, chat mode)",
            ci=(lo, hi),
            threshold=threshold,
            n_requests=n_requests,
            metrics=metrics,
            findings=[Finding(severity=sev, message=msg)],
            notes=[
                *notes,
                "chat mode: compared on the endpoint's own temperature-0 replies, aligned to the "
                "reference by token bytes; the server's chat template is part of the comparison",
                "KL is a lower bound from truncated top-k lists; true divergence may be larger",
            ],
        )
