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
"""

from __future__ import annotations

import asyncio
import math

import numpy as np

from parity.client import EndpointError
from parity.probes.base import Finding, Probe, ProbeContext, ProbeResult, Status
from parity.schema import PositionLogprobs
from parity.stats.bootstrap import cluster_bootstrap_mean
from parity.stats.kl import truncated_kl
from parity.suites import load_corpus

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


class DistributionProbe(Probe):
    id = "distribution.teacher_forced"
    family = "distribution"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        caps = ctx.adapter.capabilities
        if ctx.reference is None:
            return self.result(Status.SKIPPED, notes=["no reference implementation configured"])
        if not (caps.token_id_prompts and caps.top_logprobs):
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
        threshold = float(opts.get("thresholds", {}).get("kl_mean", DEFAULT_KL_THRESHOLD))

        docs = [
            ctx.kit.encode(d.text)[:max_tokens] for d in load_corpus(opts.get("suite", "core-v1"))
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

        for look in range(n_looks):
            batch = docs[look * chunk : (look + 1) * chunk]
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
