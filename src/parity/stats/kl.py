"""Truncated KL divergence between two next-token distributions.

Providers return only the top-k log-probabilities, so the full KL(P || Q) is not
observable. Let S be the set of token ids reported by *both* sides. Merging every other
token into one "rest" bin coarse-grains P and Q with the same partition, and by the
data-processing inequality

    KL(P || Q)  >=  sum_{i in S} p_i log(p_i / q_i)  +  p_r log(p_r / q_r)

with p_r = 1 - sum_S p_i and q_r = 1 - sum_S q_i. So the estimate is a rigorous *lower
bound* on the true KL, never an over-estimate. Its bias is downward and grows when the
reported sets are small or disagree (ids missing from one side are merged into the
rest bin and their divergence is lost). The same partition gives a lower bound on the
total variation distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class TruncatedKL:
    kl: float  # lower bound on KL(P || Q), in nats
    tv: float  # lower bound on TV(P, Q)
    shared: int  # number of ids in both reported sets
    ref_mass_shared: float  # mass of P on the shared ids (how much of P the bound sees)
    top1_match: bool  # both sides rank the same token first


def truncated_kl(
    p_ids: np.ndarray, p_lp: np.ndarray, q_ids: np.ndarray, q_lp: np.ndarray
) -> TruncatedKL:
    """KL lower bound of ``P`` (reference) against ``Q`` (target) from reported top-k lists."""
    q_map = {int(i): float(lp) for i, lp in zip(q_ids, q_lp, strict=True)}
    shared_p: list[float] = []
    shared_q: list[float] = []
    for i, lp in zip(p_ids, p_lp, strict=True):
        if int(i) in q_map:
            shared_p.append(float(np.exp(lp)))
            shared_q.append(float(np.exp(q_map[int(i)])))
    p = np.array(shared_p)
    q = np.array(shared_q)
    p_r = max(1.0 - float(p.sum()), _EPS)
    q_r = max(1.0 - float(q.sum()), _EPS)
    safe_q = np.maximum(q, _EPS)
    kl = float(np.sum(p * np.log(np.maximum(p, _EPS) / safe_q))) + p_r * float(np.log(p_r / q_r))
    tv = 0.5 * (float(np.abs(p - q).sum()) + abs(p_r - q_r))
    top1 = bool(
        len(p_ids) and len(q_ids) and int(p_ids[np.argmax(p_lp)]) == int(q_ids[np.argmax(q_lp)])
    )
    return TruncatedKL(
        kl=max(kl, 0.0), tv=tv, shared=len(p), ref_mass_shared=float(p.sum()), top1_match=top1
    )
