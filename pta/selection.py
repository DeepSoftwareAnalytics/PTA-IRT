#!/usr/bin/env python3
"""DSF item selection: Privileged Deep 4PL Fisher × reliability, difficulty-stratified."""

from __future__ import annotations

import numpy as np
import torch

from pta.models.Privileged_Deep_4PL_IRT import privileged_4pl_fisher_item_scores
from pta.models.Privileged_Deep_LUPI_IRT import (
    DSF_DELTA_L2,
    DSF_DELTA_SCALE,
    DSF_MIN_WEIGHT,
)

RELIABILITY_MODES = ("log_ess", "sqrt_ess", "sqrt_n", "none")
DSF_RELIABILITY_MODE = "log_ess"


def compute_item_information_reliability(
    summary_mask: np.ndarray,
    quality_w: np.ndarray,
    train_agents: list[int],
    n_items: int,
    *,
    min_weight: float = 0.05,
    mode: str = "log_ess",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability weight from privileged-cell ESS / count (not summary embeddings)."""
    if mode not in RELIABILITY_MODES:
        raise ValueError(f"mode must be one of {RELIABILITY_MODES}")
    aids = np.asarray(train_agents, dtype=int)
    privileged = (summary_mask[aids, :n_items] >= 0.5) & (quality_w[aids, :n_items] >= min_weight)
    n_priv = privileged.sum(axis=0).astype(np.float64)
    ess = np.zeros(n_items, dtype=np.float64)
    for j in range(n_items):
        wj = quality_w[aids, j][privileged[:, j]]
        ess[j] = float(wj.sum()) if wj.size else 0.0

    if mode == "sqrt_ess":
        rel = np.sqrt(np.maximum(ess, 0.0))
    elif mode == "log_ess":
        rel = np.log1p(ess)
    elif mode == "sqrt_n":
        rel = np.sqrt(n_priv)
    else:
        rel = np.ones(n_items, dtype=np.float64)
    return rel, n_priv, ess


def privileged_4pl_summary_fisher_scores(
    model,
    train_agents: list[int],
    n_students: int,
    n_items: int,
    device: torch.device,
    s_emb: torch.Tensor,
    summary_mask: np.ndarray,
    quality_w: np.ndarray,
    *,
    min_weight: float = 0.05,
    reliability_mode: str = "log_ess",
) -> np.ndarray:
    """score_j = Fisher_j × reliability_j."""
    fisher = privileged_4pl_fisher_item_scores(
        model,
        train_agents,
        n_students,
        n_items,
        device,
        s_emb,
        summary_mask,
        quality_w,
        min_weight=min_weight,
    )
    rel, _, _ = compute_item_information_reliability(
        summary_mask,
        quality_w,
        train_agents,
        n_items,
        min_weight=min_weight,
        mode=reliability_mode,
    )
    return fisher * rel


def train_item_pass_rates(y: np.ndarray, train_agents: list[int], n_items: int) -> np.ndarray:
    """Per-item mean pass rate on train agents."""
    aids = np.asarray(train_agents, dtype=int)
    return y[np.ix_(aids, np.arange(n_items))].mean(axis=0).astype(np.float64)


def _allocate_bin_quotas(bin_sizes: np.ndarray, n_select: int) -> np.ndarray:
    """Largest-remainder quotas proportional to bin sizes."""
    k = len(bin_sizes)
    if k == 0 or n_select <= 0:
        return np.zeros(k, dtype=int)
    total = int(bin_sizes.sum())
    if total <= 0:
        return np.zeros(k, dtype=int)
    raw = bin_sizes.astype(np.float64) / total * n_select
    base = np.floor(raw).astype(int)
    rem = n_select - int(base.sum())
    order = np.argsort(-(raw - base))
    for i in range(rem):
        base[order[i % k]] += 1
    nonempty = np.where(bin_sizes > 0)[0]
    if len(nonempty) <= n_select:
        for b in nonempty:
            if base[b] == 0:
                donor = int(np.argmax(base))
                if base[donor] > 1:
                    base[donor] -= 1
                    base[b] += 1
    for b in range(k):
        if base[b] > bin_sizes[b]:
            overflow = int(base[b] - bin_sizes[b])
            base[b] = int(bin_sizes[b])
            for _ in range(overflow):
                spare = [i for i in range(k) if bin_sizes[i] > base[i]]
                if not spare:
                    break
                j = max(spare, key=lambda i: int(bin_sizes[i] - base[i]))
                base[j] += 1
    while int(base.sum()) < n_select:
        spare = [i for i in range(k) if bin_sizes[i] > base[i]]
        if not spare:
            break
        j = max(spare, key=lambda i: (int(bin_sizes[i] - base[i]), int(bin_sizes[i])))
        base[j] += 1
    return base


def _allocate_equal_quotas(bin_sizes: np.ndarray, n_select: int) -> np.ndarray:
    """Near-equal quotas across nonempty bins, capped by bin size."""
    k = len(bin_sizes)
    base = np.zeros(k, dtype=int)
    nonempty = [i for i in range(k) if bin_sizes[i] > 0]
    if not nonempty or n_select <= 0:
        return base
    q, rem = divmod(n_select, len(nonempty))
    for i, b in enumerate(nonempty):
        base[b] = q + (1 if i < rem else 0)
    for b in nonempty:
        if base[b] > bin_sizes[b]:
            overflow = int(base[b] - bin_sizes[b])
            base[b] = int(bin_sizes[b])
            for _ in range(overflow):
                spare = [i for i in nonempty if bin_sizes[i] > base[i]]
                if not spare:
                    break
                j = max(spare, key=lambda i: int(bin_sizes[i] - base[i]))
                base[j] += 1
    while int(base.sum()) < n_select:
        spare = [i for i in nonempty if bin_sizes[i] > base[i]]
        if not spare:
            break
        j = max(spare, key=lambda i: int(bin_sizes[i] - base[i]))
        base[j] += 1
    return base


def _bin_ids_by_difficulty(
    item_pass_rates: np.ndarray,
    n_bins: int,
    *,
    binning: str = "quantile",
) -> np.ndarray:
    n_items = len(item_pass_rates)
    bin_id = np.empty(n_items, dtype=int)
    if binning == "quantile":
        order_p = np.argsort(item_pass_rates)
        edges = np.linspace(0, n_items, n_bins + 1, dtype=int)
        for b in range(n_bins):
            members = order_p[edges[b] : edges[b + 1]]
            bin_id[members] = b
        return bin_id
    if binning == "width":
        pmin = float(item_pass_rates.min())
        pmax = float(item_pass_rates.max())
        if pmax <= pmin + 1e-12:
            return np.zeros(n_items, dtype=int)
        scaled = (item_pass_rates - pmin) / (pmax - pmin)
        bin_id = np.floor(scaled * n_bins).astype(int)
        return np.clip(bin_id, 0, n_bins - 1)
    raise ValueError(f"binning must be 'quantile' or 'width', got {binning!r}")


def select_items_difficulty_stratified(
    scores: np.ndarray,
    item_pass_rates: np.ndarray,
    n_select: int,
    *,
    n_bins: int = 5,
    binning: str = "quantile",
    quota: str = "prop",
    candidate_mult: float | None = None,
) -> list[int]:
    """
    High-information + difficulty-stratified selection.

    binning: 'quantile' (equal-frequency) | 'width' (equal pass-rate width)
    quota: 'prop' (proportional to bin size) | 'equal'
    """
    n_items = len(scores)
    n_select = max(1, min(int(n_select), n_items))
    pool = np.arange(n_items)
    if candidate_mult is not None and float(candidate_mult) > 1.0:
        m = max(n_select, min(n_items, int(round(float(candidate_mult) * n_select))))
        pool = np.argsort(scores)[::-1][:m]
    n_bins = max(1, min(int(n_bins), n_select, len(pool)))
    rates_pool = item_pass_rates[pool]
    local_bins = _bin_ids_by_difficulty(rates_pool, n_bins, binning=binning)
    sizes = np.array([(local_bins == b).sum() for b in range(n_bins)], dtype=int)
    if quota == "equal":
        quotas = _allocate_equal_quotas(sizes, n_select)
    elif quota == "prop":
        quotas = _allocate_bin_quotas(sizes, n_select)
    else:
        raise ValueError(f"quota must be 'prop' or 'equal', got {quota!r}")
    selected: list[int] = []
    for b in range(n_bins):
        members_local = np.where(local_bins == b)[0]
        if len(members_local) == 0 or quotas[b] <= 0:
            continue
        members = pool[members_local]
        ranked = members[np.argsort(scores[members])[::-1]]
        selected.extend(int(i) for i in ranked[: int(quotas[b])])
    if len(selected) < n_select:
        pool_rest = [i for i in pool.tolist() if i not in set(selected)]
        pool_rest.sort(key=lambda i: scores[i], reverse=True)
        selected.extend(pool_rest[: n_select - len(selected)])
    elif len(selected) > n_select:
        selected = sorted(selected, key=lambda i: scores[i], reverse=True)[:n_select]
    return sorted(selected)


__all__ = [
    "DSF_DELTA_L2",
    "DSF_DELTA_SCALE",
    "DSF_MIN_WEIGHT",
    "DSF_RELIABILITY_MODE",
    "RELIABILITY_MODES",
    "compute_item_information_reliability",
    "privileged_4pl_summary_fisher_scores",
    "select_items_difficulty_stratified",
    "train_item_pass_rates",
]
