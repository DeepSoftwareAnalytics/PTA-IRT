"""
Agent-level evaluation metrics for PTA-IRT.

1. MAE — mean |ŷ − y|; smaller is better (score fidelity).
2. Kendall’s τ ∈ [-1, 1] — pairwise order concordance; closer to 1 is better.
3. Spearman’s ρ ∈ [-1, 1] — Pearson corr of ranks; closer to 1 is better.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.stats import kendalltau, spearmanr
except ImportError:  # pragma: no cover
    kendalltau = None  # type: ignore
    spearmanr = None  # type: ignore

METRIC_KEYS = ("mae", "kendall_tau", "spearman_rho")


def _is_constant(x: np.ndarray, *, eps: float = 1e-12) -> bool:
    """True if all finite values are (near) identical — corr/τ undefined."""
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2:
        return True
    return float(np.max(v) - np.min(v)) <= eps


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error between predicted and true agent scores."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yp - yt)))


def kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """
    Kendall’s τ and p-value.

    If either score vector is constant (zero variance), ranking correlation is
    undefined in scipy; return τ = 0 (no ranking signal) and p = NaN.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if kendalltau is None or yt.size < 2:
        return float("nan"), float("nan")
    if _is_constant(yt) or _is_constant(yp):
        return 0.0, float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        tau, p = kendalltau(yp, yt)
    return (
        float(tau) if tau == tau else 0.0,
        float(p) if p == p else float("nan"),
    )


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """
    Spearman’s ρ = corr(rank(ŷ), rank(y)) and p-value.

    Constant vectors → ρ = 0 (no ranking signal), matching kendall_tau.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if spearmanr is None or yt.size < 2:
        return float("nan"), float("nan")
    if _is_constant(yt) or _is_constant(yp):
        return 0.0, float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        rho, p = spearmanr(yp, yt)
    return (
        float(rho) if rho == rho else 0.0,
        float(p) if p == p else float("nan"),
    )


def score_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """
    Compute MAE, Kendall’s τ, and Spearman’s ρ for agent score vectors.

    Returns keys: mae, kendall_tau, kendall_p, spearman_rho, spearman_p, n_agents.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: true {yt.shape} vs pred {yp.shape}")
    tau, tau_p = kendall_tau(yt, yp)
    rho, rho_p = spearman_rho(yt, yp)
    return {
        "mae": mae(yt, yp),
        "kendall_tau": tau,
        "kendall_p": tau_p,
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "n_agents": int(yt.size),
    }


agent_ranking_metrics = score_metrics
