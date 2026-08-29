# -*- coding: utf-8 -*-
"""
Privileged Deep 4PL IRT for trajectory-aware DSF + residual LUPI estimation.

Deep-IRT style one-hot student/item nets with a 4PL response (default).
Summary embedding produces a restricted interaction residual (Δa, Δb) only:

  4PL (default, add): a_ij = a_j + Δa,     b_ij = b_j + Δb; c_j, d_j global
  4PL (ablation mul): a_ij = a_j * exp(Δa), b_ij = b_j + Δb
  1PL (ablation):     b_ij = b_j + Δb; a=1, c=0, d=1 fixed

Δa, Δb = α tanh(g(s)); Student deploy uses (a_j,b_j,c_j,d_j) with no summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

RESPONSE_TYPES = ("1pl", "4pl")
A_DELTA_MODES = ("mul", "add")


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def prob_4pl(
    theta: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Elementwise 4PL probability; arrays broadcast like (n_agents, n_items)."""
    z = a * (theta - b)
    return c + (d - c) * _sigmoid_np(z)


class Privileged_Deep_4PL_IRT(nn.Module):
    """Deep-IRT 4PL backbone; summary residual (Δa, Δb) on privileged cells."""

    def __init__(
        self,
        num_students: int,
        num_items: int,
        embed_dim: int = 384,
        hidden_dim: int = 64,
        delta_scale: float = 0.5,
        response_type: str = "4pl",
        a_delta_mode: str = "add",
        use_summary_corr: bool = True,
    ):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.response_type = str(response_type).lower()
        self.a_delta_mode = str(a_delta_mode).lower()
        self.use_summary_corr = bool(use_summary_corr)
        if self.response_type not in RESPONSE_TYPES:
            raise ValueError(f"response_type must be one of {RESPONSE_TYPES}")
        if self.a_delta_mode not in A_DELTA_MODES:
            raise ValueError(f"a_delta_mode must be one of {A_DELTA_MODES}")

        self.student_net = nn.Sequential(
            nn.Linear(num_students, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        item_out = 1 if self.response_type == "1pl" else 4
        self.item_net = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, item_out),
        )
        if self.use_summary_corr:
            corr_out = 1 if self.response_type == "1pl" else 2
            self.summary_corr = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, corr_out),
            )
        else:
            self.summary_corr = None

    def _item_params(self, item: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.item_net(item)
        if self.response_type == "1pl":
            b = raw[:, 0:1]
            a = torch.ones_like(b)
            c = torch.zeros_like(b)
            d = torch.ones_like(b)
            return a, b, c, d
        b = raw[:, 0:1]
        a = F.softplus(raw[:, 1:2]) + 0.1
        c = torch.sigmoid(raw[:, 2:3]) * 0.3
        d = c + torch.sigmoid(raw[:, 3:4]) * (1.0 - c)
        return a, b, c, d

    def _summary_delta(
        self, summary_embed: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if not self.use_summary_corr or self.summary_corr is None:
            n = summary_embed.shape[0]
            device = summary_embed.device
            db = torch.zeros(n, 1, device=device, dtype=summary_embed.dtype)
            if self.response_type == "1pl":
                return None, db
            return torch.zeros(n, 1, device=device, dtype=summary_embed.dtype), db
        out = self.summary_corr(summary_embed)
        if self.response_type == "1pl":
            db = torch.tanh(out[:, 0:1]) * self.delta_scale
            return None, db
        da = torch.tanh(out[:, 0:1]) * self.delta_scale
        db = torch.tanh(out[:, 1:2]) * self.delta_scale
        return da, db

    @staticmethod
    def _irt_prob(
        theta: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
        z = a * (theta - b)
        return (c + (d - c) * torch.sigmoid(z)).clamp(1e-6, 1.0 - 1e-6)

    def forward(
        self,
        student: torch.Tensor,
        item: torch.Tensor,
        summary_embed: torch.Tensor | None = None,
        privileged: torch.Tensor | None = None,
    ) -> torch.Tensor:
        theta = self.student_net(student)
        a, b, c, d = self._item_params(item)
        if (
            self.use_summary_corr
            and self.summary_corr is not None
            and summary_embed is not None
            and privileged is not None
        ):
            da, db = self._summary_delta(summary_embed)
            mask = privileged.unsqueeze(1) if privileged.ndim == 1 else privileged
            if da is not None:
                if self.a_delta_mode == "mul":
                    a = a * torch.exp(da * mask)
                else:
                    a = a + da * mask
            b = b + db * mask
        return self._irt_prob(theta, a, b, c, d)


class PrivilegedDeepTrainDataset(Dataset):
    """Train cells with optional summary for privileged parameter correction."""

    def __init__(
        self,
        df: pd.DataFrame,
        s_emb: torch.Tensor,
        summary_mask: torch.Tensor,
        quality_w: torch.Tensor,
        train_agents: list[int],
        min_weight: float = 0.05,
    ):
        self.num_students = len(df)
        self.num_items = len(df.columns)
        self.s_emb = s_emb
        self.data: list[tuple[int, int, float, float, float]] = []
        allowed = set(train_agents)
        sm = summary_mask.numpy() if isinstance(summary_mask, torch.Tensor) else summary_mask
        qw = quality_w.numpy() if isinstance(quality_w, torch.Tensor) else quality_w
        for sid in range(self.num_students):
            if sid not in allowed:
                continue
            for qid in range(self.num_items):
                v = df.iloc[sid, qid]
                if pd.isnull(v):
                    continue
                w = float(qw[sid, qid]) if sm[sid, qid] > 0.5 else 1.0
                priv = 1.0 if sm[sid, qid] > 0.5 and w >= min_weight else 0.0
                self.data.append((sid, qid, float(v), w, priv))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sid, qid, label, w, priv = self.data[idx]
        student = torch.zeros(self.num_students, dtype=torch.float32)
        student[sid] = 1.0
        item = torch.zeros(self.num_items, dtype=torch.float32)
        item[qid] = 1.0
        return {
            "student": student,
            "item": item,
            "summary_embed": self.s_emb[sid, qid].clone(),
            "privileged": torch.tensor(priv, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "weight": torch.tensor(w, dtype=torch.float32),
        }


def train_privileged_deep_4pl(
    df: pd.DataFrame,
    s_emb: torch.Tensor,
    summary_mask: np.ndarray,
    quality_w: np.ndarray,
    train_agents: list[int],
    device: torch.device,
    max_epochs: int,
    seed: int,
    *,
    min_weight: float = 0.05,
    delta_scale: float = 0.5,
    delta_l2: float = 0.01,
    response_type: str = "4pl",
    early_stop_patience: int = 4,
    lupi_lambda_teacher: float | None = None,
    lupi_lambda_kl: float | None = None,
    a_delta_mode: str | None = None,
    lr: float = 0.001,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    use_summary_corr: bool = True,
) -> Privileged_Deep_4PL_IRT:
    """
    Train the shared Privileged Deep 4PL backbone (Student + Teacher heads).

    Default a_delta_mode follows Privileged_Deep_LUPI_IRT.DEFAULT_A_DELTA_MODE (mul).
    """
    from pta.models.Privileged_Deep_LUPI_IRT import (  # noqa: WPS433
        DEFAULT_A_DELTA_MODE,
        DEFAULT_LAMBDA_KL,
        DEFAULT_LAMBDA_TEACHER,
        train_privileged_deep_4pl_backbone,
    )

    if a_delta_mode is None:
        a_delta_mode = DEFAULT_A_DELTA_MODE

    lam_t = DEFAULT_LAMBDA_TEACHER if lupi_lambda_teacher is None else float(lupi_lambda_teacher)
    lam_kl = DEFAULT_LAMBDA_KL if lupi_lambda_kl is None else float(lupi_lambda_kl)
    return train_privileged_deep_4pl_backbone(
        df,
        s_emb,
        summary_mask,
        quality_w,
        train_agents,
        device,
        max_epochs,
        seed,
        min_weight=min_weight,
        delta_scale=delta_scale,
        delta_l2=delta_l2,
        lupi_lambda_teacher=lam_t,
        lupi_lambda_kl=lam_kl,
        response_type=response_type,
        early_stop_patience=early_stop_patience,
        a_delta_mode=a_delta_mode,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        use_summary_corr=use_summary_corr,
    )


@torch.no_grad()
def item_base_params(
    model: Privileged_Deep_4PL_IRT, n_items: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_list, b_list, c_list, d_list = [], [], [], []
    for qid in range(n_items):
        ivec = torch.zeros(n_items, dtype=torch.float32, device=device)
        ivec[qid] = 1.0
        a, b, c, d = model._item_params(ivec.unsqueeze(0))
        a_list.append(float(a.item()))
        b_list.append(float(b.item()))
        c_list.append(float(c.item()))
        d_list.append(float(d.item()))
    return (
        np.array(a_list, dtype=np.float64),
        np.array(b_list, dtype=np.float64),
        np.array(c_list, dtype=np.float64),
        np.array(d_list, dtype=np.float64),
    )


@torch.no_grad()
def _cell_fisher_4pl(
    theta: float,
    a_ij: np.ndarray,
    b_ij: np.ndarray,
    c_j: np.ndarray,
    d_j: np.ndarray,
) -> np.ndarray:
    """Item-wise Fisher I_j(θ) = (dP/dθ)² / (P(1-P)) for 4PL with privileged (a,b)."""
    z = np.clip(a_ij * (theta - b_ij), -40.0, 40.0)
    sigma = 1.0 / (1.0 + np.exp(-z))
    p = c_j + (d_j - c_j) * sigma
    dp_dtheta = (d_j - c_j) * a_ij * sigma * (1.0 - sigma)
    denom = np.maximum(p * (1.0 - p), 1e-12)
    return (dp_dtheta ** 2) / denom


@torch.no_grad()
def privileged_4pl_fisher_item_scores(
    model: Privileged_Deep_4PL_IRT,
    train_agents: list[int],
    n_students: int,
    n_items: int,
    device: torch.device,
    s_emb: torch.Tensor,
    summary_mask: np.ndarray,
    quality_w: np.ndarray,
    *,
    min_weight: float = 0.05,
) -> np.ndarray:
    """Cell-wise 4PL Fisher I_ij(θ) = (P')²/(P(1-P)), mean over train agents per item."""
    a_base, b_base, c_base, d_base = item_base_params(model, n_items, device)
    aids = np.asarray(train_agents, dtype=int)

    se = s_emb if isinstance(s_emb, torch.Tensor) else torch.tensor(s_emb, dtype=torch.float32)
    se = se.to(device)
    sm = summary_mask
    qw = quality_w

    fisher_sum = np.zeros(n_items, dtype=np.float64)
    weight_sum = np.zeros(n_items, dtype=np.float64)

    for aid in aids:
        svec = torch.zeros(n_students, dtype=torch.float32, device=device)
        svec[aid] = 1.0
        theta = float(model.student_net(svec.unsqueeze(0)).item())

        summ_row = se[aid]
        if getattr(model, "use_summary_corr", True) and getattr(model, "summary_corr", None) is not None:
            corr_out = model.summary_corr(summ_row)
            if model.response_type == "1pl":
                da = np.zeros(n_items, dtype=np.float64)
                db = (torch.tanh(corr_out[:, 0]) * model.delta_scale).cpu().numpy()
            else:
                da = (torch.tanh(corr_out[:, 0]) * model.delta_scale).cpu().numpy()
                db = (torch.tanh(corr_out[:, 1]) * model.delta_scale).cpu().numpy()
            priv = (sm[aid, :n_items] > 0.5) & (qw[aid, :n_items] >= min_weight)
            a_ij = a_base.copy()
            b_ij = b_base.copy()
            if getattr(model, "a_delta_mode", "mul") == "mul":
                a_ij[priv] = a_base[priv] * np.exp(da[priv])
            else:
                a_ij[priv] = a_base[priv] + da[priv]
            b_ij[priv] = b_base[priv] + db[priv]
            w = np.where(priv, qw[aid, :n_items], 1.0).astype(np.float64)
        else:
            a_ij, b_ij = a_base, b_base
            w = np.ones(n_items, dtype=np.float64)

        i_ij = _cell_fisher_4pl(theta, a_ij, b_ij, c_base, d_base)
        fisher_sum += i_ij * w
        weight_sum += w

    weight_sum = np.maximum(weight_sum, 1e-8)
    return fisher_sum / weight_sum
