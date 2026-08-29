# -*- coding: utf-8 -*-
"""
Unified Privileged Deep 4PL backbone: Teacher / Student / Fisher heads.

Shared: student_net, item_net, summary_corr
  - forward_teacher()  -> BCE + KL (training / calib distillation)
  - forward_student()  -> BCE (deploy / test calibration)
  - fisher_item_scores() -> 4PL Fisher readout (item selection only; not in loss)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from pta.models.Privileged_Deep_4PL_IRT import (  # noqa: I001
    PrivilegedDeepTrainDataset,
    Privileged_Deep_4PL_IRT,
    privileged_4pl_fisher_item_scores,
)


def bernoulli_kl(p_teacher: torch.Tensor, p_student: torch.Tensor) -> torch.Tensor:
    pt = p_teacher.clamp(1e-6, 1.0 - 1e-6)
    ps = p_student.clamp(1e-6, 1.0 - 1e-6)
    return pt * (pt.log() - ps.log()) + (1.0 - pt) * ((1.0 - pt).log() - (1.0 - ps).log())


# Privileged-LUPI defaults (unified a_delta_mode=add; Student BCE unweighted by summary quality)
DEFAULT_DELTA_SCALE = 1.5
DEFAULT_DELTA_L2 = 0.01
DEFAULT_MIN_WEIGHT = 0.03
DEFAULT_LAMBDA_TEACHER = 0.7  # t07_kc045 (_hp_ref_lupi)
# Keep standalone DSF training aligned with the canonical Privileged-LUPI
# estimator defaults (unified a_delta_mode=add).
DEFAULT_LAMBDA_KL = 0.75  # t07_kc045 (_hp_ref_lupi)
DEFAULT_A_DELTA_MODE = "add"

DSF_DELTA_SCALE = DEFAULT_DELTA_SCALE
DSF_DELTA_L2 = DEFAULT_DELTA_L2
DSF_MIN_WEIGHT = DEFAULT_MIN_WEIGHT


class Privileged_Deep_LUPI_IRT(Privileged_Deep_4PL_IRT):
    """Privileged Deep 4PL with Student / Teacher / Fisher readouts."""

    def forward_student(self, student: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """Student head: global (a_j, b_j), no summary."""
        return super().forward(student, item, None, None)

    def forward_teacher(
        self,
        student: torch.Tensor,
        item: torch.Tensor,
        summary_embed: torch.Tensor,
        privileged: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher head: cell-level (a_ij, b_ij) when privileged."""
        if privileged is None:
            privileged = torch.ones(
                student.shape[0], dtype=torch.float32, device=student.device
            )
        return super().forward(student, item, summary_embed, privileged)

    @torch.no_grad()
    def fisher_item_scores(
        self,
        train_agents: list[int],
        n_students: int,
        n_items: int,
        device: torch.device,
        s_emb: torch.Tensor,
        summary_mask: np.ndarray,
        quality_w: np.ndarray,
        *,
        min_weight: float = DEFAULT_MIN_WEIGHT,
    ) -> np.ndarray:
        """Fisher head: strict 4PL I(θ); used for item selection only."""
        return privileged_4pl_fisher_item_scores(
            self,
            train_agents,
            n_students,
            n_items,
            device,
            s_emb,
            summary_mask,
            quality_w,
            min_weight=min_weight,
        )


def evaluate_student(
    model: Privileged_Deep_LUPI_IRT, dataloader: DataLoader, device: torch.device
) -> dict:
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            p = model.forward_student(
                batch["student"].to(device),
                batch["item"].to(device),
            ).squeeze()
            probs.extend(p.cpu().numpy().tolist())
            labels.extend(batch["label"].cpu().numpy().tolist())
    preds = [1 if x > 0.5 else 0 for x in probs]
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.5
    return {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": auc,
    }


def train_privileged_deep_4pl_backbone(
    df: pd.DataFrame,
    s_emb: torch.Tensor,
    summary_mask: np.ndarray,
    quality_w: np.ndarray,
    train_agents: list[int],
    device: torch.device,
    max_epochs: int,
    seed: int,
    *,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    delta_scale: float = DEFAULT_DELTA_SCALE,
    delta_l2: float = DEFAULT_DELTA_L2,
    lupi_lambda_teacher: float = DEFAULT_LAMBDA_TEACHER,
    lupi_lambda_kl: float = DEFAULT_LAMBDA_KL,
    hidden_dim: int = 64,
    response_type: str = "4pl",
    early_stop_patience: int = 4,
    a_delta_mode: str = DEFAULT_A_DELTA_MODE,
    lr: float = 0.001,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    use_summary_corr: bool = True,
) -> Privileged_Deep_LUPI_IRT:
    """
    Single training loop for the shared backbone.

    Student head: BCE on all cells (q-only / one-hot item params).
    Teacher head: BCE + optional weak KL on privileged cells (summary → Δa, Δb).
    Fisher head: not used in loss; call model.fisher_item_scores() after training.
    If use_summary_corr=False: no summary_corr module; Student BCE only (same HP otherwise).
    """
    torch.manual_seed(seed)
    sm = torch.tensor(summary_mask, dtype=torch.float32)
    qw = torch.tensor(quality_w, dtype=torch.float32)
    se = s_emb if isinstance(s_emb, torch.Tensor) else torch.tensor(s_emb, dtype=torch.float32)
    rt = str(response_type).lower()
    bs = max(1, int(batch_size))
    with_corr = bool(use_summary_corr)

    full_train = PrivilegedDeepTrainDataset(df, se, sm, qw, train_agents, min_weight=min_weight)
    if len(full_train) == 0:
        raise RuntimeError("No training cells for Privileged_Deep_LUPI_IRT")

    train_size = max(1, int(0.8 * len(full_train)))
    train_set, val_set = random_split(full_train, [train_size, len(full_train) - train_size])
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs)

    model = Privileged_Deep_LUPI_IRT(
        len(df),
        len(df.columns),
        embed_dim=int(se.shape[-1]),
        hidden_dim=hidden_dim,
        delta_scale=delta_scale,
        response_type=rt,
        a_delta_mode=a_delta_mode,
        use_summary_corr=with_corr,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.BCELoss(reduction="none")
    best_f1, best_state, stale = 0.0, None, 0
    tag = "Priv4PL-StuOnly" if not with_corr else "Priv4PL-Backbone"

    for epoch in range(max_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"{tag} e{epoch+1}", leave=False):
            optimizer.zero_grad()
            student = batch["student"].to(device)
            item = batch["item"].to(device)
            y = batch["label"].to(device)

            p_s = model.forward_student(student, item).squeeze()
            # Student = deploy head: weight 1 (do not bias by summary quality_w).
            loss = criterion(p_s, y)

            if with_corr:
                summ = batch["summary_embed"].to(device)
                priv = batch["privileged"].to(device)
                w = batch["weight"].to(device)
                if priv.sum() > 0:
                    p_t = model.forward_teacher(student, item, summ, priv).squeeze()
                    priv_w = w * priv
                    loss = loss + lupi_lambda_teacher * criterion(p_t, y) * priv_w
                    kl = bernoulli_kl(p_t.detach(), p_s) * priv_w
                    loss = loss + lupi_lambda_kl * kl
                    da, db = model._summary_delta(summ)
                    reg = db.square().squeeze() * priv
                    if da is not None:
                        reg = reg + (da.square().squeeze() * priv)
                    loss = loss + delta_l2 * reg

            loss.mean().backward()
            optimizer.step()

        val_metrics = evaluate_student(model, val_loader, device)
        if val_metrics["f1"] > best_f1 + 1e-4:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= early_stop_patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


__all__ = [
    "Privileged_Deep_LUPI_IRT",
    "DEFAULT_DELTA_SCALE",
    "DEFAULT_DELTA_L2",
    "DEFAULT_MIN_WEIGHT",
    "DEFAULT_LAMBDA_TEACHER",
    "DEFAULT_LAMBDA_KL",
    "DEFAULT_A_DELTA_MODE",
    "DSF_DELTA_SCALE",
    "DSF_DELTA_L2",
    "DSF_MIN_WEIGHT",
    "evaluate_student",
    "train_privileged_deep_4pl_backbone",
]
