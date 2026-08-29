# -*- coding: utf-8 -*-
"""PTA-IRT estimation / CV (Privileged Deep 4PL-LUPI main path only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pta.frozen_calib import load_frozen_calib_by_frac
from pta.metrics.ranking import score_metrics
from pta.metrics.ranking import score_metrics as agent_ranking_metrics
from pta.models.Privileged_Deep_4PL_IRT import item_base_params, prob_4pl
from pta.models.Privileged_Deep_LUPI_IRT import (
    DEFAULT_A_DELTA_MODE,
    DEFAULT_DELTA_L2,
    DEFAULT_DELTA_SCALE,
    Privileged_Deep_LUPI_IRT,
    bernoulli_kl,
    train_privileged_deep_4pl_backbone,
)
from pta.selection import DSF_DELTA_L2, DSF_DELTA_SCALE, DSF_MIN_WEIGHT

MODEL_NAME = "Privileged-Deep-4PL-LUPI-IRT"
CALIB_FRACS = [0.05, 0.10, 0.15, 0.20]

# Defaults aligned with results/_hp_ref_lupi.json
LUPI_LAMBDA_TEACHER = 0.7
LUPI_LAMBDA_KL = 0.75
LUPI_LAMBDA_KL_CALIB = 0.45
BACKBONE_LR = 0.001
BACKBONE_BATCH_SIZE = 256
BACKBONE_WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 4
CALIB_LR = 0.1
CALIB_OPTIMIZER = "lbfgs"
CALIB_LBFGS_MAX_ITER = 20
CALIB_THETA_L2 = 0.01
HIDDEN_DIM = 64
MIN_TRAIN_WEIGHT = 0.05

HYPERPARAM_DEFAULTS = {
    "lupi_lambda_teacher": LUPI_LAMBDA_TEACHER,
    "lupi_lambda_kl": LUPI_LAMBDA_KL,
    "lupi_lambda_kl_calib": LUPI_LAMBDA_KL_CALIB,
    "backbone_lr": BACKBONE_LR,
    "backbone_batch_size": BACKBONE_BATCH_SIZE,
    "backbone_weight_decay": BACKBONE_WEIGHT_DECAY,
    "early_stop_patience": EARLY_STOP_PATIENCE,
    "calib_lr": CALIB_LR,
    "calib_optimizer": CALIB_OPTIMIZER,
    "calib_lbfgs_max_iter": CALIB_LBFGS_MAX_ITER,
    "calib_theta_l2": CALIB_THETA_L2,
    "hidden_dim": HIDDEN_DIM,
    "min_train_weight": MIN_TRAIN_WEIGHT,
}


def get_hyperparams() -> dict:
    return {
        "lupi_lambda_teacher": LUPI_LAMBDA_TEACHER,
        "lupi_lambda_kl": LUPI_LAMBDA_KL,
        "lupi_lambda_kl_calib": LUPI_LAMBDA_KL_CALIB,
        "backbone_lr": BACKBONE_LR,
        "backbone_batch_size": BACKBONE_BATCH_SIZE,
        "backbone_weight_decay": BACKBONE_WEIGHT_DECAY,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "calib_lr": CALIB_LR,
        "calib_optimizer": CALIB_OPTIMIZER,
        "calib_lbfgs_max_iter": CALIB_LBFGS_MAX_ITER,
        "calib_theta_l2": CALIB_THETA_L2,
        "hidden_dim": HIDDEN_DIM,
        "min_train_weight": MIN_TRAIN_WEIGHT,
    }


def configure_hyperparams(**overrides) -> dict:
    """Override main-path training/calibration hyperparameters."""
    global LUPI_LAMBDA_TEACHER, LUPI_LAMBDA_KL, LUPI_LAMBDA_KL_CALIB
    global BACKBONE_LR, BACKBONE_BATCH_SIZE, BACKBONE_WEIGHT_DECAY, EARLY_STOP_PATIENCE
    global CALIB_LR, CALIB_OPTIMIZER, CALIB_LBFGS_MAX_ITER, CALIB_THETA_L2
    global HIDDEN_DIM, MIN_TRAIN_WEIGHT

    known = set(HYPERPARAM_DEFAULTS)
    cfg = {**get_hyperparams(), **{k: v for k, v in overrides.items() if k in known}}
    LUPI_LAMBDA_TEACHER = float(cfg["lupi_lambda_teacher"])
    LUPI_LAMBDA_KL = float(cfg["lupi_lambda_kl"])
    LUPI_LAMBDA_KL_CALIB = float(cfg["lupi_lambda_kl_calib"])
    BACKBONE_LR = float(cfg["backbone_lr"])
    BACKBONE_BATCH_SIZE = int(cfg["backbone_batch_size"])
    BACKBONE_WEIGHT_DECAY = float(cfg["backbone_weight_decay"])
    EARLY_STOP_PATIENCE = int(cfg["early_stop_patience"])
    CALIB_LR = float(cfg["calib_lr"])
    CALIB_OPTIMIZER = str(cfg["calib_optimizer"]).lower()
    CALIB_LBFGS_MAX_ITER = int(cfg["calib_lbfgs_max_iter"])
    CALIB_THETA_L2 = float(cfg["calib_theta_l2"])
    HIDDEN_DIM = int(cfg["hidden_dim"])
    MIN_TRAIN_WEIGHT = float(cfg["min_train_weight"])
    return dict(cfg)


def _indices_from_manifest(manifest: dict, folder_to_idx: dict[str, int]) -> tuple[list[int], list[int]]:
    train = [folder_to_idx[m["folder"]] for m in manifest["train_models"]]
    test = [folder_to_idx[m["folder"]] for m in manifest["test_models"]]
    return train, test


def _filter_agents_with_summary(
    agent_indices: list[int],
    summary_mask: np.ndarray,
    *,
    min_cells: int = 1,
) -> list[int]:
    return [i for i in agent_indices if int(summary_mask[i].sum()) >= min_cells]


def _frac_bucket(frac: float) -> float:
    for t in (0.05, 0.10, 0.15, 0.20):
        if abs(float(frac) - t) < 0.02:
            return t
    return round(float(frac), 2)


def model_base_snapshot(model: Privileged_Deep_LUPI_IRT) -> dict:
    return {
        "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "test_theta": dict(getattr(model, "test_theta", {}) or {}),
    }


def restore_model_base(model: Privileged_Deep_LUPI_IRT, base: dict) -> None:
    model.load_state_dict(base["state_dict"])
    model.test_theta = dict(base.get("test_theta") or {})
    if hasattr(model, "test_delta"):
        model.test_delta = {}
    if hasattr(model, "_item_params_cache"):
        model._item_params_cache = None


def train_privileged_deep_lupi(
    df: pd.DataFrame,
    s_emb: torch.Tensor,
    summary_mask: np.ndarray,
    quality_w: np.ndarray,
    train_agents: list[int],
    device: torch.device,
    max_epochs: int,
    seed: int,
    *,
    min_weight: float = MIN_TRAIN_WEIGHT,
    delta_scale: float = DEFAULT_DELTA_SCALE,
    delta_l2: float = DEFAULT_DELTA_L2,
    a_delta_mode: str | None = None,
    use_summary_corr: bool = True,
) -> Privileged_Deep_LUPI_IRT:
    mw = DSF_MIN_WEIGHT if min_weight == MIN_TRAIN_WEIGHT else float(min_weight)
    adm = DEFAULT_A_DELTA_MODE if a_delta_mode is None else str(a_delta_mode)
    return train_privileged_deep_4pl_backbone(
        df,
        s_emb,
        summary_mask,
        quality_w,
        train_agents,
        device,
        max_epochs,
        seed,
        min_weight=mw,
        delta_scale=delta_scale,
        delta_l2=delta_l2,
        lupi_lambda_teacher=LUPI_LAMBDA_TEACHER,
        lupi_lambda_kl=LUPI_LAMBDA_KL,
        hidden_dim=HIDDEN_DIM,
        a_delta_mode=adm,
        early_stop_patience=EARLY_STOP_PATIENCE,
        lr=BACKBONE_LR,
        batch_size=BACKBONE_BATCH_SIZE,
        weight_decay=BACKBONE_WEIGHT_DECAY,
        use_summary_corr=use_summary_corr,
    )


def _freeze_all_privileged_nets(model: Privileged_Deep_LUPI_IRT) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _prob_4pl_torch(
    theta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    z = torch.clamp(a * (theta - b), -40.0, 40.0)
    return c + (d - c) * torch.sigmoid(z)


def _run_scalar_param_optimizer(
    param: nn.Parameter,
    *,
    build_loss,
    steps: int,
    lr: float,
) -> None:
    if CALIB_OPTIMIZER == "lbfgs":
        opt = optim.LBFGS(
            [param],
            lr=lr,
            max_iter=CALIB_LBFGS_MAX_ITER,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = build_loss(LUPI_LAMBDA_KL_CALIB)
            loss.backward()
            return loss

        opt.step(closure)
        return

    opt = optim.Adam([param], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = build_loss(LUPI_LAMBDA_KL_CALIB)
        loss.backward()
        opt.step()


def _privileged_item_params_numpy(
    model: Privileged_Deep_LUPI_IRT, n_items: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = getattr(model, "_item_params_cache", None)
    if cache is None:
        cache = item_base_params(model, n_items, device)
        model._item_params_cache = cache
    return cache


def _privileged_teacher_ab_for_agent(
    model: Privileged_Deep_LUPI_IRT,
    agent_idx: int,
    item_ids: list[int],
    s_emb: torch.Tensor,
    summary_mask: np.ndarray | None,
    n_items: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Student (a,b,c,d) with Teacher Δa/Δb on privileged calib cells."""
    a, b, c, d = _privileged_item_params_numpy(model, n_items, device)
    a_eff, b_eff = a.copy(), b.copy()
    if not item_ids:
        return a_eff, b_eff, c, d
    se = s_emb[agent_idx, item_ids].to(device)
    da, db = model._summary_delta(se)
    da_np = da.squeeze(-1).cpu().numpy() if da is not None else None
    db_np = db.squeeze(-1).cpu().numpy()
    for i, qid in enumerate(item_ids):
        if summary_mask is not None and not bool(summary_mask[agent_idx, qid] > 0.5):
            continue
        if da_np is not None:
            if getattr(model, "a_delta_mode", "add") == "mul":
                a_eff[qid] = a[qid] * float(np.exp(da_np[i]))
            else:
                a_eff[qid] = a[qid] + float(da_np[i])
        b_eff[qid] = b[qid] + float(db_np[i])
    return a_eff, b_eff, c, d


def calibrate_test_agent_theta(
    model: Privileged_Deep_LUPI_IRT,
    df: pd.DataFrame,
    s_emb: torch.Tensor,
    agent_idx: int,
    calib_items: list[int],
    device: torch.device,
    n_items: int,
    steps: int,
    lr: float,
    summary_mask: np.ndarray | None = None,
) -> None:
    """Freeze nets; fit scalar θ with Student BCE + Teacher KL + θ L2."""
    if not hasattr(model, "test_theta") or model.test_theta is None:
        model.test_theta = {}
    _freeze_all_privileged_nets(model)
    model.eval()
    if not calib_items:
        model.test_theta[agent_idx] = 0.0
        return

    a_s, b_s, c_arr, d_arr = _privileged_item_params_numpy(model, n_items, device)
    a_t, b_t, _, _ = _privileged_teacher_ab_for_agent(
        model, agent_idx, calib_items, s_emb, summary_mask, n_items, device
    )
    if summary_mask is not None:
        priv = torch.tensor(
            summary_mask[agent_idx, calib_items], dtype=torch.float32, device=device
        )
    else:
        priv = torch.zeros(len(calib_items), dtype=torch.float32, device=device)

    idx = np.asarray(calib_items, dtype=int)
    ys = torch.tensor(
        [float(df.iloc[agent_idx, qid]) for qid in calib_items],
        dtype=torch.float32,
        device=device,
    )
    aa_s = torch.tensor(a_s[idx], dtype=torch.float32, device=device)
    bb_s = torch.tensor(b_s[idx], dtype=torch.float32, device=device)
    aa_t = torch.tensor(a_t[idx], dtype=torch.float32, device=device)
    bb_t = torch.tensor(b_t[idx], dtype=torch.float32, device=device)
    cc = torch.tensor(c_arr[idx], dtype=torch.float32, device=device)
    dd = torch.tensor(d_arr[idx], dtype=torch.float32, device=device)

    emp = float(np.clip(ys.detach().cpu().numpy().mean(), 1e-3, 1.0 - 1e-3))
    theta0 = float(np.log(emp / (1.0 - emp)))
    theta = nn.Parameter(torch.tensor([theta0], dtype=torch.float32, device=device))
    criterion = nn.BCELoss()

    def build_loss(kl_lambda: float) -> torch.Tensor:
        p_s = _prob_4pl_torch(theta, aa_s, bb_s, cc, dd)
        loss = criterion(p_s, ys)
        if priv.sum() > 0 and kl_lambda > 0:
            p_t = _prob_4pl_torch(theta, aa_t, bb_t, cc, dd).detach()
            kl = bernoulli_kl(p_t, p_s)
            loss = loss + kl_lambda * (kl * priv).sum() / priv.sum().clamp(min=1.0)
        loss = loss + float(CALIB_THETA_L2) * theta.square().sum()
        return loss

    _run_scalar_param_optimizer(theta, build_loss=build_loss, steps=steps, lr=lr)
    model.test_theta[agent_idx] = float(theta.detach().cpu().item())


def predict_item_probs_for_agent(
    model: Privileged_Deep_LUPI_IRT,
    agent_idx: int,
    item_ids: list[int],
    n_items: int,
    device: torch.device,
) -> np.ndarray:
    if not item_ids:
        return np.zeros(0, dtype=np.float64)
    theta = None
    if hasattr(model, "test_theta") and model.test_theta is not None:
        theta = model.test_theta.get(agent_idx)
    if theta is not None:
        a, b, c, d = _privileged_item_params_numpy(model, n_items, device)
        idx = np.asarray(item_ids, dtype=int)
        return (
            prob_4pl(
                np.array([theta], dtype=np.float64),
                a[idx],
                b[idx],
                c[idx],
                d[idx],
            )
            .reshape(-1)
            .astype(np.float64)
        )

    # Fallback if θ was not calibrated.
    n_agents = int(model.student_net[0].in_features)
    svec = torch.zeros(n_agents, dtype=torch.float32, device=device)
    svec[agent_idx] = 1.0
    sid_batch = svec.unsqueeze(0).expand(len(item_ids), -1)
    items_batch = torch.stack(
        [F.one_hot(torch.tensor(qid), n_items).float().to(device) for qid in item_ids]
    )
    probs_t = model.forward_student(sid_batch, items_batch).squeeze()
    probs = probs_t.detach().cpu().numpy()
    if np.ndim(probs) == 0:
        return np.asarray([float(probs)], dtype=np.float64)
    return np.asarray(probs, dtype=np.float64)


def evaluate_subset_vs_full_at_frac(
    model: Privileged_Deep_LUPI_IRT,
    df: pd.DataFrame,
    y: np.ndarray,
    test_agents: list[int],
    calib_items: list[int],
    s_emb: torch.Tensor,
    n_items: int,
    device: torch.device,
    base_state: dict,
    calib_steps: int,
    summary_mask: np.ndarray | None = None,
) -> dict:
    """Calibrate on S; score predicted S rate vs full-benchmark truth."""
    all_items = list(range(n_items))
    true_full = y[test_agents].mean(axis=1)
    true_subset = y[np.ix_(test_agents, calib_items)].mean(axis=1)
    pred_full_rates: list[float] = []
    pred_subset_rates: list[float] = []

    n_agents = len(test_agents)
    for i, aid in enumerate(test_agents):
        if i == 0 or (i + 1) % 5 == 0 or i + 1 == n_agents:
            print(f"  subset→full eval: agent {i + 1}/{n_agents}", flush=True)
        restore_model_base(model, base_state)
        calibrate_test_agent_theta(
            model,
            df,
            s_emb,
            aid,
            calib_items,
            device,
            n_items,
            steps=calib_steps,
            lr=CALIB_LR,
            summary_mask=summary_mask,
        )
        probs_all = predict_item_probs_for_agent(model, aid, all_items, n_items, device)
        probs_sub = probs_all[np.asarray(calib_items, dtype=int)] if calib_items else probs_all[:0]
        pred_full_rates.append(float(probs_all.mean()))
        pred_subset_rates.append(float(probs_sub.mean()) if len(probs_sub) else float("nan"))

    pred_full = np.asarray(pred_full_rates, dtype=np.float64)
    pred_subset = np.asarray(pred_subset_rates, dtype=np.float64)
    return {
        "calib_frac": len(calib_items) / n_items,
        "n_calib_items": len(calib_items),
        "n_items_full": n_items,
        "calib_items": calib_items,
        "true_full": true_full.tolist(),
        "pred_subset": pred_subset.tolist(),
        "true_subset": true_subset.tolist(),
        "pred_full": pred_full.tolist(),
        "subset_vs_full": {
            "full_extrapolation": agent_ranking_metrics(true_full, pred_full),
            "subset_estimate": agent_ranking_metrics(true_full, pred_subset),
            "true_subset_proxy": agent_ranking_metrics(true_full, true_subset),
        },
    }


def aggregate_cv(fold_rows: list[dict], *, eval_mode: str = "subset_vs_full") -> list[dict]:
    """Mean ± std per calib_frac, plus pooled OOF ranking metrics."""
    if eval_mode != "subset_vs_full":
        raise ValueError("This package only supports eval_mode=subset_vs_full")

    by_model: dict[str, dict[float, dict[str, list]]] = {}
    for fold in fold_rows:
        for model_block in fold["results"]:
            name = model_block["model"]
            by_model.setdefault(name, {})
            for row in model_block["by_calib_frac"]:
                frac = _frac_bucket(row["calib_frac"])
                full = row["subset_vs_full"]["full_extrapolation"]
                sub = row["subset_vs_full"]["subset_estimate"]
                proxy = row["subset_vs_full"]["true_subset_proxy"]
                slot = by_model[name].setdefault(
                    frac,
                    {
                        "full_mae": [],
                        "full_tau": [],
                        "subset_mae": [],
                        "subset_tau": [],
                        "subset_rho": [],
                        "proxy_mae": [],
                        "proxy_tau": [],
                        "proxy_rho": [],
                        "oof_true": [],
                        "oof_pred": [],
                        "oof_proxy": [],
                        "oof_pred_full": [],
                    },
                )
                slot["full_mae"].append(full["mae"])
                slot["full_tau"].append(full["kendall_tau"])
                slot["subset_mae"].append(sub["mae"])
                slot["subset_tau"].append(sub["kendall_tau"])
                slot["subset_rho"].append(sub["spearman_rho"])
                slot["proxy_mae"].append(proxy["mae"])
                slot["proxy_tau"].append(proxy["kendall_tau"])
                slot["proxy_rho"].append(proxy["spearman_rho"])
                true_full = row.get("true_full")
                pred_sub = row.get("pred_subset")
                true_sub = row.get("true_subset")
                pred_full = row.get("pred_full")
                if true_full is not None and pred_sub is not None:
                    slot["oof_true"].extend(true_full)
                    slot["oof_pred"].extend(pred_sub)
                if true_full is not None and true_sub is not None:
                    slot["oof_proxy"].extend(true_sub)
                if true_full is not None and pred_full is not None:
                    slot["oof_pred_full"].extend(pred_full)

    agg: list[dict] = []
    for name, frac_map in by_model.items():
        rows = []
        for frac in sorted(frac_map):
            m = frac_map[frac]
            row_out = {
                "calib_frac": frac,
                "mae_mean": float(np.mean(m["subset_mae"])),
                "mae_std": float(np.std(m["subset_mae"], ddof=0)),
                "kendall_tau_mean": float(np.mean(m["subset_tau"])),
                "kendall_tau_std": float(np.std(m["subset_tau"], ddof=0)),
                "spearman_rho_mean": float(np.mean(m["subset_rho"])),
                "spearman_rho_std": float(np.std(m["subset_rho"], ddof=0)),
                "true_subset_proxy_mae_mean": float(np.mean(m["proxy_mae"])),
                "true_subset_proxy_tau_mean": float(np.mean(m["proxy_tau"])),
                "true_subset_proxy_rho_mean": float(np.mean(m["proxy_rho"])),
                "full_extrapolation_mae_mean": float(np.mean(m["full_mae"])),
                "full_extrapolation_tau_mean": float(np.mean(m["full_tau"])),
            }
            if m["oof_true"] and m["oof_pred"]:
                oof = score_metrics(
                    np.asarray(m["oof_true"], dtype=np.float64),
                    np.asarray(m["oof_pred"], dtype=np.float64),
                )
                row_out.update(
                    {
                        "oof_n_agents": int(oof["n_agents"]),
                        "oof_mae": float(oof["mae"]),
                        "oof_kendall_tau": float(oof["kendall_tau"]),
                        "oof_spearman_rho": float(oof["spearman_rho"]),
                    }
                )
            if m["oof_true"] and m["oof_proxy"]:
                oof_p = score_metrics(
                    np.asarray(m["oof_true"], dtype=np.float64),
                    np.asarray(m["oof_proxy"], dtype=np.float64),
                )
                row_out.update(
                    {
                        "oof_proxy_mae": float(oof_p["mae"]),
                        "oof_proxy_kendall_tau": float(oof_p["kendall_tau"]),
                        "oof_proxy_spearman_rho": float(oof_p["spearman_rho"]),
                    }
                )
            rows.append(row_out)
        agg.append({"model": name, "by_calib_frac": rows})
    return agg


def run_one_fold(
    fold_manifest: dict,
    *,
    df: pd.DataFrame,
    y: np.ndarray,
    s_emb,
    summary_mask: np.ndarray,
    quality_w: np.ndarray,
    folder_to_idx: dict[str, int],
    device: torch.device,
    max_epochs: int,
    seed: int,
    frozen_calib_manifest: Path | str,
    calib_steps: int = 80,
    require_summary: bool = True,
    min_summary_cells: int = 1,
    calib_fracs: list[float] | None = None,
    min_train_weight: float | None = None,
    a_delta_mode: str | None = None,
    use_summary_corr: bool = True,
    privileged_delta_scale: float | None = None,
    privileged_delta_l2: float | None = None,
) -> dict:
    if min_train_weight is not None:
        configure_hyperparams(min_train_weight=min_train_weight)

    train_agents, test_agents = _indices_from_manifest(fold_manifest, folder_to_idx)
    n_train_raw, n_test_raw = len(train_agents), len(test_agents)
    if require_summary:
        train_agents = _filter_agents_with_summary(
            train_agents, summary_mask, min_cells=min_summary_cells
        )
        test_agents = _filter_agents_with_summary(
            test_agents, summary_mask, min_cells=min_summary_cells
        )
    if not train_agents or not test_agents:
        raise RuntimeError(
            f"Fold {fold_manifest.get('fold')}: empty train/test after summary filter"
        )

    n_items = y.shape[1]
    calib_fracs = list(calib_fracs or CALIB_FRACS)
    fold_idx = int(fold_manifest.get("fold", 0))
    calib_by_frac = load_frozen_calib_by_frac(
        fold_idx,
        calib_fracs,
        manifest_path=frozen_calib_manifest,
        fold_manifest=fold_manifest,
        strict_agent_hash=True,
    )

    pkw = {
        "delta_scale": privileged_delta_scale or DSF_DELTA_SCALE,
        "delta_l2": privileged_delta_l2 or DSF_DELTA_L2,
        "a_delta_mode": a_delta_mode or DEFAULT_A_DELTA_MODE,
        "use_summary_corr": use_summary_corr,
    }
    if min_train_weight is not None:
        pkw["min_weight"] = min_train_weight

    model = train_privileged_deep_lupi(
        df, s_emb, summary_mask, quality_w, train_agents, device, max_epochs, seed, **pkw
    )
    base_state = model_base_snapshot(model)
    frac_rows = []
    for frac in calib_fracs:
        row = evaluate_subset_vs_full_at_frac(
            model,
            df,
            y,
            test_agents,
            calib_by_frac[frac],
            s_emb,
            n_items,
            device,
            base_state,
            calib_steps,
            summary_mask=summary_mask,
        )
        frac_rows.append(row)

    return {
        "fold": fold_manifest.get("fold"),
        "n_train": len(train_agents),
        "n_test": len(test_agents),
        "n_train_before_filter": n_train_raw,
        "n_test_before_filter": n_test_raw,
        "require_summary": require_summary,
        "calib_strategy": "privileged_4pl_fisher_frozen",
        "min_train_weight": min_train_weight,
        "eval_mode": "subset_vs_full",
        "test_name_groups": fold_manifest.get("test_name_groups"),
        "results": [{"model": MODEL_NAME, "by_calib_frac": frac_rows}],
    }
