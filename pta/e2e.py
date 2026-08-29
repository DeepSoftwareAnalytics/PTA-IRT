# -*- coding: utf-8 -*-
"""PTA-IRT end-to-end: strata frozen calib + LUPI estimate (four SWE benches)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]

from pta.estimate import (  # noqa: E402
    CALIB_FRACS,
    aggregate_cv,
    configure_hyperparams,
    run_one_fold,
    _indices_from_manifest,
)
from pta.frozen_calib import fold_agent_hashes_from_manifest  # noqa: E402
from pta.metrics.ranking import score_metrics  # noqa: E402
from pta.models.Privileged_Deep_4PL_IRT import train_privileged_deep_4pl  # noqa: E402
from pta.selection import (  # noqa: E402
    DSF_DELTA_L2,
    DSF_DELTA_SCALE,
    DSF_MIN_WEIGHT,
    DSF_RELIABILITY_MODE,
    privileged_4pl_summary_fisher_scores,
    select_items_difficulty_stratified,
    train_item_pass_rates,
)

BENCH = {
    "lite": {
        "data": CODE_ROOT / "data" / "swe_lite_summary",
        "fold": CODE_ROOT / "data" / "folds" / "lite",
        "min_cells": 300,
    },
    "verified": {
        "data": CODE_ROOT / "data" / "swe_verified_summary",
        "fold": CODE_ROOT / "data" / "folds" / "verified",
        "min_cells": 500,
    },
    "full": {
        "data": CODE_ROOT / "data" / "swe_test_summary",
        "fold": CODE_ROOT / "data" / "folds" / "full",
        "min_cells": 2200,
    },
    "pro": {
        "data": CODE_ROOT / "data" / "swe_pro_summary",
        "fold": CODE_ROOT / "data" / "folds" / "pro",
        "min_cells": 100,
    },
}

# Fallback frozen manifests when `{bench}_{variant}_{adm}.json` is absent.
FROZEN_FALLBACK = {
    "lite": CODE_ROOT / "data" / "frozen_calib" / "lite_strata_k7_width_prop_add.json",
    "verified": CODE_ROOT / "data" / "frozen_calib" / "verified_strata_k7_width_prop.json",
    "full": CODE_ROOT / "data" / "frozen_calib" / "full_strata_k7_width_prop.json",
    "pro": CODE_ROOT / "data" / "frozen_calib" / "pro_strata_k7_width_prop.json",
}


def build_strata_frozen(
    *,
    bench: str,
    n_bins: int,
    binning: str,
    quota: str,
    max_epochs: int,
    seed: int,
    dest: Path,
    a_delta_mode: str = "add",
) -> Path:
    cfg = BENCH[bench]
    data_dir = cfg["data"]
    fold_dir = cfg["fold"]
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    folder_to_idx = {a["folder"]: i for i, a in enumerate(meta["agents"])}
    df = pd.read_csv(data_dir / "matrix.csv", header=None)
    y = df.to_numpy(dtype=np.float64)
    s_emb = torch.load(data_dir / "summary_embeddings.pt", map_location="cpu", weights_only=False)
    summary_mask = np.load(data_dir / "summary_mask.npy")
    quality_w = np.load(data_dir / "quality_weight.npy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adm = str(a_delta_mode).lower()

    folds: dict[str, dict[str, list[int]]] = {}
    proxy_tau: dict[str, list[float]] = {}
    proxy_mae: dict[str, list[float]] = {}
    fold_agent_hashes: dict[str, dict[str, str]] = {}

    for path in sorted(fold_dir.glob("fold_*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        train_agents, test_agents = _indices_from_manifest(manifest, folder_to_idx)
        n_students, n_items = y.shape
        fold_i = str(manifest.get("fold", path.stem))
        print(f"[{bench}] build S fold {fold_i} a_delta={adm}", flush=True)
        model = train_privileged_deep_4pl(
            df,
            s_emb,
            summary_mask,
            quality_w,
            train_agents,
            device,
            max_epochs,
            seed,
            min_weight=DSF_MIN_WEIGHT,
            delta_scale=DSF_DELTA_SCALE,
            delta_l2=DSF_DELTA_L2,
            a_delta_mode=adm,
            lupi_lambda_teacher=0.5,
            lupi_lambda_kl=0.5,
        )
        scores = privileged_4pl_summary_fisher_scores(
            model,
            train_agents,
            n_students,
            n_items,
            device,
            s_emb,
            summary_mask,
            quality_w,
            min_weight=DSF_MIN_WEIGHT,
            reliability_mode=DSF_RELIABILITY_MODE,
        )
        rates = train_item_pass_rates(y, train_agents, n_items)
        folds[fold_i] = {}
        for frac in CALIB_FRACS:
            n_select = max(1, int(round(n_items * frac)))
            items = select_items_difficulty_stratified(
                scores,
                rates,
                n_select,
                n_bins=n_bins,
                binning=binning,
                quota=quota,
            )
            folds[fold_i][str(frac)] = items
            true_full = y[test_agents].mean(axis=1)
            true_sub = y[np.ix_(test_agents, items)].mean(axis=1)
            m = score_metrics(true_full, true_sub)
            proxy_tau.setdefault(str(frac), []).append(float(m["kendall_tau"]))
            proxy_mae.setdefault(str(frac), []).append(float(m["mae"]))
        fold_agent_hashes[fold_i] = fold_agent_hashes_from_manifest(manifest)

    out = {
        "label": f"{bench}_strata_k{n_bins}_{binning}_{quota}_{adm}",
        "fold_agent_hashes": fold_agent_hashes,
        "folds": folds,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {dest}", flush=True)
    print(
        "proxy MAE",
        {k: float(np.mean(v)) for k, v in sorted(proxy_mae.items(), key=lambda x: float(x[0]))},
        flush=True,
    )
    print(
        "proxy tau",
        {k: float(np.mean(v)) for k, v in sorted(proxy_tau.items(), key=lambda x: float(x[0]))},
        flush=True,
    )
    return dest


def run_pta_e2e(
    *,
    bench: str,
    frozen: Path,
    out_json: Path,
    hp_json: Path,
    max_epochs: int,
    seed: int,
    a_delta_mode: str = "add",
) -> dict:
    cfg = BENCH[bench]
    data_dir = cfg["data"]
    fold_dir = cfg["fold"]
    min_cells = cfg["min_cells"]
    hp = json.loads(hp_json.read_text(encoding="utf-8"))
    configure_hyperparams(**hp)
    min_w = float(hp.get("min_train_weight", 0.03))
    adm = str(a_delta_mode).lower()

    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    folder_to_idx = {a["folder"]: i for i, a in enumerate(meta["agents"])}
    df = pd.read_csv(data_dir / "matrix.csv", header=None)
    y = df.to_numpy(dtype=np.float64)
    s_emb = torch.load(data_dir / "summary_embeddings.pt", map_location="cpu", weights_only=False)
    summary_mask = np.load(data_dir / "summary_mask.npy")
    quality_w = np.load(data_dir / "quality_weight.npy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_rows = []
    for path in sorted(fold_dir.glob("fold_*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        print(f"[{bench}] PTA fold {manifest.get('fold')} a_delta={adm}", flush=True)
        row = run_one_fold(
            manifest,
            df=df,
            y=y,
            s_emb=s_emb,
            summary_mask=summary_mask,
            quality_w=quality_w,
            folder_to_idx=folder_to_idx,
            device=device,
            max_epochs=max_epochs,
            seed=seed,
            frozen_calib_manifest=frozen,
            calib_steps=80,
            require_summary=True,
            min_summary_cells=min_cells,
            calib_fracs=CALIB_FRACS,
            min_train_weight=min_w,
            a_delta_mode=adm,
        )
        fold_rows.append(row)

    aggregated = aggregate_cv(fold_rows, eval_mode="subset_vs_full")
    payload = {
        "bench": bench,
        "frozen_manifest": str(frozen),
        "a_delta_mode": adm,
        "hyperparams": {**hp, "a_delta_mode": adm},
        "folds": fold_rows,
        "aggregated": aggregated,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out_json}", flush=True)
    for block in aggregated:
        if "Privileged" not in block["model"]:
            continue
        for r in block["by_calib_frac"]:
            print(
                f"  @{r['calib_frac']:.0%}: MAE={r['mae_mean']:.4f}±{r['mae_std']:.4f} "
                f"OOF τ={r.get('oof_kendall_tau', float('nan')):.4f} "
                f"ρ={r.get('oof_spearman_rho', float('nan')):.4f}",
                flush=True,
            )
    return payload


def resolve_frozen(bench: str, variant: str, adm: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    tagged = CODE_ROOT / "data" / "frozen_calib" / f"{bench}_{variant}_{adm}.json"
    if tagged.is_file():
        return tagged
    fallback = FROZEN_FALLBACK.get(bench)
    if fallback is not None and fallback.is_file():
        return fallback
    return tagged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", choices=list(BENCH), required=True)
    p.add_argument("--n-bins", type=int, default=7)
    p.add_argument("--binning", choices=("quantile", "width"), default="width")
    p.add_argument("--quota", choices=("prop", "equal"), default="prop")
    p.add_argument("--a-delta-mode", choices=("add", "mul"), default="add")
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-build", action="store_true", default=False)
    p.add_argument("--frozen", type=Path, default=None)
    p.add_argument("--hp-json", type=Path, default=CODE_ROOT / "results" / "_hp_ref_lupi.json")
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    adm = args.a_delta_mode
    variant = f"strata_k{args.n_bins}_{args.binning}_{args.quota}"
    if args.skip_build:
        frozen = resolve_frozen(args.bench, variant, adm, args.frozen)
    else:
        dest = (
            args.frozen
            if args.frozen is not None
            else CODE_ROOT / "data" / "frozen_calib" / f"{args.bench}_{variant}_{adm}.json"
        )
        frozen = build_strata_frozen(
            bench=args.bench,
            n_bins=args.n_bins,
            binning=args.binning,
            quota=args.quota,
            max_epochs=args.max_epochs,
            seed=args.seed,
            dest=dest,
            a_delta_mode=adm,
        )
    out = args.out_json or (
        CODE_ROOT / "results" / f"{args.bench}_pta_{variant}_{adm}.json"
    )
    run_pta_e2e(
        bench=args.bench,
        frozen=frozen,
        out_json=out,
        hp_json=args.hp_json,
        max_epochs=args.max_epochs,
        seed=args.seed,
        a_delta_mode=adm,
    )


if __name__ == "__main__":
    main()
