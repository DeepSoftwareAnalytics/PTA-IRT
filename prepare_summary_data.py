#!/usr/bin/env python3
"""Build PTA-IRT matrices and MiniLM summary embeddings from process summaries.

Merges the Lite / Verified / Full / Pro prepare scripts into one entry point.

Typical (rebuild embeddings aligned to packaged metadata + matrix):

  python prepare_summary_data.py \\
    --summary-root traj_summaries/lite \\
    --metadata data/swe_lite_summary/metadata.json \\
    --matrix data/swe_lite_summary/matrix.csv \\
    --out-dir data/swe_lite_summary

From scratch (resolved labels + instance order):

  python prepare_summary_data.py \\
    --summary-root traj_summaries/lite \\
    --resolved-dir /path/to/evaluation/lite \\
    --instance-ids instance_ids.json \\
    --out-dir data/swe_lite_summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

QUALITY_WEIGHT = {
    "full": 1.0,
    "degraded": 0.1,
    "eval_only": 0.1,
    "no_traj": 0.0,
}
MIN_SUMMARY_CHARS = 40
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def _summary_text(data: dict) -> str:
    s = data.get("summary")
    if s and s != "(dry-run: no LLM call)":
        return str(s)[:6000].strip()
    return ""


def _resolved_set(results_path: Path) -> set[str]:
    data = json.loads(results_path.read_text(encoding="utf-8-sig"))
    return set(data.get("resolved") or [])


def _load_json_list(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict) and "instance_ids" in raw:
        raw = raw["instance_ids"]
    if not isinstance(raw, list):
        raise SystemExit(f"{path} must be a JSON list (or an object with instance_ids)")
    return [str(x) for x in raw]


def _agents_and_items(args: argparse.Namespace) -> tuple[list[dict], list[str], dict | None]:
    meta = None
    if args.metadata:
        meta = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
        agents = list(meta["agents"])
        items = list(meta["instance_ids"])
        return agents, items, meta

    if args.instance_ids:
        items = _load_json_list(args.instance_ids)
    else:
        raise SystemExit("Provide --metadata or --instance-ids so item order is defined")

    if args.results_csv:
        df = pd.read_csv(args.results_csv)
        folders = sorted(df["model_slug"].astype(str).unique().tolist())
        agents = [{"folder": f, "name": f} for f in folders]
        return agents, items, None

    if args.summary_root.is_dir():
        folders = sorted(p.name for p in args.summary_root.iterdir() if p.is_dir())
        if folders:
            agents = [{"folder": f, "name": f} for f in folders]
            return agents, items, None

    raise SystemExit("Could not infer agents: pass --metadata, --results-csv, or a --summary-root with agent folders")


def _build_matrix(
    agents: list[dict],
    items: list[str],
    *,
    matrix_path: Path | None,
    resolved_dir: Path | None,
    results_csv: Path | None,
) -> np.ndarray:
    n_a, n_i = len(agents), len(items)
    idx = {iid: j for j, iid in enumerate(items)}

    if matrix_path:
        y = pd.read_csv(matrix_path, header=None).to_numpy(dtype=np.float64)
        if y.shape != (n_a, n_i):
            raise SystemExit(f"matrix shape {y.shape} != ({n_a}, {n_i}) from agents/items")
        return y

    y = np.zeros((n_a, n_i), dtype=np.float64)
    if results_csv:
        df = pd.read_csv(results_csv)
        a_idx = {a["folder"]: i for i, a in enumerate(agents)}
        for row in df.itertuples(index=False):
            i = a_idx.get(str(row.model_slug))
            j = idx.get(str(row.instance_id))
            if i is None or j is None:
                continue
            y[i, j] = float(row.resolved)
        return y

    if resolved_dir:
        for i, agent in enumerate(agents):
            res = resolved_dir / agent["folder"] / "results" / "results.json"
            if not res.is_file():
                continue
            for iid in _resolved_set(res):
                j = idx.get(iid)
                if j is not None:
                    y[i, j] = 1.0
        return y

    raise SystemExit("Provide --matrix, --resolved-dir, or --results-csv for outcome labels")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-root", type=Path, required=True, help="traj_summaries/<bench>/")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--metadata", type=Path, default=None, help="Existing metadata.json (keeps agent/item order)")
    p.add_argument("--instance-ids", type=Path, default=None, help="JSON list of instance ids")
    p.add_argument("--matrix", type=Path, default=None, help="Existing agent×item matrix.csv")
    p.add_argument("--resolved-dir", type=Path, default=None, help="evaluation/<bench>/{folder}/results/results.json")
    p.add_argument("--results-csv", type=Path, default=None, help="Pro-style CSV: model_slug,instance_id,resolved")
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    args = p.parse_args()

    agents, items, meta_in = _agents_and_items(args)
    y = _build_matrix(
        agents,
        items,
        matrix_path=args.matrix,
        resolved_dir=args.resolved_dir,
        results_csv=args.results_csv,
    )
    n_a, n_i = y.shape
    summary_mask = np.zeros((n_a, n_i), dtype=np.float32)
    quality_w = np.zeros((n_a, n_i), dtype=np.float32)
    texts: list[str] = []
    text_index: list[tuple[int, int]] = []

    for i, agent in enumerate(agents):
        sdir = args.summary_root / agent["folder"]
        if not sdir.is_dir():
            continue
        for j, iid in enumerate(items):
            path = sdir / f"{iid}_summary.json"
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError:
                continue
            w = QUALITY_WEIGHT.get(str(data.get("traj_quality") or "full"), 0.0)
            if w <= 0:
                continue
            text = _summary_text(data)
            if len(text) < MIN_SUMMARY_CHARS:
                continue
            summary_mask[i, j] = 1.0
            quality_w[i, j] = w
            texts.append(text)
            text_index.append((i, j))

    print(f"Encoding {len(texts)} summaries with {args.encoder} ...")
    encoder = SentenceTransformer(args.encoder)
    dim = int(encoder.get_sentence_embedding_dimension())
    s_emb = np.zeros((n_a, n_i, dim), dtype=np.float32)
    if texts:
        vecs = encoder.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        for (i, j), vec in zip(text_index, vecs):
            s_emb[i, j] = vec.astype(np.float32)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(y).to_csv(out / "matrix.csv", header=False, index=False)
    np.save(out / "summary_mask.npy", summary_mask)
    np.save(out / "quality_weight.npy", quality_w)
    torch.save(torch.tensor(s_emb, dtype=torch.float32), out / "summary_embeddings.pt")

    meta = {
        "encoder": args.encoder,
        "embed_dim": dim,
        "n_agents": n_a,
        "n_items": n_i,
        "n_summary_cells": int(summary_mask.sum()),
        "instance_ids": items,
        "agents": agents,
        "quality_weights": QUALITY_WEIGHT,
    }
    if meta_in:
        for key in (
            "split",
            "n_train_agents",
            "n_test_agents",
            "train_agent_indices",
            "test_agent_indices",
            "train_has_summary",
            "test_has_summary",
        ):
            if key in meta_in:
                meta[key] = meta_in[key]
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"agents={n_a} items={n_i} summary_cells={int(summary_mask.sum())} dim={dim}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
