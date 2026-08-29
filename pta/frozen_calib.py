# -*- coding: utf-8 -*-
"""Load frozen stratified calibration item lists."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def folders_hash(folders: list[str]) -> str:
    payload = "\n".join(sorted(str(f) for f in folders))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def fold_agent_hashes_from_manifest(fold_manifest: dict) -> dict[str, str]:
    train = [m["folder"] for m in fold_manifest["train_models"]]
    test = [m["folder"] for m in fold_manifest["test_models"]]
    return {"train": folders_hash(train), "test": folders_hash(test)}


def load_frozen_calib_manifest(path: Path | str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Frozen calib manifest not found: {p}")
    return json.loads(p.read_text(encoding="utf-8-sig"))


def validate_frozen_fold_agents(
    manifest: dict,
    fold: int,
    fold_manifest: dict,
    *,
    strict: bool = True,
) -> None:
    """Ensure frozen S was built on the same train/test agents as this CV fold."""
    expected = (manifest.get("fold_agent_hashes") or {}).get(str(fold))
    if not expected:
        msg = (
            f"Frozen manifest {manifest.get('label')!r} missing fold_agent_hashes[{fold}]"
        )
        if strict:
            raise KeyError(msg)
        print(f"WARNING: {msg}", flush=True)
        return
    got = fold_agent_hashes_from_manifest(fold_manifest)
    if got["train"] != expected.get("train") or got["test"] != expected.get("test"):
        msg = (
            f"Frozen DSF agent-hash mismatch on fold {fold}: "
            f"manifest train/test={expected.get('train')}/{expected.get('test')} "
            f"vs fold train/test={got['train']}/{got['test']}."
        )
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg}", flush=True)


def load_frozen_calib_by_frac(
    fold: int,
    calib_fracs: list[float],
    *,
    manifest_path: Path | str,
    fold_manifest: dict | None = None,
    strict_agent_hash: bool = True,
) -> dict[float, list[int]]:
    """Return calib item lists for one CV fold."""
    manifest = load_frozen_calib_manifest(manifest_path)
    if fold_manifest is not None:
        validate_frozen_fold_agents(
            manifest, fold, fold_manifest, strict=strict_agent_hash
        )
    fold_key = str(fold)
    if fold_key not in manifest["folds"]:
        raise KeyError(f"Fold {fold} not in frozen manifest {manifest.get('label')!r}")
    fold_data = manifest["folds"][fold_key]
    out: dict[float, list[int]] = {}
    for frac in calib_fracs:
        matched = None
        for k, items in fold_data.items():
            if abs(float(k) - float(frac)) < 1e-9:
                matched = items
                break
        if matched is None:
            raise KeyError(
                f"calib_frac={frac} missing for fold {fold} in frozen manifest "
                f"{manifest.get('label')!r}; available={list(fold_data)}"
            )
        out[float(frac)] = [int(i) for i in matched]
    return out
