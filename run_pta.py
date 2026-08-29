#!/usr/bin/env python3
"""Reproduce the PTA-IRT main experiment (RQ1) on one SWE-bench suite.

Examples (from this Code/ directory):

  python run_pta.py --bench lite --skip-build
  python run_pta.py --bench all --skip-build

Default protocol matches the paper main table:
  strata k7 width_prop, a_delta_mode=add, HP=results/_hp_ref_lupi.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pta.e2e import BENCH  # noqa: E402
import pta.e2e as e2e  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bench",
        choices=[*BENCH, "all"],
        default="lite",
        help="Benchmark suite (or all four)",
    )
    p.add_argument("--n-bins", type=int, default=7)
    p.add_argument("--binning", choices=("quantile", "width"), default="width")
    p.add_argument("--quota", choices=("prop", "equal"), default="prop")
    p.add_argument("--a-delta-mode", choices=("add", "mul"), default="add")
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-build",
        action="store_true",
        default=True,
        help="Use packaged frozen calibration sets (default: on)",
    )
    p.add_argument(
        "--rebuild-selection",
        action="store_true",
        help="Rebuild stratified S before estimation (disables --skip-build)",
    )
    p.add_argument("--frozen", type=Path, default=None)
    p.add_argument(
        "--hp-json",
        type=Path,
        default=CODE_ROOT / "results" / "_hp_ref_lupi.json",
    )
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    skip = not args.rebuild_selection
    benches = list(BENCH) if args.bench == "all" else [args.bench]
    for b in benches:
        argv = [
            "e2e",
            "--bench",
            b,
            "--n-bins",
            str(args.n_bins),
            "--binning",
            args.binning,
            "--quota",
            args.quota,
            "--a-delta-mode",
            args.a_delta_mode,
            "--max-epochs",
            str(args.max_epochs),
            "--seed",
            str(args.seed),
            "--hp-json",
            str(args.hp_json),
        ]
        if skip:
            argv.append("--skip-build")
        if args.frozen is not None:
            argv.extend(["--frozen", str(args.frozen)])
        if args.out_json is not None and args.bench != "all":
            argv.extend(["--out-json", str(args.out_json)])
        sys.argv = argv
        e2e.main()


if __name__ == "__main__":
    main()
