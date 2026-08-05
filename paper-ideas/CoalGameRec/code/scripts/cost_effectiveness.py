#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description="Create cost/effectiveness summary for CoalGameRec runs.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--baseline", default="uniform")
    ap.add_argument("--loo", default="loo-marginal")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    summary_path = run_dir / "tables" / "summary_mean_std.csv"
    seed_summary_path = run_dir / "tables" / "summary_by_seed_family.csv"
    if not seed_summary_path.exists():
        raise FileNotFoundError(seed_summary_path)
    summary = pd.read_csv(seed_summary_path)
    means = summary.groupby("family")[["NDCG@20", "HitRate@20", "Coverage@20", "ILD@20"]].mean()

    runtime_rows = []
    for p in sorted((run_dir / "raw").glob("seed_*/runtime.json")):
        r = read_json(p)
        row = {"seed": r.get("seed"), "total_seconds": r.get("seconds", 0.0)}
        row.update({f"stage_{k}": v for k, v in r.get("stages", {}).items()})
        runtime_rows.append(row)
    runtime = pd.DataFrame(runtime_rows)
    total_seconds = float(runtime["total_seconds"].sum()) if not runtime.empty else 0.0
    shapley_seconds = float(runtime.get("stage_shapley_seconds", pd.Series(dtype=float)).sum()) if not runtime.empty else 0.0
    loo_seconds = float(runtime.get("stage_loo_seconds", pd.Series(dtype=float)).sum()) if not runtime.empty else 0.0

    rows = []
    base_ndcg = float(means.loc[args.baseline, "NDCG@20"]) if args.baseline in means.index else None
    loo_ndcg = float(means.loc[args.loo, "NDCG@20"]) if args.loo in means.index else None
    for fam, vals in means.iterrows():
        ndcg = float(vals["NDCG@20"])
        hr = float(vals["HitRate@20"])
        row = {
            "family": fam,
            "NDCG@20_mean": ndcg,
            "HitRate@20_mean": hr,
            "Coverage@20_mean": float(vals["Coverage@20"]),
            "ILD@20_mean": float(vals["ILD@20"]),
            "delta_NDCG_vs_uniform": ndcg - base_ndcg if base_ndcg is not None else None,
            "delta_NDCG_vs_loo": ndcg - loo_ndcg if loo_ndcg is not None else None,
        }
        if fam == "shapley-mc":
            row["attribution_seconds"] = shapley_seconds
        elif fam == args.loo:
            row["attribution_seconds"] = loo_seconds
        else:
            row["attribution_seconds"] = 0.0
        denom = row["attribution_seconds"] or 1.0
        row["delta_NDCG_vs_uniform_per_attribution_hour"] = (row["delta_NDCG_vs_uniform"] or 0.0) / (denom / 3600.0)
        rows.append(row)
    out_dir = run_dir / "tables"
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "cost_effectiveness.csv", index=False)
    runtime.to_csv(out_dir / "runtime_by_seed.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
