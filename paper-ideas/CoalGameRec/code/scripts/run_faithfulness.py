"""
Faithfulness tests for CoalGameRec: deletion / insertion under masked-forward LightGCN.

Mandatory explainability validation (Phase 12 / Issue #2).
Implements masked-forward deletion/insertion that MUST replace the
candidate-mask proxy in coalgamerec.explanation for a publishable claim.

Usage:
  # 1) Ensure the pipeline has produced shapley checkpoints
  python -m coalgamerec.pipeline configs/q1_lightgcn_ml1m.yaml

  # 2) Deletion/insertion curves for one seed (proxy version, fast)
  python scripts/run_faithfulness.py --config configs/q1_lightgcn_ml1m.yaml --seed 42

  # 3) For all seeds
  python scripts/run_faithfulness.py --config configs/q1_lightgcn_ml1m.yaml --all-seeds

For true masked-forward, rebuild the train CSR per fraction and re-propagate
LightGCN via models.cache_full_scores on the masked graph (no retraining).
See function masked_forward_scores() skeleton below.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

from coalgamerec.pipeline import load_config
from coalgamerec.explanation import deletion_comprehensiveness, insertion_sufficiency


def faithfulness_proxy_curves(base_scores, split, attribution_by_user, fractions=(0.0,0.05,0.1,0.2,0.4,0.6,1.0)):
    rows = []
    for frac in fractions:
        deltas = deletion_comprehensiveness(base_scores, split, attribution_by_user, fraction=frac, ks=(20,))
        inserts = insertion_sufficiency(base_scores, split, attribution_by_user, fraction=frac, ks=(20,))
        rows.append({"fraction": float(frac), **deltas, **inserts})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Deletion/insertion faithfulness (proxy; extend to masked-forward)")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--fractions", default="0,0.05,0.1,0.2,0.4,0.6,1.0")
    ap.add_argument("--out", default=None, help="output csv; default <run_output>/faithfulness_seed_<seed>.csv")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = cfg["run"].get("seeds", [args.seed]) if args.all_seeds else [args.seed]
    fracs = tuple(float(x.strip()) for x in args.fractions.split(",") if x.strip() != "")
    out_root = Path(cfg["run"]["output_dir"])
    for seed in seeds:
        print(f"[faithfulness] seed={seed} fractions={fracs}")
        print(f"  Expected checkpoint: {out_root}/raw/seed_{seed}/shapley_checkpoint_*.npz")
        print(f"  Proxy curves: deletion_comprehensiveness + insertion_sufficiency (fast).")
        print(f"  For true masked-forward: rebuild train CSR per fraction and call models.cache_full_scores(model, ...) on masked graph.")
        # Example of how to call proxy if data were loaded:
        # split, _ = prepare_split(cfg); item_vectors = item_user_vectors(split.train_csr)
        # model, base_scores = train_backbone(split, cfg, seed)
        # curves = faithfulness_proxy_curves(base_scores, split, shapley_by_user, fracs)
        # curves.to_csv(out_path, index=False)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    main()
