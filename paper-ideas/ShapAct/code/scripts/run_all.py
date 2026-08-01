#!/usr/bin/env python3
"""Run everything: audits (all seeds), significance tests, and the validation
comparison against published Q1 numbers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_audit import main as run_audit_main
from shapact.config import CONFIGS, RESULT_DIR, TABLE_DIR, SEEDS


def run_significance():
    """Pairwise per-user comparisons of realized NDCG across decision rules."""
    from shapact.stats import compare_rules

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in CONFIGS:
        p = RESULT_DIR / f"audit_{name}_seed42.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        per_user = {r: data["decisions"]["per_user"][r]
                    for r in data["decisions"]["per_user"]}
        import numpy as np
        per_user = {r: np.array(v) for r, v in per_user.items()}
        rows = compare_rules(per_user)
        out[name] = rows
        (TABLE_DIR / f"significance_{name}.json").write_text(
            json.dumps(rows, indent=1))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-seeds", action="store_true")
    args = ap.parse_args()

    if args.no_seeds:
        run_audit_main()
    else:
        import scripts.run_audit as ra
        saved = sys.argv[:]
        sys.argv = ["run_all", "--seeds", *[n for n in CONFIGS]]
        run_audit_main()
        sys.argv = saved
    run_significance()
    print("done.")
