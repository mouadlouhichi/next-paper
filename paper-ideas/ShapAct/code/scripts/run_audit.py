#!/usr/bin/env python3
"""Run the full ShapAct audit for one or all datasets and dump JSON results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shapact.config import CONFIGS, RESULT_DIR, SEEDS
from shapact.pipeline import audit_summary, run_dataset

RESULT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", action="store_true",
                    help="run all five seeds and dump a mean summary")
    args = ap.parse_args()

    for name in args.datasets:
        cfg = CONFIGS[name]()
        if args.seeds:
            summaries = []
            for s in SEEDS:
                audit = run_dataset(cfg, seed=s)
                summaries.append(audit_summary(audit))
                p = RESULT_DIR / f"audit_{name}_seed{s}.json"
                p.write_text(json.dumps(summaries[-1], indent=1))
                print(f"saved {p}")
            # mean summary across seeds for the headline tables
            keys = ["v_grand", "recall"]
            mean = {"dataset": name, "seeds": list(SEEDS),
                    "phi": {}, "predicted": {}, "realized": {}, "fidelity": {},
                    "order": {}, "decisions": {}}
            for k in keys:
                mean[k] = float(
                    sum(s[k] for s in summaries) / len(summaries))
            for g in summaries[0]["phi"]:
                mean["phi"][g] = float(
                    sum(s["phi"][g] for s in summaries) / len(summaries))
                mean["predicted"][g] = float(
                    sum(s["predicted"][g] for s in summaries) / len(summaries))
                mean["realized"][g] = float(
                    sum(s["realized"][g] for s in summaries) / len(summaries))
                mean["fidelity"][g] = {
                    "F": float(sum(s["fidelity"][g]["F"] for s in summaries)
                               / len(summaries))}
            mean["order"]["kendall_tau"] = float(
                sum(s["order"]["kendall_tau"] for s in summaries)
                / len(summaries))
            mean["decisions"]["realized_mean"] = {
                rule: float(sum(s["decisions"]["realized_mean"][rule]
                                for s in summaries) / len(summaries))
                for rule in summaries[0]["decisions"]["realized_mean"]}
            p = RESULT_DIR / f"audit_{name}_mean.json"
            p.write_text(json.dumps(mean, indent=1))
            print(f"saved {p}")
        else:
            audit = run_dataset(cfg, seed=args.seed)
            summary = audit_summary(audit)
            p = RESULT_DIR / f"audit_{name}_seed{args.seed}.json"
            p.write_text(json.dumps(summary, indent=1))
            print(f"saved {p}")


if __name__ == "__main__":
    main()
