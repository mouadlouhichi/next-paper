#!/usr/bin/env python3
"""Reviewer-facing semantic checks beyond schema validation."""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path
METHODS = {"shapley_mc", "lime", "loo", "greedy_cf", "random"}
def rows(path: Path):
    with path.open(newline="") as f: return list(csv.DictReader(f))
def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--paper-root",default="../paper"); args=ap.parse_args()
    code=Path(__file__).resolve().parents[1]; root=(code/args.paper_root).resolve(); tables=root/"final"/"tables"; tex=(root/"paper.tex").read_text(); errors=[]
    gap=rows(tables/"actionability_gap_robustness.csv"); comps=rows(tables/"aia_components.csv")
    if {r["method"] for r in gap} != METHODS: errors.append("gap table does not contain exactly all five declared methods")
    if any(re.search(r"budget|B=1|B=3",r.get("condition_label",""),re.I) for r in gap): errors.append("budget sensitivity leaked into singleton gap table")
    keys=["dataset","model","evaluation_mode","utility","analysis_role","condition","method"]; by={}
    for r in comps: by.setdefault(tuple(r[k] for k in keys),{})[r["component"]]=float(r["mean"])
    for k,v in by.items():
        if set(v)=={"Deletion AIA","Bounded AIA","Gap (bounded - deletion)"} and abs(v["Gap (bounded - deletion)"]-(v["Bounded AIA"]-v["Deletion AIA"]))>1e-8: errors.append(f"gap algebra mismatch for {k}")
    for required in ("aia_components.tex","intervention_outcomes.tex","aia_permutation_null.tex"):
        if not (tables/required).exists(): errors.append(f"missing required table {required}")
    for phrase in ("only method with a positive","22 comparisons","uniquely intervention-robust","Shapley alone improves"):
        if phrase.lower() in tex.lower(): errors.append(f"unsupported headline phrase remains: {phrase}")
    if "\\operatorname{NRegret}" not in tex: errors.append("normalized regret equation missing")
    if "\\label{fig:aia-components}" not in tex: errors.append("component figure label missing")
    if re.search(r"\\ref\{[^}]*\?",tex): errors.append("unresolved reference")
    print({"status":"PASS" if not errors else "FAIL","gap_rows":len(gap),"component_rows":len(comps),"errors":errors})
    if errors: raise SystemExit(1)
if __name__ == "__main__": main()
