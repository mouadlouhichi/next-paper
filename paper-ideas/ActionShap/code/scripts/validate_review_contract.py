#!/usr/bin/env python3
"""Reviewer-facing semantic and production checks for the canonical paper-v3."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

METHODS = {"shapley_mc", "lime", "loo", "greedy_cf", "random"}
BUDGET_CONDITIONS = {"budget1", "budget3"}
POINTWISE_METRICS = {
    "aia",
    "aia_ndcg",
    "faithfulness_alignment",
    "actionability_gap",
}


def rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def resolve_paper_root(code_root: Path, value: str) -> Path:
    candidate = (code_root / value).resolve()
    if (candidate / "actionshap.tex").exists():
        return candidate
    if candidate.name == "actionshap.tex":
        return candidate.parent
    raise FileNotFoundError(f"canonical paper root not found: {candidate}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", default="../paper-v3")
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parents[1]
    paper_root = resolve_paper_root(code_root, args.paper_root)
    tables = paper_root / "final" / "tables"
    tex = (paper_root / "actionshap.tex").read_text()
    errors: list[str] = []

    gap = rows(tables / "actionability_gap_robustness.csv")
    components = rows(tables / "aia_components.csv")
    if {row["method"] for row in gap} != METHODS:
        errors.append("gap table does not contain exactly all five declared methods")
    if any(
        re.search(r"budget|B=1|B=3", row.get("condition_label", ""), re.I)
        for row in gap
    ):
        errors.append("budget sensitivity leaked into singleton gap table")

    component_keys = [
        "dataset",
        "model",
        "evaluation_mode",
        "utility",
        "analysis_role",
        "condition",
        "method",
    ]
    component_map: dict[tuple[str, ...], dict[str, float]] = {}
    for row in components:
        key = tuple(row[column] for column in component_keys)
        component_map.setdefault(key, {})[row["component"]] = float(row["mean"])
    for key, values in component_map.items():
        expected = {"Deletion AIA", "Bounded AIA", "Gap (bounded - deletion)"}
        if set(values) == expected:
            observed_gap = values["Gap (bounded - deletion)"]
            if abs(observed_gap - (values["Bounded AIA"] - values["Deletion AIA"])) > 1e-8:
                errors.append(f"gap algebra mismatch for {key}")
            if key[-1] == "loo" and abs(values["Deletion AIA"] - 1.0) > 1e-12:
                errors.append(f"LOO deletion AIA invariant failed for {key}")

    for required in (
        "aia_components.tex",
        "intervention_outcomes.tex",
        "aia_permutation_null.tex",
        "sensitivity_results.tex",
    ):
        if not (tables / required).exists():
            errors.append(f"missing required table {required}")

    # Budget rows must be action-only in the publication sensitivity table.
    sensitivity_tex = (tables / "sensitivity_results.tex").read_text()
    if re.search(r"budget|B=1|B=3", sensitivity_tex, re.I):
        if re.search(r"&\s*(?:AIA|Bounded AIA|Deletion AIA|Actionability Gap|Bounded-minus-deletion|Gap)\s*&", sensitivity_tex, re.I):
            errors.append("pointwise metric appears in budget sensitivity table")

    for phrase in (
        "only method with a positive",
        "22 comparisons",
        "uniquely intervention-robust",
        "Shapley alone improves",
    ):
        if phrase.lower() in tex.lower():
            errors.append(f"unsupported headline phrase remains: {phrase}")
    if "\\operatorname{NRegret}" not in tex:
        errors.append("normalized regret equation missing")
    if "M_{\\mathrm{pair}}" not in tex or "T=2M_{\\mathrm{pair}}" not in tex:
        errors.append("base-permutation/total-order notation is incomplete")
    if "B_u=H_u^{\\mathrm{train}}\\setminus P_u" not in tex:
        errors.append("older-history context B_u is not defined")
    if "no $B_u$ term" not in tex:
        errors.append("player-window scoring semantics are not explicit")
    if "architecture-agnostic protocol for history-conditioned recommenders" not in tex:
        errors.append("model-agnostic scope wording was not narrowed")
    if "paper-v2" in tex:
        errors.append("stale paper-v2 path remains in canonical manuscript")
    if re.search(r"\\safeinput\{[^}]+\}", tex):
        included = re.findall(r"\\safeinput\{([^}]+)\}", tex)
        if len(included) != len(set(included)):
            errors.append("a generated table is included more than once")

    print(
        {
            "status": "PASS" if not errors else "FAIL",
            "gap_rows": len(gap),
            "component_rows": len(components),
            "errors": errors,
        }
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
