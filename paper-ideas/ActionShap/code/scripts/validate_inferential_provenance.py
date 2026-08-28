#!/usr/bin/env python3
"""Recompute every printed inferential quantity from the record it claims to read.

Three checks, each pinning a defect class the review rounds actually found:

1. ``p`` must sit on the declared plus-one permutation grid, ``(1+#)/(R+1)``, for the ``R`` stored
   with the test --- no printed permutation ``p`` may be smaller than ``1/(R+1)`` or land between
   grid points.
2. ``p_holm`` must equal the Holm step-down correction recomputed over the family the release
   declares: the ``dataset x model x condition x metric`` group of ten pairs in
   ``paired_tests.csv``, and the single predeclared 12-contrast family in
   ``review7_success_abstention_tests.csv``.
3. One contrast can carry two *different* corrected values because the two families are two
   separate released records with different draw counts and different success estimands.  That is
   legitimate only while every table that prints one of them names its source record in its
   caption, so a reader cannot mistake one for a correction of the other.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "actionshap-ipm" / "release" / "matrices"
TABLES = ROOT / "acmart-primary" / "tables"
PER_METRIC = "paired_tests.csv"
TWELVE = "review7_success_abstention_tests.csv"
PER_FAMILY = ["dataset", "model", "condition", "metric"]
GRID_TOL = 1e-6


def read_caption(text: str) -> str:
    """First caption block, brace-matched: captions themselves contain braces."""
    start = text.find("\\caption{")
    if start < 0:
        return ""
    depth, position = 0, start + len("\\caption") - 1
    while position < len(text):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start + len("\\caption{"):position]
        position += 1
    return text[start:]


def holm(pvalues):
    order = np.argsort(np.asarray(pvalues, dtype=float), kind="stable")
    adjusted = np.zeros(len(pvalues), dtype=float)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * float(pvalues[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def grid_check(frame, p_column, draws_column, label, problems):
    draws = frame[draws_column].astype(int) if draws_column in frame else pd.Series(10_000, index=frame.index)
    if draws.nunique() != 1:
        problems.append(f"{label}: draw counts are not constant ({sorted(draws.unique())[:5]})")
    draw_value = int(draws.mode().iat[0])
    grid = frame[p_column].astype(float) * (draw_value + 1)
    off = (grid - grid.round()).abs()
    bad = int((off > GRID_TOL).sum())
    below = int((frame[p_column].astype(float) < 1.0 / (draw_value + 1) - GRID_TOL).sum())
    print(f"{label}: R={draw_value}, floor {1.0 / (draw_value + 1):.6f}; "
          f"{bad} off-grid p-values, {below} below the floor")
    if bad:
        problems.append(f"{label}: {bad} printed p-values are not multiples of 1/(R+1)={1.0/(draw_value+1):.6f}")
    if below:
        problems.append(f"{label}: {below} printed p-values fall below the attainable minimum")
    return draw_value


def holm_check(frame, families, p_column, holm_column, label, problems):
    expected = np.zeros(len(frame), dtype=float)
    for _, group in frame.groupby(families, sort=False):
        adjusted = holm(group[p_column].tolist())
        expected[frame.index.get_indexer(group.index)] = adjusted
    delta = np.abs(frame[holm_column].astype(float).to_numpy() - expected)
    mismatched = frame[delta > 5e-5]
    print(f"{label}: {frame.groupby(families, sort=False).ngroups} families, "
          f"{len(mismatched)} Holm mismatches (max |delta| {delta.max():.2e})")
    for _, row in mismatched.head(6).iterrows():
        keys = "/".join(str(row[c]) for c in families)
        problems.append(f"{label}: Holm mismatch at {keys}: printed {row[holm_column]}, "
                        f"recomputed {expected[frame.index.get_loc(row.name)]:.6f}")
    if len(mismatched) > 6:
        problems.append(f"{label}: {len(mismatched)} Holm mismatches in total")


def main() -> int:
    problems: list[str] = []
    per_metric = pd.read_csv(RELEASE / PER_METRIC)
    twelve = pd.read_csv(RELEASE / TWELVE)

    grid_check(per_metric, "permutation_p", "permutation_draws", PER_METRIC, problems)
    holm_check(per_metric, PER_FAMILY, "permutation_p", "p_holm", PER_METRIC, problems)
    grid_check(twelve, "p_plusone", "permutation_draws", TWELVE, problems)
    twelve["_all"] = 0          # one predeclared family over all twelve contrasts
    holm_check(twelve, ["_all"], "p_plusone", "holm_p", TWELVE + " (pooled 12-contrast family)", problems)
    if len(twelve) != 12:
        problems.append(f"{TWELVE}: the predeclared family has {len(twelve)} tests, not 12")

    # Cross-record conflicts: the same contrast corrected in two different families.
    metric_to_quantity = {"intervention_success_ndcg": "Success", "abstention": "Abstention"}
    conflicts = []
    for _, row in twelve.iterrows():
        wanted = [m for m, q in metric_to_quantity.items() if q == row["quantity"]]
        for metric in wanted:
            match = per_metric[(per_metric.dataset == row["dataset"]) & (per_metric.metric == metric)
                               & (per_metric.condition == "primary")
                               & (per_metric.left == row["left"]) & (per_metric.right == row["right"])]
            if match.empty:
                continue
            other = float(match.p_holm.iloc[0])
            if abs(other - float(row["holm_p"])) > 5e-5:
                conflicts.append((row["dataset"], metric, row["left"], row["right"],
                                  float(row["holm_p"]), other))
    print(f"contrasts whose two family corrections differ: {len(conflicts)}")

    # A generated table that publishes a corrected (Holm) column must say which released
    # family it corrected within, and the sentence that declares the family size must name the
    # record carrying that family.  This is what stops a 12-contrast value from being read as a
    # correction of a ten-pair value, or the other way round.
    for path in sorted(TABLES.glob("*.tex")):
        text = path.read_text(encoding="utf-8")
        if not re.search(r"Holm", text):
            continue
        caption = read_caption(text)
        flat = caption.replace("\\_", "_")
        claims_twelve = re.search(r"12[- ](test|contrast)", caption) is not None
        claims_ten = re.search(r"(ten|10)[ -]pair|family of ten", caption) is not None
        if claims_twelve and TWELVE not in flat:
            problems.append(f"{path.name}: declares a 12-test/contrast family but does not name "
                            f"{TWELVE} as its source record")
        if claims_ten and PER_METRIC not in flat:
            problems.append(f"{path.name}: declares the ten-pair per-metric family but does not "
                            f"name {PER_METRIC} as its source record")
        if (claims_twelve and claims_ten) and not ((TWELVE in flat) and (PER_METRIC in flat)):
            problems.append(f"{path.name}: publishes both families and must name both records")
    if conflicts:
        print("  (both records are cited side by side only where their captions say so)")

    if problems:
        print("\n".join(f"PROBLEM: {p}" for p in problems), file=sys.stderr)
        return 1
    print("inferential provenance: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
