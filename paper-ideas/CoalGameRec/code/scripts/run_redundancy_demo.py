#!/usr/bin/env python3
"""Round-4 mandatory experiment (#8): synthetic redundancy/complementarity game.

Demonstrates the mechanism behind Proposition 2 and its complement, using the
exact same solution concepts as the paper (exact Shapley vs grand-coalition
LOO marginal) on a synthetic coverage game with known ground-truth structure:

  - two REDUNDANT pairs:  {a1,a2} both cover need A; {b1,b2} both cover need B
  - one COMPLEMENTARY pair: need C is covered only if BOTH c1 and c2 are present
  - one NULL player z (covers nothing)
  - one singleton s covering need D

v(S) = fraction of the 4 needs covered by the items in S (exact coverage).

Predictions:
  - Redundant pairs: LOO_j = v(N)-v(N\\{j}) ~ 0 for each member (the partner
    still covers the need), so LOO assigns ~0 to jointly load-bearing items,
    while Shapley splits their joint credit evenly (efficiency holds).
  - Complementary pair: removing either member kills need C, so LOO credits
    EACH member with the full value of need C (double counting), while Shapley
    splits it; LOO over-allocates.
  - Null player: both concepts assign 0.
  - The LOO shares do NOT sum to v(N) (efficiency gap); Shapley sums exactly.

Outputs: results/journal_runs/synthetic/redundancy_demo.csv (+ printed table)
"""
from __future__ import annotations
import itertools
import math
from pathlib import Path
import csv

OUT = Path("results/journal_runs/synthetic")
OUT.mkdir(parents=True, exist_ok=True)

ITEMS = ["a1", "a2", "b1", "b2", "c1", "c2", "z", "s"]
N = len(ITEMS)
# needs covered by each item
COVERS = {
    "a1": {"A"}, "a2": {"A"},          # redundant pair for A
    "b1": {"B"}, "b2": {"B"},          # redundant pair for B
    "c1": {"C"}, "c2": {"C"},          # complementary: C needs BOTH
    "z":  set(),                        # null player
    "s":  {"D"},                        # singleton
}
NEEDS = {"A", "B", "C", "D"}
COMPLEMENTARY = {"C"}  # need covered only if ALL its items are present


def v(coalition: frozenset) -> float:
    covered = set()
    for need in NEEDS:
        holders = {it for it in coalition if need in COVERS[it]}
        if need in COMPLEMENTARY:
            need_items = {it for it in ITEMS if need in COVERS[it]}
            if need_items.issubset(coalition):
                covered.add(need)
        else:
            if holders:
                covered.add(need)
    return len(covered) / len(NEEDS)


def exact_shapley() -> dict[str, float]:
    phi = {it: 0.0 for it in ITEMS}
    for it in ITEMS:
        others = [o for o in ITEMS if o != it]
        for r in range(N):
            for S in itertools.combinations(others, r):
                Ss = frozenset(S)
                w = math.factorial(r) * math.factorial(N - r - 1) / math.factorial(N)
                phi[it] += w * (v(Ss | {it}) - v(Ss))
    return phi


def loo_marginals() -> dict[str, float]:
    vN = v(frozenset(ITEMS))
    return {it: vN - v(frozenset(ITEMS) - {it}) for it in ITEMS}


def main():
    phi = exact_shapley()
    loo = loo_marginals()
    vN = v(frozenset(ITEMS))

    rows = []
    print(f"{'item':6s} {'role':14s} {'Shapley':>10s} {'LOO':>10s}")
    role = {"a1": "redundant (A)", "a2": "redundant (A)", "b1": "redundant (B)",
            "b2": "redundant (B)", "c1": "complementary (C)", "c2": "complementary (C)",
            "z": "null", "s": "singleton (D)"}
    for it in ITEMS:
        print(f"{it:6s} {role[it]:14s} {phi[it]:10.4f} {loo[it]:10.4f}")
        rows.append(dict(item=it, role=role[it], shapley=round(phi[it], 6), loo=loo[it]))

    shap_sum = sum(phi.values())
    loo_sum = sum(loo.values())
    print(f"\nv(N) = {vN:.4f}")
    print(f"sum(Shapley) = {shap_sum:.6f}   (efficiency residual {abs(shap_sum - vN):.2e})")
    print(f"sum(LOO)     = {loo_sum:.6f}   (efficiency gap {loo_sum - vN:+.4f} = {100*(loo_sum-vN)/vN:+.1f}% of v(N))")
    red_shap = phi["a1"] + phi["a2"] + phi["b1"] + phi["b2"]
    red_loo = loo["a1"] + loo["a2"] + loo["b1"] + loo["b2"]
    comp_shap = phi["c1"] + phi["c2"]
    comp_loo = loo["c1"] + loo["c2"]
    print(f"redundant pairs:  Shapley {red_shap:.4f} vs LOO {red_loo:.4f}  (LOO under-allocates)")
    print(f"complementary:    Shapley {comp_shap:.4f} vs LOO {comp_loo:.4f}  (LOO double-counts)")

    rows.append(dict(item="SUM", role="efficiency", shapley=round(shap_sum, 6), loo=loo_sum))
    with open(OUT / "redundancy_demo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "role", "shapley", "loo"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSAVED {OUT/'redundancy_demo.csv'}")


if __name__ == "__main__":
    main()
