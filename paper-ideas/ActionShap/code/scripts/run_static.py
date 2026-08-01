#!/usr/bin/env python
"""Run the full ActionShap evaluation over a static-regime dataset.

    python scripts/run_static.py --dataset wine

Fits the clustering pipeline, runs every attribution method under R seeds,
measures feasible-intervention effects, and reports the alignment, Actionability
Score, and decision-level metrics from Section 4.5, plus the misalignment
decomposition from Section 4.7.2.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from actionshap.attribution import (  # noqa: E402
    KernelShapAttributor, LimeAttributor, PermutationAttributor,
    RandomAttributor, TreeShapAttributor, feature_direction, run_repeated,
)
from actionshap.data import load_air_quality, load_wine  # noqa: E402
from actionshap.decomposition import PurifiedGA2M, decompose_misalignment  # noqa: E402
from actionshap.intervention import (  # noqa: E402
    InterventionBudget, intervention_profile,
)
from actionshap.metrics import (  # noqa: E402
    actionability_score, actionability_score_heldout, alignment,
    intervention_regret, topk_intervention_precision,
)
from actionshap.models import StaticPipeline  # noqa: E402
from actionshap.modifiability import load_modifiability  # noqa: E402
from actionshap.rerank import eta_sweep  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("wine", "air_quality"), default="wine")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--budget-sd", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=0.01,
                    help="minimum practically meaningful effect (Definition 6)")
    ap.add_argument("--subsample", type=int, default=None,
                    help="rows to use; essential for air quality")
    ap.add_argument("--skip-slow", action="store_true",
                    help="omit KernelSHAP and LIME")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "raw")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)

    data = load_wine() if args.dataset == "wine" else load_air_quality()
    X = data.X
    if args.subsample and args.subsample < len(X):
        X = X[np.random.default_rng(0).choice(len(X), args.subsample, replace=False)]

    print(f"\n{'=' * 72}\n{data.name}: {X.shape[0]} x {X.shape[1]}\n{'=' * 72}")

    # -- modifiability -----------------------------------------------------
    mod_path = _resolve_annotation(args.dataset)
    provisional = mod_path.name.endswith(".provisional.yaml")
    table = load_modifiability(
        mod_path, expected_factors=list(data.feature_names),
        verify_freeze=not provisional,
    )
    m = table.m
    if provisional:
        print("\n*** PROVISIONAL ANNOTATION -- results are a smoke test, not")
        print("*** findings. See annotations/RUBRIC.md before citing anything.")
    print(f"\nmodifiability      {mod_path.name}")
    print(f"  immutable (m=0)  {table.n_immutable} of {len(m)}: "
          f"{[f for f, v in zip(table.factors, m) if v == 0]}")
    print(f"  agreement alpha  {table.agreement():.3f}  (intra-team; rubric clarity)")

    # -- model -------------------------------------------------------------
    pipe = StaticPipeline(k=args.k).fit(X)
    q = pipe.quality(X)
    print(f"\nclustering         {q}")
    print(f"surrogate fidelity {json.dumps(pipe.fidelity_, indent=None)}")

    # -- the game ----------------------------------------------------------
    # P(designated cluster), not P(own cluster). An accurate surrogate puts
    # P(own cluster) at ~0.99, so every feasible perturbation can only drive it
    # down, sign(Delta_j) is -1 for all j, and the sign condition in Definition
    # 6 becomes untestable. Naming a target cluster keeps the baseline near that
    # cluster's share of the data, where the output can move either way.
    target, target_note = _choose_target(pipe, data)
    print(f"\nintervention target cluster {target}   ({target_note})")

    budget = InterventionBudget.from_std(X, args.budget_sd)
    output_fn = pipe.target_fn(target)
    profile = intervention_profile(output_fn, X, budget)
    delta = profile.best
    print(f"baseline P(target)  {profile.baseline:.4f}")
    print(f"\nintervention effects at {args.budget_sd} sd "
          f"(best of +/- tau, both measured):")
    for f, d, mv in sorted(zip(data.feature_names, delta, m),
                           key=lambda t: -abs(t[1]))[:5]:
        print(f"  {f:<22} {d:+.5f}   m={mv}")
    print(f"  {(delta > 0).sum()} of {len(delta)} effects positive")

    # Sign of the effect each factor is *expected* to have, needed before P@k
    # can apply its sign condition. See attribution.feature_direction for why
    # sign(phi) will not serve. Taken for the same cluster the effects use.
    tree = TreeShapAttributor(pipe.surrogate_, target_class=target)
    per_instance, rows = tree.per_instance(X, seed=SEEDS[0])
    direction = feature_direction(per_instance, X[rows])
    agree = int(((np.sign(delta) == direction) | (direction == 0)).sum())
    print(f"  attribution direction matches realized effect for "
          f"{agree}/{len(m)} factors")

    # -- attribution -------------------------------------------------------
    attributors = [tree]
    if not args.skip_slow:
        attributors += [
            KernelShapAttributor(pipe.predict_proba),
            LimeAttributor(pipe.predict_proba, data.feature_names),
        ]
    # Permutation importance scores recovery of the cluster label, which is a
    # different question from P(target) and is reported as such.
    attributors += [
        PermutationAttributor(
            predict=lambda Z: pipe.surrogate_.predict(Z),
            score=lambda y, p: float((y == p).mean()),
            y=pipe.labels_,
        ),
        RandomAttributor(X.shape[1]),
    ]

    print(f"\n{'method':<16}{'AIA':>8}{'AIA|m>0':>10}{'P@3':>8}{'regret':>9}{'stab':>7}")
    print("-" * 72)

    results = {}
    for att in attributors:
        runs = run_repeated(att, X, SEEDS)
        phi, s = runs.phi, runs.s

        aia = alignment(phi, delta).spearman
        aia_r = alignment(phi, delta, m, restrict_to_modifiable=True).spearman
        p3 = topk_intervention_precision(
            phi, delta, m, k=3, delta=args.delta, direction=direction
        )
        reg = intervention_regret(phi, delta, m)
        a_score = actionability_score(m, delta, s)

        results[att.name] = {
            "aia": aia, "aia_restricted": aia_r, "p_at_3": p3,
            "regret_normalized": reg.normalized,
            "chosen": data.feature_names[reg.chosen_index],
            "optimal": data.feature_names[reg.optimal_index],
            "mean_stability": float(s.mean()),
            "mass_on_immutable": float(
                np.abs(phi)[m == 0].sum() / np.abs(phi).sum()
            ),
            "phi": phi.tolist(),
            "action_score": a_score.tolist(),
            "action_score_heldout": actionability_score_heldout(delta, s).tolist(),
        }
        print(f"{att.name:<16}{aia:>8.3f}{aia_r:>10.3f}{p3:>8.2f}"
              f"{reg.normalized:>9.3f}{s.mean():>7.3f}")

    print("\nattribution mass placed on immutable factors "
          "(the Section 4.5.2 headline):")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["mass_on_immutable"]):
        print(f"  {name:<16} {r['mass_on_immutable']:>6.1%}   "
              f"picks {r['chosen']!r}, best is {r['optimal']!r}")

    # -- reranking ---------------------------------------------------------
    primary = "TreeSHAP"
    phi = np.array(results[primary]["phi"])
    a_score = np.array(results[primary]["action_score"])
    print(f"\nactionability-guided reranking of {primary}:")
    print(f"  {'eta':>5}{'switched':>10}{'regret':>9}{'AIA':>8}")
    for o in eta_sweep(phi, a_score, delta, m):
        print(f"  {o.eta:>5.2f}{str(o.switched):>10}"
              f"{o.normalized_regret:>9.3f}{o.aia:>8.3f}")

    # -- decomposition -----------------------------------------------------
    print("\nmisalignment decomposition:")
    try:
        shares = _decompose(pipe, X, m, budget.values, target)
        for c in ("H1", "H2", "H3"):
            print(f"  psi_{c}  {shares.psi[c]:+.4f}")
        print(f"  gap    {shares.closed_gap:+.4f}   "
              f"(AIA {shares.aia_baseline:.3f} -> {shares.aia_grand:.3f})")
        print(f"  resid   {shares.residual:.4f}   dominant: {shares.dominant}")
        results["_decomposition"] = {
            "psi": shares.psi,
            "aia_baseline": shares.aia_baseline,
            "aia_grand": shares.aia_grand,
            "residual": shares.residual,
        }
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"  FAILED: {type(exc).__name__}: {exc}")

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"{data.name}_static.json"
    dest.write_text(json.dumps(
        {"dataset": data.name, "n": int(X.shape[0]),
         "features": list(data.feature_names),
         "provisional_annotation": provisional,
         "modifiability": m.tolist(),
         "quality": {"silhouette": q.silhouette, "db": q.davies_bouldin},
         "fidelity": pipe.fidelity_,
         "target_cluster": int(target),
         "target_rationale": target_note,
         "budget_sd": args.budget_sd,
         "effects": delta.tolist(),
         "effect_profile": profile.as_dict(),
         "attribution_direction": direction.tolist(),
         "methods": results},
        indent=2,
    ))
    print(f"\nwrote {dest}")
    return 0


def _choose_target(pipe, data):
    """Pick the cluster interventions should aim at.

    Prefers a held-out descriptor that names the desirable regime -- wine
    quality, which is excluded from the clustering but recorded alongside it.
    Falls back to the largest cluster, which carries no notion of desirable and
    makes the target an arbitrary but stated convention.
    """
    for key in ("quality",):
        if key in data.aux:
            ranked = pipe.rank_clusters_by(data.aux[key])
            best, value = ranked[0]
            return best, f"highest mean {key} {value:.3f}, from held-out {key}"

    sizes = np.bincount(pipe.labels_, minlength=pipe.k)
    best = int(sizes.argmax())
    return best, f"largest cluster, {sizes[best] / sizes.sum():.1%} of rows; " \
                 "no held-out descriptor available to rank by"


def _decompose(pipe, X, m, budgets, target):
    """Fit the purified surrogate and allocate the gap across H1, H2, H3."""
    from interpret.glassbox import ExplainableBoostingRegressor

    y = pipe.target_fn(target)(X)
    ebm = ExplainableBoostingRegressor(
        interactions=min(10, X.shape[1]),
        max_bins=32,
        max_interaction_bins=32,   # shared grid; PurifiedGA2M requires it
        outer_bags=4,
        random_state=42,
        n_jobs=1,
    ).fit(X, y)
    g = PurifiedGA2M(ebm, X)
    print(f"  surrogate R^2 {g.fidelity(X, y):.4f}   "
          f"max concurvity {g.concurvity().max():.3f}")

    def attribute(f, Xa):
        base = f(Xa)
        out = np.empty(Xa.shape[1])
        for j in range(Xa.shape[1]):
            Xp = Xa.copy()
            Xp[:, j] = Xa[:, j].mean()
            out[j] = np.abs(base - f(Xp)).mean()
        return out

    return decompose_misalignment(
        g, X, attribute, m, budgets, n_matched_draws=100, seed=42
    )


def _resolve_annotation(dataset: str) -> Path:
    d = ROOT / "annotations"
    for candidate in (d / f"{dataset}.yaml", d / f"{dataset}.provisional.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No annotation for {dataset!r}. Copy {dataset}.template.yaml to "
        f"{dataset}.yaml, complete it per annotations/RUBRIC.md, and freeze it."
    )


if __name__ == "__main__":
    raise SystemExit(main())
