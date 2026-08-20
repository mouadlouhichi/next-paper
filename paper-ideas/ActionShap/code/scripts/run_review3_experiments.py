#!/usr/bin/env python3
"""Review-3 replication experiments (run on a machine with the datasets).

Produces, per dataset/model, the evidence requested by the third review:

* exact-Shapley validation on users with n_u <= exact_max (issue 5 / crit. 5);
* pair-interaction diagnostics and additive-vs-realized comparison (issue 4);
* intervention-aware baselines: bounded LIME (binary + continuous), finite
  differences, integrated gradients (issue 2 / crit. 2);
* prospective, non-target-conditioned audits (issue 1 / crit. 1);
* protocol ablations: forced action, magnitude-only, interaction-aware
  selection (issue 11);
* per-method runtime / peak-RSS benchmarks (issue 9).

Outputs are written to ``--out`` (default ``results/review3``) as one JSON per
dataset/model plus a combined summary. Push these results so the manuscript
tables can be completed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

# Allow direct execution (``python scripts/run_review3_experiments.py``) by
# putting the ``code/`` directory (package root) on sys.path, mirroring
# run_recommendation.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from actionshap.ablations import (
    ablation_report,
    select_forced,
    select_interaction_aware,
    select_magnitude,
)
from actionshap.baselines import lime_attribution, leave_one_out
from actionshap.bounded_baselines import (
    bounded_lime,
    finite_difference,
    integrated_gradients,
)
from actionshap.candidates import (
    fixed_evaluation_sets,
    global_item_priorities,
    tie_break_for_candidates,
)
from actionshap.evaluation import aia, joint_effect, single_player_effects
from actionshap.exact_shapley import exact_shapley, mc_error_report
from actionshap.interactions import additive_vs_realized_rank, interaction_summary
from actionshap.models.itemknn import fit_item_knn
from actionshap.prospective import build_prospective_game
from actionshap.recommendation import (
    UserGame,
    mc_shapley,
    target_margin_utility,
)
from actionshap.recommendation_data import (
    load_interactions_csv,
    load_movielens_1m,
    sample_evaluation_users,
    truncate_histories,
)
from actionshap.runtime_bench import bench
from actionshap import sasrec
from actionshap import lightgcn


def build_games(args, data, model, users):
    players = truncate_histories(data, args.n_max)
    eval_sets, _ = fixed_evaluation_sets(
        {u: data.seen_before_test(u) for u in users},
        {u: data.test[u] for u in users},
        data.n_items,
        size=args.evaluation_size,
        seed=args.candidate_seed,
    )
    games = {}
    for u in users:
        cands = eval_sets[u]
        games[u] = UserGame(
            players=players[u],
            candidate_items=cands,
            target_item=int(data.test[u]),
            tie_break=tie_break_for_candidates(
            cands, global_item_priorities(data.n_items, args.tie_seed)),
        )
    return games, players


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["movielens", "amazon", "gowalla"])
    ap.add_argument("--ml-path", default="data/ml-1m/ratings.dat")
    ap.add_argument("--amazon-path", default="data/amazon-digital-music/interactions.csv")
    ap.add_argument("--gowalla-path", default="data/gowalla/interactions.csv")
    ap.add_argument("--model", default="itemknn", choices=["itemknn", "sasrec", "lightgcn"])
    ap.add_argument("--users", type=int, default=250)
    ap.add_argument("--exact-users", type=int, default=100)
    ap.add_argument("--exact-max", type=int, default=12)
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--evaluation-size", type=int, default=200)
    ap.add_argument("--permutations", type=int, default=250)
    ap.add_argument("--candidate-seed", type=int, default=1729)
    ap.add_argument("--tie-seed", type=int, default=31415)
    ap.add_argument("--user-seed", type=int, default=2718)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--out", default="results/review3")
    args = ap.parse_args()

    if args.dataset == "movielens":
        data = load_movielens_1m(args.ml_path, minimum_interactions=4)
    elif args.dataset == "gowalla":
        data = load_interactions_csv(args.gowalla_path, minimum_interactions=4)
    else:
        data = load_interactions_csv(args.amazon_path, minimum_interactions=4)

    users = sample_evaluation_users(data, args.users, seed=args.user_seed)
    histories = {u: data.seen_before_test(u) for u in users}

    if args.model == "itemknn":
        model = fit_item_knn(histories, data.n_items, neighbours=200)
    elif args.model == "lightgcn":
        model = lightgcn.fit_lightgcn(histories, data.n_items, verbose=True)
    else:
        model = sasrec.fit_sasrec(histories, data.n_items)

    games, players = build_games(args, data, model, users)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for idx, u in enumerate(users):
        game = games[u]
        n = game.players.size
        rec: dict = {"user": int(u), "n_players": int(n)}

        def utility(coalition):
            return target_margin_utility(model, game, coalition)

        # headline attribution + effects
        phi_mc, _ = mc_shapley(utility, n, permutations=args.permutations, seed=42)
        effects = single_player_effects(model, game, rho=args.rho, utility="target_margin")
        rec["aia_shapley"] = aia(phi_mc, effects)

        # exact-Shapley validation subset
        if idx < args.exact_users and n <= args.exact_max:
            exact = exact_shapley(utility, n)
            rec["exact_mc_error"] = mc_error_report(exact, phi_mc)

        # intervention-aware baselines
        rec["aia_bounded_lime_bin"] = aia(
            bounded_lime(model, game, rho=args.rho, continuous=False), effects)
        rec["aia_bounded_lime_cont"] = aia(
            bounded_lime(model, game, rho=args.rho, continuous=True), effects)
        rec["aia_finite_diff"] = aia(finite_difference(model, game), effects)
        rec["aia_ig"] = aia(integrated_gradients(model, game, rho=args.rho), effects)
        rec["aia_binary_lime"] = aia(lime_attribution(utility, n), effects)

        # pair interactions + additive-vs-realized (B=2)
        pair_effects = {
            (p, q): joint_effect(model, game, (p, q), rho=args.rho, utility="target_margin")
            for p, q in combinations(range(n), 2)
        }
        rec["interaction_summary"] = interaction_summary(effects, pair_effects)
        rec["additive_vs_realized"] = additive_vs_realized_rank(
            phi_mc, effects, pair_effects)

        # ablations on realized target-margin effects
        realized = {(): 0.0}
        for p in range(n):
            realized[(p,)] = float(effects[p])
        realized.update({k: float(v) for k, v in pair_effects.items()})
        benefit = -effects
        pair_benefit = {k: float(v) for k, v in realized.items() if len(k) == 2}
        default = select_interaction_aware(benefit, pair_benefit)
        variants = {
            "forced": select_forced(phi_mc, 2),
            "magnitude": select_magnitude(phi_mc, 2),
            "additive_shapley": tuple(
                int(i) for i in np.argsort(benefit, kind="stable")[:2]
                if benefit[np.argsort(benefit, kind="stable")[:2][0]] > 0
            ) or (),
        }
        rec["ablations"] = ablation_report(default, realized, variants)

        # unconditional (full-cohort) regret: inactive oracles count as zero gain
        oracle_eff = max(0.0, float(np.max(effects)) if n else 0.0,
                         max(pair_effects.values(), default=0.0))
        rec["unconditional"] = {
            "oracle_effect": oracle_eff,
            "default_action_effect": realized[default],
            "unconditional_regret": oracle_eff - realized[default],
        }

        # prospective audit (non-target-conditioned) for ALL principal methods
        pg = build_prospective_game(
            model, game.players, game.candidate_items, game.tie_break)
        if pg.target_item != game.target_item:
            peffects = single_player_effects(model, pg, rho=args.rho,
                                             utility="target_margin")
            putil = lambda c: target_margin_utility(model, pg, c)
            pattrs = {
                "shapley": mc_shapley(putil, n, permutations=args.permutations, seed=42)[0],
                "lime_binary": lime_attribution(putil, n),
                "bounded_lime": bounded_lime(model, pg, rho=args.rho, continuous=False),
                "finite_diff": finite_difference(model, pg),
                "ig": integrated_gradients(model, pg, rho=args.rho),
            }
            rec["prospective"] = {
                "target_is_generated_top1": True,
                **{f"aia_{k}_prospective": aia(v, peffects)
                   for k, v in pattrs.items()},
            }
        else:
            rec["prospective"] = {"target_is_generated_top1": False}

        records.append(rec)
        if (idx + 1) % 25 == 0:
            print(f"[review3] {idx + 1}/{len(users)} users done", flush=True)

    # equal-scorer-budget, rho-response, and LIME-kappa curves (review-3 items)
    subsample = users[: min(100, len(users))]
    subsample_n_max = max(games[u].players.size for u in subsample)
    curves: dict = {"equal_budget": [], "rho_curve": [], "kappa_curve": []}
    for M in (100, 250, 500, 1000):
        vals = []
        for u in subsample:
            g = games[u]
            phi, _ = mc_shapley(lambda c: target_margin_utility(model, g, c),
                                g.players.size, permutations=M, seed=42)
            eff = single_player_effects(model, g, rho=args.rho, utility="target_margin")
            vals.append(aia(phi, eff))
        curves["equal_budget"].append(
            {"method": "shapley", "budget": 2 * M * (subsample_n_max + 1),
             "M_pair": M, "mean_bounded_aia": float(np.nanmean(vals))})
    for masks in (128, 512, 1024, 2048):
        vals = []
        for u in subsample:
            g = games[u]
            vals.append(aia(lime_attribution(
                lambda c: target_margin_utility(model, g, c),
                g.players.size, samples=masks),
                single_player_effects(model, g, rho=args.rho,
                                      utility="target_margin")))
        curves["equal_budget"].append(
            {"method": "lime", "budget": masks, "M_pair": None,
             "mean_bounded_aia": float(np.nanmean(vals))})
    for rho in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9):
        vals, mabs = [], []
        for u in subsample:
            g = games[u]
            phi, _ = mc_shapley(lambda c: target_margin_utility(model, g, c),
                                g.players.size, permutations=args.permutations, seed=42)
            eff = single_player_effects(model, g, rho=rho, utility="target_margin")
            vals.append(aia(phi, eff)); mabs.append(float(np.abs(eff).mean()))
        curves["rho_curve"].append({"rho": rho, "mean_bounded_aia": float(np.nanmean(vals)),
                                    "mean_abs_effect": float(np.nanmean(mabs))})
    for kappa in (0.1, 0.25, 0.5, 1.0):
        vals = []
        for u in subsample:
            g = games[u]
            vals.append(aia(bounded_lime(model, g, rho=args.rho, kernel_width=kappa),
                            single_player_effects(model, g, rho=args.rho,
                                                  utility="target_margin")))
        curves["kappa_curve"].append({"kappa": kappa,
                                      "mean_bounded_aia": float(np.nanmean(vals))})

    # dataset audit: eligibility and history-length distribution
    eligible = [u for u in sorted(data.test) if len(data.train[u]) >= 4]
    hlens = [len(data.train[u]) for u in eligible]
    curves["dataset_audit"] = {
        "eligible_users": len(eligible),
        "history_median": float(np.median(hlens)),
        "history_iqr": [float(np.quantile(hlens, .25)), float(np.quantile(hlens, .75))],
    }

    # runtime / memory benchmark on a small user sample
    timings = {}
    sample_game = games[users[0]]
    n0 = sample_game.players.size
    timings["mc_shapley"] = bench(lambda: mc_shapley(
        lambda c: target_margin_utility(model, sample_game, c),
        n0, permutations=args.permutations, seed=42))
    timings["bounded_lime"] = bench(lambda: bounded_lime(model, sample_game))
    timings["loo"] = bench(lambda: leave_one_out(
        lambda c: target_margin_utility(model, sample_game, c), n0))
    timings["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    subsample_n_max = max(games[u].players.size for u in subsample)
    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "config": vars(args),
        "n_users": len(records),
        "records": records,
        "timings": timings,
        "curves": curves,
    }
    target = out_dir / f"review3_{args.dataset}_{args.model}.json"
    target.write_text(json.dumps(payload, indent=1))
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
