#!/usr/bin/env python3
"""Review-9 revision experiments (run on a machine with the datasets).

Addresses the mandatory experimental items of the KBS-style peer review of
the acmart-primary (ACM TORS) manuscript:

* fixed-denominator  [Critical #1] compare normalized relative reweighting
                     with a fixed-denominator pure-suppression scorer;
* utility-factorial  [Critical #4] attribution-utility x outcome-utility
                     factorial to isolate utility mismatch from
                     interaction/nonadditivity;
* prospective        [Critical #5] non-target-conditioned audit of the
                     model's own top-1 recommendation on the full cohort;
* candidate-redraw   [High #8]    repeated independent candidate-set
                     resamples and between-resample variability;
* stratified-null    [High #12]   recency- and popularity-stratified
                     within-user nulls for the AIA calibration;
* compute-matched    [High #13]   Shapley vs LIME at matched scorer-call
                     counts (separate budget-response curves, not equal
                     budgets);
* hardware           [repro]      processor/RAM/peak-RSS capture and
                     repeated per-method timing with variability.

Each subcommand writes one JSON per dataset into --out (default
results/review9). Push the resulting JSONs to the repository so the
manuscript tables can be completed (docs/REVIEW9_EXPERIMENT_GUIDE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from actionshap.baselines import leave_one_out, lime_attribution
from actionshap.candidates import (
    fixed_evaluation_sets,
    global_item_priorities,
    tie_break_for_candidates,
)
from actionshap.evaluation import (
    aia,
    joint_effect,
    signed_alignment,
    single_player_effects,
)
from actionshap.models.itemknn import FixedDenominatorItemKNN, fit_item_knn
from actionshap.recommendation import (
    UserGame,
    mc_shapley,
    profile_utility,
    target_margin_utility,
)
from actionshap.recommendation_data import (
    load_interactions_csv,
    load_movielens_1m,
    sample_evaluation_users,
    truncate_histories,
)


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------

def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dataset", required=True, choices=["movielens", "amazon", "gowalla"])
    ap.add_argument("--ml-path", default="data/ml-1m/ratings.dat")
    ap.add_argument("--amazon-path", default="data/amazon-digital-music/interactions.csv")
    ap.add_argument("--gowalla-path", default="data/gowalla/interactions.csv")
    ap.add_argument("--users", type=int, default=1000)
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--evaluation-size", type=int, default=200)
    ap.add_argument("--permutations", type=int, default=250,
                    help="M_pair base orders for MC Shapley (T=2*M_pair walks)")
    ap.add_argument("--candidate-seed", type=int, default=1729)
    ap.add_argument("--tie-seed", type=int, default=31415)
    ap.add_argument("--user-seed", type=int, default=2718)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--out", default="results/review9")


def load_data(args):
    """Load the temporal dataset for the requested benchmark."""
    if args.dataset == "movielens":
        return load_movielens_1m(args.ml_path, minimum_interactions=4)
    if args.dataset == "gowalla":
        return load_interactions_csv(args.gowalla_path, minimum_interactions=4)
    return load_interactions_csv(args.amazon_path, minimum_interactions=4)


def build_games(args, data, users, candidate_seed=None):
    players = truncate_histories(data, args.n_max)
    eval_sets, _ = fixed_evaluation_sets(
        {u: data.seen_before_test(u) for u in users},
        {u: data.test[u] for u in users},
        data.n_items,
        size=args.evaluation_size,
        seed=args.candidate_seed if candidate_seed is None else candidate_seed,
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


def finite(x):
    return float(x) if x is not None and np.isfinite(x) else None


def summarize(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
    }


def write_json(out_dir: Path, name: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    # Written via a temporary file and os.replace(): a run killed by a timeout, a
    # suspended laptop or a full disk otherwise leaves a truncated JSON behind, and
    # the queue's resume guard treats *any* existing file as a completed job -- so the
    # corrupted run would never be re-run and a table could be typeset from half a file.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------------
# Critical #1: fixed-denominator (pure suppression) vs normalized reweighting
# --------------------------------------------------------------------------

def cmd_fixed_denominator(args) -> None:
    data = load_data(args)
    users = sample_evaluation_users(data, args.users, seed=args.user_seed)
    histories = {u: data.seen_before_test(u) for u in users}
    base_model = fit_item_knn(histories, data.n_items, neighbours=200)
    fixed_model = FixedDenominatorItemKNN(base_model)

    records = []
    for label, model in (("normalized", base_model), ("fixed_denominator", fixed_model)):
        games, _ = build_games(args, data, users)
        for u in users:
            game = games[u]
            n = game.players.size

            def utility(coalition, _m=model, _g=game):
                return target_margin_utility(_m, _g, coalition)

            phi_mc, _ = mc_shapley(
                utility, n, permutations=args.permutations, seed=42)
            phi_lime = lime_attribution(utility, n, samples=512, seed=42)
            phi_loo = leave_one_out(utility, n)
            effects = single_player_effects(
                model, game, rho=args.rho, utility="target_margin")
            effects_del = single_player_effects(
                model, game, rho=0.0, utility="target_margin")
            rec = {
                "user": int(u),
                "n_players": int(n),
                "scorer": label,
                "aia_shapley_bounded": finite(aia(phi_mc, effects)),
                "aia_lime_bounded": finite(aia(phi_lime, effects)),
                "aia_loo_bounded": finite(aia(phi_loo, effects)),
                "aia_shapley_deletion": finite(aia(phi_mc, effects_del)),
                "signed_shapley_bounded": finite(signed_alignment(phi_mc, effects)),
                "signed_lime_bounded": finite(signed_alignment(phi_lime, effects)),
                "mean_abs_effect": float(np.abs(effects).mean()),
            }
            # Review-9 Critical #1: the bounded-minus-deletion gap is the
            # quantity whose interpretation the normalization objection
            # concerns, so record it per user alongside the effect scale.
            bounded = rec["aia_shapley_bounded"]
            deletion = rec["aia_shapley_deletion"]
            rec["gap_shapley"] = (
                float(bounded - deletion)
                if bounded is not None and deletion is not None
                else None
            )
            records.append(rec)

    payload = {
        "dataset": args.dataset,
        "experiment": "fixed_denominator",
        "config": vars(args) | {"seed": 42, "lime_samples": 512},
        "n_users_sampled": int(len(users)),
        "n_users_with_defined_aia": int(
            len({r["user"] for r in records if r["aia_shapley_bounded"] is not None})
        ),
        "records": records,
        "summary": {},
    }
    for label in ("normalized", "fixed_denominator"):
        subset = [r for r in records if r["scorer"] == label]
        payload["summary"][label] = {
            key: summarize([r[key] for r in subset])
            for key in (
                "aia_shapley_bounded",
                "aia_lime_bounded",
                "aia_loo_bounded",
                "aia_shapley_deletion",
                "gap_shapley",
                "signed_shapley_bounded",
                "signed_lime_bounded",
                "mean_abs_effect",
            )
        }
    # Paired user-level contrast: normalized relative reweighting versus the
    # fixed-denominator pure-suppression scorer.  Both scorers share the user
    # set, candidate sets, permutation seed and intervention strength, so the
    # only difference is the normalization of the retained weight sum.
    by_user: dict[str, dict[int, dict]] = {
        label: {r["user"]: r for r in records if r["scorer"] == label}
        for label in ("normalized", "fixed_denominator")
    }
    shared = sorted(set(by_user["normalized"]) & set(by_user["fixed_denominator"]))
    paired: dict[str, dict[str, float]] = {}
    for key in (
        "aia_shapley_bounded",
        "aia_lime_bounded",
        "aia_shapley_deletion",
        "gap_shapley",
        "signed_shapley_bounded",
        "mean_abs_effect",
    ):
        diff = [
            by_user["normalized"][u][key] - by_user["fixed_denominator"][u][key]
            for u in shared
            if by_user["normalized"][u][key] is not None
            and by_user["fixed_denominator"][u][key] is not None
        ]
        d = np.asarray(diff, dtype=float)
        if d.size == 0:
            continue
        rng = np.random.default_rng(
            (42, sum(ord(character) for character in key) + len(key))
        )
        draws = 10_000
        idx = rng.integers(0, d.size, size=(draws, d.size))
        boots = d[idx].mean(axis=1)
        # Plus-one paired sign-flip test: independent Rademacher signs on the
        # user-level differences, valid under symmetry of the difference
        # distribution about its null value (stated in the manuscript).
        signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, d.size))
        flipped = (signs * d).mean(axis=1)
        paired[key] = {
            "n": int(d.size),
            "mean_difference": float(d.mean()),
            "ci95_low": float(np.percentile(boots, 2.5)),
            "ci95_high": float(np.percentile(boots, 97.5)),
            "sign_flip_p": float((1 + int((np.abs(flipped) >= abs(d.mean())).sum())) / (draws + 1)),
            "cohens_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
        }
    payload["paired"] = paired
    write_json(Path(args.out), f"fixed_denominator_{args.dataset}.json", payload)


# --------------------------------------------------------------------------
# Critical #4: attribution-utility x outcome-utility factorial
# --------------------------------------------------------------------------

def _utility_fn(name, model, game):
    if name == "target_margin":
        return lambda coalition: target_margin_utility(model, game, coalition)
    if name == "ndcg":
        return lambda coalition: profile_utility(model, game, coalition)
    raise ValueError(name)


def cmd_utility_factorial(args) -> None:
    data = load_data(args)
    users = sample_evaluation_users(data, args.users, seed=args.user_seed)
    histories = {u: data.seen_before_test(u) for u in users}
    model = fit_item_knn(histories, data.n_items, neighbours=200)
    games, _ = build_games(args, data, users)

    records = []
    for u in users:
        game = games[u]
        n = game.players.size
        utilities = {
            name: _utility_fn(name, model, game)
            for name in ("target_margin", "ndcg")
        }
        outcome_effects = {
            name: single_player_effects(
                model, game, rho=args.rho, utility=name)
            for name in ("target_margin", "ndcg")
        }
        # exact B=2 oracle effects per outcome utility (no action included)
        oracle_effect = {}
        for name in ("target_margin", "ndcg"):
            best = 0.0
            eff = outcome_effects[name]
            for p in range(n):
                best = max(best, float(eff[p]))
            for p, q in combinations(range(n), 2):
                best = max(
                    best,
                    joint_effect(model, game, (p, q), rho=args.rho, utility=name),
                )
            oracle_effect[name] = best
        for attr_util in ("target_margin", "ndcg"):
            phi, _ = mc_shapley(
                utilities[attr_util], n, permutations=args.permutations, seed=42)
            benefit = -np.asarray(phi, dtype=float)
            order = np.argsort(-benefit, kind="stable")
            # deterministic additive selection: top-B strictly positive benefits
            eligible = [int(p) for p in order if benefit[p] > 1e-12]
            selected = tuple(eligible[:2])
            for out_util in ("target_margin", "ndcg"):
                realized = joint_effect(
                    model, game, selected, rho=args.rho, utility=out_util)
                records.append({
                    "user": int(u),
                    "n_players": int(n),
                    "attr_utility": attr_util,
                    "outcome_utility": out_util,
                    "aia_matched": finite(aia(phi, outcome_effects[attr_util])),
                    "aia_cross": finite(aia(phi, outcome_effects[out_util])),
                    "action": list(selected),
                    "realized_effect": finite(realized),
                    "oracle_effect": finite(oracle_effect[out_util]),
                    "regret": finite(oracle_effect[out_util] - realized),
                })

    payload = {
        "dataset": args.dataset,
        "experiment": "utility_factorial",
        "config": vars(args) | {"seed": 42},
        "records": records,
        "summary": {},
    }
    for attr_util in ("target_margin", "ndcg"):
        for out_util in ("target_margin", "ndcg"):
            cell = [
                r for r in records
                if r["attr_utility"] == attr_util
                and r["outcome_utility"] == out_util
            ]
            payload["summary"][f"{attr_util}__x__{out_util}"] = {
                "aia_matched": summarize([r["aia_matched"] for r in cell]),
                "aia_cross": summarize([r["aia_cross"] for r in cell]),
                "realized_effect": summarize([r["realized_effect"] for r in cell]),
                "regret": summarize([r["regret"] for r in cell]),
                "active_oracles": int(sum(
                    1 for r in cell
                    if r["oracle_effect"] is not None and r["oracle_effect"] > 1e-12
                )),
            }
    write_json(Path(args.out), f"utility_factorial_{args.dataset}.json", payload)


# --------------------------------------------------------------------------
# Critical #5: prospective (non-target-conditioned) audit, full cohort
# --------------------------------------------------------------------------

def cmd_prospective(args) -> None:
    from actionshap.prospective import build_prospective_game

    data = load_data(args)
    users = sample_evaluation_users(data, args.users, seed=args.user_seed)
    histories = {u: data.seen_before_test(u) for u in users}
    model = fit_item_knn(histories, data.n_items, neighbours=200)
    games, players = build_games(args, data, users)

    records = []
    for u in users:
        game = games[u]
        n = game.players.size
        try:
            pgame = build_prospective_game(
                model, players[u], game.candidate_items, game.tie_break)
        except ValueError:
            records.append({"user": int(u), "n_players": int(n), "status": "no_candidate"})
            continue
        covers_target = int(pgame.target_item) == int(game.target_item)
        if pgame.target_item not in set(int(c) for c in game.candidate_items):
            records.append({
                "user": int(u), "n_players": int(n),
                "status": "top1_outside_candidates",
                "prospective_target": int(pgame.target_item),
                "covers_heldout_target": int(covers_target),
            })
            continue

        def utility(coalition, _g=pgame):
            return target_margin_utility(model, _g, coalition)

        phi, _ = mc_shapley(utility, n, permutations=args.permutations, seed=42)
        phi_lime = lime_attribution(utility, n, samples=512, seed=42)
        phi_loo = leave_one_out(utility, n)
        effects = single_player_effects(
            model, pgame, rho=args.rho, utility="target_margin")
        records.append({
            "user": int(u),
            "n_players": int(n),
            "status": "audited",
            "prospective_target": int(pgame.target_item),
            "covers_heldout_target": int(covers_target),
            "aia_shapley": finite(aia(phi, effects)),
            "aia_lime": finite(aia(phi_lime, effects)),
            "aia_loo": finite(aia(phi_loo, effects)),
            "signed_shapley": finite(signed_alignment(phi, effects)),
        })

    audited = [r for r in records if r["status"] == "audited"]
    payload = {
        "dataset": args.dataset,
        "experiment": "prospective_full_cohort",
        "config": vars(args) | {"seed": 42, "lime_samples": 512},
        "users_total": len(users),
        "users_audited": len(audited),
        "summary": {
            key: summarize([r[key] for r in audited])
            for key in ("aia_shapley", "aia_lime", "aia_loo", "signed_shapley")
        },
        "covers_heldout_target_fraction": (
            float(np.mean([r["covers_heldout_target"] for r in audited]))
            if audited else None
        ),
        "records": records,
    }
    write_json(Path(args.out), f"prospective_{args.dataset}.json", payload)


# --------------------------------------------------------------------------
# High #8: independent candidate-set resamples
# --------------------------------------------------------------------------

def cmd_candidate_redraw(args) -> None:
    data = load_data(args)
    users = sample_evaluation_users(data, args.users, seed=args.user_seed)
    histories = {u: data.seen_before_test(u) for u in users}
    model = fit_item_knn(histories, data.n_items, neighbours=200)

    redraws = []
    for k in range(args.redraws):
        seed_k = args.candidate_seed + 1000 * (k + 1)
        games, _ = build_games(args, data, users, candidate_seed=seed_k)
        per_method = {"shapley": [], "lime": [], "loo": []}
        for u in users:
            game = games[u]
            n = game.players.size

            def utility(coalition, _g=game):
                return target_margin_utility(model, _g, coalition)

            phi, _ = mc_shapley(
                utility, n, permutations=args.permutations, seed=42)
            phi_lime = lime_attribution(utility, n, samples=512, seed=42)
            phi_loo = leave_one_out(utility, n)
            effects = single_player_effects(
                model, game, rho=args.rho, utility="target_margin")
            per_method["shapley"].append(finite(aia(phi, effects)))
            per_method["lime"].append(finite(aia(phi_lime, effects)))
            per_method["loo"].append(finite(aia(phi_loo, effects)))
        redraws.append({
            "redraw": k,
            "candidate_seed": seed_k,
            "summary": {
                m: summarize(vals) for m, vals in per_method.items()
            },
            "per_user": {
                m: [finite(v) for v in vals] for m, vals in per_method.items()
            },
        })

    payload = {
        "dataset": args.dataset,
        "experiment": "candidate_redraw",
        "config": vars(args),
        "redraws": redraws,
        "between_redraw": {
            # A redraw can be entirely undefined for a tiny cohort (aia() needs a
            # non-degenerate attribution spectrum), so read the mean defensively:
            # summarize() already drops the Nones this leaves behind.
            m: summarize([r["summary"][m].get("mean") for r in redraws])
            | {"redraws_with_data": sum(1 for r in redraws if r["summary"][m].get("n"))}
            for m in ("shapley", "lime", "loo")
        },
    }
    write_json(Path(args.out), f"candidate_redraw_{args.dataset}.json", payload)


# --------------------------------------------------------------------------
# High #12: stratified within-user nulls
# --------------------------------------------------------------------------

def _stratified_shuffle_indices(rng, n, strata):
    """Permute positions within each stratum (block)."""
    idx = np.arange(n)
    for block in np.unique(strata):
        pos = np.flatnonzero(strata == block)
        shuffled = rng.permutation(pos)
        idx[pos] = shuffled
    return idx


def cmd_stratified_null(args) -> None:
    from scipy.stats import spearmanr

    data = load_data(args)
    users = sample_evaluation_users(data, args.users, seed=args.user_seed)
    histories = {u: data.seen_before_test(u) for u in users}
    model = fit_item_knn(histories, data.n_items, neighbours=200)
    games, players = build_games(args, data, users)

    # item popularity ranks from complete training histories
    counts = np.zeros(data.n_items, dtype=int)
    for u in users:
        for it in np.unique(data.seen_before_test(u)):
            counts[int(it)] += 1
    pop_rank = np.argsort(np.argsort(-counts, kind="stable"), kind="stable")

    schemes = ("free", "recency_blocks", "popularity_blocks")
    observed = {s: [] for s in schemes}
    nulls = {s: [] for s in schemes}

    for u in users:
        game = games[u]
        n = game.players.size
        if n < 4:
            continue

        def utility(coalition, _g=game):
            return target_margin_utility(model, _g, coalition)

        phi, _ = mc_shapley(utility, n, permutations=args.permutations, seed=42)
        effects = single_player_effects(
            model, game, rho=args.rho, utility="target_margin")
        if np.std(effects) == 0 or np.std(phi) == 0:
            continue
        obs = float(spearmanr(np.abs(phi), np.abs(effects)).statistic)

        positions = np.arange(n)
        recency = np.digitize(
            positions, [n // 3, 2 * n // 3])  # 0 oldest .. 2 newest
        popularity = np.digitize(
            pop_rank[players[u]],
            [np.quantile(pop_rank[players[u]], 1 / 3),
             np.quantile(pop_rank[players[u]], 2 / 3)])
        strata_map = {
            "free": np.zeros(n, dtype=int),  # single block = free shuffle
            "recency_blocks": recency,
            "popularity_blocks": popularity,
        }
        rng = np.random.default_rng((42, int(u), 9))
        for scheme in schemes:
            observed[scheme].append(obs)
            draws = []
            for _ in range(args.r_null):
                idx = _stratified_shuffle_indices(rng, n, strata_map[scheme])
                shuffled = np.abs(effects)[idx]
                val = spearmanr(np.abs(phi), shuffled).statistic
                if np.isfinite(val):
                    draws.append(float(val))
            nulls[scheme].append(draws)

    summary = {}
    for scheme in schemes:
        obs_mean = float(np.mean(observed[scheme])) if observed[scheme] else None
        # dataset-level null: mean over users of matched null draws
        null_means = []
        max_r = max((len(d) for d in nulls[scheme]), default=0)
        for r in range(max_r):
            vals = [d[r] for d in nulls[scheme] if r < len(d)]
            if vals:
                null_means.append(float(np.mean(vals)))
        summary[scheme] = {
            "n_users": len(observed[scheme]),
            "observed_mean": obs_mean,
            "null_mean": summarize(null_means),
            "plus_one_p": (
                float((1 + sum(1 for m in null_means if m >= obs_mean))
                      / (len(null_means) + 1))
                if null_means and obs_mean is not None else None
            ),
        }
    payload = {
        "dataset": args.dataset,
        "experiment": "stratified_null",
        "config": vars(args),
        "summary": summary,
    }
    write_json(Path(args.out), f"stratified_null_{args.dataset}.json", payload)


# --------------------------------------------------------------------------
# High #13: matched scorer-call budget-response curves
# --------------------------------------------------------------------------

def cmd_compute_matched(args) -> None:
    data = load_data(args)
    users = sample_evaluation_users(data, args.users, seed=args.user_seed)
    histories = {u: data.seen_before_test(u) for u in users}
    model = fit_item_knn(histories, data.n_items, neighbours=200)
    games, _ = build_games(args, data, users)

    curves = []
    for m_pair in args.mpair_grid:
        shapley_aia, lime_aia, calls = [], [], []
        for u in users:
            game = games[u]
            n = game.players.size

            def utility(coalition, _g=game):
                return target_margin_utility(model, _g, coalition)

            phi, _ = mc_shapley(utility, n, permutations=m_pair, seed=42)
            # matched scorer-call count: reverse-paired prefix walks use
            # T*(n+1) coalition requests before cache reuse; give LIME the
            # same number of mask evaluations.
            matched_masks = int(2 * m_pair * (n + 1))
            phi_lime = lime_attribution(
                utility, n, samples=matched_masks, seed=42)
            effects = single_player_effects(
                model, game, rho=args.rho, utility="target_margin")
            shapley_aia.append(finite(aia(phi, effects)))
            lime_aia.append(finite(aia(phi_lime, effects)))
            calls.append(matched_masks)
        curves.append({
            "m_pair": m_pair,
            "matched_scorer_calls_per_user": int(np.mean(calls)),
            "shapley": summarize(shapley_aia),
            "lime_matched": summarize(lime_aia),
            "lime_primary_512": None,
        })
    # primary LIME reference at its declared 512 masks
    lime_primary = []
    for u in users:
        game = games[u]
        n = game.players.size

        def utility(coalition, _g=game):
            return target_margin_utility(model, _g, coalition)

        phi_lime = lime_attribution(utility, n, samples=512, seed=42)
        effects = single_player_effects(
            model, game, rho=args.rho, utility="target_margin")
        lime_primary.append(finite(aia(phi_lime, effects)))
    for curve in curves:
        curve["lime_primary_512"] = summarize(lime_primary)

    payload = {
        "dataset": args.dataset,
        "experiment": "compute_matched_budget_response",
        "note": (
            "Each row is a separate budget-response point; Shapley and "
            "matched LIME use the same scorer-call count at that row. Rows "
            "are not equal-budget comparisons across methods at the primary "
            "configuration."
        ),
        "config": vars(args),
        "curves": curves,
    }
    write_json(Path(args.out), f"compute_matched_{args.dataset}.json", payload)


# --------------------------------------------------------------------------
# reproducibility: hardware + repeated per-method timings
# --------------------------------------------------------------------------

def peak_rss_mb(ru_maxrss: int, sys_platform: str) -> float:
    """Convert ``getrusage().ru_maxrss`` to megabytes.

    Linux reports kilobytes and Darwin reports bytes, so a single divisor is wrong by 1024x on a Mac.
    """
    return round(ru_maxrss / (1024 * 1024 if sys_platform.startswith("darwin") else 1024), 4)


def cmd_hardware(args) -> None:
    import resource

    data = load_data(args)
    users = sample_evaluation_users(data, min(args.users, 50), seed=args.user_seed)
    histories = {u: data.seen_before_test(u) for u in users}
    model = fit_item_knn(histories, data.n_items, neighbours=200)
    games, _ = build_games(args, data, users)

    timings = {"shapley": [], "lime": [], "loo": []}
    for rep in range(args.timing_repeats):
        for u in users[: args.timing_users]:
            game = games[u]
            n = game.players.size

            def utility(coalition, _g=game):
                return target_margin_utility(model, _g, coalition)

            t0 = time.perf_counter()
            mc_shapley(utility, n, permutations=args.permutations, seed=42)
            timings["shapley"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            lime_attribution(utility, n, samples=512, seed=42)
            timings["lime"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            leave_one_out(utility, n)
            timings["loo"].append(time.perf_counter() - t0)

    rss_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_mb = peak_rss_mb(rss_raw, sys.platform)
    payload = {
        "dataset": args.dataset,
        "experiment": "hardware_and_timing",
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": __import__("os").cpu_count(),
            "numpy": np.__version__,
        },
        "peak_rss_mb": peak_mb,
        "peak_rss_raw": rss_raw,
        "timing_repeats": args.timing_repeats,
        "timings_seconds": {
            m: summarize(vals) for m, vals in timings.items()
        },
        "config": vars(args),
    }
    write_json(Path(args.out), f"hardware_{args.dataset}.json", payload)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    parsers = {
        "fixed-denominator": cmd_fixed_denominator,
        "utility-factorial": cmd_utility_factorial,
        "prospective": cmd_prospective,
        "candidate-redraw": cmd_candidate_redraw,
        "stratified-null": cmd_stratified_null,
        "compute-matched": cmd_compute_matched,
        "hardware": cmd_hardware,
    }
    for name in parsers:
        sp = sub.add_parser(name)
        add_common_args(sp)
        if name == "candidate-redraw":
            sp.add_argument("--redraws", type=int, default=20)
        if name == "stratified-null":
            sp.add_argument("--r-null", type=int, default=1000)
        if name == "compute-matched":
            sp.add_argument("--mpair-grid", type=int, nargs="+",
                            default=[25, 50, 100, 250])
        if name == "hardware":
            sp.add_argument("--timing-repeats", type=int, default=3)
            sp.add_argument("--timing-users", type=int, default=20)

    args = ap.parse_args()
    parsers[args.command](args)


if __name__ == "__main__":
    main()
