#!/usr/bin/env python3
"""Review-5 revision experiments (run on a machine with the datasets).

Subcommands (each writes one JSON into --out, default results/review5):

* lime-masks           mask-design ablation for LIME (bernoulli / unique /
                       enumerate) plus ridge-alpha sweep  [mandatory #6]
* sasrec-quality       SASRec recommendation quality vs popularity and the
                       masking gate, five seeds            [mandatory #1]
* exact-dist           per-user exact-Shapley error distribution, top-2
                       action Jaccard, sign error         [mandatory #3 ext]
* variance-components  profile-model variance decomposition: fixed model x
                       attribution seeds vs model seeds x fixed attribution
                                                            [stats #4]
* convergence-quantiles post-process an existing raw convergence JSON into
                       per-budget quantiles + coverage panels [mandatory #4]

Push the resulting JSONs to the repository so the manuscript can integrate
them (see docs/REVIEW5_EXPERIMENT_GUIDE.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from actionshap.baselines import lime_attribution
from actionshap.candidates import (
    fixed_evaluation_sets,
    global_item_priorities,
    tie_break_for_candidates,
)
from actionshap.evaluation import aia, single_player_effects
from actionshap.exact_shapley import exact_shapley
from actionshap.models.itemknn import fit_item_knn
from actionshap.models.profile import fit_item_embeddings
from actionshap.recommendation import UserGame, mc_shapley, target_margin_utility
from actionshap.recommendation_data import (
    load_interactions_csv,
    load_movielens_1m,
    sample_evaluation_users,
    truncate_histories,
)


def load_data(args):
    if args.dataset == "movielens":
        return load_movielens_1m(args.ml_path, minimum_interactions=4)
    return load_interactions_csv(args.amazon_path, minimum_interactions=4)


def build_games(args, data, users):
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
                cands, global_item_priorities(data.n_items, args.tie_seed)
            ),
        )
    return games, players


def quantiles(values, qs=(0.5, 0.9, 0.95)):
    values = np.asarray(values, dtype=float)
    return {f"p{int(q*100)}": float(np.quantile(values, q)) for q in qs}


def save(out_dir: Path, name: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=1, allow_nan=False))
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------------
# lime-masks
# --------------------------------------------------------------------------
def cmd_lime_masks(args) -> None:
    data = load_data(args)
    users = sample_evaluation_users(data, args.users, seed=args.user_seed)
    histories = {u: data.seen_before_test(u) for u in users}
    model = fit_item_knn(histories, data.n_items, neighbours=200)
    games, _ = build_games(args, data, users)
    designs = ["bernoulli", "unique", "enumerate"]
    records = []
    for i, u in enumerate(users):
        g = games[u]
        n = g.players.size
        eff = single_player_effects(model, g, rho=args.rho, utility="target_margin")
        for design in designs:
            if design == "enumerate" and 2 ** n > args.samples:
                continue  # enumeration infeasible at the design budget
            phi = lime_attribution(
                lambda c: target_margin_utility(model, g, c),
                n, samples=args.samples, seed=42, mask_design=design,
            )
            records.append(dict(user=int(u), n_players=int(n), design=design,
                                ridge_alpha=1.0, bounded_aia=float(aia(phi, eff))))
        for alpha in args.ridge_sweep:
            phi = lime_attribution(
                lambda c: target_margin_utility(model, g, c),
                n, samples=args.samples, seed=42, ridge_alpha=alpha,
            )
            records.append(dict(user=int(u), n_players=int(n), design="bernoulli",
                                ridge_alpha=float(alpha),
                                bounded_aia=float(aia(phi, eff))))
        if (i + 1) % 25 == 0:
            print(f"[lime-masks] {i+1}/{len(users)} users done", flush=True)
    payload = {
        "dataset": args.dataset, "model": "itemknn",
        "config": {"users": args.users, "samples": args.samples, "rho": args.rho,
                   "ridge_sweep": args.ridge_sweep, "user_seed": args.user_seed},
        "records": records,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save(Path(args.out), f"lime_mask_ablation_{args.dataset}.json", payload)


# --------------------------------------------------------------------------
# sasrec-quality
# --------------------------------------------------------------------------
def cmd_sasrec_quality(args) -> None:
    from actionshap import sasrec as srec

    data = load_data(args)
    cohort = sample_evaluation_users(data, args.users, seed=args.user_seed)
    if args.train_all:
        train_users = [u for u in data.test if len(data.train[u]) >= 4]
        histories = {u: data.seen_before_test(u) for u in train_users}
        print(f"[sasrec-quality] training on all {len(histories)} eligible users", flush=True)
    else:
        histories = {u: data.seen_before_test(u) for u in cohort}
    popularity = np.zeros(data.n_items)
    for u in data.train:
        for it in np.unique(data.train[u]):
            popularity[it] += 1
    results = []
    for seed in range(args.seeds):
        adapter = srec.fit_sasrec(histories, data.n_items, seed=42 + seed,
                                  epochs=args.epochs)
        stats = dict(ndcg=[], hr=[], mrr=[], ndcg_pop=[], hr_pop=[], mrr_pop=[],
                     mask_delta=[])
        for u in cohort:
            hist = np.asarray(data.train[u][-args.max_len:])
            seen = set(int(x) for x in data.seen_before_test(u))
            all_items = np.asarray([i for i in range(data.n_items) if i not in seen])
            target = int(data.test[u])
            scores = adapter.score(hist, all_items)
            rank = int(np.sum(scores > scores[all_items == target][0])) + 1
            stats["ndcg"].append(1.0 / np.log2(rank + 1) if rank <= 10 else 0.0)
            stats["hr"].append(float(rank <= 10))
            stats["mrr"].append(1.0 / rank)
            pop = popularity[all_items]
            rank_pop = int(np.sum(pop > popularity[target])) + 1
            stats["ndcg_pop"].append(1.0 / np.log2(rank_pop + 1) if rank_pop <= 10 else 0.0)
            stats["hr_pop"].append(float(rank_pop <= 10))
            stats["mrr_pop"].append(1.0 / rank_pop)
            # masking gate: random half-history zeroed, mean |dNDCG@10|
            m_rng = np.random.default_rng(args.tie_seed + u)
            mask = m_rng.binomial(1, 0.5, size=len(hist)).astype(float)
            if mask.sum() >= 1:
                s_m = adapter.score(hist, all_items, weights=mask)
                rank_m = int(np.sum(s_m > s_m[all_items == target][0])) + 1
                nd_m = 1.0 / np.log2(rank_m + 1) if rank_m <= 10 else 0.0
                stats["mask_delta"].append(abs(nd_m - stats["ndcg"][-1]))
        results.append(dict(
            seed=42 + seed,
            ndcg10=float(np.mean(stats["ndcg"])), hr10=float(np.mean(stats["hr"])),
            mrr=float(np.mean(stats["mrr"])),
            ndcg10_popularity=float(np.mean(stats["ndcg_pop"])),
            hr10_popularity=float(np.mean(stats["hr_pop"])),
            mrr_popularity=float(np.mean(stats["mrr_pop"])),
            mean_abs_masking_delta=float(np.mean(stats["mask_delta"])),
            masking_gate_pass=bool(np.mean(stats["mask_delta"]) > 0.001),
            beats_popularity=bool(np.mean(stats["ndcg"]) > np.mean(stats["ndcg_pop"])),
        ))
        print(f"[sasrec-quality] seed {42+seed}: NDCG@10 {results[-1]['ndcg10']:.4f} "
              f"vs popularity {results[-1]['ndcg10_popularity']:.4f}", flush=True)
    payload = {
        "dataset": args.dataset, "n_users": len(cohort), "seeds": args.seeds,
        "max_len": args.max_len, "user_seed": args.user_seed,
        "train_all": bool(args.train_all), "epochs": args.epochs,
        "results": results,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    suffix = "trainall" if args.train_all else args.dataset
    name = (f"sasrec_quality_trainall_{args.dataset}.json" if args.train_all
            else f"sasrec_quality_{args.dataset}.json")
    save(Path(args.out), name, payload)


# --------------------------------------------------------------------------
# exact-dist
# --------------------------------------------------------------------------
def signed_top_action(phi: np.ndarray, budget: int = 2, eps: float = 1e-12):
    benefit = -phi
    eligible = [p for p in range(len(phi)) if benefit[p] > eps]
    eligible.sort(key=lambda p: (-benefit[p], p))
    return set(eligible[:budget])


def cmd_exact_dist(args) -> None:
    data = load_data(args)
    users = sample_evaluation_users(data, args.users, seed=args.user_seed)
    histories = {u: data.seen_before_test(u) for u in users}
    model = fit_item_knn(histories, data.n_items, neighbours=200)
    games, players = build_games(args, data, users)
    records = []
    for u in users:
        g = games[u]
        n = g.players.size
        if n > args.exact_max:
            continue
        util = lambda c: target_margin_utility(model, g, c)
        phi_exact = exact_shapley(util, n)
        phi_mc, _ = mc_shapley(util, n, permutations=args.permutations, seed=42)
        eff = single_player_effects(model, g, rho=args.rho, utility="target_margin")
        a_exact = signed_top_action(phi_exact)
        a_mc = signed_top_action(phi_mc)
        union = a_exact | a_mc
        records.append(dict(
            user=int(u), n_players=int(n),
            max_abs_error=float(np.max(np.abs(phi_mc - phi_exact))),
            rank_spearman=(float(aia(phi_mc, phi_exact)) if np.std(phi_exact) > 0
                           and np.std(phi_mc) > 0 else None),
            top2_jaccard=(len(a_exact & a_mc) / len(union)) if union else 1.0,
            sign_error=float(np.mean(np.sign(-phi_mc + 0.0) != np.sign(-phi_exact + 0.0))),
            bounded_aia_exact=float(aia(phi_exact, eff)) if np.std(eff) > 0 else None,
        ))
        if len(records) % 25 == 0:
            print(f"[exact-dist] {len(records)} enumerated users", flush=True)
        if len(records) >= args.exact_users:
            break
    errs = [r["max_abs_error"] for r in records]
    payload = {
        "dataset": args.dataset, "n_users": len(records),
        "config": {"exact_max": args.exact_max, "permutations": args.permutations},
        "aggregate": {
            "max_abs_error": {"mean": float(np.mean(errs)), **quantiles(errs)},
            "top2_jaccard": {"mean": float(np.mean([r["top2_jaccard"] for r in records])),
                             **quantiles([r["top2_jaccard"] for r in records])},
            "sign_error": {"mean": float(np.mean([r["sign_error"] for r in records]))},
        },
        "records": records,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save(Path(args.out), f"exact_shapley_distribution_{args.dataset}.json", payload)


# --------------------------------------------------------------------------
# variance-components
# --------------------------------------------------------------------------
def cmd_variance_components(args) -> None:
    data = load_data(args)
    users = sample_evaluation_users(data, args.users, seed=args.user_seed)
    histories = {u: data.seen_before_test(u) for u in users}
    games, players = build_games(args, data, users)
    attr_seeds = list(range(42, 42 + args.attr_seeds))
    model_seeds = list(range(42, 42 + args.model_seeds))

    def aias(model):
        out = {}
        for u in games:
            g = games[u]
            eff = single_player_effects(model, g, rho=args.rho, utility="target_margin")
            out[u] = eff
        return out

    fixed_model = fit_item_embeddings(histories, data.n_items, seed=42)
    eff_fixed = aias(fixed_model)
    attr_matrix = []
    for s in attr_seeds:
        row = {}
        for u, g in games.items():
            phi, _ = mc_shapley(lambda c: target_margin_utility(fixed_model, g, c),
                                g.players.size, permutations=args.permutations, seed=s)
            row[u] = float(aia(phi, eff_fixed[u]))
        attr_matrix.append(row)
        print(f"[variance] fixed model, attribution seed {s} done", flush=True)
    model_matrix = []
    for ms in model_seeds:
        m = fit_item_embeddings(histories, data.n_items, seed=ms)
        eff_m = aias(m)
        row = {}
        for u, g in games.items():
            phi, _ = mc_shapley(lambda c: target_margin_utility(m, g, c),
                                g.players.size, permutations=args.permutations, seed=42)
            row[u] = float(aia(phi, eff_m[u]))
        model_matrix.append(row)
        print(f"[variance] model seed {ms} done", flush=True)

    def varcomp(matrix):
        arr = np.array([[row[u] for u in sorted(row)] for row in matrix])
        per_user_var = np.nanvar(arr, axis=0, ddof=1)
        return {"mean_of_user_variances": float(np.nanmean(per_user_var)),
                "variance_of_user_means": float(np.nanvar(np.nanmean(arr, axis=0), ddof=1)),
                "cell_mean": float(np.nanmean(arr))}

    payload = {
        "dataset": args.dataset, "model": "profile", "n_users": len(games),
        "config": {"attr_seeds": attr_seeds, "model_seeds": model_seeds,
                   "permutations": args.permutations, "rho": args.rho},
        "fixed_model_x_attr_seeds": varcomp(attr_matrix),
        "model_seeds_x_fixed_attr": varcomp(model_matrix),
        "attr_matrix": attr_matrix, "model_matrix": model_matrix,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save(Path(args.out), f"variance_components_{args.dataset}.json", payload)


# --------------------------------------------------------------------------
# convergence-quantiles
# --------------------------------------------------------------------------
def cmd_convergence_quantiles(args) -> None:
    raw = json.loads(Path(args.raw).read_text())
    rows = raw.get("per_user_rows", [])
    budgets = sorted({r["permutations"] for r in rows})
    out_rows = []
    for b in budgets:
        sub = [r for r in rows if r["permutations"] == b]
        ranks = [r["mean_rank_correlation_to_reference"] for r in sub
                 if r.get("mean_rank_correlation_to_reference") is not None]
        jacs = [r["mean_top2_jaccard"] for r in sub
                if r.get("mean_top2_jaccard") is not None]
        both = [r for r in sub
                if r.get("mean_rank_correlation_to_reference") is not None
                and r.get("mean_top2_jaccard") is not None]
        passing = [r for r in both
                   if r["mean_rank_correlation_to_reference"] >= 0.95
                   and r["mean_top2_jaccard"] >= 0.80]
        out_rows.append(dict(
            budget=b, n_rank_valid=len(ranks),
            rank_mean=float(np.mean(ranks)) if ranks else None,
            rank_median=float(np.median(ranks)) if ranks else None,
            rank_p5=float(np.percentile(ranks, 5)) if ranks else None,
            rank_p25=float(np.percentile(ranks, 25)) if ranks else None,
            jaccard_mean=float(np.mean(jacs)) if jacs else None,
            jaccard_median=float(np.median(jacs)) if jacs else None,
            threshold_coverage=len(passing) / len(both) if both else None,
            rank_valid_fraction=len(ranks) / len(sub) if sub else None,
        ))
    payload = {"source": str(Path(args.raw).name), "budgets": out_rows,
               "criterion": {"rank": 0.95, "jaccard": 0.80},
               "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save(Path(args.out), f"convergence_quantiles_{Path(args.raw).stem}.json", payload)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--dataset", required=True, choices=["movielens", "amazon"])
        p.add_argument("--ml-path", default="data/ml-1m/ratings.dat")
        p.add_argument("--amazon-path", default="data/amazon-digital-music/interactions.csv")
        p.add_argument("--n-max", type=int, default=20)
        p.add_argument("--evaluation-size", type=int, default=200)
        p.add_argument("--candidate-seed", type=int, default=1729)
        p.add_argument("--tie-seed", type=int, default=31415)
        p.add_argument("--user-seed", type=int, default=2718)
        p.add_argument("--rho", type=float, default=0.5)
        p.add_argument("--out", default="results/review5")

    p = sub.add_parser("lime-masks")
    common(p)
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--ridge-sweep", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    p.set_defaults(func=cmd_lime_masks)

    p = sub.add_parser("sasrec-quality")
    common(p)
    p.add_argument("--users", type=int, default=1000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max-len", type=int, default=20)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train-all", action="store_true",
                   help="train on every eligible user's history (competitive-model audit)")
    p.set_defaults(func=cmd_sasrec_quality)

    p = sub.add_parser("exact-dist")
    common(p)
    p.add_argument("--users", type=int, default=1000)
    p.add_argument("--exact-users", type=int, default=300)
    p.add_argument("--exact-max", type=int, default=12)
    p.add_argument("--permutations", type=int, default=250)
    p.set_defaults(func=cmd_exact_dist)

    p = sub.add_parser("variance-components")
    common(p)
    p.add_argument("--users", type=int, default=100)
    p.add_argument("--attr-seeds", type=int, default=5)
    p.add_argument("--model-seeds", type=int, default=5)
    p.add_argument("--permutations", type=int, default=100)
    p.set_defaults(func=cmd_variance_components)

    p = sub.add_parser("convergence-quantiles")
    p.add_argument("--raw", required=True, help="path to raw convergence JSON")
    p.add_argument("--out", default="results/review5")
    p.set_defaults(func=cmd_convergence_quantiles)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
