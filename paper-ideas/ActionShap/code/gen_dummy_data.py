#!/usr/bin/env python3
"""Generate minimal schema-v2 dummy data that passes enough of validate()
to reach make_actionability_gap_assets (for testing the offsets/colors fix).
"""
import json
from pathlib import Path
import numpy as np

out = Path("results/schema-v2")
out.mkdir(parents=True, exist_ok=True)
SHA = "506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f"
methods = ["shapley_mc", "lime", "loo", "greedy_cf", "random"]
np.random.seed(0)

def mk(ds, role, cond, seed, nu=5, nmax=20, rho=0.5, budget=2, evalsz=200, utility="target_margin", model="itemknn", model_role="primary", gate_evalsz=200):
    users = []
    for u in range(nu):
        np_ = 5
        md = {}
        for m in methods:
            md[m] = {
                "aia": 0.12 if m == "shapley_mc" else -0.08,
                "aia_ndcg": 0.05,
                "faithfulness_alignment": 0.1,
                "actionability_gap": 0.13 if m == "shapley_mc" else -0.06,
                "signed_alignment": 0.0,
                "signed_alignment_ndcg": 0.0,
                "direction_accuracy": 0.7,
                "direction_accuracy_ndcg": 0.7,
                "topk_precision": {"1": 0.5, "3": 0.4, "5": 0.3},
                "aia_null": {"null_mean": 0, "null_p95": 0.05, "p_value": 0.001 if m != "random" else 0.6},
                "effect_primary": 0.0,
                "effect_ndcg": 0.0,
                "effect_target_margin": 0.0,
                "success": True,
                "success_ndcg": True,
                "action": {"abstained": False},
                "regret_primary": 0.01,
                "normalized_regret_primary": 0.01,
                "regret_ndcg": 0.01,
                "normalized_regret_ndcg": 0.01,
                "attribution": list(np.random.randn(np_))
            }
        users.append({
            "user": u, "n_players": np_, "evaluation_size": evalsz,
            "recommendation_quality": {"model_ndcg@10": 0.25, "popularity_ndcg@10": 0.15, "model_recall@10": 0.4, "popularity_recall@10": 0.3},
            "effects": {"deletion_primary": list(np.random.randn(np_)), "feasible_primary": list(np.random.randn(np_)), "deletion_ndcg": list(np.random.randn(np_)), "feasible_ndcg": list(np.random.randn(np_))},
            "oracle": {"type": "exact", "primary_utility": 0.1, "ndcg": 0.25},
            "methods": md
        })
    p = {
        "schema_version": 2,
        "dataset": {"name": ds, "evaluation_mode": "sampled_unseen_negatives_with_target", "primary_utility": utility, "users_after_filter": 1200, "items": 5000, "selected_user_ids": list(range(nu)), "target_coverage": 1.0, "users_with_repeated_target": 0},
        "config": {
            "model": model, "analysis_role": role, "condition": cond, "seed": seed,
            "k": 10, "n_max": nmax, "action_rho": rho, "budget": budget,
            "epochs": 5, "embedding_dim": 64, "profile_samples_per_user": 1,
            "profile_learning_rate": 0.01, "profile_regularization": 0.0,
            "itemknn_neighbours": 50, "model_role": model_role,
            "permutations": 500, "dataset_format": "csv",
            "evaluation_size": evalsz,
            "utility": utility
        },
        "status": "paper_eligible",
        "masking_gate": {"passed": True, "evaluation_mode": "fixed_sampled_gate", "evaluation_size": gate_evalsz, "changed_fraction": 0.6, "mean_abs_ndcg_change": 0.01},
        "recommendation_quality": {"model_ndcg@10": 0.22, "popularity_ndcg@10": 0.14, "model_recall@10": 0.4, "popularity_recall@10": 0.29},
        "provenance": {"input_sha256": SHA},
        "users": users
    }
    return p

# Primary + full catalogue (5 seeds) for BOTH models to satisfy "architecture robustness"
for seed in range(42, 47):
    for ds in ["Amazon-Digital-Music", "MovieLens-1M"]:
        for role, cond in [("primary", "primary"), ("full_catalogue", "full_catalogue")]:
            for mname, mrole in [("itemknn", "primary"), ("profile", "robustness")]:
                fname = out / ("r_" + ds.replace("-", "_") + "_" + role + "_" + cond + "_" + mname + "_s" + str(seed) + ".json")
                fname.write_text(json.dumps(mk(ds, role, cond, seed, nu=5, model=mname, model_role=mrole)))

# All required sensitivities (correct controlled values) -- gate ALWAYS 200, only config eval_size varies for candidates
sens = [
    ("budget1", {"budget": 1}),
    ("budget3", {"budget": 3}),
    ("candidates100", {"evaluation_size": 100}),
    ("candidates500", {"evaluation_size": 500}),
    ("nmax50", {"n_max": 50}),
    ("nmax100", {"n_max": 100}),
    ("rho025", {"action_rho": 0.25}),
    ("utility-ndcg", {"utility": "ndcg"}),
]
for scond, overrides in sens:
    for seed in range(42, 47):
        params = dict(nu=5, nmax=20, rho=0.5, budget=2, evalsz=200, utility="target_margin", model="itemknn", model_role="primary", gate_evalsz=200)
        if "budget" in overrides:
            params["budget"] = overrides["budget"]
        if "evaluation_size" in overrides:
            params["evalsz"] = overrides["evaluation_size"]
        if "n_max" in overrides:
            params["nmax"] = overrides["n_max"]
        if "action_rho" in overrides:
            params["rho"] = overrides["action_rho"]
        if "utility" in overrides:
            params["utility"] = overrides["utility"]
        fname = out / ("r_ML_" + scond + "_s" + str(seed) + ".json")
        fname.write_text(json.dumps(mk("MovieLens-1M", "sensitivity", scond, seed, **params)))

# Convergence
for ds in ["Amazon-Digital-Music", "MovieLens-1M"]:
    c = {
        "schema_version": 2,
        "config": {"dataset_name": ds, "model": "itemknn", "utility": "target_margin"},
        "selected_permutations": 500,
        "user_threshold_coverage": {"500": 1.0},
        "rank_valid_fraction": {"500": 0.99},
        "rows": [{"permutations": p, "mean_rank_correlation_to_reference": 0.96, "mean_top1_agreement": 0.8, "mean_top2_jaccard": 0.81, "mean_top2_exact_agreement": 0.7, "mean_efficiency_error": 0.001} for p in [25, 50, 100, 250, 500, 1000]],
        "criterion": {"mean_rank_correlation": 0.95, "mean_top2_jaccard": 0.80, "minimum_rank_valid_fraction": 0.95},
        "provenance": {"input_sha256": SHA}
    }
    (out / ("conv_" + ds.replace("-", "_") + ".json")).write_text(json.dumps(c))

print("data ready, count:", len(list(out.glob("*.json"))))
