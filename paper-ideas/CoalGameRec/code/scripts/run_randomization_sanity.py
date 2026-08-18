#!/usr/bin/env python3
"""Attribution stability + model-randomization sanity checks (round 6).

Three diagnostics requested by the reviewers:

A. Cross-seed attribution stability. LOO attributions are computed on two
   independently trained models (seeds 42 and 43, same frozen protocol). We
   report per-user Spearman rank correlation and top-12 overlap between the
   two seeds' attributions. High stability supports interpreting attributions
   as properties of the data/model family rather than of one random fit.

B. Model randomization test. LOO attributions are computed from an UNTRAINED
   LightGCN (protocol initialization, zero gradient steps). If attributions
   from the randomized model closely matched the trained-model attributions,
   the "explanations" would not depend on what the model learned and the
   faithfulness framing would be undermined. We report the trained-vs-random
   Spearman correlation and the reranked TEST NDCG@20 of random-model LOO
   weights versus trained LOO weights.

C. Perturbation stability (input noise). LOO attributions are recomputed
   after adding N(0, sigma) noise to the trained layer-0 embeddings
   (sigma = 0.1 * mean|E0|, frozen seed), and compared with the clean
   attributions (Spearman + top-12 overlap).

Usage:
  COALGAME_DEVICE=mps python scripts/run_randomization_sanity.py --dataset ml1m \
      --source-run results/journal_runs/ml1m_lightgcn_v3_prospective \
      --out results/journal_runs/ml1m_lightgcn_v6_randomization_sanity \
      [--max-users N]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_matched_controls as rmc  # noqa: E402
import run_second_backbone as rsb   # noqa: E402
from coalgamerec.attribution import compute_attribution_for_users  # noqa: E402
from coalgamerec.data import item_user_vectors  # noqa: E402
from coalgamerec.metrics import evaluate  # noqa: E402
from coalgamerec.models import LightGCN, TrainConfig, build_lightgcn_graph, cache_full_scores, get_item_embeddings  # noqa: E402
from coalgamerec.rerank import rerank_all  # noqa: E402
from coalgamerec.utils import write_json  # noqa: E402

BACKBONE = rmc.BACKBONE
ATTR = rmc.ATTR
TAU_ATT = rmc.TAU_ATT
LAMBDA_ATTR = rmc.LAMBDA_ATTR
KS = rmc.KS

try:
    from run_negset_sensitivity import stability_report
except ImportError:  # pragma: no cover
    stability_report = None


def loo_attr(split, base_scores, item_vectors, seed, max_users):
    return compute_attribution_for_users(
        split, base_scores, item_vectors, method="loo-marginal",
        max_users=max_users, exact_threshold=8, seed=seed,
        max_players_per_user=ATTR["max_players_per_user"],
        player_selection=ATTR["player_selection"],
        checkpoint_path=None, save_every=25,
        alpha=1.0, beta=0.0, lambda_pref=0.0,
        lambda_attr_value=ATTR["lambda_attr_value"],
        value_mode=ATTR["value_mode"], n_val_negatives=ATTR["n_val_negatives"])


def train_one(split, seed):
    train_cfg = TrainConfig(dim=BACKBONE["dim"], lr=BACKBONE["lr"], weight_decay=BACKBONE["weight_decay"],
                            epochs=BACKBONE["epochs"], batch_size=BACKBONE["batch_size"],
                            n_neg=BACKBONE["n_neg"], seed=seed, device=os.environ.get("COALGAME_DEVICE", "cpu"))
    device = torch.device(os.environ.get("COALGAME_DEVICE", "cpu"))
    edge_index, edge_weight = build_lightgcn_graph(split.train, split.n_users, split.n_items, device)
    model = LightGCN(split.n_users, split.n_items, edge_index, edge_weight,
                     dim=BACKBONE["dim"], n_layers=BACKBONE["n_layers"]).to(device)
    model = rsb.train_shared_prop(model, split.train, split.n_users, split.n_items, train_cfg)
    base_scores = cache_full_scores(model, split.n_users, batch_size=BACKBONE["score_batch_size"],
                                    chunk_items=4096 if split.n_items > 8000 else None)
    return model, base_scores, get_item_embeddings(model)


def rerank_loo_eval(split, base_scores, item_vectors, loo, item_embeddings):
    scores = rerank_all(base_scores, split, item_vectors, "loo-marginal",
                        shapley_by_user=None, loo_by_user=loo,
                        lambda_attr=LAMBDA_ATTR, tau_att=TAU_ATT,
                        intervention="native", item_embeddings=item_embeddings)
    summary, _ = evaluate(scores, split, item_vectors, ks=KS)
    return summary["NDCG@20"], summary["HitRate@20"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml1m", "amazon"])
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--noise-sigma-frac", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = rmc.load_split_from_run(Path(args.source_run))
    item_vectors = item_user_vectors(split.train_csr)
    report = {"dataset": split.name}

    # --- seed 42 trained model (primary) ---
    model42, base42, emb42 = train_one(split, 42)
    loo42 = loo_attr(split, base42, item_vectors, 42, args.max_users)
    ndcg_trained, hr_trained = rerank_loo_eval(split, base42, item_vectors, loo42, emb42)
    report["trained_seed42"] = {"NDCG@20": ndcg_trained, "HitRate@20": hr_trained}

    # --- A: cross-seed stability (seed 43) ---
    model43, base43, emb43 = train_one(split, 43)
    loo43 = loo_attr(split, base43, item_vectors, 43, args.max_users)
    report["cross_seed_stability_42_vs_43"] = stability_report(loo42, loo43)

    # --- B: model randomization (untrained model, seed-42 initialization) ---
    device = torch.device(os.environ.get("COALGAME_DEVICE", "cpu"))
    edge_index, edge_weight = build_lightgcn_graph(split.train, split.n_users, split.n_items, device)
    torch.manual_seed(42)
    model_rand = LightGCN(split.n_users, split.n_items, edge_index, edge_weight,
                          dim=BACKBONE["dim"], n_layers=BACKBONE["n_layers"]).to(device)
    base_rand = cache_full_scores(model_rand, split.n_users, batch_size=BACKBONE["score_batch_size"],
                                  chunk_items=4096 if split.n_items > 8000 else None)
    emb_rand = get_item_embeddings(model_rand)
    loo_rand = loo_attr(split, base_rand, item_vectors, 42, args.max_users)
    ndcg_rand, hr_rand = rerank_loo_eval(split, base_rand, item_vectors, loo_rand, emb_rand)
    report["model_randomization"] = {
        "untrained_rerank_NDCG@20": ndcg_rand, "untrained_rerank_HitRate@20": hr_rand,
        "trained_vs_untrained_attribution": stability_report(loo42, loo_rand)}

    # --- C: perturbation stability (layer-0 embedding noise on trained model) ---
    rng = np.random.default_rng(20260818)
    with torch.no_grad():
        scale_u = float(model42.user_emb.weight.abs().mean())
        scale_i = float(model42.item_emb.weight.abs().mean())
        model42.user_emb.weight.add_(torch.as_tensor(
            rng.normal(0, args.noise_sigma_frac * scale_u, model42.user_emb.weight.shape),
            dtype=torch.float32, device=device))
        model42.item_emb.weight.add_(torch.as_tensor(
            rng.normal(0, args.noise_sigma_frac * scale_i, model42.item_emb.weight.shape),
            dtype=torch.float32, device=device))
    base_noisy = cache_full_scores(model42, split.n_users, batch_size=BACKBONE["score_batch_size"],
                                   chunk_items=4096 if split.n_items > 8000 else None)
    loo_noisy = loo_attr(split, base_noisy, item_vectors, 42, args.max_users)
    report["perturbation_stability"] = {
        "sigma_fraction": args.noise_sigma_frac,
        "clean_vs_noisy_attribution": stability_report(loo42, loo_noisy)}

    write_json(out_dir / "randomization_sanity.json", report)
    write_json(out_dir / "manifest.json", {
        "note": "Attribution stability / model-randomization / perturbation sanity checks.",
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "device": os.environ.get("COALGAME_DEVICE", "cpu")})
    print(json.dumps(report, indent=1))
    print(f"DONE -> {out_dir / 'randomization_sanity.json'}", flush=True)


if __name__ == "__main__":
    main()
