#!/usr/bin/env python3
"""Second-backbone confirmatory run (round-6 mandatory experiment).

Runs the IDENTICAL matched-controls protocol of run_matched_controls.py
(frozen hyperparameters, archived temporal splits, seeds 42-46, full-catalog
ranking, same families) but with a structurally different graph backbone:

  --backbone ngcf      NGCF-style nonlinear aggregation
                       m = W1 e_j + W2 (e_j * e_i), LeakyReLU after each
                       normalized aggregation (Wang et al., SIGIR 2019);
                       layer readout = mean over layers (documented deviation
                       from concatenation, keeps the attribution interface).
  --backbone lightgcn  re-execution control (should reproduce v4b within noise)

Everything downstream of training (item-vector integrity gate, stratified
k=24 player selection, 100 validation negatives, pairwise log-sigmoid value,
LOO + antithetic MC Shapley M=64, native reranking at lambda=0.10, matched
valid-sim / valid-linear controls, full-catalog evaluation) is imported from
the frozen matched-controls implementation, so the only changed factor is the
propagation scheme.

Usage (Apple Silicon example):
  COALGAME_DEVICE=mps C1_WITH_SHAPLEY=1 python scripts/run_second_backbone.py \
      --dataset ml1m --backbone ngcf \
      --source-run results/journal_runs/ml1m_lightgcn_v3_prospective \
      --out results/journal_runs/ml1m_ngcf_v6_second_backbone

Output schema mirrors *_v4b_matched_controls (per_user_metrics_all.csv.gz,
summary tables, runtime.json per seed) so analyze_round6_stats.py-style
inference applies unchanged.
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
import pandas as pd
import torch
from scipy import sparse

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_matched_controls as rmc  # noqa: E402  (frozen protocol constants + IO helpers)
from coalgamerec.attribution import compute_attribution_for_users, compute_shapley_for_users  # noqa: E402
from coalgamerec.data import item_user_vectors  # noqa: E402
from coalgamerec.explanation import deletion_comprehensiveness, insertion_sufficiency  # noqa: E402
from coalgamerec.metrics import evaluate  # noqa: E402
from coalgamerec.models import NGCF, LightGCN, TrainConfig, build_lightgcn_graph, cache_full_scores, get_item_embeddings  # noqa: E402
from coalgamerec.rerank import rerank_all, valid_linear_scores_all, valid_sim_scores_all  # noqa: E402
from coalgamerec.utils import sparse_fingerprint, write_json  # noqa: E402

BACKBONE = rmc.BACKBONE
ATTR = rmc.ATTR
LAMBDA_ATTR = rmc.LAMBDA_ATTR
TAU_ATT = rmc.TAU_ATT
KS = rmc.KS
FAITH_FRACTIONS = rmc.FAITH_FRACTIONS


def train_shared_prop(model: torch.nn.Module, train_df: pd.DataFrame, n_users: int,
                      n_items: int, cfg: TrainConfig) -> torch.nn.Module:
    """BPR training loop with one shared propagation per batch.

    Same equivalent-gradients argument as train_lightgcn_shared_prop in the
    frozen matched-controls script: within a batch the layer-0 embeddings are
    constant across the n_neg forward calls, so one shared propagation yields
    identical forward values and the same summed gradient. Works for any
    backbone exposing propagate().
    """
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    if os.environ.get("COALGAME_DEVICE", "cpu") == "cpu":
        torch.set_num_threads(max(1, int(os.environ.get("COALGAME_THREADS", "2"))))
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_csr = sparse.csr_matrix((np.ones(len(train_df)), (train_df.user, train_df.item)),
                                  shape=(n_users, n_items)).tocsr()
    item_deg = np.asarray(train_csr.sum(axis=0)).ravel().astype(np.float64)
    item_probs = item_deg + 1e-6
    item_probs /= item_probs.sum()
    cdf = np.cumsum(item_probs)
    seen_sets = [set(train_csr[u].indices) for u in range(n_users)]

    def sample_neg_fast(u: np.ndarray) -> np.ndarray:
        neg = np.searchsorted(cdf, rng.random(len(u)), side="right").clip(0, n_items - 1)
        for i in range(len(u)):
            s = seen_sets[int(u[i])]
            while neg[i] in s:
                neg[i] = min(int(np.searchsorted(cdf, rng.random(), side="right")), n_items - 1)
        return neg

    users_all = train_df.user.values.astype(np.int64)
    pos_all = train_df.item.values.astype(np.int64)
    n = len(train_df)
    for epoch in range(cfg.epochs):
        t0 = time.time()
        order = rng.permutation(n)
        for start in range(0, n, cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            u = users_all[idx]
            pos = pos_all[idx]
            opt.zero_grad(set_to_none=True)
            U, I = model.propagate()
            u_scores = U[torch.as_tensor(u, dtype=torch.long, device=device)]
            pos_scores = (u_scores * I[torch.as_tensor(pos, dtype=torch.long, device=device)]).sum(-1)
            loss = torch.zeros((), device=device)
            for _k in range(cfg.n_neg):
                neg = sample_neg_fast(u)
                neg_scores = (u_scores * I[torch.as_tensor(neg, dtype=torch.long, device=device)]).sum(-1)
                loss = loss + (-torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean() / cfg.n_neg)
            loss.backward()
            opt.step()
        print(f"  epoch {epoch+1}/{cfg.epochs} done in {time.time()-t0:.1f}s loss={float(loss):.4f}", flush=True)
    return model


def build_backbone(name: str, split, device) -> torch.nn.Module:
    edge_index, edge_weight = build_lightgcn_graph(split.train, split.n_users, split.n_items, device)
    if name == "ngcf":
        return NGCF(split.n_users, split.n_items, edge_index, edge_weight,
                    dim=BACKBONE["dim"], n_layers=BACKBONE["n_layers"]).to(device)
    return LightGCN(split.n_users, split.n_items, edge_index, edge_weight,
                    dim=BACKBONE["dim"], n_layers=BACKBONE["n_layers"]).to(device)


def run_seed(split, item_vectors, seed: int, out_dir: Path, backbone: str) -> None:
    seed_dir = out_dir / "raw" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if (seed_dir / "summary_by_family.csv").exists() and (seed_dir / "faithfulness_curves.csv").exists():
        print(f"seed {seed} already complete, skipping", flush=True)
        return
    stages = {}

    t = time.time()
    train_cfg = TrainConfig(dim=BACKBONE["dim"], lr=BACKBONE["lr"], weight_decay=BACKBONE["weight_decay"],
                            epochs=BACKBONE["epochs"], batch_size=BACKBONE["batch_size"],
                            n_neg=BACKBONE["n_neg"], seed=seed, device=os.environ.get("COALGAME_DEVICE", "cpu"))
    device = torch.device(os.environ.get("COALGAME_DEVICE", "cpu"))
    model = build_backbone(backbone, split, device)
    model = train_shared_prop(model, split.train, split.n_users, split.n_items, train_cfg)
    base_scores = cache_full_scores(model, split.n_users, batch_size=BACKBONE["score_batch_size"],
                                    chunk_items=4096 if split.n_items > 8000 else None)
    item_embeddings = get_item_embeddings(model)
    stages["train_and_score_seconds"] = time.time() - t

    base_summary, _ = evaluate(base_scores, split, item_vectors, ks=KS)
    write_json(seed_dir / "base_summary.json", base_summary)

    val_by_user = split.val.sort_values("user").set_index("user").item.to_dict()

    t = time.time()
    loo = compute_attribution_for_users(
        split, base_scores, item_vectors, method="loo-marginal",
        max_users=None, exact_threshold=8, seed=seed,
        max_players_per_user=ATTR["max_players_per_user"],
        player_selection=ATTR["player_selection"],
        checkpoint_path=None, save_every=25,
        alpha=1.0, beta=0.0, lambda_pref=0.0,
        lambda_attr_value=ATTR["lambda_attr_value"],
        value_mode=ATTR["value_mode"], n_val_negatives=ATTR["n_val_negatives"])
    stages["loo_seconds"] = time.time() - t

    shap = None
    if os.environ.get("C1_WITH_SHAPLEY") == "1":
        t = time.time()
        shap = compute_shapley_for_users(
            split, base_scores, item_vectors, max_users=None, m=64, exact_threshold=8, seed=seed,
            max_players_per_user=ATTR["max_players_per_user"], player_selection=ATTR["player_selection"],
            checkpoint_path=None, save_every=25, alpha=1.0, beta=0.0, lambda_pref=0.0,
            lambda_attr_value=ATTR["lambda_attr_value"], value_mode=ATTR["value_mode"],
            n_val_negatives=ATTR["n_val_negatives"], antithetic=True)
        stages["shapley_seconds"] = time.time() - t

    t = time.time()
    rows, per_user_rows = [], []

    def record(fam: str, scores: np.ndarray):
        summary, per_user = evaluate(scores, split, item_vectors, ks=KS)
        summary.update({"seed": seed, "family": fam, "backbone": backbone, "dataset": split.name})
        rows.append(summary)
        for metric, values in per_user.items():
            per_user_rows.append(pd.DataFrame({"seed": seed, "family": fam, "metric": metric,
                                               "user": np.arange(len(values)), "value": values}))

    record("unreranked", base_scores)
    for fam in ["uniform", "additive-pref", "attention", "heuristic-pop", "loo-marginal"]:
        scores = rerank_all(base_scores, split, item_vectors, fam,
                            shapley_by_user=None, loo_by_user=loo,
                            lambda_attr=LAMBDA_ATTR, tau_att=TAU_ATT,
                            intervention="native", item_embeddings=item_embeddings)
        record(fam, scores)
    record("valid-sim", valid_sim_scores_all(base_scores, split, val_by_user, item_embeddings, LAMBDA_ATTR))
    record("valid-linear", valid_linear_scores_all(base_scores, split, val_by_user, item_embeddings, LAMBDA_ATTR))
    if shap is not None:
        scores = rerank_all(base_scores, split, item_vectors, "shapley-mc",
                            shapley_by_user=shap, loo_by_user=None,
                            lambda_attr=LAMBDA_ATTR, tau_att=TAU_ATT,
                            intervention="native", item_embeddings=item_embeddings)
        record("shapley-mc", scores)
    stages["rerank_eval_seconds"] = time.time() - t

    t = time.time()
    train_csr = split.train_csr
    uniform_attr = {u: np.ones(len(train_csr[int(u)].indices), dtype=np.float32) for u in range(split.n_users)}
    rng = np.random.default_rng(seed * 100000 + 7)
    random_attr = {u: rng.standard_normal(len(train_csr[int(u)].indices)).astype(np.float32)
                   for u in range(split.n_users)}
    faith_rows = []
    for fam_name, attr in [("loo-marginal", loo), ("uniform", uniform_attr), ("random", random_attr)]:
        for frac in FAITH_FRACTIONS:
            d = deletion_comprehensiveness(base_scores, split, attr, fraction=frac, ks=(20,))
            i = insertion_sufficiency(base_scores, split, attr, fraction=frac, ks=(20,))
            faith_rows.append({"seed": seed, "family": fam_name, "fraction": frac, **d, **i})
    pd.DataFrame(faith_rows).to_csv(seed_dir / "faithfulness_curves.csv", index=False)
    stages["faithfulness_seconds"] = time.time() - t

    pd.DataFrame(rows).to_csv(seed_dir / "summary_by_family.csv", index=False)
    pd.concat(per_user_rows, ignore_index=True).to_csv(seed_dir / "per_user_metrics.csv.gz", index=False, compression="gzip")
    write_json(seed_dir / "runtime.json", {"seed": seed, "backbone": backbone, "stages": stages})
    print(f"seed {seed} done: " + ", ".join(f"{k}={v:.0f}s" for k, v in stages.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml1m", "amazon"])
    ap.add_argument("--backbone", required=True, choices=["ngcf", "lightgcn"])
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    source_run = Path(args.source_run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = rmc.load_split_from_run(source_run)
    item_vectors = item_user_vectors(split.train_csr)

    src_report = json.loads((rmc.resolve_source_dir(source_run) / "item_vectors_report.json").read_text())
    fp = sparse_fingerprint(item_vectors)
    iv_sorted = item_vectors.copy(); iv_sorted.sort_indices()
    fp_sorted = sparse_fingerprint(iv_sorted)
    iv_match = (fp == src_report.get("hash")) or (fp_sorted == src_report.get("hash"))
    shape_nnz_match = ("shape" in src_report and "nnz" in src_report
                       and list(item_vectors.shape) == src_report["shape"]
                       and int(item_vectors.nnz) == int(src_report["nnz"]))
    assert iv_match or shape_nnz_match, "item-vector integrity check failed"
    write_json(out_dir / "item_vectors_report.json", {
        "hash": fp, "hash_exact_match_with_source_run": bool(iv_match),
        "shape_nnz_match_with_source_run": bool(shape_nnz_match)})

    write_json(out_dir / "config.resolved.json", {
        "note": "Round-6 second-backbone run: frozen matched-controls protocol, backbone is the only changed factor.",
        "backbone": {"name": args.backbone, **BACKBONE,
                     "layer_readout": "mean over layers (documented deviation for NGCF)"},
        "attribution": ATTR,
        "reranking": {"lambda_attr": LAMBDA_ATTR, "tau_att": TAU_ATT, "intervention": "native"},
        "run": {"seeds": args.seeds, "device": os.environ.get("COALGAME_DEVICE", "cpu"), "ks": list(KS)},
        "dataset": {"name": split.name, "source_run": str(source_run)},
    })

    t0 = time.time()
    for seed in args.seeds:
        run_seed(split, item_vectors, seed, out_dir, args.backbone)

    frames = [pd.read_csv(out_dir / "raw" / f"seed_{s}" / "summary_by_family.csv") for s in args.seeds]
    summary = pd.concat(frames, ignore_index=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    summary.to_csv(out_dir / "tables" / "summary_by_seed_family.csv", index=False)
    metric_cols = [c for c in summary.columns if c.startswith(("HitRate@", "NDCG@", "Coverage@", "ILD@"))]
    summary.groupby(["dataset", "backbone", "family"])[metric_cols].agg(["mean", "std"]) \
        .to_csv(out_dir / "tables" / "summary_mean_std.csv")
    per_user = pd.concat([pd.read_csv(out_dir / "raw" / f"seed_{s}" / "per_user_metrics.csv.gz")
                          for s in args.seeds], ignore_index=True)
    per_user.to_csv(out_dir / "raw" / "per_user_metrics_all.csv.gz", index=False, compression="gzip")

    write_json(out_dir / "manifest.json", {
        "note": f"Second-backbone confirmatory run ({args.backbone}) under the frozen matched-controls protocol.",
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "device": os.environ.get("COALGAME_DEVICE", "cpu"),
        "n_users": split.n_users, "n_items": split.n_items,
        "train_interactions": int(len(split.train)),
        "total_seconds": time.time() - t0,
        "item_vectors_hash_exact_match": bool(iv_match), "item_vectors_shape_nnz_match": bool(shape_nnz_match),
    })
    print(f"ALL DONE {args.dataset} ({args.backbone}) in {time.time()-t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
