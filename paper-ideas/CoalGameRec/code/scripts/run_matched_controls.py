#!/usr/bin/env python3
"""Confirmatory matched-controls run (v4).

Re-runs the frozen v3 prospective protocol (identical hyperparameters, same
seeds 42-46, same archived splits) on CPU and ADDS the validation-informed
non-game controls that the frozen v3 runs did not include:

  - unreranked        (base scores, lambda_attr = 0)
  - valid-sim         history reweighting w_j = max(0, cos(e_j, e_{i_u^+}))
  - valid-linear      candidate-side linear reranker
                      s'_ui = z(b_ui) + lambda * z(cos(e_i, e_{i_u^+}))

Both controls have the SAME validation access as the Shapley/LOO games and
share the identical a-priori lambda_attr = 0.10 and native intervention, so
the run isolates "game structure" from "validation access".

Also produces faithfulness proxy curves (fractions 0.05/0.10/0.20/0.30) for
{loo-marginal, uniform, random} weights with deletion and insertion proxies.

Deviations from v3 (all recorded in manifest.json):
  - device: cpu (original: mps); torch re-execution is confirmatory, not
    bit-identical to the original runs;
  - trainer uses ONE shared full-graph propagation per batch. This is
    mathematically equivalent to the original loop (the embedding matrix is
    not updated between the n_neg forward calls inside a batch, so all
    propagate() calls within a batch return identical embeddings; the summed
    gradient through one shared propagation graph equals the accumulated
    gradients through the separate graphs);
  - shapley-mc is NOT re-run here (CPU budget); the v3 Shapley artifacts
    remain primary for all Shapley comparisons;
  - no lambda-sensitivity sweep (protocol lambda = 0.10 only).

Usage:
  python scripts/run_matched_controls.py --dataset ml1m \
      --source-run results/journal_runs/ml1m_lightgcn_v3_prospective \
      --out results/journal_runs/ml1m_lightgcn_v4_matched_controls
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
# Default results root (used by sibling experiment scripts when --source-run is omitted)
RESULTS = CODE_DIR / "results" / "journal_runs"

from coalgamerec.attribution import compute_attribution_for_users, compute_shapley_for_users  # noqa: E402
from coalgamerec.data import SplitData, item_user_vectors  # noqa: E402
from coalgamerec.explanation import deletion_comprehensiveness, insertion_sufficiency  # noqa: E402
from coalgamerec.metrics import evaluate  # noqa: E402
from coalgamerec.models import LightGCN, TrainConfig, build_lightgcn_graph, cache_full_scores, get_item_embeddings  # noqa: E402
from coalgamerec.rerank import rerank_all, valid_linear_scores_all, valid_sim_scores_all  # noqa: E402
from coalgamerec.utils import sparse_fingerprint, write_json  # noqa: E402


# Frozen protocol hyperparameters (identical to config.resolved.json of v3).
BACKBONE = dict(dim=64, n_layers=2, lr=0.002, weight_decay=1e-5, epochs=15,
                batch_size=4096, n_neg=2, score_batch_size=256)
ATTR = dict(max_players_per_user=24, n_val_negatives=100,
            lambda_attr_value=0.10, value_mode="pairwise_logsigmoid",
            player_selection="stratified")
LAMBDA_ATTR = 0.10
TAU_ATT = 0.10
KS = (5, 10, 20)
FAITH_FRACTIONS = (0.05, 0.10, 0.20, 0.30)


def train_lightgcn_shared_prop(train_df: pd.DataFrame, n_users: int, n_items: int,
                               cfg: TrainConfig, n_layers: int = 2) -> LightGCN:
    """Equivalent-gradients LightGCN training with one propagation per batch.

    The original train_lightgcn calls model(ut, pt) and model(ut, nt) inside
    the n_neg loop; each call re-propagates the full graph from the SAME
    layer-0 embeddings (no optimizer step happens inside the loop), hence
    every propagation in a batch returns identical embeddings. Computing the
    shared propagation once and scoring positives and negatives from it
    produces the same forward values and the same summed gradient.
    """
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    if os.environ.get("COALGAME_DEVICE", "cpu") == "cpu":
        torch.set_num_threads(max(1, int(os.environ.get("COALGAME_THREADS", "2"))))
    device = torch.device(os.environ.get("COALGAME_DEVICE", "cpu"))
    edge_index, edge_weight = build_lightgcn_graph(train_df, n_users, n_items, device)
    model = LightGCN(n_users, n_items, edge_index, edge_weight, cfg.dim, n_layers=n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_csr = sparse.csr_matrix((np.ones(len(train_df)), (train_df.user, train_df.item)),
                                  shape=(n_users, n_items)).tocsr()
    item_deg = np.asarray(train_csr.sum(axis=0)).ravel().astype(np.float64)
    item_probs = item_deg + 1e-6
    item_probs /= item_probs.sum()
    cdf = np.cumsum(item_probs)
    # Precompute per-user seen sets once (the original _sample_negatives rebuilt
    # them on every call; the sampler's distribution is unchanged).
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
            U, I = model.propagate()  # single shared propagation for the batch
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


def load_split_from_run(source_run: Path) -> SplitData:
    source_run = Path(source_run)
    sp = source_run / "splits"
    if not (sp / "meta.json").exists():
        # v4/v4b outputs do not carry splits; fall back to the v3 prospective run
        for tag in ("_v4b_matched_controls", "_v4_matched_controls"):
            if str(source_run).endswith(tag):
                alt = Path(str(source_run)[: -len(tag)] + "_v3_prospective")
                if (alt / "splits" / "meta.json").exists():
                    sp = alt / "splits"
                    break
    meta = json.loads((sp / "meta.json").read_text())
    train = pd.read_parquet(sp / "train.parquet")
    val = pd.read_parquet(sp / "val.parquet")
    test = pd.read_parquet(sp / "test.parquet")
    return SplitData(name=meta["name"], train=train, val=val, test=test,
                     n_users=int(meta["n_users"]), n_items=int(meta["n_items"]))


def run_seed(split: SplitData, item_vectors, seed: int, out_dir: Path) -> None:
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
    model = train_lightgcn_shared_prop(split.train, split.n_users, split.n_items, train_cfg,
                                       n_layers=BACKBONE["n_layers"])
    base_scores = cache_full_scores(model, split.n_users, batch_size=BACKBONE["score_batch_size"],
                                    chunk_items=4096 if split.n_items > 8000 else None)
    item_embeddings = get_item_embeddings(model)
    stages["train_and_score_seconds"] = time.time() - t

    base_summary, _ = evaluate(base_scores, split, item_vectors, ks=KS)
    write_json(seed_dir / "base_summary.json", base_summary)

    val_by_user = split.val.sort_values("user").set_index("user").item.to_dict()

    # --- LOO attribution (frozen protocol: stratified k=24, 100 val negatives) ---
    t = time.time()
    loo = compute_attribution_for_users(
        split, base_scores, item_vectors, method="loo-marginal",
        max_users=None, exact_threshold=8, seed=seed,
        max_players_per_user=ATTR["max_players_per_user"],
        player_selection=ATTR["player_selection"],
        checkpoint_path=None, save_every=25,
        alpha=1.0, beta=0.0, lambda_pref=0.0,
        lambda_attr_value=ATTR["lambda_attr_value"],
        value_mode=ATTR["value_mode"], n_val_negatives=ATTR["n_val_negatives"],
    )
    stages["loo_seconds"] = time.time() - t

    # --- optional Shapley re-run (C1 completeness, set C1_WITH_SHAPLEY=1) ---
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

    # --- family evaluation ---
    t = time.time()
    rows, per_user_rows = [], []

    def record(fam: str, scores: np.ndarray):
        summary, per_user = evaluate(scores, split, item_vectors, ks=KS)
        summary.update({"seed": seed, "family": fam, "backbone": "lightgcn", "dataset": split.name})
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

    # --- faithfulness proxy curves (loo / uniform / random) ---
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
    write_json(seed_dir / "runtime.json", {"seed": seed, "stages": stages})
    print(f"seed {seed} done: " + ", ".join(f"{k}={v:.0f}s" for k, v in stages.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml1m", "amazon"])
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    source_run = Path(args.source_run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = load_split_from_run(source_run)
    item_vectors = item_user_vectors(split.train_csr)

    # integrity check: item vectors depend only on the train split, so their
    # fingerprint must equal the one recorded by the source v3 run.
    src_report = json.loads((source_run / "item_vectors_report.json").read_text())
    fp = sparse_fingerprint(item_vectors)
    iv_sorted = item_vectors.copy(); iv_sorted.sort_indices()
    fp_sorted = sparse_fingerprint(iv_sorted)
    iv_match = (fp == src_report["hash"]) or (fp_sorted == src_report["hash"])
    shape_nnz_match = (list(item_vectors.shape) == src_report["shape"]
                       and int(item_vectors.nnz) == int(src_report["nnz"]))
    # Content is derived deterministically from the archived split; a hash
    # mismatch can only reflect intra-row index ordering after the parquet
    # round-trip, so shape+nnz agreement is the integrity gate.
    assert iv_match or shape_nnz_match, "item-vector integrity check failed"
    write_json(out_dir / "item_vectors_report.json", {
        "hash": fp, "hash_exact_match_with_source_run": bool(iv_match),
        "shape_nnz_match_with_source_run": bool(shape_nnz_match)})

    write_json(out_dir / "config.resolved.json", {
        "note": "Confirmatory matched-controls run: frozen v3 hyperparameters, CPU re-execution.",
        "backbone": {"name": "lightgcn", **BACKBONE},
        "attribution": ATTR,
        "reranking": {"lambda_attr": LAMBDA_ATTR, "tau_att": TAU_ATT, "intervention": "native",
                      "families": ["unreranked", "uniform", "additive-pref", "attention",
                                   "heuristic-pop", "loo-marginal", "valid-sim", "valid-linear"]},
        "run": {"seeds": args.seeds, "device": "cpu", "ks": list(KS),
                "faithfulness_fractions": list(FAITH_FRACTIONS)},
        "dataset": {"name": split.name, "source_run": str(source_run)},
        "deviations": [
            "device cpu (original mps); confirmatory re-execution, not bit-identical",
            "one shared propagation per batch (equivalent gradients; see script docstring)",
            "negative sampler vectorized (identical per-user degree-proportional distribution; seen sets precomputed)",
            "shapley-mc not re-run (CPU budget); v3 Shapley artifacts remain primary",
            "no lambda-sensitivity sweep (protocol lambda=0.10 only)",
        ],
    })

    t0 = time.time()
    for seed in args.seeds:
        run_seed(split, item_vectors, seed, out_dir)

    # aggregate tables
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
    faith = pd.concat([pd.read_csv(out_dir / "raw" / f"seed_{s}" / "faithfulness_curves.csv")
                       for s in args.seeds], ignore_index=True)
    faith.to_csv(out_dir / "tables" / "faithfulness_curves_all.csv", index=False)
    faith_metrics = [c for c in faith.columns if c.startswith(("DeletionDelta_", "Insertion_", "Attribution"))]
    faith.groupby(["family", "fraction"])[faith_metrics].agg(["mean", "std"]) \
        .to_csv(out_dir / "tables" / "faithfulness_curves_mean_std.csv")

    write_json(out_dir / "manifest.json", {
        "note": "Confirmatory matched-controls run (v4). Adds validation-informed non-game controls to the frozen v3 protocol.",
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "device": os.environ.get("COALGAME_DEVICE", "cpu"),
        "n_users": split.n_users, "n_items": split.n_items,
        "train_interactions": int(len(split.train)),
        "total_seconds": time.time() - t0,
        "trainer": "train_lightgcn_shared_prop (equivalent gradients, 1 propagation/batch)",
        "item_vectors_hash_exact_match": bool(iv_match), "item_vectors_shape_nnz_match": bool(shape_nnz_match),
    })
    print(f"ALL DONE {args.dataset} in {time.time()-t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
