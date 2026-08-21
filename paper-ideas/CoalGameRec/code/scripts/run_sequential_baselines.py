#!/usr/bin/env python3
"""Round-9 REQUIRED FIX #10: stronger same-information sequential baselines.

Adds the cheaper competitors that consume the same calibration event i_u^+ as
the game methods, under the corrected (v7) candidate exclusions:

  last-knn        rank by similarity of candidates to the calibration item
                  (item-item kNN with k=1; the pure sequential signal)
  updated-profile add the calibration item to the user's train-only profile
                  vector and rerank by profile-candidate similarity
  edge-update     add the (u, i_u^+) edge to the frozen graph, re-propagate
                  the frozen backbone, and rank by the updated scores
                  (no attribution; the true graph-update sequential baseline)
  recency         history weights = recency ranks (most recent = highest),
                  through the shared native intervention

All families use lambda_attr = 0.10 and the v7 candidate set
(I \\ (H_train U {i_u^+})). Edge-update re-propagates once per user batch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_matched_controls as rmc  # noqa: E402
from coalgamerec.metrics import evaluate  # noqa: E402
from coalgamerec.models import LightGCN, TrainConfig, build_lightgcn_graph, cache_full_scores, get_item_embeddings  # noqa: E402
from coalgamerec.rerank import rerank_user_scores, zscore_candidates  # noqa: E402
from coalgamerec.utils import write_json  # noqa: E402

LAMBDA = 0.10
KS = (5, 10, 20)


def _candidate_idx(n_items: int, train_items: np.ndarray, excl: int) -> np.ndarray:
    m = np.ones(n_items, dtype=bool)
    m[train_items] = False
    m[int(excl)] = False
    return np.flatnonzero(m)


def last_knn_scores(base_scores, split, val_by_user, item_embeddings):
    out = np.empty_like(base_scores, dtype=np.float32)
    train_csr = split.train_csr
    norms = np.linalg.norm(item_embeddings, axis=1)
    Inorm = item_embeddings / (norms[:, None] + 1e-12)
    for u in range(split.n_users):
        vu = val_by_user.get(u)
        items = train_csr[u].indices
        if vu is None:
            out[u] = zscore_candidates(base_scores[u], _candidate_idx(split.n_items, items, -1))
            continue
        ev = item_embeddings[int(vu)]
        nv = float(np.linalg.norm(ev))
        adj = Inorm.dot(ev / (nv + 1e-12)).astype(np.float32)
        cand = _candidate_idx(split.n_items, items, int(vu))
        out[u] = zscore_candidates(base_scores[u], cand) + LAMBDA * zscore_candidates(adj, cand)
    return out


def updated_profile_scores(base_scores, split, val_by_user, item_vectors):
    out = np.empty_like(base_scores, dtype=np.float32)
    train_csr = split.train_csr
    Iv = item_vectors.tocsr()
    inorms = np.sqrt(np.asarray(Iv.multiply(Iv).sum(axis=1)).ravel()) + 1e-12
    for u in range(split.n_users):
        vu = val_by_user.get(u)
        items = train_csr[u].indices
        if vu is None or len(items) == 0:
            out[u] = zscore_candidates(base_scores[u], _candidate_idx(split.n_items, items, -1))
            continue
        prof = np.asarray(Iv[items].mean(axis=0)).ravel() + np.asarray(Iv[int(vu)]).ravel()
        prof = prof / (np.linalg.norm(prof) + 1e-12)
        adj = (Iv.dot(prof) / inorms).astype(np.float32)
        cand = _candidate_idx(split.n_items, items, int(vu))
        out[u] = zscore_candidates(base_scores[u], cand) + LAMBDA * zscore_candidates(adj, cand)
    return out


def recency_scores(base_scores, split, val_by_user, item_embeddings):
    out = np.empty_like(base_scores, dtype=np.float32)
    train_csr = split.train_csr
    for u in range(split.n_users):
        vu = val_by_user.get(u)
        items = train_csr[u].indices
        if len(items) == 0:
            out[u] = zscore_candidates(base_scores[u], _candidate_idx(split.n_items, items, -1))
            continue
        n = len(items)
        w = (np.arange(n, dtype=np.float32) + 1.0) / n  # later index = more recent = higher weight
        excl = None if vu is None else int(vu)
        out[u] = rerank_user_scores(base_scores[u], items, w, None, lambda_attr=LAMBDA,
                                    intervention="native", item_embeddings=item_embeddings,
                                    exclude_item=excl)
    return out


def edge_update_scores(model, split, val_by_user, device):
    """Add (u, i_u^+) to the frozen graph, re-propagate, rank by updated scores."""
    train_csr = split.train_csr
    users = np.arange(split.n_users)
    out = np.full((split.n_users, split.n_items), -np.inf, dtype=np.float32)
    n_users, n_items = split.n_users, split.n_items
    batch = 512
    with torch.no_grad():
        for start in range(0, n_users, batch):
            uu = users[start:start + batch]
            extra_src, extra_dst = [], []
            for u in uu:
                vu = val_by_user.get(int(u))
                if vu is not None:
                    extra_src += [int(u), n_items + int(vu)]
                    extra_dst += [n_items + int(vu), int(u)]
            rows = []
            for u in uu:
                items = train_csr[int(u)].indices
                vu = val_by_user.get(int(u))
                src = np.concatenate([np.full(len(items), int(u)), [int(u)] if vu is not None else []])
                dst = np.concatenate([items + n_users, [n_items + int(vu)] if vu is not None else []])
                all_src = np.concatenate([src, dst])
                all_dst = np.concatenate([dst, src])
                deg = np.bincount(all_src, minlength=n_users + n_items).astype(np.float32)
                deg[deg == 0] = 1.0
                wgt = 1.0 / np.sqrt(deg[all_src] * deg[all_dst])
                ei = torch.as_tensor(np.vstack([all_src, all_dst]), dtype=torch.long, device=device)
                ew = torch.as_tensor(wgt, dtype=torch.float32, device=device)
                m = LightGCN(n_users, n_items, ei, ew, dim=model.user_emb.weight.shape[1],
                             n_layers=model.n_layers).to(device)
                m.user_emb.weight.copy_(model.user_emb.weight)
                m.item_emb.weight.copy_(model.item_emb.weight)
                U, I = m.propagate()
                row = U[int(u)] @ I.T
                vu = val_by_user.get(int(u))
                row[items] = -np.inf
                if vu is not None:
                    row[int(vu)] = -np.inf
                rows.append(row.cpu().numpy())
            out[start:start + len(uu)] = np.stack(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml1m", "amazon"])
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--skip-edge-update", action="store_true", help="skip the expensive per-user re-propagation baseline")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = rmc.load_split_from_run(Path(args.source_run))
    item_vectors = rmc.item_user_vectors(split.train_csr)
    val_by_user = dict(zip(split.val.user.astype(int), split.val.item.astype(int)))
    device = torch.device(os.environ.get("COALGAME_DEVICE", "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(os.environ.get("COALGAME_THREADS", "2"))))

    rows = []
    for seed in args.seeds:
        sd = out_dir / f"seed_{seed}"
        sd.mkdir(exist_ok=True)
        if (sd / "summary.csv").exists():
            print(f"seed {seed} already complete, skipping")
            continue
        torch.manual_seed(seed)
        rng_cfg = TrainConfig(dim=64, lr=0.002, weight_decay=1e-5, epochs=15,
                              batch_size=4096, n_neg=2, seed=seed, device=str(device))
        t0 = time.time()
        model = rmc.train_lightgcn_shared_prop(split.train, split.n_users, split.n_items, rng_cfg, n_layers=2)
        base_scores = cache_full_scores(model, split.n_users, batch_size=256,
                                        chunk_items=4096 if split.n_items > 8000 else None)
        item_embeddings = get_item_embeddings(model)
        print(f"seed {seed}: trained in {time.time()-t0:.0f}s")

        excl = {u: val_by_user[u] for u in val_by_user}
        for fam, scores in [
            ("unreranked", base_scores),
            ("last-knn", last_knn_scores(base_scores, split, val_by_user, item_embeddings)),
            ("updated-profile", updated_profile_scores(base_scores, split, val_by_user, item_vectors)),
            ("recency", recency_scores(base_scores, split, val_by_user, item_embeddings)),
        ]:
            summary, _ = evaluate(scores, split, item_vectors, ks=KS, exclude_by_user=excl)
            summary.update(seed=seed, family=fam)
            rows.append(summary)
            print(f"  {fam}: NDCG@20={summary['NDCG@20']:.5f}")
        if not args.skip_edge_update:
            t0 = time.time()
            eu = edge_update_scores(model, split, val_by_user, device)
            summary, _ = evaluate(eu, split, item_vectors, ks=KS, exclude_by_user=excl)
            summary.update(seed=seed, family="edge-update")
            rows.append(summary)
            print(f"  edge-update: NDCG@20={summary['NDCG@20']:.5f} ({time.time()-t0:.0f}s)")
        pd.DataFrame(rows).to_csv(sd / "summary.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary_by_seed_family.csv", index=False)
    metric_cols = [c for c in df.columns if c.startswith(("HitRate@", "NDCG@"))]
    df.groupby("family")[metric_cols].agg(["mean", "std"]).to_csv(out_dir / "summary_mean_std.csv")
    write_json(out_dir / "manifest.json", {
        "note": "Round-9 fix #10: same-information sequential baselines under the corrected v7 "
                "candidate exclusions (calibration item excluded for every family).",
        "candidate_exclusion": "candidates = I \\ (H_u_train U {i_u^+})",
        "families": ["unreranked", "last-knn", "updated-profile", "recency", "edge-update"],
        "seeds": args.seeds, "skip_edge_update": bool(args.skip_edge_update),
    })
    print("DONE ->", out_dir)


if __name__ == "__main__":
    main()
