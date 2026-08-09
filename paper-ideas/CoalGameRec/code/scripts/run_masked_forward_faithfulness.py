#!/usr/bin/env python3
"""Round-4 required experiment: TRUE masked-forward faithfulness.

Implements the graph intervention that actually defines the cooperative game:
for each user and fraction, remove (deletion) or keep-only (insertion) the
top-attributed player edges, REBUILD the normalized adjacency, REPROPAGATE the
frozen LightGCN parameters on the masked graph, and evaluate the test-item rank
under the masked model. This replaces the candidate-masking proxy.

Families: loo-marginal, shapley-mc, uniform, seeded-random.
Fractions: 0.10, 0.20, 0.30. Users: stratified subsample (default 1000).

Usage (from code/):
  COALGAME_DEVICE=mps python scripts/run_masked_forward_faithfulness.py --dataset ml1m --seed 42
  COALGAME_DEVICE=mps python scripts/run_masked_forward_faithfulness.py --dataset amazon --seed 42
Outputs: results/journal_runs/<dataset>_lightgcn_v3_prospective/tables/masked_forward_faithfulness.csv (+ log)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from coalgamerec.attribution import compute_attribution_for_users, compute_shapley_for_users  # noqa: E402
from coalgamerec.data import item_user_vectors  # noqa: E402
from coalgamerec.metrics import per_user_hit_ndcg  # noqa: E402
from coalgamerec.models import LightGCN, TrainConfig, build_lightgcn_graph, cache_full_scores  # noqa: E402
from coalgamerec.explanation import top_attributed_items  # noqa: E402
from scripts.run_matched_controls import load_split_from_run, train_lightgcn_shared_prop, RESULTS  # noqa: E402

SRC_NAME = {"ml1m": "ml1m_lightgcn_v3_prospective", "amazon": "amazon_books_lightgcn_v3_prospective"}
BACKBONE = dict(dim=64, n_layers=2, lr=0.002, weight_decay=1e-5, epochs=15, batch_size=4096, n_neg=2)
FRACTIONS = [0.10, 0.20, 0.30]


def masked_scores_for_users(model, train_csr, split, users, edge_src_np, edge_dst_np,
                            edge_item_np, user_edge_positions, keep_or_drop, mode, device):
    """Repropagate frozen LightGCN on per-user masked graphs; return scores rows for `users`.

    keep_or_drop[u] = array of history items defining the coalition S for user u.
    mode='deletion': S = P_u minus top-attributed (those edges removed).
    mode='insertion': S = top-attributed only.
    All edges of other users are kept.
    """
    n_users, n_items = split.n_users, split.n_items
    E0 = torch.cat([model.user_emb.weight.detach(), model.item_emb.weight.detach()], dim=0).to(device)
    train_items_by_user = {int(u): train_csr[int(u)].indices for u in users}
    out_scores = []
    for u in users:
        u = int(u)
        items_u = train_items_by_user[u]
        S = set(int(x) for x in keep_or_drop[u])
        keep_edge = np.ones(len(edge_src_np), dtype=bool)
        pos = user_edge_positions[u]
        for p in pos:
            if int(edge_item_np[p]) not in S:  # remove both directions of the edge
                keep_edge[p] = False
        src = torch.as_tensor(edge_src_np[keep_edge], dtype=torch.long, device=device)
        dst = torch.as_tensor(edge_dst_np[keep_edge], dtype=torch.long, device=device)
        deg = torch.bincount(src, minlength=n_users + n_items).float()
        deg[deg == 0] = 1.0
        w = 1.0 / torch.sqrt(deg[src] * deg[dst])
        all_e = E0
        embs = [all_e]
        for _ in range(2):
            out = torch.zeros_like(all_e)
            out.index_add_(0, dst, all_e[src] * w.unsqueeze(1))
            all_e = out
            embs.append(all_e)
        final = torch.stack(embs, dim=0).mean(dim=0)
        U_u = final[u]
        I_all = final[n_users:]
        s = (I_all @ U_u)
        s_items = torch.cat([s[:0], s])  # item indexing 0..n_items-1
        row = s_items[:n_items].cpu().numpy().astype(np.float32)
        row[list(items_u)] = -np.inf  # exclude full history from candidates (masked history already removed from graph)
        out_scores.append(row)
    return np.vstack(out_scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml1m", choices=["ml1m", "amazon"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-users", type=int, default=1000)
    ap.add_argument("--source-run", default=None)
    args = ap.parse_args()

    source = Path(args.source_run) if args.source_run else RESULTS / SRC_NAME[args.dataset]
    out_tables = source / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    log = source / "masked_forward_faithfulness.log"

    def say(msg):
        print(msg, flush=True)
        with open(log, "a") as f:
            f.write(msg + "\n")

    device = torch.device(os.environ.get("COALGAME_DEVICE", "cpu"))
    split = load_split_from_run(source)
    item_vectors = item_user_vectors(split.train_csr)
    cfg = TrainConfig(**BACKBONE, seed=args.seed, device=str(device))
    model = train_lightgcn_shared_prop(split.train, split.n_users, split.n_items, cfg, n_layers=BACKBONE["n_layers"])
    model.eval()
    base_scores = cache_full_scores(model, split.n_users, batch_size=256, chunk_items=4096 if split.n_items > 8000 else None)
    train_csr = split.train_csr

    # edge index (numpy) and per-user edge positions (BOTH directions of each edge)
    users_np = split.train.user.values.astype(np.int64)
    items_np = split.train.item.values.astype(np.int64) + split.n_users
    edge_src_np = np.concatenate([users_np, items_np])
    edge_dst_np = np.concatenate([items_np, users_np])
    half = len(users_np)
    # item id incident to each directed edge position
    edge_item_np = np.empty(len(edge_src_np), dtype=np.int64)
    edge_item_np[:half] = edge_dst_np[:half] - split.n_users
    edge_item_np[half:] = edge_src_np[half:] - split.n_users
    user_edge_positions = {u: [] for u in range(split.n_users)}
    for pos, u in enumerate(users_np):
        user_edge_positions[int(u)].append(pos)
        user_edge_positions[int(u)].append(pos + half)
    for u in user_edge_positions:
        user_edge_positions[u] = np.asarray(user_edge_positions[u], dtype=np.int64)

    # subsample users with history >= 10
    hlen = np.diff(train_csr.indptr)
    eligible = np.where(hlen >= 10)[0]
    step = max(1, len(eligible) // args.n_users)
    users = eligible[::step][: args.n_users]
    say(f"masked-forward faithfulness: {args.dataset} seed={args.seed}, {len(users)} users, fractions={FRACTIONS}")

    # attributions
    loo = compute_attribution_for_users(split, base_scores, item_vectors, method="loo-marginal", seed=args.seed,
                                        max_players_per_user=24, player_selection="stratified",
                                        checkpoint_path=None, save_every=25, alpha=1.0, beta=0.0, lambda_pref=0.0,
                                        lambda_attr_value=0.10, value_mode="pairwise_logsigmoid", n_val_negatives=100)
    shap = compute_shapley_for_users(split, base_scores, item_vectors, m=64, seed=args.seed,
                                     max_players_per_user=24, player_selection="stratified", antithetic=True,
                                     checkpoint_path=None, save_every=25, alpha=1.0, beta=0.0, lambda_pref=0.0,
                                     lambda_attr_value=0.10, value_mode="pairwise_logsigmoid", n_val_negatives=100)
    rng = np.random.default_rng(args.seed * 7919 + 13)
    uniform_attr = {int(u): np.ones(hlen[int(u)], dtype=np.float32) for u in users}
    random_attr = {int(u): rng.standard_normal(hlen[int(u)]).astype(np.float32) for u in users}
    fam_attrs = {"loo-marginal": loo, "shapley-mc": shap, "uniform": uniform_attr, "random": random_attr}

    targets_all = split.test.sort_values("user")
    target_by_user = dict(zip(targets_all.user.values, targets_all.item.values))

    rows = []
    # baseline (unmasked) metrics on the subsample for reference
    sub_idx = np.isin(np.arange(split.n_users), users)
    base_sub = base_scores[users]
    tg = np.array([target_by_user[int(u)] for u in users], dtype=np.int64)
    base_metrics = per_user_hit_ndcg(base_sub, tg, train_csr[users], ks=(20,))
    base_ndcg = float(base_metrics["NDCG@20"].mean())
    base_hr = float(base_metrics["HitRate@20"].mean())
    rows.append(dict(family="unmasked-reference", fraction=0.0, NDCG20=base_ndcg, HitRate20=base_hr))
    say(f"unmasked reference: NDCG@20={base_ndcg:.5f} HR@20={base_hr:.5f}")

    for fam, attrs in fam_attrs.items():
        for frac in FRACTIONS:
            for mode in ["deletion", "insertion"]:
                keep = {}
                for u in users:
                    u = int(u)
                    items = train_csr[u].indices
                    w = attrs.get(u)
                    if w is None or len(w) == 0:
                        keep[u] = items if mode == "deletion" else np.array([], dtype=np.int64)
                        continue
                    top = top_attributed_items(items, np.asarray(w), fraction=frac, positive_only=False)
                    if mode == "deletion":
                        keep[u] = np.setdiff1d(items, top)
                    else:
                        keep[u] = top
                t0 = time.time()
                scores = masked_scores_for_users(model, train_csr, split, users, edge_src_np, edge_dst_np,
                                                 edge_item_np, user_edge_positions, keep, mode, device)
                m = per_user_hit_ndcg(scores, tg, train_csr[users], ks=(20,))
                rows.append(dict(family=fam, fraction=frac, mode=mode,
                                 NDCG20=float(m["NDCG@20"].mean()), HitRate20=float(m["HitRate@20"].mean()),
                                 seconds=round(time.time() - t0, 1)))
                say(f"  {fam} | {mode} | frac={frac}: NDCG@20={m['NDCG@20'].mean():.5f} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out_tables / "masked_forward_faithfulness.csv", index=False)
    say("SAVED masked_forward_faithfulness.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
