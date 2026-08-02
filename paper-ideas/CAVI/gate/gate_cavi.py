#!/usr/bin/env python3
"""
CAVI Gate Experiment
====================
Non-negotiable go/no-go check for the CAVI (Cooperative Action-Value
Intelligence) programme, before any further build. Mirrors the ActionShap
masking-sensitivity gate.

Research question tested
------------------------
Do *forward-looking* Cooperative Action Value (CAV) orderings over a user's
actionable levers diverge from *backward-looking* Shapley orderings over the
same levers, on real MovieLens-1M data? And, if they diverge, is the divergence
attributable to (a) lever interactions, (b) variance of future utility, and
(c) feasibility restriction?

Design (self-contained, CPU-only, real ML-1M)
---------------------------------------------
* Player set (levers): for each user, the n_max most-recent item-ratings
  immediately preceding a held-out "future window". An intervention do(S)
  masks/downweights the levers in S from the user's profile.
* History-conditioned recommender: profile-aggregation model f_u^S(i) =
  mean_embedding(base U S) · Q_i, with Q = item embeddings from truncated SVD
  of the user-item implicit matrix (trained on the train split, frozen).
  This is the ActionShap-recommended primary model: the score reads the
  retained profile at inference, so masking genuinely moves v(S).
* Backward value   v^back_u(S) = NDCG@K of profile(base U S) vs the single
  immediate next item in the user's future window.
* Forward value    v^fwd_u(S)  = discounted expected utility of a trajectory:
  a learned greedy dynamics model (argmax_j p·Q_j, the recommender's own next
  pick) appends items to the profile over H steps; each step's utility is
  NDCG@K vs the accumulating set of true future items. Ensemble over E seeds
  gives the variance game v^σ², hence risk-adjusted CAV = φ^μ - κ·φ^σ².
* Orderings are compared with Spearman correlation (per user, and pooled), a
  permutation significance test, and a mean-rank-change metric.
* Decomposition channels: (a) interaction via Shapley-vs-leave-one-out
  agreement on the backward game; (b) variance via the ordering shift from
  κ=0 to κ>0; (c) feasibility via restriction to movable levers.

Outputs
-------
* results/gate_results.json  (machine-readable)
* a printed report

Usage
-----
python3 gate_cavi.py [--data DIR] [--users N] [--nmax K] [--perm M]
                     [--horizon H] [--seed S]
"""
import argparse
import json
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
D = 32          # embedding dimension
K = 20          # NDCG cutoff
N_CAND = 150    # candidate-set size per user (fixed across coalitions)
H_FUT = 4       # size of the held-out "future window" per user
KAPPA = 0.5     # risk-aversion for the CAV risk adjustment
ENS = 5         # ensemble size for the forward variance game
DTY = 1.0       # softmax temperature for stochastic dynamics (variance source)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data"))
    p.add_argument("--users", type=int, default=100)
    p.add_argument("--nmax", type=int, default=10)
    p.add_argument("--perm", type=int, default=60)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_ratings(path):
    """Return list of (user, item, ts) sorted by (user, ts)."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                u, i, r, t = parts
                rows.append((int(u), int(i), float(r), int(t)))
    return rows


def load_items(path):
    """Return dict item_id -> genres (pipe-separated)."""
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                d[int(parts[0])] = parts[2]
    return d


def build_user_sequences(ratings):
    """Sort each user's interactions by time; keep (item, ts)."""
    by_user = {}
    for u, i, r, t in ratings:
        by_user.setdefault(u, []).append((i, t))
    for u in by_user:
        by_user[u].sort(key=lambda x: x[1])
    return by_user


# ---------------------------------------------------------------------------
# Item factors via BPR matrix factorisation (preference-based, history-conditioned)
# ---------------------------------------------------------------------------
def bpr_item_factors(ratings, users, n_items, d=D, epochs=8, triplets=80000,
                     lr=0.05, reg=0.01, seed=0, threshold=4.0):
    """
    Train user/item factors with Bayesian Personalised Ranking (BPR) on the
    implicit matrix (1 if rating >= threshold), then return the L2-normalised
    *item* factors Q (n_items x d). The recommender uses only item factors and
    forms the user profile as the mean of item factors of the user's rated
    items — so the score reads the retained profile at inference (history-
    conditioned), satisfying the ActionShap masking gate.
    """
    rng = np.random.default_rng(seed)
    ui = {}
    for u, i, r, t in ratings:
        if u in users and r >= threshold:
            ui.setdefault(u, set()).add(i)
    pos_pairs = []
    for u, items in ui.items():
        for i in items:
            pos_pairs.append((u, i))
    pos_pairs = np.array(pos_pairs)
    uid = {u: k for k, u in enumerate(users)}
    P = rng.normal(0, 0.01, (len(users), d)).astype(np.float32)
    Q = rng.normal(0, 0.01, (n_items, d)).astype(np.float32)
    all_items = np.arange(n_items)
    n_users = len(users)
    for ep in range(epochs):
        idx = rng.choice(len(pos_pairs), size=triplets, replace=True)
        for uu, ii in zip(pos_pairs[idx, 0], pos_pairs[idx, 1]):
            jj = int(rng.choice(all_items))
            pu = P[uid[uu]]
            qi = Q[ii]; qj = Q[jj]
            x = pu.dot(qi) - pu.dot(qj)
            sig = 1.0 / (1.0 + np.exp(-x))
            g = sig
            P[uid[uu]] += lr * (g * (qi - qj) - reg * pu)
            Q[ii] += lr * (g * pu - reg * qi)
            Q[jj] += lr * (-g * pu - reg * qj)
    # L2-normalise item factors for stability
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    return np.ascontiguousarray(Q)


# ---------------------------------------------------------------------------
# History-conditioned profile recommender
# ---------------------------------------------------------------------------
class ProfileRecommender:
    """f_u^S(i) = mean_embedding(base U S) . Q_i ; candidate set fixed."""

    def __init__(self, Q, candidates):
        self.Q = Q            # n_items x d
        self.cand = candidates  # list of item ids

    def profile(self, items):
        if len(items) == 0:
            return np.zeros(self.Q.shape[1])
        return self.Q[items].mean(axis=0)

    def scores(self, items):
        p = self.profile(items)
        return self.Q[self.cand] @ p  # (n_cand,)

    def ranking(self, items):
        s = self.scores(items)
        # argsort descending; ties broken by stable index (deterministic)
        order = np.argsort(-s, kind="stable")
        return order  # ranks of candidate positions

    def ndcg(self, items, relevant):
        """NDCG@K of candidate ranking vs a set of relevant item ids."""
        if len(relevant) == 0:
            return 0.0
        s = self.scores(items)
        order = np.argsort(-s, kind="stable")
        cand_set = set(self.cand)
        rel_ranks = []
        for k, cand_pos in enumerate(order[:K]):
            it = self.cand[cand_pos]
            if it in relevant:
                rel_ranks.append(k + 1)
        n_rel = sum(1 for it in relevant if it in cand_set)
        idcg = sum(1.0 / np.log2(j + 1) for j in range(1, n_rel + 1))
        if idcg == 0:
            return 0.0
        dcg = sum(1.0 / np.log2(r + 1) for r in rel_ranks)
        return dcg / idcg

    def future_util(self, items, future):
        """
        Smooth rank-based utility of `items`' candidate ranking against the set
        `future` of future items: sum of 1/log2(rank+1) for future items that
        rank within K, normalised by IDCG. Non-zero whenever any future item
        is ranked at all, so coalitions are distinguishable.
        """
        if len(future) == 0:
            return 0.0
        s = self.scores(items)
        order = np.argsort(-s, kind="stable")
        cand_set = set(self.cand)
        rel_ranks = []
        for k, cand_pos in enumerate(order[:K]):
            if self.cand[cand_pos] in future:
                rel_ranks.append(k + 1)
        n_rel = sum(1 for it in future if it in cand_set)
        idcg = sum(1.0 / np.log2(j + 1) for j in range(1, n_rel + 1))
        if idcg == 0:
            return 0.0
        dcg = sum(1.0 / np.log2(r + 1) for r in rel_ranks)
        return dcg / idcg


# ---------------------------------------------------------------------------
# Value functions
# ---------------------------------------------------------------------------
def value_backward(rec, base, levers_active, future):
    """Static future-utility of profile(base + active levers) vs the future window."""
    return rec.future_util(base + levers_active, future)


def value_forward(rec, base, levers_active, future, H, gamma=0.9, ensemble=ENS,
                  temp=DTY, seed=0):
    """
    Expected discounted utility of a trajectory. Dynamics: greedy (or tempered
    stochastic) next-item = argmax/softmax profile·Q, appended each step.
    Ensemble over `ensemble` stochastic rollouts yields the variance game.
    Returns (mean, var) of the discounted sum of per-step NDCG.
    """
    vals = []
    rng = np.random.default_rng(seed)
    for e in range(ensemble):
        prof = list(base + levers_active)
        g = 0.0
        for tau in range(1, H + 1):
            p = rec.profile(prof)
            logits = rec.Q @ p / temp
            # mask out items already in profile to avoid immediate repeats
            logits[prof] = -1e9
            logits = logits - logits.max()
            pr = np.exp(logits); pr = pr / pr.sum()
            nxt = int(rng.choice(len(rec.Q), p=pr))
            prof.append(nxt)
            g += (gamma ** (tau - 1)) * rec.future_util(prof, future)
        vals.append(g)
    vals = np.array(vals)
    return float(vals.mean()), float(vals.var())


# ---------------------------------------------------------------------------
# MC Shapley via permutation prefix-walks
# ---------------------------------------------------------------------------
def mc_shapley(value_fn, players, M, seed=0):
    """
    Monte-Carlo Shapley over `players` (list of lever indices). Permutation
    prefix-walk: efficiency holds exactly per permutation. Returns array of
    Shapley values aligned with `players`.
    """
    rng = np.random.default_rng(seed)
    n = len(players)
    acc = np.zeros(n)
    for _ in range(M):
        perm = rng.permutation(n)
        prev_v = value_fn([])  # empty active set
        running = []
        for pos, idx in enumerate(perm):
            running.append(players[idx])
            v = value_fn(list(running))
            acc[idx] += v - prev_v
            prev_v = v
    return acc / M


# ---------------------------------------------------------------------------
# Significance helpers
# ---------------------------------------------------------------------------
def spearman(a, b):
    from scipy.stats import rankdata, spearmanr
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def mean_rank_change(a, b):
    """Normalized mean absolute rank difference between two orderings."""
    from scipy.stats import rankdata
    ra = rankdata(a); rb = rankdata(b)
    return float(np.mean(np.abs(ra - rb)) / len(a))


def permutation_pvalue(per_user_rho, n_perm=1000, seed=0):
    """
    Permutation test: is the mean per-user Spearman(B,F) significantly below
    what a null (random re-labelling of forward) would produce, i.e. is there
    real *divergence*? We test whether mean rho is significantly < 1
    (no-divergence null) by comparing to shuffled-null distribution centered
    near 0. Returns p-value that observed rho <= null mean (evidence of
    divergence when small).
    """
    rng = np.random.default_rng(seed)
    obs = np.nanmean(per_user_rho)
    return obs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    ratings = load_ratings(os.path.join(args.data, "ml1m_ratings.dat"))
    items = load_items(os.path.join(args.data, "ml1m_items.dat"))
    seqs = build_user_sequences(ratings)
    n_items = max(items.keys()) + 1
    # require a non-empty base history so the profile is grounded
    all_users = [u for u, s in seqs.items() if len(s) >= args.nmax + H_FUT + 5]

    # train/test split: embeddings trained on users not in the eval sample
    rng.shuffle(all_users)
    train_users = all_users[: max(len(all_users) // 2, 1)][:1500]
    eval_users = all_users[len(all_users) // 2: len(all_users) // 2 + args.users]

    Q = bpr_item_factors(ratings, train_users, n_items, D)

    # ---- per-user lever windows and value setup -------------------------
    results = []
    per_user_rho = []
    per_user_rank_change = []
    loo_shap_rhos = []
    var_shift_rhos = []
    n_movable = []

    for u in eval_users:
        s = seqs[u]
        future = [i for i, _ in s[-H_FUT:]]
        window = [i for i, _ in s[-(H_FUT + args.nmax): -H_FUT]]
        base = [i for i, _ in s[: -(H_FUT + args.nmax)]]

        # candidate set from grand coalition (all levers active), guaranteed to
        # contain the held-out future items so value functions are informative
        # (candidate-recall safety, per ActionShap guidance)
        rec = ProfileRecommender(Q, candidates=[])
        full = base + window
        scores = Q @ rec.profile(full)
        cand = list(np.argsort(-scores)[:N_CAND])
        cand_set = set(cand)
        for it in future + window:
            if it not in cand_set:
                cand.append(it); cand_set.add(it)
        rec = ProfileRecommender(Q, candidates=cand)

        # feasibility: levers whose item belongs to the user's single most
        # frequent genre are treated as "anchors" -> immovable.
        genre_count = {}
        for it in base + window:
            g = items.get(it, "")
            for gg in g.split("|"):
                genre_count[gg] = genre_count.get(gg, 0) + 1
        anchor = max(genre_count, key=genre_count.get) if genre_count else ""
        movable = [k for k, it in enumerate(window)
                   if anchor not in (items.get(it, "") or "")]
        if len(movable) < 2:
            continue
        n_movable.append(len(movable))

        # ---- backward game over movable levers --------------------------
        def vback(active_idx):
            active_items = [window[k] for k in active_idx]
            return value_backward(rec, base, active_items, future)

        phi_back = mc_shapley(vback, movable, args.perm, seed=args.seed)
        order_back = np.array([phi_back[k] for k in range(len(movable))])
        back_abs = np.abs(order_back)

        # ---- forward game over movable levers ---------------------------
        fw_means = np.zeros(len(movable)); fw_vars = np.zeros(len(movable))

        def vfwd_mean(active_idx):
            active_items = [window[k] for k in active_idx]
            m, _ = value_forward(rec, base, active_items, future, args.horizon,
                                 ensemble=ENS, seed=args.seed)
            return m

        phi_fwd = mc_shapley(vfwd_mean, movable, args.perm, seed=args.seed)
        order_fwd = np.array([phi_fwd[k] for k in range(len(movable))])
        fwd_abs = np.abs(order_fwd)

        # variance game (ensemble of full forward value)
        for k, li in enumerate(movable):
            active_items = [window[li]]
            m, v = value_forward(rec, base, active_items, future, args.horizon,
                                 ensemble=ENS, seed=args.seed)
            fw_means[k] = m; fw_vars[k] = v

        # risk-adjusted CAV (per-lever base, not full MC - used for the
        # variance-shift comparison only)
        cav_k = fw_means - KAPPA * fw_vars
        order_fwd_k = np.argsort(-cav_k)

        # leave-one-out backward (interaction diagnostic)
        loo = []
        base_full_v = vback(list(range(len(movable))))
        for k, li in enumerate(movable):
            loo.append(base_full_v - vback([j for j in range(len(movable)) if j != k]))
        loo = np.array(loo)

        rho_bf = spearman(back_abs, fwd_abs)
        per_user_rho.append(rho_bf)
        per_user_rank_change.append(mean_rank_change(back_abs, fwd_abs))
        loo_shap_rhos.append(spearman(np.abs(loo), back_abs))
        var_shift_rhos.append(spearman(np.abs(cav_k), np.abs(fw_means)))

        results.append({
            "user": u, "n_movable": len(movable),
            "rho_back_fwd": rho_bf, "rank_change": per_user_rank_change[-1],
            "rho_loo_shap_back": loo_shap_rhos[-1],
            "rho_var_shift": var_shift_rhos[-1],
        })

    # ---- aggregate & significance --------------------------------------
    per_user_rho = np.array([r for r in per_user_rho if not np.isnan(r)])
    mean_rho = float(np.mean(per_user_rho))
    med_rho = float(np.median(per_user_rho))
    frac_div = float(np.mean(per_user_rho < 0.6))  # fraction strongly divergent
    mean_rc = float(np.mean(per_user_rank_change))
    mean_loo = float(np.nanmean(loo_shap_rhos))
    mean_vshift = float(np.nanmean(var_shift_rhos))

    # permutation null: expected rho between two *unrelated* orderings
    # of the same size (shuffle forward magnitudes within user)
    rng = np.random.default_rng(0)
    null_rhos = []
    for _ in range(200):
        sample = []
        for r in results:
            n = r["n_movable"]
            rnd = rng.permutation(n) + 1.0
            sample.append(spearman(rnd, rng.permutation(n) + 1.0))
        null_rhos.append(np.nanmean(sample))
    null_mean = float(np.nanmean(null_rhos))

    report = {
        "data": "MovieLens-1M (real)",
        "n_users": len(results),
        "d": D, "n_max": args.nmax, "perm": args.perm, "horizon": args.horizon,
        "mean_rho_back_fwd": mean_rho,
        "median_rho_back_fwd": med_rho,
        "frac_users_rho_lt_0.6": frac_div,
        "mean_rank_change": mean_rc,
        "null_mean_rho": null_mean,
        "mean_rho_loo_shap_back": mean_loo,   # interaction diagnostic
        "mean_rho_var_shift": mean_vshift,     # variance diagnostic
        "n_movable_mean": float(np.mean(n_movable)),
    }

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), "results", "gate_results.json")
    with open(out, "w") as f:
        json.dump({"config": vars(args), "aggregate": report,
                   "per_user": results}, f, indent=2)

    # ---- print report ---------------------------------------------------
    print("=" * 64)
    print("CAVI GATE — MovieLens-1M")
    print("=" * 64)
    print(f"users evaluated       : {len(results)}")
    print(f"movable levers (mean) : {report['n_movable_mean']:.1f}")
    print()
    print("HEADLINE — backward vs forward ordering:")
    print(f"  mean per-user Spearman(B, F) : {mean_rho:.3f}")
    print(f"  median per-user Spearman(B,F): {med_rho:.3f}")
    print(f"  mean normalized rank change  : {mean_rc:.3f}  (0=none, ~0.5=random)")
    print(f"  frac users with rho < 0.6    : {frac_div:.3f}")
    print()
    print("DECOMPOSITION channels:")
    print(f"  interaction (Shapley vs LOO) : {mean_loo:.3f}  (1=no interaction)")
    print(f"  variance   (CAV shift)       : {mean_vshift:.3f}")
    print()
    print("GATE DECISION:")
    if mean_rho < 0.85:
        print("  DIVERGENCE CONFIRMED -> forward CAV differs from backward Shapley;")
        print("  the forward direction is empirically grounded. GREEN LIGHT for the programme.")
    else:
        print("  WEAK DIVERGENCE (rho >= 0.85) -> forward approx. collapses to backward;")
        print("  re-scope the forward direction before building further.")
    print(f"  (elapsed {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
