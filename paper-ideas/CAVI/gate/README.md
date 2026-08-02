# CAVI Gate Experiment

**Non-negotiable go/no-go check for the CAVI (Cooperative Action-Value Intelligence) programme**, run before any further build — the half-day MovieLens divergence check that the proposal's §3.13 makes a hard gate.

**Verdict: DIVERGENCE CONFIRMED → GREEN LIGHT.**

The forward-looking Cooperative Action Value ordering over a user's actionable levers is **essentially unrelated** to the backward-looking Shapley ordering over the same levers, on real MovieLens-1M data. The forward direction is empirically grounded, not assumed.

---

## What it tests

The entire novelty of CAVI rests on one claim: a *forward-looking* cooperative game (value = expected discounted **future** utility under a learned dynamics model) yields a different, and more decision-relevant, ranking of actions than the *backward-looking* game (value = **current/immediate** utility) that the thesis and all existing Shapley recommenders use. If the two orderings coincide, the forward game is a tautology and the programme should be re-scoped. The gate measures whether they diverge.

## Design (real ML-1M, CPU-only, reproducible)

| Component | Choice |
|---|---|
| Data | MovieLens-1M (1,000,209 ratings, 3,883 movies w/ genres), from a GitHub mirror (grouplens host is TLS-blocked in this sandbox). |
| Player set (levers) | per user, the `n_max=10` most recent item-ratings immediately preceding a held-out future window of `H_FUT=4`. Intervention `do(S)` = mask levers in `S` from the profile. |
| Recommender | **History-conditioned profile-aggregation** `f_u^S(i) = mean_embedding(base ∪ S) · Q_i`, `Q` = L2-normalised **BPR-MF item factors** (d=32, trained on ~1500 held-out users). The score reads the retained profile at inference → masking genuinely moves `v(S)` (satisfies the ActionShap masking-sensitivity gate). |
| Candidate set | fixed per user: top-150 by grand coalition + guaranteed to contain the future/window items (candidate-recall safety). |
| Backward value | `v^back_u(S) = future_util(profile(base ∪ S), future)` — static rank utility of the profile vs the user's real future items. |
| Forward value | `v^fwd_u(S)` = discounted (`γ=0.9`) expected utility of a **trajectory**: a learned greedy/tempered dynamics model (`argmax/softmax p·Q_j`) appends its predicted next item over `H=3` steps, scoring each step vs the accumulating future set; ensemble of `ENS=5` rollouts gives the variance game `v^σ²`, hence risk-adjusted `CAV = φ^μ − κ·φ^σ²`. |
| Allocation | Monte-Carlo Shapley via permutation prefix-walks (`M=60`), restricted to **movable** levers (feasibility = anchors in the user's dominant genre are immutable). |
| Divergence metric | per-user Spearman(`|φ^back|`, `|φ^fwd|`), plus a permutation null, mean normalized rank change, and the fraction of users with ρ<0.6. |

## Results (canonical run, seed 7)

```
users evaluated        : 73
movable levers (mean)  : 6.3

HEADLINE
  mean per-user Spearman(B, F) : 0.093   (null ~ 0.002)
  median per-user Spearman(B,F): 0.108
  mean normalized rank change  : 0.163   (0=none, 0.5=random)
  frac users with rho < 0.6    : 0.867

DECOMPOSITION channels
  interaction (Shapley vs LOO) : -0.217  (negative ⇒ strong lever interaction)
  variance   (CAV risk shift)  : 1.000   (risk-adjustment did NOT reorder)
```

**Robustness:** seed 42 gives mean ρ=0.123, 0.895 of users divergent — the divergence is not a fluke of one seed.

## Interpretation

- **Forward ≠ backward, and strongly.** A mean Spearman of ≈0.1 against a ~0 random null means the two orderings are essentially uncorrelated. The levers that earn a user's *immediate* next-item utility are not the levers that seed a good *future trajectory* — exactly the CAVI premise. **GREEN LIGHT.**
- **Interaction channel:** backward-Shapley disagrees with backward-leave-one-out (negative correlation), i.e. levers interact strongly. This is consistent with the redundancy/complementarity story that justifies a coalition-aware (Shapley) forward allocation over additive heuristics. *(Caution: this diagnostic is noisy at ~6 levers/user with many near-zero values.)*
- **Variance channel (honest negative):** risk-adjusting the forward value (`CAV = φ^μ − κφ^σ²`) did **not** reorder the levers in this gate (ρ≈1.0). As measured by the cheap *singleton* proxy, the variance of future utility is roughly proportional to its mean, so the risk term is a scaling, not a reordering, factor here. A full Shapley-of-the-variance-game would be needed to conclude whether variance ever drives ordering on other data/lever spaces; this gate does not claim it does.

## Limitations (be honest)

1. **Value spikiness.** Single-future-item rank utility is coarse; many coalitions score 0, so Shapley estimates are noisy at small `n_max`. The extreme divergence is robust across seeds, but the *precise* ordering is not trustworthy at this lever scale.
2. **Variance-channel proxy is weak.** It compares singleton means to risk-adjusted singletons, not the Shapley of the variance game.
3. **One dataset, one recommender family, one dynamics model.** This is a gate, not the Paper-A experiment; it bounds the forward-vs-backward *premise*, not the full method.
4. **BPR item factors, mean-pooling** are a simple recommender; the divergence is a property of this (realistic) setup, and should be re-checked on the actual Paper-A backbone.

## Reproduce

```
python3 gate_cavi.py --users 100 --nmax 10 --perm 60 --horizon 3 --seed 7
```

Requires `data/ml1m_ratings.dat` + `data/ml1m_items.dat` (gitignored; ~20 MB). Full results in `results/gate_results.json`.

## Gate decision & next step

**Go. Proceed to Paper A** (the forward certainty-equivalent game, CAV allocation, and provable A1/A2/A4 properties — Theorem 1, Myerson-restricted construction) using the real divergence as its empirical motivation. In the eventual Paper-A experiment, strengthen the variance channel to the full Shapley-of-variance-game and add a second dataset (rebuilt Amazon-Book) before making quantitative ordering claims.
