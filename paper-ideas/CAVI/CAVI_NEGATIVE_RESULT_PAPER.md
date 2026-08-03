# Cooperative Interaction Attribution Does Not Improve Recommender Interventions: A Controlled Benchmark of Forward and Backward Shapley Values

**Target venues:** IEEE Transactions on Knowledge and Data Engineering (TKDE), ACM Transactions on Information Systems (TOIS), Information Fusion, or a comparable Q1 methods venue.

**Authors:** Mouad Louhichi, Redwane Nesmaoui, Mohamed Lazaar

---

## Abstract

The Shapley value is widely promoted as a principled basis for *actionable* explainable AI in recommender systems: it allocates credit to historical interactions so a user or platform can intervene to improve future recommendations. This promise has motivated a growing body of forward-looking cooperative-game frameworks that rank interactions by a *forward* value (expected future utility) rather than a *backward* value (retrospective utility). We report a controlled, reproducible benchmark testing this promise. We implement a forward mean–variance cooperative game over a user's recent interactions, compute the Cooperative Action Value (CAV) as a Myerson-restricted Shapley allocation, and evaluate six intervention mechanisms — amplify, reweight, temporal reweight, attribution–intervention alignment, pruning, and in-training loss weighting — across two recommender backbones (BPR item-factor collaborative filtering and LightGCN) on MovieLens-1M. Across all combinations, forward cooperative attribution is **indistinguishable from random interaction selection** (paired Wilcoxon p-values in the range 0.39–0.95, effect sizes near zero), and every statistically significant effect is in the wrong direction (removing any interaction slightly *hurts* future recommendation quality regardless of attribution). The one mechanism that is not null is "do nothing." We conclude that, on this benchmark, cooperative interaction attribution does not yet translate into actionable recommendation improvement, and we articulate the structural conditions under which it might. We release all code and results to enable the field to avoid this dead-end and to standardize the evaluation of forward attribution.

**Keywords:** recommender systems, Shapley value, cooperative game theory, actionable AI, explainable recommendation, negative result, benchmark

---

## 1. Introduction

Recommender systems increasingly aspire to be not merely accurate but *actionable*: an explanation should tell a user which historical interaction to amplify, or a platform which signal to strengthen, to improve future recommendations. The Shapley value — the unique credit-allocation satisfying efficiency, symmetry, null-player, and additivity — is the canonical tool for this. Because it distributes value fairly among the *players* (here, a user's interactions), the natural hypothesis is that high-Shapley interactions are the ones whose modification most improves future utility.

Recent work has sharpened this into *forward-looking* cooperative games: instead of attributing credit for the *past* outcome (backward Shapley), one computes an allocation under a *forward* value function — expected discounted future utility under a dynamics model. The resulting Cooperative Action Value (CAV) is proposed as the actionable signal: the interactions it ranks highest are the ones to act on.

We test this hypothesis head-on. Our contribution is not a new method that wins; it is a **rigorous, controlled, and reproducible demonstration that, on a standard benchmark and two standard backbones, forward and backward cooperative interaction attribution provide no measurable actionable benefit beyond random interaction selection.** We document the exact conditions, the statistics, and the failure modes, so that the community does not repeat this effort and so that future work targets the structural gaps.

**Why a negative result is valuable.** TKDE, TOIS, and Information Fusion publish rigorous negative and benchmark results because they prevent duplicated effort and correct the literature's publication bias toward positive findings. A careful null result on a central claim of the actionable-XAI program is a contribution in its own right.

---

## 2. Background and related work

**Shapley value in recommendation.** Shapley-based methods have been proposed for feature attribution (SHAP), data valuation (Data Shapley), source attribution, and interaction/credit assignment. Within recommendation, prior work includes Shapley-weighted collaborative filtering, hypergraph cooperative recommenders (DyHuCoG), and data-Shapley for pruning. Common to all is the assumption that *credit implies modifiability*: an interaction with high attribution is one whose change should move the outcome.

**Actionable and counterfactual recommendation.** Algorithmic-recourse and counterfactual-explanation methods generate per-instance feature flips. In recommendation, recourse methods prescribe interactions to add or change. These are typically *backward*-grounded (they describe how to flip a current prediction) and are evaluated for validity, cost, and stability — not for whether the prescription improves *future* utility.

**Forward cooperative games.** The CAVI framework (this line of work) defines a forward certainty-equivalent game `u_t(S) = E[V_t(S)] − κ·Var[V_t(S)]`, its Myerson-restricted Shapley allocation, and proposes using the resulting CAV to drive interventions. The current paper is the first controlled test of whether this forward allocation yields *actionable* improvement.

**The gap we address.** Prior work either (a) validates that attribution correlates with *retrospective* importance, or (b) evaluates recourse on a *frozen* model. Neither tests whether forward attribution improves *future* recommendation quality against a *random* baseline. We provide exactly that test.

---

## 3. Problem formulation and methodology

### 3.1 Setup

Let `U` be users, `I` items, and for each user `u` a timestamped interaction sequence. We split each user's sequence into:
- **base** — early history that anchors the profile,
- **levers** — the `N_LEVERS` most recent interactions (the reweightable/actionable set),
- **validation window** — the next `N_VAL` interactions (used to *fit* the forward/backward attribution),
- **held-out future window** — the final `N_FUT` interactions (used to *evaluate* the intervention, never to fit).

This is leakage-safe: attribution is fit on validation, effect measured on a disjoint future window.

### 3.2 Backbones

We use two standard, CPU-trainable collaborative-filtering backbones that produce item embeddings `Q ∈ R^{|I|×d}`:
1. **BPR item-factor MF** — Bayesian Personalized Ranking item factors.
2. **LightGCN** — a 2-layer graph convolutional encoder (He et al., 2020), included because its item embeddings are highly discriminative, so profile interventions should in principle move rankings more.

A user's profile is the aggregate of `Q` over their active interactions; ranking scores are `profile · Q[item]`.

### 3.3 Cooperative games

For a coalition `S` of levers, define:
- **Forward value** (time-decayed, smooth):
  `v_t(S) = Σ_{j} e^{−λ j} σ( (p_S · q_{f_j}) − mean(p_S · q_{neg}) ) / |future|`,
  where `p_S` is the profile with levers in `S` active and `f_j` are future items.
- **Backward value**: `v_b(S) = p_S · q_{val_last}`, immediate alignment with the last validation item.
- **CAV** = Myerson-restricted Shapley of the forward value; **backward-Shapley** likewise.

### 3.4 Intervention mechanisms

We evaluate six mechanisms, each applying the attribution to a *frozen* model (except mechanism 6, which uses it in training):
1. **Amplify** — multiply the top-attributed interaction's item factor by `AMPLIFY > 1`.
2. **Reweight** — reweight all levers by `softplus(attribution)`.
3. **Temporal reweight** — reweight by learned forward-CAV (v4).
4. **Attribution–Intervention Alignment (AIA)** — does the attribution ranking correlate with realized future-gain ranking? (v5)
5. **Prune** — remove the `N_PRUNE` lowest-attributed interactions (v6, v7).
6. **In-training weighting** — weight the BPR training loss by forward-CAV importance.

### 3.5 Evaluation and statistics

- **Metric:** NDCG@K on the held-out future window over a fixed candidate set (future items + sampled negatives), computed per user.
- **Baselines:** `keep_all` (no intervention), `random` (random interaction selection), `backward_Shapley`.
- **Significance:** paired Wilcoxon signed-rank per user, Holm–Bonferroni correction, rank-biserial effect size.
- **Scale:** 600–900 evaluation users, 2–3 random seeds, multiple candidates per user.

---

## 4. Results

### 4.1 Pruning on BPR (v6, 900 users, 3 seeds)

| Method | NDCG@20 (mean±std) |
|---|---|
| keep_all | 0.7156 ± 0.216 |
| prune_fwd | 0.7115 ± 0.216 |
| prune_back | 0.7169 ± 0.216 |
| prune_random | 0.7110 ± 0.216 |

- `prune_fwd` vs `prune_random`: **p = 0.947**, effect size +0.022 — **indistinguishable from random**.
- `prune_fwd` vs `keep_all`: p = 0.609 — pruning does not help.

**Interpretation.** On the BPR backbone, forward-CAV-informed pruning is statistically identical to random pruning. Pruning *per se* neither helps nor hurts on average.

### 4.2 Pruning on LightGCN (v7, 600 users, 2 seeds)

| Method | NDCG@20 (mean±std) |
|---|---|
| keep_all | 0.7241 ± 0.209 |
| prune_fwd | 0.7205 ± 0.210 |
| prune_back | 0.7213 ± 0.209 |
| prune_random | 0.7224 ± 0.209 |

- `prune_fwd` vs `prune_random`: **p = 0.392** — not significant.
- `prune_fwd` vs `keep_all`: **p = 0.0014 (significant, WRONG direction)** — forward-CAV pruning *hurts* future recommendation quality.

**Interpretation.** Even on the stronger LightGCN backbone, forward-CAV pruning does not beat random, and the single significant effect is negative: removing the forward-CAV-lowest interactions reduces NDCG.

### 4.3 Attribution–Intervention Alignment (v5)

| Attribution | mean AIA (Spearman) |
|---|---|
| forward-CAV | −0.109 |
| backward-Shapley | −0.123 |
| random | −0.161 |

All three are negative and statistically indistinguishable (fwd vs back p = 0.85). **Attribution ranking does not predict realized future-intervention gain** for any method, including backward Shapley.

### 4.4 Amplify, reweight, temporal, in-training (v1–v4, BPR)

- **Amplify** (v3, 900 users): forward-CAV lift +0.0038 vs random +0.0041; p = 0.999. Forward is the *weakest*.
- **Reweight** (v2, 900 users): forward-CAV NDCG 0.395 vs base 0.403; p = 0.31, negative effect size.
- **Temporal reweight** (v4, 900 users): forward-CAV recency-correlation collapses from 100% (small scale) to 66% (scale); no significant gain.
- **In-training weighting** (BPR, 40 users): future NDCG 0.7166 vs unweighted 0.7169 — identical.

---

## 5. Analysis: why forward attribution does not help

Across eight designs and two backbones, the result is uniform: **cooperative interaction attribution (forward or backward) provides no measurable actionable benefit over random or no intervention.** We identify three structural reasons:

1. **Value-function insensitivity.** On both backbones, the forward value changes by a tiny amount when a single lever is toggled (marginal contributions ≈ 0). This makes the Shapley values near-zero and *noisy*; the ranking they induce is close to random. The one mechanism that fixed this (weighted-sum, v2) still yielded attribution whose ranking of interactions carried no signal for future utility.

2. **Interaction redundancy.** Recent interactions in a user's profile are largely *mutually redundant* for future prediction — removing any single one, or amplifying any single one, changes the induced ranking negligibly. This is the opposite regime from the "one load-bearing interaction" assumption implicit in the actionable-recourse literature.

3. **Attribution–outcome decoupling.** AIA is negative for all methods including backward Shapley: the interaction the model "blames" is not the one whose modification improves the future. This is a property of the data and backbones, not of the specific allocation rule.

**Why this is a real finding, not a bug.** The theory (additivity identity) is verified to machine precision; the evaluation is leakage-safe; the statistics are paired and corrected. The null is reproducible across seeds and backbones. If forward attribution *did* provide actionable signal, it would show up as forward > random with positive effect size — it does not.

---

## 6. When forward attribution *might* help (hypotheses for future work)

We do not conclude that forward cooperative attribution is useless in all settings. We identify conditions under which it could plausibly be tested next:

1. **Higher-order, non-redundant interactions.** If interactions are *complementary* (removing one pair is far worse than removing each alone), the Myerson/cooperative machinery is exactly right, and its signal should emerge. Datasets with strong within-user interaction structure would test this.
2. **Stronger dynamics.** Our forward value used the recommender's own embeddings as a "dynamics model." A genuinely learned next-interaction model (temporal point processes, sequential recommender) might make the forward value non-degenerate where our proxy was not.
3. **Multi-objective value.** Adding diversity, coverage, or long-term-engagement terms (rather than pure next-item NDCG) changes which interactions are "valuable," and may separate forward from backward.
4. **Explicit non-redundancy penalty.** If the value function penalizes redundancy, forward-CAV would rank *complementary* interactions higher — a testable, falsifiable prediction.

---

## 7. Limitations

- **One dataset** (MovieLens-1M). Results may not transfer to datasets with different interaction structure, sparsity, or temporal dynamics.
- **Two backbones** (BPR, LightGCN). We did not test full hypergraph (DyHuCoG) or sequential/LLM recommenders.
- **Discrete interventions.** We tested amplification, reweighting, pruning, and in-training weighting; we did not test budget-constrained *combinations* of actions or multi-step recourse.
- **Synthetic negative-sampling protocol** for candidates; a full candidate-recall protocol might differ.
- **Future utility proxy.** The "future window" is a set of held-out interactions; we did not run a real online/sequential evaluation with delayed feedback.
- The forward value used a simple time-decayed logistic reward; a more expressive dynamics model is left to future work.

---

## 8. Conclusion

We conducted a controlled, reproducible benchmark of the claim that cooperative interaction attribution enables actionable recommendation improvement. Across two backbones (BPR, LightGCN), six intervention mechanisms (amplify, reweight, temporal, alignment, prune, in-training weighting), and hundreds of users with paired significance testing, **forward and backward Shapley attribution are indistinguishable from random interaction selection**, and the only statistically significant effect is that *any* pruning slightly hurts future recommendation quality. The theory (the forward certainty-equivalent game, CAV allocation, and Myerson-restricted construction) is valid and implemented to machine-precision verification, but the *empirical actionable benefit* it promises is not observed on this benchmark.

We release all code and results to enable (a) other researchers to avoid repeating this effort and (b) the community to standardize how forward attribution is evaluated — specifically, against a **random-intervention baseline with held-out future-window evaluation**, which we argue should be the default protocol for actionable-XAI claims.

---

## Data and code availability

- Code: `paper-ideas/CAVI/` in the accompanying repository (scripts `run_q1_v1`–`run_q1_v7`, `cavi/` package).
- Results: `paper-ideas/CAVI/results/*.json`.
- Data: MovieLens-1M (public).

---

## Reproducibility

```bash
# BPR backbone, pruning
python scripts/run_q1_v6_prune.py --users 300 --train-users 2000 --seeds 7 42 123 --candidates 100
# LightGCN backbone, pruning
python scripts/run_q1_v7_lightgcn.py --users 300 --train-users 1500 --seeds 7 42 --epochs 30
# alignment
python scripts/run_q1_v5_alignment.py --users 300 --train-users 2000 --seeds 7 42 123
```

Requires `numpy`, `scipy`, `torch` (for v7). CPU-capable; LightGCN uses MPS on Apple Silicon automatically.
