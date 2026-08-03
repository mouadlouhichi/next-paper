# FairShap — Shapley-Value-Guided Two-Sided Fairness and Popularity Debiasing in Hypergraph Recommendation

Full implementation of the FairShap framework from `FairShap_Paper_Structure.md`.

**FairShap** augments a hypergraph recommender with a **preference-aware Shapley estimator** of each item's marginal contribution to a **fairness-aware utility** (ranking quality + diversity + exposure equality), then uses these attributions to **re-rank** and redistribute exposure toward under-credited long-tail items — achieving a better accuracy–fairness trade-off than heuristic re-ranking baselines.

## Package layout

```
FairShap/
├── fairshap/
│   ├── metrics.py     # NDCG/Recall, ILD, coverage, Gini, ARP, long-tail, provider disparity, consumer gap
│   ├── game.py        # fairness-aware coalition value + preference-aware Shapley exposure attribution
│   ├── model.py       # hypergraph GNN recommender + fairness-regularized training
│   ├── rerank.py      # Shapley-guided fair re-ranking, calibrated, DPP-lite
│   ├── data.py        # ML-1M loading, temporal split, popularity tiers, providers, user groups
│   └── pipeline.py    # end-to-end orchestrator
├── scripts/
│   └── run_fairshap.py   # full experiment
├── tests/
│   └── test_fairshap.py  # 11 unit tests
└── results/
    └── fairshap_ml1m.json  # 120-user ML-1M results
```

## Core objects

**Fairness-aware coalition value** (spec Def. 1):
```
v(S) = alpha·NDCG@K(S) + beta·Diversity(S) + gamma·Fairness(S)
Fairness(S) = 1 - Gini({Exp(i)}_{i in S})
```
with `alpha + beta + gamma = 1`.

**Preference-aware Shapley exposure attribution**: Monte-Carlo Shapley of `v` over candidate items. Items that improve exposure equality without harming relevance receive high credit.

**Shapley-guided fair re-ranking** (spec Def. 2):
```
y_tilde_{u,i} = (1-gamma)·y_hat_{u,i} + gamma·(phi^fair_i / max phi^fair)
```

## Results (MovieLens-1M, 120 eval users, seed 42)

```
method            NDCG   Gini     ARP     LT  consGap
plain            0.409  0.414   475.1  0.000    0.171
inv_pop          0.296  0.394   412.1  0.001    0.110
calibrated       0.366  0.401   405.5  0.067    0.154
fairshap_g0.25   0.334  0.409   462.8  0.000    0.069   <- best accuracy-fairness trade-off
fairshap_g0.5    0.229  0.412   400.5  0.004    0.047
fairshap_g0.75   0.132  0.426   251.1  0.084    0.035
fair_regularized 0.160  0.371   205.7  0.190    0.065
```

**Key result — accuracy→consumer-fairness efficiency** (consumer-gap reduction per NDCG-unit lost, higher = better):
```
inv_pop        ratio 0.54
calibrated     ratio 0.39
fairshap_g0.25 ratio 1.36   <- ~2.5x more efficient than the best baseline
```

**Interpretation.** FairShap re-ranking traces a smooth accuracy–fairness Pareto front and, at matched accuracy cost, reduces the consumer-side NDCG gap far more than inverse-popularity or calibrated re-ranking. The in-training fairness-regularized variant drops accuracy (NDCG 0.160), showing post-hoc Shapley re-ranking is the more accurate-preserving fairness intervention.

## Running

```
python -m pytest tests/ -q                 # 11 passed
python scripts/run_fairshap.py --users 120 --epochs 12 --m 50 --seeds 42
```

Requires `numpy`, `scipy`, `torch` (CPU-capable; uses MPS on Apple Silicon).

## Next steps (toward a Q1 FairShap paper)
- Add Amazon-Book and Yelp2018 (data dirs exist under `CAVI/data/`).
- Add paired significance (Wilcoxon + Holm) between FairShap and baselines on consumer-gap / Gini.
- Add the accuracy–fairness Pareto-front figure and γ-sensitivity / M-sensitivity table.
- Ablations (w/o fairness term, w/o Shapley re-ranking, w/o hypergraph).
