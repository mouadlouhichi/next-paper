# ShapAct — Construction-Level Audit of Exact Shapley Source Attribution

This directory contains the **complete, reproducible implementation** and the **manuscript** for
*ShapAct: A Construction-Level Audit of Exact Shapley Source Attribution for Actionable
Decisions in Hybrid Recommenders*.

```
paper-ideas/ShapAct/
├── ShapAct_Paper_Structure.md        # paper blueprint (positioning, RQs, falsification)
├── ShapAct_Implementation_Spec.md    # implementation spec + REGISTERED PREDICTIONS
│                                     #   + Part C: predictions vs. actual outcomes
├── paper.tex                         # manuscript (Discover AI / Springer sn-jnl, IJACSA layout)
├── paper.bib                         # all references
├── code/
│   ├── shapact/                      # the implementation
│   │   ├── config.py                 # dataset/hyperparameter config
│   │   ├── data.py                   # loaders, iterative 5-core, temporal leave-one-out split
│   │   ├── sources.py                # the five signal sources (CF, CB, POP, REC, SEQ)
│   │   ├── fusion.py                 # candidates, z-normalization, pairwise-logistic fusion
│   │   ├── game.py                   # exact 5-player Shapley game (32 coalitions), per-user
│   │   ├── counterfactuals.py        # L0 / L1 / L2 worlds
│   │   ├── audit.py                  # fidelity gap, order validity, reflexivity
│   │   ├── decisions.py              # Shapley / LOO / Feature-Shapley / Random rules
│   │   ├── metrics.py                # NDCG@k, Recall@k, MRR, coverage
│   │   ├── stats.py                  # paired t, Holm-Bonferroni, Wilcoxon, Cohen's d_z
│   │   └── pipeline.py               # end-to-end orchestration + validation block
│   ├── scripts/
│   │   ├── run_audit.py              # run one/all datasets, one or five seeds
│   │   └── run_all.py                # audits + significance tests
│   ├── tests/                        # pytest suite (unit + integration invariants)
│   ├── data/raw/                     # MovieLens-1M, LastFM-2K (see below)
│   ├── results/raw/                  # audit_*.json (all seeds) + audit_*_mean.json
│   ├── results/tables/               # significance_*.json
│   └── requirements.txt
```

## Reproduce

```bash
pip install -r requirements.txt
cd code
python scripts/run_all.py          # audits (5 seeds) + significance for both datasets
python -m pytest tests/ -q        # invariant tests (fast, synthetic)
```

Raw data (download once, place in `code/data/raw/`):

- **MovieLens-1M** — `ml1m_ratings.dat`, `ml1m_items.dat`, `ml1m_users.dat`
  (GroupLens; the mirror used for this run normalizes `::` to tabs, which the loader handles).
- **LastFM-2K (HetRec 2011)** — `lastfm_user_artists.dat`, `lastfm_artists.dat`,
  `lastfm_user_taggedartists.dat`, `lastfm_user_taggedartists-timestamps.dat`.

The pipeline is dataset-agnostic; Amazon-Book (raw Amazon Reviews 2018) is the intended
second benchmark per the SignalShap protocol, but its corpus was not reachable from the
compute environment used for this run, so LastFM-2K was substituted (documented in the
manuscript, Appendix A4).

## Protocol (summary)

- Implicit positives: rating >= 4 (ML-1M); any listening event (LastFM).
- Iterative 5-core filtering to a fixed point; per-user temporal split
  (last = test, second-to-last = validation, rest = train; ties broken by row order).
- Candidate set: union of each source's top-500 lists, truncated to 500 by best
  cross-source rank; recall ceiling reported.
- Fusion: regularized pairwise logistic ranker trained on **validation** pairs
  (disclosed instantiation decision — training pairs were measured to halve fused NDCG@10).
- Game: exact Shapley over 2^5 = 32 coalitions; v(C) = NDCG@10(fusion_C) − NDCG@10(null).
- Audit: L0 masked / L1 regenerated / L2 never-built counterfactuals; fidelity gap F,
  order validity (Kendall τ, top-k), reflexivity ρ, four decision rules under L2.
- 5 seeds {42..46}; all invariants verified to machine precision.

## Results (headline, 5 seeds, mean ± std)

| Quantity | MovieLens-1M | LastFM-2K |
|---|---|---|
| Recall@500 | 0.7187 ± 0.0002 | 0.6025 ± 0.0009 |
| Uplift v(G) | 0.0678 ± 0.0006 | 0.0724 ± 0.0030 |
| Fidelity gap \|F_g\| | ≤ 0.001 | ≤ 0.002 |
| Kendall τ (credit vs realized) | 0.60 | 0.44 ± 0.08 |
| Reflexivity ρ | 0.09–0.27 | 0.18–0.57 |
| Shapley rule realized NDCG@10 | 0.0676 | 0.0768 |
| Random rule realized NDCG@10 | 0.0586 | 0.0680 |
| Shapley vs Random (Holm p) | 3.7e-19 | 2.0e-05 |

Full per-seed numbers in `code/results/raw/`; the registered predictions and their
outcomes (including the falsified hypotheses) are in `ShapAct_Implementation_Spec.md` Part C.

## Validation against published Q1 numbers

The paper reports full-catalog NDCG@20 on MovieLens-1M for calibration against published
baselines (MF 0.120, NCF 0.130, LightGCN 0.213, HCCF 0.247, HPCF 0.253, DyHuCoG 0.278 —
from the group's DyHuCoG paper). The protocol difference (temporal leave-one-out vs random
splits) makes absolute values lower; the relative ordering and the internal invariants are
the validation anchors, together with the dataset-stat check (574,376 interactions after
`rating>3` + iterative 5-core, matching the value independently predicted in the DyHuCoG
extraction audit).
