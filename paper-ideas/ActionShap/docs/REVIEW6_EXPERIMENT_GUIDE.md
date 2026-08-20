# Review-6 (Knowledge-Based Systems) experiment guide — runs on your machine

Paper-side responses (statistics, writing, restructure, references) are already
integrated on the branch. The items below require new compute.

## R6-1. LightGCN quality gate (CRITICAL item #1 — competitive model)

LightGCN is implemented (`actionshap/lightgcn.py`, inference-time history
weighting interface, exact linear propagation, BPR training; smoke-tested).

```bash
# quality gate, 5 seeds, cohort + full-corpus variants (mirrors the SASRec runs)
python scripts/run_review5_experiments.py sasrec-quality --model lightgcn --dataset movielens
python scripts/run_review5_experiments.py sasrec-quality --model lightgcn --dataset amazon
python scripts/run_review5_experiments.py sasrec-quality --model lightgcn --dataset movielens --train-all --epochs 30
python scripts/run_review5_experiments.py sasrec-quality --model lightgcn --dataset amazon    --train-all --epochs 30
```

If it passes the quality gate (beats popularity on NDCG@10 + masking gate),
run the full primary audit with it:

```bash
python scripts/run_review3_experiments.py --dataset movielens --model lightgcn --users 1000 --out results/review6
python scripts/run_review3_experiments.py --dataset amazon    --model lightgcn --users 1000 --out results/review6
```

## R6-2. Third dataset from a different domain (HIGH item #3)

Gowalla (location check-ins) prep is implemented (`scripts/prepare_gowalla.py`):

```bash
python scripts/prepare_gowalla.py --out data/gowalla/interactions.csv
# ItemKNN primary audit on the new domain
python scripts/run_review3_experiments.py --dataset gowalla --model itemknn --users 1000 --out results/review6
# (optional) LightGCN on the new domain
python scripts/run_review3_experiments.py --dataset gowalla --model lightgcn --users 1000 --out results/review6
```

Note: Gowalla has no absolute timestamps; the prep script preserves per-user
interaction order as synthetic timestamps (documented in the paper).

## R6-3. Qualitative case studies (MEDIUM item)

Pick 2–3 users where the bounded gap matters. Suggested selection: users with
high bounded AIA but low deletion AIA (or vice versa) under Shapley. Output
per user: profile items, top attributions, deletion vs bounded effects,
selected pair vs oracle pair, rank before/after. (Small run; reuse
run_review3 with --users on a hand-picked user list if needed — tell me which
users and I'll generate the case-study table.)

## R6-4. Gradient-based baseline in the primary comparison (HIGH item #5)

Already covered by existing results: finite-difference and integrated-gradient
attributions along the weight path were run for BOTH models in the review-3
replication; the paper now points to them from the methods section. No new run
required unless you want FD/IG in the primary (1,000-user) table — then:
extend run_review3_experiments to compute them in the primary loop (ask me).

## Push-back contract

Push result JSONs to `code/results/review6/` and ping me; I will validate,
generate tables, integrate, and re-verify.
