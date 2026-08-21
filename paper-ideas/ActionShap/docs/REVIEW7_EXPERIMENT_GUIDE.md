# Review-7 (KBS round 2) experiment guide — runs on your machine

Paper-side fixes (consistency reconciliation, masking-gate/player-floor
definitions, hypothesis adjudication + falsification analysis, success/
abstention paired tests, bibliography completion, KernelSHAP implementation)
are already integrated on the branch. Remaining items need compute.

## R7-1. KernelSHAP baseline (mandatory #6 — IMPLEMENTED, run it)

```bash
python scripts/run_review5_experiments.py kernelschap --dataset movielens --users 1000 --ks-samples 256 512
python scripts/run_review5_experiments.py kernelschap --dataset amazon    --users 1000 --ks-samples 256 512
```
Writes results/review5/kernelschap_<dataset>.json (per-user KernelSHAP vs MC
Shapley bounded AIA at 256/512 masks — comparable to the LIME design size).

## R7-2. Exact-Shapley validation at n=20 (mandatory #5)

Exact enumeration at n=20 needs ~1M coalition evaluations per user; use a
stratified subsample with batched scoring. Suggested: 50 ML-1M users at
n_u=20 + 50 Amazon users across n_u strata. Report per-user MC SEs and
bias bounds. (If too heavy, an independent M_pair=5000 reference for the same
users is an acceptable fallback — say which you ran.)

## R7-3. Amazon rho=0.25 cell + exhaustive B=3 (mandatory #8)

```bash
# rho=0.25 sensitivity on Amazon (mirrors the MovieLens cell in C.1)
# (reuse the sensitivity run configuration with --rho 0.25 on Amazon)
# exhaustive B=3 oracle (1,351 actions/user — computationally trivial)
# set the B=3 job to exhaustive enumeration instead of greedy lower bound
```

## R7-4. CEERS-style head-to-head (mandatory #6 second part)

Compute at least one published counterfactual-evaluation metric (CEERS,
Baklanov et al. 2025 fidelity metrics, or Yao et al. 2022) on the identical
1,000 primary users, and report whether it orders the five methods
differently from the ActionShap decomposition. If implementing the exact
published metric is out of scope, run the deletion-only evaluation protocol
(deletion AIA as the single metric) and show its method ordering vs the
three-axis ActionShap view (data already exists — I can generate this
comparison; tell me if you want it done without a new run).

## R7-5. Worked user-level example (mandatory #10)

Pick 2–3 users (suggest: one high-gap Shapley user, one abstention user).
Output: profile items, top attributions, deletion vs bounded effects,
selected pair vs oracle pair, rank before/after. Small run — tell me which
users and I'll generate the case-study table.

## Push-back contract

Push result JSONs to code/results/review5/ (or results/review7/) and ping me.
