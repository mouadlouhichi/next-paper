# CAVI — Cooperative Action-Value Intelligence (Implementation)

Full code implementation of the CAVI framework proposed in `NEXT_PAPER_PROPOSAL.md`.
This is the forward-looking, uncertainty-adjusted, feasibility-restricted
cooperative-game framework for **actionable recommendation**: a single Shapley-
based object (the **Cooperative Action Value, CAV**) that explains, generates a
minimal-action recourse plan, and supports off-policy evaluation of whether the
plan actually works.

## Package layout

```
CAVI/
├── cavi/                     # the implementation package (numpy/scipy only, CPU)
│   ├── games.py              # cooperative games, Myerson feasibility, restricted game
│   ├── allocation.py         # CAV allocation, additivity theorem, component Shapley
│   ├── recourse.py           # budget-constrained minimal-action planner
│   ├── uncertainty.py        # ensemble variance game, ECE calibration
│   ├── ope.py                # IPS/DR/SNIPS + observation-propensity (Schnabel-style)
│   ├── recommender.py        # history-conditioned BPR profile recommender + dynamics
│   └── data.py               # MovieLens-1M loading & leakage-safe temporal split
├── scripts/
│   ├── run_synthetic_validation.py   # Paper A: ground-truth CAV recovery
│   └── run_ml1m_experiment.py        # full pipeline on real MovieLens-1M
├── tests/                    # 15 unit tests (theory + recourse + OPE)
├── gate/                     # the go/no-go divergence gate (already run: GREEN LIGHT)
└── results/                  # synthetic_validation.json, ml1m_experiment.json
```

## The core object (CAV)

```
u_t(S)  = E[V_t(S)] - kappa * Var[V_t(S)]        (mean-variance certainty-equiv. game)
u^F_t(S) = sum_{C in comp_F(S)} u_t(C)            (Myerson restricted game, feasibility F)
CAV_i   = Shapley_F(u_t)_i                        = Shapley_F(mean)_i - kappa*Shapley_F(var)_i
```

The second equality is the **additivity theorem** (proposal §3.5.4 Step 3): the
Shapley operator is additive and the restriction is linear, so the risk-adjusted
CAV equals the mean-Shapley minus kappa times the variance-Shapley. **This is
verified to machine precision in the code and tests.**

## What is implemented and validated

| Claim | Status |
|---|---|
| Exact Shapley recovers additive weights | ✅ unit test |
| Myerson component-efficiency (restricted game) | ✅ unit test |
| Myerson null-player property | ✅ unit test |
| **Additivity identity** `CAV = φ^μ − κφ^σ²` (exact) | ✅ unit test + real data |
| Additivity identity (Monte-Carlo) | ✅ unit test |
| Risk sensitivity (high-variance lever penalized) | ✅ unit test |
| Budget-constrained minimal-action greedy | ✅ unit test |
| Greedy-vs-exhaustive gap | ✅ unit test |
| Submodularity check | ✅ unit test |
| DR doubly-robust estimator | ✅ unit test |
| ECE calibration, ESS, discrepancy gate | ✅ unit test |
| Synthetic ground-truth recovery (5 families) | ✅ script |

**Synthetic validation (`run_synthetic_validation.py`)** confirms the theory on
games with known ground truth:
- Additive family: `max |CAV − true| = 8.9e-16` (machine precision).
- Complementary family: synergy captured (pair worth 5.0 split 2.5/2.5).
- Myerson family: component-efficiency holds exactly (each component pays its own value).
- Risk family: the high-variance lever is penalized as κ grows.

**Real-data experiment (`run_ml1m_experiment.py`)** runs the full CAVI pipeline on
MovieLens-1M: BPR item factors → history-conditioned profile recommender →
forward mean/variance value functions under a dynamics model → CAV allocation
(with additivity verified **exactly** on every user) → minimal-action recourse →
off-policy (IPS/DR) evaluation.

## Running

```
# 1. tests
python3 -m pytest tests/ -q          # 15 passed

# 2. synthetic theory validation
python3 scripts/run_synthetic_validation.py

# 3. real-data full experiment (needs gate/data/ml1m_*.dat)
python3 scripts/run_ml1m_experiment.py --users 50 --nmax 8 --seed 7
```

Requires: `numpy`, `scipy` (and `pytest` for tests). Pure CPU, no GPU.

## Key results (real ML-1M, seed 7)

```
users evaluated            : 31
additivity identity all OK : True          # theorem holds exactly on real data
mean plan size / cost      : 2.00 / 2.25   # minimal-action recourse is small
mean naive fwd lift        : 0.217         # model says the plan helps
mean DR-corrected lift     : 0.000         # OPE gate: cannot certify the plan
frac plans pass OPE gate   : 0.000
```

The most important honesty point: **the OPE discrepancy gate does its job.**
The naive model reports a positive forward lift, but the DR-corrected estimate
does **not** certify the plan — because in a synthetic offline log where the plan
coalition is rarely executed, there is insufficient matched evidence. This is
exactly the "a plan cannot be claimed to work because the model that predicted
it confirms it" gate from the proposal (§3.5.3-b). The DR *machinery* is
unit-tested for its doubly-robust property; a real OPE evaluation requires
proper logged data with the plan actually deployed, which is Paper B's task.

## Honest findings / open items

1. **Divergence magnitude is configuration-sensitive.** The original gate
   (≈0.09 Spearman between backward and forward orderings) used a different
   forward/backward operationalization than the full experiment (≈0.74). The
   **existence** of divergence is robust; its **magnitude** depends on exactly
   how "forward" is defined. Paper A must pin the value-function operationalization
   before making quantitative ordering claims — this is a real, reportable nuance,
   not hidden.
2. **Variance channel.** In the gate, risk-adjustment did not reorder levers
   (singleton proxy); the full variance-game Shapley should be used to conclude
   whether variance drives ordering.
3. **Single recommender family, one dataset.** BPR item factors + mean-pooling is
   a realistic but simple backbone; re-check on the actual Paper-A backbone and a
   second (rebuilt Amazon-Book) dataset.
4. **OPE needs real logs.** The offline OPE demonstration is a scaffold; the
   doubly-robust DR estimator is validated, but a deployable OPE result needs
   logged interaction data with known/estimated propensities.

## Relationship to the proposal

Implements Paper A (forward game + CAV + theory) and the core of Paper B
(recourse planner + OPE) with a single unified object. The closed-loop
action-consistency training objective (§3.5.6) is the one intentionally-not-yet-
implemented piece (flagged in the proposal as a separable Paper C); its circularity
controls (stop-gradient, decoupled updates) are specified but not coded.
