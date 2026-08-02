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
python3 -m pytest tests/ -q          # 20 passed

# 2. synthetic theory validation
python3 scripts/run_synthetic_validation.py

# 3. real-data full experiment (needs gate/data/ml1m_*.dat)
python3 scripts/run_ml1m_experiment.py --users 50 --nmax 8 --seed 7

# 4. cross-dataset experiments (Amazon-Book, Yelp2018)
python3 scripts/run_cross_dataset.py --dataset amazon-book --users 40 --seed 7
python3 scripts/run_cross_dataset.py --dataset yelp2018 --users 40 --seed 7

# 5. interactive walkthrough notebook (loads all 3 datasets)
jupyter notebook CAVI_walkthrough.ipynb
```

Requires: `numpy`, `scipy` (and `pytest` for tests). Pure CPU, no GPU.

## Datasets

| Dataset | Format | Files | Metadata |
|---|---|---|---|
| **MovieLens-1M** | tab-separated ratings/items | `gate/data/ml1m_ratings.dat`, `ml1m_items.dat` | timestamps + genres ✅ |
| **Amazon-Book** | LightGCN split | `data/amazon-book/{train,test,user_list,item_list}.txt` | **none** (remapped ids) |
| **Yelp2018** | LightGCN split | `data/yelp2018/{train,test,user_list,item_list}.txt` | **none** (remapped ids) |

The Amazon-Book and Yelp2018 splits are the canonical LightGCN/HCCF/HPCF/DyHuCoG
format. **Caveat (proposal SignalShap §4.1):** these canonical splits carry **no
timestamps and no item metadata**, so per-user sequences come from the train-split
order, the held-out future targets from the test split, and *feasibility* uses a
**popularity-based anchor** (top-decile-popular items are immovable) instead of the
genre-anchor used for ML-1M. The proposal recommends rebuilding Amazon-Book from
the raw Amazon Reviews 2018 corpus to recover timestamps/metadata — the loaders
here support the canonical shared split and document the limitation.

### Cross-dataset results (ML-1M-backed pipeline, seed 7)

```
Amazon-Book : users=14, additivity_all_ok=True, rho(back,forward)=0.68, 25% divergent
Yelp2018    : users=17, additivity_all_ok=True, rho(back,forward)=0.92,  0% divergent
```

The **additivity theorem holds exactly on every user** across all three datasets.
Forward-vs-backward divergence is dataset-dependent (stronger on the sparse
Amazon-Book than Yelp2018), consistent with the Paper-A finding that the forward
game is a meaningful but not radical correction to backward Shapley.

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

1. **Divergence magnitude is configuration-sensitive — now resolved by the Paper-A study.** The original gate (≈0.09) vs full experiment (≈0.74) differed because of operationalization *and* a near-degenerate forward value under the weak recommender. `run_paperA_divergence.py` uses a stronger BPR recommender and matches all settings while sweeping only the operationalization, and measures the **full variance-game Shapley** (not the singleton proxy). Results (ML-1M, seed 7, 17 users):

   ```
   forward value signal: mean coalition-range = 0.114, frac users with signal>1e-3 = 0.412
   config           rho(B,F)   rho(mean,var)   rho(risk-reorder)  frac-reorder
   H1-fullfut          0.473          0.147              0.956         0.000
   H3-fullfut          0.331          0.370              0.991         0.000
   H3-nextonly         0.119          0.370              0.991         0.000
   ```

   **Key takeaways (honest, Paper-A-shaping):**
   - **Forward divergence is real but moderate, not dramatic.** ρ(B,F) ∈ [0.12, 0.47] across operationalizations — the forward ordering is *related to but not identical with* the backward ordering. The earlier "≈0.09" and "≈0.74" extremes were artifacts (degenerate value + unmatched settings). This tempers the novelty claim: the forward game is a *meaningful correction*, not a radical departure.
   - **Variance carries independent information.** ρ(mean,var) is low (0.15–0.37), so the variance game is *not* redundant with the mean game — the risk channel is real.
   - **But risk-adjustment does not reorder at κ=0.5** (ρ(risk-reorder)≈0.96–0.99, frac-reorder≈0). The variance Shapley differs in *value* but not in *ordering* at this κ. A larger κ or a genuinely risk-relevant lever space is needed before claiming risk changes which actions to take. This is a *negative result* the Paper-A paper should report.
   - **Degeneracy caveat:** only 41% of users have measurable forward signal. On the other 59%, the forward value is ≈0 and the ordering is noise. Divergence claims must be reported *signal-filtered* (restricted to users with genuine forward signal), or they will be dominated by degenerate users.

2. **Single recommender family, one dataset.** BPR item factors + mean-pooling is
   a realistic but simple backbone; re-check on the actual Paper-A backbone and a
   second (rebuilt Amazon-Book) dataset. Amazon/grouplens hosts are TLS-blocked in
   this sandbox, so a second dataset is not reachable here.
3. **OPE needs real logs.** The offline OPE demonstration is a scaffold; the
   doubly-robust DR estimator is validated, but a deployable OPE result needs
   logged interaction data with known/estimated propensities.

## Relationship to the proposal

Implements Paper A (forward game + CAV + theory) and the core of Paper B
(recourse planner + OPE) with a single unified object. The closed-loop
action-consistency training objective (§3.5.6) is the one intentionally-not-yet-
implemented piece (flagged in the proposal as a separable Paper C); its circularity
controls (stop-gradient, decoupled updates) are specified but not coded.
