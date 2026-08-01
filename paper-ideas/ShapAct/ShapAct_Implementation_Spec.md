# ShapAct — Technical Implementation Specification and Registered Predictions

**Companion to:** `ShapAct_Paper_Structure.md` (the paper blueprint). That file says *what the paper argues*; this file says *what to build and what to expect when it runs*.
**Status:** pre-implementation. Every number in Part B is a **prediction made before running anything**, not a result.
**Dependency:** ShapAct is built **on the SignalShap codebase**, after its gates (SignalShap Implementation Spec §B.7, milestones 1–4) have passed, and **on the ActionShap codebase now in the repository** (`paper-ideas/ActionShap/code/actionshap/`): `stats.py`, `metrics.py` (AIA / top-$k$ precision / regret — adapt to retirement outcomes), the `attribution.py` Attributor interface, and the freeze-hash pattern from `modifiability.py` for pre-registering the L0/L1/L2 protocols. Nothing from DyHuCoG is used — this is now load-bearing twice over: the full 63-gap extraction audit in `paper-ideas/ActionShap/code/docs/dyhucog_spec.md` concludes the DyHuCoG paper cannot be faithfully reimplemented, and ActionShap's own dynamic regime (which planned to reuse the DyHuCoG codebase) has no implemented dynamic code in the repository. ShapAct's exact five-source host is the only recsys intervention pipeline in the group that is both exact and reproducible.

---

# PART A — IMPLEMENTATION

## A.1 Repository layout

ShapAct lives in the SignalShap repository, in a sibling package, and reuses two existing packages:

```
SignalShap/code/
├── signalshap/               # unchanged from SignalShap (data, sources, normalize, fusion, game, segments, adaptive, baselines, metrics, report)
├── shapact/
│   ├── __init__.py
│   ├── counterfactuals.py    # L0/L1/L2 execution: the counterfactual ladder
│   ├── audit.py              # P_g, R_g, F_g, fidelity decomposition (Prop. 1)
│   ├── order.py              # Kendall tau, top-k agreement, co-monotonicity violations (Prop. 2)
│   ├── reflexivity.py        # post-intervention exact game, rho_g, aggregate identity (Prop. 3)
│   ├── decisions.py          # the four decision rules + L2 realized evaluation
│   ├── variants.py           # L2 full-retraining (top-1), cost-adjusted rules, N sensitivity
│   ├── stats.py              # IMPORT from paper-ideas/ActionShap/code/actionshap/stats.py (do not vendor a second copy)
│   └── report.py             # LaTeX table + figure emitters
├── scripts/
│   ├── run_audit_l1.py       # L1 regeneration harness (stage 2)
│   ├── run_audit_l2.py       # L2 never-built + fidelity (stages 3–4)
│   ├── run_reflexivity.py    # post-intervention game (stage 6)
│   ├── run_decisions.py      # decision-rule executor (stage 7)
│   └── run_all.py
├── tests/
│   └── test_shapact.py
└── results/{raw,tables,figures}/
```

Reused without modification: `paper-ideas/ActionShap/code/actionshap/{stats,metrics,attribution,modifiability}.py` (import; never vendor a second copy). The ActionShap `metrics.py` alignment/top-$k$/regret functions are adapted at the call site to retirement outcomes (ordering by $R_g$ instead of by $\Delta_j$); do not modify the upstream module. The freeze-hash pattern from `modifiability.py` is reused to pre-register the L0/L1/L2 protocol YAML (ShapAct needs no modifiability elicitation — every source is modifiable by construction — but the audit-trail discipline carries over).

## A.2 Environment

Identical to SignalShap (Python 3.12; numpy 2.4–2.5; scipy ≥ 1.18; scikit-learn ≥ 1.6; pandas ≥ 2.2; implicit ≥ 0.7.2; matplotlib; pyyaml; tqdm; pytest). No GPU, no PyTorch — the paper's laptop-CPU claim must stay true. The only new computational step that touches trained models is the §A.6 full-retraining variant, and it is restricted to one source per dataset.

## A.3 Data layer

**Reuse SignalShap's `data.py` outputs verbatim** — the frozen splits, the subsampled/5-cored/iterative-filtered datasets, and the temporal leave-one-out protocol (last interaction test, penultimate validation). Do not re-derive splits; read from the cached Parquet keyed by the SignalShap config hash. Key facts to keep in mind:

- MovieLens-1M: 6,040 users / ~3,700 items / 1,000,209 interactions, dense (~4.5%).
- Amazon-Book: **rebuilt** from raw Amazon Reviews 2018 Books, user-subsampled to ~50k, iterative 5-core, day-resolution timestamps with deterministic tie-breaking. Counts will not match the canonical 52,643/91,599/2,984,108 (used in DyHuCoG/thesis) — this is disclosed, not a bug.
- Implicit conversion: rating ≥ 4 (coincides with "rating > 3" on integer ratings — note it once).

## A.4 Candidate generation

**Reuse SignalShap's `candidates.py`** (union of source top-lists, round-robin truncation to N=200, recall ceiling diagnostic). The audit adds one new operation: **regeneration without a source** (L1). For each user: take the top-$N_g$ lists of the remaining four sources, union, round-robin to $N=200$. Diagnostics to record: union size before truncation (does removing $g$ shrink the pool?), and the L0≡L1 no-op check (see §A.9 test 8).

## A.5 The five source scorers

**Reuse SignalShap's `sources.py` unchanged** (BPR-MF 64 factors; TF-IDF content similarity; log-popularity; time-decayed popularity with 30-day half-life; first-order Markov). All fitted on train only, cached to `.npy`. The audit adds **no new scorers**. The never-built condition for source $g$ simply never calls $g$'s scorer — it is not trained, not scored, and its cached matrix is not read.

## A.6 Normalization, fusion, and the game

**Reuse `normalize.py`, `fusion.py`, `game.py` unchanged.** The characteristic function, for every coalition $C \subseteq \mathcal{G}$:

```
v(C) = ndcg_at_10(rank_by(fusion_C)) - ndcg_at_10(null_ranker)
```

with `f_theta[()] = null_ranker` (so v(∅)=0), z-normalization per user per source with the sigma=0 fallback, pairwise-logistic fusion with drop-not-zero column masking, and exact Shapley over all 2^5 = 32 coalitions. Per-user values come free by linearity (SignalShap Prop. 3) — the audit does not need them for its primary results but they are used for the per-user significance tests in §A.11.

## A.7 The counterfactual ladder (`counterfactuals.py`)

Three levels, executed per source $g$:

| Level | Implementation | Output |
|---|---|---|
| **L0 — masked** | Already in SignalShap's game: coalition value of $\mathcal{G}\setminus\{g\}$ with fixed scorers and fixed candidates | $P_g = v(\mathcal{G}) - v(\mathcal{G}\setminus\{g\})$ |
| **L1 — regenerated** | Rebuild candidate sets from the four remaining sources (round-robin, N=200); **re-score candidates with the same trained scorers** (note: score columns for the regenerated candidate set must be recomputed from the cached scorers — this is the L1 cost); refit fusion; evaluate | $P^{\text{reg}}_g = v^{\text{reg}}(\mathcal{G}) - v^{\text{reg}}(\mathcal{G}\setminus\{g\})$ (compute $v^{\text{reg}}(\mathcal{G})$ too: regenerated candidates can change the grand coalition's value) |
| **L2 — never-built** | Same as L1 but with $g$'s scorer never trained and its scores never present; the four surviving scorers as trained | $R_g = v^{\text{nb}}_{-g}(\mathcal{G}\setminus\{g\}) - v(\mathcal{G})$ |

Implementation requirements:

1. **L0 first, always.** Verify the SignalShap coalition table reproduces (tests 1–7 of SignalShap's spec still pass) before any L1/L2 run.
2. **L1 regeneration must be a pure function of the remaining sources' top lists** — no randomness except the fixed seeds; store the regenerated candidate sets to disk keyed by (dataset, retired source, config hash).
3. **L2 bookkeeping:** the never-built game for $g$ uses the *grand-coalition* candidate set of the four-source world for every coalition evaluation (the four-source world's candidates, since $g$ never contributed any), and the fusion is refit over the four score columns. Note that in the L2 world the "grand coalition" *is* $\mathcal{G}\setminus\{g\}$ — the L2 game is a four-player game, which is exact and cheap (16 coalitions).
4. **Fidelity decomposition (Prop. 1) as an invariant:** compute $F_g$ both as $P_g - R_g$ and as $v^{\text{nb}}_{-g}(\mathcal{G}\setminus\{g\}) - v(\mathcal{G}\setminus\{g\})$; assert equality to machine precision. This single assertion catches most L1/L2 bookkeeping errors.
5. Report the L0/L1/L2 marginal matrix (Fig. 7 / Appendix B) in full — 5 sources × 3 levels × 2 datasets.

## A.8 The audit module (`audit.py`)

Computes, per source per dataset: $P_g$ (L0), $P^{\text{reg}}_g$ (L1), $R_g$ (L2), $F_g$, and the candidate-set effect $P_g - P^{\text{reg}}_g$ vs. the retraining effect $P^{\text{reg}}_g - R_g$ (both components of $F_g$). Emits the RQ1 table.

**Exact vs. sampled Shapley — the decision, stated once.** **Exact.** Player set $K=5$ ⇒ $2^5=32$ coalition evaluations for the L0 game and $2^4=16$ for each L2 world, each a sub-second convex fusion refit over cached columns. We do not use Monte-Carlo estimation, and the paper must say why in one paragraph: (a) exactness makes every fidelity gap attributable to *intervention semantics* rather than estimator noise — the audit's core inference requires point-identifiable predictions; (b) SignalShap already established the exact design's cost and vetting, so approximating it would forfeit the strongest reviewer defense for no benefit; (c) the MC route (DyHuCoG's $M=50$, MSE $\approx 1.4\times10^{-5}$ precedent) would force a convergence table into a paper whose entire point is that discrepancies are real. The degradation condition to state honestly: if a system fused ~20+ sources, exactness fails and the audit would need the MC estimator with a reported convergence table (DyHuCoG Table 7 as the protocol precedent); K=5 is the design choice that keeps this paper exact.

## A.9 Test suite (`tests/test_shapact.py`)

Non-negotiable, in priority order. Tests 1–7 are SignalShap's, run unchanged (they guard the inputs the audit consumes):

1. **Efficiency identity** (SignalShap): $\sum_g \varphi_g = v(\mathcal{G})$ to machine precision.
2. **Per-user consistency** (SignalShap): $\frac{1}{|\mathcal{U}|}\sum_u \varphi_g(u) = \varphi_g$.
3. **Null ranker calibration** (SignalShap): empirical NDCG@10 of $\pi$ matches the analytic value times recall.
4. **Empty coalition** (SignalShap): $v(\emptyset) = 0$ exactly.
5. **Symmetry on synthetic data** (SignalShap): identical score columns ⇒ equal Shapley, LOO zero.
6. **Degenerate normalization** (SignalShap): constant row ⇒ all-zero column, no NaN.
7. **Dummy source** (SignalShap): pure-noise column ⇒ $\varphi \approx 0$.

New audit tests (the ones that would catch a wrong paper):

8. **L0≡L1 no-op check.** On a synthetic dataset where the retired source contributes zero unique candidates for every user, the L1 pipeline must reproduce the L0 coalition values exactly. Validates the regeneration harness itself.
9. **Fidelity decomposition invariant.** $F_g = P_g - R_g = v^{\text{nb}}_{-g}(\mathcal{G}\setminus\{g\}) - v(\mathcal{G}\setminus\{g\})$ to machine precision, for every source and dataset (Prop. 1 as an executable identity).
10. **Reflexivity aggregate identity.** After retiring $g^*$: $\sum_{h \neq g^*} \varphi_h^{\text{nb}} = v(\mathcal{G}) - R_{g^*}$ to machine precision (Prop. 3 as an executable identity).
11. **Synthetic decision test (Prop. 2 analog).** Build a synthetic five-source game with one engineered perfectly-redundant pair (identical score columns). Assert: the LOO rule and the Shapley rule recommend *different* retirements (LOO retires one member of the pair on a coin flip of ties; Shapley never singles out either member), and the realized loss under the Shapley rule is ≤ the realized loss under the LOO rule, analytically verified in the synthetic game. This is the decision-level companion to SignalShap's test 5.

Tests 8, 9, 10 are the ones that would catch a wrong audit rather than a crashed run; test 11 is the one that would catch a wrong paper.

## A.10 Runtime budget (CPU, single laptop; additions over SignalShap)

| Stage | ML-1M | Amazon-Book (50k sample) |
|---|---|---|
| SignalShap full pipeline, one seed | ~15 min | ~40 min |
| L1 regeneration + re-scoring, all 5 sources | 5–10 min | 15–25 min |
| L2 never-built, all 5 sources (4-player games, 16 coalitions each) | 3–5 min | 8–15 min |
| Reflexivity (5 more exact games on intervened systems) | 5–10 min | 10–20 min |
| Decision rules + per-user significance | 3–5 min | 5–10 min |
| L2 full-retraining variant (top-1: retrain surviving CF + regenerate + refit) | +5–10 min | +10–20 min |
| **Total added by ShapAct, one seed** | **~25–40 min** | **~50–90 min** |
| Five seeds, both datasets, all audit stages | ~1.5–2 days (parallelizable per dataset×seed) | |

The full-retraining variant is the only stage that trains a base scorer; everything else is candidate regeneration and fusion refits. If any stage runs an order of magnitude over these, the likely bug is dense full-matrix work instead of the candidate slice — same failure mode SignalShap warns about.

## A.11 Statistics and reporting

Use the ported `stats.py` (paired t-test, Holm–Bonferroni, Wilcoxon signed-rank, Cohen's $d_z$), the house protocol (DyHuCoG Appendix A / thesis Chapter 4): per-user realized NDCG@10 under each decision rule, paired over users, Holm across the rule family, effect sizes, 95% CIs, five seeds {42..46}, mean ± std. Emitters produce the LaTeX tables for §4.3–§4.6 and Appendix B/E.

---

# PART B — REGISTERED PREDICTIONS

> **Read this as a pre-registration.** Everything below is what we expect *before* running anything. Report real numbers against this table and **flag every miss explicitly** — a missed prediction that is discussed is a strength; a quietly revised prediction is misconduct. The house precedent is the SignalShap spec Part B; same rules apply.

## B.1 Pipeline-level quantities (audit additions; L0 quantities inherited from SignalShap)

| Quantity | ML-1M | Amazon-Book | Confidence |
|---|---|---|---|
| L1 regenerated-union shrinkage (mean candidates lost per user when the least-contributing source is removed) | < 5% | < 10% | medium |
| $v^{\text{reg}}(\mathcal{G}) - v(\mathcal{G})$ (does regeneration move the grand coalition?) | ±0.005 | ±0.01 | low |
| L2 grand-coalition NDCG@10 after retiring the lowest-credit source | 0.30–0.40 | 0.10–0.20 | low (mirrors SignalShap's fusion range) |
| Reflexivity $\rho_{g^*}$ (mean relative credit shift of survivors) | 0.05–0.20 | 0.10–0.30 | medium |

Confidence on Amazon-Book is deliberately lower: the rebuilt split has no published counterpart (same caveat as SignalShap §B.1).

## B.2 RQ1 — intervention fidelity (the headline result)

| Source | Predicted $F_g$ on ML-1M | Predicted $F_g$ on Amazon-Book | Reasoning |
|---|---|---|---|
| **POP ↔ REC** | **$F > 0$, largest of the five** | **$F > 0$, largest of the five** | Engineered redundancy (SignalShap spec §A.5): never-building one member leaves the other to carry the popularity load, so the realized loss is smaller than the masked marginal predicts — the decision-level face of SignalShap Prop. 2 |
| **CB** | $F \approx 0$ (small) | **$F < 0$** | On the sparse catalogue with rich metadata, content has no substitute: the never-built world loses *more* than masking predicts, because candidates that only CB could retrieve are gone |
| CF | $F \gtrsim 0$ (small positive) | $F \gtrsim 0$ | Partial substitution via POP's popularity leakage into CF scores; smaller than the POP/REC gap |
| SEQ | $F \approx 0$ | $F \approx 0$ | Orthogonal signal; no close substitute, no adaptation |

**The prediction that carries the paper:** the *sign structure* — positive fidelity for substitutable sources, negative for the un-substitutable one. If all $F_g$ come out ≈ 0, the audit's finding is the clean one that masking is a faithful proxy and SignalShap's attribution is decision-valid as-is; that is a legitimate positive result, and the paper reframes accordingly (see B.6). If $F_g < 0$ appears on ML-1M's POP/REC (where we predict $>0$), the redundancy story is wrong on the dense dataset and RQ2's framing must soften.

## B.3 RQ2 — order validity of the credit ranking

| Quantity | ML-1M | Amazon-Book | Confidence |
|---|---|---|---|
| Kendall $\tau$ between $\varphi_g$-order and $R_g$-order | 0.6–0.9 | 0.4–0.7 | medium |
| Top-1 agreement (lowest credit = best retirement) | yes | yes | medium-high |
| Top-2 agreement | yes | yes, unless CB anomaly ranks top-2 | low |
| Co-monotonicity violations (Prop. 2) | POP↔REC pair, if any | CB vs. everyone | medium |

Prediction: the ordering is decision-valid at the top (the decision-maker retires the right source) and degrades near the bottom, where null-player sources have near-zero credit and near-zero realized effect. The interesting failure mode to watch: on Amazon-Book, CB may be *mid-credit* under L0 (substitutable-looking under masking) but *largest realized loss* under L2 — the one inversion that makes the audit worth having.

## B.4 RQ3 — post-intervention reflexivity

Retire the lowest-credit source $g^*$ per dataset; predict:

- The **largest relative credit shift among survivors lands on the source most redundant with $g^*$** (POP if REC retired, REC if POP retired), with $\rho$ concentrated rather than diffuse.
- Aggregate identity $\sum_{h\neq g^*}\varphi_h^{\text{nb}} = v(\mathcal{G}) - R_{g^*}$ holds by construction (test 10); the informative quantity is the *distribution*.
- $\rho_{g^*}$ within 0.05–0.20 (ML-1M) / 0.10–0.30 (Amazon-Book).

If $\rho$ is large and diffuse (say > 0.5 everywhere), the explanation is self-defeating: acting on it destroys the description's validity. Report that outcome plainly — it is a substantive finding about explanation fragility, and it is exactly the failure the paper exists to detect.

## B.5 RQ4 — realized quality under decision rules

| Rule | Predicted realized NDCG@10 ordering | Reasoning |
|---|---|---|
| Shapley | best | Splits redundant credit correctly (SignalShap Prop. 2) ⇒ retires a genuinely low-value source |
| LOO | second | Collapses on POP↔REC ⇒ may retire one member of a load-bearing pair (near-zero LOO) and lose real quality |
| Feature-SHAP | third | Answers a feature-level question (which normalized score matters) as a source-level decision; expected to be noisier than either source-level rule |
| Random | worst | Floor |

Predicted gap: Shapley over LOO concentrated on the POP↔REC decision — expected relative NDCG@10 difference between the two rules of +1% to +4% on ML-1M, larger on Amazon-Book (where the CB anomaly may make LOO actively destructive). **If the Shapley rule does not win, the decision-level claim softens to "exact attribution is decision-competitive," and the paper's Contribution 4 is re-scoped accordingly (B.6).**

## B.6 Falsification and contingencies

| If this happens | What it means | Contingency |
|---|---|---|
| All $F_g \approx 0$ | Masking is a faithful proxy; SignalShap's attribution is decision-valid as-is | Reframe as a *validation* result: the audit certifies the exact game's actionability; keep RQ1–RQ4, drop the "attribution misleads" framing from §1.2 |
| $F_g > 0$ for POP/REC fails to appear | Redundancy does not produce substitutability at the construction level | Report honestly; the RQ1 sign-structure claim weakens to CB-only; §5.1 must not overclaim |
| CB $F_g \geq 0$ on Amazon-Book | No un-substitutable source in the system | CB anomaly dropped; the two-dataset contrast story thins, paper still stands on RQ2–RQ4 |
| Shapley rule loses to LOO on realized NDCG@10 | Exact attribution is not decision-optimal in this system | Report; Contribution 4 becomes "decision-competitive, not decision-optimal"; do not re-run seeds until it wins |
| $\rho$ large and diffuse (> 0.5) | Explanations are self-defeating | Report as a headline *negative* finding — it is the paper's most interesting possible outcome and should be led with, not hidden |
| Decomposition invariant (test 9) fails | Bookkeeping bug | Stop; do not interpret anything until it passes |
| L1 regeneration collapses the candidate pool (< 30% of users keep ≥ 100 candidates) | Round-robin truncation too aggressive after source removal | Lower N to 150 for L1/L2 runs, report, and do not switch back afterwards |

The two outcomes that would genuinely wound the paper are all-zero fidelity gaps (dulls RQ1 into a validation paper — survivable) and Shapley losing the decision comparison (re-scopes Contribution 4 — survivable). Neither is fatal; the framing adjustments above are pre-committed so they happen without improvising under deadline pressure.

## B.7 Suggested milestone order

0. **Confirm the two upstream gates.** (a) SignalShap gates 1–4 passed (its spec §B.7): data/splits, candidate recall > 0.5, scorers cached with sane degenerate rates, coalition sweep with tests 1–7 green. (b) ActionShap's `code/actionshap` pytest suite passes and its `stats.py`/`metrics.py` import cleanly under the pinned environment (its own `requirements.txt`: numpy 2.4–2.5, scikit-learn 1.5.1, lightgbm 4.5.0, shap ≥ 0.52, interpret-core 0.6.3, krippendorff 0.8.0; note shap < 0.48 breaks against numpy 2 — keep the pins when importing). **Gate: all green; otherwise stop and finish the upstream papers first.** Also record: ActionShap's `results/raw/wine_static.json` is a provisional smoke test (its annotation header says so) and must never be cited as a result.
1. L1 regeneration harness + re-scoring. **Gate: test 8 (L0≡L1 no-op) passes on the synthetic case; union shrinkage diagnostic recorded.**
2. L2 never-built for all five sources. **Gate: test 9 (fidelity decomposition invariant) passes for all sources on both datasets.**
3. RQ1 numbers. **Gate: does the sign structure of B.2 appear? Run before writing any prose.**
4. Order-validity statistics (test: none; report $\tau$, top-k, violations).
5. Reflexivity (exact game on intervened systems). **Gate: test 10 aggregate identity passes; is $\rho$ concentrated or diffuse?**
6. Decision rules + per-user significance. **Gate: does the Shapley rule win (B.5)? If not, re-frame per B.6 before running the full-retraining variant.**
7. Full-retraining variant (top-1), cost-adjusted results, emitters, full draft.

Steps 3, 5, and 6 are the decision points. If the sign structure holds, $\rho$ is concentrated, and the Shapley rule wins, the paper is essentially written. If any fails, the contingency table says what to do.

---

# PART C — REGISTERED PREDICTIONS vs. ACTUAL OUTCOMES (added 2026-08-01, after the runs)

Every prediction in Part B was made before running anything. This part records the actual
outcomes and flags every miss, per the falsification protocol.

## C.1 Pipeline-level quantities

| Quantity | Predicted | Actual (mean±std, 5 seeds) | Verdict |
|---|---|---|---|
| Recall@500 ML-1M | 0.85–0.95 (at N=200) | 0.7187±0.0002 | **MISS** — the temporal-LOO protocol retrieves the last-interaction item less often than anticipated; N raised to 500 per the spec's own contingency; union hit rate 0.93 reported as the diagnostic |
| Recall@500 LastFM | (not pre-registered) | 0.6025±0.0009 | — |
| Null NDCG@10 (analytic 4.5436/N·recall) | ~0.0227·recall (N=200) | 0.0058 (ML-1M), 0.0042 (LastFM) at N=500; matches 4.5436/500·recall within Monte-Carlo error | ✓ (N=500) |
| Fusion NDCG@10 ML-1M | 0.30–0.40 | 0.0742 | **MISS** — pre-registered range assumed a random-split-style protocol; under temporal LOO the best source (SEQ) reaches 0.074 and the fusion ties it. The full-catalog anchor (fusion 0.0681 NDCG@20 vs published MF 0.12, LightGCN 0.21 on ML-1M) is reported with protocol caveats |
| Degenerate-normalization rate | CB <2% (ML-1M) | 0.0% (both datasets) | ✓ (better than expected) |

## C.2 RQ1 — fidelity (the headline prediction)

**Prediction:** $F_g > 0$ for the redundant POP↔REC pair (never-building one member loses less than masking predicts), $F_g < 0$ for CB on the sparse dataset.

**Outcome:** **MISS — all |F_g| ≤ 0.001 (ML-1M) and ≤ 0.002 (LastFM-2K) across all five seeds.** The masked marginal is a faithful predictor of the never-built outcome for every source on every dataset. The falsification table's pre-committed reframing applies: *"masking is a faithful proxy and the exact game is decision-valid as-is"* — the audit becomes a validation result (it certifies the exact game's actionability) rather than a gap-exposing result. The redundancy and substitutability structure still shows up, but in the **credit** (POP/REC ≈ 0 or negative; SEQ dominant) and in the **order-validity** results, not in F.

## C.3 RQ2 — order validity

| Quantity | Predicted | Actual | Verdict |
|---|---|---|---|
| Kendall τ ML-1M | 0.6–0.9 | 0.60 (all seeds) | ✓ (lower bound) |
| Kendall τ LastFM | 0.4–0.7 | 0.44±0.08 | ✓ |
| Top-1 agreement ML-1M | yes | **no** (credit-min = REC, best retirement = POP) | **MISS** — but the gap is between two near-null sources (P_REC ≈ 0, P_POP ≈ −0.0007); reported as a genuine partial-order-validity finding |
| Top-1 agreement LastFM | yes | yes (POP) | ✓ |
| Co-monotonicity violations | POP↔REC pair, if any | numerous small pairs when |F| ≈ 0 | **not interpretable** — violations are only meaningful when F is material; stated in the paper |

## C.4 RQ3 — reflexivity

**Prediction:** ρ concentrated on the source most redundant with the retired one; 0.05–0.20 (ML-1M) / 0.10–0.30 (LastFM).

**Outcome:** ρ = 0.09–0.27 (ML-1M), 0.18–0.57 (LastFM) — **partially met**. Concentration holds (retiring CB/SEQ moves the other's credit most, ρ_SEQ up to 0.57 when others are retired), but the magnitudes exceed the upper bound on LastFM. The aggregate identity Σφ_nb = v(G) − R_g holds to machine precision in every case.

## C.5 RQ4 — decision rules

**Prediction:** Shapley rule ≥ LOO ≥ FeatureSHAP ≥ random; Shapley over LOO concentrated on the POP↔REC decision.

**Outcome:** all attribution rules beat Random significantly on both datasets (ML-1M p<10⁻¹⁷, LastFM p<10⁻³ after Holm). **Shapley ≈ LOO ≈ FeatureSHAP** — differences are not significant (p ≥ 0.63) or the rules pick the same source; LOO's pick is unstable on LastFM (POP for seed 42, SEQ otherwise). The predicted Shapley-vs-LOO gap on POP↔REC did not materialize because POP/REC have ≈0/negative credit on both datasets (popularity does not predict the temporal-LOO target), so the redundancy divergence SignalShap Proposition 2 predicts is not decision-relevant here.

## C.6 What stands

The machine-precision invariants (efficiency, per-user consistency, fidelity decomposition, reflexivity identity), the null calibration, the dataset-stat check (574,376 positives ≈ the dyhucog-spec's predicted ~575k), and the audit framework itself all validated. The empirical headline is the certification outcome: the exact source game is decision-valid on both benchmarks, its credit ranking is a good-but-not-perfect decommissioning rule (τ = 0.6 / 0.44, top-1 wrong on ML-1M), and attribution-guided retirement significantly beats random. The pre-registered gap hypotheses (F > 0 redundancy, F < 0 un-substitutability) were not observed and are reported as such.
