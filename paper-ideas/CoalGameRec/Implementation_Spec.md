# CoalGameRec — Technical Implementation Specification and Registered Predictions

**Companion to:** `Paper_Structure.md` (the paper blueprint) and `spec.md` (the scope/venue/methodology spec). `spec.md` says *what the paper argues and where it is submitted*; `Paper_Structure.md` says *how the manuscript is laid out*; this file says *what to build for the benchmark and what to expect when it runs*.
**Status:** pre-implementation. Every number in Part B is a **prediction made before running anything**, not a result.
**Reuse:** `stats.py` (paired tests, Holm–Bonferroni, Cohen's d_z) and the clustering/quality diagnostics from `ActionShap/code/` port over with essentially no change. The DyHuCoG-style hypergraph backbone is reused **only if the reproducibility gaps identified in the internal SignalShap audit are fixed first** (see §A.4). Otherwise build the benchmark on LightGCN + an independently documented hypergraph GNN.
**Target venue:** *Discover Artificial Intelligence* — the benchmark is a **separately-scoped empirical case study** that grounds one slice of the survey taxonomy; it is a supporting, secondary contribution, not a method bake-off.

---

# PART A — IMPLEMENTATION

## A.0 What this implementation is for

The paper is a **survey first**. The benchmark exists to do three things and nothing more:

1. **Ground one slice of the taxonomy** — instantiate the *interaction-player / ranking-utility* cell (the survey's Axis 1 × Axis 2 intersection) on real recommenders, so claims about what that specific game formulation does rest on a reproducible artifact rather than only on cited papers. It does **not** empirically validate the whole five-axis taxonomy (features/items/users/contexts/providers/agents, or all solution concepts) — that breadth is covered by the survey corpus, not by this benchmark (review 1.7/2.3/4.3).
2. **Provide one clean intervention comparison** between game-theoretic and non-game-theoretic attribution reweighting under a shared protocol (the BQs in `spec.md` §2). This is an *intervention/reranking* study, not an explanation-faithfulness evaluation (§A.7).
3. **Continuity with the author's work** is limited and must be stated precisely (C11): the benchmark reuses the same **source domains** (MovieLens, Amazon Books) and some metrics, but Amazon-Book is rebuilt, subsampled, and temporally split rather than the reported canonical protocol, so it is **not** the same experimental setting as DyHuCoG.

It is **not** a method bake-off, a new-architecture contribution, a comprehensive empirical validation, or a large-scale study. Keep the method set small and the compute modest.

## A.1 Repository layout

```
CoalGameRec/code/
├── requirements.txt
├── configs/
│   ├── ml1m.yaml
│   └── amazon_book.yaml   # includes the subsample scheme, size, and seed
├── coalgamerec/
│   ├── __init__.py
│   ├── data.py            # loaders, 5-core filtering, temporal split, freeze splits
│   ├── backbone.py        # hypergraph GNN (fixed/pinned) + LightGCN
│   ├── attribution/
│   │   ├── base.py        # uniform / attention / heuristic-pop wrappers
│   │   ├── shapley.py     # preference-aware MC Shapley (players = interactions)
│   │   ├── myerson.py     # communication-graph-restricted Shapley (optional)
│   │   └── est_shapley.py # sampling/importance Shapley variant
│   ├── game.py            # characteristic function v(S), multi-objective
│   ├── rerank.py          # attribution-guided re-ranking
│   ├── baselines.py       # LightGCN, uniform-weight, heuristic reweighting
│   ├── metrics.py         # NDCG, Recall, coverage, ILD
│   ├── stats.py           # PORTED from ActionShap
│   └── report.py          # LaTeX table + figure emitters
├── scripts/
│   ├── build_data.py      # stage 1: data + splits
│   ├── train_backbone.py  # stage 2: train backbones (with/without attribution)
│   ├── run_game.py        # stage 3: coalitions, Shapley, per-family attribution
│   ├── run_rerank.py      # stage 4: attribution-guided rerank + metrics
│   └── run_all.py
├── tests/
└── results/{raw,tables,figures}/
```

## A.2 Environment

```
python = 3.12 (exact patch pinned in a lockfile)
numpy = 2.4.x            # exact minor pinned (C10)
scipy = 1.18.x
scikit-learn = 1.6.x
pandas = 2.2.x
torch = 2.x              # exact version + CUDA/driver recorded; deterministic kernels where available
dgl = 2.x (optional)     # only if the hypergraph backbone is built on DGL
pyyaml, tqdm, pytest, matplotlib
```

GPU optional but recommended (RTX 4090 or equivalent, matching the thesis). The benchmark should also be runnable on CPU at reduced scale for reproducibility testing.

## A.3 Data layer (`data.py`)

The two benchmarks are **MovieLens-1M and Amazon-Book**, matching the group's published DyHuCoG evaluation so the survey reads as a continuation of the same experimental line.

| | MovieLens-1M | Amazon-Book |
|---|---|---|
| Source | GroupLens `ml-1m.zip` | Amazon Reviews 2018 `Books_5.json.gz` (the 5-core subset; 27,164,983 reviews) + `meta_Books.json.gz` |
| Raw interactions | 1,000,209 | ~27M before subsampling |
| Implicit conversion | rating ≥ 4 counts as positive | rating ≥ 4 counts as positive |
| Subsampling | none | random **50,000 users**, fixed seed, before re-filtering |
| Filtering | 5-core, iterative to convergence | 5-core, iterative, after subsampling |
| Timestamps | present | present in the raw corpus |

> ### Do not use the canonical Amazon-Book split.
> The canonical split used in the DyHuCoG paper and throughout the LightGCN/HCCF literature (52,643 users / 91,599 items / 2,984,108 interactions) is a fixed 80/20 random split of anonymized index pairs — **no ratings, no timestamps, no item metadata**. The benchmark's temporal leave-one-out protocol and (if used) context features need timestamps, and reproducibility demands a rebuildable split. **Build from the raw Amazon Reviews 2018 Books corpus** (`Books_5.json.gz` + `meta_Books.json.gz`), which carries `overall`, `unixReviewTime`, and metadata. State in the paper's §4.1 that the split is rebuilt rather than reused, and why — a reviewer familiar with the canonical split will otherwise assume the temporal protocol is impossible.

> ### Subsample before anything else.
> Raw Books 5-core is ~27M reviews over millions of users. Target ~50,000 users under a fixed reported seed, then re-apply the iterative 5-core filter. Sample **users, not interactions** (sampling interactions destroys the interaction structure the game uses). Check the filter does not collapse the sample (see the snowball/time-window fallbacks in the SignalShap spec §A.3); run that feasibility spike before writing scorers.

Implementation requirements:
1. **Subsample Amazon-Book users first**, seed recorded in the config hash, then filter. Reverse order changes the result.
2. **Iterate the 5-core filter to a fixed point**; record final counts and put *those* in the paper's Table 2 (never the canonical numbers).
3. **Temporal leave-one-out split per user**: last interaction = test, second-last = validation, rest = train; break timestamp ties deterministically by row order (Amazon timestamps are day-resolution).
4. **Freeze splits to disk** as user/item integer indices with a config hash, read from there in every stage.
5. Emit a `DatasetStats` record (users, items, interactions, density, per-user interaction statistics).

## A.4 Recommender backbones (`backbone.py`)

Two backbones host the attribution modules, chosen to be small and defensible:

| Backbone | Graph type | Role |
|---|---|---|
| **Hypergraph GNN** (DyHuCoG-style message passing) | hypergraph | **primary** |
| **LightGCN** | homogeneous bipartite | secondary (transfer across graph types) |

> **Critical dependency caveat — read before reusing DyHuCoG code.** The repo's own SignalShap audit found 63 gaps that make faithful DyHuCoG reimplementation impossible (undefined hypergraph construction, unspecified similarity functions, dimensionally inconsistent propagation equations). **Options, in order of preference:**
> 1. Fix and pin the hypergraph construction, the similarity functions, and the propagation equations explicitly in this repo, and document them in the paper's methodology (this also improves the survey's credibility, since DyHuCoG is a flagship case study).
> 2. Use an **independently documented hypergraph GNN** (e.g., a standard Hypergraph Neural Network / HNN or HCCF-style model with a public implementation) and describe the attribution module generically.
>
> Do **not** silently build on unresolved DyHuCoG gaps. The survey's reproducibility claims must survive scrutiny.

## A.5 Attribution families to compare

**Players = interactions (u,i). Value = multi-objective ranking/diversity utility** (mirrors the DyHuCoG game and the thesis notation).

| Label | Attribution | Nature | `v(S)` baseline |
|---|---|---|---|
| `uniform` | uniform edge weight (no attribution) | degenerate baseline | — |
| `attention` | learned interaction-level attention gate | non-game-theoretic | — |
| `heuristic-pop` | popularity/degree weighting | heuristic baseline | — |
| `shapley-mc` | preference-aware Monte-Carlo Shapley | **game-theoretic (primary)** | $v(S)=\alpha\mathrm{NDCG}+\beta\mathrm{Diversity}+\gamma\mathrm{Context}$; $v_{pref}(S)=v(S)+\lambda_{pref}\sum_{(u,i)\in S}\mathrm{sim}(u,i)$ |
| `shapley-ai` | **precisely-defined** sampling/importance Shapley variant (§A.6d) | game-theoretic (estimator ablation) | same as `shapley-mc` |
| `myerson` (exploratory, optional) | communication-graph-restricted Shapley on the hypergraph projection | game-theoretic (structure-aware) | **Myerson value**: the Shapley value of the graph-restricted game `v^g(S) = Σ_{C∈𝒞(g[S])} v(C)`, with `g[S]` the subgraph induced by `S` and `𝒞(g[S])` its connected components; the projection (2-section / incidence / line graph) is a stated method parameter |

Default coalition weights (from DyHuCoG/thesis) for the primary `shapley-mc` family: $\alpha=0.60,\ \beta=0.25,\ \gamma=0.15,\ \lambda_{pref}=0.20$. These are **not** treated as fixed facts: an explicit weight-sensitivity analysis is a **required** robustness check (§A.6c), because the survey itself raises the value-function-arbitrariness critique and the benchmark must not inherit an unexamined weighting scheme across regimes.

> **Keep the method set small and coherent.** Default scope = `uniform` (no-attribution control), `attention`, `heuristic-pop`, `shapley-mc` on both datasets. `shapley-ai` and `myerson` are **exploratory**: if included, every promised cell must be run (no partial factorial); otherwise drop them from headline claims (see §C4 resolution).

## A.6 The game (`game.py`)

Define the characteristic function over a coalition of interactions `S ⊆ N` as the multi-objective utility:

```
v(S) = α·NDCG@20(S) + β·Diversity(S) + γ·Context(S)
v_pref(S) = v(S) + λ_pref·Σ_{(u,i)∈S} sim(u,i)
```

- Compute exact Shapley where the player set is tractably small (illustrative subgames, synthetic coalitions); use the **Monte-Carlo estimator** for the full interaction player set:
  `φ̂_j = (1/M) Σ_{m=1..M} [v(S_m ∪ {j}) − v(S_m)]`
  with the sampling law made explicit (§A.6a) and `M` chosen by an empirical convergence criterion (§A.6a) rather than an unsupported "99%" claim.
- Refresh Shapley values every `f` batches (default f=10) during training; apply light temporal smoothing and clip extremes before normalization. Provide the exact equations and detach/gradient behaviour (§A.6b).
- **Efficiency identity test:** `Σ_j φ_j(v) = v(N) − v(∅)` must hold for the exact, un-smoothed, unnormalized case to machine precision (§A.8). Note that Monte-Carlo estimates, smoothing, clipping, and normalization generally do **not** satisfy exact efficiency; report the residual and label exact vs. approximate values.

## A.6a Monte-Carlo estimator — explicit sampling law

Averaging `[v(S_m∪{j}) − v(S_m)]` over arbitrary subsets `S_m` is not automatically a Shapley estimator. Specify the sampling law:
- **Preferred:** random-permutation sampling — draw a uniform permutation of `N`, take `S_m` as the set of players preceding `j`, weight each term by `1/|N|!` order probability. This is the standard permutation-MC Shapley.
- **Or:** size-weighted subset sampling with the correct Shapley weights `(|S|!(|N|−|S|−1)!/|N|!)`.
- **Or:** an importance-sampling distribution with a defined proposal and correction factor.
State the law, the weights, the variance estimator, and an empirical convergence criterion (e.g., running MSE against a high-`M` reference, reported as a curve) — do not assert a universal 99%-at-50 claim.

## A.6b Refresh / smoothing / clipping / normalization

Give exact equations for: what is refreshed every `f` batches (and what is held fixed); the temporal-smoothing update (weights and decay); the clipping rule (bounds, and that it applies to extremes only); and the normalization used before Eq. 8 in propagation. Specify detach/gradient behaviour for the attribution weights during backprop and cache scope.

## A.6c Required value-function weight sensitivity

Because the survey (SRQ3) itself criticizes the arbitrariness of `v(S)` and the allocation rule cannot correct a poorly chosen value function, the benchmark must include a **required** weight-sensitivity analysis as its own reported result, not an afterthought: report NDCG@20 / Recall@20 for `shapley-mc` under a grid over `(α, β, γ, λ_pref)` (e.g., around the defaults and at least the two extreme weighting schemes), on both datasets, and state whether the ranking of attribution families is stable. If the headline ordering flips under plausible weights, that is a finding to report (it is the arbitrariness critique applied to the benchmark), not a tuning artefact to hide.

## A.6d `shapley-ai` must be a precisely-defined estimator

`shapley-ai` is **not** a named published method. Before it can be a benchmark family it needs a full algorithmic identity: the proposal/importance distribution, the correction factor, the seed handling, the exact estimator equation, and a citation or a clear statement that it is an original estimator defined in this paper. If it cannot be precisely specified, drop it from the benchmark rather than leave an ambiguous cell.

## A.7 Attribution-guided re-ranking (`rerank.py`) — an intervention study, not an explanation-evaluation

This applies attribution-derived weights to re-weight propagation/ranking and measures **Recall@K and NDCG@K** (primary) plus coverage/ILD (secondary) vs. the non-attribution controls. **Scope note (review 1.7/2.5):** this is an *intervention/reranking* study — it tests whether the derived weights are useful ranking features, **not** whether the attribution is a faithful or sufficient *explanation*. It does not by itself answer SRQ3 ("what game theory buys") unless explanation-quality metrics are added (§A.7b) or the claim is explicitly narrowed to intervention. Keep it labeled as an intervention study.

## A.7a Primary metric definitions (used in all result tables)

Report **Recall@K and NDCG@K for K ∈ {5, 10, 20}** as the paper's primary results. Define them explicitly (mirror the thesis Ch. 4.4 so the survey's protocol is auditable):

- **NDCG@K** = `(1/|U|) Σ_u DCG_u@K / IDCG_u@K`, with `DCG_u@K = Σ_{k=1..K} rel_u,k / log2(k+1)` (binary relevance for implicit feedback; IDCG per user).
- **Recall@K** = `(1/|U|) Σ_u |relevant_u ∩ R_u@K| / |relevant_u|`.

Also compute **Catalogue Coverage** = `|∪_u R_u@K| / |I|` and **Intra-List Diversity (ILD)** = `(2/K(K−1)) Σ_{1≤k<l≤K} [1 − sim(i_k,i_l)]` — these are secondary; they never replace Recall@K/NDCG@K as the headline.

## A.7b If explanation quality is claimed, add explanation metrics

If the paper claims the benchmark validates *explainability* (not just reranking), it must also report at least one explanation-quality metric: ranking fidelity to the explained model, deletion/insertion or sufficiency/comprehensiveness, stability under perturbation, sparsity, sign-consistency, or a model-randomization sanity check. Without these, the benchmark is an intervention study only and §7.3/§8 framing must say so. Coverage/ILD/popularity-shift are exposure metrics, not explanation-faithfulness metrics; fairness claims need explicit group/provider metrics, not coverage alone.

## A.8 Test suite (`tests/`)

Non-negotiable, in priority order:
1. **Efficiency identity.** `Σ_j φ_j(v) = v(N) − v(∅)` to machine precision (exact, un-smoothed subgames). Catches most implementation errors.
2. **Empty coalition.** `v(∅)` well-defined (e.g., 0 by the null-rank/projection convention) and non-degenerate.
3. **Symmetry on synthetic data.** Two literally identical score columns/players receive equal Shapley values.
4. **Dummy player.** A pure-noise interaction receives `φ ≈ 0`.
5. **Null/projection calibration.** The empty-coalition ranking (e.g., uniform or popularity baseline) produces the expected NDCG.
6. **Degenerate normalization / weighting.** Constant rows yield zero attribution, no NaN, no inf.
7. **Split reproducibility.** Frozen splits reload identically across stages (config-hash check).
8. **Backbone determinism.** Same seed → same embeddings/weights (to the extent the backbone allows).

Tests 1, 3, and 4 are the ones that would catch a wrong paper rather than a crashed run.

## A.9 Runtime budget

| Stage | ML-1M | Amazon-Book (50k-user sample) |
|---|---|---|
| Raw download + parse | < 1 min | 20–40 min, **one-off** |
| Subsample, filter, split | < 1 min | 2–5 min |
| Backbone training (LightGCN) | 5–15 min | 30–60 min |
| Backbone training (hypergraph GNN) | 15–30 min | 1–2 h |
| Shapley MC estimation (refresh period) | 10–25 min | 30–90 min |
| Re-ranking + metric eval | 5–15 min | 20–60 min |
| **Full pipeline, one seed** | **~45 min–1.5 h** | **~2–4 h** |
| Five seeds, both datasets | ~1–2 GPU-days | |

Cache parsed/subsampled/filtered frames to Parquet keyed by the config hash. If any stage runs an order of magnitude over these, something is wrong (most likely dense ops on the full user–item matrix instead of the candidate slice).

## A.10 Statistical analysis — unit of analysis and inference plan

This section resolves the pairing-unit ambiguity that invalidates a bare "paired t-test" claim. The unit of analysis and the inference plan are **fixed before analysis**:

- **Unit of analysis: per-user.** Primary metrics (NDCG@20, Recall@20) are computed per user on the shared held-out split; the comparison between two attribution families is a **paired difference per user** (same users, same split). The sample for inference is therefore the set of users, not the 5 seeds.
- **Seeds.** The 5 seeds affect the mean ± std summary and quantify training variability; they are **not** the pairing unit for the primary significance tests. Report per-seed means and the distribution of per-user differences, and treat per-seed pooling as a secondary/descriptive view.
- **Primary contrasts (predeclared):** `shapley-mc` vs `uniform` on NDCG@20 and Recall@20, per dataset/backbone. These four contrasts form the **Holm–Bonferroni family**; all other cutoffs/metrics/families/ablations are secondary or exploratory and are flagged as such (no correction applied to exploratory rows, or a separate stated family).
- **Tests.** Paired differences over users; report effect size (Cohen's d_z) with 95% CI and a permutation/bootstrap interval that resamples users while preserving seed structure. Use paired t-test and Wilcoxon signed-rank as **sensitivity analyses**, with assumptions stated, not as the sole basis.
- **Caveats to state:** users are not strictly independent after shared model training, and per-user n is large, so very small p-values can arise for trivial effects; the predeclared contrasts and correction family, plus reporting of effect sizes with intervals, mitigate this. A mixed-effects model over users with seed as a random effect is the fallback if formal cross-seed inference is required.
- Report per-seed values, not only pooled per-user rows, and state the number of comparisons in every significance caption.

---

# PART B — REGISTERED PREDICTIONS

> **Read this as a pre-registration.** Everything below is what I expect *before* running. Recording it now is what makes the results falsifiable rather than rationalized afterwards. When real numbers arrive, report them against this table and **flag every miss explicitly** — a missed prediction that is discussed is a strength, a quietly revised prediction is misconduct.

## B.0 External pre-registration (required before running)

A local Markdown file is not public pre-registration. Before running the benchmark, deposit an **immutable, timestamped, external pre-registration** (OSF, AsPredicted, or Zenodo) containing: the frozen protocol, the directional hypotheses in §B.2–§B.5, the primary contrasts and correction family (§A.10), the estimand (§A.6), and the falsification table (§B.6). Record the commit hash, code version, and a planned-deviations policy. Prediction tables and realized results live in separate, clearly-distinguished artifacts — never in the same file (review 1.1/2.1). The manuscript reports only realized results; hypotheses and the pre-registration link go in supplementary material.

## B.1 Pipeline-level quantities

| Quantity | ML-1M | Amazon-Book | Confidence |
|---|---|---|---|
| Users after filter | ~6,040 | 25,000–50,000 | **low** — depends on sampling |
| Items after filter | ~3,400–3,700 | 30,000–90,000 | **low** — same caveat |
| Density | ~4.5% | 0.03–0.08% | medium |
| NDCG@20, LightGCN backbone | 0.20–0.22 | 0.02–0.03 | medium — order-of-magnitude sanity check only; the rebuilt Amazon split will not match published canonical-split numbers (C11) |
| NDCG@20, hypergraph GNN backbone | 0.22–0.28 | 0.025–0.035 | low |
| Uplift of `shapley-mc` over `uniform` (NDCG@20) | +3% to +8% relative | +5% to +12% relative | low |

**Confidence on the Amazon-Book column is deliberately low** — the rebuilt, subsampled split has no published counterpart, so these are extrapolations from the density regime. The prediction worth holding is the **direction** of the contrast, not the levels.

## B.1a Primary result tables — Recall@K and NDCG@K (filled only with realized results)

The benchmark's primary deliverable is a result table reporting **Recall@K and NDCG@K (K ∈ {5,10,20})** for every attribution family, on both backbones and both datasets, **after** the experiments run. These tables are **populated only with realized numbers**; no predicted point values are placed here. All hypotheses below are directional only ("we test whether X > Y"); no exact figures, no bolded winners, and no mean ± std placeholders are pre-filled, so there is no risk of a placeholder being mistaken for a result.

- **Result table (main text, §7.2 of `Paper_Structure.md`):** one table per dataset/backbone with columns NDCG@5/10/20 and Recall@5/10/20; cells filled with realized mean ± std over seeds; significant differences flagged after correction (§A.10). Generated by `report.py` from raw outputs only.
- **Prediction register (this Part B):** directional hypotheses + falsification contingencies, stored separately from the results and referenced by the external pre-registration (see §B.0).
- LightGCN (Table C) and the hypergraph backbone share the same layout; no cells are pre-computed.

## B.2 BQ1 — ranking quality of game-theoretic vs. non-game-theoretic attribution

**Directional hypotheses (we test whether), stated neutrally — not results.**

| Comparison | Hypothesis | Confidence |
|---|---|---|
| `shapley-mc` vs `uniform` (no-attribution control) | `shapley-mc` ≥ `uniform` on NDCG@20 and Recall@20 | **high** (motivated by the DyHuCoG premise; to be tested) |
| `shapley-mc` vs `attention` | `shapley-mc` ≥ `attention` | medium |
| `shapley-mc` vs `heuristic-pop` | `shapley-mc` ≥ `heuristic-pop` | medium |
| `shapley-ai` vs `shapley-mc` | no directional claim; an estimator-ablation comparison | medium (estimator ablation) |
| `myerson` vs `shapley-mc` | no directional claim; exploratory | low |

**Primary contrast (predeclared):** `shapley-mc` versus the `uniform` no-attribution control on **NDCG@20 and Recall@20** for each dataset/backbone. All other cutoffs, metrics, families, and ablations are secondary/exploratory. This contrast set and its correction family are fixed before analysis (§A.10). Do not draft the Results section narrative around an assumed winner.

## B.3 BQ2 — coverage and diversity (directional hypotheses)

**Hypotheses (we test whether):** attribution-guided weighting broadens catalogue coverage and intra-list diversity without degrading accuracy (the accuracy–diversity trade-off is *partially* resolvable per the thesis RQ4). No magnitude is predicted; the direction of any coverage/ILD difference and whether it is significant (as a secondary outcome, not part of the primary contrast family) are measured. Head/tail exposure shift is reported descriptively, not predicted a priori.

## B.4 BQ3 — training stability and cost (hypotheses, not results)

**Hypotheses:** (a) MC-Shapley refresh increases training time relative to the plain backbone; (b) `shapley-mc` per-run metric variance is comparable to the backbone's own seed variance (i.e., the attribution module does not introduce disproportionate instability); (c) the attribution module adds some memory for weight caches. These are **testable hypotheses with defined measurement** (§A.10 / runtime budget), not claimed facts. A variance-ratio or equivalence check, not an impression, is required to claim "comparable variance."

## B.5 BQ4 — cross-dataset robustness

**Hypothesis (neutral):** if the interaction-player attribution provides a gain, we expect it to be *at least as large* in the sparse regime as in the dense regime, because attribution-guided weighting may matter most where interactions are thin and popularity dominates. This is framed as a **descriptive cross-dataset comparison**, not a causal "density" claim — MovieLens-1M and Amazon-Book differ on many axes (domain, rating process, catalogue size, metadata) beyond density alone. No rank is predicted a priori; the ordering is measured.

## B.6 Falsification and contingencies

| If this happens | What it means | Contingency |
|---|---|---|
| `shapley-mc` does **not** beat `uniform` on NDCG@20 | the game-theoretic premise does not transfer to the benchmark backbones | Report honestly; the survey's empirical grounding weakens to the clustering/prior-work lineage, and the benchmark becomes a negative result (still informative) |
| `shapley-mc` = `attention` everywhere | attribution adds nothing over learned attention | Report as a finding; soften BQ1 framing; the survey's critical analysis (SRQ3) must then emphasize where game theory is relabeled reweighting |
| Coverage/ILD gain but no NDCG gain | diversity without accuracy | Report as the trade-off result the thesis anticipated |
| `shapley-ai` diverges wildly from `shapley-mc` | estimator instability | Investigate M and the value-function variance; report both with estimator diagnostics |
| Efficiency test fails | implementation bug | Stop; do not interpret anything until test 1 passes |
| Amazon-Book subsample collapses under 5-core | uniform user sampling destroyed item support | Switch to snowball or time-window sampling per §A.3, re-report, do not switch again afterwards |
| Hypergraph backbone reproduces no stable result | DyHuCoG-gap issue | Fall back to the independently documented hypergraph GNN (§A.1 option 2) and say so |

The two failures that would genuinely wound the benchmark are (a) `shapley-mc` ≈ `uniform` and (b) the Amazon-Book subsample collapsing. Both are cheap to check early, so **run BQ1 and the Amazon-Book feasibility spike before writing any prose.**

## B.7 Suggested milestone order

0. **Amazon-Book feasibility spike.** Parse raw Books, apply the chosen subsample, run iterative 5-core, verify a workable split survives (users/items/density near target). **Gate:** viable split at ~0.05–0.08% density.
1. Data + splits frozen. **Gate:** split reproducibility test (test 7) passes.
2. LightGCN backbone trains to sane NDCG. **Gate:** reaches a sane absolute level; do **not** calibrate against published numbers from the different canonical split (C11).
3. Hypergraph backbone pinned/reproducible (or fallback chosen). **Gate:** backbone determinism test (test 8).
4. Attribution families + game module. **Gate:** tests 1–4 pass (efficiency, empty, symmetry, dummy).
5. Re-ranking + metrics. **Gate:** BQ2 coverage/ILD computable.
6. Statistics + table/figure emitters.
7. Survey-content integration: map results into the taxonomy/comparison tables; draft the critical-analysis tie-ins.

Steps 4 and 5 are the decision points. The benchmark grounds the survey regardless of the direction of BQ1/BQ2 (a null result is still an informative finding for SRQ3); the contingency table says how to respond without improvising.
