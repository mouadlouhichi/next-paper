# CoalGameRec — Technical Implementation Specification and Registered Predictions

**Companion to:** `Paper_Structure.md` (the paper blueprint) and `spec.md` (the scope/venue/methodology spec). `spec.md` says *what the paper argues and where it is submitted*; `Paper_Structure.md` says *how the manuscript is laid out*; this file says *what to build for the benchmark and what to expect when it runs*.
**Status:** pre-implementation. Every number in Part B is a **prediction made before running anything**, not a result.
**Reuse:** `stats.py` (paired tests, Holm–Bonferroni, Cohen's d_z) and the clustering/quality diagnostics from `ActionShap/code/` port over with essentially no change. **The benchmark does NOT use any DyHuCoG code** (decision by the authors, review 1.3): it uses an **independently documented hypergraph GNN** plus LightGCN (see §A.4).
**Target venue:** *Discover Artificial Intelligence* — the benchmark is a **separately-scoped empirical case study** that grounds one slice of the survey taxonomy; it is a supporting, secondary contribution, not a method bake-off.

---

# PART A — IMPLEMENTATION

## A.0 What this implementation is for

The paper is a **survey first**. The benchmark exists to do three things and nothing more:

1. **Ground one slice of the taxonomy** — instantiate the *interaction-player / ranking-utility* cell (the survey's Axis 1 × Axis 2 intersection) on real recommenders, so claims about what that specific game formulation does rest on a reproducible artifact rather than only on cited papers. It does **not** empirically validate the whole five-axis taxonomy (features/items/users/contexts/providers/agents, or all solution concepts) — that breadth is covered by the survey corpus, not by this benchmark (review 1.7/2.3/4.3).
2. **Provide one clean intervention comparison** between game-theoretic and non-game-theoretic attribution reweighting under a shared protocol (the BQs in `spec.md` §2). This is an *intervention/reranking* study, not an explanation-faithfulness evaluation (§A.7).
3. **Continuity with the author's work** is limited and must be stated precisely (C11): the benchmark shares the same **source domains** (MovieLens, Amazon Books) and some metrics, but it uses an **independently documented hypergraph GNN (not DyHuCoG code)**, Amazon-Book is rebuilt/subsampled/temporally split rather than the canonical protocol, so it is **not** the same experimental setting as DyHuCoG.

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
│   │   ├── additive_pref.py # matched non-game additive-similarity baseline
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
python = 3.12.x          # exact patch pinned in a lockfile (C10)
numpy = 2.4.x            # exact minor pinned
scipy = 1.18.x
scikit-learn = 1.6.x
pandas = 2.2.x
torch = 2.x              # exact version + CUDA/driver recorded; deterministic kernels where available
# DGL version recorded ONLY if the pinned HCCF/HGNN implementation requires it (exact, not "optional")
pyyaml, tqdm, pytest, matplotlib
```

This is the **single pinned environment** used in `spec.md` §7.6 and the config; it is a **new pinned environment for the independent HCCF backbone**, not "the same environment as the thesis." GPU: RTX 4090 (or equivalent). CPU at reduced scale may be used for **reproducibility testing only**, not for the reported numbers.

## A.3 Data layer (`data.py`)

The two benchmarks are **MovieLens-1M** (standard public dataset) and **Amazon-Book** (a **custom subsample of the public source**), chosen for a dense/sparse contrast. Both are processed under this paper's own protocol (§A.3); the Amazon-Book sample is **not** the standard Amazon-Book benchmark and is **not** a reproduction of any prior paper's evaluation.

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

> ### Why the 2018 Amazon release (dataset audit #5).
> We use the **2018 Amazon Reviews `Books_5.json.gz`** rather than the newer 2023 release. Justification to state in §4.1: the 2018 5-core corpus is the de-facto standard for reproducibility against the large existing Amazon-Book literature, has the same `overall`/`unixReviewTime`/metadata fields this protocol needs, and the 2023 release would make the custom split non-comparable to any published numbers. The paper must cite the UCSD page (Ni, Li & McAuley 2019), state the exact file and 5-core count, and disclose that this is an older release retained mainly for reproducing past results.

> ### Subsample before anything else.
> Raw Books 5-core is ~27M reviews over millions of users. Target ~50,000 users under a fixed reported seed, then re-apply the iterative 5-core filter. Sample **users, not interactions** (sampling interactions destroys the interaction structure the game uses). Check the filter does not collapse the sample (use the snowball/time-window fallbacks described here if it does); run that feasibility spike before writing scorers. (The earlier cross-reference to a `SignalShap` spec dependency is removed — the fallbacks are specified in this file.)

Implementation requirements:
1. **Subsample Amazon-Book users first**, seed recorded in the config hash, then filter. Reverse order changes the result.
2. **Iterate the 5-core filter to a fixed point**; record final counts and put *those* in the paper's dataset-statistics table (Table 8 / Online Resource in `Paper_Structure.md`), never the canonical numbers.
3. **Temporal leave-one-out split per user**: last interaction = test, second-last = validation, rest = train. **Timestamp ties (fixed):** use a **stable secondary key** (the original line index, preserved on disk and released), **not** re-parse-dependent row order (Amazon timestamps are day-resolution).
4. **Freeze splits to disk** as user/item integer indices with a config hash, read from there in every stage.
5. Emit a `DatasetStats` record (users, items, interactions, density, per-user interaction statistics).

## A.3a Dataset, split, and leakage controls (dataset/split audit)

**Final frozen choices (P0.2 — synchronized with `spec.md` §7.3).** One deterministic preprocessing algorithm and one config table with all final values are written before registration. The choices are:
- **Filtering order (audit #1, fixed):** convert ratings to positives (`rating ≥ 4`) **first**, then apply iterative 5-core to the positive interaction graph to a fixed point. (No "choose and document" language — this order is fixed.)
- **Leakage (audit #3, fixed):** the 5-core eligibility filter uses **training-period interactions only**; no future interactions are used for eligibility. Label the protocol accordingly (not transductive).
- **Candidate evaluation (audit #8, fixed):** full-catalogue scoring for top-K; candidate pool = top-200 items by model score excluding train items; **held-out positives excluded from negative sampling**; candidate pools **fixed across methods**; ties broken deterministically.
- **Recall vs. HitRate (audit #9, fixed):** under leave-one-out with one test item per user, report **HitRate@K** as the metric (Recall@K with one relevant item equals HitRate); state the single-test-positive convention.
- **NDCG edge cases (audit #10, fixed):** users with no test positives are excluded and reported; IDCG when no relevant item exists is defined (1.0 if the item is in the pool, else 0 handling stated); binary relevance; positive conversion occurs **before** the split.
- **ILD `sim(i,j)` (audit #11, fixed):** cosine similarity over a **fixed, interaction-only item-feature representation shared across all methods**; missing-metadata rule defined (feature-absent → 0 similarity); no learned method-dependent embeddings.
- **Coverage denominator (audit #12, fixed):** the **eligible item catalogue after seen-item filtering**.
- **Negative sampling (audit #14, fixed):** popularity (item-degree-weighted) distribution; N=4 negatives per positive; fixed hard-negative refresh schedule; validation/test excluded; a **separate random stream** from split/init/coalition.
- **Temporal ties (audit #13, fixed):** stable secondary key (original line index, released), not re-parse-dependent row order.
- **Cross-dataset feature mismatch (audit #15, fixed):** use a **common interaction-only similarity** so MovieLens-1M and Amazon-Book are feature-matched; do not rely on genres/demographics that differ across datasets.

## A.4 Recommender backbones (`backbone.py`) — pinned, with a preregistered fallback rule (P0.1)

**The benchmark does NOT use DyHuCoG code** (authors' decision, review 1.3). Exactly one primary model is pinned **before** implementation and recorded in the config; selecting after feasibility results is prohibited (creates researcher degrees of freedom).

| Backbone | Graph type | Role | Pinning |
|---|---|---|---|
| **HCCF (Hypergraph Contrastive Collaborative Filtering)** — Xia, Huang, Xu, Zhao, Yin & Huang, SIGIR 2022 | hypergraph | **primary** | Official public implementation pinned to a specific commit/tag; DOI, repository URL, commit hash, and license recorded in the config |
| **LightGCN** | homogeneous bipartite | secondary (transfer across graph types) | Official public implementation pinned to a commit/tag |

**Pinned-before-implementation spec for the primary backbone (must all be fixed and recorded):** formal model name; primary paper + DOI; code repository + immutable commit/tag; license; hypergraph construction (incidence matrix definition); propagation/normalization equations; input features and initialization; number of layers/dimensions; optimizer and regularization; the exact attribution integration point (which message-passing step the Shapley weights enter); supported Python/PyTorch/CUDA versions; deterministic settings and numerical tolerances.

**Preregistered fallback rule (only rule, decided in advance):** if HCCF's official code cannot be run reproducibly under the exact pinned environment (determinism, licensing, or dependency failure), fall back to a **self-contained, independently implemented standard Hypergraph Neural Network (HGNN, Feng et al. 2019)** with fully documented equations. The fallback triggers **only** on the predeclared failure condition, is disclosed as a protocol deviation, and is never chosen after inspecting results.

> This decouples the benchmark from the survey's §4.6 worked example (DyHuCoG as a literature case only) and keeps reproducibility claims defensible. Do **not** write "HNN/HGCN/HCCF-style" or "pick a small, defensible set" in the executable protocol — the backbone is HCCF, pinned.

## A.5 Attribution families to compare

**Players = interactions (u,i). Value = multi-objective ranking/diversity utility**, defined independently in this paper (§A.6); no prior codebase or game implementation is reused.

| Label | Attribution | Nature | `v(S)` baseline |
|---|---|---|---|
| `uniform` | uniform edge weight | **no-attribution control** (not "an attribution method") | — |
| `attention` | learned interaction-level attention gate (matched parameter count) | non-game-theoretic control | — |
| `heuristic-pop` | popularity/degree weighting | heuristic control | — |
| `additive-pref` | additive preference-similarity prior without game aggregation | **matched non-game heuristic (required, review 2.5)** | same preference term, no Shapley averaging |
| `shapley-mc` | preference-aware Monte-Carlo Shapley | **game-theoretic (primary)** | $v(S)=\alpha\mathrm{NDCG}+\beta\mathrm{Diversity}+\gamma\mathrm{Context}$; $v_{pref}(S)=v(S)+\lambda_{pref}\sum_{(u,i)\in S}\mathrm{sim}(u,i)$ |
| `shapley-ai` | **candidate estimator** (sampling/importance Shapley variant) — NOT yet fully defined (§A.6d); kept only if precisely specified before registration | exploratory estimator ablation | same as `shapley-mc` |
| `myerson` (exploratory, optional) | communication-graph-restricted Shapley on the hypergraph projection | game-theoretic (structure-aware) | **Myerson value**: the Shapley value of the graph-restricted game `v^g(S) = Σ_{C∈𝒞(g[S])} v(C)`, with `g[S]` the subgraph induced by `S` and `𝒞(g[S])` its connected components; the projection (2-section / incidence / line graph) is a stated method parameter |

Default coalition weights for the primary `shapley-mc` family (chosen here, not inherited from any prior codebase): $\alpha=0.60,\ \beta=0.25,\ \gamma=0.15,\ \lambda_{pref}=0.20$. These are **not** treated as fixed facts: an explicit weight-sensitivity analysis is a **required** robustness check (§A.6c), because the survey itself raises the value-function-arbitrariness critique and the benchmark must not inherit an unexamined weighting scheme across regimes.

> **Keep the method set small and coherent — primary vs. exploratory (review §3).** **Primary set (run on every cell):** `uniform` (no-attribution control), `additive-pref` (matched non-game heuristic), and `shapley-mc` (game-theoretic). `attention` and `heuristic-pop` are primary controls where stated. `shapley-ai` and `myerson` are **exploratory**: if included, every promised cell must be run (no partial factorial); otherwise drop them from headline claims. (The earlier broken "§C4 resolution" cross-reference is removed — the rule is stated here.)

## A.6 The game (`game.py`) — exact estimand (P0.3)

Define the estimand precisely (no alternatives, no template):

**Game type (fixed): frozen-model interaction-mask game.** For each user `u`, the **player set** `N_u` is the set of observed **training** interactions in `u`'s receptive field (the edges incident to `u` in the frozen graph). A **coalition** `S ⊆ N_u` is realized as an **edge mask** on the frozen model's graph at inference (mask out the training edges not in `S`); the model is **not** retrained per coalition. `v_u(S)` is the value of a **specified ranking functional** on a candidate set constructed **without test information**, computed at the **final, frozen model snapshot**. `S` contains **training edges only** (never evaluation/test interactions).

**Coalition value (per user):**

```
v_u(S) = α·NDCG@20_u(S) + β·Diversity_u(S) + γ·Context_u(S)
v_pref,u(S) = v_u(S) + λ_pref·Σ_{(u,i)∈S} sim(u,i)
```

Global `v(S) = (1/|U|) Σ_u v_u(S)`. Attribution is converted to propagation/reranking weights by a **fixed, predeclared rule** (§A.7) — state exactly which message-passing step the Shapley weights enter (§A.4) and how.

- **Moving-game semantics (5.5, fixed):** the reported attribution is computed on the **final frozen model snapshot** (post-hoc, at a fixed game snapshot). **No in-training refresh is part of the primary protocol.** The in-training refresh described below is an **optional exploratory training-control study (Study C)**, clearly separated, with its own time-indexed definition and efficiency test for that snapshot; it is not the primary attribution. The primary efficiency test is for the fixed snapshot only.
- **Leakage controls (4.2 #13):** coalition values, tuning, and `sim` use training data or a clearly separated validation set; graph construction never sees held-out edges; no hyperparameters inherited from prior evaluation. State that `S` contains training interactions only.
- **Value-function terms (4.2 #10, fixed):** `NDCG@20_u(S)` per §A.7a over the coalition's candidate set; `Diversity_u(S)` = mean Intra-List Diversity over the coalition's recommended lists (§A.7a); `Context_u(S)` = mean context-alignment score **only if context features exist in both datasets — otherwise `γ` is dropped (not silently set)**; `sim(u,i)` = a **fixed, feature-fixed similarity** (cosine over the common interaction-only item representation). Data source for each stated for both datasets; cross-dataset feature availability reconciled (audit #15).
- **Objective normalization (4.2 #11, fixed):** NDCG, Diversity, Context, and similarity are normalized to comparable [0,1]-style ranges before the weighted sum; state the normalization and the resulting range of `v(S)`.
- **Same metric as value and outcome (4.2 #12):** because NDCG appears in both `v(S)` and the headline outcome, disclose this and add at least one **held-out/alternative outcome** (e.g., Recall@20 on a separate held-out set, or a distinct metric) and matched non-game objectives so the evaluation is not circular.
- **Empty-coalition semantics (4.2 #5):** define NDCG, Diversity, Context, and Coverage for the empty ranking/graph (baseline = the null/popularity ranker, not an arbitrary zero that creates artificial marginals); justify `v(∅)`.
- Compute exact Shapley where the player set is tractably small (illustrative subgames, synthetic coalitions); use the **Monte-Carlo estimator** for the full interaction player set:
  `φ̂_j = (1/M) Σ_{m=1..M} [v(S_m ∪ {j}) − v(S_m)]`
  with the sampling law made explicit (§A.6a) and `M` chosen by an empirical convergence criterion (§A.6a) rather than an unsupported "99%" claim.
- **Primary attribution: frozen snapshot (post-hoc), no in-training refresh.** The optional **exploratory Study C** may refresh Shapley values every `f` batches during training; if so, give exact equations, detach/gradient behaviour, cache scope, and define the reported value as a **time-indexed online attribution** on a fixed game snapshot per refresh (§A.6b). It is not the primary attribution and does not validate the primary efficiency test.
- **Efficiency identity test:** `Σ_j φ_j(v) = v(N) − v(∅)` must hold for the exact, un-smoothed, unnormalized case to machine precision (§A.8). Note that Monte-Carlo estimates, smoothing, clipping, and normalization generally do **not** satisfy exact efficiency; report the residual and label exact vs. approximate values.

## A.6a Monte-Carlo estimator — explicit sampling law

Averaging `[v(S_m∪{j}) − v(S_m)]` over arbitrary subsets `S_m` is not automatically a Shapley estimator. Specify the sampling law (5.3):
- **Preferred: random-permutation sampling.** Draw a uniform permutation of `N`; for each player `j`, take `S_m` as the set of players preceding `j` and record the marginal `[v(S_m∪{j}) − v(S_m)]`. **For uniformly sampled permutations, the estimator is the sample mean of these marginals over `M` permutations** — the `1/|N|!` factor belongs to the *exact* permutation expectation and is **not** multiplied into each ordinary Monte-Carlo sample mean. State explicitly whether a single permutation supplies predecessor sets for all players, and how many model evaluations that requires (per user or per coalition).
- **Or:** size-weighted subset sampling with the correct Shapley weights `(|S|!(|N|−|S|−1)!/|N|!)`.
- **Or:** an importance-sampling distribution with a defined proposal and correction factor.
State the law, the weights, the variance estimator, and an **empirical convergence criterion** — do not assert a universal 99%-at-50 claim. **A high-`M` reference demonstrates numerical stability, not accuracy against the true Shapley value (5.4);** validate convergence against (a) exact analytic synthetic games, (b) exact enumeration on small real subgames, plus efficiency residuals, convergence curves, and common-random-number handling across methods.

## A.6b Refresh / smoothing / clipping / normalization (optional Study C only)

These apply only to the **optional exploratory training-control study (Study C)**, not the primary frozen-snapshot attribution. If Study C is run, give exact equations for: what is refreshed every `f` batches (and what is held fixed); the temporal-smoothing update (weights and decay); the clipping rule (bounds, extremes only); the normalization used before propagation weighting; and detach/gradient behaviour for the attribution weights during backprop and cache scope. Define whether the reported value is a fixed-snapshot or time-indexed online attribution, and report the efficiency residual for each refresh snapshot.

## A.6c Required value-function weight sensitivity

Because the survey (SRQ3) itself criticizes the arbitrariness of `v(S)` and the allocation rule cannot correct a poorly chosen value function, the benchmark must include a **required** weight-sensitivity analysis as its own reported result. **Fix an exact grid before registration (5.7)** — a finite, enumerated set of tuples, e.g. `(α, β, γ, λ_pref) ∈ { (0.60,0.25,0.15,0.20), (1.0,0,0,0), (0,1.0,0,0), (0.5,0.5,0,0), (0.33,0.33,0.33,0.33), (0.6,0.25,0.15,0.0), (0.6,0.25,0.15,0.5) }` — and state **which tuples run on which datasets/backbones** (decide whether all tuples run on all cells or a stated subset). Report NDCG@20 / Recall@20 for `shapley-mc` under this grid and whether the ranking of attribution families is stable. Include the sensitivity cost in the runtime budget, and state whether it is descriptive or inferential. If the headline ordering flips under plausible weights, that is a finding to report (the arbitrariness critique applied to the benchmark), not a tuning artefact to hide.

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
| Five seeds × both backbones × both datasets × default families | ~1–2 GPU-days (ML-1M dominated) | ~2–4 GPU-days (Amazon-Book, 50k-user sample) |

Cache parsed/subsampled/filtered frames to Parquet keyed by the config hash. If any stage runs an order of magnitude over these, something is wrong (most likely dense ops on the full user–item matrix instead of the candidate slice).

**Scope note (5.9):** "one seed" in the table = one backbone on one dataset for the **primary family set only** (uniform/additive-pref/shapley-mc), excluding the sensitivity grid and any exploratory methods. All numbers are **pilot estimates**; state the hardware, concurrency, and included cells when measured. The weight-sensitivity grid and any exploratory cells are budgeted separately and must be added to the total if run.

## A.10 Statistical analysis — unit of analysis and inference plan

This section resolves the pairing-unit ambiguity that invalidates a bare "paired t-test" claim. The unit of analysis and the inference plan are **fixed before analysis**:

- **Unit of analysis: per-user paired differences**, paired **within the same seed** (same split, candidate set, negative-sampling stream, and user). Users evaluated by one trained model share model-level randomness, so they are **not** independent model replicates; per-user n is large, so tiny p-values can arise for trivial effects.
- **Seeds.** The 5 seeds quantify training variability; they are **not** the pairing unit. Report per-seed means and the distribution of per-user differences; treat per-seed pooling as secondary/descriptive.
- **Primary contrast family (P0.4 correction).** The full design is `shapley-mc` vs `uniform` on NDCG@20 and Recall@20 × 2 datasets × 2 backbones = **8 metric-by-condition tests**. **Choose exactly one plan before registration and apply it consistently:**
  - **(A) 8 tests in one Holm–Bonferroni family** (all primary, corrected jointly), **or**
  - **(B) 4 predeclared joint hypotheses** (e.g., per dataset on the primary backbone only, with the other backbone/metrics/cutoffs declared secondary/exploratory and not jointly corrected), **or**
  - **(C) a hierarchical/clustered model** over users with seed as a cluster, prespecified.
  Do **not** call it "four contrasts" while the design yields eight tests. Update every reference (this section, `spec.md` §7.5, captions, preregistration) to the chosen count.
- **Seed-clustered inference (required, 5.8):** define a cluster/bootstrap that resamples users while preserving seed-level dependence — e.g., bootstrap users within each seed and aggregate seed estimates, or fit a mixed-effects/hierarchical model over users with seed as a random effect. Reporting per-seed means does not repair primary p-values that treat users as independent.
- **Tests.** Paired differences over users; report effect size (Cohen's d_z) with 95% CI and a permutation/bootstrap interval that resamples users while preserving seed structure. Paired t-test and Wilcoxon signed-rank are **sensitivity analyses** (assumptions stated), not the sole basis.
- Report per-seed values, and state the exact unit/test/alternative/correction and number of comparisons in every significance caption.

---

# PART B — REGISTERED PREDICTIONS

> **Read this as a pre-registration.** Everything below is what I expect *before* running. Recording it now is what makes the results falsifiable rather than rationalized afterwards. When real numbers arrive, report them against this table and **flag every miss explicitly** — a missed prediction that is discussed is a strength, a quietly revised prediction is misconduct.

## B.0 External pre-registration and ethics determination (planned — P0.6)

**As of this revision, the benchmark is NOT yet registered.** A local Markdown file is not public pre-registration. **Planned (before processing confirmatory data):** deposit an **immutable, timestamped, external pre-registration** (OSF, AsPredicted, or Zenodo) containing the frozen protocol, the directional hypotheses in §B.2–§B.5, the primary contrasts and correction family (§A.10), the estimand (§A.6), and the falsification table (§B.6). Record the commit hash, code version, and a planned-deviations policy. Also **obtain the institutional ethics determination** for the human-generated data before any confirmatory processing (see `spec.md` §1.1). Until these exist: label Part B as a **prediction draft** (not a registered record), and do **not** claim in the abstract/article-type plan that the case study is pre-registered. Prediction tables and realized results live in separate, clearly-distinguished artifacts — never in the same file.

## B.1 Pipeline-level quantities

| Quantity | ML-1M | Amazon-Book | Confidence |
|---|---|---|---|
| Users after filter | ~6,040 | 25,000–50,000 | **low** — depends on sampling |
| Items after filter | ~3,400–3,700 | 30,000–90,000 | **low** — same caveat |
| Density | ~4.5% | 0.03–0.08% | medium |
| NDCG@20, LightGCN backbone | 0.20–0.22 | 0.02–0.03 | medium — order-of-magnitude sanity check only; the rebuilt Amazon split will not match published canonical-split numbers (C11) |
| NDCG@20, HCCF (pinned primary) backbone | 0.22–0.28 | 0.025–0.035 | low |
| Uplift of `shapley-mc` over `uniform` (NDCG@20) | +3% to +8% relative | +5% to +12% relative | low |

**Confidence on the Amazon-Book column is deliberately low** — the rebuilt, subsampled split has no published counterpart, so these are extrapolations from the density regime. The prediction worth holding is the **direction** of the contrast, not the levels.

## B.1a Primary result tables — Recall@K and NDCG@K (filled only with realized results)

The benchmark's primary deliverable is a result table reporting **Recall@K and NDCG@K (K ∈ {5,10,20})** for the **primary set** (`uniform`, `additive-pref`, `shapley-mc`) on both backbones and both datasets, **after** the experiments run; exploratory rows (`attention`, `heuristic-pop`, and any `shapley-ai`/`myerson` kept) are clearly labelled. These tables are **populated only with realized numbers**; no predicted point values are placed here. All hypotheses below are directional only ("we test whether X > Y"); no exact figures, no bolded winners, and no mean ± std placeholders are pre-filled, so there is no risk of a placeholder being mistaken for a result.

- **Result table (main text, §7.2 of `Paper_Structure.md`):** one table per dataset/backbone with columns NDCG@5/10/20 and Recall@5/10/20; cells filled with realized mean ± std over seeds; significant differences flagged after correction (§A.10). Generated by `report.py` from raw outputs only.
- **Prediction register (this Part B):** directional hypotheses + falsification contingencies, stored separately from the results and referenced by the external pre-registration (see §B.0).
- LightGCN (Table C) and the hypergraph backbone share the same layout; no cells are pre-computed.

## B.2 BQ1 — ranking quality of game-theoretic vs. non-game-theoretic attribution

**Directional hypotheses (we test whether), stated neutrally — not results.**

| Comparison | Hypothesis | Confidence |
|---|---|---|
| `shapley-mc` vs `uniform` (no-attribution control) | `shapley-mc` ≥ `uniform` on NDCG@20 and Recall@20 | **medium-high** (motivated by the hypothesis that principled contribution weighting helps ranking; to be tested) |
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
| HCCF (pinned primary backbone) reproduces no stable result | backbone/implementation issue (not a DyHuCoG issue, since no DyHuCoG code is used) | Apply the **preregistered fallback rule only** (§A.4): fall back to a self-contained HGNN, disclose as a protocol deviation; do not switch after inspecting results |

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
