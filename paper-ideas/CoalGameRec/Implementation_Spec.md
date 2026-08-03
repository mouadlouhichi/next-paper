# CoalGameRec — Technical Implementation Specification and Registered Predictions

**Companion to:** `Paper_Structure.md` (the paper blueprint) and `spec.md` (the scope/venue/methodology spec). `spec.md` says *what the paper argues and where it is submitted*; `Paper_Structure.md` says *how the manuscript is laid out*; this file says *what to build for the benchmark and what to expect when it runs*.
**Status:** pre-implementation. Every number in Part B is a **prediction made before running anything**, not a result.
**Reuse:** `stats.py` components for paired differences, Holm–Bonferroni correction, bootstrap utilities, and descriptive user-conditional Cohen's `d_z` can be adapted from `ActionShap/code/`; the reporting labels and conditional-user estimand must be updated for this benchmark. **The benchmark does NOT use any DyHuCoG code** (decision by the authors, review 1.3): it uses an **independently documented hypergraph GNN** plus LightGCN (see §A.4).
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

## A.2 Environment — resolved via HCCF port (Option B, P0.1)

The official HCCF code documents Python 3.6.12/TensorFlow 1.14 **or** Python 3.10.4/PyTorch 1.11.0 with NumPy 1.22.3 and SciPy 1.7.3, which conflicts with a modern PyTorch stack. **Decision (P0.1, Option B): port HCCF to the pinned environment below**, keeping a documented fork and validating it before use. This is a **ported HCCF implementation**, not the official binary.

```
python = 3.12.x          # exact patch pinned in a lockfile (C10)
numpy = 2.4.x            # exact minor pinned (after verifying HCCF port compatibility)
scipy = 1.18.x
scikit-learn = 1.6.x
pandas = 2.2.x
torch = 2.x              # exact version + CUDA/driver recorded; deterministic kernels where available
pyyaml, tqdm, pytest, matplotlib
```

**Port requirements (recorded in a `PORT.md` and the config):** pin the fork commit; document every code change relative to the official HCCF repo; adapt only the data interface and the attribution wrapper; **validate the port against an official-code rerun under a prespecified HCCF dataset/protocol** before using it for the case study. **Validation tolerance (fixed before preregistration):** the port passes only if mean Recall@20 and mean NDCG@20 over five fixed seeds are each within the larger of **5% relative difference** or **0.005 absolute difference** from the independently rerun official-code reference under the same split, preprocessing, candidate protocol, and metrics. The validation artifact records the validation dataset, split, preprocessing, seeds, metrics, candidate protocol, hardware, allowed implementation differences, logs, and pass/fail decision. The environment and port status are part of the preregistration. This is the **single pinned environment** for the benchmark, not "the same environment as the thesis." GPU: RTX 4090 (or equivalent); CPU at reduced scale for reproducibility testing only.

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
> The canonical split used in the DyHuCoG paper and throughout the LightGCN/HCCF literature (52,643 users / 91,599 items / 2,984,108 interactions) is a fixed 80/20 random split of anonymized index pairs — **no ratings, no timestamps, no item metadata**. The benchmark's temporal leave-one-out protocol needs timestamps, and reproducibility demands a rebuildable split. **Build from the raw Amazon Reviews 2018 Books corpus** (`Books_5.json.gz` + `meta_Books.json.gz`), which carries `overall` and `unixReviewTime`. State in the case-study data section (§7.1) that the split is rebuilt rather than reused, and why — a reviewer familiar with the canonical split will otherwise assume the temporal protocol is impossible. **No canonical-split sanity check is included** in the primary or secondary benchmark because it would be a separate experimental protocol and cannot support the temporal BQs.

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
- **Deterministic pipeline order (4.1, fixed — resolves the training-period circularity):** (1) sample users from the source; (2) convert ratings to positives (`rating ≥ 4`); (3) create a **preliminary temporal split** (train/val/test by timestamp, stable tie key); (4) apply iterative 5-core to the **training-period positive graph** to a fixed point; (5) remove users/items that fall below threshold and **rebuild the split** from the surviving positives; (6) verify the fixed point and the minimum-history rule, and report how many users/items were removed at each step. This ordering makes "training-period 5-core" well-defined rather than circular.
- **Filtering order (audit #1, fixed):** positives (`rating ≥ 4`) are defined **before** 5-core (per step 2 above).
- **Leakage (audit #3, fixed):** the 5-core eligibility filter uses **training-period interactions only**; no future interactions are used for eligibility. Label the protocol accordingly (not transductive).
- **Candidate evaluation (audit #8, fixed; P0.3 resolution):** **full-catalogue ranking over all eligible unseen items** for every method — no method-generated top-K pool. Because every held-out positive is always in the full eligible item set, IDCG is always well-defined and no test item can be missing from a model-dependent candidate pool. For the coalition value, the candidate set is the **full eligible item catalogue, fixed across all coalitions** (a coalition changes the ranking, not the candidate set). Ties broken deterministically. Negative sampling for training is separate (below).
- **Recall vs. HitRate (audit #9, fixed):** under leave-one-out with one test item per user, report **HitRate@K** as the metric (Recall@K with one relevant item equals HitRate); state the single-test-positive convention.
- **NDCG edge cases (audit #10, fixed):** users with no test positives are excluded and reported; binary relevance; positive conversion occurs **before** the split. **IDCG rule (complete, P0/4.3):** with the single test positive always in the full-catalogue candidate set, IDCG@K = 1.0 for every evaluated user (the relevant item is guaranteed present). A user whose test positive is outside the eligible item set is excluded and reported **before** candidate membership is inspected (never silently dropped after seeing scores).
- **ILD `sim(i,j)` (audit #11, fixed):** cosine similarity over a **fixed, interaction-only item-feature representation shared across all methods**; missing-metadata rule defined (feature-absent → 0 similarity); no learned method-dependent embeddings.
- **Fixed item representation `x_i` (P0.5, fixed):** for each item `i`, `x_i` is the L2-normalized sparse binary vector over **training-period users only**: `x_i[u]=1` iff user `u` has a positive training interaction with item `i`, else `0`, followed by L2 normalization; zero-degree items after filtering are excluded from `I_eligible`. No metadata, text, demographics, validation/test interactions, or learned embeddings enter `x_i`. Store the item-user matrix as sparse CSR/CSC and compute candidate scores as sparse matrix products: for a user prototype `h_u=Σ_{(u,j)∈N_u} w_{u,(u,j)} x_j`, all `κ`-weighted candidate scores are `h_u X_C^T`, avoiding dense all-item cosine matrices. The representation dimensions, sparsity, checksums, and memory footprint are recorded in the config and dataset report.
- **Preference term `sim(u,i)` (4.4, fixed):** the user vector is the **normalized mean of the user's training-item vectors** (from the same fixed item representation); `sim(u,i)` = cosine between that user vector and item `i`'s vector; the representation is **fixed (not learned)**; the user vector uses **training interactions only**; cold/missing cases (no training items) are handled by a defined rule (similarity = 0 and the user flagged).
- **Coverage denominator (audit #12, fixed):** the **global eligible item catalogue** `I_eligible` after train-period filtering and 5-core preprocessing. Per-user seen-item filtering defines each user's candidate set, but catalogue coverage is `|∪_u R_u@K| / |I_eligible|` with one fixed denominator per dataset.
- **Negative sampling (audit #14, fixed):** popularity (item-degree-weighted) distribution; `n_neg=4` negatives per positive; hard-negative refresh every 5 epochs after epoch 10; validation/test excluded; a **separate random stream** from split/init/coalition.
- **Temporal ties (audit #13, fixed):** stable secondary key (original line index, released), not re-parse-dependent row order.
- **Cross-dataset feature mismatch (audit #15, fixed):** use a **common interaction-only similarity** so MovieLens-1M and Amazon-Book are feature-matched; do not rely on genres/demographics that differ across datasets.

## A.4 Recommender backbones (`backbone.py`) — pinned port of HCCF, with a preregistered fallback rule (P0.1)

**The benchmark does NOT use DyHuCoG code** (authors' decision, review 1.3). Exactly one primary model is pinned **before** implementation and recorded in the config; selecting after feasibility results is prohibited (creates researcher degrees of freedom).

| Backbone | Graph type | Role | Pinning |
|---|---|---|---|
| **HCCF (Hypergraph Contrastive Collaborative Filtering)** — Xia, Huang, Xu, Zhao, Yin & Huang, "Hypergraph Contrastive Collaborative Filtering," SIGIR 2022, **DOI 10.1145/3477495.3532058**; official repo `https://github.com/akaxlh/HCCF` | hypergraph | **primary** | **Ported (Option B):** fork the official repo, pin the fork commit, document all changes in `PORT.md`, validate against an official HCCF dataset/protocol (A.2). Record DOI, repo URL, fork commit, and license in the config |
| **LightGCN** | homogeneous bipartite | secondary (transfer across graph types) | Official public implementation pinned to a commit/tag |

**Loss reconciliation (P0.2).** HCCF's standard training is **not** a bare BPR loss: it includes a hypergraph contrastive objective with parameters `ssl_reg`, `temp`, and `keepRate` (graph/feature dropout and contrastive views). **Decision: retain HCCF's full standard training (CF/BPR loss + the contrastive objective) with fixed, identical hyperparameters for every attribution family.** All families are trained with the **exact same HCCF loss and contrastive settings**; the families differ **only** in the attribution/reranking module applied at evaluation (frozen-model post-hoc), never in the backbone training procedure. Do **not** remove the contrastive term (that would be a different, modified backbone and must be stated as such if ever chosen). The loss equation, `ssl_reg`, `temp`, `keepRate`, and contrastive-view construction are recorded in the config.

**Pinned-before-implementation spec for the primary backbone (all fixed and recorded):** formal model name; primary paper + DOI; ported repo + fork commit; license; hypergraph construction (incidence matrix definition); propagation/normalization equations; input features and initialization; number of layers/dimensions; optimizer and regularization; the exact attribution integration point (which message-passing/reranking step the Shapley weights enter — see §A.6); supported Python/PyTorch/CUDA versions; deterministic settings and numerical tolerances.

**Preregistered fallback rule (only rule, decided in advance):** if the HCCF port cannot pass the prespecified validation tolerance (mean Recall@20 and NDCG@20 over five fixed seeds each within max(5% relative, 0.005 absolute) of an official-code rerun under the same protocol), or fails determinism/licensing under the pinned environment, fall back to a **self-contained, independently implemented standard Hypergraph Neural Network (HGNN, Feng et al., AAAI 2019, DOI 10.1609/aaai.v33i01.33013558)** with fully documented equations. The fallback triggers **only** on the predeclared failure condition, is disclosed as a protocol deviation, and is never chosen after inspecting results.

> This decouples the benchmark from the survey's §4.6 worked example (DyHuCoG as a literature case only) and keeps reproducibility claims defensible. Do **not** write "HNN/HGCN/HCCF-style" or "pick a small, defensible set" in the executable protocol. Until the port validation artifact exists, write: **HCCF is selected as the primary backbone; a port will be pinned and validated before preregistration**. After validation, replace this with the exact fork commit, license, lockfile/container hash, validation protocol, tolerance, and validation result.

## A.5 Attribution families to compare

**Players = interactions (u,i). Value = multi-objective ranking/diversity utility**, defined independently in this paper (§A.6); no prior codebase or game implementation is reused.

| Label | Attribution | Nature | `v(S)` baseline |
|---|---|---|---|
| `uniform` | uniform edge weight | **no-attribution control** (not "an attribution method") | — |
| `attention` | **fixed post-hoc attention-style similarity weighting**: `w_{u,(u,j)} = softmax_j(sim(u,j)/τ_att)` with `τ_att=0.1`, computed from frozen training-only item vectors; no learned parameters, no separate training | non-game-theoretic secondary control | — |
| `heuristic-pop` | popularity/degree weighting | heuristic control | — |
| `additive-pref` | additive preference-similarity prior without game aggregation | **matched non-game heuristic (required, review 2.5)** | same preference term, no Shapley averaging |
| `shapley-mc` | preference-aware Monte-Carlo Shapley | **game-theoretic (primary)** | per-user benchmark game $v_u(S_u)=\alpha\mathrm{NDCG@20}_u(S_u)+\beta\mathrm{Diversity}_u(S_u)$ with $\alpha+\beta=1$; $v_{pref,u}(S_u)=v_u(S_u)+\lambda_{pref}\sum_{(u,j)\in S_u}\mathrm{sim}(u,j)$ |
| `shapley-ai` | **candidate estimator** (sampling/importance Shapley variant) — NOT yet fully defined (§A.6d); kept only if precisely specified before registration | exploratory estimator ablation | same as `shapley-mc` |
| `myerson` (exploratory, optional) | communication-graph-restricted Shapley on the hypergraph projection | game-theoretic (structure-aware) | **Myerson value**: the Shapley value of the graph-restricted game `v^g(S) = Σ_{C∈𝒞(g[S])} v(C)`, with `g[S]` the subgraph induced by `S` and `𝒞(g[S])` its connected components; the projection (2-section / incidence / line graph) is a stated method parameter |

Default coalition weights for the primary `shapley-mc` family (chosen here, not inherited from any prior codebase): $\alpha=0.70,\ \beta=0.30,\ \lambda_{pref}=0.20$. There is **no context term** in the benchmark game; `Context` remains only a survey-taxonomy category. These defaults are **not** treated as fixed facts: the seven-tuple sensitivity grid in §A.6c is a **required** robustness check, because the survey itself raises the value-function-arbitrariness critique and the benchmark must not inherit an unexamined weighting scheme across regimes.

> **Keep the method set small and coherent — primary vs. exploratory (review §3).** **Primary set (run on every cell):** `uniform` (no-attribution control), `additive-pref` (matched non-game heuristic), and `shapley-mc` (game-theoretic). `attention` and `heuristic-pop` are secondary controls; `attention` is fixed post-hoc, not learned, so all HCCF training remains identical. `shapley-ai` and `myerson` are **exploratory**: if included, every promised cell must be run (no partial factorial); otherwise drop them from headline claims. (The earlier broken "§C4 resolution" cross-reference is removed — the rule is stated here.)

## A.5a Family-specific weights before common reranking

Every attribution family produces a per-user historical-interaction weight `w_{u,p}` for `p=(u,j)∈N_u`, then the common reranking operator in §A.6 transforms those weights into candidate-item score adjustments. The executable definitions are:

- **`uniform`:** `w_{u,p}=1`.
- **`additive-pref`:** `w_{u,(u,j)}=max(0, sim(u,j))`, where `sim(u,j)=cosine(x_j, x̄_u)` and `x̄_u` is the normalized mean of user `u`'s training-item vectors. This uses the same preference signal as `v_pref,u` but without Shapley marginalization.
- **`attention`:** fixed post-hoc attention-style weighting, `w_{u,(u,j)}=exp(sim(u,j)/τ_att) / Σ_{q=(u,l)∈N_u} exp(sim(u,l)/τ_att)`, with `τ_att=0.1`; no learned parameters and no separate optimization.
- **`heuristic-pop`:** `w_{u,(u,j)}=log(1+deg_train(j)) / max_{q=(u,l)∈N_u} log(1+deg_train(l))`, where `deg_train(j)` is the training-period positive item degree. This deliberately gives higher weight to popular historical items; it is a popularity-amplifying control, not a novelty-promoting control. If the denominator is zero, all weights are set to zero and the user is flagged.
- **`shapley-mc`:** `w_{u,p}=φ_p(v_pref,u)` for the primary setting. Shapley weights can be negative and are retained.

**Common normalization before reranking:** the raw `w_{u,p}` values are not independently z-scored per family before the kernel aggregation. Instead, §A.6 uses the same scale control for every family: divide the kernel sum by `Σ_{p∈N_u}|w_{u,p}|+ε`, then z-score the resulting candidate-level adjustment `a_u(i)` over `C_u`. This preserves Shapley signs while keeping the post-hoc adjustment comparable across families.

## A.6 The game (`game.py`) — exact estimand (P0.4)

Define the estimand precisely (no alternatives, no template):

**Game type (fixed): per-user, frozen-model interaction-mask game.** For each user `u`, the **player set** `N_u` is the set of **training** interactions in `u`'s receptive field (the edges incident to `u` in the frozen graph). A **coalition** `S_u ⊆ N_u` is realized by a **mask operator** on the frozen model's graph at inference: mask the training edges `N_u ∖ S_u` in the **raw user–item incidence matrix**; **all other users' edges remain** in the global graph. The trained HCCF parameters are frozen. For each masked evaluation pass, recompute only the deterministic normalized incidence/propagation operators that are functions of the masked graph and needed to score candidates. Do **not** recompute stochastic contrastive views, dropout masks, or any training-only augmentation during value evaluation unless the final HCCF port proves those tensors enter inference scoring; if they do, the exact deterministic evaluation-time tensors must be specified in `PORT.md`. The model is **not** retrained per coalition and no training signal uses the masked evaluation. `S_u` contains **training edges only** (never evaluation/test interactions).

**Per-user coalition value (well-typed local game — each user has its own `N_u` and `S_u`):**

```
v_u(S_u) = α·NDCG@20_u(S_u) + β·Diversity_u(S_u)          [Context term REMOVED — see below]
v_pref,u(S_u) = v_u(S_u) + λ_pref·Σ_{(u,j)∈S_u} sim(u,j)
```

**Primary attribution estimand (fixed):** the confirmatory `shapley-mc` attribution is the Shapley value of the preference-aware game, `φ_{u,p}=φ_p(v_pref,u)`, with `α=0.70`, `β=0.30`, and `λ_pref=0.20`. The base game `v_u` is used only when the sensitivity tuple sets `λ_pref=0` or when explicitly reporting a diagnostic decomposition. The matched `additive-pref` control uses the same `sim(u,j)` preference term as a direct additive weight without Shapley averaging; `uniform`, `attention`, and `heuristic-pop` do not define a coalition value and are passed through the common reranking equation using their family-specific weights.

**Coalition-value relevance target (fixed, leakage control):** for each user `u`, `v_pref,u(S_u)` is computed exclusively against the **validation relevance target**: the second-last temporally ordered positive interaction. The last positive interaction is reserved exclusively for final testing. Test interactions are never used to define coalition values, estimate attribution weights, tune hyperparameters, construct item vectors, select models, choose `λ_attr`, or select value-function weights. Final Recall/HitRate and NDCG are evaluated once against the frozen test target after all design and tuning decisions have been completed.

**No context term (4.5, fixed):** the benchmark uses a **common interaction-only representation** with no context features, so `Context(S)` is **removed** and `γ` is **dropped**. The primary tuple is fixed as `α = 0.70`, `β = 0.30`, `λ_pref = 0.20`; the sensitivity grid in §A.6c is fixed separately. (If timestamps were ever used as context, the alignment score and its no-future-information property would be defined separately — they are not used here.)

**Aggregation (well-typed):** because each user has a different `N_u` and `S_u`, there is no single global coalition `S`, and a user-specific interaction `p=(u,j)` does not generally exist in another user's game. The primary artifact is therefore the **per-user attribution** `φ_{u,p}`. Any aggregate analysis must be over explicitly defined attributes, e.g. item-level `Φ_i = mean{φ_{u,(u,i)} : (u,i) is a training interaction}`, popularity decile, interaction age, or player-count bin, with denominators reported. Do **not** average incompatible interaction identities as a global `φ_p`, and do not write `v(S)` for a single global interaction set.

**Attribution-to-reranking rule (fixed, P0.4; nonzero influence operator):** the Shapley-family weights enter the **ranking step only** through a leakage-safe item-kernel influence from a user's historical item to each unseen candidate. For a player `p=(u,j)∈N_u` and candidate unseen item `i`, define

```
κ(j,i) = cosine(x_j, x_i)
a_u(i) = Σ_{p=(u,j)∈N_u} w_{u,p} · κ(j,i) / (Σ_{p∈N_u} |w_{u,p}| + ε)
z_base,u(i) = zscore_{i∈C_u}(base_score(u,i))
z_attr,u(i) = zscore_{i∈C_u}(a_u(i))     # if sd=0, z_attr=0
score(u,i) = z_base,u(i) + λ_attr · z_attr,u(i)
```

where `x_i` is the **fixed interaction-only item representation** from §A.3a, `C_u` is the full eligible unseen-item candidate set for user `u`, `ε=1e-12`, and `base_score(u,i)` is the cached frozen backbone's eval-mode score (HCCF/LightGCN: dot product of the final frozen user and item embeddings, with any official inference-time normalization retained and documented in `PORT.md`). **Base-score timing (fixed):** for every trained backbone and user, full-catalogue base scores are computed once in deterministic evaluation mode on the complete frozen training graph and cached before attribution estimation. Coalition masking is used only inside characteristic-value evaluation; base scores are not recomputed for each coalition. The final reranking score combines the cached full-graph base score with the attribution-derived post-hoc adjustment. The primary protocol fixes `λ_attr=0.10`. The secondary, preregistered reranking-strength sensitivity runs **all three values** `λ_attr∈{0.05,0.10,0.20}` on the HCCF primary backbone for both datasets and primary families; it is descriptive and not part of the Holm confirmatory family. For `shapley-mc`, `w_{u,p}=φ_p(v_pref,u)`; for `uniform`, `additive-pref`, `attention`, and `heuristic-pop`, `w_{u,p}` is the family-specific interaction weight transformed through the **same** equation. Negative attributions are retained before normalization; they may decrease the score of candidates similar to negatively attributed historical items. The kernel uses training-derived item vectors only and never uses validation/test labels, review text, or method-dependent learned embeddings. This is a **post-hoc rerank**, not an in-training weight; it does not change the frozen backbone. A unit test in §A.8 must prove that the term changes at least some unseen-item scores on a synthetic graph.

- **Moving-game semantics (5.5, fixed):** the reported attribution is computed on the **final frozen model snapshot** (post-hoc, at a fixed game snapshot). **No in-training refresh is part of the primary protocol.** In-training refresh is an **optional exploratory Study C**, clearly separated, with its own time-indexed definition and efficiency test for that snapshot; it is not the primary attribution and does not validate the primary efficiency test.
- **Leakage controls (4.2 #13):** coalition values use validation relevance only; tuning uses validation data only; `sim` and `x_i` use training interactions only; graph construction never sees validation/test edges; no hyperparameters are inherited from prior test evaluation. `S_u` contains training interactions only.
- **Value-function terms (4.2 #10, fixed):** `NDCG@20_u(S_u)` per §A.7a over the full-catalogue candidate set; `Diversity_u(S_u)` = mean Intra-List Diversity over the coalition's recommended list (§A.7a); `sim(u,i)` = cosine over the fixed interaction-only item representation with the user vector defined in §A.3a. Data source stated for both datasets; cross-dataset feature availability reconciled (audit #15).
- **Objective normalization (4.2 #11, fixed):** `NDCG@20_u` is already in [0,1]; `Diversity_u` is in [0,1]; normalize both to [0,1] and state the resulting range of `v_u(S_u)` (a finite, documented range).
- **Same metric as value and outcome (4.2 #12):** NDCG appears in both `v_u` and the headline outcome, and ILD/Diversity appears in both the value function and secondary diversity outcomes; disclose these as objective-aligned outcomes, not independent validation. **Recall@20 (HitRate@20) on the held-out set** is the primary alternative outcome not directly optimized by the value function; coverage is secondary. Matched non-game objectives are run for every family.
- **Empty-coalition semantics (4.2 #5):** `v_u(∅)` is computed by the **same frozen model with all of user `u`'s training edges masked** (the empty mask), evaluated on the same full-catalogue set — **not** a null/popularity ranker. The baseline value and its rationale are reported.
- Compute exact Shapley where the player set is tractably small (illustrative subgames, synthetic coalitions); use the **Monte-Carlo estimator** for the full interaction player set:
  `φ̂_{u,p} = (1/M) Σ_{m=1..M} [v_pref,u(S_m(p) ∪ {p}) − v_pref,u(S_m(p))]`
  with the sampling law fixed as random-permutation MC and `M=128` (§A.6a), with convergence diagnostics reported rather than an unsupported "99%" claim.
- **Primary attribution: frozen snapshot (post-hoc), no in-training refresh.** The optional **exploratory Study C** may refresh Shapley values every `f` batches during training; if so, give exact equations, detach/gradient behaviour, cache scope, and define the reported value as a **time-indexed online attribution** on a fixed game snapshot per refresh (§A.6b). It is not the primary attribution and does not validate the primary efficiency test.
- **Efficiency identity test:** `Σ_j φ_j(v) = v(N) − v(∅)` must hold for the exact, un-smoothed, unnormalized case to machine precision (§A.8). Note that Monte-Carlo estimates, smoothing, clipping, and normalization generally do **not** satisfy exact efficiency; report the residual and label exact vs. approximate values.

## A.6a Monte-Carlo estimator — explicit sampling law

Averaging `[v(S_m∪{j}) − v(S_m)]` over arbitrary subsets `S_m` is not automatically a Shapley estimator. Specify the sampling law (5.3):
**Exact vs. MC threshold (5.6, fixed):** for a user `u` with `|N_u| ≤ 8`, compute the Shapley values **exactly** by enumeration over all `2^{|N_u|}` coalitions. For `|N_u| > 8`, use the permutation-MC estimator. Report the distribution of `|N_u|` across users, the fraction of users in each regime, and verify that exact enumeration on small real subgames matches the MC estimate within tolerance. Because users differ in `|N_u|`, state that the reported attribution is the **per-user `φ_{u,p}`** and that comparing across users with different player counts is done only in the aggregate, with the player-count distribution reported.

- **Selected MC law: random-permutation sampling.** Draw `M=128` independent uniform permutations of `N_u` for every MC-evaluated user. For each player `p`, take `S_m(p)` as the set of players preceding `p` in permutation `m` and record the marginal `[v_pref,u(S_m(p)∪{p}) − v_pref,u(S_m(p))]` for the primary estimator (or `v_u` only for `λ_pref=0` sensitivity tuples). **For uniformly sampled permutations, the estimator is the sample mean of these marginals over `M` permutations** — the `1/|N|!` factor belongs to the *exact* permutation expectation and is **not** multiplied into each ordinary Monte-Carlo sample mean. One sampled permutation supplies predecessor sets for all players; the implementation caches coalition evaluations within a user/permutation and reports the realized number of forward evaluations per user.
- **Variance and convergence diagnostics (fixed):** report the sample variance of marginal contributions per player, user-level MC standard errors, efficiency residuals, and convergence curves for `M∈{16,32,64,128}`. Before confirmatory runs, validate the estimator against (a) exact analytic synthetic games and (b) exact enumeration on all real subgames with `|N_u|≤8`; additionally compute a `M=512` reference on a fixed pilot subset of at most 500 users per dataset to document numerical stability. The primary estimator remains `M=128` regardless of pilot direction; the pilot may trigger a documented feasibility stop before preregistration, not an after-results tuning change. **Feasibility-stop rule before preregistration:** stop and revise the protocol (without confirmatory claims) if any of the following hold on the pilot: estimated full primary Shapley run time exceeds 14 GPU-days on the declared hardware; masked forward-pass failures/NaNs exceed 0.5% of evaluated users; median absolute player-level error versus exact enumeration on `|N_u|≤8` real subgames exceeds 0.01 or the 95th percentile exceeds 0.05; median absolute efficiency residual exceeds 0.01; or the `M=128` versus `M=512` pilot rank correlation of item-level aggregate attributions falls below 0.90. The triggered criterion and revised protocol must be reported before external preregistration.

## A.6b Refresh / smoothing / clipping / normalization (optional Study C only)

These apply only to the **optional exploratory training-control study (Study C)**, not the primary frozen-snapshot attribution. If Study C is run, give exact equations for: what is refreshed every `f` batches (and what is held fixed); the temporal-smoothing update (weights and decay); the clipping rule (bounds, extremes only); the normalization used before propagation weighting; and detach/gradient behaviour for the attribution weights during backprop and cache scope. Define whether the reported value is a fixed-snapshot or time-indexed online attribution, and report the efficiency residual for each refresh snapshot.

## A.6c Required value-function weight sensitivity

Because the survey (SRQ3) itself criticizes the arbitrariness of `v(S)` and the allocation rule cannot correct a poorly chosen value function, the benchmark must include a **required** weight-sensitivity analysis as its own reported result. **Fix an exact grid before registration (5.5/5.7) — no "e.g.":** with `Context` removed and `α + β = 1`, the grid over `(α, β, λ_pref)` is fixed to the **seven tuples** `(0.7,0.3,0.2), (1.0,0.0,0.0), (0.0,1.0,0.0), (0.5,0.5,0.0), (0.5,0.5,0.2), (0.7,0.3,0.0), (0.7,0.3,0.5)`. **Decision: all seven tuples run on the HCCF primary backbone for both datasets** (not on LightGCN — that is declared secondary). **Multiplicity:** this is a **descriptive robustness analysis**, not a confirmatory test; no correction is applied, and it is reported as exploratory. **Control reruns:** `uniform`, `additive-pref`, and `shapley-mc` are all evaluated under each value-function condition so family-rank stability is assessed on a common footing. Include the sensitivity cost in the runtime budget. If the ordering flips under plausible weights, that is a finding (the arbitrariness critique applied to the benchmark), not a tuning artefact.

## A.6d `shapley-ai` must be a precisely-defined estimator

`shapley-ai` is **not** a named published method. Before it can be a benchmark family it needs a full algorithmic identity: the proposal/importance distribution, the correction factor, the seed handling, the exact estimator equation, and a citation or a clear statement that it is an original estimator defined in this paper. If it cannot be precisely specified, drop it from the benchmark rather than leave an ambiguous cell.

## A.7 Attribution-guided re-ranking (`rerank.py`) — an intervention study, not an explanation-evaluation

This applies attribution-derived weights to re-weight propagation/ranking and measures **Recall@K/HitRate@K and NDCG@K** (primary) plus coverage/ILD (secondary) vs. the non-attribution controls. **Scope note (review 1.7/2.5):** this is an *intervention/reranking* study — it tests whether the derived weights are useful ranking features, **not** whether the attribution is a faithful or sufficient *explanation*. It does not by itself answer SRQ3 ("what game theory buys") unless explanation-quality metrics are added (§A.7b) or the claim is explicitly narrowed to intervention. Keep it labeled as an intervention study.

## A.7a Primary metric definitions (used in all result tables)

Report **Recall@K/HitRate@K and NDCG@K for K ∈ {5, 10, 20}** as the paper's primary results. Define them explicitly (mirror the thesis Ch. 4.4 so the survey's protocol is auditable):

- **NDCG@K** = `(1/|U|) Σ_u DCG_u@K / IDCG_u@K`, with `DCG_u@K = Σ_{k=1..K} rel_u,k / log2(k+1)` (binary relevance for implicit feedback; IDCG per user).
- **Recall@K/HitRate@K** = `(1/|U|) Σ_u |relevant_u ∩ R_u@K| / |relevant_u|`. Under the temporal leave-one-out protocol there is one test positive per user, so Recall@K equals HitRate@K; tables label this as `Recall@K (HitRate@K)` for comparability.

Also compute **Catalogue Coverage** = `|∪_u R_u@K| / |I_eligible|` and **Intra-List Diversity (ILD)** = `(2/K(K−1)) Σ_{1≤k<l≤K} [1 − sim(i_k,i_l)]` — these are secondary; they never replace Recall@K/NDCG@K as the headline.

## A.7b If explanation quality is claimed, add explanation metrics

If the paper claims the benchmark validates *explainability* (not just reranking), it must also report at least one explanation-quality metric: ranking fidelity to the explained model, deletion/insertion or sufficiency/comprehensiveness, stability under perturbation, sparsity, sign-consistency, or a model-randomization sanity check. Without these, the benchmark is an intervention study only and §7.3/§8 framing must say so. Coverage/ILD/popularity-shift are exposure metrics, not explanation-faithfulness metrics; fairness claims need explicit group/provider metrics, not coverage alone.

## A.8 Test suite (`tests/`)

Non-negotiable, in priority order:
1. **Efficiency identity.** `Σ_j φ_j(v) = v(N) − v(∅)` to machine precision (exact, un-smoothed subgames). Catches most implementation errors.
2. **Empty coalition.** `v_u(∅)` is computed by the same frozen model with all of user `u`'s training edges masked (the empty-mask convention in §A.6), is finite, and is non-degenerate.
3. **Symmetry on synthetic data.** Two literally identical score columns/players receive equal Shapley values.
4. **Dummy player.** A synthetic player that the implemented value function provably ignores (its mask does not alter scores, candidate set, `x_i`, diversity, or preference term) receives `φ = 0` in exact enumeration and `φ ≈ 0` under MC. A noisy real edge is tested separately as robustness, not as the formal dummy axiom test.
5. **Empty-mask calibration.** The empty-coalition ranking from the same frozen model with all of user `u`'s training edges masked produces the expected deterministic NDCG/HitRate under the full-catalogue candidate set.
6. **Degenerate normalization / weighting.** Constant rows yield zero attribution, no NaN, no inf.
7. **Split reproducibility.** Frozen splits reload identically across stages (config-hash check).
8. **Backbone determinism.** Same seed → same embeddings/weights (to the extent the backbone allows).
9. **Reranking nonzero intervention.** On a synthetic graph with at least one unseen candidate similar to a historical item, the kernel attribution term changes at least one unseen-item score relative to `z_base`; if `z_attr` has zero variance, the test must fail unless the synthetic fixture is degenerate by construction.
10. **Mask locality.** Masking one user's edge changes only the intended raw incidence entries before deterministic propagation reconstruction; other users' raw edges remain unchanged.
11. **Item-vector isolation.** Modifying validation or test records without changing training data leaves every item vector `x_i` and the item-matrix hash unchanged.
12. **Cached base-score invariance.** Coalition-value masking never mutates the cached full-graph `base_score`; repeated deterministic eval-mode scoring on the complete frozen graph gives identical scores.
13. **HCCF inference-mode determinism.** Repeated masked and unmasked HCCF evaluation passes under fixed model state produce identical scores and do not invoke dropout, stochastic contrastive augmentation, or training-state updates.

Tests 1, 3, and 4 are the ones that would catch a wrong paper rather than a crashed run.

## A.9 Runtime budget

| Stage | ML-1M | Amazon-Book (50k-user sample) |
|---|---|---|
| Raw download + parse | < 1 min | 20–40 min, **one-off** |
| Subsample, filter, split | < 1 min | 2–5 min |
| Backbone training (LightGCN) | 5–15 min | 30–60 min |
| Backbone training (hypergraph GNN) | 15–30 min | 1–2 h |
| Post-hoc Shapley estimation (frozen snapshot, no refresh) | 10–25 min | 30–90 min |
| Re-ranking + metric eval | 5–15 min | 20–60 min |
| **Full pipeline, one seed** | **~45 min–1.5 h** | **~2–4 h** |
| Five seeds × both backbones × both datasets × default families | ~1–2 GPU-days (ML-1M dominated) | ~2–4 GPU-days (Amazon-Book, 50k-user sample) |

Cache parsed/subsampled/filtered frames to Parquet keyed by the config hash. If any stage runs an order of magnitude over these, something is wrong (most likely dense ops on the full user–item matrix instead of the candidate slice).

**Scope note (5.9):** "one seed" in the table = one backbone on one dataset for the **primary family set only** (uniform/additive-pref/shapley-mc), excluding the sensitivity grid and any exploratory methods. All numbers are **pilot estimates**; state the hardware, concurrency, and included cells when measured. The weight-sensitivity grid and any exploratory cells are budgeted separately and must be added to the total if run.

## A.10 Statistical analysis — unit of analysis and inference plan

This section resolves the pairing-unit ambiguity that invalidates a bare "paired t-test" claim. The unit of analysis and the inference plan are **fixed before analysis**:

- **Primary statistical estimand (fixed): conditional user-population effect.** The primary inferential target is the distribution of paired user-level outcome differences **conditional on the five preregistered trained models**. Users are resampled within each training seed, and seed-specific estimates are aggregated. Variation across the five seeds is reported separately and descriptively. The resulting confidence intervals must **not** be interpreted as fully representing uncertainty over the population of all possible training initializations.
- **Unit of analysis: per-user paired differences**, paired **within the same seed** (same split, candidate set, negative-sampling stream, and user). Users evaluated by one trained model share model-level randomness, so they are **not** independent model replicates; per-user n is large, so tiny p-values can arise for trivial effects.
- **Seeds.** The 5 seeds quantify training variability descriptively; they are **not** the pairing unit and are too few for strong model-level generalization claims. Report each seed's effect separately, mean ± std across seeds, and the distribution of per-user differences; treat per-seed pooling as secondary/descriptive.
- **Primary contrast family (P0.5 — SELECTED, Option B, concrete):** the **confirmatory primary family is `shapley-mc` vs `uniform` on NDCG@20 and Recall@20 × 2 datasets on the HCCF primary backbone = 4 tests** in one Holm–Bonferroni family. **Every other cell is declared secondary/exploratory**: the LightGCN backbone, all other cutoffs (@5, @10), and the `additive-pref`/`attention`/`heuristic-pop` comparisons are secondary and are **not** jointly corrected (a separate, stated secondary family, or reported without correction and labelled exploratory). This is the **single selected plan** — no A/B/C menu remains. (The 8-test framing is the full design; the confirmatory family is deliberately narrowed to the 4 primary tests.)
- **Inference procedure (fixed):** per-user paired differences are computed **within the same seed** (same split, candidate set, and user). Confirmatory inference uses a **seed-clustered bootstrap**: resample users within each seed, aggregate the seed-level estimates, and build a 95% cluster bootstrap CI; the 4 primary tests are corrected with Holm–Bonferroni. A mixed-effects/hierarchical model over users with seed as a random effect is a **fixed secondary sensitivity analysis**, not a conditional replacement for the bootstrap primary analysis. Paired t-test and Wilcoxon are **sensitivity analyses only** (assumptions stated).
- Report per-seed values, the cluster bootstrap CI, the exact unit/test/alternative/correction, and the number of comparisons in every significance caption.
- **Seed-clustered inference (required, 5.8):** use the fixed cluster bootstrap that resamples users within each seed and aggregates seed-level estimates while preserving seed-level dependence. The mixed-effects/hierarchical model is reported only as a secondary sensitivity analysis. Reporting per-seed means does not repair primary p-values that treat users as independent.
- **Reporting.** Report user-level paired mean difference, bootstrap confidence interval, median and distribution of user-level differences, each seed's effect separately, mean and standard deviation across seeds, and user-conditional Cohen's `d_z` explicitly labelled descriptive. Paired t-test and Wilcoxon signed-rank are **sensitivity analyses** (assumptions stated), not the sole basis.
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

## B.1a Primary result tables — Recall@K/HitRate@K and NDCG@K (filled only with realized results)

The benchmark's main-text primary deliverable is a result table reporting **Recall@K/HitRate@K and NDCG@K (K ∈ {5,10,20})** for the **primary family set** (`uniform`, `additive-pref`, `shapley-mc`) on the **HCCF primary backbone** and both datasets, **after** the experiments run; LightGCN is a secondary/supplementary backbone; exploratory rows (`attention`, `heuristic-pop`, and any `shapley-ai`/`myerson` kept) are clearly labelled. These tables are **populated only with realized numbers**; no predicted point values are placed here. All hypotheses below are directional only ("we test whether X > Y"); no exact figures, no bolded winners, and no mean ± std placeholders are pre-filled, so there is no risk of a placeholder being mistaken for a result.

- **Result table (main text, §7.2 of `Paper_Structure.md`):** one table per dataset for the **primary families on the HCCF primary backbone**, with columns NDCG@5/10/20 and Recall@5/10/20; cells filled with realized mean ± std over seeds; significant differences flagged after the selected Holm correction (§A.10). Generated by `report.py` from raw outputs only.
- **Prediction register (this Part B):** directional hypotheses + falsification contingencies, stored separately from the results and referenced by the external pre-registration (see §B.0).
- LightGCN results (secondary backbone) are reported in the supplementary Online Resource, not the main primary table; no cells are pre-computed.

## B.2 BQ1 — ranking quality of game-theoretic vs. non-game-theoretic attribution

**Directional hypotheses (we test whether), stated neutrally — not results.**

| Comparison | Hypothesis | Confidence |
|---|---|---|
| `shapley-mc` vs `uniform` (no-attribution control) | `shapley-mc` ≥ `uniform` on NDCG@20 and Recall@20 | **medium-high** (motivated by the hypothesis that principled contribution weighting helps ranking; to be tested) |
| `shapley-mc` vs `attention` | `shapley-mc` ≥ `attention` | medium |
| `shapley-mc` vs `heuristic-pop` | `shapley-mc` ≥ `heuristic-pop` | medium |
| `shapley-ai` vs `shapley-mc` | no directional claim; an estimator-ablation comparison | medium (estimator ablation) |
| `myerson` vs `shapley-mc` | no directional claim; exploratory | low |

**Primary contrast (predeclared):** `shapley-mc` versus the `uniform` no-attribution control on **NDCG@20 and Recall@20/HitRate@20** for each dataset on the **HCCF primary backbone**. All other cutoffs, metrics, families, and ablations are secondary/exploratory. This contrast set and its correction family are fixed before analysis (§A.10). Do not draft the Results section narrative around an assumed winner.

## B.3 BQ2 — coverage and diversity (directional hypotheses)

**Hypotheses (we test whether):** attribution-guided weighting changes catalogue coverage and intra-list diversity relative to the uniform control. **Because "without degrading accuracy" needs a defined tolerance (5.3), we do not predeclare a non-inferiority margin; accuracy (NDCG@20/Recall@20) and diversity (coverage/ILD) are reported as **separate descriptive outcomes**.** No magnitude is predicted; the direction of any coverage/ILD difference and whether it is significant (as a secondary outcome, not part of the primary contrast family) are measured. Head/tail exposure shift is reported descriptively, not predicted a priori.

## B.4 BQ3 — post-hoc attribution cost and stability (hypotheses, not results; 5.2)

Because the primary protocol is a **frozen-model post-hoc mask game** (no in-training refresh), BQ3 is redefined as **post-hoc cost and variance decomposition**, not training-refresh cost:

**Revised BQ3:** What computational cost, Monte Carlo approximation error, training-seed variability, and reranking-sensitivity are observed under the frozen protocol? Report separately: `Var_seed` (training-seed variability, descriptive over the five seeds), `Var_MC` (MC permutation variability with model/data fixed), `Var_lambda` (reranking-strength sensitivity over `λ_attr∈{0.05,0.10,0.20}`), and `Var_sample` only if multiple Amazon user samples are actually generated. Do not collapse these into a single vague "per-run variance" claim.

**In-training refresh is NOT a BQ3 primary hypothesis.** The optional **Study C** (training-refresh attribution) is a separately labelled exploratory experiment with its own estimand, losses, seeds, runtime, and statistical plan; it is reported separately and does not feed the primary BQ3, main tables, preregistration requirements, or required runtime budget.

## B.5 BQ4 — cross-dataset comparison (descriptive, 5.4)

**Hypothesis (neutral, descriptive):** we compare the effect of attribution-guided reranking across the two datasets. This is a **descriptive cross-dataset comparison, not a causal density claim** — MovieLens-1M and Amazon-Book differ on many axes (domain, rating process, catalogue size, metadata) and Amazon-Book is a custom sample. **No effect scale is assumed a priori**; absolute and relative effect sizes are reported descriptively. Because the domains differ, "the sparse regime causes a larger effect" is **not** claimed; only the observed direction/magnitude is reported.

## B.6 Falsification and contingencies

| If this happens | What it means | Contingency |
|---|---|---|
| `shapley-mc` does **not** beat `uniform` on NDCG@20 | the game-theoretic premise does not transfer to the benchmark backbones | Report honestly; the survey's empirical grounding weakens to the clustering/prior-work lineage, and the benchmark becomes a negative result (still informative) |
| `shapley-mc` = `attention` everywhere | attribution adds nothing over the fixed post-hoc attention-style similarity control | Report as a finding; soften BQ1 framing; the survey's critical analysis (SRQ3) must then emphasize where game theory is relabeled reweighting |
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
