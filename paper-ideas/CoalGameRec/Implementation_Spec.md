# CoalGameRec — Technical Implementation Specification and Registered Predictions

**Companion to:** `Paper_Structure.md` (the paper blueprint) and `spec.md` (the scope/venue/methodology spec). `spec.md` says *what the paper argues and where it is submitted*; `Paper_Structure.md` says *how the manuscript is laid out*; this file says *what to build for the benchmark and what to expect when it runs*.
**Status:** pre-implementation. Every number in Part B is a **prediction made before running anything**, not a result.
**Reuse:** `stats.py` (paired tests, Holm–Bonferroni, Cohen's d_z) and the clustering/quality diagnostics from `ActionShap/code/` port over with essentially no change. The DyHuCoG-style hypergraph backbone is reused **only if the reproducibility gaps identified in the SignalShap audit are fixed first** (see §A.1 and §R). Otherwise build the benchmark on LightGCN + an independently documented hypergraph GNN.
**Target venue:** *Discover Artificial Intelligence* (Q1) — the benchmark is a **secondary, supporting** deliverable that grounds the survey taxonomy; it must stay small.

---

# PART A — IMPLEMENTATION

## A.0 What this implementation is for

The paper is a **survey first**. The benchmark exists to do three things and nothing more:

1. **Ground the taxonomy** — instantiate the game-theoretic attribution families the survey categorizes (Shapley, structure-aware/Myerson, heuristic, attention) on real recommenders, so the survey's claims about what these methods *do* rest on a reproducible artifact rather than only on cited papers.
2. **Provide one clean empirical comparison** between game-theoretic and non-game-theoretic attribution under a shared protocol (the BQs in `spec.md` §2).
3. **Be a continuity artifact** with the author's published DyHuCoG work (same two datasets, same metrics), so the survey reads as a continuation of the group's experimental line.

It is **not** a method bake-off, a new-architecture contribution, or a large-scale study. Keep the method set small and the compute modest.

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
python = 3.12
numpy >= 2.4, < 2.5
scipy >= 1.18
scikit-learn >= 1.6
pandas >= 2.2
torch >= 2.0            # PyTorch for the GNN backbones (CUDA if available)
dgl >= 2.0 (optional)   # only if the hypergraph backbone is built on DGL
pyyaml, tqdm, pytest, matplotlib
```

GPU optional but recommended (RTX 4090 or equivalent, matching the thesis). The benchmark should also be runnable on CPU at reduced scale for reproducibility testing.

## A.3 Data layer (`data.py`)

The two benchmarks are **MovieLens-1M and Amazon-Book**, matching the group's published DyHuCoG evaluation so the survey reads as a continuation of the same experimental line.

| | MovieLens-1M | Amazon-Book |
|---|---|---|
| Source | GroupLens `ml-1m.zip` | **Raw** Amazon Reviews 2018, Books 5-core + `meta_Books` |
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
| `shapley-ai` | sampling/importance-based Shapley variant | game-theoretic (estimator ablation) | same as `shapley-mc` |
| `myerson` (optional) | communication-graph-restricted Shapley on the hypergraph projection | game-theoretic (structure-aware) | Shapley restricted to connected coalitions |

Default coalition weights (from DyHuCoG/thesis): $\alpha=0.60,\ \beta=0.25,\ \gamma=0.15,\ \lambda_{pref}=0.20$. Report sensitivity to these in the paper (§7.7 / sensitivity).

> **Keep the method set small.** 6 attribution families × 2 backbones × 2 datasets is already the upper bound. Drop `myerson` and/or `shapley-ai` if the runtime or review scope demands it. The benchmark is illustrative.

## A.6 The game (`game.py`)

Define the characteristic function over a coalition of interactions `S ⊆ N` as the multi-objective utility:

```
v(S) = α·NDCG@20(S) + β·Diversity(S) + γ·Context(S)
v_pref(S) = v(S) + λ_pref·Σ_{(u,i)∈S} sim(u,i)
```

- Compute exact Shapley where the player set is tractably small (illustrative subgames, synthetic coalitions); use the **Monte-Carlo estimator** for the full interaction player set:
  `φ̂_j = (1/M) Σ_{m=1..M} [v(S_m ∪ {j}) − v(S_m)]`
  with `M` chosen from the convergence analysis (predict M≈50 at ~99% accuracy per DyHuCoG).
- Refresh Shapley values every `f` batches (default f=10) during training; apply light temporal smoothing and clip extremes before normalization.
- **Efficiency identity test** `Σ_j φ_j = v(N)` must hold for the exact case to machine precision (see §A.8).

## A.7 Attribution-guided re-ranking (`rerank.py`)

To test the *explanation→improvement* direction that the survey's critical analysis discusses, apply attribution-guided re-ranking: use Shapley-derived interaction weights to re-weight propagation/ranking and measure NDCG/Recall/coverage/ILD vs. the non-attribution backbones. This is a small, contained experiment that grounds the survey's "attribution→intervention" discussion without claiming a new architecture.

## A.8 Test suite (`tests/`)

Non-negotiable, in priority order:
1. **Efficiency identity.** `Σ_j φ_j = v(N)` to machine precision (exact subgames). Catches most implementation errors.
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

---

# PART B — REGISTERED PREDICTIONS

> **Read this as a pre-registration.** Everything below is what I expect *before* running. Recording it now is what makes the results falsifiable rather than rationalized afterwards. When real numbers arrive, report them against this table and **flag every miss explicitly** — a missed prediction that is discussed is a strength, a quietly revised prediction is misconduct.

## B.1 Pipeline-level quantities

| Quantity | ML-1M | Amazon-Book | Confidence |
|---|---|---|---|
| Users after filter | ~6,040 | 25,000–50,000 | **low** — depends on sampling |
| Items after filter | ~3,400–3,700 | 30,000–90,000 | **low** — same caveat |
| Density | ~4.5% | 0.03–0.08% | medium |
| NDCG@20, LightGCN backbone | 0.20–0.22 | 0.02–0.03 | medium (matches published baselines) |
| NDCG@20, hypergraph GNN backbone | 0.22–0.28 | 0.025–0.035 | low |
| Uplift of `shapley-mc` over `uniform` (NDCG@20) | +3% to +8% relative | +5% to +12% relative | low |

**Confidence on the Amazon-Book column is deliberately low** — the rebuilt, subsampled split has no published counterpart, so these are extrapolations from the density regime. The prediction worth holding is the **direction** of the contrast, not the levels.

## B.2 BQ1 — ranking quality of game-theoretic vs. non-game-theoretic attribution

| Comparison | Predicted NDCG@20 ordering | Confidence |
|---|---|---|
| `shapley-mc` vs `uniform` | `shapley-mc` > `uniform` | **high** (the whole DyHuCoG premise) |
| `shapley-mc` vs `attention` | `shapley-mc` ≥ `attention` | medium |
| `shapley-mc` vs `heuristic-pop` | `shapley-mc` > `heuristic-pop` | high |
| `shapley-ai` vs `shapley-mc` | within ±1% of each other | medium (estimator ablation) |
| `myerson` vs `shapley-mc` | near-parity, `myerson` may help on sparse | low |

**Headline prediction:** the game-theoretic attribution (`shapley-mc`) improves NDCG@20 and Recall@20 over all non-game-theoretic baselines, with the largest relative gain on the sparse Amazon-Book regime (the density-contrast argument).

## B.3 BQ2 — coverage and diversity

| Quantity | ML-1M | Amazon-Book | Prediction |
|---|---|---|---|
| Coverage gain of `shapley-mc` vs `uniform` | +8% to +16% relative | +15% to +30% relative | high |
| ILD gain of `shapley-mc` vs `uniform` | +4% to +10% relative | +8% to +15% relative | medium |
| Head/tail coverage shift | attribution pushes toward tail | stronger on sparse | medium |

**Prediction:** game-theoretic weighting broadens catalogue coverage and intra-list diversity without sacrificing accuracy (the accuracy–diversity trade-off is *partially* resolvable, per the thesis RQ4). Watch for whether the ILD gain is significant or merely descriptive.

## B.4 BQ3 — training stability and cost

- MC-Shapley refresh adds ~1.5–2× training time over the plain backbone (matching DyHuCoG's ~1.78×).
- **Variance:** `shapley-mc` per-run NDCG std should be comparable to the backbone's own seed variance (no instability introduced). If variance balloons, the smoothing/clipping in §A.6 needs attention.
- Memory: +20–40% over plain backbone (Shapley/attention weight caches).

## B.5 BQ4 — cross-regime generalization

| Dataset | Predicted NDCG@20 rank of `shapley-mc` | Prediction |
|---|---|---|
| MovieLens-1M (dense) | 1st | high |
| Amazon-Book (sparse) | 1st | medium |

**Prediction:** the game-theoretic method holds the top ranking on both, with the *relative* margin over baselines larger on sparse Amazon-Book — supporting the survey's claim that principled contribution weighting is most valuable where interactions are thin and popularity dominates.

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
2. LightGCN backbone trains to sane NDCG. **Gate:** matches published baseline within tolerance.
3. Hypergraph backbone pinned/reproducible (or fallback chosen). **Gate:** backbone determinism test (test 8).
4. Attribution families + game module. **Gate:** tests 1–4 pass (efficiency, empty, symmetry, dummy).
5. Re-ranking + metrics. **Gate:** BQ2 coverage/ILD computable.
6. Statistics + table/figure emitters.
7. Survey-content integration: map results into the taxonomy/comparison tables; draft the critical-analysis tie-ins.

Steps 4 and 5 are the decision points. If BQ1 and BQ2 land as predicted, the benchmark cleanly grounds the survey. If they fail, the contingency table says what to do without improvising.
