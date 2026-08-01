# SignalShap — Technical Implementation Specification and Registered Predictions

**Companion to:** `SignalShap_Paper_Structure.md` (the paper blueprint). That file says *what the paper argues*; this file says *what to build and what to expect when it runs*.
**Status:** pre-implementation. Every number in Part B is a **prediction made before running anything**, not a result.
**Reuse:** `stats.py` and the clustering/quality diagnostics from `ActionShap/code/` port over with essentially no change. Nothing from DyHuCoG is used.

---

# PART A — IMPLEMENTATION

## A.1 Repository layout

```
SignalShap/code/
├── requirements.txt
├── configs/
│   ├── ml1m.yaml
│   └── amazon_book.yaml   # includes the subsample scheme, size, and seed
├── signalshap/
│   ├── __init__.py
│   ├── data.py            # loaders, subsampling, 5-core filtering, temporal split
│   ├── candidates.py      # candidate generation + recall ceiling diagnostic
│   ├── sources.py         # the five scorers, each cached to disk
│   ├── normalize.py       # per-user per-source z-normalization + sigma=0 guard
│   ├── fusion.py          # pairwise logistic ranker, coalition masking
│   ├── game.py            # characteristic function, exact Shapley, per-user
│   ├── segments.py        # behavioural + attribution segmentation
│   ├── adaptive.py        # SignalShap-Fuse
│   ├── baselines.py       # LOO, forward selection, permutation, MC-Shapley
│   ├── metrics.py         # NDCG, Recall, MRR, HR, coverage, Gini
│   ├── stats.py           # PORTED from ActionShap
│   └── report.py          # LaTeX table + figure emitters
├── scripts/
│   ├── build_cache.py     # stages 1-2: data + cached source scores
│   ├── run_game.py        # stages 3-5: coalitions, Shapley, per-user
│   ├── run_segments.py    # stages 6-7: segments + adaptive fusion
│   └── run_all.py
├── tests/
└── results/{raw,tables,figures}/
```

## A.2 Environment

```
python = 3.12
numpy >= 2.4, < 2.5          # same pin as ActionShap; avoids the macOS Accelerate segfault
scipy >= 1.18
scikit-learn >= 1.6
pandas >= 2.2
implicit >= 0.7.2            # BPR-MF
scipy.sparse                 # all score matrices stay sparse until the candidate stage
matplotlib >= 3.9
pyyaml, tqdm, pytest
```

No GPU. No PyTorch. Everything runs on a laptop CPU — this is a claim the paper makes, so keep it true.

## A.3 Data layer (`data.py`)

The two benchmarks are **MovieLens-1M and Amazon-Book**, matching the group's published DyHuCoG evaluation so that SignalShap reads as a continuation of the same experimental line.

| | MovieLens-1M | Amazon-Book |
|---|---|---|
| Source | GroupLens `ml-1m.zip` | **Raw** Amazon Reviews 2018, Books 5-core + `meta_Books` |
| Raw interactions | 1,000,209 | ~27.2M before subsampling |
| Implicit conversion | rating $\ge 4$ counts as positive | rating $\ge 4$ counts as positive |
| Subsampling | none | random **50,000 users**, fixed seed, before re-filtering |
| Filtering | 5-core, applied **iteratively to convergence** | 5-core, iterative, **after** subsampling |
| Item metadata | genres, release year, title | category path, brand, title (from `meta_Books`) |
| Timestamps | present | present in the raw corpus |

> ### Do not use the canonical Amazon-Book split. This is the single most likely way to waste a week.
>
> The Amazon-Book split used in the DyHuCoG paper and throughout the LightGCN/HCCF literature (52,643 users / 91,599 items / 2,984,108 interactions) is distributed as a **fixed 80/20 random split of anonymized user–item index pairs**. It contains no ratings, **no timestamps**, and no item metadata. Against SignalShap's requirements that is fatal in four places at once:
>
> | Component | Needs | Canonical split provides |
> |---|---|---|
> | **REC** | event timestamps for time decay | nothing |
> | **SEQ** | interaction order for Markov transitions | nothing (the split is unordered) |
> | **CB** | item title / category / brand text | nothing (items are integers) |
> | Temporal leave-one-out | timestamps to pick the last interaction | nothing (the split is random, not temporal) |
>
> Three of the five players and the entire evaluation protocol would be unimplementable. **Build from the raw corpus instead** — `Books_5.json.gz` plus `meta_Books.json.gz` from the Amazon Reviews 2018 release — which carries `overall`, `unixReviewTime`, and the metadata fields. Say in §4.1 of the paper that the split is rebuilt rather than reused, and say why; a reviewer familiar with the canonical split will otherwise assume the sequential and recency sources cannot exist.

> ### Subsample before you do anything else.
>
> Raw Books 5-core is on the order of **27M reviews over millions of users** — verify the exact figures against the release page rather than trusting this line. The caching design of §A.5 stores a dense `(n_users, 200)` `float32` array per source, so at millions of users the cache runs to **tens of gigabytes**. The score cache is the enabling design choice of the whole method; at that scale it does not fit, and the laptop-CPU claim goes down with it.
>
> Target roughly **50,000 users**, under a fixed and reported seed, then re-apply the iterative 5-core filter. Three reasons for that number: the cache lands near 200 MB, which is comfortable; the scale is close to the published Amazon-Book split (52,643 users), so continuity with the group's prior work is real rather than nominal; and the density should stay in the ~0.05–0.06% regime, preserving the two-orders-of-magnitude contrast against ML-1M that justifies the dataset pair. Sample **users, not interactions** — sampling interactions would destroy the sequential structure SEQ depends on.
>
> **Do not assume uniform random user sampling will work, and check before building on it.** Drawing 50k users uniformly from a multi-million-user pool retains only a percent or two of each book's reviews, so most items fall below the 5-core threshold and the re-filter can cascade until the sample collapses to a small, unrepresentative core. Run the filter and inspect the survivors *first*, as a ten-minute check, before writing any scorer. If it collapses, the fix is **snowball sampling**: start from a seed set of users, add the items they reviewed, add the other users of those items, and iterate until the user count reaches the target. That preserves the local co-occurrence density that CF needs, at the cost of a sample that is not uniform — which is fine provided the paper says so plainly in §4.1 and reports the procedure. A time-window restriction (for example Books reviews from 2014 onward, then 5-core) is the third option and is the easiest of the three to describe, though it shifts the catalogue toward recent titles. Pick one, report it, and do not switch after seeing results.

Implementation requirements:

1. **Subsample Amazon-Book users first**, with the seed recorded in the config hash, then filter. Doing it in the other order changes the result and is not reproducible from the reported numbers.
2. **Iterate the 5-core filter to a fixed point.** A single pass leaves users below threshold after items are dropped. Loop until neither users nor items change, then record final counts and put *those* in Table 2 — never the pre-filter numbers, and never the canonical Amazon-Book numbers, which this pipeline will not reproduce.
3. **Temporal leave-one-out split per user.** Sort each user's interactions by timestamp; last is test, second-last is validation, rest is train. Break timestamp ties deterministically by original row order. Amazon timestamps are day-resolution, so ties are common and the tie-break rule is load-bearing rather than cosmetic — state it in the paper.
4. **Freeze splits to disk** as user/item integer indices with a config hash, so every downstream stage reads identical splits. Re-deriving splits per stage is how silent inconsistencies enter.
5. Emit a `DatasetStats` record: users, items, interactions, density, mean and median interactions per user, the subsample rate where applicable, and the activity quantile boundaries used later for segmentation.

## A.4 Candidate generation (`candidates.py`)

For each user, retrieve the union of each source's top-$N_g$ items excluding the user's train items, then truncate to $N = 200$ by round-robin across sources so no single source dominates the pool.

**The recall ceiling is the most important diagnostic in the whole pipeline.** If a user's held-out test item is not in their candidate set, their NDCG is zero for *every* coalition, and no fusion weighting can recover it. Therefore:

- Compute and report **Recall@200 of the candidate generator** per dataset. Every NDCG figure in the paper is bounded above by it.
- Report it *before* any attribution result, in §4.2 of the paper, so the reader can calibrate.
- The candidate set is built **once from the grand coalition** and reused for all 32 coalitions, per the blueprint's pre-commitment. This is what keeps the ceiling identical across coalitions and makes $v(C)$ comparisons clean.
- The regenerate-per-coalition variant goes in the appendix. Note it changes the ceiling per coalition, which is exactly why it is not the main result.

```python
def candidate_recall(candidates: dict[int, np.ndarray], test_item: dict[int, int]) -> float:
    """Fraction of users whose held-out item survived retrieval. Upper-bounds every NDCG."""
    return np.mean([test_item[u] in candidates[u] for u in candidates])
```

## A.5 The five source scorers (`sources.py`)

Each returns a dense `float32` array of shape `(n_users, 200)` aligned to the candidate matrix, cached to `.npy`. All are fitted on **train only**.

| $g$ | Method | Key hyperparameters | Score definition |
|---|---|---|---|
| **CF** | BPR-MF via `implicit` | factors 64, reg 0.01, lr 0.01, 100 iters | $\mathbf{p}_u^\top \mathbf{q}_i$ |
| **CB** | TF-IDF content similarity | sublinear tf, min_df 2 | $\cos(\bar{\mathbf{c}}_u, \mathbf{c}_i)$ where $\bar{\mathbf{c}}_u$ is the mean TF-IDF of the user's train items |
| **POP** | Static popularity | — | $\log(1 + \text{train count}_i)$ |
| **REC** | Time-decayed popularity | half-life $\tau$ = 30 days | $\log\big(1 + \sum_{t \in \mathcal{T}_i} e^{-\ln 2\,\Delta t/\tau}\big)$ |
| **SEQ** | First-order Markov | additive smoothing 1 | $\log\big(1 + \text{count}(\text{last}_u \rightarrow i)\big)$ |

**A deliberate design note that de-risks RQ2.** POP and REC are both popularity-flavoured and differ only in time weighting, so they are *engineered* to be substantially redundant. This is intentional. The blueprint's §4.4 predicts the leave-one-out failure using POP↔CF, but POP↔REC is the far more reliable candidate and should be named as the primary expected pair. Including both sources means the paper's central empirical claim does not depend on a redundancy relationship happening to exist in the data — one is built in by construction, while remaining entirely defensible, since time-decayed and static popularity genuinely are two signals a real system would maintain separately.

## A.6 Normalization (`normalize.py`)

Per user, per source, across that user's 200 candidates:

```python
def znorm(scores: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Per-row z-normalization with an explicit constant-row fallback.

    A source that returns one value for every candidate carries no ranking
    information for that user, so the correct normalized column is zero.
    Adding eps to sigma instead would amplify float noise into a fake signal.
    """
    mu = scores.mean(axis=1, keepdims=True)
    sigma = scores.std(axis=1, keepdims=True)
    degenerate = sigma < eps
    z = np.where(degenerate, 0.0, (scores - mu) / np.where(degenerate, 1.0, sigma))
    return z.astype(np.float32), degenerate.ravel()
```

Log the degenerate rate per source per dataset and report it. It is a one-line diagnostic that doubles as direct evidence for which sources are uninformative for cold users, which is the §4.5 story.

## A.7 Fusion (`fusion.py`)

Fusion is a **pairwise logistic ranker**, which is BPR with a logistic loss and, crucially, a **convex problem with a deterministic solution**.

For each training user, sample $R = 10$ (positive, negative) candidate pairs. Build features $\Delta z = z_{\text{pos}} - z_{\text{neg}}$, and for symmetry add $(-\Delta z)$ with the opposite label. Fit `LogisticRegression(fit_intercept=False, C=1.0)`.

Two consequences worth stating in the paper:

- **The fusion layer contributes zero seed variance.** The only stochastic elements are base-scorer training and pair sampling; fix the sampling seed and refits become exactly reproducible. This is what lets §4.7 say the sole variance source is base-model training.
- A refit costs **well under a second** on five features, which is what makes 32 coalitions trivial.

```python
def fit_fusion(z: np.ndarray, coalition: tuple[str, ...]) -> np.ndarray:
    """Refit fusion weights using only the sources in `coalition`.

    Columns outside the coalition are dropped rather than zeroed. The two are
    mathematically equivalent under L2 regularization: the objective is
    separable in an all-zero column's weight, which the penalty drives to
    exactly zero while leaving the retained coefficients untouched. Dropping
    is chosen only to keep the feature matrix small.
    """
```

Dropping rather than zeroing is implementation hygiene, **not** a modelling choice. The two give identical fitted weights on the retained sources: under L2 regularization the objective is separable in the weight of an all-zero column, the data-fit term does not depend on it, so the penalty sends it to exactly zero and the remaining coefficients are unaffected. Verified numerically — agreement to machine precision with and without an intercept, and across regularization strengths spanning four orders of magnitude. Drop the columns because it keeps the feature matrix small, not because zeroing would bias anything.

Do not let this become a source of doubt later: if a coalition result ever looks wrong, the masking convention is not the cause, and the efficiency test in §A.9 is the place to look instead.

## A.8 The game (`game.py`)

```python
v(C) = ndcg_at_10(rank_by(fusion_C)) - ndcg_at_10(null_ranker)
```

with `f_theta[()] = null_ranker` by definition, so $v(\emptyset) = 0$ holds rather than being imposed.

The null ranker $\pi$ shuffles each user's candidate list with a fixed seed. Its expected NDCG@10 over 200 candidates with one relevant item is analytically

$$\mathbb{E}[\mathrm{NDCG@10}(\pi)] = \frac{1}{200}\sum_{r=1}^{10}\frac{1}{\log_2(r+1)} = \frac{4.543}{200} \approx 0.0227,$$

scaled by the candidate recall. **Assert the empirical value matches this to within Monte-Carlo error** — it is a free correctness check on the whole evaluation path, and if it fails, something upstream is broken.

Exact Shapley over all $2^5 = 32$ coalitions, plus per-user values from the same sweep by Proposition 3.

## A.9 Test suite (`tests/`)

Non-negotiable, in priority order:

1. **Efficiency identity.** $\sum_g \varphi_g = v(\mathcal{G})$ to machine precision. Catches most implementation errors on its own.
2. **Per-user consistency.** $\frac{1}{|\mathcal{U}|}\sum_u \varphi_g(u) = \varphi_g$ to machine precision.
3. **Null ranker calibration.** Empirical NDCG@10 of $\pi$ matches $0.0227 \times \text{recall}$.
4. **Empty coalition.** $v(\emptyset) = 0$ exactly.
5. **Symmetry on synthetic data.** Two literally identical score columns must receive equal Shapley values, and their LOO must be zero. This is Proposition 2 as an executable test.
6. **Degenerate normalization.** A constant row yields an all-zero column, no `NaN`, no `inf`.
7. **Dummy source.** A pure-noise column must receive $\varphi \approx 0$.

Tests 1, 2 and 5 are the ones that would catch a wrong paper rather than a crashed run.

## A.10 Runtime budget (CPU, single laptop)

| Stage | ML-1M | Amazon-Book (50k-user sample) |
|---|---|---|
| Raw download + parse | < 1 min | 20–40 min, **one-off** |
| Subsample, filter, split | < 1 min | 2–5 min |
| BPR-MF fit | 2–5 min | 5–15 min |
| Other four scorers | 2–3 min | 5–12 min (TF-IDF over ~90k titles dominates) |
| Candidate generation + caching | 1–2 min | 5–10 min |
| 32 coalition refits | 3–8 min | 8–20 min |
| Segments + adaptive fusion | 2–4 min | 4–8 min |
| **Full pipeline, one seed** | **~15 min** | **~40 min** |
| Five seeds, both datasets | ~5 hours, excluding the one-off parse | |

The Amazon-Book column is the reason the raw parse is listed separately: decompressing and parsing the Books corpus is a one-off cost paid before the first run, and it must **not** be repeated per seed. Cache the parsed, subsampled, filtered frame to Parquet keyed by the config hash and read from it thereafter.

If any stage runs an order of magnitude over these, something is wrong — most likely dense operations on the full user-item matrix instead of the candidate slice.

---

# PART B — REGISTERED PREDICTIONS

> **Read this as a pre-registration.** Everything below is what I expect *before* running. Recording it now is what makes the results falsifiable rather than rationalized afterwards. When real numbers arrive, report them against this table and **flag every miss explicitly** — a missed prediction that is discussed is a strength, a quietly revised prediction is misconduct.

## B.1 Pipeline-level quantities

| Quantity | ML-1M | Amazon-Book | Confidence |
|---|---|---|---|
| Users after subsample + 5-core | ~6,040 | 25,000–50,000 | **low** — depends entirely on the sampling scheme |
| Items after 5-core | ~3,400–3,700 | 30,000–90,000 | **low** — same caveat |
| Density | ~4.5% | 0.03–0.08% | medium |
| Candidate Recall@200 | 0.85–0.95 | 0.45–0.70 | low |
| NDCG@10, null ranker $\pi$ | ~0.020 | ~0.015 | high (analytic) |
| NDCG@10, full fusion | 0.30–0.40 | 0.10–0.20 | low |
| Uplift $v(\mathcal{G})$ | 0.28–0.38 | 0.09–0.19 | low |
| Degenerate-normalization rate, CB | < 2% | 10–30% | medium |

**Confidence on the Amazon-Book column is deliberately marked low, and that is the honest label.** These figures were originally calibrated for Amazon-Beauty, whose post-filter shape is well documented across dozens of papers. The rebuilt, subsampled Books split has no published counterpart, so the ranges are extrapolations from the density regime rather than reads off prior work. Treat the ML-1M column as a genuine pre-registration and the Amazon-Book column as an order-of-magnitude sanity check: the prediction worth holding yourself to is the **direction** of the contrast in §B.2, not these levels.

Recall@200 on Amazon-Book is the number most likely to disappoint, and it matters more here than it did on Beauty because the item catalogue is several times larger. If it lands below 0.5, raise $N$ to 500 and rerun — that is a cheap fix and compromises nothing, since the ceiling applies uniformly across coalitions. Budget for the possibility that $N=500$ is needed and the cache grows accordingly (~500 MB, still fine).

## B.2 Source shares $\bar\varphi_g$ (the headline result)

| Source | ML-1M (dense) | Amazon-Book (sparse) | Reasoning |
|---|---|---|---|
| **CF** | **45–60%** | 20–30% | dense co-occurrence is where matrix factorization thrives; collapses under sparsity |
| **SEQ** | 15–25% | 8–18% | ML-1M sessions are long and ordered; book purchases are further apart in time and closer to independent |
| **POP** | 8–15% | 20–30% | popularity is the fallback when personalization has nothing to work with, and the book catalogue is heavily long-tailed |
| **CB** | 5–12% | 22–35% | title, category and brand text carries the load when interactions are thin, and book metadata is unusually rich |
| **REC** | 3–8% | 5–12% | largely subsumed by POP — see B.3 |

**The prediction that carries the paper:** the ordering *inverts* between datasets. CF leads on ML-1M; CB and POP jointly exceed CF on Amazon-Book. This is the density-contrast argument that justified choosing these two datasets, and it is the single result most worth checking first.

One adjustment relative to the earlier Beauty-based calibration is worth noting: **CB should do somewhat better on Books than it would on Beauty.** Book titles and category paths are long, descriptive, and genuinely discriminative under TF-IDF, whereas beauty-product titles are short and repetitive. If CB comes out weak on Amazon-Book, suspect the metadata join before suspecting the method — a high null rate on `meta_Books` lookups is the likeliest cause, and the degenerate-normalization diagnostic from §A.6 will show it immediately.

## B.3 Shapley versus leave-one-out

| Pair | Predicted rank correlation | Predicted LOO | Predicted $\varphi$ | Confidence |
|---|---|---|---|---|
| **POP ↔ REC** | **0.75–0.95** | both < 0.01 | both 3–15% | **high** |
| POP ↔ CF | 0.35–0.60 | POP small | POP moderate | medium |
| CB ↔ SEQ | < 0.3 | — | — | high (no redundancy expected) |

POP↔REC is the primary expected instance and is close to guaranteed by construction. Expect leave-one-out to report **near zero for both** while Shapley assigns each a real share — the disagreement Proposition 2 describes, appearing on real data.

Predicted headline: leave-one-out under-attributes the redundant pair by **a factor of 5 or more**, and the sum of LOO values falls **30–60% short** of $v(\mathcal{G})$, visibly violating efficiency. That efficiency gap is the cleanest single number in the paper and should be reported explicitly.

## B.4 Segment heterogeneity

Using activity quartiles, from Q1 (coldest) to Q4 (heaviest):

| Segment | Predicted dominant sources | Predicted CF share |
|---|---|---|
| Q1 cold | POP, CB | 10–25% |
| Q2 | POP, CF | 25–40% |
| Q3 | CF, SEQ | 40–55% |
| Q4 heavy | CF, SEQ | 55–70% |

Prediction: **the global ordering inverts in Q1 on at least one dataset**, with POP or CB overtaking CF. Permutation test on between-segment variance expected to give $p < 0.01$, with high confidence — the heterogeneity is large and the test is well powered at these sample sizes.

Lower confidence on attribution segments matching behavioural segments. Adjusted Rand index predicted **0.3–0.6**: related but not identical, which is the interesting outcome. Be genuinely prepared for this to come out near zero, in which case, per the blueprint, drop the claim rather than defend it.

## B.5 Segment-adaptive fusion

| Metric | Predicted gain over global fusion |
|---|---|
| NDCG@10, overall | **+1% to +3% relative** |
| NDCG@10, Q1 cold | +5% to +12% relative |
| NDCG@10, Q4 heavy | −1% to +1% (parity) |
| Best $\lambda$ | 0.4–0.7 |

Be honest that the overall gain is modest. The *shape* — large gains where sources differ most from the global average, parity where they do not — is the convincing part, and it is also exactly what the mechanism predicts, so reporting the shape is stronger evidence than reporting the aggregate.

## B.6 Falsification and contingencies

| If this happens | What it means | Contingency |
|---|---|---|
| POP↔REC correlation < 0.4 | the engineered redundancy did not materialize | report measured redundancy honestly; RQ2 weakens to a demonstration on synthetic data in the appendix |
| LOO and Shapley agree everywhere | no redundancy in this system | **this is a genuine negative result** — report it; the paper still stands on RQ1, RQ3, RQ4, but the framing in §1.3 must soften |
| Shares identical across datasets | density does not drive attribution | the two-dataset justification collapses; add LastFM-2K and reframe around a different axis |
| Segments homogeneous ($p > 0.05$) | attribution is population-uniform | drop Contribution 3; paper becomes three contributions and noticeably thinner |
| Adaptive fusion gives no gain | attribution does not transfer to improvement | drop Contribution 4; report the negative result, which is still informative about the gap between explanation and intervention |
| Some $\varphi_g < 0$ | a source actively harms ranking | **do not hide this** — it is a finding, and normalized shares become uninterpretable, so report raw values only |
| Efficiency check fails | implementation bug | stop; do not interpret anything until test 1 passes |
| Amazon-Book subsample collapses under 5-core | uniform user sampling destroyed item support | switch to snowball or time-window sampling per §A.3, re-report, and do not switch again afterwards |
| `meta_Books` join leaves many items without text | metadata coverage gap, not a CB failure | report the coverage rate; if below ~80%, restrict the catalogue to items with metadata and say so |

The two failures that would genuinely wound the paper are homogeneous segments and no redundancy. Both are cheap to check early, so **run RQ2 and RQ3 before writing any prose** — B.3 and B.4 are answerable within the first full pipeline run.

## B.7 Suggested milestone order

0. **Amazon-Book feasibility spike, before writing any pipeline code.** Parse the raw Books corpus, apply the chosen subsample, run the iterative 5-core filter, and join `meta_Books`. **Gate: the surviving split has a workable user and item count, a density near 0.05%, and metadata coverage above ~80%.** This is a half-day of work that determines whether the dataset choice is viable at all, and every later stage depends on it.
1. Data, splits, candidate recall. **Gate: recall > 0.5 on both.**
2. Five scorers cached. **Gate: degenerate rates sane, no NaN.**
3. Fusion + coalition sweep. **Gate: tests 1–4 pass.**
4. Shapley + B.2 check. **Gate: does the ordering invert between datasets?**
5. LOO comparison + B.3 check. **Gate: does redundancy appear?**
6. Segments + B.4 check. **Gate: is heterogeneity significant?**
7. Adaptive fusion, statistics, emitters.

Steps 4, 5 and 6 are the decision points. If all three land as predicted, the paper is essentially written. If any fails, the contingency table above says what to do without improvising under deadline pressure.
