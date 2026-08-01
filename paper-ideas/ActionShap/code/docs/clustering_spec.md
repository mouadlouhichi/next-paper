# Static Regime: Implementation Specification

Extracted from the two published clustering papers so the pipeline can be
rebuilt without their code (none was released).

| Ref | Paper | Venue |
|---|---|---|
| **[P]** PRIMARY | *Game Theory Meets Explainable AI* | IJACSA 16(7), 2025, 716–725 |
| **[E]** EARLIER | *Shapley Values for Explaining the Black Box Nature of ML Model Clustering* | Procedia CS 220 (2023) 806–811 |

[P] cites [E] and formally supersedes it, but [E] closes several gaps in [P] —
most importantly the wine variant.

---

## 0. WARNING: the advertised pipeline is not the implemented one

[P] Table V advertises `PCA → K-means → LightGBM surrogate → multi-level SHAP`.
**k-means does not run in PCA space.** Algorithm 1 reads:

```
X_scaled ← StandardScaler(X)
X_PCA    ← PCA(X_scaled)          ▹ computed, then NEVER CONSUMED
k*       ← determine via silhouette and Davies-Bouldin
C        ← KMeans(X_scaled, k*)   ▹ full standardized matrix, not X_PCA
M        ← LightGBM(X, {Ci})      ▹ ORIGINAL unscaled features
φ        ← SHAP
```

Corroborated independently by [P] §IV.F: *"Clustering was performed using
k = 3 … on the full 11-feature matrix."* The reported wine silhouette of 0.144
is far too low for a 2-D projection and is consistent with 11-D space.

The operative pipeline:

```
X_raw ─ StandardScaler ─ X_scaled ─┬─ PCA(2) ────→ 2-D scatter plots ONLY
                                   └─ KMeans(k=3) → labels
X_raw ─────────────────────────────── LightGBM(X_raw, labels) → TreeSHAP → φ
```

Two consequences for our code: PCA is a visualization device with no modelling
role, and because the surrogate consumes original unscaled features, SHAP
values are natively in original units with no inverse transform needed.

---

## 1. Preprocessing

Scaling strictly precedes PCA (Algorithm 1; §III.A *"X is the standardized
data matrix"*). `StandardScaler`, defaults implied.

**CONTRADICTION — missing values.** [P] Table III says *"handled by omission"*;
[P] §IV.F says *"imputation of missing values"*. No imputer is ever named.
**Omission is better supported**: 420,768 UCI rows − 383,585 reported = 37,183
dropped (8.8%), which reconciles exactly with `dropna()` on the 11 retained
columns. Imputation would have preserved all rows.

**GAP — wind direction is categorical and never addressed.** `wd` is a string
with 16 compass levels. `StandardScaler` and `PCA` both raise on raw strings,
so the stated pipeline cannot execute. Encoding is NOT STATED — not one-hot,
ordinal, label, nor cyclical.

**Outliers:** raised in [E] §4.1 only to be set aside; no removal performed.

**Temporal columns silently dropped.** Feature types are called *"Numeric and
temporal"* but no year/month/day/hour appears among the 11 features. `RAIN` is
excluded without comment. Rows are treated as i.i.d.

---

## 2. PCA

| Parameter | Value |
|---|---|
| `n_components` | **2** |
| Retained variance at 2 components | NOT STATED |
| `svd_solver`, `whiten`, `random_state` | NOT STATED |
| Input | `X_scaled` |

Chosen for plottability, not variance: [E] §4.2.2 *"to be able to draw all the
wines in one graph."* No scree plot, elbow, Kaiser criterion, or variance
threshold was applied.

**AMBIGUITY.** [P] Table VI reads `2 components (9 PCs ≈ 97 % var.)` for *both*
datasets — the identical figure for two unrelated 11-feature datasets is
suspicious as an independent result, and no variance figure is ever given for
the 2 components actually used.

Eq. 1: $Y = XW$, with $W$ the covariance eigenvectors.
Eq. 2: $r_k = \lambda_k / \sum_i \lambda_i$ — stated but never evaluated. [P]
§III.B glosses this as *"determines feature importance"*, which is wrong: an
explained-variance ratio is a property of a component, not of an input feature.
Nothing downstream depends on the claim.

---

## 3. Clustering

| Parameter | Value |
|---|---|
| Algorithm | `KMeans` (Lloyd) |
| `n_clusters` | **3**, both datasets |
| Input | **`X_scaled`**, full 11-D |
| `init`, `n_init`, `max_iter`, `tol`, `algorithm` | NOT STATED |
| `random_state` | NOT STATED |
| MiniBatchKMeans at 383k rows | not mentioned |

Objective (Eq. 3): $J = \sum_{i=1}^{k} \sum_{x \in C_i} \lVert x - \mu_i \rVert^2$

### Selecting k

[E] sweeps k = 2…11 on inertia, silhouette, and Davies-Bouldin, then applies
the elbow method and picks k = 3. [P] describes the same decision four
inconsistent ways: *"silhouette and Davies-Bouldin"* (Alg. 1), *"DB, silhouette,
and elbow"* (Table IV), *"CH elbow"* (Table V — the only appearance of
Calinski-Harabasz anywhere, and no CH value is ever reported), and a
*"multicriteria evaluation"* (§IV.F). No combination rule, weighting, or
tie-break is given.

### CONTRADICTION — k = 3 is not optimal for wine

| k | Wine Silh. / DB | Air Silh. / DB |
|---|---|---|
| 2 | **0.214 / 1.775** | 0.265 / 1.503 |
| 3 | 0.144 / 2.097 | **0.626 / 0.553** |

For wine, k = 2 dominates k = 3 on *both* metrics, yet k = 3 was adopted as the
*"best overall balance"*. The stated criteria do not produce the stated choice;
k = 3 appears fixed a priori, carried over from [E].

[E] compounds this: *"we obtained a high Davies-Bouldin score and a low
silhouette score for k=3, and therefore chose to cluster into three clusters."*
Both are signals of *poor* clustering; the sentence justifies k = 3 with
evidence against it.

### Non-determinism acknowledged, unresolved

[E] §4.2.3 says multiple restarts *"should"* be considered — prescriptive, not
a report of what was done. No seed, no `n_init`, no variance across runs. All
reported metrics are single-run point estimates.

### No alternative clustering algorithm was run

[P] Table V compares against numbers **quoted from three unrelated published
papers** on different datasets with different pipelines. It is a literature
table, not a controlled comparison, and must not be cited as one.

### UNSPECIFIED — "multi-level clustering", the headline contribution

Claimed in [P]'s abstract, contributions list, Table V, §III.E, and §V.A.
**Never defined and absent from Algorithm 1**, which fits one flat k-means. No
sub-clustering, dendrogram, linkage, recursion, depth, or per-level aggregation
rule exists. There is nothing to reproduce.

---

## 4. LightGBM surrogate

Maps **original unscaled features → cluster label**, 3-class.

| Hyperparameter | Value |
|---|---|
| `num_leaves` | **31** (stated, default kept) |
| `n_estimators` | **100** (stated, default kept) |
| `objective` | multiclass — implied by *"cross-entropy loss"*; literal string NOT STATED |
| everything else | NOT STATED — `learning_rate`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `boosting_type`, `class_weight`, `random_state`, early stopping |

Both stated values *are* the `LGBMClassifier` defaults, so the faithful
reconstruction is stock LightGBM with `objective="multiclass", num_class=3`.

Eq. 6: $\mathcal{L}(\phi) = \sum_i l(y_i,\hat y_i) + \sum_k \Omega(f_k)$, with
$\Omega$ never instantiated. Note $\phi$ denotes the model here, colliding with
$\phi_i$ for Shapley values in Eq. 7.

**Hyperparameter search: none.** The *"brief sensitivity sweep"* covers three
parameters at two values each, one at a time: k ∈ {2,3}, PCA d (only 2
reported), `num_leaves` ∈ {31,63}.

**Fidelity: F1 only.** 0.82 at 31 leaves (4 min), 0.84 at 63 leaves (8 min) —
reported *identically* for both datasets, which is either coincidence or a copy
artifact. Averaging mode NOT STATED (macro/micro/weighted differ substantially
for 3 classes). Train or test set NOT STATED. No accuracy, confusion matrix, or
fidelity metric in the surrogate sense.

**Train/test split: NOT STATED anywhere.** Algorithm 1 fits on full `X`, so
F1 = 0.82 is most likely in-sample.

---

## 5. SHAP attribution

| Aspect | Value |
|---|---|
| Explainer | NOT STATED — `TreeExplainer` implied |
| `feature_perturbation` | **NOT STATED** — interventional vs tree_path_dependent |
| Background data | **NOT STATED** |
| Instances explained | NOT STATED — matters at 383,585 rows |
| Output | multiclass, per-class arrays |

Because perturbation mode and background are both unstated, the reported SHAP
magnitudes cannot be reproduced even with an identical surrogate.

### No exact-vs-Monte-Carlo comparison exists

**Neither paper contains one.** No KernelSHAP baseline, no permutation
sampling, no sample counts. [P] is internally inconsistent: the abstract claims
*"approximated Shapley values"* while §V.C/§VI describe exact computation as a
present cost and approximation as *future work*. The symbol $M$ in the
complexity bound $\mathcal{O}(nd^2 + nk|\mathcal{F}| + k|\mathcal{F}|M)$ is
never assigned a value.

> The file `Figure 2.10- Exact versus Monte Carlo Shapley computation.png` in
> the phd folder is **not from either paper** — it belongs to the thesis.

### Aggregation

Multiclass, then sliced per cluster (Algorithm 1):
`φ_j ← φ[j]`, then `P_j ← argsort(Σ|φ_j|)`. Three levels are produced: global
mean-|SHAP| bar, per-cluster beeswarm, and per-instance force plots.

### CRITICAL — the value function is defined wrongly

[P] §III.C.2 states $v(S)$ *"measures the clustering quality when only features
in S are considered."* That describes refitting the clustering for all
$2^{11} = 2048$ subsets and scoring cluster quality — a different and far more
expensive method, needing no surrogate at all.

What is actually computed is standard SHAP on the surrogate, where
$v(S) = \mathbb{E}[f(x) \mid x_S]$. **Implement the surrogate-based
definition**; treat §III.C.2 as a drafting error. This matters for what the
attributions *mean*: they explain the surrogate's imitation of the partition,
not the partition's intrinsic quality.

### Altair claimed but unreproducible

[P] §III.F lists interactive Altair visualization as a contribution, but all
figures are stock `shap` matplotlib output, which Altair cannot render. No
chart specification is given.

---

## 6. Validation measures

Silhouette (Eq. 4) and Davies-Bouldin (Eq. 5) only, despite §IV.C claiming
*"four widely accepted cluster-quality indices"* and then naming two.

$$s(x) = \frac{b(x)-a(x)}{\max\{a(x),b(x)\}} \qquad
DB = \frac{1}{k}\sum_i \max_{j\neq i}\frac{\sigma_i+\sigma_j}{d(\mu_i,\mu_j)}$$

Distance metric NOT STATED (assume Euclidean). Whether silhouette used a
subsample at 383,585 rows is NOT STATED.

**CONTRADICTION — headline metrics are air-quality-only.** Table V attributes
`Silhouette 0.63, DB 0.55` to *both* datasets. Table VI shows these are the air
values alone; wine k=3 is **0.144 / 2.097**. Table V quotes competing work at
silhouette 0.37 / DB ≈ 1.1, so **the wine result is substantially worse than
the baseline it is favourably compared against** — a fact the aggregation
hides. Benchmark against Table VI per dataset, never Table V.

**No interpretability or stability metric exists.** No faithfulness, no
sufficiency/comprehensiveness, no deletion curves, no SHAP variance across
seeds, no ARI across restarts, no confidence intervals, no significance tests.
Claims of *"thorough statistical validation"* (§VI) are unsupported. Hardware
behind the 4/8 min runtimes is NOT STATED.

---

## 7. Datasets

### Wine Quality — WHITE, 4,898 × 11

Confirmed white on two independent grounds: [E] §4.1 says so explicitly, and
n = 4,898 is exactly `winequality-white.csv` (red = 1,599; combined = 6,497).
[P] never states the variant — **this is the most important gap [E] closes.**
[E]'s reported means (`fixed acidity` 6.854788, `volatile acidity` 0.278241)
match the white file exactly.

Canonical UCI order, `quality` excluded as the 12th column:
`fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol`

> [E] Table 1's sample matrix (`fixed acidity` 1.4, etc.) contains values
> outside the real data range. It is illustrative filler — do not use as a
> fixture.

### Beijing Multi-Site Air Quality — 383,585 × 11

`PM2.5, PM10, NO2, SO2, CO, O3, TEMP, PRES, DEWP, wd, WSPM`. `RAIN` excluded
without comment; `year, month, day, hour, No, station` also excluded.
12 stations × 35,064 rows = 420,768, less 37,183 dropped, gives the reported
count — which supports omission over imputation and implies all stations were
concatenated, though neither is stated.

**Unresolved:** §IV.A says *"PM2.5 was selected as a proxy target"*, but the
pipeline is unsupervised, the surrogate predicts cluster labels, and PM2.5 is
both a listed feature and SHAP-attributed. The three statements cannot all
hold; treat the proxy-target sentence as vestigial.

### Reported feature importance — validation targets

**Wine global:** density, pH, fixed acidity dominant; sulfur dioxide compounds
and residual sugar moderate.

**Air quality global: TEMP, DEWP, PRES dominant** — meteorological, not
pollutant — with CO, NO₂, PM10, PM2.5 only moderate.

> This is directly load-bearing for ActionShap. The dominant attributed drivers
> in the air-quality task are exactly the variables an operator cannot control,
> which is the phenomenon the Actionability Score exists to expose. It also
> means the negative control in §4.5.2 has a real effect to find rather than a
> hypothetical one.

**CONTRADICTION — abstract vs results.** [P]'s abstract claims *"density and
total sulfur dioxide"* for wine and *"PM2.5 and NO2"* for air. Both conflict
with §IV, which ranks density/pH/fixed acidity and TEMP/DEWP/PRES respectively.
Trust §IV. Note the abstract's framing is what makes the air result sound
domain-plausible; the actual finding is the more surprising one and receives no
discussion.

> Figs. 7 and 8 assign incompatible regimes to the same cluster indices — the
> narratives appear rotated by one. Since k-means indices are arbitrary, do not
> validate index-level agreement; check that three regimes of these *types*
> emerge.

---

## 8. Ambiguities requiring an implementer decision

### Tier 1 — blocks reimplementation

1. "Multi-level clustering" never defined; nothing to implement.
2. Clustering input space contradictory across three statements → use `X_scaled`.
3. `wd` encoding NOT STATED → pipeline cannot execute on raw UCI data.
4. Missing-value strategy contradictory → use omission.
5. SHAP `feature_perturbation` NOT STATED.
6. SHAP background dataset NOT STATED.
7. $v(S)$ defined as cluster quality but implemented as surrogate output.
8. Train/test split NOT STATED; F1 may be in-sample.

### Tier 2 — blocks reproducing reported numbers

9. No `random_state` anywhere, for any component.
10. All KMeans control parameters NOT STATED.
11. All LightGBM hyperparameters beyond two NOT STATED.
12. F1 averaging mode NOT STATED.
13. k=3 is not optimal for wine yet was adopted.
14. "9 PCs ≈ 97%" inconsistent with `n_components=2`.
15. Headline 0.63/0.55 are air-only, presented as both.
16. Four inconsistent statements of the k-selection criterion.
17. k-sweep range differs between papers (2–11 vs 2–3).
18. Number of instances explained by SHAP NOT STATED.

### Tier 3 — documentation gaps

19. Station handling NOT STATED. 20. Temporal columns dropped silently.
21. PM2.5 target/feature contradiction. 22. Abstract contradicts results.
23. Figure cluster profiles rotated. 24. Altair unreproducible.

Also unstated: [E]'s scaler; all library versions; hardware; silhouette
distance metric and subsampling; the point at which `quality` is dropped;
force-plot "median SHAP magnitude" axis. No code released by either paper.

---

## 9. Reimplementation defaults

Every `IMPLEMENTER CHOICE` below is a selection made here, **not** paper
content. Cross-reference §8 before publishing any comparison.

| Decision | Value chosen | Why |
|---|---|---|
| Seed | 42 | none stated (§8.9) |
| KMeans | `k-means++`, `n_init=10`, `max_iter=300`, `tol=1e-4` | sklearn defaults (§8.10) |
| `wd` encoding | cyclical sin/cos | preserves the 11-feature count and the circular topology; one-hot would break the count (§8.3) |
| Missing values | `dropna()` | reconciles with 383,585 (§8.4) |
| Split | stratified 80/20 | none stated (§8.8) |
| LightGBM | stock defaults + `objective="multiclass"` | matches "default kept" (§4) |
| SHAP | `TreeExplainer`, interventional, background = k-means summary of X | interventional matches the estimand Definition 2 targets (§8.5, §8.6) |
| F1 | report macro, micro, and weighted | mode unstated (§8.12) |

### Validation checklist

| Target | Value | Confidence |
|---|---|---|
| Wine shape | 4,898 × 11 | High — exact, cross-confirmed |
| Beijing shape | 383,585 × 11 | High — arithmetic reconciles |
| Wine silhouette / DB @ k=3 | 0.144 / 2.097 | Medium — seed-dependent |
| Air silhouette / DB @ k=3 | 0.626 / 0.553 | Medium — also `wd`-encoding dependent |
| Surrogate F1 | ≈ 0.82 | Low — averaging and split both unstated |
| Wine top-3 | density, pH, fixed acidity | High — both papers agree |
| Air top-3 | TEMP, DEWP, PRES | Medium — contradicts [P]'s own abstract |
| Cluster index ↔ profile | — | **Do not validate**; arbitrary and internally inconsistent |

---

## 10. Consequences for the ActionShap manuscript

1. **§3.1.1 corrected.** The pipeline was described as
   "PCA–k-means–LightGBM–TreeSHAP". PCA is not in the modelling path; the text
   now says so and notes attributions are natively in original feature units.
2. **§4.1.1 corrected.** Wine is now identified as the *white* variant with
   `quality` excluded, and the air-quality count is the exact 383,585.
3. **Surrogate fidelity must be reported.** Attributions explain the
   surrogate's imitation of the partition, so a fidelity number belongs beside
   every static-regime result. Added to §3.1.1.
4. **Do not cite [P] Table V as a controlled baseline comparison.** It quotes
   unrelated papers on different datasets.
5. **The air-quality negative control is real**, not hypothetical: the
   published attributions already rank meteorological variables first.
