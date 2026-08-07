# Cooperative-game attribution for explainable graph-based recommendation: a systematic review, taxonomy, and critical LightGCN case study

**Draft status:** manuscript draft for internal development. The empirical case-study sections use completed LightGCN MovieLens-1M and Amazon-Book prototype results. The systematic-review sections still require the registered PRISMA search, screening log, coded corpus, quality assessment, ethics determination, and external archive before submission.

## Abstract

Graph-based recommender systems improve ranking accuracy by propagating user and item signals through interaction graphs, but their predictions remain difficult to explain at the level of individual historical interactions. Cooperative-game methods, especially Shapley-value attribution, offer an axiomatic vocabulary for assigning credit to such interactions, yet the literature remains fragmented across feature attribution, graph explanation, data valuation, provider incentives, and recommender-specific ranking tasks. This article develops a systematic review and taxonomy of cooperative-game attribution for explainable graph-based recommendation. The taxonomy organizes the field by player definition, coalition value, solution concept, attribution role, and graph structure, while distinguishing core recommender evidence from adjacent graph-explanation and data-valuation work. To ground one taxonomy slice, we report a controlled LightGCN case study on MovieLens-1M and a custom temporal Amazon-Book split. The case study evaluates bounded interaction-level Shapley attribution against uniform, additive-preference, attention-style, popularity, and leave-one-out marginal controls under temporal leave-one-out evaluation. Shapley improves over uniform, additive-preference, and attention-style weighting on both datasets. However, leave-one-out marginal attribution matches or exceeds Shapley on NDCG@20, coverage, diversity, and cost-effectiveness. These results show that validation-guided marginal attribution is useful, but full Shapley coalition averaging does not automatically provide superior ranking utility. The findings support a critical view: cooperative-game attribution is valuable when player sets, value functions, baselines, and interventions are explicit, but its axiomatic appeal must be tested against simpler marginal baselines.

**Keywords:** recommender systems; graph neural networks; LightGCN; explainable artificial intelligence; cooperative game theory; Shapley value; leave-one-out attribution; systematic review; taxonomy

## 1. Introduction

Graph-based recommender systems model user preferences through observed interactions, social links, knowledge graphs, or higher-order relational structures. In top-N recommendation, graph neural methods such as LightGCN propagate user and item representations over a bipartite interaction graph and often outperform matrix-factorization baselines. Their success creates a familiar problem: the model can rank items effectively, but it is rarely clear which historical interactions are responsible for a recommendation.

Explainable recommendation addresses this gap by producing evidence for a recommendation in terms of features, items, paths, neighbors, providers, or user actions. Cooperative-game attribution is a particularly attractive candidate because it defines a value function over coalitions of players and allocates the resulting utility through solution concepts such as the Shapley value, Banzhaf value, Myerson value, or interaction indices. For recommendation, the players may be historical user-item interactions, graph edges, item features, signal sources, providers, or training data points. The value may be ranking quality, score margin, exposure, diversity, or another task-specific objective.

Despite this appeal, cooperative-game attribution in graph recommendation is not yet a settled methodology. Published methods differ in what counts as a player, which coalition semantics are used, whether the model is retrained or masked, whether the value function is smooth or discontinuous, and how attribution is used after it is computed. A Shapley value over one game may be theoretically well-defined but scientifically uninformative if the players, value function, or intervention do not correspond to the explanation question.

This paper has two aims. First, it develops a systematic review and taxonomy of cooperative-game attribution for explainable graph-based recommendation. Second, it uses a controlled empirical case study to test one central taxonomy slice: historical interactions as players, validation-based ranking utility as coalition value, and post-hoc attribution-guided reranking as intervention.

The empirical case study is deliberately critical rather than promotional. We do not ask only whether Shapley attribution beats a uniform baseline. We also compare it to additive preference weighting, fixed attention-style weighting, popularity weighting, and leave-one-out marginal attribution. This last comparison is crucial. Leave-one-out marginal attribution measures the effect of removing each interaction from the full coalition. If it performs as well as Shapley, then much of the ranking benefit comes from validation-guided marginal attribution rather than from averaging marginal contributions over many coalition contexts.

### Contributions

1. **Systematic review framework.** We specify a reproducible review protocol for cooperative-game attribution in graph-based recommendation, with core and adjacent evidence tiers.
2. **Five-axis taxonomy.** We organize methods by player set, coalition value function, solution concept, attribution role, and graph structure.
3. **Critical synthesis.** We distinguish what cooperative-game theory genuinely adds from cases where it degenerates into ordinary reweighting or marginal ablation.
4. **Controlled LightGCN case study.** We evaluate bounded interaction Shapley attribution on MovieLens-1M and a custom temporal Amazon-Book split using five seeds, temporal leave-one-out splits, full-catalogue ranking, and paired user-level bootstrap analysis.
5. **Boundary-condition finding.** Shapley improves over uniform, additive-preference, and attention-style controls, but leave-one-out marginal attribution is stronger or equivalent for ranking and substantially more cost-effective.

## 2. Background and preliminaries

### 2.1 Graph-based top-N recommendation

Let `U` be the set of users and `I` the set of items. In implicit-feedback top-N recommendation, the observed training graph contains positive user-item interactions. A graph recommender learns user and item representations and produces a score `s(u,i)` for each candidate item. Evaluation ranks all unseen eligible items for each user and measures whether the held-out item appears near the top.

This case study uses LightGCN as a graph recommender backbone. LightGCN removes feature transformations and nonlinearities from graph convolution and propagates user and item embeddings through the normalized user-item graph. In this paper, LightGCN provides the graph-based empirical setting, while the survey covers broader graph and hypergraph recommender methods.

### 2.2 Cooperative games and Shapley attribution

A cooperative game is defined by a player set `N` and a characteristic function `v: 2^N -> R`. The Shapley value allocates the total value of the grand coalition to players by averaging each player's marginal contribution over all coalition contexts. For player `p`,

```text
phi_p(v) = sum_{S subset N\{p}} |S|! (|N|-|S|-1)! / |N|! * [v(S union {p}) - v(S)].
```

In recommendation, `N` may be a user's historical interactions. The value function may measure validation ranking utility when only a subset of those interactions is active. The Shapley value then assigns credit to historical interactions according to their expected contribution across contexts.

### 2.3 Why Shapley may fail to add ranking value

The Shapley value is not a guarantee of better recommendations. It is an allocation rule conditional on a chosen game. If the value function contains an additive similarity term, then Shapley linearity decomposes that term into the same per-interaction similarity scores used by non-game baselines. If the value function uses hard top-20 NDCG with one validation item, most coalitions may receive zero utility, producing sparse marginal signals. If the intervention uses an external similarity kernel that is misaligned with the graph backbone, the attribution signal may be distorted before it affects ranking.

These issues motivated the case-study design used here. The primary game removes the additive preference term, uses a smooth validation-only pairwise utility, selects a stratified bounded player set, and applies attribution through the backbone-native item embedding space.

## 3. Systematic review methodology

This section will be completed after the registered search is executed. The final paper will report the external protocol registry, search dates, databases, database-specific queries, deduplication rules, eligibility criteria, evidence tiers, screening counts, full-text exclusion reasons, extraction codebook, quality assessment, and inter-coder agreement.

The planned evidence tiers are:

- **Core:** top-N recommendation where a graph or graph-like interaction structure is central and a cooperative-game formulation attributes recommendation-relevant outputs or training signals.
- **Adjacent A:** ranking or recommendation attribution without a graph model.
- **Adjacent B:** graph explanation outside recommendation.
- **Adjacent C:** provider, data, or agent incentive games without model explanation.
- **Background:** general cooperative-game theory and general Shapley-XAI surveys.

The synthesis will be descriptive and structured. A pooled meta-analysis is not planned because the expected corpus is heterogeneous in tasks, metrics, splits, baselines, and attribution targets.

## 4. Taxonomy

The taxonomy has five axes.

### 4.1 Player set

Players may be features, historical interactions, items, users, graph nodes, edges, hyperedges, contexts, signal sources, providers, or agents. In the case study, each player's identity is a historical training interaction `(u,j)` for a fixed user `u`.

### 4.2 Coalition value function

The value function determines what a coalition earns. Typical values include ranking utility, score margin, diversity, coverage, fairness, exposure, regret, model loss, or data value. In the case study, the primary value is a validation-only pairwise log-sigmoid utility comparing the validation positive against fixed validation negatives.

### 4.3 Solution concept

The Shapley value is the primary solution concept considered here. The survey also covers Banzhaf values, Myerson values, Harsanyi dividends, interaction indices, core, and nucleolus where they appear in the literature.

### 4.4 Attribution role

Attribution can be used for post-hoc explanation, model debugging, data valuation, provider credit, fairness auditing, actionability, or intervention. The case study uses attribution as a post-hoc reranking intervention and also reports preliminary explanation diagnostics.

### 4.5 Graph structure

Graph structures include bipartite user-item graphs, knowledge graphs, heterogeneous graphs, hypergraphs, and dynamic graphs. The empirical case study uses a bipartite LightGCN graph. Hypergraph methods are treated in the survey and future-work sections unless a validated HCCF artifact is added.

## 5. Empirical case study

### 5.1 Scope

The empirical study tests one taxonomy slice: interaction players, validation ranking utility, Shapley solution concept, post-hoc reranking intervention, and bipartite graph recommendation. It does not validate the entire taxonomy. It also does not prove that Shapley values are faithful human explanations. Instead, it tests whether cooperative-game attribution improves ranking and whether Shapley averaging adds value beyond simpler baselines.

### 5.2 Data and splits

We use MovieLens-1M and a custom temporal Amazon-Book split built from Amazon Reviews 2018 Books 5-core data. Ratings are converted to implicit positives using `rating >= 4`. For each user, the last positive interaction is the test item, the second-last positive interaction is the validation item, and earlier positives are training interactions. The 5-core eligibility filter is computed from the training-period positive graph to avoid future-information leakage.

**Dataset statistics.** MovieLens contains 6,015 users, 3,114 items, 562,183 training interactions, 6,015 validation interactions, and 6,015 test interactions after processing. Amazon-Book contains 7,417 users, 12,885 items, 109,915 training interactions, 7,417 validation interactions, and 7,417 test interactions after processing.

### 5.3 Backbone and attribution families

The graph backbone is LightGCN with two propagation layers. We train for five seeds `{42,43,44,45,46}`. The attribution families are:

- **uniform:** all historical interactions receive equal weight.
- **additive-pref:** weights are direct positive similarity to the user's training profile.
- **attention:** fixed attention-style softmax over profile similarity.
- **heuristic-pop:** popularity-weighted historical interactions.
- **loo-marginal:** leave-one-out marginal value `v(N)-v(N\{j})`.
- **shapley-mc:** antithetic permutation Monte Carlo Shapley over a bounded stratified set of 24 historical interactions.

The Shapley game uses no additive preference term. Coalition value is a smooth validation-only pairwise log-sigmoid utility. The final reranker applies attribution through the native item embedding space.

### 5.4 Metrics and inference

Final test metrics are HitRate@K and NDCG@K for `K in {5,10,20}`. Since each user has one test positive, HitRate@K is the formal label. We also report catalogue coverage and intra-list diversity at 20.

Inference uses paired user-level differences within each seed and a seed-clustered user bootstrap. The estimand is conditional on the five trained models. We report mean paired differences, 95% bootstrap intervals, user-conditional descriptive `d_z`, and proportions of improved, harmed, and unchanged users.

## 6. Results

### 6.1 Main LightGCN results

The main results are shown in Table 1.

**Table 1. LightGCN test performance, mean +/- standard deviation over five seeds.**

| Dataset      | Backbone   | Method        | HitRate@20          | NDCG@20             | Coverage@20         | ILD@20              |
|:-------------|:-----------|:--------------|:--------------------|:--------------------|:--------------------|:--------------------|
| MovieLens-1M | LightGCN   | uniform       | 0.11737 +/- 0.00102 | 0.04601 +/- 0.00030 | 0.61452 +/- 0.00196 | 0.72176 +/- 0.00022 |
| MovieLens-1M | LightGCN   | additive-pref | 0.11751 +/- 0.00087 | 0.04610 +/- 0.00020 | 0.61137 +/- 0.00281 | 0.72011 +/- 0.00026 |
| MovieLens-1M | LightGCN   | attention     | 0.11800 +/- 0.00165 | 0.04648 +/- 0.00046 | 0.60501 +/- 0.00367 | 0.71703 +/- 0.00021 |
| MovieLens-1M | LightGCN   | heuristic-pop | 0.11754 +/- 0.00110 | 0.04612 +/- 0.00035 | 0.61040 +/- 0.00231 | 0.72051 +/- 0.00023 |
| MovieLens-1M | LightGCN   | loo-marginal  | 0.12519 +/- 0.00230 | 0.04976 +/- 0.00041 | 0.64123 +/- 0.00296 | 0.73496 +/- 0.00018 |
| MovieLens-1M | LightGCN   | shapley-mc    | 0.12555 +/- 0.00084 | 0.04922 +/- 0.00033 | 0.63372 +/- 0.00396 | 0.72782 +/- 0.00019 |
| Amazon-Book  | LightGCN   | uniform       | 0.06679 +/- 0.00253 | 0.02978 +/- 0.00078 | 0.23387 +/- 0.00438 | 0.92104 +/- 0.00142 |
| Amazon-Book  | LightGCN   | additive-pref | 0.06655 +/- 0.00246 | 0.02968 +/- 0.00075 | 0.23387 +/- 0.00423 | 0.92091 +/- 0.00144 |
| Amazon-Book  | LightGCN   | attention     | 0.06604 +/- 0.00249 | 0.02954 +/- 0.00075 | 0.23454 +/- 0.00388 | 0.92054 +/- 0.00145 |
| Amazon-Book  | LightGCN   | heuristic-pop | 0.06687 +/- 0.00284 | 0.02995 +/- 0.00085 | 0.23379 +/- 0.00433 | 0.92098 +/- 0.00142 |
| Amazon-Book  | LightGCN   | loo-marginal  | 0.07089 +/- 0.00234 | 0.03237 +/- 0.00077 | 0.23551 +/- 0.00418 | 0.92038 +/- 0.00145 |
| Amazon-Book  | LightGCN   | shapley-mc    | 0.07019 +/- 0.00289 | 0.03187 +/- 0.00085 | 0.23424 +/- 0.00394 | 0.92071 +/- 0.00138 |

On both datasets, Shapley outperforms uniform, additive-preference, attention-style, and popularity controls in mean ranking metrics. On MovieLens, Shapley achieves the highest HitRate@20, while LOO marginal achieves the highest NDCG@20, coverage, and ILD. On Amazon-Book, LOO marginal achieves the highest HitRate@20, NDCG@20, and coverage.

### 6.2 Paired contrasts

Table 2 reports paired conditional user-level contrasts for Shapley against the main controls.

**Table 2. Paired Shapley contrasts under the conditional user-population estimand.**

| Dataset      | Contrast                  | Metric     | Mean difference   | 95% CI                | p        | Holm reject   | Improved   | Harmed   | Unchanged   | d_z     |
|:-------------|:--------------------------|:-----------|:------------------|:----------------------|:---------|:--------------|:-----------|:---------|:------------|:--------|
| MovieLens-1M | shapley-mc vs uniform     | NDCG@20    | 0.003216          | [0.002707, 0.003715]  | < 0.0005 | True          | 6.01%      | 2.92%    | 91.07%      | 0.0716  |
| MovieLens-1M | shapley-mc vs uniform     | HitRate@20 | 0.008180          | [0.006517, 0.009875]  | < 0.0005 | True          | 1.52%      | 0.70%    | 97.79%      | 0.0550  |
| MovieLens-1M | shapley-mc vs additive-pref | NDCG@20  | 0.003121          | [0.002606, 0.003628]  | < 0.0005 | True          | 5.95%      | 2.93%    | 91.13%      | 0.0693  |
| MovieLens-1M | shapley-mc vs additive-pref | HitRate@20 | 0.008047        | [0.006384, 0.009742]  | < 0.0005 | True          | 1.51%      | 0.70%    | 97.79%      | 0.0542  |
| MovieLens-1M | shapley-mc vs attention   | NDCG@20    | 0.002740          | [0.002207, 0.003271]  | < 0.0005 | True          | 5.89%      | 3.11%    | 91.00%      | 0.0582  |
| MovieLens-1M | shapley-mc vs attention   | HitRate@20 | 0.007548          | [0.005786, 0.009211]  | < 0.0005 | True          | 1.54%      | 0.79%    | 97.67%      | 0.0495  |
| MovieLens-1M | shapley-mc vs loo-marginal | NDCG@20   | -0.000532         | [-0.000963, -0.000119] | 0.008   | True          | 3.19%      | 4.97%    | 91.84%      | -0.0143 |
| MovieLens-1M | shapley-mc vs loo-marginal | HitRate@20 | 0.000366        | [-0.000899, 0.001696] | 0.575    | False         | 0.68%      | 0.65%    | 98.67%      | 0.0032  |
| Amazon-Book  | shapley-mc vs uniform     | NDCG@20    | 0.002096          | [0.001703, 0.002474]  | < 0.0005 | True          | 2.33%      | 1.36%    | 96.31%      | 0.0568  |
| Amazon-Book  | shapley-mc vs uniform     | HitRate@20 | 0.003398          | [0.002427, 0.004395]  | < 0.0005 | True          | 0.65%      | 0.31%    | 99.05%      | 0.0348  |
| Amazon-Book  | shapley-mc vs additive-pref | NDCG@20  | 0.002191          | [0.001789, 0.002575]  | < 0.0005 | True          | 2.32%      | 1.33%    | 96.36%      | 0.0591  |
| Amazon-Book  | shapley-mc vs additive-pref | HitRate@20 | 0.003640        | [0.002670, 0.004638]  | < 0.0005 | True          | 0.65%      | 0.29%    | 99.06%      | 0.0377  |
| Amazon-Book  | shapley-mc vs attention   | NDCG@20    | 0.002338          | [0.001935, 0.002730]  | < 0.0005 | True          | 2.30%      | 1.28%    | 96.42%      | 0.0616  |
| Amazon-Book  | shapley-mc vs attention   | HitRate@20 | 0.004153          | [0.003128, 0.005097]  | < 0.0005 | True          | 0.69%      | 0.27%    | 99.04%      | 0.0424  |
| Amazon-Book  | shapley-mc vs loo-marginal | NDCG@20   | -0.000494         | [-0.000709, -0.000281] | < 0.0005 | True          | 1.04%      | 1.63%    | 97.34%      | -0.0228 |
| Amazon-Book  | shapley-mc vs loo-marginal | HitRate@20 | -0.000701       | [-0.001375, -0.000054] | 0.036    | True          | 0.18%      | 0.25%    | 99.58%      | -0.0108 |

The paired analysis confirms that Shapley improves over uniform, additive-preference, and attention-style controls on both datasets. However, Shapley does not beat LOO marginal. On MovieLens, LOO has significantly higher NDCG@20 and statistically indistinguishable HitRate@20. On Amazon-Book, LOO significantly outperforms Shapley on both metrics.

### 6.3 Cost-effectiveness

Table 3 compares ranking gains and attribution cost for Shapley and LOO marginal attribution.

**Table 3. Cost-effectiveness of Shapley and LOO marginal attribution.**

| Dataset      | Method       | NDCG@20   | HitRate@20   | Delta NDCG vs uniform   | Delta NDCG vs LOO   | Attribution seconds   | NDCG gain per attr hour   |
|:-------------|:-------------|:----------|:-------------|:------------------------|:--------------------|:----------------------|:--------------------------|
| MovieLens-1M | loo-marginal | 0.049755  | 0.125187     | 0.003748                | 0.000000            | 2010.5                | 0.006712                  |
| MovieLens-1M | shapley-mc   | 0.049223  | 0.125553     | 0.003216                | -0.000532           | 31657.7               | 0.000366                  |
| Amazon-Book  | loo-marginal | 0.032367  | 0.070891     | 0.002589                | 0.000000            | 637.2                 | 0.014630                  |
| Amazon-Book  | shapley-mc   | 0.031873  | 0.070190     | 0.002096                | -0.000494           | 8283.2                | 0.000911                  |

LOO is substantially more cost-effective. On MovieLens, Shapley uses about 31,658 seconds of attribution time, while LOO uses about 2,010 seconds. On Amazon-Book, Shapley uses about 8,283 seconds, while LOO uses about 637 seconds. In both datasets, LOO produces higher NDCG@20 at much lower cost.

## 7. Discussion

The case study yields three main findings.

First, validation-guided interaction attribution is useful. Shapley consistently improves over uniform, additive-preference, attention-style, and popularity controls on ranking metrics across MovieLens-1M and Amazon-Book. This supports the broader claim that explicitly attributing historical interactions can improve graph-based recommendation beyond simple similarity or attention-style weighting.

Second, the distinctive contribution of full Shapley averaging is limited for ranking utility. The LOO marginal baseline is stronger than Shapley on NDCG@20 in both datasets, ties or beats Shapley on HitRate@20, and is much cheaper. This suggests that much of the useful signal comes from validation-guided marginal attribution rather than from averaging marginal contributions over many coalition contexts.

Third, ranking utility alone may be the wrong place to expect Shapley to dominate. Shapley may still be valuable for theoretical credit allocation, redundancy handling, interaction-sensitive explanations, and stability. Those benefits require explanation-focused tests, such as deletion, insertion, perturbation stability, model randomization, and controlled redundancy/complementarity games. Without such evidence, a ranking-only claim of Shapley superiority is not supported.

These findings align with the paper's critical survey thesis: cooperative-game attribution offers a disciplined language for specifying players, values, and allocations, but its empirical value depends on the full design of the game and intervention. A Shapley value is not automatically better than simpler marginal attribution.

## 8. Limitations

The empirical case study has several limitations. First, although LightGCN is a graph recommender, the HCCF hypergraph port remains unavailable. Hypergraph-specific empirical claims should therefore remain outside the present case study. Second, the results were produced during design development and were not externally preregistered before execution. Third, the Amazon split is custom and temporally reconstructed, so it is not directly comparable to canonical random-split Amazon-Book numbers. Fourth, the Shapley estimator is bounded to 24 selected historical interactions per user. This makes the experiment feasible and auditable, but it is not full-history Shapley. Fifth, explanation diagnostics remain preliminary and do not yet establish human usefulness or full explanation faithfulness.

## 9. Conclusion

This paper argues that cooperative-game attribution should be treated as a precise design language rather than a guaranteed performance enhancer. The LightGCN case study shows that a carefully specified bounded Shapley intervention improves ranking over uniform and similarity-based controls on MovieLens-1M and Amazon-Book. However, leave-one-out marginal attribution captures most or all of the ranking benefit at a fraction of the computational cost. This boundary condition is central: the axiomatic appeal of Shapley does not by itself justify its use unless coalition-context averaging adds measurable ranking, explanation, stability, or interaction-sensitive benefits beyond simpler marginal baselines.

A Q1-ready final manuscript should therefore present these results as a critical empirical finding, not as a simple Shapley victory. The strongest claim is that cooperative-game attribution improves the clarity and structure of interaction-credit assignment, while the practical necessity of full Shapley averaging must be demonstrated against strong marginal baselines.
