| Dataset      | Contrast                    | Metric     |   Mean difference | 95% CI                 | p        | Holm reject   | Improved   | Harmed   | Unchanged   |     d_z |
|:-------------|:----------------------------|:-----------|------------------:|:-----------------------|:---------|:--------------|:-----------|:---------|:------------|--------:|
| MovieLens-1M | shapley-mc vs uniform       | NDCG@20    |          0.003216 | [0.002707, 0.003715]   | < 0.0005 | True          | 6.01%      | 2.92%    | 91.07%      |  0.0716 |
| MovieLens-1M | shapley-mc vs uniform       | HitRate@20 |          0.00818  | [0.006517, 0.009875]   | < 0.0005 | True          | 1.52%      | 0.70%    | 97.79%      |  0.055  |
| MovieLens-1M | shapley-mc vs additive-pref | NDCG@20    |          0.003121 | [0.002606, 0.003628]   | < 0.0005 | True          | 5.95%      | 2.93%    | 91.13%      |  0.0693 |
| MovieLens-1M | shapley-mc vs additive-pref | HitRate@20 |          0.008047 | [0.006384, 0.009742]   | < 0.0005 | True          | 1.51%      | 0.70%    | 97.79%      |  0.0542 |
| MovieLens-1M | shapley-mc vs attention     | NDCG@20    |          0.00274  | [0.002207, 0.003271]   | < 0.0005 | True          | 5.89%      | 3.11%    | 91.00%      |  0.0582 |
| MovieLens-1M | shapley-mc vs attention     | HitRate@20 |          0.007548 | [0.005786, 0.009211]   | < 0.0005 | True          | 1.54%      | 0.79%    | 97.67%      |  0.0495 |
| MovieLens-1M | shapley-mc vs loo-marginal  | NDCG@20    |         -0.000532 | [-0.000963, -0.000119] | 0.008    | True          | 3.19%      | 4.97%    | 91.84%      | -0.0143 |
| MovieLens-1M | shapley-mc vs loo-marginal  | HitRate@20 |          0.000366 | [-0.000899, 0.001696]  | 0.575    | False         | 0.68%      | 0.65%    | 98.67%      |  0.0032 |
| Amazon-Book  | shapley-mc vs uniform       | NDCG@20    |          0.002096 | [0.001703, 0.002474]   | < 0.0005 | True          | 2.33%      | 1.36%    | 96.31%      |  0.0568 |
| Amazon-Book  | shapley-mc vs uniform       | HitRate@20 |          0.003398 | [0.002427, 0.004395]   | < 0.0005 | True          | 0.65%      | 0.31%    | 99.05%      |  0.0348 |
| Amazon-Book  | shapley-mc vs additive-pref | NDCG@20    |          0.002191 | [0.001789, 0.002575]   | < 0.0005 | True          | 2.32%      | 1.33%    | 96.36%      |  0.0591 |
| Amazon-Book  | shapley-mc vs additive-pref | HitRate@20 |          0.00364  | [0.002670, 0.004638]   | < 0.0005 | True          | 0.65%      | 0.29%    | 99.06%      |  0.0377 |
| Amazon-Book  | shapley-mc vs attention     | NDCG@20    |          0.002338 | [0.001935, 0.002730]   | < 0.0005 | True          | 2.30%      | 1.28%    | 96.42%      |  0.0616 |
| Amazon-Book  | shapley-mc vs attention     | HitRate@20 |          0.004153 | [0.003128, 0.005097]   | < 0.0005 | True          | 0.69%      | 0.27%    | 99.04%      |  0.0424 |
| Amazon-Book  | shapley-mc vs loo-marginal  | NDCG@20    |         -0.000494 | [-0.000709, -0.000281] | < 0.0005 | True          | 1.04%      | 1.63%    | 97.34%      | -0.0228 |
| Amazon-Book  | shapley-mc vs loo-marginal  | HitRate@20 |         -0.000701 | [-0.001375, -0.000054] | 0.036    | True          | 0.18%      | 0.25%    | 99.58%      | -0.0108 |