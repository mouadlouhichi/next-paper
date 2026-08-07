| Dataset      | Method       |   NDCG@20 |   HitRate@20 |   Delta NDCG vs uniform |   Delta NDCG vs LOO |   Attribution seconds |   NDCG gain per attr hour |
|:-------------|:-------------|----------:|-------------:|------------------------:|--------------------:|----------------------:|--------------------------:|
| MovieLens-1M | loo-marginal |  0.049755 |     0.125187 |                0.003748 |            0        |                2010.5 |                  0.006712 |
| MovieLens-1M | shapley-mc   |  0.049223 |     0.125553 |                0.003216 |           -0.000532 |               31657.7 |                  0.000366 |
| Amazon-Book  | loo-marginal |  0.032367 |     0.070891 |                0.002589 |            0        |                 637.2 |                  0.01463  |
| Amazon-Book  | shapley-mc   |  0.031873 |     0.07019  |                0.002096 |           -0.000494 |                8283.2 |                  0.000911 |