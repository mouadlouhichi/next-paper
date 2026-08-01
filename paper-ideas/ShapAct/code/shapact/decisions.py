"""Decision rules and their realized outcomes (ShapAct Structure Sec. 3.5).

Rules (each a policy: retire the source with the smallest score):
    Shapley     min exact source Shapley credit
    LOO         min leave-one-out (masked) marginal
    FeatureSHAP min mean |feature SHAP| of the fused ranker (exact for a
                linear fusion: |theta_j| * mean|z_j - zbar_j|)
    Random      expected outcome over a uniform source draw

Each rule is executed under L2 (never-built) semantics and evaluated by the
realized NDCG@10 of the resulting system, per user (for significance tests).
"""

from __future__ import annotations

import numpy as np

from .config import SOURCES


def feature_shap_mean(theta: np.ndarray, zcache):
    """Exact mean |Shapley| of each fusion feature for a linear ranker.

    For f(z) = theta . z, the Shapley value of feature j is
    theta_j (z_j - mean z_j); averaged absolute value over users.
    """
    out = {}
    for j, g in enumerate(SOURCES):
        z = zcache.z[g]
        mu = z.mean(axis=1, keepdims=True)
        out[g] = abs(theta[j]) * float(np.mean(np.abs(z - mu)))
    return out


def rule_recommendations(audit, zcache_grand):
    """Return {rule: source} for the four decision rules."""
    G = tuple(sorted(SOURCES))
    theta = audit.thetas[G] if audit.thetas is not None else None

    sh_rule = min(SOURCES, key=lambda g: audit.phi[g])
    loo = {g: audit.predicted_effect(g) for g in SOURCES}
    loo_rule = min(SOURCES, key=lambda g: loo[g])

    fsh_rule = None
    if theta is not None:
        fshap = feature_shap_mean(theta, zcache_grand)
        fsh_rule = min(SOURCES, key=lambda g: fshap[g])

    return {"Shapley": sh_rule, "LOO": loo_rule, "FeatureSHAP": fsh_rule}


def realized_ndcg_per_user(audit, g_retired):
    """Per-user NDCG@10 in the never-built world for g_retired (L2)."""
    key = tuple(sorted(set(SOURCES) - {g_retired}))
    return audit.per_user_nb[g_retired][key]


def realized_mean(audit, g_retired):
    return float(np.mean(realized_ndcg_per_user(audit, g_retired)))


def random_expected(audit):
    """Expected realized NDCG of a uniform-random retirement."""
    return float(np.mean([realized_mean(audit, g) for g in SOURCES]))
