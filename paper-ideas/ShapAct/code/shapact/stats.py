"""Significance machinery (house protocol: paired t, Holm-Bonferroni,
Wilcoxon signed-rank, Cohen's d_z). Reuses the convention of the DyHuCoG
Appendix A and the ActionShap stats module.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def holm_bonferroni(p: np.ndarray) -> np.ndarray:
    """Holm step-down: adj[i] = max_{k<=rank(i)} p_(k) * (m - k + 1), capped at 1."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.zeros_like(p)
    prev = 0.0
    for rank, i in enumerate(order):
        val = p[i] * (m - rank)          # multiplier m-rank (1-based rank)
        prev = max(prev, val)
        adj[i] = min(1.0, prev)
    return adj


def paired_test(diff: np.ndarray):
    """Paired t-test, Wilcoxon signed-rank, and Cohen's d_z on one diff vector."""
    diff = np.asarray(diff, dtype=float)
    t, p_t = stats.ttest_rel(diff, np.zeros_like(diff))
    w, p_w = stats.wilcoxon(diff, zero_method="wilcox")
    sd = diff.std(ddof=1)
    dz = float(diff.mean() / sd) if sd > 0 else 0.0
    return {
        "t": float(t), "df": int(len(diff) - 1), "p_t": float(p_t),
        "w": float(w), "p_w": float(p_w), "d_z": float(dz),
        "mean": float(diff.mean()),
    }


def compare_rules(per_user: dict[str, np.ndarray]):
    """Pairwise per-user NDCG comparisons across decision rules, Holm-corrected.

    per_user: {rule: per-user realized NDCG@10 array}.
    Returns list of comparison dicts; degenerate pairs (identical per-user
    vectors, e.g. two rules retiring the same source) are marked `identical`
    and excluded from the Holm family.
    """
    rules = list(per_user)
    rows = []
    ps = []
    for i, a in enumerate(rules):
        for b in rules[i + 1:]:
            d = np.asarray(per_user[a]) - np.asarray(per_user[b])
            row = {"a": a, "b": b, "identical": bool(np.allclose(d, 0.0))}
            if row["identical"]:
                row.update({"mean": 0.0, "t": None, "df": 0, "p_t": None,
                            "w": None, "p_w": None, "d_z": 0.0,
                            "p_holm": None, "sig": False})
                rows.append(row)
                continue
            r = paired_test(d)
            r["a"], r["b"] = a, b
            r["identical"] = False
            rows.append(r)
            ps.append(r["p_t"])
    adj = holm_bonferroni(np.array(ps)) if ps else np.array([])
    k = 0
    for r in rows:
        if r["identical"]:
            continue
        r["p_holm"] = float(adj[k])
        r["sig"] = bool(adj[k] < 0.05)
        k += 1
    return rows
