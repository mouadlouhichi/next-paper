from __future__ import annotations

import numpy as np


def paired_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float) - np.asarray(b, dtype=float)


def cohen_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def bootstrap_ci(diff_by_seed: list[np.ndarray], n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float, float]:
    """User-within-seed bootstrap for conditional user-population estimand."""
    rng = np.random.default_rng(seed)
    obs_seed_means = np.array([d.mean() for d in diff_by_seed if len(d) > 0], dtype=float)
    obs = float(obs_seed_means.mean())
    boots = []
    for _ in range(n_boot):
        vals = []
        for d in diff_by_seed:
            if len(d) == 0:
                continue
            idx = rng.integers(0, len(d), size=len(d))
            vals.append(d[idx].mean())
        boots.append(np.mean(vals))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return obs, float(lo), float(hi)


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, dict[str, float | bool]]:
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    for rank, (name, p) in enumerate(items, start=1):
        threshold = 0.05 / (m - rank + 1)
        out[name] = {"p": float(p), "holm_threshold": threshold, "reject_0.05": bool(p <= threshold)}
    return out
