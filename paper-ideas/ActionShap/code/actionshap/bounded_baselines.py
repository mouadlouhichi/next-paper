"""Intervention-aware attribution baselines (review 3, issue 2).

These baselines match the *deployment* perturbation distribution of the
bounded-weight protocol, removing the deletion-distribution confound of
binary-mask LIME:

* ``bounded_lime``      - locally weighted ridge over weight vectors drawn from
                          {rho, 1}^n (Bernoulli) or U[rho, 1]^n (continuous);
* ``finite_difference`` - one-sided sensitivity at the full profile;
* ``integrated_gradients`` - path attribution along the weight path from the
                          bounded baseline rho to the full weight 1.

All attributions follow the package sign convention: ``-phi`` is the predicted
benefit of bounded downweighting.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .recommendation import UserGame, target_margin_from_scores


def utility_of_weights(model, game: UserGame, weights: np.ndarray, L: int = 10) -> float:
    scores = model.score(game.players, game.candidate_items, weights)
    return target_margin_from_scores(scores, game, L)


def bounded_lime(
    model,
    game: UserGame,
    *,
    rho: float = 0.5,
    samples: int = 512,
    seed: int = 0,
    continuous: bool = False,
    kernel_width: float = 0.25,
    ridge_alpha: float = 1.0,
) -> np.ndarray:
    n = game.players.size
    rng = np.random.default_rng(seed)
    full = np.ones(n)
    masks = [full]
    for p in range(n):  # leave-one-bounded-out neighbours
        w = np.ones(n)
        w[p] = rho
        masks.append(w)
    remaining = samples - len(masks)
    if continuous:
        masks += [rng.uniform(rho, 1.0, n) for _ in range(remaining)]
    else:
        masks += [
            np.where(rng.random(n) < 0.5, rho, 1.0) for _ in range(remaining)
        ]
    X = np.vstack(masks)
    y = np.array([utility_of_weights(model, game, w) for w in masks])
    distance = np.mean((1.0 - X) / (1.0 - rho), axis=1)
    weight = np.exp(-np.square(distance) / np.square(kernel_width))
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
    ridge.fit(X, y, sample_weight=weight)
    return np.asarray(ridge.coef_, dtype=float)  # phi; -phi predicts downweight benefit


def finite_difference(model, game: UserGame, *, h: float = 0.05) -> np.ndarray:
    """One-sided weight sensitivity at the full profile (attribution = -d)."""
    n = game.players.size
    base = utility_of_weights(model, game, np.ones(n))
    d = np.zeros(n)
    for p in range(n):
        w = np.ones(n)
        w[p] = 1.0 - h
        d[p] = (utility_of_weights(model, game, w) - base) / h
    return -d


def integrated_gradients(
    model, game: UserGame, *, rho: float = 0.5, steps: int = 20, dh: float = 1e-3
) -> np.ndarray:
    """Path attribution from the bounded baseline rho to full weight 1."""
    n = game.players.size
    ts = np.linspace(0.0, 1.0, steps + 1)
    ig = np.zeros(n)
    span = 1.0 - rho
    for p in range(n):
        grads = []
        for t in ts:
            w = np.full(n, rho + t * span)
            up = w.copy()
            up[p] = min(1.0, up[p] + dh)
            dn = w.copy()
            dn[p] = max(rho, dn[p] - dh)
            denom = float(up[p] - dn[p]) or dh
            grads.append(
                (utility_of_weights(model, game, up)
                 - utility_of_weights(model, game, dn)) / denom
            )
        integral = _trapz(grads, ts) * span
        ig[p] = integral
    return ig  # phi convention: -phi predicts the bounded-downweight benefit


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(np.asarray(y), np.asarray(x)))
