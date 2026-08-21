"""KernelSHAP attribution (review-7 mandatory baseline).

Weighted-least-squares Shapley estimation with the Shapley kernel
:math:`\\pi(z) = (M-1) / (\\binom{M}{|z|} |z| (M-|z|))`, solved under the
efficiency constraint :math:`\\sum_p \\phi_p = v(P) - v(\\varnothing)` via the
standard Lagrangian (KKT) system. Masks of size ``s`` are sampled uniformly
and paired with their complements, which leaves the kernel invariant and
halves the number of distinct utility evaluations needed per pair.
"""
from __future__ import annotations

from math import comb

import numpy as np


def kernelschap_attribution(
    utility,
    n_players: int,
    samples: int = 512,
    seed: int = 0,
) -> np.ndarray:
    """Estimate Shapley values with KernelSHAP at the given mask budget.

    ``samples`` counts mask draws (each draw and its complement are both
    evaluated), matching the design size convention of the LIME comparison
    (512 masks). The full and empty coalitions enter through the efficiency
    constraint rather than as regression rows.
    """
    M = int(n_players)
    if M < 1:
        raise ValueError("n_players must be positive")
    if M == 1:
        v_full = float(utility(frozenset({0})))
        v_empty = float(utility(frozenset()))
        return np.array([v_full - v_empty])
    if samples < 2:
        raise ValueError("samples must be at least 2")

    rng = np.random.default_rng(seed)
    v_empty = float(utility(frozenset()))
    v_full = float(utility(frozenset(range(M))))
    delta = v_full - v_empty

    sizes = np.arange(1, M)
    # kernel mass per coalition of size s
    mass = (M - 1) / (np.array([comb(M, s) for s in sizes], dtype=float)
                      * sizes * (M - sizes))
    size_probs = mass / mass.sum()

    Z: list[np.ndarray] = []
    W: list[float] = []
    seen: set[tuple[int, ...]] = set()
    n_pairs = max(1, samples // 2)
    attempts = 0
    while len(Z) < n_pairs and attempts < n_pairs * 40:
        attempts += 1
        s = int(rng.choice(sizes, p=size_probs))
        idx = rng.choice(M, size=s, replace=False)
        z = np.zeros(M)
        z[idx] = 1.0
        key = tuple(z.astype(int))
        if key in seen:
            continue
        seen.add(key)
        seen.add(tuple(1 - z.astype(int)))
        w = float((M - 1) / (comb(M, s) * s * (M - s)))
        Z.append(z)
        W.append(w)
        Z.append(1.0 - z)
        W.append(w)
    Zm = np.vstack(Z)
    Wv = np.asarray(W)
    y = np.array([float(utility(frozenset(np.flatnonzero(z)))) for z in Zm]) - v_empty

    # constrained WLS: min (y - Z phi)' W (y - Z phi)  s.t.  1' phi = delta
    Wd = Wv[:, None] * Zm
    A = Zm.T @ Wd                      # M x M
    b = Zm.T @ (Wv * y)
    ones = np.ones(M)
    # KKT system [[A, 1], [1', 0]] [phi, lam] = [b, delta]
    K = np.zeros((M + 1, M + 1))
    K[:M, :M] = A
    K[:M, M] = ones
    K[M, :M] = ones
    rhs = np.concatenate([b, [delta]])
    try:
        sol = np.linalg.solve(K + 1e-12 * np.eye(M + 1), rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    phi = sol[:M]
    return phi
