"""
uncertainty.py — variance game via ensemble dynamics, plus calibration.

The variance game v^sigma2_t(S) = Var[ discounted future utility | do(S) ] is
estimated from an ensemble of E dynamics models (or E stochastic rollouts).
Calibration of future-utility quantiles is measured by Expected Calibration
Error (ECE).
"""
from __future__ import annotations
from typing import Callable, List, Optional, Sequence

import numpy as np


def ensemble_variance(sample_fn: Callable[[int], np.ndarray],
                      S: Sequence[int], E: int = 8, seed: int = 0) -> float:
    """
    sample_fn(seed) returns an array of E* independent draws of the discounted
    future utility for coalition S (or E stochastic rollouts). Returns the
    variance of the pooled draws.
    """
    draws = np.concatenate([sample_fn(seed + e) for e in range(E)])
    return float(np.var(draws))


def ece_from_bins(acc: np.ndarray, conf: np.ndarray, counts: np.ndarray) -> float:
    """Expected Calibration Error over pre-computed bins."""
    total = max(int(counts.sum()), 1)
    return float(np.sum(counts / total * np.abs(acc - conf)))


def calibration_ece(pred_probs: np.ndarray, outcomes: np.ndarray,
                    n_bins: int = 10) -> dict:
    """
    ECE of predicted probabilities vs binary outcomes.
    pred_probs: (N,) predicted confidence; outcomes: (N,) binary.
    Returns dict with ece, bin accuracies, confidences, counts.
    """
    pred_probs = np.asarray(pred_probs)
    outcomes = np.asarray(outcomes)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    acc = np.zeros(n_bins); conf = np.zeros(n_bins); cnt = np.zeros(n_bins)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        if b == n_bins - 1:
            m = pred_probs >= lo
        else:
            m = (pred_probs >= lo) & (pred_probs < hi)
        cnt[b] = m.sum()
        if cnt[b] > 0:
            conf[b] = pred_probs[m].mean()
            acc[b] = outcomes[m].mean()
    ece = ece_from_bins(acc, conf, cnt)
    return {"ece": ece, "acc": acc, "conf": conf, "counts": cnt}


def qq_coverage(true_quantiles: np.ndarray, nominal: float) -> float:
    """Fraction of true outcomes falling below their predicted quantile."""
    return float(np.mean(true_quantiles >= nominal))
