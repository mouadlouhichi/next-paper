"""
ope.py — off-policy evaluation of the forward value.

Implements IPS, DR, SNIPS with the *observation/selection propensity* of
Schnabel et al. (2016) "Recommendations as Treatments", appropriate for MNAR
observational rating matrices (MovieLens-1M etc.), and the discrepancy gate
that keeps a plan from being "validated by the model that predicted it".

The observation-propensity is a *proxy* for the action/intervention propensity
(flagged explicitly in the proposal §3.5.3-b) and is bounded by the synthetic
known-propensity positive control and the DR-robustness analysis.
"""
from __future__ import annotations
from typing import Callable, Optional, Sequence

import numpy as np


def naive_bayes_propensity(users: Sequence[int], items: Sequence[int],
                           observed: Sequence[bool]) -> np.ndarray:
    """
    Schnabel-style naive-Bayes observation propensity:
        P(observed | u, i) ~ P(observed|u) * P(observed|i)
    Normalized to a valid probability via a logistic link on the log product.
    Returns a per-pair propensity array aligned with the input rows.
    """
    users = np.asarray(users); items = np.asarray(items)
    observed = np.asarray(observed, dtype=float)
    # user marginal
    u_obs = np.zeros(int(users.max()) + 1); u_cnt = np.zeros(int(users.max()) + 1)
    i_obs = np.zeros(int(items.max()) + 1); i_cnt = np.zeros(int(items.max()) + 1)
    for u, i, o in zip(users, items, observed):
        u_obs[u] += o; u_cnt[u] += 1
        i_obs[i] += o; i_cnt[i] += 1
    pu = np.divide(u_obs + 0.5, u_cnt + 1.0, out=np.ones_like(u_obs), where=u_cnt > 0)
    pi = np.divide(i_obs + 0.5, i_cnt + 1.0, out=np.ones_like(i_obs), where=i_cnt > 0)
    logit = np.log(pu[users] + 1e-9) + np.log(pi[items] + 1e-9)
    # logistic link to [0,1]
    return 1.0 / (1.0 + np.exp(-logit))


def logistic_propensity(features: np.ndarray, observed: np.ndarray,
                        max_iter: int = 200, seed: int = 0) -> np.ndarray:
    """
    Logistic-regression observation propensity over observable covariates.
    Returns fitted probabilities. (Gradient descent on log-likelihood.)
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(features, dtype=float)
    X = np.column_stack([np.ones(X.shape[0]), X])
    y = np.asarray(observed, dtype=float)
    theta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-X @ theta))
        grad = X.T @ (p - y) + 1e-4 * theta
        hess = X.T @ (p * (1 - p))[:, None] * X
        try:
            step = np.linalg.solve(hess + 1e-6 * np.eye(X.shape[1]), grad)
        except np.linalg.LinAlgError:
            step = grad * 0.01
        theta = theta - 0.5 * step
    return 1.0 / (1.0 + np.exp(-X @ theta))


def ips_estimate(rewards: np.ndarray, propensity: np.ndarray,
                 indicator: np.ndarray, cap: Optional[float] = None) -> float:
    """Inverse-propensity-score estimate (optionally clipped)."""
    prop = np.clip(propensity, 1e-6, 1.0 - 1e-6)
    w = 1.0 / prop
    if cap is not None:
        w = np.minimum(w, cap)
    return float(np.sum(rewards * w * indicator) / max(len(rewards), 1))


def snips_estimate(rewards: np.ndarray, propensity: np.ndarray,
                   indicator: np.ndarray, cap: Optional[float] = None) -> float:
    """Self-normalized IPS."""
    prop = np.clip(propensity, 1e-6, 1.0 - 1e-6)
    w = 1.0 / prop
    if cap is not None:
        w = np.minimum(w, cap)
    sel = indicator.astype(float)
    num = np.sum(rewards * w * sel)
    den = np.sum(w * sel)
    return float(num / den) if den > 0 else 0.0


def dr_estimate(rewards: np.ndarray, propensity: np.ndarray,
                outcome_model: np.ndarray, indicator: np.ndarray,
                cap: Optional[float] = None) -> float:
    """
    Doubly-robust estimate. Unbiased if the propensity model OR the outcome
    model is correct. This is the workhorse estimator.
    """
    prop = np.clip(propensity, 1e-6, 1.0 - 1e-6)
    w = 1.0 / prop
    if cap is not None:
        w = np.minimum(w, cap)
    sel = indicator.astype(float)
    resid = w * (rewards - outcome_model)
    return float(np.sum(sel * (resid + outcome_model)) / max(len(rewards), 1))


def effective_sample_size(propensity: np.ndarray, indicator: np.ndarray) -> float:
    """Effective sample size of the reweighting (ESS)."""
    prop = np.clip(np.asarray(propensity), 1e-6, 1.0 - 1e-6)
    w = (1.0 / prop) * np.asarray(indicator, dtype=float)
    if np.sum(w) == 0:
        return 0.0
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def discrepancy_gate(dr_lift: float, naive_lift: float,
                     tolerance: float = 0.5) -> dict:
    """
    Gate on the "does the plan work" claim: a plan is only claimed to work when
    its DR-corrected lift is positive AND the naive model number is within
    `tolerance` of it (so the model is not validating itself).
    """
    gap = float(np.abs(dr_lift - naive_lift))
    pass_gate = (dr_lift > 0) and (gap <= tolerance * max(abs(dr_lift), 1e-9))
    return {"dr_lift": dr_lift, "naive_lift": naive_lift, "gap": gap,
            "pass_gate": pass_gate}
