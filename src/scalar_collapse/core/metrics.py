"""Metric computation: entropy, alive fraction, divergence, sigma, damage onset."""

from __future__ import annotations

import numpy as np


def policy_entropy(arm_probs: np.ndarray) -> float:
    """Shannon entropy of the arm selection distribution.

    Returns 0 for degenerate distributions (single arm = 1.0).
    """
    p = np.asarray(arm_probs, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def alive_fraction(alive: np.ndarray) -> float:
    """Fraction of users still alive."""
    a = np.asarray(alive)
    if a.size == 0:
        return 0.0
    return float(np.mean(a))


def proxy_true_divergence(
    proxy_cumulative: np.ndarray, true_cumulative: np.ndarray
) -> float:
    """Gap between cumulative proxy and true objectives (AUC of difference)."""
    proxy = np.asarray(proxy_cumulative, dtype=np.float64)
    true = np.asarray(true_cumulative, dtype=np.float64)
    n = min(len(proxy), len(true))
    return float(np.sum(proxy[:n] - true[:n]))


def sigma_count(proxy_series: np.ndarray, true_series: np.ndarray) -> int:
    """Count timesteps where proxy improves but true degrades.

    Both series are per-step reward values (not cumulative).
    """
    proxy = np.asarray(proxy_series, dtype=np.float64)
    true = np.asarray(true_series, dtype=np.float64)
    n = min(len(proxy), len(true))
    if n < 2:
        return 0
    proxy_delta = np.diff(proxy[:n])
    true_delta = np.diff(true[:n])
    return int(np.sum((proxy_delta > 0) & (true_delta < 0)))


def damage_onset(true_series: np.ndarray, floor: float) -> int:
    """First timestep where true objective crosses below the floor.

    Returns -1 if the floor is never crossed.
    """
    true = np.asarray(true_series, dtype=np.float64)
    below = np.where(true < floor)[0]
    if len(below) == 0:
        return -1
    return int(below[0])
