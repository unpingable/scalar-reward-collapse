"""Analytic regime boundary prediction for §3.8.

Predicts stable/metastable/unstable from config parameters without
running a simulation. Two independent clocks:

1. Controller clock (t_cross): how many rounds until proxy-only
   updates cause the agent to switch from safe arm to harmful arm.

2. Plant clock (t_damage): how many rounds of harmful-arm exploitation
   until cumulative hazard violates C_k.

Critical insight: Q starts at [0,0], not at true values. The first
D-correction doesn't arrive until t=D. So the first interval is
essentially uncorrected proxy learning — the agent discovers arm 0
(high CTR) almost immediately and exploits it for ~D rounds.

D_crit is determined by two conditions:
A) First-interval collapse: D rounds of uncorrected exploitation
   cause enough hazard to violate C_k before the first correction.
B) Steady-state collapse: over many corrected intervals, the drift
   cycle accumulates enough average burnout to violate C_k within T.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scalar_collapse.core.config import ExperimentConfig


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


@dataclass
class BoundaryPrediction:
    """Output of predict_regime()."""

    # First interval (uncorrected)
    n_cross_initial: int  # updates until arm 0 dominates from Q=[0,0]
    t_damage_first: int  # rounds of arm-0 exploitation until C_k violated
    first_interval_collapses: bool  # does D > t_damage_first?

    # Post-correction drift cycle
    n_cross: int  # updates until Q[0] >= Q[1] from true values
    t_cross: int  # rounds until crossover (n_cross * W)
    t_damage_post: int  # rounds after crossover until C_k violated
    B_at_crossover: float  # burnout level when crossover happens
    D_crit: int  # t_cross + t_damage_post (single-interval)

    # Steady-state analysis (multi-interval)
    delta_avg_per_interval: float
    B_steady: float
    h_steady: float
    tau_steady: float  # expected damage onset from steady-state hazard

    # Prediction
    regime_pred: str  # stable / metastable / unstable
    margin: float  # how far D is from boundary
    D: int
    W: int


def _q_crossover(
    c0: float,
    c1: float,
    u0: float,
    u1: float,
    step_size: float,
    epsilon: float,
    n_arms: int = 2,
    max_n: int = 50_000,
) -> int:
    """Iterate mean-field Q dynamics with state-dependent selection probs.

    Returns n_cross: number of proxy updates until Q[0] >= Q[1].
    p(n) depends on current Q gap, so crossover accelerates as
    Q[0] approaches Q[1].
    """
    Q0, Q1 = u0, u1

    if Q0 >= Q1:
        return 0

    for n in range(max_n):
        if Q0 > Q1:
            p0 = (1.0 - epsilon) + epsilon / n_arms
        elif Q1 > Q0:
            p0 = epsilon / n_arms
        else:
            p0 = 1.0 / n_arms

        Q0 += p0 * step_size * (c0 - Q0)
        Q1 += (1.0 - p0) * step_size * (c1 - Q1)

        if Q0 >= Q1:
            return n + 1

    return max_n


def _rounds_to_damage(
    B0: float,
    alpha: float,
    delta_bar: float,
    h_scale: float,
    h_k: float,
    h_theta: float,
    alive_floor: float = 0.5,
    max_rounds: int = 50_000,
) -> int:
    """Deterministic burnout iteration from B0 under constant delta_bar.

    Returns rounds until cumulative hazard >= -ln(alive_floor).
    """
    target = -math.log(alive_floor)
    B = B0
    cum_hazard = 0.0

    for t in range(max_rounds):
        h = h_scale * _sigmoid(h_k * (B - h_theta))
        cum_hazard += h
        if cum_hazard >= target:
            return t + 1
        B = alpha * B + delta_bar

    return max_rounds


def _burnout_after(
    B0: float,
    alpha: float,
    delta_bar: float,
    rounds: int,
) -> float:
    """Compute burnout level after `rounds` of constant delta_bar."""
    B = B0
    for _ in range(rounds):
        B = alpha * B + delta_bar
    return B


def predict_regime(config: ExperimentConfig) -> BoundaryPrediction:
    """Predict regime from config without simulation."""
    w = config.world
    D = config.retention_delay_D
    W = config.update_cadence_W
    T = config.n_rounds
    step_size = config.agent_kwargs.get("step_size", 0.05)
    epsilon = config.agent_kwargs.get("epsilon", 0.1)
    n_arms = w.n_arms
    alive_floor = 0.5

    c0, c1 = w.click_probs[0], w.click_probs[1]
    d0, d1 = w.burnout_deltas[0], w.burnout_deltas[1]

    # True arm values
    ss0 = d0 / (1.0 - w.alpha)
    ss1 = d1 / (1.0 - w.alpha)
    u0 = c0 - w.churn_penalty * w.hazard_scale * _sigmoid(w.hazard_k * (ss0 - w.hazard_theta))
    u1 = c1 - w.churn_penalty * w.hazard_scale * _sigmoid(w.hazard_k * (ss1 - w.hazard_theta))

    # --- FIRST INTERVAL: Q starts at [0,0], no correction until t=D ---
    # From [0,0], arm 0 dominates almost immediately (higher CTR)
    n_cross_initial = _q_crossover(c0, c1, 0.0, 0.0, step_size, epsilon, n_arms)

    # Post-crossover exploitation probabilities
    p0_post = (1.0 - epsilon) + epsilon / n_arms
    delta_post = p0_post * d0 + (1.0 - p0_post) * d1

    # How many rounds of arm-0 exploitation until C_k violated?
    # Start from B=0 (fresh population)
    t_damage_first = _rounds_to_damage(
        0.0, w.alpha, delta_post,
        w.hazard_scale, w.hazard_k, w.hazard_theta,
        alive_floor,
    )

    first_interval_collapses = D > t_damage_first

    # --- POST-CORRECTION DRIFT CYCLE ---
    # After first D-correction, Q forced to [u0, u1]
    # Then drifts back toward proxy-optimal over D/W updates
    n_cross = _q_crossover(c0, c1, u0, u1, step_size, epsilon, n_arms)
    t_cross = n_cross * W

    # Pre-crossover burnout rate (arm 1 dominant)
    p0_pre = epsilon / n_arms
    delta_pre = p0_pre * d0 + (1.0 - p0_pre) * d1
    B_pre_steady = delta_pre / (1.0 - w.alpha)

    t_damage_post = _rounds_to_damage(
        B_pre_steady, w.alpha, delta_post,
        w.hazard_scale, w.hazard_k, w.hazard_theta,
        alive_floor,
    )

    D_crit = t_cross + t_damage_post

    # --- STEADY-STATE (multi-interval, post first correction) ---
    updates_per_interval = D // W if W > 0 else D

    if n_cross < updates_per_interval:
        rounds_safe = min(n_cross * W, D)
        rounds_bad = D - rounds_safe
        delta_avg = (rounds_safe * delta_pre + rounds_bad * delta_post) / D
    else:
        delta_avg = delta_pre

    B_steady = delta_avg / (1.0 - w.alpha)
    h_steady = w.hazard_scale * _sigmoid(w.hazard_k * (B_steady - w.hazard_theta))

    if h_steady > 1e-15:
        tau_steady = -math.log(alive_floor) / h_steady
    else:
        tau_steady = float("inf")

    # --- REGIME PREDICTION ---
    # First interval is the most dangerous (no correction, Q=[0,0]).
    # If first interval alone causes collapse, that's unstable:
    # the correction arrives too late, burnout is irreversible.
    # Near the boundary (D ≈ t_damage_first), stochastic variation
    # can make the outcome seed-dependent → metastable.
    if first_interval_collapses:
        if D > t_damage_first * 1.15:
            regime = "unstable"
        else:
            regime = "metastable"
    elif n_cross >= updates_per_interval and tau_steady > 1.5 * T:
        # No crossover in corrected intervals, low steady-state hazard
        regime = "stable"
    elif tau_steady > 1.5 * T:
        regime = "stable"
    elif tau_steady < 0.7 * T:
        regime = "unstable"
    else:
        regime = "metastable"

    margin = (D - D_crit) / D_crit if D_crit > 0 else -1.0

    return BoundaryPrediction(
        n_cross_initial=n_cross_initial,
        t_damage_first=t_damage_first,
        first_interval_collapses=first_interval_collapses,
        n_cross=n_cross,
        t_cross=t_cross,
        t_damage_post=t_damage_post,
        B_at_crossover=B_pre_steady,
        D_crit=D_crit,
        delta_avg_per_interval=delta_avg,
        B_steady=B_steady,
        h_steady=h_steady,
        tau_steady=tau_steady,
        regime_pred=regime,
        margin=margin,
        D=D,
        W=W,
    )
