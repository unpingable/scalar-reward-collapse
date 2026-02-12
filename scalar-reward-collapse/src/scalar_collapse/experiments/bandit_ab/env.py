"""HazardChurnEnv: population-level bandit environment with burnout and churn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scalar_collapse.core.config import WorldConfig


@dataclass
class StepResult:
    """Result of a single environment step."""

    clicks: int
    new_churns: int
    alive_count: int
    mean_burnout: float
    std_burnout: float
    ctr: float  # clicks / alive_count (proxy reward)
    alive_fraction: float


class HazardChurnEnv:
    """Population-level bandit with burnout accumulation and churn hazard.

    Each round:
    1. Serve an arm to all alive users
    2. Users click with Bernoulli(click_prob[arm])
    3. Burnout updates: B_{t+1} = alpha * B_t + delta[arm]
    4. Churn hazard: h = hazard_scale * sigmoid(k * (B - theta))
    5. Users churn with Bernoulli(h)
    """

    def __init__(self, world: WorldConfig, rng: np.random.Generator):
        self.world = world
        self.rng = rng
        self.burnout = np.zeros(world.n_users)
        self.alive = np.ones(world.n_users, dtype=bool)
        self._initial_n = world.n_users

    def step(self, arm: int) -> StepResult:
        """Serve arm to all alive users. Returns step metrics."""
        w = self.world
        alive_mask = self.alive
        alive_count = int(alive_mask.sum())

        if alive_count == 0:
            return StepResult(
                clicks=0,
                new_churns=0,
                alive_count=0,
                mean_burnout=0.0,
                std_burnout=0.0,
                ctr=0.0,
                alive_fraction=0.0,
            )

        # 1. Generate clicks
        click_prob = w.click_probs[arm]
        clicks_mask = self.rng.random(w.n_users) < click_prob
        clicks_mask &= alive_mask
        clicks = int(clicks_mask.sum())

        # 2. Update burnout for alive users
        self.burnout[alive_mask] = (
            w.alpha * self.burnout[alive_mask] + w.burnout_deltas[arm]
        )

        # 3. Compute churn hazard: hazard_scale * sigmoid(k * (B - theta))
        hazard = np.zeros(w.n_users)
        hazard[alive_mask] = w.hazard_scale * _sigmoid(
            w.hazard_k * (self.burnout[alive_mask] - w.hazard_theta)
        )

        # 4. Sample churn
        churn_mask = self.rng.random(w.n_users) < hazard
        churn_mask &= alive_mask
        new_churns = int(churn_mask.sum())

        # 5. Update alive mask
        self.alive[churn_mask] = False
        alive_count_after = int(self.alive.sum())

        mean_b = float(self.burnout[self.alive].mean()) if alive_count_after > 0 else 0.0
        std_b = float(self.burnout[self.alive].std()) if alive_count_after > 0 else 0.0

        return StepResult(
            clicks=clicks,
            new_churns=new_churns,
            alive_count=alive_count_after,
            mean_burnout=mean_b,
            std_burnout=std_b,
            ctr=clicks / alive_count if alive_count > 0 else 0.0,
            alive_fraction=alive_count_after / self._initial_n,
        )

    def reset(self) -> None:
        """Reset population to initial state."""
        self.burnout = np.zeros(self.world.n_users)
        self.alive = np.ones(self.world.n_users, dtype=bool)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (vectorized)."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def sigmoid_scalar(x: float) -> float:
    """Numerically stable sigmoid (scalar)."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)
