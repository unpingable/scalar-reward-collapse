"""Bandit agents: EpsilonGreedy, UCB, ThompsonSampling, GradientBandit."""

from __future__ import annotations

import numpy as np


class EpsilonGreedy:
    """Epsilon-greedy bandit agent.

    Supports an optional constant step_size for non-stationary environments.
    When step_size is None, uses the standard 1/n sample average.
    When set (e.g., 0.05), uses an exponential moving average so recent
    observations carry weight — critical for detecting reward drift.
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1, rng: np.random.Generator | None = None,
                 step_size: float | None = None):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.rng = rng or np.random.default_rng()
        self.step_size = step_size
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        if self.step_size is not None:
            self.values[arm] += self.step_size * (reward - self.values[arm])
        else:
            self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    def force_value(self, arm: int, value: float) -> None:
        """Override value estimate for an arm (used by D-correction).

        Models the retention team forcing a re-evaluation based on
        ground-truth data, overriding the optimizer's proxy-based estimate.
        """
        self.values[arm] = value

    def arm_probs(self) -> np.ndarray:
        """Return the probability of selecting each arm."""
        probs = np.full(self.n_arms, self.epsilon / self.n_arms)
        best = int(np.argmax(self.values))
        probs[best] += 1.0 - self.epsilon
        return probs


class UCB:
    """Upper Confidence Bound agent."""

    def __init__(self, n_arms: int, c: float = 2.0, rng: np.random.Generator | None = None):
        self.n_arms = n_arms
        self.c = c
        self.rng = rng or np.random.default_rng()
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_count = 0

    def select_arm(self) -> int:
        # Play each arm once first
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        ucb_values = self.values + self.c * np.sqrt(
            np.log(self.total_count) / self.counts
        )
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.total_count += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def arm_probs(self) -> np.ndarray:
        """Approximate: returns 1-hot for greedy, uniform if unexplored."""
        if np.any(self.counts == 0):
            return np.ones(self.n_arms) / self.n_arms
        probs = np.zeros(self.n_arms)
        probs[self.select_arm()] = 1.0
        return probs


class ThompsonSampling:
    """Thompson Sampling with Beta posteriors (for Bernoulli rewards)."""

    def __init__(self, n_arms: int, rng: np.random.Generator | None = None):
        self.n_arms = n_arms
        self.rng = rng or np.random.default_rng()
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        if reward > 0.5:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def arm_probs(self) -> np.ndarray:
        """Mean of Beta posteriors, normalized."""
        means = self.alpha / (self.alpha + self.beta)
        return means / means.sum()


class GradientBandit:
    """Gradient bandit with softmax action selection."""

    def __init__(self, n_arms: int, lr: float = 0.1, rng: np.random.Generator | None = None):
        self.n_arms = n_arms
        self.lr = lr
        self.rng = rng or np.random.default_rng()
        self.preferences = np.zeros(n_arms)
        self.avg_reward = 0.0
        self.total_count = 0

    def select_arm(self) -> int:
        probs = self.arm_probs()
        return int(self.rng.choice(self.n_arms, p=probs))

    def update(self, arm: int, reward: float) -> None:
        self.total_count += 1
        probs = self.arm_probs()
        baseline = self.avg_reward

        for a in range(self.n_arms):
            if a == arm:
                self.preferences[a] += self.lr * (reward - baseline) * (1 - probs[a])
            else:
                self.preferences[a] -= self.lr * (reward - baseline) * probs[a]

        self.avg_reward += (reward - self.avg_reward) / self.total_count

    def arm_probs(self) -> np.ndarray:
        """Softmax over preferences."""
        # Numerically stable softmax
        p = self.preferences - self.preferences.max()
        exp_p = np.exp(p)
        return exp_p / exp_p.sum()


class ControlledEpsilonGreedy(EpsilonGreedy):
    """Epsilon-greedy with a safety governor that throttles proxy updates
    when alive_fraction drops below a threshold.

    Mechanism: when alive_fraction < alive_threshold, the controller
    blocks proxy updates (the agent stops learning from CTR). This
    freezes value estimates at whatever the last D-correction set —
    which favors the safe arm. The agent still acts, but stops
    drifting back toward the proxy-optimal harmful arm.

    This is the minimal "controller baseline" for the paper: one line of
    defense that shifts the regime boundary.
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        rng: np.random.Generator | None = None,
        step_size: float | None = None,
        alive_threshold: float = 0.9,
    ):
        super().__init__(n_arms, epsilon, rng, step_size=step_size)
        self.alive_threshold = alive_threshold
        self._current_alive = 1.0

    def observe_alive(self, alive_fraction: float) -> None:
        """Called by the simulation loop with the current alive fraction."""
        self._current_alive = alive_fraction

    def should_update_proxy(self) -> bool:
        """Returns False when the safety governor is throttling updates."""
        return self._current_alive >= self.alive_threshold


AGENT_REGISTRY: dict[str, type] = {
    "epsilon_greedy": EpsilonGreedy,
    "ucb": UCB,
    "thompson_sampling": ThompsonSampling,
    "gradient_bandit": GradientBandit,
    "controlled_epsilon_greedy": ControlledEpsilonGreedy,
}


def make_agent(agent_type: str, n_arms: int, rng: np.random.Generator, **kwargs):
    """Factory for creating agents by name."""
    cls = AGENT_REGISTRY[agent_type]
    return cls(n_arms=n_arms, rng=rng, **kwargs)
