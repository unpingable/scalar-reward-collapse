"""Experiment and sweep configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorldConfig:
    """Parameters defining the hazard/churn world."""

    n_users: int = 10_000
    n_arms: int = 2
    # Per-arm parameters (indexed by arm)
    click_probs: tuple[float, ...] = (0.08, 0.04)
    burnout_deltas: tuple[float, ...] = (0.015, 0.003)
    alpha: float = 0.98  # burnout recovery rate per round
    hazard_k: float = 6.0  # sigmoid steepness for churn hazard
    hazard_theta: float = 1.0  # burnout threshold for 50% *scaled* churn
    hazard_scale: float = 0.02  # max per-round churn probability
    churn_penalty: float = 15.0  # weight of churn in true reward signal

    def __post_init__(self) -> None:
        assert len(self.click_probs) == self.n_arms
        assert len(self.burnout_deltas) == self.n_arms


@dataclass
class ExperimentConfig:
    """Full configuration for a single experiment run."""

    world: WorldConfig = field(default_factory=WorldConfig)
    n_rounds: int = 1000
    update_cadence_W: int = 5  # agent updates policy every W rounds
    retention_delay_D: int = 50  # true objective observed every D rounds
    agent_type: str = "epsilon_greedy"
    agent_kwargs: dict = field(default_factory=lambda: {"epsilon": 0.1, "step_size": 0.05})
    seed: int = 42


@dataclass
class SweepConfig:
    """Grid sweep over D and W values with multiple seeds."""

    D_values: list[int] = field(default_factory=lambda: [10, 50, 200, 500])
    W_values: list[int] = field(default_factory=lambda: [1, 5, 20])
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    base: ExperimentConfig = field(default_factory=ExperimentConfig)
