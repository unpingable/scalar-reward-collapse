"""Threshold throttle policies for the kappa sweep.

Each policy wraps the simulation loop's update decision. The simulation
loop consults policy.should_throttle(t, state) each timestep to decide
whether to skip the proxy update. This keeps agents clean and policies
composable.

State dict populated by the simulation loop each timestep:
    alive_fraction: float — fraction of users still alive
    sigma_rate_window: float — rolling sigma rate over last w steps
    D: int — retention delay parameter
    proxy_delta: float — change in proxy reward vs previous step
    true_delta: float — change in true reward vs previous step
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThrottlePolicy:
    """Base: never throttle."""

    def should_throttle(self, t: int, state: dict) -> bool:
        return False


@dataclass
class AliveThrottlePolicy(ThrottlePolicy):
    """Throttle proxy updates when alive fraction drops below threshold."""

    alive_thr: float = 0.9

    def should_throttle(self, t: int, state: dict) -> bool:
        return state["alive_fraction"] < self.alive_thr


@dataclass
class SigmaRateThrottlePolicy(ThrottlePolicy):
    """Throttle when rolling sigma rate exceeds threshold."""

    sigma_thr: float = 0.1
    window: int = 50

    def should_throttle(self, t: int, state: dict) -> bool:
        return state.get("sigma_rate_window", 0.0) > self.sigma_thr


@dataclass
class PredictiveThrottlePolicy(ThrottlePolicy):
    """Throttle based on analytic D-margin predictor.

    Uses the critical delay D_crit from Step A analysis. When the
    configured D exceeds D_crit by more than margin_thr (normalized),
    the policy throttles from t=0 — a purely structural decision.
    """

    margin_thr: float = 0.0
    D_crit: float = 317.0  # from Step A analysis

    def should_throttle(self, t: int, state: dict) -> bool:
        D = state["D"]
        margin = (D - self.D_crit) / self.D_crit
        return margin > self.margin_thr


POLICY_REGISTRY: dict[str, type] = {
    "alive": AliveThrottlePolicy,
    "sigma": SigmaRateThrottlePolicy,
    "predictive": PredictiveThrottlePolicy,
}
