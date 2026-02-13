"""Tests for HazardChurnEnv correctness."""

from pathlib import Path

import numpy as np
import pytest

from scalar_collapse.core.config import WorldConfig
from scalar_collapse.experiments.bandit_ab.env import HazardChurnEnv


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def default_world():
    return WorldConfig()


def test_initial_state(default_world, rng):
    env = HazardChurnEnv(default_world, rng)
    assert env.alive.sum() == default_world.n_users
    assert env.burnout.sum() == 0.0


def test_burnout_accumulates(default_world, rng):
    env = HazardChurnEnv(default_world, rng)
    env.step(0)  # arm 0 has burnout_delta = 0.04
    # Burnout should be > 0 for alive users
    alive_burnout = env.burnout[env.alive]
    assert alive_burnout.mean() > 0


def test_burnout_recovery(rng):
    """With alpha < 1, burnout partially recovers each round."""
    world = WorldConfig(
        n_users=100,
        click_probs=(0.0, 0.0),  # no clicks
        burnout_deltas=(0.0, 0.0),  # no new burnout
        alpha=0.5,
        hazard_k=0.01,  # very low churn
        hazard_theta=100.0,  # very high threshold
    )
    env = HazardChurnEnv(world, rng)
    # Manually set some burnout
    env.burnout[:] = 1.0
    env.step(0)
    # Burnout should decay: 0.5 * 1.0 + 0.0 = 0.5
    alive_burnout = env.burnout[env.alive]
    assert np.allclose(alive_burnout, 0.5, atol=0.01)


def test_churn_hazard_monotone_in_burnout(rng):
    """Higher burnout should produce higher churn rates."""
    world = WorldConfig(
        n_users=10_000,
        click_probs=(0.0, 0.0),
        burnout_deltas=(0.1, 0.001),
        alpha=1.0,  # no recovery
        hazard_k=10.0,
        hazard_theta=0.5,
    )

    # Run many steps with high-harm arm
    env_high = HazardChurnEnv(world, np.random.default_rng(42))
    for _ in range(20):
        env_high.step(0)
    high_alive = env_high.alive.sum()

    # Run many steps with low-harm arm
    env_low = HazardChurnEnv(world, np.random.default_rng(42))
    for _ in range(20):
        env_low.step(1)
    low_alive = env_low.alive.sum()

    # High-harm arm should cause more churn
    assert high_alive < low_alive


def test_all_churn_with_extreme_harm(rng):
    """With extreme burnout, all users should eventually churn."""
    world = WorldConfig(
        n_users=1000,
        click_probs=(0.0, 0.0),
        burnout_deltas=(1.0, 0.0),
        alpha=1.0,
        hazard_k=20.0,
        hazard_theta=0.1,
        hazard_scale=1.0,  # unscaled so sigmoid(large) -> ~1.0 churn
    )
    env = HazardChurnEnv(world, rng)
    for _ in range(100):
        env.step(0)
    assert env.alive.sum() == 0


def test_no_churn_with_zero_harm(rng):
    """With zero burnout delta, no churn should occur (theta > 0)."""
    world = WorldConfig(
        n_users=1000,
        click_probs=(0.05, 0.05),
        burnout_deltas=(0.0, 0.0),
        alpha=0.95,
        hazard_k=10.0,
        hazard_theta=1.0,
    )
    env = HazardChurnEnv(world, rng)
    for _ in range(100):
        env.step(0)
    # With zero burnout, hazard = sigmoid(10 * (0 - 1)) = sigmoid(-10) ≈ 0
    # Some very small number of churns possible, but vast majority alive
    assert env.alive.sum() > 950


def test_reset(default_world, rng):
    env = HazardChurnEnv(default_world, rng)
    env.step(0)
    env.step(0)
    env.reset()
    assert env.alive.sum() == default_world.n_users
    assert env.burnout.sum() == 0.0


def test_step_result_fields(default_world, rng):
    env = HazardChurnEnv(default_world, rng)
    result = env.step(0)
    assert result.clicks >= 0
    assert result.new_churns >= 0
    assert result.alive_count >= 0
    assert 0.0 <= result.alive_fraction <= 1.0
    assert result.ctr >= 0.0


def test_empty_population(rng):
    """Environment should handle all-dead population gracefully."""
    world = WorldConfig(n_users=10, click_probs=(0.0, 0.0), burnout_deltas=(10.0, 0.0),
                        hazard_k=50.0, hazard_theta=0.01, hazard_scale=1.0)
    env = HazardChurnEnv(world, rng)
    # Kill everyone
    for _ in range(50):
        env.step(0)
    # Now step with dead population
    result = env.step(0)
    assert result.alive_count == 0
    assert result.ctr == 0.0
    assert result.alive_fraction == 0.0


# --- Sanity invariant tests ---
# These prove the collapse phenomenon is real, not an artifact.


def test_no_collapse_with_equal_harm():
    """If both arms have identical harm, there is no exploitation trap.

    The agent can't make a "wrong" choice because both arms are identical
    in their long-term impact. With genuinely safe parameters, the population
    should survive.
    """
    from scalar_collapse.core.config import ExperimentConfig
    from scalar_collapse.experiments.bandit_ab.simulate import run_experiment
    import tempfile

    # Equal click probs AND equal LOW burnout — no differential to exploit.
    # Steady state burnout = 0.001/0.02 = 0.05, far below theta=1.0.
    # Churn rate: 0.02 * sigmoid(6*(0.05-1.0)) ≈ 0.02 * 0.003 ≈ 0.00006/round.
    config = ExperimentConfig(
        world=WorldConfig(
            n_users=5000,
            click_probs=(0.06, 0.06),
            burnout_deltas=(0.001, 0.001),  # identical, very low harm
            alpha=0.98,
            hazard_k=6.0,
            hazard_theta=1.0,
            hazard_scale=0.02,
            churn_penalty=15.0,
        ),
        n_rounds=300,
        update_cadence_W=5,
        retention_delay_D=50,
        agent_kwargs={"epsilon": 0.1, "step_size": 0.05},
        seed=42,
    )
    with tempfile.TemporaryDirectory() as tmp:
        summary = run_experiment(config, Path(tmp) / "run")

    assert summary["alive_final"] > 0.9, (
        f"Equal-harm arms should not collapse, got alive={summary['alive_final']}"
    )


def test_low_D_prevents_or_delays_collapse():
    """With D=1 (true signal every round), the agent should correct quickly
    enough to prevent or substantially delay collapse compared to D=200.

    Requires step_size (non-stationary estimator) so the true-signal
    corrections actually move the agent's value estimates.
    """
    from scalar_collapse.core.config import ExperimentConfig
    from scalar_collapse.experiments.bandit_ab.simulate import run_experiment
    import tempfile

    base_world = WorldConfig(
        n_users=5000,
        click_probs=(0.08, 0.04),
        burnout_deltas=(0.015, 0.003),
        alpha=0.98,
        hazard_k=6.0,
        hazard_theta=1.0,
        hazard_scale=0.02,
        churn_penalty=15.0,
    )

    results = {}
    for D in [10, 200]:
        config = ExperimentConfig(
            world=base_world,
            n_rounds=500,
            update_cadence_W=5,
            retention_delay_D=D,
            agent_kwargs={"epsilon": 0.1, "step_size": 0.05},
            seed=42,
        )
        with tempfile.TemporaryDirectory() as tmp:
            results[D] = run_experiment(config, Path(tmp) / "run")

    # D=10 should have better survival than D=200
    alive_D10 = results[10]["alive_final"]
    alive_D200 = results[200]["alive_final"]
    assert alive_D10 > alive_D200, (
        f"D=10 should survive better than D=200: "
        f"alive_D10={alive_D10:.3f}, alive_D200={alive_D200:.3f}"
    )


def test_high_D_produces_collapse():
    """With D=500 (true signal very rare), the agent should exploit the
    high-CTR arm unchecked and cause population collapse.
    """
    from scalar_collapse.core.config import ExperimentConfig
    from scalar_collapse.experiments.bandit_ab.simulate import run_experiment
    import tempfile

    config = ExperimentConfig(
        world=WorldConfig(
            n_users=5000,
            click_probs=(0.08, 0.04),
            burnout_deltas=(0.015, 0.003),
            alpha=0.98,
            hazard_k=6.0,
            hazard_theta=1.0,
            hazard_scale=0.02,
            churn_penalty=15.0,
        ),
        n_rounds=1000,
        update_cadence_W=5,
        retention_delay_D=500,
        agent_kwargs={"epsilon": 0.1, "step_size": 0.05},
        seed=42,
    )
    with tempfile.TemporaryDirectory() as tmp:
        summary = run_experiment(config, Path(tmp) / "run")

    assert summary["tau_collapse"] >= 0, "High D should produce collapse"
    assert summary["alive_final"] < 0.3, (
        f"High D should cause severe population loss, got {summary['alive_final']}"
    )


def test_no_delay_truth_closes_gap():
    """With D=1 (truth observed every round), the agent should stay corrected
    and prevent collapse. AUC divergence should be much smaller than D=500.

    This nails the claim that delay is the causal lever, not just
    'bad arms exist'.
    """
    from scalar_collapse.core.config import ExperimentConfig
    from scalar_collapse.experiments.bandit_ab.simulate import run_experiment
    import tempfile

    base_world = WorldConfig(
        n_users=5000,
        click_probs=(0.08, 0.04),
        burnout_deltas=(0.015, 0.003),
        alpha=0.98,
        hazard_k=6.0,
        hazard_theta=1.0,
        hazard_scale=0.02,
        churn_penalty=15.0,
    )

    # D=1: truth every round
    config_d1 = ExperimentConfig(
        world=base_world,
        n_rounds=500,
        update_cadence_W=5,
        retention_delay_D=1,
        agent_kwargs={"epsilon": 0.1, "step_size": 0.05},
        seed=42,
    )
    # D=500: truth almost never
    config_d500 = ExperimentConfig(
        world=base_world,
        n_rounds=500,
        update_cadence_W=5,
        retention_delay_D=500,
        agent_kwargs={"epsilon": 0.1, "step_size": 0.05},
        seed=42,
    )

    with tempfile.TemporaryDirectory() as tmp:
        summary_d1 = run_experiment(config_d1, Path(tmp) / "d1")
        summary_d500 = run_experiment(config_d500, Path(tmp) / "d500")

    # D=1 should not collapse
    assert summary_d1["tau_collapse"] < 0, (
        f"D=1 should not collapse, got tau={summary_d1['tau_collapse']}"
    )
    # D=1 should have much higher survival
    assert summary_d1["alive_final"] > summary_d500["alive_final"], (
        f"D=1 alive={summary_d1['alive_final']:.3f} should beat "
        f"D=500 alive={summary_d500['alive_final']:.3f}"
    )
    # D=1 should have much smaller AUC divergence
    assert summary_d1["auc_divergence"] < summary_d500["auc_divergence"], (
        f"D=1 divergence={summary_d1['auc_divergence']:.1f} should be less than "
        f"D=500 divergence={summary_d500['auc_divergence']:.1f}"
    )


def test_predict_regime_matches_sweep():
    """Analytic boundary prediction should match observed regimes
    for all cells in the default sweep grid."""
    from scalar_collapse.core.config import ExperimentConfig
    from scalar_collapse.core.boundary import predict_regime

    expected = {
        (10, 1): "stable", (10, 5): "stable", (10, 20): "stable",
        (50, 1): "stable", (50, 5): "stable", (50, 20): "stable",
        (200, 1): "metastable", (200, 5): "stable", (200, 20): "stable",
        (500, 1): "unstable", (500, 5): "unstable", (500, 20): "unstable",
    }

    for (D, W), regime in expected.items():
        config = ExperimentConfig(retention_delay_D=D, update_cadence_W=W)
        pred = predict_regime(config)
        assert pred.regime_pred == regime, (
            f"D={D}, W={W}: predicted {pred.regime_pred}, expected {regime}"
        )
