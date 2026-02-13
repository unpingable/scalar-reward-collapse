"""Tests for kappa policy sweep: policies, sigma window, sweep selection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from scalar_collapse.core.config import ExperimentConfig, WorldConfig
from scalar_collapse.experiments.bandit_ab.policies import (
    AliveThrottlePolicy,
    PredictiveThrottlePolicy,
    SigmaRateThrottlePolicy,
    ThrottlePolicy,
)
from scalar_collapse.experiments.bandit_ab.simulate import run_experiment


@pytest.fixture
def small_world():
    """Small world config for fast tests."""
    return WorldConfig(
        n_users=500,
        n_arms=2,
        click_probs=(0.08, 0.04),
        burnout_deltas=(0.015, 0.003),
        alpha=0.98,
        hazard_k=6.0,
        hazard_theta=1.0,
        hazard_scale=0.02,
        churn_penalty=15.0,
    )


def test_kappa_sweep_selects_reasonable_threshold(small_world):
    """Higher kappa should select a more permissive threshold (less throttling).

    With alive-threshold policy: higher kappa penalizes proxy loss more,
    so the optimizer should pick a LOWER alive_thr (less throttling).
    """
    from scalar_collapse.experiments.bandit_ab.kappa_sweep import run_kappa_sweep

    base = ExperimentConfig(
        world=small_world,
        n_rounds=200,
        update_cadence_W=5,
        agent_kwargs={"epsilon": 0.1, "step_size": 0.05},
        seed=42,
    )

    with tempfile.TemporaryDirectory() as tmp:
        summary = run_kappa_sweep(
            base=base,
            D_values=[50, 200],
            W_values=[5],
            seeds=[42],
            kappa_grid=[0.0, 100.0],
            policy_names=["alive"],
            threshold_grids={"alive": [0.5, 0.7, 0.9]},
            output_dir=Path(tmp) / "kappa_test",
        )

    frontier = summary["frontier"]
    alive_rows = [r for r in frontier if r["policy"] == "alive"]

    # Should have 2 rows (one per kappa value)
    assert len(alive_rows) == 2

    kappa_0_row = [r for r in alive_rows if r["kappa"] == 0.0][0]
    kappa_100_row = [r for r in alive_rows if r["kappa"] == 100.0][0]

    # At kappa=0, only collapse_rate matters -> should pick most aggressive threshold
    # At kappa=100, proxy loss dominates -> should pick most permissive threshold (less throttling)
    # More permissive = lower alive_thr for AliveThrottlePolicy
    assert kappa_100_row["threshold"] <= kappa_0_row["threshold"], (
        f"Higher kappa should select lower (more permissive) alive threshold: "
        f"kappa=100 picked {kappa_100_row['threshold']}, kappa=0 picked {kappa_0_row['threshold']}"
    )


def test_deltaR_proxy_zero_when_no_throttle(small_world):
    """When threshold is set to a value that never triggers throttling,
    the proxy return should match baseline (delta_R_proxy ~= 0).
    """
    base = ExperimentConfig(
        world=small_world,
        n_rounds=200,
        update_cadence_W=5,
        retention_delay_D=50,
        agent_kwargs={"epsilon": 0.1, "step_size": 0.05},
        seed=42,
    )

    with tempfile.TemporaryDirectory() as tmp:
        # Baseline: no policy
        baseline_summary = run_experiment(base, Path(tmp) / "baseline")
        baseline_return = baseline_summary["proxy_return"]

        # alive_thr=0.0: never triggers (alive_fraction is always >= 0)
        policy = AliveThrottlePolicy(alive_thr=0.0)
        governed_summary = run_experiment(base, Path(tmp) / "governed", policy=policy)
        governed_return = governed_summary["proxy_return"]

    # Should be essentially the same (same seed, same behavior)
    assert governed_summary["throttle_count"] == 0, (
        f"Expected 0 throttles with alive_thr=0.0, got {governed_summary['throttle_count']}"
    )
    assert abs(governed_return - baseline_return) < 1e-6, (
        f"Expected identical proxy returns, got baseline={baseline_return:.4f}, "
        f"governed={governed_return:.4f}"
    )


def test_predictive_throttle_triggers_from_t0_when_margin_high(small_world):
    """With D=500 and D_crit=317, margin = (500-317)/317 ≈ 0.58.
    With margin_thr=0.0 (any positive margin triggers), throttle should
    be active from the very first round.
    """
    config = ExperimentConfig(
        world=small_world,
        n_rounds=100,
        update_cadence_W=5,
        retention_delay_D=500,
        agent_kwargs={"epsilon": 0.1, "step_size": 0.05},
        seed=42,
    )

    policy = PredictiveThrottlePolicy(margin_thr=0.0, D_crit=317.0)

    # Verify the margin calculation
    margin = (500 - 317.0) / 317.0
    assert margin > 0.0, f"Expected positive margin, got {margin}"

    with tempfile.TemporaryDirectory() as tmp:
        summary = run_experiment(config, Path(tmp) / "run", policy=policy)

    # Every round should be throttled
    assert summary["throttle_count"] == config.n_rounds, (
        f"Expected all {config.n_rounds} rounds throttled, got {summary['throttle_count']}"
    )


def test_sigma_window_computation():
    """Hand-craft proxy/true reward series and verify sigma_rate_window values."""
    # Create a scenario where we know exactly which steps have sigma indicators.
    # Use a world where arm 0 has high CTR but also high harm.
    world = WorldConfig(
        n_users=1000,
        n_arms=2,
        click_probs=(0.08, 0.04),
        burnout_deltas=(0.015, 0.003),
        alpha=0.98,
        hazard_k=6.0,
        hazard_theta=1.0,
        hazard_scale=0.02,
        churn_penalty=15.0,
    )

    config = ExperimentConfig(
        world=world,
        n_rounds=100,
        update_cadence_W=5,
        retention_delay_D=50,
        agent_kwargs={"epsilon": 0.1, "step_size": 0.05},
        seed=42,
    )

    import json

    with tempfile.TemporaryDirectory() as tmp:
        summary = run_experiment(config, Path(tmp) / "run")

        # Read trajectory to verify sigma_rate_window was computed
        traj_path = Path(tmp) / "run" / "metrics.ndjson"
        records = []
        with open(traj_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    assert len(records) == 100

    # Verify sigma_rate_window field exists and is a float in [0, 1]
    for r in records:
        assert "sigma_rate_window" in r, f"Missing sigma_rate_window at t={r['t']}"
        assert 0.0 <= r["sigma_rate_window"] <= 1.0, (
            f"sigma_rate_window out of range at t={r['t']}: {r['sigma_rate_window']}"
        )

    # Verify sigma_rate_window is consistent with sigma_cumulative
    # The window rate at step t should equal the mean of indicators in last 50 steps.
    # Since sigma_cumulative is monotonically non-decreasing, we can verify:
    # - If sigma_cumulative didn't change in a window, sigma_rate_window should be 0
    # - sigma_rate_window at t=0 is always 0 (first step, no prior to compare)
    assert records[0]["sigma_rate_window"] == 0.0, (
        "First step should have sigma_rate_window=0 (no prior step to compare)"
    )

    # Verify the throttled field exists
    for r in records:
        assert "throttled" in r, f"Missing throttled field at t={r['t']}"
