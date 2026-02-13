"""Main simulation loop: serve -> click -> burnout -> churn -> update."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scalar_collapse.core.config import ExperimentConfig
from scalar_collapse.core.metrics import (
    damage_onset,
    policy_entropy,
    proxy_true_divergence,
    sigma_count,
)
from scalar_collapse.core.rng import seed_all
from scalar_collapse.core.template import derive_template
from scalar_collapse.experiments.bandit_ab.agents import make_agent
from scalar_collapse.experiments.bandit_ab.env import HazardChurnEnv, sigmoid_scalar
from scalar_collapse.runstore.trajectory import TrajectoryWriter
from scalar_collapse.runstore.writer import store_run


def _compute_true_arm_values(config: ExperimentConfig) -> list[float]:
    """Compute model-based true reward for each arm.

    Uses the world parameters to compute steady-state burnout and expected
    churn hazard. This represents the retention team's best estimate of
    each arm's long-term impact.

    true_r[arm] = click_prob[arm] - churn_penalty * expected_hazard[arm]
    """
    w = config.world
    values = []
    for a in range(w.n_arms):
        ss_burnout = w.burnout_deltas[a] / (1.0 - w.alpha)
        expected_hazard = w.hazard_scale * sigmoid_scalar(
            w.hazard_k * (ss_burnout - w.hazard_theta)
        )
        true_r = w.click_probs[a] - w.churn_penalty * expected_hazard
        values.append(true_r)
    return values


def run_experiment(config: ExperimentConfig, run_dir: Path, policy=None) -> dict:
    """Run a single bandit experiment.

    Main loop:
    - Each round: serve arm to population, collect clicks, update burnout/churn
    - Every W rounds: agent updates policy using proxy (CTR)
    - Every D rounds: retention team overrides agent's value estimates with
      model-based true rewards (CTR minus expected churn cost). Between
      overrides, the optimizer drifts back toward proxy-optimal on CTR alone.
    - Log per-step metrics to trajectory
    - Compute summary + template at end
    - Store run via runstore

    If policy is provided, policy.should_throttle(t, state) overrides
    the agent's own update decision. When policy is None, behavior is
    identical to v0.2 — backward compatible.

    Returns the summary dict.
    """
    started_at = datetime.now(timezone.utc)
    run_dir = Path(run_dir)
    rng = seed_all(config.seed)

    env = HazardChurnEnv(config.world, rng)
    agent = make_agent(
        config.agent_type, config.world.n_arms, rng, **config.agent_kwargs
    )

    w = config.world

    # Pre-compute model-based true rewards for each arm
    true_arm_values = _compute_true_arm_values(config)

    # Tracking arrays
    proxy_rewards = []  # per-round CTR
    true_rewards = []  # per-round churn-penalized reward
    proxy_cumulative = []
    true_cumulative = []
    entropies = []
    alive_fracs = []
    sigma_cumulative = []

    proxy_cum = 0.0
    true_cum = 0.0
    sigma_cum = 0

    # Rolling sigma rate window
    sigma_window_size = 50
    sigma_indicators: list[int] = []  # per-step sigma indicator (0 or 1)
    sigma_rate_window = 0.0
    throttle_count = 0

    writer = TrajectoryWriter(run_dir)

    try:
        for t in range(config.n_rounds):
            arm = agent.select_arm()

            # Inform controlled agents of population state
            if hasattr(agent, "observe_alive"):
                prev_alive = env.alive.sum() / env._initial_n
                agent.observe_alive(prev_alive)

            result = env.step(arm)

            # Proxy reward: CTR (what the agent optimizes)
            proxy_r = result.ctr

            # True reward for logging: use model-based value for the played arm
            true_r = true_arm_values[arm]

            proxy_rewards.append(proxy_r)
            true_rewards.append(true_r)
            proxy_cum += proxy_r
            true_cum += true_r
            proxy_cumulative.append(proxy_cum)
            true_cumulative.append(true_cum)

            # Sigma: proxy improves but true degrades
            proxy_delta = 0.0
            true_delta = 0.0
            indicator = 0
            if len(proxy_rewards) >= 2:
                proxy_delta = proxy_rewards[-1] - proxy_rewards[-2]
                true_delta = true_rewards[-1] - true_rewards[-2]
                if proxy_delta > 0 and true_delta < 0:
                    sigma_cum += 1
                    indicator = 1
            sigma_cumulative.append(sigma_cum)

            # Rolling sigma rate
            sigma_indicators.append(indicator)
            window_start = max(0, len(sigma_indicators) - sigma_window_size)
            window_slice = sigma_indicators[window_start:]
            sigma_rate_window = sum(window_slice) / len(window_slice)

            # Entropy of agent's arm selection distribution
            ent = policy_entropy(agent.arm_probs())
            entropies.append(ent)
            alive_fracs.append(result.alive_fraction)

            # Build state dict for policy
            state = {
                "alive_fraction": result.alive_fraction,
                "sigma_rate_window": sigma_rate_window,
                "D": config.retention_delay_D,
                "proxy_delta": proxy_delta,
                "true_delta": true_delta,
            }

            # Agent updates on proxy every W rounds
            # Policy throttle overrides agent's own update decision
            throttled = False
            if policy is not None:
                throttled = policy.should_throttle(t, state)
            elif hasattr(agent, "should_update_proxy"):
                # v0.2 backward compat: ControlledEpsilonGreedy path
                throttled = not agent.should_update_proxy()

            if not throttled and (t + 1) % config.update_cadence_W == 0:
                agent.update(arm, proxy_r)

            if throttled:
                throttle_count += 1

            # True-signal correction every D rounds:
            # Retention team overrides the optimizer's value estimates
            # with model-based true rewards for ALL arms.
            # Small D → frequent overrides → agent stays corrected → stable
            # Large D → rare overrides → agent drifts on proxy → collapse
            if (t + 1) % config.retention_delay_D == 0:
                for a in range(w.n_arms):
                    if hasattr(agent, "force_value"):
                        agent.force_value(a, true_arm_values[a])
                    else:
                        agent.update(a, true_arm_values[a])

            # Log trajectory
            writer.append({
                "t": t,
                "arm": arm,
                "clicks": result.clicks,
                "new_churns": result.new_churns,
                "alive_count": result.alive_count,
                "alive_fraction": result.alive_fraction,
                "ctr": result.ctr,
                "proxy_reward": proxy_r,
                "true_reward": true_r,
                "proxy_cumulative": proxy_cum,
                "true_cumulative": true_cum,
                "entropy": ent,
                "sigma_cumulative": sigma_cum,
                "sigma_rate_window": sigma_rate_window,
                "throttled": throttled,
                "burnout_mean": result.mean_burnout,
                "burnout_std": result.std_burnout,
            })

            # Periodic flush
            if (t + 1) % 100 == 0:
                writer.flush()
    finally:
        writer.close()

    # Compute summary
    proxy_arr = np.array(proxy_rewards)
    true_arr = np.array(true_rewards)
    proxy_cum_arr = np.array(proxy_cumulative)
    true_cum_arr = np.array(true_cumulative)
    sigma_arr = np.array(sigma_cumulative)

    alive_floor = 0.5
    tau = damage_onset(np.array(alive_fracs), alive_floor)
    auc_div = proxy_true_divergence(proxy_cum_arr, true_cum_arr)
    sigma_total = sigma_count(proxy_arr, true_arr)

    # sigma at damage onset: the sigma value when tau_collapse fires
    sigma_at_tau = int(sigma_arr[tau]) if tau >= 0 and tau < len(sigma_arr) else 0

    # sigma_rate: normalized sigma per round up to damage onset
    sigma_rate = float(sigma_at_tau) / float(tau) if tau > 0 else 0.0

    # Failure signature
    if tau >= 0:
        failure_sig = f"Long Quiet -> Flicker -> Snap at t={tau}"
    else:
        failure_sig = "no_collapse"

    # Explicit alive metrics (continuous) separate from regime label (categorical)
    alive_final = alive_fracs[-1] if alive_fracs else 0.0
    collapsed = tau >= 0

    proxy_return = float(proxy_cum)

    summary = {
        "T": config.n_rounds,
        "seed": config.seed,
        "agent_type": config.agent_type,
        "D": config.retention_delay_D,
        "W": config.update_cadence_W,
        "alive_final": alive_final,
        "alive_floor": alive_floor,
        "collapsed": collapsed,
        "tau_collapse": tau,
        "auc_divergence": auc_div,
        "sigma_total": sigma_total,
        "sigma_at_tau": sigma_at_tau,
        "sigma_rate": sigma_rate,
        "sigma_threshold": float(sigma_at_tau) if tau >= 0 else 0.0,
        "sigma_threshold_method": "sigma_value_at_damage_onset",
        "kappa": 0.0,  # no FP/abort budget modeled yet
        "kappa_note": "not modeled; reserved for future false-positive abort budget",
        "final_entropy": entropies[-1] if entropies else 0.0,
        "final_proxy_cumulative": proxy_cumulative[-1] if proxy_cumulative else 0.0,
        "final_true_cumulative": true_cumulative[-1] if true_cumulative else 0.0,
        "proxy_return": proxy_return,
        "throttle_count": throttle_count,
        "failure_signature": failure_sig,
        "true_arm_values": true_arm_values,
    }

    # Derive template
    template = derive_template(summary, config)
    summary["template"] = {
        k: v for k, v in template.__dict__.items()
    }

    # Store run bundle
    manifest = store_run(
        run_dir=run_dir,
        config=config,
        trajectory_path=writer.path,
        summary=summary,
        started_at=started_at,
    )
    summary["run_id"] = manifest.run_id

    return summary
