"""Paper template dataclass and derivation logic for CFDD §3.8."""

from __future__ import annotations

from dataclasses import dataclass

from scalar_collapse.core.config import ExperimentConfig


@dataclass
class TemplateInst:
    """v0.2 canonical instantiation block for §3.8."""

    domain: str = "scalar_reward_collapse"
    # Plant dynamics (the actual system being controlled)
    burnout_dynamics: str = ""  # B_{t+1} = alpha * B_t + delta(a)
    H: str = ""  # harm model (churn hazard function)
    C_k: str = ""  # constraint definition (e.g., "alive_fraction >= 0.5")
    T: int = 0  # horizon (n_rounds); regime labels are always "within T"
    # Timing parameters
    T_commit: str = ""  # when policy updates take effect
    W: float = 0.0  # update cadence (rounds)
    D: float = 0.0  # retention observability delay
    A: float = 0.0  # actuation time (= W; updates are synchronous)
    # Sigma accounting
    sigma_spec: str = ""  # how sigma is counted
    sigma_value: float = 0.0  # total sigma over the run
    sigma_max: float = 0.0  # theoretical max (= n_rounds)
    sigma_at_tau: float = 0.0  # sigma at the moment of damage onset
    sigma_rate: float = 0.0  # sigma_at_tau / tau_collapse (normalized)
    kappa: float = 0.0  # FP/abort budget (0 = not modeled)
    kappa_note: str = ""  # explanation of kappa status
    sigma_threshold: float = 0.0  # = sigma_at_tau when collapse occurs
    sigma_threshold_method: str = "sigma_value_at_damage_onset"
    # Outcome
    tau_collapse: int = 0  # first timestep constraint violated (-1 = none)
    auc_divergence: float = 0.0  # integral(proxy - true) dt
    failure_signature: str = ""


def derive_template(summary: dict, config: ExperimentConfig) -> TemplateInst:
    """Derive paper template block from run summary + config."""
    w = config.world
    tau = summary.get("tau_collapse", -1)
    sigma_at_tau = summary.get("sigma_at_tau", 0)

    # sigma_rate: normalized sigma per round up to damage onset
    if tau > 0:
        sigma_rate = float(sigma_at_tau) / float(tau)
    else:
        sigma_rate = 0.0

    return TemplateInst(
        domain="scalar_reward_collapse",
        burnout_dynamics=(
            f"B_{{t+1}} = {w.alpha} * B_t + delta(a); "
            f"delta = {w.burnout_deltas}; "
            f"steady-state B* = delta / (1 - {w.alpha})"
        ),
        H=f"h = {w.hazard_scale} * sigmoid({w.hazard_k} * (B - {w.hazard_theta}))",
        C_k=f"alive_fraction >= {summary.get('alive_floor', 0.5)}",
        T=config.n_rounds,
        T_commit=f"every {config.update_cadence_W} rounds (proxy); every {config.retention_delay_D} rounds (true correction)",
        W=float(config.update_cadence_W),
        D=float(config.retention_delay_D),
        A=float(config.update_cadence_W),
        sigma_spec="count(t : proxy_reward_delta[t] > 0 AND true_reward_delta[t] < 0)",
        sigma_value=summary.get("sigma_total", 0.0),
        sigma_max=float(config.n_rounds),
        sigma_at_tau=float(sigma_at_tau),
        sigma_rate=sigma_rate,
        kappa=0.0,
        kappa_note="not modeled; reserved for future false-positive abort budget",
        sigma_threshold=float(sigma_at_tau) if tau >= 0 else 0.0,
        sigma_threshold_method="sigma_value_at_damage_onset",
        tau_collapse=tau,
        auc_divergence=summary.get("auc_divergence", 0.0),
        failure_signature=summary.get("failure_signature", ""),
    )
