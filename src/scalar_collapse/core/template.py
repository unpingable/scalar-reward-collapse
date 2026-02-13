"""Paper template dataclass and derivation logic for CFDD §3.8."""

from __future__ import annotations

from dataclasses import dataclass

from scalar_collapse.core.config import ExperimentConfig


@dataclass
class TemplateInst:
    """v0.2 canonical instantiation block for §3.8.

    Internal representation stays rich (JSON-serializable).
    Use to_spec_block() for the paper-ready presentation layer.
    """

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
    # Control policy
    controller: str = "None"  # "None" or description of controller
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
    alive_final: float = 0.0  # alive fraction at end of horizon
    failure_signature: str = ""

    def to_spec_block(self) -> str:
        """Format as v0.2 canonical instantiation block for the paper.

        This is the presentation layer — field renames for spec readability
        while internal JSON stays unchanged.
        """
        lines = []
        lines.append("§3.8 Canonical Instantiation: Scalar Reward Collapse")
        lines.append("=" * 55)
        lines.append("")

        # Domain / Plant / Metrics
        lines.append(f"  Domain:           {self.domain.replace('_', ' ').title()}")
        lines.append(f"  Plant:            Burnout–Hazard Population Model")
        lines.append(f"  Proxy metric:     CTR (immediate)")
        lines.append(f"  True objective:   Survival-weighted utility")
        lines.append(f"  Controller:       {self.controller}")
        lines.append("")

        # Plant dynamics
        lines.append(f"  Plant dynamics:   {self.burnout_dynamics}")
        lines.append(f"  H (harm model):   {self.H}")
        lines.append("")

        # Constraint & horizon
        lines.append(f"  C_k (constraint): {self.C_k} over horizon T={self.T}")
        lines.append("")

        # Timing
        lines.append(f"  T_commit:         {self.T_commit}")
        lines.append(f"  W (loop period):  {self.W:.0f} rounds")
        lines.append(f"  D (truth delay):  {self.D:.0f} rounds")
        lines.append(f"  A (actuation):    {self.A:.0f} rounds")
        lines.append("")

        # Sigma
        lines.append(f"  σ spec:           {self.sigma_spec}")
        lines.append(f"  σ_total:          {self.sigma_value:.0f}")
        if self.tau_collapse >= 0:
            lines.append(f"  σ_damage:         {self.sigma_at_tau:.0f} (σ at τ_damage)")
            lines.append(f"  σ̇_damage:         {self.sigma_rate:.4f} (σ_damage / τ_damage)")
        else:
            lines.append(f"  σ_damage:         n/a (no collapse)")
            lines.append(f"  σ̇_damage:         n/a")
        lines.append(f"  σ_threshold:      {self.sigma_threshold:.0f} (calibration: damage onset)")
        lines.append(f"  κ:                {self.kappa:.1f} ({self.kappa_note})")
        lines.append("")

        # Outcome
        tau_str = str(self.tau_collapse) if self.tau_collapse >= 0 else "none (stable within T)"
        lines.append(f"  τ_damage:         {tau_str}")
        lines.append(f"  alive(T):         {self.alive_final:.4f}")
        lines.append(f"  Proxy–True divergence (AUC): {self.auc_divergence:.1f}")
        lines.append(f"  Failure signature: {self.failure_signature}")

        return "\n".join(lines)


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

    # Determine controller description from agent type
    agent_type = config.agent_type
    if agent_type == "controlled_epsilon_greedy":
        threshold = config.agent_kwargs.get("alive_threshold", 0.9)
        controller_desc = f"Throttle-on-Alive (alive < {threshold} stops proxy learning)"
    else:
        controller_desc = "None"

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
        controller=controller_desc,
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
        alive_final=summary.get("alive_final", 0.0),
        failure_signature=summary.get("failure_signature", ""),
    )
