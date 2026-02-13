# Scalar Reward Collapse

**A canonical instantiation of scalar reward collapse for CFDD paper section 3.8.**

A bandit/AB simulator with a hazard-churn population model that produces phase-change behavior: proxy optimization outruns delayed ground-truth feedback, causing population collapse. The system exhibits three regimes (stable, metastable, unstable) controlled by the retention delay D and update cadence W.

**Origin:** this repo implements the "scalar reward collapse" domain instantiation from the Cybernetic Feedback Dynamics and Drift (CFDD) framework. It demonstrates that proxy reward optimization under delayed true-objective feedback is a phase transition, not a moral failing.

## What it models

A recommendation system optimizing click-through rate (CTR) while users accumulate burnout and churn:

- **Proxy metric (fast):** CTR — observed every round
- **True objective (slow):** Survival-weighted utility — corrected every D rounds
- **Harm model:** Burnout accumulates per user; churn hazard = h_scale * sigmoid(k * (B - theta))
- **Collapse:** When alive_fraction drops below 0.5 (the constraint C_k)

## Why this exists

Scalar proxy optimization with delayed ground-truth feedback produces collapse as a **predictable phase transition**. This repo:

1. **Demonstrates** three regimes (stable / metastable / unstable) via D/W sweep
2. **Predicts** the regime boundary analytically from config parameters (no simulation needed)
3. **Shows** that a trivial controller (throttle proxy learning when population drops) shifts the phase boundary
4. **Exports** paper-ready spec blocks, heatmaps, and structured signals for governor integration

## Installation

```bash
git clone https://github.com/unpingable/scalar-reward-collapse.git
cd scalar-reward-collapse
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

### Run a single experiment

```bash
python -m scalar_collapse run
```

### Run the default D/W sweep (36 runs)

```bash
python -m scalar_collapse sweep
```

### Predict regime without simulation

```bash
python -m scalar_collapse predict --D 500 --W 5
# → unstable (first-interval collapse at t=317, D=500 > 317)

python -m scalar_collapse predict --D 10 --W 5
# → stable (correction arrives before damage)
```

### Print the v0.2 spec block

```bash
python -m scalar_collapse spec --run-dir runs/sweep/D500_W5_s42
```

### Generate plots

```bash
python -m scalar_collapse plot --run-dir runs/sweep/D500_W5_s42
```

### Run tests

```bash
pytest tests/
```

## Three regimes

The sweep over D (retention delay) and W (update cadence) produces:

| D | W=1 | W=5 | W=20 | Regime |
|---|-----|-----|------|--------|
| **10** | 0.87 | 0.87 | 0.87 | **Stable** — no collapse within T=1000 |
| **50** | 0.83 | 0.84 | 0.84 | **Stable** |
| **200** | 0.40 | 0.60 | 0.60 | **Metastable** — 2/9 seeds collapse, tau=550-611 |
| **500** | 0.11 | 0.24 | 0.25 | **Unstable** — 9/9 collapse, tau~314 |

Values are `alive_final_mean` (fraction of population surviving at T=1000).

### Regime predicates (within horizon T)

- **Stable:** No runs violate C_k within horizon T
- **Metastable:** Some seeds violate C_k; tau large or high variance across seeds
- **Unstable:** All seeds violate C_k; tau bounded above

## Analytic boundary prediction

`predict_regime()` predicts the regime from config parameters without simulation. Two independent clocks:

1. **Controller clock** (`t_cross`): Mean-field Q iteration with state-dependent selection probabilities finds when the agent's value estimates cross over from true-optimal to proxy-optimal arm.

2. **Plant clock** (`t_damage_post`): Deterministic burnout iteration computes rounds until cumulative hazard violates C_k.

**Key insight:** The agent starts at Q=[0,0] with the first D-correction at t=D. If D > t_damage_first (approximately 317 rounds), the population collapses before any correction arrives. This is the dominant mechanism — W is irrelevant when D exceeds the first-interval kill threshold.

Validated 12/12 against the observed sweep grid.

## Controller baseline

The `ControlledEpsilonGreedy` agent throttles proxy learning when `alive_fraction < 0.9`. This shifts the phase boundary:

| D | Vanilla | Controlled |
|---|---------|------------|
| 200 | 2/9 collapse (metastable) | 0/9 collapse (stable) |
| 500 | 9/9 collapse (unstable) | 9/9 collapse (unstable) |

Proof that governance can shift regimes — collapse is a control problem, not an inevitability.

## Canonical instantiation (v0.2 spec block)

```
§3.8 Canonical Instantiation: Scalar Reward Collapse

  Domain:           Scalar Reward Collapse
  Plant:            Burnout-Hazard Population Model
  Proxy metric:     CTR (immediate)
  True objective:   Survival-weighted utility
  Controller:       None

  Plant dynamics:   B_{t+1} = 0.98 * B_t + delta(a); delta = (0.015, 0.003)
  H (harm model):   h = 0.02 * sigmoid(6.0 * (B - 1.0))

  C_k (constraint): alive_fraction >= 0.5 over horizon T=1000

  W (loop period):  5 rounds
  D (truth delay):  500 rounds

  sigma spec:       count(t : proxy_reward_delta[t] > 0 AND true_reward_delta[t] < 0)
  sigma_total:      59
  sigma_damage:     16 (at tau_damage)
  sigma_rate:       0.051 (sigma_damage / tau_damage)

  tau_damage:       316
  alive(T):         0.2206
  Failure signature: Long Quiet -> Flicker -> Snap at t=316
```

## Project structure

```
scalar-reward-collapse/
  src/scalar_collapse/
    core/
      config.py         # WorldConfig, ExperimentConfig, SweepConfig dataclasses
      metrics.py        # policy_entropy, alive_fraction, sigma_count, damage_onset
      rng.py            # Deterministic seeding via numpy.random.Generator
      template.py       # TemplateInst v0.2 + to_spec_block() formatter
      boundary.py       # predict_regime() analytic boundary prediction
    experiments/bandit_ab/
      env.py            # HazardChurnEnv (burnout + sigmoid churn hazard)
      agents.py         # EpsilonGreedy, UCB, Thompson, Gradient, Controlled
      simulate.py       # Main simulation loop (proxy updates + D-corrections)
      sweep.py          # D x W x seed grid sweep with regime classification
      policies.py       # Throttle policies (alive, sigma-rate, predictive)
      kappa_sweep.py    # Kappa policy sweep runner + offline L computation
      kappa_plots.py    # Efficient frontier, threshold vs kappa, boundary shift
      plots.py          # Heatmaps, proxy vs true, churn cliff, sigma trace
    runstore/
      manifest.py       # Run ID, config hash, artifact hashes
      writer.py         # Atomic writes, crash-safe manifest (written last)
      trajectory.py     # NDJSON trajectory logging
      diff.py           # Config/metric diff between runs
    cli.py              # CLI: run, sweep, predict, spec, summarize, diff, plot, kappa-sweep
  tests/
    test_env.py         # Environment + simulation + boundary prediction tests
    test_metrics.py     # Metric function unit tests
    test_runstore.py    # Run store, manifest, trajectory tests
    test_kappa_sweep.py # Kappa sweep policies + selection tests
  scripts/
    run_bandit_sweep.sh # Shell helper for default sweep
```

## Run bundles

Every experiment produces an immutable bundle:

```
runs/<run_dir>/
  manifest.json     # Hash-pinned metadata (written last = crash-safety gate)
  config.json       # Full experiment configuration
  summary.json      # Metrics + TemplateInst + regime classification
  metrics.ndjson    # Per-round trajectory (NDJSON)
```

## Key concepts

| Symbol | Name | Definition |
|--------|------|------------|
| D | Retention delay | Rounds between true-objective corrections |
| W | Update cadence | Rounds between proxy (CTR) updates |
| T | Horizon | Total rounds; regime labels are "within T" |
| sigma | Divergence count | Timesteps where proxy improves but true degrades |
| sigma_rate | Normalized sigma | sigma_at_tau / tau_damage |
| tau_damage | Damage onset | First timestep alive_fraction < 0.5 |
| kappa | Abort budget | Cost of false-positive governor intervention (v0.3) |
| C_k | Constraint | alive_fraction >= 0.5 |

## Conventions

- All randomness flows through `numpy.random.Generator` seeded via `core.rng`
- Run artifacts go to `runs/<run_id>/` with manifest written last (crash-safety gate)
- Trajectory data is NDJSON (one JSON object per line)
- Config dataclasses live in `core/config.py`; never use raw dicts for configuration

## Non-goals

- We don't model real recommendation systems (this is a minimal instantiation)
- We don't optimize the controller (that's the governor's job)
- We focus on **demonstrating the phase transition** and **predicting the boundary**

## Research lineage

This repo is the companion code for two papers:

> Beck, J. (2025). "Scalar Reward Collapse: A General Theory of Eigenstructure Evaporation in Closed-Loop Systems." [DOI: 10.5281/zenodo.17791872](https://zenodo.org/records/17791873)

> Beck, J. (2026). "Cybernetic Fault Domains: When Commitment Outruns Verification." Section 3.8. [DOI: 10.5281/zenodo.18518894](https://zenodo.org/records/18518895)

Paper #03 (Scalar Reward Collapse) provides the mathematical theory — eigenstructure evaporation under scalar optimization. Paper #15 (Cybernetic Fault Domains) provides the CFDD framework template (C_k, H, sigma, tau, kappa) that this repo instantiates with a concrete bandit/AB environment.

## License

Apache 2.0
