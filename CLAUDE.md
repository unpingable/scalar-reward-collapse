# scalar-reward-collapse

Canonical instantiation of scalar reward collapse for CFDD paper section 3.8.

## Project Structure

- `src/scalar_collapse/core/` — Config, RNG, template, metrics
- `src/scalar_collapse/runstore/` — Run manifest, atomic writes, trajectory logging, diff
- `src/scalar_collapse/experiments/bandit_ab/` — Hazard/churn bandit environment, agents, simulation, sweep, plots
- `src/scalar_collapse/cli.py` — CLI entry point
- `tests/` — Pytest suite
- `scripts/` — Shell helpers

## Conventions

- All randomness flows through `numpy.random.Generator` seeded via `core.rng`.
- Run artifacts go to `runs/<run_id>/` with manifest written last (crash-safety gate).
- Trajectory data is NDJSON (one JSON object per line).
- Config dataclasses live in `core/config.py`; never use raw dicts for configuration.
- Tests run with `pytest tests/`.
- Entry point: `python -m scalar_collapse <subcommand>`.

## Key Concepts

- **Proxy reward**: CTR (click-through rate) — fast, observable every round.
- **True reward**: Survival-weighted utility — slow, observable every D rounds.
- **sigma (σ)**: Count of timesteps where proxy improves but true degrades.
- **tau_collapse**: First timestep the true objective crosses a damage floor.
- **D**: Retention observability delay. **W**: Optimizer update cadence.

## Commands

```bash
pip install -e ".[dev]"
pytest tests/
python -m scalar_collapse run
python -m scalar_collapse sweep --sweep-config sweep.json
python -m scalar_collapse plot --run-dir runs/<id>
```
