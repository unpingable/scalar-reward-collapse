"""Grid sweep over D/W values with multiple seeds."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from scalar_collapse.core.config import ExperimentConfig, SweepConfig, WorldConfig
from scalar_collapse.experiments.bandit_ab.simulate import run_experiment


def run_sweep(sweep_config: SweepConfig, output_dir: Path) -> list[dict]:
    """Run a grid sweep over D_values x W_values x seeds.

    Each cell runs one experiment. Returns a list of summary dicts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(sweep_config.D_values) * len(sweep_config.W_values) * len(sweep_config.seeds)
    count = 0

    for D in sweep_config.D_values:
        for W in sweep_config.W_values:
            for seed in sweep_config.seeds:
                count += 1
                print(f"[{count}/{total}] D={D}, W={W}, seed={seed}")

                cell_config = ExperimentConfig(
                    world=sweep_config.base.world,
                    n_rounds=sweep_config.base.n_rounds,
                    update_cadence_W=W,
                    retention_delay_D=D,
                    agent_type=sweep_config.base.agent_type,
                    agent_kwargs=sweep_config.base.agent_kwargs,
                    seed=seed,
                )

                run_dir = output_dir / f"D{D}_W{W}_s{seed}"
                summary = run_experiment(cell_config, run_dir)
                summary["sweep_D"] = D
                summary["sweep_W"] = W
                summary["sweep_seed"] = seed
                results.append(summary)

    # Compute per-cell aggregates
    cells: dict[tuple[int, int], list[dict]] = {}
    for r in results:
        key = (r["sweep_D"], r["sweep_W"])
        cells.setdefault(key, []).append(r)

    cell_summaries = []
    T = sweep_config.base.n_rounds
    for (D, W), cell_runs in sorted(cells.items()):
        n_seeds = len(cell_runs)
        n_collapsed = sum(1 for r in cell_runs if r["collapsed"])
        alive_vals = [r["alive_final"] for r in cell_runs]
        alive_final_mean = sum(alive_vals) / len(alive_vals)
        collapse_rate = n_collapsed / n_seeds

        # Regime predicate (within horizon T)
        if collapse_rate == 0.0:
            regime = "stable"
        elif collapse_rate < 1.0:
            regime = "metastable"
        else:
            regime = "unstable"

        cell_summaries.append({
            "D": D, "W": W, "T": T,
            "alive_final_mean": round(alive_final_mean, 4),
            "collapse_rate": round(collapse_rate, 3),
            "n_collapsed": n_collapsed,
            "n_seeds": n_seeds,
            "regime": regime,
        })

    # Determine controller type from first result
    agent_type = results[0].get("agent_type", "epsilon_greedy") if results else "unknown"
    if agent_type == "controlled_epsilon_greedy":
        controller = "Throttle-on-Alive"
    else:
        controller = "None"

    # Write sweep summary (per-run + per-cell)
    sweep_output = {
        "per_run": results,
        "per_cell": cell_summaries,
        "controller": controller,
        "regime_definitions": {
            "stable": "No runs violate C_k within horizon T",
            "metastable": "Some seeds violate C_k; tau large or high variance across seeds",
            "unstable": "All seeds violate C_k; tau bounded above",
            "T": T,
            "C_k": f"alive_fraction >= {results[0].get('alive_floor', 0.5) if results else 0.5}",
        },
    }
    sweep_summary_path = output_dir / "sweep_summary.json"
    with open(sweep_summary_path, "w") as f:
        json.dump(sweep_output, f, indent=2, default=str)

    print(f"Sweep complete: {len(results)} runs. Summary: {sweep_summary_path}")
    return results
