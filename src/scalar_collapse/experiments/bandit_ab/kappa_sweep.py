"""Kappa policy sweep: threshold policies evaluated across a kappa-weighted loss.

Phase 1: Baseline sweep (no governor) to establish R_proxy_baseline.
Phase 2: Threshold sweep per policy family to collect (collapse_rate, proxy_return).
Phase 3: Offline kappa selection — for each kappa, select threshold minimizing L.

Runtime is O(#thresholds_per_family x D x W x seeds). Kappa selection is
pure offline computation on cached aggregates.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from scalar_collapse.core.config import ExperimentConfig, SweepConfig
from scalar_collapse.experiments.bandit_ab.policies import (
    POLICY_REGISTRY,
    AliveThrottlePolicy,
    PredictiveThrottlePolicy,
    SigmaRateThrottlePolicy,
    ThrottlePolicy,
)
from scalar_collapse.experiments.bandit_ab.simulate import run_experiment


def _run_cell(
    base: ExperimentConfig,
    D: int,
    W: int,
    seed: int,
    run_dir: Path,
    policy: ThrottlePolicy | None = None,
) -> dict:
    """Run a single D/W/seed cell and return its summary."""
    config = ExperimentConfig(
        world=base.world,
        n_rounds=base.n_rounds,
        update_cadence_W=W,
        retention_delay_D=D,
        agent_type=base.agent_type,
        agent_kwargs=base.agent_kwargs,
        seed=seed,
    )
    return run_experiment(config, run_dir, policy=policy)


def _run_grid(
    base: ExperimentConfig,
    D_values: list[int],
    W_values: list[int],
    seeds: list[int],
    output_dir: Path,
    policy: ThrottlePolicy | None = None,
    label: str = "",
) -> list[dict]:
    """Run a full D x W x seeds grid, return list of summaries."""
    results = []
    total = len(D_values) * len(W_values) * len(seeds)
    count = 0
    for D in D_values:
        for W in W_values:
            for seed in seeds:
                count += 1
                tag = f" [{label}]" if label else ""
                print(f"  [{count}/{total}]{tag} D={D}, W={W}, seed={seed}")
                run_dir = output_dir / f"D{D}_W{W}_s{seed}"
                summary = _run_cell(base, D, W, seed, run_dir, policy=policy)
                summary["sweep_D"] = D
                summary["sweep_W"] = W
                summary["sweep_seed"] = seed
                results.append(summary)
    return results


def _aggregate_results(results: list[dict]) -> dict:
    """Compute collapse_rate and proxy_return_mean from a list of run summaries."""
    if not results:
        return {"collapse_rate": 0.0, "proxy_return_mean": 0.0, "alive_min": 1.0}
    n = len(results)
    n_collapsed = sum(1 for r in results if r.get("collapsed", False))
    proxy_returns = [r.get("proxy_return", r.get("final_proxy_cumulative", 0.0)) for r in results]
    alive_finals = [r.get("alive_final", 1.0) for r in results]
    return {
        "collapse_rate": n_collapsed / n,
        "proxy_return_mean": float(np.mean(proxy_returns)),
        "alive_min": float(np.min(alive_finals)),
    }


def _make_threshold_grid(policy_name: str, thresholds: list[float]) -> list[ThrottlePolicy]:
    """Create a list of policy instances for the given threshold values."""
    if policy_name == "alive":
        return [AliveThrottlePolicy(alive_thr=t) for t in thresholds]
    elif policy_name == "sigma":
        return [SigmaRateThrottlePolicy(sigma_thr=t) for t in thresholds]
    elif policy_name == "predictive":
        return [PredictiveThrottlePolicy(margin_thr=t) for t in thresholds]
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def _get_threshold_value(policy: ThrottlePolicy) -> float:
    """Extract the threshold parameter value from a policy instance."""
    if isinstance(policy, AliveThrottlePolicy):
        return policy.alive_thr
    elif isinstance(policy, SigmaRateThrottlePolicy):
        return policy.sigma_thr
    elif isinstance(policy, PredictiveThrottlePolicy):
        return policy.margin_thr
    return 0.0


def run_kappa_sweep(
    base: ExperimentConfig,
    D_values: list[int],
    W_values: list[int],
    seeds: list[int],
    kappa_grid: list[float],
    policy_names: list[str],
    threshold_grids: dict[str, list[float]],
    output_dir: Path,
) -> dict:
    """Run the full kappa policy sweep.

    Returns a summary dict with frontier data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Baseline sweep (no governor)
    print("Phase 1: Baseline sweep (no governor)...")
    baseline_dir = output_dir / "baseline"
    baseline_results = _run_grid(
        base, D_values, W_values, seeds, baseline_dir, policy=None, label="baseline"
    )
    baseline_agg = _aggregate_results(baseline_results)
    R_proxy_baseline = baseline_agg["proxy_return_mean"]
    print(f"  Baseline proxy_return_mean = {R_proxy_baseline:.2f}")
    print(f"  Baseline collapse_rate = {baseline_agg['collapse_rate']:.3f}")

    # Phase 2: Threshold sweep per policy family
    print("\nPhase 2: Threshold sweep per policy family...")
    threshold_results: dict[str, list[dict]] = {}  # policy_name -> list of threshold records

    for policy_name in policy_names:
        thresholds = threshold_grids[policy_name]
        policies = _make_threshold_grid(policy_name, thresholds)
        policy_records = []

        print(f"\n  Policy: {policy_name} ({len(thresholds)} thresholds)")
        for i, (thr_val, pol) in enumerate(zip(thresholds, policies)):
            print(f"    Threshold {i+1}/{len(thresholds)}: {thr_val:.4f}")
            thr_dir = output_dir / "threshold_runs" / policy_name / f"thr_{thr_val:.4f}"
            thr_results = _run_grid(
                base, D_values, W_values, seeds, thr_dir,
                policy=pol, label=f"{policy_name}@{thr_val:.4f}"
            )
            thr_agg = _aggregate_results(thr_results)
            delta_R = (thr_agg["proxy_return_mean"] - R_proxy_baseline) / abs(R_proxy_baseline) if R_proxy_baseline != 0 else 0.0

            record = {
                "threshold": thr_val,
                "collapse_rate": thr_agg["collapse_rate"],
                "proxy_return_mean": thr_agg["proxy_return_mean"],
                "delta_R_proxy": delta_R,
                "alive_min": thr_agg["alive_min"],
            }
            policy_records.append(record)

        threshold_results[policy_name] = policy_records

        # Write per-policy NDJSON
        ndjson_dir = output_dir / "threshold_results"
        ndjson_dir.mkdir(parents=True, exist_ok=True)
        ndjson_path = ndjson_dir / f"{policy_name}.ndjson"
        with open(ndjson_path, "w") as f:
            for rec in policy_records:
                f.write(json.dumps(rec) + "\n")

    # Phase 3: Offline kappa selection
    print("\nPhase 3: Offline kappa selection...")
    frontier_rows = []

    for kappa in kappa_grid:
        for policy_name in policy_names:
            records = threshold_results[policy_name]
            if not records:
                continue

            # For each threshold, compute L = collapse_rate + kappa * |delta_R_proxy|
            # delta_R_proxy is typically negative (throttling reduces proxy return),
            # so we use the absolute value to penalize proxy loss
            best_L = float("inf")
            best_rec = None

            for rec in records:
                # Use abs(delta_R_proxy) so that proxy loss is always penalized
                L = rec["collapse_rate"] + kappa * abs(rec["delta_R_proxy"])
                if L < best_L:
                    best_L = L
                    best_rec = rec

            if best_rec is not None:
                frontier_rows.append({
                    "kappa": kappa,
                    "policy": policy_name,
                    "threshold": best_rec["threshold"],
                    "collapse_rate": best_rec["collapse_rate"],
                    "delta_R_proxy": best_rec["delta_R_proxy"],
                    "L": best_L,
                })

    # Write frontier.csv
    frontier_path = output_dir / "frontier.csv"
    if frontier_rows:
        fieldnames = ["kappa", "policy", "threshold", "collapse_rate", "delta_R_proxy", "L"]
        with open(frontier_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(frontier_rows)
        print(f"  Wrote frontier.csv with {len(frontier_rows)} rows")

    # Write summary JSON
    summary = {
        "R_proxy_baseline": R_proxy_baseline,
        "baseline_collapse_rate": baseline_agg["collapse_rate"],
        "kappa_grid": kappa_grid,
        "policy_names": policy_names,
        "threshold_grids": {k: [float(v) for v in vs] for k, vs in threshold_grids.items()},
        "D_values": D_values,
        "W_values": W_values,
        "seeds": seeds,
        "frontier": frontier_rows,
        "n_total_runs": (
            len(D_values) * len(W_values) * len(seeds) *
            (1 + sum(len(threshold_grids[p]) for p in policy_names))
        ),
    }
    summary_path = output_dir / "kappa_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nKappa sweep complete. Summary: {summary_path}")
    return summary
