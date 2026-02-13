"""Minimal CLI: run, sweep, summarize, diff, plot, predict, kappa-sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> None:
    """Run a single experiment."""
    from scalar_collapse.core.config import ExperimentConfig, WorldConfig
    from scalar_collapse.experiments.bandit_ab.simulate import run_experiment

    if args.config:
        with open(args.config) as f:
            raw = json.load(f)
        world = WorldConfig(**raw.get("world", {}))
        # Remove 'world' from raw to avoid double-passing
        exp_kwargs = {k: v for k, v in raw.items() if k != "world"}
        config = ExperimentConfig(world=world, **exp_kwargs)
    else:
        config = ExperimentConfig()

    run_dir = Path(args.output) / f"run_s{config.seed}"
    print(f"Running experiment: seed={config.seed}, D={config.retention_delay_D}, W={config.update_cadence_W}")
    summary = run_experiment(config, run_dir)
    print(f"Run complete: {run_dir}")
    print(f"  run_id: {summary['run_id']}")
    print(f"  tau_collapse: {summary['tau_collapse']}")
    print(f"  alive_final: {summary['alive_final']:.4f}")
    print(f"  sigma_total: {summary['sigma_total']}")
    print(f"  auc_divergence: {summary['auc_divergence']:.2f}")


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run a grid sweep."""
    from scalar_collapse.core.config import ExperimentConfig, SweepConfig, WorldConfig
    from scalar_collapse.experiments.bandit_ab.sweep import run_sweep

    if args.sweep_config:
        with open(args.sweep_config) as f:
            raw = json.load(f)
        base_raw = raw.get("base", {})
        world = WorldConfig(**base_raw.pop("world", {}))
        base = ExperimentConfig(world=world, **base_raw)
        sweep = SweepConfig(
            D_values=raw.get("D_values", [10, 50]),
            W_values=raw.get("W_values", [1, 5]),
            seeds=raw.get("seeds", [42]),
            base=base,
        )
    else:
        sweep = SweepConfig()

    results = run_sweep(sweep, Path(args.output))
    collapsed = sum(1 for r in results if r["tau_collapse"] >= 0)
    print(f"\n{collapsed}/{len(results)} runs showed collapse.")


def cmd_summarize(args: argparse.Namespace) -> None:
    """Print summary of a run."""
    summary_path = Path(args.run_dir) / "summary.json"
    if not summary_path.exists():
        print(f"No summary.json found in {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    print(json.dumps(summary, indent=2))


def cmd_diff(args: argparse.Namespace) -> None:
    """Compare two runs."""
    from scalar_collapse.runstore.diff import config_diff, format_diff, metric_diff

    cd = config_diff(Path(args.run_a), Path(args.run_b))
    md = metric_diff(Path(args.run_a), Path(args.run_b))

    print(format_diff(cd, "Config diff"))
    print()
    print(format_diff(md, "Metric diff"))


def cmd_spec(args: argparse.Namespace) -> None:
    """Print v0.2 canonical instantiation spec block for a run."""
    from scalar_collapse.core.template import TemplateInst

    summary_path = Path(args.run_dir) / "summary.json"
    if not summary_path.exists():
        print(f"No summary.json found in {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    template_data = summary.get("template", {})
    template = TemplateInst(**template_data)
    print(template.to_spec_block())


def cmd_predict(args: argparse.Namespace) -> None:
    """Predict regime from config without simulation."""
    from scalar_collapse.core.boundary import predict_regime
    from scalar_collapse.core.config import ExperimentConfig, WorldConfig

    if args.config:
        with open(args.config) as f:
            raw = json.load(f)
        world = WorldConfig(**raw.get("world", {}))
        exp_kwargs = {k: v for k, v in raw.items() if k != "world"}
        config = ExperimentConfig(world=world, **exp_kwargs)
    else:
        config = ExperimentConfig(
            retention_delay_D=int(args.D),
            update_cadence_W=int(args.W),
        )

    pred = predict_regime(config)
    tau_str = f"{pred.tau_steady:.0f}" if pred.tau_steady < 1e6 else "inf"
    print(f"predict_regime(D={pred.D}, W={pred.W})")
    print(f"  Regime:              {pred.regime_pred}")
    print(f"  First interval:      {'COLLAPSES' if pred.first_interval_collapses else 'safe'} (t_damage_first={pred.t_damage_first})")
    print(f"  Crossover (Q drift): n_cross={pred.n_cross} updates, t_cross={pred.t_cross} rounds")
    print(f"  Plant clock:         t_damage_post={pred.t_damage_post} rounds")
    print(f"  D_crit:              {pred.D_crit} (t_cross + t_damage_post)")
    print(f"  Steady-state:        B={pred.B_steady:.4f}, h={pred.h_steady:.6f}, tau={tau_str}")
    print(f"  Margin:              {pred.margin:+.2f}")


def _parse_float_list(s: str) -> list[float]:
    """Parse comma-separated float list from CLI string."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    """Parse comma-separated int list from CLI string."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def cmd_kappa_sweep(args: argparse.Namespace) -> None:
    """Run the kappa policy sweep."""
    import numpy as np

    from scalar_collapse.core.config import ExperimentConfig, WorldConfig
    from scalar_collapse.experiments.bandit_ab.kappa_plots import generate_kappa_plots
    from scalar_collapse.experiments.bandit_ab.kappa_sweep import run_kappa_sweep

    base = ExperimentConfig()
    kappa_grid = _parse_float_list(args.kappa_grid)
    D_values = _parse_int_list(args.D)
    W_values = _parse_int_list(args.W)
    seeds = _parse_int_list(args.seeds)

    # Determine which policies to run
    if args.policy == "all":
        policy_names = ["alive", "sigma", "predictive"]
    else:
        policy_names = [args.policy]

    # Build threshold grids
    threshold_grids = {}
    if "alive" in policy_names:
        threshold_grids["alive"] = list(np.linspace(
            *_parse_float_list(args.alive_thresholds)[:2],
            int(_parse_float_list(args.alive_thresholds)[2])
        )) if "," in args.alive_thresholds and len(args.alive_thresholds.split(",")) == 3 else _parse_float_list(args.alive_thresholds)
    if "sigma" in policy_names:
        threshold_grids["sigma"] = list(np.linspace(
            *_parse_float_list(args.sigma_thresholds)[:2],
            int(_parse_float_list(args.sigma_thresholds)[2])
        )) if "," in args.sigma_thresholds and len(args.sigma_thresholds.split(",")) == 3 else _parse_float_list(args.sigma_thresholds)
    if "predictive" in policy_names:
        threshold_grids["predictive"] = list(np.linspace(
            *_parse_float_list(args.margin_thresholds)[:2],
            int(_parse_float_list(args.margin_thresholds)[2])
        )) if "," in args.margin_thresholds and len(args.margin_thresholds.split(",")) == 3 else _parse_float_list(args.margin_thresholds)

    output_dir = Path(args.out)

    summary = run_kappa_sweep(
        base=base,
        D_values=D_values,
        W_values=W_values,
        seeds=seeds,
        kappa_grid=kappa_grid,
        policy_names=policy_names,
        threshold_grids=threshold_grids,
        output_dir=output_dir,
    )

    # Generate plots
    print("\nGenerating plots...")
    plots = generate_kappa_plots(output_dir)
    print(f"Generated {len(plots)} plots:")
    for p in plots:
        print(f"  {p}")

    # Print summary
    n_frontier = len(summary.get("frontier", []))
    print(f"\nKappa sweep complete:")
    print(f"  Total runs: {summary.get('n_total_runs', 0)}")
    print(f"  Frontier points: {n_frontier}")
    print(f"  Baseline collapse rate: {summary.get('baseline_collapse_rate', 0):.3f}")


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate plots for a run."""
    from scalar_collapse.experiments.bandit_ab.plots import generate_all_plots

    output_dir = Path(args.output) if args.output else Path(args.run_dir) / "plots"
    paths = generate_all_plots(Path(args.run_dir), output_dir)
    print(f"Generated {len(paths)} plots in {output_dir}:")
    for p in paths:
        print(f"  {p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="scalar_collapse",
        description="Scalar reward collapse experiments",
    )
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run a single experiment")
    p_run.add_argument("--config", type=str, default=None, help="Path to config JSON")
    p_run.add_argument("--output", type=str, default="runs", help="Output directory")

    # sweep
    p_sweep = sub.add_parser("sweep", help="Run a grid sweep")
    p_sweep.add_argument("--sweep-config", type=str, default=None, help="Path to sweep config JSON")
    p_sweep.add_argument("--output", type=str, default="runs/sweep", help="Output directory")

    # summarize
    p_sum = sub.add_parser("summarize", help="Print run summary")
    p_sum.add_argument("--run-dir", type=str, required=True, help="Path to run directory")

    # diff
    p_diff = sub.add_parser("diff", help="Compare two runs")
    p_diff.add_argument("run_a", type=str, help="First run directory")
    p_diff.add_argument("run_b", type=str, help="Second run directory")

    # spec
    p_spec = sub.add_parser("spec", help="Print v0.2 spec block")
    p_spec.add_argument("--run-dir", type=str, required=True, help="Path to run directory")

    # predict
    p_pred = sub.add_parser("predict", help="Predict regime from config (no simulation)")
    p_pred.add_argument("--config", type=str, default=None, help="Path to config JSON")
    p_pred.add_argument("--D", type=int, default=50, help="Retention delay D")
    p_pred.add_argument("--W", type=int, default=5, help="Update cadence W")

    # plot
    p_plot = sub.add_parser("plot", help="Generate plots")
    p_plot.add_argument("--run-dir", type=str, required=True, help="Path to run directory")
    p_plot.add_argument("--output", type=str, default=None, help="Output directory for plots")

    # kappa-sweep
    p_kappa = sub.add_parser("kappa-sweep", help="Run kappa policy sweep")
    p_kappa.add_argument("--kappa-grid", type=str, default="0,0.1,0.3,1,3,10,30,100",
                         help="Comma-separated kappa values")
    p_kappa.add_argument("--policy", type=str, default="all",
                         choices=["alive", "sigma", "predictive", "all"],
                         help="Policy family to sweep (default: all)")
    p_kappa.add_argument("--alive-thresholds", type=str, default="0.50,0.99,50",
                         help="start,stop,n for alive thresholds (linspace) or explicit list")
    p_kappa.add_argument("--sigma-thresholds", type=str, default="0.0,0.2,50",
                         help="start,stop,n for sigma thresholds (linspace) or explicit list")
    p_kappa.add_argument("--margin-thresholds", type=str, default="-0.5,1.0,50",
                         help="start,stop,n for margin thresholds (linspace) or explicit list")
    p_kappa.add_argument("--D", type=str, default="10,50,200,500",
                         help="Comma-separated D values")
    p_kappa.add_argument("--W", type=str, default="1,5,20",
                         help="Comma-separated W values")
    p_kappa.add_argument("--seeds", type=str, default="42,123,456",
                         help="Comma-separated seed values")
    p_kappa.add_argument("--out", type=str, default="runs/kappa_sweep",
                         help="Output directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "run": cmd_run,
        "sweep": cmd_sweep,
        "summarize": cmd_summarize,
        "spec": cmd_spec,
        "predict": cmd_predict,
        "diff": cmd_diff,
        "plot": cmd_plot,
        "kappa-sweep": cmd_kappa_sweep,
    }
    dispatch[args.command](args)
