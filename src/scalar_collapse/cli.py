"""Minimal CLI: run, sweep, summarize, diff, plot."""

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
    }
    dispatch[args.command](args)
