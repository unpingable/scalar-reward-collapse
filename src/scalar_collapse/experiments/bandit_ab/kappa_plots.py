"""Kappa sweep plots: efficient frontier, threshold vs kappa, boundary shift."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_kappa_summary(sweep_dir: Path) -> dict:
    """Load kappa_sweep_summary.json."""
    with open(Path(sweep_dir) / "kappa_sweep_summary.json") as f:
        return json.load(f)


def plot_efficient_frontier(sweep_dir: Path, output_dir: Path) -> Path:
    """Plot the efficient frontier: collapse_rate vs delta_R_proxy.

    Each curve = one policy family. Points indexed by kappa.
    """
    summary = load_kappa_summary(sweep_dir)
    frontier = summary["frontier"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    policy_names = summary["policy_names"]
    colors = {"alive": "tab:blue", "sigma": "tab:orange", "predictive": "tab:green"}
    markers = {"alive": "o", "sigma": "s", "predictive": "^"}

    for policy_name in policy_names:
        rows = [r for r in frontier if r["policy"] == policy_name]
        if not rows:
            continue
        x = [abs(r["delta_R_proxy"]) for r in rows]
        y = [r["collapse_rate"] for r in rows]
        kappas = [r["kappa"] for r in rows]

        color = colors.get(policy_name, "tab:gray")
        marker = markers.get(policy_name, "o")
        ax.plot(x, y, marker=marker, color=color, label=policy_name, linewidth=2, markersize=8)

        # Annotate kappa values
        for xi, yi, ki in zip(x, y, kappas):
            ax.annotate(f"k={ki}", (xi, yi), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, alpha=0.7)

    ax.set_xlabel("Proxy Loss |delta_R_proxy| (normalized)")
    ax.set_ylabel("Collapse Rate Pr[violate C_k]")
    ax.set_title("Efficient Frontier: Safety vs Proxy Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = output_dir / "frontier.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_threshold_vs_kappa(sweep_dir: Path, output_dir: Path) -> Path:
    """Plot chosen threshold vs kappa for each policy family."""
    summary = load_kappa_summary(sweep_dir)
    frontier = summary["frontier"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    policy_names = summary["policy_names"]
    colors = {"alive": "tab:blue", "sigma": "tab:orange", "predictive": "tab:green"}
    markers = {"alive": "o", "sigma": "s", "predictive": "^"}

    for policy_name in policy_names:
        rows = [r for r in frontier if r["policy"] == policy_name]
        if not rows:
            continue
        kappas = [r["kappa"] for r in rows]
        thresholds = [r["threshold"] for r in rows]

        color = colors.get(policy_name, "tab:gray")
        marker = markers.get(policy_name, "o")
        ax.plot(kappas, thresholds, marker=marker, color=color,
                label=policy_name, linewidth=2, markersize=8)

    ax.set_xlabel("kappa (log scale)")
    ax.set_ylabel("Chosen Threshold")
    ax.set_title("Threshold Selection vs kappa")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = output_dir / "threshold_vs_kappa.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _load_sweep_collapse_grid(sweep_dir: Path) -> tuple[np.ndarray, list[int], list[int]]:
    """Load collapse rates per D/W cell from a sweep directory's run summaries."""
    results = []
    # Collect all run summaries from subdirectories
    for run_dir in sorted(sweep_dir.iterdir()):
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            # Extract D/W from directory name pattern D{D}_W{W}_s{seed}
            parts = run_dir.name.split("_")
            d_val = None
            w_val = None
            for p in parts:
                if p.startswith("D") and p[1:].isdigit():
                    d_val = int(p[1:])
                elif p.startswith("W") and p[1:].isdigit():
                    w_val = int(p[1:])
            if d_val is not None and w_val is not None:
                results.append({"D": d_val, "W": w_val, "collapsed": s.get("collapsed", False)})

    if not results:
        return np.array([[]]), [], []

    D_vals = sorted(set(r["D"] for r in results))
    W_vals = sorted(set(r["W"] for r in results))

    # Compute collapse rate per cell
    cells: dict[tuple[int, int], list[bool]] = {}
    for r in results:
        key = (r["D"], r["W"])
        cells.setdefault(key, []).append(r["collapsed"])

    grid = np.zeros((len(D_vals), len(W_vals)))
    for i, D in enumerate(D_vals):
        for j, W in enumerate(W_vals):
            vals = cells.get((D, W), [])
            grid[i, j] = sum(vals) / len(vals) if vals else 0.0

    return grid, D_vals, W_vals


def plot_boundary_shift_heatmap(sweep_dir: Path, output_dir: Path) -> Path:
    """Side-by-side collapse heatmaps: baseline vs governed at kappa=1.

    For each policy family, picks the threshold chosen at kappa=1 and
    shows the collapse rate grid before/after governance.
    """
    summary = load_kappa_summary(sweep_dir)
    frontier = summary["frontier"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline grid
    baseline_dir = Path(sweep_dir) / "baseline"
    baseline_grid, D_vals, W_vals = _load_sweep_collapse_grid(baseline_dir)

    # Find governed grids at kappa=1
    kappa1_rows = [r for r in frontier if r["kappa"] == 1.0 or r["kappa"] == 1]
    policy_names = summary["policy_names"]

    # One row per policy family, 2 columns (baseline, governed)
    n_policies = len(policy_names)
    if n_policies == 0:
        n_policies = 1

    fig, axes = plt.subplots(n_policies, 2, figsize=(14, 5 * n_policies), squeeze=False)

    for row_idx, policy_name in enumerate(policy_names):
        # Find the threshold at kappa=1 for this policy
        k1_row = [r for r in kappa1_rows if r["policy"] == policy_name]
        if k1_row:
            thr_val = k1_row[0]["threshold"]
            thr_dir = Path(sweep_dir) / "threshold_runs" / policy_name / f"thr_{thr_val:.4f}"
            gov_grid, _, _ = _load_sweep_collapse_grid(thr_dir)
        else:
            gov_grid = np.zeros_like(baseline_grid)
            thr_val = 0.0

        for col_idx, (grid, title) in enumerate([
            (baseline_grid, "Baseline (no governor)"),
            (gov_grid, f"{policy_name} @ thr={thr_val:.3f}"),
        ]):
            ax = axes[row_idx, col_idx]
            if grid.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)
                continue

            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1)
            ax.set_xticks(range(len(W_vals)))
            ax.set_xticklabels(W_vals)
            ax.set_yticks(range(len(D_vals)))
            ax.set_yticklabels(D_vals)
            ax.set_xlabel("Update Cadence W")
            ax.set_ylabel("Retention Delay D")
            ax.set_title(title)
            for i in range(len(D_vals)):
                for j in range(len(W_vals)):
                    if i < grid.shape[0] and j < grid.shape[1]:
                        ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Collapse Rate", shrink=0.8)
    fig.suptitle("Boundary Shift at kappa=1", fontsize=14, y=1.02)

    out = output_dir / "boundary_shift_kappa1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_kappa_plots(sweep_dir: Path, output_dir: Path | None = None) -> list[Path]:
    """Generate all kappa sweep plots."""
    sweep_dir = Path(sweep_dir)
    if output_dir is None:
        output_dir = sweep_dir / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_efficient_frontier(sweep_dir, output_dir),
        plot_threshold_vs_kappa(sweep_dir, output_dir),
        plot_boundary_shift_heatmap(sweep_dir, output_dir),
    ]
    return paths
