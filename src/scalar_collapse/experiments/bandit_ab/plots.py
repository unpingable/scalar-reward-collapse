"""Paper-ready figure generators for bandit/AB experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_trajectory(run_dir: Path) -> list[dict]:
    """Load NDJSON trajectory from a run directory."""
    records = []
    traj_path = Path(run_dir) / "metrics.ndjson"
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_summary(run_dir: Path) -> dict:
    """Load summary.json from a run directory."""
    with open(Path(run_dir) / "summary.json") as f:
        return json.load(f)


def plot_proxy_vs_true(run_dir: Path, output_dir: Path) -> Path:
    """Plot cumulative proxy vs true objective over time."""
    records = load_trajectory(run_dir)
    t = [r["t"] for r in records]
    proxy = [r["proxy_cumulative"] for r in records]
    true = [r["true_cumulative"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, proxy, label="Proxy (cumulative CTR)", color="tab:blue")
    ax.plot(t, true, label="True (survival-weighted)", color="tab:red")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Proxy vs True Objective — Divergence Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(output_dir) / "proxy_vs_true.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_entropy(run_dir: Path, output_dir: Path) -> Path:
    """Plot policy entropy over time."""
    records = load_trajectory(run_dir)
    t = [r["t"] for r in records]
    entropy = [r["entropy"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, entropy, color="tab:green")
    ax.set_xlabel("Round")
    ax.set_ylabel("Policy Entropy (nats)")
    ax.set_title("Policy Entropy — Distribution Collapse")
    ax.grid(True, alpha=0.3)

    out = Path(output_dir) / "entropy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sigma(run_dir: Path, output_dir: Path) -> Path:
    """Plot cumulative sigma (unverified boundary crossings) over time."""
    records = load_trajectory(run_dir)
    t = [r["t"] for r in records]
    sigma = [r["sigma_cumulative"] for r in records]

    summary = load_summary(run_dir)
    threshold = summary.get("sigma_threshold", 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, sigma, color="tab:purple", label="sigma(t)")
    if threshold > 0:
        ax.axhline(y=threshold, color="tab:orange", linestyle="--", label=f"threshold = {threshold:.0f}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative sigma")
    ax.set_title("sigma(t) — Unverified Boundary Crossings")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(output_dir) / "sigma_trace.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_alive_fraction(run_dir: Path, output_dir: Path) -> Path:
    """Plot alive fraction over time — the churn cliff."""
    records = load_trajectory(run_dir)
    t = [r["t"] for r in records]
    alive = [r["alive_fraction"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, alive, color="tab:red")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="floor = 0.5")
    ax.set_xlabel("Round")
    ax.set_ylabel("Alive Fraction")
    ax.set_title("Population Survival — Churn Cliff")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(output_dir) / "alive_fraction.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_regime_heatmap(sweep_dir: Path, output_dir: Path) -> list[Path]:
    """Plot D/W regime heatmaps from sweep results.

    Generates two heatmaps:
    1. tau_collapse (higher = safer) — shows when collapse happens
    2. alive_final (mean across seeds) — shows severity of population loss
    """
    summary_path = Path(sweep_dir) / "sweep_summary.json"
    with open(summary_path) as f:
        raw = json.load(f)

    # Support both old (flat list) and new (nested) sweep_summary formats
    results = raw["per_run"] if isinstance(raw, dict) and "per_run" in raw else raw

    D_vals = sorted(set(r["sweep_D"] for r in results))
    W_vals = sorted(set(r["sweep_W"] for r in results))

    paths = []

    # --- Heatmap 1: tau_collapse ---
    tau_cells: dict[tuple[int, int], list[int]] = {}
    for r in results:
        key = (r["sweep_D"], r["sweep_W"])
        tau = r["tau_collapse"]
        tau_cells.setdefault(key, []).append(tau if tau >= 0 else r["T"])

    tau_grid = np.zeros((len(D_vals), len(W_vals)))
    for i, D in enumerate(D_vals):
        for j, W in enumerate(W_vals):
            tau_grid[i, j] = np.mean(tau_cells.get((D, W), [0]))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(tau_grid, aspect="auto", origin="lower", cmap="RdYlGn")
    ax.set_xticks(range(len(W_vals)))
    ax.set_xticklabels(W_vals)
    ax.set_yticks(range(len(D_vals)))
    ax.set_yticklabels(D_vals)
    ax.set_xlabel("Update Cadence W")
    ax.set_ylabel("Retention Delay D")
    ax.set_title("Regime Heatmap — tau_collapse (higher = safer)")
    for i in range(len(D_vals)):
        for j in range(len(W_vals)):
            ax.text(j, i, f"{tau_grid[i, j]:.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Mean tau_collapse")

    out = Path(output_dir) / "regime_heatmap_tau.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(out)

    # --- Heatmap 2: alive_final ---
    alive_cells: dict[tuple[int, int], list[float]] = {}
    for r in results:
        key = (r["sweep_D"], r["sweep_W"])
        alive_cells.setdefault(key, []).append(r["alive_final"])

    alive_grid = np.zeros((len(D_vals), len(W_vals)))
    for i, D in enumerate(D_vals):
        for j, W in enumerate(W_vals):
            alive_grid[i, j] = np.mean(alive_cells.get((D, W), [1.0]))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(alive_grid, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(W_vals)))
    ax.set_xticklabels(W_vals)
    ax.set_yticks(range(len(D_vals)))
    ax.set_yticklabels(D_vals)
    ax.set_xlabel("Update Cadence W")
    ax.set_ylabel("Retention Delay D")
    ax.set_title("Regime Heatmap — alive_final_mean (higher = safer)")
    for i in range(len(D_vals)):
        for j in range(len(W_vals)):
            ax.text(j, i, f"{alive_grid[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="alive_final_mean")

    out = Path(output_dir) / "regime_heatmap_alive.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(out)

    return paths


def plot_sigma_threshold_sensitivity(run_dir: Path, output_dir: Path) -> Path:
    """Plot how sigma-threshold choice affects detection timing."""
    records = load_trajectory(run_dir)
    sigma = [r["sigma_cumulative"] for r in records]
    t_arr = [r["t"] for r in records]

    if not sigma:
        return Path(output_dir) / "sigma_threshold_sensitivity.png"

    max_sigma = max(sigma)
    if max_sigma == 0:
        max_sigma = 1

    thresholds = np.linspace(0, max_sigma, 20)
    detection_times = []
    for thr in thresholds:
        detected = False
        for i, s in enumerate(sigma):
            if s >= thr:
                detection_times.append(t_arr[i])
                detected = True
                break
        if not detected:
            detection_times.append(t_arr[-1] if t_arr else 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, detection_times, color="tab:brown", marker="o", markersize=3)
    ax.set_xlabel("sigma Threshold")
    ax.set_ylabel("Detection Time (round)")
    ax.set_title("sigma-Threshold Sensitivity")
    ax.grid(True, alpha=0.3)

    out = Path(output_dir) / "sigma_threshold_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_controller_comparison(
    vanilla_sweep_dir: Path, controlled_sweep_dir: Path, output_dir: Path
) -> Path:
    """Side-by-side heatmaps showing how the controller shifts the regime boundary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _load_alive_grid(sweep_dir: Path):
        with open(Path(sweep_dir) / "sweep_summary.json") as f:
            raw = json.load(f)
        results = raw["per_run"] if isinstance(raw, dict) and "per_run" in raw else raw
        D_vals = sorted(set(r["sweep_D"] for r in results))
        W_vals = sorted(set(r["sweep_W"] for r in results))
        cells: dict[tuple[int, int], list[float]] = {}
        for r in results:
            key = (r["sweep_D"], r["sweep_W"])
            cells.setdefault(key, []).append(r["alive_final"])
        grid = np.zeros((len(D_vals), len(W_vals)))
        for i, D in enumerate(D_vals):
            for j, W in enumerate(W_vals):
                grid[i, j] = np.mean(cells.get((D, W), [1.0]))
        return grid, D_vals, W_vals

    grid_v, D_vals, W_vals = _load_alive_grid(vanilla_sweep_dir)
    grid_c, _, _ = _load_alive_grid(controlled_sweep_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, grid, title in [
        (ax1, grid_v, "Vanilla Agent"),
        (ax2, grid_c, "Controlled Agent (throttle)"),
    ]:
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(W_vals)))
        ax.set_xticklabels(W_vals)
        ax.set_yticks(range(len(D_vals)))
        ax.set_yticklabels(D_vals)
        ax.set_xlabel("Update Cadence W")
        ax.set_ylabel("Retention Delay D")
        ax.set_title(title)
        for i in range(len(D_vals)):
            for j in range(len(W_vals)):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=[ax1, ax2], label="alive_final_mean", shrink=0.8)
    fig.suptitle("Controller Shifts the Phase Boundary", fontsize=14, y=1.02)

    out = output_dir / "controller_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_all_plots(run_dir: Path, output_dir: Path) -> list[Path]:
    """Generate all standard plots for a single run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_proxy_vs_true(run_dir, output_dir),
        plot_entropy(run_dir, output_dir),
        plot_sigma(run_dir, output_dir),
        plot_alive_fraction(run_dir, output_dir),
        plot_sigma_threshold_sensitivity(run_dir, output_dir),
    ]
