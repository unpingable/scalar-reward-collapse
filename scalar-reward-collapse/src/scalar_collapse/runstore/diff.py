"""Config diff and metric rollup diff between runs."""

from __future__ import annotations

import json
from pathlib import Path


def config_diff(run_dir_a: Path, run_dir_b: Path) -> dict:
    """Compare config.json between two runs. Returns dict of changed keys."""
    config_a = _load_json(Path(run_dir_a) / "config.json")
    config_b = _load_json(Path(run_dir_b) / "config.json")
    return _deep_diff(config_a, config_b)


def metric_diff(run_dir_a: Path, run_dir_b: Path) -> dict:
    """Compare summary.json between two runs.

    Returns dict with keys that differ, each mapping to (value_a, value_b).
    """
    summary_a = _load_json(Path(run_dir_a) / "summary.json")
    summary_b = _load_json(Path(run_dir_b) / "summary.json")
    return _shallow_diff(summary_a, summary_b)


def format_diff(diff: dict, label: str = "diff") -> str:
    """Format a diff dict into a readable string."""
    if not diff:
        return f"No differences in {label}."
    lines = [f"{label}:"]
    for key, val in sorted(diff.items()):
        if isinstance(val, tuple) and len(val) == 2:
            lines.append(f"  {key}: {val[0]} -> {val[1]}")
        else:
            lines.append(f"  {key}: {val}")
    return "\n".join(lines)


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _deep_diff(a: dict, b: dict, prefix: str = "") -> dict:
    """Recursively diff two dicts. Returns {dotted_key: (val_a, val_b)}."""
    diffs = {}
    all_keys = set(a.keys()) | set(b.keys())
    for key in sorted(all_keys):
        full_key = f"{prefix}.{key}" if prefix else key
        va = a.get(key)
        vb = b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            diffs.update(_deep_diff(va, vb, full_key))
        elif va != vb:
            diffs[full_key] = (va, vb)
    return diffs


def _shallow_diff(a: dict, b: dict) -> dict:
    """Flat diff of two dicts. Returns {key: (val_a, val_b)} for changed keys."""
    diffs = {}
    all_keys = set(a.keys()) | set(b.keys())
    for key in sorted(all_keys):
        va = a.get(key)
        vb = b.get(key)
        if va != vb:
            diffs[key] = (va, vb)
    return diffs
