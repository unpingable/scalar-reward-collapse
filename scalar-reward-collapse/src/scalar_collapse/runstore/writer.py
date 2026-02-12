"""Atomic write helpers and run storage orchestration."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from scalar_collapse.core.config import ExperimentConfig
from scalar_collapse.runstore.manifest import RunManifest, build_manifest


def atomic_write(path: Path, data: str | bytes) -> None:
    """Write data to a temp file, fsync, then rename into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, mode) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def store_run(
    run_dir: Path,
    config: ExperimentConfig,
    trajectory_path: Path,
    summary: dict,
    started_at: datetime,
) -> RunManifest:
    """Orchestrate writing a complete run bundle.

    Write order (crash-safety):
    1. config.json
    2. metrics.ndjson (already written by TrajectoryWriter)
    3. summary.json
    4. manifest.json (LAST — presence means run completed)
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_dict = _config_to_dict(config)
    atomic_write(run_dir / "config.json", json.dumps(config_dict, indent=2))
    atomic_write(run_dir / "summary.json", json.dumps(summary, indent=2))

    # Build artifact list (trajectory may have been written already)
    artifact_files = ["config.json", "summary.json"]
    if trajectory_path.exists():
        # Copy or reference — trajectory is already in run_dir
        artifact_files.append(trajectory_path.name)

    manifest = build_manifest(
        config_dict=config_dict,
        seed=config.seed,
        run_dir=run_dir,
        started_at=started_at,
        artifact_files=artifact_files,
    )
    # Manifest written LAST
    atomic_write(run_dir / "manifest.json", manifest.to_json())
    return manifest


def _config_to_dict(config: ExperimentConfig) -> dict:
    """Convert ExperimentConfig to a JSON-serializable dict."""
    d = asdict(config)
    # Convert tuples to lists for JSON
    if "world" in d:
        w = d["world"]
        w["click_probs"] = list(w["click_probs"])
        w["burnout_deltas"] = list(w["burnout_deltas"])
    return d
