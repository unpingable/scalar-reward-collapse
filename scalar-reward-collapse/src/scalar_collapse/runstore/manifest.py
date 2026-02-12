"""Run manifest: unique run IDs, schema, and integrity checking."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass
class RunManifest:
    """Schema for a run manifest file."""

    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    config_hash: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)  # filename -> content hash
    git_commit: str = ""
    git_dirty: bool = False
    environment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> RunManifest:
        return cls(**d)


def _git_info() -> tuple[str, bool]:
    """Get current git commit and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
            != ""
        )
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False


def _environment_info() -> dict[str, str]:
    """Capture environment details."""
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def compute_config_hash(config_dict: dict) -> str:
    """SHA-256 of the JSON-serialized config."""
    blob = json.dumps(config_dict, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def derive_run_id(config_hash: str, git_commit: str, seed: int) -> str:
    """Derive a deterministic run ID from config hash, git commit, and seed."""
    blob = f"{config_hash}|{git_commit}|{seed}".encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def compute_file_hash(path: Path) -> str:
    """SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    config_dict: dict,
    seed: int,
    run_dir: Path,
    started_at: datetime,
    artifact_files: list[str],
) -> RunManifest:
    """Build a complete manifest after a run finishes."""
    config_hash = compute_config_hash(config_dict)
    git_commit, git_dirty = _git_info()
    run_id = derive_run_id(config_hash, git_commit, seed)

    artifacts = {}
    for fname in artifact_files:
        fpath = run_dir / fname
        if fpath.exists():
            artifacts[fname] = compute_file_hash(fpath)

    return RunManifest(
        run_id=run_id,
        started_at=started_at.isoformat(),
        finished_at=datetime.now(timezone.utc).isoformat(),
        config_hash=config_hash,
        artifacts=artifacts,
        git_commit=git_commit,
        git_dirty=git_dirty,
        environment=_environment_info(),
    )
