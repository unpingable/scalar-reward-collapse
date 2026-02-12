"""Tests for run store: manifest integrity, crash safety, diff."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from scalar_collapse.core.config import ExperimentConfig
from scalar_collapse.runstore.diff import config_diff, metric_diff
from scalar_collapse.runstore.manifest import (
    RunManifest,
    compute_config_hash,
    compute_file_hash,
    derive_run_id,
)
from scalar_collapse.runstore.trajectory import TrajectoryWriter
from scalar_collapse.runstore.writer import atomic_write, store_run


class TestManifest:
    def test_run_id_deterministic(self):
        """Same inputs should produce the same run_id."""
        id1 = derive_run_id("abc", "def", 42)
        id2 = derive_run_id("abc", "def", 42)
        assert id1 == id2
        assert len(id1) == 12

    def test_run_id_differs_with_seed(self):
        id1 = derive_run_id("abc", "def", 42)
        id2 = derive_run_id("abc", "def", 43)
        assert id1 != id2

    def test_config_hash_deterministic(self):
        d = {"a": 1, "b": 2}
        h1 = compute_config_hash(d)
        h2 = compute_config_hash(d)
        assert h1 == h2

    def test_config_hash_order_independent(self):
        h1 = compute_config_hash({"a": 1, "b": 2})
        h2 = compute_config_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_manifest_roundtrip(self):
        m = RunManifest(run_id="abc123", started_at="2024-01-01T00:00:00Z")
        d = m.to_dict()
        m2 = RunManifest.from_dict(d)
        assert m2.run_id == "abc123"

    def test_file_hash(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex


class TestAtomicWrite:
    def test_write_string(self, tmp_path):
        p = tmp_path / "test.txt"
        atomic_write(p, "hello")
        assert p.read_text() == "hello"

    def test_write_bytes(self, tmp_path):
        p = tmp_path / "test.bin"
        atomic_write(p, b"hello")
        assert p.read_bytes() == b"hello"

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "a" / "b" / "test.txt"
        atomic_write(p, "hello")
        assert p.read_text() == "hello"


class TestStoreRun:
    def test_manifest_written_last(self, tmp_path):
        """Manifest should exist and contain valid JSON after store_run."""
        config = ExperimentConfig(seed=42, n_rounds=10)
        run_dir = tmp_path / "test_run"

        # Create a fake trajectory file
        run_dir.mkdir(parents=True)
        traj_path = run_dir / "metrics.ndjson"
        traj_path.write_text('{"t": 0}\n')

        manifest = store_run(
            run_dir=run_dir,
            config=config,
            trajectory_path=traj_path,
            summary={"tau_collapse": -1},
            started_at=datetime.now(timezone.utc),
        )

        # All files should exist
        assert (run_dir / "config.json").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "manifest.json").exists()

        # Manifest should be valid JSON
        with open(run_dir / "manifest.json") as f:
            m = json.load(f)
        assert m["run_id"] == manifest.run_id
        assert "config.json" in m["artifacts"]

    def test_hashes_match(self, tmp_path):
        """Artifact hashes in manifest should match actual file hashes."""
        config = ExperimentConfig(seed=99)
        run_dir = tmp_path / "hash_test"
        run_dir.mkdir(parents=True)
        traj_path = run_dir / "metrics.ndjson"
        traj_path.write_text('{"t": 0}\n')

        store_run(
            run_dir=run_dir,
            config=config,
            trajectory_path=traj_path,
            summary={},
            started_at=datetime.now(timezone.utc),
        )

        with open(run_dir / "manifest.json") as f:
            m = json.load(f)

        for fname, expected_hash in m["artifacts"].items():
            actual_hash = compute_file_hash(run_dir / fname)
            assert actual_hash == expected_hash, f"Hash mismatch for {fname}"

    def test_incomplete_run_detectable(self, tmp_path):
        """A run without manifest.json is incomplete."""
        run_dir = tmp_path / "incomplete"
        run_dir.mkdir()
        (run_dir / "config.json").write_text("{}")
        assert not (run_dir / "manifest.json").exists()


class TestTrajectoryWriter:
    def test_basic_write(self, tmp_path):
        with TrajectoryWriter(tmp_path) as tw:
            tw.append({"t": 0, "value": 1.0})
            tw.append({"t": 1, "value": 2.0})

        lines = (tmp_path / "metrics.ndjson").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["t"] == 0
        assert json.loads(lines[1])["t"] == 1

    def test_chunk_rollover(self, tmp_path):
        with TrajectoryWriter(tmp_path, chunk_size=5) as tw:
            for i in range(12):
                tw.append({"t": i})

        # Should have original file + at least one chunk
        assert (tmp_path / "metrics.ndjson").exists()
        assert (tmp_path / "metrics_0001.ndjson").exists()

    def test_numpy_serialization(self, tmp_path):
        with TrajectoryWriter(tmp_path) as tw:
            tw.append({"t": np.int64(0), "value": np.float64(1.5)})

        line = (tmp_path / "metrics.ndjson").read_text().strip()
        record = json.loads(line)
        assert record["t"] == 0
        assert record["value"] == 1.5


class TestDiff:
    def test_config_diff(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        (dir_a / "config.json").write_text(json.dumps({"seed": 42, "n_rounds": 100}))
        (dir_b / "config.json").write_text(json.dumps({"seed": 43, "n_rounds": 100}))

        diff = config_diff(dir_a, dir_b)
        assert "seed" in diff
        assert diff["seed"] == (42, 43)
        assert "n_rounds" not in diff

    def test_metric_diff(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        (dir_a / "summary.json").write_text(json.dumps({"tau": 100, "sigma": 5}))
        (dir_b / "summary.json").write_text(json.dumps({"tau": 200, "sigma": 5}))

        diff = metric_diff(dir_a, dir_b)
        assert "tau" in diff
        assert diff["tau"] == (100, 200)
        assert "sigma" not in diff
