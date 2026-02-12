"""NDJSON trajectory writer with periodic flush and chunk rollover."""

from __future__ import annotations

import json
from pathlib import Path


class TrajectoryWriter:
    """Append NDJSON lines with periodic flush and optional chunk rollover."""

    def __init__(
        self, run_dir: Path, chunk_size: int = 50_000, filename: str = "metrics.ndjson"
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.filename = filename
        self._lines_in_chunk = 0
        self._chunk_index = 0
        self._file = None
        self._open_file()

    def _current_path(self) -> Path:
        if self._chunk_index == 0:
            return self.run_dir / self.filename
        stem = Path(self.filename).stem
        suffix = Path(self.filename).suffix
        return self.run_dir / f"{stem}_{self._chunk_index:04d}{suffix}"

    def _open_file(self) -> None:
        self._file = open(self._current_path(), "a")

    def append(self, record: dict) -> None:
        """Append a single record as a JSON line."""
        line = json.dumps(record, default=_json_default)
        self._file.write(line + "\n")
        self._lines_in_chunk += 1
        if self._lines_in_chunk >= self.chunk_size:
            self._rollover()

    def flush(self) -> None:
        """Force flush to disk."""
        if self._file and not self._file.closed:
            self._file.flush()

    def close(self) -> None:
        """Close the current file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    @property
    def path(self) -> Path:
        """Path to the current (or first) trajectory file."""
        return self.run_dir / self.filename

    def _rollover(self) -> None:
        """Close current chunk and open a new one."""
        self.close()
        self._chunk_index += 1
        self._lines_in_chunk = 0
        self._open_file()

    def __enter__(self) -> TrajectoryWriter:
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
