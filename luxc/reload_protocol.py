"""Sentinel file format for hot-reload communication between compiler and engine.

The compiler writes sentinel JSON files on successful/failed recompilation.
The engine polls these files once per frame to detect shader changes.
"""

import json
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ReloadSentinel:
    """Successful recompilation sentinel."""
    version: int
    spv_files: list[str]
    json_files: list[str]
    timestamp: float
    source: str


@dataclass
class ErrorSentinel:
    """Failed recompilation sentinel."""
    error: str
    timestamp: float
    source: str


def write_sentinel(
    sentinel_path: Path,
    version: int,
    spv_files: list[Path],
    json_files: list[Path],
    source: str,
) -> None:
    """Atomically write a success sentinel JSON file.

    Uses write-to-temp-then-rename to prevent engines from reading partial data.
    """
    data = {
        "version": version,
        "spv_files": [str(p) for p in spv_files],
        "json_files": [str(p) for p in json_files],
        "timestamp": time.time(),
        "source": source,
    }
    _atomic_write_json(sentinel_path, data)


def read_sentinel(sentinel_path: Path) -> Optional[ReloadSentinel]:
    """Read a sentinel JSON file. Returns None if the file doesn't exist or is invalid."""
    try:
        text = sentinel_path.read_text(encoding="utf-8")
        data = json.loads(text)
        # Check if this is an error sentinel
        if "error" in data:
            return None
        return ReloadSentinel(
            version=data["version"],
            spv_files=data["spv_files"],
            json_files=data["json_files"],
            timestamp=data["timestamp"],
            source=data["source"],
        )
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def write_error(sentinel_path: Path, error_message: str, source: str) -> None:
    """Atomically write an error sentinel JSON file."""
    data = {
        "error": error_message,
        "timestamp": time.time(),
        "source": source,
    }
    _atomic_write_json(sentinel_path, data)


def read_error(sentinel_path: Path) -> Optional[ErrorSentinel]:
    """Read an error sentinel JSON file. Returns None if not an error or missing."""
    try:
        text = sentinel_path.read_text(encoding="utf-8")
        data = json.loads(text)
        if "error" not in data:
            return None
        return ErrorSentinel(
            error=data["error"],
            timestamp=data["timestamp"],
            source=data["source"],
        )
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def clear_error(sentinel_path: Path) -> None:
    """Remove the sentinel file if it exists."""
    try:
        sentinel_path.unlink()
    except OSError:
        pass


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON data atomically using temp-file + rename pattern.

    On Windows, os.replace() is atomic for NTFS. We write to a temp file
    in the same directory, then rename to the target path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create temp file in the same directory to ensure same filesystem (required for rename)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=".lux_sentinel_",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # Atomic rename (os.replace is atomic on both POSIX and Windows/NTFS)
        os.replace(tmp_path, str(path))
    except BaseException:
        # Clean up temp file on any error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
