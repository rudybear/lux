"""Live recompilation orchestrator for Lux shader hot-reload.

Wraps the compilation pipeline to support watch-mode recompilation:
- On success: writes sentinel file, outputs new .spv paths
- On failure: writes error sentinel, preserves old .spv files
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from luxc.compiler import compile_source
from luxc.reload_protocol import write_sentinel, write_error
from luxc.watcher import ImportGraph


@dataclass
class ReloadResult:
    """Result of a hot-reload recompilation attempt."""
    success: bool
    spv_paths: list[Path] = field(default_factory=list)
    json_paths: list[Path] = field(default_factory=list)
    error_message: str = ""
    elapsed_ms: float = 0.0


class HotReloader:
    """Orchestrates recompilation with hot-reload semantics.

    - Tracks compilation version (monotonically increasing)
    - Writes sentinel files for engine communication
    - Preserves last good .spv files on compilation failure
    - Re-scans import graph on each recompile
    """

    def __init__(
        self,
        source_path: Path,
        output_dir: Path,
        validate: bool = True,
        emit_reflection: bool = True,
        debug: bool = False,
        pipeline: Optional[str] = None,
        features: Optional[set[str]] = None,
        defines: Optional[dict[str, int]] = None,
        bindless: bool = False,
    ) -> None:
        self._source_path = source_path.resolve()
        self._output_dir = output_dir
        self._validate = validate
        self._emit_reflection = emit_reflection
        self._debug = debug
        self._pipeline = pipeline
        self._features = features
        self._defines = defines
        self._bindless = bindless
        self._version = 0
        self._last_good_spv: list[Path] = []
        self._last_good_json: list[Path] = []
        self._sentinel_path = output_dir / f"{source_path.stem}.reload.json"

    @property
    def version(self) -> int:
        """Current compilation version number."""
        return self._version

    @property
    def sentinel_path(self) -> Path:
        """Path to the sentinel JSON file."""
        return self._sentinel_path

    def get_import_paths(self) -> set[Path]:
        """Collect all file paths in the import graph (including root)."""
        graph = ImportGraph(self._source_path)
        return graph.collect()

    def recompile(self) -> ReloadResult:
        """Recompile the source file and write sentinel.

        On success: increments version, writes success sentinel, returns spv paths.
        On failure: writes error sentinel, preserves old .spv files.
        """
        start = time.monotonic()
        stem = self._source_path.stem
        source_dir = self._source_path.parent

        try:
            source = self._source_path.read_text(encoding="utf-8")
        except OSError as e:
            elapsed = (time.monotonic() - start) * 1000
            error_msg = f"Cannot read source file: {e}"
            write_error(self._sentinel_path, error_msg, str(self._source_path))
            return ReloadResult(
                success=False,
                error_message=error_msg,
                elapsed_ms=elapsed,
            )

        try:
            compile_source(
                source=source,
                stem=stem,
                output_dir=self._output_dir,
                source_dir=source_dir,
                validate=self._validate,
                emit_reflection=self._emit_reflection,
                debug=self._debug,
                source_name=self._source_path.name,
                pipeline=self._pipeline,
                features=self._features,
                defines=self._defines,
                bindless=self._bindless,
            )
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            error_msg = str(e)
            write_error(self._sentinel_path, error_msg, str(self._source_path))
            return ReloadResult(
                success=False,
                spv_paths=list(self._last_good_spv),
                json_paths=list(self._last_good_json),
                error_message=error_msg,
                elapsed_ms=elapsed,
            )

        elapsed = (time.monotonic() - start) * 1000
        self._version += 1

        # Collect output files
        spv_files = sorted(self._output_dir.glob(f"{stem}*.spv"))
        json_files = sorted(self._output_dir.glob(f"{stem}*.json"))
        # Exclude the sentinel file from json_files
        json_files = [j for j in json_files if j.name != self._sentinel_path.name]

        self._last_good_spv = list(spv_files)
        self._last_good_json = list(json_files)

        write_sentinel(
            self._sentinel_path,
            version=self._version,
            spv_files=spv_files,
            json_files=json_files,
            source=str(self._source_path),
        )

        return ReloadResult(
            success=True,
            spv_paths=spv_files,
            json_paths=json_files,
            elapsed_ms=elapsed,
        )
