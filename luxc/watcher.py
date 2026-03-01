"""File watcher and dependency tracker for Lux shader hot-reload.

Monitors .lux source files and their transitive imports for changes,
triggering recompilation callbacks with debouncing.
"""

import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

# Standard library search path (same as compiler.py)
_STDLIB_DIR = Path(__file__).parent / "stdlib"

# Regex to extract import declarations from Lux source files
_IMPORT_RE = re.compile(r"^\s*import\s+(\w+)\s*;", re.MULTILINE)


class ImportGraph:
    """Collects all transitive import file paths from a root .lux file.

    Uses the same import resolution order as the compiler:
    1. luxc/stdlib/<name>.lux
    2. <source_dir>/<name>.lux
    """

    def __init__(self, root_path: Path) -> None:
        self._root = root_path.resolve()

    def collect(self) -> set[Path]:
        """Return the set of all resolved file paths (including root) that the
        root .lux file transitively depends on."""
        visited: set[Path] = set()
        self._walk(self._root, visited)
        return visited

    def _walk(self, file_path: Path, visited: set[Path]) -> None:
        resolved = file_path.resolve()
        if resolved in visited:
            return
        visited.add(resolved)

        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError:
            return

        source_dir = file_path.parent
        for match in _IMPORT_RE.finditer(source):
            name = match.group(1)
            imp_path = self._resolve_import(name, source_dir)
            if imp_path is not None:
                self._walk(imp_path, visited)

    @staticmethod
    def _resolve_import(name: str, source_dir: Path) -> Optional[Path]:
        """Resolve an import name to a file path.

        Search order matches luxc/compiler.py _resolve_imports():
        1. stdlib directory
        2. source file directory
        """
        stdlib_path = _STDLIB_DIR / f"{name}.lux"
        if stdlib_path.exists():
            return stdlib_path.resolve()

        local_path = source_dir / f"{name}.lux"
        if local_path.exists():
            return local_path.resolve()

        return None


class LuxFileWatcher:
    """Watches a set of .lux files for changes, calling a callback on modification.

    Uses polling (checking mtime) with configurable interval. Falls back gracefully
    on all platforms including Windows.

    Debounces rapid changes (e.g., editor save storms) with a configurable delay.
    """

    def __init__(
        self,
        files: set[Path],
        callback: Callable[[Path], None],
        poll_interval_ms: int = 500,
        debounce_ms: int = 100,
    ) -> None:
        self._files = {p.resolve() for p in files}
        self._callback = callback
        self._poll_interval = poll_interval_ms / 1000.0
        self._debounce = debounce_ms / 1000.0
        self._mtimes: dict[Path, float] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Initialize mtimes
        for f in self._files:
            try:
                self._mtimes[f] = f.stat().st_mtime
            except OSError:
                self._mtimes[f] = 0.0

    @property
    def file_count(self) -> int:
        """Number of files being watched."""
        return len(self._files)

    def update_files(self, files: set[Path]) -> None:
        """Update the set of watched files (e.g., after import graph changes)."""
        with self._lock:
            new_files = {p.resolve() for p in files}
            self._files = new_files
            # Add mtimes for new files
            for f in new_files:
                if f not in self._mtimes:
                    try:
                        self._mtimes[f] = f.stat().st_mtime
                    except OSError:
                        self._mtimes[f] = 0.0

    def start(self) -> None:
        """Start watching in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the watcher and wait for the thread to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def poll_once(self) -> Optional[Path]:
        """Check all files for changes. Returns the first changed file path, or None.

        Useful for single-step testing without starting a thread.
        """
        with self._lock:
            files = set(self._files)

        for f in files:
            try:
                current_mtime = f.stat().st_mtime
            except OSError:
                continue

            old_mtime = self._mtimes.get(f, 0.0)
            if current_mtime > old_mtime:
                self._mtimes[f] = current_mtime
                return f
        return None

    def _poll_loop(self) -> None:
        """Internal polling loop that runs in a background thread."""
        pending_change: Optional[Path] = None
        pending_time: float = 0.0

        while self._running:
            changed = self.poll_once()

            if changed is not None:
                # Start/reset debounce timer
                pending_change = changed
                pending_time = time.monotonic()

            if pending_change is not None:
                elapsed = time.monotonic() - pending_time
                if elapsed >= self._debounce:
                    # Debounce period expired, fire callback
                    self._callback(pending_change)
                    pending_change = None

            time.sleep(self._poll_interval)
