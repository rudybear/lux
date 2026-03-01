"""Tests for P24: Shader Hot-Reload.

Tests cover:
- ImportGraph transitive import collection
- Sentinel file write/read round-trip
- Atomic write pattern
- Error sentinel handling
- HotReloader recompile success/failure
- Failure preserves old .spv files
- Failure-then-fix recovery
- Watcher detects file changes
- Debounce coalesces rapid changes
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from luxc.watcher import ImportGraph, LuxFileWatcher
from luxc.reload_protocol import (
    write_sentinel,
    read_sentinel,
    write_error,
    read_error,
    clear_error,
)
from luxc.hot_reload import HotReloader, ReloadResult
from luxc.compiler import collect_import_paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_VERTEX_SRC = """\
vertex {
    in position: vec3;
    out frag_color: vec3;

    fn main() {
        frag_color = position;
        builtin_position = vec4(position, 1.0);
    }
}
"""

MINIMAL_FRAGMENT_SRC = """\
fragment {
    in frag_color: vec3;
    out color: vec4;

    fn main() {
        color = vec4(frag_color, 1.0);
    }
}
"""

INVALID_SRC = """\
vertex {
    this is not valid lux code !!!
}
"""


def _has_spirv_tools() -> bool:
    try:
        subprocess.run(["spirv-as", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


requires_spirv_tools = pytest.mark.skipif(
    not _has_spirv_tools(), reason="spirv-as/spirv-val not found on PATH"
)


# ---------------------------------------------------------------------------
# 1. ImportGraph collection
# ---------------------------------------------------------------------------

class TestImportGraphCollection:
    def test_no_imports(self, tmp_path):
        """A file with no imports should only include itself."""
        root = tmp_path / "main.lux"
        root.write_text(MINIMAL_VERTEX_SRC, encoding="utf-8")

        graph = ImportGraph(root)
        paths = graph.collect()

        assert root.resolve() in paths
        assert len(paths) == 1

    def test_direct_import(self, tmp_path):
        """A file importing a local module should include both files."""
        lib = tmp_path / "helpers.lux"
        lib.write_text("fn helper() -> float { return 1.0; }\n", encoding="utf-8")

        root = tmp_path / "main.lux"
        root.write_text(
            "import helpers;\n" + MINIMAL_VERTEX_SRC,
            encoding="utf-8",
        )

        graph = ImportGraph(root)
        paths = graph.collect()

        assert root.resolve() in paths
        assert lib.resolve() in paths
        assert len(paths) == 2

    def test_transitive_imports(self, tmp_path):
        """Transitive imports should all be collected."""
        a = tmp_path / "a.lux"
        a.write_text("fn a_fn() -> float { return 1.0; }\n", encoding="utf-8")

        b = tmp_path / "b.lux"
        b.write_text("import a;\nfn b_fn() -> float { return 2.0; }\n", encoding="utf-8")

        root = tmp_path / "main.lux"
        root.write_text(
            "import b;\n" + MINIMAL_VERTEX_SRC,
            encoding="utf-8",
        )

        graph = ImportGraph(root)
        paths = graph.collect()

        assert root.resolve() in paths
        assert a.resolve() in paths
        assert b.resolve() in paths
        assert len(paths) == 3

    def test_circular_imports_handled(self, tmp_path):
        """Circular imports should not cause infinite recursion."""
        a = tmp_path / "a.lux"
        b = tmp_path / "b.lux"
        a.write_text("import b;\nfn a_fn() -> float { return 1.0; }\n", encoding="utf-8")
        b.write_text("import a;\nfn b_fn() -> float { return 2.0; }\n", encoding="utf-8")

        root = tmp_path / "main.lux"
        root.write_text("import a;\n" + MINIMAL_VERTEX_SRC, encoding="utf-8")

        graph = ImportGraph(root)
        paths = graph.collect()

        assert root.resolve() in paths
        assert a.resolve() in paths
        assert b.resolve() in paths
        assert len(paths) == 3

    def test_missing_import_skipped(self, tmp_path):
        """Missing imports should be skipped without error."""
        root = tmp_path / "main.lux"
        root.write_text("import nonexistent;\n" + MINIMAL_VERTEX_SRC, encoding="utf-8")

        graph = ImportGraph(root)
        paths = graph.collect()

        assert root.resolve() in paths
        assert len(paths) == 1


# ---------------------------------------------------------------------------
# 2. collect_import_paths (compiler.py addition)
# ---------------------------------------------------------------------------

class TestCollectImportPaths:
    def test_no_imports(self, tmp_path):
        paths = collect_import_paths(MINIMAL_VERTEX_SRC, tmp_path)
        assert len(paths) == 0

    def test_local_import(self, tmp_path):
        lib = tmp_path / "helpers.lux"
        lib.write_text("fn helper() -> float { return 1.0; }\n", encoding="utf-8")

        source = "import helpers;\n" + MINIMAL_VERTEX_SRC
        paths = collect_import_paths(source, tmp_path)

        assert lib.resolve() in paths


# ---------------------------------------------------------------------------
# 3. Sentinel write/read round-trip
# ---------------------------------------------------------------------------

class TestSentinelWriteRead:
    def test_round_trip(self, tmp_path):
        """Write a sentinel and read it back."""
        sentinel_path = tmp_path / "test.reload.json"
        spv_files = [tmp_path / "shader.vert.spv", tmp_path / "shader.frag.spv"]
        json_files = [tmp_path / "shader.vert.json"]

        write_sentinel(sentinel_path, version=3, spv_files=spv_files,
                       json_files=json_files, source="test.lux")

        result = read_sentinel(sentinel_path)
        assert result is not None
        assert result.version == 3
        assert len(result.spv_files) == 2
        assert len(result.json_files) == 1
        assert result.source == "test.lux"
        assert result.timestamp > 0

    def test_read_missing_file(self, tmp_path):
        """Reading a non-existent sentinel returns None."""
        result = read_sentinel(tmp_path / "nonexistent.json")
        assert result is None

    def test_read_invalid_json(self, tmp_path):
        """Reading invalid JSON returns None."""
        path = tmp_path / "bad.json"
        path.write_text("not json at all", encoding="utf-8")
        result = read_sentinel(path)
        assert result is None

    def test_read_error_sentinel_as_success_returns_none(self, tmp_path):
        """Reading an error sentinel via read_sentinel returns None."""
        path = tmp_path / "err.json"
        write_error(path, "compile failed", "test.lux")
        result = read_sentinel(path)
        assert result is None


# ---------------------------------------------------------------------------
# 4. Sentinel atomic write
# ---------------------------------------------------------------------------

class TestSentinelAtomicWrite:
    def test_no_temp_files_left(self, tmp_path):
        """After writing, no .tmp files should remain in the directory."""
        sentinel_path = tmp_path / "test.reload.json"
        write_sentinel(sentinel_path, version=1, spv_files=[], json_files=[],
                       source="test.lux")

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0
        assert sentinel_path.exists()

    def test_valid_json_output(self, tmp_path):
        """Written file should be valid JSON."""
        sentinel_path = tmp_path / "test.reload.json"
        write_sentinel(sentinel_path, version=1, spv_files=[], json_files=[],
                       source="test.lux")

        data = json.loads(sentinel_path.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert "timestamp" in data

    def test_overwrite_existing(self, tmp_path):
        """Writing to an existing sentinel should overwrite it atomically."""
        sentinel_path = tmp_path / "test.reload.json"

        write_sentinel(sentinel_path, version=1, spv_files=[], json_files=[],
                       source="test.lux")
        write_sentinel(sentinel_path, version=2, spv_files=[], json_files=[],
                       source="test.lux")

        result = read_sentinel(sentinel_path)
        assert result is not None
        assert result.version == 2


# ---------------------------------------------------------------------------
# 5. Error sentinel
# ---------------------------------------------------------------------------

class TestErrorSentinel:
    def test_write_and_read_error(self, tmp_path):
        """Write an error sentinel and read it back."""
        path = tmp_path / "test.reload.json"
        write_error(path, "unexpected token at line 5", "test.lux")

        err = read_error(path)
        assert err is not None
        assert "unexpected token" in err.error
        assert err.source == "test.lux"
        assert err.timestamp > 0

    def test_read_success_as_error_returns_none(self, tmp_path):
        """Reading a success sentinel via read_error returns None."""
        path = tmp_path / "test.reload.json"
        write_sentinel(path, version=1, spv_files=[], json_files=[],
                       source="test.lux")

        err = read_error(path)
        assert err is None

    def test_clear_error(self, tmp_path):
        """clear_error should remove the sentinel file."""
        path = tmp_path / "test.reload.json"
        write_error(path, "error", "test.lux")
        assert path.exists()

        clear_error(path)
        assert not path.exists()

    def test_clear_nonexistent(self, tmp_path):
        """clear_error on a nonexistent file should not raise."""
        clear_error(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# 6. HotReloader recompile success
# ---------------------------------------------------------------------------

@requires_spirv_tools
class TestRecompileSuccess:
    def test_recompile_produces_spv(self, tmp_path):
        """Successful recompile should produce .spv files and a sentinel."""
        src_path = tmp_path / "shader.lux"
        src_path.write_text(MINIMAL_VERTEX_SRC, encoding="utf-8")
        out_dir = tmp_path / "out"

        reloader = HotReloader(src_path, out_dir)
        result = reloader.recompile()

        assert result.success
        assert result.error_message == ""
        assert result.elapsed_ms > 0
        assert len(result.spv_paths) > 0
        assert any(p.suffix == ".spv" for p in result.spv_paths)

        # Sentinel should exist
        sentinel = read_sentinel(reloader.sentinel_path)
        assert sentinel is not None
        assert sentinel.version == 1

    def test_version_increments(self, tmp_path):
        """Each successful recompile increments the version."""
        src_path = tmp_path / "shader.lux"
        src_path.write_text(MINIMAL_VERTEX_SRC, encoding="utf-8")
        out_dir = tmp_path / "out"

        reloader = HotReloader(src_path, out_dir)
        reloader.recompile()
        reloader.recompile()
        result = reloader.recompile()

        assert result.success
        assert reloader.version == 3

        sentinel = read_sentinel(reloader.sentinel_path)
        assert sentinel.version == 3


# ---------------------------------------------------------------------------
# 7. Recompile failure preserves old .spv
# ---------------------------------------------------------------------------

@requires_spirv_tools
class TestRecompileFailurePreservesOld:
    def test_failure_keeps_old_spv(self, tmp_path):
        """Compilation failure should not overwrite existing .spv files."""
        src_path = tmp_path / "shader.lux"
        src_path.write_text(MINIMAL_VERTEX_SRC, encoding="utf-8")
        out_dir = tmp_path / "out"

        reloader = HotReloader(src_path, out_dir)

        # First compile: success
        result1 = reloader.recompile()
        assert result1.success
        old_spv_paths = result1.spv_paths
        old_spv_contents = {p: p.read_bytes() for p in old_spv_paths}

        # Break the source
        src_path.write_text(INVALID_SRC, encoding="utf-8")

        # Second compile: should fail
        result2 = reloader.recompile()
        assert not result2.success
        assert result2.error_message != ""

        # Old .spv files should still exist and be unchanged
        for spv_path, old_content in old_spv_contents.items():
            assert spv_path.exists()
            assert spv_path.read_bytes() == old_content

        # Error sentinel should exist
        err = read_error(reloader.sentinel_path)
        assert err is not None

        # Version should NOT have incremented
        assert reloader.version == 1

    def test_failure_result_has_old_paths(self, tmp_path):
        """Failed recompile result should still reference old spv paths."""
        src_path = tmp_path / "shader.lux"
        src_path.write_text(MINIMAL_VERTEX_SRC, encoding="utf-8")
        out_dir = tmp_path / "out"

        reloader = HotReloader(src_path, out_dir)
        result1 = reloader.recompile()
        assert result1.success

        src_path.write_text(INVALID_SRC, encoding="utf-8")
        result2 = reloader.recompile()
        assert not result2.success
        # The failed result should carry forward the last good spv paths
        assert len(result2.spv_paths) == len(result1.spv_paths)


# ---------------------------------------------------------------------------
# 8. Recompile failure then fix
# ---------------------------------------------------------------------------

@requires_spirv_tools
class TestRecompileFailureThenFix:
    def test_fail_fix_recompile(self, tmp_path):
        """After a failure, fixing the source and recompiling should succeed."""
        src_path = tmp_path / "shader.lux"
        src_path.write_text(MINIMAL_VERTEX_SRC, encoding="utf-8")
        out_dir = tmp_path / "out"

        reloader = HotReloader(src_path, out_dir)

        # First compile: success
        result1 = reloader.recompile()
        assert result1.success

        # Break it
        src_path.write_text(INVALID_SRC, encoding="utf-8")
        result2 = reloader.recompile()
        assert not result2.success

        # Fix it
        src_path.write_text(MINIMAL_VERTEX_SRC, encoding="utf-8")
        result3 = reloader.recompile()
        assert result3.success
        assert reloader.version == 2  # version 1 from first, version 2 from fix

        sentinel = read_sentinel(reloader.sentinel_path)
        assert sentinel is not None
        assert sentinel.version == 2


# ---------------------------------------------------------------------------
# 9. Watcher detects change
# ---------------------------------------------------------------------------

class TestWatchDetectsChange:
    def test_poll_once_detects_change(self, tmp_path):
        """poll_once() should detect when a file's mtime changes."""
        f = tmp_path / "shader.lux"
        f.write_text("original", encoding="utf-8")

        callback_log = []
        watcher = LuxFileWatcher(
            files={f},
            callback=lambda p: callback_log.append(p),
            poll_interval_ms=50,
        )

        # No change yet
        assert watcher.poll_once() is None

        # Ensure new mtime by sleeping briefly (filesystem granularity)
        time.sleep(0.05)
        f.write_text("modified", encoding="utf-8")

        changed = watcher.poll_once()
        assert changed is not None
        assert changed == f.resolve()

    def test_threaded_watcher_fires_callback(self, tmp_path):
        """Start watcher in thread, modify file, callback fires."""
        f = tmp_path / "shader.lux"
        f.write_text("original", encoding="utf-8")

        event = threading.Event()
        changed_files = []

        def on_change(path):
            changed_files.append(path)
            event.set()

        watcher = LuxFileWatcher(
            files={f},
            callback=on_change,
            poll_interval_ms=50,
            debounce_ms=50,
        )
        watcher.start()

        try:
            # Ensure mtime difference
            time.sleep(0.1)
            f.write_text("modified", encoding="utf-8")

            # Wait for callback (with timeout)
            assert event.wait(timeout=5.0), "Callback was not fired within timeout"
            assert len(changed_files) >= 1
        finally:
            watcher.stop()

    def test_file_count(self, tmp_path):
        """file_count should reflect the number of watched files."""
        f1 = tmp_path / "a.lux"
        f2 = tmp_path / "b.lux"
        f1.write_text("a", encoding="utf-8")
        f2.write_text("b", encoding="utf-8")

        watcher = LuxFileWatcher(
            files={f1, f2},
            callback=lambda p: None,
        )
        assert watcher.file_count == 2

    def test_update_files(self, tmp_path):
        """update_files() should change the set of watched files."""
        f1 = tmp_path / "a.lux"
        f2 = tmp_path / "b.lux"
        f1.write_text("a", encoding="utf-8")
        f2.write_text("b", encoding="utf-8")

        watcher = LuxFileWatcher(files={f1}, callback=lambda p: None)
        assert watcher.file_count == 1

        watcher.update_files({f1, f2})
        assert watcher.file_count == 2


# ---------------------------------------------------------------------------
# 10. Debounce
# ---------------------------------------------------------------------------

class TestDebounce:
    def test_rapid_changes_single_callback(self, tmp_path):
        """Rapid modifications should trigger a single callback (debounced)."""
        f = tmp_path / "shader.lux"
        f.write_text("v0", encoding="utf-8")

        event = threading.Event()
        callback_count = []

        def on_change(path):
            callback_count.append(1)
            event.set()

        watcher = LuxFileWatcher(
            files={f},
            callback=on_change,
            poll_interval_ms=30,
            debounce_ms=200,
        )
        watcher.start()

        try:
            # Rapid-fire modifications within debounce window
            time.sleep(0.1)
            for i in range(5):
                f.write_text(f"v{i+1}", encoding="utf-8")
                time.sleep(0.02)  # 20ms between writes, well within 200ms debounce

            # Wait for the single debounced callback
            assert event.wait(timeout=5.0), "Callback was not fired within timeout"

            # Give a bit more time then check count
            time.sleep(0.5)

            # Should have fired only once (or at most twice due to timing)
            # The key is that 5 rapid writes don't produce 5 callbacks
            assert len(callback_count) <= 2, (
                f"Expected at most 2 callbacks for 5 rapid writes, got {len(callback_count)}"
            )
        finally:
            watcher.stop()
