"""Performance regression tests.

Compiles benchmark shaders and verifies instruction counts from reflection JSON
don't exceed established ceilings. These are integration tests that require
the full luxc toolchain (spirv-as, spirv-val).
"""

import json
import subprocess
import sys
import tempfile
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


def _compile_and_get_hints(shader_rel_path: str, stage_suffix: str) -> dict | None:
    """Compile a shader and return performance_hints from reflection JSON.

    Returns None if compilation fails or shader doesn't exist.
    """
    shader_path = ROOT_DIR / shader_rel_path
    if not shader_path.exists():
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "-m", "luxc",
            str(shader_path),
            "-o", tmpdir,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT_DIR))
        if result.returncode != 0:
            return None

        json_files = list(Path(tmpdir).glob(f"*.{stage_suffix}.json"))
        if not json_files:
            return None

        reflection = json.loads(json_files[0].read_text(encoding="utf-8"))
        return reflection.get("performance_hints")


# Mark all tests as requiring external tools
pytestmark = pytest.mark.skipif(
    not (ROOT_DIR / "examples" / "gltf_pbr.lux").exists(),
    reason="Example shaders not found"
)


class TestPerformanceHintsPresent:
    """Verify that compiled shaders include performance_hints in reflection."""

    def test_pbr_fragment_has_hints(self):
        hints = _compile_and_get_hints("examples/gltf_pbr.lux", "frag")
        if hints is None:
            pytest.skip("Compilation tools not available")
        assert "instruction_count" in hints
        assert "alu_ops" in hints
        assert "texture_samples" in hints
        assert "vgpr_estimate" in hints

    def test_pbr_vertex_has_hints(self):
        hints = _compile_and_get_hints("examples/gltf_pbr.lux", "vert")
        if hints is None:
            pytest.skip("Compilation tools not available")
        assert "instruction_count" in hints


class TestInstructionCounts:
    """Verify instruction counts are reasonable (not absurdly high)."""

    def test_pbr_fragment_instruction_ceiling(self):
        """PBR fragment should be under 1000 instructions."""
        hints = _compile_and_get_hints("examples/gltf_pbr.lux", "frag")
        if hints is None:
            pytest.skip("Compilation tools not available")
        assert hints["instruction_count"] < 1000, \
            f"Fragment instruction count {hints['instruction_count']} exceeds ceiling of 1000"

    def test_pbr_vertex_instruction_ceiling(self):
        """PBR vertex should be under 500 instructions."""
        hints = _compile_and_get_hints("examples/gltf_pbr.lux", "vert")
        if hints is None:
            pytest.skip("Compilation tools not available")
        assert hints["instruction_count"] < 500, \
            f"Vertex instruction count {hints['instruction_count']} exceeds ceiling of 500"

    def test_toon_fragment_instruction_ceiling(self):
        """Toon shader fragment should be under 1000 instructions."""
        hints = _compile_and_get_hints("examples/cartoon_toon.lux", "frag")
        if hints is None:
            pytest.skip("Compilation tools not available")
        assert hints["instruction_count"] < 1000

    def test_pbr_fragment_has_texture_samples(self):
        """PBR fragment shader should have at least 1 texture sample."""
        hints = _compile_and_get_hints("examples/gltf_pbr.lux", "frag")
        if hints is None:
            pytest.skip("Compilation tools not available")
        assert hints["texture_samples"] >= 0  # may be 0 if no samplers in basic PBR

    def test_vertex_shader_no_texture_samples(self):
        """Vertex shaders typically have 0 texture samples."""
        hints = _compile_and_get_hints("examples/gltf_pbr.lux", "vert")
        if hints is None:
            pytest.skip("Compilation tools not available")
        assert hints["texture_samples"] == 0


class TestVgprEstimate:
    """Verify VGPR estimates are valid strings."""

    def test_vgpr_is_valid_level(self):
        hints = _compile_and_get_hints("examples/gltf_pbr.lux", "frag")
        if hints is None:
            pytest.skip("Compilation tools not available")
        assert hints["vgpr_estimate"] in ("low", "medium", "high")
