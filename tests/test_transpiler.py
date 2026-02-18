"""Tests for the GLSL-to-Lux transpiler."""

import subprocess
import pytest
from pathlib import Path

from luxc.transpiler.glsl_to_lux import transpile_glsl_to_lux
from luxc.builtins.types import clear_type_aliases


GLSL_FIXTURES = Path(__file__).parent / "fixtures" / "glsl"


def _has_spirv_tools() -> bool:
    try:
        subprocess.run(["spirv-as", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


requires_spirv_tools = pytest.mark.skipif(
    not _has_spirv_tools(), reason="spirv-as/spirv-val not found on PATH"
)


class TestGlslParsing:
    def test_simple_fragment(self):
        """UV->color GLSL transpiles, output contains fragment."""
        src = (GLSL_FIXTURES / "simple_frag.glsl").read_text()
        result = transpile_glsl_to_lux(src)
        assert "fragment" in result.lux_source
        assert "uv" in result.lux_source

    def test_type_mapping(self):
        """float -> scalar in transpiled output."""
        src = """
        #version 450
        layout(location = 0) in float x;
        layout(location = 0) out vec4 outColor;
        void main() {
            float y = x * 2.0;
            outColor = vec4(y, y, y, 1.0);
        }
        """
        result = transpile_glsl_to_lux(src)
        assert "scalar" in result.lux_source
        # Should not contain "float" as a type (only in comments)
        lines = [l for l in result.lux_source.split("\n") if not l.strip().startswith("//")]
        code_text = "\n".join(lines)
        assert "float" not in code_text

    def test_gl_position_mapping(self):
        """gl_Position -> builtin_position."""
        src = """
        #version 450
        layout(location = 0) in vec3 pos;
        void main() {
            gl_Position = vec4(pos, 1.0);
        }
        """
        result = transpile_glsl_to_lux(src)
        assert "builtin_position" in result.lux_source
        assert "gl_Position" not in result.lux_source

    def test_loop_warning(self):
        """for-loop generates warning."""
        src = (GLSL_FIXTURES / "has_loop.glsl").read_text()
        result = transpile_glsl_to_lux(src)
        assert any("loop" in w.lower() for w in result.warnings)
        assert "UNSUPPORTED" in result.lux_source

    def test_math_functions(self):
        """sin/cos pass through unchanged."""
        src = """
        #version 450
        layout(location = 0) in float x;
        layout(location = 0) out vec4 outColor;
        void main() {
            float s = sin(x);
            float c = cos(x);
            outColor = vec4(s, c, 0.0, 1.0);
        }
        """
        result = transpile_glsl_to_lux(src)
        assert "sin(" in result.lux_source
        assert "cos(" in result.lux_source

    def test_texture_function_mapping(self):
        """texture() -> sample() in transpiled output."""
        src = """
        #version 450
        layout(location = 0) in vec2 uv;
        layout(location = 0) out vec4 outColor;
        uniform sampler2D tex;
        void main() {
            outColor = texture(tex, uv);
        }
        """
        result = transpile_glsl_to_lux(src)
        assert "sample(" in result.lux_source

    def test_compound_assignment(self):
        """x += e -> x = x + e."""
        src = """
        #version 450
        layout(location = 0) in float x;
        layout(location = 0) out vec4 outColor;
        void main() {
            float y = x;
            y += 1.0;
            outColor = vec4(y, 0.0, 0.0, 1.0);
        }
        """
        result = transpile_glsl_to_lux(src)
        # Should contain "y = y + 1.0" or similar
        assert "y = y +" in result.lux_source

    def test_phong_fixture(self):
        """Phong shader transpiles successfully."""
        src = (GLSL_FIXTURES / "phong.glsl").read_text()
        result = transpile_glsl_to_lux(src)
        assert "fragment" in result.lux_source
        assert "normalize" in result.lux_source
        assert "dot" in result.lux_source


@requires_spirv_tools
class TestTranspileRoundtrip:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_simple_roundtrip(self, tmp_path):
        """GLSL -> Lux -> SPIR-V compiles and validates."""
        from luxc.compiler import compile_source
        src = (GLSL_FIXTURES / "simple_frag.glsl").read_text()
        result = transpile_glsl_to_lux(src)
        compile_source(result.lux_source, "roundtrip", tmp_path, validate=True)
        assert (tmp_path / "roundtrip.frag.spv").exists()
