"""Tests for the training data pipeline."""

import json
import pytest
from pathlib import Path

from luxc.transpiler.glsl_to_lux import transpile_glsl_to_lux


GLSL_FIXTURES = Path(__file__).parent / "fixtures" / "glsl"


class TestTranspileFixture:
    def test_transpile_simple_frag(self):
        """Transpile simple_frag.glsl to Lux and verify basic structure."""
        src = (GLSL_FIXTURES / "simple_frag.glsl").read_text()
        result = transpile_glsl_to_lux(src)
        assert len(result.lux_source) > 0
        assert "fragment" in result.lux_source
        assert "uv" in result.lux_source

    def test_transpile_phong(self):
        """Transpile phong.glsl to Lux."""
        src = (GLSL_FIXTURES / "phong.glsl").read_text()
        result = transpile_glsl_to_lux(src)
        assert "normalize" in result.lux_source
        assert "scalar" in result.lux_source


class TestJsonlOutput:
    def test_jsonl_output(self, tmp_path):
        """Verify output is valid JSONL with expected keys."""
        # Simulate what generate_training_data.py does
        src = (GLSL_FIXTURES / "simple_frag.glsl").read_text()
        result = transpile_glsl_to_lux(src)

        record = {
            "glsl_source": src,
            "lux": result.lux_source,
            "source_file": str(GLSL_FIXTURES / "simple_frag.glsl"),
            "warnings": result.warnings,
        }

        output_path = tmp_path / "test.jsonl"
        with open(output_path, "w") as f:
            f.write(json.dumps(record) + "\n")

        # Read back and verify
        with open(output_path) as f:
            line = f.readline()
            data = json.loads(line)

        assert "glsl_source" in data
        assert "lux" in data
        assert "source_file" in data
        assert "warnings" in data
        assert isinstance(data["warnings"], list)
        assert len(data["lux"]) > 0

    def test_multiple_records(self, tmp_path):
        """Verify multiple GLSL files produce valid JSONL."""
        output_path = tmp_path / "multi.jsonl"
        records = 0

        with open(output_path, "w") as f:
            for glsl_file in GLSL_FIXTURES.glob("*.glsl"):
                src = glsl_file.read_text()
                result = transpile_glsl_to_lux(src)
                record = {
                    "glsl_source": src,
                    "lux": result.lux_source,
                    "source_file": str(glsl_file),
                    "warnings": result.warnings,
                }
                f.write(json.dumps(record) + "\n")
                records += 1

        assert records >= 3  # We have 3 fixtures

        # Verify all lines are valid JSON
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                assert "glsl_source" in data
                assert "lux" in data
