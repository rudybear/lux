"""Tests for AI benchmark evaluation (Phase 16)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from luxc.ai.benchmark import (
    BenchmarkCase,
    BenchmarkScore,
    load_benchmark_cases,
    extract_surface_parameters,
    score_generation,
)


_BENCHMARK_FILE = Path(__file__).parent.parent / "data" / "benchmark_cases.json"

# ---------------------------------------------------------------------------
# Sample Lux sources used across multiple tests
# ---------------------------------------------------------------------------

_SAMPLE_METAL = """\
surface TestMetal {
    layers [
        base(
            albedo: vec3(1.00, 0.77, 0.34),
            roughness: 0.15,
            metallic: 1.0
        ),
        coat(factor: 0.8, roughness: 0.05),
    ]
}
"""

_SAMPLE_DIELECTRIC = """\
surface SimplePlastic {
    layers [
        base(
            albedo: vec3(0.80, 0.10, 0.10),
            roughness: 0.25,
            metallic: 0.0
        ),
    ]
}
"""

_SAMPLE_WITH_TRANSMISSION = """\
surface ClearGlass {
    layers [
        base(
            albedo: vec3(0.95, 0.95, 0.95),
            roughness: 0.02,
            metallic: 0.0
        ),
        transmission(factor: 1.0, ior: 1.52),
    ]
}
"""


class TestLoadBenchmarkCases:
    def test_load_benchmark_cases(self):
        """Load from data/benchmark_cases.json, verify at least 20 cases."""
        cases = load_benchmark_cases(_BENCHMARK_FILE)

        assert isinstance(cases, list)
        assert len(cases) >= 20

        for case in cases:
            assert isinstance(case, BenchmarkCase)
            assert len(case.id) > 0
            assert len(case.description) > 0

    def test_load_benchmark_cases_missing(self):
        """Non-existent file returns empty list."""
        cases = load_benchmark_cases(Path("/nonexistent/benchmark.json"))
        assert cases == []


class TestExtractSurfaceParameters:
    def test_extract_surface_parameters(self):
        """Parse known Lux source, extract roughness, metallic, layers."""
        params = extract_surface_parameters(_SAMPLE_METAL)

        assert params["roughness"] == pytest.approx(0.15)
        assert params["metallic"] == pytest.approx(1.0)
        assert "albedo_r" in params
        assert params["albedo_r"] == pytest.approx(1.00)
        assert params["albedo_g"] == pytest.approx(0.77)
        assert params["albedo_b"] == pytest.approx(0.34)

        # Layer detection
        assert "base" in params["_layers"]
        assert "coat" in params["_layers"]

    def test_extract_surface_parameters_with_layers(self):
        """Parse source with coat/transmission, verify _layers list."""
        params = extract_surface_parameters(_SAMPLE_WITH_TRANSMISSION)

        assert "base" in params["_layers"]
        assert "transmission" in params["_layers"]
        assert params["roughness"] == pytest.approx(0.02)
        assert params["metallic"] == pytest.approx(0.0)

        # coat_factor comes from explicit coat(...) call — not present here
        assert "coat" not in params["_layers"]

        # Coat factor from the metal sample
        metal_params = extract_surface_parameters(_SAMPLE_METAL)
        assert "coat_factor" in metal_params
        assert metal_params["coat_factor"] == pytest.approx(0.8)


class TestScoreGeneration:
    def _make_result(self, source, success=True, attempts=1):
        """Create a mock GenerateResult."""
        result = MagicMock()
        result.lux_source = source
        result.compilation_success = success
        result.attempts = attempts
        return result

    def test_score_generation_pass(self):
        """Correct params within expected ranges produce high accuracy."""
        case = BenchmarkCase(
            id="test-metal",
            description="Gold metal",
            expected_properties={
                "metallic": 1.0,
                "roughness": [0.0, 0.3],
                "albedo_r": [0.8, 1.0],
            },
            required_layers=["base", "coat"],
            category="basic-metal",
        )

        result = self._make_result(_SAMPLE_METAL)
        score = score_generation(case, result)

        assert isinstance(score, BenchmarkScore)
        assert score.compilation_success is True
        assert score.parameter_accuracy > 0.8
        assert score.layer_correctness is True
        assert score.energy_conserving is True

    def test_score_generation_compile_fail(self):
        """Failed compilation produces zero scores."""
        case = BenchmarkCase(
            id="test-fail",
            description="Should fail",
            expected_properties={"metallic": 1.0},
            required_layers=["base"],
        )

        result = self._make_result("invalid source", success=False, attempts=3)
        score = score_generation(case, result)

        assert score.compilation_success is False
        assert score.parameter_accuracy == 0.0
        assert score.energy_conserving is False
        assert score.layer_correctness is False
        assert score.attempts == 3

    def test_score_generation_wrong_params(self):
        """Wrong params produce low accuracy."""
        # Expect dielectric (metallic=0) but the source is metallic=1.0
        case = BenchmarkCase(
            id="test-wrong",
            description="Expecting a dielectric",
            expected_properties={
                "metallic": 0.0,
                "roughness": [0.5, 1.0],  # sample has 0.15 — out of range
            },
            required_layers=["base"],
        )

        result = self._make_result(_SAMPLE_METAL)
        score = score_generation(case, result)

        assert score.compilation_success is True
        # Both metallic and roughness are wrong -> 0/2 = 0.0
        assert score.parameter_accuracy < 0.5


class TestBenchmarkCaseCategories:
    def test_benchmark_case_categories(self):
        """Loaded cases span at least 4 different categories."""
        cases = load_benchmark_cases(_BENCHMARK_FILE)
        categories = {case.category for case in cases}

        assert len(categories) >= 4, (
            f"Expected at least 4 categories, found {len(categories)}: {categories}"
        )
