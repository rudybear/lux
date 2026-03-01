"""Evaluation benchmark for AI shader generation quality.

Provides test cases with expected properties and scoring logic
to measure generation accuracy across materials, layers, and parameters.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


_BENCHMARK_DIR = Path(__file__).parent.parent.parent / "data"


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    id: str
    description: str
    expected_properties: dict  # {param: [min, max], ...}
    required_layers: list[str] = field(default_factory=list)
    category: str = "general"


@dataclass
class BenchmarkScore:
    """Score for a single benchmark case."""
    case_id: str
    compilation_success: bool = False
    parameter_accuracy: float = 0.0    # 0-1
    energy_conserving: bool = False
    layer_correctness: bool = False
    attempts: int = 1


def load_benchmark_cases(path: Path | None = None) -> list[BenchmarkCase]:
    """Load benchmark cases from a JSON file.

    Parameters
    ----------
    path : Path or None
        Path to the JSON file. Defaults to data/benchmark_cases.json.

    Returns
    -------
    list[BenchmarkCase]
    """
    if path is None:
        path = _BENCHMARK_DIR / "benchmark_cases.json"

    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for entry in data:
        cases.append(BenchmarkCase(
            id=entry["id"],
            description=entry["description"],
            expected_properties=entry.get("expected_properties", {}),
            required_layers=entry.get("required_layers", []),
            category=entry.get("category", "general"),
        ))
    return cases


def extract_surface_parameters(lux_source: str) -> dict:
    """Parse a surface declaration and extract literal parameter values.

    Extracts albedo, roughness, metallic, and layer information from
    the Lux source code using regex pattern matching.

    Parameters
    ----------
    lux_source : str
        Lux shader source code.

    Returns
    -------
    dict
        Extracted parameters (e.g., {"roughness": 0.5, "metallic": 1.0, ...}).
    """
    params: dict = {}

    # Extract roughness from base layer or properties
    rough_match = re.search(r"roughness\s*[:=]\s*(\d+\.?\d*)", lux_source)
    if rough_match:
        params["roughness"] = float(rough_match.group(1))

    # Extract metallic from base layer
    metal_match = re.search(r"metallic\s*[:=]\s*(\d+\.?\d*)", lux_source)
    if metal_match:
        params["metallic"] = float(metal_match.group(1))

    # Extract albedo components (look for vec3 in albedo context)
    albedo_match = re.search(
        r"(?:albedo|base_color|reflectance)\s*[:=]\s*vec3\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\)",
        lux_source,
    )
    if albedo_match:
        params["albedo_r"] = float(albedo_match.group(1))
        params["albedo_g"] = float(albedo_match.group(2))
        params["albedo_b"] = float(albedo_match.group(3))

    # Detect layers
    layers_found = []
    for layer_name in ["base", "coat", "sheen", "transmission", "emission", "normal_map"]:
        if re.search(rf"\b{layer_name}\s*\(", lux_source):
            layers_found.append(layer_name)
    params["_layers"] = layers_found

    # Extract IOR if present
    ior_match = re.search(r"ior\s*[:=]\s*(\d+\.?\d*)", lux_source)
    if ior_match:
        params["ior"] = float(ior_match.group(1))

    # Extract coat factor
    coat_match = re.search(r"coat\s*\(\s*factor\s*:\s*(\d+\.?\d*)", lux_source)
    if coat_match:
        params["coat_factor"] = float(coat_match.group(1))

    return params


def score_generation(
    case: BenchmarkCase,
    result,  # GenerateResult
) -> BenchmarkScore:
    """Score a generation result against a benchmark case.

    Parameters
    ----------
    case : BenchmarkCase
        The benchmark case with expected properties.
    result : GenerateResult
        The AI generation result.

    Returns
    -------
    BenchmarkScore
    """
    if not result.compilation_success:
        return BenchmarkScore(
            case_id=case.id,
            compilation_success=False,
            attempts=result.attempts,
        )

    params = extract_surface_parameters(result.lux_source)

    # Parameter accuracy: check each expected property
    matches = 0
    total = 0
    for prop, expected in case.expected_properties.items():
        if prop not in params:
            total += 1
            continue

        actual = params[prop]
        if isinstance(expected, list) and len(expected) == 2:
            # Range check [min, max]
            if expected[0] <= actual <= expected[1]:
                matches += 1
        elif isinstance(expected, (int, float)):
            # Exact value with tolerance
            if abs(actual - expected) < 0.1:
                matches += 1
        total += 1

    accuracy = matches / total if total > 0 else 0.0

    # Layer correctness
    found_layers = params.get("_layers", [])
    required_present = all(
        layer in found_layers for layer in case.required_layers
    )

    # Energy conservation check
    albedo_r = params.get("albedo_r", 0.5)
    albedo_g = params.get("albedo_g", 0.5)
    albedo_b = params.get("albedo_b", 0.5)
    metallic = params.get("metallic", 0.0)
    max_albedo = max(albedo_r, albedo_g, albedo_b)
    energy_ok = max_albedo <= 1.0
    if metallic < 0.5:
        # Dielectrics should have albedo < 0.95
        energy_ok = energy_ok and max_albedo <= 0.95

    return BenchmarkScore(
        case_id=case.id,
        compilation_success=True,
        parameter_accuracy=accuracy,
        energy_conserving=energy_ok,
        layer_correctness=required_present,
        attempts=result.attempts,
    )


def run_benchmark(
    cases: list[BenchmarkCase] | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict:
    """Run the full benchmark suite.

    Parameters
    ----------
    cases : list[BenchmarkCase] | None
        Cases to run. Loads from data/benchmark_cases.json if None.
    provider, model, base_url : str | None
        AI provider overrides.

    Returns
    -------
    dict
        Aggregate results with per-case scores and summary metrics.
    """
    from luxc.ai.generate import generate_lux_shader

    if cases is None:
        cases = load_benchmark_cases()

    scores: list[BenchmarkScore] = []
    for case in cases:
        result = generate_lux_shader(
            case.description,
            verify=True,
            model=model,
            max_retries=2,
            provider=provider,
            base_url=base_url,
        )
        score = score_generation(case, result)
        scores.append(score)

    # Aggregate
    total = len(scores)
    compiled = sum(1 for s in scores if s.compilation_success)
    avg_accuracy = (
        sum(s.parameter_accuracy for s in scores if s.compilation_success)
        / max(compiled, 1)
    )
    energy_ok = sum(1 for s in scores if s.energy_conserving)
    layers_ok = sum(1 for s in scores if s.layer_correctness)

    return {
        "total_cases": total,
        "compilation_rate": compiled / max(total, 1),
        "average_parameter_accuracy": avg_accuracy,
        "energy_conservation_rate": energy_ok / max(total, 1),
        "layer_correctness_rate": layers_ok / max(total, 1),
        "scores": [
            {
                "case_id": s.case_id,
                "compiled": s.compilation_success,
                "accuracy": s.parameter_accuracy,
                "energy_ok": s.energy_conserving,
                "layers_ok": s.layer_correctness,
                "attempts": s.attempts,
            }
            for s in scores
        ],
    }
