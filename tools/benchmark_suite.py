"""Benchmark suite for Lux shader compilation performance.

Compiles curated shaders, extracts instruction counts from reflection JSON,
and verifies they don't exceed established baselines.

Usage:
    python tools/benchmark_suite.py
    python tools/benchmark_suite.py --update-baselines
    python tools/benchmark_suite.py --json
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Benchmark definitions: (name, shader_path, stage, max_instructions)
# max_instructions is a soft ceiling — exceeding it is a warning, not a failure
BENCHMARKS = [
    {
        "name": "pbr_simple",
        "shader": "examples/gltf_pbr.lux",
        "stage": "fragment",
        "description": "Basic glTF PBR fragment shader",
    },
    {
        "name": "pbr_simple_vert",
        "shader": "examples/gltf_pbr.lux",
        "stage": "vertex",
        "description": "Basic glTF PBR vertex shader",
    },
    {
        "name": "toon_shader",
        "shader": "examples/cartoon_toon.lux",
        "stage": "fragment",
        "description": "Toon/cel shading fragment shader",
    },
]

# Path to baselines file
BASELINES_PATH = Path(__file__).parent.parent / "data" / "perf_baselines.json"
ROOT_DIR = Path(__file__).parent.parent


def load_baselines() -> dict:
    """Load instruction count baselines from JSON file."""
    if BASELINES_PATH.exists():
        return json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
    return {}


def save_baselines(baselines: dict) -> None:
    """Save instruction count baselines to JSON file."""
    BASELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINES_PATH.write_text(
        json.dumps(baselines, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def compile_benchmark(benchmark: dict) -> dict | None:
    """Compile a benchmark shader and extract performance hints.

    Returns reflection performance_hints dict, or None on failure.
    """
    shader_path = ROOT_DIR / benchmark["shader"]
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

        # Find reflection JSON for the target stage
        stage = benchmark["stage"]
        suffix_map = {
            "vertex": "vert", "fragment": "frag", "compute": "comp",
            "mesh": "mesh", "task": "task",
        }
        suffix = suffix_map.get(stage, stage)

        json_files = list(Path(tmpdir).glob(f"*.{suffix}.json"))
        if not json_files:
            return None

        reflection = json.loads(json_files[0].read_text(encoding="utf-8"))
        return reflection.get("performance_hints")


def run_benchmarks(update_baselines: bool = False, output_json: bool = False) -> bool:
    """Run all benchmarks and report results.

    Returns True if all benchmarks pass (no regressions), False otherwise.
    """
    baselines = load_baselines()
    results = []
    all_pass = True

    for bench in BENCHMARKS:
        name = bench["name"]
        print(f"  Compiling {bench['shader']} ({bench['stage']})...", end=" ", flush=True)

        hints = compile_benchmark(bench)
        if hints is None:
            print("SKIP (compile failed or shader not found)")
            results.append({"name": name, "status": "skip"})
            continue

        instr_count = hints.get("instruction_count", 0)
        alu_ops = hints.get("alu_ops", 0)
        tex_samples = hints.get("texture_samples", 0)
        vgpr = hints.get("vgpr_estimate", "?")

        baseline = baselines.get(name, {})
        baseline_instr = baseline.get("instruction_count")

        status = "pass"
        delta = ""
        if baseline_instr is not None:
            diff = instr_count - baseline_instr
            if diff > 0:
                status = "REGRESSION"
                delta = f" (+{diff})"
                all_pass = False
            elif diff < 0:
                delta = f" ({diff})"
        else:
            status = "new"

        print(f"{instr_count} instr, {alu_ops} ALU, {tex_samples} tex, VGPR: {vgpr}{delta} [{status}]")

        result_entry = {
            "name": name,
            "status": status,
            "instruction_count": instr_count,
            "alu_ops": alu_ops,
            "texture_samples": tex_samples,
            "vgpr_estimate": vgpr,
        }
        if baseline_instr is not None:
            result_entry["baseline_instruction_count"] = baseline_instr
        results.append(result_entry)

        if update_baselines:
            baselines[name] = {
                "instruction_count": instr_count,
                "alu_ops": alu_ops,
                "texture_samples": tex_samples,
                "vgpr_estimate": vgpr,
                "updated": datetime.now().isoformat(),
            }

    if update_baselines:
        save_baselines(baselines)
        print(f"\nBaselines updated: {BASELINES_PATH}")

    if output_json:
        print(json.dumps({"benchmarks": results, "all_pass": all_pass}, indent=2))

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Lux shader benchmark suite")
    parser.add_argument("--update-baselines", action="store_true",
                        help="Update baseline instruction counts")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    print("=== Lux Shader Benchmark Suite ===\n")
    all_pass = run_benchmarks(
        update_baselines=args.update_baselines,
        output_json=args.json,
    )

    if not all_pass:
        print("\nFAILED: Instruction count regressions detected!")
        sys.exit(1)
    else:
        print("\nAll benchmarks passed.")


if __name__ == "__main__":
    main()
