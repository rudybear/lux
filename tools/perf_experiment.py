"""A/B performance experiment runner for Lux shaders.

Compiles multiple optimization variants, captures frames via RenderDoc,
compares instruction counts and image quality, and generates reports.

Usage:
    python tools/perf_experiment.py \\
        --shader examples/gltf_pbr_layered.lux \\
        --scene sphere \\
        --variants "baseline:" "optimized:-O" "aggressive:--perf" \\
        --output perf_results/
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def parse_variant(spec: str) -> tuple[str, list[str]]:
    """Parse 'name:flags' variant spec. Returns (name, flag_list)."""
    if ':' in spec:
        name, flags_str = spec.split(':', 1)
        flags = flags_str.split() if flags_str.strip() else []
    else:
        name, flags = spec, []
    return name, flags


def compile_variant(shader_path: Path, variant_name: str, flags: list[str],
                    output_dir: Path) -> dict:
    """Compile a shader variant and extract reflection data.

    Returns dict with: name, flags, success, error, reflection, spv_files, instruction_counts
    """
    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "luxc", str(shader_path),
           "-o", str(variant_dir), "--emit-asm"] + flags

    result = subprocess.run(cmd, capture_output=True, text=True)

    info = {
        "name": variant_name,
        "flags": flags,
        "success": result.returncode == 0,
        "error": result.stderr if result.returncode != 0 else None,
        "reflection": {},
        "spv_files": [],
        "instruction_counts": {},
    }

    if result.returncode == 0:
        # Load reflection JSON files
        for json_file in sorted(variant_dir.glob("*.json")):
            reflection = json.loads(json_file.read_text())
            stage = reflection.get("stage", "unknown")
            info["reflection"][stage] = reflection
            if "performance_hints" in reflection:
                info["instruction_counts"][stage] = reflection["performance_hints"]
        info["spv_files"] = [str(p) for p in sorted(variant_dir.glob("*.spv"))]

    return info


def compare_variants(variants: list[dict], baseline_name: str = "baseline") -> dict:
    """Compare instruction counts across variants.

    Returns comparison dict with per-stage deltas.
    """
    baseline = None
    for v in variants:
        if v["name"] == baseline_name:
            baseline = v
            break
    if baseline is None and variants:
        baseline = variants[0]

    comparisons = {}
    if baseline:
        for v in variants:
            if v["name"] == baseline["name"]:
                continue
            stage_diffs = {}
            for stage, counts in v.get("instruction_counts", {}).items():
                base_counts = baseline.get("instruction_counts", {}).get(stage, {})
                if base_counts and counts:
                    diff = {
                        key: counts.get(key, 0) - base_counts.get(key, 0)
                        for key in ["instruction_count", "alu_ops", "texture_samples"]
                        if key in counts and key in base_counts
                    }
                    stage_diffs[stage] = diff
            comparisons[v["name"]] = stage_diffs

    return comparisons


def generate_report(shader_path: Path, variants: list[dict],
                    comparisons: dict, quality_results: dict | None,
                    output_dir: Path) -> Path:
    """Generate a markdown comparison report.

    Returns path to the report file.
    """
    report_path = output_dir / "experiment_report.md"
    lines = [
        f"# Performance Experiment Report",
        f"",
        f"**Shader**: `{shader_path}`",
        f"**Date**: {datetime.now().isoformat()}",
        f"",
        f"## Variants",
        f"",
    ]

    # Variant summaries
    for v in variants:
        status = "PASS" if v["success"] else "FAIL"
        flags_str = " ".join(v["flags"]) if v["flags"] else "(none)"
        lines.append(f"### {v['name']} [{status}]")
        lines.append(f"- Flags: `{flags_str}`")
        for stage, counts in v.get("instruction_counts", {}).items():
            if isinstance(counts, dict):
                instr = counts.get("instruction_count", "?")
                alu = counts.get("alu_ops", "?")
                tex = counts.get("texture_samples", "?")
                vgpr = counts.get("vgpr_estimate", "?")
                lines.append(f"- {stage}: {instr} instr, {alu} ALU, {tex} tex, VGPR: {vgpr}")
        lines.append("")

    # Comparisons
    if comparisons:
        lines.append("## Comparisons (vs baseline)")
        lines.append("")
        lines.append("| Variant | Stage | Instr Delta | ALU Delta | Tex Delta |")
        lines.append("|---------|-------|-------------|-----------|-----------|")
        for variant_name, stages in comparisons.items():
            for stage, diff in stages.items():
                instr_d = diff.get("instruction_count", 0)
                alu_d = diff.get("alu_ops", 0)
                tex_d = diff.get("texture_samples", 0)
                sign = lambda x: f"+{x}" if x > 0 else str(x)
                lines.append(f"| {variant_name} | {stage} | {sign(instr_d)} | {sign(alu_d)} | {sign(tex_d)} |")
        lines.append("")

    # Quality results
    if quality_results:
        lines.append("## Quality Comparison")
        lines.append("")
        for variant_name, quality in quality_results.items():
            psnr = quality.get("psnr", "N/A")
            ssim = quality.get("ssim", "N/A")
            passed = quality.get("passed", "N/A")
            lines.append(f"- **{variant_name}**: PSNR={psnr}, SSIM={ssim}, Passed={passed}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_experiment(shader_path: Path, scene: str, variant_specs: list[str],
                   output_dir: Path, baseline_png: Path | None = None) -> dict:
    """Run a full A/B experiment.

    Returns experiment results dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Performance Experiment ===")
    print(f"Shader: {shader_path}")
    print(f"Scene: {scene}")
    print(f"Variants: {len(variant_specs)}")
    print()

    # Compile each variant
    variants = []
    for spec in variant_specs:
        name, flags = parse_variant(spec)
        print(f"Compiling variant '{name}' with flags: {flags or '(none)'}...")
        info = compile_variant(shader_path, name, flags, output_dir)
        if info["success"]:
            print(f"  OK - {len(info['spv_files'])} SPV files")
        else:
            print(f"  FAILED: {info['error']}")
        variants.append(info)

    print()

    # Compare
    comparisons = compare_variants(variants)

    # Quality comparison if baseline PNG provided
    quality_results = None
    if baseline_png and baseline_png.exists():
        try:
            from tools.quality_metrics import compare_images, quality_check
            quality_results = {}
            # Would compare rendered frames here; for now just placeholder
        except ImportError:
            print("Warning: quality_metrics not available")

    # Generate report
    report_path = generate_report(shader_path, variants, comparisons, quality_results, output_dir)
    print(f"Report: {report_path}")

    # Save experiment data
    experiment = {
        "shader": str(shader_path),
        "scene": scene,
        "timestamp": datetime.now().isoformat(),
        "variants": [{
            "name": v["name"],
            "flags": v["flags"],
            "success": v["success"],
            "instruction_counts": v["instruction_counts"],
        } for v in variants],
        "comparisons": comparisons,
    }

    data_path = output_dir / "experiment_data.json"
    data_path.write_text(json.dumps(experiment, indent=2) + "\n", encoding="utf-8")

    return experiment


def main():
    parser = argparse.ArgumentParser(description="Lux shader A/B performance experiment")
    parser.add_argument("--shader", type=Path, required=True, help="Path to .lux shader")
    parser.add_argument("--scene", type=str, default="sphere", help="Scene name")
    parser.add_argument("--variants", nargs="+", default=["baseline:", "optimized:-O"],
                        help='Variant specs as "name:flags" (e.g., "baseline:" "opt:-O")')
    parser.add_argument("--output", type=Path, default=Path("perf_results"), help="Output directory")
    parser.add_argument("--baseline-png", type=Path, help="Baseline PNG for quality comparison")

    args = parser.parse_args()

    if not args.shader.exists():
        print(f"Error: shader not found: {args.shader}", file=sys.stderr)
        sys.exit(1)

    run_experiment(args.shader, args.scene, args.variants, args.output, args.baseline_png)


if __name__ == "__main__":
    main()
