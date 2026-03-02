"""Run shader optimization experiments across all Lux shaders + original GLSL.

Compiles each shader with 3 optimization levels:
  1. default (AST-level optimizations only)
  2. -O (AST + spirv-opt -O)
  3. --perf (AST + performance-oriented spirv-opt)

Also measures original GLSL-compiled SPIR-V for baseline comparison.
Records instruction counts and SPIR-V binary sizes.
"""
import json
import os
import struct
import sys
from pathlib import Path

# Add project root to path so we can import luxc
sys.path.insert(0, "D:/shaderlang")

from luxc.compiler import compile_source

SHADERS = ["pbr", "skybox", "tonemap", "spbrdf"]
LUX_DIR = Path("D:/shaderlang/projects/nadrin-pbr/lux-variant/data/shaders/lux")
ORIG_SPIRV_DIR = Path("D:/shaderlang/projects/nadrin-pbr/upstream/data/shaders/spirv")
RESULTS_DIR = Path("D:/shaderlang/projects/nadrin-pbr/optimization/results")

CONFIGS = [
    {"name": "default", "optimize": False, "perf_optimize": False},
    {"name": "opt-O", "optimize": True, "perf_optimize": False},
    {"name": "perf", "optimize": False, "perf_optimize": True},
]

# Per-shader compile defines (e.g., spbrdf needs custom workgroup size)
SHADER_DEFINES = {
    "spbrdf": {"workgroup_size_x": 32, "workgroup_size_y": 32},
}

# Map from our shader names to original GLSL SPIR-V filenames
ORIG_SPV_MAP = {
    "pbr": {"vert": "pbr_vs.spv", "frag": "pbr_fs.spv"},
    "skybox": {"vert": "skybox_vs.spv", "frag": "skybox_fs.spv"},
    "tonemap": {"vert": "tonemap_vs.spv", "frag": "tonemap_fs.spv"},
    "spbrdf": {"comp": "spbrdf_cs.spv"},
}


def count_spirv_instructions(spv_path):
    """Count SPIR-V instructions by parsing the binary."""
    try:
        with open(spv_path, "rb") as f:
            data = f.read()
        if len(data) < 20:
            return 0
        words = len(data) // 4
        word_data = struct.unpack(f"<{words}I", data)
        pos = 5  # skip 5-word header
        count = 0
        while pos < words:
            word_count = word_data[pos] >> 16
            if word_count == 0:
                break
            count += 1
            pos += word_count
        return count
    except Exception:
        return -1


def get_spirv_stats(spv_path):
    """Get instruction count and file size for a SPIR-V binary."""
    size = os.path.getsize(spv_path)
    instr_count = count_spirv_instructions(spv_path)
    return {"spv_size": size, "spv_instructions": instr_count}


def measure_original():
    """Measure original GLSL-compiled SPIR-V files."""
    print("\n" + "=" * 60)
    print("Configuration: ORIGINAL (GLSL compiled via glslangValidator)")
    print("=" * 60)

    results = {}
    for shader in SHADERS:
        shader_result = {"stages": {}}
        stage_map = ORIG_SPV_MAP.get(shader, {})
        for stage, filename in sorted(stage_map.items()):
            spv_path = ORIG_SPIRV_DIR / filename
            if spv_path.exists():
                stats = get_spirv_stats(str(spv_path))
                shader_result["stages"][stage] = stats
                print(f"  {shader}.{stage}: {stats['spv_instructions']} instructions, "
                      f"{stats['spv_size']} bytes")
            else:
                print(f"  {shader}.{stage}: NOT FOUND ({spv_path})")
        results[shader] = shader_result
    return results


def measure_lux_config(config):
    """Compile and measure all Lux shaders with given config."""
    config_name = config["name"]
    config_dir = RESULTS_DIR / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Configuration: Lux {config_name}")
    print(f"  optimize={config['optimize']}, perf_optimize={config['perf_optimize']}")
    print(f"{'=' * 60}")

    config_results = {}

    for shader in SHADERS:
        shader_dir = config_dir / shader
        shader_dir.mkdir(parents=True, exist_ok=True)

        src_path = LUX_DIR / f"{shader}.lux"
        source = src_path.read_text(encoding="utf-8")

        defines = SHADER_DEFINES.get(shader, {})
        print(f"\n  Compiling {shader}.lux...{' (defines: ' + str(defines) + ')' if defines else ''}")
        try:
            compile_source(
                source=source,
                stem=shader,
                output_dir=shader_dir,
                source_dir=src_path.parent,
                emit_asm=True,
                validate=False,
                emit_reflection=True,
                optimize=config["optimize"],
                perf_optimize=config["perf_optimize"],
                defines=defines,
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            config_results[shader] = {"error": str(e)}
            continue

        # Collect metrics
        shader_result = {"stages": {}}
        for f in sorted(os.listdir(shader_dir)):
            if f.endswith(".spv"):
                stage = f.replace(f"{shader}.", "").replace(".spv", "")
                spv_path = shader_dir / f
                stats = get_spirv_stats(str(spv_path))

                # Also get reflection data
                json_path = shader_dir / f.replace(".spv", ".json")
                if json_path.exists():
                    with open(json_path) as jf:
                        refl = json.load(jf)
                    hints = refl.get("performance_hints", {})
                    stats.update({
                        "reflection_instruction_count": hints.get("instruction_count", 0),
                        "alu_ops": hints.get("alu_ops", 0),
                        "texture_samples": hints.get("texture_samples", 0),
                        "branches": hints.get("branches", 0),
                        "function_calls": hints.get("function_calls", 0),
                        "vgpr_estimate": hints.get("vgpr_estimate", "unknown"),
                    })

                shader_result["stages"][stage] = stats
                print(f"    {stage}: {stats['spv_instructions']} SPIR-V instrs, "
                      f"{stats.get('alu_ops', '?')} ALU, "
                      f"{stats['spv_size']} bytes")

        config_results[shader] = shader_result

    return config_results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Measure original GLSL SPIR-V
    all_results["original"] = measure_original()

    # Measure each Lux configuration
    for config in CONFIGS:
        all_results[config["name"]] = measure_lux_config(config)

    # Save full results
    results_path = RESULTS_DIR / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nFull results saved to {results_path}")

    # Print comparison tables
    configs_ordered = ["original", "default", "opt-O", "perf"]

    print(f"\n{'=' * 95}")
    print("SPIR-V INSTRUCTION COUNT COMPARISON")
    print(f"{'=' * 95}")
    header = f"{'Shader':<12} {'Stage':<6}"
    for cfg in configs_ordered:
        header += f" {cfg:>12}"
    header += f" {'Orig->Def':>12} {'Def->Perf':>12}"
    print(header)
    print("-" * 95)

    for shader in SHADERS:
        stages_set = set()
        for cfg in configs_ordered:
            if shader in all_results.get(cfg, {}):
                stages_set.update(all_results[cfg][shader].get("stages", {}).keys())

        for stage in sorted(stages_set):
            row = f"{shader:<12} {stage:<6}"
            values = {}
            for cfg in configs_ordered:
                val = (all_results.get(cfg, {}).get(shader, {})
                       .get("stages", {}).get(stage, {})
                       .get("spv_instructions", "?"))
                values[cfg] = val
                row += f" {str(val):>12}"

            # Compute original -> default gain
            orig_val = values.get("original", "?")
            def_val = values.get("default", "?")
            if isinstance(orig_val, int) and isinstance(def_val, int) and orig_val > 0:
                pct = (orig_val - def_val) / orig_val * 100
                row += f" {pct:>+11.1f}%"
            else:
                row += f" {'N/A':>12}"

            # Compute default -> perf gain
            perf_val = values.get("perf", "?")
            if isinstance(def_val, int) and isinstance(perf_val, int) and def_val > 0:
                pct = (def_val - perf_val) / def_val * 100
                row += f" {pct:>+11.1f}%"
            else:
                row += f" {'N/A':>12}"

            print(row)

    print(f"\n{'=' * 95}")
    print("SPIR-V BINARY SIZE COMPARISON (bytes)")
    print(f"{'=' * 95}")
    header = f"{'Shader':<12} {'Stage':<6}"
    for cfg in configs_ordered:
        header += f" {cfg:>12}"
    header += f" {'Orig->Def':>12} {'Def->Perf':>12}"
    print(header)
    print("-" * 95)

    total_sizes = {cfg: 0 for cfg in configs_ordered}

    for shader in SHADERS:
        stages_set = set()
        for cfg in configs_ordered:
            if shader in all_results.get(cfg, {}):
                stages_set.update(all_results[cfg][shader].get("stages", {}).keys())

        for stage in sorted(stages_set):
            row = f"{shader:<12} {stage:<6}"
            values = {}
            for cfg in configs_ordered:
                val = (all_results.get(cfg, {}).get(shader, {})
                       .get("stages", {}).get(stage, {})
                       .get("spv_size", "?"))
                values[cfg] = val
                if isinstance(val, int):
                    total_sizes[cfg] += val
                row += f" {str(val):>12}"

            orig_val = values.get("original", "?")
            def_val = values.get("default", "?")
            if isinstance(orig_val, int) and isinstance(def_val, int) and orig_val > 0:
                pct = (orig_val - def_val) / orig_val * 100
                row += f" {pct:>+11.1f}%"
            else:
                row += f" {'N/A':>12}"

            perf_val = values.get("perf", "?")
            if isinstance(def_val, int) and isinstance(perf_val, int) and def_val > 0:
                pct = (def_val - perf_val) / def_val * 100
                row += f" {pct:>+11.1f}%"
            else:
                row += f" {'N/A':>12}"

            print(row)

    # Totals
    row = f"{'TOTAL':<12} {'':>6}"
    for cfg in configs_ordered:
        row += f" {total_sizes[cfg]:>12}"
    orig_t = total_sizes.get("original", 0)
    def_t = total_sizes.get("default", 0)
    perf_t = total_sizes.get("perf", 0)
    if orig_t > 0 and def_t > 0:
        pct = (orig_t - def_t) / orig_t * 100
        row += f" {pct:>+11.1f}%"
    else:
        row += f" {'N/A':>12}"
    if def_t > 0 and perf_t > 0:
        pct = (def_t - perf_t) / def_t * 100
        row += f" {pct:>+11.1f}%"
    print(row)


if __name__ == "__main__":
    main()
