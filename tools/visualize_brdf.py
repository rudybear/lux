#!/usr/bin/env python3
"""BRDF & Layer Visualization Tool — compile, render, and composite BRDF viz shaders.

Usage:
    python -m tools.visualize_brdf [OPTIONS]

Options:
    --shader NAME     Render only named shader (transfer, polar, sweep, furnace, layers)
    --output DIR      Output directory (default: screenshots/brdf_viz/)
    --composite       Generate composite image combining all panels
    --width N         Render width per shader (default: 512)
    --height N        Render height per shader (default: 512)
    --skip-compile    Use cached SPIR-V
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
SHADERCACHE = PROJECT_ROOT / "shadercache"

# Shader registry: name -> (lux file, spv file)
SHADERS = {
    "transfer": ("viz_transfer_functions.lux", "viz_transfer_functions.frag.spv"),
    "polar": ("viz_brdf_polar.lux", "viz_brdf_polar.frag.spv"),
    "sweep": ("viz_param_sweep.lux", "viz_param_sweep.frag.spv"),
    "furnace": ("viz_furnace_test.lux", "viz_furnace_test.frag.spv"),
    "layers": ("viz_layer_energy.lux", "viz_layer_energy.frag.spv"),
}


def compile_shader(name: str) -> bool:
    """Compile a visualization shader to SPIR-V."""
    lux_file, _ = SHADERS[name]
    src_path = EXAMPLES_DIR / lux_file

    if not src_path.exists():
        print(f"  ERROR: source not found: {src_path}")
        return False

    cmd = [sys.executable, "-m", "luxc", str(src_path), "-o", str(SHADERCACHE)]
    print(f"  Compiling {lux_file}...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"  FAIL: {result.stderr.strip()}")
        return False

    print(f"  OK: {result.stdout.strip()}")
    return True


def render_shader(name: str, output_dir: Path, width: int, height: int) -> Path | None:
    """Render a compiled visualization shader to PNG."""
    _, spv_file = SHADERS[name]
    spv_path = SHADERCACHE / spv_file

    if not spv_path.exists():
        print(f"  ERROR: compiled shader not found: {spv_path}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.png"

    try:
        sys.path.insert(0, str(PROJECT_ROOT / "playground"))
        from render_harness import render_fullscreen, save_png

        print(f"  Rendering {name} ({width}x{height})...")
        pixels = render_fullscreen(spv_path, width=width, height=height)
        save_png(pixels, output_path)
        print(f"  OK: {output_path}")
        return output_path
    except Exception as exc:
        print(f"  FAIL: rendering {name}: {exc}")
        return None


def create_composite(output_dir: Path, width: int, height: int) -> Path | None:
    """Combine all rendered panels into a 3x2 composite report."""
    try:
        from PIL import Image
    except ImportError:
        print("  ERROR: PIL not available for compositing")
        return None

    # Composite layout: 3 columns x 2 rows
    # Row 0: transfer, polar, sweep
    # Row 1: furnace, layers, (empty)
    layout = [
        ["transfer", "polar", "sweep"],
        ["furnace", "layers", None],
    ]

    comp_w = width * 3
    comp_h = height * 2
    composite = Image.new("RGBA", (comp_w, comp_h), (20, 20, 28, 255))

    for row_idx, row in enumerate(layout):
        for col_idx, name in enumerate(row):
            if name is None:
                continue
            panel_path = output_dir / f"{name}.png"
            if not panel_path.exists():
                print(f"  WARNING: missing panel {panel_path}")
                continue
            panel = Image.open(panel_path).resize((width, height))
            composite.paste(panel, (col_idx * width, row_idx * height))

    output_path = output_dir / "brdf_visualization_report.png"
    composite.save(output_path)
    print(f"  Composite saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="BRDF & Layer Visualization Tool")
    parser.add_argument(
        "--shader",
        choices=list(SHADERS.keys()),
        help="Render only the named shader",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "screenshots" / "brdf_viz",
        help="Output directory (default: screenshots/brdf_viz/)",
    )
    parser.add_argument(
        "--composite",
        action="store_true",
        help="Generate composite image combining all panels",
    )
    parser.add_argument("--width", type=int, default=512, help="Render width")
    parser.add_argument("--height", type=int, default=512, help="Render height")
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip compilation, use cached SPIR-V",
    )

    args = parser.parse_args()

    # Determine which shaders to process
    names = [args.shader] if args.shader else list(SHADERS.keys())

    print("=" * 60)
    print("  BRDF & Layer Visualization")
    print("=" * 60)

    # Step 1: Compile
    if not args.skip_compile:
        print("\n--- Compilation ---")
        SHADERCACHE.mkdir(parents=True, exist_ok=True)
        for name in names:
            if not compile_shader(name):
                print(f"  Stopping: compilation failed for {name}")
                sys.exit(1)

    # Step 2: Render
    print("\n--- Rendering ---")
    rendered = []
    for name in names:
        path = render_shader(name, args.output, args.width, args.height)
        if path:
            rendered.append(path)

    # Step 3: Composite
    if args.composite and len(rendered) > 0:
        print("\n--- Compositing ---")
        create_composite(args.output, args.width, args.height)

    print(f"\n{'=' * 60}")
    print(f"  Done: {len(rendered)}/{len(names)} shaders rendered")
    print(f"  Output: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
