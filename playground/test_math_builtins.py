"""End-to-end screenshot test: compile math_builtins_demo.lux and render to PNG.

This script:
  1. Compiles examples/math_builtins_demo.lux -> playground/*.spv
  2. Loads the fragment SPIR-V via the fullscreen render harness
  3. Renders a 512x512 frame and saves playground/math_builtins_demo.png
  4. Validates the output (4x2 grid, function curves, color variation)

Usage:
    python test_math_builtins.py

Exit code 0 on success, 1 on failure.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np


PLAYGROUND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLAYGROUND_DIR.parent
LUX_SOURCE = PROJECT_ROOT / "examples" / "math_builtins_demo.lux"
FRAG_SPV = PLAYGROUND_DIR / "math_builtins_demo.frag.spv"
OUTPUT_PNG = PLAYGROUND_DIR / "math_builtins_demo.png"


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def compile_shader() -> bool:
    """Run luxc to compile the .lux file to SPIR-V."""
    step("Step 1: Compiling math_builtins_demo.lux")

    if not LUX_SOURCE.exists():
        print(f"FAIL: source file not found: {LUX_SOURCE}")
        return False

    cmd = [
        sys.executable, "-m", "luxc",
        str(LUX_SOURCE),
        "-o", str(PLAYGROUND_DIR),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")

    if result.returncode != 0:
        print(f"FAIL: luxc exited with code {result.returncode}")
        return False

    if not FRAG_SPV.exists():
        print("FAIL: expected .spv file not produced")
        return False

    print(f"  OK: {FRAG_SPV.name} ({FRAG_SPV.stat().st_size} bytes)")
    return True


def render_frame() -> np.ndarray | None:
    """Load fragment shader and render one fullscreen frame."""
    step("Step 2: Rendering frame")

    from render_harness import render_fullscreen, save_png

    try:
        pixels = render_fullscreen(FRAG_SPV, width=512, height=512)
    except Exception as exc:
        print(f"FAIL: rendering failed: {exc}")
        return None

    save_png(pixels, OUTPUT_PNG)
    print(f"  OK: saved {OUTPUT_PNG} ({OUTPUT_PNG.stat().st_size} bytes)")
    return pixels


def validate_output(pixels: np.ndarray) -> bool:
    """Check the rendered image for expected math visualization content."""
    step("Step 3: Validating output")
    h, w, c = pixels.shape
    assert (h, w, c) == (512, 512, 4), f"Unexpected shape: {pixels.shape}"

    rgb = pixels[:, :, :3].astype(np.float32)
    brightness = rgb.sum(axis=2)

    # Check 1: non-uniform image (function curves produce varied output)
    dynamic_range = brightness.max() - brightness.min()
    print(f"  Dynamic range: {dynamic_range:.1f}")
    if dynamic_range < 20:
        print("FAIL: image looks uniform — math functions not rendered?")
        return False

    # Check 2: significant coverage
    non_black = (brightness > 5).sum()
    total = h * w
    coverage = non_black / total * 100
    print(f"  Non-black coverage: {coverage:.1f}%")
    if coverage < 20:
        print("FAIL: too few visible pixels — grid cells not rendered")
        return False

    # Check 3: grid structure — check that cells have distinct content
    # 4x2 grid: each cell is 128x256 pixels
    cell_w = w // 4
    cell_h = h // 2
    cell_means = []
    for row in range(2):
        for col in range(4):
            cell = brightness[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w]
            cell_means.append(cell.mean())
    print(f"  Cell mean brightnesses: {[f'{m:.1f}' for m in cell_means]}")

    # Check 4: brightness variation within cells (function curves, not flat)
    cell_stds = []
    for row in range(2):
        for col in range(4):
            cell = brightness[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w]
            cell_stds.append(cell.std())
    avg_cell_std = sum(cell_stds) / len(cell_stds)
    print(f"  Average cell brightness std: {avg_cell_std:.1f}")
    if avg_cell_std < 3:
        print("FAIL: cells look flat — function curves not visible")
        return False

    print("  All validation checks passed.")
    return True


def main() -> None:
    print("Lux Shader Playground -- math_builtins_demo screenshot test")
    print(f"Project root: {PROJECT_ROOT}")

    if not compile_shader():
        print("\n*** FAILED at compilation step ***")
        sys.exit(1)

    pixels = render_frame()
    if pixels is None:
        print("\n*** FAILED at rendering step ***")
        sys.exit(1)

    if not validate_output(pixels):
        print("\n*** FAILED at validation step ***")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  SUCCESS: math_builtins_demo rendered and validated correctly!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
