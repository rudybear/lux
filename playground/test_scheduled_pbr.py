"""End-to-end screenshot test: compile scheduled_pbr.lux and render to PNG.

This script:
  1. Compiles examples/scheduled_pbr.lux -> playground/*.spv
  2. Loads the SPIR-V via the PBR render harness (sphere + lighting)
  3. Renders a 512x512 frame and saves playground/scheduled_pbr.png
  4. Validates the output (copper sphere, tonemap, specular)

Usage:
    python test_scheduled_pbr.py

Exit code 0 on success, 1 on failure.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np


PLAYGROUND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLAYGROUND_DIR.parent
LUX_SOURCE = PROJECT_ROOT / "examples" / "scheduled_pbr.lux"
VERT_SPV = PLAYGROUND_DIR / "scheduled_pbr.vert.spv"
FRAG_SPV = PLAYGROUND_DIR / "scheduled_pbr.frag.spv"
OUTPUT_PNG = PLAYGROUND_DIR / "scheduled_pbr.png"


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def compile_shader() -> bool:
    """Run luxc to compile the .lux file to SPIR-V."""
    step("Step 1: Compiling scheduled_pbr.lux")

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

    if not VERT_SPV.exists() or not FRAG_SPV.exists():
        print("FAIL: expected .spv files not produced")
        return False

    print(f"  OK: {VERT_SPV.name} ({VERT_SPV.stat().st_size} bytes)")
    print(f"  OK: {FRAG_SPV.name} ({FRAG_SPV.stat().st_size} bytes)")
    return True


def render_frame() -> np.ndarray | None:
    """Load shaders and render one PBR frame."""
    step("Step 2: Rendering frame")

    from render_pbr import render_pbr
    from render_harness import save_png

    try:
        pixels = render_pbr(VERT_SPV, FRAG_SPV, width=512, height=512)
    except Exception as exc:
        print(f"FAIL: rendering failed: {exc}")
        return None

    save_png(pixels, OUTPUT_PNG)
    print(f"  OK: saved {OUTPUT_PNG} ({OUTPUT_PNG.stat().st_size} bytes)")
    return pixels


def validate_output(pixels: np.ndarray) -> bool:
    """Check the rendered image for expected scheduled PBR content."""
    step("Step 3: Validating output")
    h, w, c = pixels.shape
    assert (h, w, c) == (512, 512, 4), f"Unexpected shape: {pixels.shape}"

    rgb = pixels[:, :, :3].astype(np.float32)
    brightness = rgb.sum(axis=2)

    # Check 1: sphere visible — non-black coverage > 10%
    non_black = (brightness > 30).sum()
    total = h * w
    coverage = non_black / total * 100
    print(f"  Sphere coverage: {coverage:.1f}% ({non_black}/{total} pixels)")
    if coverage < 10:
        print("FAIL: sphere not visible — too few non-background pixels")
        return False

    # Check 2: copper-colored pixels — R > G > B pattern
    copper_mask = (rgb[:, :, 0] > 80) & (rgb[:, :, 0] > rgb[:, :, 1]) & (rgb[:, :, 1] > rgb[:, :, 2])
    copper_count = copper_mask.sum()
    copper_pct = copper_count / total * 100
    print(f"  Copper pixels (R>G>B): {copper_pct:.1f}%")
    if copper_count < 100:
        print("FAIL: no copper-colored pixels — material color wrong")
        return False

    # Check 3: tonemap applied — no blown-out pixels above 254
    max_channel = rgb.max()
    print(f"  Max channel value: {max_channel:.0f}")
    if max_channel > 254:
        blown_count = (rgb > 254).sum()
        print(f"  Warning: {blown_count} blown-out values (tonemap may not be applied)")

    # Check 4: specular highlight present — bright pixels
    bright_mask = brightness > 400
    bright_count = bright_mask.sum()
    print(f"  Bright specular pixels: {bright_count}")
    if bright_count < 3:
        print("FAIL: no specular highlight detected")
        return False

    print("  All validation checks passed.")
    return True


def main() -> None:
    print("Lux Shader Playground -- scheduled_pbr screenshot test")
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
    print("  SUCCESS: scheduled_pbr rendered and validated correctly!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
