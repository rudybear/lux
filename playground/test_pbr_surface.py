"""End-to-end screenshot test: compile pbr_surface.lux and render to PNG.

This script:
  1. Compiles examples/pbr_surface.lux -> playground/*.spv
  2. Loads the SPIR-V via the PBR render harness (sphere + texture + lighting)
  3. Renders a 512x512 frame and saves playground/pbr_surface.png
  4. Validates the output (sphere visible, specular highlight, diffuse shading)

Usage:
    python test_pbr_surface.py

Exit code 0 on success, 1 on failure.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np


PLAYGROUND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLAYGROUND_DIR.parent
LUX_SOURCE = PROJECT_ROOT / "examples" / "pbr_surface.lux"
VERT_SPV = PLAYGROUND_DIR / "pbr_surface.vert.spv"
FRAG_SPV = PLAYGROUND_DIR / "pbr_surface.frag.spv"
OUTPUT_PNG = PLAYGROUND_DIR / "pbr_surface.png"


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def compile_shader() -> bool:
    """Run luxc to compile the .lux file to SPIR-V."""
    step("Step 1: Compiling pbr_surface.lux")

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
    """Check the rendered image for expected PBR surface content."""
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

    # Check 2: specular highlight — bright pixels above the average
    avg_brightness = brightness[brightness > 30].mean() if (brightness > 30).any() else 0
    bright_threshold = max(avg_brightness * 1.5, 150)
    bright_mask = brightness > bright_threshold
    bright_count = bright_mask.sum()
    print(f"  Avg brightness: {avg_brightness:.1f}, highlight threshold: {bright_threshold:.1f}")
    print(f"  Bright specular pixels: {bright_count}")
    if bright_count < 5:
        print("FAIL: no specular highlight detected")
        return False

    # Check 3: brightness variation (shading, not flat color)
    brightness_std = brightness[brightness > 30].std() if (brightness > 30).any() else 0
    print(f"  Brightness std dev: {brightness_std:.1f}")
    if brightness_std < 5:
        print("FAIL: no shading variation — sphere looks flat")
        return False

    # Check 4: diffuse shading gradient (brightness variation across sphere)
    mid_row = rgb[h // 2, :, :]
    mid_brightness = mid_row.sum(axis=1)
    non_zero = mid_brightness[mid_brightness > 10]
    if len(non_zero) > 10:
        grad_range = non_zero.max() - non_zero.min()
        print(f"  Mid-row brightness range: {grad_range:.1f}")
        if grad_range < 20:
            print("FAIL: no diffuse shading gradient visible")
            return False
    else:
        print("  Mid-row has few visible pixels (ok for small sphere)")

    print("  All validation checks passed.")
    return True


def main() -> None:
    print("Lux Shader Playground -- pbr_surface screenshot test")
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
    print("  SUCCESS: pbr_surface rendered and validated correctly!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
