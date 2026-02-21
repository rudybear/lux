"""End-to-end screenshot test: compile colorspace_demo.lux and render to PNG.

This script:
  1. Compiles examples/colorspace_demo.lux -> shadercache/*.spv
  2. Loads the fragment SPIR-V via the fullscreen render harness
  3. Renders a 512x512 frame and saves screenshots/colorspace_demo.png
  4. Validates the output (full color range, smooth gradients, hue variation)

Usage:
    python test_colorspace.py

Exit code 0 on success, 1 on failure.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np


PLAYGROUND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLAYGROUND_DIR.parent
SHADERCACHE = PROJECT_ROOT / "shadercache"
SCREENSHOTS = PROJECT_ROOT / "screenshots"
LUX_SOURCE = PROJECT_ROOT / "examples" / "colorspace_demo.lux"
FRAG_SPV = SHADERCACHE / "colorspace_demo.frag.spv"
OUTPUT_PNG = SCREENSHOTS / "colorspace_demo.png"


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def compile_shader() -> bool:
    """Run luxc to compile the .lux file to SPIR-V."""
    step("Step 1: Compiling colorspace_demo.lux")

    if not LUX_SOURCE.exists():
        print(f"FAIL: source file not found: {LUX_SOURCE}")
        return False

    cmd = [
        sys.executable, "-m", "luxc",
        str(LUX_SOURCE),
        "-o", str(SHADERCACHE),
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
    """Check the rendered image for expected colorspace content."""
    step("Step 3: Validating output")
    h, w, c = pixels.shape
    assert (h, w, c) == (512, 512, 4), f"Unexpected shape: {pixels.shape}"

    rgb = pixels[:, :, :3].astype(np.float32)

    # Check 1: full color range — R, G, B channels all significant
    mean_r = rgb[:, :, 0].mean()
    mean_g = rgb[:, :, 1].mean()
    mean_b = rgb[:, :, 2].mean()
    print(f"  Mean RGB: ({mean_r:.1f}, {mean_g:.1f}, {mean_b:.1f})")
    if mean_r < 10 or mean_g < 10 or mean_b < 10:
        print("FAIL: one or more color channels too dark — rainbow not visible")
        return False

    # Check 2: smooth gradients — brightness standard deviation
    brightness = rgb.sum(axis=2)
    std_dev = brightness.std()
    print(f"  Brightness std dev: {std_dev:.2f}")
    if std_dev < 10:
        print("FAIL: brightness variation too low — gradients not visible")
        return False

    # Check 3: hue variation across image — compare left and right halves
    left_rgb = rgb[:, :w // 4, :].mean(axis=(0, 1))
    right_rgb = rgb[:, 3 * w // 4:, :].mean(axis=(0, 1))
    hue_diff = np.abs(left_rgb - right_rgb).max()
    print(f"  Left mean: ({left_rgb[0]:.1f}, {left_rgb[1]:.1f}, {left_rgb[2]:.1f})")
    print(f"  Right mean: ({right_rgb[0]:.1f}, {right_rgb[1]:.1f}, {right_rgb[2]:.1f})")
    print(f"  Max channel difference (left vs right): {hue_diff:.1f}")
    if hue_diff < 5:
        print("FAIL: no hue variation across image — HSV sweep not working")
        return False

    # Check 4: non-black coverage
    non_black = (brightness > 0).sum()
    total = h * w
    coverage = non_black / total * 100
    print(f"  Non-black coverage: {coverage:.1f}%")
    if coverage < 80:
        print("FAIL: too many black pixels")
        return False

    print("  All validation checks passed.")
    return True


def main() -> None:
    print("Lux Shader Playground -- colorspace screenshot test")
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
    print("  SUCCESS: colorspace_demo rendered and validated correctly!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
