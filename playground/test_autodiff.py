"""End-to-end screenshot test: compile autodiff_demo.lux and render to PNG.

This script:
  1. Compiles examples/autodiff_demo.lux -> playground/*.spv
  2. Loads the fragment SPIR-V via the fullscreen render harness
  3. Renders a 512x512 frame and saves playground/autodiff_demo.png
  4. Validates the output (top/bottom halves different, gradient visualization)

Usage:
    python test_autodiff.py

Exit code 0 on success, 1 on failure.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np


PLAYGROUND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLAYGROUND_DIR.parent
LUX_SOURCE = PROJECT_ROOT / "examples" / "autodiff_demo.lux"
FRAG_SPV = PLAYGROUND_DIR / "autodiff_demo.frag.spv"
OUTPUT_PNG = PLAYGROUND_DIR / "autodiff_demo.png"


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def compile_shader() -> bool:
    """Run luxc to compile the .lux file to SPIR-V."""
    step("Step 1: Compiling autodiff_demo.lux")

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
    """Check the rendered image for expected autodiff content."""
    step("Step 3: Validating output")
    h, w, c = pixels.shape
    assert (h, w, c) == (512, 512, 4), f"Unexpected shape: {pixels.shape}"

    rgb = pixels[:, :, :3].astype(np.float32)
    brightness = rgb.sum(axis=2)

    # Check 1: non-black coverage > 80%
    non_black = (brightness > 0).sum()
    total = h * w
    coverage = non_black / total * 100
    print(f"  Non-black coverage: {coverage:.1f}%")
    if coverage < 80:
        print("FAIL: too many black pixels — autodiff output not visible")
        return False

    # Check 2: top and bottom halves visually different
    top_half = rgb[:h // 2, :, :]
    bot_half = rgb[h // 2:, :, :]
    top_mean = top_half.mean(axis=(0, 1))
    bot_mean = bot_half.mean(axis=(0, 1))
    half_diff = np.abs(top_mean - bot_mean).max()
    print(f"  Top half mean: ({top_mean[0]:.1f}, {top_mean[1]:.1f}, {top_mean[2]:.1f})")
    print(f"  Bot half mean: ({bot_mean[0]:.1f}, {bot_mean[1]:.1f}, {bot_mean[2]:.1f})")
    print(f"  Max half difference: {half_diff:.1f}")
    if half_diff < 5:
        print("FAIL: top and bottom halves look identical — f(x) and f'(x) should differ")
        return False

    # Check 3: gradient visualization varies with x (horizontal variation)
    left_col = brightness[:, :w // 4].mean()
    mid_col = brightness[:, w // 4 : 3 * w // 4].mean()
    right_col = brightness[:, 3 * w // 4:].mean()
    x_range = max(left_col, mid_col, right_col) - min(left_col, mid_col, right_col)
    print(f"  Horizontal regions: left={left_col:.1f} mid={mid_col:.1f} right={right_col:.1f}")
    print(f"  Horizontal variation: {x_range:.1f}")
    if x_range < 1.0:
        print("FAIL: no horizontal variation — wave function not varying with x")
        return False

    # Check 4: function uses warm colors (R channel), gradient uses cool (B channel)
    top_r = top_half[:, :, 0].mean()
    bot_b = bot_half[:, :, 2].mean()
    print(f"  Top R mean: {top_r:.1f}, Bot B mean: {bot_b:.1f}")
    if top_r < 5 and bot_b < 5:
        print("FAIL: expected warm top (R) and cool bottom (B)")
        return False

    print("  All validation checks passed.")
    return True


def main() -> None:
    print("Lux Shader Playground -- autodiff screenshot test")
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
    print("  SUCCESS: autodiff_demo rendered and validated correctly!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
