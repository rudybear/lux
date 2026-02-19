"""End-to-end screenshot test: compile procedural_noise.lux and render to PNG.

This script:
  1. Compiles examples/procedural_noise.lux -> playground/*.spv
  2. Loads the fragment SPIR-V via the fullscreen render harness
  3. Renders a 512x512 frame and saves playground/procedural_noise.png
  4. Validates the output (coverage, variation, color range)

Usage:
    python test_procedural_noise.py

Exit code 0 on success, 1 on failure.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np


PLAYGROUND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLAYGROUND_DIR.parent
LUX_SOURCE = PROJECT_ROOT / "examples" / "procedural_noise.lux"
FRAG_SPV = PLAYGROUND_DIR / "procedural_noise.frag.spv"
OUTPUT_PNG = PLAYGROUND_DIR / "procedural_noise.png"


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def compile_shader() -> bool:
    """Run luxc to compile the .lux file to SPIR-V."""
    step("Step 1: Compiling procedural_noise.lux")

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
    """Check the rendered image for expected noise content."""
    step("Step 3: Validating output")
    h, w, c = pixels.shape
    assert (h, w, c) == (512, 512, 4), f"Unexpected shape: {pixels.shape}"

    rgb = pixels[:, :, :3].astype(np.float32)
    brightness = rgb.sum(axis=2)

    # Check 1: image should not be all black
    non_black = (brightness > 0).sum()
    total = h * w
    coverage = non_black / total * 100
    print(f"  Non-black coverage: {coverage:.1f}% ({non_black}/{total} pixels)")
    if coverage < 80:
        print("FAIL: too many black pixels — noise not rendered?")
        return False

    # Check 2: image should not be uniform (noise should vary)
    min_b, max_b = brightness.min(), brightness.max()
    dynamic_range = max_b - min_b
    print(f"  Brightness range: {min_b:.1f} - {max_b:.1f} (dynamic range: {dynamic_range:.1f})")
    if dynamic_range < 30:
        print("FAIL: image looks uniform — noise variation not visible")
        return False

    # Check 3: standard deviation of brightness should be non-trivial
    # Noise produces varied patterns; a flat color would have stddev near 0
    std_dev = brightness.std()
    print(f"  Brightness std dev: {std_dev:.2f}")
    if std_dev < 5.0:
        print("FAIL: brightness variation too low — noise not generating patterns")
        return False

    # Check 4: check color channels — noise shader uses earth-tone palette
    # (browns/tans), so R channel should generally be significant
    mean_r = rgb[:, :, 0].mean()
    mean_g = rgb[:, :, 1].mean()
    mean_b = rgb[:, :, 2].mean()
    print(f"  Mean RGB: ({mean_r:.1f}, {mean_g:.1f}, {mean_b:.1f})")
    if mean_r < 10 and mean_g < 10 and mean_b < 10:
        print("FAIL: mean color too dark — shader output not visible")
        return False

    # Check 5: spatial variation — compare top half vs bottom half
    # Noise should produce different patterns in different regions
    top_mean = brightness[:h // 2, :].mean()
    bot_mean = brightness[h // 2:, :].mean()
    left_mean = brightness[:, :w // 2].mean()
    right_mean = brightness[:, w // 2:].mean()
    print(f"  Quadrant means: top={top_mean:.1f} bot={bot_mean:.1f} left={left_mean:.1f} right={right_mean:.1f}")

    # At least some spatial variation expected (quadrants shouldn't be identical)
    quadrant_range = max(top_mean, bot_mean, left_mean, right_mean) - min(top_mean, bot_mean, left_mean, right_mean)
    print(f"  Quadrant variation: {quadrant_range:.1f}")
    if quadrant_range < 1.0:
        print("FAIL: all quadrants identical — noise has no spatial variation")
        return False

    print("  All validation checks passed.")
    return True


def main() -> None:
    print("Lux Shader Playground -- procedural_noise screenshot test")
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
    print("  SUCCESS: procedural_noise rendered and validated correctly!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
