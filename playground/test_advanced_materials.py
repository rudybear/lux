"""End-to-end screenshot test: compile advanced_materials_demo.lux and render to PNG.

This script:
  1. Compiles examples/advanced_materials_demo.lux -> shadercache/*.spv
  2. Loads the fragment SPIR-V via the fullscreen render harness
  3. Renders a 512x512 frame and saves screenshots/advanced_materials_demo.png
  4. Validates the output (4 quadrants with distinct color signatures)

Usage:
    python test_advanced_materials.py

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
LUX_SOURCE = PROJECT_ROOT / "examples" / "advanced_materials_demo.lux"
FRAG_SPV = SHADERCACHE / "advanced_materials_demo.frag.spv"
OUTPUT_PNG = SCREENSHOTS / "advanced_materials_demo.png"


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def compile_shader() -> bool:
    """Run luxc to compile the .lux file to SPIR-V."""
    step("Step 1: Compiling advanced_materials_demo.lux")

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
    """Check the rendered image for expected advanced materials content."""
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
        print("FAIL: too many black pixels — material quadrants not visible")
        return False

    # Check 2: each quadrant has distinct color signature
    qh, qw = h // 2, w // 2
    quadrants = [
        ("BL-transmission", rgb[qh:, :qw, :]),
        ("BR-iridescence", rgb[qh:, qw:, :]),
        ("TL-dispersion", rgb[:qh, :qw, :]),
        ("TR-volume", rgb[:qh, qw:, :]),
    ]

    quad_colors = []
    for name, q in quadrants:
        r_mean = q[:, :, 0].mean()
        g_mean = q[:, :, 1].mean()
        b_mean = q[:, :, 2].mean()
        quad_colors.append((r_mean, g_mean, b_mean))
        print(f"  {name}: R={r_mean:.1f} G={g_mean:.1f} B={b_mean:.1f}")

    # Check 3: color channel variation across image
    all_r = rgb[:, :, 0].std()
    all_g = rgb[:, :, 1].std()
    all_b = rgb[:, :, 2].std()
    print(f"  Channel std devs: R={all_r:.1f} G={all_g:.1f} B={all_b:.1f}")
    if all_r < 3 and all_g < 3 and all_b < 3:
        print("FAIL: no color variation — all quadrants look the same")
        return False

    # Check 4: at least 2 quadrant pairs should be visually distinct
    distinct_count = 0
    for i in range(4):
        for j in range(i + 1, 4):
            diff = max(
                abs(quad_colors[i][0] - quad_colors[j][0]),
                abs(quad_colors[i][1] - quad_colors[j][1]),
                abs(quad_colors[i][2] - quad_colors[j][2]),
            )
            if diff > 5:
                distinct_count += 1
    print(f"  Distinct quadrant pairs: {distinct_count}/6")
    if distinct_count < 2:
        print("FAIL: quadrants not distinct enough — different materials should look different")
        return False

    print("  All validation checks passed.")
    return True


def main() -> None:
    print("Lux Shader Playground -- advanced_materials screenshot test")
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
    print("  SUCCESS: advanced_materials_demo rendered and validated correctly!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
