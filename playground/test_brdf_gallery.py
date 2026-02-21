"""End-to-end screenshot test: compile brdf_gallery.lux and render to PNG.

This script:
  1. Compiles examples/brdf_gallery.lux -> shadercache/*.spv
  2. Loads the fragment SPIR-V via the fullscreen render harness
  3. Renders a 512x512 frame and saves screenshots/brdf_gallery.png
  4. Validates the output (4 distinct BRDF bands, color variation)

Usage:
    python test_brdf_gallery.py

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
LUX_SOURCE = PROJECT_ROOT / "examples" / "brdf_gallery.lux"
FRAG_SPV = SHADERCACHE / "brdf_gallery.frag.spv"
OUTPUT_PNG = SCREENSHOTS / "brdf_gallery.png"


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def compile_shader() -> bool:
    """Run luxc to compile the .lux file to SPIR-V."""
    step("Step 1: Compiling brdf_gallery.lux")

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
    """Check the rendered image for expected BRDF gallery content."""
    step("Step 3: Validating output")
    h, w, c = pixels.shape
    assert (h, w, c) == (512, 512, 4), f"Unexpected shape: {pixels.shape}"

    rgb = pixels[:, :, :3].astype(np.float32)
    brightness = rgb.sum(axis=2)

    # Check 1: non-black coverage > 90% (fullscreen bands)
    non_black = (brightness > 0).sum()
    total = h * w
    coverage = non_black / total * 100
    print(f"  Non-black coverage: {coverage:.1f}%")
    if coverage < 90:
        print("FAIL: too many black pixels — BRDF bands not filling screen")
        return False

    # Check 2: each band has distinct average brightness
    band_h = h // 4
    band_means = []
    for i in range(4):
        band = rgb[i * band_h : (i + 1) * band_h, :, :]
        mean = band.mean()
        band_means.append(mean)
        print(f"  Band {i} mean brightness: {mean:.1f}")

    # Check 3: spatial variation between bands
    band_range = max(band_means) - min(band_means)
    print(f"  Band brightness range: {band_range:.1f}")
    if band_range < 3:
        print("FAIL: bands look identical — different BRDFs should produce distinct results")
        return False

    # Check 4: color channel variation (different BRDFs have different colors)
    band_colors = []
    for i in range(4):
        band = rgb[i * band_h : (i + 1) * band_h, :, :]
        r_mean = band[:, :, 0].mean()
        g_mean = band[:, :, 1].mean()
        b_mean = band[:, :, 2].mean()
        band_colors.append((r_mean, g_mean, b_mean))
        print(f"  Band {i} color: R={r_mean:.1f} G={g_mean:.1f} B={b_mean:.1f}")

    # At least 2 bands should have different dominant colors
    distinct_count = 0
    for i in range(4):
        for j in range(i + 1, 4):
            r_diff = abs(band_colors[i][0] - band_colors[j][0])
            g_diff = abs(band_colors[i][1] - band_colors[j][1])
            b_diff = abs(band_colors[i][2] - band_colors[j][2])
            if r_diff > 5 or g_diff > 5 or b_diff > 5:
                distinct_count += 1
    print(f"  Distinct band pairs: {distinct_count}/6")
    if distinct_count < 2:
        print("FAIL: bands not distinct enough — BRDFs should produce different colors")
        return False

    print("  All validation checks passed.")
    return True


def main() -> None:
    print("Lux Shader Playground -- brdf_gallery screenshot test")
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
    print("  SUCCESS: brdf_gallery rendered and validated correctly!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
