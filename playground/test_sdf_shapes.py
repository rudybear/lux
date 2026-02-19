"""End-to-end screenshot test: compile sdf_shapes.lux and render to PNG.

This script:
  1. Compiles examples/sdf_shapes.lux -> playground/*.spv
  2. Loads the fragment SPIR-V via the fullscreen render harness
  3. Renders a 512x512 frame and saves playground/sdf_shapes.png
  4. Validates the output (coverage, color regions, distance field edges)

Usage:
    python test_sdf_shapes.py

Exit code 0 on success, 1 on failure.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np


PLAYGROUND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLAYGROUND_DIR.parent
LUX_SOURCE = PROJECT_ROOT / "examples" / "sdf_shapes.lux"
FRAG_SPV = PLAYGROUND_DIR / "sdf_shapes.frag.spv"
OUTPUT_PNG = PLAYGROUND_DIR / "sdf_shapes.png"


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def compile_shader() -> bool:
    """Run luxc to compile the .lux file to SPIR-V."""
    step("Step 1: Compiling sdf_shapes.lux")

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
    """Check the rendered image for expected SDF content."""
    step("Step 3: Validating output")
    h, w, c = pixels.shape
    assert (h, w, c) == (512, 512, 4), f"Unexpected shape: {pixels.shape}"

    rgb = pixels[:, :, :3].astype(np.float32)

    # Check 1: image should not be uniform — SDFs produce varied output
    brightness = rgb.sum(axis=2)
    min_b, max_b = brightness.min(), brightness.max()
    dynamic_range = max_b - min_b
    print(f"  Brightness range: {min_b:.1f} - {max_b:.1f} (dynamic range: {dynamic_range:.1f})")
    if dynamic_range < 30:
        print("FAIL: image looks uniform — SDF shapes not rendered?")
        return False

    # Check 2: significant portion of the image should be non-black
    # The SDF shader colors both inside (warm) and outside (cool) regions,
    # so most of the screen should have some color
    non_black = (brightness > 0).sum()
    total = h * w
    coverage = non_black / total * 100
    print(f"  Non-black coverage: {coverage:.1f}% ({non_black}/{total} pixels)")
    if coverage < 50:
        print("FAIL: too few non-black pixels — expected near full coverage")
        return False

    # Check 3: there should be warm-colored pixels (inside SDF shapes)
    # Warm colors: R > G and R > B
    warm_mask = (rgb[:, :, 0] > 100) & (rgb[:, :, 0] > rgb[:, :, 2])
    warm_count = warm_mask.sum()
    warm_pct = warm_count / total * 100
    print(f"  Warm pixels (inside shapes): {warm_pct:.1f}%")
    if warm_count < 100:
        print("FAIL: no warm-colored pixels — SDF inside regions not visible")
        return False

    # Check 4: there should be cool-colored pixels (outside SDF shapes)
    # Cool colors: B > R
    cool_mask = (rgb[:, :, 2] > 50) & (rgb[:, :, 2] > rgb[:, :, 0])
    cool_count = cool_mask.sum()
    cool_pct = cool_count / total * 100
    print(f"  Cool pixels (outside shapes): {cool_pct:.1f}%")
    if cool_count < 100:
        print("FAIL: no cool-colored pixels — SDF outside regions not visible")
        return False

    # Check 5: there should be bright edge pixels (SDF boundary)
    # Edge color is white-ish: all channels high
    edge_mask = (rgb[:, :, 0] > 200) & (rgb[:, :, 1] > 200) & (rgb[:, :, 2] > 200)
    edge_count = edge_mask.sum()
    edge_pct = edge_count / total * 100
    print(f"  Bright edge pixels: {edge_pct:.1f}%")
    if edge_count < 50:
        print("FAIL: no bright edge pixels — SDF boundary not visible")
        return False

    print("  All validation checks passed.")
    return True


def main() -> None:
    print("Lux Shader Playground -- sdf_shapes screenshot test")
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
    print("  SUCCESS: sdf_shapes rendered and validated correctly!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
