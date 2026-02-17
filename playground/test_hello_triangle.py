"""End-to-end test: compile hello_triangle.lux and render to PNG.

This script:
  1. Compiles examples/hello_triangle.lux -> playground/*.spv
  2. Loads the SPIR-V files via the render harness
  3. Renders a 512x512 frame and saves playground/hello_triangle.png
  4. Validates the output (non-black pixels, expected colors at vertices)

Usage:
    python test_hello_triangle.py

Exit code 0 on success, 1 on failure.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np


# Resolve paths relative to this script's location
PLAYGROUND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLAYGROUND_DIR.parent
LUX_SOURCE = PROJECT_ROOT / "examples" / "hello_triangle.lux"
VERT_SPV = PLAYGROUND_DIR / "hello_triangle.vert.spv"
FRAG_SPV = PLAYGROUND_DIR / "hello_triangle.frag.spv"
OUTPUT_PNG = PLAYGROUND_DIR / "hello_triangle.png"


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def compile_shader() -> bool:
    """Run luxc to compile the .lux file to SPIR-V."""
    step("Step 1: Compiling hello_triangle.lux")

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
        print(f"FAIL: expected .spv files not produced")
        return False

    print(f"  OK: {VERT_SPV.name} ({VERT_SPV.stat().st_size} bytes)")
    print(f"  OK: {FRAG_SPV.name} ({FRAG_SPV.stat().st_size} bytes)")
    return True


def render_frame() -> np.ndarray | None:
    """Load shaders and render one frame, returning pixel data."""
    step("Step 2: Rendering frame")

    # Import locally so compilation step doesn't need wgpu
    from render_harness import render_triangle, save_png

    try:
        pixels = render_triangle(VERT_SPV, FRAG_SPV, width=512, height=512)
    except Exception as exc:
        print(f"FAIL: rendering failed: {exc}")
        return None

    save_png(pixels, OUTPUT_PNG)
    print(f"  OK: saved {OUTPUT_PNG} ({OUTPUT_PNG.stat().st_size} bytes)")
    return pixels


def validate_output(pixels: np.ndarray) -> bool:
    """Check the rendered image for expected triangle content."""
    step("Step 3: Validating output")
    h, w, c = pixels.shape
    assert (h, w, c) == (512, 512, 4), f"Unexpected shape: {pixels.shape}"

    # Check 1: there must be some non-black pixels (the triangle)
    rgb = pixels[:, :, :3].astype(np.float32)
    brightness = rgb.sum(axis=2)
    non_black = (brightness > 0).sum()
    total = h * w
    coverage = non_black / total * 100

    print(f"  Triangle coverage: {coverage:.1f}% ({non_black}/{total} pixels)")
    if non_black < 1000:
        print("FAIL: too few non-black pixels -- triangle not rendered?")
        return False

    # Check 2: top-center pixel should be reddish (top vertex is red)
    top_pixel = pixels[int(h * 0.27), w // 2, :3]  # slightly below top vertex
    print(f"  Top-center pixel (near red vertex): RGB = {tuple(top_pixel)}")
    if top_pixel[0] < 100:
        print("FAIL: top pixel should be predominantly red")
        return False

    # Check 3: bottom-left should be greenish
    bl_pixel = pixels[int(h * 0.73), int(w * 0.3), :3]
    print(f"  Bottom-left pixel (near green vertex): RGB = {tuple(bl_pixel)}")
    if bl_pixel[1] < 50:
        print("FAIL: bottom-left pixel should have significant green")
        return False

    # Check 4: bottom-right should be bluish
    br_pixel = pixels[int(h * 0.73), int(w * 0.7), :3]
    print(f"  Bottom-right pixel (near blue vertex): RGB = {tuple(br_pixel)}")
    if br_pixel[2] < 50:
        print("FAIL: bottom-right pixel should have significant blue")
        return False

    # Check 5: corners should be black (background)
    corner = pixels[0, 0, :3]
    print(f"  Corner pixel (background): RGB = {tuple(corner)}")
    if corner.sum() != 0:
        print("FAIL: corner pixel should be black (background)")
        return False

    print("  All validation checks passed.")
    return True


def main() -> None:
    print("Lux Shader Playground -- hello_triangle end-to-end test")
    print(f"Project root: {PROJECT_ROOT}")

    ok = True

    # Step 1: Compile
    if not compile_shader():
        print("\n*** FAILED at compilation step ***")
        sys.exit(1)

    # Step 2: Render
    pixels = render_frame()
    if pixels is None:
        print("\n*** FAILED at rendering step ***")
        sys.exit(1)

    # Step 3: Validate
    if not validate_output(pixels):
        print("\n*** FAILED at validation step ***")
        sys.exit(1)

    # Success
    print("\n" + "=" * 60)
    print("  SUCCESS: hello_triangle rendered and validated correctly!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
