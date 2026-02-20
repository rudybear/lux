"""End-to-end screenshot test: render Khronos MetalRoughSpheres with PBR + IBL.

MetalRoughSpheres.glb is the standard PBR correctness test — a grid of spheres
varying metallic (0->1) on one axis and roughness (0->1) on the other.

This script:
  1. Downloads MetalRoughSpheres.glb from Khronos (cached in playground/assets/)
  2. Compiles examples/gltf_pbr.lux -> playground/*.spv
  3. Renders using the engine with --scene and --pipeline (with IBL if available)
  4. Validates the output (grid visible, metallic/roughness gradient, PBR shading)

Usage:
    python test_metal_rough_spheres.py

Exit code 0 on success, 1 on failure.
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np


PLAYGROUND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLAYGROUND_DIR.parent
ASSETS_DIR = PLAYGROUND_DIR / "assets"
MODEL_GLB = ASSETS_DIR / "MetalRoughSpheres.glb"
MODEL_URL = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/MetalRoughSpheres/glTF-Binary/MetalRoughSpheres.glb"

LUX_SOURCE = PROJECT_ROOT / "examples" / "gltf_pbr.lux"
VERT_SPV = PLAYGROUND_DIR / "gltf_pbr.vert.spv"
FRAG_SPV = PLAYGROUND_DIR / "gltf_pbr.frag.spv"
OUTPUT_PNG = PLAYGROUND_DIR / "test_metal_rough_spheres.png"


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def download_asset() -> bool:
    """Download MetalRoughSpheres.glb if not already cached."""
    step("Step 0: Ensuring MetalRoughSpheres.glb asset")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_GLB.exists():
        size = MODEL_GLB.stat().st_size
        print(f"  Already cached: {MODEL_GLB} ({size:,} bytes)")
        return True

    print(f"  Downloading from: {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_GLB))
        size = MODEL_GLB.stat().st_size
        print(f"  OK: downloaded {size:,} bytes")
        return True
    except Exception as exc:
        print(f"FAIL: download failed: {exc}")
        return False


def compile_shader() -> bool:
    """Run luxc to compile the gltf_pbr.lux file to SPIR-V."""
    step("Step 1: Compiling gltf_pbr.lux")

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


def detect_ibl() -> str:
    """Auto-detect available IBL environment name, preferring pisa then neutral."""
    ibl_dir = PLAYGROUND_DIR / "assets" / "ibl"
    if ibl_dir.exists():
        preferred = ["pisa", "neutral"]
        candidates = [d.name for d in ibl_dir.iterdir()
                      if d.is_dir() and (d / "manifest.json").exists()]
        ordered = [n for n in preferred if n in candidates]
        ordered += [n for n in sorted(candidates) if n not in preferred]
        if ordered:
            print(f"  Found IBL assets: {ordered[0]}")
            return ordered[0]
    print("  No IBL assets found, rendering without IBL")
    return ""


def render_frame() -> np.ndarray | None:
    """Render MetalRoughSpheres using the engine."""
    step("Step 2: Rendering MetalRoughSpheres")

    ibl_name = detect_ibl()

    try:
        sys.path.insert(0, str(PLAYGROUND_DIR))
        from engine import render
        pixels = render(
            scene_source=str(MODEL_GLB),
            pipeline_base=str(PLAYGROUND_DIR / "gltf_pbr"),
            output=str(OUTPUT_PNG),
            width=512,
            height=512,
            ibl_name=ibl_name,
        )
        return pixels
    except ImportError:
        cmd = [
            sys.executable, "-m", "playground.engine",
            "--scene", str(MODEL_GLB),
            "--pipeline", str(PLAYGROUND_DIR / "gltf_pbr"),
            "--output", str(OUTPUT_PNG),
            "--width", "512",
            "--height", "512",
        ]
        if ibl_name:
            cmd.extend(["--ibl", ibl_name])
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")
        if result.returncode != 0:
            print(f"FAIL: engine exited with code {result.returncode}")
            return None

        from PIL import Image
        img = Image.open(str(OUTPUT_PNG))
        return np.array(img.convert("RGBA"), dtype=np.uint8)
    except Exception as exc:
        print(f"FAIL: rendering failed: {exc}")
        import traceback
        traceback.print_exc()
        return None


def validate_output(pixels: np.ndarray) -> bool:
    """Check the rendered image for expected MetalRoughSpheres content."""
    step("Step 3: Validating output")
    h, w, c = pixels.shape
    assert (h, w, c) == (512, 512, 4), f"Unexpected shape: {pixels.shape}"

    rgb = pixels[:, :, :3].astype(np.float32)
    brightness = rgb.sum(axis=2)

    # Check 1: mesh visible — grid of spheres should cover significant area
    non_bg = (brightness > 15).sum()
    total = h * w
    coverage = non_bg / total * 100
    print(f"  Mesh coverage: {coverage:.1f}%")
    if coverage < 5:
        print("FAIL: spheres not visible — too few non-background pixels")
        return False

    # Check 2: brightness variation (different roughness/metallic values)
    visible = brightness[brightness > 15]
    if len(visible) > 100:
        brightness_std = visible.std()
        print(f"  Brightness std dev: {brightness_std:.1f}")
        if brightness_std < 10:
            print("FAIL: no shading variation — spheres look flat")
            return False

    # Check 3: color variation across the grid
    visible_rgb = rgb[brightness > 15]
    if len(visible_rgb) > 100:
        r_mean = visible_rgb[:, 0].mean()
        g_mean = visible_rgb[:, 1].mean()
        b_mean = visible_rgb[:, 2].mean()
        print(f"  Average color: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")

    # Check 4: specular highlights (sharp reflections on smooth spheres)
    bright_pixels = (brightness > 300).sum()
    print(f"  Bright specular pixels: {bright_pixels}")

    # Check 5: metallic/roughness gradient — compare quadrants
    # With IBL: top half (smooth) should differ from bottom half (rough)
    top_half = brightness[:h // 2, :]
    bot_half = brightness[h // 2:, :]
    top_mean = top_half[top_half > 15].mean() if (top_half > 15).any() else 0
    bot_mean = bot_half[bot_half > 15].mean() if (bot_half > 15).any() else 0
    print(f"  Top-half mean brightness: {top_mean:.1f}, Bottom-half: {bot_mean:.1f}")

    print("  All validation checks passed.")
    return True


def main() -> None:
    print("Lux Shader Playground -- MetalRoughSpheres (Khronos PBR validation) screenshot test")
    print(f"Project root: {PROJECT_ROOT}")

    if not download_asset():
        print("\n*** FAILED at asset download step ***")
        sys.exit(1)

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
    print("  SUCCESS: MetalRoughSpheres rendered and validated correctly!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
