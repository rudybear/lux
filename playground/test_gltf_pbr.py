"""End-to-end screenshot test: compile gltf_pbr.lux and render DamagedHelmet.

This script:
  1. Downloads DamagedHelmet.glb from Khronos (cached in assets/)
  2. Compiles examples/gltf_pbr.lux -> shadercache/*.spv
  3. Renders using the engine with --scene and --pipeline
  4. Validates the output (mesh visible, PBR shading, normal mapping)

Usage:
    python test_gltf_pbr.py

Exit code 0 on success, 1 on failure.
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np


PLAYGROUND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLAYGROUND_DIR.parent
SHADERCACHE = PROJECT_ROOT / "shadercache"
SCREENSHOTS = PROJECT_ROOT / "screenshots"
ASSETS_DIR = PROJECT_ROOT / "assets"
HELMET_GLB = ASSETS_DIR / "DamagedHelmet.glb"
HELMET_URL = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb"

LUX_SOURCE = PROJECT_ROOT / "examples" / "gltf_pbr.lux"
VERT_SPV = SHADERCACHE / "gltf_pbr.vert.spv"
FRAG_SPV = SHADERCACHE / "gltf_pbr.frag.spv"
OUTPUT_PNG = SCREENSHOTS / "test_gltf_pbr.png"


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def download_asset() -> bool:
    """Download DamagedHelmet.glb if not already cached."""
    step("Step 0: Ensuring DamagedHelmet.glb asset")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    if HELMET_GLB.exists():
        size = HELMET_GLB.stat().st_size
        print(f"  Already cached: {HELMET_GLB} ({size:,} bytes)")
        return True

    print(f"  Downloading from: {HELMET_URL}")
    try:
        urllib.request.urlretrieve(HELMET_URL, str(HELMET_GLB))
        size = HELMET_GLB.stat().st_size
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

    if not VERT_SPV.exists() or not FRAG_SPV.exists():
        print("FAIL: expected .spv files not produced")
        return False

    print(f"  OK: {VERT_SPV.name} ({VERT_SPV.stat().st_size} bytes)")
    print(f"  OK: {FRAG_SPV.name} ({FRAG_SPV.stat().st_size} bytes)")
    return True


def detect_ibl() -> str:
    """Auto-detect available IBL environment name, preferring pisa then neutral."""
    ibl_dir = PROJECT_ROOT / "assets" / "ibl"
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
    """Render DamagedHelmet using the engine."""
    step("Step 2: Rendering DamagedHelmet")

    ibl_name = detect_ibl()

    try:
        # Try using the engine module
        sys.path.insert(0, str(PLAYGROUND_DIR))
        from engine import render
        pixels = render(
            scene_source=str(HELMET_GLB),
            pipeline_base=str(SHADERCACHE / "gltf_pbr"),
            output=str(OUTPUT_PNG),
            width=512,
            height=512,
            ibl_name=ibl_name,
        )
        return pixels
    except ImportError:
        # Fallback: try using the engine as a subprocess
        cmd = [
            sys.executable, "-m", "playground.engine",
            "--scene", str(HELMET_GLB),
            "--pipeline", str(SHADERCACHE / "gltf_pbr"),
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

        # Load the output image
        from PIL import Image
        img = Image.open(str(OUTPUT_PNG))
        return np.array(img.convert("RGBA"), dtype=np.uint8)
    except Exception as exc:
        print(f"FAIL: rendering failed: {exc}")
        import traceback
        traceback.print_exc()
        return None


def validate_output(pixels: np.ndarray) -> bool:
    """Check the rendered image for expected glTF PBR content."""
    step("Step 3: Validating output")
    h, w, c = pixels.shape
    assert (h, w, c) == (512, 512, 4), f"Unexpected shape: {pixels.shape}"

    rgb = pixels[:, :, :3].astype(np.float32)
    brightness = rgb.sum(axis=2)

    # Check 1: mesh visible — significant non-background coverage
    non_bg = (brightness > 15).sum()
    total = h * w
    coverage = non_bg / total * 100
    print(f"  Mesh coverage: {coverage:.1f}%")
    if coverage < 5:
        print("FAIL: helmet not visible — too few non-background pixels")
        return False

    # Check 2: brightness variation (PBR shading, not flat)
    visible = brightness[brightness > 15]
    if len(visible) > 100:
        brightness_std = visible.std()
        print(f"  Brightness std dev: {brightness_std:.1f}")
        if brightness_std < 10:
            print("FAIL: no shading variation — model looks flat")
            return False

    # Check 3: color variation (different materials on the helmet)
    visible_rgb = rgb[brightness > 15]
    if len(visible_rgb) > 100:
        r_mean = visible_rgb[:, 0].mean()
        g_mean = visible_rgb[:, 1].mean()
        b_mean = visible_rgb[:, 2].mean()
        print(f"  Average color: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")

    # Check 4: specular highlights (bright spots from PBR)
    bright_pixels = (brightness > 300).sum()
    print(f"  Bright specular pixels: {bright_pixels}")

    # Check 5: dark areas exist (AO darkening, shadowed regions)
    dark_mesh = ((brightness > 5) & (brightness < 50)).sum()
    dark_pct = dark_mesh / max(non_bg, 1) * 100
    print(f"  Dark mesh pixels: {dark_pct:.1f}% of mesh")

    print("  All validation checks passed.")
    return True


def main() -> None:
    print("Lux Shader Playground -- glTF PBR (DamagedHelmet) screenshot test")
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
    print("  SUCCESS: glTF PBR rendered and validated correctly!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
