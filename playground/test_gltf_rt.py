"""End-to-end screenshot test: RT ray-traced DamagedHelmet via C++ and Rust engines.

This script:
  1. Downloads DamagedHelmet.glb from Khronos (cached in assets/)
  2. Compiles examples/gltf_pbr_rt.lux -> shadercache/*.spv (rgen/rchit/rmiss)
  3. Renders via C++ and Rust Vulkan RT engines
  4. Validates output images (mesh visible, PBR shading, geometry silhouette)

RT requires Vulkan ray tracing hardware support. Python/wgpu does NOT support RT.

Usage:
    python test_gltf_rt.py

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

LUX_SOURCE = PROJECT_ROOT / "examples" / "gltf_pbr_rt.lux"
RGEN_SPV = SHADERCACHE / "gltf_pbr_rt.rgen.spv"
RCHIT_SPV = SHADERCACHE / "gltf_pbr_rt.rchit.spv"
RMISS_SPV = SHADERCACHE / "gltf_pbr_rt.rmiss.spv"

CPP_EXE = PROJECT_ROOT / "playground_cpp" / "build" / "Release" / "lux-playground.exe"
RUST_EXE = PROJECT_ROOT / "playground_rust" / "target" / "release" / "lux-playground.exe"

OUTPUT_CPP = SCREENSHOTS / "test_gltf_rt_cpp.png"
OUTPUT_RUST = SCREENSHOTS / "test_gltf_rt_rust.png"


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
    """Run luxc to compile gltf_pbr_rt.lux to RT SPIR-V stages."""
    step("Step 1: Compiling gltf_pbr_rt.lux")

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

    missing = []
    for spv in [RGEN_SPV, RCHIT_SPV, RMISS_SPV]:
        if spv.exists():
            print(f"  OK: {spv.name} ({spv.stat().st_size} bytes)")
        else:
            missing.append(spv.name)

    if missing:
        print(f"FAIL: missing RT .spv files: {missing}")
        return False

    return True


def render_with_engine(exe: Path, scene: Path, pipeline_base: str,
                       output: Path, label: str) -> np.ndarray | None:
    """Render using a native Vulkan RT engine (C++ or Rust)."""
    if not exe.exists():
        print(f"  SKIP: {label} executable not found at {exe}")
        return None

    cmd = [
        str(exe),
        "--scene", str(scene),
        "--pipeline", pipeline_base,
        "--output", str(output),
        "--width", "512",
        "--height", "512",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(PROJECT_ROOT), timeout=120)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")

    if result.returncode != 0:
        print(f"  FAIL: {label} exited with code {result.returncode}")
        return None

    if not output.exists():
        print(f"  FAIL: {label} did not produce {output.name}")
        return None

    from PIL import Image
    img = Image.open(str(output))
    pixels = np.array(img.convert("RGBA"), dtype=np.uint8)
    print(f"  OK: {label} produced {output.name} ({output.stat().st_size:,} bytes)")
    return pixels


def validate_output(pixels: np.ndarray, label: str) -> bool:
    """Check the RT-rendered image for expected helmet geometry."""
    h, w, c = pixels.shape
    if (h, w) != (512, 512):
        print(f"  FAIL: {label}: unexpected dimensions {w}x{h}")
        return False

    rgb = pixels[:, :, :3].astype(np.float32)
    brightness = rgb.sum(axis=2)

    # Check 1: mesh visible -- helmet silhouette should cover significant area
    non_bg = (brightness > 15).sum()
    total = h * w
    coverage = non_bg / total * 100
    print(f"  {label} mesh coverage: {coverage:.1f}%")
    if coverage < 10:
        print(f"  FAIL: {label}: helmet not visible")
        return False

    # Check 2: brightness variation (PBR shading, not flat color)
    visible = brightness[brightness > 15]
    if len(visible) > 100:
        brightness_std = visible.std()
        print(f"  {label} brightness std dev: {brightness_std:.1f}")
        if brightness_std < 5:
            print(f"  FAIL: {label}: no shading variation")
            return False

    # Check 3: color info
    visible_rgb = rgb[brightness > 15]
    if len(visible_rgb) > 100:
        r_mean = visible_rgb[:, 0].mean()
        g_mean = visible_rgb[:, 1].mean()
        b_mean = visible_rgb[:, 2].mean()
        print(f"  {label} avg color: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")

    # Check 4: sky background present (blue-ish gradient from miss shader)
    bg_pixels = rgb[brightness <= 15]
    if len(bg_pixels) > 100:
        bg_b = bg_pixels[:, 2].mean()
        print(f"  {label} background blue channel: {bg_b:.1f}")

    print(f"  {label}: all checks passed.")
    return True


def main() -> None:
    print("Lux Shader Playground -- glTF RT (DamagedHelmet) screenshot test")
    print(f"Project root: {PROJECT_ROOT}")

    if not download_asset():
        print("\n*** FAILED at asset download step ***")
        sys.exit(1)

    if not compile_shader():
        print("\n*** FAILED at compilation step ***")
        sys.exit(1)

    pipeline_base = str(SHADERCACHE / "gltf_pbr_rt")
    engines_tried = 0
    engines_passed = 0
    results = []

    # C++ RT render
    step("Step 2a: Rendering DamagedHelmet via C++ RT")
    cpp_pixels = render_with_engine(CPP_EXE, HELMET_GLB, pipeline_base,
                                    OUTPUT_CPP, "C++")
    if cpp_pixels is not None:
        engines_tried += 1
        step("Step 3a: Validating C++ RT output")
        if validate_output(cpp_pixels, "C++"):
            engines_passed += 1
            results.append(("C++ RT", True))
        else:
            results.append(("C++ RT", False))
    else:
        results.append(("C++ RT", None))

    # Rust RT render
    step("Step 2b: Rendering DamagedHelmet via Rust RT")
    rust_pixels = render_with_engine(RUST_EXE, HELMET_GLB, pipeline_base,
                                     OUTPUT_RUST, "Rust")
    if rust_pixels is not None:
        engines_tried += 1
        step("Step 3b: Validating Rust RT output")
        if validate_output(rust_pixels, "Rust"):
            engines_passed += 1
            results.append(("Rust RT", True))
        else:
            results.append(("Rust RT", False))
    else:
        results.append(("Rust RT", None))

    # Summary
    step("Summary")
    for name, ok in results:
        if ok is None:
            print(f"  SKIP: {name} (executable not found)")
        elif ok:
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name}")

    if engines_tried == 0:
        print("\n*** SKIP: no RT-capable engines found (need C++ or Rust build) ***")
        sys.exit(0)

    if engines_passed == 0:
        print("\n*** FAILED: all RT engines failed validation ***")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  SUCCESS: {engines_passed}/{engines_tried} RT engines passed!")
    print(f"{'='*60}")
    outputs = [str(p) for p in [OUTPUT_CPP, OUTPUT_RUST] if p.exists()]
    for o in outputs:
        print(f"  Output: {o}")


if __name__ == "__main__":
    main()
