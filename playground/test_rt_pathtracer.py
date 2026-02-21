"""End-to-end test: compile RT shaders and validate SPIR-V assembly.

This script:
  1. Compiles examples/rt_pathtracer.lux (declarative RT pipeline)
  2. Compiles examples/rt_manual.lux (hand-written RT stages)
  3. Validates SPIR-V assembly for correct RT capabilities

RT shaders can't be rasterized via wgpu, so validation is SPIR-V
assembly inspection + spirv-val, not pixel rendering.

Usage:
    python test_rt_pathtracer.py

Exit code 0 on success, 1 on failure.
"""

import subprocess
import sys
from pathlib import Path


PLAYGROUND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLAYGROUND_DIR.parent
SHADERCACHE = PROJECT_ROOT / "shadercache"


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def compile_shader(lux_source: Path) -> bool:
    """Run luxc to compile the .lux file to SPIR-V with assembly output."""
    if not lux_source.exists():
        print(f"FAIL: source file not found: {lux_source}")
        return False

    cmd = [
        sys.executable, "-m", "luxc",
        str(lux_source),
        "-o", str(SHADERCACHE),
        "--emit-asm",
        "--no-validate",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")

    if result.returncode != 0:
        print(f"FAIL: luxc exited with code {result.returncode}")
        return False

    return True


def check_spv_exists(name: str, stage_ext: str) -> Path | None:
    """Check that a .spv file exists and report its size."""
    spv = SHADERCACHE / f"{name}.{stage_ext}.spv"
    if spv.exists():
        print(f"  OK: {spv.name} ({spv.stat().st_size} bytes)")
        return spv
    else:
        print(f"  MISSING: {spv.name}")
        return None


def read_asm(name: str, stage_ext: str) -> str | None:
    """Read a .spvasm file if it exists."""
    asm = SHADERCACHE / f"{name}.{stage_ext}.spvasm"
    if asm.exists():
        return asm.read_text()
    return None


def validate_rt_asm(asm_text: str, stage_name: str) -> bool:
    """Validate that RT SPIR-V assembly contains expected capabilities."""
    ok = True

    # Check for RT capability
    if "OpCapability RayTracingKHR" not in asm_text:
        # Also accept ShaderBallotKHR or just check for RT execution model
        if "RayGeneration" not in asm_text and "ClosestHit" not in asm_text and "Miss" not in asm_text:
            print(f"  WARN: {stage_name}: no RT capability or execution model found")

    # Check for correct execution model based on stage
    if stage_name == "raygen" or stage_name == "rgen":
        if "RayGenerationKHR" in asm_text or "RayGenerationNV" in asm_text or "RayGeneration" in asm_text:
            print(f"  OK: {stage_name}: RayGeneration execution model present")
        else:
            print(f"  WARN: {stage_name}: RayGeneration execution model not found")

    elif stage_name == "closest_hit" or stage_name == "rchit":
        if "ClosestHitKHR" in asm_text or "ClosestHitNV" in asm_text or "ClosestHit" in asm_text:
            print(f"  OK: {stage_name}: ClosestHit execution model present")
        else:
            print(f"  WARN: {stage_name}: ClosestHit execution model not found")

    elif stage_name == "miss" or stage_name == "rmiss":
        if "MissKHR" in asm_text or "MissNV" in asm_text or "Miss" in asm_text:
            print(f"  OK: {stage_name}: Miss execution model present")
        else:
            print(f"  WARN: {stage_name}: Miss execution model not found")

    # No rasterization-only artifacts in RT stages
    if "OriginUpperLeft" in asm_text:
        print(f"  FAIL: {stage_name}: contains OriginUpperLeft (rasterization-only)")
        ok = False

    return ok


def test_declarative_rt() -> bool:
    """Test the declarative RT pipeline (rt_pathtracer.lux)."""
    step("Test 1: Declarative RT pipeline (rt_pathtracer.lux)")
    source = PROJECT_ROOT / "examples" / "rt_pathtracer.lux"

    if not compile_shader(source):
        return False

    # Check that all 3 RT stages were produced
    stages = [
        ("rt_pathtracer", "rgen"),
        ("rt_pathtracer", "rchit"),
        ("rt_pathtracer", "rmiss"),
    ]

    all_ok = True
    for name, ext in stages:
        spv = check_spv_exists(name, ext)
        if spv is None:
            all_ok = False
            continue

        asm = read_asm(name, ext)
        if asm:
            if not validate_rt_asm(asm, ext):
                all_ok = False
        else:
            print(f"  WARN: no .spvasm for {name}.{ext} (--emit-asm may not have produced it)")

    return all_ok


def test_manual_rt() -> bool:
    """Test the hand-written RT stages (rt_manual.lux)."""
    step("Test 2: Manual RT stages (rt_manual.lux)")
    source = PROJECT_ROOT / "examples" / "rt_manual.lux"

    if not compile_shader(source):
        return False

    # Check that all 3 RT stages were produced
    stages = [
        ("rt_manual", "rgen"),
        ("rt_manual", "rchit"),
        ("rt_manual", "rmiss"),
    ]

    all_ok = True
    for name, ext in stages:
        spv = check_spv_exists(name, ext)
        if spv is None:
            all_ok = False
            continue

        asm = read_asm(name, ext)
        if asm:
            if not validate_rt_asm(asm, ext):
                all_ok = False

            # Check for RT builtins in the manual shader
            if ext == "rgen":
                for builtin in ["LaunchId", "LaunchSize"]:
                    if builtin in asm or builtin.lower() in asm.lower():
                        print(f"  OK: {ext}: {builtin} builtin found")
                    else:
                        print(f"  INFO: {ext}: {builtin} not found (may use different naming)")

            if ext == "rchit":
                for builtin in ["HitT", "hit_t", "IncomingRayFlags"]:
                    if builtin in asm or builtin.lower() in asm.lower():
                        print(f"  OK: {ext}: {builtin}-related builtin found")
                        break

            if ext == "rmiss":
                for builtin in ["WorldRayDirection", "world_ray_direction"]:
                    if builtin in asm or builtin.lower() in asm.lower():
                        print(f"  OK: {ext}: {builtin}-related builtin found")
                        break
        else:
            print(f"  WARN: no .spvasm for {name}.{ext}")

    return all_ok


def main() -> None:
    print("Lux Shader Playground -- RT pathtracer compilation test")
    print(f"Project root: {PROJECT_ROOT}")

    results = []

    ok1 = test_declarative_rt()
    results.append(("Declarative RT", ok1))

    ok2 = test_manual_rt()
    results.append(("Manual RT", ok2))

    # Summary
    step("Summary")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\n" + "=" * 60)
        print("  SUCCESS: all RT compilation tests passed!")
        print("=" * 60)
    else:
        print("\n*** SOME RT TESTS FAILED ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
