#!/usr/bin/env python3
"""Validation script: compile every example × pipeline × flag combination and report pass/fail.

Usage:
    python validate_all.py          # full matrix
    python validate_all.py --quick  # fast subset (one permutation each)
"""

import subprocess
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_luxc(*args: str) -> tuple[bool, str]:
    """Run luxc with given args. Returns (success, stderr)."""
    cmd = [sys.executable, "-m", "luxc"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    success = result.returncode == 0
    return success, result.stderr.strip()


def main():
    parser = argparse.ArgumentParser(description="Validate all Lux shader compilations")
    parser.add_argument("--quick", action="store_true", help="Fast subset only")
    args = parser.parse_args()

    out = Path("/tmp/lux_validate") if sys.platform != "win32" else Path("C:/temp/lux_validate")
    out.mkdir(parents=True, exist_ok=True)
    out_str = str(out)

    # Define the compilation matrix
    tests: list[tuple[str, list[str]]] = []

    # Simple examples (all engines)
    for ex in ["hello_triangle", "pbr_surface", "sdf_shapes", "procedural_noise"]:
        tests.append((f"{ex}", [f"examples/{ex}.lux", "-o", out_str]))

    # Cartoon shader (all pipelines)
    tests.append(("cartoon_toon/Forward", ["examples/cartoon_toon.lux", "--pipeline", "CartoonForward", "-o", out_str]))
    tests.append(("cartoon_toon/RT", ["examples/cartoon_toon.lux", "--pipeline", "CartoonRT", "-o", out_str]))

    # gltf_pbr Forward + RT
    tests.append(("gltf_pbr/Forward/perms", ["examples/gltf_pbr.lux", "--pipeline", "GltfForward", "--all-permutations", "-o", out_str]))
    tests.append(("gltf_pbr/RT", ["examples/gltf_pbr.lux", "--pipeline", "GltfRT", "-o", out_str]))

    # gltf_pbr_layered: Forward
    tests.append(("layered/Forward/perms", ["examples/gltf_pbr_layered.lux", "--pipeline", "GltfForward", "--all-permutations", "-o", out_str]))
    tests.append(("layered/Forward/bindless", ["examples/gltf_pbr_layered.lux", "--pipeline", "GltfForward", "--all-permutations", "--bindless", "-o", out_str]))

    # gltf_pbr_layered: RT
    tests.append(("layered/RT/perms", ["examples/gltf_pbr_layered.lux", "--pipeline", "GltfRT", "--all-permutations", "-o", out_str]))
    tests.append(("layered/RT/bindless", ["examples/gltf_pbr_layered.lux", "--pipeline", "GltfRT", "--all-permutations", "--bindless", "-o", out_str]))

    # gltf_pbr_layered: Mesh
    tests.append(("layered/Mesh/perms", ["examples/gltf_pbr_layered.lux", "--pipeline", "GltfMesh", "--all-permutations", "-o", out_str]))
    tests.append(("layered/Mesh/bindless", ["examples/gltf_pbr_layered.lux", "--pipeline", "GltfMesh", "--all-permutations", "--bindless", "-o", out_str]))

    # rt_manual
    tests.append(("rt_manual", ["examples/rt_manual.lux", "-o", out_str]))

    if args.quick:
        # Only run one representative from each category
        tests = [t for t in tests if "perms" not in t[0] and "bindless" not in t[0]] + [
            ("layered/Forward/single", ["examples/gltf_pbr_layered.lux", "--pipeline", "GltfForward", "-o", out_str]),
            ("layered/RT/single", ["examples/gltf_pbr_layered.lux", "--pipeline", "GltfRT", "-o", out_str]),
        ]

    passed = 0
    failed = 0
    errors = []

    print(f"Running {len(tests)} validation tests...")
    print(f"Output: {out}")
    print()

    for name, luxc_args in tests:
        ok, stderr = run_luxc(*luxc_args)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if ok:
            passed += 1
        else:
            failed += 1
            errors.append((name, stderr))

    print()
    print(f"{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")

    if errors:
        print("\nFailed tests:")
        for name, stderr in errors:
            print(f"\n  {name}:")
            for line in stderr.split("\n")[:5]:
                print(f"    {line}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
