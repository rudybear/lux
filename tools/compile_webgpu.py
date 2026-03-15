#!/usr/bin/env python3
"""Compile Lux examples to WebGPU-compatible WGSL + reflection JSON.

Usage:
    python tools/compile_webgpu.py                    # compile all compatible examples
    python tools/compile_webgpu.py examples/pbr_basic.lux  # compile specific file

Outputs go to playground_web/shaders/.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "playground_web" / "public" / "shaders"

# Examples that use features incompatible with WebGPU (RT, mesh, bindless)
_SKIP_PATTERNS = {"rt_", "mesh_shader", "gltf_pbr_rt", "bindless"}


def should_skip(name: str) -> bool:
    return any(pat in name for pat in _SKIP_PATTERNS)


def compile_one(lux_path: Path) -> bool:
    from luxc.compiler import compile_source

    source = lux_path.read_text(encoding="utf-8")
    stem = lux_path.stem
    try:
        compile_source(
            source=source,
            stem=stem,
            output_dir=OUTPUT_DIR,
            source_dir=lux_path.parent,
            validate=True,
            emit_reflection=True,
            source_name=lux_path.name,
            target="wgsl",
            webgpu=True,
        )
        return True
    except Exception as e:
        print(f"  SKIP {lux_path.name}: {e}", file=sys.stderr)
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        files = [Path(a) for a in sys.argv[1:]]
    else:
        examples_dir = ROOT / "examples"
        files = sorted(examples_dir.glob("*.lux"))

    ok = 0
    fail = 0
    skip = 0
    for f in files:
        if should_skip(f.stem):
            print(f"  skip {f.name} (unsupported for WebGPU)")
            skip += 1
            continue
        print(f"Compiling {f.name} -> WGSL ...")
        if compile_one(f):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} compiled, {fail} failed, {skip} skipped -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
