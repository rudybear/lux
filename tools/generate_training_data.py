#!/usr/bin/env python3
"""Generate training data by transpiling GLSL shaders to Lux.

Usage:
    python tools/generate_training_data.py \\
        --input-dir ./glsl_shaders \\
        --output data/training.jsonl \\
        [--describe] [--verify]
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


def collect_glsl_files(input_dir: Path) -> list[Path]:
    """Collect all GLSL shader files from a directory tree."""
    extensions = {".glsl", ".frag", ".vert", ".fs", ".vs"}
    files = []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*{ext}"))
    return sorted(set(files))


def transpile_file(path: Path) -> dict | None:
    """Transpile a single GLSL file to Lux. Returns a record dict or None on error."""
    from luxc.transpiler.glsl_to_lux import transpile_glsl_to_lux

    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        result = transpile_glsl_to_lux(source)
        return {
            "glsl_source": source,
            "lux": result.lux_source,
            "source_file": str(path),
            "warnings": result.warnings,
        }
    except Exception as e:
        return None


def verify_lux(lux_source: str) -> bool:
    """Check if transpiled Lux parses and type-checks."""
    try:
        from luxc.parser.tree_builder import parse_lux
        from luxc.analysis.type_checker import type_check
        from luxc.builtins.types import clear_type_aliases

        clear_type_aliases()
        module = parse_lux(lux_source)
        type_check(module)
        return True
    except Exception:
        return False


def describe_shader(glsl_source: str) -> str:
    """Use Claude API to generate a one-sentence description of the shader."""
    try:
        import anthropic
    except ImportError:
        return ""

    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    "Describe this GLSL shader in one sentence "
                    "(what visual effect it produces):\n\n"
                    + glsl_source[:2000]
                ),
            }],
        )
        return msg.content[0].text.strip()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Generate Lux training data from GLSL shaders"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing GLSL shader files",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/training.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify transpiled Lux parses and type-checks",
    )
    parser.add_argument(
        "--describe", action="store_true",
        help="Generate descriptions using Claude API",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    files = collect_glsl_files(args.input_dir)
    print(f"Found {len(files)} GLSL files in {args.input_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    errors = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for path in files:
            record = transpile_file(path)
            if record is None:
                errors += 1
                print(f"  ERROR: {path}")
                continue

            if args.verify and not verify_lux(record["lux"]):
                skipped += 1
                print(f"  SKIP (verify failed): {path}")
                continue

            if args.describe:
                record["description"] = describe_shader(record["glsl_source"])

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            success += 1
            print(f"  OK: {path}")

    print(f"\nDone: {success} success, {skipped} skipped, {errors} errors")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
