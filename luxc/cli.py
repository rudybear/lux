"""Command-line interface for the Lux compiler."""

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="luxc",
        description="Lux shader compiler — compiles .lux files to SPIR-V",
    )
    parser.add_argument("input", nargs="?", help="Input .lux file")
    parser.add_argument(
        "--dump-ast", action="store_true", help="Dump the AST and exit"
    )
    parser.add_argument(
        "--emit-asm", action="store_true", help="Write .spvasm text files"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip spirv-val validation",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Output directory (default: same as input file)",
    )
    parser.add_argument(
        "--version", action="version", version="luxc 0.1.0"
    )

    args = parser.parse_args(argv)

    if args.input is None:
        parser.print_help()
        sys.exit(0)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    source = input_path.read_text(encoding="utf-8")
    stem = input_path.stem
    output_dir = args.output_dir or input_path.parent

    from luxc.compiler import compile_source

    try:
        compile_source(
            source=source,
            stem=stem,
            output_dir=output_dir,
            dump_ast=args.dump_ast,
            emit_asm=args.emit_asm,
            validate=not args.no_validate,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
