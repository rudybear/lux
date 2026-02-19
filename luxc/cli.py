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
        "--transpile", action="store_true",
        help="Transpile GLSL input to Lux",
    )
    parser.add_argument(
        "--ai", type=str, metavar="DESCRIPTION",
        help='Generate shader from description (e.g., --ai "frosted glass")',
    )
    parser.add_argument(
        "--ai-model", type=str, default="claude-sonnet-4-20250514",
        help="Model to use for AI generation",
    )
    parser.add_argument(
        "--ai-no-verify", action="store_true",
        help="Skip compilation verification of AI-generated code",
    )
    parser.add_argument(
        "--no-reflection",
        action="store_true",
        help="Skip .lux.json reflection metadata emission",
    )
    parser.add_argument(
        "-g", "--debug",
        action="store_true",
        help="Emit OpLine/OpSource debug info in SPIR-V",
    )
    parser.add_argument(
        "--version", action="version", version="luxc 0.1.0"
    )

    args = parser.parse_args(argv)

    # --- AI generation mode ---
    if args.ai:
        from luxc.ai.generate import generate_lux_shader
        try:
            result = generate_lux_shader(
                args.ai,
                verify=not args.ai_no_verify,
                model=args.ai_model,
            )
            if args.input:
                output_path = Path(args.input)
            else:
                output_path = Path("generated.lux")
            if args.output_dir:
                output_path = args.output_dir / output_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.lux_source, encoding="utf-8")
            print(f"Generated: {output_path}")
            if result.errors:
                for err in result.errors:
                    print(f"  Warning: {err}", file=sys.stderr)
            if not result.compilation_success and not args.ai_no_verify:
                print("  Note: generated code did not pass compilation check", file=sys.stderr)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

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

    # --- Transpile mode ---
    if args.transpile:
        from luxc.transpiler.glsl_to_lux import transpile_glsl_to_lux
        try:
            result = transpile_glsl_to_lux(source)
            lux_path = output_dir / f"{stem}.lux"
            lux_path.parent.mkdir(parents=True, exist_ok=True)
            lux_path.write_text(result.lux_source, encoding="utf-8")
            print(f"Transpiled: {lux_path}")
            for w in result.warnings:
                print(f"  Warning: {w}", file=sys.stderr)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- Normal compilation ---
    from luxc.compiler import compile_source

    try:
        compile_source(
            source=source,
            stem=stem,
            output_dir=output_dir,
            source_dir=input_path.parent,
            dump_ast=args.dump_ast,
            emit_asm=args.emit_asm,
            validate=not args.no_validate,
            emit_reflection=not args.no_reflection,
            debug=args.debug,
            source_name=input_path.name,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
