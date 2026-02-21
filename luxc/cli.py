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
        "--pipeline", type=str, metavar="NAME",
        help="Compile only the named pipeline (e.g., --pipeline GltfForward)",
    )
    parser.add_argument(
        "--features", type=str, metavar="FEAT,FEAT,...",
        help="Enable compile-time features (comma-separated)",
    )
    parser.add_argument(
        "--all-permutations", action="store_true",
        help="Compile all 2^N feature permutations",
    )
    parser.add_argument(
        "--list-features", action="store_true",
        help="List available features and exit",
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

    # --- List features mode ---
    if args.list_features:
        from luxc.parser.tree_builder import parse_lux
        from luxc.features.evaluator import collect_feature_names
        module = parse_lux(source)
        names = collect_feature_names(module)
        if names:
            print("Available features:")
            for name in names:
                print(f"  {name}")
        else:
            print("No features declared in this file.")
        return

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

    # Parse feature flags
    feature_set = None
    if args.features:
        feature_set = set(f.strip() for f in args.features.split(",") if f.strip())

    if args.all_permutations:
        # Compile all 2^N feature permutations
        from luxc.parser.tree_builder import parse_lux
        from luxc.features.evaluator import collect_feature_names
        import itertools, json

        module = parse_lux(source)
        all_names = collect_feature_names(module)
        if not all_names:
            print("No features declared — compiling normally.", file=sys.stderr)
            try:
                compile_source(
                    source=source, stem=stem, output_dir=output_dir,
                    source_dir=input_path.parent, dump_ast=args.dump_ast,
                    emit_asm=args.emit_asm, validate=not args.no_validate,
                    emit_reflection=not args.no_reflection, debug=args.debug,
                    source_name=input_path.name, pipeline=args.pipeline,
                )
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            return

        permutations = []
        for r in range(len(all_names) + 1):
            for combo in itertools.combinations(all_names, r):
                permutations.append(set(combo))

        print(f"Compiling {len(permutations)} permutations for features: {', '.join(all_names)}")
        manifest_perms = []
        for perm in permutations:
            suffix = "+" + "+".join(
                n.removeprefix("has_") for n in sorted(perm)
            ) if perm else ""
            manifest_perms.append({
                "suffix": suffix,
                "features": {n: n in perm for n in all_names},
            })
            try:
                compile_source(
                    source=source, stem=stem, output_dir=output_dir,
                    source_dir=input_path.parent, dump_ast=args.dump_ast,
                    emit_asm=args.emit_asm, validate=not args.no_validate,
                    emit_reflection=not args.no_reflection, debug=args.debug,
                    source_name=input_path.name, pipeline=args.pipeline,
                    features=perm,
                )
            except Exception as e:
                print(f"Error (features={sorted(perm)}): {e}", file=sys.stderr)

        # Write manifest
        manifest = {
            "pipeline": args.pipeline or "*",
            "features": all_names,
            "permutations": manifest_perms,
        }
        manifest_path = output_dir / f"{stem}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {manifest_path}")
        return

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
            pipeline=args.pipeline,
            features=feature_set,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
