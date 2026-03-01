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
        "--ai-setup", action="store_true",
        help="Run interactive AI provider setup wizard",
    )
    parser.add_argument(
        "--ai-provider", type=str, metavar="NAME",
        help="Override AI provider (anthropic, openai, gemini, ollama, lm-studio)",
    )
    parser.add_argument(
        "--ai-base-url", type=str, metavar="URL",
        help="Override AI provider base URL (for OpenAI-compatible endpoints)",
    )
    parser.add_argument(
        "--ai-model", type=str, default=None,
        help="Model to use for AI generation (default: from config)",
    )
    parser.add_argument(
        "--ai-no-verify", action="store_true",
        help="Skip compilation verification of AI-generated code",
    )
    parser.add_argument(
        "--ai-retries", type=int, default=2, metavar="N",
        help="Max retry attempts for AI generation (default: 2)",
    )
    parser.add_argument(
        "--ai-from-image", type=str, metavar="IMAGE_PATH",
        help="Generate surface material from an image (e.g., --ai-from-image photo.jpg)",
    )
    parser.add_argument(
        "--ai-skills", type=str, metavar="SKILL,...",
        help="Load specific skills into the AI prompt (comma-separated)",
    )
    parser.add_argument(
        "--ai-list-skills", action="store_true",
        help="List available AI skills and exit",
    )
    parser.add_argument(
        "--ai-critique", type=str, metavar="FILE",
        help="AI review of a .lux file (e.g., --ai-critique material.lux)",
    )
    parser.add_argument(
        "--ai-modify", type=str, metavar="INSTRUCTION",
        help='Modify existing material (e.g., --ai-modify "add weathering")',
    )
    parser.add_argument(
        "--ai-batch", type=str, metavar="DESCRIPTION",
        help='Generate batch of materials (e.g., --ai-batch "medieval tavern")',
    )
    parser.add_argument(
        "--ai-batch-count", type=int, default=None, metavar="N",
        help="Number of materials in batch (default: AI decides)",
    )
    parser.add_argument(
        "--ai-from-video", type=str, metavar="VIDEO",
        help="Generate animated shader from video (e.g., --ai-from-video fire.mp4)",
    )
    parser.add_argument(
        "--ai-match-reference", type=str, metavar="IMAGE",
        help="Iteratively match a reference image",
    )
    parser.add_argument(
        "--ai-match-iterations", type=int, default=5, metavar="N",
        help="Max refinement iterations for reference matching (default: 5)",
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
        "--define", type=str, action="append", metavar="KEY=VALUE",
        help="Define compile-time integer constant (e.g., --define max_vertices=64)",
    )
    parser.add_argument(
        "--bindless", action="store_true",
        help="Emit bindless descriptor uber-shaders (requires VK_EXT_descriptor_indexing)",
    )
    parser.add_argument(
        "--warn-nan", action="store_true", default=False,
        help="Enable static analysis warnings for potential NaN/Inf operations",
    )
    parser.add_argument(
        "--version", action="version", version="luxc 0.1.0"
    )

    args = parser.parse_args(argv)

    # --- AI setup wizard ---
    if args.ai_setup:
        from luxc.ai.setup import run_setup
        run_setup()
        return

    # --- AI list skills ---
    if args.ai_list_skills:
        from luxc.ai.skills import discover_skills
        skills = discover_skills()
        if skills:
            print("Available AI skills:")
            for name, skill in sorted(skills.items()):
                print(f"  {name:<25} {skill.title}")
        else:
            print("No skills found.")
        return

    # --- AI critique mode ---
    if args.ai_critique:
        from luxc.ai.generate import critique_lux_file
        critique_path = Path(args.ai_critique)
        if not critique_path.exists():
            print(f"Error: file not found: {critique_path}", file=sys.stderr)
            sys.exit(1)
        try:
            source = critique_path.read_text(encoding="utf-8")
            result = critique_lux_file(
                source,
                model=args.ai_model,
                provider=args.ai_provider,
                base_url=args.ai_base_url,
            )
            if result.issues:
                for issue in result.issues:
                    icon = {"error": "E", "warning": "W", "info": "I"}.get(issue.severity, "?")
                    loc = f":{issue.line}" if issue.line else ""
                    print(f"  [{icon}] {issue.category}{loc}: {issue.message}")
                    if issue.suggestion:
                        print(f"      -> {issue.suggestion}")
                print()
            if result.summary:
                print(f"Summary: {result.summary}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- AI modify mode ---
    if args.ai_modify:
        from luxc.ai.generate import modify_material
        if not args.input:
            print("Error: --ai-modify requires an input file", file=sys.stderr)
            sys.exit(1)
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        try:
            source = input_path.read_text(encoding="utf-8")
            result = modify_material(
                source,
                args.ai_modify,
                verify=not args.ai_no_verify,
                model=args.ai_model,
                max_retries=args.ai_retries,
                provider=args.ai_provider,
                base_url=args.ai_base_url,
            )
            output_path = input_path
            if args.output_dir:
                output_path = args.output_dir / input_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.lux_source, encoding="utf-8")
            print(f"Modified: {output_path}")
            if result.attempts > 1:
                print(f"  (took {result.attempts} attempts)", file=sys.stderr)
            if not result.compilation_success and not args.ai_no_verify:
                print("  Note: modified code did not pass compilation check", file=sys.stderr)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- AI batch generation mode ---
    if args.ai_batch:
        from luxc.ai.generate import generate_material_batch
        try:
            result = generate_material_batch(
                args.ai_batch,
                count=args.ai_batch_count,
                verify=not args.ai_no_verify,
                model=args.ai_model,
                max_retries=args.ai_retries,
                provider=args.ai_provider,
                base_url=args.ai_base_url,
            )
            output_dir = args.output_dir or Path("batch_materials")
            output_dir.mkdir(parents=True, exist_ok=True)
            for name, mat_result in zip(result.material_names, result.materials):
                out_path = output_dir / f"{name}.lux"
                out_path.write_text(mat_result.lux_source, encoding="utf-8")
                status = "ok" if mat_result.compilation_success else "FAIL"
                print(f"  [{status}] {out_path}")
            print(f"\nGenerated {len(result.materials)} materials in {output_dir}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- AI video-to-animation mode ---
    if args.ai_from_video:
        from luxc.ai.generate import generate_animated_shader_from_video
        try:
            result = generate_animated_shader_from_video(
                Path(args.ai_from_video),
                description=args.ai or "",
                verify=not args.ai_no_verify,
                model=args.ai_model,
                max_retries=args.ai_retries,
                provider=args.ai_provider,
                base_url=args.ai_base_url,
            )
            if args.input:
                output_path = Path(args.input)
            else:
                output_path = Path("animated.lux")
            if args.output_dir:
                output_path = args.output_dir / output_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.lux_source, encoding="utf-8")
            print(f"Generated animated shader: {output_path}")
            if result.attempts > 1:
                print(f"  (took {result.attempts} attempts)", file=sys.stderr)
            if not result.compilation_success and not args.ai_no_verify:
                print("  Note: generated code did not pass compilation check", file=sys.stderr)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- AI reference matching mode ---
    if args.ai_match_reference:
        from luxc.ai.reference_match import match_reference_image
        try:
            result = match_reference_image(
                Path(args.ai_match_reference),
                description=args.ai or "",
                max_iterations=args.ai_match_iterations,
                verify=not args.ai_no_verify,
                model=args.ai_model,
                provider=args.ai_provider,
                base_url=args.ai_base_url,
            )
            if args.input:
                output_path = Path(args.input)
            else:
                output_path = Path("matched.lux")
            if args.output_dir:
                output_path = args.output_dir / output_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.final_result.lux_source, encoding="utf-8")
            print(f"Reference match: {output_path}")
            print(f"  Iterations: {len(result.iterations)}")
            if result.comparison:
                print(f"  PSNR: {result.comparison.psnr:.1f}")
                print(f"  SSIM: {result.comparison.ssim:.3f}")
            print(f"  Converged: {result.converged}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- AI image-to-material mode ---
    if args.ai_from_image:
        from luxc.ai.generate import generate_material_from_image
        try:
            result = generate_material_from_image(
                args.ai_from_image,
                description=args.ai or "",
                verify=not args.ai_no_verify,
                model=args.ai_model,
                max_retries=args.ai_retries,
                provider=args.ai_provider,
                base_url=args.ai_base_url,
            )
            if args.input:
                output_path = Path(args.input)
            else:
                output_path = Path("generated.lux")
            if args.output_dir:
                output_path = args.output_dir / output_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.lux_source, encoding="utf-8")
            print(f"Generated material: {output_path}")
            if result.attempts > 1:
                print(f"  (took {result.attempts} attempts)", file=sys.stderr)
            if result.errors:
                for err in result.errors:
                    print(f"  Warning: {err.phase}: {err.message}", file=sys.stderr)
            if not result.compilation_success and not args.ai_no_verify:
                print("  Note: generated code did not pass compilation check", file=sys.stderr)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- AI generation mode ---
    if args.ai:
        from luxc.ai.generate import generate_lux_shader
        try:
            result = generate_lux_shader(
                args.ai,
                verify=not args.ai_no_verify,
                model=args.ai_model,
                max_retries=args.ai_retries,
                provider=args.ai_provider,
                base_url=args.ai_base_url,
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
            if result.attempts > 1:
                print(f"  (took {result.attempts} attempts)", file=sys.stderr)
            if result.errors:
                for err in result.errors:
                    print(f"  Warning: {err.phase}: {err.message}", file=sys.stderr)
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

    # Parse --define arguments
    defines = {}
    if args.define:
        for d in args.define:
            if '=' not in d:
                print(f"Error: --define requires KEY=VALUE format, got '{d}'", file=sys.stderr)
                sys.exit(1)
            key, val = d.split('=', 1)
            try:
                defines[key.strip()] = int(val.strip())
            except ValueError:
                print(f"Error: --define value must be integer, got '{val}' for key '{key}'", file=sys.stderr)
                sys.exit(1)

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
                    defines=defines, bindless=args.bindless,
                    warn_nan=args.warn_nan,
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
                    features=perm, defines=defines, bindless=args.bindless,
                    warn_nan=args.warn_nan,
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
            defines=defines,
            bindless=args.bindless,
            warn_nan=args.warn_nan,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
