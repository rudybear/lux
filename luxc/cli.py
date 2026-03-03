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
        "--rich-debug", action="store_true",
        help="Emit NonSemantic.Shader.DebugInfo.100 for RenderDoc variable inspection (requires -g)",
    )
    parser.add_argument(
        "--debug-print", action="store_true",
        help="Preserve debug_print/assert in release mode (without full -g debug info)",
    )
    parser.add_argument(
        "--assert-kill", action="store_true",
        help="Demote fragment invocation on assertion failure (OpDemoteToHelperInvocation)",
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
        "-O", "--optimize", action="store_true", default=False,
        help="Run spirv-opt -O on generated SPIR-V binaries (requires spirv-opt on PATH)",
    )
    parser.add_argument(
        "--perf", action="store_true", default=False,
        help="Run performance-oriented spirv-opt passes (loop unroll, strength reduction)",
    )
    parser.add_argument(
        "--analyze", action="store_true", default=False,
        help="Print per-stage instruction cost analysis after compilation",
    )
    parser.add_argument(
        "--debug-run", action="store_true",
        help="Run the shader in the CPU-side AST debugger (no GPU required)",
    )
    parser.add_argument(
        "--stage", type=str, default="fragment", metavar="STAGE",
        help="Stage to debug with --debug-run (default: fragment)",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run debugger in batch mode (JSON output)",
    )
    parser.add_argument(
        "--dump-vars", action="store_true",
        help="Dump all variable assignments in batch mode",
    )
    parser.add_argument(
        "--check-nan", action="store_true",
        help="Check for NaN/Inf values in batch mode",
    )
    parser.add_argument(
        "--break", type=int, action="append", metavar="LINE", dest="break_lines",
        help="Set breakpoint at line (batch mode)",
    )
    parser.add_argument(
        "--dump-at-break", action="store_true",
        help="Dump variables at each breakpoint hit (batch mode)",
    )
    parser.add_argument(
        "--input", type=Path, metavar="JSON", dest="debug_input",
        help="Input values JSON file for --debug-run",
    )
    parser.add_argument(
        "--pixel", type=str, metavar="X,Y",
        help="Debug a specific pixel (e.g., --pixel 320,240). Sets uv from pixel coords.",
    )
    parser.add_argument(
        "--resolution", type=str, metavar="WxH", default="1920x1080",
        help="Screen resolution for --pixel (default: 1920x1080)",
    )
    parser.add_argument(
        "--set", type=str, action="append", metavar="VAR=VALUE", dest="debug_set",
        help="Override a single input variable (e.g., --set roughness=0.1 --set \"normal=0,1,0\")",
    )
    parser.add_argument(
        "--export-inputs", type=Path, metavar="JSON",
        help="Export default shader inputs to JSON for --debug-run replay",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Watch source files for changes and recompile automatically",
    )
    parser.add_argument(
        "--watch-poll", type=int, default=500, metavar="MS",
        help="Polling interval in milliseconds for watch mode (default: 500)",
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

    # --- Export inputs mode ---
    if args.export_inputs:
        from luxc.parser.tree_builder import parse_lux
        from luxc.analysis.type_checker import type_check
        from luxc.debug.io import build_default_inputs, export_inputs_to_json
        try:
            module = parse_lux(source)
            if module.surfaces or module.pipelines or module.environments or module.procedurals:
                from luxc.expansion.surface_expander import expand_surfaces
                expand_surfaces(module)
            type_check(module)
            stage = None
            for s in module.stages:
                if s.stage_type == args.stage:
                    stage = s
                    break
            if stage is None:
                print(f"Error: stage '{args.stage}' not found", file=sys.stderr)
                sys.exit(1)
            inputs = build_default_inputs(stage)
            export_inputs_to_json(inputs, args.export_inputs)
            print(f"Exported inputs to {args.export_inputs}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- Debug run mode ---
    if args.debug_run:
        from luxc.debug.cli import run_interactive, run_batch

        # Build inline overrides from --pixel and --set
        inline_overrides: dict[str, object] = {}

        if args.pixel:
            try:
                px, py = (int(x) for x in args.pixel.split(","))
                rw, rh = (int(x) for x in args.resolution.split("x"))
                u = (px + 0.5) / rw
                v = (py + 0.5) / rh
                inline_overrides["uv"] = [u, v]
                inline_overrides["texcoord"] = [u, v]
                # Synthesize a world position on a unit quad centered at origin
                # Maps pixel (0,0)=top-left to (-1,1,0), (W,H)=bottom-right to (1,-1,0)
                wx = u * 2.0 - 1.0
                wy = 1.0 - v * 2.0
                inline_overrides["world_position"] = [wx, wy, 0.0]
                inline_overrides["position"] = [wx, wy, 0.0]
                # Derive a normal from screen position (hemisphere for visual variety)
                import math
                r2 = wx * wx + wy * wy
                if r2 < 1.0:
                    nz = math.sqrt(1.0 - r2)
                    inline_overrides["normal"] = [wx, wy, nz]
                    inline_overrides["world_normal"] = [wx, wy, nz]
                print(f"Pixel ({px}, {py}) @ {rw}x{rh} -> uv=({u:.4f}, {v:.4f}), world=({wx:.3f}, {wy:.3f}, 0.0)")
            except ValueError:
                print(f"Error: invalid --pixel format '{args.pixel}'. Use: --pixel X,Y", file=sys.stderr)
                sys.exit(1)

        if args.debug_set:
            for assignment in args.debug_set:
                if "=" not in assignment:
                    print(f"Error: --set requires VAR=VALUE format, got '{assignment}'", file=sys.stderr)
                    sys.exit(1)
                name, val_str = assignment.split("=", 1)
                name = name.strip()
                val_str = val_str.strip()
                # Parse value: number, comma-separated vector, or bool
                if "," in val_str:
                    inline_overrides[name] = [float(x) for x in val_str.split(",")]
                elif val_str.lower() in ("true", "false"):
                    inline_overrides[name] = val_str.lower() == "true"
                else:
                    try:
                        inline_overrides[name] = float(val_str)
                    except ValueError:
                        print(f"Error: cannot parse value '{val_str}' for --set {name}", file=sys.stderr)
                        sys.exit(1)

        # Write inline overrides to a temp JSON file if any
        debug_input = args.debug_input
        if inline_overrides:
            import tempfile, json as json_mod
            # If there's also a --input file, merge with it
            merged = {}
            if args.debug_input:
                with open(args.debug_input, "r") as f:
                    merged = json_mod.load(f)
            merged.update(inline_overrides)
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
            json_mod.dump(merged, tmp, indent=2)
            tmp.close()
            debug_input = Path(tmp.name)

        try:
            if args.batch:
                result = run_batch(
                    source=source,
                    stage_name=args.stage,
                    source_name=input_path.name,
                    input_path=debug_input,
                    dump_vars=args.dump_vars,
                    check_nan=args.check_nan,
                    break_lines=args.break_lines,
                    dump_at_break=args.dump_at_break,
                )
                import json
                print(json.dumps(result, indent=2))
            else:
                run_interactive(
                    source=source,
                    stage_name=args.stage,
                    source_name=input_path.name,
                    input_path=debug_input,
                    source_lines=source.splitlines(),
                )
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
                    optimize=args.optimize,
                    perf_optimize=args.perf,
                    analyze=args.analyze,
                    debug_print=args.debug_print,
                    assert_kill=args.assert_kill,
                    rich_debug=args.rich_debug,
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
                    optimize=args.optimize,
                    perf_optimize=args.perf,
                    analyze=args.analyze,
                    debug_print=args.debug_print,
                    assert_kill=args.assert_kill,
                    rich_debug=args.rich_debug,
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
            optimize=args.optimize,
            perf_optimize=args.perf,
            analyze=args.analyze,
            debug_print=args.debug_print,
            assert_kill=args.assert_kill,
            rich_debug=args.rich_debug,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if not args.watch:
            sys.exit(1)

    # --- Watch mode ---
    if args.watch:
        from luxc.hot_reload import HotReloader
        from luxc.watcher import LuxFileWatcher

        reloader = HotReloader(
            source_path=input_path,
            output_dir=output_dir,
            validate=not args.no_validate,
            emit_reflection=not args.no_reflection,
            debug=args.debug,
            pipeline=args.pipeline,
            features=feature_set,
            defines=defines,
            bindless=args.bindless,
        )

        watched_files = reloader.get_import_paths()
        print(f"[watch] Monitoring {len(watched_files)} files...", flush=True)

        def on_change(changed_path):
            print(f"[watch] Changed: {changed_path.name}", flush=True)
            result = reloader.recompile()
            if result.success:
                print(f"[watch] Recompiling... done ({result.elapsed_ms:.0f}ms)", flush=True)
                # Re-scan imports in case they changed
                new_files = reloader.get_import_paths()
                watcher.update_files(new_files)
                if len(new_files) != len(watched_files):
                    print(f"[watch] Monitoring {len(new_files)} files...", flush=True)
            else:
                print(f"[watch] FAILED: {result.error_message}", flush=True)

        watcher = LuxFileWatcher(
            files=watched_files,
            callback=on_change,
            poll_interval_ms=args.watch_poll,
        )
        watcher.start()

        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[watch] Stopped.", flush=True)
            watcher.stop()
