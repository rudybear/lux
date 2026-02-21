"""Top-level compiler orchestration."""

from pathlib import Path
from luxc.parser.tree_builder import parse_lux
from luxc.analysis.type_checker import type_check
from luxc.optimization.const_fold import constant_fold
from luxc.analysis.layout_assigner import assign_layouts
from luxc.codegen.spirv_builder import generate_spirv
from luxc.codegen.spv_assembler import assemble_and_validate
from luxc.codegen.reflection import generate_reflection, emit_reflection_json
from luxc.builtins.types import clear_type_aliases
from luxc.features.evaluator import collect_feature_names, strip_features

# Standard library search path
_STDLIB_DIR = Path(__file__).parent / "stdlib"


def _resolve_imports(module, source_dir: Path | None = None):
    """Resolve import declarations by finding, parsing, and merging imported modules.

    Search order:
    1. stdlib directory (luxc/stdlib/<name>.lux)
    2. Source file directory (if provided)
    """
    resolved = set()
    for imp in module.imports:
        name = imp.module_name
        if name in resolved:
            continue
        resolved.add(name)

        # Find the .lux file
        candidates = [_STDLIB_DIR / f"{name}.lux"]
        if source_dir:
            candidates.append(source_dir / f"{name}.lux")

        found = None
        for path in candidates:
            if path.exists():
                found = path
                break

        if found is None:
            searched = ", ".join(str(c.parent) for c in candidates)
            raise ImportError(f"Cannot find module '{name}' (searched: {searched})")

        # Parse the imported module
        imported_source = found.read_text(encoding="utf-8")
        imported = parse_lux(imported_source)

        # Merge: type aliases, constants, functions, schedules (but not stages, surfaces, etc.)
        module.type_aliases.extend(imported.type_aliases)
        module.constants.extend(imported.constants)
        module.functions.extend(imported.functions)
        module.schedules.extend(imported.schedules)

        # Recursively resolve imports from the imported module
        if imported.imports:
            _resolve_imports(imported, found.parent)
            # After recursive resolution, merge any additional items
            # that were added to the imported module
            # (already handled by extend above since we recurse first)


def compile_source(
    source: str,
    stem: str,
    output_dir: Path,
    source_dir: Path | None = None,
    dump_ast: bool = False,
    emit_asm: bool = False,
    validate: bool = True,
    emit_reflection: bool = True,
    debug: bool = False,
    source_name: str = "",
    pipeline: str | None = None,
    features: set[str] | None = None,
    defines: dict[str, int] | None = None,
) -> None:
    # Clear type aliases from previous compilations
    clear_type_aliases()

    module = parse_lux(source)

    module._defines = defines or {}

    # Strip compile-time features (before import resolution)
    if module.features_decls or hasattr(module, '_conditional_blocks'):
        active = features if features is not None else set()
        strip_features(module, active)

    # Resolve imports before type checking
    _resolve_imports(module, source_dir)

    # Expand surface/geometry/pipeline declarations into stage blocks
    if module.surfaces or module.pipelines or module.environments or module.procedurals:
        from luxc.expansion.surface_expander import expand_surfaces
        expand_surfaces(module, pipeline_filter=pipeline)

    # Expand @differentiable functions into gradient functions
    from luxc.autodiff.forward_diff import autodiff_expand
    autodiff_expand(module)

    if dump_ast:
        _dump_ast(module)
        return

    type_check(module)
    constant_fold(module)
    assign_layouts(module)

    output_dir.mkdir(parents=True, exist_ok=True)

    _SUFFIX_MAP = {
        "vertex": "vert",
        "fragment": "frag",
        "raygen": "rgen",
        "closest_hit": "rchit",
        "any_hit": "rahit",
        "miss": "rmiss",
        "intersection": "rint",
        "callable": "rcall",
        "mesh": "mesh",
        "task": "task",
    }

    # Compute feature suffix for output filenames
    feature_suffix = ""
    all_features = {}
    if module.features_decls:
        all_feature_names = collect_feature_names(module)
        active = features if features is not None else set()
        feature_suffix = "+" + "+".join(
            n.removeprefix("has_") for n in sorted(active)
        ) if active else ""
        all_features = {name: name in active for name in all_feature_names}

    for stage in module.stages:
        stage_name = stage.stage_type
        suffix = _SUFFIX_MAP[stage_name]
        out_stem = f"{stem}{feature_suffix}"

        asm_text = generate_spirv(module, stage, debug=debug, source_name=source_name or f"{stem}.lux")

        if emit_asm:
            asm_path = output_dir / f"{out_stem}.{suffix}.spvasm"
            asm_path.write_text(asm_text, encoding="utf-8")
            print(f"Wrote {asm_path}")

        spv_path = output_dir / f"{out_stem}.{suffix}.spv"
        assemble_and_validate(asm_text, spv_path, validate=validate)
        print(f"Wrote {spv_path}")

        if emit_reflection:
            reflection = generate_reflection(
                module, stage,
                source_name=source_name or f"{stem}.lux",
                features=all_features,
                feature_suffix=feature_suffix,
            )
            json_path = output_dir / f"{out_stem}.{suffix}.json"
            json_path.write_text(emit_reflection_json(reflection), encoding="utf-8")
            print(f"Wrote {json_path}")


def _dump_ast(module):
    import dataclasses, json

    def _ser(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            d = {"_type": type(obj).__name__}
            d.update(dataclasses.asdict(obj))
            return d
        if isinstance(obj, list):
            return [_ser(x) for x in obj]
        return obj

    print(json.dumps(_ser(module), indent=2, default=str))
