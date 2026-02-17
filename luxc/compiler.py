"""Top-level compiler orchestration."""

from pathlib import Path
from luxc.parser.tree_builder import parse_lux
from luxc.analysis.type_checker import type_check
from luxc.analysis.layout_assigner import assign_layouts
from luxc.codegen.spirv_builder import generate_spirv
from luxc.codegen.spv_assembler import assemble_and_validate


def compile_source(
    source: str,
    stem: str,
    output_dir: Path,
    dump_ast: bool = False,
    emit_asm: bool = False,
    validate: bool = True,
) -> None:
    module = parse_lux(source)

    if dump_ast:
        _dump_ast(module)
        return

    type_check(module)
    assign_layouts(module)

    output_dir.mkdir(parents=True, exist_ok=True)

    for stage in module.stages:
        stage_name = stage.stage_type  # "vertex" or "fragment"
        suffix = {"vertex": "vert", "fragment": "frag"}[stage_name]

        asm_text = generate_spirv(module, stage)

        if emit_asm:
            asm_path = output_dir / f"{stem}.{suffix}.spvasm"
            asm_path.write_text(asm_text, encoding="utf-8")
            print(f"Wrote {asm_path}")

        spv_path = output_dir / f"{stem}.{suffix}.spv"
        assemble_and_validate(asm_text, spv_path, validate=validate)
        print(f"Wrote {spv_path}")


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
