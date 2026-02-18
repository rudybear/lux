"""Auto-assign locations, descriptor set/binding numbers."""

from __future__ import annotations
from luxc.parser.ast_nodes import Module, StageBlock
from luxc.builtins.types import resolve_type, MatrixType


def assign_layouts(module: Module) -> None:
    binding_counter = 0
    for stage in module.stages:
        _assign_stage_layouts(stage, binding_counter)


def _assign_stage_layouts(stage: StageBlock, binding_start: int) -> None:
    # Auto-assign input locations by declaration order
    loc = 0
    for inp in stage.inputs:
        inp.location = loc
        loc += 1

    # Auto-assign output locations by declaration order
    loc = 0
    for out in stage.outputs:
        out.location = loc
        loc += 1

    # Auto-assign uniform block set/binding
    set_num = getattr(stage, '_descriptor_set_offset', 0)
    binding = 0
    for ub in stage.uniforms:
        ub.set_number = set_num
        ub.binding = binding
        binding += 1

    # Auto-assign sampler bindings (same set, continuing binding numbers)
    for sam in stage.samplers:
        sam.set_number = set_num
        sam.binding = binding
        binding += 1


def compute_std140_offsets(fields: list) -> list[int]:
    """Compute std140 offsets for block fields.

    Returns list of byte offsets, one per field.
    """
    offsets = []
    current_offset = 0

    for f in fields:
        t = resolve_type(f.type_name)
        size, align = _std140_size_align(f.type_name)
        # Align current offset
        current_offset = _align_up(current_offset, align)
        offsets.append(current_offset)
        current_offset += size

    return offsets


def _std140_size_align(type_name: str) -> tuple[int, int]:
    """Return (size, alignment) in bytes for std140 layout."""
    t = resolve_type(type_name)
    if t is None:
        return (4, 4)

    if t.name == "scalar" or t.name == "int" or t.name == "uint":
        return (4, 4)
    elif t.name == "vec2":
        return (8, 8)
    elif t.name == "vec3":
        return (12, 16)  # vec3 has alignment of vec4 in std140
    elif t.name == "vec4":
        return (16, 16)
    elif t.name == "mat2":
        return (32, 16)  # 2 columns of vec4 (padded vec2)
    elif t.name == "mat3":
        return (48, 16)  # 3 columns of vec4 (padded vec3)
    elif t.name == "mat4":
        return (64, 16)  # 4 columns of vec4
    elif t.name == "bool":
        return (4, 4)
    else:
        return (4, 4)


def _align_up(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)
