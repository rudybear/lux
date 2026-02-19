"""Reflection metadata emitter.

Generates .lux.json sidecar files alongside .spv binaries, providing
data-driven pipeline setup information. Runs after assign_layouts() so
all set/binding/location numbers are filled in.

The JSON schema is consumed by all three runtimes (Python/wgpu, C++/Vulkan,
Rust/ash) to create descriptor set layouts, pipeline layouts, and vertex
buffer layouts without any hardcoded values.
"""

from __future__ import annotations

import json
from luxc.parser.ast_nodes import (
    Module, StageBlock, VarDecl, UniformBlock, PushBlock, BlockField,
    SamplerDecl, AccelDecl, StorageImageDecl,
    RayPayloadDecl, HitAttributeDecl, CallableDataDecl,
)
from luxc.analysis.layout_assigner import compute_std140_offsets, _std140_size_align
from luxc.builtins.types import resolve_type, resolve_alias_chain, VectorType, MatrixType, ScalarType


# Execution model mapping (matches spirv_builder.py)
_EXEC_MODELS = {
    "vertex": "Vertex",
    "fragment": "Fragment",
    "raygen": "RayGenerationKHR",
    "closest_hit": "ClosestHitKHR",
    "any_hit": "AnyHitKHR",
    "miss": "MissKHR",
    "intersection": "IntersectionKHR",
    "callable": "CallableKHR",
}

# Vulkan format strings for common Lux types
_TYPE_TO_VK_FORMAT = {
    "scalar": "R32_SFLOAT",
    "vec2": "R32G32_SFLOAT",
    "vec3": "R32G32B32_SFLOAT",
    "vec4": "R32G32B32A32_SFLOAT",
    "int": "R32_SINT",
    "ivec2": "R32G32_SINT",
    "ivec3": "R32G32B32_SINT",
    "ivec4": "R32G32B32A32_SINT",
    "uint": "R32_UINT",
    "uvec2": "R32G32_UINT",
    "uvec3": "R32G32B32_UINT",
    "uvec4": "R32G32B32A32_UINT",
}

# Size in bytes for vertex attribute types
_TYPE_BYTE_SIZE = {
    "scalar": 4, "int": 4, "uint": 4, "bool": 4,
    "vec2": 8, "vec3": 12, "vec4": 16,
    "ivec2": 8, "ivec3": 12, "ivec4": 16,
    "uvec2": 8, "uvec3": 12, "uvec4": 16,
    "mat2": 32, "mat3": 48, "mat4": 64,
}


def generate_reflection(module: Module, stage: StageBlock, source_name: str = "") -> dict:
    """Generate reflection metadata dict for a single stage.

    Args:
        module: The compiled module (for accessing shared definitions).
        stage: The stage block to reflect.
        source_name: Original .lux source filename for metadata.

    Returns:
        A dict matching the .lux.json schema.
    """
    result = {
        "version": 1,
        "source": source_name,
        "stage": stage.stage_type,
        "execution_model": _EXEC_MODELS.get(stage.stage_type, "Unknown"),
    }

    # --- Inputs ---
    result["inputs"] = [
        _reflect_var_decl(inp) for inp in stage.inputs
    ]

    # --- Outputs ---
    result["outputs"] = [
        _reflect_var_decl(out) for out in stage.outputs
    ]

    # --- Descriptor sets ---
    descriptor_sets: dict[str, list[dict]] = {}

    # Uniform blocks
    for ub in stage.uniforms:
        set_key = str(ub.set_number)
        if set_key not in descriptor_sets:
            descriptor_sets[set_key] = []

        offsets = compute_std140_offsets(ub.fields)
        total_size = _compute_block_size(ub.fields, offsets)

        descriptor_sets[set_key].append({
            "binding": ub.binding,
            "type": "uniform_buffer",
            "name": ub.name,
            "fields": [
                {
                    "name": f.name,
                    "type": f.type_name,
                    "offset": offsets[i],
                    "size": _std140_size_align(f.type_name)[0],
                }
                for i, f in enumerate(ub.fields)
            ],
            "size": total_size,
            "stage_flags": [stage.stage_type],
        })

    # Samplers (separate sampler + texture, 2 bindings each)
    for sam in stage.samplers:
        set_key = str(sam.set_number)
        if set_key not in descriptor_sets:
            descriptor_sets[set_key] = []

        sam_type_name = getattr(sam, 'type_name', 'sampler2d')
        is_cube = sam_type_name == "samplerCube"
        descriptor_sets[set_key].append({
            "binding": sam.binding,
            "type": "sampler",
            "name": sam.name,
            "stage_flags": [stage.stage_type],
        })
        descriptor_sets[set_key].append({
            "binding": sam.texture_binding,
            "type": "sampled_cube_image" if is_cube else "sampled_image",
            "name": sam.name,
            "stage_flags": [stage.stage_type],
        })

    # Acceleration structures
    for accel in stage.accel_structs:
        set_key = str(accel.set_number)
        if set_key not in descriptor_sets:
            descriptor_sets[set_key] = []

        descriptor_sets[set_key].append({
            "binding": accel.binding,
            "type": "acceleration_structure",
            "name": accel.name,
            "stage_flags": [stage.stage_type],
        })

    # Storage images
    for si in getattr(stage, 'storage_images', []):
        set_key = str(si.set_number)
        if set_key not in descriptor_sets:
            descriptor_sets[set_key] = []

        descriptor_sets[set_key].append({
            "binding": si.binding,
            "type": "storage_image",
            "name": si.name,
            "stage_flags": [stage.stage_type],
        })

    result["descriptor_sets"] = descriptor_sets

    # --- Push constants ---
    result["push_constants"] = []
    for pb in stage.push_constants:
        offsets = compute_std140_offsets(pb.fields)
        total_size = _compute_block_size(pb.fields, offsets)
        result["push_constants"].append({
            "name": pb.name,
            "fields": [
                {
                    "name": f.name,
                    "type": f.type_name,
                    "offset": offsets[i],
                    "size": _std140_size_align(f.type_name)[0],
                }
                for i, f in enumerate(pb.fields)
            ],
            "size": total_size,
            "stage_flags": [stage.stage_type],
        })

    # --- Vertex attributes (only for vertex shaders) ---
    if stage.stage_type == "vertex":
        attrs = []
        offset = 0
        for inp in stage.inputs:
            resolved = resolve_alias_chain(inp.type_name)
            fmt = _TYPE_TO_VK_FORMAT.get(resolved, "R32G32B32A32_SFLOAT")
            byte_size = _TYPE_BYTE_SIZE.get(resolved, 16)
            attrs.append({
                "location": inp.location,
                "type": inp.type_name,
                "name": inp.name,
                "format": fmt,
                "offset": offset,
            })
            offset += byte_size

        result["vertex_attributes"] = attrs
        result["vertex_stride"] = offset
    else:
        result["vertex_attributes"] = []
        result["vertex_stride"] = 0

    # --- RT-specific metadata ---
    if stage.ray_payloads:
        result["ray_payloads"] = [
            {"name": rp.name, "type": rp.type_name, "location": rp.location}
            for rp in stage.ray_payloads
        ]

    if stage.hit_attributes:
        result["hit_attributes"] = [
            {"name": ha.name, "type": ha.type_name}
            for ha in stage.hit_attributes
        ]

    if stage.callable_data:
        result["callable_data"] = [
            {"name": cd.name, "type": cd.type_name, "location": cd.location}
            for cd in stage.callable_data
        ]

    return result


def emit_reflection_json(reflection: dict) -> str:
    """Serialize reflection metadata to a JSON string."""
    return json.dumps(reflection, indent=2, sort_keys=False) + "\n"


def _reflect_var_decl(v: VarDecl) -> dict:
    """Reflect an input/output variable declaration."""
    return {
        "name": v.name,
        "type": v.type_name,
        "location": v.location,
    }


def _compute_block_size(fields: list[BlockField], offsets: list[int]) -> int:
    """Compute total size of a uniform/push block in bytes (std140 padded)."""
    if not fields:
        return 0
    last_offset = offsets[-1]
    last_size = _std140_size_align(fields[-1].type_name)[0]
    total = last_offset + last_size
    # Round up to 16-byte alignment (std140 struct alignment)
    return (total + 15) & ~15
