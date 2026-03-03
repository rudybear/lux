"""Reflection-driven input loading for the Lux shader debugger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from luxc.debug.values import (
    LuxValue, LuxScalar, LuxVec, LuxMat, LuxInt, LuxBool, LuxStruct,
    default_value,
)
from luxc.parser.ast_nodes import StageBlock


# Semantic defaults for common shader input names
_SEMANTIC_DEFAULTS: dict[str, LuxValue] = {
    "position": LuxVec([0.0, 0.0, 0.0]),
    "world_position": LuxVec([0.0, 0.0, 0.0]),
    "normal": LuxVec([0.0, 0.0, 1.0]),
    "world_normal": LuxVec([0.0, 0.0, 1.0]),
    "uv": LuxVec([0.5, 0.5]),
    "texcoord": LuxVec([0.5, 0.5]),
    "texcoord0": LuxVec([0.5, 0.5]),
    "color": LuxVec([1.0, 1.0, 1.0, 1.0]),
    "vertex_color": LuxVec([1.0, 1.0, 1.0, 1.0]),
    "frag_color": LuxVec([1.0, 1.0, 1.0]),
    "tangent": LuxVec([1.0, 0.0, 0.0, 1.0]),
    "bitangent": LuxVec([0.0, 1.0, 0.0]),
    "roughness": LuxScalar(0.5),
    "metallic": LuxScalar(0.0),
    "metalness": LuxScalar(0.0),
    "albedo": LuxVec([0.8, 0.8, 0.8]),
    "baseColor": LuxVec([0.8, 0.8, 0.8]),
    "ao": LuxScalar(1.0),
    "occlusion": LuxScalar(1.0),
    "emission": LuxVec([0.0, 0.0, 0.0]),
    "opacity": LuxScalar(1.0),
    "alpha": LuxScalar(1.0),
    "exposure": LuxScalar(1.0),
    "intensity": LuxScalar(1.0),
    "ior": LuxScalar(1.5),
}


def _json_to_value(data, type_name: str | None = None) -> LuxValue:
    """Convert a JSON value to a LuxValue."""
    if isinstance(data, bool):
        return LuxBool(data)
    if isinstance(data, int):
        if type_name in ("scalar", "float"):
            return LuxScalar(float(data))
        return LuxInt(data)
    if isinstance(data, float):
        return LuxScalar(data)
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], list):
            # Matrix (list of columns)
            return LuxMat([list(map(float, col)) for col in data])
        floats = [float(x) for x in data]
        return LuxVec(floats)
    if isinstance(data, dict):
        # Typed value: {"type": "vec3", "value": [1, 2, 3]}
        if "type" in data and "value" in data:
            return _json_to_value(data["value"], data["type"])
        # Struct-like
        fields = {k: _json_to_value(v) for k, v in data.items()}
        return LuxStruct("unknown", fields)
    return LuxScalar(0.0)


def load_inputs_from_json(path: Path) -> dict[str, LuxValue]:
    """Load input values from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    inputs: dict[str, LuxValue] = {}
    for name, value in data.items():
        inputs[name] = _json_to_value(value)
    return inputs


def load_inputs_from_reflection(reflection_path: Path, stage: StageBlock) -> dict[str, LuxValue]:
    """Load default inputs based on reflection JSON and stage declarations."""
    inputs: dict[str, LuxValue] = {}

    # Use stage input declarations to determine types
    for inp in stage.inputs:
        name = inp.name
        type_name = inp.type_name

        # Check semantic defaults
        if name in _SEMANTIC_DEFAULTS:
            inputs[name] = _SEMANTIC_DEFAULTS[name]
        else:
            inputs[name] = default_value(type_name)

    # Load uniform defaults
    for ub in stage.uniforms:
        for field in ub.fields:
            name = field.name
            type_name = field.type_name
            if name in _SEMANTIC_DEFAULTS:
                inputs[name] = _SEMANTIC_DEFAULTS[name]
            elif type_name == "mat4":
                inputs[name] = default_value("mat4")  # identity
            elif type_name in ("vec3", "vec4"):
                inputs[name] = default_value(type_name)
            else:
                inputs[name] = default_value(type_name)

    # Merge with reflection JSON if it exists
    if reflection_path.exists():
        try:
            with open(reflection_path, "r", encoding="utf-8") as f:
                refl = json.load(f)
            # Override with any explicit defaults from reflection
            if "defaults" in refl:
                for name, value in refl["defaults"].items():
                    inputs[name] = _json_to_value(value)
        except (json.JSONDecodeError, KeyError):
            pass

    return inputs


def build_default_inputs(stage: StageBlock) -> dict[str, LuxValue]:
    """Build default inputs purely from stage declarations and semantic heuristics."""
    inputs: dict[str, LuxValue] = {}

    for inp in stage.inputs:
        name = inp.name
        if name in _SEMANTIC_DEFAULTS:
            inputs[name] = _SEMANTIC_DEFAULTS[name]
        else:
            inputs[name] = default_value(inp.type_name)

    # Add uniform fields as inputs
    for ub in stage.uniforms:
        for uf in ub.fields:
            if uf.name in _SEMANTIC_DEFAULTS:
                inputs[uf.name] = _SEMANTIC_DEFAULTS[uf.name]
            else:
                inputs[uf.name] = default_value(uf.type_name)

    # Push constant fields
    for pb in stage.push_constants:
        for pf in pb.fields:
            if pf.name in _SEMANTIC_DEFAULTS:
                inputs[pf.name] = _SEMANTIC_DEFAULTS[pf.name]
            else:
                inputs[pf.name] = default_value(pf.type_name)

    return inputs


def export_inputs_to_json(inputs: dict[str, LuxValue], path: Path) -> None:
    """Export current input values to a JSON file for later replay."""
    data = {}
    for name, val in inputs.items():
        data[name] = _value_to_json_simple(val)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _value_to_json_simple(val: LuxValue):
    """Convert a LuxValue to a simple JSON-serializable form."""
    if isinstance(val, LuxScalar):
        return val.value
    if isinstance(val, LuxInt):
        return val.value
    if isinstance(val, LuxBool):
        return val.value
    if isinstance(val, LuxVec):
        return val.components
    if isinstance(val, LuxMat):
        return val.columns
    if isinstance(val, LuxStruct):
        return {k: _value_to_json_simple(v) for k, v in val.fields.items()}
    return 0.0
