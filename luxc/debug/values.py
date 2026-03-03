"""Value types for the Lux shader debugger interpreter."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union


@dataclass
class LuxScalar:
    value: float

    def __repr__(self) -> str:
        return f"{self.value:.6f}"


@dataclass
class LuxInt:
    value: int
    signed: bool = True

    def __repr__(self) -> str:
        return str(self.value)


@dataclass
class LuxBool:
    value: bool

    def __repr__(self) -> str:
        return "true" if self.value else "false"


@dataclass
class LuxVec:
    components: list[float]

    @property
    def size(self) -> int:
        return len(self.components)

    def __repr__(self) -> str:
        inner = ", ".join(f"{c:.3f}" for c in self.components)
        return f"vec{self.size}({inner})"


@dataclass
class LuxMat:
    columns: list[list[float]]  # column-major NxN

    @property
    def size(self) -> int:
        return len(self.columns)

    def __repr__(self) -> str:
        cols = "; ".join(
            ", ".join(f"{v:.3f}" for v in col) for col in self.columns
        )
        return f"mat{self.size}([{cols}])"


@dataclass
class LuxStruct:
    type_name: str
    fields: dict[str, LuxValue]

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self.fields.items())
        return f"{self.type_name}{{{inner}}}"


LuxValue = Union[LuxScalar, LuxInt, LuxBool, LuxVec, LuxMat, LuxStruct]


def is_nan(val: LuxValue) -> bool:
    """Check if a value contains NaN."""
    if isinstance(val, LuxScalar):
        return math.isnan(val.value)
    if isinstance(val, LuxVec):
        return any(math.isnan(c) for c in val.components)
    if isinstance(val, LuxMat):
        return any(math.isnan(v) for col in val.columns for v in col)
    return False


def is_inf(val: LuxValue) -> bool:
    """Check if a value contains Inf."""
    if isinstance(val, LuxScalar):
        return math.isinf(val.value)
    if isinstance(val, LuxVec):
        return any(math.isinf(c) for c in val.components)
    if isinstance(val, LuxMat):
        return any(math.isinf(v) for col in val.columns for v in col)
    return False


def value_to_json(val: LuxValue) -> dict:
    """Convert a LuxValue to a JSON-serializable dict."""
    if isinstance(val, LuxScalar):
        return {"type": "scalar", "value": val.value}
    if isinstance(val, LuxInt):
        return {"type": "int" if val.signed else "uint", "value": val.value}
    if isinstance(val, LuxBool):
        return {"type": "bool", "value": val.value}
    if isinstance(val, LuxVec):
        return {"type": f"vec{val.size}", "value": val.components}
    if isinstance(val, LuxMat):
        return {"type": f"mat{val.size}", "value": val.columns}
    if isinstance(val, LuxStruct):
        return {
            "type": val.type_name,
            "value": {k: value_to_json(v) for k, v in val.fields.items()},
        }
    return {"type": "unknown", "value": str(val)}


def value_type_name(val: LuxValue) -> str:
    """Return the Lux type name for a value."""
    if isinstance(val, LuxScalar):
        return "scalar"
    if isinstance(val, LuxInt):
        return "int" if val.signed else "uint"
    if isinstance(val, LuxBool):
        return "bool"
    if isinstance(val, LuxVec):
        return f"vec{val.size}"
    if isinstance(val, LuxMat):
        return f"mat{val.size}"
    if isinstance(val, LuxStruct):
        return val.type_name
    return "unknown"


def default_value(type_name: str) -> LuxValue:
    """Create a default-initialized value for the given type."""
    if type_name in ("scalar", "float"):
        return LuxScalar(0.0)
    if type_name == "int":
        return LuxInt(0)
    if type_name == "uint":
        return LuxInt(0, signed=False)
    if type_name == "bool":
        return LuxBool(False)
    if type_name == "vec2":
        return LuxVec([0.0, 0.0])
    if type_name == "vec3":
        return LuxVec([0.0, 0.0, 0.0])
    if type_name == "vec4":
        return LuxVec([0.0, 0.0, 0.0, 0.0])
    if type_name == "ivec2":
        return LuxVec([0.0, 0.0])
    if type_name == "ivec3":
        return LuxVec([0.0, 0.0, 0.0])
    if type_name == "ivec4":
        return LuxVec([0.0, 0.0, 0.0, 0.0])
    if type_name == "uvec2":
        return LuxVec([0.0, 0.0])
    if type_name == "uvec3":
        return LuxVec([0.0, 0.0, 0.0])
    if type_name == "uvec4":
        return LuxVec([0.0, 0.0, 0.0, 0.0])
    if type_name == "mat2":
        return LuxMat([[1.0, 0.0], [0.0, 1.0]])
    if type_name == "mat3":
        return LuxMat([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    if type_name == "mat4":
        return LuxMat([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    return LuxScalar(0.0)
