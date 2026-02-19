"""Built-in type definitions for Lux."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class LuxType:
    name: str

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class ScalarType(LuxType):
    pass


@dataclass(frozen=True)
class VectorType(LuxType):
    component: str   # "scalar", "int", "uint"
    size: int        # 2, 3, 4


@dataclass(frozen=True)
class MatrixType(LuxType):
    size: int  # 2, 3, 4 (cols)
    rows: int = 0  # 0 means square (rows == size); non-zero for non-square (e.g., mat4x3)


@dataclass(frozen=True)
class SamplerType(LuxType):
    pass


@dataclass(frozen=True)
class BoolType(LuxType):
    pass


@dataclass(frozen=True)
class VoidType(LuxType):
    pass


@dataclass(frozen=True)
class AccelerationStructureType(LuxType):
    pass


@dataclass(frozen=True)
class StorageImageType(LuxType):
    pass


# Singleton type instances
VOID = VoidType("void")
BOOL = BoolType("bool")
SCALAR = ScalarType("scalar")
INT = ScalarType("int")
UINT = ScalarType("uint")

VEC2 = VectorType("vec2", "scalar", 2)
VEC3 = VectorType("vec3", "scalar", 3)
VEC4 = VectorType("vec4", "scalar", 4)

IVEC2 = VectorType("ivec2", "int", 2)
IVEC3 = VectorType("ivec3", "int", 3)
IVEC4 = VectorType("ivec4", "int", 4)

UVEC2 = VectorType("uvec2", "uint", 2)
UVEC3 = VectorType("uvec3", "uint", 3)
UVEC4 = VectorType("uvec4", "uint", 4)

MAT2 = MatrixType("mat2", 2)
MAT3 = MatrixType("mat3", 3)
MAT4 = MatrixType("mat4", 4)
MAT4X3 = MatrixType("mat4x3", 4, 3)

SAMPLER2D = SamplerType("sampler2d")
SAMPLER_CUBE = SamplerType("samplerCube")
ACCELERATION_STRUCTURE = AccelerationStructureType("acceleration_structure")
STORAGE_IMAGE = StorageImageType("storage_image")

# Lookup table: type name string -> LuxType
TYPE_MAP: dict[str, LuxType] = {
    "void": VOID,
    "bool": BOOL,
    "scalar": SCALAR,
    "int": INT,
    "uint": UINT,
    "vec2": VEC2, "vec3": VEC3, "vec4": VEC4,
    "ivec2": IVEC2, "ivec3": IVEC3, "ivec4": IVEC4,
    "uvec2": UVEC2, "uvec3": UVEC3, "uvec4": UVEC4,
    "mat2": MAT2, "mat3": MAT3, "mat4": MAT4, "mat4x3": MAT4X3,
    "sampler2d": SAMPLER2D,
    "samplerCube": SAMPLER_CUBE,
    "acceleration_structure": ACCELERATION_STRUCTURE,
    "storage_image": STORAGE_IMAGE,
}


# User-defined type aliases: name -> target type name
_type_aliases: dict[str, str] = {}


def register_type_alias(alias_name: str, target_name: str) -> None:
    """Register a user-defined type alias (e.g., Radiance -> vec3)."""
    _type_aliases[alias_name] = target_name


def clear_type_aliases() -> None:
    """Clear all user-defined type aliases (for test isolation)."""
    _type_aliases.clear()


def resolve_type(name: str) -> LuxType | None:
    # Check built-in types first
    t = TYPE_MAP.get(name)
    if t is not None:
        return t
    # Check user-defined type aliases (resolve transitively)
    seen = set()
    current = name
    while current in _type_aliases and current not in seen:
        seen.add(current)
        current = _type_aliases[current]
    return TYPE_MAP.get(current)


def resolve_alias_chain(name: str) -> str:
    """Resolve a type alias chain to the final built-in type name."""
    seen = set()
    current = name
    while current in _type_aliases and current not in seen:
        seen.add(current)
        current = _type_aliases[current]
    return current


def is_type_alias(name: str) -> bool:
    """Check if a name is a registered type alias."""
    return name in _type_aliases


def vector_component_type(vt: VectorType) -> LuxType:
    return TYPE_MAP[vt.component]


def vector_size(type_name: str) -> int | None:
    t = TYPE_MAP.get(type_name)
    if isinstance(t, VectorType):
        return t.size
    return None


def is_numeric(t: LuxType) -> bool:
    return isinstance(t, (ScalarType, VectorType, MatrixType))


def is_float_based(t: LuxType) -> bool:
    if isinstance(t, ScalarType) and t.name == "scalar":
        return True
    if isinstance(t, VectorType) and t.component == "scalar":
        return True
    if isinstance(t, MatrixType):
        return True
    return False
