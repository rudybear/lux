"""AST node definitions for Lux."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SourceLocation:
    line: int
    column: int


# --- Module ---

@dataclass
class Module:
    constants: list[ConstDecl] = field(default_factory=list)
    functions: list[FunctionDef] = field(default_factory=list)
    structs: list[StructDef] = field(default_factory=list)
    stages: list[StageBlock] = field(default_factory=list)
    type_aliases: list[TypeAlias] = field(default_factory=list)
    imports: list[ImportDecl] = field(default_factory=list)
    surfaces: list[SurfaceDecl] = field(default_factory=list)
    geometries: list[GeometryDecl] = field(default_factory=list)
    pipelines: list[PipelineDecl] = field(default_factory=list)
    schedules: list[ScheduleDecl] = field(default_factory=list)
    environments: list[EnvironmentDecl] = field(default_factory=list)
    procedurals: list[ProceduralDecl] = field(default_factory=list)


# --- Top-level declarations ---

@dataclass
class ConstDecl:
    name: str
    type_name: str
    value: Expr
    loc: Optional[SourceLocation] = None


@dataclass
class StructDef:
    name: str
    fields: list[StructField]
    loc: Optional[SourceLocation] = None


@dataclass
class StructField:
    name: str
    type_name: str


@dataclass
class TypeAlias:
    name: str
    target_type: str
    loc: Optional[SourceLocation] = None


@dataclass
class ImportDecl:
    module_name: str
    loc: Optional[SourceLocation] = None


# --- Surface declarations ---

@dataclass
class SurfaceDecl:
    name: str
    members: list[SurfaceMember]
    samplers: list[str] = field(default_factory=list)
    loc: Optional[SourceLocation] = None


@dataclass
class SurfaceMember:
    name: str
    value: Expr


# --- Geometry declarations ---

@dataclass
class GeometryDecl:
    name: str
    fields: list[GeometryField]
    transform: Optional[GeometryTransform] = None
    outputs: Optional[GeometryOutputs] = None
    loc: Optional[SourceLocation] = None


@dataclass
class GeometryField:
    name: str
    type_name: str


@dataclass
class GeometryTransform:
    name: str
    fields: list[BlockField]


@dataclass
class GeometryOutputs:
    bindings: list[OutputBinding]


@dataclass
class OutputBinding:
    name: str
    value: Expr


# --- Pipeline declarations ---

@dataclass
class PipelineDecl:
    name: str
    members: list[PipelineMember]
    loc: Optional[SourceLocation] = None


@dataclass
class PipelineMember:
    name: str
    value: Expr


# --- Schedule declarations ---

@dataclass
class ScheduleDecl:
    name: str
    members: list[ScheduleMember]
    loc: Optional[SourceLocation] = None


@dataclass
class ScheduleMember:
    name: str
    value: str


# --- Environment declarations (RT miss shader) ---

@dataclass
class EnvironmentDecl:
    name: str
    members: list[SurfaceMember]
    samplers: list[str] = field(default_factory=list)
    loc: Optional[SourceLocation] = None


# --- Procedural declarations (RT intersection shader) ---

@dataclass
class ProceduralDecl:
    name: str
    members: list[ProceduralMember]
    loc: Optional[SourceLocation] = None


@dataclass
class ProceduralMember:
    name: str
    value: Expr


# --- Stage blocks ---

@dataclass
class StageBlock:
    stage_type: str  # "vertex", "fragment", "raygen", "closest_hit", "any_hit", "miss", "intersection", "callable"
    inputs: list[VarDecl] = field(default_factory=list)
    outputs: list[VarDecl] = field(default_factory=list)
    uniforms: list[UniformBlock] = field(default_factory=list)
    push_constants: list[PushBlock] = field(default_factory=list)
    samplers: list[SamplerDecl] = field(default_factory=list)
    functions: list[FunctionDef] = field(default_factory=list)
    ray_payloads: list[RayPayloadDecl] = field(default_factory=list)
    hit_attributes: list[HitAttributeDecl] = field(default_factory=list)
    callable_data: list[CallableDataDecl] = field(default_factory=list)
    accel_structs: list[AccelDecl] = field(default_factory=list)
    storage_images: list[StorageImageDecl] = field(default_factory=list)
    loc: Optional[SourceLocation] = None


@dataclass
class VarDecl:
    name: str
    type_name: str
    location: Optional[int] = None  # assigned later
    loc: Optional[SourceLocation] = None


@dataclass
class UniformBlock:
    name: str
    fields: list[BlockField]
    set_number: Optional[int] = None
    binding: Optional[int] = None
    loc: Optional[SourceLocation] = None


@dataclass
class PushBlock:
    name: str
    fields: list[BlockField]
    loc: Optional[SourceLocation] = None


@dataclass
class BlockField:
    name: str
    type_name: str


@dataclass
class SamplerDecl:
    name: str
    type_name: str = "sampler2d"           # "sampler2d" or "samplerCube"
    set_number: Optional[int] = None
    binding: Optional[int] = None          # sampler state binding
    texture_binding: Optional[int] = None  # texture image binding
    loc: Optional[SourceLocation] = None


# --- RT-specific declarations ---

@dataclass
class RayPayloadDecl:
    name: str
    type_name: str
    location: Optional[int] = None
    loc: Optional[SourceLocation] = None


@dataclass
class HitAttributeDecl:
    name: str
    type_name: str
    loc: Optional[SourceLocation] = None


@dataclass
class CallableDataDecl:
    name: str
    type_name: str
    location: Optional[int] = None
    loc: Optional[SourceLocation] = None


@dataclass
class AccelDecl:
    name: str
    set_number: Optional[int] = None
    binding: Optional[int] = None
    loc: Optional[SourceLocation] = None


@dataclass
class StorageImageDecl:
    name: str
    set_number: Optional[int] = None
    binding: Optional[int] = None
    loc: Optional[SourceLocation] = None


# --- Functions ---

@dataclass
class FunctionDef:
    name: str
    params: list[Param]
    return_type: Optional[str]
    body: list[Stmt]
    loc: Optional[SourceLocation] = None
    attributes: list[str] = field(default_factory=list)


@dataclass
class Param:
    name: str
    type_name: str


# --- Statements ---

Stmt = "LetStmt | AssignStmt | ReturnStmt | IfStmt | ExprStmt"


@dataclass
class LetStmt:
    name: str
    type_name: str
    value: Expr
    loc: Optional[SourceLocation] = None


@dataclass
class AssignStmt:
    target: Expr
    value: Expr
    loc: Optional[SourceLocation] = None


@dataclass
class ReturnStmt:
    value: Expr
    loc: Optional[SourceLocation] = None


@dataclass
class IfStmt:
    condition: Expr
    then_body: list
    else_body: list
    loc: Optional[SourceLocation] = None


@dataclass
class ExprStmt:
    expr: Expr
    loc: Optional[SourceLocation] = None


# --- Expressions ---

Expr = "any expression node"


@dataclass
class NumberLit:
    value: str  # keep as string to preserve precision
    loc: Optional[SourceLocation] = None
    resolved_type: Optional[str] = None


@dataclass
class BoolLit:
    value: bool
    loc: Optional[SourceLocation] = None
    resolved_type: Optional[str] = None


@dataclass
class VarRef:
    name: str
    loc: Optional[SourceLocation] = None
    resolved_type: Optional[str] = None


@dataclass
class BinaryOp:
    op: str
    left: Expr
    right: Expr
    loc: Optional[SourceLocation] = None
    resolved_type: Optional[str] = None


@dataclass
class UnaryOp:
    op: str
    operand: Expr
    loc: Optional[SourceLocation] = None
    resolved_type: Optional[str] = None


@dataclass
class CallExpr:
    func: Expr
    args: list[Expr]
    loc: Optional[SourceLocation] = None
    resolved_type: Optional[str] = None


@dataclass
class ConstructorExpr:
    type_name: str
    args: list[Expr]
    loc: Optional[SourceLocation] = None
    resolved_type: Optional[str] = None


@dataclass
class FieldAccess:
    object: Expr
    field: str
    loc: Optional[SourceLocation] = None
    resolved_type: Optional[str] = None


@dataclass
class SwizzleAccess:
    object: Expr
    components: str  # e.g. "xyz", "rg"
    loc: Optional[SourceLocation] = None
    resolved_type: Optional[str] = None


@dataclass
class IndexAccess:
    object: Expr
    index: Expr
    loc: Optional[SourceLocation] = None
    resolved_type: Optional[str] = None


@dataclass
class TernaryExpr:
    condition: Expr
    then_expr: Expr
    else_expr: Expr
    loc: Optional[SourceLocation] = None
    resolved_type: Optional[str] = None


@dataclass
class AssignTarget:
    """Wrapper for assignment LHS — can be a VarRef, FieldAccess, SwizzleAccess, or IndexAccess."""
    expr: Expr
