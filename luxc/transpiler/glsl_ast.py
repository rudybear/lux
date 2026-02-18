"""Lightweight GLSL AST nodes used by the transpiler."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Module-level
# ---------------------------------------------------------------------------

@dataclass
class GlslModule:
    version: Optional[int] = None
    inputs: list[GlslInOut] = field(default_factory=list)
    outputs: list[GlslInOut] = field(default_factory=list)
    uniforms: list[GlslUniform] = field(default_factory=list)
    globals: list[GlslGlobalVar] = field(default_factory=list)
    functions: list[GlslFunction] = field(default_factory=list)


@dataclass
class GlslInOut:
    qualifier: str  # "in" or "out"
    type_name: str
    name: str
    layout: Optional[int] = None


@dataclass
class GlslUniform:
    type_name: str
    name: str


@dataclass
class GlslGlobalVar:
    type_name: str
    name: str
    value: Optional[object] = None  # expression node
    is_const: bool = False


@dataclass
class GlslFunction:
    return_type: str
    name: str
    params: list[GlslParam] = field(default_factory=list)
    body: list = field(default_factory=list)  # list of statement nodes
    has_unsupported: bool = False


@dataclass
class GlslParam:
    type_name: str
    name: str
    qualifier: Optional[str] = None  # "in", "out", "inout"


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass
class GlslVarDecl:
    type_name: str
    name: str
    value: Optional[object] = None


@dataclass
class GlslAssign:
    target: object
    value: object


@dataclass
class GlslCompoundAssign:
    target: object
    op: str  # "+=", "-=", etc.
    value: object


@dataclass
class GlslReturn:
    value: Optional[object] = None


@dataclass
class GlslIf:
    condition: object
    then_body: list = field(default_factory=list)
    else_body: list = field(default_factory=list)


@dataclass
class GlslFor:
    """Represents a for loop (flagged as unsupported in output)."""
    pass


@dataclass
class GlslWhile:
    """Represents a while loop (flagged as unsupported in output)."""
    pass


@dataclass
class GlslIncrDecr:
    """Represents ++/-- (flagged as unsupported in output)."""
    target: object
    op: str


@dataclass
class GlslExprStmt:
    expr: object


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

@dataclass
class GlslNumberLit:
    value: str


@dataclass
class GlslBoolLit:
    value: bool


@dataclass
class GlslVarRef:
    name: str


@dataclass
class GlslBinaryOp:
    op: str
    left: object
    right: object


@dataclass
class GlslUnaryOp:
    op: str
    operand: object


@dataclass
class GlslCall:
    func: str
    args: list = field(default_factory=list)


@dataclass
class GlslFieldAccess:
    obj: object
    field: str


@dataclass
class GlslSwizzle:
    obj: object
    components: str


@dataclass
class GlslIndex:
    obj: object
    index: object


@dataclass
class GlslTernary:
    condition: object
    then_expr: object
    else_expr: object
