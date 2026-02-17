"""Type checker for Lux programs."""

from __future__ import annotations
from luxc.parser.ast_nodes import (
    Module, StageBlock, FunctionDef, LetStmt, AssignStmt, ReturnStmt,
    IfStmt, ExprStmt, NumberLit, BoolLit, VarRef, BinaryOp, UnaryOp,
    CallExpr, ConstructorExpr, FieldAccess, SwizzleAccess, IndexAccess,
    TernaryExpr, AssignTarget,
    TypeAlias, ImportDecl, SurfaceDecl, GeometryDecl, PipelineDecl,
)
from luxc.builtins.types import (
    LuxType, resolve_type, TYPE_MAP, SCALAR, BOOL, VOID,
    VEC2, VEC3, VEC4, MAT2, MAT3, MAT4,
    VectorType, MatrixType, ScalarType, is_numeric,
    register_type_alias, clear_type_aliases,
)
from luxc.builtins.functions import lookup_builtin
from luxc.analysis.symbols import Scope, Symbol


class TypeCheckError(Exception):
    pass


def type_check(module: Module) -> None:
    checker = TypeChecker(module)
    checker.check()


class TypeChecker:
    def __init__(self, module: Module):
        self.module = module
        self.global_scope = Scope()
        self.current_scope: Scope = self.global_scope
        self.current_stage: StageBlock | None = None
        self._register_type_aliases()
        self._register_constants()

    def _register_type_aliases(self):
        for ta in self.module.type_aliases:
            # Verify the target type resolves
            t = resolve_type(ta.target_type)
            if t is None:
                raise TypeCheckError(f"Unknown target type '{ta.target_type}' in type alias '{ta.name}'")
            register_type_alias(ta.name, ta.target_type)

    def _register_constants(self):
        for c in self.module.constants:
            t = resolve_type(c.type_name)
            if t is None:
                raise TypeCheckError(f"Unknown type '{c.type_name}' for constant '{c.name}'")
            self.global_scope.define(Symbol(c.name, t, "constant"))

    def check(self):
        for fn in self.module.functions:
            self._check_function(fn, self.global_scope)
        for stage in self.module.stages:
            self._check_stage(stage)

    def _check_stage(self, stage: StageBlock):
        self.current_stage = stage
        scope = self.global_scope.child()

        # Register inputs
        for inp in stage.inputs:
            t = resolve_type(inp.type_name)
            if t is None:
                raise TypeCheckError(f"Unknown type '{inp.type_name}'")
            scope.define(Symbol(inp.name, t, "input"))

        # Register outputs
        for out in stage.outputs:
            t = resolve_type(out.type_name)
            if t is None:
                raise TypeCheckError(f"Unknown type '{out.type_name}'")
            scope.define(Symbol(out.name, t, "output"))

        # Register builtin_position for vertex shaders
        if stage.stage_type == "vertex":
            scope.define(Symbol("builtin_position", VEC4, "builtin_position"))

        # Register uniform block fields as directly accessible
        for ub in stage.uniforms:
            for field in ub.fields:
                t = resolve_type(field.type_name)
                if t is None:
                    raise TypeCheckError(f"Unknown type '{field.type_name}' in uniform block '{ub.name}'")
                scope.define(Symbol(field.name, t, "uniform_field"))

        # Register push constant fields
        for pb in stage.push_constants:
            for field in pb.fields:
                t = resolve_type(field.type_name)
                if t is None:
                    raise TypeCheckError(f"Unknown type '{field.type_name}' in push block '{pb.name}'")
                scope.define(Symbol(field.name, t, "push_field"))

        # Register samplers
        for sam in stage.samplers:
            from luxc.builtins.types import SAMPLER2D
            scope.define(Symbol(sam.name, SAMPLER2D, "sampler"))

        for fn in stage.functions:
            self._check_function(fn, scope)

        self.current_stage = None

    def _check_function(self, fn: FunctionDef, parent_scope: Scope):
        scope = parent_scope.child()
        for p in fn.params:
            t = resolve_type(p.type_name)
            if t is None:
                raise TypeCheckError(f"Unknown param type '{p.type_name}'")
            scope.define(Symbol(p.name, t, "param"))

        ret_type = resolve_type(fn.return_type) if fn.return_type else VOID

        for stmt in fn.body:
            self._check_stmt(stmt, scope, ret_type)

    def _check_stmt(self, stmt, scope: Scope, ret_type: LuxType):
        if isinstance(stmt, LetStmt):
            t = resolve_type(stmt.type_name)
            if t is None:
                raise TypeCheckError(f"Unknown type '{stmt.type_name}'")
            vt = self._check_expr(stmt.value, scope)
            if vt.name != t.name:
                # Allow scalar literal in vector context etc — but for now strict
                if not _implicit_convert(vt, t):
                    raise TypeCheckError(
                        f"Type mismatch in let '{stmt.name}': declared {t.name}, got {vt.name}"
                    )
            scope.define(Symbol(stmt.name, t, "variable"))

        elif isinstance(stmt, AssignStmt):
            target_type = self._check_assign_target(stmt.target, scope)
            val_type = self._check_expr(stmt.value, scope)
            if target_type.name != val_type.name:
                if not _implicit_convert(val_type, target_type):
                    raise TypeCheckError(
                        f"Type mismatch in assignment: target is {target_type.name}, value is {val_type.name}"
                    )

        elif isinstance(stmt, ReturnStmt):
            vt = self._check_expr(stmt.value, scope)
            if ret_type != VOID and vt.name != ret_type.name:
                raise TypeCheckError(f"Return type mismatch: expected {ret_type.name}, got {vt.name}")

        elif isinstance(stmt, IfStmt):
            ct = self._check_expr(stmt.condition, scope)
            if not isinstance(ct, type(BOOL)):
                pass  # Allow non-bool conditions for now
            for s in stmt.then_body:
                self._check_stmt(s, scope, ret_type)
            for s in stmt.else_body:
                self._check_stmt(s, scope, ret_type)

        elif isinstance(stmt, ExprStmt):
            self._check_expr(stmt.expr, scope)

    def _check_assign_target(self, target, scope: Scope) -> LuxType:
        if isinstance(target, AssignTarget):
            return self._check_expr(target.expr, scope)
        return self._check_expr(target, scope)

    def _check_expr(self, expr, scope: Scope) -> LuxType:
        if isinstance(expr, NumberLit):
            # If it has a decimal point, it's a scalar; otherwise could be int
            # For simplicity, all numeric literals are scalar in v1
            expr.resolved_type = "scalar"
            return SCALAR

        elif isinstance(expr, BoolLit):
            expr.resolved_type = "bool"
            return BOOL

        elif isinstance(expr, VarRef):
            sym = scope.lookup(expr.name)
            if sym is None:
                raise TypeCheckError(f"Undefined variable '{expr.name}'")
            expr.resolved_type = sym.type.name
            return sym.type

        elif isinstance(expr, BinaryOp):
            lt = self._check_expr(expr.left, scope)
            rt = self._check_expr(expr.right, scope)
            result = _check_binary_op(expr.op, lt, rt)
            expr.resolved_type = result.name
            return result

        elif isinstance(expr, UnaryOp):
            ot = self._check_expr(expr.operand, scope)
            if expr.op == "-":
                if not is_numeric(ot):
                    raise TypeCheckError(f"Cannot negate type {ot.name}")
                expr.resolved_type = ot.name
                return ot
            elif expr.op == "!":
                expr.resolved_type = "bool"
                return BOOL
            return ot

        elif isinstance(expr, CallExpr):
            # func could be a VarRef to a built-in function name
            if isinstance(expr.func, VarRef):
                fname = expr.func.name
                arg_types = [self._check_expr(a, scope) for a in expr.args]
                sig = lookup_builtin(fname, arg_types)
                if sig is None:
                    # Check user-defined functions
                    fn = self._find_user_function(fname)
                    if fn is not None:
                        ret = resolve_type(fn.return_type) if fn.return_type else VOID
                        expr.resolved_type = ret.name
                        return ret
                    raise TypeCheckError(
                        f"No matching overload for '{fname}' with args ({', '.join(t.name for t in arg_types)})"
                    )
                expr.resolved_type = sig.return_type.name
                return sig.return_type
            # Method-style calls not supported in v1
            arg_types = [self._check_expr(a, scope) for a in expr.args]
            raise TypeCheckError(f"Unsupported call expression")

        elif isinstance(expr, ConstructorExpr):
            arg_types = [self._check_expr(a, scope) for a in expr.args]
            t = resolve_type(expr.type_name)
            if t is None:
                raise TypeCheckError(f"Unknown constructor type '{expr.type_name}'")
            # Validate constructor args (permissive for v1)
            expr.resolved_type = t.name
            return t

        elif isinstance(expr, FieldAccess):
            obj_type = self._check_expr(expr.object, scope)
            # For uniform block field access, etc.
            expr.resolved_type = SCALAR.name  # simplified
            return SCALAR

        elif isinstance(expr, SwizzleAccess):
            obj_type = self._check_expr(expr.object, scope)
            n = len(expr.components)
            if n == 1:
                result = SCALAR
            else:
                result = {2: VEC2, 3: VEC3, 4: VEC4}.get(n)
                if result is None:
                    raise TypeCheckError(f"Invalid swizzle length {n}")
            expr.resolved_type = result.name
            return result

        elif isinstance(expr, IndexAccess):
            obj_type = self._check_expr(expr.object, scope)
            self._check_expr(expr.index, scope)
            # mat[i] -> vecN, vec[i] -> scalar
            if isinstance(obj_type, MatrixType):
                result = {2: VEC2, 3: VEC3, 4: VEC4}[obj_type.size]
            elif isinstance(obj_type, VectorType):
                result = resolve_type(obj_type.component)
            else:
                result = SCALAR
            expr.resolved_type = result.name
            return result

        elif isinstance(expr, TernaryExpr):
            self._check_expr(expr.condition, scope)
            tt = self._check_expr(expr.then_expr, scope)
            et = self._check_expr(expr.else_expr, scope)
            if tt.name != et.name:
                raise TypeCheckError(f"Ternary branches have different types: {tt.name} vs {et.name}")
            expr.resolved_type = tt.name
            return tt

        raise TypeCheckError(f"Unknown expression type: {type(expr).__name__}")

    def _find_user_function(self, name: str) -> FunctionDef | None:
        for fn in self.module.functions:
            if fn.name == name:
                return fn
        if self.current_stage:
            for fn in self.current_stage.functions:
                if fn.name == name:
                    return fn
        return None


def _check_binary_op(op: str, left: LuxType, right: LuxType) -> LuxType:
    if op in ("==", "!=", "<", ">", "<=", ">="):
        return BOOL
    if op in ("&&", "||"):
        return BOOL

    # Arithmetic: +, -, *, /, %
    if op in ("+", "-", "/", "%"):
        if left.name == right.name and is_numeric(left):
            return left
        # scalar op vec -> vec, vec op scalar -> vec
        if isinstance(left, ScalarType) and isinstance(right, VectorType):
            return right
        if isinstance(left, VectorType) and isinstance(right, ScalarType):
            return left
        raise TypeCheckError(f"Cannot apply '{op}' to {left.name} and {right.name}")

    if op == "*":
        # Same types
        if left.name == right.name and is_numeric(left):
            return left
        # scalar * vec -> vec
        if isinstance(left, ScalarType) and isinstance(right, VectorType):
            return right
        if isinstance(left, VectorType) and isinstance(right, ScalarType):
            return left
        # mat * vec -> vec
        if isinstance(left, MatrixType) and isinstance(right, VectorType):
            return right
        # mat * mat -> mat
        if isinstance(left, MatrixType) and isinstance(right, MatrixType):
            return left
        # vec * mat -> vec
        if isinstance(left, VectorType) and isinstance(right, MatrixType):
            return left
        # scalar * mat -> mat
        if isinstance(left, ScalarType) and isinstance(right, MatrixType):
            return right
        if isinstance(left, MatrixType) and isinstance(right, ScalarType):
            return left
        raise TypeCheckError(f"Cannot apply '*' to {left.name} and {right.name}")

    raise TypeCheckError(f"Unknown operator '{op}'")


def _implicit_convert(from_type: LuxType, to_type: LuxType) -> bool:
    # No implicit conversions in v1
    return False
