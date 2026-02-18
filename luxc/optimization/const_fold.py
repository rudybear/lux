"""AST-level constant folding pass.

Walks all functions in the module and folds constant expressions:
- Arithmetic on literals (1.0 + 2.0 -> 3.0)
- Const variable inlining (const PI -> literal value)
- Known builtin calls on literals (sin(0.0) -> 0.0)
- Dead branch elimination (if(true) -> taken branch)
"""

from __future__ import annotations
import math
from luxc.parser.ast_nodes import (
    Module, FunctionDef, ConstDecl,
    LetStmt, AssignStmt, ReturnStmt, IfStmt, ExprStmt,
    NumberLit, BoolLit, VarRef, BinaryOp, UnaryOp, CallExpr,
    ConstructorExpr, FieldAccess, SwizzleAccess, IndexAccess, TernaryExpr,
)

# Builtin functions that can be folded when all args are literal floats.
_FOLD_BUILTINS_1 = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "exp": math.exp,
    "exp2": lambda x: 2.0 ** x,
    "log": math.log,
    "log2": math.log2,
    "sqrt": math.sqrt,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "fract": lambda x: x - math.floor(x),
    "sign": lambda x: 0.0 if x == 0.0 else (1.0 if x > 0 else -1.0),
}

_FOLD_BUILTINS_2 = {
    "min": min,
    "max": max,
    "pow": math.pow,
    "step": lambda edge, x: 0.0 if x < edge else 1.0,
}

_FOLD_BUILTINS_3 = {
    "mix": lambda x, y, a: x * (1.0 - a) + y * a,
    "clamp": lambda x, lo, hi: max(lo, min(x, hi)),
    "smoothstep": lambda e0, e1, x: (
        0.0 if x <= e0 else (
            1.0 if x >= e1 else (
                (lambda t: t * t * (3.0 - 2.0 * t))((x - e0) / (e1 - e0))
            )
        )
    ),
}


def _format_float(val: float) -> str:
    """Format a float ensuring a decimal point is preserved."""
    s = repr(val)
    if "." not in s and "e" not in s and "E" not in s and "inf" not in s.lower() and "nan" not in s.lower():
        s += ".0"
    return s


def _as_float(node) -> float | None:
    """Extract float value from a NumberLit, or None."""
    if isinstance(node, NumberLit):
        try:
            return float(node.value)
        except ValueError:
            return None
    return None


def _as_bool(node) -> bool | None:
    """Extract bool value from a BoolLit, or None."""
    if isinstance(node, BoolLit):
        return node.value
    return None


def _make_number(val: float, resolved_type: str | None = None) -> NumberLit:
    """Create a NumberLit from a computed float value."""
    return NumberLit(_format_float(val), resolved_type=resolved_type or "scalar")


def constant_fold(module: Module) -> None:
    """Run constant folding on all functions in the module."""
    # Build const environment from module-level constants
    const_env: dict[str, NumberLit | BoolLit] = {}
    for c in module.constants:
        if isinstance(c.value, NumberLit):
            const_env[c.name] = c.value
        elif isinstance(c.value, BoolLit):
            const_env[c.name] = c.value

    # Fold constants in all functions (module-level and stage-level)
    all_functions = list(module.functions)
    for stage in module.stages:
        all_functions.extend(stage.functions)

    for func in all_functions:
        func.body = _fold_stmts(func.body, const_env)


def _fold_stmts(stmts: list, const_env: dict) -> list:
    """Fold constants in a list of statements, with dead branch elimination."""
    result = []
    for stmt in stmts:
        if isinstance(stmt, LetStmt):
            stmt.value = _try_fold_expr(stmt.value, const_env)
            result.append(stmt)
        elif isinstance(stmt, AssignStmt):
            stmt.value = _try_fold_expr(stmt.value, const_env)
            result.append(stmt)
        elif isinstance(stmt, ReturnStmt):
            stmt.value = _try_fold_expr(stmt.value, const_env)
            result.append(stmt)
        elif isinstance(stmt, ExprStmt):
            stmt.expr = _try_fold_expr(stmt.expr, const_env)
            result.append(stmt)
        elif isinstance(stmt, IfStmt):
            stmt.condition = _try_fold_expr(stmt.condition, const_env)
            cond_val = _as_bool(stmt.condition)
            if cond_val is True:
                # Dead branch elimination: take then branch
                result.extend(_fold_stmts(stmt.then_body, const_env))
            elif cond_val is False:
                # Dead branch elimination: take else branch
                result.extend(_fold_stmts(stmt.else_body, const_env))
            else:
                stmt.then_body = _fold_stmts(stmt.then_body, const_env)
                stmt.else_body = _fold_stmts(stmt.else_body, const_env)
                result.append(stmt)
        else:
            result.append(stmt)
    return result


def _try_fold_expr(expr, const_env: dict):
    """Try to fold an expression to a literal. Returns the folded or original expr."""
    if isinstance(expr, (NumberLit, BoolLit)):
        return expr

    if isinstance(expr, VarRef):
        if expr.name in const_env:
            lit = const_env[expr.name]
            if isinstance(lit, NumberLit):
                return NumberLit(lit.value, resolved_type=expr.resolved_type or lit.resolved_type)
            if isinstance(lit, BoolLit):
                return BoolLit(lit.value, resolved_type=expr.resolved_type or lit.resolved_type)
        return expr

    if isinstance(expr, UnaryOp):
        expr.operand = _try_fold_expr(expr.operand, const_env)
        if expr.op == "-":
            val = _as_float(expr.operand)
            if val is not None:
                return _make_number(-val, expr.resolved_type or expr.operand.resolved_type)
        elif expr.op == "!":
            bval = _as_bool(expr.operand)
            if bval is not None:
                return BoolLit(not bval, resolved_type=expr.resolved_type)
        return expr

    if isinstance(expr, BinaryOp):
        expr.left = _try_fold_expr(expr.left, const_env)
        expr.right = _try_fold_expr(expr.right, const_env)
        lval = _as_float(expr.left)
        rval = _as_float(expr.right)
        if lval is not None and rval is not None:
            result = _fold_binary(expr.op, lval, rval)
            if result is not None:
                if isinstance(result, bool):
                    return BoolLit(result, resolved_type=expr.resolved_type or "bool")
                return _make_number(result, expr.resolved_type or expr.left.resolved_type)
        return expr

    if isinstance(expr, CallExpr):
        if isinstance(expr.func, VarRef):
            expr.args = [_try_fold_expr(a, const_env) for a in expr.args]
            fname = expr.func.name
            float_args = [_as_float(a) for a in expr.args]
            if all(v is not None for v in float_args):
                folded = _fold_builtin(fname, float_args)
                if folded is not None:
                    return _make_number(folded, expr.resolved_type)
        else:
            expr.args = [_try_fold_expr(a, const_env) for a in expr.args]
        return expr

    if isinstance(expr, ConstructorExpr):
        expr.args = [_try_fold_expr(a, const_env) for a in expr.args]
        return expr

    if isinstance(expr, TernaryExpr):
        expr.condition = _try_fold_expr(expr.condition, const_env)
        cond_val = _as_bool(expr.condition)
        if cond_val is True:
            return _try_fold_expr(expr.then_expr, const_env)
        elif cond_val is False:
            return _try_fold_expr(expr.else_expr, const_env)
        expr.then_expr = _try_fold_expr(expr.then_expr, const_env)
        expr.else_expr = _try_fold_expr(expr.else_expr, const_env)
        return expr

    if isinstance(expr, SwizzleAccess):
        expr.object = _try_fold_expr(expr.object, const_env)
        return expr

    if isinstance(expr, FieldAccess):
        expr.object = _try_fold_expr(expr.object, const_env)
        return expr

    if isinstance(expr, IndexAccess):
        expr.object = _try_fold_expr(expr.object, const_env)
        expr.index = _try_fold_expr(expr.index, const_env)
        return expr

    return expr


def _fold_binary(op: str, lval: float, rval: float):
    """Fold a binary operation on two float literals. Returns float, bool, or None."""
    try:
        if op == "+":
            return lval + rval
        elif op == "-":
            return lval - rval
        elif op == "*":
            return lval * rval
        elif op == "/":
            if rval == 0.0:
                return None  # safety: leave unfolded
            return lval / rval
        elif op == "%":
            if rval == 0.0:
                return None
            return lval % rval
        elif op == "==":
            return lval == rval
        elif op == "!=":
            return lval != rval
        elif op == "<":
            return lval < rval
        elif op == ">":
            return lval > rval
        elif op == "<=":
            return lval <= rval
        elif op == ">=":
            return lval >= rval
    except (ArithmeticError, ValueError):
        return None
    return None


def _fold_builtin(fname: str, args: list[float]) -> float | None:
    """Try to fold a known builtin call. Returns float or None."""
    try:
        if len(args) == 1 and fname in _FOLD_BUILTINS_1:
            return _FOLD_BUILTINS_1[fname](args[0])
        if len(args) == 2 and fname in _FOLD_BUILTINS_2:
            return _FOLD_BUILTINS_2[fname](args[0], args[1])
        if len(args) == 3 and fname in _FOLD_BUILTINS_3:
            return _FOLD_BUILTINS_3[fname](args[0], args[1], args[2])
    except (ArithmeticError, ValueError):
        return None  # domain error — leave unfolded
    return None
