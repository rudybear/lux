"""Static analysis pass to warn about potential NaN/Inf-producing operations."""

from __future__ import annotations
import warnings
from luxc.parser.ast_nodes import (
    CallExpr, BinaryOp, FunctionDef, IfStmt, LetStmt, ReturnStmt,
    AssignStmt, DebugBlock, DebugPrintStmt, AssertStmt, ExprStmt,
    VarRef, NumberLit,
)


def check_nan_warnings(module) -> None:
    """Walk all functions and stage blocks, emitting warnings for risky float ops."""
    # Walk module-level functions
    for fn in module.functions:
        _check_function(fn)
    # Walk stage block functions
    for stage in module.stages:
        for fn in stage.functions:
            _check_function(fn)


def _check_function(fn: FunctionDef) -> None:
    for stmt in fn.body:
        _check_stmt(stmt)


def _check_stmt(stmt) -> None:
    """Recursively check a statement for risky operations."""
    if isinstance(stmt, LetStmt):
        _check_expr(stmt.value, _loc(stmt))
    elif isinstance(stmt, ReturnStmt):
        _check_expr(stmt.value, _loc(stmt))
    elif isinstance(stmt, AssignStmt):
        _check_expr(stmt.value, _loc(stmt))
    elif isinstance(stmt, IfStmt):
        _check_expr(stmt.condition, _loc(stmt))
        for s in stmt.then_body:
            _check_stmt(s)
        for s in stmt.else_body:
            _check_stmt(s)
    elif isinstance(stmt, DebugBlock):
        for s in stmt.body:
            _check_stmt(s)
    elif isinstance(stmt, DebugPrintStmt):
        for arg in stmt.args:
            _check_expr(arg, _loc(stmt))
    elif isinstance(stmt, AssertStmt):
        _check_expr(stmt.condition, _loc(stmt))
    elif isinstance(stmt, ExprStmt):
        _check_expr(stmt.expr, _loc(stmt))


def _check_expr(expr, loc: str) -> None:
    """Check an expression tree for risky float operations."""
    if expr is None:
        return

    if isinstance(expr, BinaryOp):
        if expr.op == "/":
            if not _is_guarded_denominator(expr.right):
                _warn(loc, "division by potentially zero value — consider using max(x, epsilon)")
        _check_expr(expr.left, loc)
        _check_expr(expr.right, loc)

    elif isinstance(expr, CallExpr):
        fname = _get_call_name(expr)
        if fname == "sqrt" and len(expr.args) == 1:
            if not _is_guarded_non_negative(expr.args[0]):
                _warn(loc, "sqrt() of potentially negative value — consider max(x, 0.0)")
        elif fname == "normalize" and len(expr.args) == 1:
            _warn(loc, "normalize() of potentially zero-length vector — consider length check")
        elif fname == "pow" and len(expr.args) == 2:
            _warn(loc, "pow() with potentially negative base — may produce NaN")
        elif fname in ("log", "log2") and len(expr.args) == 1:
            if not _is_guarded_positive(expr.args[0]):
                _warn(loc, f"{fname}() of potentially non-positive value — consider max(x, epsilon)")
        elif fname == "inversesqrt" and len(expr.args) == 1:
            if not _is_guarded_positive(expr.args[0]):
                _warn(loc, "inversesqrt() of potentially non-positive value")
        # Recurse into call args
        for arg in expr.args:
            _check_expr(arg, loc)

    # Recurse into sub-expressions for other node types
    elif hasattr(expr, 'left') and hasattr(expr, 'right'):
        _check_expr(expr.left, loc)
        _check_expr(expr.right, loc)
    elif hasattr(expr, 'operand'):
        _check_expr(expr.operand, loc)
    elif hasattr(expr, 'condition') and hasattr(expr, 'then_expr'):
        # TernaryExpr
        _check_expr(expr.condition, loc)
        _check_expr(expr.then_expr, loc)
        _check_expr(expr.else_expr, loc)
    elif hasattr(expr, 'object') and hasattr(expr, 'field'):
        # FieldAccess
        _check_expr(expr.object, loc)
    elif hasattr(expr, 'object') and hasattr(expr, 'components'):
        # SwizzleAccess
        _check_expr(expr.object, loc)
    elif hasattr(expr, 'object') and hasattr(expr, 'index'):
        # IndexAccess
        _check_expr(expr.object, loc)
        _check_expr(expr.index, loc)


def _get_call_name(expr: CallExpr) -> str | None:
    """Extract the function name from a CallExpr node."""
    func = expr.func
    if isinstance(func, VarRef):
        return func.name
    # For method-style calls or other patterns, we can't easily resolve the name
    if hasattr(func, 'func_name'):
        return func.func_name
    if hasattr(func, 'name'):
        return func.name
    return None


def _is_guarded_denominator(expr) -> bool:
    """Check if expression is guarded against zero (e.g., max(x, 0.001))."""
    if isinstance(expr, CallExpr):
        fname = _get_call_name(expr)
        if fname == "max" and len(expr.args) == 2:
            # max(x, literal) where literal > 0
            if _is_positive_literal(expr.args[1]) or _is_positive_literal(expr.args[0]):
                return True
        if fname == "abs":
            return False  # abs(x) can still be 0
        if fname == "clamp" and len(expr.args) == 3:
            # clamp(x, min, max) where min > 0
            if _is_positive_literal(expr.args[1]):
                return True
    # Literal number > 0
    if _is_positive_literal(expr):
        return True
    return False


def _is_guarded_non_negative(expr) -> bool:
    """Check if expression is guaranteed non-negative."""
    if isinstance(expr, CallExpr):
        fname = _get_call_name(expr)
        if fname == "max" and len(expr.args) == 2:
            if _is_non_negative_literal(expr.args[1]) or _is_non_negative_literal(expr.args[0]):
                return True
        if fname == "abs":
            return True
        if fname == "clamp" and len(expr.args) == 3:
            if _is_non_negative_literal(expr.args[1]):
                return True
        if fname == "dot":
            # dot(x, x) is always non-negative — conservative: treat all dot as non-negative
            return True
    if _is_non_negative_literal(expr):
        return True
    return False


def _is_guarded_positive(expr) -> bool:
    """Check if expression is guaranteed positive (> 0)."""
    return _is_guarded_denominator(expr)


def _is_positive_literal(expr) -> bool:
    """Check if expr is a numeric literal > 0."""
    if isinstance(expr, NumberLit):
        try:
            return float(expr.value) > 0
        except (ValueError, TypeError):
            return False
    return False


def _is_non_negative_literal(expr) -> bool:
    """Check if expr is a numeric literal >= 0."""
    if isinstance(expr, NumberLit):
        try:
            return float(expr.value) >= 0
        except (ValueError, TypeError):
            return False
    return False


def _loc(node) -> str:
    """Get location string from a node."""
    if hasattr(node, 'loc') and node.loc is not None:
        loc = node.loc
        if hasattr(loc, 'file') and loc.file is not None:
            return f"{loc.file}:{loc.line}"
        return f"<unknown>:{loc.line}"
    return "<unknown>"


def _warn(loc: str, msg: str) -> None:
    warnings.warn(f"[NaN warning] {loc}: {msg}", stacklevel=3)
