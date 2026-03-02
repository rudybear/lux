"""AST-level function inlining pass.

Inlines user-defined function calls into the call site before CSE runs,
allowing CSE to catch duplicate expressions that result from inlining
the same function multiple times (e.g., dot(N, Lo) inlined 4 times).

This pass runs in release mode only. In debug mode, codegen-time inlining
is used instead to preserve call structure for debugger step-through.
"""

from __future__ import annotations
import copy
from typing import Any

from luxc.parser.ast_nodes import (
    Module, FunctionDef, StageBlock, Param,
    LetStmt, AssignStmt, ReturnStmt, IfStmt, ExprStmt,
    NumberLit, BoolLit, VarRef, BinaryOp, UnaryOp, CallExpr,
    ConstructorExpr, FieldAccess, SwizzleAccess, IndexAccess, TernaryExpr,
    ForStmt, WhileStmt, BreakStmt, ContinueStmt,
    DebugPrintStmt, AssertStmt, DebugBlock, AssignTarget,
)

_counter = [0]


def _unique_prefix() -> str:
    _counter[0] += 1
    return f"_inl{_counter[0]}"


def inline_functions(module: Module) -> None:
    """Inline user-defined functions at the AST level.

    Processes each stage's main function body, replacing CallExpr nodes
    to user-defined functions with their inlined body.
    """
    _counter[0] = 0

    # Build function lookup (module-level + per-stage)
    module_fns: dict[str, FunctionDef] = {}
    for fn in module.functions:
        module_fns[fn.name] = fn

    # Deep-copy function bodies before inlining to avoid aliasing:
    # pipeline expansion may share expression objects across stages,
    # so in-place modifications in one stage can corrupt another.
    for stage in module.stages:
        for fn in stage.functions:
            fn.body = copy.deepcopy(fn.body)

    for stage in module.stages:
        stage_fns = dict(module_fns)
        for fn in stage.functions:
            if fn.name != "main":
                stage_fns[fn.name] = fn

        # Inline in non-main stage functions first (they might call each other)
        for fn in stage.functions:
            if fn.name != "main":
                fn.body = _inline_body(fn.body, stage_fns, fn.name)

        # Then inline in main
        for fn in stage.functions:
            if fn.name == "main":
                fn.body = _inline_body(fn.body, stage_fns, fn.name)

    # Also inline within module-level functions (they call each other)
    for fn in module.functions:
        fn.body = _inline_body(fn.body, module_fns, fn.name)


def _inline_body(
    stmts: list, fn_map: dict[str, FunctionDef], current_fn: str
) -> list:
    """Process a statement list, inlining function calls."""
    result = []
    for stmt in stmts:
        result.extend(_inline_stmt(stmt, fn_map, current_fn))
    return result


def _inline_stmt(
    stmt, fn_map: dict[str, FunctionDef], current_fn: str
) -> list:
    """Inline function calls within a statement. Returns a list of statements
    (may expand a single statement into multiple when inlining)."""

    if isinstance(stmt, LetStmt):
        expanded, new_value = _inline_expr(stmt.value, fn_map, current_fn)
        stmt.value = new_value
        return expanded + [stmt]

    elif isinstance(stmt, AssignStmt):
        expanded, new_value = _inline_expr(stmt.value, fn_map, current_fn)
        stmt.value = new_value
        return expanded + [stmt]

    elif isinstance(stmt, ReturnStmt):
        expanded, new_value = _inline_expr(stmt.value, fn_map, current_fn)
        stmt.value = new_value
        return expanded + [stmt]

    elif isinstance(stmt, ExprStmt):
        expanded, new_expr = _inline_expr(stmt.expr, fn_map, current_fn)
        stmt.expr = new_expr
        return expanded + [stmt]

    elif isinstance(stmt, IfStmt):
        expanded, new_cond = _inline_expr(stmt.condition, fn_map, current_fn)
        stmt.condition = new_cond
        stmt.then_body = _inline_body(stmt.then_body, fn_map, current_fn)
        stmt.else_body = _inline_body(stmt.else_body, fn_map, current_fn)
        return expanded + [stmt]

    elif isinstance(stmt, ForStmt):
        # Inline in init value
        exp_init, new_init = _inline_expr(stmt.init_value, fn_map, current_fn)
        stmt.init_value = new_init
        # Inline in condition
        exp_cond, new_cond = _inline_expr(stmt.condition, fn_map, current_fn)
        stmt.condition = new_cond
        # Inline in update value
        exp_upd, new_upd = _inline_expr(stmt.update_value, fn_map, current_fn)
        stmt.update_value = new_upd
        # Inline in body
        stmt.body = _inline_body(stmt.body, fn_map, current_fn)
        return exp_init + exp_cond + exp_upd + [stmt]

    elif isinstance(stmt, WhileStmt):
        exp_cond, new_cond = _inline_expr(stmt.condition, fn_map, current_fn)
        stmt.condition = new_cond
        stmt.body = _inline_body(stmt.body, fn_map, current_fn)
        return exp_cond + [stmt]

    elif isinstance(stmt, DebugBlock):
        stmt.body = _inline_body(stmt.body, fn_map, current_fn)
        return [stmt]

    return [stmt]


def _inline_expr(
    expr, fn_map: dict[str, FunctionDef], current_fn: str
) -> tuple[list, Any]:
    """Process an expression, inlining any CallExpr to user functions.

    Returns (prefix_stmts, new_expr) where prefix_stmts is a list of
    LetStmt nodes that must precede the expression.
    """
    if expr is None:
        return [], expr

    if isinstance(expr, (NumberLit, BoolLit, VarRef)):
        return [], expr

    if isinstance(expr, BinaryOp):
        exp_l, new_l = _inline_expr(expr.left, fn_map, current_fn)
        exp_r, new_r = _inline_expr(expr.right, fn_map, current_fn)
        expr.left = new_l
        expr.right = new_r
        return exp_l + exp_r, expr

    if isinstance(expr, UnaryOp):
        exp, new_op = _inline_expr(expr.operand, fn_map, current_fn)
        expr.operand = new_op
        return exp, expr

    if isinstance(expr, CallExpr):
        # First, inline within arguments
        all_prefix = []
        for i, arg in enumerate(expr.args):
            exp, new_arg = _inline_expr(arg, fn_map, current_fn)
            all_prefix.extend(exp)
            expr.args[i] = new_arg

        # Check if callee is a user-defined function
        if isinstance(expr.func, VarRef):
            fname = expr.func.name
            if fname in fn_map and fname != current_fn:
                fn_def = fn_map[fname]
                # Only inline functions with a single return at the end
                if _is_inlineable(fn_def):
                    prefix, result_expr = _do_inline(fn_def, expr.args, expr.resolved_type)
                    return all_prefix + prefix, result_expr

        return all_prefix, expr

    if isinstance(expr, ConstructorExpr):
        all_prefix = []
        for i, arg in enumerate(expr.args):
            exp, new_arg = _inline_expr(arg, fn_map, current_fn)
            all_prefix.extend(exp)
            expr.args[i] = new_arg
        return all_prefix, expr

    if isinstance(expr, FieldAccess):
        exp, new_obj = _inline_expr(expr.object, fn_map, current_fn)
        expr.object = new_obj
        return exp, expr

    if isinstance(expr, SwizzleAccess):
        exp, new_obj = _inline_expr(expr.object, fn_map, current_fn)
        expr.object = new_obj
        return exp, expr

    if isinstance(expr, IndexAccess):
        exp_obj, new_obj = _inline_expr(expr.object, fn_map, current_fn)
        exp_idx, new_idx = _inline_expr(expr.index, fn_map, current_fn)
        expr.object = new_obj
        expr.index = new_idx
        return exp_obj + exp_idx, expr

    if isinstance(expr, TernaryExpr):
        exp_c, new_c = _inline_expr(expr.condition, fn_map, current_fn)
        exp_t, new_t = _inline_expr(expr.then_expr, fn_map, current_fn)
        exp_e, new_e = _inline_expr(expr.else_expr, fn_map, current_fn)
        expr.condition = new_c
        expr.then_expr = new_t
        expr.else_expr = new_e
        return exp_c + exp_t + exp_e, expr

    return [], expr


def _is_inlineable(fn: FunctionDef) -> bool:
    """Check if a function is suitable for AST-level inlining.

    Currently inlines functions that have a simple structure:
    a sequence of LetStmts followed by a ReturnStmt (with optional IfStmts).
    """
    if not fn.body:
        return False
    # Must end with ReturnStmt
    last = fn.body[-1]
    if not isinstance(last, ReturnStmt):
        return False
    # All preceding stmts must be LetStmt or IfStmt (no loops, breaks, etc.)
    for stmt in fn.body[:-1]:
        if not isinstance(stmt, (LetStmt, IfStmt)):
            return False
    return True


def _do_inline(
    fn: FunctionDef, call_args: list, resolved_type: str | None
) -> tuple[list, Any]:
    """Inline a function call, returning (prefix_stmts, result_expr).

    Creates unique variable names to avoid conflicts and deep-copies
    the function body to avoid aliasing issues.
    """
    prefix = _unique_prefix()
    body = copy.deepcopy(fn.body)

    # Build param name -> arg mapping
    rename_map: dict[str, str] = {}
    prefix_stmts: list = []

    for param, arg in zip(fn.params, call_args):
        # Create a unique let binding for each parameter
        unique_name = f"{prefix}_{param.name}"
        rename_map[param.name] = unique_name
        prefix_stmts.append(LetStmt(
            name=unique_name,
            type_name=param.type_name,
            value=copy.deepcopy(arg),
        ))

    # Rename locals in the body
    for stmt in body:
        if isinstance(stmt, LetStmt):
            old_name = stmt.name
            new_name = f"{prefix}_{old_name}"
            rename_map[old_name] = new_name
            stmt.name = new_name

    # Apply renames to all expressions in the body
    for stmt in body:
        _rename_in_stmt(stmt, rename_map)

    # Collect non-return stmts as prefix, return value as result expr
    for stmt in body[:-1]:
        prefix_stmts.append(stmt)

    # The last stmt is ReturnStmt — its value is the inlined result
    ret_stmt = body[-1]
    assert isinstance(ret_stmt, ReturnStmt)
    result_expr = ret_stmt.value
    if resolved_type and not getattr(result_expr, 'resolved_type', None):
        result_expr.resolved_type = resolved_type

    return prefix_stmts, result_expr


def _rename_in_stmt(stmt, rename_map: dict[str, str]) -> None:
    """Rename variables in a statement according to rename_map."""
    if isinstance(stmt, LetStmt):
        _rename_in_expr(stmt.value, rename_map)
    elif isinstance(stmt, AssignStmt):
        _rename_in_expr(stmt.value, rename_map)
        _rename_in_target(stmt.target, rename_map)
    elif isinstance(stmt, ReturnStmt):
        _rename_in_expr(stmt.value, rename_map)
    elif isinstance(stmt, ExprStmt):
        _rename_in_expr(stmt.expr, rename_map)
    elif isinstance(stmt, IfStmt):
        _rename_in_expr(stmt.condition, rename_map)
        for s in stmt.then_body:
            _rename_in_stmt(s, rename_map)
        for s in stmt.else_body:
            _rename_in_stmt(s, rename_map)
    elif isinstance(stmt, ForStmt):
        if stmt.loop_var in rename_map:
            stmt.loop_var = rename_map[stmt.loop_var]
        _rename_in_expr(stmt.init_value, rename_map)
        _rename_in_expr(stmt.condition, rename_map)
        _rename_in_expr(stmt.update_value, rename_map)
        if isinstance(stmt.update_target, AssignTarget):
            _rename_in_target(stmt.update_target, rename_map)
        for s in stmt.body:
            _rename_in_stmt(s, rename_map)
    elif isinstance(stmt, WhileStmt):
        _rename_in_expr(stmt.condition, rename_map)
        for s in stmt.body:
            _rename_in_stmt(s, rename_map)


def _rename_in_expr(expr, rename_map: dict[str, str]) -> None:
    """Rename VarRef nodes in an expression tree in-place."""
    if expr is None:
        return
    if isinstance(expr, VarRef):
        if expr.name in rename_map:
            expr.name = rename_map[expr.name]
    elif isinstance(expr, BinaryOp):
        _rename_in_expr(expr.left, rename_map)
        _rename_in_expr(expr.right, rename_map)
    elif isinstance(expr, UnaryOp):
        _rename_in_expr(expr.operand, rename_map)
    elif isinstance(expr, CallExpr):
        # Don't rename the function name itself (only local vars)
        for arg in expr.args:
            _rename_in_expr(arg, rename_map)
    elif isinstance(expr, ConstructorExpr):
        for arg in expr.args:
            _rename_in_expr(arg, rename_map)
    elif isinstance(expr, FieldAccess):
        _rename_in_expr(expr.object, rename_map)
    elif isinstance(expr, SwizzleAccess):
        _rename_in_expr(expr.object, rename_map)
    elif isinstance(expr, IndexAccess):
        _rename_in_expr(expr.object, rename_map)
        _rename_in_expr(expr.index, rename_map)
    elif isinstance(expr, TernaryExpr):
        _rename_in_expr(expr.condition, rename_map)
        _rename_in_expr(expr.then_expr, rename_map)
        _rename_in_expr(expr.else_expr, rename_map)


def _rename_in_target(target, rename_map: dict[str, str]) -> None:
    """Rename VarRef nodes in an assignment target."""
    if isinstance(target, AssignTarget):
        _rename_in_target(target.expr, rename_map)
    elif isinstance(target, VarRef):
        if target.name in rename_map:
            target.name = rename_map[target.name]
    elif isinstance(target, (SwizzleAccess, FieldAccess)):
        _rename_in_target(target.object, rename_map)
    elif isinstance(target, IndexAccess):
        _rename_in_target(target.object, rename_map)
        _rename_in_expr(target.index, rename_map)
