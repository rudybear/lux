"""Dead code elimination pass.

Removes unreferenced let bindings and assignments to unused variables.
Iterates until no more dead code is found.
"""

from __future__ import annotations
from luxc.parser.ast_nodes import (
    Module, FunctionDef, StageBlock,
    LetStmt, AssignStmt, ReturnStmt, IfStmt, ExprStmt,
    DebugPrintStmt, AssertStmt, DebugBlock,
    NumberLit, BoolLit, VarRef, BinaryOp, UnaryOp, CallExpr,
    ConstructorExpr, FieldAccess, SwizzleAccess, IndexAccess, TernaryExpr,
    ForStmt, WhileStmt, BreakStmt, ContinueStmt, AssignTarget,
)

# Calls that have side effects and must be preserved even if result is unused
_SIDE_EFFECT_CALLS = frozenset({
    "sample", "sample_lod", "sample_bindless", "sample_bindless_lod",
    "sample_compare", "sample_array", "sample_grad",
    "trace_ray", "barrier", "memoryBarrier", "memoryBarrierShared",
    "atomicAdd", "atomicMin", "atomicMax", "atomicAnd", "atomicOr", "atomicXor",
    "atomicExchange", "atomicCompareExchange",
    "emit_vertex", "end_primitive", "emit_mesh_tasks",
    "debug_printf", "imageStore",
})


def dead_code_elim(module: Module) -> None:
    """Run dead code elimination on all functions in the module."""
    all_functions: list[tuple[FunctionDef, StageBlock | None]] = []
    for func in module.functions:
        all_functions.append((func, None))
    for stage in module.stages:
        for func in stage.functions:
            all_functions.append((func, stage))
    for func, stage in all_functions:
        _dce_function(func, stage)


def _dce_function(func: FunctionDef, stage: StageBlock | None) -> None:
    """Remove dead code from a function, iterating until stable."""
    # Collect the names of stage output variables so we never eliminate
    # assignments to them from the main entry point.
    protected_vars: set[str] = set()
    if func.name == "main" and stage is not None:
        for out in stage.outputs:
            protected_vars.add(out.name)
        # Builtin variables that are implicitly outputs
        if stage.stage_type == "vertex":
            protected_vars.add("builtin_position")

    # Iterate until no changes
    for _ in range(20):  # safety bound
        used = _collect_used_vars(func.body)
        # Stage outputs are always considered used
        used |= protected_vars
        new_body, changed = _eliminate_dead(func.body, used)
        func.body = new_body
        if not changed:
            break


# ---------------------------------------------------------------------------
# Variable usage collection
# ---------------------------------------------------------------------------

def _collect_used_vars(stmts: list) -> set[str]:
    """Collect all variable names that are READ (referenced) in the statement list.

    Note: a LetStmt defining 'x' does NOT count as a use of 'x'.
    We want to find variables that are read somewhere.
    Walk into IfStmt bodies, ForStmt bodies, WhileStmt bodies, and all expressions.
    """
    used: set[str] = set()
    for stmt in stmts:
        _collect_used_in_stmt(stmt, used)
    return used


def _collect_used_in_stmt(stmt, used: set[str]) -> None:
    """Collect variable references from a single statement."""
    if isinstance(stmt, LetStmt):
        # The variable being defined is NOT a use.  But its initialiser is.
        _collect_used_in_expr(stmt.value, used)

    elif isinstance(stmt, AssignStmt):
        # The assignment target may contain reads (e.g. a[idx].x — 'a' and
        # 'idx' are read).  We collect from the target expression too.
        _collect_used_in_assign_target(stmt.target, used)
        _collect_used_in_expr(stmt.value, used)

    elif isinstance(stmt, ReturnStmt):
        _collect_used_in_expr(stmt.value, used)

    elif isinstance(stmt, IfStmt):
        _collect_used_in_expr(stmt.condition, used)
        for s in stmt.then_body:
            _collect_used_in_stmt(s, used)
        for s in stmt.else_body:
            _collect_used_in_stmt(s, used)

    elif isinstance(stmt, ExprStmt):
        _collect_used_in_expr(stmt.expr, used)

    elif isinstance(stmt, ForStmt):
        # The loop variable itself is implicitly used by the loop machinery.
        used.add(stmt.loop_var)
        _collect_used_in_expr(stmt.init_value, used)
        _collect_used_in_expr(stmt.condition, used)
        # update_target may read through indexing/field chains
        _collect_used_in_assign_target(stmt.update_target, used)
        _collect_used_in_expr(stmt.update_value, used)
        for s in stmt.body:
            _collect_used_in_stmt(s, used)

    elif isinstance(stmt, WhileStmt):
        _collect_used_in_expr(stmt.condition, used)
        for s in stmt.body:
            _collect_used_in_stmt(s, used)

    elif isinstance(stmt, DebugPrintStmt):
        for arg in stmt.args:
            _collect_used_in_expr(arg, used)

    elif isinstance(stmt, AssertStmt):
        _collect_used_in_expr(stmt.condition, used)

    elif isinstance(stmt, DebugBlock):
        for s in stmt.body:
            _collect_used_in_stmt(s, used)

    # BreakStmt, ContinueStmt — no variables to collect


def _collect_used_in_assign_target(target, used: set[str]) -> None:
    """Collect variable references that are READ inside an assignment target.

    An AssignTarget wraps an expression that is the LHS of an assignment.
    For ``a.x = ...``, the base ``a`` is being written to, not read.
    But for ``a[idx].x = ...``, ``idx`` is being read.
    We must walk the chain and collect index expressions, but NOT the
    base VarRef (which is the write destination).

    However, we still add the base VarRef to ``used`` here because an
    assignment to a variable means that variable is *relevant* — the
    actual dead-assignment check is done in ``_eliminate_dead`` by
    looking at whether anyone reads it.  Here we need to capture reads
    embedded in the target expression (like index sub-expressions).
    """
    if isinstance(target, AssignTarget):
        _collect_reads_in_target_expr(target.expr, used)
    else:
        # Might be a raw expression
        _collect_reads_in_target_expr(target, used)


def _collect_reads_in_target_expr(expr, used: set[str]) -> None:
    """Walk an assignment target expression collecting reads from index sub-exprs.

    The base VarRef of the target chain is NOT added (it is the write dest).
    But any IndexAccess indices are full reads.
    """
    if isinstance(expr, VarRef):
        # This is the base — it is being written to, not read.
        # Do NOT add to used.
        pass
    elif isinstance(expr, FieldAccess):
        _collect_reads_in_target_expr(expr.object, used)
    elif isinstance(expr, SwizzleAccess):
        _collect_reads_in_target_expr(expr.object, used)
    elif isinstance(expr, IndexAccess):
        _collect_reads_in_target_expr(expr.object, used)
        # The index expression is a full read
        _collect_used_in_expr(expr.index, used)
    else:
        # Fallback — treat the whole thing as a read
        _collect_used_in_expr(expr, used)


def _collect_used_in_expr(expr, used: set[str]) -> None:
    """Recursively collect VarRef names from an expression tree."""
    if expr is None:
        return

    if isinstance(expr, VarRef):
        used.add(expr.name)

    elif isinstance(expr, (NumberLit, BoolLit)):
        pass  # literals reference no variables

    elif isinstance(expr, BinaryOp):
        _collect_used_in_expr(expr.left, used)
        _collect_used_in_expr(expr.right, used)

    elif isinstance(expr, UnaryOp):
        _collect_used_in_expr(expr.operand, used)

    elif isinstance(expr, CallExpr):
        _collect_used_in_expr(expr.func, used)
        for arg in expr.args:
            _collect_used_in_expr(arg, used)

    elif isinstance(expr, ConstructorExpr):
        for arg in expr.args:
            _collect_used_in_expr(arg, used)

    elif isinstance(expr, FieldAccess):
        _collect_used_in_expr(expr.object, used)

    elif isinstance(expr, SwizzleAccess):
        _collect_used_in_expr(expr.object, used)

    elif isinstance(expr, IndexAccess):
        _collect_used_in_expr(expr.object, used)
        _collect_used_in_expr(expr.index, used)

    elif isinstance(expr, TernaryExpr):
        _collect_used_in_expr(expr.condition, used)
        _collect_used_in_expr(expr.then_expr, used)
        _collect_used_in_expr(expr.else_expr, used)

    elif isinstance(expr, AssignTarget):
        # Can appear in ForStmt update_target
        _collect_used_in_assign_target(expr, used)


# ---------------------------------------------------------------------------
# Dead code elimination
# ---------------------------------------------------------------------------

def _eliminate_dead(stmts: list, used: set[str]) -> tuple[list, bool]:
    """Remove dead let bindings and dead assignments.

    A LetStmt is dead if its name is not in ``used``.
    An AssignStmt is dead if its target base variable is not in ``used``.

    BUT: if the RHS contains a side-effecting call, convert to ExprStmt
    instead of removing entirely so the side effect is preserved.

    Also recursively process IfStmt/ForStmt/WhileStmt/DebugBlock bodies.

    Returns (new_stmts, changed) where changed is True if anything was
    removed or transformed.
    """
    result: list = []
    changed = False
    for stmt in stmts:
        if isinstance(stmt, LetStmt):
            if stmt.name not in used:
                changed = True
                # Dead let — but keep side effects
                if _has_side_effects(stmt.value):
                    result.append(ExprStmt(stmt.value, loc=stmt.loc))
                # else: drop entirely
                continue
            result.append(stmt)

        elif isinstance(stmt, AssignStmt):
            base = _extract_base_var(stmt.target)
            if base is not None and base not in used:
                changed = True
                # Dead assignment — but keep side effects
                if _has_side_effects(stmt.value):
                    result.append(ExprStmt(stmt.value, loc=stmt.loc))
                # else: drop entirely
                continue
            result.append(stmt)

        elif isinstance(stmt, IfStmt):
            stmt.then_body, c1 = _eliminate_dead(stmt.then_body, used)
            stmt.else_body, c2 = _eliminate_dead(stmt.else_body, used)
            changed = changed or c1 or c2
            result.append(stmt)

        elif isinstance(stmt, ForStmt):
            stmt.body, c = _eliminate_dead(stmt.body, used)
            changed = changed or c
            result.append(stmt)

        elif isinstance(stmt, WhileStmt):
            stmt.body, c = _eliminate_dead(stmt.body, used)
            changed = changed or c
            result.append(stmt)

        elif isinstance(stmt, DebugBlock):
            stmt.body, c = _eliminate_dead(stmt.body, used)
            changed = changed or c
            result.append(stmt)

        else:
            # ReturnStmt, ExprStmt, DebugPrintStmt, AssertStmt,
            # BreakStmt, ContinueStmt — always keep
            result.append(stmt)

    return result, changed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_base_var(target) -> str | None:
    """Extract the base variable name from an assignment target.

    Walks through FieldAccess, SwizzleAccess, IndexAccess chains to
    find the root VarRef.  Returns None if the structure is unexpected.
    """
    if isinstance(target, AssignTarget):
        return _extract_base_var_from_expr(target.expr)
    return _extract_base_var_from_expr(target)


def _extract_base_var_from_expr(expr) -> str | None:
    """Walk an expression to find the root VarRef name."""
    if isinstance(expr, VarRef):
        return expr.name
    if isinstance(expr, (FieldAccess, SwizzleAccess)):
        return _extract_base_var_from_expr(expr.object)
    if isinstance(expr, IndexAccess):
        return _extract_base_var_from_expr(expr.object)
    return None


def _has_side_effects(expr) -> bool:
    """Check if an expression tree contains any side-effecting calls."""
    if expr is None:
        return False

    if isinstance(expr, (NumberLit, BoolLit, VarRef)):
        return False

    if isinstance(expr, BinaryOp):
        return _has_side_effects(expr.left) or _has_side_effects(expr.right)

    if isinstance(expr, UnaryOp):
        return _has_side_effects(expr.operand)

    if isinstance(expr, CallExpr):
        # Check if this call itself is side-effecting
        if isinstance(expr.func, VarRef) and expr.func.name in _SIDE_EFFECT_CALLS:
            return True
        # Even if the call name is not in the known set, check arguments
        if _has_side_effects(expr.func):
            return True
        return any(_has_side_effects(arg) for arg in expr.args)

    if isinstance(expr, ConstructorExpr):
        return any(_has_side_effects(arg) for arg in expr.args)

    if isinstance(expr, FieldAccess):
        return _has_side_effects(expr.object)

    if isinstance(expr, SwizzleAccess):
        return _has_side_effects(expr.object)

    if isinstance(expr, IndexAccess):
        return _has_side_effects(expr.object) or _has_side_effects(expr.index)

    if isinstance(expr, TernaryExpr):
        return (
            _has_side_effects(expr.condition)
            or _has_side_effects(expr.then_expr)
            or _has_side_effects(expr.else_expr)
        )

    if isinstance(expr, AssignTarget):
        return _has_side_effects(expr.expr)

    return False
