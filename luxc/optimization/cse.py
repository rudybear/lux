"""Common Subexpression Elimination (CSE) pass.

Identifies duplicate expression trees within function bodies and replaces
redundant computations with references to synthetic let bindings.

Constraints:
- Never CSE across control-flow boundaries (if/for/while bodies)
- Never CSE texture sampling or other side-effecting calls
- Only CSE non-trivial expressions (skip VarRef, NumberLit, BoolLit)
- Synthetic bindings use names like _cse_0, _cse_1, etc.
"""

from __future__ import annotations
from typing import Any

from luxc.parser.ast_nodes import (
    Module, FunctionDef, StageBlock,
    LetStmt, AssignStmt, ReturnStmt, IfStmt, ExprStmt,
    NumberLit, BoolLit, VarRef, BinaryOp, UnaryOp, CallExpr,
    ConstructorExpr, FieldAccess, SwizzleAccess, IndexAccess, TernaryExpr,
    ForStmt, WhileStmt, DebugPrintStmt, AssertStmt, DebugBlock,
    AssignTarget,
)

# ---------------------------------------------------------------------------
# Calls with side effects that must never be CSE'd
# ---------------------------------------------------------------------------

_NO_CSE_CALLS = frozenset({
    # Texture sampling (derivative side effects)
    "sample", "sample_lod", "sample_bindless", "sample_bindless_lod",
    "sample_compare", "sample_array", "sample_grad",
    # Ray tracing
    "trace_ray",
    # Barriers / synchronisation
    "barrier", "memoryBarrier", "memoryBarrierBuffer",
    "memoryBarrierShared", "memoryBarrierImage",
    "groupMemoryBarrier",
    # Atomics
    "atomicAdd", "atomicMin", "atomicMax", "atomicAnd", "atomicOr",
    "atomicXor", "atomicExchange", "atomicCompareExchange",
    # Geometry / mesh output
    "emit_vertex", "end_primitive", "emit_mesh_tasks",
    # Debug
    "debug_printf",
})


# ---------------------------------------------------------------------------
# Structural hashing
# ---------------------------------------------------------------------------

def _expr_hash(expr: Any) -> int:
    """Compute a structural hash for an expression tree.

    The hash is used as a fast pre-filter: two expressions with different
    hashes are definitely not equal; two with the same hash *may* be equal
    and must be confirmed with ``_exprs_equal``.
    """
    if isinstance(expr, NumberLit):
        return hash(("NumberLit", expr.value))
    if isinstance(expr, BoolLit):
        return hash(("BoolLit", expr.value))
    if isinstance(expr, VarRef):
        return hash(("VarRef", expr.name))
    if isinstance(expr, BinaryOp):
        return hash(("BinaryOp", expr.op, _expr_hash(expr.left),
                      _expr_hash(expr.right)))
    if isinstance(expr, UnaryOp):
        return hash(("UnaryOp", expr.op, _expr_hash(expr.operand)))
    if isinstance(expr, CallExpr):
        args_hash = tuple(_expr_hash(a) for a in expr.args)
        return hash(("CallExpr", _expr_hash(expr.func), args_hash))
    if isinstance(expr, ConstructorExpr):
        args_hash = tuple(_expr_hash(a) for a in expr.args)
        return hash(("ConstructorExpr", expr.type_name, args_hash))
    if isinstance(expr, FieldAccess):
        return hash(("FieldAccess", expr.field, _expr_hash(expr.object)))
    if isinstance(expr, SwizzleAccess):
        return hash(("SwizzleAccess", expr.components,
                      _expr_hash(expr.object)))
    if isinstance(expr, IndexAccess):
        return hash(("IndexAccess", _expr_hash(expr.object),
                      _expr_hash(expr.index)))
    if isinstance(expr, TernaryExpr):
        return hash(("TernaryExpr", _expr_hash(expr.condition),
                      _expr_hash(expr.then_expr),
                      _expr_hash(expr.else_expr)))
    # Fallback for unknown node types — use id so they never collide
    return id(expr)


# ---------------------------------------------------------------------------
# Structural equality
# ---------------------------------------------------------------------------

def _exprs_equal(a: Any, b: Any) -> bool:
    """Check structural equality of two expression trees."""
    if type(a) is not type(b):
        return False
    if isinstance(a, NumberLit):
        return a.value == b.value
    if isinstance(a, BoolLit):
        return a.value == b.value
    if isinstance(a, VarRef):
        return a.name == b.name
    if isinstance(a, BinaryOp):
        return (a.op == b.op
                and _exprs_equal(a.left, b.left)
                and _exprs_equal(a.right, b.right))
    if isinstance(a, UnaryOp):
        return a.op == b.op and _exprs_equal(a.operand, b.operand)
    if isinstance(a, CallExpr):
        if not _exprs_equal(a.func, b.func):
            return False
        if len(a.args) != len(b.args):
            return False
        return all(_exprs_equal(x, y) for x, y in zip(a.args, b.args))
    if isinstance(a, ConstructorExpr):
        if a.type_name != b.type_name or len(a.args) != len(b.args):
            return False
        return all(_exprs_equal(x, y) for x, y in zip(a.args, b.args))
    if isinstance(a, FieldAccess):
        return a.field == b.field and _exprs_equal(a.object, b.object)
    if isinstance(a, SwizzleAccess):
        return (a.components == b.components
                and _exprs_equal(a.object, b.object))
    if isinstance(a, IndexAccess):
        return (_exprs_equal(a.object, b.object)
                and _exprs_equal(a.index, b.index))
    if isinstance(a, TernaryExpr):
        return (_exprs_equal(a.condition, b.condition)
                and _exprs_equal(a.then_expr, b.then_expr)
                and _exprs_equal(a.else_expr, b.else_expr))
    return False


# ---------------------------------------------------------------------------
# Candidacy check
# ---------------------------------------------------------------------------

_STRUCT_TYPES = frozenset({"LightData", "BindlessMaterialData", "ShadowEntry"})


def _is_cse_candidate(expr: Any) -> bool:
    """Return True if *expr* is worth CSE-ing.

    Trivial leaf nodes (VarRef, NumberLit, BoolLit) are not worth hoisting
    into a separate let binding.  Side-effecting calls are excluded as well.
    """
    # Trivial leaves — never CSE
    if isinstance(expr, (VarRef, NumberLit, BoolLit)):
        return False

    # SwizzleAccess (e.g. v.x, v.xy) is very cheap in SPIR-V (a single
    # OpCompositeExtract) — hoisting it into a function-scope variable
    # adds more overhead than it saves and can trigger naga/wgpu issues
    # with large shaders.
    if isinstance(expr, SwizzleAccess):
        return False

    # Side-effecting calls — never CSE
    if isinstance(expr, CallExpr):
        if isinstance(expr.func, VarRef) and expr.func.name in _NO_CSE_CALLS:
            return False

    # Never CSE struct-typed SSBO element access — the struct type carries
    # StorageBuffer layout decorations that are invalid for Function scope.
    if isinstance(expr, IndexAccess):
        resolved = getattr(expr, "resolved_type", None)
        if resolved in _STRUCT_TYPES:
            return False

    return True


# ---------------------------------------------------------------------------
# Sub-expression collection helpers
# ---------------------------------------------------------------------------

# An occurrence is a tuple (stmt_index, setter) where setter is a callable
# that, given a new expression, replaces the sub-expression at that location.

_Occurrence = tuple[int, Any]  # (stmt_index, setter_callback)


def _collect_subexprs(expr: Any, stmt_idx: int, setter,
                      out: dict[int, list[tuple[Any, _Occurrence]]]) -> None:
    """Recursively walk *expr* collecting every CSE-candidate sub-expression.

    For each candidate we store its hash, the expression reference, the
    statement index it appeared in, and a *setter* callback that can replace
    the expression in its parent.

    ``out`` maps expr_hash -> list of (expr, (stmt_idx, setter)).
    """
    if expr is None:
        return

    if isinstance(expr, (NumberLit, BoolLit, VarRef)):
        # Leaves — still recurse nowhere, but record nothing (trivial).
        return

    if isinstance(expr, BinaryOp):
        # Children first (bottom-up) so inner sub-expressions are found too
        _collect_subexprs(expr.left, stmt_idx,
                          lambda e, _e=expr: setattr(_e, "left", e), out)
        _collect_subexprs(expr.right, stmt_idx,
                          lambda e, _e=expr: setattr(_e, "right", e), out)
        if _is_cse_candidate(expr):
            h = _expr_hash(expr)
            out.setdefault(h, []).append((expr, (stmt_idx, setter)))
        return

    if isinstance(expr, UnaryOp):
        _collect_subexprs(expr.operand, stmt_idx,
                          lambda e, _e=expr: setattr(_e, "operand", e), out)
        if _is_cse_candidate(expr):
            h = _expr_hash(expr)
            out.setdefault(h, []).append((expr, (stmt_idx, setter)))
        return

    if isinstance(expr, CallExpr):
        _collect_subexprs(expr.func, stmt_idx,
                          lambda e, _e=expr: setattr(_e, "func", e), out)
        for i, arg in enumerate(expr.args):
            _collect_subexprs(arg, stmt_idx,
                              lambda e, _e=expr, _i=i: _e.args.__setitem__(_i, e),
                              out)
        if _is_cse_candidate(expr):
            h = _expr_hash(expr)
            out.setdefault(h, []).append((expr, (stmt_idx, setter)))
        return

    if isinstance(expr, ConstructorExpr):
        for i, arg in enumerate(expr.args):
            _collect_subexprs(arg, stmt_idx,
                              lambda e, _e=expr, _i=i: _e.args.__setitem__(_i, e),
                              out)
        if _is_cse_candidate(expr):
            h = _expr_hash(expr)
            out.setdefault(h, []).append((expr, (stmt_idx, setter)))
        return

    if isinstance(expr, FieldAccess):
        _collect_subexprs(expr.object, stmt_idx,
                          lambda e, _e=expr: setattr(_e, "object", e), out)
        if _is_cse_candidate(expr):
            h = _expr_hash(expr)
            out.setdefault(h, []).append((expr, (stmt_idx, setter)))
        return

    if isinstance(expr, SwizzleAccess):
        _collect_subexprs(expr.object, stmt_idx,
                          lambda e, _e=expr: setattr(_e, "object", e), out)
        if _is_cse_candidate(expr):
            h = _expr_hash(expr)
            out.setdefault(h, []).append((expr, (stmt_idx, setter)))
        return

    if isinstance(expr, IndexAccess):
        _collect_subexprs(expr.object, stmt_idx,
                          lambda e, _e=expr: setattr(_e, "object", e), out)
        _collect_subexprs(expr.index, stmt_idx,
                          lambda e, _e=expr: setattr(_e, "index", e), out)
        if _is_cse_candidate(expr):
            h = _expr_hash(expr)
            out.setdefault(h, []).append((expr, (stmt_idx, setter)))
        return

    if isinstance(expr, TernaryExpr):
        _collect_subexprs(expr.condition, stmt_idx,
                          lambda e, _e=expr: setattr(_e, "condition", e), out)
        _collect_subexprs(expr.then_expr, stmt_idx,
                          lambda e, _e=expr: setattr(_e, "then_expr", e), out)
        _collect_subexprs(expr.else_expr, stmt_idx,
                          lambda e, _e=expr: setattr(_e, "else_expr", e), out)
        if _is_cse_candidate(expr):
            h = _expr_hash(expr)
            out.setdefault(h, []).append((expr, (stmt_idx, setter)))
        return

    if isinstance(expr, AssignTarget):
        # Walk inside the assign target expression
        _collect_subexprs(expr.expr, stmt_idx,
                          lambda e, _e=expr: setattr(_e, "expr", e), out)
        return


# ---------------------------------------------------------------------------
# Collect sub-expressions from a single statement
# ---------------------------------------------------------------------------

def _collect_from_stmt(stmt: Any, stmt_idx: int,
                       out: dict[int, list[tuple[Any, _Occurrence]]]) -> None:
    """Collect all CSE-candidate sub-expressions from a flat statement.

    Does NOT recurse into sub-bodies of if/for/while — those are treated as
    opaque for CSE purposes (they get their own independent CSE pass).
    """
    if isinstance(stmt, LetStmt):
        _collect_subexprs(
            stmt.value, stmt_idx,
            lambda e, _s=stmt: setattr(_s, "value", e), out)
    elif isinstance(stmt, AssignStmt):
        # Do not CSE the target itself (it is an l-value), but do collect
        # from within index expressions on the target.
        if isinstance(stmt.target, (IndexAccess, FieldAccess, SwizzleAccess)):
            _collect_subexprs(
                stmt.target, stmt_idx,
                lambda e, _s=stmt: setattr(_s, "target", e), out)
        _collect_subexprs(
            stmt.value, stmt_idx,
            lambda e, _s=stmt: setattr(_s, "value", e), out)
    elif isinstance(stmt, ReturnStmt):
        _collect_subexprs(
            stmt.value, stmt_idx,
            lambda e, _s=stmt: setattr(_s, "value", e), out)
    elif isinstance(stmt, ExprStmt):
        _collect_subexprs(
            stmt.expr, stmt_idx,
            lambda e, _s=stmt: setattr(_s, "expr", e), out)
    elif isinstance(stmt, IfStmt):
        # Only the *condition* is part of this flat scope.
        _collect_subexprs(
            stmt.condition, stmt_idx,
            lambda e, _s=stmt: setattr(_s, "condition", e), out)
    elif isinstance(stmt, ForStmt):
        # init_value, condition, update parts live in the flat scope
        _collect_subexprs(
            stmt.init_value, stmt_idx,
            lambda e, _s=stmt: setattr(_s, "init_value", e), out)
        _collect_subexprs(
            stmt.condition, stmt_idx,
            lambda e, _s=stmt: setattr(_s, "condition", e), out)
        _collect_subexprs(
            stmt.update_value, stmt_idx,
            lambda e, _s=stmt: setattr(_s, "update_value", e), out)
    elif isinstance(stmt, WhileStmt):
        _collect_subexprs(
            stmt.condition, stmt_idx,
            lambda e, _s=stmt: setattr(_s, "condition", e), out)
    elif isinstance(stmt, DebugPrintStmt):
        for i, arg in enumerate(stmt.args):
            _collect_subexprs(
                arg, stmt_idx,
                lambda e, _s=stmt, _i=i: _s.args.__setitem__(_i, e), out)
    elif isinstance(stmt, AssertStmt):
        _collect_subexprs(
            stmt.condition, stmt_idx,
            lambda e, _s=stmt: setattr(_s, "condition", e), out)
    # DebugBlock, BreakStmt, ContinueStmt — nothing to collect


# ---------------------------------------------------------------------------
# Core CSE on a flat statement list
# ---------------------------------------------------------------------------

def _collect_and_replace(stmts: list, counter: list[int]) -> list:
    """Process a flat list of statements, finding and replacing common
    sub-expressions.

    Algorithm
    ---------
    1. Walk each statement collecting all candidate sub-expressions together
       with their structural hashes and setter callbacks.
    2. Group by hash.  For each group with 2+ entries, confirm structural
       equality (to guard against hash collisions).  If a true duplicate
       set is found:
       a. Create a synthetic ``let _cse_N: <type> = <first_occurrence>``
       b. Replace *every* occurrence (including the first) with a VarRef to
          the synthetic variable.
       c. Insert the let before the statement containing the first
          occurrence.
    3. Return the (possibly expanded) list of statements.
    """
    # --- Phase 1: collect ---------------------------------------------------
    # Map hash -> list of (expr_ref, (stmt_idx, setter))
    expr_map: dict[int, list[tuple[Any, _Occurrence]]] = {}

    for idx, stmt in enumerate(stmts):
        _collect_from_stmt(stmt, idx, expr_map)

    # --- Phase 2: identify true duplicates ----------------------------------
    # We need to process larger expressions before their sub-expressions so
    # that when we replace a large expression, its sub-expressions (which may
    # also be duplicated) are replaced together rather than independently.
    # Sort candidates by descending tree size for greedy selection.

    # Collect groups: list of (hash, representative_expr, occurrences_list)
    cse_groups: list[tuple[int, Any, list[_Occurrence]]] = []

    for h, entries in expr_map.items():
        if len(entries) < 2:
            continue

        # Cluster entries by structural equality (handles hash collisions)
        clusters: list[list[tuple[Any, _Occurrence]]] = []
        for expr, occ in entries:
            placed = False
            for cluster in clusters:
                if _exprs_equal(expr, cluster[0][0]):
                    cluster.append((expr, occ))
                    placed = True
                    break
            if not placed:
                clusters.append([(expr, occ)])

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            representative = cluster[0][0]
            occurrences = [occ for _, occ in cluster]
            cse_groups.append((h, representative, occurrences))

    if not cse_groups:
        return stmts

    # --- Phase 3: replace, largest first ------------------------------------
    # To avoid replacing a sub-expression of something we already plan to
    # replace, we track which (stmt_idx, setter) pairs have been consumed.
    consumed_setters: set[int] = set()  # ids of setter callbacks

    # Sort groups by descending expression "size" (approximate via hash
    # depth — we use a simple node-count heuristic).
    cse_groups.sort(key=lambda g: -_expr_size(g[1]))

    # Records of insertions: list of (insert_before_stmt_idx, LetStmt)
    insertions: list[tuple[int, LetStmt]] = []

    for _h, representative, occurrences in cse_groups:
        # Filter out occurrences whose setters have already been consumed
        live = [(si, setter) for si, setter in occurrences
                if id(setter) not in consumed_setters]
        if len(live) < 2:
            continue

        # Determine type from the representative expression
        resolved = getattr(representative, "resolved_type", None)
        type_name = resolved if resolved else "auto"

        name = f"_cse_{counter[0]}"
        counter[0] += 1

        # Find the earliest statement index among the live occurrences
        first_stmt_idx = min(si for si, _ in live)

        # Create the let binding
        let_stmt = LetStmt(name=name, type_name=type_name,
                           value=representative)

        insertions.append((first_stmt_idx, let_stmt))

        # Replace all occurrences with VarRef
        for _si, setter in live:
            ref = VarRef(name=name, resolved_type=resolved)
            setter(ref)
            consumed_setters.add(id(setter))

    if not insertions:
        return stmts

    # --- Phase 4: build the new statement list with insertions --------------
    # Group insertions by their target statement index
    insert_map: dict[int, list[LetStmt]] = {}
    for idx, let_stmt in insertions:
        insert_map.setdefault(idx, []).append(let_stmt)

    new_stmts: list = []
    for idx, stmt in enumerate(stmts):
        if idx in insert_map:
            new_stmts.extend(insert_map[idx])
        new_stmts.append(stmt)

    return new_stmts


def _expr_size(expr: Any) -> int:
    """Return an approximate node count for an expression tree."""
    if isinstance(expr, (NumberLit, BoolLit, VarRef)):
        return 1
    if isinstance(expr, BinaryOp):
        return 1 + _expr_size(expr.left) + _expr_size(expr.right)
    if isinstance(expr, UnaryOp):
        return 1 + _expr_size(expr.operand)
    if isinstance(expr, CallExpr):
        return 1 + _expr_size(expr.func) + sum(_expr_size(a) for a in expr.args)
    if isinstance(expr, ConstructorExpr):
        return 1 + sum(_expr_size(a) for a in expr.args)
    if isinstance(expr, FieldAccess):
        return 1 + _expr_size(expr.object)
    if isinstance(expr, SwizzleAccess):
        return 1 + _expr_size(expr.object)
    if isinstance(expr, IndexAccess):
        return 1 + _expr_size(expr.object) + _expr_size(expr.index)
    if isinstance(expr, TernaryExpr):
        return (1 + _expr_size(expr.condition) + _expr_size(expr.then_expr)
                + _expr_size(expr.else_expr))
    return 1


# ---------------------------------------------------------------------------
# Per-function CSE driver
# ---------------------------------------------------------------------------

def _cse_function(func: FunctionDef, counter: list[int]) -> None:
    """Run CSE on a single function body."""
    func.body = _cse_stmts(func.body, counter)


def _cse_stmts(stmts: list, counter: list[int]) -> list:
    """Run CSE on a list of statements.

    1. Perform CSE on the flat (top-level) statement sequence.
    2. Recurse into the bodies of any control-flow statements so that each
       nested scope gets its own independent CSE pass.
    """
    # First, CSE the flat sequence
    stmts = _collect_and_replace(stmts, counter)

    # Then recurse into sub-bodies
    for stmt in stmts:
        if isinstance(stmt, IfStmt):
            stmt.then_body = _cse_stmts(stmt.then_body, counter)
            stmt.else_body = _cse_stmts(stmt.else_body, counter)
        elif isinstance(stmt, ForStmt):
            stmt.body = _cse_stmts(stmt.body, counter)
        elif isinstance(stmt, WhileStmt):
            stmt.body = _cse_stmts(stmt.body, counter)
        elif isinstance(stmt, DebugBlock):
            stmt.body = _cse_stmts(stmt.body, counter)


    return stmts


# ---------------------------------------------------------------------------
# Module-level entry point
# ---------------------------------------------------------------------------

def cse(module: Module) -> None:
    """Run Common Subexpression Elimination on all functions in the module.

    Processes every function in both the module-level function list and
    each stage block's function list.  A single monotonic counter is shared
    across all functions to guarantee unique synthetic variable names.
    """
    import copy

    counter = [0]

    for func in module.functions:
        _cse_function(func, counter)

    # Deep-copy function bodies before CSE to avoid aliasing: pipeline
    # expansion may share expression objects across stages, so in-place
    # CSE replacements in one stage can corrupt another.
    for stage in module.stages:
        for func in stage.functions:
            func.body = copy.deepcopy(func.body)

    for stage in module.stages:
        for func in stage.functions:
            _cse_function(func, counter)
