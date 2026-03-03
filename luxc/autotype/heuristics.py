"""Heuristic precision classification based on variable names and usage patterns."""

from __future__ import annotations

import re

from luxc.parser.ast_nodes import (
    Module, StageBlock, FunctionDef, LetStmt, AssignStmt,
    BinaryOp, CallExpr, VarRef, IfStmt, ForStmt, WhileStmt,
    DebugBlock,
)
from luxc.autotype.types import Precision, PrecisionDecision


# --- Name-based patterns ---

_FP16_NAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^normal"), "normal vector"),
    (re.compile(r"^world_normal"), "normal vector"),
    (re.compile(r"^color$"), "color value"),
    (re.compile(r"^frag_color$"), "color value"),
    (re.compile(r"^vertex_color$"), "color value"),
    (re.compile(r"^albedo"), "albedo color"),
    (re.compile(r"^base_?[Cc]olor"), "base color"),
    (re.compile(r"^roughness"), "material param [0,1]"),
    (re.compile(r"^metallic"), "material param [0,1]"),
    (re.compile(r"^metalness"), "material param [0,1]"),
    (re.compile(r"^ao$"), "ambient occlusion [0,1]"),
    (re.compile(r"^occlusion"), "ambient occlusion [0,1]"),
    (re.compile(r"^opacity"), "opacity [0,1]"),
    (re.compile(r"^alpha$"), "alpha [0,1]"),
    (re.compile(r"^uv"), "texture coordinate"),
    (re.compile(r"^texcoord"), "texture coordinate"),
    (re.compile(r"^emission$"), "emission color"),
    (re.compile(r"^emissive$"), "emissive color"),
    (re.compile(r"^diffuse"), "diffuse color"),
    (re.compile(r"^specular$"), "specular color"),
    (re.compile(r"^ambient$"), "ambient color"),
]

_FP32_NAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"position"), "position value"),
    (re.compile(r"world_pos"), "world-space position"),
    (re.compile(r"matrix"), "matrix value"),
    (re.compile(r"mvp"), "MVP matrix"),
    (re.compile(r"projection"), "projection matrix"),
    (re.compile(r"^depth"), "depth value"),
    (re.compile(r"^z_"), "depth/z value"),
    (re.compile(r"^distance"), "distance value"),
    (re.compile(r"^view_"), "view-space value"),
    (re.compile(r"^clip_"), "clip-space value"),
]

# CSE temporaries — don't classify by name, defer to range data
_CSE_PATTERN = re.compile(r"^_cse_\d+$")


class HeuristicClassifier:
    """Classify variable precision using name patterns and usage context."""

    def classify(self, module: Module, stage: StageBlock) -> dict[str, PrecisionDecision]:
        """Classify all variables in a stage using heuristic rules."""
        decisions: dict[str, PrecisionDecision] = {}

        # Collect all variable names and types from the main function
        main_fn = _find_main(stage)
        if main_fn is None:
            return decisions

        var_types: dict[str, str] = {}
        _collect_var_types(main_fn.body, var_types)

        # Name-based classification
        for name, type_name in var_types.items():
            # Skip integer types — not relevant for fp16
            if type_name in ("int", "uint", "ivec2", "ivec3", "ivec4",
                             "uvec2", "uvec3", "uvec4", "bool"):
                decisions[name] = PrecisionDecision(
                    Precision.FP32, 1.0, "integer/bool type"
                )
                continue

            # Skip matrix types — always fp32
            if type_name.startswith("mat"):
                decisions[name] = PrecisionDecision(
                    Precision.FP32, 1.0, "matrix type"
                )
                continue

            # CSE vars: defer to range data
            if _CSE_PATTERN.match(name):
                continue

            result = self._classify_by_name(name, type_name)
            if result is not None:
                decisions[name] = result

        # Usage-based classification (can override name-based)
        usage_decisions = self._classify_by_usage(main_fn.body, var_types)
        for name, decision in usage_decisions.items():
            if decision.precision == Precision.FP32:
                # fp32 usage always wins
                decisions[name] = decision
            elif name not in decisions:
                decisions[name] = decision

        # Stage outputs are always fp32
        for out in stage.outputs:
            decisions[out.name] = PrecisionDecision(
                Precision.FP32, 1.0, "stage output"
            )

        return decisions

    def _classify_by_name(self, name: str, type_name: str) -> PrecisionDecision | None:
        """Classify by variable name pattern."""
        name_lower = name.lower()

        for pattern, reason in _FP16_NAME_PATTERNS:
            if pattern.search(name_lower):
                return PrecisionDecision(Precision.FP16, 0.8, f"name heuristic: {reason}")

        for pattern, reason in _FP32_NAME_PATTERNS:
            if pattern.search(name_lower):
                return PrecisionDecision(Precision.FP32, 1.0, f"name heuristic: {reason}")

        return None

    def _classify_by_usage(self, stmts: list, var_types: dict[str, str]) -> dict[str, PrecisionDecision]:
        """Classify variables by how they're used in expressions."""
        decisions: dict[str, PrecisionDecision] = {}
        fp32_usages: set[str] = set()
        fp16_usages: dict[str, str] = {}  # name -> reason

        _scan_usage(stmts, fp32_usages, fp16_usages)

        for name in fp32_usages:
            decisions[name] = PrecisionDecision(
                Precision.FP32, 1.0, "usage: fp32-sensitive operation"
            )

        for name, reason in fp16_usages.items():
            if name not in fp32_usages and name not in decisions:
                decisions[name] = PrecisionDecision(
                    Precision.FP16, 0.7, f"usage: {reason}"
                )

        return decisions


def _find_main(stage: StageBlock) -> FunctionDef | None:
    for fn in stage.functions:
        if fn.name == "main":
            return fn
    return None


def _collect_var_types(stmts: list, var_types: dict[str, str]) -> None:
    """Collect variable name -> type_name from LetStmts."""
    for stmt in stmts:
        if isinstance(stmt, LetStmt):
            var_types[stmt.name] = stmt.type_name
        elif isinstance(stmt, IfStmt):
            _collect_var_types(stmt.then_body, var_types)
            _collect_var_types(stmt.else_body, var_types)
        elif isinstance(stmt, ForStmt):
            var_types[stmt.loop_var] = stmt.loop_var_type
            _collect_var_types(stmt.body, var_types)
        elif isinstance(stmt, WhileStmt):
            _collect_var_types(stmt.body, var_types)
        elif isinstance(stmt, DebugBlock):
            _collect_var_types(stmt.body, var_types)


def _scan_usage(stmts: list, fp32_usages: set[str], fp16_usages: dict[str, str]) -> None:
    """Scan statements for usage patterns that inform precision."""
    for stmt in stmts:
        if isinstance(stmt, LetStmt):
            _scan_expr_usage(stmt.value, stmt.name, fp32_usages, fp16_usages)
        elif isinstance(stmt, AssignStmt):
            _scan_expr_usage(stmt.value, None, fp32_usages, fp16_usages)
        elif isinstance(stmt, IfStmt):
            _scan_usage(stmt.then_body, fp32_usages, fp16_usages)
            _scan_usage(stmt.else_body, fp32_usages, fp16_usages)
        elif isinstance(stmt, ForStmt):
            _scan_usage(stmt.body, fp32_usages, fp16_usages)
        elif isinstance(stmt, WhileStmt):
            _scan_usage(stmt.body, fp32_usages, fp16_usages)
        elif isinstance(stmt, DebugBlock):
            _scan_usage(stmt.body, fp32_usages, fp16_usages)


def _scan_expr_usage(expr, let_name: str | None, fp32_usages: set[str], fp16_usages: dict[str, str]) -> None:
    """Scan an expression for usage patterns."""
    if expr is None:
        return

    if isinstance(expr, CallExpr):
        func_name = expr.func.name if isinstance(expr.func, VarRef) else ""

        # Output of normalize/clamp/smoothstep → fp16-safe
        if func_name in ("normalize", "smoothstep"):
            if let_name is not None:
                fp16_usages[let_name] = f"output of {func_name}()"

        if func_name == "clamp" and len(expr.args) >= 3:
            if let_name is not None:
                fp16_usages[let_name] = "output of clamp()"

        # pow() exponent → fp32
        if func_name == "pow" and len(expr.args) >= 2:
            _mark_vars_fp32(expr.args[1], fp32_usages)

        # Division denominator → fp32
        for arg in expr.args:
            _scan_expr_usage(arg, None, fp32_usages, fp16_usages)

    elif isinstance(expr, BinaryOp):
        if expr.op == "/":
            _mark_vars_fp32(expr.right, fp32_usages)
        _scan_expr_usage(expr.left, None, fp32_usages, fp16_usages)
        _scan_expr_usage(expr.right, None, fp32_usages, fp16_usages)


def _mark_vars_fp32(expr, fp32_usages: set[str]) -> None:
    """Mark all VarRefs in expression as requiring fp32."""
    if isinstance(expr, VarRef):
        fp32_usages.add(expr.name)
    elif isinstance(expr, BinaryOp):
        _mark_vars_fp32(expr.left, fp32_usages)
        _mark_vars_fp32(expr.right, fp32_usages)
    elif isinstance(expr, CallExpr):
        for arg in expr.args:
            _mark_vars_fp32(arg, fp32_usages)
