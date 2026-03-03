"""Static interval analysis — forward dataflow propagation of [lo, hi] ranges."""

from __future__ import annotations

import math

from luxc.parser.ast_nodes import (
    Module, StageBlock, FunctionDef,
    LetStmt, AssignStmt, IfStmt, ForStmt, WhileStmt, ReturnStmt,
    ExprStmt, DebugBlock, DebugPrintStmt, AssertStmt,
    NumberLit, BoolLit, VarRef, BinaryOp, UnaryOp,
    CallExpr, ConstructorExpr, FieldAccess, SwizzleAccess,
    IndexAccess, TernaryExpr,
)
from luxc.autotype.types import Interval, VarRange


# --- Default input ranges for stage variables by semantic name ---

_INPUT_RANGES: dict[str, list[Interval]] = {
    "normal":           [Interval(-1, 1), Interval(-1, 1), Interval(-1, 1)],
    "world_normal":     [Interval(-1, 1), Interval(-1, 1), Interval(-1, 1)],
    "uv":               [Interval(0, 1), Interval(0, 1)],
    "texcoord":         [Interval(0, 1), Interval(0, 1)],
    "texcoord0":        [Interval(0, 1), Interval(0, 1)],
    "roughness":        [Interval(0, 1)],
    "metallic":         [Interval(0, 1)],
    "metalness":        [Interval(0, 1)],
    "ao":               [Interval(0, 1)],
    "occlusion":        [Interval(0, 1)],
    "opacity":          [Interval(0, 1)],
    "alpha":            [Interval(0, 1)],
    "color":            [Interval(0, 1)] * 4,
    "albedo":           [Interval(0, 1)] * 3,
    "baseColor":        [Interval(0, 1)] * 3,
    "emission":         [Interval(0, 1)] * 3,
    "position":         [Interval(-1000, 1000)] * 3,
    "world_position":   [Interval(-1000, 1000)] * 3,
}


class IntervalAnalysis:
    """Forward dataflow analysis propagating [lo, hi] intervals through AST expressions."""

    def __init__(self):
        self.var_intervals: dict[str, list[Interval]] = {}

    def analyze_stage(self, stage: StageBlock, module: Module,
                      input_ranges: dict[str, list[Interval]] | None = None) -> dict[str, VarRange]:
        """Analyze a stage, returning range info for all variables."""
        self.var_intervals = {}

        # Initialize input variable ranges
        for inp in stage.inputs:
            name = inp.name
            if input_ranges and name in input_ranges:
                self.var_intervals[name] = _deep_copy_intervals(input_ranges[name])
            elif name in _INPUT_RANGES:
                self.var_intervals[name] = _deep_copy_intervals(_INPUT_RANGES[name])
            else:
                n = _type_component_count(inp.type_name)
                self.var_intervals[name] = [Interval(float('-inf'), float('inf'))] * n

        # Initialize uniform/push constant ranges (conservative)
        for ub in stage.uniforms:
            for uf in ub.fields:
                name = uf.name
                if name in _INPUT_RANGES:
                    self.var_intervals[name] = _deep_copy_intervals(_INPUT_RANGES[name])
                else:
                    n = _type_component_count(uf.type_name)
                    # Matrices → very large range (forces fp32)
                    if uf.type_name.startswith("mat"):
                        self.var_intervals[name] = [Interval(-1e6, 1e6)] * n
                    else:
                        self.var_intervals[name] = [Interval(float('-inf'), float('inf'))] * n

        for pb in stage.push_constants:
            for pf in pb.fields:
                name = pf.name
                if name in _INPUT_RANGES:
                    self.var_intervals[name] = _deep_copy_intervals(_INPUT_RANGES[name])
                else:
                    n = _type_component_count(pf.type_name)
                    if pf.type_name.startswith("mat"):
                        self.var_intervals[name] = [Interval(-1e6, 1e6)] * n
                    else:
                        self.var_intervals[name] = [Interval(float('-inf'), float('inf'))] * n

        # Initialize module constants
        for const in module.constants:
            intervals = self._analyze_expr(const.value)
            if intervals:
                self.var_intervals[const.name] = intervals

        # Analyze main function body
        main_fn = None
        for fn in stage.functions:
            if fn.name == "main":
                main_fn = fn
                break

        if main_fn is not None:
            self._analyze_stmts(main_fn.body)

        # Build result
        result: dict[str, VarRange] = {}
        for name, intervals in self.var_intervals.items():
            # Determine type name
            type_name = _guess_type_name(len(intervals))
            result[name] = VarRange(
                name=name,
                type_name=type_name,
                intervals=intervals,
            )
        return result

    def _analyze_stmts(self, stmts: list) -> None:
        for stmt in stmts:
            self._analyze_stmt(stmt)

    def _analyze_stmt(self, stmt) -> None:
        if isinstance(stmt, LetStmt):
            intervals = self._analyze_expr(stmt.value)
            n = _type_component_count(stmt.type_name)
            if intervals:
                # Pad or truncate to match declared type
                while len(intervals) < n:
                    intervals.append(Interval(0, 0))
                intervals = intervals[:n]
            else:
                intervals = [Interval(float('-inf'), float('inf'))] * n
            self.var_intervals[stmt.name] = intervals

        elif isinstance(stmt, AssignStmt):
            # For simplicity, widen the existing variable
            target_name = _extract_root_name(stmt.target)
            if target_name and target_name in self.var_intervals:
                new_intervals = self._analyze_expr(stmt.value)
                if new_intervals:
                    existing = self.var_intervals[target_name]
                    for i in range(min(len(existing), len(new_intervals))):
                        existing[i].merge(new_intervals[i])

        elif isinstance(stmt, IfStmt):
            # Save state, analyze both branches, union results
            saved = _snapshot(self.var_intervals)
            self._analyze_stmts(stmt.then_body)
            then_state = _snapshot(self.var_intervals)
            self.var_intervals = saved
            self._analyze_stmts(stmt.else_body)
            # Union then-branch with else-branch
            _merge_states(self.var_intervals, then_state)

        elif isinstance(stmt, (ForStmt, WhileStmt)):
            # Conservative: widen loop-modified variables to [-inf, inf] after 2 iterations
            body = stmt.body
            if isinstance(stmt, ForStmt):
                init_intervals = self._analyze_expr(stmt.init_value)
                n = _type_component_count(stmt.loop_var_type)
                if init_intervals:
                    while len(init_intervals) < n:
                        init_intervals.append(Interval(0, 0))
                    self.var_intervals[stmt.loop_var] = init_intervals[:n]
                else:
                    self.var_intervals[stmt.loop_var] = [Interval(float('-inf'), float('inf'))] * n

            # Iterate twice to detect widening
            for _ in range(2):
                self._analyze_stmts(body)

            # After loop: any variable modified inside the loop gets widened
            modified = _find_modified_vars(body)
            for name in modified:
                if name in self.var_intervals:
                    for iv in self.var_intervals[name]:
                        iv.lo = float('-inf')
                        iv.hi = float('inf')

        elif isinstance(stmt, (DebugBlock,)):
            self._analyze_stmts(stmt.body)

        elif isinstance(stmt, (ReturnStmt, ExprStmt, DebugPrintStmt, AssertStmt)):
            pass  # No new variable definitions

    def _analyze_expr(self, expr) -> list[Interval]:
        """Analyze an expression and return its interval per component."""
        if expr is None:
            return []

        if isinstance(expr, NumberLit):
            try:
                v = float(expr.value)
                return [Interval(v, v)]
            except (ValueError, TypeError):
                return [Interval(float('-inf'), float('inf'))]

        if isinstance(expr, BoolLit):
            v = 1.0 if expr.value else 0.0
            return [Interval(v, v)]

        if isinstance(expr, VarRef):
            if expr.name in self.var_intervals:
                return _deep_copy_intervals(self.var_intervals[expr.name])
            return [Interval(float('-inf'), float('inf'))]

        if isinstance(expr, BinaryOp):
            return self._analyze_binary(expr)

        if isinstance(expr, UnaryOp):
            return self._analyze_unary(expr)

        if isinstance(expr, CallExpr):
            return self._analyze_call(expr)

        if isinstance(expr, ConstructorExpr):
            # vec3(a, b, c) → concatenate component intervals
            result = []
            for arg in expr.args:
                result.extend(self._analyze_expr(arg))
            return result

        if isinstance(expr, FieldAccess):
            # Conservative: unknown struct field
            return [Interval(float('-inf'), float('inf'))]

        if isinstance(expr, SwizzleAccess):
            base = self._analyze_expr(expr.object)
            _SWIZZLE_MAP = {'x': 0, 'y': 1, 'z': 2, 'w': 3,
                            'r': 0, 'g': 1, 'b': 2, 'a': 3}
            result = []
            for c in expr.components:
                idx = _SWIZZLE_MAP.get(c, 0)
                if idx < len(base):
                    result.append(Interval(base[idx].lo, base[idx].hi,
                                           base[idx].has_nan, base[idx].has_inf))
                else:
                    result.append(Interval(float('-inf'), float('inf')))
            return result

        if isinstance(expr, IndexAccess):
            base = self._analyze_expr(expr.object)
            if base:
                # Union of all components (conservative)
                merged = Interval()
                for iv in base:
                    merged.merge(iv)
                return [merged]
            return [Interval(float('-inf'), float('inf'))]

        if isinstance(expr, TernaryExpr):
            then_iv = self._analyze_expr(expr.then_expr)
            else_iv = self._analyze_expr(expr.else_expr)
            # Union of both branches
            result = []
            n = max(len(then_iv), len(else_iv))
            for i in range(n):
                iv = Interval()
                if i < len(then_iv):
                    iv.merge(then_iv[i])
                if i < len(else_iv):
                    iv.merge(else_iv[i])
                result.append(iv)
            return result

        return [Interval(float('-inf'), float('inf'))]

    def _analyze_binary(self, expr: BinaryOp) -> list[Interval]:
        left = self._analyze_expr(expr.left)
        right = self._analyze_expr(expr.right)

        if not left or not right:
            return [Interval(float('-inf'), float('inf'))]

        # Scalar-broadcast: expand shorter operand
        n = max(len(left), len(right))
        while len(left) < n:
            left.append(Interval(left[0].lo, left[0].hi, left[0].has_nan, left[0].has_inf) if left else Interval())
        while len(right) < n:
            right.append(Interval(right[0].lo, right[0].hi, right[0].has_nan, right[0].has_inf) if right else Interval())

        result = []
        for i in range(n):
            a, b = left[i], right[i]
            iv = _apply_binary_op(expr.op, a, b)
            result.append(iv)
        return result

    def _analyze_unary(self, expr: UnaryOp) -> list[Interval]:
        operand = self._analyze_expr(expr.operand)
        if not operand:
            return [Interval(float('-inf'), float('inf'))]

        if expr.op == "-":
            return [Interval(-iv.hi, -iv.lo, iv.has_nan, iv.has_inf) for iv in operand]
        if expr.op == "!":
            return [Interval(0, 1)]
        return operand

    def _analyze_call(self, expr: CallExpr) -> list[Interval]:
        func_name = expr.func.name if isinstance(expr.func, VarRef) else ""
        args_intervals = [self._analyze_expr(a) for a in expr.args]

        # Known function rules
        if func_name in ("sin", "cos"):
            return [Interval(-1, 1)]

        if func_name == "normalize":
            n = len(args_intervals[0]) if args_intervals else 3
            return [Interval(-1, 1) for _ in range(n)]

        if func_name == "clamp" and len(args_intervals) >= 3:
            lo_iv = args_intervals[1]
            hi_iv = args_intervals[2]
            if lo_iv and hi_iv:
                lo_val = lo_iv[0].lo if lo_iv[0].lo != float('inf') else 0.0
                hi_val = hi_iv[0].hi if hi_iv[0].hi != float('-inf') else 1.0
                n = len(args_intervals[0]) if args_intervals[0] else 1
                return [Interval(lo_val, hi_val) for _ in range(n)]
            return args_intervals[0] if args_intervals else [Interval(float('-inf'), float('inf'))]

        if func_name == "smoothstep":
            return [Interval(0, 1)]

        if func_name == "mix" and len(args_intervals) >= 3:
            a_ivs = args_intervals[0]
            b_ivs = args_intervals[1]
            t_ivs = args_intervals[2]
            # If t in [0,1], result in [min(a,b), max(a,b)]
            t_safe = t_ivs and all(iv.lo >= 0 and iv.hi <= 1 for iv in t_ivs)
            if t_safe and a_ivs and b_ivs:
                result = []
                n = max(len(a_ivs), len(b_ivs))
                for i in range(n):
                    a = a_ivs[i] if i < len(a_ivs) else a_ivs[-1]
                    b = b_ivs[i] if i < len(b_ivs) else b_ivs[-1]
                    result.append(Interval(min(a.lo, b.lo), max(a.hi, b.hi)))
                return result
            # Conservative fallback
            if a_ivs and b_ivs:
                result = []
                n = max(len(a_ivs), len(b_ivs))
                for i in range(n):
                    a = a_ivs[i] if i < len(a_ivs) else a_ivs[-1]
                    b = b_ivs[i] if i < len(b_ivs) else b_ivs[-1]
                    result.append(Interval(float('-inf'), float('inf'),
                                           a.has_nan or b.has_nan, a.has_inf or b.has_inf))
                return result

        if func_name == "abs" and args_intervals:
            result = []
            for iv in args_intervals[0]:
                abs_max = max(abs(iv.lo) if iv.lo != float('inf') else 0,
                              abs(iv.hi) if iv.hi != float('-inf') else 0)
                result.append(Interval(0, abs_max, iv.has_nan, iv.has_inf))
            return result

        if func_name in ("sample", "texture"):
            # LDR texture assumption
            return [Interval(0, 1)] * 4

        if func_name == "sqrt" and args_intervals:
            result = []
            for iv in args_intervals[0]:
                lo = math.sqrt(max(0, iv.lo)) if iv.lo != float('inf') else 0
                hi = math.sqrt(max(0, iv.hi)) if iv.hi != float('-inf') else 0
                result.append(Interval(lo, hi, iv.has_nan, iv.has_inf))
            return result

        if func_name == "exp" and args_intervals:
            result = []
            for iv in args_intervals[0]:
                try:
                    lo = math.exp(iv.lo) if iv.lo > -700 else 0.0
                except OverflowError:
                    lo = float('inf')
                try:
                    hi = math.exp(iv.hi) if iv.hi < 700 else float('inf')
                except OverflowError:
                    hi = float('inf')
                has_inf = iv.has_inf or hi == float('inf')
                result.append(Interval(lo, hi, iv.has_nan, has_inf))
            return result

        if func_name == "dot" and len(args_intervals) >= 2:
            a_ivs = args_intervals[0]
            b_ivs = args_intervals[1]
            if a_ivs and b_ivs:
                # Conservative: sum of per-component products
                lo_sum = 0.0
                hi_sum = 0.0
                n = min(len(a_ivs), len(b_ivs))
                for i in range(n):
                    prod = _apply_binary_op("*", a_ivs[i], b_ivs[i])
                    lo_sum += prod.lo if prod.lo != float('inf') else 0
                    hi_sum += prod.hi if prod.hi != float('-inf') else 0
                return [Interval(lo_sum, hi_sum)]

        if func_name == "length" and args_intervals:
            ivs = args_intervals[0]
            # [0, sqrt(sum(comp.hi^2))]
            sum_sq = sum(
                max(iv.lo ** 2, iv.hi ** 2) if not iv.is_empty else 0
                for iv in ivs
            )
            return [Interval(0, math.sqrt(sum_sq))]

        if func_name == "max" and len(args_intervals) >= 2:
            a_ivs = args_intervals[0]
            b_ivs = args_intervals[1]
            if a_ivs and b_ivs:
                result = []
                n = max(len(a_ivs), len(b_ivs))
                for i in range(n):
                    a = a_ivs[i] if i < len(a_ivs) else a_ivs[-1]
                    b = b_ivs[i] if i < len(b_ivs) else b_ivs[-1]
                    result.append(Interval(max(a.lo, b.lo), max(a.hi, b.hi)))
                return result

        if func_name == "min" and len(args_intervals) >= 2:
            a_ivs = args_intervals[0]
            b_ivs = args_intervals[1]
            if a_ivs and b_ivs:
                result = []
                n = max(len(a_ivs), len(b_ivs))
                for i in range(n):
                    a = a_ivs[i] if i < len(a_ivs) else a_ivs[-1]
                    b = b_ivs[i] if i < len(b_ivs) else b_ivs[-1]
                    result.append(Interval(min(a.lo, b.lo), min(a.hi, b.hi)))
                return result

        if func_name == "pow" and len(args_intervals) >= 2:
            base_ivs = args_intervals[0]
            exp_ivs = args_intervals[1]
            if base_ivs and exp_ivs:
                # Conservative: pow can blow up
                result = []
                for iv in base_ivs:
                    if iv.lo >= 0 and exp_ivs[0].lo >= 0:
                        try:
                            lo = iv.lo ** exp_ivs[0].hi if iv.lo > 0 else 0.0
                            hi = iv.hi ** exp_ivs[0].hi if iv.hi > 0 else 0.0
                        except (OverflowError, ValueError):
                            lo, hi = 0.0, float('inf')
                        result.append(Interval(min(lo, hi), max(lo, hi)))
                    else:
                        result.append(Interval(float('-inf'), float('inf')))
                return result

        if func_name in ("step",):
            return [Interval(0, 1)]

        if func_name in ("sign",):
            return [Interval(-1, 1)]

        if func_name == "fract" and args_intervals:
            return [Interval(0, 1) for _ in args_intervals[0]]

        if func_name in ("floor", "ceil", "round", "trunc") and args_intervals:
            return args_intervals[0]  # Same range (conservative)

        if func_name in ("atan", "asin"):
            return [Interval(-math.pi / 2, math.pi / 2)]

        if func_name in ("acos",):
            return [Interval(0, math.pi)]

        if func_name == "atan2":
            return [Interval(-math.pi, math.pi)]

        if func_name == "tan" and args_intervals:
            return [Interval(float('-inf'), float('inf'))]

        if func_name in ("log", "log2") and args_intervals:
            # log(x) → [-inf, inf] for general x
            return [Interval(float('-inf'), float('inf'))]

        if func_name == "inversesqrt" and args_intervals:
            return [Interval(0, float('inf'))]

        if func_name == "reflect" and len(args_intervals) >= 2:
            # Reflected vector has same magnitude
            n = len(args_intervals[0]) if args_intervals[0] else 3
            return [Interval(-1, 1) for _ in range(n)]  # Assuming unit input

        if func_name == "cross" and len(args_intervals) >= 2:
            # Conservative: could be any direction
            return [Interval(float('-inf'), float('inf'))] * 3

        # Unknown function: conservative
        n = len(args_intervals[0]) if args_intervals and args_intervals[0] else 1
        return [Interval(float('-inf'), float('inf'))] * n


def _apply_binary_op(op: str, a: Interval, b: Interval) -> Interval:
    """Apply a binary operation to two intervals."""
    has_nan = a.has_nan or b.has_nan
    has_inf = a.has_inf or b.has_inf

    if a.is_empty or b.is_empty:
        return Interval(has_nan=has_nan, has_inf=has_inf)

    if op == "+":
        return Interval(a.lo + b.lo, a.hi + b.hi, has_nan, has_inf)

    if op == "-":
        return Interval(a.lo - b.hi, a.hi - b.lo, has_nan, has_inf)

    if op == "*":
        products = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
        # Filter out NaN from inf * 0
        valid = [p for p in products if not math.isnan(p)]
        if not valid:
            return Interval(has_nan=True, has_inf=has_inf)
        return Interval(min(valid), max(valid), has_nan, has_inf or any(math.isinf(p) for p in valid))

    if op == "/":
        if b.lo <= 0 <= b.hi:
            # Division by zero possible
            return Interval(float('-inf'), float('inf'), has_nan, True)
        # Safe division: multiply by reciprocal
        recip = Interval(1.0 / b.hi, 1.0 / b.lo)
        return _apply_binary_op("*", a, recip)

    # Comparison operators return bool [0, 1]
    if op in ("<", ">", "<=", ">=", "==", "!="):
        return Interval(0, 1)

    # Logical operators
    if op in ("&&", "||"):
        return Interval(0, 1)

    return Interval(float('-inf'), float('inf'), has_nan, has_inf)


def _type_component_count(type_name: str) -> int:
    if type_name in ("scalar", "float"):
        return 1
    if type_name == "vec2":
        return 2
    if type_name == "vec3":
        return 3
    if type_name == "vec4":
        return 4
    if type_name == "mat2":
        return 4
    if type_name == "mat3":
        return 9
    if type_name == "mat4":
        return 16
    if type_name in ("int", "uint", "bool"):
        return 1
    if type_name in ("ivec2", "uvec2"):
        return 2
    if type_name in ("ivec3", "uvec3"):
        return 3
    if type_name in ("ivec4", "uvec4"):
        return 4
    return 1


def _guess_type_name(n_components: int) -> str:
    _MAP = {1: "scalar", 2: "vec2", 3: "vec3", 4: "vec4",
            9: "mat3", 16: "mat4"}
    return _MAP.get(n_components, "scalar")


def _deep_copy_intervals(intervals: list[Interval]) -> list[Interval]:
    return [Interval(iv.lo, iv.hi, iv.has_nan, iv.has_inf) for iv in intervals]


def _snapshot(state: dict[str, list[Interval]]) -> dict[str, list[Interval]]:
    return {name: _deep_copy_intervals(ivs) for name, ivs in state.items()}


def _merge_states(target: dict[str, list[Interval]], other: dict[str, list[Interval]]) -> None:
    """Merge other state into target (union of intervals)."""
    for name, ivs in other.items():
        if name in target:
            for i in range(min(len(target[name]), len(ivs))):
                target[name][i].merge(ivs[i])
        else:
            target[name] = _deep_copy_intervals(ivs)


def _find_modified_vars(stmts: list) -> set[str]:
    """Find variable names that are assigned inside a statement list."""
    modified: set[str] = set()
    for stmt in stmts:
        if isinstance(stmt, AssignStmt):
            name = _extract_root_name(stmt.target)
            if name:
                modified.add(name)
        elif isinstance(stmt, LetStmt):
            modified.add(stmt.name)
        elif isinstance(stmt, IfStmt):
            modified |= _find_modified_vars(stmt.then_body)
            modified |= _find_modified_vars(stmt.else_body)
        elif isinstance(stmt, (ForStmt, WhileStmt)):
            modified |= _find_modified_vars(stmt.body)
        elif isinstance(stmt, DebugBlock):
            modified |= _find_modified_vars(stmt.body)
    return modified


def _extract_root_name(target) -> str | None:
    """Extract root variable name from an assignment target."""
    from luxc.parser.ast_nodes import AssignTarget
    if isinstance(target, AssignTarget):
        return _extract_root_name(target.expr)
    if isinstance(target, VarRef):
        return target.name
    if isinstance(target, (SwizzleAccess, FieldAccess)):
        return _extract_root_name(target.object)
    if isinstance(target, IndexAccess):
        return _extract_root_name(target.object)
    return None
