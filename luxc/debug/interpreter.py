"""AST tree-walking interpreter for the Lux shader debugger."""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field
from typing import Optional

from luxc.parser.ast_nodes import (
    Module, StageBlock, FunctionDef, LetStmt, AssignStmt, ReturnStmt,
    IfStmt, ExprStmt, NumberLit, BoolLit, VarRef, BinaryOp, UnaryOp,
    CallExpr, ConstructorExpr, FieldAccess, SwizzleAccess, IndexAccess,
    TernaryExpr, AssignTarget,
    ForStmt, WhileStmt, BreakStmt, ContinueStmt,
    DebugPrintStmt, AssertStmt, DebugBlock,
)
from luxc.debug.values import (
    LuxValue, LuxScalar, LuxVec, LuxMat, LuxInt, LuxBool, LuxStruct,
    is_nan, is_inf, default_value, value_to_json, value_type_name,
)
from luxc.debug.environment import Environment
from luxc.debug.builtins import BUILTIN_FUNCTIONS


class BreakSignal(Exception):
    """Control flow signal for break statements."""
    pass


class ContinueSignal(Exception):
    """Control flow signal for continue statements."""
    pass


class ReturnSignal(Exception):
    """Control flow signal for return statements."""
    def __init__(self, value: LuxValue):
        self.value = value


@dataclass
class NanEvent:
    """Records when a NaN/Inf is produced."""
    line: int
    variable: str
    operation: str
    value: LuxValue


@dataclass
class VarTrace:
    """Records a variable assignment for tracing."""
    line: int
    name: str
    type_name: str
    value: LuxValue


@dataclass
class InterpResult:
    """Result from running the interpreter."""
    output: LuxValue | None = None
    nan_detected: bool = False
    nan_events: list[NanEvent] = field(default_factory=list)
    statements_executed: int = 0
    variable_trace: list[VarTrace] = field(default_factory=list)
    debug_prints: list[str] = field(default_factory=list)
    assert_failures: list[str] = field(default_factory=list)


class Interpreter:
    """AST tree-walking interpreter for Lux shaders."""

    def __init__(self, module: Module, stage: StageBlock, source_lines: list[str] | None = None):
        self.module = module
        self.stage = stage
        self.source_lines = source_lines or []
        self.env = Environment()
        self.result = InterpResult()
        self._trace_all = False
        self._max_iterations = 10000  # guard against infinite loops

        # Register module-level functions
        for fn in module.functions:
            self.env.register_function(fn)
        # Register stage-local functions
        for fn in stage.functions:
            self.env.register_function(fn)

        # Register struct definitions
        for struct in module.structs:
            self.env.register_struct(
                struct.name,
                [(f.name, f.type_name) for f in struct.fields],
            )

    def run(self, inputs: dict[str, LuxValue] | None = None,
            trace_all: bool = False) -> InterpResult:
        """Run the shader stage's main function."""
        self._trace_all = trace_all

        # Set up inputs as global variables
        if inputs:
            for name, value in inputs.items():
                self.env.define(name, value)

        # Set up default inputs from stage declarations
        for inp in self.stage.inputs:
            if self.env.get(inp.name) is None:
                self.env.define(inp.name, default_value(inp.type_name))

        # Set up output variables with defaults
        for out in self.stage.outputs:
            if self.env.get(out.name) is None:
                self.env.define(out.name, default_value(out.type_name))

        # Set up uniform fields as accessible variables
        for ub in self.stage.uniforms:
            for uf in ub.fields:
                if self.env.get(uf.name) is None:
                    self.env.define(uf.name, default_value(uf.type_name))

        # Set up push constant fields
        for pb in self.stage.push_constants:
            for pf in pb.fields:
                if self.env.get(pf.name) is None:
                    self.env.define(pf.name, default_value(pf.type_name))

        # Set up sampler variables as mock handles
        for sampler in getattr(self.stage, 'samplers', []):
            if self.env.get(sampler.name) is None:
                self.env.define(sampler.name, LuxInt(0))  # mock sampler handle

        # Set up storage images, accel structs etc. as mock handles
        for si in getattr(self.stage, 'storage_images', []):
            if self.env.get(si.name) is None:
                self.env.define(si.name, LuxInt(0))

        # Set up constants
        for const in self.module.constants:
            val = self.eval_expr(const.value)
            self.env.constants[const.name] = val

        # Find and run the main function
        main_fn = None
        for fn in self.stage.functions:
            if fn.name == "main":
                main_fn = fn
                break

        if main_fn is None:
            raise RuntimeError("No main function found in stage")

        self.env.push_scope("main")
        try:
            for stmt in main_fn.body:
                self.exec_stmt(stmt)
        except ReturnSignal:
            pass
        finally:
            self.env.pop_scope()

        # Collect output from output variables
        for out in self.stage.outputs:
            val = self.env.get(out.name)
            if val is not None:
                self.result.output = val
                break

        return self.result

    def exec_stmt(self, stmt) -> None:
        """Execute a statement."""
        self.result.statements_executed += 1
        line = getattr(stmt, 'loc', None)
        line_num = line.line if line else 0

        if isinstance(stmt, LetStmt):
            val = self.eval_expr(stmt.value)
            val = self._coerce(val, stmt.type_name)
            self.env.define(stmt.name, val)
            self._check_nan(val, stmt.name, "let", line_num)
            if self._trace_all:
                self.result.variable_trace.append(
                    VarTrace(line_num, stmt.name, stmt.type_name, copy.deepcopy(val))
                )

        elif isinstance(stmt, AssignStmt):
            val = self.eval_expr(stmt.value)
            self._exec_assign(stmt.target, val)

        elif isinstance(stmt, ReturnStmt):
            val = self.eval_expr(stmt.value)
            raise ReturnSignal(val)

        elif isinstance(stmt, IfStmt):
            cond = self.eval_expr(stmt.condition)
            if self._is_truthy(cond):
                self.env.push_scope("if_then")
                try:
                    for s in stmt.then_body:
                        self.exec_stmt(s)
                finally:
                    self.env.pop_scope()
            elif stmt.else_body:
                self.env.push_scope("if_else")
                try:
                    for s in stmt.else_body:
                        self.exec_stmt(s)
                finally:
                    self.env.pop_scope()

        elif isinstance(stmt, ForStmt):
            init_val = self.eval_expr(stmt.init_value)
            self.env.push_scope("for")
            self.env.define(stmt.loop_var, init_val)
            iterations = 0
            try:
                while True:
                    cond = self.eval_expr(stmt.condition)
                    if not self._is_truthy(cond):
                        break
                    try:
                        for s in stmt.body:
                            self.exec_stmt(s)
                    except ContinueSignal:
                        pass
                    except BreakSignal:
                        break
                    # Update
                    update_val = self.eval_expr(stmt.update_value)
                    self._exec_assign(stmt.update_target, update_val)
                    iterations += 1
                    if iterations >= self._max_iterations:
                        break
            finally:
                self.env.pop_scope()

        elif isinstance(stmt, WhileStmt):
            iterations = 0
            self.env.push_scope("while")
            try:
                while True:
                    cond = self.eval_expr(stmt.condition)
                    if not self._is_truthy(cond):
                        break
                    try:
                        for s in stmt.body:
                            self.exec_stmt(s)
                    except ContinueSignal:
                        pass
                    except BreakSignal:
                        break
                    iterations += 1
                    if iterations >= self._max_iterations:
                        break
            finally:
                self.env.pop_scope()

        elif isinstance(stmt, BreakStmt):
            raise BreakSignal()

        elif isinstance(stmt, ContinueStmt):
            raise ContinueSignal()

        elif isinstance(stmt, ExprStmt):
            self.eval_expr(stmt.expr)

        elif isinstance(stmt, DebugPrintStmt):
            args = [self.eval_expr(a) for a in stmt.args]
            msg = self._format_debug_print(stmt.format_string, args)
            self.result.debug_prints.append(msg)

        elif isinstance(stmt, AssertStmt):
            cond = self.eval_expr(stmt.condition)
            if not self._is_truthy(cond):
                msg = stmt.message or "assertion failed"
                loc_str = f" [line {line_num}]" if line_num else ""
                self.result.assert_failures.append(f"ASSERT FAILED{loc_str}: {msg}")

        elif isinstance(stmt, DebugBlock):
            for s in stmt.body:
                self.exec_stmt(s)

    def _exec_assign(self, target, value: LuxValue) -> None:
        """Execute an assignment to a target (VarRef, FieldAccess, SwizzleAccess, etc.)."""
        if isinstance(target, AssignTarget):
            target = target.expr

        if isinstance(target, VarRef):
            if not self.env.set(target.name, value):
                self.env.define(target.name, value)

        elif isinstance(target, FieldAccess):
            obj = self.eval_expr(target.object)
            if isinstance(obj, LuxStruct):
                obj.fields[target.field] = value

        elif isinstance(target, SwizzleAccess):
            obj = self.eval_expr(target.object)
            if isinstance(obj, LuxVec):
                comp_map = {'x': 0, 'y': 1, 'z': 2, 'w': 3, 'r': 0, 'g': 1, 'b': 2, 'a': 3}
                src_vals = value.components if isinstance(value, LuxVec) else [_to_float(value)]
                for i, ch in enumerate(target.components):
                    idx = comp_map.get(ch, 0)
                    if idx < len(obj.components) and i < len(src_vals):
                        obj.components[idx] = src_vals[i]

        elif isinstance(target, IndexAccess):
            obj = self.eval_expr(target.object)
            idx_val = self.eval_expr(target.index)
            idx = int(_to_float(idx_val))
            if isinstance(obj, LuxVec) and 0 <= idx < len(obj.components):
                obj.components[idx] = _to_float(value)
            elif isinstance(obj, LuxMat) and 0 <= idx < len(obj.columns):
                if isinstance(value, LuxVec):
                    obj.columns[idx] = list(value.components)
            elif isinstance(obj, list) and 0 <= idx < len(obj):
                obj[idx] = value

    def eval_expr(self, expr) -> LuxValue:
        """Evaluate an expression and return its value."""
        if expr is None:
            return LuxScalar(0.0)

        if isinstance(expr, NumberLit):
            val_str = expr.value
            if '.' in val_str or 'e' in val_str.lower():
                return LuxScalar(float(val_str))
            return LuxInt(int(val_str))

        if isinstance(expr, BoolLit):
            return LuxBool(expr.value)

        if isinstance(expr, VarRef):
            val = self.env.get(expr.name)
            if val is not None:
                return val
            # Check if it's a builtin constant
            if expr.name == "PI":
                return LuxScalar(math.pi)
            if expr.name == "TAU":
                return LuxScalar(math.tau)
            if expr.name == "E":
                return LuxScalar(math.e)
            raise NameError(f"Undefined variable: {expr.name}")

        if isinstance(expr, BinaryOp):
            return self._eval_binary(expr)

        if isinstance(expr, UnaryOp):
            return self._eval_unary(expr)

        if isinstance(expr, CallExpr):
            return self._eval_call(expr)

        if isinstance(expr, ConstructorExpr):
            return self._eval_constructor(expr)

        if isinstance(expr, FieldAccess):
            obj = self.eval_expr(expr.object)
            if isinstance(obj, LuxStruct):
                if expr.field in obj.fields:
                    return obj.fields[expr.field]
                raise AttributeError(f"Struct {obj.type_name} has no field '{expr.field}'")
            # Swizzle on vec (single component like .x, .r)
            if isinstance(obj, LuxVec):
                comp_map = {'x': 0, 'y': 1, 'z': 2, 'w': 3, 'r': 0, 'g': 1, 'b': 2, 'a': 3}
                if expr.field in comp_map:
                    idx = comp_map[expr.field]
                    return LuxScalar(obj.components[idx])
            raise AttributeError(f"Cannot access field '{expr.field}' on {type(obj).__name__}")

        if isinstance(expr, SwizzleAccess):
            obj = self.eval_expr(expr.object)
            if isinstance(obj, LuxVec):
                comp_map = {'x': 0, 'y': 1, 'z': 2, 'w': 3, 'r': 0, 'g': 1, 'b': 2, 'a': 3}
                components = [obj.components[comp_map[c]] for c in expr.components]
                if len(components) == 1:
                    return LuxScalar(components[0])
                return LuxVec(components)
            raise TypeError(f"Cannot swizzle {type(obj).__name__}")

        if isinstance(expr, IndexAccess):
            obj = self.eval_expr(expr.object)
            idx = self.eval_expr(expr.index)
            i = int(_to_float(idx))
            if isinstance(obj, LuxVec):
                return LuxScalar(obj.components[i])
            if isinstance(obj, LuxMat):
                return LuxVec(list(obj.columns[i]))
            if isinstance(obj, list) and 0 <= i < len(obj):
                return obj[i]
            raise TypeError(f"Cannot index {type(obj).__name__}")

        if isinstance(expr, TernaryExpr):
            cond = self.eval_expr(expr.condition)
            if self._is_truthy(cond):
                return self.eval_expr(expr.then_expr)
            return self.eval_expr(expr.else_expr)

        raise TypeError(f"Unknown expression type: {type(expr).__name__}")

    def _eval_binary(self, expr: BinaryOp) -> LuxValue:
        left = self.eval_expr(expr.left)
        right = self.eval_expr(expr.right)
        op = expr.op

        # Comparison operators
        if op in ("==", "!=", "<", ">", "<=", ">="):
            return self._eval_comparison(op, left, right)

        # Logical operators
        if op == "&&":
            return LuxBool(self._is_truthy(left) and self._is_truthy(right))
        if op == "||":
            return LuxBool(self._is_truthy(left) or self._is_truthy(right))

        # Arithmetic
        lf = self._to_floats(left)
        rf = self._to_floats(right)
        size = max(len(lf), len(rf))
        if len(lf) == 1:
            lf = lf * size
        if len(rf) == 1:
            rf = rf * size

        if op == "+":
            result = [a + b for a, b in zip(lf, rf)]
        elif op == "-":
            result = [a - b for a, b in zip(lf, rf)]
        elif op == "*":
            # Matrix multiplication special case
            if isinstance(left, LuxMat) and isinstance(right, LuxVec):
                return self._mat_vec_mul(left, right)
            if isinstance(left, LuxMat) and isinstance(right, LuxMat):
                return self._mat_mat_mul(left, right)
            result = [a * b for a, b in zip(lf, rf)]
        elif op == "/":
            result = [a / b if b != 0.0 else float('inf') for a, b in zip(lf, rf)]
        elif op == "%":
            result = [math.fmod(a, b) if b != 0.0 else float('nan') for a, b in zip(lf, rf)]
        else:
            raise ValueError(f"Unknown binary operator: {op}")

        if size == 1:
            if isinstance(left, LuxInt) and isinstance(right, LuxInt):
                return LuxInt(int(result[0]))
            return LuxScalar(result[0])
        return LuxVec(result)

    def _eval_unary(self, expr: UnaryOp) -> LuxValue:
        val = self.eval_expr(expr.operand)
        if expr.op == "-":
            if isinstance(val, LuxScalar):
                return LuxScalar(-val.value)
            if isinstance(val, LuxInt):
                return LuxInt(-val.value)
            if isinstance(val, LuxVec):
                return LuxVec([-c for c in val.components])
            raise TypeError(f"Cannot negate {type(val).__name__}")
        if expr.op == "!":
            return LuxBool(not self._is_truthy(val))
        raise ValueError(f"Unknown unary operator: {expr.op}")

    def _eval_call(self, expr: CallExpr) -> LuxValue:
        # Get function name
        if isinstance(expr.func, VarRef):
            name = expr.func.name
        else:
            raise TypeError(f"Cannot call {type(expr.func).__name__}")

        # Evaluate arguments
        args = [self.eval_expr(a) for a in expr.args]

        # Try builtin first
        if name in BUILTIN_FUNCTIONS:
            return BUILTIN_FUNCTIONS[name](args)

        # Try user-defined function
        fn = self.env.lookup_function(name)
        if fn is not None:
            return self._call_user_function(fn, args)

        # Unknown function — return zero
        return LuxScalar(0.0)

    def _call_user_function(self, fn: FunctionDef, args: list[LuxValue]) -> LuxValue:
        self.env.push_scope(fn.name)
        try:
            # Bind parameters
            for param, arg in zip(fn.params, args):
                self.env.define(param.name, copy.deepcopy(arg))

            # Execute body
            for stmt in fn.body:
                self.exec_stmt(stmt)

            # If no return was hit, return default
            return LuxScalar(0.0)
        except ReturnSignal as ret:
            return ret.value
        finally:
            self.env.pop_scope()

    def _eval_constructor(self, expr: ConstructorExpr) -> LuxValue:
        args = [self.eval_expr(a) for a in expr.args]
        type_name = expr.type_name

        # Vector constructors
        if type_name in ("vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "uvec2", "uvec3", "uvec4"):
            size = int(type_name[-1])
            components: list[float] = []
            for arg in args:
                if isinstance(arg, LuxVec):
                    components.extend(arg.components)
                elif isinstance(arg, LuxScalar):
                    components.append(arg.value)
                elif isinstance(arg, LuxInt):
                    components.append(float(arg.value))
                elif isinstance(arg, LuxBool):
                    components.append(1.0 if arg.value else 0.0)
            # Scalar broadcast
            if len(components) == 1:
                components = components * size
            return LuxVec(components[:size])

        # Matrix constructors
        if type_name in ("mat2", "mat3", "mat4"):
            size = int(type_name[-1])
            if len(args) == 1 and isinstance(args[0], LuxScalar):
                # Diagonal constructor
                d = args[0].value
                cols = [[d if r == c else 0.0 for r in range(size)] for c in range(size)]
                return LuxMat(cols)
            if len(args) == 1 and isinstance(args[0], LuxMat):
                return copy.deepcopy(args[0])
            # Per-column construction
            if all(isinstance(a, LuxVec) for a in args) and len(args) == size:
                return LuxMat([list(a.components[:size]) for a in args])
            # Flat float list
            floats: list[float] = []
            for arg in args:
                if isinstance(arg, LuxScalar):
                    floats.append(arg.value)
                elif isinstance(arg, LuxInt):
                    floats.append(float(arg.value))
                elif isinstance(arg, LuxVec):
                    floats.extend(arg.components)
            if len(floats) >= size * size:
                cols = [floats[c * size:(c + 1) * size] for c in range(size)]
                return LuxMat(cols)
            return default_value(type_name)

        # Scalar constructors
        if type_name in ("scalar", "float"):
            return LuxScalar(_to_float(args[0]) if args else 0.0)
        if type_name == "int":
            return LuxInt(int(_to_float(args[0])) if args else 0)
        if type_name == "uint":
            return LuxInt(int(_to_float(args[0])) if args else 0, signed=False)
        if type_name == "bool":
            return LuxBool(self._is_truthy(args[0]) if args else False)

        # Struct constructor
        struct_fields = self.env.lookup_struct(type_name)
        if struct_fields is not None:
            fields = {}
            for i, (fname, ftype) in enumerate(struct_fields):
                if i < len(args):
                    fields[fname] = args[i]
                else:
                    fields[fname] = default_value(ftype)
            return LuxStruct(type_name, fields)

        raise TypeError(f"Unknown constructor type: {type_name}")

    def _eval_comparison(self, op: str, left: LuxValue, right: LuxValue) -> LuxValue:
        lf = _to_float(left)
        rf = _to_float(right)
        if op == "==":
            return LuxBool(lf == rf)
        if op == "!=":
            return LuxBool(lf != rf)
        if op == "<":
            return LuxBool(lf < rf)
        if op == ">":
            return LuxBool(lf > rf)
        if op == "<=":
            return LuxBool(lf <= rf)
        if op == ">=":
            return LuxBool(lf >= rf)
        raise ValueError(f"Unknown comparison: {op}")

    def _mat_vec_mul(self, m: LuxMat, v: LuxVec) -> LuxVec:
        n = m.size
        result = [0.0] * n
        for c in range(n):
            for r in range(n):
                if c < len(v.components):
                    result[r] += m.columns[c][r] * v.components[c]
        return LuxVec(result)

    def _mat_mat_mul(self, a: LuxMat, b: LuxMat) -> LuxMat:
        n = a.size
        result = [[0.0] * n for _ in range(n)]
        for c in range(n):
            for r in range(n):
                for k in range(n):
                    result[c][r] += a.columns[k][r] * b.columns[c][k]
        return LuxMat(result)

    def _is_truthy(self, val: LuxValue) -> bool:
        if isinstance(val, LuxBool):
            return val.value
        if isinstance(val, LuxScalar):
            return val.value != 0.0
        if isinstance(val, LuxInt):
            return val.value != 0
        return True

    def _to_floats(self, val: LuxValue) -> list[float]:
        if isinstance(val, LuxScalar):
            return [val.value]
        if isinstance(val, LuxInt):
            return [float(val.value)]
        if isinstance(val, LuxVec):
            return list(val.components)
        if isinstance(val, LuxBool):
            return [1.0 if val.value else 0.0]
        raise TypeError(f"Cannot convert {type(val).__name__} to float list")

    def _coerce(self, val: LuxValue, type_name: str) -> LuxValue:
        """Coerce a value to the target type if needed."""
        if type_name in ("scalar", "float") and isinstance(val, LuxInt):
            return LuxScalar(float(val.value))
        if type_name == "int" and isinstance(val, LuxScalar):
            return LuxInt(int(val.value))
        return val

    def _check_nan(self, val: LuxValue, name: str, op: str, line: int) -> None:
        if is_nan(val):
            event = NanEvent(line, name, op, copy.deepcopy(val))
            self.result.nan_events.append(event)
            self.result.nan_detected = True
        elif is_inf(val):
            event = NanEvent(line, name, f"{op}(inf)", copy.deepcopy(val))
            self.result.nan_events.append(event)

    def _format_debug_print(self, fmt: str, args: list[LuxValue]) -> str:
        """Format a debug_print message."""
        result = fmt
        for arg in args:
            placeholder_idx = result.find("%")
            if placeholder_idx >= 0:
                # Replace one placeholder
                end = placeholder_idx + 1
                while end < len(result) and result[end] in "0123456789.":
                    end += 1
                if end < len(result):
                    end += 1
                result = result[:placeholder_idx] + repr(arg) + result[end:]
            else:
                result += f" {arg!r}"
        return result


def _to_float(v: LuxValue) -> float:
    if isinstance(v, LuxScalar):
        return v.value
    if isinstance(v, LuxInt):
        return float(v.value)
    if isinstance(v, LuxBool):
        return 1.0 if v.value else 0.0
    raise TypeError(f"Expected scalar, got {type(v).__name__}")
