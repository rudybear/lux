"""GLSL-to-Lux transpiler — converts a GlslModule into Lux source text."""

from __future__ import annotations
from dataclasses import dataclass, field

from luxc.transpiler.glsl_ast import (
    GlslModule, GlslInOut, GlslUniform, GlslGlobalVar,
    GlslFunction, GlslParam,
    GlslVarDecl, GlslAssign, GlslCompoundAssign, GlslReturn,
    GlslIf, GlslFor, GlslWhile, GlslIncrDecr, GlslExprStmt,
    GlslNumberLit, GlslBoolLit, GlslVarRef, GlslBinaryOp, GlslUnaryOp,
    GlslCall, GlslFieldAccess, GlslSwizzle, GlslIndex, GlslTernary,
)


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, str] = {
    "float": "scalar",
    "sampler2D": "sampler2d",
    "samplerCube": "sampler2d",  # approximate
}

# Builtin function mapping
_FUNC_MAP: dict[str, str] = {
    "texture": "sample",
    "texture2D": "sample",
}

# Variable mapping
_VAR_MAP: dict[str, str] = {
    "gl_Position": "builtin_position",
    "gl_FragColor": "color",
    "gl_FragCoord": "frag_coord",
}

# BRDF-related function names that suggest importing brdf module
_BRDF_FUNCS = {"fresnel_schlick", "lambert_brdf", "pbr_brdf", "microfacet_brdf"}

# Color-related function names
_COLOR_FUNCS = {"linear_to_srgb", "srgb_to_linear", "tonemap_aces", "tonemap_reinhard"}


def _map_type(glsl_type: str) -> str:
    return _TYPE_MAP.get(glsl_type, glsl_type)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class TranspileResult:
    lux_source: str
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------

def transpile_glsl_to_lux(source: str) -> TranspileResult:
    """Transpile GLSL source text to Lux source text."""
    from luxc.transpiler.glsl_parser import parse_glsl
    module = parse_glsl(source)
    emitter = LuxEmitter(module)
    return emitter.emit()


class LuxEmitter:
    """Generates formatted Lux source from a GlslModule."""

    def __init__(self, module: GlslModule):
        self.module = module
        self.warnings: list[str] = []
        self.lines: list[str] = []
        self._indent = 0
        self._used_funcs: set[str] = set()
        # Collect all function calls for import detection
        for fn in module.functions:
            self._collect_func_calls(fn.body)

    def emit(self) -> TranspileResult:
        """Generate the complete Lux source."""
        # Determine stage type
        stage_type = self._infer_stage_type()

        # Emit header comment
        self._line("// Transpiled from GLSL to Lux")
        if self.module.version:
            self._line(f"// Original GLSL version: {self.module.version}")
        self._line("")

        # Emit imports if needed
        if self._used_funcs & _BRDF_FUNCS:
            self._line("import brdf;")
        if self._used_funcs & _COLOR_FUNCS:
            self._line("import color;")
        if (self._used_funcs & _BRDF_FUNCS) or (self._used_funcs & _COLOR_FUNCS):
            self._line("")

        # Emit const globals
        for g in self.module.globals:
            if g.is_const and g.value is not None:
                self._line(f"const {g.name}: {_map_type(g.type_name)} = {self._emit_expr(g.value)};")
        if any(g.is_const for g in self.module.globals):
            self._line("")

        # Emit non-main helper functions as module-level functions
        helper_fns = [f for f in self.module.functions if f.name != "main"]
        for fn in helper_fns:
            self._emit_function(fn, module_level=True)
            self._line("")

        # Emit stage block
        self._line(f"{stage_type} {{")
        self._indent += 1

        # Inputs
        for inp in self.module.inputs:
            self._line(f"in {self._map_var_name(inp.name)}: {_map_type(inp.type_name)};")

        # Outputs
        for out in self.module.outputs:
            self._line(f"out {self._map_var_name(out.name)}: {_map_type(out.type_name)};")

        # Uniforms
        for uni in self.module.uniforms:
            if uni.type_name in ("sampler2D", "samplerCube"):
                self._line(f"sampler2d {uni.name};")
            else:
                self._line(f"// uniform {uni.name}: {_map_type(uni.type_name)}")
                self.warnings.append(f"Non-sampler uniform '{uni.name}' needs manual conversion")

        # Blank line before functions
        if self.module.inputs or self.module.outputs or self.module.uniforms:
            self._line("")

        # Emit main function
        main_fn = next((f for f in self.module.functions if f.name == "main"), None)
        if main_fn:
            self._emit_function(main_fn, module_level=False)

        self._indent -= 1
        self._line("}")

        return TranspileResult("\n".join(self.lines), self.warnings)

    def _infer_stage_type(self) -> str:
        """Infer whether this is a vertex or fragment shader."""
        # Check for gl_Position usage in function bodies
        for fn in self.module.functions:
            if self._uses_gl_position(fn.body):
                return "vertex"

        # Check output types — vec4 named color/fragColor suggests fragment
        for out in self.module.outputs:
            if out.name.lower() in ("color", "fragcolor", "gl_fragcolor", "outcolor"):
                return "fragment"

        # Default to fragment
        return "fragment"

    def _uses_gl_position(self, stmts) -> bool:
        """Check if any statement references gl_Position."""
        for stmt in stmts:
            if isinstance(stmt, GlslAssign):
                if isinstance(stmt.target, GlslVarRef) and stmt.target.name == "gl_Position":
                    return True
            elif isinstance(stmt, GlslIf):
                if self._uses_gl_position(stmt.then_body) or self._uses_gl_position(stmt.else_body):
                    return True
        return False

    def _collect_func_calls(self, stmts):
        """Recursively collect function names used in statements."""
        for stmt in stmts:
            if isinstance(stmt, GlslVarDecl) and stmt.value:
                self._collect_expr_calls(stmt.value)
            elif isinstance(stmt, (GlslAssign, GlslCompoundAssign)):
                self._collect_expr_calls(stmt.value)
            elif isinstance(stmt, GlslReturn) and stmt.value:
                self._collect_expr_calls(stmt.value)
            elif isinstance(stmt, GlslIf):
                self._collect_expr_calls(stmt.condition)
                self._collect_func_calls(stmt.then_body)
                self._collect_func_calls(stmt.else_body)
            elif isinstance(stmt, GlslExprStmt):
                self._collect_expr_calls(stmt.expr)

    def _collect_expr_calls(self, expr):
        """Collect function names from an expression."""
        if isinstance(expr, GlslCall):
            self._used_funcs.add(expr.func)
            for a in expr.args:
                self._collect_expr_calls(a)
        elif isinstance(expr, GlslBinaryOp):
            self._collect_expr_calls(expr.left)
            self._collect_expr_calls(expr.right)
        elif isinstance(expr, GlslUnaryOp):
            self._collect_expr_calls(expr.operand)
        elif isinstance(expr, GlslTernary):
            self._collect_expr_calls(expr.condition)
            self._collect_expr_calls(expr.then_expr)
            self._collect_expr_calls(expr.else_expr)
        elif isinstance(expr, (GlslFieldAccess, GlslSwizzle)):
            self._collect_expr_calls(expr.obj)
        elif isinstance(expr, GlslIndex):
            self._collect_expr_calls(expr.obj)
            self._collect_expr_calls(expr.index)

    def _map_var_name(self, name: str) -> str:
        return _VAR_MAP.get(name, name)

    def _emit_function(self, fn: GlslFunction, module_level: bool):
        ret = _map_type(fn.return_type)
        params_str = ", ".join(
            f"{p.name}: {_map_type(p.type_name)}" for p in fn.params
        )
        ret_clause = f" -> {ret}" if ret != "void" else ""
        self._line(f"fn {fn.name}({params_str}){ret_clause} {{")
        self._indent += 1
        self._emit_stmts(fn.body)
        self._indent -= 1
        self._line("}")

    def _emit_stmts(self, stmts):
        for stmt in stmts:
            self._emit_stmt(stmt)

    def _emit_stmt(self, stmt):
        if isinstance(stmt, GlslVarDecl):
            t = _map_type(stmt.type_name)
            if stmt.value is not None:
                self._line(f"let {stmt.name}: {t} = {self._emit_expr(stmt.value)};")
            else:
                # Uninitialized — emit with zero init
                zero = self._zero_init(t)
                self._line(f"let {stmt.name}: {t} = {zero};")
                self.warnings.append(f"Uninitialized variable '{stmt.name}' zero-initialized")

        elif isinstance(stmt, GlslAssign):
            target = self._emit_lvalue(stmt.target)
            self._line(f"{target} = {self._emit_expr(stmt.value)};")

        elif isinstance(stmt, GlslCompoundAssign):
            # Convert x += e -> x = x + e
            target = self._emit_lvalue(stmt.target)
            op = stmt.op[0]  # "+=" -> "+"
            self._line(f"{target} = {target} {op} {self._emit_expr(stmt.value)};")

        elif isinstance(stmt, GlslReturn):
            if stmt.value is not None:
                self._line(f"return {self._emit_expr(stmt.value)};")
            else:
                self._line("return;")

        elif isinstance(stmt, GlslIf):
            self._line(f"if ({self._emit_expr(stmt.condition)}) {{")
            self._indent += 1
            self._emit_stmts(stmt.then_body)
            self._indent -= 1
            if stmt.else_body:
                self._line("} else {")
                self._indent += 1
                self._emit_stmts(stmt.else_body)
                self._indent -= 1
            self._line("}")

        elif isinstance(stmt, GlslFor):
            self._line("// UNSUPPORTED: for loop (Lux does not support loops)")
            self.warnings.append("for loop is not supported in Lux")

        elif isinstance(stmt, GlslWhile):
            self._line("// UNSUPPORTED: while loop (Lux does not support loops)")
            self.warnings.append("while loop is not supported in Lux")

        elif isinstance(stmt, GlslIncrDecr):
            target = self._emit_lvalue(stmt.target)
            self._line(f"// UNSUPPORTED: {target}{stmt.op}")
            self.warnings.append(f"increment/decrement operator '{stmt.op}' not supported")

        elif isinstance(stmt, GlslExprStmt):
            self._line(f"{self._emit_expr(stmt.expr)};")

        elif isinstance(stmt, list):
            # Compound statement body
            self._emit_stmts(stmt)

    def _emit_expr(self, expr) -> str:
        if isinstance(expr, GlslNumberLit):
            val = expr.value
            # Strip f suffix
            if val.endswith("f") or val.endswith("F"):
                val = val[:-1]
            return val

        elif isinstance(expr, GlslBoolLit):
            return "true" if expr.value else "false"

        elif isinstance(expr, GlslVarRef):
            return self._map_var_name(expr.name)

        elif isinstance(expr, GlslBinaryOp):
            left = self._emit_expr(expr.left)
            right = self._emit_expr(expr.right)
            return f"({left} {expr.op} {right})"

        elif isinstance(expr, GlslUnaryOp):
            operand = self._emit_expr(expr.operand)
            return f"({expr.op}{operand})"

        elif isinstance(expr, GlslCall):
            func = _FUNC_MAP.get(expr.func, expr.func)
            args = ", ".join(self._emit_expr(a) for a in expr.args)
            # Handle type constructors
            if func == "float":
                func = "scalar"
            return f"{func}({args})"

        elif isinstance(expr, GlslFieldAccess):
            obj = self._emit_expr(expr.obj)
            return f"{obj}.{expr.field}"

        elif isinstance(expr, GlslSwizzle):
            obj = self._emit_expr(expr.obj)
            return f"{obj}.{expr.components}"

        elif isinstance(expr, GlslIndex):
            obj = self._emit_expr(expr.obj)
            idx = self._emit_expr(expr.index)
            return f"{obj}[{idx}]"

        elif isinstance(expr, GlslTernary):
            cond = self._emit_expr(expr.condition)
            then = self._emit_expr(expr.then_expr)
            els = self._emit_expr(expr.else_expr)
            return f"({cond} ? {then} : {els})"

        return str(expr)

    def _emit_lvalue(self, lv) -> str:
        if isinstance(lv, GlslVarRef):
            return self._map_var_name(lv.name)
        elif isinstance(lv, GlslFieldAccess):
            return f"{self._emit_lvalue(lv.obj)}.{lv.field}"
        elif isinstance(lv, GlslSwizzle):
            return f"{self._emit_lvalue(lv.obj)}.{lv.components}"
        elif isinstance(lv, GlslIndex):
            return f"{self._emit_lvalue(lv.obj)}[{self._emit_expr(lv.index)}]"
        return str(lv)

    def _zero_init(self, type_name: str) -> str:
        """Generate a zero-value for a type."""
        if type_name == "scalar":
            return "0.0"
        elif type_name in ("int", "uint"):
            return "0"
        elif type_name == "bool":
            return "false"
        elif type_name.startswith("vec"):
            return f"{type_name}(0.0)"
        elif type_name.startswith("mat"):
            return f"{type_name}(0.0)"
        return "0.0"

    def _line(self, text: str):
        indent = "    " * self._indent
        self.lines.append(f"{indent}{text}")
