"""GLSL subset parser — builds a GlslModule from GLSL source text."""

from __future__ import annotations
from pathlib import Path
from lark import Lark, Transformer, Token, Tree

from luxc.transpiler.glsl_ast import (
    GlslModule, GlslInOut, GlslUniform, GlslGlobalVar,
    GlslFunction, GlslParam,
    GlslVarDecl, GlslAssign, GlslCompoundAssign, GlslReturn,
    GlslIf, GlslFor, GlslWhile, GlslIncrDecr, GlslExprStmt,
    GlslNumberLit, GlslBoolLit, GlslVarRef, GlslBinaryOp, GlslUnaryOp,
    GlslCall, GlslFieldAccess, GlslSwizzle, GlslIndex, GlslTernary,
)

_GRAMMAR_PATH = Path(__file__).parent.parent / "grammar" / "glsl_subset.lark"

_CONSTRUCTOR_TYPES = frozenset({
    "vec2", "vec3", "vec4",
    "ivec2", "ivec3", "ivec4",
    "uvec2", "uvec3", "uvec4",
    "mat2", "mat3", "mat4",
    "float", "int", "uint", "bool",
})

_parser = Lark(
    _GRAMMAR_PATH.read_text(encoding="utf-8"),
    parser="earley",
    propagate_positions=True,
)


class GlslTransformer(Transformer):
    """Transforms a Lark parse tree into GlslModule."""

    def start(self, items):
        mod = GlslModule()
        for item in items:
            if item is None:
                continue
            if isinstance(item, int):
                mod.version = item
            elif isinstance(item, GlslInOut):
                if item.qualifier == "in":
                    mod.inputs.append(item)
                else:
                    mod.outputs.append(item)
            elif isinstance(item, GlslUniform):
                mod.uniforms.append(item)
            elif isinstance(item, GlslGlobalVar):
                mod.globals.append(item)
            elif isinstance(item, GlslFunction):
                mod.functions.append(item)
        return mod

    # --- Directives ---

    def version_directive(self, args):
        return int(str(args[0]))

    def precision_decl(self, args):
        return None  # ignored

    def preprocessor_line(self, args):
        return None  # ignored

    # --- Layout / IO declarations ---

    def layout_decl(self, args):
        layout_loc = None
        type_name = None
        name = None
        qualifier = None
        is_uniform = False

        for a in args:
            if isinstance(a, int):
                layout_loc = a
            elif isinstance(a, Token):
                if a.type == "GLSL_TYPE_NAME":
                    type_name = str(a)
                elif a.type == "IDENT":
                    name = str(a)
            elif isinstance(a, str):
                if a in ("in", "out", "varying", "attribute"):
                    qualifier = a
                elif a == "uniform":
                    is_uniform = True
                else:
                    type_name = type_name or a

        if type_name is None or name is None:
            return None

        if is_uniform:
            return GlslUniform(type_name, name)

        if qualifier in ("in", "attribute"):
            return GlslInOut("in", type_name, name, layout_loc)
        elif qualifier in ("out", "varying"):
            return GlslInOut("out", type_name, name, layout_loc)
        return GlslInOut("in", type_name, name, layout_loc)

    def layout_qualifier(self, args):
        # Extract location from layout params
        for a in args:
            if isinstance(a, int):
                return a
        return None

    def layout_param(self, args):
        name = str(args[0])
        val = int(str(args[1]))
        if name == "location":
            return val
        return None

    def io_qualifier(self, args):
        quals = [str(a) for a in args if isinstance(a, Token)]
        # Return the meaningful qualifier
        for q in quals:
            if q in ("in", "out", "varying", "attribute"):
                return q
        return quals[0] if quals else "in"

    # --- Global variable declarations ---

    def const_global_decl(self, args):
        is_const = False
        type_name = None
        name = None
        value = None
        for a in args:
            if isinstance(a, Token):
                s = str(a)
                if s == "const":
                    is_const = True
                elif a.type == "GLSL_TYPE_NAME":
                    type_name = s
                elif a.type == "IDENT":
                    name = s
                else:
                    type_name = type_name or s
            elif not isinstance(a, str) or a == "const":
                if a == "const":
                    is_const = True
                elif type_name is None and isinstance(a, str):
                    type_name = a
                else:
                    value = a
            else:
                type_name = type_name or a
        # Remaining non-token, non-string items are the value expression
        for a in args:
            if not isinstance(a, (Token, str)) and a is not None:
                value = a
                break
        if type_name and name:
            return GlslGlobalVar(type_name, name, value, is_const)
        return None

    def plain_global_decl(self, args):
        type_name = None
        name = None
        for a in args:
            if isinstance(a, Token):
                s = str(a)
                if a.type == "GLSL_TYPE_NAME":
                    type_name = s
                elif a.type == "IDENT":
                    name = s
        if type_name and name:
            return GlslGlobalVar(type_name, name)
        return None

    # --- Functions ---

    def function_def(self, args):
        ret_type = str(args[0])
        name = str(args[1])
        params = []
        body = []
        for a in args[2:]:
            if isinstance(a, list):
                if all(isinstance(x, GlslParam) for x in a):
                    params = a
                else:
                    body = a
        return GlslFunction(ret_type, name, params, body)

    def func_param_list(self, args):
        return [a for a in args if isinstance(a, GlslParam)]

    def func_param(self, args):
        qualifier = None
        type_name = None
        name = None
        for a in args:
            if isinstance(a, Token):
                s = str(a)
                if s in ("in", "out", "inout"):
                    qualifier = s
                elif a.type == "GLSL_TYPE_NAME":
                    type_name = s
                elif a.type == "IDENT":
                    name = s
        if type_name and name:
            return GlslParam(type_name, name, qualifier)
        return None

    # --- Statements ---

    def compound_stmt(self, args):
        return [a for a in args if a is not None]

    def var_decl_stmt(self, args):
        type_name = str(args[0])
        name = str(args[1])
        value = args[2] if len(args) > 2 else None
        return GlslVarDecl(type_name, name, value)

    def assign_stmt(self, args):
        return GlslAssign(args[0], args[1])

    def compound_assign_stmt(self, args):
        return GlslCompoundAssign(args[0], str(args[1]), args[2])

    def return_stmt(self, args):
        return GlslReturn(args[0] if args else None)

    def if_stmt(self, args):
        cond = args[0]
        then_body = args[1] if len(args) > 1 else []
        else_body = args[2] if len(args) > 2 else []
        # Normalize: single statement -> list
        if not isinstance(then_body, list):
            then_body = [then_body]
        if not isinstance(else_body, list):
            else_body = [else_body]
        return GlslIf(cond, then_body, else_body)

    def for_stmt(self, args):
        return GlslFor()

    def for_init_decl(self, args):
        return None

    def for_init_expr(self, args):
        return None

    def for_update_expr(self, args):
        return None

    def for_update_compound(self, args):
        return None

    def for_update_incr(self, args):
        return None

    def for_update_decr(self, args):
        return None

    def while_stmt(self, args):
        return GlslWhile()

    def incr_decr_stmt(self, args):
        return GlslIncrDecr(args[0], str(args[1]))

    def expr_stmt(self, args):
        return GlslExprStmt(args[0])

    # --- Lvalue ---

    def lvalue_ident(self, args):
        return GlslVarRef(str(args[0]))

    def lvalue_field(self, args):
        return GlslFieldAccess(args[0], str(args[1]))

    def lvalue_swizzle(self, args):
        return GlslSwizzle(args[0], str(args[1]))

    def lvalue_index(self, args):
        return GlslIndex(args[0], args[1])

    # --- Expressions ---

    def float_lit(self, args):
        # Strip trailing 'f' suffix
        val = str(args[0])
        if val.endswith("f") or val.endswith("F"):
            val = val[:-1]
        return GlslNumberLit(val)

    def int_lit(self, args):
        val = str(args[0])
        # Strip unsigned suffix
        if val.endswith("u") or val.endswith("U"):
            val = val[:-1]
        return GlslNumberLit(val)

    def bool_true(self, args):
        return GlslBoolLit(True)

    def bool_false(self, args):
        return GlslBoolLit(False)

    def var_ref(self, args):
        return GlslVarRef(str(args[0]))

    def or_expr(self, args):
        return _left_assoc(args, "||")

    def and_expr(self, args):
        return _left_assoc(args, "&&")

    def equality_expr(self, args):
        return _left_assoc_ops(args)

    def comparison_expr(self, args):
        return _left_assoc_ops(args)

    def additive_expr(self, args):
        return _left_assoc_ops(args)

    def multiplicative_expr(self, args):
        return _left_assoc_ops(args)

    def unary(self, args):
        op = str(args[0])
        return GlslUnaryOp(op, args[1])

    def prefix_incr(self, args):
        return GlslIncrDecr(args[1], str(args[0]) + "pre")

    def call_expr(self, args):
        func = args[0]
        call_args = args[1] if len(args) > 1 and isinstance(args[1], list) else []
        # Detect constructor calls
        if isinstance(func, GlslVarRef) and func.name in _CONSTRUCTOR_TYPES:
            return GlslCall(func.name, call_args)
        if isinstance(func, GlslVarRef):
            return GlslCall(func.name, call_args)
        # Nested call
        return GlslCall(str(func), call_args)

    def swizzle_access(self, args):
        return GlslSwizzle(args[0], str(args[1]))

    def field_access(self, args):
        return GlslFieldAccess(args[0], str(args[1]))

    def index_access(self, args):
        return GlslIndex(args[0], args[1])

    def ternary_expr(self, args):
        if len(args) == 1:
            return args[0]
        return GlslTernary(args[0], args[1], args[2])

    def postfix_incr(self, args):
        return GlslIncrDecr(args[0], str(args[1]))

    def arg_list(self, args):
        return list(args)


def _left_assoc(args, op):
    result = args[0]
    for i in range(1, len(args)):
        result = GlslBinaryOp(op, result, args[i])
    return result


def _left_assoc_ops(args):
    if len(args) == 1:
        return args[0]
    result = args[0]
    i = 1
    while i < len(args):
        op = str(args[i])
        right = args[i + 1]
        result = GlslBinaryOp(op, result, right)
        i += 2
    return result


def parse_glsl(source: str) -> GlslModule:
    """Parse GLSL source text into a GlslModule."""
    tree = _parser.parse(source)
    return GlslTransformer().transform(tree)
