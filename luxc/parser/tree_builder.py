"""Lark Transformer that builds our AST from the parse tree."""

from __future__ import annotations
from pathlib import Path
from lark import Lark, Transformer, Token, Tree

from luxc.parser.ast_nodes import (
    Module, ConstDecl, StructDef, StructField, StageBlock,
    VarDecl, UniformBlock, PushBlock, BlockField, SamplerDecl,
    FunctionDef, Param, LetStmt, AssignStmt, ReturnStmt, IfStmt, ExprStmt,
    NumberLit, BoolLit, VarRef, BinaryOp, UnaryOp, CallExpr,
    ConstructorExpr, FieldAccess, SwizzleAccess, IndexAccess, TernaryExpr,
    AssignTarget, SourceLocation,
    TypeAlias, ImportDecl,
    SurfaceDecl, SurfaceMember,
    GeometryDecl, GeometryField, GeometryTransform, GeometryOutputs, OutputBinding,
    PipelineDecl, PipelineMember,
)

_CONSTRUCTOR_TYPES = frozenset({
    "vec2", "vec3", "vec4",
    "ivec2", "ivec3", "ivec4",
    "uvec2", "uvec3", "uvec4",
    "mat2", "mat3", "mat4",
})

_GRAMMAR_PATH = Path(__file__).parent.parent / "grammar" / "lux.lark"

_parser = Lark(
    _GRAMMAR_PATH.read_text(encoding="utf-8"),
    parser="earley",
    propagate_positions=True,
)


def _loc(meta) -> SourceLocation | None:
    if hasattr(meta, "line"):
        return SourceLocation(meta.line, meta.column)
    return None


def _tok_loc(tok: Token) -> SourceLocation | None:
    if tok is not None and hasattr(tok, "line"):
        return SourceLocation(tok.line, tok.column)
    return None


class LuxTransformer(Transformer):
    # --- Module ---

    def start(self, items):
        mod = Module()
        for item in items:
            if isinstance(item, ConstDecl):
                mod.constants.append(item)
            elif isinstance(item, FunctionDef):
                mod.functions.append(item)
            elif isinstance(item, StructDef):
                mod.structs.append(item)
            elif isinstance(item, StageBlock):
                mod.stages.append(item)
            elif isinstance(item, TypeAlias):
                mod.type_aliases.append(item)
            elif isinstance(item, ImportDecl):
                mod.imports.append(item)
            elif isinstance(item, SurfaceDecl):
                mod.surfaces.append(item)
            elif isinstance(item, GeometryDecl):
                mod.geometries.append(item)
            elif isinstance(item, PipelineDecl):
                mod.pipelines.append(item)
        return mod

    # --- Top-level declarations ---

    def const_decl(self, args):
        name, type_node, value = args[0], args[1], args[2]
        return ConstDecl(str(name), _extract_type(type_node), value, _tok_loc(name))

    def struct_def(self, args):
        name = args[0]
        fields = [a for a in args[1:] if isinstance(a, StructField)]
        return StructDef(str(name), fields, _tok_loc(name))

    def struct_field(self, args):
        return StructField(str(args[0]), _extract_type(args[1]))

    def type_alias(self, args):
        name = args[0]
        target = _extract_type(args[1])
        return TypeAlias(str(name), target, _tok_loc(name))

    def import_decl(self, args):
        name = args[0]
        return ImportDecl(str(name), _tok_loc(name))

    # --- Surface declarations ---

    def surface_decl(self, args):
        name = args[0]
        members = [a for a in args[1:] if isinstance(a, SurfaceMember)]
        samplers = [str(a) for a in args[1:] if isinstance(a, Token) and a.type == "IDENT" and a not in [name]]
        # Collect sampler names from surface_sampler rules (returned as strings)
        sampler_names = [a for a in args[1:] if isinstance(a, str) and a not in [str(name)]]
        return SurfaceDecl(str(name), members, samplers=sampler_names, loc=_tok_loc(name))

    def surface_sampler(self, args):
        return str(args[0])  # Return sampler name as string

    def surface_member(self, args):
        return SurfaceMember(str(args[0]), args[1])

    # --- Geometry declarations ---

    def geometry_decl(self, args):
        name = args[0]
        fields = []
        transform = None
        outputs = None
        for a in args[1:]:
            if isinstance(a, GeometryField):
                fields.append(a)
            elif isinstance(a, GeometryTransform):
                transform = a
            elif isinstance(a, GeometryOutputs):
                outputs = a
        return GeometryDecl(str(name), fields, transform, outputs, _tok_loc(name))

    def geometry_field(self, args):
        return GeometryField(str(args[0]), _extract_type(args[1]))

    def geometry_transform(self, args):
        name = args[0]
        fields = [a for a in args[1:] if isinstance(a, BlockField)]
        return GeometryTransform(str(name), fields)

    def geometry_outputs(self, args):
        bindings = [a for a in args if isinstance(a, OutputBinding)]
        return GeometryOutputs(bindings)

    def output_binding(self, args):
        return OutputBinding(str(args[0]), args[1])

    # --- Pipeline declarations ---

    def pipeline_decl(self, args):
        name = args[0]
        members = [a for a in args[1:] if isinstance(a, PipelineMember)]
        return PipelineDecl(str(name), members, _tok_loc(name))

    def pipeline_member(self, args):
        return PipelineMember(str(args[0]), args[1])

    # --- Stage blocks ---

    def stage_block(self, args):
        stage_type = str(args[0])
        block = StageBlock(stage_type=stage_type)
        for item in args[1:]:
            if isinstance(item, VarDecl):
                if item._is_input:
                    block.inputs.append(item)
                else:
                    block.outputs.append(item)
            elif isinstance(item, UniformBlock):
                block.uniforms.append(item)
            elif isinstance(item, PushBlock):
                block.push_constants.append(item)
            elif isinstance(item, SamplerDecl):
                block.samplers.append(item)
            elif isinstance(item, FunctionDef):
                block.functions.append(item)
        return block

    def in_decl(self, args):
        name, type_node = args[0], args[1]
        v = VarDecl(str(name), _extract_type(type_node), loc=_tok_loc(name))
        v._is_input = True
        return v

    def out_decl(self, args):
        name, type_node = args[0], args[1]
        v = VarDecl(str(name), _extract_type(type_node), loc=_tok_loc(name))
        v._is_input = False
        return v

    def uniform_block(self, args):
        name = args[0]
        fields = [a for a in args[1:] if isinstance(a, BlockField)]
        return UniformBlock(str(name), fields, loc=_tok_loc(name))

    def push_block(self, args):
        name = args[0]
        fields = [a for a in args[1:] if isinstance(a, BlockField)]
        return PushBlock(str(name), fields, loc=_tok_loc(name))

    def block_field(self, args):
        return BlockField(str(args[0]), _extract_type(args[1]))

    def sampler_decl(self, args):
        return SamplerDecl(str(args[0]), loc=_tok_loc(args[0]))

    # --- Functions ---

    def function_def(self, args):
        name = args[0]
        idx = 1
        params = []
        if idx < len(args) and isinstance(args[idx], list):
            params = args[idx]
            idx += 1
        ret_type = None
        if idx < len(args) and isinstance(args[idx], str):
            ret_type = args[idx]
            idx += 1
        body = list(args[idx:])
        return FunctionDef(str(name), params, ret_type, body, _tok_loc(name))

    def param_list(self, args):
        return list(args)

    def param(self, args):
        return Param(str(args[0]), _extract_type(args[1]))

    # --- Type ---

    def base_type(self, args):
        return str(args[0])

    # --- Statements ---

    def let_stmt(self, args):
        name, type_node, value = args[0], args[1], args[2]
        return LetStmt(str(name), _extract_type(type_node), value, _tok_loc(name))

    def assign_stmt(self, args):
        target, value = args[0], args[1]
        return AssignStmt(target, value)

    def return_stmt(self, args):
        return ReturnStmt(args[0])

    def if_stmt(self, args):
        condition = args[0]
        then_body = []
        else_body = []
        collecting_else = False
        for a in args[1:]:
            if a is None:
                collecting_else = True
            elif isinstance(a, list):
                if not collecting_else:
                    then_body = a
                else:
                    else_body = a
            else:
                if not collecting_else:
                    then_body.append(a)
                else:
                    else_body.append(a)
        return IfStmt(condition, then_body, else_body)

    def expr_stmt(self, args):
        return ExprStmt(args[0])

    # --- Assignment targets ---

    def assign_ident(self, args):
        return AssignTarget(VarRef(str(args[0]), _tok_loc(args[0])))

    def assign_swizzle(self, args):
        return AssignTarget(args[0])

    def assign_field(self, args):
        return AssignTarget(args[0])

    def assign_index(self, args):
        return AssignTarget(args[0])

    # --- Expressions ---

    def number_lit(self, args):
        return NumberLit(str(args[0]), _tok_loc(args[0]))

    def bool_true(self, args):
        return BoolLit(True)

    def bool_false(self, args):
        return BoolLit(False)

    def var_ref(self, args):
        return VarRef(str(args[0]), _tok_loc(args[0]))

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
        op, operand = str(args[0]), args[1]
        return UnaryOp(op, operand)

    def call_expr(self, args):
        func = args[0]
        call_args = args[1] if len(args) > 1 and isinstance(args[1], list) else []
        # Detect type constructors parsed as function calls
        if isinstance(func, VarRef) and func.name in _CONSTRUCTOR_TYPES:
            return ConstructorExpr(func.name, call_args, func.loc)
        return CallExpr(func, call_args)

    def constructor_expr(self, args):
        type_name = str(args[0])
        call_args = args[1] if len(args) > 1 and isinstance(args[1], list) else []
        return ConstructorExpr(type_name, call_args)

    def field_access(self, args):
        return FieldAccess(args[0], str(args[1]))

    def swizzle_access(self, args):
        return SwizzleAccess(args[0], str(args[1]))

    def index_access(self, args):
        return IndexAccess(args[0], args[1])

    def ternary_expr(self, args):
        if len(args) == 1:
            return args[0]
        return TernaryExpr(args[0], args[1], args[2])

    def arg_list(self, args):
        return list(args)


def _left_assoc(args, op):
    result = args[0]
    for i in range(1, len(args)):
        result = BinaryOp(op, result, args[i])
    return result


def _left_assoc_ops(args):
    """Handle interleaved value/op/value/op/value lists."""
    if len(args) == 1:
        return args[0]
    result = args[0]
    i = 1
    while i < len(args):
        op = str(args[i])
        right = args[i + 1]
        result = BinaryOp(op, result, right)
        i += 2
    return result


def _extract_type(node) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, Token):
        return str(node)
    if isinstance(node, Tree):
        return str(node.children[0])
    return str(node)


def parse_lux(source: str) -> Module:
    tree = _parser.parse(source)
    return LuxTransformer().transform(tree)
