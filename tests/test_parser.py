"""Tests for the Lux parser and AST construction."""

import pytest
from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import (
    Module, StageBlock, FunctionDef, LetStmt, AssignStmt, ReturnStmt,
    NumberLit, BoolLit, VarRef, BinaryOp, UnaryOp, CallExpr,
    ConstructorExpr, SwizzleAccess, AssignTarget,
)


class TestBasicParsing:
    def test_empty_module(self):
        m = parse_lux("")
        assert isinstance(m, Module)
        assert m.stages == []
        assert m.constants == []

    def test_const_decl(self):
        m = parse_lux('const PI: scalar = 3.14159;')
        assert len(m.constants) == 1
        assert m.constants[0].name == "PI"
        assert m.constants[0].type_name == "scalar"

    def test_minimal_vertex(self):
        src = """
        vertex {
            in position: vec3;
            out frag_color: vec3;
            fn main() {
                frag_color = position;
                builtin_position = vec4(position, 1.0);
            }
        }
        """
        m = parse_lux(src)
        assert len(m.stages) == 1
        stage = m.stages[0]
        assert stage.stage_type == "vertex"
        assert len(stage.inputs) == 1
        assert stage.inputs[0].name == "position"
        assert stage.inputs[0].type_name == "vec3"
        assert len(stage.outputs) == 1
        assert len(stage.functions) == 1
        assert stage.functions[0].name == "main"

    def test_minimal_fragment(self):
        src = """
        fragment {
            in frag_color: vec3;
            out color: vec4;
            fn main() {
                color = vec4(frag_color, 1.0);
            }
        }
        """
        m = parse_lux(src)
        assert len(m.stages) == 1
        assert m.stages[0].stage_type == "fragment"

    def test_two_stages(self):
        src = """
        vertex {
            in pos: vec3;
            fn main() {
                builtin_position = vec4(pos, 1.0);
            }
        }
        fragment {
            out color: vec4;
            fn main() {
                color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        assert len(m.stages) == 2
        assert m.stages[0].stage_type == "vertex"
        assert m.stages[1].stage_type == "fragment"


class TestExpressions:
    def test_number_literal(self):
        src = """
        vertex {
            out x: vec4;
            fn main() {
                x = vec4(1.0, 2.0, 3.0, 4.0);
            }
        }
        """
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        assert len(fn.body) == 1
        stmt = fn.body[0]
        assert isinstance(stmt, AssignStmt)
        assert isinstance(stmt.value, ConstructorExpr)
        assert stmt.value.type_name == "vec4"
        assert len(stmt.value.args) == 4

    def test_binary_ops(self):
        src = """
        vertex {
            in a: vec3;
            in b: vec3;
            out r: vec3;
            fn main() {
                r = a + b;
                builtin_position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        stmt = fn.body[0]
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == "+"

    def test_swizzle(self):
        src = """
        fragment {
            in v: vec4;
            out color: vec4;
            fn main() {
                color = vec4(v.xyz, 1.0);
            }
        }
        """
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        ctor = fn.body[0].value
        assert isinstance(ctor, ConstructorExpr)
        assert isinstance(ctor.args[0], SwizzleAccess)
        assert ctor.args[0].components == "xyz"

    def test_constructor_parsed_as_constructor(self):
        src = """
        fragment {
            out color: vec4;
            fn main() {
                color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        value = fn.body[0].value
        assert isinstance(value, ConstructorExpr)
        assert value.type_name == "vec4"

    def test_function_call(self):
        src = """
        fragment {
            in n: vec3;
            out color: vec4;
            fn main() {
                let nn: vec3 = normalize(n);
                color = vec4(nn, 1.0);
            }
        }
        """
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        let_stmt = fn.body[0]
        assert isinstance(let_stmt, LetStmt)
        assert isinstance(let_stmt.value, CallExpr)

    def test_unary_negation(self):
        src = """
        vertex {
            in a: scalar;
            out b: scalar;
            fn main() {
                b = -a;
                builtin_position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        assert isinstance(fn.body[0].value, UnaryOp)
        assert fn.body[0].value.op == "-"


class TestDeclarations:
    def test_uniform_block(self):
        src = """
        vertex {
            in pos: vec3;
            uniform MVP {
                model: mat4,
                view: mat4,
                projection: mat4,
            }
            fn main() {
                builtin_position = projection * view * model * vec4(pos, 1.0);
            }
        }
        """
        m = parse_lux(src)
        stage = m.stages[0]
        assert len(stage.uniforms) == 1
        ub = stage.uniforms[0]
        assert ub.name == "MVP"
        assert len(ub.fields) == 3
        assert ub.fields[0].name == "model"
        assert ub.fields[0].type_name == "mat4"

    def test_push_block(self):
        src = """
        fragment {
            out color: vec4;
            push Camera { view_pos: vec3 }
            fn main() {
                color = vec4(view_pos, 1.0);
            }
        }
        """
        m = parse_lux(src)
        stage = m.stages[0]
        assert len(stage.push_constants) == 1
        pb = stage.push_constants[0]
        assert pb.name == "Camera"
        assert len(pb.fields) == 1

    def test_sampler_decl(self):
        src = """
        fragment {
            in uv: vec2;
            out color: vec4;
            sampler2d tex;
            fn main() {
                color = sample(tex, uv);
            }
        }
        """
        m = parse_lux(src)
        stage = m.stages[0]
        assert len(stage.samplers) == 1
        assert stage.samplers[0].name == "tex"

    def test_module_level_function(self):
        src = """
        fn helper(x: vec3) -> vec3 {
            return normalize(x);
        }
        vertex {
            in n: vec3;
            out r: vec3;
            fn main() {
                r = helper(n);
                builtin_position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        assert len(m.functions) == 1
        assert m.functions[0].name == "helper"
        assert m.functions[0].return_type == "vec3"
