"""Tests for the Lux parser and AST construction."""

import pytest
from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import (
    Module, StageBlock, FunctionDef, LetStmt, AssignStmt, ReturnStmt,
    NumberLit, BoolLit, VarRef, BinaryOp, UnaryOp, CallExpr,
    ConstructorExpr, SwizzleAccess, AssignTarget,
    TypeAlias, ImportDecl, SurfaceDecl, SurfaceMember,
    GeometryDecl, GeometryField, GeometryTransform, GeometryOutputs,
    PipelineDecl, PipelineMember,
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


class TestTypeAliases:
    def test_simple_type_alias(self):
        m = parse_lux("type Radiance = vec3;")
        assert len(m.type_aliases) == 1
        ta = m.type_aliases[0]
        assert isinstance(ta, TypeAlias)
        assert ta.name == "Radiance"
        assert ta.target_type == "vec3"

    def test_multiple_type_aliases(self):
        src = """
        type Radiance = vec3;
        type Reflectance = vec3;
        type Direction = vec3;
        type Normal = vec3;
        """
        m = parse_lux(src)
        assert len(m.type_aliases) == 4
        names = [ta.name for ta in m.type_aliases]
        assert names == ["Radiance", "Reflectance", "Direction", "Normal"]

    def test_type_alias_with_stages(self):
        src = """
        type Radiance = vec3;
        vertex {
            in pos: vec3;
            fn main() {
                builtin_position = vec4(pos, 1.0);
            }
        }
        """
        m = parse_lux(src)
        assert len(m.type_aliases) == 1
        assert len(m.stages) == 1


class TestImportDecl:
    def test_simple_import(self):
        m = parse_lux("import brdf;")
        assert len(m.imports) == 1
        assert isinstance(m.imports[0], ImportDecl)
        assert m.imports[0].module_name == "brdf"

    def test_multiple_imports(self):
        src = """
        import brdf;
        import noise;
        import sdf;
        """
        m = parse_lux(src)
        assert len(m.imports) == 3
        assert [i.module_name for i in m.imports] == ["brdf", "noise", "sdf"]


class TestSurfaceDecl:
    def test_simple_surface(self):
        src = """
        surface CopperMetal {
            brdf: lambert,
            roughness: 0.3,
        }
        """
        m = parse_lux(src)
        assert len(m.surfaces) == 1
        s = m.surfaces[0]
        assert isinstance(s, SurfaceDecl)
        assert s.name == "CopperMetal"
        assert len(s.members) == 2
        assert s.members[0].name == "brdf"
        assert s.members[1].name == "roughness"

    def test_surface_with_function_call_value(self):
        src = """
        surface Metal {
            brdf: fresnel_blend(reflection, base),
            normal: sample(normal_tex, uv),
        }
        """
        m = parse_lux(src)
        s = m.surfaces[0]
        assert isinstance(s.members[0].value, CallExpr)
        assert isinstance(s.members[1].value, CallExpr)


class TestGeometryDecl:
    def test_geometry_fields(self):
        src = """
        geometry StandardMesh {
            position: vec3,
            normal: vec3,
            uv: vec2,
        }
        """
        m = parse_lux(src)
        assert len(m.geometries) == 1
        g = m.geometries[0]
        assert isinstance(g, GeometryDecl)
        assert g.name == "StandardMesh"
        assert len(g.fields) == 3
        assert g.fields[0].name == "position"
        assert g.fields[0].type_name == "vec3"

    def test_geometry_with_transform_and_outputs(self):
        src = """
        geometry StandardMesh {
            position: vec3,
            normal: vec3,
            transform: MVP {
                model: mat4,
                view: mat4,
                projection: mat4,
            }
            outputs {
                world_pos: position,
                clip_pos: position,
            }
        }
        """
        m = parse_lux(src)
        g = m.geometries[0]
        assert g.transform is not None
        assert g.transform.name == "MVP"
        assert len(g.transform.fields) == 3
        assert g.outputs is not None
        assert len(g.outputs.bindings) == 2
        assert g.outputs.bindings[0].name == "world_pos"


class TestPipelineDecl:
    def test_simple_pipeline(self):
        src = """
        pipeline PBRForward {
            geometry: StandardMesh,
            surface: CopperMetal,
        }
        """
        m = parse_lux(src)
        assert len(m.pipelines) == 1
        p = m.pipelines[0]
        assert isinstance(p, PipelineDecl)
        assert p.name == "PBRForward"
        assert len(p.members) == 2
        assert p.members[0].name == "geometry"
        assert p.members[1].name == "surface"
