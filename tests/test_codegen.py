"""Tests for SPIR-V code generation."""

import pytest
from luxc.parser.tree_builder import parse_lux
from luxc.analysis.type_checker import type_check
from luxc.optimization.const_fold import constant_fold
from luxc.analysis.layout_assigner import assign_layouts
from luxc.codegen.spirv_builder import generate_spirv


def _compile_stage(src: str, stage_idx: int = 0) -> str:
    """Parse, type-check, constant-fold, and generate SPIR-V assembly for a stage."""
    m = parse_lux(src)
    type_check(m)
    constant_fold(m)
    assign_layouts(m)
    return generate_spirv(m, m.stages[stage_idx])


class TestVertexShaderOutput:
    def test_has_capability(self):
        asm = _compile_stage("""
        vertex {
            in pos: vec3;
            fn main() { builtin_position = vec4(pos, 1.0); }
        }
        """)
        assert "OpCapability Shader" in asm

    def test_has_entry_point(self):
        asm = _compile_stage("""
        vertex {
            in pos: vec3;
            fn main() { builtin_position = vec4(pos, 1.0); }
        }
        """)
        assert 'OpEntryPoint Vertex %main "main"' in asm

    def test_has_memory_model(self):
        asm = _compile_stage("""
        vertex {
            in pos: vec3;
            fn main() { builtin_position = vec4(pos, 1.0); }
        }
        """)
        assert "OpMemoryModel Logical GLSL450" in asm

    def test_builtin_position_decoration(self):
        asm = _compile_stage("""
        vertex {
            in pos: vec3;
            fn main() { builtin_position = vec4(pos, 1.0); }
        }
        """)
        assert "BuiltIn Position" in asm

    def test_input_location_decoration(self):
        asm = _compile_stage("""
        vertex {
            in pos: vec3;
            in color: vec3;
            out frag_color: vec3;
            fn main() {
                frag_color = color;
                builtin_position = vec4(pos, 1.0);
            }
        }
        """)
        assert "Location 0" in asm
        assert "Location 1" in asm

    def test_gl_per_vertex_block(self):
        asm = _compile_stage("""
        vertex {
            in pos: vec3;
            fn main() { builtin_position = vec4(pos, 1.0); }
        }
        """)
        assert "gl_PerVertex" in asm
        assert "Block" in asm


class TestFragmentShaderOutput:
    def test_execution_mode(self):
        asm = _compile_stage("""
        fragment {
            out color: vec4;
            fn main() { color = vec4(1.0, 0.0, 0.0, 1.0); }
        }
        """)
        assert "OpExecutionMode %main OriginUpperLeft" in asm
        assert "OpEntryPoint Fragment" in asm

    def test_no_gl_per_vertex(self):
        asm = _compile_stage("""
        fragment {
            out color: vec4;
            fn main() { color = vec4(1.0, 0.0, 0.0, 1.0); }
        }
        """)
        assert "gl_PerVertex" not in asm


class TestUniformBlocks:
    def test_uniform_block_decorations(self):
        asm = _compile_stage("""
        vertex {
            in pos: vec3;
            uniform MVP { model: mat4, view: mat4, projection: mat4 }
            fn main() { builtin_position = projection * view * model * vec4(pos, 1.0); }
        }
        """)
        assert "OpDecorate" in asm
        assert "Block" in asm
        assert "DescriptorSet 0" in asm
        assert "Binding 0" in asm
        assert "Offset 0" in asm
        assert "Offset 64" in asm
        assert "Offset 128" in asm
        assert "MatrixStride 16" in asm

    def test_access_chain_for_uniform(self):
        asm = _compile_stage("""
        vertex {
            in pos: vec3;
            uniform MVP { model: mat4 }
            fn main() {
                builtin_position = model * vec4(pos, 1.0);
            }
        }
        """)
        assert "OpAccessChain" in asm


class TestBuiltinFunctions:
    def test_normalize(self):
        asm = _compile_stage("""
        fragment {
            in n: vec3;
            out color: vec4;
            fn main() {
                let nn: vec3 = normalize(n);
                color = vec4(nn, 1.0);
            }
        }
        """)
        assert "OpExtInst" in asm
        assert "Normalize" in asm

    def test_dot_product(self):
        asm = _compile_stage("""
        fragment {
            in a: vec3;
            in b: vec3;
            out color: vec4;
            fn main() {
                let d: scalar = dot(a, b);
                color = vec4(d, d, d, 1.0);
            }
        }
        """)
        assert "OpDot" in asm

    def test_math_functions(self):
        asm = _compile_stage("""
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                let a: scalar = sin(x);
                let b: scalar = cos(x);
                let c: scalar = pow(a, b);
                let d: scalar = max(a, b);
                color = vec4(c, d, a, 1.0);
            }
        }
        """)
        assert "Sin" in asm
        assert "Cos" in asm
        assert "Pow" in asm
        assert "FMax" in asm


class TestConstructors:
    def test_vec4_from_vec3_scalar(self):
        asm = _compile_stage("""
        vertex {
            in pos: vec3;
            fn main() { builtin_position = vec4(pos, 1.0); }
        }
        """)
        assert "OpCompositeExtract" in asm
        assert "OpCompositeConstruct" in asm

    def test_vec3_splat(self):
        asm = _compile_stage("""
        fragment {
            out color: vec4;
            fn main() {
                let v: vec3 = vec3(0.5);
                color = vec4(v, 1.0);
            }
        }
        """)
        assert "OpCompositeConstruct" in asm


class TestSwizzle:
    def test_xyz_swizzle(self):
        asm = _compile_stage("""
        fragment {
            in v: vec4;
            out color: vec4;
            fn main() {
                let xyz: vec3 = v.xyz;
                color = vec4(xyz, 1.0);
            }
        }
        """)
        assert "OpVectorShuffle" in asm

    def test_single_component(self):
        asm = _compile_stage("""
        fragment {
            in v: vec4;
            out color: vec4;
            fn main() {
                let x: scalar = v.x;
                color = vec4(x, x, x, 1.0);
            }
        }
        """)
        assert "OpCompositeExtract" in asm


class TestConstantFolding:
    def test_literal_arithmetic_folded(self):
        """1.0 + 2.0 should be folded to a constant 3.0 in SPIR-V."""
        asm = _compile_stage("""
        fragment {
            out color: vec4;
            fn main() {
                let v: scalar = 1.0 + 2.0;
                color = vec4(v, v, v, 1.0);
            }
        }
        """)
        # After folding, the literal 3.0 should appear as a constant
        assert "OpConstant" in asm and "3.0" in asm
        # No OpFAdd needed — the addition was folded at compile time
        assert "OpFAdd" not in asm

    def test_constant_reference_inlined(self):
        """Module-level const should be inlined as a literal."""
        asm = _compile_stage("""
        const HALF: scalar = 0.5;
        fragment {
            out color: vec4;
            fn main() {
                let v: scalar = HALF;
                color = vec4(v, v, v, 1.0);
            }
        }
        """)
        # HALF should be inlined to 0.5 constant
        assert "OpConstant" in asm and "0.5" in asm
