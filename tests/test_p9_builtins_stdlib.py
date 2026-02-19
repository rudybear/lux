"""Tests for P9: new built-in functions, samplerCube, and stdlib modules."""

import pytest
from luxc.parser.tree_builder import parse_lux
from luxc.analysis.type_checker import type_check, TypeCheckError
from luxc.optimization.const_fold import constant_fold
from luxc.analysis.layout_assigner import assign_layouts
from luxc.codegen.spirv_builder import generate_spirv
from luxc.builtins.functions import lookup_builtin
from luxc.builtins.types import (
    SCALAR, VEC2, VEC3, VEC4, MAT2, MAT3, MAT4,
    SAMPLER_CUBE, clear_type_aliases,
)


def _compile_stage(src: str, stage_idx: int = 0) -> str:
    """Parse, type-check, constant-fold, and generate SPIR-V assembly."""
    clear_type_aliases()
    m = parse_lux(src)
    from luxc.compiler import _resolve_imports
    _resolve_imports(m)
    from luxc.expansion.surface_expander import expand_surfaces
    if m.surfaces or m.pipelines or m.environments or m.procedurals:
        expand_surfaces(m)
    from luxc.autodiff.forward_diff import autodiff_expand
    autodiff_expand(m)
    type_check(m)
    constant_fold(m)
    assign_layouts(m)
    return generate_spirv(m, m.stages[stage_idx])


# ── New math built-ins: registration ──

class TestNewMathBuiltins:
    @pytest.mark.parametrize("fn", [
        "round", "trunc", "radians", "degrees",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    ])
    def test_scalar_overload(self, fn):
        sig = lookup_builtin(fn, [SCALAR])
        assert sig is not None
        assert sig.return_type.name == "scalar"

    @pytest.mark.parametrize("fn", [
        "round", "trunc", "radians", "degrees",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    ])
    def test_vec3_overload(self, fn):
        sig = lookup_builtin(fn, [VEC3])
        assert sig is not None
        assert sig.return_type.name == "vec3"

    def test_faceforward_vec3(self):
        sig = lookup_builtin("faceforward", [VEC3, VEC3, VEC3])
        assert sig is not None
        assert sig.return_type.name == "vec3"

    def test_faceforward_scalar(self):
        sig = lookup_builtin("faceforward", [SCALAR, SCALAR, SCALAR])
        assert sig is not None
        assert sig.return_type.name == "scalar"


class TestMatrixBuiltins:
    @pytest.mark.parametrize("mat_type", [MAT2, MAT3, MAT4])
    def test_determinant(self, mat_type):
        sig = lookup_builtin("determinant", [mat_type])
        assert sig is not None
        assert sig.return_type.name == "scalar"

    @pytest.mark.parametrize("mat_type", [MAT2, MAT3, MAT4])
    def test_inverse(self, mat_type):
        sig = lookup_builtin("inverse", [mat_type])
        assert sig is not None
        assert sig.return_type.name == mat_type.name

    @pytest.mark.parametrize("mat_type", [MAT2, MAT3, MAT4])
    def test_transpose(self, mat_type):
        sig = lookup_builtin("transpose", [mat_type])
        assert sig is not None
        assert sig.return_type.name == mat_type.name


# ── New math built-ins: codegen ──

class TestNewBuiltinCodegen:
    def test_round_generates_spirv(self):
        src = """
        fragment {
            in val: scalar;
            out color: vec4;
            fn main() {
                let r: scalar = round(val);
                color = vec4(r, 0.0, 0.0, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "Round" in asm

    def test_sinh_generates_spirv(self):
        src = """
        fragment {
            in val: scalar;
            out color: vec4;
            fn main() {
                let s: scalar = sinh(val);
                color = vec4(s, 0.0, 0.0, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "Sinh" in asm

    def test_radians_generates_spirv(self):
        src = """
        fragment {
            in val: scalar;
            out color: vec4;
            fn main() {
                let r: scalar = radians(val);
                color = vec4(r, 0.0, 0.0, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "Radians" in asm

    def test_faceforward_generates_spirv(self):
        src = """
        fragment {
            in n: vec3;
            in i: vec3;
            in nref: vec3;
            out color: vec4;
            fn main() {
                let ff: vec3 = faceforward(n, i, nref);
                color = vec4(ff, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "FaceForward" in asm

    def test_transpose_generates_spirv(self):
        src = """
        vertex {
            in position: vec3;
            uniform MVP { model: mat4, view: mat4, proj: mat4, }
            fn main() {
                let mt: mat4 = transpose(model);
                builtin_position = mt * vec4(position, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "OpTranspose" in asm

    def test_determinant_generates_spirv(self):
        src = """
        fragment {
            in val: scalar;
            out color: vec4;
            uniform M { m: mat4, }
            fn main() {
                let d: scalar = determinant(m);
                color = vec4(d, 0.0, 0.0, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "Determinant" in asm

    def test_inverse_generates_spirv(self):
        src = """
        vertex {
            in position: vec3;
            uniform MVP { model: mat4, }
            fn main() {
                let inv: mat4 = inverse(model);
                builtin_position = inv * vec4(position, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "MatrixInverse" in asm


# ── samplerCube ──

class TestSamplerCube:
    def test_sampler_cube_lookup(self):
        sig = lookup_builtin("sample", [SAMPLER_CUBE, VEC3])
        assert sig is not None
        assert sig.return_type.name == "vec4"

    def test_sampler_cube_parse(self):
        src = """
        fragment {
            in dir: vec3;
            out color: vec4;
            samplerCube env_map;
            fn main() {
                color = sample(env_map, dir);
            }
        }
        """
        clear_type_aliases()
        m = parse_lux(src)
        assert len(m.stages[0].samplers) == 1
        assert m.stages[0].samplers[0].type_name == "samplerCube"

    def test_sampler_cube_type_checks(self):
        src = """
        fragment {
            in dir: vec3;
            out color: vec4;
            samplerCube env_map;
            fn main() {
                color = sample(env_map, dir);
            }
        }
        """
        asm = _compile_stage(src)
        assert "Cube" in asm
        assert "OpImageSampleImplicitLod" in asm


# ── stdlib: lighting ──

class TestLightingStdlib:
    def test_import_lighting(self):
        src = """
        import lighting;

        fragment {
            in frag_pos: vec3;
            in frag_normal: vec3;
            out color: vec4;
            fn main() {
                let n: vec3 = normalize(frag_normal);
                let light: vec3 = evaluate_directional_light(
                    vec3(0.0, 1.0, 0.0), vec3(1.0), 1.0, n
                );
                color = vec4(light, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "OpEntryPoint" in asm

    def test_point_light(self):
        src = """
        import lighting;

        fragment {
            in frag_pos: vec3;
            in frag_normal: vec3;
            out color: vec4;
            fn main() {
                let n: vec3 = normalize(frag_normal);
                let light: vec3 = evaluate_point_light(
                    vec3(2.0, 2.0, 2.0), vec3(1.0), 1.0, 10.0,
                    frag_pos, n
                );
                color = vec4(light, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "OpEntryPoint" in asm

    def test_spot_light(self):
        src = """
        import lighting;

        fragment {
            in frag_pos: vec3;
            in frag_normal: vec3;
            out color: vec4;
            fn main() {
                let n: vec3 = normalize(frag_normal);
                let light: vec3 = evaluate_spot_light(
                    vec3(2.0, 3.0, 2.0), vec3(0.0, -1.0, 0.0),
                    vec3(1.0), 1.0, 20.0, 0.9, 0.7,
                    frag_pos, n
                );
                color = vec4(light, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "OpEntryPoint" in asm


# ── stdlib: ibl ──

class TestIBLStdlib:
    def test_import_ibl(self):
        src = """
        import brdf;
        import ibl;

        fragment {
            in n_dot_v: scalar;
            out color: vec4;
            fn main() {
                let spec: vec3 = ibl_specular_contribution(
                    n_dot_v, 0.5, vec3(0.04), vec2(0.8, 0.1)
                );
                color = vec4(spec, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "OpEntryPoint" in asm

    def test_ibl_diffuse(self):
        src = """
        import brdf;
        import ibl;

        fragment {
            in n_dot_v: scalar;
            out color: vec4;
            fn main() {
                let diff: vec3 = ibl_diffuse_contribution(
                    vec3(0.5), vec3(0.8, 0.2, 0.1), 0.0,
                    vec3(0.04), n_dot_v
                );
                color = vec4(diff, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "OpEntryPoint" in asm

    def test_sh_irradiance(self):
        src = """
        import brdf;
        import ibl;

        fragment {
            in frag_normal: vec3;
            out color: vec4;
            fn main() {
                let n: vec3 = normalize(frag_normal);
                let l0: vec3 = sh_irradiance_l0(vec3(0.5));
                let l1: vec3 = sh_irradiance_l1(
                    vec3(0.1), vec3(0.2), vec3(0.3), n
                );
                color = vec4(l0 + l1, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "OpEntryPoint" in asm

    def test_brdf_lut_approximation(self):
        src = """
        import brdf;
        import ibl;

        fragment {
            in n_dot_v: scalar;
            out color: vec4;
            fn main() {
                let lut: vec2 = importance_sample_ggx_approx(n_dot_v, 0.5);
                color = vec4(lut.x, lut.y, 0.0, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "OpEntryPoint" in asm


# ── sample_lod ──

class TestSampleLod:
    def test_sample_lod_2d_lookup(self):
        from luxc.builtins.types import SAMPLER2D
        sig = lookup_builtin("sample_lod", [SAMPLER2D, VEC2, SCALAR])
        assert sig is not None
        assert sig.return_type.name == "vec4"

    def test_sample_lod_cube_lookup(self):
        sig = lookup_builtin("sample_lod", [SAMPLER_CUBE, VEC3, SCALAR])
        assert sig is not None
        assert sig.return_type.name == "vec4"

    def test_sample_lod_2d_codegen(self):
        src = """
        fragment {
            in uv: vec2;
            out color: vec4;
            sampler2d tex;
            fn main() {
                color = sample_lod(tex, uv, 2.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "OpImageSampleExplicitLod" in asm
        assert "Lod" in asm

    def test_sample_lod_cube_codegen(self):
        src = """
        fragment {
            in dir: vec3;
            out color: vec4;
            samplerCube env_map;
            fn main() {
                color = sample_lod(env_map, dir, 3.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "OpImageSampleExplicitLod" in asm
        assert "Lod" in asm
        assert "Cube" in asm
