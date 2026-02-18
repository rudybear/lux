"""Tests for P5: new built-in functions and stdlib expansions."""

import pytest
from luxc.parser.tree_builder import parse_lux
from luxc.analysis.type_checker import type_check, TypeCheckError
from luxc.optimization.const_fold import constant_fold
from luxc.analysis.layout_assigner import assign_layouts
from luxc.codegen.spirv_builder import generate_spirv
from luxc.builtins.functions import lookup_builtin
from luxc.builtins.types import SCALAR, VEC2, VEC3, VEC4, clear_type_aliases


def _compile_stage(src: str, stage_idx: int = 0) -> str:
    """Parse, type-check, constant-fold, and generate SPIR-V assembly."""
    m = parse_lux(src)
    type_check(m)
    constant_fold(m)
    assign_layouts(m)
    return generate_spirv(m, m.stages[stage_idx])


# ── P5.1: Built-in function registration ──


class TestRefractBuiltin:
    def test_refract_signature_vec3(self):
        sig = lookup_builtin("refract", [VEC3, VEC3, SCALAR])
        assert sig is not None
        assert sig.return_type.name == "vec3"

    def test_refract_signature_vec2(self):
        sig = lookup_builtin("refract", [VEC2, VEC2, SCALAR])
        assert sig is not None
        assert sig.return_type.name == "vec2"

    def test_refract_type_checks(self):
        src = """
        fragment {
            in I: vec3;
            in N: vec3;
            out color: vec4;
            fn main() {
                let r: vec3 = refract(I, N, 1.5);
                color = vec4(r, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)  # should not raise

    def test_refract_codegen(self):
        src = """
        fragment {
            in I: vec3;
            in N: vec3;
            out color: vec4;
            fn main() {
                let r: vec3 = refract(I, N, 1.5);
                color = vec4(r, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "Refract" in asm


class TestAtan2Builtin:
    def test_atan2_signature(self):
        sig = lookup_builtin("atan", [SCALAR, SCALAR])
        assert sig is not None
        assert sig.return_type.name == "scalar"

    def test_atan2_vec3_signature(self):
        sig = lookup_builtin("atan", [VEC3, VEC3])
        assert sig is not None
        assert sig.return_type.name == "vec3"

    def test_atan_1arg_still_works(self):
        sig = lookup_builtin("atan", [SCALAR])
        assert sig is not None
        assert sig.return_type.name == "scalar"

    def test_atan2_type_checks(self):
        src = """
        fragment {
            in x: scalar;
            in y: scalar;
            out color: vec4;
            fn main() {
                let angle: scalar = atan(y, x);
                color = vec4(angle, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)

    def test_atan2_codegen(self):
        src = """
        fragment {
            in x: scalar;
            in y: scalar;
            out color: vec4;
            fn main() {
                let angle: scalar = atan(y, x);
                color = vec4(angle, 0.0, 0.0, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "Atan2" in asm

    def test_atan_1arg_codegen(self):
        src = """
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                let a: scalar = atan(x);
                color = vec4(a, 0.0, 0.0, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        # 1-arg atan should use "Atan" not "Atan2"
        assert "Atan " in asm or "Atan\n" in asm


class TestInverseSqrtBuiltin:
    def test_inversesqrt_signature(self):
        sig = lookup_builtin("inversesqrt", [SCALAR])
        assert sig is not None
        assert sig.return_type.name == "scalar"

    def test_inversesqrt_vec3(self):
        sig = lookup_builtin("inversesqrt", [VEC3])
        assert sig is not None
        assert sig.return_type.name == "vec3"

    def test_inversesqrt_codegen(self):
        src = """
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                let r: scalar = inversesqrt(x);
                color = vec4(r, 0.0, 0.0, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "InverseSqrt" in asm


class TestModBuiltin:
    def test_mod_signature(self):
        sig = lookup_builtin("mod", [SCALAR, SCALAR])
        assert sig is not None
        assert sig.return_type.name == "scalar"

    def test_mod_vec3(self):
        sig = lookup_builtin("mod", [VEC3, VEC3])
        assert sig is not None
        assert sig.return_type.name == "vec3"

    def test_mod_codegen(self):
        src = """
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                let r: scalar = mod(x, 2.0);
                color = vec4(r, 0.0, 0.0, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "OpFMod" in asm


# ── P5.2: stdlib BRDF expansions ──


class TestBrdfStdlib:
    """Test that new BRDF functions compile when imported."""

    @pytest.fixture(autouse=True)
    def clear_aliases(self):
        clear_type_aliases()

    def _compile_with_brdf(self, body: str) -> str:
        src = f"""
        import brdf;
        fragment {{
            in n: vec3;
            in v: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {{
                {body}
            }}
        }}
        """
        from luxc.compiler import compile_source
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            clear_type_aliases()
            m = parse_lux(src)
            from luxc.compiler import _resolve_imports
            _resolve_imports(m)
            type_check(m)
            constant_fold(m)
            assign_layouts(m)
            return generate_spirv(m, m.stages[0])

    def test_v_ggx_correlated(self):
        asm = self._compile_with_brdf("""
            let result: scalar = v_ggx_correlated(0.5, 0.5, 0.3);
            color = vec4(result, 0.0, 0.0, 1.0);
        """)
        assert "OpFunction" in asm

    def test_oren_nayar_diffuse(self):
        asm = self._compile_with_brdf("""
            let result: vec3 = oren_nayar_diffuse(vec3(0.8), 0.5, 0.5, 0.5);
            color = vec4(result, 1.0);
        """)
        assert "OpFunction" in asm

    def test_burley_diffuse(self):
        asm = self._compile_with_brdf("""
            let result: vec3 = burley_diffuse(vec3(0.8), 0.5, 0.5, 0.5, 0.5);
            color = vec4(result, 1.0);
        """)
        assert "OpFunction" in asm

    def test_conductor_fresnel(self):
        asm = self._compile_with_brdf("""
            let result: vec3 = conductor_fresnel(vec3(0.95, 0.64, 0.54), vec3(0.96, 0.63, 0.53), 0.5);
            color = vec4(result, 1.0);
        """)
        assert "OpFunction" in asm

    def test_charlie_ndf(self):
        asm = self._compile_with_brdf("""
            let result: scalar = charlie_ndf(0.5, 0.8);
            color = vec4(result, 0.0, 0.0, 1.0);
        """)
        assert "OpFunction" in asm

    def test_sheen_brdf(self):
        asm = self._compile_with_brdf("""
            let result: vec3 = sheen_brdf(vec3(1.0), 0.5, 0.8, 0.5, 0.5);
            color = vec4(result, 1.0);
        """)
        assert "OpFunction" in asm

    def test_clearcoat_brdf(self):
        asm = self._compile_with_brdf("""
            let result: scalar = clearcoat_brdf(n, v, l, 1.0, 0.1);
            color = vec4(result, 0.0, 0.0, 1.0);
        """)
        assert "OpFunction" in asm

    def test_anisotropic_ggx_ndf(self):
        asm = self._compile_with_brdf("""
            let result: scalar = anisotropic_ggx_ndf(0.8, 0.3, 0.2, 0.1, 0.5);
            color = vec4(result, 0.0, 0.0, 1.0);
        """)
        assert "OpFunction" in asm

    def test_anisotropic_v_ggx(self):
        asm = self._compile_with_brdf("""
            let result: scalar = anisotropic_v_ggx(0.5, 0.5, 0.3, 0.2, 0.3, 0.2, 0.1, 0.5);
            color = vec4(result, 0.0, 0.0, 1.0);
        """)
        assert "OpFunction" in asm

    def test_volume_attenuation(self):
        asm = self._compile_with_brdf("""
            let result: vec3 = volume_attenuation(1.0, vec3(0.5, 0.2, 0.1), 2.0);
            color = vec4(result, 1.0);
        """)
        assert "OpFunction" in asm

    def test_ior_to_f0(self):
        asm = self._compile_with_brdf("""
            let result: scalar = ior_to_f0(1.5);
            color = vec4(result, 0.0, 0.0, 1.0);
        """)
        assert "OpFunction" in asm

    def test_gltf_pbr(self):
        asm = self._compile_with_brdf("""
            let result: vec3 = gltf_pbr(n, v, l, vec3(0.8, 0.2, 0.1), 0.3, 0.9);
            color = vec4(result, 1.0);
        """)
        assert "OpFunction" in asm


# ── P5.2: Colorspace stdlib ──


class TestColorspaceStdlib:
    @pytest.fixture(autouse=True)
    def clear_aliases(self):
        clear_type_aliases()

    def _compile_with_colorspace(self, body: str) -> str:
        src = f"""
        import colorspace;
        fragment {{
            in c: vec3;
            out color: vec4;
            fn main() {{
                {body}
            }}
        }}
        """
        m = parse_lux(src)
        from luxc.compiler import _resolve_imports
        _resolve_imports(m)
        type_check(m)
        constant_fold(m)
        assign_layouts(m)
        return generate_spirv(m, m.stages[0])

    def test_contrast(self):
        asm = self._compile_with_colorspace("""
            let result: vec3 = contrast(c, 0.5, 1.5);
            color = vec4(result, 1.0);
        """)
        assert "OpFunction" in asm

    def test_saturate_color(self):
        asm = self._compile_with_colorspace("""
            let result: vec3 = saturate_color(c, 1.5);
            color = vec4(result, 1.0);
        """)
        assert "OpFunction" in asm

    def test_brightness(self):
        asm = self._compile_with_colorspace("""
            let result: vec3 = brightness(c, 2.0);
            color = vec4(result, 1.0);
        """)
        assert "OpFunction" in asm

    def test_gamma_correct(self):
        asm = self._compile_with_colorspace("""
            let result: vec3 = gamma_correct(c, 2.2);
            color = vec4(result, 1.0);
        """)
        assert "OpFunction" in asm


# ── E2E: Full SPIR-V compilation + validation ──


class TestP5E2E:
    """End-to-end tests that compile to SPIR-V and validate."""

    @pytest.fixture(autouse=True)
    def clear_aliases(self):
        clear_type_aliases()

    def test_refract_glass_shader(self):
        """A simple glass refraction shader compiles end-to-end."""
        src = """
        fragment {
            in world_normal: vec3;
            in view_dir: vec3;
            out color: vec4;
            fn main() {
                let n: vec3 = normalize(world_normal);
                let v: vec3 = normalize(view_dir);
                let refracted: vec3 = refract(v, n, 0.66);
                color = vec4(refracted * 0.5 + vec3(0.5), 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "Refract" in asm
        assert "Normalize" in asm

    def test_anisotropy_rotation_shader(self):
        """A shader using atan2 for anisotropy rotation compiles."""
        src = """
        fragment {
            in tangent: vec3;
            in bitangent: vec3;
            in aniso_rotation: scalar;
            out color: vec4;
            fn main() {
                let angle: scalar = atan(tangent.y, tangent.x) + aniso_rotation;
                let rotated_x: scalar = cos(angle);
                let rotated_y: scalar = sin(angle);
                color = vec4(rotated_x, rotated_y, 0.0, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "Atan2" in asm
        assert "Sin" in asm
        assert "Cos" in asm

    def test_pbr_with_inversesqrt(self):
        """inversesqrt used in a PBR-style calculation compiles."""
        src = """
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                let fast_recip: scalar = inversesqrt(x);
                let result: scalar = fast_recip * fast_recip;
                color = vec4(result, 0.0, 0.0, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "InverseSqrt" in asm

    def test_mod_pattern(self):
        """mod used in a tiling pattern compiles."""
        src = """
        fragment {
            in uv: vec2;
            out color: vec4;
            fn main() {
                let tile_x: scalar = mod(uv.x * 10.0, 1.0);
                let tile_y: scalar = mod(uv.y * 10.0, 1.0);
                color = vec4(tile_x, tile_y, 0.0, 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "OpFMod" in asm

    def test_all_new_builtins_together(self):
        """All four new builtins used in one shader."""
        src = """
        fragment {
            in x: scalar;
            in n: vec3;
            in v: vec3;
            out color: vec4;
            fn main() {
                let a: scalar = atan(x, 1.0);
                let b: scalar = inversesqrt(x + 1.0);
                let c: scalar = mod(a, 3.14159);
                let r: vec3 = refract(v, n, 1.5);
                color = vec4(r * b + vec3(c), 1.0);
            }
        }
        """
        asm = _compile_stage(src)
        assert "Atan2" in asm
        assert "InverseSqrt" in asm
        assert "OpFMod" in asm
        assert "Refract" in asm
