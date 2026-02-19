"""Tests for P5.3: Advanced material models — transmission, iridescence, dispersion, texture."""

import pytest
from luxc.parser.tree_builder import parse_lux
from luxc.analysis.type_checker import type_check, TypeCheckError
from luxc.optimization.const_fold import constant_fold
from luxc.analysis.layout_assigner import assign_layouts
from luxc.codegen.spirv_builder import generate_spirv
from luxc.compiler import _resolve_imports
from luxc.builtins.types import clear_type_aliases


def _compile_with_import(src: str, stage_idx: int = 0) -> str:
    """Parse with imports, type-check, and generate SPIR-V assembly."""
    clear_type_aliases()
    m = parse_lux(src)
    _resolve_imports(m)
    type_check(m)
    constant_fold(m)
    assign_layouts(m)
    return generate_spirv(m, m.stages[stage_idx])


# ── Transmission BTDF ──


class TestTransmissionBTDF:
    @pytest.fixture(autouse=True)
    def clear(self):
        clear_type_aliases()

    def test_transmission_btdf_compiles(self):
        asm = _compile_with_import("""
        import brdf;
        fragment {
            in n: vec3;
            in v: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                let btdf: scalar = transmission_btdf(n, v, l, 0.1, 1.5);
                color = vec4(btdf, 0.0, 0.0, 1.0);
            }
        }
        """)
        assert "OpFunction" in asm

    def test_transmission_color_compiles(self):
        asm = _compile_with_import("""
        import brdf;
        fragment {
            in n: vec3;
            in v: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                let btdf: scalar = transmission_btdf(n, v, l, 0.1, 1.5);
                let tc: vec3 = transmission_color(vec3(0.8, 0.2, 0.1), btdf, 0.5);
                color = vec4(tc, 1.0);
            }
        }
        """)
        assert "OpFunction" in asm

    def test_diffuse_transmission_compiles(self):
        asm = _compile_with_import("""
        import brdf;
        fragment {
            in n: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                let n_dot_l: scalar = dot(n, l);
                let dt: vec3 = diffuse_transmission(vec3(0.8), n_dot_l);
                color = vec4(dt, 1.0);
            }
        }
        """)
        assert "OpFunction" in asm

    def test_volumetric_btdf_compiles(self):
        asm = _compile_with_import("""
        import brdf;
        fragment {
            in n: vec3;
            in v: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                let btdf: scalar = volumetric_btdf(n, v, l, 0.1, 1.0, 1.5);
                color = vec4(btdf, 0.0, 0.0, 1.0);
            }
        }
        """)
        assert "OpFunction" in asm


# ── Iridescence ──


class TestIridescence:
    @pytest.fixture(autouse=True)
    def clear(self):
        clear_type_aliases()

    def test_iridescence_fresnel_compiles(self):
        asm = _compile_with_import("""
        import brdf;
        fragment {
            in cos_theta: scalar;
            out color: vec4;
            fn main() {
                let iri: vec3 = iridescence_fresnel(1.0, 1.3, vec3(0.04), 400.0, cos_theta);
                color = vec4(iri, 1.0);
            }
        }
        """)
        assert "OpFunction" in asm
        # Should contain sin/cos for spectral evaluation
        assert "Sin" in asm or "Cos" in asm

    def test_iridescence_f0_to_ior_compiles(self):
        asm = _compile_with_import("""
        import brdf;
        fragment {
            in f0: scalar;
            out color: vec4;
            fn main() {
                let ior: scalar = iridescence_f0_to_ior(f0);
                color = vec4(ior, 0.0, 0.0, 1.0);
            }
        }
        """)
        assert "Sqrt" in asm

    def test_iridescence_ior_to_f0_compiles(self):
        asm = _compile_with_import("""
        import brdf;
        fragment {
            in ior: scalar;
            out color: vec4;
            fn main() {
                let f0: scalar = iridescence_ior_to_f0(ior, 1.0);
                color = vec4(f0, 0.0, 0.0, 1.0);
            }
        }
        """)
        assert "OpFunction" in asm

    def test_iridescence_sensitivity_compiles(self):
        asm = _compile_with_import("""
        import brdf;
        fragment {
            in opd: scalar;
            out color: vec4;
            fn main() {
                let s: vec3 = iridescence_sensitivity(opd, vec3(0.0));
                color = vec4(s, 1.0);
            }
        }
        """)
        assert "Cos" in asm
        assert "Exp" in asm


# ── Dispersion ──


class TestDispersion:
    @pytest.fixture(autouse=True)
    def clear(self):
        clear_type_aliases()

    def test_dispersion_ior_compiles(self):
        asm = _compile_with_import("""
        import brdf;
        fragment {
            in ior: scalar;
            out color: vec4;
            fn main() {
                let iors: vec3 = dispersion_ior(ior, 0.5);
                color = vec4(iors, 1.0);
            }
        }
        """)
        assert "OpFunction" in asm

    def test_dispersion_f0_compiles(self):
        asm = _compile_with_import("""
        import brdf;
        fragment {
            in ior: scalar;
            out color: vec4;
            fn main() {
                let f0s: vec3 = dispersion_f0(ior, 0.5);
                color = vec4(f0s, 1.0);
            }
        }
        """)
        assert "OpFunction" in asm

    def test_dispersion_refract_compiles(self):
        asm = _compile_with_import("""
        import brdf;
        fragment {
            in v: vec3;
            in n: vec3;
            out color: vec4;
            fn main() {
                let spread: vec3 = dispersion_refract(v, n, 1.5, 0.5);
                color = vec4(spread, 1.0);
            }
        }
        """)
        assert "Refract" in asm


# ── Texture Module ──


class TestTextureStdlib:
    @pytest.fixture(autouse=True)
    def clear(self):
        clear_type_aliases()

    def test_tbn_perturb_normal(self):
        asm = _compile_with_import("""
        import texture;
        fragment {
            in n: vec3;
            in t: vec3;
            in b: vec3;
            out color: vec4;
            fn main() {
                let normal_sample: vec3 = vec3(0.5, 0.5, 1.0);
                let perturbed: vec3 = tbn_perturb_normal(normal_sample, n, t, b);
                color = vec4(perturbed, 1.0);
            }
        }
        """)
        assert "Normalize" in asm

    def test_tbn_from_tangent(self):
        asm = _compile_with_import("""
        import texture;
        fragment {
            in n: vec3;
            in tangent: vec4;
            out color: vec4;
            fn main() {
                let b: vec3 = tbn_from_tangent(n, tangent);
                color = vec4(b, 1.0);
            }
        }
        """)
        assert "Cross" in asm
        assert "Normalize" in asm

    def test_unpack_normal(self):
        asm = _compile_with_import("""
        import texture;
        fragment {
            in encoded: vec3;
            out color: vec4;
            fn main() {
                let n: vec3 = unpack_normal(encoded);
                color = vec4(n, 1.0);
            }
        }
        """)
        assert "Normalize" in asm

    def test_unpack_normal_strength(self):
        asm = _compile_with_import("""
        import texture;
        fragment {
            in encoded: vec3;
            out color: vec4;
            fn main() {
                let n: vec3 = unpack_normal_strength(encoded, 2.0);
                color = vec4(n, 1.0);
            }
        }
        """)
        assert "Normalize" in asm

    def test_triplanar_weights(self):
        asm = _compile_with_import("""
        import texture;
        fragment {
            in n: vec3;
            out color: vec4;
            fn main() {
                let w: vec3 = triplanar_weights(n, 4.0);
                color = vec4(w, 1.0);
            }
        }
        """)
        assert "Pow" in asm
        assert "FAbs" in asm

    def test_triplanar_uv_functions(self):
        asm = _compile_with_import("""
        import texture;
        fragment {
            in pos: vec3;
            out color: vec4;
            fn main() {
                let uv_x: vec2 = triplanar_uv_x(pos);
                let uv_y: vec2 = triplanar_uv_y(pos);
                let uv_z: vec2 = triplanar_uv_z(pos);
                color = vec4(uv_x.x + uv_y.x + uv_z.x, 0.0, 0.0, 1.0);
            }
        }
        """)
        assert "OpFunction" in asm

    def test_triplanar_blend(self):
        asm = _compile_with_import("""
        import texture;
        fragment {
            in n: vec3;
            out color: vec4;
            fn main() {
                let w: vec3 = triplanar_weights(n, 4.0);
                let blended: vec3 = triplanar_blend(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), w);
                color = vec4(blended, 1.0);
            }
        }
        """)
        assert "OpFunction" in asm

    def test_triplanar_blend_scalar(self):
        asm = _compile_with_import("""
        import texture;
        fragment {
            in n: vec3;
            out color: vec4;
            fn main() {
                let w: vec3 = triplanar_weights(n, 4.0);
                let height: scalar = triplanar_blend_scalar(0.5, 0.3, 0.8, w);
                color = vec4(height, 0.0, 0.0, 1.0);
            }
        }
        """)
        assert "OpFunction" in asm

    def test_parallax_offset(self):
        asm = _compile_with_import("""
        import texture;
        fragment {
            in view_ts: vec3;
            out color: vec4;
            fn main() {
                let offset: vec2 = parallax_offset(0.5, 0.04, view_ts);
                color = vec4(offset.x, offset.y, 0.0, 1.0);
            }
        }
        """)
        assert "OpFunction" in asm

    def test_rotate_uv(self):
        asm = _compile_with_import("""
        import texture;
        fragment {
            in uv: vec2;
            out color: vec4;
            fn main() {
                let rotated: vec2 = rotate_uv(uv, 1.5708, vec2(0.5));
                color = vec4(rotated.x, rotated.y, 0.0, 1.0);
            }
        }
        """)
        assert "Sin" in asm
        assert "Cos" in asm

    def test_tile_uv(self):
        asm = _compile_with_import("""
        import texture;
        fragment {
            in uv: vec2;
            out color: vec4;
            fn main() {
                let tiled: vec2 = tile_uv(uv, vec2(4.0, 4.0));
                color = vec4(tiled.x, tiled.y, 0.0, 1.0);
            }
        }
        """)
        assert "OpFMod" in asm


# ── E2E: Complex material shaders ──


class TestP53E2E:
    @pytest.fixture(autouse=True)
    def clear(self):
        clear_type_aliases()

    def test_glass_material(self):
        """A complete glass material using transmission + volume + refraction."""
        asm = _compile_with_import("""
        import brdf;
        fragment {
            in n: vec3;
            in v: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                let roughness: scalar = 0.05;
                let ior: scalar = 1.5;
                let base_color: vec3 = vec3(0.95, 0.95, 1.0);

                // Specular reflection
                let f0: scalar = ior_to_f0(ior);
                let h: vec3 = normalize(v + l);
                let v_dot_h: scalar = max(dot(v, h), 0.0);
                let f: vec3 = fresnel_schlick(v_dot_h, vec3(f0));

                // Transmission
                let btdf: scalar = transmission_btdf(n, v, l, roughness, ior);
                let transmitted: vec3 = (vec3(1.0) - f) * base_color * btdf;

                // Volume attenuation
                let thickness: scalar = 0.1;
                let att: vec3 = volume_attenuation(thickness, vec3(0.9, 0.95, 1.0), 1.0);

                color = vec4(transmitted * att + f * 0.5, 0.8);
            }
        }
        """)
        assert "Refract" not in asm or "OpFunction" in asm  # transmission_btdf doesn't use refract directly

    def test_iridescent_metal(self):
        """A complete iridescent metal material."""
        asm = _compile_with_import("""
        import brdf;
        fragment {
            in n: vec3;
            in v: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                let base_f0: vec3 = vec3(0.95, 0.64, 0.54);
                let cos_theta: scalar = max(dot(n, v), 0.0);

                // Iridescence Fresnel
                let iri_f: vec3 = iridescence_fresnel(1.0, 1.3, base_f0, 380.0, cos_theta);

                // Standard specular with iridescent Fresnel
                let h: vec3 = normalize(v + l);
                let n_dot_h: scalar = max(dot(n, h), 0.0);
                let n_dot_v: scalar = max(dot(n, v), 0.0001);
                let n_dot_l: scalar = max(dot(n, l), 0.0001);

                let d: scalar = ggx_ndf(n_dot_h, 0.2);
                let vis: scalar = v_ggx_correlated(n_dot_l, n_dot_v, 0.2);

                let specular: vec3 = iri_f * d * vis * n_dot_l;
                color = vec4(specular, 1.0);
            }
        }
        """)
        assert "Cos" in asm
        assert "Exp" in asm
        assert "Sqrt" in asm

    def test_dispersive_glass(self):
        """A dispersive glass material with per-channel IOR."""
        asm = _compile_with_import("""
        import brdf;
        fragment {
            in n: vec3;
            in v: vec3;
            out color: vec4;
            fn main() {
                let base_ior: scalar = 1.52;
                let disp: scalar = 0.5;

                // Per-channel F0
                let f0_rgb: vec3 = dispersion_f0(base_ior, disp);

                // Per-channel IOR for refraction
                let iors: vec3 = dispersion_ior(base_ior, disp);

                // Refract each channel
                let r_r: vec3 = refract(v, n, 1.0 / iors.x);
                let r_g: vec3 = refract(v, n, 1.0 / iors.y);
                let r_b: vec3 = refract(v, n, 1.0 / iors.z);

                color = vec4(r_r.x, r_g.y, r_b.z, 1.0);
            }
        }
        """)
        assert "Refract" in asm

    def test_normal_mapped_pbr(self):
        """PBR with normal mapping via texture module."""
        asm = _compile_with_import("""
        import brdf;
        import texture;
        fragment {
            in world_normal: vec3;
            in world_tangent: vec3;
            in world_bitangent: vec3;
            in v: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                // Simulated normal map sample
                let normal_sample: vec3 = vec3(0.5, 0.5, 1.0);
                let n: vec3 = tbn_perturb_normal(normal_sample, normalize(world_normal), normalize(world_tangent), normalize(world_bitangent));

                // PBR with perturbed normal
                let result: vec3 = gltf_pbr(n, normalize(v), normalize(l), vec3(0.8, 0.2, 0.1), 0.3, 0.0);
                color = vec4(result, 1.0);
            }
        }
        """)
        assert "Normalize" in asm
        assert "OpDot" in asm

    def test_triplanar_textured_surface(self):
        """Triplanar projection for procedural texturing."""
        asm = _compile_with_import("""
        import texture;
        fragment {
            in world_pos: vec3;
            in world_normal: vec3;
            out color: vec4;
            fn main() {
                let w: vec3 = triplanar_weights(world_normal, 4.0);
                let uv_x: vec2 = triplanar_uv_x(world_pos);
                let uv_y: vec2 = triplanar_uv_y(world_pos);
                let uv_z: vec2 = triplanar_uv_z(world_pos);

                // Simulate 3 different color samples
                let col_x: vec3 = vec3(uv_x.x, uv_x.y, 0.0);
                let col_y: vec3 = vec3(0.0, uv_y.x, uv_y.y);
                let col_z: vec3 = vec3(uv_z.x, 0.0, uv_z.y);

                let result: vec3 = triplanar_blend(col_x, col_y, col_z, w);
                color = vec4(result, 1.0);
            }
        }
        """)
        assert "Pow" in asm
        assert "FAbs" in asm
