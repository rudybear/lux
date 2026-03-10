"""Tests for P21: Shared Stdlib Refactoring.

Validates:
- Non-bindless path now uses compose_pbr_layers (AST inspection)
- Coat IBL fix in bindless (new AST nodes present)
- Shared helpers (_detect_multi_light, _emit_multi_light_from_lighting)
- stdlib/pbr_pipeline.lux compiles correctly
"""

import pytest
from pathlib import Path
from luxc.parser.tree_builder import parse_lux
from luxc.expansion.surface_expander import expand_surfaces
from luxc.compiler import compile_source, _resolve_imports
from luxc.builtins.types import clear_type_aliases
from luxc.features.evaluator import strip_features

import subprocess

EXAMPLES = Path(__file__).parent.parent / "examples"


def _has_spirv_tools() -> bool:
    try:
        subprocess.run(["spirv-as", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


requires_spirv_tools = pytest.mark.skipif(
    not _has_spirv_tools(), reason="spirv-as/spirv-val not found on PATH"
)


# --- Shared source fragments ---

_LAYERED_SURFACE_WITH_IBL = """
import brdf;
import compositing;
import ibl;
import color;

geometry Quad {
    position: vec3,
    normal: vec3,
    uv: vec2,
    transform: MVP { model: mat4, view: mat4, projection: mat4 }
    outputs {
        world_pos: (model * vec4(position, 1.0)).xyz,
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        frag_uv: uv,
        clip_pos: projection * view * model * vec4(position, 1.0),
    }
}

surface PbrSurface {
    sampler2d albedo_tex,
    samplerCube env_specular,
    samplerCube env_irradiance,
    sampler2d brdf_lut,

    layers [
        base(albedo: sample(albedo_tex, uv).xyz, roughness: 0.5, metallic: 0.0),
        ibl(specular_map: env_specular, irradiance_map: env_irradiance, brdf_lut: brdf_lut),
    ]
}

pipeline TestForward {
    mode: forward,
    geometry: Quad,
    surface: PbrSurface,
}
"""

_LAYERED_SURFACE_WITH_ALL = """
import brdf;
import compositing;
import ibl;
import color;

geometry Quad {
    position: vec3,
    normal: vec3,
    uv: vec2,
    transform: MVP { model: mat4, view: mat4, projection: mat4 }
    outputs {
        world_pos: (model * vec4(position, 1.0)).xyz,
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        frag_uv: uv,
        clip_pos: projection * view * model * vec4(position, 1.0),
    }
}

surface AllLayers {
    sampler2d albedo_tex,
    samplerCube env_specular,
    samplerCube env_irradiance,
    sampler2d brdf_lut,

    layers [
        base(albedo: sample(albedo_tex, uv).xyz, roughness: 0.5, metallic: 0.0),
        transmission(factor: 0.3, ior: 1.5),
        ibl(specular_map: env_specular, irradiance_map: env_irradiance, brdf_lut: brdf_lut),
        sheen(color: vec3(0.1, 0.05, 0.0), roughness: 0.5),
        coat(factor: 0.8, roughness: 0.1),
        emission(color: vec3(0.5, 0.0, 0.0)),
    ]
}

pipeline AllForward {
    mode: forward,
    geometry: Quad,
    surface: AllLayers,
}
"""


def _parse_and_resolve(source):
    """Parse source and resolve stdlib imports."""
    mod = parse_lux(source)
    _resolve_imports(mod)
    return mod


class TestNonBindlessUsesCompose:
    """P21.1: Non-bindless path now calls compose_pbr_layers."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def _get_frag_vars(self, source, pipeline_name):
        mod = _parse_and_resolve(source)
        expand_surfaces(mod, pipeline_filter=pipeline_name)
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        main = frag.functions[0]
        return [stmt.name for stmt in main.body if hasattr(stmt, 'name')]

    def test_ibl_produces_compose_call(self):
        """IBL layer triggers compose_pbr_layers call."""
        var_names = self._get_frag_vars(
            _LAYERED_SURFACE_WITH_IBL, "TestForward")
        assert "composed" in var_names
        assert "bl_ambient" in var_names

    def test_all_layers_produce_compose_call(self):
        """All optional layers trigger compose_pbr_layers call."""
        var_names = self._get_frag_vars(
            _LAYERED_SURFACE_WITH_ALL, "AllForward")
        assert "composed" in var_names
        assert "bl_trans_factor" in var_names
        assert "bl_sheen_color" in var_names
        assert "bl_coat_factor" in var_names
        assert "bl_emission" in var_names
        assert "bl_ambient" in var_names

    def test_coat_ibl_in_non_bindless(self):
        """Coat + IBL produces coat IBL contribution."""
        var_names = self._get_frag_vars(
            _LAYERED_SURFACE_WITH_ALL, "AllForward")
        assert "bl_coat_ibl" in var_names
        assert "prefiltered_coat" in var_names

    def test_simple_shader_no_compose(self):
        """Shader without compositing layers does NOT call compose_pbr_layers."""
        src = """
        import brdf;
        geometry Quad {
            position: vec3,
            normal: vec3,
            uv: vec2,
            transform: MVP { model: mat4, view: mat4, projection: mat4 }
            outputs {
                world_pos: (model * vec4(position, 1.0)).xyz,
                world_normal: normalize((model * vec4(normal, 0.0)).xyz),
                frag_uv: uv,
                clip_pos: projection * view * model * vec4(position, 1.0),
            }
        }
        surface Simple {
            layers [
                base(albedo: vec3(0.5), roughness: 0.5, metallic: 0.0),
            ]
        }
        pipeline SimpleForward {
            mode: forward,
            geometry: Quad,
            surface: Simple,
        }
        """
        var_names = self._get_frag_vars(src, "SimpleForward")
        assert "composed" not in var_names
        assert "bl_ambient" not in var_names


@requires_spirv_tools
class TestCompilationRegression:
    """P21.5: Compilation regression tests."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_layered_ibl_compiles(self, tmp_path):
        """Non-bindless with IBL compiles to valid SPIR-V."""
        compile_source(_LAYERED_SURFACE_WITH_IBL, "ibl_test", tmp_path,
                       validate=True)
        assert (tmp_path / "ibl_test.frag.spv").exists()

    def test_layered_all_compiles(self, tmp_path):
        """Non-bindless with all layers compiles to valid SPIR-V."""
        compile_source(_LAYERED_SURFACE_WITH_ALL, "all_test", tmp_path,
                       validate=True)
        assert (tmp_path / "all_test.frag.spv").exists()

    def test_gltf_layered_all_features(self, tmp_path):
        """gltf_pbr_layered.lux with all features compiles."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        compile_source(
            src, "layered_all", tmp_path, validate=True,
            source_dir=EXAMPLES,
            pipeline="GltfForward",
            features={"has_normal_map", "has_emission", "has_clearcoat",
                       "has_sheen", "has_transmission"},
        )
        assert len(list(tmp_path.glob("*.spv"))) >= 1

    def test_gltf_layered_bindless(self, tmp_path):
        """gltf_pbr_layered.lux bindless with all features compiles."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        compile_source(
            src, "layered_bl", tmp_path, validate=True,
            source_dir=EXAMPLES,
            pipeline="GltfForward",
            features={"has_normal_map", "has_emission", "has_clearcoat",
                       "has_sheen", "has_transmission"},
            bindless=True,
        )
        assert len(list(tmp_path.glob("*.spv"))) >= 1

    def test_pbr_pipeline_stdlib_compiles(self, tmp_path):
        """stdlib/pbr_pipeline.lux pbr_shade() function compiles."""
        src = """
        import pbr_pipeline;
        import compositing;
        import ibl;
        import brdf;

        fragment {
            out color: vec4;
            uniform Light { light_dir: vec3, view_pos: vec3 }
            samplerCube env_specular;
            samplerCube env_irradiance;
            sampler2d brdf_lut;

            fn main() -> void {
                let n: vec3 = vec3(0.0, 1.0, 0.0);
                let v: vec3 = normalize(view_pos);
                let l: vec3 = normalize(light_dir);

                let prefiltered: vec3 = sample_lod(env_specular, n, 0.0).xyz;
                let irradiance: vec3 = sample_lod(env_irradiance, n, 0.0).xyz;
                let brdf_s: vec2 = sample(brdf_lut, vec2(0.5, 0.5)).xy;

                let result: vec3 = pbr_shade(
                    vec3(0.1),
                    n, v, l,
                    vec3(0.8), 0.5, 0.0,
                    prefiltered, irradiance, brdf_s,
                    0.0, 1.5, 0.0, vec3(1.0), 1000000.0,
                    vec3(0.0), 0.0,
                    0.0, 0.0, vec3(0.0),
                    vec3(0.0)
                );
                color = vec4(result, 1.0);
            }
        }
        """
        compile_source(src, "pbr_shade_test", tmp_path, validate=True)
        assert (tmp_path / "pbr_shade_test.frag.spv").exists()


class TestBindlessCoatIblFix:
    """P21.3: Coat IBL is now computed in bindless path."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_bindless_has_coat_ibl(self):
        """Bindless expansion includes bl_coat_ibl and prefiltered_coat."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        _resolve_imports(mod, EXAMPLES)
        # Strip features to enable clearcoat
        strip_features(mod, {"has_clearcoat"})
        expand_surfaces(mod, pipeline_filter="GltfForward", bindless=True)
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        main = frag.functions[0]
        # Flatten all statement names including those inside if-blocks
        all_names = set()
        def _collect(stmts):
            for s in stmts:
                if hasattr(s, 'name'):
                    all_names.add(s.name)
                if hasattr(s, 'then_body'):
                    _collect(s.then_body)
                if hasattr(s, 'else_body'):
                    _collect(s.else_body)
        _collect(main.body)
        assert "bl_coat_ibl" in all_names
        assert "prefiltered_coat" in all_names


class TestSharedHelpers:
    """P21.2: Shared helper functions work correctly."""

    def test_detect_multi_light_true(self):
        from luxc.expansion.surface_expander import _detect_multi_light
        from luxc.parser.ast_nodes import LightingDecl, LayerCall
        lighting = LightingDecl("TestLighting", [], [], [
            LayerCall("multi_light", []),
        ], None)
        assert _detect_multi_light(lighting) is True

    def test_detect_multi_light_false(self):
        from luxc.expansion.surface_expander import _detect_multi_light
        from luxc.parser.ast_nodes import LightingDecl, LayerCall
        lighting = LightingDecl("TestLighting", [], [], [
            LayerCall("directional", []),
        ], None)
        assert _detect_multi_light(lighting) is False

    def test_detect_multi_light_none(self):
        from luxc.expansion.surface_expander import _detect_multi_light
        assert _detect_multi_light(None) is False
