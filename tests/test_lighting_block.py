"""Tests for P17.1 lighting block — lighting declaration, IBL migration, directional layer."""

import json
import pytest
from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import (
    LightingDecl, PropertiesBlock, SurfaceSampler, LayerCall, VarRef,
)
from luxc.features.evaluator import strip_features
from luxc.expansion.surface_expander import expand_surfaces


# --- Test sources ---

_MINIMAL_LIGHTING = """
lighting TestLighting {
    samplerCube env_specular,
    samplerCube env_irradiance,
    sampler2d brdf_lut,

    properties Light {
        light_dir: vec3 = vec3(0.0, -1.0, 0.0),
        view_pos: vec3 = vec3(0.0, 0.0, 3.0),
    },

    layers [
        directional(direction: Light.light_dir,
                    color: vec3(1.0, 0.98, 0.95)),
        ibl(specular_map: env_specular, irradiance_map: env_irradiance,
            brdf_lut: brdf_lut),
    ]
}
"""

_LIGHTING_NO_LAYERS = """
lighting SimpleLighting {
    samplerCube env_specular,

    properties Light {
        light_dir: vec3 = vec3(0.0, -1.0, 0.0),
        view_pos: vec3 = vec3(0.0, 0.0, 3.0),
    },
}
"""

_SURFACE_WITH_LIGHTING_PIPELINE = """
import brdf;
import color;
import compositing;
import ibl;
import texture;

surface TestPBR {
    sampler2d base_color_tex,

    properties Material {
        base_color_factor: vec4 = vec4(1.0),
    },

    layers [
        base(albedo: sample(base_color_tex, uv).xyz,
             roughness: 0.5,
             metallic: 0.0),
    ]
}

lighting TestLighting {
    samplerCube env_specular,
    samplerCube env_irradiance,
    sampler2d brdf_lut,

    properties Light {
        light_dir: vec3 = vec3(0.0, -1.0, 0.0),
        view_pos: vec3 = vec3(0.0, 0.0, 3.0),
    },

    layers [
        directional(direction: Light.light_dir,
                    color: vec3(1.0, 0.98, 0.95)),
        ibl(specular_map: env_specular, irradiance_map: env_irradiance,
            brdf_lut: brdf_lut),
    ]
}

geometry SimpleGeo {
    position: vec3,
    normal: vec3,
    uv: vec2,
    transform: MVP {
        model: mat4,
        view: mat4,
        projection: mat4,
    }
    outputs {
        world_pos: (model * vec4(position, 1.0)).xyz,
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        frag_uv: uv,
        clip_pos: projection * view * model * vec4(position, 1.0),
    }
}

schedule HQ {
    tonemap: aces,
}

pipeline TestForward {
    geometry: SimpleGeo,
    surface: TestPBR,
    lighting: TestLighting,
    schedule: HQ,
}
"""

_SURFACE_NO_LIGHTING_PIPELINE = """
import brdf;
import color;
import compositing;
import ibl;
import texture;

surface TestPBR {
    sampler2d base_color_tex,
    samplerCube env_specular,
    samplerCube env_irradiance,
    sampler2d brdf_lut,

    properties Material {
        base_color_factor: vec4 = vec4(1.0),
    },

    layers [
        base(albedo: sample(base_color_tex, uv).xyz,
             roughness: 0.5,
             metallic: 0.0),
        ibl(specular_map: env_specular, irradiance_map: env_irradiance,
            brdf_lut: brdf_lut),
    ]
}

geometry SimpleGeo {
    position: vec3,
    normal: vec3,
    uv: vec2,
    transform: MVP {
        model: mat4,
        view: mat4,
        projection: mat4,
    }
    outputs {
        world_pos: (model * vec4(position, 1.0)).xyz,
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        frag_uv: uv,
        clip_pos: projection * view * model * vec4(position, 1.0),
    }
}

schedule HQ {
    tonemap: aces,
}

pipeline TestForward {
    geometry: SimpleGeo,
    surface: TestPBR,
    schedule: HQ,
}
"""

_LIGHTING_WITH_FEATURES = """
features {
    use_ibl: bool,
}

lighting ConditionalLighting {
    samplerCube env_specular if use_ibl,
    samplerCube env_irradiance if use_ibl,
    sampler2d brdf_lut if use_ibl,

    properties Light {
        light_dir: vec3 = vec3(0.0, -1.0, 0.0),
        view_pos: vec3 = vec3(0.0, 0.0, 3.0),
    },

    layers [
        directional(direction: Light.light_dir,
                    color: vec3(1.0)),
        ibl(specular_map: env_specular, irradiance_map: env_irradiance,
            brdf_lut: brdf_lut) if use_ibl,
    ]
}
"""


# --- Parsing tests ---

class TestLightingParsing:

    def test_parse_lighting_block(self):
        """Grammar accepts 'lighting Name { ... }' syntax."""
        mod = parse_lux(_MINIMAL_LIGHTING)
        assert len(mod.lightings) == 1
        assert isinstance(mod.lightings[0], LightingDecl)

    def test_lighting_name(self):
        """Lighting block has correct name."""
        mod = parse_lux(_MINIMAL_LIGHTING)
        assert mod.lightings[0].name == "TestLighting"

    def test_lighting_with_properties(self):
        """Lighting block properties generate correct PropertiesBlock."""
        mod = parse_lux(_MINIMAL_LIGHTING)
        lighting = mod.lightings[0]
        assert lighting.properties is not None
        assert isinstance(lighting.properties, PropertiesBlock)
        assert lighting.properties.name == "Light"
        assert len(lighting.properties.fields) == 2
        assert lighting.properties.fields[0].name == "light_dir"
        assert lighting.properties.fields[0].type_name == "vec3"
        assert lighting.properties.fields[1].name == "view_pos"

    def test_lighting_with_layers(self):
        """Lighting block layers parsed correctly."""
        mod = parse_lux(_MINIMAL_LIGHTING)
        lighting = mod.lightings[0]
        assert lighting.layers is not None
        assert len(lighting.layers) == 2
        assert lighting.layers[0].name == "directional"
        assert lighting.layers[1].name == "ibl"

    def test_lighting_with_samplers(self):
        """Lighting block samplers parsed correctly."""
        mod = parse_lux(_MINIMAL_LIGHTING)
        lighting = mod.lightings[0]
        assert len(lighting.samplers) == 3
        names = [s.name for s in lighting.samplers]
        assert "env_specular" in names
        assert "env_irradiance" in names
        assert "brdf_lut" in names

    def test_lighting_sampler_types(self):
        """Lighting samplers have correct types."""
        mod = parse_lux(_MINIMAL_LIGHTING)
        lighting = mod.lightings[0]
        by_name = {s.name: s for s in lighting.samplers}
        assert by_name["env_specular"].type_name == "samplerCube"
        assert by_name["env_irradiance"].type_name == "samplerCube"
        assert by_name["brdf_lut"].type_name == "sampler2d"

    def test_lighting_no_layers(self):
        """Lighting block without layers parses fine."""
        mod = parse_lux(_LIGHTING_NO_LAYERS)
        assert len(mod.lightings) == 1
        assert mod.lightings[0].layers is None

    def test_directional_layer_args(self):
        """Directional layer has direction and color args."""
        mod = parse_lux(_MINIMAL_LIGHTING)
        layers = mod.lightings[0].layers
        dir_layer = layers[0]
        arg_names = [a.name for a in dir_layer.args]
        assert "direction" in arg_names
        assert "color" in arg_names

    def test_ibl_layer_args(self):
        """IBL layer in lighting has specular_map, irradiance_map, brdf_lut args."""
        mod = parse_lux(_MINIMAL_LIGHTING)
        layers = mod.lightings[0].layers
        ibl_layer = layers[1]
        arg_names = [a.name for a in ibl_layer.args]
        assert "specular_map" in arg_names
        assert "irradiance_map" in arg_names
        assert "brdf_lut" in arg_names


# --- Feature stripping tests ---

class TestLightingFeatureStripping:

    def test_lighting_feature_stripping_samplers(self):
        """Conditional lighting samplers are stripped when feature is off."""
        mod = parse_lux(_LIGHTING_WITH_FEATURES)
        strip_features(mod, set())  # no features active
        lighting = mod.lightings[0]
        assert len(lighting.samplers) == 0

    def test_lighting_feature_stripping_samplers_active(self):
        """Conditional lighting samplers are kept when feature is on."""
        mod = parse_lux(_LIGHTING_WITH_FEATURES)
        strip_features(mod, {"use_ibl"})
        lighting = mod.lightings[0]
        assert len(lighting.samplers) == 3

    def test_lighting_feature_stripping_layers(self):
        """Conditional lighting layers are stripped when feature is off."""
        mod = parse_lux(_LIGHTING_WITH_FEATURES)
        strip_features(mod, set())
        lighting = mod.lightings[0]
        # Only directional should remain (no condition)
        assert len(lighting.layers) == 1
        assert lighting.layers[0].name == "directional"

    def test_lighting_feature_stripping_layers_active(self):
        """Conditional lighting layers are kept when feature is on."""
        mod = parse_lux(_LIGHTING_WITH_FEATURES)
        strip_features(mod, {"use_ibl"})
        lighting = mod.lightings[0]
        assert len(lighting.layers) == 2


# --- Pipeline wiring and expansion tests ---

class TestLightingExpansion:

    def test_pipeline_references_lighting(self):
        """Pipeline with lighting: SceneLighting parses correctly."""
        mod = parse_lux(_SURFACE_WITH_LIGHTING_PIPELINE)
        pipeline = mod.pipelines[0]
        members = {m.name: m for m in pipeline.members}
        assert "lighting" in members
        assert isinstance(members["lighting"].value, VarRef)
        assert members["lighting"].value.name == "TestLighting"

    def test_directional_layer_in_generated_code(self):
        """Directional layer produces light_color variable in generated fragment."""
        mod = parse_lux(_SURFACE_WITH_LIGHTING_PIPELINE)
        expand_surfaces(mod, pipeline_filter="TestForward")
        frag_stages = [s for s in mod.stages if s.stage_type == "fragment"]
        assert len(frag_stages) == 1
        frag = frag_stages[0]
        # Check that the main function body contains light_color
        main_fn = frag.functions[0]
        var_names = [stmt.name for stmt in main_fn.body
                     if hasattr(stmt, 'name')]
        assert "light_color" in var_names
        assert "l" in var_names

    def test_ibl_in_lighting_block(self):
        """IBL from lighting block generates reflection/irradiance sampling."""
        mod = parse_lux(_SURFACE_WITH_LIGHTING_PIPELINE)
        expand_surfaces(mod, pipeline_filter="TestForward")
        frag_stages = [s for s in mod.stages if s.stage_type == "fragment"]
        assert len(frag_stages) == 1
        frag = frag_stages[0]
        main_fn = frag.functions[0]
        var_names = [stmt.name for stmt in main_fn.body
                     if hasattr(stmt, 'name')]
        assert "prefiltered" in var_names
        assert "irradiance" in var_names
        assert "brdf_sample" in var_names
        assert "bl_ambient" in var_names

    def test_lighting_samplers_in_fragment(self):
        """Lighting block samplers appear in generated fragment stage."""
        mod = parse_lux(_SURFACE_WITH_LIGHTING_PIPELINE)
        expand_surfaces(mod, pipeline_filter="TestForward")
        frag_stages = [s for s in mod.stages if s.stage_type == "fragment"]
        frag = frag_stages[0]
        sampler_names = [s.name for s in frag.samplers]
        assert "env_specular" in sampler_names
        assert "env_irradiance" in sampler_names
        assert "brdf_lut" in sampler_names

    def test_lighting_properties_generate_ubo(self):
        """Lighting properties block generates Light UBO in fragment."""
        mod = parse_lux(_SURFACE_WITH_LIGHTING_PIPELINE)
        expand_surfaces(mod, pipeline_filter="TestForward")
        frag_stages = [s for s in mod.stages if s.stage_type == "fragment"]
        frag = frag_stages[0]
        uniform_names = [u.name for u in frag.uniforms]
        assert "Light" in uniform_names
        light_ubo = next(u for u in frag.uniforms if u.name == "Light")
        field_names = [f.name for f in light_ubo.fields]
        assert "light_dir" in field_names
        assert "view_pos" in field_names

    def test_backward_compat_no_lighting(self):
        """Pipeline without lighting block uses legacy hardcoded Light UBO."""
        mod = parse_lux(_SURFACE_NO_LIGHTING_PIPELINE)
        expand_surfaces(mod, pipeline_filter="TestForward")
        frag_stages = [s for s in mod.stages if s.stage_type == "fragment"]
        frag = frag_stages[0]
        uniform_names = [u.name for u in frag.uniforms]
        assert "Light" in uniform_names
        light_ubo = next(u for u in frag.uniforms if u.name == "Light")
        field_names = [f.name for f in light_ubo.fields]
        assert "light_dir" in field_names
        assert "view_pos" in field_names

    def test_backward_compat_ibl_in_surface(self):
        """IBL layer in surface still works when there's no lighting block."""
        mod = parse_lux(_SURFACE_NO_LIGHTING_PIPELINE)
        expand_surfaces(mod, pipeline_filter="TestForward")
        frag_stages = [s for s in mod.stages if s.stage_type == "fragment"]
        frag = frag_stages[0]
        main_fn = frag.functions[0]
        var_names = [stmt.name for stmt in main_fn.body
                     if hasattr(stmt, 'name')]
        # IBL should still be generated from surface layers
        assert "prefiltered" in var_names
        assert "irradiance" in var_names
        assert "bl_ambient" in var_names


# --- Cross-block interaction tests ---

class TestCrossBlockInteraction:

    _COAT_IBL_SOURCE = """
import brdf;
import color;
import compositing;
import ibl;
import texture;

surface CoatedPBR {
    sampler2d base_color_tex,
    sampler2d clearcoat_tex,

    layers [
        base(albedo: sample(base_color_tex, uv).xyz,
             roughness: 0.5,
             metallic: 0.0),
        coat(factor: sample(clearcoat_tex, uv).x,
             roughness: 0.1),
    ]
}

lighting SceneLighting {
    samplerCube env_specular,
    samplerCube env_irradiance,
    sampler2d brdf_lut,

    properties Light {
        light_dir: vec3 = vec3(0.0, -1.0, 0.0),
        view_pos: vec3 = vec3(0.0, 0.0, 3.0),
    },

    layers [
        directional(direction: Light.light_dir,
                    color: vec3(1.0)),
        ibl(specular_map: env_specular, irradiance_map: env_irradiance,
            brdf_lut: brdf_lut),
    ]
}

geometry SimpleGeo {
    position: vec3,
    normal: vec3,
    uv: vec2,
    transform: MVP {
        model: mat4,
        view: mat4,
        projection: mat4,
    }
    outputs {
        world_pos: (model * vec4(position, 1.0)).xyz,
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        frag_uv: uv,
        clip_pos: projection * view * model * vec4(position, 1.0),
    }
}

schedule HQ {
    tonemap: aces,
}

pipeline CoatedForward {
    geometry: SimpleGeo,
    surface: CoatedPBR,
    lighting: SceneLighting,
    schedule: HQ,
}
"""

    def test_coat_ibl_cross_block(self):
        """Coat in surface + IBL in lighting generates coat_ibl contribution."""
        mod = parse_lux(self._COAT_IBL_SOURCE)
        expand_surfaces(mod, pipeline_filter="CoatedForward")
        frag_stages = [s for s in mod.stages if s.stage_type == "fragment"]
        assert len(frag_stages) == 1
        frag = frag_stages[0]
        main_fn = frag.functions[0]
        var_names = [stmt.name for stmt in main_fn.body
                     if hasattr(stmt, 'name')]
        # Should have coat_ibl contribution from cross-block interaction
        # (coat params loaded into bl_ vars, coat IBL computed, passed to compose)
        assert "bl_coat_factor" in var_names
        assert "bl_coat_roughness" in var_names
        assert "prefiltered_coat" in var_names
        assert "bl_coat_ibl" in var_names
        assert "composed" in var_names
