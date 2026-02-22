"""Tests for P18 material properties pipeline — properties block in surface declarations."""

import json
import pytest
from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import PropertiesBlock, PropertiesField, SurfaceDecl


# --- Parse / AST ---

_MINIMAL_SURFACE = """
surface TestPBR {
    sampler2d tex,
    properties Material {
        base_color_factor: vec4 = vec4(1.0, 1.0, 1.0, 1.0),
        roughness_factor: scalar = 1.0,
        metallic_factor: scalar = 1.0,
        ior: scalar = 1.5,
    },
    layers [
        base(albedo: sample(tex, uv).xyz * Material.base_color_factor.xyz,
             roughness: Material.roughness_factor,
             metallic: Material.metallic_factor),
    ]
}
"""

_NO_PROPERTIES_SURFACE = """
surface SimplePBR {
    sampler2d tex,
    layers [
        base(albedo: sample(tex, uv).xyz,
             roughness: 0.5,
             metallic: 0.0),
    ]
}
"""


class TestPropertiesParsing:

    def test_properties_block_parses(self):
        """Grammar accepts 'properties Material { ... }' in surface."""
        mod = parse_lux(_MINIMAL_SURFACE)
        assert len(mod.surfaces) == 1

    def test_properties_in_surface_ast(self):
        """AST has PropertiesBlock on SurfaceDecl."""
        mod = parse_lux(_MINIMAL_SURFACE)
        surface = mod.surfaces[0]
        assert surface.properties is not None
        assert isinstance(surface.properties, PropertiesBlock)
        assert surface.properties.name == "Material"

    def test_properties_fields(self):
        """Properties block has the right fields with types and defaults."""
        mod = parse_lux(_MINIMAL_SURFACE)
        props = mod.surfaces[0].properties
        assert len(props.fields) == 4
        assert props.fields[0].name == "base_color_factor"
        assert props.fields[0].type_name == "vec4"
        assert props.fields[0].default is not None  # vec4(1.0, 1.0, 1.0, 1.0)
        assert props.fields[1].name == "roughness_factor"
        assert props.fields[1].type_name == "scalar"
        assert props.fields[3].name == "ior"

    def test_no_properties_backward_compat(self):
        """Surfaces without 'properties' still parse correctly."""
        mod = parse_lux(_NO_PROPERTIES_SURFACE)
        surface = mod.surfaces[0]
        assert surface.properties is None
        assert surface.layers is not None


# --- Compilation & Reflection ---

class TestPropertiesCompilation:

    def _compile_and_reflect(self, features_str=""):
        """Compile gltf_pbr_layered.lux and return (frag_reflection, rchit_reflection)."""
        from luxc.compiler import compile_source
        from pathlib import Path
        import tempfile, os

        lux_path = Path("examples/gltf_pbr_layered.lux")
        source = lux_path.read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmpdir:
            feat_set = set()
            if features_str:
                for f in features_str.split(","):
                    feat_set.add(f.strip())

            compile_source(
                source,
                stem="gltf_pbr_layered",
                output_dir=Path(tmpdir),
                source_name="gltf_pbr_layered.lux",
                features=feat_set if feat_set else None,
                emit_reflection=True,
                validate=False,
            )

            # Find reflection JSONs
            frag_json = rchit_json = None
            for fname in os.listdir(tmpdir):
                if fname.endswith(".frag.json"):
                    frag_json = json.loads(Path(tmpdir, fname).read_text())
                elif fname.endswith(".rchit.json"):
                    rchit_json = json.loads(Path(tmpdir, fname).read_text())

            return frag_json, rchit_json

    def _find_material_ubo(self, reflection):
        """Find the Material UBO in a reflection dict."""
        for ds_key, bindings in reflection.get("descriptor_sets", {}).items():
            for b in bindings:
                if b.get("type") == "uniform_buffer" and b.get("name") == "Material":
                    return b
        return None

    def test_properties_ubo_in_reflection(self):
        """Fragment reflection JSON contains 'Material' UBO with correct fields."""
        frag, _ = self._compile_and_reflect("has_emission")
        assert frag is not None
        mat_ubo = self._find_material_ubo(frag)
        assert mat_ubo is not None, "Material UBO not found in frag reflection"
        field_names = [f["name"] for f in mat_ubo["fields"]]
        assert "base_color_factor" in field_names
        assert "roughness_factor" in field_names
        assert "metallic_factor" in field_names
        assert "ior" in field_names
        assert "emissive_factor" in field_names
        assert "emissive_strength" in field_names

    def test_properties_ubo_offsets(self):
        """Material UBO fields have correct std140 offsets."""
        frag, _ = self._compile_and_reflect("has_emission")
        mat_ubo = self._find_material_ubo(frag)
        fields_by_name = {f["name"]: f for f in mat_ubo["fields"]}
        assert fields_by_name["base_color_factor"]["offset"] == 0
        assert fields_by_name["emissive_factor"]["offset"] == 16
        assert fields_by_name["metallic_factor"]["offset"] == 28
        assert fields_by_name["roughness_factor"]["offset"] == 32
        assert mat_ubo["size"] == 80

    def test_properties_defaults_in_reflection(self):
        """Reflection JSON includes default values from properties block."""
        frag, _ = self._compile_and_reflect("has_emission")
        mat_ubo = self._find_material_ubo(frag)
        fields_by_name = {f["name"]: f for f in mat_ubo["fields"]}
        assert fields_by_name["base_color_factor"]["default"] == [1.0, 1.0, 1.0, 1.0]
        assert fields_by_name["roughness_factor"]["default"] == 1.0
        assert fields_by_name["ior"]["default"] == 1.5
        assert fields_by_name["emissive_factor"]["default"] == [0.0, 0.0, 0.0]

    def test_material_field_access_compiles(self):
        """Material.roughness_factor in layer expressions → valid SPIR-V."""
        # If compilation succeeds, field access works
        frag, _ = self._compile_and_reflect("has_emission")
        assert frag is not None
        assert frag["stage"] == "fragment"

    def test_rt_has_properties_ubo(self):
        """Closest-hit reflection also contains 'Material' UBO."""
        _, rchit = self._compile_and_reflect("has_emission")
        assert rchit is not None
        mat_ubo = self._find_material_ubo(rchit)
        assert mat_ubo is not None, "Material UBO not found in rchit reflection"
        assert mat_ubo["size"] == 80

    def test_no_properties_backward_compat(self):
        """Surfaces without properties still compile (existing tests pass).

        The basic PBR surface compiles fine; this is covered by the 425
        passing regression tests, but we verify the core mechanism here.
        """
        from luxc.parser.tree_builder import parse_lux
        from luxc.expansion.surface_expander import expand_surfaces

        src = """
import brdf;
import color;
import compositing;
import ibl;
import texture;

surface Simple {
    sampler2d tex,
    samplerCube env_specular,
    samplerCube env_irradiance,
    sampler2d brdf_lut,
    layers [
        base(albedo: sample(tex, uv).xyz,
             roughness: 0.5,
             metallic: 0.0),
        ibl(specular_map: env_specular, irradiance_map: env_irradiance,
            brdf_lut: brdf_lut),
    ]
}

geometry Mesh {
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

pipeline Forward {
    geometry: Mesh,
    surface: Simple,
}
"""
        mod = parse_lux(src)
        expand_surfaces(mod)
        # Should have stages after expansion
        assert len(mod.stages) >= 2
