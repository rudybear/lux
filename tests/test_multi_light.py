"""Tests for multi-light system (Phase G1).

Covers:
  G1.1 - Parsing lighting block with multi_light layer
  G1.2 - Multi-light unrolling generates correct AST
  G1.3 - LightData SSBO in generated stage
  G1.4 - sampler2DArray / samplerCubeArray parse and validate
  G1.5 - Backward compatibility: directional() still works
  G1.6 - All 6 pipeline paths compile with multi_light
  G1.7 - Shadow args: shadow SSBO and sampler2DArray with has_shadows
"""

import subprocess
import pytest
from pathlib import Path

from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import (
    LightingDecl, PropertiesBlock, SurfaceSampler, LayerCall,
    IfStmt, LetStmt, StorageBufferDecl, SamplerDecl,
    CallExpr, VarRef,
)
from luxc.features.evaluator import strip_features
from luxc.expansion.surface_expander import expand_surfaces
from luxc.compiler import compile_source, _resolve_imports
from luxc.builtins.types import clear_type_aliases


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


# ---------------------------------------------------------------------------
# Shared sources
# ---------------------------------------------------------------------------

_DIRECTIONAL_PIPELINE = """\
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


# ---------------------------------------------------------------------------
# G1.1 — Parse lighting block with multi_light layer
# ---------------------------------------------------------------------------

class TestMultiLightParsing:
    """G1.1: Parse lighting block with multi_light layer."""

    def test_gltf_pbr_layered_has_lighting_block(self):
        """gltf_pbr_layered.lux parses a lighting block named SceneLighting."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        assert len(mod.lightings) == 1
        assert isinstance(mod.lightings[0], LightingDecl)
        assert mod.lightings[0].name == "SceneLighting"

    def test_multi_light_layer_present_with_shadows(self):
        """With has_shadows active, multi_light layer has count, shadow_map, max_lights args."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        strip_features(mod, {"has_shadows"})
        lighting = mod.lightings[0]
        layer_names = [l.name for l in lighting.layers]
        assert "multi_light" in layer_names
        ml_layer = next(l for l in lighting.layers if l.name == "multi_light")
        arg_names = [a.name for a in ml_layer.args]
        assert "count" in arg_names
        assert "shadow_map" in arg_names
        assert "max_lights" in arg_names

    def test_multi_light_layer_present_without_shadows(self):
        """Without has_shadows, multi_light layer has count and max_lights but no shadow_map."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        strip_features(mod, set())
        lighting = mod.lightings[0]
        layer_names = [l.name for l in lighting.layers]
        assert "multi_light" in layer_names
        ml_layer = next(l for l in lighting.layers if l.name == "multi_light")
        arg_names = [a.name for a in ml_layer.args]
        assert "count" in arg_names
        assert "max_lights" in arg_names
        assert "shadow_map" not in arg_names

    def test_scene_light_properties(self):
        """SceneLight properties have view_pos (vec3) and light_count (int)."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        lighting = mod.lightings[0]
        props = lighting.properties
        assert isinstance(props, PropertiesBlock)
        assert props.name == "SceneLight"
        field_map = {f.name: f.type_name for f in props.fields}
        assert field_map["view_pos"] == "vec3"
        assert field_map["light_count"] == "int"

    def test_shadow_maps_sampler2DArray_with_shadows(self):
        """With has_shadows, shadow_maps sampler2DArray is present in lighting samplers."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        strip_features(mod, {"has_shadows"})
        lighting = mod.lightings[0]
        sampler_map = {s.name: s.type_name for s in lighting.samplers}
        assert "shadow_maps" in sampler_map
        assert sampler_map["shadow_maps"] == "sampler2DArray"

    def test_shadow_maps_stripped_without_shadows(self):
        """Without has_shadows, shadow_maps sampler is stripped from lighting."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        strip_features(mod, set())
        lighting = mod.lightings[0]
        sampler_names = [s.name for s in lighting.samplers]
        assert "shadow_maps" not in sampler_names

    def test_multi_light_demo_parses(self):
        """multi_light_demo.lux parses with multi_light layer in MultiLighting block."""
        src = (EXAMPLES / "multi_light_demo.lux").read_text()
        mod = parse_lux(src)
        assert len(mod.lightings) == 1
        lighting = mod.lightings[0]
        assert lighting.name == "MultiLighting"
        layer_names = [l.name for l in lighting.layers]
        assert "multi_light" in layer_names
        assert "ibl" in layer_names


# ---------------------------------------------------------------------------
# G1.2 — Multi-light unrolling generates correct AST
# ---------------------------------------------------------------------------

class TestMultiLightUnrolling:
    """G1.2: Multi-light unrolling generates correct AST."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def _expand_gltf_forward(self, features=None):
        """Expand gltf_pbr_layered.lux for GltfForward and return fragment stage."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        strip_features(mod, features or set())
        _resolve_imports(mod, EXAMPLES)
        expand_surfaces(mod, pipeline_filter="GltfForward")
        frag_stages = [s for s in mod.stages if s.stage_type == "fragment"]
        assert len(frag_stages) == 1
        return frag_stages[0]

    def test_16_guarded_iterations(self):
        """Multi-light loop emits 16 IfStmt guards (max_lights: 16)."""
        frag = self._expand_gltf_forward()
        main_fn = frag.functions[0]
        # Count IfStmt nodes whose then_body contains light field loads (lt_N)
        ml_ifs = 0
        for stmt in main_fn.body:
            if isinstance(stmt, IfStmt):
                inner_names = [s.name for s in stmt.then_body
                               if isinstance(s, LetStmt)]
                if any(n.startswith("lt_") for n in inner_names):
                    ml_ifs += 1
        assert ml_ifs == 16

    def test_light_count_guard_variable(self):
        """_ml_light_count variable is emitted before the unrolled loop."""
        frag = self._expand_gltf_forward()
        main_fn = frag.functions[0]
        let_names = [s.name for s in main_fn.body if isinstance(s, LetStmt)]
        assert "_ml_light_count" in let_names

    def test_total_direct_accumulator(self):
        """total_direct accumulator is initialized in main body."""
        frag = self._expand_gltf_forward()
        main_fn = frag.functions[0]
        let_names = [s.name for s in main_fn.body if isinstance(s, LetStmt)]
        assert "total_direct" in let_names

    def test_ssbo_field_loads_per_iteration(self):
        """Each iteration loads light_type, position, direction, color, intensity, etc."""
        frag = self._expand_gltf_forward()
        main_fn = frag.functions[0]
        # Inspect first iteration (suffix _0)
        first_if = None
        for stmt in main_fn.body:
            if isinstance(stmt, IfStmt):
                inner_names = [s.name for s in stmt.then_body
                               if isinstance(s, LetStmt)]
                if "lt_0" in inner_names:
                    first_if = stmt
                    break
        assert first_if is not None, "First multi-light iteration not found"
        inner_names = [s.name for s in first_if.then_body
                       if isinstance(s, LetStmt)]
        # Light field loads
        assert "lt_0" in inner_names     # light_type
        assert "li_0" in inner_names     # intensity
        assert "lr_0" in inner_names     # range
        assert "lic_0" in inner_names    # inner_cone
        assert "lp_0" in inner_names     # position
        assert "loc_0" in inner_names    # outer_cone
        assert "ld_0" in inner_names     # direction
        assert "lsi_0" in inner_names    # shadow_index
        assert "lc_0" in inner_names     # color

    def test_evaluate_light_direction_call(self):
        """Each iteration calls evaluate_light_direction -> l_N."""
        frag = self._expand_gltf_forward()
        main_fn = frag.functions[0]
        first_if = None
        for stmt in main_fn.body:
            if isinstance(stmt, IfStmt):
                inner_names = [s.name for s in stmt.then_body
                               if isinstance(s, LetStmt)]
                if "l_0" in inner_names:
                    first_if = stmt
                    break
        assert first_if is not None
        inner_names = [s.name for s in first_if.then_body
                       if isinstance(s, LetStmt)]
        assert "l_0" in inner_names

    def test_gltf_pbr_call(self):
        """Each iteration calls gltf_pbr -> brdf_N."""
        frag = self._expand_gltf_forward()
        main_fn = frag.functions[0]
        first_if = None
        for stmt in main_fn.body:
            if isinstance(stmt, IfStmt):
                inner_names = [s.name for s in stmt.then_body
                               if isinstance(s, LetStmt)]
                if "brdf_0" in inner_names:
                    first_if = stmt
                    break
        assert first_if is not None
        inner_names = [s.name for s in first_if.then_body
                       if isinstance(s, LetStmt)]
        assert "brdf_0" in inner_names

    def test_evaluate_light_call(self):
        """Each iteration calls evaluate_light -> radiance_N."""
        frag = self._expand_gltf_forward()
        main_fn = frag.functions[0]
        first_if = None
        for stmt in main_fn.body:
            if isinstance(stmt, IfStmt):
                inner_names = [s.name for s in stmt.then_body
                               if isinstance(s, LetStmt)]
                if "radiance_0" in inner_names:
                    first_if = stmt
                    break
        assert first_if is not None
        inner_names = [s.name for s in first_if.then_body
                       if isinstance(s, LetStmt)]
        assert "radiance_0" in inner_names

    def test_last_iteration_has_suffix_15(self):
        """The 16th (last) iteration uses suffix _15."""
        frag = self._expand_gltf_forward()
        main_fn = frag.functions[0]
        found = False
        for stmt in main_fn.body:
            if isinstance(stmt, IfStmt):
                inner_names = [s.name for s in stmt.then_body
                               if isinstance(s, LetStmt)]
                if "lt_15" in inner_names:
                    found = True
                    assert "brdf_15" in inner_names
                    assert "radiance_15" in inner_names
                    break
        assert found, "Iteration 15 (the last) not found"


# ---------------------------------------------------------------------------
# G1.3 — LightData SSBO in generated stage
# ---------------------------------------------------------------------------

class TestLightDataSSBO:
    """G1.3: LightData SSBO in generated stage."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_lights_ssbo_in_raster_fragment(self):
        """GltfForward fragment stage has StorageBufferDecl('lights', 'LightData')."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        strip_features(mod, set())
        _resolve_imports(mod, EXAMPLES)
        expand_surfaces(mod, pipeline_filter="GltfForward")
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        ssbo_map = {sb.name: sb.element_type for sb in frag.storage_buffers}
        assert "lights" in ssbo_map
        assert ssbo_map["lights"] == "LightData"

    def test_lights_ssbo_in_multi_light_demo(self):
        """multi_light_demo.lux fragment has LightData SSBO."""
        src = (EXAMPLES / "multi_light_demo.lux").read_text()
        mod = parse_lux(src)
        strip_features(mod, set())
        _resolve_imports(mod, EXAMPLES)
        expand_surfaces(mod, pipeline_filter="MultiLightForward")
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        ssbo_map = {sb.name: sb.element_type for sb in frag.storage_buffers}
        assert "lights" in ssbo_map
        assert ssbo_map["lights"] == "LightData"

    def test_no_lights_ssbo_with_directional(self):
        """directional() lighting does NOT add a lights SSBO."""
        mod = parse_lux(_DIRECTIONAL_PIPELINE)
        _resolve_imports(mod, None)
        expand_surfaces(mod, pipeline_filter="TestForward")
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        ssbo_names = [sb.name for sb in frag.storage_buffers]
        assert "lights" not in ssbo_names


# ---------------------------------------------------------------------------
# G1.4 — sampler2DArray / samplerCubeArray parse and validate
# ---------------------------------------------------------------------------

class TestArraySamplerTypes:
    """G1.4: sampler2DArray and samplerCubeArray type support."""

    def test_sampler2DArray_parses_in_lighting(self):
        """sampler2DArray is accepted in a lighting block sampler declaration."""
        src = """
        lighting TestLighting {
            sampler2DArray shadow_maps,
            sampler2d brdf_lut,

            properties Light {
                view_pos: vec3 = vec3(0.0),
            },
        }
        """
        mod = parse_lux(src)
        lighting = mod.lightings[0]
        sampler_map = {s.name: s.type_name for s in lighting.samplers}
        assert sampler_map["shadow_maps"] == "sampler2DArray"

    def test_samplerCubeArray_parses_in_lighting(self):
        """samplerCubeArray is accepted in a lighting block sampler declaration."""
        src = """
        lighting TestLighting {
            samplerCubeArray env_array,
            sampler2d brdf_lut,

            properties Light {
                view_pos: vec3 = vec3(0.0),
            },
        }
        """
        mod = parse_lux(src)
        lighting = mod.lightings[0]
        sampler_map = {s.name: s.type_name for s in lighting.samplers}
        assert sampler_map["env_array"] == "samplerCubeArray"

    def test_sampler2DArray_parses_in_surface(self):
        """sampler2DArray is accepted in a surface sampler declaration."""
        src = """
        surface TestSurface {
            sampler2DArray tex_array,

            layers [
                base(albedo: vec3(1.0), roughness: 0.5, metallic: 0.0),
            ]
        }
        """
        mod = parse_lux(src)
        surface = mod.surfaces[0]
        sampler_map = {s.name: s.type_name for s in surface.samplers}
        assert sampler_map["tex_array"] == "sampler2DArray"


@requires_spirv_tools
class TestArraySamplerSPIRV:
    """G1.4: sampler2DArray generates valid SPIR-V."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_sampler2DArray_in_compiled_shader(self, tmp_path):
        """Shader with sampler2DArray shadow_maps produces valid SPIR-V."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        compile_source(
            src, "sampler2DArray_test", tmp_path, validate=True,
            source_dir=EXAMPLES, pipeline="GltfForward",
            features={"has_shadows"},
        )
        spv_files = list(tmp_path.glob("*.frag.spv"))
        assert len(spv_files) >= 1, "No fragment .spv file produced"


# ---------------------------------------------------------------------------
# G1.5 — Backward compatibility: directional() still works
# ---------------------------------------------------------------------------

@requires_spirv_tools
class TestDirectionalBackwardCompat:
    """G1.5: Backward compatibility -- directional() layer still works."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_directional_compiles_valid_spirv(self, tmp_path):
        """Pipeline with directional() lighting compiles and validates SPIR-V."""
        compile_source(
            _DIRECTIONAL_PIPELINE, "compat_directional", tmp_path,
            validate=True, pipeline="TestForward",
        )
        assert (tmp_path / "compat_directional.vert.spv").exists()
        assert (tmp_path / "compat_directional.frag.spv").exists()

    def test_directional_generates_light_color(self):
        """directional() layer still produces light_color and l variables."""
        clear_type_aliases()
        mod = parse_lux(_DIRECTIONAL_PIPELINE)
        _resolve_imports(mod, None)
        expand_surfaces(mod, pipeline_filter="TestForward")
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        main_fn = frag.functions[0]
        let_names = [s.name for s in main_fn.body if isinstance(s, LetStmt)]
        assert "light_color" in let_names
        assert "l" in let_names

    def test_directional_does_not_emit_multi_light_loop(self):
        """directional() does NOT emit total_direct or _ml_light_count."""
        clear_type_aliases()
        mod = parse_lux(_DIRECTIONAL_PIPELINE)
        _resolve_imports(mod, None)
        expand_surfaces(mod, pipeline_filter="TestForward")
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        main_fn = frag.functions[0]
        let_names = [s.name for s in main_fn.body if isinstance(s, LetStmt)]
        assert "total_direct" not in let_names
        assert "_ml_light_count" not in let_names


# ---------------------------------------------------------------------------
# G1.6 — All 6 pipeline paths compile with multi_light
# ---------------------------------------------------------------------------

@requires_spirv_tools
class TestMultiLightAllPipelines:
    """G1.6: All 6 pipeline paths (3 modes x 2 bindless) compile with multi_light."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def _compile_layered(self, tmp_path, pipeline, bindless=False):
        """Compile gltf_pbr_layered.lux for a given pipeline and bindless mode."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        stem = f"ml_{pipeline}"
        if bindless:
            stem += "_bindless"
        compile_source(
            src, stem, tmp_path, validate=True,
            source_dir=EXAMPLES, pipeline=pipeline,
            bindless=bindless,
        )
        spv_files = list(tmp_path.glob("*.spv"))
        assert len(spv_files) >= 1, f"No .spv files for {stem}"
        return spv_files

    # --- GltfForward (raster) ---

    def test_forward_non_bindless(self, tmp_path):
        """GltfForward non-bindless with multi_light produces valid SPIR-V."""
        spvs = self._compile_layered(tmp_path, "GltfForward")
        assert any("vert" in str(f) for f in spvs)
        assert any("frag" in str(f) for f in spvs)

    def test_forward_bindless(self, tmp_path):
        """GltfForward bindless with multi_light produces valid SPIR-V."""
        spvs = self._compile_layered(tmp_path, "GltfForward", bindless=True)
        assert any("vert" in str(f) for f in spvs)
        assert any("frag" in str(f) for f in spvs)

    # --- GltfRT (raytracing) ---

    def test_rt_non_bindless(self, tmp_path):
        """GltfRT non-bindless with multi_light produces valid SPIR-V."""
        spvs = self._compile_layered(tmp_path, "GltfRT")
        assert any("rgen" in str(f) for f in spvs)
        assert any("rchit" in str(f) for f in spvs)
        assert any("rmiss" in str(f) for f in spvs)

    def test_rt_bindless(self, tmp_path):
        """GltfRT bindless with multi_light produces valid SPIR-V."""
        spvs = self._compile_layered(tmp_path, "GltfRT", bindless=True)
        assert any("rgen" in str(f) for f in spvs)
        assert any("rchit" in str(f) for f in spvs)
        assert any("rmiss" in str(f) for f in spvs)

    # --- GltfMesh (mesh shader) ---

    def test_mesh_non_bindless(self, tmp_path):
        """GltfMesh non-bindless with multi_light produces valid SPIR-V."""
        spvs = self._compile_layered(tmp_path, "GltfMesh")
        assert any("mesh" in str(f) for f in spvs)
        assert any("frag" in str(f) for f in spvs)

    def test_mesh_bindless(self, tmp_path):
        """GltfMesh bindless with multi_light produces valid SPIR-V."""
        spvs = self._compile_layered(tmp_path, "GltfMesh", bindless=True)
        assert any("mesh" in str(f) for f in spvs)
        assert any("frag" in str(f) for f in spvs)


# ---------------------------------------------------------------------------
# G1.6 (extended) — multi_light_demo.lux compilation
# ---------------------------------------------------------------------------

@requires_spirv_tools
class TestMultiLightDemo:
    """Compile multi_light_demo.lux to verify the standalone multi-light example."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_multi_light_demo_compiles(self, tmp_path):
        """multi_light_demo.lux compiles with valid SPIR-V."""
        src = (EXAMPLES / "multi_light_demo.lux").read_text()
        compile_source(
            src, "multi_light_demo", tmp_path, validate=True,
            source_dir=EXAMPLES, pipeline="MultiLightForward",
        )
        assert (tmp_path / "multi_light_demo.vert.spv").exists()
        assert (tmp_path / "multi_light_demo.frag.spv").exists()


# ---------------------------------------------------------------------------
# G1.7 — Shadow args: shadow SSBO and sampler2DArray with has_shadows
# ---------------------------------------------------------------------------

class TestShadowArgs:
    """G1.7: Shadow-related resources when has_shadows is enabled."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_shadow_maps_sampler_in_fragment(self):
        """With has_shadows, shadow_maps sampler2DArray appears in the fragment stage."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        strip_features(mod, {"has_shadows"})
        _resolve_imports(mod, EXAMPLES)
        expand_surfaces(mod, pipeline_filter="GltfForward")
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        sampler_map = {s.name: getattr(s, "type_name", "sampler2d")
                       for s in frag.samplers}
        assert "shadow_maps" in sampler_map
        assert sampler_map["shadow_maps"] == "sampler2DArray"

    def test_shadow_maps_absent_without_shadows(self):
        """Without has_shadows, shadow_maps is NOT in fragment samplers."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        strip_features(mod, set())
        _resolve_imports(mod, EXAMPLES)
        expand_surfaces(mod, pipeline_filter="GltfForward")
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        sampler_names = [s.name for s in frag.samplers]
        assert "shadow_maps" not in sampler_names

    def test_lights_ssbo_present_with_shadows(self):
        """With has_shadows, lights SSBO is still present (multi_light layer active)."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        strip_features(mod, {"has_shadows"})
        _resolve_imports(mod, EXAMPLES)
        expand_surfaces(mod, pipeline_filter="GltfForward")
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        ssbo_map = {sb.name: sb.element_type for sb in frag.storage_buffers}
        assert "lights" in ssbo_map
        assert ssbo_map["lights"] == "LightData"

    def test_multi_light_with_shadow_map_arg(self):
        """With has_shadows, multi_light layer has shadow_map argument wired to shadow_maps."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        strip_features(mod, {"has_shadows"})
        lighting = mod.lightings[0]
        ml_layer = next(l for l in lighting.layers if l.name == "multi_light")
        arg_map = {a.name: a for a in ml_layer.args}
        assert "shadow_map" in arg_map
        # The shadow_map arg should reference the shadow_maps sampler
        shadow_arg = arg_map["shadow_map"]
        assert shadow_arg.value.name == "shadow_maps"


@requires_spirv_tools
class TestShadowsSPIRV:
    """G1.7: has_shadows compilation produces valid SPIR-V."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_forward_with_shadows_compiles(self, tmp_path):
        """GltfForward with has_shadows feature produces valid SPIR-V."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        compile_source(
            src, "forward_shadows", tmp_path, validate=True,
            source_dir=EXAMPLES, pipeline="GltfForward",
            features={"has_shadows"},
        )
        spv_files = list(tmp_path.glob("*.spv"))
        assert len(spv_files) >= 1

    def test_forward_bindless_with_shadows_compiles(self, tmp_path):
        """GltfForward bindless with has_shadows produces valid SPIR-V."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        compile_source(
            src, "forward_bl_shadows", tmp_path, validate=True,
            source_dir=EXAMPLES, pipeline="GltfForward",
            features={"has_shadows"}, bindless=True,
        )
        spv_files = list(tmp_path.glob("*.spv"))
        assert len(spv_files) >= 1

    def test_rt_with_shadows_compiles(self, tmp_path):
        """GltfRT with has_shadows produces valid SPIR-V."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        compile_source(
            src, "rt_shadows", tmp_path, validate=True,
            source_dir=EXAMPLES, pipeline="GltfRT",
            features={"has_shadows"},
        )
        spv_files = list(tmp_path.glob("*.spv"))
        assert len(spv_files) >= 1

    def test_mesh_with_shadows_compiles(self, tmp_path):
        """GltfMesh with has_shadows produces valid SPIR-V."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        compile_source(
            src, "mesh_shadows", tmp_path, validate=True,
            source_dir=EXAMPLES, pipeline="GltfMesh",
            features={"has_shadows"},
        )
        spv_files = list(tmp_path.glob("*.spv"))
        assert len(spv_files) >= 1


# ---------------------------------------------------------------------------
# P17.3 — Shadow sampling wiring tests
# ---------------------------------------------------------------------------

class TestShadowSamplingWiring:
    """P17.3: Shadow sampling wired into multi-light loop."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def _expand_with_shadows(self, pipeline="GltfForward"):
        """Expand gltf_pbr_layered.lux with has_shadows and return fragment stage."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        strip_features(mod, {"has_shadows"})
        _resolve_imports(mod, EXAMPLES)
        expand_surfaces(mod, pipeline_filter=pipeline)
        frag_stages = [s for s in mod.stages if s.stage_type == "fragment"]
        if not frag_stages:
            # For RT pipelines, check closest_hit
            frag_stages = [s for s in mod.stages if s.stage_type == "closest_hit"]
        assert len(frag_stages) >= 1
        return frag_stages[0]

    def test_shadow_matrices_ssbo_emitted(self):
        """ShadowEntry SSBO present in fragment stage when shadow_map arg present."""
        frag = self._expand_with_shadows()
        ssbo_map = {sb.name: sb.element_type for sb in frag.storage_buffers}
        assert "shadow_matrices" in ssbo_map
        assert ssbo_map["shadow_matrices"] == "ShadowEntry"

    def test_shadow_factor_in_accumulation(self):
        """shadow_0 variable is emitted in first multi-light iteration."""
        frag = self._expand_with_shadows()
        main_fn = frag.functions[0]
        # Find the first multi-light iteration (contains lt_0)
        for stmt in main_fn.body:
            if isinstance(stmt, IfStmt):
                inner_names = [s.name for s in stmt.then_body
                               if isinstance(s, LetStmt)]
                if "lt_0" in inner_names:
                    assert "shadow_0" in inner_names
                    return
        pytest.fail("shadow_0 not found in first multi-light iteration")

    def test_shadow_guard_on_shadow_index(self):
        """Shadow evaluation is guarded by shadow_index >= 0.0."""
        frag = self._expand_with_shadows()
        main_fn = frag.functions[0]
        # Find the first multi-light iteration, then look for nested IfStmt
        for stmt in main_fn.body:
            if isinstance(stmt, IfStmt):
                inner_names = [s.name for s in stmt.then_body
                               if isinstance(s, LetStmt)]
                if "lt_0" in inner_names:
                    # Look for a nested IfStmt that guards shadow evaluation
                    nested_ifs = [s for s in stmt.then_body if isinstance(s, IfStmt)]
                    assert len(nested_ifs) >= 1, "No nested IfStmt guard for shadow eval"
                    # The guard should compare lsi_0 >= 0.0
                    guard = nested_ifs[0]
                    from luxc.parser.ast_nodes import BinaryOp
                    assert isinstance(guard.condition, BinaryOp)
                    assert guard.condition.op == ">="
                    return
        pytest.fail("Shadow guard IfStmt not found")

    def test_no_shadow_without_shadow_map_arg(self):
        """No shadow code when shadow_map arg is absent."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        strip_features(mod, set())  # no has_shadows
        _resolve_imports(mod, EXAMPLES)
        expand_surfaces(mod, pipeline_filter="GltfForward")
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        main_fn = frag.functions[0]
        # Check no shadow_0 variable exists
        for stmt in main_fn.body:
            if isinstance(stmt, IfStmt):
                inner_names = [s.name for s in stmt.then_body
                               if isinstance(s, LetStmt)]
                if "shadow_0" in inner_names:
                    pytest.fail("shadow_0 found when has_shadows is not enabled")
        # Also verify no shadow_matrices SSBO
        ssbo_names = [sb.name for sb in frag.storage_buffers]
        assert "shadow_matrices" not in ssbo_names


# ---------------------------------------------------------------------------
# P17.3 — Builtin type checks
# ---------------------------------------------------------------------------

class TestShadowBuiltins:
    """P17.3: sample_compare and sample_array builtin type checks."""

    def test_sample_compare_type_checks(self):
        """sample_compare(sampler2DArray, vec3, scalar) -> scalar."""
        from luxc.builtins.functions import lookup_builtin
        from luxc.builtins.types import SAMPLER_2D_ARRAY, VEC3, SCALAR
        sig = lookup_builtin("sample_compare", [SAMPLER_2D_ARRAY, VEC3, SCALAR])
        assert sig is not None
        assert sig.return_type == SCALAR

    def test_sample_array_type_checks(self):
        """sample_array(sampler2DArray, vec2, scalar) -> vec4."""
        from luxc.builtins.functions import lookup_builtin
        from luxc.builtins.types import SAMPLER_2D_ARRAY, VEC2, SCALAR, VEC4
        sig = lookup_builtin("sample_array", [SAMPLER_2D_ARRAY, VEC2, SCALAR])
        assert sig is not None
        assert sig.return_type == VEC4

    def test_sample_compare_no_match_wrong_sampler(self):
        """sample_compare rejects sampler2d (requires sampler2DArray)."""
        from luxc.builtins.functions import lookup_builtin
        from luxc.builtins.types import SAMPLER2D, VEC3, SCALAR
        sig = lookup_builtin("sample_compare", [SAMPLER2D, VEC3, SCALAR])
        assert sig is None


# ---------------------------------------------------------------------------
# P17.3 — Filter tests
# ---------------------------------------------------------------------------

class TestShadowFilters:
    """P17.3: Shadow filter emitters produce correct code patterns."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def _get_shadow_body(self, shadow_filter="pcf"):
        """Extract shadow evaluation body from first iteration of multi-light loop."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        # Inject the desired shadow_filter
        src = src.replace("shadow_filter: pcf", f"shadow_filter: {shadow_filter}")
        mod = parse_lux(src)
        strip_features(mod, {"has_shadows"})
        _resolve_imports(mod, EXAMPLES)
        expand_surfaces(mod, pipeline_filter="GltfForward")
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        main_fn = frag.functions[0]
        # Find first iteration
        for stmt in main_fn.body:
            if isinstance(stmt, IfStmt):
                inner_names = [s.name for s in stmt.then_body
                               if isinstance(s, LetStmt)]
                if "lt_0" in inner_names:
                    # Find the nested shadow guard IfStmt
                    for inner_stmt in stmt.then_body:
                        if isinstance(inner_stmt, IfStmt):
                            return inner_stmt.then_body
        pytest.fail("Shadow body not found")

    def test_hard_filter_emits_1_sample_compare(self):
        """hard filter emits 1 sample_compare call per light."""
        shadow_body = self._get_shadow_body("hard")
        from luxc.parser.ast_nodes import CallExpr
        sample_compares = [s for s in shadow_body if isinstance(s, LetStmt)
                          and isinstance(s.value, CallExpr)
                          and isinstance(s.value.func, VarRef)
                          and s.value.func.name == "sample_compare"]
        assert len(sample_compares) == 1

    def test_pcf_filter_emits_4_sample_compare(self):
        """PCF filter emits 4 sample_compare calls per light."""
        shadow_body = self._get_shadow_body("pcf")
        from luxc.parser.ast_nodes import CallExpr
        sample_compares = [s for s in shadow_body if isinstance(s, LetStmt)
                          and isinstance(s.value, CallExpr)
                          and isinstance(s.value.func, VarRef)
                          and s.value.func.name == "sample_compare"]
        assert len(sample_compares) == 4

    def test_pcss_filter_emits_sample_array_and_sample_compare(self):
        """PCSS filter emits 16 sample_array + 16 sample_compare per light."""
        shadow_body = self._get_shadow_body("pcss")
        from luxc.parser.ast_nodes import CallExpr
        sample_compares = [s for s in shadow_body if isinstance(s, LetStmt)
                          and isinstance(s.value, CallExpr)
                          and isinstance(s.value.func, VarRef)
                          and s.value.func.name == "sample_compare"]
        sample_arrays = [s for s in shadow_body if isinstance(s, LetStmt)
                        and isinstance(s.value, CallExpr)
                        and isinstance(s.value.func, VarRef)
                        and s.value.func.name == "sample_array"]
        assert len(sample_arrays) == 16
        assert len(sample_compares) == 16

    def test_shadow_filter_arg_parsed(self):
        """shadow_filter: pcf is extracted from layer args."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        mod = parse_lux(src)
        strip_features(mod, {"has_shadows"})
        lighting = mod.lightings[0]
        ml_layer = next(l for l in lighting.layers if l.name == "multi_light")
        arg_map = {a.name: a for a in ml_layer.args}
        assert "shadow_filter" in arg_map
        assert arg_map["shadow_filter"].value.name == "pcf"


# ---------------------------------------------------------------------------
# P17.3 — All 6 pipelines with shadows compile
# ---------------------------------------------------------------------------

@requires_spirv_tools
class TestAllPipelinesWithShadows:
    """P17.3: All 6 pipeline paths (3 modes x 2 bindless) compile with shadows."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def _compile_with_shadows(self, tmp_path, pipeline, bindless=False):
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        stem = f"p173_{pipeline}"
        if bindless:
            stem += "_bl"
        compile_source(
            src, stem, tmp_path, validate=True,
            source_dir=EXAMPLES, pipeline=pipeline,
            features={"has_shadows"}, bindless=bindless,
        )
        return list(tmp_path.glob("*.spv"))

    def test_forward_shadows(self, tmp_path):
        spvs = self._compile_with_shadows(tmp_path, "GltfForward")
        assert any("frag" in str(f) for f in spvs)

    def test_forward_bindless_shadows(self, tmp_path):
        spvs = self._compile_with_shadows(tmp_path, "GltfForward", bindless=True)
        assert any("frag" in str(f) for f in spvs)

    def test_rt_shadows(self, tmp_path):
        spvs = self._compile_with_shadows(tmp_path, "GltfRT")
        assert any("rchit" in str(f) for f in spvs)

    def test_rt_bindless_shadows(self, tmp_path):
        spvs = self._compile_with_shadows(tmp_path, "GltfRT", bindless=True)
        assert any("rchit" in str(f) for f in spvs)

    def test_mesh_shadows(self, tmp_path):
        spvs = self._compile_with_shadows(tmp_path, "GltfMesh")
        assert any("frag" in str(f) for f in spvs)

    def test_mesh_bindless_shadows(self, tmp_path):
        spvs = self._compile_with_shadows(tmp_path, "GltfMesh", bindless=True)
        assert any("frag" in str(f) for f in spvs)


# ---------------------------------------------------------------------------
# P17.3 — Shadow stdlib compilation
# ---------------------------------------------------------------------------

@requires_spirv_tools
class TestShadowStdlib:
    """P17.3: Shadow stdlib functions compile."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_shadow_stdlib_compiles(self, tmp_path):
        """All shadow stdlib functions compile to valid SPIR-V via has_shadows pipeline."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        compile_source(
            src, "shadow_stdlib", tmp_path, validate=True,
            source_dir=EXAMPLES, pipeline="GltfForward",
            features={"has_shadows"},
        )
        frag_spvs = list(tmp_path.glob("*.frag.spv"))
        assert len(frag_spvs) >= 1
