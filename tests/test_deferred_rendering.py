"""Deferred rendering pipeline tests.

Tests cover:
- Parsing: mode: deferred parsed from pipeline declarations
- Schedule: deferred-specific schedule slots accepted and validated
- Expansion: 4 stages produced, MRT outputs, fullscreen vertex, descriptor sets
- Compilation: full compile to SPIR-V, reflection JSON, variants
- Edge cases: no surface raises, default geometry, coexistence with forward
"""

import json
import pytest
from pathlib import Path
from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import Module, VarRef, NumberLit
from luxc.expansion.surface_expander import expand_surfaces
from luxc.builtins.types import clear_type_aliases


def _has_spirv_tools() -> bool:
    try:
        import subprocess
        subprocess.run(["spirv-as", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


requires_spirv_tools = pytest.mark.skipif(
    not _has_spirv_tools(), reason="spirv-as/spirv-val not found on PATH"
)


# ---------------------------------------------------------------------------
# Source snippets
# ---------------------------------------------------------------------------

_BASIC_DEFERRED = """
import brdf;
import color;

geometry Geo {
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

surface PBR {
    sampler2d albedo_tex,

    properties Material {
        base_color: vec4 = vec4(1.0, 1.0, 1.0, 1.0),
        roughness: scalar = 0.5,
        metallic: scalar = 0.0,
    },

    layers [
        base(albedo: sample(albedo_tex, uv).xyz * Material.base_color.xyz,
             roughness: Material.roughness,
             metallic: Material.metallic),
    ]
}

pipeline Deferred {
    mode: deferred,
    geometry: Geo,
    surface: PBR,
}
"""

_DEFERRED_WITH_SCHEDULE = """
import brdf;
import color;

geometry Geo {
    position: vec3,
    normal: vec3,
    transform: MVP {
        model: mat4,
        view: mat4,
        projection: mat4,
    }
    outputs {
        world_pos: (model * vec4(position, 1.0)).xyz,
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        clip_pos: projection * view * model * vec4(position, 1.0),
    }
}

surface PBR {
    layers [
        base(albedo: vec3(0.8, 0.2, 0.1),
             roughness: 0.5,
             metallic: 0.0),
    ]
}

schedule HQ {
    tonemap: aces,
    gbuffer_precision: standard,
    normal_encoding: octahedron,
}

pipeline Deferred {
    mode: deferred,
    geometry: Geo,
    surface: PBR,
    schedule: HQ,
}
"""

_DEFERRED_WITH_EMISSION = """
import brdf;
import color;

geometry Geo {
    position: vec3,
    normal: vec3,
    transform: MVP {
        model: mat4,
        view: mat4,
        projection: mat4,
    }
    outputs {
        world_pos: (model * vec4(position, 1.0)).xyz,
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        clip_pos: projection * view * model * vec4(position, 1.0),
    }
}

surface EmissivePBR {
    layers [
        base(albedo: vec3(0.5),
             roughness: 0.5,
             metallic: 0.0),
        emission(color: vec3(1.0, 0.5, 0.0)),
    ]
}

pipeline Deferred {
    mode: deferred,
    geometry: Geo,
    surface: EmissivePBR,
}
"""

_DEFERRED_WITH_LIGHTING = """
import brdf;
import color;
import lighting;

geometry Geo {
    position: vec3,
    normal: vec3,
    transform: MVP {
        model: mat4,
        view: mat4,
        projection: mat4,
    }
    outputs {
        world_pos: (model * vec4(position, 1.0)).xyz,
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        clip_pos: projection * view * model * vec4(position, 1.0),
    }
}

surface PBR {
    layers [
        base(albedo: vec3(0.8),
             roughness: 0.5,
             metallic: 0.0),
    ]
}

lighting SceneLights {
    properties SceneLight {
        view_pos: vec3 = vec3(0.0, 0.0, 3.0),
        light_count: int = 0,
    },

    layers [
        multi_light(count: SceneLight.light_count, max_lights: 4),
    ]
}

pipeline Deferred {
    mode: deferred,
    geometry: Geo,
    surface: PBR,
    lighting: SceneLights,
}
"""

_DEFERRED_NO_SURFACE = """
geometry Geo {
    position: vec3,
    normal: vec3,
    transform: MVP {
        model: mat4,
    }
    outputs {
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        clip_pos: model * vec4(position, 1.0),
    }
}

pipeline Deferred {
    mode: deferred,
    geometry: Geo,
}
"""

_DEFERRED_NO_GEOMETRY = """
import brdf;
import color;

surface SimplePBR {
    layers [
        base(albedo: vec3(0.5),
             roughness: 0.5,
             metallic: 0.0),
    ]
}

pipeline Deferred {
    mode: deferred,
    surface: SimplePBR,
}
"""

_DEFERRED_PLUS_FORWARD = """
import brdf;
import color;

geometry Geo {
    position: vec3,
    normal: vec3,
    transform: MVP {
        model: mat4,
        view: mat4,
        projection: mat4,
    }
    outputs {
        world_pos: (model * vec4(position, 1.0)).xyz,
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        clip_pos: projection * view * model * vec4(position, 1.0),
    }
}

surface PBR {
    layers [
        base(albedo: vec3(0.8),
             roughness: 0.5,
             metallic: 0.0),
    ]
}

pipeline Forward {
    geometry: Geo,
    surface: PBR,
}

pipeline Deferred {
    mode: deferred,
    geometry: Geo,
    surface: PBR,
}
"""


def _compile_deferred(tmp_path, source, stem="test_deferred"):
    """Helper: compile deferred source to SPIR-V via compile_source."""
    from luxc.compiler import compile_source
    clear_type_aliases()
    compile_source(
        source, stem, tmp_path,
        emit_reflection=True, validate=True,
    )


def _expand(source):
    """Parse source, expand surfaces, return module stages."""
    clear_type_aliases()
    module = parse_lux(source)
    if not hasattr(module, '_defines'):
        module._defines = {}
    from luxc.compiler import _resolve_imports
    _resolve_imports(module, Path(__file__).parent.parent / "luxc" / "stdlib")
    expand_surfaces(module)
    return module.stages


# =========================================================================
# 1. Parser Tests
# =========================================================================

class TestDeferredParsing:
    """Test that mode: deferred is parsed correctly from pipeline declarations."""

    def test_parse_deferred_mode(self):
        """Pipeline with mode: deferred should parse correctly."""
        module = parse_lux(_BASIC_DEFERRED)
        assert len(module.pipelines) == 1
        pipeline = module.pipelines[0]
        mode_member = next(m for m in pipeline.members if m.name == "mode")
        assert isinstance(mode_member.value, VarRef)
        assert mode_member.value.name == "deferred"

    def test_parse_deferred_with_schedule(self):
        """Pipeline with mode: deferred and schedule should parse both."""
        module = parse_lux(_DEFERRED_WITH_SCHEDULE)
        pipeline = module.pipelines[0]
        member_names = {m.name for m in pipeline.members}
        assert "mode" in member_names
        assert "schedule" in member_names

    def test_parse_deferred_with_lighting(self):
        """Pipeline with mode: deferred and lighting block should parse."""
        module = parse_lux(_DEFERRED_WITH_LIGHTING)
        pipeline = module.pipelines[0]
        member_names = {m.name for m in pipeline.members}
        assert "mode" in member_names
        assert "lighting" in member_names


# =========================================================================
# 2. Schedule Tests
# =========================================================================

class TestDeferredSchedule:
    """Test deferred-specific schedule slots."""

    def test_schedule_deferred_slots_accepted(self):
        """Deferred schedule slots (gbuffer_precision, normal_encoding) should be valid."""
        module = parse_lux(_DEFERRED_WITH_SCHEDULE)
        assert len(module.schedules) == 1
        sched = module.schedules[0]
        member_names = {m.name for m in sched.members}
        assert "gbuffer_precision" in member_names
        assert "normal_encoding" in member_names

    def test_schedule_defaults_applied(self):
        """Default schedule values should be applied for deferred slots."""
        stages = _expand(_DEFERRED_WITH_SCHEDULE)
        # Should produce 4 stages without error
        assert len(stages) == 4

    def test_invalid_schedule_slot_raises(self):
        """An invalid schedule slot should raise ValueError."""
        source = """
import brdf;
import color;

surface PBR {
    layers [
        base(albedo: vec3(0.5), roughness: 0.5, metallic: 0.0),
    ]
}

schedule Bad {
    nonexistent_slot: value,
}

pipeline P {
    mode: deferred,
    surface: PBR,
    schedule: Bad,
}
"""
        with pytest.raises(ValueError, match="Unknown schedule slot"):
            _expand(source)


# =========================================================================
# 3. Expansion Tests
# =========================================================================

class TestDeferredExpansion:
    """Test expand_deferred_pipeline stage generation."""

    def test_produces_four_stages(self):
        """Expansion should produce exactly 4 stages."""
        stages = _expand(_BASIC_DEFERRED)
        assert len(stages) == 4

    def test_stage_types(self):
        """Stages should be: vertex, fragment, vertex, fragment."""
        stages = _expand(_BASIC_DEFERRED)
        types = [s.stage_type for s in stages]
        assert types == ["vertex", "fragment", "vertex", "fragment"]

    def test_gbuffer_vertex_has_geometry_inputs(self):
        """G-buffer vertex should have geometry inputs (position, normal, uv)."""
        stages = _expand(_BASIC_DEFERRED)
        gbuf_vert = stages[0]
        input_names = {v.name for v in gbuf_vert.inputs}
        assert "position" in input_names
        assert "normal" in input_names
        assert "uv" in input_names

    def test_gbuffer_frag_has_three_outputs(self):
        """G-buffer fragment should have 3 MRT outputs."""
        stages = _expand(_BASIC_DEFERRED)
        gbuf_frag = stages[1]
        assert len(gbuf_frag.outputs) == 3
        out_names = {v.name for v in gbuf_frag.outputs}
        assert out_names == {"gbuf_rt0", "gbuf_rt1", "gbuf_rt2"}

    def test_gbuffer_frag_no_light_uniform(self):
        """G-buffer fragment should NOT have a Light uniform block."""
        stages = _expand(_BASIC_DEFERRED)
        gbuf_frag = stages[1]
        uniform_names = {ub.name for ub in gbuf_frag.uniforms}
        assert "Light" not in uniform_names

    def test_gbuffer_frag_has_material_uniform(self):
        """G-buffer fragment should have the surface Material uniform."""
        stages = _expand(_BASIC_DEFERRED)
        gbuf_frag = stages[1]
        uniform_names = {ub.name for ub in gbuf_frag.uniforms}
        assert "Material" in uniform_names

    def test_lighting_vertex_no_inputs(self):
        """Fullscreen lighting vertex should have no vertex inputs."""
        stages = _expand(_BASIC_DEFERRED)
        light_vert = stages[2]
        assert len(light_vert.inputs) == 0

    def test_lighting_vertex_has_frag_uv_output(self):
        """Fullscreen lighting vertex should output frag_uv."""
        stages = _expand(_BASIC_DEFERRED)
        light_vert = stages[2]
        out_names = {v.name for v in light_vert.outputs}
        assert "frag_uv" in out_names

    def test_lighting_frag_has_gbuffer_samplers(self):
        """Lighting fragment should have G-buffer texture samplers."""
        stages = _expand(_BASIC_DEFERRED)
        light_frag = stages[3]
        sam_names = {s.name for s in light_frag.samplers}
        assert "gbuf_tex0" in sam_names
        assert "gbuf_tex1" in sam_names
        assert "gbuf_tex2" in sam_names
        assert "gbuf_depth" in sam_names

    def test_lighting_frag_has_deferred_camera(self):
        """Lighting fragment should have DeferredCamera uniform."""
        stages = _expand(_BASIC_DEFERRED)
        light_frag = stages[3]
        uniform_names = {ub.name for ub in light_frag.uniforms}
        assert "DeferredCamera" in uniform_names

    def test_lighting_frag_single_output(self):
        """Lighting fragment should have a single color output."""
        stages = _expand(_BASIC_DEFERRED)
        light_frag = stages[3]
        assert len(light_frag.outputs) == 1
        assert light_frag.outputs[0].name == "color"

    def test_descriptor_set_offsets(self):
        """Each stage should have the correct descriptor set offset.
        G-buffer stages use sets 0/1 (same pipeline), lighting stages use set 0
        (separate pipeline, independent set numbering)."""
        stages = _expand(_BASIC_DEFERRED)
        assert getattr(stages[0], '_descriptor_set_offset', None) == 0
        assert getattr(stages[1], '_descriptor_set_offset', None) == 1
        assert getattr(stages[2], '_descriptor_set_offset', None) == 0
        assert getattr(stages[3], '_descriptor_set_offset', None) == 0

    def test_deferred_pass_tags(self):
        """Stages should be tagged with _deferred_pass."""
        stages = _expand(_BASIC_DEFERRED)
        assert stages[0]._deferred_pass == "gbuffer"
        assert stages[1]._deferred_pass == "gbuffer"
        assert stages[2]._deferred_pass == "lighting"
        assert stages[3]._deferred_pass == "lighting"

    def test_output_stem_suffixes(self):
        """Stages should be tagged with _output_stem_suffix."""
        stages = _expand(_BASIC_DEFERRED)
        assert stages[0]._output_stem_suffix == "gbuf"
        assert stages[1]._output_stem_suffix == "gbuf"
        assert stages[2]._output_stem_suffix == "light"
        assert stages[3]._output_stem_suffix == "light"


# =========================================================================
# 4. Compilation Tests
# =========================================================================

class TestDeferredCompilation:
    """Test full compilation of deferred pipelines to SPIR-V."""

    @requires_spirv_tools
    def test_compile_basic_deferred(self, tmp_path):
        """Basic deferred pipeline should compile without errors."""
        _compile_deferred(tmp_path, _BASIC_DEFERRED)

    @requires_spirv_tools
    def test_compile_produces_four_spv_files(self, tmp_path):
        """Compilation should produce 4 .spv files with correct stems."""
        _compile_deferred(tmp_path, _BASIC_DEFERRED)
        spv_files = sorted(p.name for p in tmp_path.glob("*.spv"))
        assert "test_deferred.gbuf.vert.spv" in spv_files
        assert "test_deferred.gbuf.frag.spv" in spv_files
        assert "test_deferred.light.vert.spv" in spv_files
        assert "test_deferred.light.frag.spv" in spv_files

    @requires_spirv_tools
    def test_compile_produces_reflection_json(self, tmp_path):
        """Compilation should produce reflection JSON files."""
        _compile_deferred(tmp_path, _BASIC_DEFERRED)
        json_files = sorted(p.name for p in tmp_path.glob("*.json"))
        assert "test_deferred.gbuf.frag.json" in json_files
        assert "test_deferred.light.frag.json" in json_files

    @requires_spirv_tools
    def test_reflection_gbuf_frag_has_deferred_section(self, tmp_path):
        """G-buffer fragment reflection should have deferred section with render_targets."""
        _compile_deferred(tmp_path, _BASIC_DEFERRED)
        with open(tmp_path / "test_deferred.gbuf.frag.json") as f:
            refl = json.load(f)
        assert "deferred" in refl
        assert refl["deferred"]["pass"] == "gbuffer"
        assert len(refl["deferred"]["render_targets"]) == 3

    @requires_spirv_tools
    def test_reflection_light_frag_has_deferred_section(self, tmp_path):
        """Lighting fragment reflection should have deferred section."""
        _compile_deferred(tmp_path, _BASIC_DEFERRED)
        with open(tmp_path / "test_deferred.light.frag.json") as f:
            refl = json.load(f)
        assert "deferred" in refl
        assert refl["deferred"]["pass"] == "lighting"

    @requires_spirv_tools
    def test_compile_with_emission(self, tmp_path):
        """Deferred pipeline with emission layer should compile."""
        _compile_deferred(tmp_path, _DEFERRED_WITH_EMISSION)

    @requires_spirv_tools
    def test_compile_with_schedule(self, tmp_path):
        """Deferred pipeline with schedule should compile."""
        _compile_deferred(tmp_path, _DEFERRED_WITH_SCHEDULE)

    @requires_spirv_tools
    def test_compile_with_multi_light(self, tmp_path):
        """Deferred pipeline with multi_light lighting should compile."""
        _compile_deferred(tmp_path, _DEFERRED_WITH_LIGHTING)


# =========================================================================
# 5. Edge Case Tests
# =========================================================================

class TestDeferredEdgeCases:
    """Test edge cases for deferred pipeline expansion."""

    def test_no_surface_raises(self):
        """Pipeline with mode: deferred but no surface should raise ValueError."""
        with pytest.raises(ValueError, match="no surface"):
            _expand(_DEFERRED_NO_SURFACE)

    def test_no_geometry_uses_defaults(self):
        """Pipeline with mode: deferred but no geometry should use default vertex."""
        stages = _expand(_DEFERRED_NO_GEOMETRY)
        assert len(stages) == 4
        gbuf_vert = stages[0]
        input_names = {v.name for v in gbuf_vert.inputs}
        assert "position" in input_names
        assert "normal" in input_names

    @requires_spirv_tools
    def test_coexists_with_forward(self, tmp_path):
        """Deferred and forward pipelines in the same file should both compile."""
        _compile_deferred(tmp_path, _DEFERRED_PLUS_FORWARD, stem="test_dual")
        spv_files = sorted(p.name for p in tmp_path.glob("*.spv"))
        # Forward: 2 stages
        assert "test_dual.vert.spv" in spv_files
        assert "test_dual.frag.spv" in spv_files
        # Deferred: 4 stages
        assert "test_dual.gbuf.vert.spv" in spv_files
        assert "test_dual.gbuf.frag.spv" in spv_files
        assert "test_dual.light.vert.spv" in spv_files
        assert "test_dual.light.frag.spv" in spv_files
