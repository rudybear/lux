"""Gaussian splatting pipeline tests.

Tests cover:
- Splat declaration grammar parsing (splat_decl, splat_member, properties)
- Configuration extraction from SplatDecl members (_get_splat_config)
- Pipeline expansion (splat → compute + vertex + fragment stages)
- Full compilation (parse → expand → type_check → codegen → SPIR-V assembly)
- Edge cases (splat + surface coexistence, vertex inputs, workgroup attribute)
"""

import json
import pytest
from pathlib import Path
from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import SplatDecl, SplatMember, Module, NumberLit, VarRef
from luxc.expansion.splat_expander import expand_splat_pipeline, _get_splat_config
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
# Splat source snippets
# ---------------------------------------------------------------------------

_BASIC_SPLAT = """
splat MyGaussians {
    sh_degree: 0,
}
"""

_SPLAT_SH3 = """
splat DetailedSplats {
    sh_degree: 3,
    kernel: ellipse,
    color_space: srgb,
    sort: camera_distance,
    alpha_cutoff: 0.01,
}
"""

_SPLAT_WITH_PROPERTIES = """
splat PropSplat {
    sh_degree: 1,
    properties Material {
        tint: vec3 = vec3(1.0, 1.0, 1.0),
        exposure: scalar = 1.0,
    },
}
"""

_MINIMAL_SPLAT = """
splat Minimal {
    sh_degree: 0,
}
"""

_SPLAT_ALL_OPTIONS = """
splat FullOptions {
    sh_degree: 2,
    kernel: ellipse,
    color_space: linear,
    sort: depth,
    alpha_cutoff: 0.002,
}
"""

_PIPELINE_WITH_SPLAT = """
splat GS {
    sh_degree: 0,
}
pipeline SplatRender {
    mode: gaussian_splat,
    splat: GS,
}
"""

_MULTIPLE_SPLATS = """
splat LowQuality {
    sh_degree: 0,
}
splat HighQuality {
    sh_degree: 3,
}
"""


def _compile_splat(tmp_path, source, stem="test_splat"):
    """Helper: compile a splat source to SPIR-V via compile_source."""
    from luxc.compiler import compile_source
    clear_type_aliases()
    compile_source(
        source, stem, tmp_path,
        emit_reflection=True, validate=True,
    )


# =========================================================================
# 1. Parser Tests
# =========================================================================

class TestSplatParsing:
    """Test that splat declarations parse correctly."""

    def test_parse_basic_splat(self):
        """A basic splat declaration should parse into a SplatDecl with the correct name and members."""
        module = parse_lux(_BASIC_SPLAT)
        assert len(module.splats) == 1
        splat = module.splats[0]
        assert isinstance(splat, SplatDecl)
        assert splat.name == "MyGaussians"
        assert len(splat.members) == 1
        assert splat.members[0].name == "sh_degree"

    def test_parse_splat_sh_degree(self):
        """sh_degree member value should be parsed as a NumberLit."""
        module = parse_lux(_BASIC_SPLAT)
        splat = module.splats[0]
        sh_member = splat.members[0]
        assert isinstance(sh_member, SplatMember)
        assert isinstance(sh_member.value, NumberLit)
        assert sh_member.value.value == "0"

    def test_parse_splat_with_properties(self):
        """A splat with a properties block should parse and retain the properties."""
        module = parse_lux(_SPLAT_WITH_PROPERTIES)
        splat = module.splats[0]
        assert splat.name == "PropSplat"
        assert splat.properties is not None
        assert splat.properties.name == "Material"
        assert len(splat.properties.fields) == 2
        field_names = [f.name for f in splat.properties.fields]
        assert "tint" in field_names
        assert "exposure" in field_names

    def test_parse_splat_minimal(self):
        """A minimal splat with only required fields should parse successfully."""
        module = parse_lux(_MINIMAL_SPLAT)
        assert len(module.splats) == 1
        assert module.splats[0].name == "Minimal"
        assert module.splats[0].properties is None

    def test_parse_splat_all_options(self):
        """A splat with all options should parse each member correctly."""
        module = parse_lux(_SPLAT_ALL_OPTIONS)
        splat = module.splats[0]
        assert splat.name == "FullOptions"
        member_names = {m.name for m in splat.members}
        assert member_names == {"sh_degree", "kernel", "color_space", "sort", "alpha_cutoff"}

    def test_parse_splat_in_pipeline(self):
        """A pipeline with mode: gaussian_splat and splat: ref should parse correctly."""
        module = parse_lux(_PIPELINE_WITH_SPLAT)
        assert len(module.splats) == 1
        assert len(module.pipelines) == 1
        pipeline = module.pipelines[0]
        member_names = {m.name for m in pipeline.members}
        assert "mode" in member_names
        assert "splat" in member_names
        # mode should be gaussian_splat
        mode_member = next(m for m in pipeline.members if m.name == "mode")
        assert isinstance(mode_member.value, VarRef)
        assert mode_member.value.name == "gaussian_splat"

    def test_parse_module_splats_list(self):
        """Module.splats should be populated with all parsed splat declarations."""
        module = parse_lux(_MULTIPLE_SPLATS)
        assert len(module.splats) == 2

    def test_parse_multiple_splats(self):
        """Multiple splat declarations should each be parsed with distinct names."""
        module = parse_lux(_MULTIPLE_SPLATS)
        names = {s.name for s in module.splats}
        assert names == {"LowQuality", "HighQuality"}
        # Verify sh_degree differs
        for s in module.splats:
            sh_member = next(m for m in s.members if m.name == "sh_degree")
            if s.name == "LowQuality":
                assert sh_member.value.value == "0"
            else:
                assert sh_member.value.value == "3"


# =========================================================================
# 2. Config Extraction Tests
# =========================================================================

class TestSplatConfig:
    """Test _get_splat_config extraction from SplatDecl members."""

    def test_get_splat_config_defaults(self):
        """An empty splat (no members) should yield all default config values."""
        splat = SplatDecl("Empty", [])
        config = _get_splat_config(splat)
        assert config["sh_degree"] == 0
        assert config["kernel"] == "ellipse"
        assert config["color_space"] == "srgb"
        assert config["sort"] == "camera_distance"
        assert config["alpha_cutoff"] == pytest.approx(0.004)

    def test_get_splat_config_sh_degree_3(self):
        """Custom sh_degree should override the default."""
        splat = SplatDecl("SH3", [SplatMember("sh_degree", NumberLit("3"))])
        config = _get_splat_config(splat)
        assert config["sh_degree"] == 3

    def test_get_splat_config_linear_color(self):
        """color_space: linear should be extracted correctly."""
        splat = SplatDecl("Linear", [SplatMember("color_space", VarRef("linear"))])
        config = _get_splat_config(splat)
        assert config["color_space"] == "linear"

    def test_get_splat_config_custom_alpha(self):
        """Custom alpha_cutoff should be extracted as a float."""
        splat = SplatDecl("Custom", [SplatMember("alpha_cutoff", NumberLit("0.01"))])
        config = _get_splat_config(splat)
        assert config["alpha_cutoff"] == pytest.approx(0.01)


# =========================================================================
# 3. Expansion Tests
# =========================================================================

class TestSplatExpansion:
    """Test expand_splat_pipeline stage generation."""

    def _expand(self, source):
        """Parse source, expand surfaces, and return the generated stages."""
        clear_type_aliases()
        module = parse_lux(source)
        if not hasattr(module, '_defines'):
            module._defines = {}
        expand_surfaces(module)
        return module.stages

    def _expand_direct(self, splat_src, sh_degree=0):
        """Directly invoke expand_splat_pipeline for fine-grained control."""
        from luxc.parser.ast_nodes import PipelineDecl, PipelineMember
        clear_type_aliases()
        members = [SplatMember("sh_degree", NumberLit(str(sh_degree)))]
        splat = SplatDecl("Test", members)
        pipeline = PipelineDecl("TestPipe", [
            PipelineMember("mode", VarRef("gaussian_splat")),
            PipelineMember("splat", VarRef("Test")),
        ])
        module = Module()
        module._defines = {}
        return expand_splat_pipeline(splat, pipeline, module)

    def test_expand_produces_three_stages(self):
        """Expansion should produce exactly three stages: compute, vertex, fragment."""
        stages = self._expand(_PIPELINE_WITH_SPLAT)
        assert len(stages) == 3
        stage_types = [s.stage_type for s in stages]
        assert stage_types == ["compute", "vertex", "fragment"]

    def test_expand_compute_storage_buffers(self):
        """Compute stage should have SSBOs for splat input, projected output, and sort keys."""
        stages = self._expand(_PIPELINE_WITH_SPLAT)
        compute = stages[0]
        sb_names = [sb.name for sb in compute.storage_buffers]
        # Input buffers
        assert "splat_pos" in sb_names
        assert "splat_rot" in sb_names
        assert "splat_scale" in sb_names
        assert "splat_opacity" in sb_names
        assert "splat_sh0" in sb_names
        # Output buffers
        assert "projected_center" in sb_names
        assert "projected_conic" in sb_names
        assert "projected_color" in sb_names
        assert "sort_keys" in sb_names
        assert "visible_count" in sb_names

    def test_expand_vertex_outputs(self):
        """Vertex stage should have outputs for conic, color, center, and offset."""
        stages = self._expand(_PIPELINE_WITH_SPLAT)
        vertex = stages[1]
        out_names = {v.name for v in vertex.outputs}
        assert out_names == {"frag_conic", "frag_color", "frag_center", "frag_offset"}

    def test_expand_fragment_inputs_match_vertex_outputs(self):
        """Fragment inputs should exactly match vertex outputs in name and type."""
        stages = self._expand(_PIPELINE_WITH_SPLAT)
        vertex = stages[1]
        fragment = stages[2]
        vert_outs = {(v.name, v.type_name) for v in vertex.outputs}
        frag_ins = {(v.name, v.type_name) for v in fragment.inputs}
        assert vert_outs == frag_ins

    def test_expand_sh_degree_0_buffers(self):
        """SH degree 0 should produce exactly 1 SH buffer (splat_sh0)."""
        stages = self._expand_direct("", sh_degree=0)
        compute = stages[0]
        sh_buffers = [sb.name for sb in compute.storage_buffers if sb.name.startswith("splat_sh")]
        assert sh_buffers == ["splat_sh0"]

    def test_expand_sh_degree_3_buffers(self):
        """SH degree 3 should produce 16 SH buffers (splat_sh0 through splat_sh15)."""
        stages = self._expand_direct("", sh_degree=3)
        compute = stages[0]
        sh_buffers = [sb.name for sb in compute.storage_buffers if sb.name.startswith("splat_sh")]
        expected = [f"splat_sh{i}" for i in range(16)]
        assert sh_buffers == expected

    def test_expand_compute_push_constants(self):
        """Compute stage push constants should include view/proj matrices and camera position."""
        stages = self._expand(_PIPELINE_WITH_SPLAT)
        compute = stages[0]
        assert len(compute.push_constants) == 1
        pc = compute.push_constants[0]
        field_names = {f.name for f in pc.fields}
        assert "view_matrix" in field_names
        assert "proj_matrix" in field_names
        assert "cam_pos" in field_names
        assert "screen_size" in field_names
        assert "total_splats" in field_names
        assert "focal_x" in field_names
        assert "focal_y" in field_names

    def test_expand_invalid_sh_degree(self):
        """SH degree 4 should raise a ValueError."""
        from luxc.parser.ast_nodes import PipelineDecl, PipelineMember
        splat = SplatDecl("Bad", [SplatMember("sh_degree", NumberLit("4"))])
        pipeline = PipelineDecl("BadPipe", [
            PipelineMember("mode", VarRef("gaussian_splat")),
            PipelineMember("splat", VarRef("Bad")),
        ])
        module = Module()
        module._defines = {}
        with pytest.raises(ValueError, match="Unsupported SH degree 4"):
            expand_splat_pipeline(splat, pipeline, module)


# =========================================================================
# 4. Full Compilation Tests
# =========================================================================

@requires_spirv_tools
class TestSplatCompilation:
    """End-to-end compilation tests for Gaussian splatting (require spirv-as/spirv-val)."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_compile_splat_sh0(self, tmp_path):
        """Full compile with sh_degree 0 should succeed."""
        src = _PIPELINE_WITH_SPLAT
        _compile_splat(tmp_path, src)

    def test_compile_splat_sh1(self, tmp_path):
        """Full compile with sh_degree 1 should succeed."""
        src = """
        splat GS { sh_degree: 1, }
        pipeline P { mode: gaussian_splat, splat: GS, }
        """
        _compile_splat(tmp_path, src)

    def test_compile_splat_sh2(self, tmp_path):
        """Full compile with sh_degree 2 should succeed."""
        src = """
        splat GS { sh_degree: 2, }
        pipeline P { mode: gaussian_splat, splat: GS, }
        """
        _compile_splat(tmp_path, src)

    def test_compile_splat_sh3(self, tmp_path):
        """Full compile with sh_degree 3 should succeed."""
        src = """
        splat GS { sh_degree: 3, }
        pipeline P { mode: gaussian_splat, splat: GS, }
        """
        _compile_splat(tmp_path, src)

    def test_compile_splat_linear_color(self, tmp_path):
        """Full compile with color_space: linear should succeed."""
        src = """
        splat GS { sh_degree: 0, color_space: linear, }
        pipeline P { mode: gaussian_splat, splat: GS, }
        """
        _compile_splat(tmp_path, src)

    def test_compile_produces_three_spv(self, tmp_path):
        """Compilation should produce .comp.spv, .vert.spv, and .frag.spv files."""
        _compile_splat(tmp_path, _PIPELINE_WITH_SPLAT)
        assert (tmp_path / "test_splat.comp.spv").exists()
        assert (tmp_path / "test_splat.vert.spv").exists()
        assert (tmp_path / "test_splat.frag.spv").exists()

    def test_compile_reflection_has_gaussian_splatting(self, tmp_path):
        """Reflection JSON for the compute stage should have a gaussian_splatting section."""
        _compile_splat(tmp_path, _PIPELINE_WITH_SPLAT)
        json_path = tmp_path / "test_splat.comp.json"
        assert json_path.exists()
        meta = json.loads(json_path.read_text())
        assert "gaussian_splatting" in meta
        gs = meta["gaussian_splatting"]
        assert gs["sh_degree"] == 0
        assert gs["kernel"] == "ellipse"
        assert "input_buffers" in gs
        assert "output_buffers" in gs

    def test_compile_via_pipeline(self, tmp_path):
        """Compile via pipeline referencing a named splat."""
        src = """
        splat Gaussians {
            sh_degree: 1,
            alpha_cutoff: 0.005,
        }
        pipeline Renderer {
            mode: gaussian_splat,
            splat: Gaussians,
        }
        """
        _compile_splat(tmp_path, src, stem="pipeline_splat")
        assert (tmp_path / "pipeline_splat.comp.spv").exists()
        assert (tmp_path / "pipeline_splat.vert.spv").exists()
        assert (tmp_path / "pipeline_splat.frag.spv").exists()


# =========================================================================
# 5. Edge Case Tests
# =========================================================================

class TestSplatEdgeCases:
    """Test edge cases and interaction with other module features."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_splat_with_surface_does_not_conflict(self):
        """A module containing both a splat and a surface should parse without conflict."""
        src = """
        splat GS {
            sh_degree: 0,
        }
        surface SimplePBR {
            layers [
                base(albedo: vec3(0.8, 0.2, 0.1), roughness: 0.5, metallic: 0.0),
            ]
        }
        """
        module = parse_lux(src)
        assert len(module.splats) == 1
        assert len(module.surfaces) == 1
        assert module.splats[0].name == "GS"
        assert module.surfaces[0].name == "SimplePBR"

    def test_splat_features_conditional(self):
        """A splat with an option based on a VarRef should parse the member correctly."""
        src = """
        splat GS {
            sh_degree: 2,
            kernel: ellipse,
        }
        """
        module = parse_lux(src)
        splat = module.splats[0]
        kernel_member = next(m for m in splat.members if m.name == "kernel")
        assert isinstance(kernel_member.value, VarRef)
        assert kernel_member.value.name == "ellipse"

    def test_vertex_has_no_inputs(self):
        """The generated vertex stage should have an empty inputs list (SSBOs supply data)."""
        module = parse_lux(_PIPELINE_WITH_SPLAT)
        module._defines = {}
        expand_surfaces(module)
        vertex = next(s for s in module.stages if s.stage_type == "vertex")
        assert vertex.inputs == []

    def test_compute_has_workgroup_attribute(self):
        """The compute main function should have a workgroup_size attribute."""
        module = parse_lux(_PIPELINE_WITH_SPLAT)
        module._defines = {}
        expand_surfaces(module)
        compute = next(s for s in module.stages if s.stage_type == "compute")
        main_fn = next(fn for fn in compute.functions if fn.name == "main")
        assert any("workgroup_size" in attr for attr in main_fn.attributes)

    def test_fragment_alpha_cutoff(self):
        """The fragment stage push constants should include alpha_cutoff."""
        module = parse_lux(_PIPELINE_WITH_SPLAT)
        module._defines = {}
        expand_surfaces(module)
        fragment = next(s for s in module.stages if s.stage_type == "fragment")
        assert len(fragment.push_constants) == 1
        pc = fragment.push_constants[0]
        field_names = {f.name for f in pc.fields}
        assert "alpha_cutoff" in field_names
