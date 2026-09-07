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
        # alpha_cutoff/alpha_min default to 1/255 (3DGS/gsplat convention,
        # SPECIFICATION.md 12.8), not the old 0.004.
        assert config["alpha_cutoff"] == pytest.approx(1.0 / 255.0)
        assert config["alpha_min"] == pytest.approx(1.0 / 255.0)
        assert config["dilation"] == pytest.approx(0.3)

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
        """Custom alpha_cutoff (legacy name) should be extracted as a float
        and mirrored into alpha_min (they're aliases of each other)."""
        splat = SplatDecl("Custom", [SplatMember("alpha_cutoff", NumberLit("0.01"))])
        config = _get_splat_config(splat)
        assert config["alpha_cutoff"] == pytest.approx(0.01)
        assert config["alpha_min"] == pytest.approx(0.01)

    def test_get_splat_config_custom_alpha_min(self):
        """alpha_min (the reference-matching name) should also set the
        legacy alpha_cutoff alias."""
        splat = SplatDecl("Custom", [SplatMember("alpha_min", NumberLit("0.02"))])
        config = _get_splat_config(splat)
        assert config["alpha_min"] == pytest.approx(0.02)
        assert config["alpha_cutoff"] == pytest.approx(0.02)

    def test_get_splat_config_custom_dilation(self):
        splat = SplatDecl("Custom", [SplatMember("dilation", NumberLit("0.5"))])
        config = _get_splat_config(splat)
        assert config["dilation"] == pytest.approx(0.5)

    def test_get_splat_config_sort_view_depth(self):
        splat = SplatDecl("Custom", [SplatMember("sort", VarRef("view_depth"))])
        config = _get_splat_config(splat)
        assert config["sort"] == "view_depth"


class TestSplatAntialiasing:
    """3DGS/gsplat antialiasing conventions (SPECIFICATION.md 12.8): no
    opacity compensation, dilation baked as a compile-time literal, alpha
    clamped to <= 0.99, only a trivial power>0 discard (no arbitrary -4.0
    cutoff), and a real (not silently-ignored) `sort` option."""

    def test_no_compensation_term_emitted(self):
        from luxc.expansion.splat_expander import _build_preprocess_stage
        stage = _build_preprocess_stage(_get_splat_config(SplatDecl("S", [])))
        main_fn = next(fn for fn in stage.functions if fn.name == "main")
        var_names = {stmt.name for stmt in main_fn.body if hasattr(stmt, "name")}
        assert "compensation" not in var_names
        assert "det_orig" not in var_names

    def test_dilation_baked_as_literal(self):
        from luxc.expansion.splat_expander import _build_preprocess_stage
        config = _get_splat_config(SplatDecl("S", [SplatMember("dilation", NumberLit("0.7"))]))
        stage = _build_preprocess_stage(config)
        main_fn = next(fn for fn in stage.functions if fn.name == "main")
        dilation_lets = [stmt for stmt in main_fn.body
                         if hasattr(stmt, "name") and stmt.name in ("cov2d_00f", "cov2d_11f")]
        assert len(dilation_lets) == 2
        for stmt in dilation_lets:
            # value is `cov2d_XX + <dilation literal>`; the literal's value
            # should be exactly the configured dilation.
            assert stmt.value.right.value == "0.7"

    def test_fragment_no_arbitrary_power_cutoff(self):
        from luxc.expansion.splat_expander import _build_fragment_body
        config = _get_splat_config(SplatDecl("S", []))
        body = _build_fragment_body(config)
        # No power-related discard at all: with the isotropic eigenbasis
        # evaluation (`d2 = dot(rel, rel)`, `power = -0.5*d2`), `power` can
        # never be positive under IEEE-754 (a sum of two non-negative
        # squares), so the old `power > 0.0` safety-net discard (itself a
        # replacement for an even older, arbitrary `power < -4.0` cutoff)
        # is now provably dead code and has been removed -- see
        # splat_expander.py's "Isotropic fragment evaluation" docstring
        # note. The only remaining discard is the real visibility cutoff,
        # `alpha < alpha_min`.
        power_conditions = [stmt.condition for stmt in body
                             if hasattr(stmt, "condition") and hasattr(stmt.condition, "right")
                             and getattr(stmt.condition.left, "name", None) == "power"]
        assert power_conditions == []

    def test_fragment_alpha_clamped_to_099(self):
        from luxc.expansion.splat_expander import _build_fragment_body
        config = _get_splat_config(SplatDecl("S", []))
        body = _build_fragment_body(config)
        alpha_let = next(stmt for stmt in body if getattr(stmt, "name", None) == "alpha")
        assert alpha_let.value.func.name == "min"
        assert any(getattr(a, "value", None) == "0.99" for a in alpha_let.value.args)


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
        assert "projected_extent" in sb_names
        assert "projected_color" in sb_names
        assert "sort_keys" in sb_names
        assert "visible_count" in sb_names

    def test_expand_vertex_outputs(self):
        """Vertex stage should have outputs for the isotropic rel + color."""
        stages = self._expand(_PIPELINE_WITH_SPLAT)
        vertex = stages[1]
        out_names = {v.name for v in vertex.outputs}
        assert out_names == {"frag_rel", "frag_color"}

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
        """The fragment stage push constants should include alpha_min (the
        reference-matching name; renamed from alpha_cutoff, SPECIFICATION.md
        12.8's 3DGS/gsplat antialiasing convention update)."""
        module = parse_lux(_PIPELINE_WITH_SPLAT)
        module._defines = {}
        expand_surfaces(module)
        fragment = next(s for s in module.stages if s.stage_type == "fragment")
        assert len(fragment.push_constants) == 1
        pc = fragment.push_constants[0]
        field_names = {f.name for f in pc.fields}
        assert "alpha_min" in field_names


# =========================================================================
# 5. Dynamic splats: motion: keyframes (morph-apply compute stage)
# =========================================================================

_PIPELINE_WITH_MOTION = """
splat GS {
    sh_degree: 0,
    motion: keyframes,
}

pipeline SplatRender {
    mode: gaussian_splat,
    splat: GS,
}
"""


class TestSplatMotionConfig:
    """_get_splat_config should recognize the optional `motion` field."""

    def test_default_motion_is_none(self):
        splat = SplatDecl("GS", [])
        config = _get_splat_config(splat)
        assert config["motion"] == "none"

    def test_motion_keyframes(self):
        splat = SplatDecl("GS", [SplatMember("motion", VarRef("keyframes"))])
        config = _get_splat_config(splat)
        assert config["motion"] == "keyframes"


class TestMorphApplyExpansion:
    """expand_splat_pipeline should prepend a morph-apply compute stage when
    motion: keyframes is set, and otherwise stay exactly as before."""

    def _expand(self, source):
        clear_type_aliases()
        module = parse_lux(source)
        module._defines = {}
        expand_surfaces(module)
        return module.stages

    def test_no_motion_still_three_stages(self):
        stages = self._expand(_PIPELINE_WITH_SPLAT)
        assert len(stages) == 3
        assert [s.stage_type for s in stages] == ["compute", "vertex", "fragment"]

    def test_motion_keyframes_produces_four_stages(self):
        stages = self._expand(_PIPELINE_WITH_MOTION)
        assert len(stages) == 4
        assert [s.stage_type for s in stages] == ["compute", "compute", "vertex", "fragment"]

    def test_morph_stage_is_first_and_tagged(self):
        stages = self._expand(_PIPELINE_WITH_MOTION)
        morph = stages[0]
        assert morph.stage_type == "compute"
        assert getattr(morph, "_output_stem_suffix", None) == "morph"
        assert getattr(morph, "_splat_motion_config", None) is not None

    def test_morph_stage_buffers(self):
        stages = self._expand(_PIPELINE_WITH_MOTION)
        morph = stages[0]
        sb_names = {sb.name for sb in morph.storage_buffers}
        assert {"splat_base_pos", "splat_base_rot", "splat_base_sh0"} <= sb_names
        assert {"morph_index", "morph_delta_pos_lo", "morph_delta_rot_lo",
                "morph_delta_sh0_lo", "morph_delta_pos_hi", "morph_delta_rot_hi",
                "morph_delta_sh0_hi"} <= sb_names
        # Output buffers are exactly the names the preprocess stage reads as input.
        assert {"splat_pos", "splat_rot", "splat_sh0"} <= sb_names

    def test_morph_stage_push_constants(self):
        stages = self._expand(_PIPELINE_WITH_MOTION)
        morph = stages[0]
        assert len(morph.push_constants) == 1
        field_names = {f.name for f in morph.push_constants[0].fields}
        assert field_names == {"segment_offset", "segment_count", "weight_lo", "weight_hi"}

    def test_preprocess_stage_unaffected_by_motion(self):
        """The preprocess stage's own buffers/push-constants must be identical
        whether or not motion: keyframes is set -- it's just fed by different
        upstream writers."""
        plain = self._expand(_PIPELINE_WITH_SPLAT)
        motion = self._expand(_PIPELINE_WITH_MOTION)
        preprocess_plain = plain[0]
        preprocess_motion = next(s for s in motion if s._splat_name == "GS" and
                                  getattr(s, "_splat_config", None) is not None)
        names_plain = [sb.name for sb in preprocess_plain.storage_buffers]
        names_motion = [sb.name for sb in preprocess_motion.storage_buffers]
        assert names_plain == names_motion


@requires_spirv_tools
class TestMorphApplyCompilation:
    """Full compile of a motion: keyframes splat pipeline."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_compile_produces_four_spv(self, tmp_path):
        _compile_splat(tmp_path, _PIPELINE_WITH_MOTION)
        assert (tmp_path / "test_splat.morph.comp.spv").exists()
        assert (tmp_path / "test_splat.comp.spv").exists()
        assert (tmp_path / "test_splat.vert.spv").exists()
        assert (tmp_path / "test_splat.frag.spv").exists()

    def test_morph_reflection_has_gaussian_splatting_morph(self, tmp_path):
        _compile_splat(tmp_path, _PIPELINE_WITH_MOTION)
        json_path = tmp_path / "test_splat.morph.comp.json"
        assert json_path.exists()
        meta = json.loads(json_path.read_text())
        assert "gaussian_splatting_morph" in meta
        gs = meta["gaussian_splatting_morph"]
        assert gs["role"] == "morph_apply"
        assert "splat_pos" in gs["output_buffers"]
        assert "splat_rot" in gs["output_buffers"]
        assert "splat_sh0" in gs["output_buffers"]

    def test_preprocess_reflection_reports_motion(self, tmp_path):
        _compile_splat(tmp_path, _PIPELINE_WITH_MOTION)
        json_path = tmp_path / "test_splat.comp.json"
        meta = json.loads(json_path.read_text())
        assert meta["gaussian_splatting"]["motion"] == "keyframes"

    def test_static_pipeline_reflection_motion_is_none(self, tmp_path):
        _compile_splat(tmp_path, _PIPELINE_WITH_SPLAT)
        json_path = tmp_path / "test_splat.comp.json"
        meta = json.loads(json_path.read_text())
        assert meta["gaussian_splatting"]["motion"] == "none"
