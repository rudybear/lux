"""RT Gaussian splatting (3DGRT) pipeline tests.

Tests cover:
- RT splat pipeline expansion (splat + raytrace mode → 4 RT stages)
- Stage content verification (storage buffers, payloads, hit attributes)
- SH degree variations (0-3)
- Full compilation to SPIR-V assembly
- Error handling (missing splat reference)
- Reflection JSON metadata
"""

import json
import pytest
from pathlib import Path
from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import (
    StageBlock, StorageBufferDecl, RayPayloadDecl, HitAttributeDecl,
    AccelDecl, StorageImageDecl, ForStmt,
)
from luxc.expansion.splat_expander import expand_splat_rt_pipeline, _get_splat_config
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

_RT_SPLAT_SH0 = """
splat GS {
    sh_degree: 0,
}
pipeline RTSplat {
    mode: raytrace,
    splat: GS,
}
"""

_RT_SPLAT_SH1 = """
splat GS {
    sh_degree: 1,
}
pipeline RTSplat {
    mode: raytrace,
    splat: GS,
}
"""

_RT_SPLAT_SH2 = """
splat GS {
    sh_degree: 2,
}
pipeline RTSplat {
    mode: raytrace,
    splat: GS,
}
"""

_RT_SPLAT_SH3 = """
splat GS {
    sh_degree: 3,
    kernel: ellipse,
    color_space: srgb,
    alpha_cutoff: 0.004,
}
pipeline RTSplat {
    mode: raytrace,
    splat: GS,
}
"""

_RT_SPLAT_CUSTOM_CUTOFF = """
splat GS {
    sh_degree: 0,
    alpha_cutoff: 0.01,
}
pipeline RTSplat {
    mode: raytrace,
    splat: GS,
}
"""

_RT_SPLAT_MISSING = """
splat GS {
    sh_degree: 0,
}
pipeline RTSplat {
    mode: raytrace,
    splat: NonExistent,
}
"""


# ---------------------------------------------------------------------------
# Expansion tests
# ---------------------------------------------------------------------------

class TestRTSplatExpansion:
    """Test that RT splat pipelines expand into the correct 4 stages."""

    def test_basic_expansion_produces_4_stages(self):
        module = parse_lux(_RT_SPLAT_SH0)
        expand_surfaces(module)
        assert len(module.stages) == 4

    def test_stage_types(self):
        module = parse_lux(_RT_SPLAT_SH0)
        expand_surfaces(module)
        types = [s.stage_type for s in module.stages]
        assert types == ["intersection", "closest_hit", "miss", "raygen"]

    def test_intersection_stage_has_splat_ssbos(self):
        module = parse_lux(_RT_SPLAT_SH0)
        expand_surfaces(module)
        isect = module.stages[0]
        ssbo_names = [sb.name for sb in isect.storage_buffers]
        assert "splat_pos" in ssbo_names
        assert "splat_rot" in ssbo_names
        assert "splat_scale" in ssbo_names
        assert "splat_opacity" in ssbo_names

    def test_intersection_stage_has_hit_attribute(self):
        module = parse_lux(_RT_SPLAT_SH0)
        expand_surfaces(module)
        isect = module.stages[0]
        assert len(isect.hit_attributes) == 1
        assert isect.hit_attributes[0].name == "hit_alpha"
        assert isect.hit_attributes[0].type_name == "scalar"

    def test_closest_hit_has_payload_and_hit_attr(self):
        module = parse_lux(_RT_SPLAT_SH0)
        expand_surfaces(module)
        chit = module.stages[1]
        assert len(chit.ray_payloads) == 1
        assert chit.ray_payloads[0].name == "payload"
        assert chit.ray_payloads[0].type_name == "vec4"
        assert len(chit.hit_attributes) == 1
        assert chit.hit_attributes[0].name == "hit_alpha"

    def test_miss_has_payload(self):
        module = parse_lux(_RT_SPLAT_SH0)
        expand_surfaces(module)
        miss = module.stages[2]
        assert len(miss.ray_payloads) == 1
        assert miss.ray_payloads[0].name == "payload"

    def test_raygen_has_accel_struct(self):
        module = parse_lux(_RT_SPLAT_SH0)
        expand_surfaces(module)
        rgen = module.stages[3]
        assert len(rgen.accel_structs) == 1
        assert rgen.accel_structs[0].name == "tlas"

    def test_raygen_has_camera_uniform(self):
        module = parse_lux(_RT_SPLAT_SH0)
        expand_surfaces(module)
        rgen = module.stages[3]
        assert len(rgen.uniforms) == 1
        assert rgen.uniforms[0].name == "Camera"
        field_names = [f.name for f in rgen.uniforms[0].fields]
        assert "inv_view" in field_names
        assert "inv_proj" in field_names

    def test_raygen_has_storage_image(self):
        module = parse_lux(_RT_SPLAT_SH0)
        expand_surfaces(module)
        rgen = module.stages[3]
        assert len(rgen.storage_images) == 1
        assert rgen.storage_images[0].name == "result_color"

    def test_raygen_has_payload(self):
        module = parse_lux(_RT_SPLAT_SH0)
        expand_surfaces(module)
        rgen = module.stages[3]
        assert len(rgen.ray_payloads) == 1
        assert rgen.ray_payloads[0].name == "payload"

    def test_raygen_has_for_loop(self):
        module = parse_lux(_RT_SPLAT_SH0)
        expand_surfaces(module)
        rgen = module.stages[3]
        main_fn = rgen.functions[0]
        for_stmts = [s for s in main_fn.body if isinstance(s, ForStmt)]
        assert len(for_stmts) == 1

    def test_splat_config_tagged(self):
        module = parse_lux(_RT_SPLAT_SH3)
        expand_surfaces(module)
        for stage in module.stages:
            assert hasattr(stage, '_splat_config')
            assert stage._splat_config["sh_degree"] == 3
            assert hasattr(stage, '_splat_name')
            assert stage._splat_name == "GS"


class TestRTSplatSHDegrees:
    """Test SH degree variations in RT splat pipeline."""

    def test_sh0_raygen_buffers(self):
        module = parse_lux(_RT_SPLAT_SH0)
        expand_surfaces(module)
        rgen = module.stages[3]
        ssbo_names = [sb.name for sb in rgen.storage_buffers]
        assert "splat_sh0" in ssbo_names
        assert "splat_sh1" not in ssbo_names

    def test_sh1_raygen_buffers(self):
        module = parse_lux(_RT_SPLAT_SH1)
        expand_surfaces(module)
        rgen = module.stages[3]
        ssbo_names = [sb.name for sb in rgen.storage_buffers]
        assert "splat_sh0" in ssbo_names
        assert "splat_sh3" in ssbo_names
        assert "splat_sh4" not in ssbo_names

    def test_sh2_raygen_buffers(self):
        module = parse_lux(_RT_SPLAT_SH2)
        expand_surfaces(module)
        rgen = module.stages[3]
        ssbo_names = [sb.name for sb in rgen.storage_buffers]
        assert "splat_sh8" in ssbo_names
        assert "splat_sh9" not in ssbo_names

    def test_sh3_raygen_buffers(self):
        module = parse_lux(_RT_SPLAT_SH3)
        expand_surfaces(module)
        rgen = module.stages[3]
        ssbo_names = [sb.name for sb in rgen.storage_buffers]
        assert "splat_sh0" in ssbo_names
        assert "splat_sh15" in ssbo_names
        # Also has splat_pos for SH direction
        assert "splat_pos" in ssbo_names

    def test_sh0_no_direction_vars(self):
        """SH degree 0 doesn't need direction components."""
        module = parse_lux(_RT_SPLAT_SH0)
        expand_surfaces(module)
        rgen = module.stages[3]
        main_fn = rgen.functions[0]
        # Find the ForStmt and check its body
        for_stmt = [s for s in main_fn.body if isinstance(s, ForStmt)][0]
        var_names = [s.name for s in for_stmt.body
                     if hasattr(s, 'name') and isinstance(s.name, str)]
        assert "dx" not in var_names


class TestRTSplatErrors:
    """Test error handling for RT splat pipelines."""

    def test_missing_splat_reference(self):
        module = parse_lux(_RT_SPLAT_MISSING)
        with pytest.raises(ValueError, match="no matching splat declaration found"):
            expand_surfaces(module)

    def test_invalid_sh_degree(self):
        """Direct call with invalid SH degree."""
        from luxc.parser.ast_nodes import SplatDecl, SplatMember, NumberLit, PipelineDecl, Module
        splat = SplatDecl("Bad", [SplatMember("sh_degree", NumberLit("5"))])
        pipeline = PipelineDecl("P", [])
        module = Module()
        with pytest.raises(ValueError, match="Unsupported SH degree"):
            expand_splat_rt_pipeline(splat, pipeline, module)


class TestRTSplatCustomConfig:
    """Test that custom config values propagate correctly."""

    def test_custom_alpha_cutoff(self):
        module = parse_lux(_RT_SPLAT_CUSTOM_CUTOFF)
        expand_surfaces(module)
        isect = module.stages[0]
        assert isect._splat_config["alpha_cutoff"] == 0.01


# ---------------------------------------------------------------------------
# Compilation tests
# ---------------------------------------------------------------------------

def _compile_rt_splat(tmp_path, source: str, stem="test_rt_splat"):
    """Compile RT splat source via compile_source."""
    from luxc.compiler import compile_source
    clear_type_aliases()
    compile_source(source, stem, tmp_path, emit_reflection=True, validate=True)


def _get_spv_asm(tmp_path, source: str, stem="test_rt_splat") -> dict:
    """Compile and return {stage_type: spv_asm_text} by reading .spvasm files."""
    from luxc.compiler import compile_source
    clear_type_aliases()
    compile_source(source, stem, tmp_path, emit_asm=True, validate=True)
    # Map file extension to stage type
    ext_to_stage = {
        ".rint": "intersection",
        ".rchit": "closest_hit",
        ".rmiss": "miss",
        ".rgen": "raygen",
    }
    result = {}
    for f in tmp_path.iterdir():
        if f.suffix == ".spvasm":
            # File names like: test_rt_splat.rint.spvasm
            # Extract the stage extension from the double-suffix
            base = f.stem  # test_rt_splat.rint
            if "." in base:
                stage_ext = "." + base.rsplit(".", 1)[1]
                stage_type = ext_to_stage.get(stage_ext)
                if stage_type:
                    result[stage_type] = f.read_text()
    return result


@requires_spirv_tools
class TestRTSplatCompilation:
    """End-to-end compilation tests for RT Gaussian splatting."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_sh0_compiles(self, tmp_path):
        _compile_rt_splat(tmp_path, _RT_SPLAT_SH0)

    def test_sh1_compiles(self, tmp_path):
        _compile_rt_splat(tmp_path, _RT_SPLAT_SH1)

    def test_sh2_compiles(self, tmp_path):
        _compile_rt_splat(tmp_path, _RT_SPLAT_SH2)

    def test_sh3_compiles(self, tmp_path):
        _compile_rt_splat(tmp_path, _RT_SPLAT_SH3)

    def test_intersection_has_report(self, tmp_path):
        result = _get_spv_asm(tmp_path, _RT_SPLAT_SH0)
        assert "OpReportIntersectionKHR" in result["intersection"]

    def test_raygen_has_trace(self, tmp_path):
        result = _get_spv_asm(tmp_path, _RT_SPLAT_SH0)
        assert "OpTraceRayKHR" in result["raygen"]

    def test_raygen_has_image_store(self, tmp_path):
        result = _get_spv_asm(tmp_path, _RT_SPLAT_SH0)
        assert "OpImageWrite" in result["raygen"]

    def test_raygen_has_loop(self, tmp_path):
        result = _get_spv_asm(tmp_path, _RT_SPLAT_SH0)
        assert "OpLoopMerge" in result["raygen"]

    def test_miss_writes_zero_payload(self, tmp_path):
        result = _get_spv_asm(tmp_path, _RT_SPLAT_SH0)
        assert "OpStore" in result["miss"]


class TestRTSplatReflection:
    """Test reflection JSON for RT splat stages."""

    def test_raygen_reflection_has_rt_metadata(self):
        from luxc.codegen.reflection import generate_reflection

        module = parse_lux(_RT_SPLAT_SH3)
        expand_surfaces(module)

        rgen = module.stages[3]
        assert rgen.stage_type == "raygen"

        refl = generate_reflection(module, rgen)
        assert "gaussian_splatting_rt" in refl
        gs_rt = refl["gaussian_splatting_rt"]
        assert gs_rt["sh_degree"] == 3
        assert gs_rt["splat_name"] == "GS"
        assert gs_rt["alpha_cutoff"] == 0.004

    def test_intersection_reflection_has_rt_metadata(self):
        from luxc.codegen.reflection import generate_reflection

        module = parse_lux(_RT_SPLAT_SH0)
        expand_surfaces(module)

        isect = module.stages[0]
        refl = generate_reflection(module, isect)
        assert "gaussian_splatting_rt" in refl

    def test_raygen_reflection_lists_input_buffers(self):
        from luxc.codegen.reflection import generate_reflection

        module = parse_lux(_RT_SPLAT_SH3)
        expand_surfaces(module)

        rgen = module.stages[3]
        refl = generate_reflection(module, rgen)
        gs_rt = refl["gaussian_splatting_rt"]
        assert "input_buffers" in gs_rt
        assert "splat_sh0" in gs_rt["input_buffers"]
        assert "splat_sh15" in gs_rt["input_buffers"]
        assert "splat_pos" in gs_rt["input_buffers"]


# ---------------------------------------------------------------------------
# SPIR-V validation tests (compile_source already runs spirv-val)
# ---------------------------------------------------------------------------

@requires_spirv_tools
class TestRTSplatSPIRVValidation:
    """Validate generated SPIR-V with spirv-val via compile_source."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_validate_sh0(self, tmp_path):
        _compile_rt_splat(tmp_path, _RT_SPLAT_SH0)

    def test_validate_sh3(self, tmp_path):
        _compile_rt_splat(tmp_path, _RT_SPLAT_SH3)
