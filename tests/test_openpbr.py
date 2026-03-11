"""Tests for P22: OpenPBR Material Model Integration.

Validates:
- Parser correctly handles `import openpbr;` and _is_openpbr detection
- openpbr.lux stdlib functions parse without errors
- Surface expander generates correct OpenPBR main body for various layer configs
- Full end-to-end compilation to valid SPIR-V for all layer combinations
- Transitive imports (brdf, compositing) resolve through openpbr
- Edge cases: missing layers default correctly, non-OpenPBR shaders unaffected
"""

import json
import subprocess
import pytest
from pathlib import Path

from luxc.parser.tree_builder import parse_lux
from luxc.expansion.surface_expander import expand_surfaces, _is_openpbr
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


def _parse_and_resolve(source):
    """Parse source and resolve stdlib imports."""
    mod = parse_lux(source)
    _resolve_imports(mod)
    return mod


# ---------------------------------------------------------------------------
# Shared Lux source fragments
# ---------------------------------------------------------------------------

_STANDARD_GEOMETRY = """
geometry StandardMesh {
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
"""

_STANDARD_LIGHTING = """
lighting SceneLighting {
    samplerCube env_specular,
    samplerCube env_irradiance,
    sampler2d brdf_lut,

    properties Light {
        light_dir: vec3 = vec3(0.5, -0.8, 0.3),
        view_pos: vec3 = vec3(0.0, 1.5, 4.0),
    },

    layers [
        directional(direction: Light.light_dir,
                    color: vec3(1.0, 0.98, 0.95)),
        ibl(specular_map: env_specular, irradiance_map: env_irradiance,
            brdf_lut: brdf_lut),
    ]
}
"""

_STANDARD_LIGHTING_NO_IBL = """
lighting SimpleLighting {
    properties Light {
        light_dir: vec3 = vec3(0.5, -0.8, 0.3),
        view_pos: vec3 = vec3(0.0, 1.5, 4.0),
    },

    layers [
        directional(direction: Light.light_dir,
                    color: vec3(1.0, 0.98, 0.95)),
    ]
}
"""

_MULTI_LIGHT_LIGHTING = """
lighting MultiLighting {
    samplerCube env_specular,
    samplerCube env_irradiance,
    sampler2d brdf_lut,

    properties Light {
        view_pos: vec3 = vec3(0.0, 1.5, 4.0),
    },

    layers [
        multi_light(max_lights: 4),
        ibl(specular_map: env_specular, irradiance_map: env_irradiance,
            brdf_lut: brdf_lut),
    ]
}
"""

_OPENPBR_MINIMAL_SURFACE = """
import openpbr;

""" + _STANDARD_GEOMETRY + """

surface MinimalPBR {
    layers [
        base(color: vec3(0.8, 0.2, 0.1),
             metalness: 0.0,
             diffuse_roughness: 0.0,
             weight: 1.0),
    ]
}

pipeline MinimalForward {
    geometry: StandardMesh,
    surface: MinimalPBR,
}
"""

_OPENPBR_FULL_SURFACE = """
import openpbr;

""" + _STANDARD_GEOMETRY + _STANDARD_LIGHTING + """

surface FullOpenPBR {
    layers [
        base(color: vec3(0.8, 0.2, 0.1),
             metalness: 0.5,
             diffuse_roughness: 0.3,
             weight: 1.0),
        specular(weight: 1.0,
                 color: vec3(1.0),
                 roughness: 0.3,
                 ior: 1.5),
        transmission(weight: 0.2,
                     color: vec3(0.9, 0.6, 0.3),
                     depth: 0.5),
        coat(weight: 0.8,
             color: vec3(1.0),
             roughness: 0.1,
             ior: 1.6,
             darkening: 1.0),
        fuzz(weight: 0.3,
             color: vec3(0.5, 0.2, 0.1),
             roughness: 0.6),
        thin_film(weight: 0.2,
                  thickness: 0.4,
                  ior: 1.3),
        emission(luminance: 100.0,
                 color: vec3(1.0, 0.5, 0.0)),
    ]
}

pipeline FullForward {
    geometry: StandardMesh,
    surface: FullOpenPBR,
    lighting: SceneLighting,
}
"""


# ===========================================================================
# 1. Parser Tests
# ===========================================================================

class TestOpenPBRParser:
    """Verify parsing of import openpbr and _is_openpbr detection."""

    def test_import_openpbr_parsed(self):
        """import openpbr; is parsed as an ImportDecl."""
        mod = parse_lux("import openpbr;")
        assert len(mod.imports) == 1
        assert mod.imports[0].module_name == "openpbr"

    def test_openpbr_stdlib_parses(self):
        """openpbr.lux stdlib file parses without errors."""
        stdlib_path = Path(__file__).parent.parent / "luxc" / "stdlib" / "openpbr.lux"
        source = stdlib_path.read_text(encoding="utf-8")
        mod = parse_lux(source)
        # openpbr.lux imports brdf and compositing
        assert len(mod.imports) >= 2
        import_names = {imp.module_name for imp in mod.imports}
        assert "brdf" in import_names
        assert "compositing" in import_names
        # Should have many function definitions
        assert len(mod.functions) >= 10

    def test_is_openpbr_true_with_import(self):
        """_is_openpbr returns True when module imports openpbr."""
        mod = parse_lux("import openpbr;")
        assert _is_openpbr(mod) is True

    def test_is_openpbr_false_without_import(self):
        """_is_openpbr returns False when module does not import openpbr."""
        mod = parse_lux("import brdf;")
        assert _is_openpbr(mod) is False

    def test_is_openpbr_false_empty_module(self):
        """_is_openpbr returns False for a module with no imports."""
        mod = parse_lux("""
        fragment {
            out color: vec4;
            fn main() { color = vec4(1.0); }
        }
        """)
        assert _is_openpbr(mod) is False


# ===========================================================================
# 2. Stdlib Function Tests
# ===========================================================================

class TestOpenPBRStdlibFunctions:
    """Verify openpbr stdlib functions exist and have correct signatures."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def _get_resolved_functions(self):
        """Parse and resolve openpbr import, return dict of function names."""
        mod = _parse_and_resolve("import openpbr;")
        return {fn.name: fn for fn in mod.functions}

    def test_fresnel_dielectric_exists(self):
        """openpbr_fresnel_dielectric function is available after import."""
        fns = self._get_resolved_functions()
        assert "openpbr_fresnel_dielectric" in fns
        fn = fns["openpbr_fresnel_dielectric"]
        # Should have 2 parameters: cos_theta, eta
        assert len(fn.params) == 2

    def test_f82_tint_exists(self):
        """openpbr_f82_tint function is available after import."""
        fns = self._get_resolved_functions()
        assert "openpbr_f82_tint" in fns
        fn = fns["openpbr_f82_tint"]
        # Should have 3 parameters: cos_theta, f0, f82
        assert len(fn.params) == 3

    def test_eon_diffuse_exists(self):
        """openpbr_eon_diffuse function is available after import."""
        fns = self._get_resolved_functions()
        assert "openpbr_eon_diffuse" in fns
        fn = fns["openpbr_eon_diffuse"]
        # Should have 5 parameters: albedo, roughness, n_dot_l, n_dot_v, v_dot_l
        assert len(fn.params) == 5

    def test_coat_brdf_exists(self):
        """openpbr_coat_brdf function is available after import."""
        fns = self._get_resolved_functions()
        assert "openpbr_coat_brdf" in fns
        fn = fns["openpbr_coat_brdf"]
        # Should have 6 parameters: n, v, l, coat_weight, coat_roughness, coat_ior
        assert len(fn.params) == 6

    def test_openpbr_compose_exists(self):
        """openpbr_compose function is available after import."""
        fns = self._get_resolved_functions()
        assert "openpbr_compose" in fns
        fn = fns["openpbr_compose"]
        # openpbr_compose has many parameters (the full layer stack)
        assert len(fn.params) >= 20


# ===========================================================================
# 3. Expansion Tests
# ===========================================================================

class TestOpenPBRExpansion:
    """Verify surface expander generates correct OpenPBR main body."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def _get_frag_var_names(self, source, pipeline_name):
        """Parse, resolve, expand, and return variable names from fragment main."""
        mod = _parse_and_resolve(source)
        expand_surfaces(mod, pipeline_filter=pipeline_name)
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        main = frag.functions[0]
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
        return all_names

    def test_minimal_openpbr_expands(self):
        """Minimal OpenPBR surface (base only) produces expected variables."""
        var_names = self._get_frag_var_names(_OPENPBR_MINIMAL_SURFACE, "MinimalForward")
        # Must have core OpenPBR parameter variables
        assert "base_color" in var_names
        assert "base_metalness" in var_names
        assert "specular_roughness" in var_names
        assert "specular_ior" in var_names
        # Must call openpbr_compose (producing 'composed')
        assert "composed" in var_names

    def test_full_openpbr_expands_all_layers(self):
        """Full OpenPBR surface (all layers) produces all layer variables."""
        var_names = self._get_frag_var_names(_OPENPBR_FULL_SURFACE, "FullForward")
        # Base parameters
        assert "base_color" in var_names
        assert "base_metalness" in var_names
        assert "base_diffuse_roughness" in var_names
        assert "base_weight" in var_names
        # Specular parameters
        assert "specular_weight" in var_names
        assert "specular_color" in var_names
        assert "specular_roughness" in var_names
        assert "specular_ior" in var_names
        # Coat parameters
        assert "coat_weight" in var_names
        assert "coat_color" in var_names
        assert "coat_roughness" in var_names
        assert "coat_ior" in var_names
        assert "coat_darkening" in var_names
        # Fuzz parameters
        assert "fuzz_weight" in var_names
        assert "fuzz_color" in var_names
        assert "fuzz_roughness" in var_names
        # Thin film parameters
        assert "thin_film_weight" in var_names
        assert "thin_film_thickness" in var_names
        assert "thin_film_ior" in var_names
        # Transmission parameters
        assert "trans_weight" in var_names
        assert "trans_color" in var_names
        assert "trans_depth" in var_names
        # Emission parameters
        assert "emission_luminance" in var_names
        assert "emission_color" in var_names
        # Composition result
        assert "composed" in var_names

    def test_openpbr_with_ibl_generates_ambient(self):
        """OpenPBR with IBL lighting generates bl_ambient and ibl sampling vars."""
        var_names = self._get_frag_var_names(_OPENPBR_FULL_SURFACE, "FullForward")
        assert "bl_ambient" in var_names
        assert "prefiltered" in var_names
        assert "irradiance" in var_names
        assert "brdf_sample" in var_names
        assert "r" in var_names

    def test_openpbr_with_coat_ibl(self):
        """OpenPBR with coat + IBL generates coat IBL contribution."""
        var_names = self._get_frag_var_names(_OPENPBR_FULL_SURFACE, "FullForward")
        assert "bl_coat_ibl" in var_names
        assert "prefiltered_coat" in var_names

    def test_openpbr_with_multi_light(self):
        """OpenPBR with multi_light lighting mode expands correctly."""
        src = """
import openpbr;

""" + _STANDARD_GEOMETRY + _MULTI_LIGHT_LIGHTING + """

surface MultiLightPBR {
    layers [
        base(color: vec3(0.8), metalness: 0.0,
             diffuse_roughness: 0.0, weight: 1.0),
        specular(weight: 1.0, color: vec3(1.0), roughness: 0.3, ior: 1.5),
    ]
}

pipeline MultiForward {
    geometry: StandardMesh,
    surface: MultiLightPBR,
    lighting: MultiLighting,
}
"""
        var_names = self._get_frag_var_names(src, "MultiForward")
        # Multi-light should produce layer compatibility aliases
        assert "layer_albedo" in var_names
        assert "layer_roughness" in var_names
        assert "layer_metallic" in var_names

    def test_openpbr_without_coat_no_coat_ibl(self):
        """OpenPBR without coat layer does not generate prefiltered_coat."""
        src = """
import openpbr;

""" + _STANDARD_GEOMETRY + _STANDARD_LIGHTING + """

surface NCoatPBR {
    layers [
        base(color: vec3(0.8), metalness: 0.0,
             diffuse_roughness: 0.0, weight: 1.0),
        specular(weight: 1.0, color: vec3(1.0), roughness: 0.3, ior: 1.5),
    ]
}

pipeline NCoatForward {
    geometry: StandardMesh,
    surface: NCoatPBR,
    lighting: SceneLighting,
}
"""
        var_names = self._get_frag_var_names(src, "NCoatForward")
        # IBL should be present
        assert "bl_ambient" in var_names
        # But no coat IBL prefilter since there is no coat layer
        assert "prefiltered_coat" not in var_names

    def test_openpbr_layer_defaults(self):
        """Missing layers get correct default values (weight=0.0)."""
        # With only base layer, all optional layer weights should default to 0
        var_names = self._get_frag_var_names(_OPENPBR_MINIMAL_SURFACE, "MinimalForward")
        # Coat, fuzz, thin_film, transmission, emission should all have vars
        # but with default values (weight 0.0)
        assert "coat_weight" in var_names
        assert "fuzz_weight" in var_names
        assert "thin_film_weight" in var_names
        assert "trans_weight" in var_names
        assert "emission_luminance" in var_names

    def test_openpbr_direct_call_emitted(self):
        """OpenPBR single-light mode emits openpbr_direct call (direct_raw var)."""
        src = """
import openpbr;

""" + _STANDARD_GEOMETRY + _STANDARD_LIGHTING_NO_IBL + """

surface DirectPBR {
    layers [
        base(color: vec3(0.5), metalness: 0.0,
             diffuse_roughness: 0.0, weight: 1.0),
    ]
}

pipeline DirectForward {
    geometry: StandardMesh,
    surface: DirectPBR,
    lighting: SimpleLighting,
}
"""
        var_names = self._get_frag_var_names(src, "DirectForward")
        assert "direct_raw" in var_names
        assert "direct_lit" in var_names


# ===========================================================================
# 4. Compilation Tests (end-to-end SPIR-V)
# ===========================================================================

@requires_spirv_tools
class TestOpenPBRCompilation:
    """Full end-to-end compilation to valid SPIR-V."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_minimal_openpbr_compiles(self, tmp_path):
        """Minimal OpenPBR shader (base layer only) compiles to valid SPIR-V."""
        compile_source(_OPENPBR_MINIMAL_SURFACE, "openpbr_minimal", tmp_path,
                       validate=True)
        assert (tmp_path / "openpbr_minimal.vert.spv").exists()
        assert (tmp_path / "openpbr_minimal.frag.spv").exists()

    def test_full_openpbr_compiles(self, tmp_path):
        """Full OpenPBR shader (all layers + IBL) compiles to valid SPIR-V."""
        compile_source(_OPENPBR_FULL_SURFACE, "openpbr_full", tmp_path,
                       validate=True)
        assert (tmp_path / "openpbr_full.vert.spv").exists()
        assert (tmp_path / "openpbr_full.frag.spv").exists()

    def test_carpaint_example_compiles(self, tmp_path):
        """Car paint example compiles and validates."""
        src = (EXAMPLES / "openpbr_carpaint.lux").read_text()
        compile_source(src, "carpaint", tmp_path, validate=True,
                       source_dir=EXAMPLES)
        assert (tmp_path / "carpaint.vert.spv").exists()
        assert (tmp_path / "carpaint.frag.spv").exists()

    def test_velvet_example_compiles(self, tmp_path):
        """Velvet example compiles and validates."""
        src = (EXAMPLES / "openpbr_velvet.lux").read_text()
        compile_source(src, "velvet", tmp_path, validate=True,
                       source_dir=EXAMPLES)
        assert (tmp_path / "velvet.vert.spv").exists()
        assert (tmp_path / "velvet.frag.spv").exists()

    def test_glass_example_compiles(self, tmp_path):
        """Glass example compiles and validates."""
        src = (EXAMPLES / "openpbr_glass.lux").read_text()
        compile_source(src, "glass", tmp_path, validate=True,
                       source_dir=EXAMPLES)
        assert (tmp_path / "glass.vert.spv").exists()
        assert (tmp_path / "glass.frag.spv").exists()

    def test_openpbr_with_ibl_compiles(self, tmp_path):
        """OpenPBR with IBL lighting compiles to valid SPIR-V."""
        src = """
import openpbr;

""" + _STANDARD_GEOMETRY + _STANDARD_LIGHTING + """

surface IblPBR {
    layers [
        base(color: vec3(0.5, 0.3, 0.2), metalness: 0.0,
             diffuse_roughness: 0.2, weight: 1.0),
        specular(weight: 1.0, color: vec3(1.0), roughness: 0.4, ior: 1.5),
    ]
}

pipeline IblForward {
    geometry: StandardMesh,
    surface: IblPBR,
    lighting: SceneLighting,
}
"""
        compile_source(src, "openpbr_ibl", tmp_path, validate=True)
        assert (tmp_path / "openpbr_ibl.vert.spv").exists()
        assert (tmp_path / "openpbr_ibl.frag.spv").exists()

    def test_openpbr_with_multi_light_compiles(self, tmp_path):
        """OpenPBR with multi-light mode compiles to valid SPIR-V."""
        src = """
import openpbr;
import lighting;

""" + _STANDARD_GEOMETRY + _MULTI_LIGHT_LIGHTING + """

surface MultiPBR {
    layers [
        base(color: vec3(0.8), metalness: 0.0,
             diffuse_roughness: 0.0, weight: 1.0),
        specular(weight: 1.0, color: vec3(1.0), roughness: 0.3, ior: 1.5),
    ]
}

pipeline MultiForward {
    geometry: StandardMesh,
    surface: MultiPBR,
    lighting: MultiLighting,
}
"""
        compile_source(src, "openpbr_multi", tmp_path, validate=True)
        assert (tmp_path / "openpbr_multi.vert.spv").exists()
        assert (tmp_path / "openpbr_multi.frag.spv").exists()

    def test_openpbr_with_emission_compiles(self, tmp_path):
        """OpenPBR with emission layer compiles to valid SPIR-V."""
        src = """
import openpbr;

""" + _STANDARD_GEOMETRY + """

surface EmissivePBR {
    layers [
        base(color: vec3(0.1), metalness: 0.0,
             diffuse_roughness: 0.0, weight: 1.0),
        emission(luminance: 500.0,
                 color: vec3(1.0, 0.3, 0.0)),
    ]
}

pipeline EmissiveForward {
    geometry: StandardMesh,
    surface: EmissivePBR,
}
"""
        compile_source(src, "openpbr_emissive", tmp_path, validate=True)
        assert (tmp_path / "openpbr_emissive.vert.spv").exists()
        assert (tmp_path / "openpbr_emissive.frag.spv").exists()

    def test_openpbr_all_layers_compiles(self, tmp_path):
        """OpenPBR with every layer type compiles to valid SPIR-V."""
        compile_source(_OPENPBR_FULL_SURFACE, "openpbr_all", tmp_path,
                       validate=True)
        assert (tmp_path / "openpbr_all.frag.spv").exists()

    def test_reflection_json_contains_uniforms(self, tmp_path):
        """Reflection JSON for OpenPBR shader includes expected metadata."""
        compile_source(_OPENPBR_FULL_SURFACE, "openpbr_refl", tmp_path,
                       validate=True, emit_reflection=True)
        json_path = tmp_path / "openpbr_refl.frag.json"
        assert json_path.exists()
        reflection = json.loads(json_path.read_text(encoding="utf-8"))
        assert reflection["stage"] == "fragment"
        # Should have outputs
        assert "outputs" in reflection

    def test_fragment_shader_has_color_output(self, tmp_path):
        """OpenPBR fragment shader declares vec4 color output."""
        compile_source(_OPENPBR_MINIMAL_SURFACE, "openpbr_out", tmp_path,
                       validate=True, emit_reflection=True)
        json_path = tmp_path / "openpbr_out.frag.json"
        assert json_path.exists()
        reflection = json.loads(json_path.read_text(encoding="utf-8"))
        assert reflection["stage"] == "fragment"
        # Check that there is at least one output
        outputs = reflection.get("outputs", [])
        assert len(outputs) >= 1
        # The first output should be vec4 color
        output_names = [o.get("name", "") for o in outputs]
        assert "color" in output_names


# ===========================================================================
# 5. Transitive Import Tests
# ===========================================================================

class TestOpenPBRTransitiveImports:
    """Verify import openpbr transitively resolves brdf and compositing."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_brdf_functions_available(self):
        """import openpbr; makes brdf functions (ggx_ndf, etc.) available."""
        mod = _parse_and_resolve("import openpbr;")
        fn_names = {fn.name for fn in mod.functions}
        # brdf.lux functions should be resolved transitively
        assert "ggx_ndf" in fn_names
        assert "v_ggx_correlated" in fn_names

    def test_compositing_functions_available(self):
        """import openpbr; makes compositing functions available."""
        mod = _parse_and_resolve("import openpbr;")
        fn_names = {fn.name for fn in mod.functions}
        # compositing.lux functions should be resolved transitively
        assert "volume_attenuation" in fn_names

    def test_openpbr_functions_available(self):
        """import openpbr; makes all openpbr_* functions available."""
        mod = _parse_and_resolve("import openpbr;")
        fn_names = {fn.name for fn in mod.functions}
        expected = [
            "openpbr_fresnel_dielectric",
            "openpbr_fresnel_avg",
            "openpbr_specular_ior_modulated",
            "openpbr_fresnel_modulated",
            "openpbr_f82_tint",
            "openpbr_coat_roughening",
            "openpbr_coat_absorption",
            "openpbr_coat_darkening",
            "openpbr_coat_brdf",
            "openpbr_eon_albedo",
            "openpbr_eon_diffuse",
            "openpbr_specular_albedo",
            "openpbr_fuzz_brdf",
            "openpbr_direct",
            "openpbr_compose",
        ]
        for name in expected:
            assert name in fn_names, f"Missing function: {name}"


# ===========================================================================
# 6. Edge Case Tests
# ===========================================================================

class TestOpenPBREdgeCases:
    """Edge cases: defaults, unknown layers, non-OpenPBR shaders."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_missing_layers_use_defaults(self):
        """OpenPBR surface with only base layer gets default values for all others."""
        mod = _parse_and_resolve(_OPENPBR_MINIMAL_SURFACE)
        expand_surfaces(mod, pipeline_filter="MinimalForward")
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        main = frag.functions[0]

        # Collect all LetStmt name -> value pairs
        let_stmts = {}
        for stmt in main.body:
            if hasattr(stmt, 'name') and hasattr(stmt, 'value'):
                let_stmts[stmt.name] = stmt.value

        # coat_weight, fuzz_weight, trans_weight should all be 0.0 (default)
        from luxc.parser.ast_nodes import NumberLit
        assert "coat_weight" in let_stmts
        coat_val = let_stmts["coat_weight"]
        assert isinstance(coat_val, NumberLit) and coat_val.value == "0.0"

        assert "fuzz_weight" in let_stmts
        fuzz_val = let_stmts["fuzz_weight"]
        assert isinstance(fuzz_val, NumberLit) and fuzz_val.value == "0.0"

        assert "trans_weight" in let_stmts
        trans_val = let_stmts["trans_weight"]
        assert isinstance(trans_val, NumberLit) and trans_val.value == "0.0"

        assert "emission_luminance" in let_stmts
        em_val = let_stmts["emission_luminance"]
        assert isinstance(em_val, NumberLit) and em_val.value == "0.0"

    def test_non_openpbr_shader_unaffected(self):
        """Non-OpenPBR layered surface uses standard gltf_pbr path, not openpbr_direct."""
        src = """
import brdf;
import compositing;
import ibl;

""" + _STANDARD_GEOMETRY + """

surface StandardPBR {
    samplerCube env_specular,
    samplerCube env_irradiance,
    sampler2d brdf_lut,

    layers [
        base(albedo: vec3(0.8, 0.2, 0.1), roughness: 0.5, metallic: 0.0),
        ibl(specular_map: env_specular, irradiance_map: env_irradiance,
            brdf_lut: brdf_lut),
    ]
}

pipeline StdForward {
    mode: forward,
    geometry: StandardMesh,
    surface: StandardPBR,
}
"""
        mod = _parse_and_resolve(src)
        expand_surfaces(mod, pipeline_filter="StdForward")
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        main = frag.functions[0]
        var_names = {s.name for s in main.body if hasattr(s, 'name')}

        # Standard path produces 'composed' from compose_pbr_layers, NOT openpbr_compose
        # It should NOT have openpbr-specific variables
        assert "base_metalness" not in var_names
        assert "specular_ior" not in var_names
        assert "fuzz_weight" not in var_names
        # It should have standard layer variables
        assert "bl_ambient" in var_names

    def test_openpbr_specular_defaults(self):
        """Specular layer defaults: weight=1.0, roughness=0.3, ior=1.5."""
        mod = _parse_and_resolve(_OPENPBR_MINIMAL_SURFACE)
        expand_surfaces(mod, pipeline_filter="MinimalForward")
        frag = [s for s in mod.stages if s.stage_type == "fragment"][0]
        main = frag.functions[0]

        let_stmts = {}
        for stmt in main.body:
            if hasattr(stmt, 'name') and hasattr(stmt, 'value'):
                let_stmts[stmt.name] = stmt.value

        from luxc.parser.ast_nodes import NumberLit
        # Specular weight defaults to 1.0
        assert "specular_weight" in let_stmts
        sw = let_stmts["specular_weight"]
        assert isinstance(sw, NumberLit) and sw.value == "1.0"

        # Specular roughness defaults to 0.3
        assert "specular_roughness" in let_stmts
        sr = let_stmts["specular_roughness"]
        assert isinstance(sr, NumberLit) and sr.value == "0.3"

        # Specular IOR defaults to 1.5
        assert "specular_ior" in let_stmts
        si = let_stmts["specular_ior"]
        assert isinstance(si, NumberLit) and si.value == "1.5"
