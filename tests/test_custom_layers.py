"""Tests for custom @layer functions (P8).

Covers parsing, validation, compilation, reflection, and integration.
"""

import json
import subprocess
import pytest
from pathlib import Path

from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import Module, FunctionDef, Param
from luxc.compiler import compile_source
from luxc.builtins.types import clear_type_aliases
from luxc.expansion.surface_expander import (
    _collect_layer_functions,
    _BUILTIN_LAYER_NAMES,
)

EXAMPLES = Path(__file__).parent.parent / "examples"
STDLIB = Path(__file__).parent.parent / "luxc" / "stdlib"


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
# Parsing tests
# ---------------------------------------------------------------------------

class TestLayerAttributeParsing:
    def test_layer_attribute_parsed(self):
        """@layer appears in FunctionDef.attributes."""
        src = """
        @layer
        fn my_effect(base: vec3, n: vec3, v: vec3, l: vec3,
                     strength: scalar) -> vec3 {
            return base * strength;
        }
        """
        m = parse_lux(src)
        assert len(m.functions) == 1
        assert "layer" in m.functions[0].attributes

    def test_layer_attribute_coexists_with_others(self):
        """@layer can coexist with other attributes."""
        src = """
        @differentiable
        @layer
        fn my_effect(base: vec3, n: vec3, v: vec3, l: vec3,
                     strength: scalar) -> vec3 {
            return base * strength;
        }
        """
        m = parse_lux(src)
        assert "layer" in m.functions[0].attributes
        assert "differentiable" in m.functions[0].attributes


# ---------------------------------------------------------------------------
# Validation tests (_collect_layer_functions)
# ---------------------------------------------------------------------------

class TestLayerValidation:
    def test_valid_layer_collected(self):
        """A properly formed @layer function is collected."""
        src = """
        @layer
        fn my_effect(base: vec3, n: vec3, v: vec3, l: vec3,
                     strength: scalar) -> vec3 {
            return base * strength;
        }
        """
        m = parse_lux(src)
        fns = _collect_layer_functions(m)
        assert "my_effect" in fns

    def test_layer_too_few_params_rejected(self):
        """@layer function with <4 params raises ValueError."""
        src = """
        @layer
        fn bad_layer(base: vec3, n: vec3) -> vec3 {
            return base;
        }
        """
        m = parse_lux(src)
        with pytest.raises(ValueError, match="needs ≥4 params"):
            _collect_layer_functions(m)

    def test_layer_wrong_return_type_rejected(self):
        """@layer function with non-vec3 return raises ValueError."""
        src = """
        @layer
        fn bad_layer(base: vec3, n: vec3, v: vec3, l: vec3) -> scalar {
            return 1.0;
        }
        """
        m = parse_lux(src)
        with pytest.raises(ValueError, match="must return vec3"):
            _collect_layer_functions(m)

    def test_layer_builtin_name_rejected(self):
        """@layer function named 'base' conflicts with built-in layer."""
        src = """
        @layer
        fn base(base: vec3, n: vec3, v: vec3, l: vec3) -> vec3 {
            return base;
        }
        """
        m = parse_lux(src)
        with pytest.raises(ValueError, match="conflicts with built-in layer"):
            _collect_layer_functions(m)

    def test_all_builtin_names_rejected(self):
        """All built-in layer names are rejected."""
        for name in _BUILTIN_LAYER_NAMES:
            fn = FunctionDef(
                name=name,
                params=[
                    Param("base", "vec3"),
                    Param("n", "vec3"),
                    Param("v", "vec3"),
                    Param("l", "vec3"),
                ],
                return_type="vec3",
                body=[],
                attributes=["layer"],
            )
            module = Module(functions=[fn])
            with pytest.raises(ValueError, match="conflicts with built-in layer"):
                _collect_layer_functions(module)

    def test_non_layer_functions_ignored(self):
        """Functions without @layer are not collected."""
        src = """
        fn helper(a: vec3) -> vec3 {
            return a;
        }
        @layer
        fn my_effect(base: vec3, n: vec3, v: vec3, l: vec3) -> vec3 {
            return base;
        }
        """
        m = parse_lux(src)
        fns = _collect_layer_functions(m)
        assert "helper" not in fns
        assert "my_effect" in fns


# ---------------------------------------------------------------------------
# Compilation tests (SPIR-V output)
# ---------------------------------------------------------------------------

@requires_spirv_tools
class TestCustomLayerCompilation:
    def test_cartoon_forward_compiles(self, tmp_path):
        """cartoon_toon.lux compiles to .vert.spv + .frag.spv with spirv-val."""
        clear_type_aliases()
        src = (EXAMPLES / "cartoon_toon.lux").read_text()
        compile_source(
            src, "cartoon_toon", tmp_path,
            source_dir=EXAMPLES,
            pipeline="ToonForward",
            validate=True,
        )
        assert (tmp_path / "cartoon_toon.vert.spv").exists()
        assert (tmp_path / "cartoon_toon.frag.spv").exists()

    def test_cartoon_reflection(self, tmp_path):
        """Reflection JSON includes albedo_tex sampler."""
        clear_type_aliases()
        src = (EXAMPLES / "cartoon_toon.lux").read_text()
        compile_source(
            src, "cartoon_toon", tmp_path,
            source_dir=EXAMPLES,
            pipeline="ToonForward",
            validate=True,
        )
        refl = json.loads(
            (tmp_path / "cartoon_toon.frag.json").read_text()
        )
        samplers = {
            b["name"]
            for bindings in refl["descriptor_sets"].values()
            for b in bindings
            if b.get("type") == "sampler"
        }
        assert "albedo_tex" in samplers

    def test_inline_custom_layer_compiles(self, tmp_path):
        """An inline @layer function (not from import) compiles."""
        clear_type_aliases()
        src = """
        import brdf;
        import color;

        @layer
        fn tint(base: vec3, n: vec3, v: vec3, l: vec3,
                color_mul: vec3) -> vec3 {
            return base * color_mul;
        }

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

        surface TintedSurface {
            layers [
                base(albedo: vec3(0.5), roughness: 0.5, metallic: 0.0),
                tint(color_mul: vec3(1.0, 0.5, 0.5)),
            ]
        }

        schedule S { tonemap: aces }

        pipeline P {
            geometry: Quad,
            surface: TintedSurface,
            schedule: S,
        }
        """
        compile_source(src, "inline_custom", tmp_path, validate=True)
        assert (tmp_path / "inline_custom.frag.spv").exists()

    def test_multiple_custom_layers(self, tmp_path):
        """Two @layer functions compose in declaration order."""
        clear_type_aliases()
        src = """
        import brdf;
        import color;

        @layer
        fn tint(base: vec3, n: vec3, v: vec3, l: vec3,
                color_mul: vec3) -> vec3 {
            return base * color_mul;
        }

        @layer
        fn boost(base: vec3, n: vec3, v: vec3, l: vec3,
                 factor: scalar) -> vec3 {
            return base * factor;
        }

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

        surface MultiLayered {
            layers [
                base(albedo: vec3(0.5), roughness: 0.5, metallic: 0.0),
                tint(color_mul: vec3(1.0, 0.5, 0.5)),
                boost(factor: 2.0),
            ]
        }

        schedule S { tonemap: aces }

        pipeline P {
            geometry: Quad,
            surface: MultiLayered,
            schedule: S,
        }
        """
        compile_source(src, "multi_custom", tmp_path, validate=True)
        assert (tmp_path / "multi_custom.frag.spv").exists()

    def test_custom_layer_missing_arg(self, tmp_path):
        """Missing layer arg raises ValueError at expansion time."""
        clear_type_aliases()
        src = """
        import brdf;
        import color;

        @layer
        fn tint(base: vec3, n: vec3, v: vec3, l: vec3,
                color_mul: vec3, amount: scalar) -> vec3 {
            return base * color_mul * amount;
        }

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

        surface BadSurface {
            layers [
                base(albedo: vec3(0.5), roughness: 0.5, metallic: 0.0),
                tint(color_mul: vec3(1.0, 0.5, 0.5)),
            ]
        }

        schedule S { tonemap: aces }

        pipeline P {
            geometry: Quad,
            surface: BadSurface,
            schedule: S,
        }
        """
        with pytest.raises(ValueError, match="missing arg 'amount'"):
            compile_source(src, "missing_arg", tmp_path, validate=False)

    def test_custom_layer_with_emission(self, tmp_path):
        """Custom layer works alongside emission (custom before emission)."""
        clear_type_aliases()
        src = """
        import brdf;
        import color;

        @layer
        fn tint(base: vec3, n: vec3, v: vec3, l: vec3,
                color_mul: vec3) -> vec3 {
            return base * color_mul;
        }

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

        surface EmitSurface {
            layers [
                base(albedo: vec3(0.5), roughness: 0.5, metallic: 0.0),
                tint(color_mul: vec3(1.0, 0.8, 0.6)),
                emission(color: vec3(0.1, 0.0, 0.0)),
            ]
        }

        schedule S { tonemap: aces }

        pipeline P {
            geometry: Quad,
            surface: EmitSurface,
            schedule: S,
        }
        """
        compile_source(src, "custom_with_emission", tmp_path, validate=True)
        assert (tmp_path / "custom_with_emission.frag.spv").exists()

    def test_cartoon_with_features(self, tmp_path):
        """Custom layer with 'if has_feature' condition compiles."""
        clear_type_aliases()
        src = """
        import brdf;
        import color;
        import toon;

        features {
            has_toon: bool,
        }

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

        surface ConditionalToon {
            layers [
                base(albedo: vec3(0.5), roughness: 0.5, metallic: 0.0),
                cartoon(bands: 4.0, rim_power: 3.0, rim_color: vec3(0.3, 0.3, 0.5)) if has_toon,
            ]
        }

        schedule S { tonemap: aces }

        pipeline P {
            geometry: Quad,
            surface: ConditionalToon,
            schedule: S,
        }
        """
        # Compile with has_toon enabled — cartoon layer included
        compile_source(
            src, "cond_toon", tmp_path,
            source_dir=EXAMPLES,
            features={"has_toon"},
            validate=True,
        )
        assert (tmp_path / "cond_toon+toon.frag.spv").exists()

        # Compile without has_toon — cartoon layer stripped
        clear_type_aliases()
        compile_source(
            src, "cond_toon_no", tmp_path,
            source_dir=EXAMPLES,
            features=set(),
            validate=True,
        )
        assert (tmp_path / "cond_toon_no.frag.spv").exists()
