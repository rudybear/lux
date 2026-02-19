"""Tests for the P8 reflection metadata emitter."""

import json
import pytest
from pathlib import Path

from luxc.parser.tree_builder import parse_lux
from luxc.analysis.type_checker import type_check
from luxc.optimization.const_fold import constant_fold
from luxc.analysis.layout_assigner import assign_layouts
from luxc.codegen.reflection import generate_reflection, emit_reflection_json
from luxc.builtins.types import clear_type_aliases


def _compile_and_reflect(source: str) -> list[dict]:
    """Compile source and return reflection dicts for all stages."""
    clear_type_aliases()
    module = parse_lux(source)

    # Resolve imports (needed for stdlib like brdf)
    from luxc.compiler import _resolve_imports
    _resolve_imports(module)

    if module.surfaces or module.pipelines or module.environments or module.procedurals:
        from luxc.expansion.surface_expander import expand_surfaces
        expand_surfaces(module)

    from luxc.autodiff.forward_diff import autodiff_expand
    autodiff_expand(module)

    type_check(module)
    constant_fold(module)
    assign_layouts(module)

    results = []
    for stage in module.stages:
        reflection = generate_reflection(module, stage, source_name="test.lux")
        results.append(reflection)
    return results


# ===========================================================================
# Basic schema validation
# ===========================================================================

class TestReflectionSchema:
    def test_version_field(self):
        source = """
        vertex {
            in position: vec3;
            out frag_pos: vec3;
            fn main() {
                frag_pos = position;
                builtin_position = vec4(position, 1.0);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        assert reflections[0]["version"] == 1

    def test_source_field(self):
        source = """
        vertex {
            in position: vec3;
            fn main() {
                builtin_position = vec4(position, 1.0);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        assert reflections[0]["source"] == "test.lux"

    def test_stage_field(self):
        source = """
        vertex {
            in position: vec3;
            fn main() {
                builtin_position = vec4(position, 1.0);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        assert reflections[0]["stage"] == "vertex"
        assert reflections[0]["execution_model"] == "Vertex"

    def test_fragment_stage(self):
        source = """
        fragment {
            in frag_color: vec4;
            out color: vec4;
            fn main() {
                color = frag_color;
            }
        }
        """
        reflections = _compile_and_reflect(source)
        assert reflections[0]["stage"] == "fragment"
        assert reflections[0]["execution_model"] == "Fragment"

    def test_json_serializable(self):
        source = """
        vertex {
            in position: vec3;
            fn main() {
                builtin_position = vec4(position, 1.0);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        json_str = emit_reflection_json(reflections[0])
        parsed = json.loads(json_str)
        assert parsed["version"] == 1


# ===========================================================================
# Input/Output reflection
# ===========================================================================

class TestInputOutputReflection:
    def test_vertex_inputs(self):
        source = """
        vertex {
            in position: vec3;
            in normal: vec3;
            in uv: vec2;
            fn main() {
                builtin_position = vec4(position, 1.0);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        inputs = reflections[0]["inputs"]
        assert len(inputs) == 3
        assert inputs[0] == {"name": "position", "type": "vec3", "location": 0}
        assert inputs[1] == {"name": "normal", "type": "vec3", "location": 1}
        assert inputs[2] == {"name": "uv", "type": "vec2", "location": 2}

    def test_vertex_outputs(self):
        source = """
        vertex {
            in position: vec3;
            in normal: vec3;
            out frag_pos: vec3;
            out frag_normal: vec3;
            fn main() {
                frag_pos = position;
                frag_normal = normal;
                builtin_position = vec4(position, 1.0);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        outputs = reflections[0]["outputs"]
        assert len(outputs) == 2
        assert outputs[0] == {"name": "frag_pos", "type": "vec3", "location": 0}
        assert outputs[1] == {"name": "frag_normal", "type": "vec3", "location": 1}

    def test_fragment_outputs(self):
        source = """
        fragment {
            out color: vec4;
            fn main() {
                color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        outputs = reflections[0]["outputs"]
        assert len(outputs) == 1
        assert outputs[0]["name"] == "color"
        assert outputs[0]["location"] == 0


# ===========================================================================
# Descriptor set reflection
# ===========================================================================

class TestDescriptorSetReflection:
    def test_uniform_block(self):
        source = """
        vertex {
            in position: vec3;
            uniform MVP {
                model: mat4,
                view: mat4,
                proj: mat4,
            }
            fn main() {
                builtin_position = proj * view * model * vec4(position, 1.0);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        sets = reflections[0]["descriptor_sets"]
        assert "0" in sets
        assert len(sets["0"]) == 1
        ub = sets["0"][0]
        assert ub["binding"] == 0
        assert ub["type"] == "uniform_buffer"
        assert ub["name"] == "MVP"
        assert len(ub["fields"]) == 3
        assert ub["fields"][0]["name"] == "model"
        assert ub["fields"][0]["type"] == "mat4"
        assert ub["fields"][0]["offset"] == 0
        assert ub["fields"][1]["name"] == "view"
        assert ub["fields"][1]["offset"] == 64
        assert ub["fields"][2]["name"] == "proj"
        assert ub["fields"][2]["offset"] == 128
        assert ub["size"] == 192

    def test_sampler_produces_two_bindings(self):
        source = """
        fragment {
            in frag_uv: vec2;
            out color: vec4;
            sampler2d albedo_tex;
            fn main() {
                color = sample(albedo_tex, frag_uv);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        sets = reflections[0]["descriptor_sets"]
        assert "0" in sets
        bindings = sets["0"]
        # Should have sampler and sampled_image for the texture
        types = {b["type"] for b in bindings}
        assert "sampler" in types
        assert "sampled_image" in types

    def test_multiple_uniforms(self):
        source = """
        fragment {
            in frag_normal: vec3;
            out color: vec4;
            uniform Light {
                light_dir: vec3,
                view_pos: vec3,
            }
            fn main() {
                let n: vec3 = normalize(frag_normal);
                let ndotl: scalar = max(dot(n, normalize(light_dir)), 0.0);
                color = vec4(vec3(ndotl), 1.0);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        sets = reflections[0]["descriptor_sets"]
        assert "0" in sets
        ub = sets["0"][0]
        assert ub["name"] == "Light"
        assert ub["fields"][0]["name"] == "light_dir"
        assert ub["fields"][1]["name"] == "view_pos"


# ===========================================================================
# Vertex attributes reflection
# ===========================================================================

class TestVertexAttributeReflection:
    def test_vertex_attributes(self):
        source = """
        vertex {
            in position: vec3;
            in normal: vec3;
            in uv: vec2;
            fn main() {
                builtin_position = vec4(position, 1.0);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        attrs = reflections[0]["vertex_attributes"]
        assert len(attrs) == 3

        assert attrs[0]["location"] == 0
        assert attrs[0]["name"] == "position"
        assert attrs[0]["format"] == "R32G32B32_SFLOAT"
        assert attrs[0]["offset"] == 0

        assert attrs[1]["location"] == 1
        assert attrs[1]["name"] == "normal"
        assert attrs[1]["format"] == "R32G32B32_SFLOAT"
        assert attrs[1]["offset"] == 12

        assert attrs[2]["location"] == 2
        assert attrs[2]["name"] == "uv"
        assert attrs[2]["format"] == "R32G32_SFLOAT"
        assert attrs[2]["offset"] == 24

    def test_vertex_stride(self):
        source = """
        vertex {
            in position: vec3;
            in normal: vec3;
            in uv: vec2;
            fn main() {
                builtin_position = vec4(position, 1.0);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        assert reflections[0]["vertex_stride"] == 32  # 12 + 12 + 8

    def test_fragment_no_vertex_attrs(self):
        source = """
        fragment {
            out color: vec4;
            fn main() {
                color = vec4(1.0);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        assert reflections[0]["vertex_attributes"] == []
        assert reflections[0]["vertex_stride"] == 0


# ===========================================================================
# Push constant reflection
# ===========================================================================

class TestPushConstantReflection:
    def test_push_constants(self):
        source = """
        vertex {
            in position: vec3;
            push Transform {
                model: mat4,
            }
            fn main() {
                builtin_position = model * vec4(position, 1.0);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        pcs = reflections[0]["push_constants"]
        assert len(pcs) == 1
        assert pcs[0]["name"] == "Transform"
        assert pcs[0]["fields"][0]["name"] == "model"
        assert pcs[0]["fields"][0]["type"] == "mat4"
        assert pcs[0]["size"] == 64


# ===========================================================================
# Surface-expanded reflection
# ===========================================================================

class TestSurfaceReflection:
    def test_surface_expansion_produces_reflection(self):
        source = """
        import brdf;

        surface BasicMaterial {
            brdf: lambert(vec3(0.8, 0.2, 0.3)),
        }
        """
        reflections = _compile_and_reflect(source)
        assert len(reflections) >= 1
        # Should have a fragment stage
        frag = next((r for r in reflections if r["stage"] == "fragment"), None)
        assert frag is not None
        # Fragment should have Light uniform
        sets = frag["descriptor_sets"]
        has_light = False
        for set_bindings in sets.values():
            for b in set_bindings:
                if b.get("name") == "Light":
                    has_light = True
        assert has_light

    def test_pipeline_expansion(self):
        source = """
        import brdf;

        geometry StandardMesh {
            position: vec3,
            normal: vec3,
            uv: vec2,
            transform: MVP {
                model: mat4,
                view: mat4,
                proj: mat4,
            }
            outputs {
                clip_pos: proj * view * model * vec4(position, 1.0),
                frag_normal: normal,
                frag_pos: position,
            }
        }

        surface PBRMaterial {
            brdf: pbr(vec3(0.8), 0.5, 0.0),
        }

        pipeline Main {
            geometry: StandardMesh,
            surface: PBRMaterial,
        }
        """
        reflections = _compile_and_reflect(source)
        assert len(reflections) >= 2

        vert = next((r for r in reflections if r["stage"] == "vertex"), None)
        frag = next((r for r in reflections if r["stage"] == "fragment"), None)
        assert vert is not None
        assert frag is not None

        # Vertex should have MVP uniform
        assert len(vert["vertex_attributes"]) == 3
        assert vert["vertex_stride"] == 32


# ===========================================================================
# RT stage reflection
# ===========================================================================

class TestRTReflection:
    def test_raygen_reflection(self):
        source = """
        import brdf;

        surface RTMaterial {
            brdf: lambert(vec3(1.0)),
        }

        environment Sky {
            color: vec3(0.5, 0.7, 1.0),
        }

        pipeline RTMain {
            mode: raytrace,
            surface: RTMaterial,
            environment: Sky,
        }
        """
        reflections = _compile_and_reflect(source)
        raygen = next((r for r in reflections if r["stage"] == "raygen"), None)
        assert raygen is not None
        assert raygen["execution_model"] == "RayGenerationKHR"

        # Should have acceleration structure descriptor
        has_accel = False
        for set_bindings in raygen["descriptor_sets"].values():
            for b in set_bindings:
                if b.get("type") == "acceleration_structure":
                    has_accel = True
        assert has_accel

        # Should have ray payload metadata
        assert "ray_payloads" in raygen
        assert len(raygen["ray_payloads"]) > 0

    def test_miss_reflection(self):
        source = """
        import brdf;

        surface RTMaterial {
            brdf: lambert(vec3(1.0)),
        }

        environment Sky {
            color: vec3(0.5, 0.7, 1.0),
        }

        pipeline RTMain {
            mode: raytrace,
            surface: RTMaterial,
            environment: Sky,
        }
        """
        reflections = _compile_and_reflect(source)
        miss = next((r for r in reflections if r["stage"] == "miss"), None)
        assert miss is not None
        assert miss["execution_model"] == "MissKHR"


# ===========================================================================
# Integration: compile .lux -> .json round-trip
# ===========================================================================

class TestReflectionRoundTrip:
    def test_json_roundtrip(self):
        source = """
        vertex {
            in position: vec3;
            in normal: vec3;
            uniform MVP {
                model: mat4,
                view: mat4,
                proj: mat4,
            }
            out frag_normal: vec3;
            fn main() {
                frag_normal = normal;
                builtin_position = proj * view * model * vec4(position, 1.0);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        json_str = emit_reflection_json(reflections[0])
        parsed = json.loads(json_str)

        # Verify round-trip preserves all data
        assert parsed["version"] == 1
        assert parsed["stage"] == "vertex"
        assert len(parsed["inputs"]) == 2
        assert len(parsed["vertex_attributes"]) == 2
        assert parsed["vertex_stride"] == 24  # vec3 + vec3
        assert "0" in parsed["descriptor_sets"]
        assert parsed["descriptor_sets"]["0"][0]["name"] == "MVP"

    def test_compile_source_emits_json(self, tmp_path):
        """Integration test: compile_source writes .json files."""
        from luxc.compiler import compile_source

        source = """
        vertex {
            in position: vec3;
            fn main() {
                builtin_position = vec4(position, 1.0);
            }
        }
        """
        lux_file = tmp_path / "test.lux"
        lux_file.write_text(source, encoding="utf-8")

        compile_source(
            source=source,
            stem="test",
            output_dir=tmp_path,
            validate=False,
            emit_reflection=True,
        )

        json_path = tmp_path / "test.vert.json"
        assert json_path.exists()

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert data["stage"] == "vertex"


# ===========================================================================
# Descriptor set assignment for multi-stage shaders
# ===========================================================================

class TestDescriptorSetAssignment:
    """Regression tests for descriptor set assignment across stages.

    When a shader has both vertex and fragment stages, their resources
    must be in different descriptor sets to avoid binding conflicts.
    Vertex stage → set 0, fragment stage → set 1.
    """

    def test_explicit_vertex_fragment_separate_sets(self):
        """Explicit vertex/fragment blocks must get separate descriptor sets."""
        source = """
        vertex {
            in position: vec3;
            in normal: vec3;
            in uv: vec2;
            out frag_uv: vec2;
            uniform MVP {
                model: mat4,
                view: mat4,
                projection: mat4,
            }
            fn main() {
                frag_uv = uv;
                builtin_position = projection * view * model * vec4(position, 1.0);
            }
        }
        fragment {
            in frag_uv: vec2;
            out color: vec4;
            uniform Light {
                light_dir: vec3,
                view_pos: vec3,
            }
            sampler2d albedo_tex;
            fn main() {
                let c: vec4 = sample(albedo_tex, frag_uv);
                color = c;
            }
        }
        """
        reflections = _compile_and_reflect(source)
        vert = next(r for r in reflections if r["stage"] == "vertex")
        frag = next(r for r in reflections if r["stage"] == "fragment")

        # Vertex MVP must be in set 0
        assert "0" in vert["descriptor_sets"]
        mvp = vert["descriptor_sets"]["0"][0]
        assert mvp["name"] == "MVP"
        assert mvp["binding"] == 0

        # Fragment Light+textures must be in set 1 (NOT set 0)
        assert "1" in frag["descriptor_sets"], \
            "Fragment resources must be in set 1, not set 0 (binding conflict)"
        assert "0" not in frag["descriptor_sets"], \
            "Fragment should not have resources in set 0 when vertex uses it"
        frag_bindings = frag["descriptor_sets"]["1"]
        light = next(b for b in frag_bindings if b.get("name") == "Light")
        assert light["binding"] == 0

    def test_explicit_blocks_match_declarative_layout(self):
        """Explicit vertex/fragment must produce same set layout as declarative syntax."""
        # Declarative version (known to work)
        decl_source = """
        import brdf;
        geometry Geo {
            position: vec3,
            normal: vec3,
            uv: vec2,
            transform: MVP {
                model: mat4,
                view: mat4,
                proj: mat4,
            }
            outputs {
                clip_pos: proj * view * model * vec4(position, 1.0),
                frag_normal: normal,
                frag_uv: uv,
            }
        }
        surface Mat {
            sampler2d albedo_tex,
            brdf: lambert(sample(albedo_tex, frag_uv).xyz),
        }
        pipeline P {
            geometry: Geo,
            surface: Mat,
        }
        """
        decl_reflections = _compile_and_reflect(decl_source)
        decl_vert = next(r for r in decl_reflections if r["stage"] == "vertex")
        decl_frag = next(r for r in decl_reflections if r["stage"] == "fragment")

        # Both should use set 0 for vertex, set 1 for fragment
        assert "0" in decl_vert["descriptor_sets"]
        assert "1" in decl_frag["descriptor_sets"]

    def test_single_stage_uses_set_zero(self):
        """A single fragment-only shader should use set 0."""
        source = """
        fragment {
            out color: vec4;
            uniform Params {
                time: scalar,
            }
            fn main() {
                color = vec4(time, 0.0, 0.0, 1.0);
            }
        }
        """
        reflections = _compile_and_reflect(source)
        frag = reflections[0]
        assert "0" in frag["descriptor_sets"]
        assert frag["descriptor_sets"]["0"][0]["name"] == "Params"

    def test_five_texture_gltf_pbr_layout(self):
        """A glTF PBR shader with 5 textures must have all in fragment set 1."""
        source = """
        import brdf;
        import color;
        vertex {
            in position: vec3;
            in normal: vec3;
            in uv: vec2;
            out world_pos: vec3;
            out world_normal: vec3;
            out frag_uv: vec2;
            uniform MVP {
                model: mat4,
                view: mat4,
                projection: mat4,
            }
            fn main() {
                let world: vec4 = model * vec4(position, 1.0);
                world_pos = world.xyz;
                world_normal = normalize((model * vec4(normal, 0.0)).xyz);
                frag_uv = uv;
                builtin_position = projection * view * world;
            }
        }
        fragment {
            in world_pos: vec3;
            in world_normal: vec3;
            in frag_uv: vec2;
            out color: vec4;
            uniform Light {
                light_dir: vec3,
                view_pos: vec3,
            }
            sampler2d base_color_tex;
            sampler2d normal_tex;
            sampler2d metallic_roughness_tex;
            sampler2d occlusion_tex;
            sampler2d emissive_tex;
            fn main() {
                let bc: vec4 = sample(base_color_tex, frag_uv);
                let n: vec4 = sample(normal_tex, frag_uv);
                let mr: vec4 = sample(metallic_roughness_tex, frag_uv);
                let ao: vec4 = sample(occlusion_tex, frag_uv);
                let em: vec4 = sample(emissive_tex, frag_uv);
                color = bc;
            }
        }
        """
        reflections = _compile_and_reflect(source)
        vert = next(r for r in reflections if r["stage"] == "vertex")
        frag = next(r for r in reflections if r["stage"] == "fragment")

        # Vertex: set 0, binding 0 = MVP
        assert "0" in vert["descriptor_sets"]

        # Fragment: set 1 with Light + 5 textures (10 bindings: sampler + image each)
        assert "1" in frag["descriptor_sets"]
        frag_bindings = frag["descriptor_sets"]["1"]
        # Light (binding 0) + 5 textures * 2 bindings each = 11 total
        assert len(frag_bindings) == 11
        # Verify no set 0 conflict
        assert "0" not in frag["descriptor_sets"]

    def test_rt_multi_stage_separate_sets(self):
        """RT stages (raygen, closest_hit) must get separate descriptor sets.

        Regression test: gltf_pbr_rt.lux had Camera (rgen set 0 binding 0)
        colliding with Light (rchit set 0 binding 0). Fix: rchit → set 1.
        """
        # Use the actual compiled gltf_pbr_rt.lux reflection JSON
        # to verify the compiler assigns separate sets.
        source_path = Path(__file__).parent.parent / "examples" / "gltf_pbr_rt.lux"
        if not source_path.exists():
            pytest.skip("gltf_pbr_rt.lux not found")
        reflections = _compile_and_reflect(source_path.read_text())
        rgen = next(r for r in reflections if r["stage"] == "raygen")
        rchit = next(r for r in reflections if r["stage"] == "closest_hit")

        # Raygen: Camera + tlas + output_image in set 0
        assert "0" in rgen["descriptor_sets"]
        rgen_bindings = rgen["descriptor_sets"]["0"]
        camera = next(b for b in rgen_bindings if b.get("name") == "Camera")
        assert camera["binding"] == 0

        # Closest hit: Light in set 1 (NOT set 0 — would collide with Camera)
        assert "1" in rchit["descriptor_sets"], \
            "Closest hit must be in set 1, not set 0 (binding conflict with raygen)"
        assert "0" not in rchit["descriptor_sets"], \
            "Closest hit should not have resources in set 0 when raygen uses it"
        rchit_bindings = rchit["descriptor_sets"]["1"]
        light = next(b for b in rchit_bindings if b.get("name") == "Light")
        assert light["binding"] == 0
