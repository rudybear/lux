"""P6 Ray Tracing Pipeline tests.

Tests cover:
- RT grammar parsing (stage types, environment, procedural, RT declarations)
- AST node creation and Module population
- RT type system (acceleration_structure)
- RT SPIR-V codegen (capabilities, execution models, storage classes, instructions)
- RT surface expansion (surface→closest_hit, environment→miss, procedural→intersection)
- RT pipeline expansion with mode: raytrace
"""

import pytest
import subprocess
import shutil
from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import (
    Module, StageBlock, EnvironmentDecl, ProceduralDecl,
    RayPayloadDecl, HitAttributeDecl, CallableDataDecl, AccelDecl,
    SurfaceMember, ProceduralMember,
)
from luxc.builtins.types import (
    resolve_type, AccelerationStructureType, ACCELERATION_STRUCTURE,
)
from luxc.codegen.spirv_types import TypeRegistry


# =========================================================================
# Grammar / Parsing Tests
# =========================================================================

class TestRTGrammar:
    """Test that RT-related grammar extensions parse correctly."""

    def test_rt_stage_types_parse(self):
        """All 6 RT stage types should parse as valid stage blocks."""
        for stage_type in ["raygen", "closest_hit", "any_hit", "miss", "intersection", "callable"]:
            src = f'{stage_type} {{ fn main() {{ }} }}'
            module = parse_lux(src)
            assert len(module.stages) == 1
            assert module.stages[0].stage_type == stage_type

    def test_environment_decl_parses(self):
        """Environment declaration should parse into EnvironmentDecl."""
        src = '''
        environment Sky {
            color: vec3(0.5, 0.7, 1.0),
        }
        '''
        module = parse_lux(src)
        assert len(module.environments) == 1
        env = module.environments[0]
        assert isinstance(env, EnvironmentDecl)
        assert env.name == "Sky"
        assert len(env.members) == 1
        assert env.members[0].name == "color"

    def test_environment_with_multiple_members(self):
        src = '''
        environment HDRISky {
            color: vec3(1.0),
            intensity: 2.5,
        }
        '''
        module = parse_lux(src)
        assert len(module.environments) == 1
        env = module.environments[0]
        assert len(env.members) == 2

    def test_procedural_decl_parses(self):
        """Procedural declaration should parse into ProceduralDecl."""
        src = '''
        procedural MetaBall {
            sdf: sdf_sphere(0.5),
        }
        '''
        module = parse_lux(src)
        assert len(module.procedurals) == 1
        proc = module.procedurals[0]
        assert isinstance(proc, ProceduralDecl)
        assert proc.name == "MetaBall"
        assert len(proc.members) == 1
        assert proc.members[0].name == "sdf"

    def test_procedural_with_surface_ref(self):
        src = '''
        procedural MetaBall {
            sdf: sdf_sphere(0.5),
            surface: Chrome,
        }
        '''
        module = parse_lux(src)
        assert len(module.procedurals[0].members) == 2

    def test_ray_payload_decl(self):
        """Ray payload declarations should parse inside RT stage blocks."""
        src = '''
        raygen {
            ray_payload color: vec4;
            fn main() { }
        }
        '''
        module = parse_lux(src)
        assert len(module.stages) == 1
        stage = module.stages[0]
        assert stage.stage_type == "raygen"
        assert len(stage.ray_payloads) == 1
        rp = stage.ray_payloads[0]
        assert isinstance(rp, RayPayloadDecl)
        assert rp.name == "color"
        assert rp.type_name == "vec4"

    def test_hit_attribute_decl(self):
        """Hit attribute declarations should parse inside intersection blocks."""
        src = '''
        intersection {
            hit_attribute barycentrics: vec2;
            fn main() { }
        }
        '''
        module = parse_lux(src)
        stage = module.stages[0]
        assert len(stage.hit_attributes) == 1
        ha = stage.hit_attributes[0]
        assert isinstance(ha, HitAttributeDecl)
        assert ha.name == "barycentrics"
        assert ha.type_name == "vec2"

    def test_callable_data_decl(self):
        src = '''
        callable {
            callable_data result: vec3;
            fn main() { }
        }
        '''
        module = parse_lux(src)
        stage = module.stages[0]
        assert len(stage.callable_data) == 1
        cd = stage.callable_data[0]
        assert isinstance(cd, CallableDataDecl)
        assert cd.name == "result"
        assert cd.type_name == "vec3"

    def test_accel_struct_decl(self):
        """Acceleration structure declarations should parse."""
        src = '''
        raygen {
            acceleration_structure tlas;
            fn main() { }
        }
        '''
        module = parse_lux(src)
        stage = module.stages[0]
        assert len(stage.accel_structs) == 1
        accel = stage.accel_structs[0]
        assert isinstance(accel, AccelDecl)
        assert accel.name == "tlas"

    def test_acceleration_structure_type_name(self):
        """acceleration_structure should be a valid type name."""
        src = '''
        fn test_func(x: acceleration_structure) -> void {
        }
        '''
        # This should parse without error (type checker will handle it)
        module = parse_lux(src)
        assert module.functions[0].params[0].type_name == "acceleration_structure"

    def test_mixed_rt_stage_items(self):
        """RT stage with mixed declaration types."""
        src = '''
        closest_hit {
            ray_payload payload: vec4;
            hit_attribute attribs: vec2;
            uniform Light {
                light_dir: vec3,
            }
            fn main() { }
        }
        '''
        module = parse_lux(src)
        stage = module.stages[0]
        assert stage.stage_type == "closest_hit"
        assert len(stage.ray_payloads) == 1
        assert len(stage.hit_attributes) == 1
        assert len(stage.uniforms) == 1

    def test_pipeline_mode_raytrace(self):
        """Pipeline with mode: raytrace should parse."""
        src = '''
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        environment Sky {
            color: vec3(0.5, 0.7, 1.0),
        }
        pipeline RTRenderer {
            mode: raytrace,
            surface: Metal,
            environment: Sky,
            max_bounces: 4,
        }
        '''
        module = parse_lux(src)
        assert len(module.surfaces) == 1
        assert len(module.environments) == 1
        assert len(module.pipelines) == 1
        # Check pipeline members
        members = {m.name for m in module.pipelines[0].members}
        assert "mode" in members
        assert "surface" in members
        assert "environment" in members


# =========================================================================
# Type System Tests
# =========================================================================

class TestRTTypes:
    """Test acceleration_structure type registration."""

    def test_acceleration_structure_type_exists(self):
        t = resolve_type("acceleration_structure")
        assert t is not None
        assert isinstance(t, AccelerationStructureType)

    def test_acceleration_structure_singleton(self):
        assert resolve_type("acceleration_structure") is ACCELERATION_STRUCTURE

    def test_acceleration_structure_name(self):
        assert ACCELERATION_STRUCTURE.name == "acceleration_structure"

    def test_spirv_type_registry(self):
        reg = TypeRegistry()
        tid = reg.acceleration_structure_type()
        assert tid is not None
        # Should be deduplicated
        tid2 = reg.acceleration_structure_type()
        assert tid == tid2
        # Check declaration
        decls = reg.emit_declarations()
        assert any("OpTypeAccelerationStructureKHR" in d for d in decls)

    def test_spirv_lux_type_mapping(self):
        """lux_type_to_spirv should map acceleration_structure correctly."""
        reg = TypeRegistry()
        tid = reg.lux_type_to_spirv("acceleration_structure")
        assert tid is not None
        decls = reg.emit_declarations()
        assert any("OpTypeAccelerationStructureKHR" in d for d in decls)


# =========================================================================
# SPIR-V Codegen Tests
# =========================================================================

class TestRTCodegen:
    """Test RT SPIR-V code generation."""

    def _get_spirv(self, src: str, stage_idx: int = 0) -> str:
        """Parse, type-check, and generate SPIR-V for a source string."""
        from luxc.analysis.type_checker import type_check
        from luxc.optimization.const_fold import constant_fold
        from luxc.analysis.layout_assigner import assign_layouts
        from luxc.codegen.spirv_builder import generate_spirv
        from luxc.builtins.types import clear_type_aliases

        clear_type_aliases()
        module = parse_lux(src)
        type_check(module)
        constant_fold(module)
        assign_layouts(module)
        stage = module.stages[stage_idx]
        return generate_spirv(module, stage)

    def test_raygen_capability(self):
        """Raygen shader should emit RayTracingKHR capability."""
        src = '''
        raygen {
            fn main() { }
        }
        '''
        asm = self._get_spirv(src)
        assert "OpCapability RayTracingKHR" in asm
        assert 'OpExtension "SPV_KHR_ray_tracing"' in asm

    def test_raygen_execution_model(self):
        src = '''
        raygen {
            fn main() { }
        }
        '''
        asm = self._get_spirv(src)
        assert "OpEntryPoint RayGenerationKHR %main" in asm

    def test_closest_hit_execution_model(self):
        src = '''
        closest_hit {
            fn main() { }
        }
        '''
        asm = self._get_spirv(src)
        assert "OpEntryPoint ClosestHitKHR %main" in asm

    def test_any_hit_execution_model(self):
        src = '''
        any_hit {
            fn main() { }
        }
        '''
        asm = self._get_spirv(src)
        assert "OpEntryPoint AnyHitKHR %main" in asm

    def test_miss_execution_model(self):
        src = '''
        miss {
            fn main() { }
        }
        '''
        asm = self._get_spirv(src)
        assert "OpEntryPoint MissKHR %main" in asm

    def test_intersection_execution_model(self):
        src = '''
        intersection {
            fn main() { }
        }
        '''
        asm = self._get_spirv(src)
        assert "OpEntryPoint IntersectionKHR %main" in asm

    def test_callable_execution_model(self):
        src = '''
        callable {
            fn main() { }
        }
        '''
        asm = self._get_spirv(src)
        assert "OpEntryPoint CallableKHR %main" in asm

    def test_no_origin_upper_left_for_rt(self):
        """RT shaders should NOT emit OriginUpperLeft execution mode."""
        src = '''
        raygen {
            fn main() { }
        }
        '''
        asm = self._get_spirv(src)
        assert "OriginUpperLeft" not in asm

    def test_ray_payload_variable(self):
        """Ray payload should generate correct storage class."""
        src = '''
        raygen {
            ray_payload color: vec4;
            fn main() { }
        }
        '''
        asm = self._get_spirv(src)
        assert "RayPayloadKHR" in asm
        assert "OpDecorate" in asm  # Should have Location decoration

    def test_incoming_ray_payload(self):
        """In closest_hit/any_hit, payload uses IncomingRayPayloadKHR."""
        src = '''
        closest_hit {
            ray_payload payload: vec4;
            fn main() { }
        }
        '''
        asm = self._get_spirv(src)
        assert "IncomingRayPayloadKHR" in asm

    def test_hit_attribute_variable(self):
        src = '''
        closest_hit {
            hit_attribute attribs: vec2;
            fn main() { }
        }
        '''
        asm = self._get_spirv(src)
        assert "HitAttributeKHR" in asm

    def test_callable_data_variable(self):
        src = '''
        callable {
            callable_data result: vec3;
            fn main() { }
        }
        '''
        asm = self._get_spirv(src)
        assert "IncomingCallableDataKHR" in asm

    def test_accel_struct_variable(self):
        src = '''
        raygen {
            acceleration_structure tlas;
            fn main() { }
        }
        '''
        asm = self._get_spirv(src)
        assert "OpTypeAccelerationStructureKHR" in asm
        assert "UniformConstant" in asm

    def test_rt_builtin_launch_id(self):
        """RT builtins (launch_id) should be declared with BuiltIn decorations."""
        src = '''
        raygen {
            fn main() {
                let id: uvec3 = launch_id;
            }
        }
        '''
        asm = self._get_spirv(src)
        assert "BuiltIn LaunchIdKHR" in asm

    def test_rt_builtin_launch_size(self):
        src = '''
        raygen {
            fn main() {
                let sz: uvec3 = launch_size;
            }
        }
        '''
        asm = self._get_spirv(src)
        assert "BuiltIn LaunchSizeKHR" in asm

    def test_rt_builtin_ray_origin(self):
        src = '''
        closest_hit {
            fn main() {
                let orig: vec3 = world_ray_origin;
            }
        }
        '''
        asm = self._get_spirv(src)
        assert "BuiltIn WorldRayOriginKHR" in asm

    def test_rt_builtin_ray_direction(self):
        src = '''
        closest_hit {
            fn main() {
                let dir: vec3 = world_ray_direction;
            }
        }
        '''
        asm = self._get_spirv(src)
        assert "BuiltIn WorldRayDirectionKHR" in asm

    def test_rt_builtin_hit_t(self):
        src = '''
        closest_hit {
            fn main() {
                let t: scalar = hit_t;
            }
        }
        '''
        asm = self._get_spirv(src)
        assert "BuiltIn HitTKHR" in asm

    def test_vertex_shader_unchanged(self):
        """Vertex shaders should NOT get RT capabilities."""
        src = '''
        vertex {
            in position: vec3;
            out color: vec3;
            fn main() {
                builtin_position = vec4(position, 1.0);
                color = position;
            }
        }
        '''
        asm = self._get_spirv(src)
        assert "OpCapability Shader" in asm
        assert "RayTracingKHR" not in asm
        assert "OpEntryPoint Vertex" in asm

    def test_fragment_shader_unchanged(self):
        """Fragment shaders should still work correctly."""
        src = '''
        fragment {
            in uv: vec2;
            out color: vec4;
            fn main() {
                color = vec4(uv, 0.0, 1.0);
            }
        }
        '''
        asm = self._get_spirv(src)
        assert "OpEntryPoint Fragment" in asm
        assert "OriginUpperLeft" in asm
        assert "RayTracingKHR" not in asm


# =========================================================================
# RT Expansion Tests
# =========================================================================

class TestRTExpansion:
    """Test RT pipeline expansion from declarative to stage blocks."""

    def _expand_and_get_stages(self, src: str) -> list[StageBlock]:
        """Parse source, expand surfaces, return stages."""
        from luxc.builtins.types import clear_type_aliases
        clear_type_aliases()
        module = parse_lux(src)
        # Resolve imports
        from luxc.compiler import _resolve_imports
        _resolve_imports(module)
        # Expand
        from luxc.expansion.surface_expander import expand_surfaces
        expand_surfaces(module)
        return module.stages

    def test_rt_pipeline_creates_raygen(self):
        """RT pipeline should generate a raygen stage."""
        src = '''
        import brdf;
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        environment Sky {
            color: vec3(0.5, 0.7, 1.0),
        }
        pipeline RT {
            mode: raytrace,
            surface: Metal,
            environment: Sky,
        }
        '''
        stages = self._expand_and_get_stages(src)
        stage_types = [s.stage_type for s in stages]
        assert "raygen" in stage_types

    def test_rt_pipeline_creates_closest_hit(self):
        """RT pipeline with surface should create closest_hit stage."""
        src = '''
        import brdf;
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        pipeline RT {
            mode: raytrace,
            surface: Metal,
        }
        '''
        stages = self._expand_and_get_stages(src)
        stage_types = [s.stage_type for s in stages]
        assert "closest_hit" in stage_types

    def test_rt_pipeline_creates_miss(self):
        """RT pipeline with environment should create miss stage."""
        src = '''
        import brdf;
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        environment Sky {
            color: vec3(0.5, 0.7, 1.0),
        }
        pipeline RT {
            mode: raytrace,
            surface: Metal,
            environment: Sky,
        }
        '''
        stages = self._expand_and_get_stages(src)
        stage_types = [s.stage_type for s in stages]
        assert "miss" in stage_types

    def test_rt_pipeline_full_stages(self):
        """Full RT pipeline should create raygen + closest_hit + miss."""
        src = '''
        import brdf;
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        environment Sky {
            color: vec3(0.5, 0.7, 1.0),
        }
        pipeline RT {
            mode: raytrace,
            surface: Metal,
            environment: Sky,
        }
        '''
        stages = self._expand_and_get_stages(src)
        stage_types = set(s.stage_type for s in stages)
        assert stage_types == {"raygen", "closest_hit", "miss"}

    def test_rt_pipeline_with_opacity_creates_any_hit(self):
        """Surface with opacity should generate an any_hit stage."""
        src = '''
        import brdf;
        surface Glass {
            brdf: lambert(vec3(0.9)),
            opacity: 0.3,
        }
        pipeline RT {
            mode: raytrace,
            surface: Glass,
        }
        '''
        stages = self._expand_and_get_stages(src)
        stage_types = [s.stage_type for s in stages]
        assert "any_hit" in stage_types

    def test_rt_pipeline_no_opacity_no_any_hit(self):
        """Surface without opacity should NOT generate any_hit."""
        src = '''
        import brdf;
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        pipeline RT {
            mode: raytrace,
            surface: Metal,
        }
        '''
        stages = self._expand_and_get_stages(src)
        stage_types = [s.stage_type for s in stages]
        assert "any_hit" not in stage_types

    def test_raygen_has_accel_struct(self):
        """Expanded raygen should have acceleration structure binding."""
        src = '''
        import brdf;
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        pipeline RT {
            mode: raytrace,
            surface: Metal,
        }
        '''
        stages = self._expand_and_get_stages(src)
        raygen = next(s for s in stages if s.stage_type == "raygen")
        assert len(raygen.accel_structs) == 1
        assert raygen.accel_structs[0].name == "tlas"

    def test_raygen_has_payload(self):
        """Expanded raygen should have ray payload."""
        src = '''
        import brdf;
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        pipeline RT {
            mode: raytrace,
            surface: Metal,
        }
        '''
        stages = self._expand_and_get_stages(src)
        raygen = next(s for s in stages if s.stage_type == "raygen")
        assert len(raygen.ray_payloads) == 1
        assert raygen.ray_payloads[0].name == "payload"
        assert raygen.ray_payloads[0].type_name == "vec4"

    def test_closest_hit_has_payload(self):
        """Expanded closest_hit should have incoming ray payload."""
        src = '''
        import brdf;
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        pipeline RT {
            mode: raytrace,
            surface: Metal,
        }
        '''
        stages = self._expand_and_get_stages(src)
        chit = next(s for s in stages if s.stage_type == "closest_hit")
        assert len(chit.ray_payloads) == 1
        assert chit.ray_payloads[0].name == "payload"

    def test_closest_hit_has_hit_attribs(self):
        """Expanded closest_hit should have hit attributes."""
        src = '''
        import brdf;
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        pipeline RT {
            mode: raytrace,
            surface: Metal,
        }
        '''
        stages = self._expand_and_get_stages(src)
        chit = next(s for s in stages if s.stage_type == "closest_hit")
        assert len(chit.hit_attributes) == 1
        assert chit.hit_attributes[0].name == "attribs"

    def test_miss_has_payload(self):
        """Expanded miss should have incoming ray payload."""
        src = '''
        import brdf;
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        environment Sky {
            color: vec3(0.5, 0.7, 1.0),
        }
        pipeline RT {
            mode: raytrace,
            surface: Metal,
            environment: Sky,
        }
        '''
        stages = self._expand_and_get_stages(src)
        miss = next(s for s in stages if s.stage_type == "miss")
        assert len(miss.ray_payloads) == 1
        assert miss.ray_payloads[0].name == "payload"

    def test_rasterize_pipeline_still_works(self):
        """Rasterization pipeline (no mode or mode: rasterize) should be unchanged."""
        src = '''
        import brdf;
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        pipeline Raster {
            surface: Metal,
        }
        '''
        stages = self._expand_and_get_stages(src)
        stage_types = [s.stage_type for s in stages]
        assert "fragment" in stage_types
        assert "raygen" not in stage_types

    def test_standalone_surface_unaffected(self):
        """Standalone surfaces without pipeline should still expand to fragment."""
        src = '''
        import brdf;
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        '''
        stages = self._expand_and_get_stages(src)
        assert len(stages) == 1
        assert stages[0].stage_type == "fragment"

    def test_procedural_creates_intersection(self):
        """Pipeline with procedural should generate intersection stage."""
        src = '''
        import brdf;
        import sdf;
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        procedural Ball {
            sdf: sdf_sphere(0.5),
        }
        pipeline RT {
            mode: raytrace,
            surface: Metal,
            procedural: Ball,
        }
        '''
        stages = self._expand_and_get_stages(src)
        stage_types = [s.stage_type for s in stages]
        assert "intersection" in stage_types

    def test_intersection_has_hit_attribs(self):
        """Expanded intersection should have hit attributes."""
        src = '''
        import brdf;
        import sdf;
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        procedural Ball {
            sdf: sdf_sphere(0.5),
        }
        pipeline RT {
            mode: raytrace,
            surface: Metal,
            procedural: Ball,
        }
        '''
        stages = self._expand_and_get_stages(src)
        isect = next(s for s in stages if s.stage_type == "intersection")
        assert len(isect.hit_attributes) == 1


# =========================================================================
# Layout Assigner Tests
# =========================================================================

class TestRTLayout:
    """Test that RT-specific layouts are assigned correctly."""

    def test_payload_location_assigned(self):
        """Ray payload should get auto-assigned location."""
        from luxc.analysis.layout_assigner import assign_layouts
        from luxc.builtins.types import clear_type_aliases
        clear_type_aliases()

        src = '''
        raygen {
            ray_payload color: vec4;
            fn main() { }
        }
        '''
        module = parse_lux(src)
        assign_layouts(module)
        rp = module.stages[0].ray_payloads[0]
        assert rp.location == 0

    def test_multiple_payload_locations(self):
        from luxc.analysis.layout_assigner import assign_layouts
        from luxc.builtins.types import clear_type_aliases
        clear_type_aliases()

        src = '''
        raygen {
            ray_payload color: vec4;
            ray_payload shadow: scalar;
            fn main() { }
        }
        '''
        module = parse_lux(src)
        assign_layouts(module)
        assert module.stages[0].ray_payloads[0].location == 0
        assert module.stages[0].ray_payloads[1].location == 1

    def test_accel_struct_binding_assigned(self):
        from luxc.analysis.layout_assigner import assign_layouts
        from luxc.builtins.types import clear_type_aliases
        clear_type_aliases()

        src = '''
        raygen {
            acceleration_structure tlas;
            fn main() { }
        }
        '''
        module = parse_lux(src)
        assign_layouts(module)
        accel = module.stages[0].accel_structs[0]
        assert accel.set_number is not None
        assert accel.binding is not None


# =========================================================================
# E2E Compilation Tests (require spirv-tools)
# =========================================================================

HAS_SPIRV_TOOLS = shutil.which("spirv-as") is not None


@pytest.mark.skipif(not HAS_SPIRV_TOOLS, reason="spirv-tools not installed")
class TestRTEndToEnd:
    """End-to-end compilation tests for RT shaders."""

    def _compile_and_check(self, src: str):
        """Parse → type check → codegen → assemble. Returns list of SPIR-V asm strings."""
        from luxc.analysis.type_checker import type_check
        from luxc.optimization.const_fold import constant_fold
        from luxc.analysis.layout_assigner import assign_layouts
        from luxc.codegen.spirv_builder import generate_spirv
        from luxc.builtins.types import clear_type_aliases

        clear_type_aliases()
        module = parse_lux(src)
        type_check(module)
        constant_fold(module)
        assign_layouts(module)

        results = []
        for stage in module.stages:
            asm = generate_spirv(module, stage)
            results.append((stage.stage_type, asm))
        return results

    def test_simple_raygen_compiles(self):
        """Minimal raygen shader should compile."""
        src = '''
        raygen {
            fn main() { }
        }
        '''
        results = self._compile_and_check(src)
        assert len(results) == 1
        assert results[0][0] == "raygen"

    def test_raygen_with_builtins(self):
        """Raygen with launch_id and launch_size should compile."""
        src = '''
        raygen {
            fn main() {
                let id: uvec3 = launch_id;
                let sz: uvec3 = launch_size;
            }
        }
        '''
        results = self._compile_and_check(src)
        asm = results[0][1]
        assert "LaunchIdKHR" in asm
        assert "LaunchSizeKHR" in asm

    def test_closest_hit_with_builtins(self):
        """Closest-hit with RT builtins should compile."""
        src = '''
        closest_hit {
            ray_payload payload: vec4;
            hit_attribute attribs: vec2;
            fn main() {
                let origin: vec3 = world_ray_origin;
                let dir: vec3 = world_ray_direction;
                let t: scalar = hit_t;
                let hit_pos: vec3 = origin + dir * t;
                payload = vec4(hit_pos, 1.0);
            }
        }
        '''
        results = self._compile_and_check(src)
        assert results[0][0] == "closest_hit"

    def test_miss_shader_compiles(self):
        """Miss shader should compile."""
        src = '''
        miss {
            ray_payload payload: vec4;
            fn main() {
                let dir: vec3 = world_ray_direction;
                let t: scalar = dir.y * 0.5 + 0.5;
                let sky: vec3 = mix(vec3(1.0), vec3(0.5, 0.7, 1.0), t);
                payload = vec4(sky, 1.0);
            }
        }
        '''
        results = self._compile_and_check(src)
        assert results[0][0] == "miss"

    def test_intersection_shader_compiles(self):
        """Intersection shader should compile."""
        src = '''
        intersection {
            hit_attribute attribs: vec2;
            fn main() {
                let origin: vec3 = world_ray_origin;
                let dir: vec3 = world_ray_direction;
                let t: scalar = ray_tmin;
                attribs = vec2(0.0);
            }
        }
        '''
        results = self._compile_and_check(src)
        assert results[0][0] == "intersection"

    def test_all_six_stages_compile(self):
        """All 6 RT stage types should produce valid SPIR-V assembly."""
        for stage_type in ["raygen", "closest_hit", "any_hit", "miss", "intersection", "callable"]:
            src = f'{stage_type} {{ fn main() {{ }} }}'
            results = self._compile_and_check(src)
            assert results[0][0] == stage_type, f"{stage_type} failed to compile"
