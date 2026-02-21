"""P13 Mesh Shader tests.

Tests cover:
- Mesh/task grammar parsing (stage types, mesh_output_decl, task_payload_decl)
- AST node creation and Module population
- Mesh shader SPIR-V codegen (MeshShadingEXT capability, MeshEXT/TaskEXT execution models)
- Mesh pipeline expansion (mode: mesh_shader -> mesh + fragment stages)
- End-to-end compilation of mesh_shader_manual.lux and GltfMesh pipeline from gltf_pbr_layered.lux
- Reflection metadata (mesh_output with max_vertices, max_primitives, workgroup_size)
"""

import pytest
import shutil
from pathlib import Path
from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import (
    Module, StageBlock, MeshOutputDecl, TaskPayloadDecl,
    StorageBufferDecl,
)
from luxc.builtins.types import clear_type_aliases

EXAMPLES = Path(__file__).parent.parent / "examples"


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


# =========================================================================
# Grammar / Parsing Tests
# =========================================================================

class TestMeshGrammar:
    """Test that mesh/task-related grammar extensions parse correctly."""

    def test_mesh_stage_type_parses(self):
        """A minimal mesh stage block should parse with stage_type='mesh'."""
        src = 'mesh { fn main() { } }'
        module = parse_lux(src)
        assert len(module.stages) == 1
        assert module.stages[0].stage_type == "mesh"

    def test_task_stage_type_parses(self):
        """A minimal task stage block should parse with stage_type='task'."""
        src = 'task { fn main() { } }'
        module = parse_lux(src)
        assert len(module.stages) == 1
        assert module.stages[0].stage_type == "task"

    def test_mesh_output_decl_parses(self):
        """mesh_output declarations should parse inside mesh stage blocks."""
        src = '''
        mesh {
            mesh_output PerVertex {
                position: vec4,
                normal: vec3,
            }
            fn main() { }
        }
        '''
        module = parse_lux(src)
        assert len(module.stages) == 1
        stage = module.stages[0]
        assert stage.stage_type == "mesh"
        assert len(stage.mesh_outputs) == 1
        mo = stage.mesh_outputs[0]
        assert isinstance(mo, MeshOutputDecl)
        assert mo.name == "PerVertex"
        assert len(mo.fields) == 2

    def test_task_payload_decl_parses(self):
        """task_payload declarations should parse inside task stage blocks."""
        src = '''
        task {
            task_payload data: uint;
            fn main() { }
        }
        '''
        module = parse_lux(src)
        assert len(module.stages) == 1
        stage = module.stages[0]
        assert stage.stage_type == "task"
        assert len(stage.task_payloads) == 1
        tp = stage.task_payloads[0]
        assert isinstance(tp, TaskPayloadDecl)
        assert tp.name == "data"
        assert tp.type_name == "uint"

    def test_mesh_with_storage_buffer(self):
        """Mesh stage should support storage_buffer declarations."""
        src = '''
        mesh {
            storage_buffer positions: vec4;
            fn main() { }
        }
        '''
        module = parse_lux(src)
        stage = module.stages[0]
        assert len(stage.storage_buffers) == 1
        sb = stage.storage_buffers[0]
        assert isinstance(sb, StorageBufferDecl)
        assert sb.name == "positions"
        assert sb.element_type == "vec4"

    def test_mesh_with_uniform_and_outputs(self):
        """Mesh stage with uniform block and outputs should parse."""
        src = '''
        mesh {
            uniform MVP {
                model: mat4,
                view: mat4,
                projection: mat4,
            }
            out frag_color: vec3;
            storage_buffer positions: vec4;
            fn main() { }
        }
        '''
        module = parse_lux(src)
        stage = module.stages[0]
        assert stage.stage_type == "mesh"
        assert len(stage.uniforms) == 1
        assert len(stage.outputs) == 1
        assert len(stage.storage_buffers) == 1

    def test_pipeline_mode_mesh_shader(self):
        """Pipeline with mode: mesh_shader should parse."""
        src = '''
        import brdf;
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        geometry Mesh {
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
        pipeline MeshPipe {
            mode: mesh_shader,
            geometry: Mesh,
            surface: Metal,
        }
        '''
        module = parse_lux(src)
        assert len(module.pipelines) == 1
        members = {m.name for m in module.pipelines[0].members}
        assert "mode" in members
        assert "geometry" in members
        assert "surface" in members


# =========================================================================
# SPIR-V Codegen Tests
# =========================================================================

class TestMeshCodegen:
    """Test mesh shader SPIR-V code generation."""

    def _get_spirv(self, src: str, stage_idx: int = 0, defines: dict = None) -> str:
        """Parse, type-check, and generate SPIR-V for a source string."""
        from luxc.analysis.type_checker import type_check
        from luxc.optimization.const_fold import constant_fold
        from luxc.analysis.layout_assigner import assign_layouts
        from luxc.codegen.spirv_builder import generate_spirv

        clear_type_aliases()
        module = parse_lux(src)
        module._defines = defines or {}
        type_check(module)
        constant_fold(module)
        assign_layouts(module)
        stage = module.stages[stage_idx]
        return generate_spirv(module, stage)

    def test_mesh_capability(self):
        """Mesh shader should emit MeshShadingEXT capability."""
        src = 'mesh { fn main() { } }'
        asm = self._get_spirv(src)
        assert "OpCapability MeshShadingEXT" in asm
        assert 'OpExtension "SPV_EXT_mesh_shader"' in asm

    def test_mesh_execution_model(self):
        """Mesh shader should use MeshEXT execution model."""
        src = 'mesh { fn main() { } }'
        asm = self._get_spirv(src)
        assert "OpEntryPoint MeshEXT %main" in asm

    def test_task_execution_model(self):
        """Task shader should use TaskEXT execution model."""
        src = 'task { fn main() { } }'
        asm = self._get_spirv(src)
        assert "OpEntryPoint TaskEXT %main" in asm
        assert "OpCapability MeshShadingEXT" in asm

    def test_mesh_execution_modes(self):
        """Mesh shader should have LocalSize, OutputVertices, OutputPrimitivesEXT, OutputTrianglesEXT."""
        src = 'mesh { fn main() { } }'
        defines = {'workgroup_size': 32, 'max_vertices': 64, 'max_primitives': 124}
        asm = self._get_spirv(src, defines=defines)
        assert "OpExecutionMode %main LocalSize 32 1 1" in asm
        assert "OpExecutionMode %main OutputVertices 64" in asm
        assert "OpExecutionMode %main OutputPrimitivesEXT 124" in asm
        assert "OpExecutionMode %main OutputTrianglesEXT" in asm

    def test_task_execution_mode(self):
        """Task shader should have LocalSize execution mode."""
        src = 'task { fn main() { } }'
        defines = {'workgroup_size': 32}
        asm = self._get_spirv(src, defines=defines)
        assert "OpExecutionMode %main LocalSize 32 1 1" in asm

    def test_no_origin_upper_left_for_mesh(self):
        """Mesh shaders should NOT emit OriginUpperLeft."""
        src = 'mesh { fn main() { } }'
        asm = self._get_spirv(src)
        assert "OriginUpperLeft" not in asm

    def test_mesh_builtin_local_invocation_index(self):
        """Mesh shader local_invocation_index builtin should resolve."""
        src = '''
        mesh {
            fn main() {
                let tid: uint = local_invocation_index;
            }
        }
        '''
        asm = self._get_spirv(src)
        assert "BuiltIn LocalInvocationIndex" in asm

    def test_mesh_builtin_workgroup_id(self):
        """Mesh shader workgroup_id builtin should resolve."""
        src = '''
        mesh {
            fn main() {
                let wg: uvec3 = workgroup_id;
            }
        }
        '''
        asm = self._get_spirv(src)
        assert "BuiltIn WorkgroupId" in asm

    def test_vertex_shader_unchanged(self):
        """Vertex shaders should NOT get mesh shader capabilities."""
        src = '''
        vertex {
            in position: vec3;
            fn main() {
                builtin_position = vec4(position, 1.0);
            }
        }
        '''
        asm = self._get_spirv(src)
        assert "MeshShadingEXT" not in asm
        assert "OpEntryPoint Vertex" in asm


# =========================================================================
# Mesh Pipeline Expansion Tests
# =========================================================================

class TestMeshExpansion:
    """Test mesh shader pipeline expansion from declarative to stage blocks."""

    def _expand_and_get_stages(self, src: str) -> list[StageBlock]:
        """Parse source, expand surfaces, return stages."""
        clear_type_aliases()
        module = parse_lux(src)
        module._defines = {'max_vertices': 64, 'max_primitives': 124, 'workgroup_size': 32}
        # Resolve imports
        from luxc.compiler import _resolve_imports
        _resolve_imports(module)
        # Expand
        from luxc.expansion.surface_expander import expand_surfaces
        expand_surfaces(module)
        return module.stages

    def test_mesh_pipeline_creates_mesh_stage(self):
        """Mesh shader pipeline should generate a mesh stage."""
        src = '''
        import brdf;
        geometry Mesh {
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
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        pipeline MeshPipe {
            mode: mesh_shader,
            geometry: Mesh,
            surface: Metal,
        }
        '''
        stages = self._expand_and_get_stages(src)
        stage_types = [s.stage_type for s in stages]
        assert "mesh" in stage_types

    def test_mesh_pipeline_creates_fragment(self):
        """Mesh shader pipeline should also generate a fragment stage."""
        src = '''
        import brdf;
        geometry Mesh {
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
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        pipeline MeshPipe {
            mode: mesh_shader,
            geometry: Mesh,
            surface: Metal,
        }
        '''
        stages = self._expand_and_get_stages(src)
        stage_types = [s.stage_type for s in stages]
        assert "fragment" in stage_types

    def test_mesh_pipeline_full_stages(self):
        """Full mesh pipeline should create mesh + fragment."""
        src = '''
        import brdf;
        geometry Mesh {
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
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        pipeline MeshPipe {
            mode: mesh_shader,
            geometry: Mesh,
            surface: Metal,
        }
        '''
        stages = self._expand_and_get_stages(src)
        stage_types = set(s.stage_type for s in stages)
        assert stage_types == {"mesh", "fragment"}

    def test_mesh_pipeline_no_vertex_stage(self):
        """Mesh shader pipeline should NOT generate a vertex stage."""
        src = '''
        import brdf;
        geometry Mesh {
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
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        pipeline MeshPipe {
            mode: mesh_shader,
            geometry: Mesh,
            surface: Metal,
        }
        '''
        stages = self._expand_and_get_stages(src)
        stage_types = [s.stage_type for s in stages]
        assert "vertex" not in stage_types

    def test_mesh_stage_has_storage_buffers(self):
        """Expanded mesh stage should have storage buffers for meshlet data."""
        src = '''
        import brdf;
        geometry Mesh {
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
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        pipeline MeshPipe {
            mode: mesh_shader,
            geometry: Mesh,
            surface: Metal,
        }
        '''
        stages = self._expand_and_get_stages(src)
        mesh = next(s for s in stages if s.stage_type == "mesh")
        sb_names = {sb.name for sb in mesh.storage_buffers}
        assert "meshlet_descriptors" in sb_names
        assert "meshlet_vertices" in sb_names
        assert "meshlet_triangles" in sb_names
        assert "positions" in sb_names

    def test_mesh_stage_has_uniforms(self):
        """Expanded mesh stage should inherit transform uniform from geometry."""
        src = '''
        import brdf;
        geometry Mesh {
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
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        pipeline MeshPipe {
            mode: mesh_shader,
            geometry: Mesh,
            surface: Metal,
        }
        '''
        stages = self._expand_and_get_stages(src)
        mesh = next(s for s in stages if s.stage_type == "mesh")
        assert len(mesh.uniforms) == 1
        assert mesh.uniforms[0].name == "MVP"

    def test_mesh_pipeline_expands_gltf_pbr_layered(self):
        """Parse gltf_pbr_layered.lux with GltfMesh pipeline, verify mesh + fragment stages."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        clear_type_aliases()
        module = parse_lux(src)
        module._defines = {'max_vertices': 64, 'max_primitives': 124, 'workgroup_size': 32}
        # Strip features (enable emission only)
        from luxc.features.evaluator import strip_features
        strip_features(module, {"has_emission"})
        from luxc.compiler import _resolve_imports
        _resolve_imports(module)
        from luxc.expansion.surface_expander import expand_surfaces
        expand_surfaces(module, pipeline_filter="GltfMesh")
        stage_types = set(s.stage_type for s in module.stages)
        assert "mesh" in stage_types
        assert "fragment" in stage_types
        assert "vertex" not in stage_types

    def test_rasterize_pipeline_still_works(self):
        """Rasterization pipeline (no mode) should not produce mesh stages."""
        src = '''
        import brdf;
        geometry Mesh {
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
        surface Metal {
            brdf: lambert(vec3(0.8)),
        }
        pipeline Raster {
            geometry: Mesh,
            surface: Metal,
        }
        '''
        stages = self._expand_and_get_stages(src)
        stage_types = [s.stage_type for s in stages]
        assert "vertex" in stage_types
        assert "fragment" in stage_types
        assert "mesh" not in stage_types


# =========================================================================
# Reflection Metadata Tests
# =========================================================================

class TestMeshReflection:
    """Test mesh shader reflection metadata."""

    def test_mesh_reflection_metadata(self):
        """Mesh stage reflection should include mesh_output metadata."""
        from luxc.analysis.type_checker import type_check
        from luxc.optimization.const_fold import constant_fold
        from luxc.analysis.layout_assigner import assign_layouts
        from luxc.codegen.reflection import generate_reflection

        clear_type_aliases()
        src = 'mesh { fn main() { } }'
        module = parse_lux(src)
        module._defines = {
            'max_vertices': 64,
            'max_primitives': 124,
            'workgroup_size': 32,
        }
        type_check(module)
        constant_fold(module)
        assign_layouts(module)
        stage = module.stages[0]
        reflection = generate_reflection(module, stage)

        assert "mesh_output" in reflection
        mo = reflection["mesh_output"]
        assert mo["max_vertices"] == 64
        assert mo["max_primitives"] == 124
        assert mo["workgroup_size"] == [32, 1, 1]
        assert mo["output_topology"] == "triangles"

    def test_task_reflection_metadata(self):
        """Task stage reflection should include task_shader metadata."""
        from luxc.analysis.type_checker import type_check
        from luxc.optimization.const_fold import constant_fold
        from luxc.analysis.layout_assigner import assign_layouts
        from luxc.codegen.reflection import generate_reflection

        clear_type_aliases()
        src = 'task { fn main() { } }'
        module = parse_lux(src)
        module._defines = {'workgroup_size': 32}
        type_check(module)
        constant_fold(module)
        assign_layouts(module)
        stage = module.stages[0]
        reflection = generate_reflection(module, stage)

        assert "task_shader" in reflection
        ts = reflection["task_shader"]
        assert ts["workgroup_size"] == [32, 1, 1]

    def test_mesh_reflection_execution_model(self):
        """Mesh stage reflection should have MeshEXT execution model."""
        from luxc.analysis.type_checker import type_check
        from luxc.optimization.const_fold import constant_fold
        from luxc.analysis.layout_assigner import assign_layouts
        from luxc.codegen.reflection import generate_reflection

        clear_type_aliases()
        src = 'mesh { fn main() { } }'
        module = parse_lux(src)
        module._defines = {}
        type_check(module)
        constant_fold(module)
        assign_layouts(module)
        stage = module.stages[0]
        reflection = generate_reflection(module, stage)

        assert reflection["execution_model"] == "MeshEXT"
        assert reflection["stage"] == "mesh"

    def test_mesh_reflection_custom_defines(self):
        """Custom --define values should be reflected in mesh_output metadata."""
        from luxc.analysis.type_checker import type_check
        from luxc.optimization.const_fold import constant_fold
        from luxc.analysis.layout_assigner import assign_layouts
        from luxc.codegen.reflection import generate_reflection

        clear_type_aliases()
        src = 'mesh { fn main() { } }'
        module = parse_lux(src)
        module._defines = {
            'max_vertices': 128,
            'max_primitives': 256,
            'workgroup_size': 64,
        }
        type_check(module)
        constant_fold(module)
        assign_layouts(module)
        stage = module.stages[0]
        reflection = generate_reflection(module, stage)

        mo = reflection["mesh_output"]
        assert mo["max_vertices"] == 128
        assert mo["max_primitives"] == 256
        assert mo["workgroup_size"] == [64, 1, 1]


# =========================================================================
# E2E Compilation Tests (require spirv-tools)
# =========================================================================

@requires_spirv_tools
class TestMeshEndToEnd:
    """End-to-end compilation tests for mesh shaders."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_minimal_mesh_compiles(self, tmp_path):
        """Minimal mesh shader should compile to .mesh.spv."""
        from luxc.compiler import compile_source
        src = 'mesh { fn main() { } }'
        compile_source(
            src, "minimal_mesh", tmp_path,
            validate=True,
            defines={'max_vertices': 64, 'max_primitives': 124, 'workgroup_size': 32},
        )
        assert (tmp_path / "minimal_mesh.mesh.spv").exists()

    def test_minimal_task_compiles(self, tmp_path):
        """Minimal task shader should compile to .task.spv."""
        from luxc.compiler import compile_source
        src = 'task { fn main() { } }'
        compile_source(
            src, "minimal_task", tmp_path,
            validate=True,
            defines={'workgroup_size': 32},
        )
        assert (tmp_path / "minimal_task.task.spv").exists()

    def test_mesh_with_builtins_compiles(self, tmp_path):
        """Mesh shader with local_invocation_index and workgroup_id should compile."""
        from luxc.compiler import compile_source
        src = '''
        mesh {
            fn main() {
                let tid: uint = local_invocation_index;
                let wg: uvec3 = workgroup_id;
            }
        }
        '''
        compile_source(
            src, "mesh_builtins", tmp_path,
            validate=True,
            defines={'max_vertices': 64, 'max_primitives': 124, 'workgroup_size': 32},
        )
        assert (tmp_path / "mesh_builtins.mesh.spv").exists()

    def test_mesh_manual_compiles(self, tmp_path):
        """Compile mesh_shader_manual.lux with --define flags, verify .mesh.spv exists."""
        from luxc.compiler import compile_source
        src = (EXAMPLES / "mesh_shader_manual.lux").read_text()
        compile_source(
            src, "mesh_shader_manual", tmp_path,
            source_dir=EXAMPLES,
            validate=True,
            defines={'max_vertices': 64, 'max_primitives': 124, 'workgroup_size': 32},
        )
        assert (tmp_path / "mesh_shader_manual.mesh.spv").exists()
        assert (tmp_path / "mesh_shader_manual.frag.spv").exists()

    def test_mesh_manual_reflection(self, tmp_path):
        """Compile mesh_shader_manual.lux and check .mesh.json has mesh_output metadata."""
        import json
        from luxc.compiler import compile_source
        src = (EXAMPLES / "mesh_shader_manual.lux").read_text()
        compile_source(
            src, "mesh_shader_manual", tmp_path,
            source_dir=EXAMPLES,
            validate=True,
            emit_reflection=True,
            defines={'max_vertices': 64, 'max_primitives': 124, 'workgroup_size': 32},
        )
        json_path = tmp_path / "mesh_shader_manual.mesh.json"
        assert json_path.exists()
        meta = json.loads(json_path.read_text())
        assert "mesh_output" in meta
        assert meta["mesh_output"]["max_vertices"] == 64
        assert meta["mesh_output"]["max_primitives"] == 124
        assert meta["mesh_output"]["workgroup_size"] == [32, 1, 1]
        assert meta["mesh_output"]["output_topology"] == "triangles"
        assert meta["execution_model"] == "MeshEXT"

    def test_mesh_fragment_compiles(self, tmp_path):
        """Mesh + fragment pipeline should produce both .mesh.spv and .frag.spv."""
        from luxc.compiler import compile_source
        src = '''
        mesh {
            storage_buffer positions: vec4;
            out frag_color: vec3;
            fn main() {
                let tid: uint = local_invocation_index;
                set_mesh_outputs(3, 1);
                if (tid < 3) {
                    let pos: vec4 = positions[tid];
                    gl_MeshVerticesEXT[tid] = vec4(pos.xyz, 1.0);
                    frag_color[tid] = pos.xyz;
                }
                if (tid < 1) {
                    gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
                }
            }
        }
        fragment {
            in frag_color: vec3;
            out color: vec4;
            fn main() {
                color = vec4(frag_color, 1.0);
            }
        }
        '''
        compile_source(
            src, "mesh_frag", tmp_path,
            validate=True,
            defines={'max_vertices': 64, 'max_primitives': 124, 'workgroup_size': 32},
        )
        assert (tmp_path / "mesh_frag.mesh.spv").exists()
        assert (tmp_path / "mesh_frag.frag.spv").exists()

    def test_gltf_mesh_pipeline_compiles(self, tmp_path):
        """Full compilation of GltfMesh pipeline from gltf_pbr_layered.lux."""
        from luxc.compiler import compile_source
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        compile_source(
            src, "gltf_pbr_layered", tmp_path,
            source_dir=EXAMPLES,
            validate=True,
            pipeline="GltfMesh",
            features={"has_emission"},
            defines={'max_vertices': 64, 'max_primitives': 124, 'workgroup_size': 32},
        )
        assert (tmp_path / "gltf_pbr_layered+emission.mesh.spv").exists()
        assert (tmp_path / "gltf_pbr_layered+emission.frag.spv").exists()

    def test_gltf_mesh_pipeline_reflection(self, tmp_path):
        """GltfMesh pipeline from gltf_pbr_layered.lux produces correct reflection."""
        import json
        from luxc.compiler import compile_source
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        compile_source(
            src, "gltf_pbr_layered", tmp_path,
            source_dir=EXAMPLES,
            validate=True,
            emit_reflection=True,
            pipeline="GltfMesh",
            features={"has_emission"},
            defines={'max_vertices': 64, 'max_primitives': 124, 'workgroup_size': 32},
        )
        # Check mesh reflection
        mesh_json = tmp_path / "gltf_pbr_layered+emission.mesh.json"
        assert mesh_json.exists()
        mesh_meta = json.loads(mesh_json.read_text())
        assert mesh_meta["stage"] == "mesh"
        assert "mesh_output" in mesh_meta

        # Check fragment reflection
        frag_json = tmp_path / "gltf_pbr_layered+emission.frag.json"
        assert frag_json.exists()
        frag_meta = json.loads(frag_json.read_text())
        assert frag_meta["stage"] == "fragment"
