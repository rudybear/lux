"""P23 Compute Shader tests.

Tests cover:
- Compute grammar parsing (stage type, storage buffers, uniforms, push constants)
- Compute SPIR-V codegen (GLCompute, LocalSize, builtins, barrier, RW SSBOs)
- Reflection metadata (execution model, workgroup_size, storage buffers)
- End-to-end compilation of compute examples
"""

import pytest
import shutil
from pathlib import Path
from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import Module, StageBlock, StorageBufferDecl
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

class TestComputeGrammar:
    """Test that compute stage blocks parse correctly."""

    def test_compute_stage_type_parses(self):
        """A minimal compute stage block should parse with stage_type='compute'."""
        src = 'compute { fn main() { } }'
        module = parse_lux(src)
        assert len(module.stages) == 1
        assert module.stages[0].stage_type == "compute"

    def test_compute_with_storage_buffer(self):
        """Compute stage should support storage_buffer declarations."""
        src = '''
        compute {
            storage_buffer data: scalar;
            fn main() { }
        }
        '''
        module = parse_lux(src)
        stage = module.stages[0]
        assert len(stage.storage_buffers) == 1
        sb = stage.storage_buffers[0]
        assert isinstance(sb, StorageBufferDecl)
        assert sb.name == "data"
        assert sb.element_type == "scalar"

    def test_compute_with_uniform_and_push(self):
        """Compute stage should support uniform and push constant blocks."""
        src = '''
        compute {
            uniform Params {
                scale: scalar,
            }
            push Constants {
                offset: uint,
            }
            fn main() { }
        }
        '''
        module = parse_lux(src)
        stage = module.stages[0]
        assert stage.stage_type == "compute"
        assert len(stage.uniforms) == 1
        assert len(stage.push_constants) == 1

    def test_compute_with_multiple_storage_buffers(self):
        """Compute stage should support multiple storage buffers."""
        src = '''
        compute {
            storage_buffer input_a: scalar;
            storage_buffer input_b: scalar;
            storage_buffer output_c: scalar;
            fn main() { }
        }
        '''
        module = parse_lux(src)
        stage = module.stages[0]
        assert len(stage.storage_buffers) == 3
        names = [sb.name for sb in stage.storage_buffers]
        assert names == ["input_a", "input_b", "output_c"]


# =========================================================================
# SPIR-V Codegen Tests
# =========================================================================

class TestComputeCodegen:
    """Test compute shader SPIR-V code generation."""

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

    def test_compute_execution_model(self):
        """Compute shader should use GLCompute execution model."""
        src = 'compute { fn main() { } }'
        asm = self._get_spirv(src)
        assert "OpEntryPoint GLCompute %main" in asm

    def test_compute_local_size_1d(self):
        """Compute shader should emit LocalSize with 1D workgroup size."""
        src = 'compute { fn main() { } }'
        defines = {'workgroup_size': 64}
        asm = self._get_spirv(src, defines=defines)
        assert "OpExecutionMode %main LocalSize 64 1 1" in asm

    def test_compute_local_size_3d(self):
        """Compute shader should support 3D workgroup size via defines."""
        src = 'compute { fn main() { } }'
        defines = {'workgroup_size_x': 16, 'workgroup_size_y': 16, 'workgroup_size_z': 1}
        asm = self._get_spirv(src, defines=defines)
        assert "OpExecutionMode %main LocalSize 16 16 1" in asm

    def test_compute_builtin_global_invocation_id(self):
        """Compute shader should declare GlobalInvocationId builtin."""
        src = '''
        compute {
            fn main() {
                let gid: uvec3 = global_invocation_id;
            }
        }
        '''
        asm = self._get_spirv(src)
        assert "BuiltIn GlobalInvocationId" in asm

    def test_compute_builtin_local_invocation_index(self):
        """Compute shader should declare LocalInvocationIndex builtin."""
        src = '''
        compute {
            fn main() {
                let lid: uint = local_invocation_index;
            }
        }
        '''
        asm = self._get_spirv(src)
        assert "BuiltIn LocalInvocationIndex" in asm

    def test_compute_builtin_workgroup_id(self):
        """Compute shader should declare WorkgroupId builtin."""
        src = '''
        compute {
            fn main() {
                let wg: uvec3 = workgroup_id;
            }
        }
        '''
        asm = self._get_spirv(src)
        assert "BuiltIn WorkgroupId" in asm

    def test_compute_no_origin_upper_left(self):
        """Compute shaders should NOT emit OriginUpperLeft (fragment-only)."""
        src = 'compute { fn main() { } }'
        asm = self._get_spirv(src)
        assert "OriginUpperLeft" not in asm

    def test_compute_no_mesh_capability(self):
        """Compute shaders should NOT emit MeshShadingEXT capability."""
        src = 'compute { fn main() { } }'
        asm = self._get_spirv(src)
        assert "MeshShadingEXT" not in asm
        assert "SPV_EXT_mesh_shader" not in asm

    def test_barrier_codegen(self):
        """barrier() should emit OpControlBarrier."""
        src = '''
        compute {
            fn main() {
                barrier();
            }
        }
        '''
        asm = self._get_spirv(src)
        assert "OpControlBarrier" in asm

    def test_storage_buffer_no_nonwritable_in_compute(self):
        """Compute shader SSBOs should NOT have NonWritable decoration."""
        src = '''
        compute {
            storage_buffer data: scalar;
            fn main() { }
        }
        '''
        asm = self._get_spirv(src)
        assert "NonWritable" not in asm

    def test_storage_buffer_write_generates_access_chain(self):
        """Compute shader should generate OpAccessChain for buffer[i] = value."""
        src = '''
        compute {
            storage_buffer data: scalar;
            push params {
                count: scalar,
            }
            fn main() {
                let gid: scalar = global_invocation_id.x;
                data[gid] = 1.0;
            }
        }
        '''
        asm = self._get_spirv(src)
        # Should have OpAccessChain for storage buffer write
        assert "OpAccessChain" in asm
        assert "OpStore" in asm


# =========================================================================
# Reflection Tests
# =========================================================================

class TestComputeReflection:
    """Test compute shader reflection metadata."""

    def _get_reflection(self, src: str, defines: dict = None) -> dict:
        """Parse and generate reflection for a compute shader."""
        from luxc.analysis.type_checker import type_check
        from luxc.optimization.const_fold import constant_fold
        from luxc.analysis.layout_assigner import assign_layouts
        from luxc.codegen.reflection import generate_reflection

        clear_type_aliases()
        module = parse_lux(src)
        module._defines = defines or {}
        type_check(module)
        constant_fold(module)
        assign_layouts(module)
        stage = module.stages[0]
        return generate_reflection(module, stage, source_name="test.lux")

    def test_compute_reflection_exec_model(self):
        """Reflection should report GLCompute execution model."""
        src = 'compute { fn main() { } }'
        ref = self._get_reflection(src)
        assert ref["execution_model"] == "GLCompute"
        assert ref["stage"] == "compute"

    def test_compute_reflection_workgroup_size(self):
        """Reflection should include compute workgroup_size."""
        src = 'compute { fn main() { } }'
        ref = self._get_reflection(src, defines={'workgroup_size': 64})
        assert "compute" in ref
        assert ref["compute"]["workgroup_size"] == [64, 1, 1]

    def test_compute_reflection_workgroup_size_3d(self):
        """Reflection should support 3D workgroup size."""
        src = 'compute { fn main() { } }'
        ref = self._get_reflection(src, defines={
            'workgroup_size_x': 8, 'workgroup_size_y': 8, 'workgroup_size_z': 4,
        })
        assert ref["compute"]["workgroup_size"] == [8, 8, 4]

    def test_compute_reflection_storage_buffers(self):
        """Reflection should include storage buffer metadata."""
        src = '''
        compute {
            storage_buffer data: scalar;
            fn main() { }
        }
        '''
        ref = self._get_reflection(src)
        ds = ref["descriptor_sets"]
        # Should have at least one descriptor set with a storage_buffer
        found = False
        for bindings in ds.values():
            for b in bindings:
                if b["type"] == "storage_buffer" and b["name"] == "data":
                    found = True
                    assert b["element_type"] == "scalar"
        assert found, "Storage buffer 'data' not found in reflection"


# =========================================================================
# End-to-End Tests
# =========================================================================

class TestComputeEndToEnd:
    """End-to-end tests requiring spirv-as/spirv-val."""

    def _compile(self, src: str, stem: str = "test", defines: dict = None,
                 emit_asm: bool = True) -> Path:
        """Compile source to SPIR-V, return output dir."""
        import tempfile
        from luxc.compiler import compile_source

        clear_type_aliases()
        out = Path(tempfile.mkdtemp())
        compile_source(
            src, stem, out,
            emit_asm=emit_asm,
            validate=_has_spirv_tools(),
            emit_reflection=True,
            defines=defines or {},
        )
        return out

    @requires_spirv_tools
    def test_minimal_compute_compiles(self):
        """Minimal compute shader should compile to .comp.spv."""
        src = 'compute { fn main() { } }'
        out = self._compile(src, defines={'workgroup_size': 64})
        assert (out / "test.comp.spv").exists()
        assert (out / "test.comp.spvasm").exists()
        shutil.rmtree(out)

    @requires_spirv_tools
    def test_compute_with_ssbo_compiles(self):
        """Compute shader with storage buffer should compile."""
        src = '''
        compute {
            storage_buffer data: scalar;
            fn main() {
                let gid: scalar = global_invocation_id.x;
            }
        }
        '''
        out = self._compile(src, defines={'workgroup_size': 64})
        assert (out / "test.comp.spv").exists()
        shutil.rmtree(out)

    @requires_spirv_tools
    def test_compute_with_barrier_compiles(self):
        """Compute shader with barrier() should compile and validate."""
        src = '''
        compute {
            fn main() {
                barrier();
            }
        }
        '''
        out = self._compile(src, defines={'workgroup_size': 64})
        assert (out / "test.comp.spv").exists()
        shutil.rmtree(out)

    @requires_spirv_tools
    def test_compute_reflection_json(self):
        """Compute shader should produce .comp.json with correct metadata."""
        import json
        src = 'compute { fn main() { } }'
        out = self._compile(src, defines={'workgroup_size': 128})
        json_path = out / "test.comp.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["execution_model"] == "GLCompute"
        assert data["compute"]["workgroup_size"] == [128, 1, 1]
        shutil.rmtree(out)

    @requires_spirv_tools
    def test_compute_double_example(self):
        """examples/compute_double.lux should compile successfully."""
        src = (EXAMPLES / "compute_double.lux").read_text()
        out = self._compile(src, stem="compute_double", defines={'workgroup_size': 64})
        assert (out / "compute_double.comp.spv").exists()
        shutil.rmtree(out)

    @requires_spirv_tools
    def test_compute_saxpy_example(self):
        """examples/compute_saxpy.lux should compile successfully."""
        src = (EXAMPLES / "compute_saxpy.lux").read_text()
        out = self._compile(src, stem="compute_saxpy", defines={'workgroup_size': 256})
        assert (out / "compute_saxpy.comp.spv").exists()
        shutil.rmtree(out)

    def test_vertex_shader_unaffected(self):
        """Vertex shaders should still work correctly after compute changes."""
        src = '''
        vertex {
            in position: vec3;
            fn main() {
                builtin_position = vec4(position, 1.0);
            }
        }
        '''
        from luxc.analysis.type_checker import type_check
        from luxc.optimization.const_fold import constant_fold
        from luxc.analysis.layout_assigner import assign_layouts
        from luxc.codegen.spirv_builder import generate_spirv

        clear_type_aliases()
        module = parse_lux(src)
        module._defines = {}
        type_check(module)
        constant_fold(module)
        assign_layouts(module)
        asm = generate_spirv(module, module.stages[0])
        assert "OpEntryPoint Vertex" in asm
        assert "GLCompute" not in asm
        assert "MeshShadingEXT" not in asm
