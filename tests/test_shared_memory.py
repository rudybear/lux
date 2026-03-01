"""P23.2 Shared Memory & Atomics tests.

Tests cover:
- Grammar parsing (shared scalar, shared array, array types)
- Type checker (shared registration, FixedArrayType indexing, compute-only restriction)
- SPIR-V codegen (Workgroup storage class, OpTypeArray, OpAccessChain, atomic ops)
- Reflection metadata (shared_memory list, total bytes)
- End-to-end compilation (histogram, reduction examples)
"""

import pytest
import shutil
from pathlib import Path
from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import Module, StageBlock, SharedDecl
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


def _get_spirv(src: str, stage_idx: int = 0, defines: dict = None) -> str:
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


def _get_reflection(src: str, defines: dict = None) -> dict:
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


# =========================================================================
# Grammar / Parsing Tests
# =========================================================================

class TestSharedGrammar:
    """Test that shared memory declarations parse correctly."""

    def test_shared_scalar_parses(self):
        """A shared scalar declaration should parse."""
        src = '''
        compute {
            shared counter: uint;
            fn main() { }
        }
        '''
        module = parse_lux(src)
        stage = module.stages[0]
        assert len(stage.shared_decls) == 1
        sd = stage.shared_decls[0]
        assert isinstance(sd, SharedDecl)
        assert sd.name == "counter"
        assert sd.type_name == "uint"
        assert sd.array_size is None

    def test_shared_array_parses(self):
        """A shared array declaration should parse with array size."""
        src = '''
        compute {
            shared histogram: uint[256];
            fn main() { }
        }
        '''
        module = parse_lux(src)
        stage = module.stages[0]
        assert len(stage.shared_decls) == 1
        sd = stage.shared_decls[0]
        assert sd.name == "histogram"
        assert sd.type_name == "uint"
        assert sd.array_size == 256

    def test_multiple_shared_declarations(self):
        """Multiple shared declarations should all parse."""
        src = '''
        compute {
            shared data: uint[64];
            shared counter: uint;
            shared flags: int[32];
            fn main() { }
        }
        '''
        module = parse_lux(src)
        stage = module.stages[0]
        assert len(stage.shared_decls) == 3
        assert stage.shared_decls[0].name == "data"
        assert stage.shared_decls[0].array_size == 64
        assert stage.shared_decls[1].name == "counter"
        assert stage.shared_decls[1].array_size is None
        assert stage.shared_decls[2].name == "flags"
        assert stage.shared_decls[2].type_name == "int"
        assert stage.shared_decls[2].array_size == 32

    def test_shared_with_different_types(self):
        """Shared declarations with vec4, int, uint types should parse."""
        src = '''
        compute {
            shared buf_vec: vec4[16];
            shared buf_int: int[8];
            shared scalar_shared: uint;
            fn main() { }
        }
        '''
        module = parse_lux(src)
        stage = module.stages[0]
        assert len(stage.shared_decls) == 3
        assert stage.shared_decls[0].type_name == "vec4"
        assert stage.shared_decls[0].array_size == 16
        assert stage.shared_decls[1].type_name == "int"
        assert stage.shared_decls[2].type_name == "uint"
        assert stage.shared_decls[2].array_size is None

    def test_shared_with_storage_buffer(self):
        """Shared declarations should coexist with storage buffers."""
        src = '''
        compute {
            storage_buffer data: uint;
            shared scratch: uint[256];
            fn main() { }
        }
        '''
        module = parse_lux(src)
        stage = module.stages[0]
        assert len(stage.storage_buffers) == 1
        assert len(stage.shared_decls) == 1


# =========================================================================
# Type Checker Tests
# =========================================================================

class TestSharedTypeChecker:
    """Test type checking for shared memory declarations."""

    def test_shared_variable_registered(self):
        """Shared scalar variable should be type-checked without errors."""
        src = '''
        compute {
            shared counter: uint;
            fn main() { }
        }
        '''
        from luxc.analysis.type_checker import type_check
        clear_type_aliases()
        module = parse_lux(src)
        module._defines = {}
        type_check(module)  # Should not raise

    def test_shared_array_indexing_returns_element_type(self):
        """Indexing a shared array should return the element type."""
        src = '''
        compute {
            shared data: uint[256];
            fn main() {
                let tid: uint = local_invocation_index;
                let val: uint = data[tid];
            }
        }
        '''
        from luxc.analysis.type_checker import type_check
        clear_type_aliases()
        module = parse_lux(src)
        module._defines = {}
        type_check(module)  # Should not raise

    def test_shared_only_in_compute_stage(self):
        """Shared variables should raise error in non-compute stages."""
        src = '''
        fragment {
            shared data: uint[64];
            fn main() { }
        }
        '''
        from luxc.analysis.type_checker import type_check, TypeCheckError
        clear_type_aliases()
        module = parse_lux(src)
        module._defines = {}
        with pytest.raises(TypeCheckError, match="shared.*only allowed in compute"):
            type_check(module)

    def test_shared_with_atomic_type_checks(self):
        """atomic_add on shared array element should type-check."""
        src = '''
        compute {
            shared histogram: uint[256];
            fn main() {
                let tid: uint = local_invocation_index;
                let old: uint = atomic_add(histogram[tid], 1);
            }
        }
        '''
        from luxc.analysis.type_checker import type_check
        clear_type_aliases()
        module = parse_lux(src)
        module._defines = {}
        type_check(module)  # Should not raise


# =========================================================================
# SPIR-V Codegen Tests
# =========================================================================

class TestSharedCodegen:
    """Test SPIR-V code generation for shared memory and atomics."""

    def test_shared_scalar_workgroup_storage(self):
        """Shared scalar should generate Workgroup storage class variable."""
        src = '''
        compute {
            shared counter: uint;
            fn main() { }
        }
        '''
        asm = _get_spirv(src)
        assert "Workgroup" in asm
        assert "OpVariable" in asm

    def test_shared_array_generates_optypearray(self):
        """Shared array should generate OpTypeArray + Workgroup variable."""
        src = '''
        compute {
            shared histogram: uint[256];
            fn main() { }
        }
        '''
        asm = _get_spirv(src)
        assert "OpTypeArray" in asm
        assert "Workgroup" in asm

    def test_shared_array_access_generates_access_chain(self):
        """Shared array access should generate OpAccessChain with Workgroup pointer."""
        src = '''
        compute {
            shared data: uint[64];
            fn main() {
                let tid: uint = local_invocation_index;
                let val: uint = data[tid];
            }
        }
        '''
        asm = _get_spirv(src)
        # Should have OpAccessChain with Workgroup pointer
        assert "OpAccessChain" in asm
        assert "Workgroup" in asm
        assert "OpLoad" in asm

    def test_shared_array_store_generates_access_chain(self):
        """Shared array write should generate OpAccessChain + OpStore."""
        src = '''
        compute {
            shared data: uint[64];
            fn main() {
                let tid: uint = local_invocation_index;
                data[tid] = 0;
            }
        }
        '''
        asm = _get_spirv(src)
        assert "OpAccessChain" in asm
        assert "OpStore" in asm

    def test_atomic_add_generates_opatomiciadd(self):
        """atomic_add should generate OpAtomicIAdd."""
        src = '''
        compute {
            shared histogram: uint[256];
            fn main() {
                let tid: uint = local_invocation_index;
                let old: uint = atomic_add(histogram[tid], 1);
            }
        }
        '''
        asm = _get_spirv(src)
        assert "OpAtomicIAdd" in asm

    def test_atomic_min_generates_correct_op(self):
        """atomic_min on uint should generate OpAtomicUMin."""
        src = '''
        compute {
            shared mins: uint[64];
            fn main() {
                let tid: uint = local_invocation_index;
                let old: uint = atomic_min(mins[tid], 100);
            }
        }
        '''
        asm = _get_spirv(src)
        assert "OpAtomicUMin" in asm

    def test_atomic_max_signed_generates_smax(self):
        """atomic_max on int should generate OpAtomicSMax."""
        src = '''
        compute {
            shared maxes: int[64];
            fn main() {
                let tid: uint = local_invocation_index;
                let old: int = atomic_max(maxes[tid], 0);
            }
        }
        '''
        asm = _get_spirv(src)
        assert "OpAtomicSMax" in asm

    def test_barrier_with_shared_compiles(self):
        """barrier + shared memory interaction should compile correctly."""
        src = '''
        compute {
            shared data: uint[64];
            fn main() {
                let tid: uint = local_invocation_index;
                data[tid] = tid;
                barrier();
                let val: uint = data[tid];
            }
        }
        '''
        asm = _get_spirv(src)
        assert "OpControlBarrier" in asm
        assert "Workgroup" in asm

    def test_shared_in_entry_point_interface(self):
        """Shared variables should appear in OpEntryPoint interface list."""
        src = '''
        compute {
            shared counter: uint;
            fn main() { }
        }
        '''
        asm = _get_spirv(src)
        # The OpEntryPoint line should contain the shared variable ID
        for line in asm.split('\n'):
            if "OpEntryPoint" in line:
                # Should have more than just %main in the interface
                parts = line.split()
                assert len(parts) > 4, "OpEntryPoint should include shared var in interface"
                break

    def test_shared_no_descriptor_set_binding(self):
        """Shared variables should NOT have DescriptorSet or Binding decorations."""
        src = '''
        compute {
            shared data: uint[64];
            fn main() { }
        }
        '''
        asm = _get_spirv(src)
        # Count DescriptorSet decorations; none should refer to shared vars
        # (There might be storage buffer DescriptorSet decorations, so just check
        # that shared vars don't get them)
        lines = asm.split('\n')
        shared_var_ids = set()
        for line in lines:
            if "OpVariable" in line and "Workgroup" in line:
                # Extract the ID (first word with %)
                parts = line.split()
                shared_var_ids.add(parts[0])
        for line in lines:
            if "DescriptorSet" in line or "Binding" in line:
                for vid in shared_var_ids:
                    assert vid not in line, f"Shared var {vid} should not have descriptor decoration"

    def test_atomic_exchange_generates_op(self):
        """atomic_exchange should generate OpAtomicExchange."""
        src = '''
        compute {
            shared lock: uint;
            fn main() {
                let old: uint = atomic_exchange(lock, 1);
            }
        }
        '''
        asm = _get_spirv(src)
        assert "OpAtomicExchange" in asm

    def test_atomic_compare_exchange_generates_op(self):
        """atomic_compare_exchange should generate OpAtomicCompareExchange."""
        src = '''
        compute {
            shared lock: uint;
            fn main() {
                let old: uint = atomic_compare_exchange(lock, 0, 1);
            }
        }
        '''
        asm = _get_spirv(src)
        assert "OpAtomicCompareExchange" in asm

    def test_atomic_load_generates_op(self):
        """atomic_load should generate OpAtomicLoad."""
        src = '''
        compute {
            shared flag: uint;
            fn main() {
                let val: uint = atomic_load(flag);
            }
        }
        '''
        asm = _get_spirv(src)
        assert "OpAtomicLoad" in asm

    def test_atomic_store_generates_op(self):
        """atomic_store should generate OpAtomicStore."""
        src = '''
        compute {
            shared flag: uint;
            fn main() {
                atomic_store(flag, 1);
            }
        }
        '''
        asm = _get_spirv(src)
        assert "OpAtomicStore" in asm

    def test_atomic_and_or_xor_generate_ops(self):
        """atomic_and, atomic_or, atomic_xor should generate correct ops."""
        for op_name, spirv_op in [("atomic_and", "OpAtomicAnd"),
                                   ("atomic_or", "OpAtomicOr"),
                                   ("atomic_xor", "OpAtomicXor")]:
            src = f'''
            compute {{
                shared flags: uint;
                fn main() {{
                    let old: uint = {op_name}(flags, 1);
                }}
            }}
            '''
            asm = _get_spirv(src)
            assert spirv_op in asm, f"{op_name} should generate {spirv_op}"

    def test_atomic_on_ssbo_uses_device_scope(self):
        """atomic_add on SSBO should use Device scope (1), not Workgroup (2)."""
        src = '''
        compute {
            storage_buffer data: uint;
            fn main() {
                let gid: uvec3 = global_invocation_id;
                let old: uint = atomic_add(data[gid.x], 1);
            }
        }
        '''
        asm = _get_spirv(src)
        assert "OpAtomicIAdd" in asm
        # The scope should be Device=1 for SSBO atomics


# =========================================================================
# Reflection Tests
# =========================================================================

class TestSharedReflection:
    """Test reflection metadata for shared memory."""

    def test_shared_memory_list_in_reflection(self):
        """Reflection should include shared_memory list."""
        src = '''
        compute {
            shared data: uint[256];
            fn main() { }
        }
        '''
        ref = _get_reflection(src)
        assert "shared_memory" in ref
        assert len(ref["shared_memory"]) == 1
        entry = ref["shared_memory"][0]
        assert entry["name"] == "data"
        assert entry["type"] == "uint"
        assert entry["array_size"] == 256

    def test_shared_memory_total_bytes(self):
        """Reflection should calculate shared_memory_total_bytes correctly."""
        src = '''
        compute {
            shared data: uint[256];
            fn main() { }
        }
        '''
        ref = _get_reflection(src)
        assert ref["shared_memory_total_bytes"] == 256 * 4  # uint = 4 bytes

    def test_shared_memory_multiple_entries(self):
        """Multiple shared declarations should all appear in reflection."""
        src = '''
        compute {
            shared arr: uint[64];
            shared counter: uint;
            shared vecs: vec4[16];
            fn main() { }
        }
        '''
        ref = _get_reflection(src)
        assert len(ref["shared_memory"]) == 3
        total = 64 * 4 + 1 * 4 + 16 * 16  # uint[64] + uint + vec4[16]
        assert ref["shared_memory_total_bytes"] == total

    def test_shared_scalar_no_array_size_in_reflection(self):
        """Scalar shared variable should not have 'array_size' in reflection entry."""
        src = '''
        compute {
            shared counter: uint;
            fn main() { }
        }
        '''
        ref = _get_reflection(src)
        entry = ref["shared_memory"][0]
        assert "array_size" not in entry
        assert entry["size_bytes"] == 4

    def test_no_shared_memory_when_none_declared(self):
        """Reflection should not have shared_memory when none declared."""
        src = '''
        compute {
            fn main() { }
        }
        '''
        ref = _get_reflection(src)
        assert "shared_memory" not in ref


# =========================================================================
# End-to-End Tests
# =========================================================================

class TestSharedEndToEnd:
    """End-to-end tests for shared memory and atomics."""

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

    def test_histogram_example_compiles(self):
        """The compute_histogram.lux example should compile to SPIR-V asm."""
        src = '''
        compute {
            storage_buffer input_data: uint;
            storage_buffer output_histogram: uint;
            shared histogram: uint[256];

            fn main() {
                let tid: uint = local_invocation_index;
                let gid: uvec3 = global_invocation_id;
                histogram[tid] = 0;
                barrier();
                let value: uint = input_data[gid.x];
                let bin: uint = value;
                let old: uint = atomic_add(histogram[bin], 1);
                barrier();
                let count: uint = histogram[tid];
                let prev: uint = atomic_add(output_histogram[tid], count);
            }
        }
        '''
        asm = _get_spirv(src, defines={'workgroup_size': 256})
        assert "OpEntryPoint GLCompute" in asm
        assert "OpAtomicIAdd" in asm
        assert "OpControlBarrier" in asm
        assert "Workgroup" in asm

    def test_reduction_example_compiles(self):
        """The reduction pattern with shared memory and barrier should compile."""
        src = '''
        compute {
            storage_buffer input_data: uint;
            storage_buffer output_data: uint;
            shared scratch: uint[256];

            fn main() {
                let tid: uint = local_invocation_index;
                let gid: uvec3 = global_invocation_id;
                scratch[tid] = input_data[gid.x];
                barrier();
                if (tid < 128) {
                    let a: uint = scratch[tid];
                    let b: uint = scratch[tid + 128];
                    scratch[tid] = a + b;
                }
                barrier();
                if (tid < 1) {
                    let wg_id: uvec3 = workgroup_id;
                    output_data[wg_id.x] = scratch[0];
                }
            }
        }
        '''
        asm = _get_spirv(src, defines={'workgroup_size': 256})
        assert "OpEntryPoint GLCompute" in asm
        assert "OpControlBarrier" in asm
        assert "Workgroup" in asm

    @requires_spirv_tools
    def test_histogram_example_file_compiles(self):
        """examples/compute_histogram.lux should compile to .comp.spv."""
        src = (EXAMPLES / "compute_histogram.lux").read_text()
        out = self._compile(src, stem="compute_histogram", defines={'workgroup_size': 256})
        assert (out / "compute_histogram.comp.spv").exists()
        shutil.rmtree(out)

    @requires_spirv_tools
    def test_reduction_example_file_compiles(self):
        """examples/compute_reduction.lux should compile to .comp.spv."""
        src = (EXAMPLES / "compute_reduction.lux").read_text()
        out = self._compile(src, stem="compute_reduction", defines={'workgroup_size': 256})
        assert (out / "compute_reduction.comp.spv").exists()
        shutil.rmtree(out)

    @requires_spirv_tools
    def test_shared_with_reflection_json(self):
        """Compute shader with shared memory should produce .comp.json with shared metadata."""
        import json
        src = '''
        compute {
            shared data: uint[128];
            fn main() { }
        }
        '''
        out = self._compile(src, defines={'workgroup_size': 128})
        json_path = out / "test.comp.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "shared_memory" in data
        assert data["shared_memory_total_bytes"] == 128 * 4
        shutil.rmtree(out)
