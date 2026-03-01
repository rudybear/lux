"""P25 Phase A: spirv-opt integration tests.

Tests cover:
- spirv-opt availability detection
- Optimized output is smaller than unoptimized
- Optimized .spv still passes spirv-val
- OpLoad count decreases with optimization (mem2reg)
- All shader stages (vertex, fragment, compute) optimize correctly
- Loops optimize correctly
- Shared memory + atomics optimize correctly
"""

import pytest
import shutil
import subprocess
import tempfile
from pathlib import Path

from luxc.compiler import compile_source
from luxc.builtins.types import clear_type_aliases
from luxc.codegen.spv_assembler import run_spirv_opt


def _has_spirv_tools() -> bool:
    try:
        subprocess.run(["spirv-as", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


def _has_spirv_opt() -> bool:
    return shutil.which("spirv-opt") is not None


requires_spirv_tools = pytest.mark.skipif(
    not _has_spirv_tools(), reason="spirv-as/spirv-val not found on PATH"
)

requires_spirv_opt = pytest.mark.skipif(
    not _has_spirv_opt(), reason="spirv-opt not found on PATH"
)


def _compile(src: str, stem: str = "test", optimize: bool = False,
             defines: dict | None = None, emit_asm: bool = True) -> Path:
    """Compile source to SPIR-V, return output dir."""
    clear_type_aliases()
    out = Path(tempfile.mkdtemp())
    compile_source(
        src, stem, out,
        emit_asm=emit_asm,
        validate=True,
        emit_reflection=False,
        defines=defines or {},
        optimize=optimize,
    )
    return out


def _spv_size(out_dir: Path, suffix: str) -> int:
    """Return the file size of the .spv file with the given suffix."""
    spv_files = list(out_dir.glob(f"*.{suffix}.spv"))
    assert len(spv_files) == 1, f"Expected 1 .{suffix}.spv file, got {len(spv_files)}"
    return spv_files[0].stat().st_size


def _disassemble(spv_path: Path) -> str:
    """Disassemble a .spv binary back to SPIR-V text."""
    result = subprocess.run(
        ["spirv-dis", str(spv_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"spirv-dis failed: {result.stderr}"
    return result.stdout


# =========================================================================
# Test: spirv-opt availability
# =========================================================================

class TestSpirvOptAvailability:
    def test_spirv_opt_available(self):
        """Check that spirv-opt is on PATH (informational)."""
        available = shutil.which("spirv-opt") is not None
        if not available:
            pytest.skip("spirv-opt not found on PATH — install Vulkan SDK")
        assert available


# =========================================================================
# Test: Optimized output is smaller
# =========================================================================

@requires_spirv_tools
@requires_spirv_opt
class TestOptimizeFlagReducesOutput:
    def test_optimize_flag_reduces_output(self):
        """Compiling with optimize=True should produce a smaller .spv file."""
        src = '''
        fragment {
            in uv: vec2;
            out color: vec4;
            fn main() {
                let x: scalar = uv.x;
                let y: scalar = uv.y;
                let r: scalar = sin(x) * cos(y);
                let g: scalar = cos(x) * sin(y);
                let b: scalar = (r + g) * 0.5;
                color = vec4(r, g, b, 1.0);
            }
        }
        '''
        out_normal = _compile(src, stem="test_normal", optimize=False)
        out_opt = _compile(src, stem="test_opt", optimize=True)

        size_normal = _spv_size(out_normal, "frag")
        size_opt = _spv_size(out_opt, "frag")

        assert size_opt < size_normal, (
            f"Optimized size ({size_opt}) should be smaller than "
            f"unoptimized ({size_normal})"
        )

        shutil.rmtree(out_normal)
        shutil.rmtree(out_opt)


# =========================================================================
# Test: Optimized .spv still passes validation
# =========================================================================

@requires_spirv_tools
@requires_spirv_opt
class TestOptimizeFlagStillValidates:
    def test_optimize_flag_still_validates(self):
        """Optimized .spv should pass spirv-val."""
        src = '''
        fragment {
            in uv: vec2;
            out color: vec4;
            uniform Params { time: scalar }
            fn main() {
                let t: scalar = time;
                let r: scalar = sin(uv.x + t);
                let g: scalar = cos(uv.y + t);
                color = vec4(r, g, 0.5, 1.0);
            }
        }
        '''
        out = _compile(src, optimize=True)
        spv_files = list(out.glob("*.frag.spv"))
        assert len(spv_files) == 1

        result = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.2", str(spv_files[0])],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"spirv-val failed: {result.stderr}"

        shutil.rmtree(out)


# =========================================================================
# Test: OpLoad count decreases (mem2reg effect)
# =========================================================================

@requires_spirv_tools
@requires_spirv_opt
class TestOptimizeRemovesRedundantLoads:
    def test_optimize_removes_redundant_loads(self):
        """Optimized output should have fewer OpLoad instructions due to mem2reg."""
        src = '''
        fragment {
            in uv: vec2;
            out color: vec4;
            fn main() {
                let a: scalar = uv.x;
                let b: scalar = uv.y;
                let c: scalar = a + b;
                let d: scalar = a * b;
                let e: scalar = c + d;
                let f: scalar = e * 0.5;
                color = vec4(f, f, f, 1.0);
            }
        }
        '''
        out_normal = _compile(src, stem="normal", optimize=False)
        out_opt = _compile(src, stem="opt", optimize=True)

        normal_spv = list(out_normal.glob("*.frag.spv"))[0]
        opt_spv = list(out_opt.glob("*.frag.spv"))[0]

        normal_asm = _disassemble(normal_spv)
        opt_asm = _disassemble(opt_spv)

        normal_loads = normal_asm.count("OpLoad")
        opt_loads = opt_asm.count("OpLoad")

        assert opt_loads < normal_loads, (
            f"Optimized OpLoad count ({opt_loads}) should be less than "
            f"unoptimized ({normal_loads})"
        )

        shutil.rmtree(out_normal)
        shutil.rmtree(out_opt)


# =========================================================================
# Test: All stages work with -O
# =========================================================================

@requires_spirv_tools
@requires_spirv_opt
class TestOptimizeFlagCompilesAllStages:
    def test_optimize_flag_compiles_all_stages(self):
        """Vertex, fragment, and compute stages should all compile with optimize=True."""
        # Vertex
        vertex_src = '''
        vertex {
            in pos: vec3;
            in color: vec3;
            out frag_color: vec3;
            fn main() {
                frag_color = color;
                builtin_position = vec4(pos, 1.0);
            }
        }
        '''
        out_v = _compile(vertex_src, stem="test_vert", optimize=True)
        assert list(out_v.glob("*.vert.spv"))
        result = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.2",
             str(list(out_v.glob("*.vert.spv"))[0])],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

        # Fragment
        frag_src = '''
        fragment {
            in frag_color: vec3;
            out color: vec4;
            fn main() {
                color = vec4(frag_color, 1.0);
            }
        }
        '''
        out_f = _compile(frag_src, stem="test_frag", optimize=True)
        assert list(out_f.glob("*.frag.spv"))
        result = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.2",
             str(list(out_f.glob("*.frag.spv"))[0])],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

        # Compute
        compute_src = '''
        compute {
            storage_buffer data: scalar;
            fn main() {
                let gid: scalar = global_invocation_id.x;
                data[gid] = gid * 2.0;
            }
        }
        '''
        out_c = _compile(compute_src, stem="test_comp", optimize=True,
                         defines={'workgroup_size': 64})
        assert list(out_c.glob("*.comp.spv"))
        result = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.2",
             str(list(out_c.glob("*.comp.spv"))[0])],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

        shutil.rmtree(out_v)
        shutil.rmtree(out_f)
        shutil.rmtree(out_c)


# =========================================================================
# Test: Loops optimize correctly
# =========================================================================

@requires_spirv_tools
@requires_spirv_opt
class TestOptimizeWithLoops:
    def test_optimize_with_loops(self):
        """For/while loops should optimize correctly and still validate."""
        src = '''
        fragment {
            in uv: vec2;
            out color: vec4;
            fn main() {
                let sum: scalar = 0.0;
                for (let i: int = 0; i < 10; i = i + 1) {
                    sum = sum + sin(uv.x + uv.y);
                }
                let x: scalar = 1.0;
                while (x < 100.0) {
                    x = x * 1.5;
                }
                color = vec4(sum, x, 0.0, 1.0);
            }
        }
        '''
        out = _compile(src, optimize=True)
        spv_files = list(out.glob("*.frag.spv"))
        assert len(spv_files) == 1

        # Validate optimized output
        result = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.2", str(spv_files[0])],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"spirv-val failed: {result.stderr}"

        shutil.rmtree(out)


# =========================================================================
# Test: Shared memory + atomics optimize correctly
# =========================================================================

@requires_spirv_tools
@requires_spirv_opt
class TestOptimizeWithSharedMemory:
    def test_optimize_with_shared_memory(self):
        """Compute shader with shared memory and atomics should optimize correctly."""
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
        out = _compile(src, optimize=True, defines={'workgroup_size': 256})
        spv_files = list(out.glob("*.comp.spv"))
        assert len(spv_files) == 1

        # Validate optimized output
        result = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.2", str(spv_files[0])],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"spirv-val failed: {result.stderr}"

        shutil.rmtree(out)
