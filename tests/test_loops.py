"""Tests for P22: Loops & Control Flow (for/while, break/continue, integer arithmetic)."""

import pytest
from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import (
    ForStmt, WhileStmt, BreakStmt, ContinueStmt, AssignTarget, VarRef,
)
from luxc.analysis.type_checker import type_check, TypeCheckError
from luxc.optimization.const_fold import constant_fold
from luxc.analysis.layout_assigner import assign_layouts
from luxc.codegen.spirv_builder import generate_spirv


def _compile_stage(src: str, stage_idx: int = 0) -> str:
    """Parse, type-check, constant-fold, and generate SPIR-V assembly for a stage."""
    m = parse_lux(src)
    type_check(m)
    constant_fold(m)
    assign_layouts(m)
    return generate_spirv(m, m.stages[stage_idx])


# ============================================================
# Grammar / Parsing tests
# ============================================================

class TestLoopParsing:
    def test_for_loop_parses(self):
        src = """
        fragment {
            out color: vec4;
            fn main() {
                let sum: scalar = 0.0;
                for (let i: int = 0; i < 10; i = i + 1) {
                    sum = sum + 1.0;
                }
                color = vec4(sum, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        for_stmt = fn.body[1]
        assert isinstance(for_stmt, ForStmt)
        assert for_stmt.loop_var == "i"
        assert for_stmt.loop_var_type == "int"

    def test_while_loop_parses(self):
        src = """
        fragment {
            out color: vec4;
            fn main() {
                let x: scalar = 1.0;
                while (x < 100.0) {
                    x = x * 2.0;
                }
                color = vec4(x, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        while_stmt = fn.body[1]
        assert isinstance(while_stmt, WhileStmt)

    def test_break_parses(self):
        src = """
        fragment {
            out color: vec4;
            fn main() {
                let x: scalar = 0.0;
                while (x < 100.0) {
                    if (x > 50.0) {
                        break;
                    }
                    x = x + 1.0;
                }
                color = vec4(x, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        while_stmt = fn.body[1]
        assert isinstance(while_stmt, WhileStmt)
        # The if body should contain a BreakStmt
        if_stmt = while_stmt.body[0]
        assert isinstance(if_stmt.then_body[0], BreakStmt)

    def test_continue_parses(self):
        src = """
        fragment {
            out color: vec4;
            fn main() {
                let sum: scalar = 0.0;
                for (let i: int = 0; i < 10; i = i + 1) {
                    if (i == 5) {
                        continue;
                    }
                    sum = sum + 1.0;
                }
                color = vec4(sum, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        for_stmt = fn.body[1]
        assert isinstance(for_stmt, ForStmt)
        if_stmt = for_stmt.body[0]
        assert isinstance(if_stmt.then_body[0], ContinueStmt)

    def test_nested_loops_parse(self):
        src = """
        fragment {
            out color: vec4;
            fn main() {
                let sum: scalar = 0.0;
                for (let i: int = 0; i < 4; i = i + 1) {
                    for (let j: int = 0; j < 4; j = j + 1) {
                        sum = sum + 1.0;
                    }
                }
                color = vec4(sum, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        outer = fn.body[1]
        assert isinstance(outer, ForStmt)
        inner = outer.body[0]
        assert isinstance(inner, ForStmt)
        assert inner.loop_var == "j"

    def test_unroll_annotation_parses(self):
        src = """
        fragment {
            out color: vec4;
            fn main() {
                let sum: scalar = 0.0;
                @[unroll] for (let i: int = 0; i < 4; i = i + 1) {
                    sum = sum + 1.0;
                }
                color = vec4(sum, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        for_stmt = fn.body[1]
        assert isinstance(for_stmt, ForStmt)
        assert for_stmt.unroll is True

    def test_unroll_with_count_annotation_parses(self):
        src = """
        fragment {
            out color: vec4;
            fn main() {
                let sum: scalar = 0.0;
                @[unroll(8)] for (let i: int = 0; i < 64; i = i + 1) {
                    sum = sum + 1.0;
                }
                color = vec4(sum, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        for_stmt = fn.body[1]
        assert isinstance(for_stmt, ForStmt)
        assert for_stmt.unroll is True
        assert for_stmt.unroll_count == 8


# ============================================================
# Type checker tests
# ============================================================

class TestLoopTypeChecker:
    def test_loop_variable_scoping(self):
        """Loop variable should be accessible inside body."""
        src = """
        fragment {
            out color: vec4;
            fn main() {
                let sum: scalar = 0.0;
                for (let i: int = 0; i < 10; i = i + 1) {
                    sum = sum + 1.0;
                }
                color = vec4(sum, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)  # Should not raise

    def test_break_outside_loop_error(self):
        src = """
        fragment {
            out color: vec4;
            fn main() {
                break;
                color = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        with pytest.raises(TypeCheckError, match="break.*outside"):
            type_check(m)

    def test_continue_outside_loop_error(self):
        src = """
        fragment {
            out color: vec4;
            fn main() {
                continue;
                color = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        with pytest.raises(TypeCheckError, match="continue.*outside"):
            type_check(m)

    def test_int_loop_variable_type_check(self):
        """Integer loop variable should pass type checking."""
        src = """
        fragment {
            out color: vec4;
            fn main() {
                for (let i: int = 0; i < 10; i = i + 1) {
                }
                color = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)  # Should not raise

    def test_break_inside_loop_ok(self):
        """Break inside a loop should pass type checking."""
        src = """
        fragment {
            out color: vec4;
            fn main() {
                let x: scalar = 0.0;
                while (x < 10.0) {
                    break;
                }
                color = vec4(x, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)  # Should not raise


# ============================================================
# SPIR-V Codegen tests
# ============================================================

class TestLoopCodegen:
    def test_for_loop_generates_loop_merge(self):
        asm = _compile_stage("""
        fragment {
            out color: vec4;
            fn main() {
                let sum: scalar = 0.0;
                for (let i: int = 0; i < 10; i = i + 1) {
                    sum = sum + 1.0;
                }
                color = vec4(sum, 0.0, 0.0, 1.0);
            }
        }
        """)
        assert "OpLoopMerge" in asm
        assert "OpBranchConditional" in asm

    def test_while_loop_generates_loop_merge(self):
        asm = _compile_stage("""
        fragment {
            out color: vec4;
            fn main() {
                let x: scalar = 1.0;
                while (x < 100.0) {
                    x = x * 2.0;
                }
                color = vec4(x, 0.0, 0.0, 1.0);
            }
        }
        """)
        assert "OpLoopMerge" in asm

    def test_break_generates_branch_to_merge(self):
        asm = _compile_stage("""
        fragment {
            out color: vec4;
            fn main() {
                let x: scalar = 0.0;
                while (x < 100.0) {
                    if (x > 50.0) {
                        break;
                    }
                    x = x + 1.0;
                }
                color = vec4(x, 0.0, 0.0, 1.0);
            }
        }
        """)
        assert "OpLoopMerge" in asm
        # Break generates OpBranch to the merge label
        # The asm should have at least two OpBranch instructions for the loop structure
        lines = asm.split("\n")
        branch_lines = [l for l in lines if "OpBranch " in l and "OpBranchConditional" not in l]
        assert len(branch_lines) >= 3  # header->header, body->continue, continue->header, break->merge

    def test_continue_generates_branch_to_continue(self):
        asm = _compile_stage("""
        fragment {
            out color: vec4;
            fn main() {
                let sum: scalar = 0.0;
                for (let i: int = 0; i < 10; i = i + 1) {
                    if (i == 5) {
                        continue;
                    }
                    sum = sum + 1.0;
                }
                color = vec4(sum, 0.0, 0.0, 1.0);
            }
        }
        """)
        assert "OpLoopMerge" in asm

    def test_integer_addition_generates_op_iadd(self):
        asm = _compile_stage("""
        fragment {
            out color: vec4;
            fn main() {
                for (let i: int = 0; i < 10; i = i + 1) {
                }
                color = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
        """)
        assert "OpIAdd" in asm

    def test_integer_comparison_generates_sless(self):
        asm = _compile_stage("""
        fragment {
            out color: vec4;
            fn main() {
                for (let i: int = 0; i < 10; i = i + 1) {
                }
                color = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
        """)
        assert "OpSLessThan" in asm

    def test_unroll_hint_in_spirv(self):
        asm = _compile_stage("""
        fragment {
            out color: vec4;
            fn main() {
                let sum: scalar = 0.0;
                @[unroll] for (let i: int = 0; i < 4; i = i + 1) {
                    sum = sum + 1.0;
                }
                color = vec4(sum, 0.0, 0.0, 1.0);
            }
        }
        """)
        assert "Unroll" in asm

    def test_no_unroll_hint_default(self):
        asm = _compile_stage("""
        fragment {
            out color: vec4;
            fn main() {
                let sum: scalar = 0.0;
                for (let i: int = 0; i < 4; i = i + 1) {
                    sum = sum + 1.0;
                }
                color = vec4(sum, 0.0, 0.0, 1.0);
            }
        }
        """)
        # Default loop control is None
        lines = asm.split("\n")
        loop_merge_lines = [l for l in lines if "OpLoopMerge" in l]
        assert len(loop_merge_lines) >= 1
        assert "None" in loop_merge_lines[0]

    def test_nested_loops_codegen(self):
        asm = _compile_stage("""
        fragment {
            out color: vec4;
            fn main() {
                let sum: scalar = 0.0;
                for (let i: int = 0; i < 3; i = i + 1) {
                    for (let j: int = 0; j < 3; j = j + 1) {
                        sum = sum + 1.0;
                    }
                }
                color = vec4(sum, 0.0, 0.0, 1.0);
            }
        }
        """)
        # Should have two OpLoopMerge instructions
        lines = asm.split("\n")
        loop_merge_count = sum(1 for l in lines if "OpLoopMerge" in l)
        assert loop_merge_count == 2


# ============================================================
# End-to-end tests
# ============================================================

class TestLoopEndToEnd:
    def test_compute_shader_with_for_loop(self):
        """Compile a compute shader with a for loop."""
        asm = _compile_stage("""
        compute {
            storage_image output_img;
            fn main() {
                let gid: uvec3 = global_invocation_id;
                let sum: scalar = 0.0;
                for (let i: int = 0; i < 8; i = i + 1) {
                    sum = sum + 1.0;
                }
                let color: vec4 = vec4(sum, sum, sum, 1.0);
                image_store(output_img, gid.xy, color);
            }
        }
        """)
        assert "OpLoopMerge" in asm
        assert "GLCompute" in asm
        assert "OpIAdd" in asm

    def test_mandelbrot_with_loop(self):
        """Compile the mandelbrot compute shader with a proper loop."""
        asm = _compile_stage("""
        compute {
            storage_image output_img;
            fn main() {
                let gid: uvec3 = global_invocation_id;
                let px: scalar = gid.x;
                let py: scalar = gid.y;
                let w: scalar = 512.0;
                let h: scalar = 512.0;
                let cx: scalar = -0.5 + (px / w - 0.5) * 3.0;
                let cy: scalar = (py / h - 0.5) * 3.0;
                let zx: scalar = cx;
                let zy: scalar = cy;
                let iter: int = 0;
                for (let i: int = 0; i < 64; i = i + 1) {
                    let mag: scalar = zx * zx + zy * zy;
                    if (mag > 4.0) {
                        break;
                    }
                    let new_zx: scalar = zx * zx - zy * zy + cx;
                    zy = 2.0 * zx * zy + cy;
                    zx = new_zx;
                    iter = i;
                }
                let t: scalar = clamp(sqrt(zx * zx + zy * zy) / 4.0, 0.0, 1.0);
                let color: vec4 = vec4(t, t, t, 1.0);
                image_store(output_img, gid.xy, color);
            }
        }
        """)
        assert "OpLoopMerge" in asm
        assert "OpIAdd" in asm
        assert "OpSLessThan" in asm

    def test_while_loop_with_break(self):
        """While loop with break compiles correctly."""
        asm = _compile_stage("""
        fragment {
            out color: vec4;
            fn main() {
                let x: scalar = 1.0;
                let count: int = 0;
                while (count < 100) {
                    x = x * 1.01;
                    count = count + 1;
                    if (x > 2.0) {
                        break;
                    }
                }
                color = vec4(x, 0.0, 0.0, 1.0);
            }
        }
        """)
        assert "OpLoopMerge" in asm
        assert "OpSLessThan" in asm
