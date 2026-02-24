"""Tests for Phase 20 debug features: debug_print, assert, @[debug] blocks,
semantic types, and any_nan/any_inf."""

import pytest
from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import (
    DebugPrintStmt, AssertStmt, DebugBlock, LetStmt,
)
from luxc.analysis.type_checker import type_check, TypeCheckError
from luxc.optimization.const_fold import constant_fold
from luxc.analysis.layout_assigner import assign_layouts
from luxc.codegen.spirv_builder import generate_spirv
from luxc.builtins.types import (
    clear_type_aliases, register_strict_type_alias, SemanticType,
    VEC3, SCALAR, resolve_type,
)
from luxc.compiler import compile_source, strip_debug_stmts


# ---------------------------------------------------------------------------
# Helper: parse, type-check, constant-fold, and generate SPIR-V assembly
# ---------------------------------------------------------------------------

def _compile_stage(src: str, stage_idx: int = 0, debug: bool = False) -> str:
    """Parse, type-check, constant-fold, assign layouts, and generate SPIR-V asm."""
    m = parse_lux(src)
    type_check(m)
    if not debug:
        strip_debug_stmts(m)
    constant_fold(m)
    assign_layouts(m)
    return generate_spirv(m, m.stages[stage_idx], debug=debug)


# ===========================================================================
# Grammar / Parser Tests
# ===========================================================================

class TestDebugPrintParse:
    def test_debug_print_parse(self):
        """debug_print('foo', x) parses to DebugPrintStmt."""
        src = '''
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                debug_print("foo", x);
                color = vec4(x, x, x, 1.0);
            }
        }
        '''
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        debug_stmt = fn.body[0]
        assert isinstance(debug_stmt, DebugPrintStmt)
        assert debug_stmt.format_string == "foo"
        assert len(debug_stmt.args) == 1

    def test_debug_print_multi_args_parse(self):
        """debug_print with multiple args parses correctly."""
        src = '''
        fragment {
            in a: scalar;
            in b: scalar;
            out color: vec4;
            fn main() {
                debug_print("a={} b={}", a, b);
                color = vec4(a, b, 0.0, 1.0);
            }
        }
        '''
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        debug_stmt = fn.body[0]
        assert isinstance(debug_stmt, DebugPrintStmt)
        assert debug_stmt.format_string == "a={} b={}"
        assert len(debug_stmt.args) == 2


class TestAssertParse:
    def test_assert_parse(self):
        """assert(x > 0.0) parses to AssertStmt."""
        src = '''
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                assert(x > 0.0);
                color = vec4(x, x, x, 1.0);
            }
        }
        '''
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        assert_stmt = fn.body[0]
        assert isinstance(assert_stmt, AssertStmt)
        assert assert_stmt.condition is not None
        assert assert_stmt.message is None

    def test_assert_with_message_parse(self):
        """assert(x > 0.0, 'x must be positive') parses with message."""
        src = '''
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                assert(x > 0.0, "x must be positive");
                color = vec4(x, x, x, 1.0);
            }
        }
        '''
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        assert_stmt = fn.body[0]
        assert isinstance(assert_stmt, AssertStmt)
        assert assert_stmt.condition is not None
        assert assert_stmt.message == "x must be positive"


class TestDebugBlockParse:
    def test_debug_block_parse(self):
        """@[debug] { ... } parses to DebugBlock."""
        src = '''
        fragment {
            out color: vec4;
            fn main() {
                @[debug] {
                    let x: scalar = 1.0;
                }
                color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
        '''
        m = parse_lux(src)
        fn = m.stages[0].functions[0]
        debug_blk = fn.body[0]
        assert isinstance(debug_blk, DebugBlock)
        assert len(debug_blk.body) == 1
        assert isinstance(debug_blk.body[0], LetStmt)


# ===========================================================================
# Type Checker Tests
# ===========================================================================

class TestDebugTypeCheck:
    def test_debug_print_type_check(self):
        """debug_print with numeric/vector args type-checks successfully."""
        src = '''
        fragment {
            in x: scalar;
            in v: vec3;
            out color: vec4;
            fn main() {
                debug_print("x={} v={}", x, v);
                color = vec4(v, 1.0);
            }
        }
        '''
        m = parse_lux(src)
        type_check(m)  # should not raise

    def test_assert_type_check(self):
        """assert with a boolean condition type-checks successfully."""
        src = '''
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                assert(x > 0.0);
                color = vec4(x, x, x, 1.0);
            }
        }
        '''
        m = parse_lux(src)
        type_check(m)  # should not raise

    def test_debug_block_type_check(self):
        """@[debug] block body is type-checked normally."""
        src = '''
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                @[debug] {
                    let y: scalar = x + 1.0;
                    debug_print("y={}", y);
                }
                color = vec4(x, x, x, 1.0);
            }
        }
        '''
        m = parse_lux(src)
        type_check(m)  # should not raise


# ===========================================================================
# Codegen Tests (SPIR-V assembly)
# ===========================================================================

class TestDebugPrintCodegen:
    def test_debug_print_codegen_debug_mode(self):
        """With debug=True, SPIR-V output contains NonSemantic.DebugPrintf."""
        src = '''
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                debug_print("x={}", x);
                color = vec4(x, x, x, 1.0);
            }
        }
        '''
        asm = _compile_stage(src, debug=True)
        assert "NonSemantic.DebugPrintf" in asm
        assert "SPV_KHR_non_semantic_info" in asm

    def test_debug_print_stripped_release(self):
        """Without debug, output does NOT contain DebugPrintf."""
        src = '''
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                debug_print("x={}", x);
                color = vec4(x, x, x, 1.0);
            }
        }
        '''
        asm = _compile_stage(src, debug=False)
        assert "DebugPrintf" not in asm
        assert "NonSemantic.DebugPrintf" not in asm


class TestAssertCodegen:
    def test_assert_codegen_debug_mode(self):
        """With debug=True, assert generates conditional branch in SPIR-V."""
        src = '''
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                assert(x > 0.0);
                color = vec4(x, x, x, 1.0);
            }
        }
        '''
        asm = _compile_stage(src, debug=True)
        assert "OpSelectionMerge" in asm
        assert "OpBranchConditional" in asm
        assert "ASSERT FAILED" in asm

    def test_assert_stripped_release(self):
        """Without debug, assert is stripped from output."""
        src = '''
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                assert(x > 0.0);
                color = vec4(x, x, x, 1.0);
            }
        }
        '''
        asm = _compile_stage(src, debug=False)
        assert "ASSERT FAILED" not in asm


class TestDebugBlockCodegen:
    def test_debug_block_stripped_release(self):
        """Without debug, entire @[debug] block is removed from output."""
        src = '''
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                @[debug] {
                    debug_print("debug only: x={}", x);
                }
                color = vec4(x, x, x, 1.0);
            }
        }
        '''
        asm = _compile_stage(src, debug=False)
        assert "DebugPrintf" not in asm
        assert "debug only" not in asm


# ===========================================================================
# Semantic Type Tests
# ===========================================================================

class TestSemanticTypeDeclaration:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_semantic_type_strict_declaration(self):
        """'type strict Foo = vec3;' creates a SemanticType via type-check."""
        src = '''
        type strict Foo = vec3;
        fragment {
            out color: vec4;
            fn main() {
                let f: Foo = vec3(1.0, 2.0, 3.0);
                color = vec4(f, 1.0);
            }
        }
        '''
        m = parse_lux(src)
        # Verify the AST has strict=True
        assert m.type_aliases[0].strict is True
        assert m.type_aliases[0].name == "Foo"
        # Type check should succeed and register the SemanticType
        type_check(m)
        resolved = resolve_type("Foo")
        assert isinstance(resolved, SemanticType)
        assert resolved.base_type.name == "vec3"

    def test_semantic_type_mismatch(self):
        """Two different strict types (A, B) wrapping vec3 are not interchangeable."""
        src = '''
        type strict TypeA = vec3;
        type strict TypeB = vec3;
        fragment {
            out color: vec4;
            fn main() {
                let a: TypeA = vec3(1.0, 0.0, 0.0);
                let b: TypeB = a;
                color = vec4(b, 1.0);
            }
        }
        '''
        m = parse_lux(src)
        with pytest.raises(TypeCheckError, match="Type mismatch"):
            type_check(m)

    def test_semantic_type_arithmetic(self):
        """A + A works for same strict type, A + B fails."""
        # Same type arithmetic should work
        src_ok = '''
        type strict Velocity = vec3;
        fragment {
            in v1: vec3;
            in v2: vec3;
            out color: vec4;
            fn main() {
                let a: Velocity = v1;
                let b: Velocity = v2;
                let c: Velocity = a + b;
                color = vec4(c, 1.0);
            }
        }
        '''
        m = parse_lux(src_ok)
        type_check(m)  # should not raise

    def test_semantic_type_arithmetic_mismatch(self):
        """A + B where A and B are different strict types should fail."""
        src_bad = '''
        type strict Velocity = vec3;
        type strict Position = vec3;
        fragment {
            in v1: vec3;
            in v2: vec3;
            out color: vec4;
            fn main() {
                let a: Velocity = v1;
                let b: Position = v2;
                let c: Velocity = a + b;
                color = vec4(c, 1.0);
            }
        }
        '''
        m = parse_lux(src_bad)
        with pytest.raises(TypeCheckError, match="incompatible semantic types"):
            type_check(m)

    def test_semantic_type_builtin_compat(self):
        """normalize(a) where a is a strict vec3 type should work."""
        src = '''
        type strict Normal = vec3;
        fragment {
            in n: vec3;
            out color: vec4;
            fn main() {
                let norm: Normal = n;
                let nn: Normal = normalize(norm);
                color = vec4(nn, 1.0);
            }
        }
        '''
        m = parse_lux(src)
        type_check(m)  # should not raise


# ===========================================================================
# any_nan / any_inf Tests
# ===========================================================================

class TestNanInfCodegen:
    def test_any_nan_codegen(self):
        """any_nan compiles to OpIsNan in SPIR-V."""
        src = '''
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                let has_nan: bool = any_nan(x);
                color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
        '''
        asm = _compile_stage(src)
        assert "OpIsNan" in asm

    def test_any_inf_codegen(self):
        """any_inf compiles to OpIsInf in SPIR-V."""
        src = '''
        fragment {
            in x: scalar;
            out color: vec4;
            fn main() {
                let has_inf: bool = any_inf(x);
                color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
        '''
        asm = _compile_stage(src)
        assert "OpIsInf" in asm

    def test_any_nan_vec3_codegen(self):
        """any_nan on vec3 emits OpIsNan + OpAny."""
        src = '''
        fragment {
            in v: vec3;
            out color: vec4;
            fn main() {
                let has_nan: bool = any_nan(v);
                color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
        '''
        asm = _compile_stage(src)
        assert "OpIsNan" in asm
        assert "OpAny" in asm

    def test_any_inf_vec3_codegen(self):
        """any_inf on vec3 emits OpIsInf + OpAny."""
        src = '''
        fragment {
            in v: vec3;
            out color: vec4;
            fn main() {
                let has_inf: bool = any_inf(v);
                color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
        '''
        asm = _compile_stage(src)
        assert "OpIsInf" in asm
        assert "OpAny" in asm
