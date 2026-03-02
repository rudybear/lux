"""Tests for Common Subexpression Elimination (CSE) pass.

Tests cover:
- Two identical expressions -> one gets replaced with _cse_N reference
- Three identical expressions -> all replaced with same _cse_N
- Different expressions -> no CSE applied
- Trivial expressions (VarRef, NumberLit) -> NOT CSE'd
- Texture sample calls -> NOT CSE'd (side effects)
- Expressions in different if-branches -> NOT CSE'd (different scopes)
- CSE within a single if-branch body works
- Nested duplicate expressions: (a+b) * (a+b) -> CSE the inner a+b
- CSE generates unique names (_cse_0, _cse_1, etc.)
- Empty function body -> no crash
"""

import pytest
from luxc.parser.ast_nodes import (
    Module, FunctionDef, Param,
    LetStmt, AssignStmt, ReturnStmt, IfStmt, ExprStmt,
    NumberLit, BoolLit, VarRef, BinaryOp, UnaryOp, CallExpr,
)
from luxc.optimization.cse import cse


def _var(name: str, resolved_type: str = "scalar") -> VarRef:
    """Create a VarRef for testing."""
    return VarRef(name, resolved_type=resolved_type)


def _num(val: float, resolved_type: str = "scalar") -> NumberLit:
    """Create a NumberLit for testing."""
    s = repr(val)
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return NumberLit(s, resolved_type=resolved_type)


def _binop(left, op: str, right, resolved_type: str = "scalar") -> BinaryOp:
    """Create a BinaryOp for testing."""
    return BinaryOp(op=op, left=left, right=right, resolved_type=resolved_type)


def _func(name: str, body: list) -> FunctionDef:
    """Create a FunctionDef with an empty param list and void return."""
    return FunctionDef(name=name, params=[], return_type="void", body=body)


def _module(functions: list[FunctionDef]) -> Module:
    """Create a Module with the given functions."""
    return Module(functions=functions)


class TestCseTwoIdentical:
    def test_two_identical_exprs(self):
        """Two identical a + b expressions should be CSE'd."""
        # let x = a + b; let y = a + b;
        body = [
            LetStmt("x", "scalar", _binop(_var("a"), "+", _var("b"))),
            LetStmt("y", "scalar", _binop(_var("a"), "+", _var("b"))),
        ]
        mod = _module([_func("test", body)])
        cse(mod)
        func_body = mod.functions[0].body
        # A _cse_N let should have been inserted and at least one
        # of the original let bindings should reference it.
        cse_lets = [s for s in func_body if isinstance(s, LetStmt) and s.name.startswith("_cse_")]
        assert len(cse_lets) >= 1
        cse_name = cse_lets[0].name
        # The original let statements should now reference the cse variable
        non_cse_lets = [s for s in func_body if isinstance(s, LetStmt) and not s.name.startswith("_cse_")]
        refs = [s for s in non_cse_lets if isinstance(s.value, VarRef) and s.value.name == cse_name]
        assert len(refs) >= 1


class TestCseThreeIdentical:
    def test_three_identical_exprs(self):
        """Three identical expressions should all reference the same _cse_N."""
        body = [
            LetStmt("x", "scalar", _binop(_var("a"), "+", _var("b"))),
            LetStmt("y", "scalar", _binop(_var("a"), "+", _var("b"))),
            LetStmt("z", "scalar", _binop(_var("a"), "+", _var("b"))),
        ]
        mod = _module([_func("test", body)])
        cse(mod)
        func_body = mod.functions[0].body
        cse_lets = [s for s in func_body if isinstance(s, LetStmt) and s.name.startswith("_cse_")]
        assert len(cse_lets) == 1
        cse_name = cse_lets[0].name
        # All three original lets should now reference the same cse variable
        non_cse_lets = [s for s in func_body if isinstance(s, LetStmt) and not s.name.startswith("_cse_")]
        refs = [s for s in non_cse_lets if isinstance(s.value, VarRef) and s.value.name == cse_name]
        assert len(refs) == len(non_cse_lets)


class TestCseDifferentExprs:
    def test_different_exprs_not_csed(self):
        """Different expressions should NOT be CSE'd."""
        body = [
            LetStmt("x", "scalar", _binop(_var("a"), "+", _var("b"))),
            LetStmt("y", "scalar", _binop(_var("c"), "+", _var("d"))),
        ]
        mod = _module([_func("test", body)])
        cse(mod)
        func_body = mod.functions[0].body
        cse_lets = [s for s in func_body if isinstance(s, LetStmt) and s.name.startswith("_cse_")]
        assert len(cse_lets) == 0


class TestCseTrivialNotCsed:
    def test_trivial_varref_not_csed(self):
        """Trivial VarRef expressions should NOT be CSE'd."""
        body = [
            LetStmt("x", "scalar", _var("a")),
            LetStmt("y", "scalar", _var("a")),
        ]
        mod = _module([_func("test", body)])
        cse(mod)
        func_body = mod.functions[0].body
        cse_lets = [s for s in func_body if isinstance(s, LetStmt) and s.name.startswith("_cse_")]
        assert len(cse_lets) == 0

    def test_trivial_numberlit_not_csed(self):
        """Trivial NumberLit expressions should NOT be CSE'd."""
        body = [
            LetStmt("x", "scalar", _num(1.0)),
            LetStmt("y", "scalar", _num(1.0)),
        ]
        mod = _module([_func("test", body)])
        cse(mod)
        func_body = mod.functions[0].body
        cse_lets = [s for s in func_body if isinstance(s, LetStmt) and s.name.startswith("_cse_")]
        assert len(cse_lets) == 0


class TestCseTextureSampleNotCsed:
    def test_sample_call_not_csed(self):
        """Texture sample calls should NOT be CSE'd (side effects)."""
        sample1 = CallExpr(VarRef("sample"), [_var("tex"), _var("uv")], resolved_type="vec4")
        sample2 = CallExpr(VarRef("sample"), [_var("tex"), _var("uv")], resolved_type="vec4")
        body = [
            LetStmt("x", "vec4", sample1),
            LetStmt("y", "vec4", sample2),
        ]
        mod = _module([_func("test", body)])
        cse(mod)
        func_body = mod.functions[0].body
        cse_lets = [s for s in func_body if isinstance(s, LetStmt) and s.name.startswith("_cse_")]
        assert len(cse_lets) == 0


class TestCseDifferentIfBranches:
    def test_exprs_in_different_branches_not_csed(self):
        """Expressions in different if-branches should NOT be CSE'd."""
        then_body = [
            LetStmt("x", "scalar", _binop(_var("a"), "+", _var("b"))),
        ]
        else_body = [
            LetStmt("y", "scalar", _binop(_var("a"), "+", _var("b"))),
        ]
        body = [
            IfStmt(
                condition=_var("cond", resolved_type="bool"),
                then_body=then_body,
                else_body=else_body,
            ),
        ]
        mod = _module([_func("test", body)])
        cse(mod)
        # Each branch is processed independently, so there should be no CSE
        # within the top-level body, and each branch alone has only one
        # occurrence so no CSE there either.
        func_body = mod.functions[0].body
        assert isinstance(func_body[0], IfStmt)
        then_cse = [s for s in func_body[0].then_body
                     if isinstance(s, LetStmt) and s.name.startswith("_cse_")]
        else_cse = [s for s in func_body[0].else_body
                     if isinstance(s, LetStmt) and s.name.startswith("_cse_")]
        assert len(then_cse) == 0
        assert len(else_cse) == 0


class TestCseWithinIfBranch:
    def test_cse_within_single_if_branch(self):
        """CSE should work within a single if-branch body."""
        then_body = [
            LetStmt("x", "scalar", _binop(_var("a"), "+", _var("b"))),
            LetStmt("y", "scalar", _binop(_var("a"), "+", _var("b"))),
        ]
        body = [
            IfStmt(
                condition=_var("cond", resolved_type="bool"),
                then_body=then_body,
                else_body=[],
            ),
        ]
        mod = _module([_func("test", body)])
        cse(mod)
        if_stmt = mod.functions[0].body[0]
        assert isinstance(if_stmt, IfStmt)
        cse_lets = [s for s in if_stmt.then_body
                     if isinstance(s, LetStmt) and s.name.startswith("_cse_")]
        assert len(cse_lets) >= 1


class TestCseNestedDuplicate:
    def test_nested_duplicate_inner_csed(self):
        """(a+b) * (a+b) should CSE the inner a+b."""
        inner1 = _binop(_var("a"), "+", _var("b"))
        inner2 = _binop(_var("a"), "+", _var("b"))
        outer = _binop(inner1, "*", inner2)
        body = [
            LetStmt("x", "scalar", outer),
        ]
        mod = _module([_func("test", body)])
        cse(mod)
        func_body = mod.functions[0].body
        # The inner a+b should have been CSE'd into a _cse_N variable
        cse_lets = [s for s in func_body if isinstance(s, LetStmt) and s.name.startswith("_cse_")]
        assert len(cse_lets) >= 1


class TestCseUniqueNames:
    def test_cse_generates_unique_names(self):
        """CSE should generate unique _cse_0, _cse_1, etc. names."""
        body = [
            LetStmt("w", "scalar", _binop(_var("a"), "+", _var("b"))),
            LetStmt("x", "scalar", _binop(_var("a"), "+", _var("b"))),
            LetStmt("y", "scalar", _binop(_var("c"), "*", _var("d"))),
            LetStmt("z", "scalar", _binop(_var("c"), "*", _var("d"))),
        ]
        mod = _module([_func("test", body)])
        cse(mod)
        func_body = mod.functions[0].body
        cse_lets = [s for s in func_body if isinstance(s, LetStmt) and s.name.startswith("_cse_")]
        assert len(cse_lets) == 2
        names = sorted(s.name for s in cse_lets)
        assert names[0] == "_cse_0"
        assert names[1] == "_cse_1"


class TestCseEmptyBody:
    def test_empty_function_body_no_crash(self):
        """CSE on an empty function body should not crash."""
        mod = _module([_func("test", [])])
        cse(mod)
        assert mod.functions[0].body == []
