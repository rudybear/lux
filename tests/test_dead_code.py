"""Tests for dead code elimination pass.

Tests cover:
- Unused let binding removed
- Used let binding kept
- Chain of dead code removed
- Side-effecting RHS preserved as ExprStmt
- Assignment to unused variable removed
- Assignment to used variable kept
- Variables used in if condition kept
- Variables used in return kept
- Empty function body -> no crash
- Multiple functions in module -> each processed independently
"""

import pytest
from luxc.parser.ast_nodes import (
    Module, FunctionDef, Param,
    LetStmt, AssignStmt, ReturnStmt, IfStmt, ExprStmt,
    NumberLit, BoolLit, VarRef, BinaryOp, UnaryOp, CallExpr,
    AssignTarget,
)
from luxc.optimization.dead_code import dead_code_elim


def _var(name: str, resolved_type: str = "scalar") -> VarRef:
    """Create a VarRef for testing."""
    return VarRef(name, resolved_type=resolved_type)


def _num(val: float, resolved_type: str = "scalar") -> NumberLit:
    """Create a NumberLit for testing."""
    s = repr(val)
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return NumberLit(s, resolved_type=resolved_type)


def _func(name: str, body: list) -> FunctionDef:
    """Create a FunctionDef with an empty param list and void return."""
    return FunctionDef(name=name, params=[], return_type="void", body=body)


def _module(functions: list[FunctionDef]) -> Module:
    """Create a Module with the given functions."""
    return Module(functions=functions)


class TestUnusedLetRemoved:
    def test_unused_let_binding_removed(self):
        """let x = 1.0; return 2.0; -> return 2.0; (x is unused)."""
        body = [
            LetStmt("x", "scalar", _num(1.0)),
            ReturnStmt(_num(2.0)),
        ]
        mod = _module([_func("test", body)])
        dead_code_elim(mod)
        func_body = mod.functions[0].body
        assert len(func_body) == 1
        assert isinstance(func_body[0], ReturnStmt)
        assert float(func_body[0].value.value) == 2.0


class TestUsedLetKept:
    def test_used_let_binding_kept(self):
        """let x = 1.0; return x; -> both statements kept."""
        body = [
            LetStmt("x", "scalar", _num(1.0)),
            ReturnStmt(_var("x")),
        ]
        mod = _module([_func("test", body)])
        dead_code_elim(mod)
        func_body = mod.functions[0].body
        assert len(func_body) == 2
        assert isinstance(func_body[0], LetStmt)
        assert func_body[0].name == "x"
        assert isinstance(func_body[1], ReturnStmt)


class TestChainOfDeadCode:
    def test_chain_of_dead_code_removed(self):
        """let x = 1.0; let y = x; return 2.0; -> both lets removed."""
        body = [
            LetStmt("x", "scalar", _num(1.0)),
            LetStmt("y", "scalar", _var("x")),
            ReturnStmt(_num(2.0)),
        ]
        mod = _module([_func("test", body)])
        dead_code_elim(mod)
        func_body = mod.functions[0].body
        # Both x and y are unused (y uses x, but y itself is unused,
        # then on second iteration x becomes unused too)
        assert len(func_body) == 1
        assert isinstance(func_body[0], ReturnStmt)


class TestSideEffectingRhsPreserved:
    def test_side_effecting_rhs_preserved_as_expr_stmt(self):
        """let x = sample(tex, uv); return 2.0; -> ExprStmt(sample(...)) + return."""
        sample_call = CallExpr(VarRef("sample"), [_var("tex"), _var("uv")], resolved_type="vec4")
        body = [
            LetStmt("x", "vec4", sample_call),
            ReturnStmt(_num(2.0)),
        ]
        mod = _module([_func("test", body)])
        dead_code_elim(mod)
        func_body = mod.functions[0].body
        # The let is dead, but sample() has side effects, so it becomes ExprStmt
        assert len(func_body) == 2
        assert isinstance(func_body[0], ExprStmt)
        assert isinstance(func_body[0].expr, CallExpr)
        assert isinstance(func_body[0].expr.func, VarRef)
        assert func_body[0].expr.func.name == "sample"
        assert isinstance(func_body[1], ReturnStmt)


class TestAssignToUnusedRemoved:
    def test_assign_to_unused_variable_removed(self):
        """Assignment to an unused variable should be removed."""
        body = [
            AssignStmt(target=VarRef("x"), value=_num(1.0)),
            ReturnStmt(_num(2.0)),
        ]
        mod = _module([_func("test", body)])
        dead_code_elim(mod)
        func_body = mod.functions[0].body
        assert len(func_body) == 1
        assert isinstance(func_body[0], ReturnStmt)


class TestAssignToUsedKept:
    def test_assign_to_used_variable_kept(self):
        """Assignment to a variable that is later read should be kept."""
        body = [
            AssignStmt(target=VarRef("x"), value=_num(1.0)),
            ReturnStmt(_var("x")),
        ]
        mod = _module([_func("test", body)])
        dead_code_elim(mod)
        func_body = mod.functions[0].body
        assert len(func_body) == 2
        assert isinstance(func_body[0], AssignStmt)
        assert isinstance(func_body[1], ReturnStmt)


class TestVarUsedInIfCondition:
    def test_var_used_in_if_condition_kept(self):
        """Variables used in if conditions should be kept."""
        body = [
            LetStmt("flag", "bool", BoolLit(True)),
            IfStmt(
                condition=_var("flag"),
                then_body=[ReturnStmt(_num(1.0))],
                else_body=[ReturnStmt(_num(0.0))],
            ),
        ]
        mod = _module([_func("test", body)])
        dead_code_elim(mod)
        func_body = mod.functions[0].body
        assert len(func_body) == 2
        assert isinstance(func_body[0], LetStmt)
        assert func_body[0].name == "flag"
        assert isinstance(func_body[1], IfStmt)


class TestVarUsedInReturn:
    def test_var_used_in_return_kept(self):
        """Variables used in return statements should be kept."""
        body = [
            LetStmt("result", "scalar", BinaryOp("+", _var("a"), _var("b"), resolved_type="scalar")),
            ReturnStmt(_var("result")),
        ]
        mod = _module([_func("test", body)])
        dead_code_elim(mod)
        func_body = mod.functions[0].body
        assert len(func_body) == 2
        assert isinstance(func_body[0], LetStmt)
        assert func_body[0].name == "result"


class TestEmptyBody:
    def test_empty_function_body_no_crash(self):
        """Dead code elimination on an empty function body should not crash."""
        mod = _module([_func("test", [])])
        dead_code_elim(mod)
        assert mod.functions[0].body == []


class TestMultipleFunctions:
    def test_multiple_functions_processed_independently(self):
        """Each function in the module should be processed independently."""
        # func1: has dead code
        body1 = [
            LetStmt("unused", "scalar", _num(42.0)),
            ReturnStmt(_num(1.0)),
        ]
        # func2: no dead code
        body2 = [
            LetStmt("used", "scalar", _num(3.14)),
            ReturnStmt(_var("used")),
        ]
        mod = _module([_func("func1", body1), _func("func2", body2)])
        dead_code_elim(mod)

        # func1: unused let removed
        f1_body = mod.functions[0].body
        assert len(f1_body) == 1
        assert isinstance(f1_body[0], ReturnStmt)

        # func2: used let kept
        f2_body = mod.functions[1].body
        assert len(f2_body) == 2
        assert isinstance(f2_body[0], LetStmt)
        assert f2_body[0].name == "used"
        assert isinstance(f2_body[1], ReturnStmt)
