"""Tests for strength reduction rules in const_fold.py.

Tests cover:
- pow(x, 2.0) -> x * x
- pow(x, 3.0) -> x * x * x
- pow(x, 0.5) -> sqrt(x)
- pow(x, -0.5) -> inversesqrt(x)
- pow(x, 1.0) -> x
- pow(x, 0.0) -> 1.0
- pow(x, 4.0) is NOT reduced
- length(v) * length(v) -> dot(v, v)
- length(a) * length(b) is NOT reduced (different args)
- _exprs_equal structural checks
- resolved_type preservation
"""

import pytest
from luxc.parser.ast_nodes import (
    NumberLit, BoolLit, BinaryOp, UnaryOp, VarRef, CallExpr,
)
from luxc.optimization.const_fold import _try_fold_expr, _exprs_equal


def _var(name: str, resolved_type: str = "scalar") -> VarRef:
    """Create a VarRef for testing."""
    return VarRef(name, resolved_type=resolved_type)


def _num(val: float, resolved_type: str = "scalar") -> NumberLit:
    """Create a NumberLit for testing."""
    s = repr(val)
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return NumberLit(s, resolved_type=resolved_type)


def _call(fname: str, args: list, resolved_type: str = "scalar") -> CallExpr:
    """Create a CallExpr for testing."""
    return CallExpr(VarRef(fname), args, resolved_type=resolved_type)


class TestPowSquare:
    def test_pow_x_2(self):
        """pow(x, 2.0) should reduce to x * x."""
        expr = _call("pow", [_var("x"), _num(2.0)])
        result = _try_fold_expr(expr, {})
        assert isinstance(result, BinaryOp)
        assert result.op == "*"
        assert isinstance(result.left, VarRef) and result.left.name == "x"
        assert isinstance(result.right, VarRef) and result.right.name == "x"


class TestPowCube:
    def test_pow_x_3(self):
        """pow(x, 3.0) should reduce to x * x * x."""
        expr = _call("pow", [_var("x"), _num(3.0)])
        result = _try_fold_expr(expr, {})
        # Result should be BinaryOp("*", BinaryOp("*", x, x), x)
        assert isinstance(result, BinaryOp)
        assert result.op == "*"
        inner = result.left
        assert isinstance(inner, BinaryOp)
        assert inner.op == "*"
        assert isinstance(inner.left, VarRef) and inner.left.name == "x"
        assert isinstance(inner.right, VarRef) and inner.right.name == "x"
        assert isinstance(result.right, VarRef) and result.right.name == "x"


class TestPowSqrt:
    def test_pow_x_half(self):
        """pow(x, 0.5) should reduce to sqrt(x)."""
        expr = _call("pow", [_var("x"), _num(0.5)])
        result = _try_fold_expr(expr, {})
        assert isinstance(result, CallExpr)
        assert isinstance(result.func, VarRef)
        assert result.func.name == "sqrt"
        assert len(result.args) == 1
        assert isinstance(result.args[0], VarRef) and result.args[0].name == "x"


class TestPowInverseSqrt:
    def test_pow_x_neg_half(self):
        """pow(x, -0.5) should reduce to inversesqrt(x)."""
        expr = _call("pow", [_var("x"), _num(-0.5)])
        result = _try_fold_expr(expr, {})
        assert isinstance(result, CallExpr)
        assert isinstance(result.func, VarRef)
        assert result.func.name == "inversesqrt"
        assert len(result.args) == 1
        assert isinstance(result.args[0], VarRef) and result.args[0].name == "x"


class TestPowIdentity:
    def test_pow_x_1(self):
        """pow(x, 1.0) should reduce to just x."""
        expr = _call("pow", [_var("x"), _num(1.0)])
        result = _try_fold_expr(expr, {})
        assert isinstance(result, VarRef)
        assert result.name == "x"


class TestPowZero:
    def test_pow_x_0(self):
        """pow(x, 0.0) should reduce to 1.0."""
        expr = _call("pow", [_var("x"), _num(0.0)])
        result = _try_fold_expr(expr, {})
        assert isinstance(result, NumberLit)
        assert float(result.value) == 1.0


class TestPowNotReduced:
    def test_pow_x_4_not_reduced(self):
        """pow(x, 4.0) should NOT be reduced (remains a pow call)."""
        expr = _call("pow", [_var("x"), _num(4.0)])
        result = _try_fold_expr(expr, {})
        assert isinstance(result, CallExpr)
        assert isinstance(result.func, VarRef)
        assert result.func.name == "pow"


class TestLengthSquaredToDot:
    def test_length_v_squared_to_dot(self):
        """length(v) * length(v) should reduce to dot(v, v)."""
        left = _call("length", [_var("v")])
        right = _call("length", [_var("v")])
        expr = BinaryOp(op="*", left=left, right=right, resolved_type="scalar")
        result = _try_fold_expr(expr, {})
        assert isinstance(result, CallExpr)
        assert isinstance(result.func, VarRef)
        assert result.func.name == "dot"
        assert len(result.args) == 2
        assert isinstance(result.args[0], VarRef) and result.args[0].name == "v"
        assert isinstance(result.args[1], VarRef) and result.args[1].name == "v"


class TestLengthDifferentArgsNotReduced:
    def test_length_a_times_length_b_not_reduced(self):
        """length(a) * length(b) should NOT be reduced (different args)."""
        left = _call("length", [_var("a")])
        right = _call("length", [_var("b")])
        expr = BinaryOp(op="*", left=left, right=right, resolved_type="scalar")
        result = _try_fold_expr(expr, {})
        # Should remain a BinaryOp multiplication, not reduced to dot
        assert isinstance(result, BinaryOp)
        assert result.op == "*"


class TestExprsEqual:
    def test_matching_varrefs(self):
        """_exprs_equal should return True for matching VarRefs."""
        a = VarRef("x", resolved_type="scalar")
        b = VarRef("x", resolved_type="scalar")
        assert _exprs_equal(a, b) is True

    def test_mismatched_types_returns_false(self):
        """_exprs_equal should return False for different node types."""
        a = VarRef("x")
        b = NumberLit("1.0")
        assert _exprs_equal(a, b) is False


class TestStrengthReductionPreservesType:
    def test_pow_square_preserves_resolved_type(self):
        """Strength reduction of pow(x, 2.0) should preserve resolved_type."""
        expr = _call("pow", [_var("x"), _num(2.0)], resolved_type="vec3")
        result = _try_fold_expr(expr, {})
        assert isinstance(result, BinaryOp)
        assert result.resolved_type == "vec3"

    def test_pow_sqrt_preserves_resolved_type(self):
        """Strength reduction of pow(x, 0.5) should preserve resolved_type."""
        expr = _call("pow", [_var("x"), _num(0.5)], resolved_type="vec2")
        result = _try_fold_expr(expr, {})
        assert isinstance(result, CallExpr)
        assert result.func.name == "sqrt"
        assert result.resolved_type == "vec2"
