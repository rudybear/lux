"""P25 Phase B: AST-level algebraic simplification tests.

Tests cover:
- Multiplicative identities (x * 1.0, x * 0.0, 1.0 * x, 0.0 * x)
- Additive identities (x + 0.0, 0.0 + x, x - 0.0)
- Division identity (x / 1.0)
- Double negation (-(-x))
- Boolean double negation (!(!x))
- Nested algebraic simplifications
- Non-identity values are NOT folded
"""

import pytest
from luxc.parser.ast_nodes import (
    NumberLit, BoolLit, BinaryOp, UnaryOp, VarRef,
)
from luxc.optimization.const_fold import _try_fold_expr, _as_float_value


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


def _unary(op: str, operand, resolved_type: str = "scalar") -> UnaryOp:
    """Create a UnaryOp for testing."""
    return UnaryOp(op=op, operand=operand, resolved_type=resolved_type)


class TestAsFloatValue:
    def test_number_lit_returns_float(self):
        assert _as_float_value(NumberLit("1.0")) == 1.0

    def test_number_lit_integer(self):
        assert _as_float_value(NumberLit("42")) == 42.0

    def test_var_ref_returns_none(self):
        assert _as_float_value(VarRef("x")) is None

    def test_bool_lit_returns_none(self):
        assert _as_float_value(BoolLit(True)) is None


class TestMultiplyByOne:
    def test_multiply_x_by_one(self):
        """x * 1.0 should fold to x."""
        expr = _binop(_var("x"), "*", _num(1.0))
        result = _try_fold_expr(expr, {})
        assert isinstance(result, VarRef)
        assert result.name == "x"

    def test_multiply_one_by_x(self):
        """1.0 * x should fold to x."""
        expr = _binop(_num(1.0), "*", _var("x"))
        result = _try_fold_expr(expr, {})
        assert isinstance(result, VarRef)
        assert result.name == "x"


class TestMultiplyByZero:
    def test_multiply_x_by_zero(self):
        """x * 0.0 should fold to 0.0."""
        expr = _binop(_var("x"), "*", _num(0.0))
        result = _try_fold_expr(expr, {})
        assert isinstance(result, NumberLit)
        assert float(result.value) == 0.0

    def test_multiply_zero_by_x(self):
        """0.0 * x should fold to 0.0."""
        expr = _binop(_num(0.0), "*", _var("x"))
        result = _try_fold_expr(expr, {})
        assert isinstance(result, NumberLit)
        assert float(result.value) == 0.0


class TestAddZero:
    def test_add_x_plus_zero(self):
        """x + 0.0 should fold to x."""
        expr = _binop(_var("x"), "+", _num(0.0))
        result = _try_fold_expr(expr, {})
        assert isinstance(result, VarRef)
        assert result.name == "x"

    def test_add_zero_plus_x(self):
        """0.0 + x should fold to x."""
        expr = _binop(_num(0.0), "+", _var("x"))
        result = _try_fold_expr(expr, {})
        assert isinstance(result, VarRef)
        assert result.name == "x"


class TestSubtractZero:
    def test_subtract_zero(self):
        """x - 0.0 should fold to x."""
        expr = _binop(_var("x"), "-", _num(0.0))
        result = _try_fold_expr(expr, {})
        assert isinstance(result, VarRef)
        assert result.name == "x"


class TestDivideByOne:
    def test_divide_by_one(self):
        """x / 1.0 should fold to x."""
        expr = _binop(_var("x"), "/", _num(1.0))
        result = _try_fold_expr(expr, {})
        assert isinstance(result, VarRef)
        assert result.name == "x"


class TestDoubleNegation:
    def test_double_negation(self):
        """-(-x) should fold to x."""
        inner = _unary("-", _var("x"))
        expr = _unary("-", inner)
        result = _try_fold_expr(expr, {})
        assert isinstance(result, VarRef)
        assert result.name == "x"


class TestBooleanDoubleNegation:
    def test_boolean_double_negation(self):
        """!(!x) should fold to x."""
        inner = _unary("!", _var("x", resolved_type="bool"), resolved_type="bool")
        expr = _unary("!", inner, resolved_type="bool")
        result = _try_fold_expr(expr, {})
        assert isinstance(result, VarRef)
        assert result.name == "x"


class TestNestedAlgebraic:
    def test_nested_algebraic(self):
        """(x * 1.0) + 0.0 should fold to x."""
        inner = _binop(_var("x"), "*", _num(1.0))
        expr = _binop(inner, "+", _num(0.0))
        result = _try_fold_expr(expr, {})
        assert isinstance(result, VarRef)
        assert result.name == "x"

    def test_deeply_nested(self):
        """((x + 0.0) * 1.0) - 0.0 should fold to x."""
        step1 = _binop(_var("x"), "+", _num(0.0))
        step2 = _binop(step1, "*", _num(1.0))
        expr = _binop(step2, "-", _num(0.0))
        result = _try_fold_expr(expr, {})
        assert isinstance(result, VarRef)
        assert result.name == "x"


class TestNoFoldNonIdentity:
    def test_no_fold_non_identity_multiply(self):
        """x * 2.0 should NOT be folded."""
        expr = _binop(_var("x"), "*", _num(2.0))
        result = _try_fold_expr(expr, {})
        assert isinstance(result, BinaryOp)
        assert result.op == "*"

    def test_no_fold_non_identity_add(self):
        """x + 1.0 should NOT be folded."""
        expr = _binop(_var("x"), "+", _num(1.0))
        result = _try_fold_expr(expr, {})
        assert isinstance(result, BinaryOp)
        assert result.op == "+"

    def test_no_fold_divide_by_two(self):
        """x / 2.0 should NOT be folded."""
        expr = _binop(_var("x"), "/", _num(2.0))
        result = _try_fold_expr(expr, {})
        assert isinstance(result, BinaryOp)
        assert result.op == "/"

    def test_no_fold_subtract_one(self):
        """x - 1.0 should NOT be folded."""
        expr = _binop(_var("x"), "-", _num(1.0))
        result = _try_fold_expr(expr, {})
        assert isinstance(result, BinaryOp)
        assert result.op == "-"
