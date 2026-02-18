"""Tests for forward-mode automatic differentiation."""

import subprocess
import pytest
from pathlib import Path

from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import (
    FunctionDef, NumberLit, VarRef, BinaryOp, CallExpr, ConstructorExpr,
)
from luxc.autodiff.forward_diff import autodiff_expand
from luxc.builtins.types import clear_type_aliases


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_and_expand(source: str):
    """Parse source and run autodiff expansion, return the module."""
    module = parse_lux(source)
    autodiff_expand(module)
    return module


def _find_fn(module, name: str) -> FunctionDef | None:
    for f in module.functions:
        if f.name == name:
            return f
    return None


def _has_spirv_tools() -> bool:
    try:
        subprocess.run(["spirv-as", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


requires_spirv_tools = pytest.mark.skipif(
    not _has_spirv_tools(), reason="spirv-as/spirv-val not found on PATH"
)


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

class TestAutodiffParsing:
    def test_attribute_parsed(self):
        m = parse_lux("@differentiable fn square(x: scalar) -> scalar { return x * x; }")
        assert len(m.functions) == 1
        assert m.functions[0].attributes == ["differentiable"]
        assert m.functions[0].name == "square"

    def test_attribute_on_regular_fn(self):
        m = parse_lux("fn foo(x: scalar) -> scalar { return x; }")
        assert m.functions[0].attributes == []

    def test_multiple_attributes(self):
        m = parse_lux("@differentiable @inline fn f(x: scalar) -> scalar { return x; }")
        assert m.functions[0].attributes == ["differentiable", "inline"]


# ---------------------------------------------------------------------------
# Expansion tests
# ---------------------------------------------------------------------------

class TestAutodiffExpansion:
    def test_linear(self):
        """return x * 2.0 -> gradient = 2.0"""
        m = _parse_and_expand("""
            @differentiable
            fn f(x: scalar) -> scalar { return x * 2.0; }
        """)
        grad = _find_fn(m, "f_d_x")
        assert grad is not None
        # Derivative of x*2.0 is 1.0*2.0 + x*0.0 = 2.0 (simplified)
        ret = grad.body[0]
        assert isinstance(ret.value, NumberLit)
        assert float(ret.value.value) == 2.0

    def test_quadratic(self):
        """return x * x -> gradient = x + x"""
        m = _parse_and_expand("""
            @differentiable
            fn f(x: scalar) -> scalar { return x * x; }
        """)
        grad = _find_fn(m, "f_d_x")
        assert grad is not None
        ret = grad.body[0]
        # Should be x + x (simplified from x*1 + 1*x)
        assert isinstance(ret.value, BinaryOp)
        assert ret.value.op == "+"

    def test_two_params(self):
        """return x * y -> d_x = y, d_y = x"""
        m = _parse_and_expand("""
            @differentiable
            fn f(x: scalar, y: scalar) -> scalar { return x * y; }
        """)
        d_x = _find_fn(m, "f_d_x")
        d_y = _find_fn(m, "f_d_y")
        assert d_x is not None
        assert d_y is not None
        # d/dx(x*y) = y, d/dy(x*y) = x
        assert isinstance(d_x.body[0].value, VarRef)
        assert d_x.body[0].value.name == "y"
        assert isinstance(d_y.body[0].value, VarRef)
        assert d_y.body[0].value.name == "x"

    def test_builtin_sin(self):
        """return sin(x) -> cos(x)"""
        m = _parse_and_expand("""
            @differentiable
            fn f(x: scalar) -> scalar { return sin(x); }
        """)
        grad = _find_fn(m, "f_d_x")
        assert grad is not None
        ret = grad.body[0]
        # Should be cos(x) * 1.0 = cos(x) (simplified)
        assert isinstance(ret.value, CallExpr)
        assert ret.value.func.name == "cos"

    def test_chain_rule(self):
        """return sin(x * x) -> cos(x*x) * (x + x)"""
        m = _parse_and_expand("""
            @differentiable
            fn f(x: scalar) -> scalar { return sin(x * x); }
        """)
        grad = _find_fn(m, "f_d_x")
        assert grad is not None
        ret = grad.body[0]
        # Should be cos(x*x) * (x + x)
        assert isinstance(ret.value, BinaryOp)
        assert ret.value.op == "*"

    def test_let_binding(self):
        """let y = x * x; return y + 1.0; -> d_x = 2*x"""
        m = _parse_and_expand("""
            @differentiable
            fn f(x: scalar) -> scalar {
                let y: scalar = x * x;
                return y + 1.0;
            }
        """)
        grad = _find_fn(m, "f_d_x")
        assert grad is not None
        # Should have: let y, let _d_y, return _d_y
        assert len(grad.body) == 3
        # _d_y should be x + x
        d_y_stmt = grad.body[1]
        assert d_y_stmt.name == "_d_y"

    def test_vec_return(self):
        """return vec3(x, x*x, sin(x)) -> vec3(1.0, 2*x, cos(x))"""
        m = _parse_and_expand("""
            @differentiable
            fn f(x: scalar) -> vec3 { return vec3(x, x * x, sin(x)); }
        """)
        grad = _find_fn(m, "f_d_x")
        assert grad is not None
        ret = grad.body[0]
        assert isinstance(ret.value, ConstructorExpr)
        assert ret.value.type_name == "vec3"
        assert len(ret.value.args) == 3

    def test_non_scalar_param_skipped(self):
        """Only scalar params get gradient functions."""
        m = _parse_and_expand("""
            @differentiable
            fn f(v: vec3, x: scalar) -> scalar { return x; }
        """)
        # Should only have f_d_x, not f_d_v
        names = [f.name for f in m.functions]
        assert "f_d_x" in names
        assert "f_d_v" not in names


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------

@requires_spirv_tools
class TestAutodiffE2E:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_gradient_compiles(self, tmp_path):
        """Fragment shader calling energy_d_x() compiles to valid SPIR-V."""
        from luxc.compiler import compile_source
        src = """
        @differentiable
        fn energy(x: scalar) -> scalar {
            return x * x + sin(x);
        }

        fragment {
            in param: scalar;
            out color: vec4;
            fn main() {
                let val: scalar = energy(param);
                let grad: scalar = energy_d_x(param);
                color = vec4(val, grad, 0.0, 1.0);
            }
        }
        """
        compile_source(src, "gradient_test", tmp_path, validate=True)
        assert (tmp_path / "gradient_test.frag.spv").exists()

    def test_differentiable_example(self, tmp_path):
        """Compile the examples/differentiable.lux example file."""
        from luxc.compiler import compile_source
        examples = Path(__file__).parent.parent / "examples"
        src = (examples / "differentiable.lux").read_text()
        compile_source(src, "differentiable", tmp_path, validate=True)
        assert (tmp_path / "differentiable.frag.spv").exists()
