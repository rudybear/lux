"""Tests for the Lux shader debugger (CPU-side AST interpreter)."""

import math
import json
import pytest

from luxc.debug.values import (
    LuxScalar, LuxVec, LuxMat, LuxInt, LuxBool, LuxStruct,
    is_nan, is_inf, value_to_json, default_value, value_type_name,
)
from luxc.debug.builtins import (
    builtin_sin, builtin_cos, builtin_normalize, builtin_dot,
    builtin_cross, builtin_mix, builtin_clamp, builtin_smoothstep,
    builtin_length, builtin_reflect, builtin_determinant, builtin_inverse,
    builtin_abs, builtin_sqrt, builtin_pow, builtin_min, builtin_max,
)
from luxc.debug.environment import Environment, Scope
from luxc.debug.interpreter import Interpreter
from luxc.debug.debugger import Debugger
from luxc.debug.io import build_default_inputs, load_inputs_from_json, _json_to_value
from luxc.debug.cli import run_batch


# ===== Value Tests =====

class TestValues:
    def test_scalar_repr(self):
        assert repr(LuxScalar(3.14)).startswith("3.14")

    def test_vec_repr(self):
        v = LuxVec([1.0, 2.0, 3.0])
        assert "vec3" in repr(v)

    def test_mat_repr(self):
        m = LuxMat([[1, 0], [0, 1]])
        assert "mat2" in repr(m)

    def test_is_nan(self):
        assert is_nan(LuxScalar(float('nan')))
        assert not is_nan(LuxScalar(3.14))
        assert is_nan(LuxVec([1.0, float('nan'), 3.0]))
        assert not is_nan(LuxVec([1.0, 2.0, 3.0]))

    def test_is_inf(self):
        assert is_inf(LuxScalar(float('inf')))
        assert not is_inf(LuxScalar(0.0))

    def test_value_to_json(self):
        j = value_to_json(LuxVec([1.0, 2.0, 3.0]))
        assert j["type"] == "vec3"
        assert j["value"] == [1.0, 2.0, 3.0]

    def test_default_values(self):
        assert isinstance(default_value("scalar"), LuxScalar)
        assert isinstance(default_value("vec3"), LuxVec)
        assert default_value("vec3").size == 3
        assert isinstance(default_value("mat4"), LuxMat)
        assert default_value("mat4").size == 4
        # mat4 identity
        m = default_value("mat4")
        assert m.columns[0][0] == 1.0
        assert m.columns[1][1] == 1.0

    def test_value_type_name(self):
        assert value_type_name(LuxScalar(1.0)) == "scalar"
        assert value_type_name(LuxVec([1, 2, 3])) == "vec3"
        assert value_type_name(LuxInt(5)) == "int"
        assert value_type_name(LuxBool(True)) == "bool"


# ===== Builtin Tests =====

class TestBuiltins:
    def test_sin_cos(self):
        s = builtin_sin([LuxScalar(0.0)])
        assert isinstance(s, LuxScalar)
        assert abs(s.value) < 1e-10

        c = builtin_cos([LuxScalar(0.0)])
        assert abs(c.value - 1.0) < 1e-10

    def test_normalize(self):
        v = builtin_normalize([LuxVec([3.0, 4.0, 0.0])])
        assert isinstance(v, LuxVec)
        assert abs(v.components[0] - 0.6) < 1e-10
        assert abs(v.components[1] - 0.8) < 1e-10

    def test_normalize_zero_is_nan(self):
        v = builtin_normalize([LuxVec([0.0, 0.0, 0.0])])
        assert is_nan(v)

    def test_dot(self):
        d = builtin_dot([LuxVec([1, 0, 0]), LuxVec([0, 1, 0])])
        assert d.value == 0.0

        d2 = builtin_dot([LuxVec([1, 2, 3]), LuxVec([4, 5, 6])])
        assert d2.value == 32.0

    def test_cross(self):
        c = builtin_cross([LuxVec([1, 0, 0]), LuxVec([0, 1, 0])])
        assert c.components == [0.0, 0.0, 1.0]

    def test_mix(self):
        m = builtin_mix([LuxScalar(0.0), LuxScalar(10.0), LuxScalar(0.5)])
        assert abs(m.value - 5.0) < 1e-10

    def test_clamp(self):
        c = builtin_clamp([LuxScalar(-1.0), LuxScalar(0.0), LuxScalar(1.0)])
        assert c.value == 0.0
        c2 = builtin_clamp([LuxScalar(5.0), LuxScalar(0.0), LuxScalar(1.0)])
        assert c2.value == 1.0

    def test_smoothstep(self):
        s = builtin_smoothstep([LuxScalar(0.0), LuxScalar(1.0), LuxScalar(0.5)])
        assert 0.4 < s.value < 0.6

    def test_length(self):
        l = builtin_length([LuxVec([3.0, 4.0])])
        assert abs(l.value - 5.0) < 1e-10

    def test_reflect(self):
        r = builtin_reflect([LuxVec([1, -1, 0]), LuxVec([0, 1, 0])])
        assert abs(r.components[0] - 1.0) < 1e-10
        assert abs(r.components[1] - 1.0) < 1e-10

    def test_determinant_2x2(self):
        m = LuxMat([[1.0, 3.0], [2.0, 4.0]])
        d = builtin_determinant([m])
        assert abs(d.value - (1*4 - 2*3)) < 1e-10

    def test_inverse_2x2(self):
        m = LuxMat([[1.0, 0.0], [0.0, 1.0]])
        inv = builtin_inverse([m])
        assert abs(inv.columns[0][0] - 1.0) < 1e-10
        assert abs(inv.columns[1][1] - 1.0) < 1e-10

    def test_abs_vec(self):
        r = builtin_abs([LuxVec([-1.0, 2.0, -3.0])])
        assert r.components == [1.0, 2.0, 3.0]

    def test_sqrt_negative_is_nan(self):
        r = builtin_sqrt([LuxScalar(-1.0)])
        assert is_nan(r)

    def test_pow(self):
        r = builtin_pow([LuxScalar(2.0), LuxScalar(3.0)])
        assert abs(r.value - 8.0) < 1e-10

    def test_min_max(self):
        assert builtin_min([LuxScalar(3.0), LuxScalar(5.0)]).value == 3.0
        assert builtin_max([LuxScalar(3.0), LuxScalar(5.0)]).value == 5.0

    def test_component_wise_vec(self):
        r = builtin_min([LuxVec([1, 5, 3]), LuxVec([4, 2, 6])])
        assert r.components == [1.0, 2.0, 3.0]


# ===== Environment Tests =====

class TestEnvironment:
    def test_define_and_get(self):
        env = Environment()
        env.define("x", LuxScalar(42.0))
        assert env.get("x").value == 42.0

    def test_scope_nesting(self):
        env = Environment()
        env.define("outer", LuxScalar(1.0))
        env.push_scope("inner")
        env.define("inner_var", LuxScalar(2.0))
        assert env.get("outer").value == 1.0
        assert env.get("inner_var").value == 2.0
        env.pop_scope()
        assert env.get("outer").value == 1.0
        assert env.get("inner_var") is None

    def test_set_updates_nearest(self):
        env = Environment()
        env.define("x", LuxScalar(1.0))
        env.push_scope()
        env.set("x", LuxScalar(99.0))
        assert env.get("x").value == 99.0
        env.pop_scope()
        assert env.get("x").value == 99.0


# ===== I/O Tests =====

class TestIO:
    def test_json_to_value_scalar(self):
        v = _json_to_value(3.14)
        assert isinstance(v, LuxScalar)
        assert v.value == 3.14

    def test_json_to_value_vec(self):
        v = _json_to_value([1.0, 2.0, 3.0])
        assert isinstance(v, LuxVec)
        assert v.size == 3

    def test_json_to_value_mat(self):
        v = _json_to_value([[1, 0], [0, 1]])
        assert isinstance(v, LuxMat)
        assert v.size == 2

    def test_json_to_value_typed(self):
        v = _json_to_value({"type": "vec3", "value": [1, 2, 3]})
        assert isinstance(v, LuxVec)


# ===== Integration Tests =====

class TestBatchMode:
    def test_hello_triangle_fragment(self):
        source = """
fragment {
    in frag_color: vec3;
    out color: vec4;
    fn main() {
        color = vec4(frag_color, 1.0);
    }
}
"""
        result = run_batch(source, "fragment")
        assert result["status"] == "completed"
        assert result["statements_executed"] == 1

    def test_let_binding(self):
        source = """
fragment {
    in uv: vec2;
    out color: vec4;
    fn main() {
        let r: scalar = uv.x;
        let g: scalar = uv.y;
        color = vec4(r, g, 0.0, 1.0);
    }
}
"""
        result = run_batch(source, "fragment", dump_vars=True)
        assert result["status"] == "completed"
        assert result["statements_executed"] == 3
        assert len(result.get("variable_trace", [])) == 2  # r and g

    def test_nan_detection(self):
        source = """
fragment {
    out color: vec4;
    fn main() {
        let bad: scalar = sqrt(-1.0);
        color = vec4(bad, 0.0, 0.0, 1.0);
    }
}
"""
        result = run_batch(source, "fragment", check_nan=True)
        assert result["nan_detected"] is True
        assert len(result["nan_events"]) >= 1

    def test_if_statement(self):
        source = """
fragment {
    out color: vec4;
    fn main() {
        let x: scalar = 0.5;
        if (x > 0.25) {
            color = vec4(1.0, 0.0, 0.0, 1.0);
        }
    }
}
"""
        result = run_batch(source, "fragment")
        assert result["status"] == "completed"
        assert result["output"]["value"][0] == 1.0  # red channel

    def test_for_loop(self):
        source = """
fragment {
    out color: vec4;
    fn main() {
        let total: scalar = 0.0;
        for (let i: int = 0; i < 10; i = i + 1) {
            total = total + 1.0;
        }
        color = vec4(total, 0.0, 0.0, 1.0);
    }
}
"""
        result = run_batch(source, "fragment")
        assert result["status"] == "completed"
        # total should be 10.0
        assert result["output"]["value"][0] == 10.0

    def test_function_call(self):
        source = """
fn my_add(a: scalar, b: scalar) -> scalar {
    return a + b;
}

fragment {
    out color: vec4;
    fn main() {
        let sum: scalar = my_add(3.0, 4.0);
        color = vec4(sum, 0.0, 0.0, 1.0);
    }
}
"""
        result = run_batch(source, "fragment")
        assert result["status"] == "completed"
        assert result["output"]["value"][0] == 7.0

    def test_builtin_call(self):
        source = """
fragment {
    out color: vec4;
    fn main() {
        let n: vec3 = normalize(vec3(3.0, 4.0, 0.0));
        color = vec4(n, 1.0);
    }
}
"""
        result = run_batch(source, "fragment")
        assert result["status"] == "completed"
        out = result["output"]["value"]
        assert abs(out[0] - 0.6) < 1e-5
        assert abs(out[1] - 0.8) < 1e-5

    def test_breakpoint_batch(self):
        source = """
fragment {
    out color: vec4;
    fn main() {
        let a: scalar = 1.0;
        let b: scalar = 2.0;
        let c: scalar = a + b;
        color = vec4(c, 0.0, 0.0, 1.0);
    }
}
"""
        result = run_batch(source, "fragment", break_lines=[6], dump_at_break=True)
        assert result["status"] == "completed"
