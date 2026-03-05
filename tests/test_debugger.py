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


# ===== Expression Parser Tests =====

class TestExprParser:
    def test_simple_comparison(self):
        from luxc.debug.expr_parser import parse_debug_expr
        ast = parse_debug_expr("x > 0.5")
        assert ast is not None
        from luxc.parser.ast_nodes import BinaryOp
        assert isinstance(ast, BinaryOp)
        assert ast.op == ">"

    def test_function_call(self):
        from luxc.debug.expr_parser import parse_debug_expr
        ast = parse_debug_expr("dot(n, l)")
        assert ast is not None
        from luxc.parser.ast_nodes import CallExpr
        assert isinstance(ast, CallExpr)

    def test_field_access(self):
        from luxc.debug.expr_parser import parse_debug_expr
        ast = parse_debug_expr("v.x")
        assert ast is not None

    def test_arithmetic(self):
        from luxc.debug.expr_parser import parse_debug_expr
        ast = parse_debug_expr("a + b * 2.0")
        assert ast is not None

    def test_invalid_expr(self):
        from luxc.debug.expr_parser import parse_debug_expr
        with pytest.raises(ValueError):
            parse_debug_expr("@#$%^")


# ===== Conditional Breakpoint Tests =====

class TestConditionalBreakpoints:
    def test_conditional_bp_stops_when_met(self):
        source = """
fragment {
    out color: vec4;
    fn main() {
        let x: scalar = 0.0;
        let y: scalar = 0.5;
        let z: scalar = 1.0;
        color = vec4(x, y, z, 1.0);
    }
}
"""
        result = run_batch(source, "fragment", break_lines=[6], dump_at_break=True)
        assert result["status"] == "completed"
        # Line 6 is "let y: scalar = 0.5;" — breakpoint should trigger
        assert len(result.get("break_dumps", [])) >= 1

    def test_conditional_bp_skips_when_not_met(self):
        """A breakpoint with condition 'x > 10.0' should not stop when x = 0.0."""
        source = """
fragment {
    out color: vec4;
    fn main() {
        let x: scalar = 0.0;
        let y: scalar = 1.0;
        color = vec4(x, y, 0.0, 1.0);
    }
}
"""
        from luxc.parser.tree_builder import parse_lux
        from luxc.analysis.type_checker import type_check
        from luxc.optimization.const_fold import constant_fold
        from luxc.debug.io import build_default_inputs

        module = parse_lux(source)
        type_check(module)
        constant_fold(module)
        stage = module.stages[0]
        lines = source.splitlines()
        interp = Interpreter(module, stage, source_lines=lines)
        dbg = Debugger(interp)

        # Add conditional breakpoint at line 7 (after x is defined at line 6)
        # x = 0.0, so "x > 10.0" should be false and bp should not fire
        bp = dbg.add_breakpoint(7, condition="x > 10.0")

        break_hits = []
        def on_break(hit):
            break_hits.append(hit.line)
        dbg._on_break = on_break

        result = dbg.run_batch(inputs=build_default_inputs(stage))
        # Breakpoint should not have been hit since x = 0.0 < 10.0
        assert bp.hit_count == 0


# ===== Watch Expression Tests =====

class TestWatchExpressions:
    def test_watch_evaluates(self):
        source = """
fragment {
    out color: vec4;
    fn main() {
        let x: scalar = 3.0;
        let y: scalar = 4.0;
        color = vec4(x, y, 0.0, 1.0);
    }
}
"""
        from luxc.parser.tree_builder import parse_lux
        from luxc.analysis.type_checker import type_check
        from luxc.optimization.const_fold import constant_fold
        from luxc.debug.io import build_default_inputs

        module = parse_lux(source)
        type_check(module)
        constant_fold(module)
        stage = module.stages[0]
        lines = source.splitlines()
        interp = Interpreter(module, stage, source_lines=lines)
        dbg = Debugger(interp)

        # Add a watch for x + y
        w = dbg.add_watch("x + y")

        # Set up to evaluate after running partway
        dbg.add_breakpoint(7)  # break before color assignment

        stopped = []
        def on_break(hit):
            results = dbg.evaluate_watches()
            stopped.append(results)
        dbg._on_break = on_break

        dbg.run_batch(inputs=build_default_inputs(stage))

        # Check that watch was evaluated
        assert len(stopped) >= 1
        watch_results = stopped[0]
        assert len(watch_results) == 1
        entry, val, changed = watch_results[0]
        assert val is not None
        assert abs(val.value - 7.0) < 1e-6


# ===== Break-on-NaN Tests =====

class TestBreakOnNaN:
    def test_break_on_nan_batch(self):
        source = """
fragment {
    out color: vec4;
    fn main() {
        let good: scalar = 1.0;
        let bad: scalar = sqrt(-1.0);
        color = vec4(bad, 0.0, 0.0, 1.0);
    }
}
"""
        result = run_batch(source, "fragment", break_on_nan=True, check_nan=True)
        assert result["nan_detected"] is True


# ===== Time-Travel Tests =====

class TestTimeTravel:
    def test_time_travel_recording(self):
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
        from luxc.parser.tree_builder import parse_lux
        from luxc.analysis.type_checker import type_check
        from luxc.optimization.const_fold import constant_fold
        from luxc.debug.io import build_default_inputs

        module = parse_lux(source)
        type_check(module)
        constant_fold(module)
        stage = module.stages[0]
        lines = source.splitlines()
        interp = Interpreter(module, stage, source_lines=lines)
        dbg = Debugger(interp)

        result = dbg.run_batch(inputs=build_default_inputs(stage))

        # Time travel should have recorded snapshots
        assert len(dbg.time_travel.history) > 0

    def test_reverse_step(self):
        source = """
fragment {
    out color: vec4;
    fn main() {
        let a: scalar = 1.0;
        let b: scalar = 2.0;
        color = vec4(a, b, 0.0, 1.0);
    }
}
"""
        from luxc.parser.tree_builder import parse_lux
        from luxc.analysis.type_checker import type_check
        from luxc.optimization.const_fold import constant_fold
        from luxc.debug.io import build_default_inputs

        module = parse_lux(source)
        type_check(module)
        constant_fold(module)
        stage = module.stages[0]
        lines = source.splitlines()
        interp = Interpreter(module, stage, source_lines=lines)
        dbg = Debugger(interp)

        dbg.run_batch(inputs=build_default_inputs(stage))

        # Should be able to reverse step
        snap = dbg.reverse_step()
        assert snap is not None
        assert snap.line > 0

    def test_goto_step(self):
        source = """
fragment {
    out color: vec4;
    fn main() {
        let x: scalar = 42.0;
        color = vec4(x, 0.0, 0.0, 1.0);
    }
}
"""
        from luxc.parser.tree_builder import parse_lux
        from luxc.analysis.type_checker import type_check
        from luxc.optimization.const_fold import constant_fold
        from luxc.debug.io import build_default_inputs

        module = parse_lux(source)
        type_check(module)
        constant_fold(module)
        stage = module.stages[0]
        lines = source.splitlines()
        interp = Interpreter(module, stage, source_lines=lines)
        dbg = Debugger(interp)

        dbg.run_batch(inputs=build_default_inputs(stage))

        # Go to first step
        snap = dbg.goto_step(1)
        assert snap is not None
        assert snap.step_index == 1


# ===== LuxImage Tests =====

class TestLuxImage:
    def test_make_solid_image(self):
        from luxc.debug.values import make_solid_image
        img = make_solid_image(255, 0, 0)
        assert img.width == 1
        assert img.height == 1
        result = img.sample_bilinear(0.5, 0.5)
        assert isinstance(result, LuxVec)
        assert abs(result.components[0] - 1.0) < 0.01  # Red
        assert abs(result.components[1] - 0.0) < 0.01  # Green
        assert abs(result.components[2] - 0.0) < 0.01  # Blue
        assert abs(result.components[3] - 1.0) < 0.01  # Alpha

    def test_sample_bilinear_clamp(self):
        from luxc.debug.values import make_solid_image
        img = make_solid_image(128, 128, 255)
        img.wrap = "clamp"
        result = img.sample_bilinear(0.5, 0.5)
        assert isinstance(result, LuxVec)
        assert abs(result.components[2] - 1.0) < 0.01  # Blue


# ===== Heatmap Tests =====

class TestHeatmap:
    def test_run_pixel_grid(self):
        from luxc.debug.heatmap import run_pixel_grid
        source = """
fragment {
    in uv: vec2;
    out color: vec4;
    fn main() {
        color = vec4(uv.x, uv.y, 0.0, 1.0);
    }
}
"""
        result = run_pixel_grid(source, "fragment", grid_size=(4, 4))
        assert result.width == 4
        assert result.height == 4
        assert len(result.pixels) == 16

    def test_write_ppm(self, tmp_path):
        from luxc.debug.heatmap import run_pixel_grid, write_ppm
        source = """
fragment {
    in uv: vec2;
    out color: vec4;
    fn main() {
        color = vec4(uv.x, uv.y, 0.0, 1.0);
    }
}
"""
        result = run_pixel_grid(source, "fragment", grid_size=(4, 4))
        ppm_path = str(tmp_path / "test.ppm")
        write_ppm(ppm_path, result)

        # Verify PPM file exists and has correct header
        with open(ppm_path, "rb") as f:
            header = f.read(10)
            assert header.startswith(b"P6\n4 4\n")

    def test_nan_heatmap(self):
        from luxc.debug.heatmap import run_pixel_grid
        source = """
fragment {
    in uv: vec2;
    out color: vec4;
    fn main() {
        let bad: scalar = sqrt(-1.0);
        color = vec4(bad, uv.y, 0.0, 1.0);
    }
}
"""
        result = run_pixel_grid(source, "fragment", grid_size=(2, 2), mode="nan")
        assert all(p.nan_count > 0 for p in result.pixels)


# ===== Default Material Tests =====

class TestDefaultMaterial:
    def test_build_default_material(self):
        from luxc.debug.io import build_default_material
        mat = build_default_material()
        assert isinstance(mat, LuxStruct)
        assert mat.type_name == "BindlessMaterialData"
        assert "baseColorFactor" in mat.fields
        assert "roughnessFactor" in mat.fields
        assert mat.fields["roughnessFactor"].value == 0.5
        assert mat.fields["base_color_tex_index"].value == -1


# ===== Texture-Aware Builtins Tests =====

class TestTextureBuiltins:
    def test_sample_with_image(self):
        from luxc.debug.builtins import builtin_sample
        from luxc.debug.values import make_solid_image
        img = make_solid_image(255, 128, 0)
        uv = LuxVec([0.5, 0.5])
        result = builtin_sample([img, uv])
        assert isinstance(result, LuxVec)
        assert abs(result.components[0] - 1.0) < 0.01  # Red = 255/255

    def test_sample_without_image(self):
        from luxc.debug.builtins import builtin_sample
        result = builtin_sample([LuxInt(0), LuxVec([0.5, 0.5])])
        assert isinstance(result, LuxVec)
        assert result.components[0] == 0.8  # Default grey

    def test_sample_bindless(self):
        from luxc.debug.builtins import builtin_sample_bindless
        from luxc.debug.values import make_solid_image
        img1 = make_solid_image(255, 0, 0)
        img2 = make_solid_image(0, 255, 0)
        tex_array = [img1, img2]
        result = builtin_sample_bindless([tex_array, LuxInt(1), LuxVec([0.5, 0.5])])
        assert isinstance(result, LuxVec)
        assert abs(result.components[1] - 1.0) < 0.01  # Green from img2


# ===== List Indexing Tests =====

class TestListIndexing:
    def test_list_indexing_in_interpreter(self):
        """Test that list[index] works for SSBO-like array access."""
        from luxc.parser.tree_builder import parse_lux
        from luxc.analysis.type_checker import type_check
        from luxc.optimization.const_fold import constant_fold

        source = """
fragment {
    out color: vec4;
    fn main() {
        color = vec4(1.0);
    }
}
"""
        module = parse_lux(source)
        type_check(module)
        constant_fold(module)
        stage = module.stages[0]
        interp = Interpreter(module, stage)

        # Manually define a list and test indexing
        materials = [
            LuxStruct("Test", {"val": LuxScalar(42.0)}),
            LuxStruct("Test", {"val": LuxScalar(99.0)}),
        ]
        interp.env.define("materials", materials)

        # Parse and eval an index expression
        from luxc.parser.ast_nodes import IndexAccess, VarRef, NumberLit
        idx_expr = IndexAccess(VarRef("materials"), NumberLit("0"))
        result = interp.eval_expr(idx_expr)
        assert isinstance(result, LuxStruct)
        assert result.fields["val"].value == 42.0
