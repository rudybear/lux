"""Tests for Auto-Type: automatic precision optimization."""

import math
import json
import pytest

from luxc.autotype.types import (
    Precision, Interval, VarRange, PrecisionDecision, PrecisionMap,
)
from luxc.autotype.heuristics import HeuristicClassifier, _collect_var_types
from luxc.autotype.range_analysis import IntervalAnalysis, _apply_binary_op
from luxc.autotype.tracer import RangeTracer, _component_count, _update_intervals
from luxc.autotype.classifier import PrecisionClassifier, run_auto_type_analysis
from luxc.autotype.report import format_report, format_report_json

from luxc.parser.tree_builder import parse_lux
from luxc.analysis.type_checker import type_check
from luxc.optimization.const_fold import constant_fold
from luxc.analysis.layout_assigner import assign_layouts
from luxc.codegen.spirv_builder import generate_spirv
from luxc.debug.values import LuxScalar, LuxVec, LuxInt, LuxBool


# ===== Interval Arithmetic =====

class TestInterval:
    def test_empty_interval(self):
        iv = Interval()
        assert iv.is_empty
        assert not iv.fits_fp16()

    def test_update_single_value(self):
        iv = Interval()
        iv.update(0.5)
        assert iv.lo == 0.5
        assert iv.hi == 0.5
        assert iv.fits_fp16()

    def test_update_range(self):
        iv = Interval()
        iv.update(-1.0)
        iv.update(1.0)
        assert iv.lo == -1.0
        assert iv.hi == 1.0
        assert iv.fits_fp16()

    def test_update_nan(self):
        iv = Interval()
        iv.update(0.5)
        iv.update(float('nan'))
        assert iv.has_nan
        assert not iv.fits_fp16()

    def test_update_inf(self):
        iv = Interval()
        iv.update(0.5)
        iv.update(float('inf'))
        assert iv.has_inf
        assert not iv.fits_fp16()

    def test_fits_fp16_within_range(self):
        iv = Interval(-100.0, 100.0)
        assert iv.fits_fp16()

    def test_exceeds_fp16(self):
        iv = Interval(-70000.0, 70000.0)
        assert not iv.fits_fp16()

    def test_fp16_boundary(self):
        iv = Interval(-65504.0, 65504.0)
        assert iv.fits_fp16()
        iv2 = Interval(-65505.0, 65505.0)
        assert not iv2.fits_fp16()

    def test_merge(self):
        iv1 = Interval(0, 1)
        iv2 = Interval(-1, 0.5)
        iv1.merge(iv2)
        assert iv1.lo == -1
        assert iv1.hi == 1

    def test_merge_nan(self):
        iv1 = Interval(0, 1)
        iv2 = Interval(0, 1, has_nan=True)
        iv1.merge(iv2)
        assert iv1.has_nan

    def test_repr(self):
        iv = Interval(0, 1)
        assert "0" in repr(iv)
        assert "1" in repr(iv)
        empty = Interval()
        assert "empty" in repr(empty)


class TestIntervalArithmetic:
    def test_add(self):
        a = Interval(1, 3)
        b = Interval(2, 5)
        result = _apply_binary_op("+", a, b)
        assert result.lo == 3
        assert result.hi == 8

    def test_subtract(self):
        a = Interval(1, 3)
        b = Interval(2, 5)
        result = _apply_binary_op("-", a, b)
        assert result.lo == -4  # 1 - 5
        assert result.hi == 1   # 3 - 2

    def test_multiply(self):
        a = Interval(-2, 3)
        b = Interval(1, 4)
        result = _apply_binary_op("*", a, b)
        assert result.lo == -8  # -2 * 4
        assert result.hi == 12  # 3 * 4

    def test_divide_safe(self):
        a = Interval(2, 6)
        b = Interval(1, 3)
        result = _apply_binary_op("/", a, b)
        assert abs(result.lo - 2/3) < 1e-10
        assert abs(result.hi - 6.0) < 1e-10

    def test_divide_by_zero(self):
        a = Interval(1, 2)
        b = Interval(-1, 1)  # Contains zero
        result = _apply_binary_op("/", a, b)
        assert result.lo == float('-inf')
        assert result.hi == float('inf')

    def test_comparison(self):
        a = Interval(1, 3)
        b = Interval(2, 5)
        result = _apply_binary_op("<", a, b)
        assert result.lo == 0
        assert result.hi == 1


# ===== VarRange =====

class TestVarRange:
    def test_all_fit_fp16(self):
        vr = VarRange("test", "vec3", [
            Interval(0, 1), Interval(-1, 1), Interval(0, 0.5),
        ])
        assert vr.all_fit_fp16()

    def test_one_component_exceeds(self):
        vr = VarRange("test", "vec3", [
            Interval(0, 1), Interval(-1e6, 1e6), Interval(0, 0.5),
        ])
        assert not vr.all_fit_fp16()


# ===== PrecisionMap =====

class TestPrecisionMap:
    def test_fp16_vars(self):
        pm = PrecisionMap("fragment", {
            "roughness": PrecisionDecision(Precision.FP16, 1.0, "test"),
            "position": PrecisionDecision(Precision.FP32, 1.0, "test"),
            "metallic": PrecisionDecision(Precision.FP16, 0.5, "test"),
        })
        fp16 = pm.fp16_vars()
        assert "roughness" in fp16
        assert "metallic" in fp16
        assert "position" not in fp16


# ===== Heuristic Classifier =====

class TestHeuristicClassifier:
    def _make_stage(self, src):
        m = parse_lux(src)
        type_check(m)
        return m, m.stages[0]

    def test_name_patterns_fp16(self):
        src = """
        fragment {
            in normal: vec3;
            out frag_color: vec4;
            fn main() {
                let roughness: scalar = 0.5;
                let albedo: vec3 = vec3(0.8, 0.8, 0.8);
                let uv_coord: vec2 = vec2(0.5, 0.5);
                frag_color = vec4(albedo, 1.0);
            }
        }
        """
        m, stage = self._make_stage(src)
        heur = HeuristicClassifier()
        decisions = heur.classify(m, stage)
        assert decisions["roughness"].precision == Precision.FP16
        assert decisions["albedo"].precision == Precision.FP16
        assert decisions["uv_coord"].precision == Precision.FP16

    def test_name_patterns_fp32(self):
        src = """
        fragment {
            in world_position: vec3;
            out frag_color: vec4;
            fn main() {
                let depth: scalar = 0.99;
                frag_color = vec4(world_position, 1.0);
            }
        }
        """
        m, stage = self._make_stage(src)
        heur = HeuristicClassifier()
        decisions = heur.classify(m, stage)
        assert decisions["depth"].precision == Precision.FP32

    def test_integer_types_fp32(self):
        src = """
        fragment {
            out frag_color: vec4;
            fn main() {
                let index: int = 0;
                frag_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
        """
        m, stage = self._make_stage(src)
        heur = HeuristicClassifier()
        decisions = heur.classify(m, stage)
        assert decisions["index"].precision == Precision.FP32

    def test_matrix_types_fp32(self):
        src = """
        fragment {
            out frag_color: vec4;
            fn main() {
                let view_matrix: mat4 = mat4(1.0);
                frag_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
        """
        m, stage = self._make_stage(src)
        heur = HeuristicClassifier()
        decisions = heur.classify(m, stage)
        # mat4 type → always fp32
        assert decisions["view_matrix"].precision == Precision.FP32

    def test_output_always_fp32(self):
        src = """
        fragment {
            out frag_color: vec4;
            fn main() {
                frag_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
        """
        m, stage = self._make_stage(src)
        heur = HeuristicClassifier()
        decisions = heur.classify(m, stage)
        assert decisions["frag_color"].precision == Precision.FP32

    def test_division_denominator_fp32(self):
        src = """
        fragment {
            out frag_color: vec4;
            fn main() {
                let a: scalar = 1.0;
                let b: scalar = 0.5;
                let c: scalar = a / b;
                frag_color = vec4(c, 0.0, 0.0, 1.0);
            }
        }
        """
        m, stage = self._make_stage(src)
        heur = HeuristicClassifier()
        decisions = heur.classify(m, stage)
        # b is used as denominator → fp32
        assert decisions["b"].precision == Precision.FP32


# ===== Static Interval Analysis =====

class TestStaticIntervalAnalysis:
    def _analyze(self, src):
        m = parse_lux(src)
        type_check(m)
        constant_fold(m)
        stage = m.stages[0]
        analyzer = IntervalAnalysis()
        return analyzer.analyze_stage(stage, m)

    def test_literal_range(self):
        src = """
        fragment {
            out frag_color: vec4;
            fn main() {
                let x: scalar = 0.5;
                frag_color = vec4(x, 0.0, 0.0, 1.0);
            }
        }
        """
        ranges = self._analyze(src)
        assert "x" in ranges
        assert ranges["x"].intervals[0].lo == 0.5
        assert ranges["x"].intervals[0].hi == 0.5
        assert ranges["x"].all_fit_fp16()

    def test_addition_range(self):
        src = """
        fragment {
            in roughness: scalar;
            out frag_color: vec4;
            fn main() {
                let x: scalar = roughness + 0.1;
                frag_color = vec4(x, 0.0, 0.0, 1.0);
            }
        }
        """
        ranges = self._analyze(src)
        # roughness ∈ [0,1], so x ∈ [0.1, 1.1]
        assert "x" in ranges
        assert ranges["x"].intervals[0].lo == pytest.approx(0.1)
        assert ranges["x"].intervals[0].hi == pytest.approx(1.1)

    def test_normalize_range(self):
        src = """
        import glsl_ext;
        fragment {
            in normal: vec3;
            out frag_color: vec4;
            fn main() {
                let n: vec3 = normalize(normal);
                frag_color = vec4(n, 1.0);
            }
        }
        """
        ranges = self._analyze(src)
        assert "n" in ranges
        for iv in ranges["n"].intervals:
            assert iv.lo >= -1.0
            assert iv.hi <= 1.0

    def test_clamp_range(self):
        src = """
        import glsl_ext;
        fragment {
            in roughness: scalar;
            out frag_color: vec4;
            fn main() {
                let r: scalar = clamp(roughness, 0.0, 1.0);
                frag_color = vec4(r, 0.0, 0.0, 1.0);
            }
        }
        """
        ranges = self._analyze(src)
        assert "r" in ranges
        assert ranges["r"].intervals[0].lo == 0.0
        assert ranges["r"].intervals[0].hi == 1.0

    def test_sample_range(self):
        src = """
        import glsl_ext;
        fragment {
            sampler2d diffuse_map;
            in uv: vec2;
            out frag_color: vec4;
            fn main() {
                let color: vec4 = sample(diffuse_map, uv);
                frag_color = color;
            }
        }
        """
        ranges = self._analyze(src)
        assert "color" in ranges
        for iv in ranges["color"].intervals:
            assert iv.lo >= 0.0
            assert iv.hi <= 1.0


# ===== Dynamic Tracer =====

class TestRangeTracer:
    def test_component_count(self):
        assert _component_count("scalar") == 1
        assert _component_count("vec2") == 2
        assert _component_count("vec3") == 3
        assert _component_count("vec4") == 4
        assert _component_count("mat4") == 16

    def test_update_intervals_scalar(self):
        intervals = [Interval()]
        _update_intervals(intervals, LuxScalar(0.5))
        assert intervals[0].lo == 0.5
        assert intervals[0].hi == 0.5

    def test_update_intervals_vec(self):
        intervals = [Interval(), Interval(), Interval()]
        _update_intervals(intervals, LuxVec([1.0, -2.0, 3.0]))
        assert intervals[0].lo == 1.0
        assert intervals[1].lo == -2.0
        assert intervals[2].hi == 3.0

    def test_trace_simple_shader(self):
        src = """
        fragment {
            in roughness: scalar;
            out frag_color: vec4;
            fn main() {
                let r: scalar = roughness;
                frag_color = vec4(r, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)
        constant_fold(m)
        stage = m.stages[0]
        tracer = RangeTracer(m, stage)
        ranges = tracer.trace(num_random=5)
        assert "r" in ranges
        # Roughness should be traced with various values
        assert ranges["r"].assignment_count > 0


# ===== Precision Classifier =====

class TestPrecisionClassifier:
    def test_both_agree_fp16(self):
        trace = {"x": VarRange("x", "scalar", [Interval(0, 1)])}
        static = {"x": VarRange("x", "scalar", [Interval(0, 1)])}
        heur = {}
        stage = _make_minimal_stage()
        classifier = PrecisionClassifier()
        result = classifier.classify(trace, static, heur, stage)
        assert result.decisions["x"].precision == Precision.FP16
        assert result.decisions["x"].confidence == 1.0

    def test_heuristic_veto(self):
        trace = {"x": VarRange("x", "scalar", [Interval(0, 1)])}
        static = {"x": VarRange("x", "scalar", [Interval(0, 1)])}
        heur = {"x": PrecisionDecision(Precision.FP32, 1.0, "position")}
        stage = _make_minimal_stage()
        classifier = PrecisionClassifier()
        result = classifier.classify(trace, static, heur, stage)
        assert result.decisions["x"].precision == Precision.FP32

    def test_trace_only(self):
        trace = {"x": VarRange("x", "scalar", [Interval(0, 1)])}
        static = {}
        heur = {}
        stage = _make_minimal_stage()
        classifier = PrecisionClassifier()
        result = classifier.classify(trace, static, heur, stage)
        assert result.decisions["x"].precision == Precision.FP16
        assert result.decisions["x"].confidence == 0.5

    def test_static_only(self):
        trace = {}
        static = {"x": VarRange("x", "scalar", [Interval(0, 1)])}
        heur = {}
        stage = _make_minimal_stage()
        classifier = PrecisionClassifier()
        result = classifier.classify(trace, static, heur, stage)
        assert result.decisions["x"].precision == Precision.FP16
        assert result.decisions["x"].confidence == 0.5

    def test_no_data_defaults_fp32(self):
        trace = {}
        static = {}
        heur = {"x": PrecisionDecision(Precision.UNKNOWN, 0.5, "unknown")}
        stage = _make_minimal_stage()
        classifier = PrecisionClassifier()
        result = classifier.classify(trace, static, heur, stage)
        assert result.decisions["x"].precision == Precision.FP32

    def test_nan_forces_fp32(self):
        trace = {"x": VarRange("x", "scalar", [Interval(0, 1, has_nan=True)])}
        static = {"x": VarRange("x", "scalar", [Interval(0, 1)])}
        heur = {}
        stage = _make_minimal_stage()
        classifier = PrecisionClassifier()
        result = classifier.classify(trace, static, heur, stage)
        assert result.decisions["x"].precision == Precision.FP32

    def test_integer_always_fp32(self):
        trace = {"idx": VarRange("idx", "int", [Interval(0, 10)])}
        static = {}
        heur = {}
        stage = _make_minimal_stage()
        classifier = PrecisionClassifier()
        result = classifier.classify(trace, static, heur, stage)
        assert result.decisions["idx"].precision == Precision.FP32


# ===== Report Formatting =====

class TestReport:
    def test_text_report(self):
        pm = PrecisionMap("fragment", {
            "roughness": PrecisionDecision(Precision.FP16, 1.0, "material param [0,1]"),
            "position": PrecisionDecision(Precision.FP32, 1.0, "world-space position"),
        })
        report = format_report({0: pm})
        assert "Fragment Stage" in report
        assert "roughness" in report
        assert "fp16" in report
        assert "fp32" in report
        assert "1/2" in report
        assert "50.0%" in report

    def test_json_report(self):
        pm = PrecisionMap("fragment", {
            "roughness": PrecisionDecision(Precision.FP16, 1.0, "test"),
            "position": PrecisionDecision(Precision.FP32, 1.0, "test"),
        })
        report = format_report_json({0: pm})
        data = json.loads(report)
        assert "0" in data
        assert data["0"]["summary"]["fp16"] == 1
        assert data["0"]["summary"]["fp32"] == 1


# ===== Integration: run_auto_type_analysis =====

class TestAutoTypeIntegration:
    def test_simple_pbr_fragment(self):
        src = """
        import glsl_ext;
        fragment {
            in normal: vec3;
            in uv: vec2;
            in world_position: vec3;
            sampler2d albedo_map;
            uniform Material { roughness: scalar, metallic: scalar }
            out frag_color: vec4;

            fn main() {
                let n: vec3 = normalize(normal);
                let albedo: vec4 = sample(albedo_map, uv);
                let color: vec3 = albedo.xyz * roughness;
                frag_color = vec4(color, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)
        constant_fold(m)
        stage = m.stages[0]
        pm = run_auto_type_analysis(m, stage, num_random=5)
        assert isinstance(pm, PrecisionMap)
        assert pm.stage_type == "fragment"
        # Output should be fp32
        assert pm.decisions["frag_color"].precision == Precision.FP32
        # n (normalized) should be fp16 or at least analyzed
        if "n" in pm.decisions:
            assert pm.decisions["n"].precision in (Precision.FP16, Precision.FP32)


# ===== SPIR-V RelaxedPrecision Emission =====

class TestRelaxedPrecisionEmission:
    def _compile_with_precision(self, src, precision_map):
        m = parse_lux(src)
        type_check(m)
        constant_fold(m)
        assign_layouts(m)
        return generate_spirv(m, m.stages[0], precision_map=precision_map)

    def test_relaxed_precision_on_ssa(self):
        """SSA-path variables should get RelaxedPrecision on their value ID."""
        src = """
        fragment {
            in roughness: scalar;
            out frag_color: vec4;
            fn main() {
                let r: scalar = roughness;
                frag_color = vec4(r, r, r, 1.0);
            }
        }
        """
        asm = self._compile_with_precision(src, {"r": "fp16"})
        assert "OpDecorate" in asm
        assert "RelaxedPrecision" in asm

    def test_no_precision_map_no_decorations(self):
        """Without precision_map, no RelaxedPrecision should appear."""
        src = """
        fragment {
            in roughness: scalar;
            out frag_color: vec4;
            fn main() {
                let r: scalar = roughness;
                frag_color = vec4(r, r, r, 1.0);
            }
        }
        """
        asm = self._compile_with_precision(src, None)
        assert "RelaxedPrecision" not in asm

    def test_relaxed_precision_on_mutable_var(self):
        """Mutable variables should get RelaxedPrecision on OpVariable and OpLoad."""
        src = """
        fragment {
            in roughness: scalar;
            out frag_color: vec4;
            fn main() {
                let r: scalar = roughness;
                r = r + 0.1;
                frag_color = vec4(r, r, r, 1.0);
            }
        }
        """
        asm = self._compile_with_precision(src, {"r": "fp16"})
        assert "RelaxedPrecision" in asm

    def test_empty_precision_map(self):
        """Empty precision map should produce no RelaxedPrecision."""
        src = """
        fragment {
            out frag_color: vec4;
            fn main() {
                frag_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
        """
        asm = self._compile_with_precision(src, {})
        assert "RelaxedPrecision" not in asm

    def test_multiple_vars_decorated(self):
        """Multiple fp16 variables should each get RelaxedPrecision."""
        src = """
        fragment {
            in roughness: scalar;
            in metallic: scalar;
            out frag_color: vec4;
            fn main() {
                let r: scalar = roughness;
                let m: scalar = metallic;
                frag_color = vec4(r, m, 0.0, 1.0);
            }
        }
        """
        asm = self._compile_with_precision(src, {"r": "fp16", "m": "fp16"})
        # Count RelaxedPrecision decorations (at least 2)
        count = asm.count("RelaxedPrecision")
        assert count >= 2


# ===== Helpers =====

def _make_minimal_stage():
    """Create a minimal StageBlock for classifier tests."""
    src = """
    fragment {
        out frag_color: vec4;
        fn main() {
            frag_color = vec4(1.0, 0.0, 0.0, 1.0);
        }
    }
    """
    m = parse_lux(src)
    return m.stages[0]
