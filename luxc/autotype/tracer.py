"""Dynamic range tracer — wraps the AST interpreter to collect min/max ranges."""

from __future__ import annotations

import copy
import math
import random

from luxc.parser.ast_nodes import Module, StageBlock
from luxc.debug.values import (
    LuxValue, LuxScalar, LuxVec, LuxMat, LuxInt, LuxBool,
    default_value,
)
from luxc.debug.interpreter import Interpreter, VarTrace
from luxc.debug.io import build_default_inputs, _SEMANTIC_DEFAULTS
from luxc.autotype.types import Interval, VarRange


class RangeTracer:
    """Run the interpreter with diverse inputs and aggregate observed value ranges."""

    def __init__(self, module: Module, stage: StageBlock):
        self.module = module
        self.stage = stage

    def trace(self, input_sets: list[dict] | None = None,
              num_random: int = 50) -> dict[str, VarRange]:
        """Run interpreter with diverse inputs, aggregate observed ranges."""
        if input_sets is None:
            input_sets = self._generate_input_sets(num_random)

        all_traces: list[list[VarTrace]] = []
        for inputs in input_sets:
            try:
                interp = Interpreter(self.module, self.stage)
                result = interp.run(inputs=inputs, trace_all=True)
                all_traces.append(result.variable_trace)
            except Exception:
                # Interpreter may fail on some edge-case inputs; skip
                continue

        return self._aggregate_traces(all_traces)

    def _generate_input_sets(self, num_random: int) -> list[dict[str, LuxValue]]:
        """Generate diverse input sets for tracing."""
        sets: list[dict[str, LuxValue]] = []

        # 1. Default inputs (semantic defaults)
        defaults = build_default_inputs(self.stage)
        sets.append(defaults)

        # 2. Zero inputs
        zeros = {}
        for name, val in defaults.items():
            zeros[name] = _zero_like(val)
        sets.append(zeros)

        # 3. Extreme inputs (near fp16 max)
        extremes = {}
        for name, val in defaults.items():
            extremes[name] = _extreme_like(val, name)
        sets.append(extremes)

        # 4. Negative inputs
        negatives = {}
        for name, val in defaults.items():
            negatives[name] = _negate_like(val)
        sets.append(negatives)

        # 5. Axis-aligned normals (6 directions)
        for axis_val in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]):
            axis_inputs = copy.deepcopy(defaults)
            for name in axis_inputs:
                if "normal" in name.lower():
                    axis_inputs[name] = LuxVec([float(v) for v in axis_val])
            sets.append(axis_inputs)

        # 6. Random inputs
        for _ in range(num_random):
            rand_inputs = {}
            for name, val in defaults.items():
                rand_inputs[name] = _random_like(val, name)
            sets.append(rand_inputs)

        return sets

    def _aggregate_traces(self, all_traces: list[list[VarTrace]]) -> dict[str, VarRange]:
        """Aggregate traced variable values into VarRange intervals."""
        ranges: dict[str, VarRange] = {}

        for trace_list in all_traces:
            for trace in trace_list:
                if trace.name not in ranges:
                    n_components = _component_count(trace.type_name)
                    ranges[trace.name] = VarRange(
                        name=trace.name,
                        type_name=trace.type_name,
                        intervals=[Interval() for _ in range(n_components)],
                    )

                vr = ranges[trace.name]
                vr.assignment_count += 1
                _update_intervals(vr.intervals, trace.value)

        return ranges


def _component_count(type_name: str) -> int:
    """Get the number of scalar components for a type."""
    if type_name in ("scalar", "float"):
        return 1
    if type_name == "vec2":
        return 2
    if type_name == "vec3":
        return 3
    if type_name == "vec4":
        return 4
    # For matrices, treat as flat component list
    if type_name == "mat2":
        return 4
    if type_name == "mat3":
        return 9
    if type_name == "mat4":
        return 16
    # Default to 1 for unknown types
    return 1


def _update_intervals(intervals: list[Interval], value: LuxValue) -> None:
    """Update interval list with components from a LuxValue."""
    if isinstance(value, LuxScalar):
        if intervals:
            intervals[0].update(value.value)
    elif isinstance(value, LuxVec):
        for i, comp in enumerate(value.components):
            if i < len(intervals):
                intervals[i].update(comp)
    elif isinstance(value, LuxMat):
        idx = 0
        for col in value.columns:
            for val in col:
                if idx < len(intervals):
                    intervals[idx].update(val)
                idx += 1
    elif isinstance(value, LuxInt):
        if intervals:
            intervals[0].update(float(value.value))
    elif isinstance(value, LuxBool):
        if intervals:
            intervals[0].update(1.0 if value.value else 0.0)


def _zero_like(val: LuxValue) -> LuxValue:
    """Create a zero-valued copy of a LuxValue."""
    if isinstance(val, LuxScalar):
        return LuxScalar(0.0)
    if isinstance(val, LuxVec):
        return LuxVec([0.0] * len(val.components))
    if isinstance(val, LuxMat):
        return LuxMat([[0.0] * len(col) for col in val.columns])
    if isinstance(val, LuxInt):
        return LuxInt(0)
    if isinstance(val, LuxBool):
        return LuxBool(False)
    return copy.deepcopy(val)


def _extreme_like(val: LuxValue, name: str) -> LuxValue:
    """Create an extreme-valued version (near fp16 max for floats)."""
    # Use contextually appropriate extremes
    name_lower = name.lower()
    if "normal" in name_lower:
        magnitude = 1.0  # unit normals
    elif "uv" in name_lower or "texcoord" in name_lower:
        magnitude = 1.0
    elif "roughness" in name_lower or "metallic" in name_lower or "ao" in name_lower:
        magnitude = 1.0
    elif "color" in name_lower or "albedo" in name_lower:
        magnitude = 1.0
    else:
        magnitude = 1000.0

    if isinstance(val, LuxScalar):
        return LuxScalar(magnitude)
    if isinstance(val, LuxVec):
        return LuxVec([magnitude] * len(val.components))
    if isinstance(val, LuxMat):
        return LuxMat([[magnitude] * len(col) for col in val.columns])
    return copy.deepcopy(val)


def _negate_like(val: LuxValue) -> LuxValue:
    """Negate all float components."""
    if isinstance(val, LuxScalar):
        return LuxScalar(-val.value)
    if isinstance(val, LuxVec):
        return LuxVec([-c for c in val.components])
    if isinstance(val, LuxMat):
        return LuxMat([[-v for v in col] for col in val.columns])
    if isinstance(val, LuxInt):
        return LuxInt(-val.value)
    return copy.deepcopy(val)


def _random_like(val: LuxValue, name: str) -> LuxValue:
    """Create a random-valued version with range biased by name."""
    name_lower = name.lower()

    # Choose appropriate random range
    if "normal" in name_lower:
        lo, hi = -1.0, 1.0
    elif "uv" in name_lower or "texcoord" in name_lower:
        lo, hi = 0.0, 1.0
    elif "roughness" in name_lower or "metallic" in name_lower:
        lo, hi = 0.0, 1.0
    elif "ao" in name_lower or "occlusion" in name_lower:
        lo, hi = 0.0, 1.0
    elif "opacity" in name_lower or "alpha" in name_lower:
        lo, hi = 0.0, 1.0
    elif "color" in name_lower or "albedo" in name_lower or "emission" in name_lower:
        lo, hi = 0.0, 1.0
    elif "position" in name_lower:
        lo, hi = -100.0, 100.0
    else:
        lo, hi = -10.0, 10.0

    if isinstance(val, LuxScalar):
        return LuxScalar(random.uniform(lo, hi))
    if isinstance(val, LuxVec):
        comps = [random.uniform(lo, hi) for _ in val.components]
        # Normalize normals
        if "normal" in name_lower and len(comps) >= 3:
            mag = math.sqrt(sum(c * c for c in comps)) or 1.0
            comps = [c / mag for c in comps]
        return LuxVec(comps)
    if isinstance(val, LuxMat):
        return LuxMat([[random.uniform(lo, hi) for _ in col] for col in val.columns])
    if isinstance(val, LuxInt):
        return LuxInt(random.randint(int(lo), int(hi)))
    return copy.deepcopy(val)
