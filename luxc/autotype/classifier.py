"""Precision classifier — combines dynamic, static, and heuristic signals."""

from __future__ import annotations

from luxc.parser.ast_nodes import Module, StageBlock
from luxc.autotype.types import (
    Precision, PrecisionDecision, PrecisionMap, VarRange,
)
from luxc.autotype.heuristics import HeuristicClassifier
from luxc.autotype.tracer import RangeTracer
from luxc.autotype.range_analysis import IntervalAnalysis


# Types that are never candidates for fp16
_ALWAYS_FP32_TYPES = frozenset({
    "int", "uint", "bool",
    "ivec2", "ivec3", "ivec4",
    "uvec2", "uvec3", "uvec4",
    "mat2", "mat3", "mat4", "mat4x3", "mat3x4",
})


class PrecisionClassifier:
    """Combine trace, static, and heuristic signals into final precision decisions."""

    def classify(
        self,
        trace_ranges: dict[str, VarRange],
        static_ranges: dict[str, VarRange],
        heuristics: dict[str, PrecisionDecision],
        stage: StageBlock,
    ) -> PrecisionMap:
        """Produce final precision map from all signals.

        Decision logic (any doubt → fp32):
        1. ANY signal says fp32 → fp32
        2. Trace AND static agree fp16-safe AND heuristic doesn't block → fp16 (confidence 1.0)
        3. Only trace OR only static says safe → fp16 (confidence 0.5)
        4. No data → fp32 (default safe)
        """
        pmap = PrecisionMap(stage_type=stage.stage_type)

        # Collect all variable names across all signals
        all_names = set(trace_ranges.keys()) | set(static_ranges.keys()) | set(heuristics.keys())

        # Stage output names — always excluded
        output_names = {out.name for out in stage.outputs}

        for name in all_names:
            # Always exclude stage outputs
            if name in output_names:
                pmap.decisions[name] = PrecisionDecision(
                    Precision.FP32, 1.0, "stage output variable"
                )
                continue

            trace_vr = trace_ranges.get(name)
            static_vr = static_ranges.get(name)
            heur = heuristics.get(name)

            # Check type: always-fp32 types
            type_name = None
            if trace_vr:
                type_name = trace_vr.type_name
            elif static_vr:
                type_name = static_vr.type_name
            if type_name and type_name in _ALWAYS_FP32_TYPES:
                pmap.decisions[name] = PrecisionDecision(
                    Precision.FP32, 1.0, f"type {type_name} always fp32"
                )
                continue

            # Rule 1: ANY signal says fp32 → fp32
            if heur and heur.precision == Precision.FP32:
                pmap.decisions[name] = PrecisionDecision(
                    Precision.FP32, 1.0, f"heuristic: {heur.reason}"
                )
                continue

            trace_safe = trace_vr is not None and trace_vr.all_fit_fp16()
            static_safe = static_vr is not None and static_vr.all_fit_fp16()

            # If trace or static explicitly shows fp32 needed
            if trace_vr and not trace_safe:
                reason = "dynamic range exceeds fp16"
                if trace_vr.intervals and (trace_vr.intervals[0].has_nan or trace_vr.intervals[0].has_inf):
                    reason = "NaN/Inf observed in dynamic trace"
                pmap.decisions[name] = PrecisionDecision(Precision.FP32, 1.0, reason)
                continue
            if static_vr and not static_safe:
                reason = "static range exceeds fp16"
                if static_vr.intervals and (static_vr.intervals[0].has_nan or static_vr.intervals[0].has_inf):
                    reason = "NaN/Inf possible in static analysis"
                pmap.decisions[name] = PrecisionDecision(Precision.FP32, 1.0, reason)
                continue

            # Rule 2: Both agree safe + heuristic doesn't block
            if trace_safe and static_safe:
                heur_reason = f" + heuristic: {heur.reason}" if heur and heur.precision == Precision.FP16 else ""
                pmap.decisions[name] = PrecisionDecision(
                    Precision.FP16, 1.0,
                    f"both dynamic and static ranges fit fp16{heur_reason}"
                )
                continue

            # Rule 3: Only one signal says safe
            if trace_safe and not static_vr:
                pmap.decisions[name] = PrecisionDecision(
                    Precision.FP16, 0.5,
                    "dynamic range fits fp16 (no static data)"
                )
                continue
            if static_safe and not trace_vr:
                pmap.decisions[name] = PrecisionDecision(
                    Precision.FP16, 0.5,
                    "static range fits fp16 (no dynamic data)"
                )
                continue

            # Heuristic-only fp16
            if heur and heur.precision == Precision.FP16:
                pmap.decisions[name] = PrecisionDecision(
                    Precision.FP16, 0.5,
                    f"heuristic only: {heur.reason}"
                )
                continue

            # Rule 4: No data → fp32
            pmap.decisions[name] = PrecisionDecision(
                Precision.FP32, 0.5, "insufficient data, defaulting to fp32"
            )

        return pmap


def run_auto_type_analysis(module: Module, stage: StageBlock,
                           num_random: int = 50) -> PrecisionMap:
    """Top-level entry point — runs all three analyses and combines."""
    # 1. Dynamic trace
    tracer = RangeTracer(module, stage)
    try:
        trace_ranges = tracer.trace(num_random=num_random)
    except Exception:
        trace_ranges = {}

    # 2. Static interval analysis
    static_analyzer = IntervalAnalysis()
    try:
        static_ranges = static_analyzer.analyze_stage(stage, module)
    except Exception:
        static_ranges = {}

    # 3. Heuristic classification
    heuristic = HeuristicClassifier()
    try:
        heur_decisions = heuristic.classify(module, stage)
    except Exception:
        heur_decisions = {}

    # Combine
    classifier = PrecisionClassifier()
    return classifier.classify(trace_ranges, static_ranges, heur_decisions, stage)
