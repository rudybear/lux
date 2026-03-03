"""Report formatting for auto-type precision analysis."""

from __future__ import annotations

import json

from luxc.autotype.types import Precision, PrecisionMap


def format_report(precision_maps: dict[int | str, PrecisionMap]) -> str:
    """Format a human-readable text report of precision decisions."""
    lines: list[str] = []

    for key, pmap in precision_maps.items():
        stage_label = pmap.stage_type.capitalize()
        lines.append(f"=== {stage_label} Stage: Auto-Type Precision Report ===")
        lines.append(f"{'Variable':<22} {'Type':<8} {'Decision':<10} {'Confidence':<12} Reason")
        lines.append("-" * 80)

        # Sort: fp16 first, then fp32, alphabetical within each
        sorted_decisions = sorted(
            pmap.decisions.items(),
            key=lambda item: (0 if item[1].precision == Precision.FP16 else 1, item[0]),
        )

        fp16_count = 0
        total = len(sorted_decisions)

        for name, decision in sorted_decisions:
            prec_str = decision.precision.value
            conf_str = f"{decision.confidence:.2f}"
            # Infer type from name for display (best-effort)
            type_str = _infer_display_type(name)
            lines.append(f"{name:<22} {type_str:<8} {prec_str:<10} {conf_str:<12} {decision.reason}")
            if decision.precision == Precision.FP16:
                fp16_count += 1

        lines.append("-" * 80)
        if total > 0:
            pct = fp16_count / total * 100
            lines.append(f"Summary: {fp16_count}/{total} variables safe for fp16 ({pct:.1f}%)")
        else:
            lines.append("Summary: no variables analyzed")
        lines.append("")

    return "\n".join(lines)


def format_report_json(precision_maps: dict[int | str, PrecisionMap]) -> str:
    """Format a JSON report of precision decisions."""
    data = {}
    for key, pmap in precision_maps.items():
        stage_data = {
            "stage_type": pmap.stage_type,
            "variables": {},
            "summary": {},
        }
        fp16_count = 0
        total = len(pmap.decisions)
        for name, decision in sorted(pmap.decisions.items()):
            stage_data["variables"][name] = {
                "precision": decision.precision.value,
                "confidence": decision.confidence,
                "reason": decision.reason,
            }
            if decision.precision == Precision.FP16:
                fp16_count += 1
        stage_data["summary"] = {
            "total": total,
            "fp16": fp16_count,
            "fp32": total - fp16_count,
            "fp16_percentage": round(fp16_count / total * 100, 1) if total > 0 else 0,
        }
        data[str(key)] = stage_data

    return json.dumps(data, indent=2)


def _infer_display_type(name: str) -> str:
    """Best-effort type display from variable name."""
    name_lower = name.lower()
    if any(k in name_lower for k in ("normal", "position", "direction", "albedo", "emission", "color")):
        if "color" in name_lower and not name_lower.endswith("color"):
            return "vec4"
        return "vec3"
    if any(k in name_lower for k in ("uv", "texcoord")):
        return "vec2"
    if any(k in name_lower for k in ("matrix", "mvp", "projection", "view")):
        return "mat4"
    if any(k in name_lower for k in ("roughness", "metallic", "ao", "alpha", "opacity", "depth")):
        return "scalar"
    return "?"
