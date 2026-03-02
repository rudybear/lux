"""CPU-side rendering analysis tool.

Analyzes draw call patterns from RenderDoc capture events to identify
CPU-side bottlenecks: batching inefficiency, descriptor churn, buffer
padding waste, and redundant pipeline binds.

Usage:
    python tools/cpu_analysis.py metrics.json
    python tools/cpu_analysis.py --format table metrics.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path


# Thresholds
MIN_DRAWS_PER_PIPELINE_BIND = 2.0
MIN_DRAWS_PER_DESCRIPTOR_BIND = 3.0
MAX_BUFFER_PADDING_RATIO = 0.15  # 15% padding waste
MAX_REDUNDANT_BIND_RATIO = 0.10  # 10% redundant binds


class CPUFinding:
    """A CPU-side performance finding."""

    def __init__(self, severity: str, category: str, message: str,
                 metric_value: float = 0.0, threshold: float = 0.0,
                 recommendation: str = ""):
        self.severity = severity        # "critical", "warning", "info"
        self.category = category        # "batching", "descriptor", "buffer", "pipeline"
        self.message = message
        self.metric_value = metric_value
        self.threshold = threshold
        self.recommendation = recommendation

    def __repr__(self):
        return f"[{self.severity}] {self.category}: {self.message}"

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "recommendation": self.recommendation,
        }


def analyze_draw_batching(metrics: dict) -> list[CPUFinding]:
    """Analyze draw call batching efficiency.

    Checks how many draws are issued per pipeline bind. A low ratio means
    frequent pipeline state changes, which stalls the command processor.
    """
    findings = []
    draw_calls = metrics.get("draw_calls", 0)
    pipeline_binds = metrics.get("pipeline_binds", 0)

    if pipeline_binds == 0 or draw_calls == 0:
        findings.append(CPUFinding("info", "batching",
            "No draw calls or pipeline binds recorded."))
        return findings

    ratio = draw_calls / pipeline_binds

    if ratio < 1.5:
        findings.append(CPUFinding("critical", "batching",
            f"Very low draw/pipeline-bind ratio ({ratio:.2f}). "
            f"Nearly every draw triggers a pipeline rebind.",
            metric_value=ratio, threshold=MIN_DRAWS_PER_PIPELINE_BIND,
            recommendation="Sort draws by pipeline/material to batch contiguous draws "
                           "that share the same PSO."))
    elif ratio < MIN_DRAWS_PER_PIPELINE_BIND:
        findings.append(CPUFinding("warning", "batching",
            f"Low draw/pipeline-bind ratio ({ratio:.2f}), "
            f"threshold is {MIN_DRAWS_PER_PIPELINE_BIND:.1f}.",
            metric_value=ratio, threshold=MIN_DRAWS_PER_PIPELINE_BIND,
            recommendation="Group draws by material/pipeline to reduce state changes."))
    else:
        findings.append(CPUFinding("info", "batching",
            f"Draw/pipeline-bind ratio is acceptable ({ratio:.2f})."))

    return findings


def analyze_descriptor_rebinds(metrics: dict) -> list[CPUFinding]:
    """Analyze descriptor set rebind rate.

    Frequent descriptor set rebinds can be expensive. Ideally per-frame
    descriptors (set 0) are bound once, per-material (set 1) once per
    material, and only per-object (set 2) changes per draw.
    """
    findings = []
    draw_calls = metrics.get("draw_calls", 0)
    descriptor_binds = metrics.get("descriptor_set_binds", 0)

    if descriptor_binds == 0 or draw_calls == 0:
        return findings

    ratio = draw_calls / descriptor_binds

    if ratio < 1.2:
        findings.append(CPUFinding("warning", "descriptor",
            f"Descriptor sets rebound nearly every draw ({ratio:.2f} draws/bind). "
            f"Consider splitting into per-frame, per-material, per-object sets.",
            metric_value=ratio, threshold=MIN_DRAWS_PER_DESCRIPTOR_BIND,
            recommendation="Use set 0 for per-frame data (camera, lights), "
                           "set 1 for per-material textures, set 2 for per-object transforms."))
    elif ratio < MIN_DRAWS_PER_DESCRIPTOR_BIND:
        findings.append(CPUFinding("warning", "descriptor",
            f"Descriptor rebind rate is elevated ({ratio:.2f} draws/bind).",
            metric_value=ratio, threshold=MIN_DRAWS_PER_DESCRIPTOR_BIND,
            recommendation="Check if per-frame descriptors are being unnecessarily rebound."))

    # Per-set breakdown if available
    per_set = metrics.get("descriptor_binds_per_set", {})
    for set_idx, bind_count in per_set.items():
        set_num = int(set_idx)
        if set_num == 0 and bind_count > 1:
            findings.append(CPUFinding("warning", "descriptor",
                f"Per-frame descriptor set (set 0) bound {bind_count} times "
                f"(expected 1 per render pass).",
                metric_value=bind_count, threshold=1,
                recommendation="Bind set 0 once at the start of each render pass."))

    return findings


def analyze_buffer_padding(metrics: dict) -> list[CPUFinding]:
    """Analyze buffer alignment padding waste.

    std140 layout can waste significant space due to vec3->vec4 padding
    and array stride requirements. Reports the ratio of padding to payload.
    """
    findings = []
    buffers = metrics.get("uniform_buffers", [])

    if not buffers:
        return findings

    total_payload = 0
    total_allocated = 0

    for buf in buffers:
        payload = buf.get("payload_bytes", 0)
        allocated = buf.get("allocated_bytes", 0)
        total_payload += payload
        total_allocated += allocated

    if total_allocated == 0:
        return findings

    padding_ratio = 1.0 - (total_payload / total_allocated)

    if padding_ratio > MAX_BUFFER_PADDING_RATIO:
        wasted_kb = (total_allocated - total_payload) / 1024
        findings.append(CPUFinding("warning", "buffer",
            f"Buffer padding waste is {padding_ratio:.1%} ({wasted_kb:.1f} KB wasted). "
            f"Threshold: {MAX_BUFFER_PADDING_RATIO:.0%}.",
            metric_value=padding_ratio, threshold=MAX_BUFFER_PADDING_RATIO,
            recommendation="Use std430 layout where supported (SSBOs). "
                           "Pack vec3 fields as vec4 or restructure to avoid padding. "
                           "Consider using push constants for small, frequently-updated data."))

    return findings


def analyze_redundant_binds(metrics: dict) -> list[CPUFinding]:
    """Detect redundant pipeline binds.

    A redundant bind is when the same pipeline is bound consecutively
    without an intervening draw that uses a different pipeline.
    """
    findings = []
    pipeline_binds = metrics.get("pipeline_binds", 0)
    redundant_binds = metrics.get("redundant_pipeline_binds", 0)

    if pipeline_binds == 0:
        return findings

    redundant_ratio = redundant_binds / pipeline_binds

    if redundant_ratio > MAX_REDUNDANT_BIND_RATIO:
        findings.append(CPUFinding("warning", "pipeline",
            f"{redundant_binds} of {pipeline_binds} pipeline binds are redundant "
            f"({redundant_ratio:.1%}). Threshold: {MAX_REDUNDANT_BIND_RATIO:.0%}.",
            metric_value=redundant_ratio, threshold=MAX_REDUNDANT_BIND_RATIO,
            recommendation="Track the currently-bound pipeline and skip vkCmdBindPipeline "
                           "when the same PSO is already active."))
    elif redundant_binds > 0:
        findings.append(CPUFinding("info", "pipeline",
            f"{redundant_binds} redundant pipeline binds detected ({redundant_ratio:.1%})."))

    return findings


def analyze_metrics(metrics: dict) -> list[CPUFinding]:
    """Run all CPU-side analyses on a metrics dict.

    Args:
        metrics: Parsed JSON from ``rdc stats --json`` or equivalent.

    Returns:
        Combined list of findings from all analysis passes.
    """
    findings = []
    findings.extend(analyze_draw_batching(metrics))
    findings.extend(analyze_descriptor_rebinds(metrics))
    findings.extend(analyze_buffer_padding(metrics))
    findings.extend(analyze_redundant_binds(metrics))
    return findings


def format_findings_text(findings: list[CPUFinding]) -> str:
    """Format findings as human-readable text."""
    if not findings:
        return "No CPU-side issues found."

    lines = []
    for f in findings:
        icon = {"critical": "!!!", "warning": "!!", "info": "i"}.get(f.severity, "?")
        lines.append(f"  [{icon}] {f.category}: {f.message}")
        if f.recommendation:
            lines.append(f"        -> {f.recommendation}")
    return "\n".join(lines)


def format_findings_table(findings: list[CPUFinding]) -> str:
    """Format findings as a markdown table."""
    if not findings:
        return "No CPU-side issues found."

    lines = [
        "| Severity | Category | Message | Value | Threshold |",
        "|----------|----------|---------|-------|-----------|",
    ]
    for f in findings:
        val = f"{f.metric_value:.3f}" if f.metric_value else "-"
        thr = f"{f.threshold:.3f}" if f.threshold else "-"
        lines.append(f"| {f.severity} | {f.category} | {f.message} | {val} | {thr} |")
    return "\n".join(lines)


def format_findings_json(findings: list[CPUFinding]) -> str:
    """Format findings as JSON."""
    return json.dumps([f.to_dict() for f in findings], indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="CPU-side rendering analysis — draw call patterns and resource binding")
    parser.add_argument("metrics_file", type=Path,
                        help="Metrics JSON file (from rdc stats --json)")
    parser.add_argument("--format", choices=["text", "table", "json"], default="text",
                        help="Output format (default: text)")
    args = parser.parse_args()

    if not args.metrics_file.exists():
        print(f"Error: {args.metrics_file} not found")
        raise SystemExit(1)

    metrics = json.loads(args.metrics_file.read_text(encoding="utf-8"))

    findings = analyze_metrics(metrics)

    if args.format == "text":
        print("\n=== CPU-Side Rendering Analysis ===\n")
        print(format_findings_text(findings))
    elif args.format == "table":
        print(format_findings_table(findings))
    elif args.format == "json":
        print(format_findings_json(findings))

    # Exit with non-zero if any critical/warning findings
    severity_counts = {"critical": 0, "warning": 0, "info": 0}
    for f in findings:
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

    if severity_counts["critical"] > 0:
        raise SystemExit(2)
    elif severity_counts["warning"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
