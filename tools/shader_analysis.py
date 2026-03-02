"""Shader bottleneck analyzer.

Detects performance anti-patterns in compiled shaders and RenderDoc captures.
"""

from __future__ import annotations
import json
from pathlib import Path


# Thresholds for flagging
FRAG_TEX_THRESHOLD = 3
FRAG_BRANCH_THRESHOLD = 5
FRAG_ALU_BUDGET_60FPS = 200
FRAG_ALU_BUDGET_VR = 100
VERT_ALU_BUDGET = 50
HIGH_FDIV_THRESHOLD = 4
LARGE_TEXTURE_BYTES = 4 * 1024 * 1024  # 4MB


class Finding:
    """A performance finding/warning."""
    def __init__(self, severity: str, category: str, message: str,
                 stage: str = "", metric_value: float = 0, threshold: float = 0):
        self.severity = severity  # "warning", "info", "critical"
        self.category = category  # "texture", "branching", "alu", "memory", "cpu"
        self.message = message
        self.stage = stage
        self.metric_value = metric_value
        self.threshold = threshold

    def __repr__(self):
        return f"[{self.severity}] {self.category}: {self.message}"


def analyze_reflection(reflection: dict, target: str = "desktop") -> list[Finding]:
    """Analyze a reflection JSON dict for performance issues.

    Args:
        reflection: Parsed reflection JSON
        target: "desktop" or "mobile" or "vr"

    Returns:
        List of Finding objects
    """
    findings = []
    stage = reflection.get("stage", "unknown")
    hints = reflection.get("performance_hints", {})

    if not hints:
        findings.append(Finding("info", "metadata",
            "No performance_hints in reflection — compile with --analyze", stage=stage))
        return findings

    alu_ops = hints.get("alu_ops", 0)
    tex_samples = hints.get("texture_samples", 0)
    branches = hints.get("branches", 0)
    instr_count = hints.get("instruction_count", 0)
    vgpr = hints.get("vgpr_estimate", "low")

    # Fragment shader checks
    if stage == "fragment":
        alu_budget = FRAG_ALU_BUDGET_VR if target == "vr" else FRAG_ALU_BUDGET_60FPS
        if alu_ops > alu_budget:
            findings.append(Finding("warning", "alu",
                f"Fragment ALU ops ({alu_ops}) exceeds {target} budget ({alu_budget})",
                stage=stage, metric_value=alu_ops, threshold=alu_budget))

        tex_limit = FRAG_TEX_THRESHOLD if target == "mobile" else FRAG_TEX_THRESHOLD * 2
        if tex_samples > tex_limit:
            findings.append(Finding("warning", "texture",
                f"Fragment texture samples ({tex_samples}) exceeds threshold ({tex_limit})",
                stage=stage, metric_value=tex_samples, threshold=tex_limit))

        if branches > FRAG_BRANCH_THRESHOLD:
            findings.append(Finding("warning", "branching",
                f"Fragment branches ({branches}) may cause divergent execution (threshold: {FRAG_BRANCH_THRESHOLD})",
                stage=stage, metric_value=branches, threshold=FRAG_BRANCH_THRESHOLD))

    # Vertex shader checks
    if stage == "vertex":
        if alu_ops > VERT_ALU_BUDGET:
            findings.append(Finding("warning", "alu",
                f"Vertex ALU ops ({alu_ops}) exceeds budget ({VERT_ALU_BUDGET})",
                stage=stage, metric_value=alu_ops, threshold=VERT_ALU_BUDGET))

    # VGPR pressure
    if vgpr == "high":
        findings.append(Finding("warning", "register_pressure",
            f"High VGPR pressure — may reduce occupancy",
            stage=stage))

    return findings


def analyze_capture_metrics(metrics: dict) -> list[Finding]:
    """Analyze RenderDoc capture metrics for CPU-side issues."""
    findings = []

    draw_count = metrics.get("draw_calls", 0)
    pipeline_binds = metrics.get("pipeline_binds", 0)

    if pipeline_binds > 0 and draw_count > 0:
        ratio = draw_count / pipeline_binds
        if ratio < 2.0:
            findings.append(Finding("warning", "cpu",
                f"Low draw/pipeline-bind ratio ({ratio:.1f}) — consider batching draws",
                metric_value=ratio, threshold=2.0))

    return findings


def format_findings(findings: list[Finding]) -> str:
    """Format findings as a readable report."""
    if not findings:
        return "No performance issues found."

    lines = []
    for f in findings:
        icon = {"critical": "!!!", "warning": "!!", "info": "i"}.get(f.severity, "?")
        stage_prefix = f"[{f.stage}] " if f.stage else ""
        lines.append(f"  [{icon}] {stage_prefix}{f.category}: {f.message}")

    return "\n".join(lines)


def analyze_shader_file(json_path: Path, target: str = "desktop") -> list[Finding]:
    """Analyze a .json reflection file."""
    reflection = json.loads(json_path.read_text(encoding="utf-8"))
    return analyze_reflection(reflection, target)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Shader bottleneck analyzer")
    parser.add_argument("json_files", nargs="+", type=Path, help="Reflection JSON files")
    parser.add_argument("--target", choices=["desktop", "mobile", "vr"], default="desktop")

    args = parser.parse_args()

    for path in args.json_files:
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        print(f"\n=== {path.name} ===")
        findings = analyze_shader_file(path, args.target)
        print(format_findings(findings))


if __name__ == "__main__":
    main()
