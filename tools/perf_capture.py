#!/usr/bin/env python3
"""Performance capture orchestration for Lux shaders.

Compiles a shader, launches the C++ engine with RenderDoc injection,
captures a frame, extracts GPU metrics, and exports the framebuffer.

Usage:
    python tools/perf_capture.py --shader examples/pbr.lux
    python tools/perf_capture.py --shader examples/pbr.lux --baseline baseline.png
    python tools/perf_capture.py --shader examples/pbr.lux --scene sphere --flags "-O --perf"
    python tools/perf_capture.py --shader examples/pbr.lux --skip-capture
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LUXC = sys.executable  # Python running luxc as a module
_DEFAULT_ENGINE = _PROJECT_ROOT / "engine" / "build" / "Release" / "lux_engine.exe"
_RENDERDOC_CAPTURE = Path("d:/renderdoc/capture_frame.py")
_RENDERDOC_CLI = "renderdoccmd"


def _run(cmd: list[str], label: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a subprocess with unified error handling and progress output."""
    print(f"  [{label}] {' '.join(str(c) for c in cmd)}", flush=True)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            detail = stderr or stdout or "(no output)"
            print(f"  [{label}] FAILED (exit {result.returncode}): {detail}", file=sys.stderr, flush=True)
        return result
    except FileNotFoundError:
        print(f"  [{label}] ERROR: command not found: {cmd[0]}", file=sys.stderr, flush=True)
        raise
    except subprocess.TimeoutExpired:
        print(f"  [{label}] ERROR: timed out after {timeout}s", file=sys.stderr, flush=True)
        raise


# ---------------------------------------------------------------------------
# Stage 1: Compile
# ---------------------------------------------------------------------------

def compile_shader(
    shader_path: Path,
    output_dir: Path,
    flags: list[str] | None = None,
) -> dict:
    """Compile a .lux shader and return reflection JSON.

    Invokes ``python -m luxc <shader> -o <output_dir>`` plus any extra flags.
    Expects the compiler to produce .spv binaries and a .lux.json reflection
    sidecar in the output directory.

    Args:
        shader_path: Path to .lux file.
        output_dir: Where to write .spv and .json outputs.
        flags: Additional luxc flags (e.g., ['-O', '--perf']).

    Returns:
        dict with keys:
            spv_files  -- list[str] of generated .spv file paths.
            reflection -- dict parsed from the .lux.json sidecar (or {} on failure).
            success    -- bool indicating compilation success.
            error      -- str or None with error message on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "luxc", str(shader_path), "-o", str(output_dir)]
    if flags:
        cmd.extend(flags)

    try:
        result = _run(cmd, "compile")
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "spv_files": [],
            "reflection": {},
            "success": False,
            "error": str(exc),
        }

    if result.returncode != 0:
        return {
            "spv_files": [],
            "reflection": {},
            "success": False,
            "error": result.stderr.strip() or result.stdout.strip() or "unknown error",
        }

    # Collect outputs
    spv_files = sorted(str(p) for p in output_dir.glob("*.spv"))

    reflection: dict = {}
    json_files = list(output_dir.glob("*.lux.json"))
    if json_files:
        try:
            reflection = json.loads(json_files[0].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [compile] warning: could not read reflection JSON: {exc}", file=sys.stderr)

    print(f"  [compile] OK -- {len(spv_files)} .spv file(s)", flush=True)
    return {
        "spv_files": spv_files,
        "reflection": reflection,
        "success": True,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Stage 2: Capture
# ---------------------------------------------------------------------------

def capture_frame(
    engine_path: Path,
    spv_dir: Path,
    scene: str,
    output_dir: Path,
) -> dict:
    """Launch engine with RenderDoc and capture a frame.

    Uses ``d:\\renderdoc\\capture_frame.py`` for RenderDoc injection.  The engine
    is started with ``--scene``, ``--pipeline``, and ``--capture`` flags, giving it
    one frame to render before the capture layer triggers.

    Args:
        engine_path: Path to the C++ engine executable.
        spv_dir: Directory containing compiled .spv + .lux.json files.
        scene: Scene name (e.g., "sphere", "fullscreen", "triangle").
        output_dir: Where to save the .rdc capture file.

    Returns:
        dict with keys:
            rdc_path -- str path to saved .rdc file (or None on failure).
            success  -- bool.
            error    -- str or None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rdc_path = output_dir / f"capture_{timestamp}.rdc"

    if not engine_path.exists():
        return {
            "rdc_path": None,
            "success": False,
            "error": f"Engine not found: {engine_path}",
        }

    # Build the command.  If the RenderDoc capture helper script exists we
    # wrap the engine invocation with it; otherwise we invoke the engine
    # directly with a --renderdoc flag (engine must self-inject).
    engine_cmd = [
        str(engine_path),
        "--scene", scene,
        "--pipeline", str(spv_dir),
        "--capture", str(rdc_path),
        "--frames", "1",
        "--headless",
    ]

    if _RENDERDOC_CAPTURE.exists():
        cmd = [
            sys.executable,
            str(_RENDERDOC_CAPTURE),
            "--output", str(rdc_path),
            "--",
        ] + engine_cmd
    else:
        cmd = engine_cmd + ["--renderdoc"]

    try:
        result = _run(cmd, "capture", timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"rdc_path": None, "success": False, "error": str(exc)}

    if result.returncode != 0:
        return {
            "rdc_path": None,
            "success": False,
            "error": result.stderr.strip() or result.stdout.strip() or "capture failed",
        }

    if not rdc_path.exists():
        return {
            "rdc_path": None,
            "success": False,
            "error": "Capture completed but .rdc file not found",
        }

    print(f"  [capture] OK -- {rdc_path} ({rdc_path.stat().st_size / 1024:.0f} KB)", flush=True)
    return {
        "rdc_path": str(rdc_path),
        "success": True,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Stage 3: Extract GPU metrics
# ---------------------------------------------------------------------------

def extract_metrics(rdc_path: Path) -> dict:
    """Extract GPU metrics from a RenderDoc capture.

    Calls ``renderdoccmd`` with ``stats``, ``draws``, and ``passes`` sub-commands
    to retrieve structured JSON data about the captured frame.

    Args:
        rdc_path: Path to a .rdc capture file.

    Returns:
        dict with GPU metrics including:
            draw_calls      -- int total draw call count.
            pipeline_binds  -- int number of pipeline state changes.
            total_vertices  -- int vertices processed.
            total_primitives -- int primitives generated.
            passes          -- list of per-pass summaries.
            raw             -- dict with the full ``stats --json`` output.
            success         -- bool.
            error           -- str or None.
    """
    if not Path(rdc_path).exists():
        return {"success": False, "error": f"Capture not found: {rdc_path}"}

    metrics: dict = {
        "draw_calls": 0,
        "pipeline_binds": 0,
        "total_vertices": 0,
        "total_primitives": 0,
        "passes": [],
        "raw": {},
        "success": False,
        "error": None,
    }

    # --- stats ---
    try:
        result = _run(
            [_RENDERDOC_CLI, "stats", str(rdc_path), "--json"],
            "metrics/stats",
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            stats = json.loads(result.stdout)
            metrics["raw"] = stats
            metrics["draw_calls"] = stats.get("draw_calls", 0)
            metrics["total_vertices"] = stats.get("total_vertices", 0)
            metrics["total_primitives"] = stats.get("total_primitives", 0)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        metrics["error"] = f"stats extraction failed: {exc}"
        return metrics

    # --- draws ---
    try:
        result = _run(
            [_RENDERDOC_CLI, "draws", str(rdc_path), "--json"],
            "metrics/draws",
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            draws = json.loads(result.stdout)
            metrics["pipeline_binds"] = sum(
                1 for d in draws if d.get("type") == "pipeline_bind"
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        metrics["error"] = f"draws extraction failed: {exc}"
        return metrics

    # --- passes ---
    try:
        result = _run(
            [_RENDERDOC_CLI, "passes", str(rdc_path), "--json"],
            "metrics/passes",
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            metrics["passes"] = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        metrics["error"] = f"passes extraction failed: {exc}"
        return metrics

    metrics["success"] = True
    print(
        f"  [metrics] OK -- {metrics['draw_calls']} draws, "
        f"{metrics['pipeline_binds']} binds, "
        f"{metrics['total_vertices']} verts",
        flush=True,
    )
    return metrics


# ---------------------------------------------------------------------------
# Stage 4: Export framebuffer
# ---------------------------------------------------------------------------

def export_framebuffer(
    rdc_path: Path,
    output_png: Path,
    event_id: int | None = None,
) -> bool:
    """Export framebuffer from RenderDoc capture as PNG.

    Calls ``renderdoccmd rt <event_id> -o <output_png>`` on the given capture.
    If *event_id* is None the last draw event is used (``renderdoccmd`` default).

    Args:
        rdc_path: Path to a .rdc capture file.
        output_png: Destination path for the exported PNG.

    Returns:
        True on success, False otherwise.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)

    cmd = [_RENDERDOC_CLI, "rt"]
    if event_id is not None:
        cmd.append(str(event_id))
    cmd.extend(["-o", str(output_png), str(rdc_path)])

    try:
        result = _run(cmd, "export", timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

    if result.returncode != 0:
        return False

    if not output_png.exists():
        print("  [export] WARNING: command succeeded but PNG not found", file=sys.stderr, flush=True)
        return False

    print(f"  [export] OK -- {output_png} ({output_png.stat().st_size / 1024:.0f} KB)", flush=True)
    return True


# ---------------------------------------------------------------------------
# Stage 5: Report
# ---------------------------------------------------------------------------

def generate_report(
    shader_path: Path,
    reflection: dict,
    metrics: dict | None,
    quality: dict | None,
    output_dir: Path,
) -> Path:
    """Generate a Markdown performance report.

    Combines compilation reflection data, GPU metrics, and optional image
    quality results into a single ``.md`` file.

    Args:
        shader_path: Path to the original .lux source.
        reflection: Parsed reflection JSON from compilation.
        metrics: GPU metrics dict from :func:`extract_metrics` (or None).
        quality: Image quality dict from ``quality_metrics.compare_images`` (or None).
        output_dir: Directory to write the report file.

    Returns:
        Path to the generated .md report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"{shader_path.stem}_perf_{timestamp}.md"
    report_path = output_dir / report_name

    lines: list[str] = []
    lines.append(f"# Performance Report: {shader_path.name}")
    lines.append(f"")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"")

    # --- Compilation info ---
    lines.append("## Compilation")
    lines.append("")
    stages = reflection.get("stages", [])
    if stages:
        for stage in stages:
            name = stage.get("stage", "unknown")
            entry = stage.get("entry_point", "main")
            lines.append(f"- **{name}** (entry: `{entry}`)")
    else:
        lines.append("- No stage information available.")
    lines.append("")

    descriptors = reflection.get("descriptor_sets", [])
    if descriptors:
        lines.append("### Descriptor Sets")
        lines.append("")
        for ds in descriptors:
            set_num = ds.get("set", "?")
            bindings = ds.get("bindings", [])
            lines.append(f"- Set {set_num}: {len(bindings)} binding(s)")
            for b in bindings:
                btype = b.get("type", "unknown")
                bname = b.get("name", "?")
                lines.append(f"  - binding {b.get('binding', '?')}: `{bname}` ({btype})")
        lines.append("")

    push_constants = reflection.get("push_constants", {})
    if push_constants:
        size = push_constants.get("size", 0)
        lines.append(f"### Push Constants: {size} bytes")
        lines.append("")

    # --- GPU metrics ---
    if metrics and metrics.get("success"):
        lines.append("## GPU Metrics")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Draw calls | {metrics.get('draw_calls', 'N/A')} |")
        lines.append(f"| Pipeline binds | {metrics.get('pipeline_binds', 'N/A')} |")
        lines.append(f"| Total vertices | {metrics.get('total_vertices', 'N/A')} |")
        lines.append(f"| Total primitives | {metrics.get('total_primitives', 'N/A')} |")
        lines.append("")

        passes = metrics.get("passes", [])
        if passes:
            lines.append("### Render Passes")
            lines.append("")
            for i, p in enumerate(passes):
                name = p.get("name", f"Pass {i}")
                draws = p.get("draw_calls", "?")
                lines.append(f"- **{name}**: {draws} draw(s)")
            lines.append("")
    elif metrics:
        lines.append("## GPU Metrics")
        lines.append("")
        lines.append(f"Extraction failed: {metrics.get('error', 'unknown error')}")
        lines.append("")

    # --- Quality comparison ---
    if quality:
        lines.append("## Image Quality")
        lines.append("")
        psnr = quality.get("psnr", 0)
        ssim = quality.get("ssim", 0)
        psnr_str = f"{psnr:.2f}" if psnr != float("inf") else "inf (identical)"
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| PSNR | {psnr_str} dB |")
        lines.append(f"| SSIM | {ssim:.6f} |")
        lines.append(f"| Max delta | {quality.get('max_delta', 'N/A')} |")
        lines.append("")

        if "passed" in quality:
            status = "PASS" if quality["passed"] else "**FAIL**"
            lines.append(f"Quality gate: {status}")
            lines.append("")

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  [report] OK -- {report_path}", flush=True)
    return report_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lux shader performance capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --shader examples/pbr.lux\n"
            "  %(prog)s --shader examples/pbr.lux --baseline baseline.png\n"
            "  %(prog)s --shader examples/pbr.lux --skip-capture --flags '-O'\n"
        ),
    )
    parser.add_argument(
        "--shader", type=Path, required=True,
        help="Path to .lux file",
    )
    parser.add_argument(
        "--scene", type=str, default="sphere",
        help="Scene name (default: sphere)",
    )
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="Baseline PNG for quality comparison",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("perf_results"),
        help="Output directory (default: perf_results)",
    )
    parser.add_argument(
        "--flags", type=str, default="",
        help="Additional luxc flags, space-separated (e.g., '-O --debug')",
    )
    parser.add_argument(
        "--engine", type=Path, default=_DEFAULT_ENGINE,
        help=f"Path to C++ engine executable (default: {_DEFAULT_ENGINE})",
    )
    parser.add_argument(
        "--skip-capture", action="store_true",
        help="Skip RenderDoc capture (compile + analysis only)",
    )
    parser.add_argument(
        "--skip-metrics", action="store_true",
        help="Skip GPU metric extraction (capture + export only)",
    )
    parser.add_argument(
        "--event-id", type=int, default=None,
        help="RenderDoc event ID for framebuffer export (default: last draw)",
    )
    parser.add_argument(
        "--psnr-threshold", type=float, default=40.0,
        help="Minimum acceptable PSNR in dB (default: 40.0)",
    )
    parser.add_argument(
        "--ssim-threshold", type=float, default=0.99,
        help="Minimum acceptable SSIM (default: 0.99)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output final results as JSON to stdout",
    )

    args = parser.parse_args()

    if not args.shader.exists():
        print(f"Error: shader not found: {args.shader}", file=sys.stderr)
        sys.exit(1)

    output_dir: Path = args.output.resolve()
    spv_dir = output_dir / "spv"
    captures_dir = output_dir / "captures"

    extra_flags = args.flags.split() if args.flags.strip() else []

    all_results: dict = {
        "shader": str(args.shader),
        "scene": args.scene,
        "timestamp": datetime.now().isoformat(),
        "stages": {},
    }

    # ------------------------------------------------------------------
    # Stage 1: Compile
    # ------------------------------------------------------------------
    print(f"\n=== Stage 1/5: Compile ===", flush=True)
    compile_result = compile_shader(args.shader, spv_dir, flags=extra_flags)
    all_results["stages"]["compile"] = compile_result

    if not compile_result["success"]:
        print(f"\nCompilation failed: {compile_result['error']}", file=sys.stderr)
        if args.json:
            print(json.dumps(all_results, indent=2, default=str))
        sys.exit(1)

    reflection = compile_result["reflection"]

    # ------------------------------------------------------------------
    # Stage 2: Capture
    # ------------------------------------------------------------------
    rdc_path: Path | None = None

    if args.skip_capture:
        print(f"\n=== Stage 2/5: Capture (SKIPPED) ===", flush=True)
        all_results["stages"]["capture"] = {"skipped": True}
    else:
        print(f"\n=== Stage 2/5: Capture ===", flush=True)
        capture_result = capture_frame(args.engine, spv_dir, args.scene, captures_dir)
        all_results["stages"]["capture"] = capture_result

        if not capture_result["success"]:
            print(f"\nCapture failed: {capture_result['error']}", file=sys.stderr)
            print("Continuing with compile-only report...", file=sys.stderr)
        else:
            rdc_path = Path(capture_result["rdc_path"])

    # ------------------------------------------------------------------
    # Stage 3: Extract metrics
    # ------------------------------------------------------------------
    metrics: dict | None = None

    if rdc_path and not args.skip_metrics:
        print(f"\n=== Stage 3/5: Extract Metrics ===", flush=True)
        metrics = extract_metrics(rdc_path)
        all_results["stages"]["metrics"] = metrics
    else:
        reason = "no capture" if not rdc_path else "skipped"
        print(f"\n=== Stage 3/5: Extract Metrics ({reason.upper()}) ===", flush=True)
        all_results["stages"]["metrics"] = {"skipped": True, "reason": reason}

    # ------------------------------------------------------------------
    # Stage 4: Export framebuffer + quality comparison
    # ------------------------------------------------------------------
    quality: dict | None = None
    framebuffer_png = output_dir / "framebuffer.png"

    if rdc_path:
        print(f"\n=== Stage 4/5: Export Framebuffer ===", flush=True)
        exported = export_framebuffer(rdc_path, framebuffer_png, event_id=args.event_id)
        all_results["stages"]["export"] = {"success": exported, "path": str(framebuffer_png)}

        if exported and args.baseline:
            print(f"\n  Comparing against baseline: {args.baseline}", flush=True)
            try:
                from tools.quality_metrics import compare_images, quality_check

                diff_path = output_dir / "diff_heatmap.png"
                quality = compare_images(
                    str(args.baseline),
                    str(framebuffer_png),
                    diff_output=str(diff_path),
                )
                check = quality_check(
                    quality["psnr"],
                    quality["ssim"],
                    psnr_threshold=args.psnr_threshold,
                    ssim_threshold=args.ssim_threshold,
                )
                quality.update(check)
                all_results["stages"]["quality"] = quality
                print(f"  Quality: {check['message']}", flush=True)
            except ImportError:
                print("  WARNING: quality_metrics not available, skipping comparison", file=sys.stderr)
            except Exception as exc:
                print(f"  WARNING: quality comparison failed: {exc}", file=sys.stderr)
    else:
        print(f"\n=== Stage 4/5: Export Framebuffer (SKIPPED) ===", flush=True)
        all_results["stages"]["export"] = {"skipped": True}

    # ------------------------------------------------------------------
    # Stage 5: Generate report
    # ------------------------------------------------------------------
    print(f"\n=== Stage 5/5: Generate Report ===", flush=True)
    report_path = generate_report(
        shader_path=args.shader,
        reflection=reflection,
        metrics=metrics,
        quality=quality,
        output_dir=output_dir,
    )
    all_results["report"] = str(report_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 50}", flush=True)
    print(f"Performance capture complete.", flush=True)
    print(f"  Shader : {args.shader}", flush=True)
    print(f"  Output : {output_dir}", flush=True)
    print(f"  Report : {report_path}", flush=True)
    if quality and "passed" in quality:
        status = "PASS" if quality["passed"] else "FAIL"
        print(f"  Quality: {status}", flush=True)
    print(f"{'=' * 50}\n", flush=True)

    # Structured output for CI / scripting
    if args.json:
        print(json.dumps(all_results, indent=2, default=str))

    # Exit with non-zero if quality gate failed
    if quality and not quality.get("passed", True):
        sys.exit(1)


if __name__ == "__main__":
    main()
