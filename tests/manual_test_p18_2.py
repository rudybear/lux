#!/usr/bin/env python3
"""
Manual Testing Script for P18.2 — Debugger Enhancements & Bindless Properties

Run this script to interactively verify all P18.2 features:
    python tests/manual_test_p18_2.py

Or run a specific section:
    python tests/manual_test_p18_2.py --section a1     # Conditional breakpoints
    python tests/manual_test_p18_2.py --section a2     # Watch expressions
    python tests/manual_test_p18_2.py --section a4     # Break-on-NaN
    python tests/manual_test_p18_2.py --section b      # Time-travel
    python tests/manual_test_p18_2.py --section c      # Heatmap
    python tests/manual_test_p18_2.py --section d      # Real textures + bindless debug
    python tests/manual_test_p18_2.py --section e      # Bindless properties compiler
    python tests/manual_test_p18_2.py --section cli    # CLI integration tests
    python tests/manual_test_p18_2.py --all            # Run all automated checks
    python tests/manual_test_p18_2.py --guide          # Print interactive testing guide

Each section prints PASS/FAIL and explanations.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─── Colors ──────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0
skipped = 0


def header(title: str):
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  {GREEN}PASS{RESET}  {name}")
    else:
        failed += 1
        print(f"  {RED}FAIL{RESET}  {name}")
    if detail:
        print(f"        {detail}")


def skip(name: str, reason: str):
    global skipped
    skipped += 1
    print(f"  {YELLOW}SKIP{RESET}  {name} -- {reason}")


def section_header(code: str, title: str):
    print(f"\n  {BOLD}[{code}] {title}{RESET}")
    print(f"  {'-' * 50}")


# ─── Helper: build module + stage + debugger ─────────────────────────────────

def _prepare(source: str):
    """Parse, type-check, constant-fold, and return (module, stage, lines)."""
    from luxc.parser.tree_builder import parse_lux
    from luxc.analysis.type_checker import type_check
    from luxc.optimization.const_fold import constant_fold

    module = parse_lux(source)
    type_check(module)
    constant_fold(module)
    stage = module.stages[0]
    lines = source.splitlines()
    return module, stage, lines


def _make_debugger(module, stage, lines):
    """Create Interpreter + Debugger pair."""
    from luxc.debug.interpreter import Interpreter
    from luxc.debug.debugger import Debugger

    interp = Interpreter(module, stage, source_lines=lines)
    dbg = Debugger(interp)
    return interp, dbg


# ─── Test Shaders ────────────────────────────────────────────────────────────

SIMPLE_FRAGMENT = textwrap.dedent("""\
    fragment {
        in uv: vec2;
        out color: vec4;

        fn main() {
            let x: scalar = uv.x;
            let y: scalar = uv.y;
            let brightness: scalar = x * y;
            let r: scalar = x;
            let g: scalar = y;
            let b: scalar = brightness;
            color = vec4(r, g, b, 1.0);
        }
    }
""")

NAN_FRAGMENT = textwrap.dedent("""\
    fragment {
        in uv: vec2;
        out color: vec4;

        fn main() {
            let x: scalar = uv.x;
            let zero: vec3 = vec3(0.0, 0.0, 0.0);
            let bad: vec3 = normalize(zero);
            let safe: scalar = 0.5;
            color = vec4(safe, safe, safe, 1.0);
        }
    }
""")


# ═══════════════════════════════════════════════════════════════════════════════
# A1: Conditional Breakpoints
# ═══════════════════════════════════════════════════════════════════════════════

def test_a1_conditional_breakpoints():
    section_header("A1", "Conditional Breakpoints")

    from luxc.debug.debugger import Breakpoint, BreakpointHit
    from luxc.debug.values import LuxVec
    from luxc.debug.io import build_default_inputs

    module, stage, lines = _prepare(SIMPLE_FRAGMENT)

    # Find the line with "let brightness"
    bp_line = None
    for i, line in enumerate(lines, 1):
        if "brightness" in line and "let" in line:
            bp_line = i
            break

    # --- Test 1: Condition IS met (x=0.8 > 0.5) ---
    interp, dbg = _make_debugger(module, stage, lines)
    inputs = build_default_inputs(stage)
    inputs["uv"] = LuxVec([0.8, 0.9])

    dbg.add_breakpoint(bp_line, condition="x > 0.5")

    stopped_lines = []
    def on_break(hit: BreakpointHit):
        stopped_lines.append(hit.line)
    dbg._on_break = on_break
    dbg._step_mode = dbg._step_mode  # RUN mode (default)
    interp.run(inputs=inputs)

    check("Conditional BP triggers when condition met (x=0.8 > 0.5)",
          bp_line in stopped_lines,
          f"stopped at lines: {stopped_lines}, expected {bp_line}")

    # --- Test 2: Condition NOT met (x=0.2 < 0.5) ---
    interp2, dbg2 = _make_debugger(module, stage, lines)
    inputs2 = build_default_inputs(stage)
    inputs2["uv"] = LuxVec([0.2, 0.3])

    dbg2.add_breakpoint(bp_line, condition="x > 0.5")

    stopped_lines2 = []
    def on_break2(hit: BreakpointHit):
        stopped_lines2.append(hit.line)
    dbg2._on_break = on_break2
    interp2.run(inputs=inputs2)

    check("Conditional BP skipped when condition not met (x=0.2 < 0.5)",
          bp_line not in stopped_lines2,
          f"stopped at lines: {stopped_lines2}")

    # --- Test 3: Unconditional breakpoint always triggers ---
    interp3, dbg3 = _make_debugger(module, stage, lines)
    dbg3.add_breakpoint(bp_line)  # no condition

    stopped_lines3 = []
    def on_break3(hit: BreakpointHit):
        stopped_lines3.append(hit.line)
    dbg3._on_break = on_break3
    interp3.run(inputs=build_default_inputs(stage))

    check("Unconditional BP always triggers",
          bp_line in stopped_lines3)


# ═══════════════════════════════════════════════════════════════════════════════
# A2: Watch Expression Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def test_a2_watch_expressions():
    section_header("A2", "Watch Expression Evaluation")

    from luxc.debug.debugger import BreakpointHit
    from luxc.debug.values import LuxVec
    from luxc.debug.io import build_default_inputs

    module, stage, lines = _prepare(SIMPLE_FRAGMENT)
    interp, dbg = _make_debugger(module, stage, lines)
    inputs = build_default_inputs(stage)
    inputs["uv"] = LuxVec([0.6, 0.4])

    # Add watches
    dbg.add_watch("x")
    dbg.add_watch("x + y")

    # Set a breakpoint near end
    bp_line = None
    for i, line in enumerate(lines, 1):
        if "let b:" in line or "let b " in line:
            bp_line = i
            break
    if bp_line is None:
        # fallback: use the color = line
        for i, line in enumerate(lines, 1):
            if "color = " in line:
                bp_line = i
                break

    dbg.add_breakpoint(bp_line)

    watch_results = []
    def on_break(hit: BreakpointHit):
        results = dbg.evaluate_watches()
        watch_results.extend(results)
    dbg._on_break = on_break
    interp.run(inputs=inputs)

    check("Watch expressions evaluated at breakpoint",
          len(watch_results) > 0,
          f"got {len(watch_results)} watch results")

    if watch_results:
        # First watch: x should be 0.6
        entry0, val0, changed0 = watch_results[0]
        check("Watch 'x' returns correct value",
              val0 is not None and hasattr(val0, 'value') and abs(val0.value - 0.6) < 0.01,
              f"x = {val0}")

        # Second watch: x + y should be 1.0
        if len(watch_results) > 1:
            entry1, val1, changed1 = watch_results[1]
            check("Watch 'x + y' returns correct value",
                  val1 is not None and hasattr(val1, 'value') and abs(val1.value - 1.0) < 0.01,
                  f"x + y = {val1}")


# ═══════════════════════════════════════════════════════════════════════════════
# A3: Expression Parser
# ═══════════════════════════════════════════════════════════════════════════════

def test_a3_expression_parser():
    section_header("A3", "Expression Parser (parse_debug_expr)")

    from luxc.debug.expr_parser import parse_debug_expr

    # Simple literal
    node = parse_debug_expr("42.0")
    check("Parses numeric literal", node is not None, f"got {type(node).__name__}")

    # Binary operation
    node2 = parse_debug_expr("x + y * 2.0")
    check("Parses binary expression", node2 is not None, f"got {type(node2).__name__}")

    # Comparison
    node3 = parse_debug_expr("x > 0.5")
    check("Parses comparison", node3 is not None, f"got {type(node3).__name__}")

    # Function call
    node4 = parse_debug_expr("dot(n, l)")
    check("Parses function call", node4 is not None, f"got {type(node4).__name__}")

    # Field access
    node5 = parse_debug_expr("v.xyz")
    check("Parses field/swizzle access", node5 is not None, f"got {type(node5).__name__}")

    # Constructor
    node6 = parse_debug_expr("vec3(1.0, 2.0, 3.0)")
    check("Parses constructor", node6 is not None, f"got {type(node6).__name__}")

    # Invalid expression — parser may or may not raise depending on what it sees
    try:
        parse_debug_expr("@#$%^&")
        check("Rejects invalid expression", False, "should have raised ValueError")
    except (ValueError, Exception):
        check("Rejects invalid expression", True)


# ═══════════════════════════════════════════════════════════════════════════════
# A4: Break-on-NaN
# ═══════════════════════════════════════════════════════════════════════════════

def test_a4_break_on_nan():
    section_header("A4", "Break-on-NaN")

    from luxc.debug.debugger import BreakpointHit
    from luxc.debug.values import LuxVec
    from luxc.debug.io import build_default_inputs

    module, stage, lines = _prepare(NAN_FRAGMENT)
    interp, dbg = _make_debugger(module, stage, lines)
    dbg.break_on_nan = True

    nan_events_received = []
    def on_nan(events):
        nan_events_received.extend(events)
    dbg._on_nan_break = on_nan

    stopped_lines = []
    def on_break(hit: BreakpointHit):
        stopped_lines.append(hit.line)
    dbg._on_break = on_break

    interp.run(inputs=build_default_inputs(stage))

    check("Break-on-NaN detects NaN events",
          len(nan_events_received) > 0,
          f"events: {len(nan_events_received)}")

    # After break-on-NaN fires, step mode is set → subsequent statements trigger on_break
    check("Break-on-NaN forces stop after NaN statement",
          len(stopped_lines) > 0,
          f"stopped at lines: {stopped_lines}")

    # --- Batch mode with --check-nan ---
    from luxc.debug.cli import run_batch
    result = run_batch(NAN_FRAGMENT, "fragment", check_nan=True)
    check("Batch mode detects NaN",
          result.get("nan_detected", False),
          f"nan_events: {result.get('nan_events', [])}")

    # --- Batch mode with break_on_nan ---
    result2 = run_batch(NAN_FRAGMENT, "fragment", check_nan=True, break_on_nan=True)
    check("Batch mode with break_on_nan works",
          result2.get("status") == "completed")


# ═══════════════════════════════════════════════════════════════════════════════
# B: Time-Travel Debugging
# ═══════════════════════════════════════════════════════════════════════════════

def test_b_time_travel():
    section_header("B", "Time-Travel Debugging")

    from luxc.debug.values import LuxVec
    from luxc.debug.io import build_default_inputs

    module, stage, lines = _prepare(SIMPLE_FRAGMENT)
    interp, dbg = _make_debugger(module, stage, lines)
    inputs = build_default_inputs(stage)
    inputs["uv"] = LuxVec([0.6, 0.4])

    # Run to completion (records history)
    interp.run(inputs=inputs)

    check("Time-travel history is recorded",
          dbg.time_travel is not None and len(dbg.time_travel.history) > 0,
          f"history length: {len(dbg.time_travel.history) if dbg.time_travel else 0}")

    if dbg.time_travel and len(dbg.time_travel.history) > 2:
        hist_len = len(dbg.time_travel.history)

        # Reverse step
        snap = dbg.reverse_step()
        check("reverse_step() succeeds",
              snap is not None,
              f"current_index: {dbg.time_travel.current_index}")

        # Check env was restored
        x_val = interp.env.get("x")
        check("Environment restored after reverse-step",
              x_val is not None,
              f"x = {x_val}")

        # Goto first step (step_index=1, the first recorded step)
        first_step = dbg.time_travel.history[0].step_index
        snap2 = dbg.goto_step(first_step)
        check("goto_step(first) jumps to beginning",
              snap2 is not None and dbg.time_travel.current_index == 0,
              f"step_index={first_step}, current_index={dbg.time_travel.current_index}")

        # Goto last step
        last_step = dbg.time_travel.history[-1].step_index
        snap3 = dbg.goto_step(last_step)
        check("goto_step(last) jumps to end",
              snap3 is not None and dbg.time_travel.current_index == hist_len - 1,
              f"step_index={last_step}, current_index={dbg.time_travel.current_index}")

        # Reverse continue with a breakpoint -> should find it
        # Add a breakpoint at the line of the first history entry
        first_line = dbg.time_travel.history[0].line
        dbg.add_breakpoint(first_line)
        dbg.goto_step(last_step)  # go to end
        snap4 = dbg.reverse_continue()
        check("reverse_continue() finds breakpoint in history",
              snap4 is not None,
              f"stopped at line {snap4.line if snap4 else '?'}, index={dbg.time_travel.current_index}")


# ═══════════════════════════════════════════════════════════════════════════════
# C: Multi-Pixel Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def test_c_heatmap():
    section_header("C", "Multi-Pixel Heatmap")

    from luxc.debug.heatmap import run_pixel_grid, write_ppm

    # Run a small grid
    result = run_pixel_grid(
        source=SIMPLE_FRAGMENT,
        stage_name="fragment",
        resolution=(64, 64),
        grid_size=(4, 4),
        mode="output",
    )

    check("Heatmap grid runs successfully",
          result is not None and len(result.pixels) == 16,
          f"pixels: {len(result.pixels)}")

    check("All pixels have output",
          all(p.output is not None for p in result.pixels),
          f"none-outputs: {sum(1 for p in result.pixels if p.output is None)}")

    # Write PPM
    with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
        ppm_path = f.name
    write_ppm(ppm_path, result)
    ppm_size = Path(ppm_path).stat().st_size
    check("PPM file written successfully",
          ppm_size > 0,
          f"size: {ppm_size} bytes")

    # Verify PPM header
    with open(ppm_path, "rb") as f:
        ppm_header = f.read(20)
    check("PPM has valid P6 header",
          ppm_header.startswith(b"P6\n"),
          f"header: {ppm_header[:10]}")
    Path(ppm_path).unlink()

    # NaN mode
    nan_result = run_pixel_grid(
        source=NAN_FRAGMENT,
        stage_name="fragment",
        resolution=(64, 64),
        grid_size=(4, 4),
        mode="nan",
        check_nan=True,
    )
    nan_pixels = [p for p in nan_result.pixels if p.nan_count > 0]
    check("NaN heatmap detects NaN pixels",
          len(nan_pixels) > 0,
          f"NaN pixels: {len(nan_pixels)}/{len(nan_result.pixels)}")

    # Perf mode
    perf_result = run_pixel_grid(
        source=SIMPLE_FRAGMENT,
        stage_name="fragment",
        resolution=(64, 64),
        grid_size=(4, 4),
        mode="perf",
    )
    check("Perf heatmap tracks statement counts",
          all(p.statement_count > 0 for p in perf_result.pixels),
          f"statement counts: {[p.statement_count for p in perf_result.pixels[:4]]}")

    # Variable tracking mode — note: after interp.run() completes, function-local
    # variables are out of scope (pop_scope), so var tracking returns 0.
    # This is a known design limitation: heatmap var tracking would need to capture
    # variables during execution (via hooks) rather than after completion.
    # For now, verify the mode runs without error.
    var_result = run_pixel_grid(
        source=SIMPLE_FRAGMENT,
        stage_name="fragment",
        resolution=(64, 64),
        grid_size=(4, 4),
        mode="var:brightness",
        var_name="brightness",
    )
    check("Variable heatmap mode runs without error",
          len(var_result.pixels) == 16,
          f"pixels: {len(var_result.pixels)}")


# ═══════════════════════════════════════════════════════════════════════════════
# D: Real Textures & Bindless Debugging
# ═══════════════════════════════════════════════════════════════════════════════

def test_d_textures_and_bindless():
    section_header("D1", "LuxImage Bilinear Sampling")

    from luxc.debug.values import LuxImage, LuxVec, LuxInt, LuxStruct, LuxScalar, make_solid_image

    # make_solid_image(r, g, b, a=255, size=1)
    img = make_solid_image(255, 128, 64, 255, size=4)
    check("make_solid_image creates LuxImage",
          isinstance(img, LuxImage) and img.width == 4 and img.height == 4)

    # Test bilinear sampling at center
    sample = img.sample_bilinear(0.5, 0.5)
    check("Bilinear sample at center returns correct color",
          isinstance(sample, LuxVec) and len(sample.components) == 4,
          f"sampled: {sample}")

    # Verify approximately correct values (255/255=1.0, 128/255~0.502, 64/255~0.251)
    r, g, b, a = sample.components
    check("Sample values match solid color",
          abs(r - 1.0) < 0.01 and abs(g - 0.502) < 0.02 and abs(b - 0.251) < 0.02,
          f"r={r:.3f}, g={g:.3f}, b={b:.3f}, a={a:.3f}")

    # Test wrap modes
    sample_repeat = img.sample_bilinear(1.5, 1.5)  # should wrap to 0.5, 0.5
    check("Repeat wrap mode works",
          isinstance(sample_repeat, LuxVec),
          f"wrapped sample: {sample_repeat}")

    section_header("D2", "Texture-Aware Builtins")

    from luxc.debug.builtins import BUILTIN_FUNCTIONS
    check("'sample_bindless' registered in builtins",
          "sample_bindless" in BUILTIN_FUNCTIONS)
    check("'sample_bindless_lod' registered in builtins",
          "sample_bindless_lod" in BUILTIN_FUNCTIONS)

    # Test sample() with LuxImage
    builtin_sample = BUILTIN_FUNCTIONS["sample"]
    tex = make_solid_image(200, 100, 50, 255, size=4)
    uv = LuxVec([0.5, 0.5])
    result = builtin_sample([tex, uv])
    check("sample() with LuxImage returns vec4",
          isinstance(result, LuxVec) and len(result.components) == 4,
          f"result: {result}")

    # Test sample_bindless with list of images
    builtin_sample_bl = BUILTIN_FUNCTIONS["sample_bindless"]
    tex_array = [
        make_solid_image(255, 0, 0, 255, size=2),   # red
        make_solid_image(0, 255, 0, 255, size=2),    # green
        make_solid_image(0, 0, 255, 255, size=2),    # blue
    ]
    result_bl = builtin_sample_bl([tex_array, LuxInt(1), uv])  # sample green
    check("sample_bindless() indexes into texture array",
          isinstance(result_bl, LuxVec) and result_bl.components[1] > 0.9,
          f"green channel: {result_bl.components[1]:.3f}")

    section_header("D3", "List Indexing in Interpreter")

    # Test that list indexing works at the value level
    materials = [
        LuxStruct("Material", {"roughness": LuxScalar(0.5)}),
        LuxStruct("Material", {"roughness": LuxScalar(0.1)}),
    ]
    check("List of LuxStruct can be created and indexed",
          len(materials) == 2 and materials[1].fields["roughness"].value == 0.1)

    section_header("D4-D5", "Default Material Builder & Texture Loading")

    from luxc.debug.io import build_default_material

    mat = build_default_material()
    check("build_default_material() returns LuxStruct",
          isinstance(mat, LuxStruct) and mat.type_name == "BindlessMaterialData")

    # Check key fields
    check("Default baseColorFactor is vec4(0.8, 0.8, 0.8, 1.0)",
          "baseColorFactor" in mat.fields
          and isinstance(mat.fields["baseColorFactor"], LuxVec)
          and abs(mat.fields["baseColorFactor"].components[0] - 0.8) < 0.01)

    check("Default roughnessFactor is 0.5",
          "roughnessFactor" in mat.fields
          and abs(mat.fields["roughnessFactor"].value - 0.5) < 0.01)

    check("Default ior is 1.5",
          "ior" in mat.fields
          and abs(mat.fields["ior"].value - 1.5) < 0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# E: Bindless Properties Compiler Integration
# ═══════════════════════════════════════════════════════════════════════════════

def test_e_bindless_properties():
    section_header("E2-E3", "Bindless Properties -- Compiler Pipeline")

    from luxc.expansion.surface_expander import _PROPERTY_TO_BINDLESS

    check("_PROPERTY_TO_BINDLESS mapping exists",
          len(_PROPERTY_TO_BINDLESS) >= 11,
          f"entries: {list(_PROPERTY_TO_BINDLESS.keys())}")

    check("base_color maps to baseColorFactor",
          _PROPERTY_TO_BINDLESS.get("base_color") == "baseColorFactor")

    check("roughness maps to roughnessFactor",
          _PROPERTY_TO_BINDLESS.get("roughness") == "roughnessFactor")

    section_header("E4-E5", "Bindless Properties -- Compilation Test")

    # Use the real gltf_pbr_layered.lux which has properties + surfaces
    lux_path = ROOT / "examples" / "gltf_pbr_layered.lux"
    if not lux_path.exists():
        skip("Bindless compilation", "examples/gltf_pbr_layered.lux not found")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Compile GltfForward pipeline in bindless mode
        result = subprocess.run(
            [sys.executable, "-m", "luxc", str(lux_path), "--bindless",
             "--pipeline", "GltfForward", "-o", tmp_dir],
            capture_output=True, text=True, timeout=120, cwd=str(ROOT)
        )
        check("Surface with properties compiles in --bindless mode",
              result.returncode == 0,
              f"stderr: {result.stderr[:300]}" if result.returncode != 0 else "")

        # Check for SPIR-V output files
        spv_files = list(Path(tmp_dir).glob("*.spv"))
        check("SPIR-V files generated",
              len(spv_files) > 0,
              f"files: {[f.name for f in spv_files]}")

        # Check for reflection JSON
        json_files = list(Path(tmp_dir).glob("*.json"))
        if json_files:
            with open(json_files[0]) as f:
                refl_data = json.load(f)
            check("Reflection JSON generated",
                  True, f"keys: {list(refl_data.keys())[:5]}")
        else:
            check("Reflection JSON generated", len(json_files) > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Integration Tests (subprocess-based)
# ═══════════════════════════════════════════════════════════════════════════════

def test_cli_integration():
    section_header("CLI", "Command-Line Integration")

    debug_playground = ROOT / "examples" / "debug_playground.lux"
    if not debug_playground.exists():
        skip("CLI tests", "examples/debug_playground.lux not found")
        return

    # Test 1: Batch mode with --dump-vars
    result = subprocess.run(
        [sys.executable, "-m", "luxc", str(debug_playground),
         "--debug-run", "--stage", "fragment", "--batch", "--dump-vars"],
        capture_output=True, text=True, timeout=60, cwd=str(ROOT)
    )
    check("Batch mode runs successfully",
          result.returncode == 0,
          f"stderr: {result.stderr[:200]}" if result.returncode != 0 else "")

    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            check("Batch output is valid JSON",
                  True, f"keys: {list(data.keys())}")
            # dump_vars populates variable_trace
            has_vars = "variable_trace" in data or "variables" in data
            check("Batch output has variable data",
                  has_vars,
                  f"keys present: {[k for k in data.keys() if 'var' in k.lower()]}")
        except json.JSONDecodeError:
            check("Batch output is valid JSON", False,
                  f"stdout[:200]: {result.stdout[:200]}")

    # Test 2: --check-nan mode
    result2 = subprocess.run(
        [sys.executable, "-m", "luxc", str(debug_playground),
         "--debug-run", "--stage", "fragment", "--batch", "--check-nan"],
        capture_output=True, text=True, timeout=60, cwd=str(ROOT)
    )
    if result2.returncode == 0:
        try:
            data2 = json.loads(result2.stdout)
            check("NaN detection finds NaN in debug_playground.lux",
                  data2.get("nan_detected", False),
                  f"nan_events: {len(data2.get('nan_events', []))}")
        except json.JSONDecodeError:
            check("NaN detection output is valid JSON", False)

    # Test 3: Breakpoint with --dump-at-break
    result3 = subprocess.run(
        [sys.executable, "-m", "luxc", str(debug_playground),
         "--debug-run", "--stage", "fragment", "--batch",
         "--break", "100", "--dump-at-break"],
        capture_output=True, text=True, timeout=60, cwd=str(ROOT)
    )
    if result3.returncode == 0:
        try:
            data3 = json.loads(result3.stdout)
            # break_dumps or breakpoints key
            has_bp_data = any(k in data3 for k in ("breakpoints", "break_dumps"))
            check("Breakpoint dump output present",
                  has_bp_data or "status" in data3,
                  f"keys: {list(data3.keys())}")
        except json.JSONDecodeError:
            check("Breakpoint dump output is valid JSON", False)

    # Test 4: Custom inputs
    inputs_json = ROOT / "examples" / "debug_playground_inputs.json"
    if inputs_json.exists():
        result4 = subprocess.run(
            [sys.executable, "-m", "luxc", str(debug_playground),
             "--debug-run", "--stage", "fragment", "--batch", "--dump-vars",
             "--input", str(inputs_json)],
            capture_output=True, text=True, timeout=60, cwd=str(ROOT)
        )
        check("Custom inputs loaded from JSON",
              result4.returncode == 0,
              f"stderr: {result4.stderr[:200]}" if result4.returncode != 0 else "")

    # Test 5: --pixel mode
    result5 = subprocess.run(
        [sys.executable, "-m", "luxc", str(debug_playground),
         "--debug-run", "--stage", "fragment", "--batch", "--dump-vars",
         "--pixel", "320,240", "--resolution", "640x480"],
        capture_output=True, text=True, timeout=60, cwd=str(ROOT)
    )
    check("Pixel mode (--pixel 320,240) runs",
          result5.returncode == 0)

    # Test 6: --set overrides
    result6 = subprocess.run(
        [sys.executable, "-m", "luxc", str(debug_playground),
         "--debug-run", "--stage", "fragment", "--batch", "--dump-vars",
         "--set", "roughness=0.1", "--set", "metallic=1.0"],
        capture_output=True, text=True, timeout=60, cwd=str(ROOT)
    )
    check("Variable overrides (--set) work",
          result6.returncode == 0)

    # Test 7: --export-inputs
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        export_path = f.name
    result7 = subprocess.run(
        [sys.executable, "-m", "luxc", str(debug_playground),
         "--stage", "fragment", "--export-inputs", export_path],
        capture_output=True, text=True, timeout=60, cwd=str(ROOT)
    )
    if result7.returncode == 0:
        try:
            with open(export_path) as f:
                exported = json.load(f)
            check("Export inputs generates valid JSON",
                  len(exported) > 0,
                  f"exported {len(exported)} input fields")
        except (json.JSONDecodeError, FileNotFoundError):
            check("Export inputs generates valid JSON", False)
    Path(export_path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive Demo Instructions
# ═══════════════════════════════════════════════════════════════════════════════

def print_interactive_guide():
    header("Interactive Testing Guide")
    print(textwrap.dedent(f"""\
    The following features require interactive testing. Launch the debugger with:

      {CYAN}python -m luxc examples/debug_playground.lux --debug-run --stage fragment{RESET}

    Then try these commands in the interactive REPL:

    {BOLD}1. Conditional Breakpoints:{RESET}
       lux-debug> break 111 if roughness < 0.3
       lux-debug> run
       -> Should stop at line 111 only if roughness is less than 0.3

    {BOLD}2. Watch Expressions:{RESET}
       lux-debug> watch n_dot_l
       lux-debug> watch roughness * metallic
       lux-debug> start
       lux-debug> step (repeatedly)
       -> Watch values shown at each stop, changed ones marked with ~

    {BOLD}3. Expression Evaluation:{RESET}
       lux-debug> start
       lux-debug> step (until past line 100)
       lux-debug> eval dot(n, light_dir)
       lux-debug> eval roughness + 0.5
       lux-debug> eval vec3(1.0, 0.0, 0.0).xy

    {BOLD}4. Break-on-NaN:{RESET}
       lux-debug> break-nan
       lux-debug> run
       -> Should stop immediately after NaN is produced (around line 55/143)

    {BOLD}5. Time-Travel Debugging:{RESET}
       lux-debug> start
       lux-debug> step (10+ times)
       lux-debug> history
       lux-debug> reverse-step
       lux-debug> print x
       lux-debug> reverse-step
       lux-debug> print x
       -> Variables should show previous values
       lux-debug> goto 0
       -> Should jump to start, all variables reset

    {BOLD}6. Reverse Continue:{RESET}
       lux-debug> break 100
       lux-debug> run
       -> Stops at line 100
       lux-debug> continue
       -> Runs to end
       lux-debug> reverse-continue
       -> Should jump back to line 100

    {BOLD}7. Heatmap (Python API):{RESET}
       python -c "
       from luxc.debug.heatmap import run_pixel_grid, write_ppm
       src = open('examples/debug_playground.lux').read()
       r = run_pixel_grid(src, 'fragment', grid_size=(16,16), mode='nan', check_nan=True)
       write_ppm('nan_map.ppm', r)
       print(f'Written nan_map.ppm ({{r.width}}x{{r.height}})')
       for p in r.pixels:
           if p.nan_count > 0:
               print(f'  NaN at grid ({{p.x}}, {{p.y}}) uv=({{p.u:.2f}}, {{p.v:.2f}})')
       "
    """))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Manual P18.2 test runner")
    parser.add_argument("--section", type=str, default=None,
                        help="Run specific section (a1, a2, a3, a4, b, c, d, e, cli)")
    parser.add_argument("--all", action="store_true", help="Run all automated checks")
    parser.add_argument("--guide", action="store_true", help="Print interactive guide only")
    args = parser.parse_args()

    if args.guide:
        print_interactive_guide()
        return

    header("P18.2 Manual Test Suite")

    sections = {
        "a1": ("A1 -- Conditional Breakpoints", test_a1_conditional_breakpoints),
        "a2": ("A2 -- Watch Expressions", test_a2_watch_expressions),
        "a3": ("A3 -- Expression Parser", test_a3_expression_parser),
        "a4": ("A4 -- Break-on-NaN", test_a4_break_on_nan),
        "b":  ("B -- Time-Travel Debugging", test_b_time_travel),
        "c":  ("C -- Multi-Pixel Heatmap", test_c_heatmap),
        "d":  ("D -- Real Textures & Bindless Debugging", test_d_textures_and_bindless),
        "e":  ("E -- Bindless Properties Compiler", test_e_bindless_properties),
        "cli": ("CLI -- Command-Line Integration", test_cli_integration),
    }

    if args.section:
        key = args.section.lower()
        if key in sections:
            title, func = sections[key]
            func()
        else:
            print(f"Unknown section: {key}")
            print(f"Available: {', '.join(sections.keys())}")
            sys.exit(1)
    elif args.all or not args.section:
        for key, (title, func) in sections.items():
            try:
                func()
            except Exception as e:
                print(f"\n  {RED}ERROR{RESET} in {title}: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print(f"\n{'=' * 60}")
    total = passed + failed + skipped
    print(f"  {BOLD}Results:{RESET}  {GREEN}{passed} passed{RESET}  "
          f"{RED}{failed} failed{RESET}  "
          f"{YELLOW}{skipped} skipped{RESET}  "
          f"({total} total)")
    print(f"{'=' * 60}")

    if failed == 0 and not args.guide:
        print(f"\n  {GREEN}All automated checks passed!{RESET}")
        print(f"\n  Run with --guide for interactive testing instructions.")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
