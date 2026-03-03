"""Interactive CLI for the Lux shader debugger (gdb-style REPL)."""

from __future__ import annotations

import sys
import json
from pathlib import Path

from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import Module, StageBlock
from luxc.debug.values import LuxValue, value_to_json, value_type_name
from luxc.debug.interpreter import Interpreter, InterpResult
from luxc.debug.debugger import Debugger, BreakpointHit, StepMode
from luxc.debug.io import load_inputs_from_json, build_default_inputs


def _find_stage(module: Module, stage_name: str) -> StageBlock | None:
    for stage in module.stages:
        if stage.stage_type == stage_name:
            return stage
    return None


def _print_value(name: str, val: LuxValue) -> None:
    type_str = value_type_name(val)
    print(f"  {name} = {val!r} ({type_str})")


def run_interactive(source: str, stage_name: str, source_name: str = "<input>",
                    input_path: Path | None = None, source_lines: list[str] | None = None) -> None:
    """Run the interactive shader debugger REPL."""
    from luxc.analysis.type_checker import type_check
    from luxc.optimization.const_fold import constant_fold

    module = parse_lux(source)

    # Light-weight pipeline for debugger (no codegen)
    if module.surfaces or module.pipelines or module.environments or module.procedurals:
        from luxc.expansion.surface_expander import expand_surfaces
        expand_surfaces(module)

    type_check(module)
    constant_fold(module)

    stage = _find_stage(module, stage_name)
    if stage is None:
        available = [s.stage_type for s in module.stages]
        print(f"Error: stage '{stage_name}' not found. Available: {available}")
        return

    lines = source_lines or source.splitlines()

    # Build inputs
    inputs = build_default_inputs(stage)
    if input_path:
        user_inputs = load_inputs_from_json(input_path)
        inputs.update(user_inputs)

    interp = Interpreter(module, stage, source_lines=lines)
    dbg = Debugger(interp)

    def _handle_inspect_command(command: str, arg: str) -> bool:
        """Handle inspection commands (print, locals, source, etc.).

        Returns True if the command was handled, False if not recognized.
        """
        if command in ("print", "p"):
            if not arg:
                print("Usage: print <variable_name>")
            else:
                val = dbg.get_variable(arg)
                if val is not None:
                    _print_value(arg, val)
                else:
                    print(f"Variable '{arg}' not found")
            return True

        elif command in ("locals", "l"):
            variables = dbg.get_locals()
            if variables:
                for name, val in sorted(variables.items()):
                    _print_value(name, val)
            else:
                print("No variables in scope")
            return True

        elif command in ("watch", "w"):
            if not arg:
                if dbg.watches:
                    for w in dbg.watches.values():
                        print(f"  Watch {w.id}: {w.expression}")
                else:
                    print("No watches set")
            else:
                w = dbg.add_watch(arg)
                print(f"Watch {w.id}: {arg}")
            return True

        elif command in ("source", "src"):
            line_num = dbg._current_line
            if arg:
                try:
                    line_num = int(arg)
                except ValueError:
                    pass
            if line_num > 0:
                _show_source_context(lines, line_num, context=5)
            else:
                print("No current line")
            return True

        elif command == "output":
            result = interp.result
            if result.output:
                print(f"Output: {result.output!r}")
            else:
                print("No output yet")
            return True

        elif command in ("breakpoints", "info"):
            if dbg.breakpoints:
                for bp in dbg.breakpoints.values():
                    status = "enabled" if bp.enabled else "disabled"
                    print(f"  Breakpoint {bp.id}: line {bp.line} [{status}] (hits: {bp.hit_count})")
            else:
                print("No breakpoints set")
            return True

        elif command in ("backtrace", "bt"):
            print(f"  (at line {dbg._current_line}, depth {dbg._current_depth})")
            return True

        elif command in ("list",):
            # Show full shader source with cursor and breakpoint markers
            bp_lines = {bp.line for bp in dbg.breakpoints.values() if bp.enabled}
            cur = dbg._current_line
            for i, line_text in enumerate(lines):
                ln = i + 1
                if ln == cur:
                    marker = " > "
                elif ln in bp_lines:
                    marker = " * "
                else:
                    marker = "   "
                print(f"{marker}{ln:4d} | {line_text}")
            return True

        elif command in ("help", "h"):
            _print_help()
            return True

        return False

    def on_break(hit: BreakpointHit):
        """Called when execution stops (breakpoint or step).

        Enters a nested command loop — blocks until user issues
        step/next/continue to resume execution.
        """
        if hit.breakpoint.id > 0:
            print(f"Hit breakpoint {hit.breakpoint.id} at line {hit.line}")
        else:
            print(f"Stopped at line {hit.line}")
        _show_source_context(lines, hit.line)

        # Nested command loop — runs until user resumes execution
        while True:
            try:
                cmd = input("lux-debug> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nContinuing...")
                dbg._step_mode = StepMode.RUN
                return

            if not cmd:
                continue

            parts = cmd.split(None, 1)
            command = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            # Step/next/continue — resume execution
            if command in ("step", "s"):
                dbg._step_mode = StepMode.STEP
                return

            elif command in ("next", "n"):
                dbg._step_mode = StepMode.NEXT
                dbg._step_depth = dbg._current_depth
                return

            elif command in ("continue", "c"):
                dbg._step_mode = StepMode.RUN
                return

            elif command in ("finish", "f"):
                dbg._step_mode = StepMode.FINISH
                dbg._step_depth = dbg._current_depth
                return

            # Break/delete — modify breakpoints mid-run
            elif command in ("break", "b"):
                if not arg:
                    print("Usage: break <line_number>")
                else:
                    try:
                        line = int(arg)
                        bp = dbg.add_breakpoint(line)
                        print(f"Breakpoint {bp.id} at line {line}")
                    except ValueError:
                        print(f"Invalid line number: {arg}")

            elif command in ("delete", "d"):
                if not arg:
                    print("Usage: delete <breakpoint_id>")
                else:
                    try:
                        bp_id = int(arg)
                        if dbg.remove_breakpoint(bp_id):
                            print(f"Deleted breakpoint {bp_id}")
                        else:
                            print(f"No breakpoint with id {bp_id}")
                    except ValueError:
                        print(f"Invalid id: {arg}")

            elif command in ("quit", "q", "exit"):
                raise KeyboardInterrupt("quit")

            # Inspection commands
            elif not _handle_inspect_command(command, arg):
                print(f"Unknown command: '{command}'. Type 'help' for commands.")

    dbg._on_break = on_break

    # Track variable state between steps to show what changed
    _prev_vars: dict[str, LuxValue] = {}

    def _show_changes(current_vars: dict[str, LuxValue]):
        """Show variables that changed since last stop."""
        nonlocal _prev_vars
        for name, val in sorted(current_vars.items()):
            if name not in _prev_vars:
                # New variable
                _print_value(f"+ {name}", val)
            elif repr(val) != repr(_prev_vars.get(name)):
                # Changed variable
                _print_value(f"~ {name}", val)
        _prev_vars = {k: v for k, v in current_vars.items()}

    # Monkey-patch on_break to include change tracking
    _orig_on_break = on_break
    def on_break_with_changes(hit: BreakpointHit):
        # Show what changed since last stop (only in step mode, not first stop)
        if _prev_vars and dbg._step_mode == StepMode.STEP:
            _show_changes(hit.variables)
        else:
            _prev_vars.update(hit.variables)
        _orig_on_break(hit)
    dbg._on_break = on_break_with_changes

    print(f"lux-debug: {source_name} ({stage_name} stage)")
    print(f"Type 'help' for available commands.\n")

    finished = False

    def _run_shader(step_mode: StepMode):
        """Execute the shader with given step mode, handling results."""
        nonlocal finished, interp, dbg

        if finished:
            # Reset for re-run
            interp = Interpreter(module, stage, source_lines=lines)
            dbg = Debugger(interp)
            dbg._on_break = on_break
            finished = False

        dbg._step_mode = step_mode
        try:
            result = interp.run(inputs=inputs, trace_all=False)
            finished = True
            if result.output:
                print(f"Output: {result.output!r}")
            if result.nan_detected:
                print(f"NaN detected! ({len(result.nan_events)} events)")
            if result.assert_failures:
                for msg in result.assert_failures:
                    print(f"  {msg}")
            print(f"Executed {result.statements_executed} statements.")
        except KeyboardInterrupt:
            print("\nExecution interrupted.")
            finished = True
        except Exception as e:
            print(f"Runtime error: {e}")
            finished = True

    # Main command loop (pre-run)
    while True:
        try:
            cmd = input("lux-debug> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nQuit.")
            break

        if not cmd:
            continue

        parts = cmd.split(None, 1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command in ("quit", "q", "exit"):
            break

        elif command in ("run", "r"):
            _run_shader(StepMode.RUN)

        elif command in ("start",):
            _run_shader(StepMode.STEP)

        elif command in ("step", "s"):
            if finished:
                # Fresh start in step mode
                _run_shader(StepMode.STEP)
            else:
                print("Not running. Use 'start' to begin stepping, or 'run' to run.")

        elif command in ("next", "n"):
            if finished:
                _run_shader(StepMode.STEP)
            else:
                print("Not running. Use 'start' to begin stepping, or 'run' to run.")

        elif command in ("break", "b"):
            if not arg:
                print("Usage: break <line_number>")
                continue
            try:
                line = int(arg)
                bp = dbg.add_breakpoint(line)
                print(f"Breakpoint {bp.id} at line {line}")
            except ValueError:
                print(f"Invalid line number: {arg}")

        elif command in ("delete", "d"):
            if not arg:
                print("Usage: delete <breakpoint_id>")
                continue
            try:
                bp_id = int(arg)
                if dbg.remove_breakpoint(bp_id):
                    print(f"Deleted breakpoint {bp_id}")
                else:
                    print(f"No breakpoint with id {bp_id}")
            except ValueError:
                print(f"Invalid id: {arg}")

        elif not _handle_inspect_command(command, arg):
            print(f"Unknown command: '{command}'. Type 'help' for commands.")


def run_batch(source: str, stage_name: str, source_name: str = "<input>",
              input_path: Path | None = None,
              dump_vars: bool = False, check_nan: bool = False,
              break_lines: list[int] | None = None,
              dump_at_break: bool = False) -> dict:
    """Run the debugger in batch mode, returning JSON output."""
    from luxc.analysis.type_checker import type_check
    from luxc.optimization.const_fold import constant_fold

    module = parse_lux(source)

    if module.surfaces or module.pipelines or module.environments or module.procedurals:
        from luxc.expansion.surface_expander import expand_surfaces
        expand_surfaces(module)

    type_check(module)
    constant_fold(module)

    stage = _find_stage(module, stage_name)
    if stage is None:
        return {"status": "error", "message": f"Stage '{stage_name}' not found"}

    lines = source.splitlines()

    # Build inputs
    inputs = build_default_inputs(stage)
    if input_path:
        user_inputs = load_inputs_from_json(input_path)
        inputs.update(user_inputs)

    interp = Interpreter(module, stage, source_lines=lines)
    dbg = Debugger(interp)

    return dbg.run_batch(
        inputs=inputs,
        dump_vars=dump_vars,
        check_nan=check_nan,
        break_lines=break_lines,
        dump_at_break=dump_at_break,
    )


def _show_source_context(lines: list[str], current_line: int, context: int = 3) -> None:
    """Display source lines around the current line."""
    start = max(0, current_line - context - 1)
    end = min(len(lines), current_line + context)
    for i in range(start, end):
        line_num = i + 1
        marker = " > " if line_num == current_line else "   "
        print(f"{marker}{line_num:4d} | {lines[i]}")


def _print_help() -> None:
    print("""Commands:
  start                Begin execution, stop at first statement
  run / r              Run to completion (or next breakpoint)
  step / s             Step into (next statement) — shows variable changes
  next / n             Step over (skip function bodies)
  continue / c         Continue to next breakpoint
  finish / f           Run until current function returns
  break / b <line>     Set breakpoint at line
  delete / d <id>      Delete breakpoint
  print / p <var>      Print variable value
  locals / l           Show all variables in scope
  list                 Show full shader with cursor (>) and breakpoints (*)
  watch / w [expr]     Add/list watches
  source / src [line]  Show source context (5 lines)
  output               Show current output
  breakpoints / info   List breakpoints
  backtrace / bt       Show current position
  quit / q             Exit debugger
  help / h             Show this help

Workflow:
  start              -> stops at first statement, then use step/next/continue
  break 42 + run     -> runs until line 42, then use step/next/continue
  run                -> runs to completion (stops only at breakpoints)

When stepping, new/changed variables are shown automatically:
  + albedo  = vec3(0.800, ...)    <- new variable
  ~ n_dot_l = 0.333 (scalar)     <- value changed""")
