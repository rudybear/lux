"""Debugger with breakpoints, stepping, NaN detection, and time-travel."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

from luxc.parser.ast_nodes import (
    LetStmt, AssignStmt, ReturnStmt, IfStmt, ForStmt, WhileStmt,
    DebugPrintStmt, AssertStmt, DebugBlock, ExprStmt,
    BreakStmt, ContinueStmt,
)
from luxc.debug.interpreter import Interpreter, ReturnSignal, BreakSignal, ContinueSignal
from luxc.debug.values import LuxValue, is_nan, is_inf, value_to_json


class StepMode(Enum):
    RUN = auto()      # run until breakpoint or end
    STEP = auto()     # step into (every statement)
    NEXT = auto()     # step over (same depth)
    FINISH = auto()   # run until current function returns


@dataclass
class Breakpoint:
    id: int
    line: int
    condition: str | None = None
    hit_count: int = 0
    enabled: bool = True


@dataclass
class WatchEntry:
    id: int
    expression: str
    last_value: LuxValue | None = None


@dataclass
class BreakpointHit:
    """Notification that a breakpoint was hit."""
    breakpoint: Breakpoint
    line: int
    variables: dict[str, LuxValue]


@dataclass
class ExecutionSnapshot:
    """A captured moment of execution for time-travel debugging."""
    step_index: int
    line: int
    env_snapshot: dict[str, LuxValue]


class TimeTravel:
    """Records execution history and allows navigating backwards."""

    def __init__(self, max_history: int = 10000):
        self.history: list[ExecutionSnapshot] = []
        self.current_index: int = -1
        self.recording: bool = True
        self.max_history = max_history
        self._full_snapshot_interval = 100

    def record(self, step_index: int, line: int, env: dict[str, LuxValue]) -> None:
        if not self.recording:
            return
        snapshot = ExecutionSnapshot(
            step_index=step_index,
            line=line,
            env_snapshot={k: copy.deepcopy(v) for k, v in env.items()},
        )
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append(snapshot)
        self.current_index = len(self.history) - 1

    def can_reverse(self) -> bool:
        return self.current_index > 0

    def reverse_step(self) -> ExecutionSnapshot | None:
        if self.current_index > 0:
            self.current_index -= 1
            return self.history[self.current_index]
        return None

    def reverse_continue(self, breakpoint_lines: set[int]) -> ExecutionSnapshot | None:
        idx = self.current_index - 1
        while idx >= 0:
            if self.history[idx].line in breakpoint_lines:
                self.current_index = idx
                return self.history[idx]
            idx -= 1
        return None

    def goto_step(self, step_n: int) -> ExecutionSnapshot | None:
        for i, snap in enumerate(self.history):
            if snap.step_index == step_n:
                self.current_index = i
                return snap
        return None

    def get_recent(self, count: int = 10) -> list[ExecutionSnapshot]:
        start = max(0, self.current_index - count + 1)
        end = self.current_index + 1
        return self.history[start:end]


class Debugger:
    """Wraps the interpreter with debugging support."""

    def __init__(self, interp: Interpreter):
        self.interp = interp
        self.breakpoints: dict[int, Breakpoint] = {}
        self.watches: dict[int, WatchEntry] = {}
        self._next_bp_id = 1
        self._next_watch_id = 1
        self._step_mode = StepMode.RUN
        self._step_depth = 0
        self._current_depth = 0
        self._current_line = 0
        self._stopped = False
        self._on_break: Callable[[BreakpointHit], None] | None = None
        self._on_step: Callable[[int, dict[str, LuxValue]], None] | None = None
        # Variable snapshots at breakpoints
        self.break_snapshots: list[dict[str, LuxValue]] = []

        # Break-on-NaN support
        self.break_on_nan: bool = False
        self._nan_event_count_before: int = 0
        self._on_nan_break: Callable[[list], None] | None = None

        # Time-travel debugging
        self.time_travel = TimeTravel()
        self._step_counter: int = 0

        # Patch the interpreter's exec_stmt to intercept
        self._orig_exec_stmt = interp.exec_stmt
        interp.exec_stmt = self._debug_exec_stmt

    def add_breakpoint(self, line: int, condition: str | None = None) -> Breakpoint:
        bp = Breakpoint(self._next_bp_id, line, condition)
        self.breakpoints[self._next_bp_id] = bp
        self._next_bp_id += 1
        return bp

    def remove_breakpoint(self, bp_id: int) -> bool:
        return self.breakpoints.pop(bp_id, None) is not None

    def add_watch(self, expression: str) -> WatchEntry:
        w = WatchEntry(self._next_watch_id, expression)
        self.watches[self._next_watch_id] = w
        self._next_watch_id += 1
        return w

    def remove_watch(self, watch_id: int) -> bool:
        return self.watches.pop(watch_id, None) is not None

    def step(self) -> None:
        self._step_mode = StepMode.STEP

    def next(self) -> None:
        self._step_mode = StepMode.NEXT
        self._step_depth = self._current_depth

    def continue_run(self) -> None:
        self._step_mode = StepMode.RUN

    def finish(self) -> None:
        self._step_mode = StepMode.FINISH
        self._step_depth = self._current_depth

    def get_locals(self) -> dict[str, LuxValue]:
        return self.interp.env.all_variables()

    def get_variable(self, name: str) -> LuxValue | None:
        return self.interp.env.get(name)

    def _eval_condition(self, expr_str: str) -> bool:
        """Evaluate a breakpoint condition expression.

        Parses the expression using the full Lux parser, evaluates it
        in the current interpreter environment, and returns whether the
        result is truthy. If the condition fails to parse or evaluate,
        the breakpoint is treated as met (returns True).
        """
        try:
            from luxc.debug.expr_parser import parse_debug_expr
            ast_node = parse_debug_expr(expr_str)
            val = self.interp.eval_expr(ast_node)
            return self.interp._is_truthy(val)
        except Exception:
            return True  # If condition fails to parse/eval, treat as met

    def evaluate_watches(self) -> list[tuple[WatchEntry, LuxValue | None, bool]]:
        """Evaluate all watch expressions in the current environment.

        Returns a list of (watch_entry, current_value, changed) tuples.
        If evaluation fails for an entry, value is None and changed is False.
        """
        results = []
        for w in self.watches.values():
            try:
                from luxc.debug.expr_parser import parse_debug_expr
                ast_node = parse_debug_expr(w.expression)
                val = self.interp.eval_expr(ast_node)
                changed = (w.last_value is None or repr(val) != repr(w.last_value))
                w.last_value = copy.deepcopy(val)
                results.append((w, val, changed))
            except Exception:
                results.append((w, None, False))
        return results

    def _debug_exec_stmt(self, stmt) -> None:
        """Intercept statement execution for debugging."""
        loc = getattr(stmt, 'loc', None)
        line_num = loc.line if loc else 0
        self._current_line = line_num

        # Record for time-travel (before execution)
        self._step_counter += 1
        self.time_travel.record(
            self._step_counter, line_num,
            self.interp.env.all_variables()
        )

        # Check breakpoints
        should_stop = False

        if self._step_mode == StepMode.STEP:
            # In step mode, always stop — but also check conditional
            # breakpoints for consistent hit_count tracking
            should_stop = True
            for bp in self.breakpoints.values():
                if bp.enabled and bp.line == line_num:
                    if bp.condition:
                        cond_val = self._eval_condition(bp.condition)
                        if not cond_val:
                            continue
                    bp.hit_count += 1
                    break
        elif self._step_mode == StepMode.NEXT:
            if self._current_depth <= self._step_depth:
                should_stop = True
        elif self._step_mode == StepMode.RUN:
            # Check breakpoints only
            for bp in self.breakpoints.values():
                if bp.enabled and bp.line == line_num:
                    if bp.condition:
                        cond_val = self._eval_condition(bp.condition)
                        if not cond_val:
                            continue
                    bp.hit_count += 1
                    should_stop = True
                    break

        if should_stop and line_num > 0:
            self._stopped = True
            variables = self.interp.env.all_variables()

            # Save snapshot
            snapshot = {k: copy.deepcopy(v) for k, v in variables.items()}
            self.break_snapshots.append(snapshot)

            if self._on_break:
                # Find matching breakpoint (if any)
                bp_match = None
                for bp in self.breakpoints.values():
                    if bp.enabled and bp.line == line_num:
                        bp_match = bp
                        break
                hit = BreakpointHit(
                    breakpoint=bp_match or Breakpoint(0, line_num),
                    line=line_num,
                    variables=snapshot,
                )
                self._on_break(hit)

            if self._on_step:
                self._on_step(line_num, snapshot)

        # Execute the original statement
        self._orig_exec_stmt(stmt)

        # Break-on-NaN: check if new NaN events appeared after execution
        if self.break_on_nan and hasattr(self.interp, 'result') and self.interp.result is not None:
            current_nan_count = len(self.interp.result.nan_events)
            if current_nan_count > self._nan_event_count_before:
                new_events = self.interp.result.nan_events[self._nan_event_count_before:]
                self._nan_event_count_before = current_nan_count
                # Force stop at NEXT statement
                self._step_mode = StepMode.STEP
                # Notify about NaN
                if self._on_nan_break:
                    self._on_nan_break(new_events)

    # -- Time-travel navigation methods --

    def reverse_step(self) -> ExecutionSnapshot | None:
        """Step backwards one statement in execution history."""
        snap = self.time_travel.reverse_step()
        if snap:
            self._restore_snapshot(snap)
        return snap

    def reverse_continue(self) -> ExecutionSnapshot | None:
        """Run backwards until hitting an enabled breakpoint."""
        bp_lines = {bp.line for bp in self.breakpoints.values() if bp.enabled}
        snap = self.time_travel.reverse_continue(bp_lines)
        if snap:
            self._restore_snapshot(snap)
        return snap

    def goto_step(self, step_n: int) -> ExecutionSnapshot | None:
        """Jump to a specific step number in execution history."""
        snap = self.time_travel.goto_step(step_n)
        if snap:
            self._restore_snapshot(snap)
        return snap

    def _restore_snapshot(self, snap: ExecutionSnapshot) -> None:
        """Restore interpreter environment from a snapshot."""
        self._current_line = snap.line
        for name, val in snap.env_snapshot.items():
            if not self.interp.env.set(name, copy.deepcopy(val)):
                self.interp.env.define(name, copy.deepcopy(val))

    def run_batch(self, inputs: dict[str, LuxValue] | None = None,
                  dump_vars: bool = False, check_nan: bool = False,
                  break_lines: list[int] | None = None,
                  dump_at_break: bool = False,
                  break_on_nan: bool = False) -> dict:
        """Run in batch mode, returning JSON-serializable results."""
        # Set up breakpoints
        if break_lines:
            for line in break_lines:
                self.add_breakpoint(line)

        # Set up break-on-NaN
        nan_break_info: list[dict] = []
        if break_on_nan:
            self.break_on_nan = True
            self._nan_event_count_before = 0

            def on_nan_break(new_events):
                for evt in new_events:
                    nan_break_info.append({
                        "line": evt.line,
                        "variable": evt.variable,
                        "operation": evt.operation,
                        "value": value_to_json(evt.value),
                    })
            self._on_nan_break = on_nan_break

        break_dumps: list[dict] = []
        if dump_at_break:
            def on_break(hit: BreakpointHit):
                dump = {
                    "line": hit.line,
                    "variables": {k: value_to_json(v) for k, v in hit.variables.items()},
                }
                break_dumps.append(dump)
            self._on_break = on_break

        # Run
        result = self.interp.run(inputs=inputs, trace_all=dump_vars)

        # Build output
        output: dict = {
            "status": "completed",
            "statements_executed": result.statements_executed,
            "nan_detected": result.nan_detected,
        }

        if result.output is not None:
            output["output"] = value_to_json(result.output)

        if result.nan_events:
            output["nan_events"] = [
                {
                    "line": e.line,
                    "variable": e.variable,
                    "operation": e.operation,
                    "value": value_to_json(e.value),
                }
                for e in result.nan_events
            ]

        if result.debug_prints:
            output["debug_prints"] = result.debug_prints

        if result.assert_failures:
            output["assert_failures"] = result.assert_failures

        if dump_vars and result.variable_trace:
            output["variable_trace"] = [
                {
                    "line": t.line,
                    "name": t.name,
                    "type": t.type_name,
                    "value": value_to_json(t.value),
                }
                for t in result.variable_trace
            ]

        if break_dumps:
            output["break_dumps"] = break_dumps

        if nan_break_info:
            output["nan_breaks"] = nan_break_info

        return output
