"""Debugger with breakpoints, stepping, and NaN detection."""

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

    def _debug_exec_stmt(self, stmt) -> None:
        """Intercept statement execution for debugging."""
        loc = getattr(stmt, 'loc', None)
        line_num = loc.line if loc else 0
        self._current_line = line_num

        # Check breakpoints
        should_stop = False

        if self._step_mode == StepMode.STEP:
            should_stop = True
        elif self._step_mode == StepMode.NEXT:
            if self._current_depth <= self._step_depth:
                should_stop = True
        elif self._step_mode == StepMode.RUN:
            # Check breakpoints only
            for bp in self.breakpoints.values():
                if bp.enabled and bp.line == line_num:
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

    def run_batch(self, inputs: dict[str, LuxValue] | None = None,
                  dump_vars: bool = False, check_nan: bool = False,
                  break_lines: list[int] | None = None,
                  dump_at_break: bool = False) -> dict:
        """Run in batch mode, returning JSON-serializable results."""
        # Set up breakpoints
        if break_lines:
            for line in break_lines:
                self.add_breakpoint(line)

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

        return output
