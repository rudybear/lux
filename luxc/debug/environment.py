"""Variable scoping and function lookup for the Lux shader debugger."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from luxc.debug.values import LuxValue
from luxc.parser.ast_nodes import FunctionDef


@dataclass
class Scope:
    """A lexical scope frame holding variable bindings."""
    variables: dict[str, LuxValue] = field(default_factory=dict)
    parent: Optional[Scope] = None
    name: str = "<block>"

    def get(self, name: str) -> LuxValue | None:
        if name in self.variables:
            return self.variables[name]
        if self.parent:
            return self.parent.get(name)
        return None

    def set(self, name: str, value: LuxValue) -> bool:
        """Set a variable in the nearest scope where it exists."""
        if name in self.variables:
            self.variables[name] = value
            return True
        if self.parent:
            return self.parent.set(name, value)
        return False

    def define(self, name: str, value: LuxValue) -> None:
        """Define a new variable in this scope."""
        self.variables[name] = value

    def all_variables(self) -> dict[str, LuxValue]:
        """Collect all visible variables (current scope outward)."""
        result = {}
        if self.parent:
            result.update(self.parent.all_variables())
        result.update(self.variables)
        return result


class Environment:
    """Manages scoping and function lookup for the interpreter."""

    def __init__(self) -> None:
        self.global_scope = Scope(name="<global>")
        self.current_scope = self.global_scope
        self.functions: dict[str, FunctionDef] = {}
        self.struct_defs: dict[str, list[tuple[str, str]]] = {}  # name -> [(field, type)]
        self.constants: dict[str, LuxValue] = {}

    def push_scope(self, name: str = "<block>") -> Scope:
        """Enter a new lexical scope."""
        scope = Scope(parent=self.current_scope, name=name)
        self.current_scope = scope
        return scope

    def pop_scope(self) -> None:
        """Leave the current scope."""
        if self.current_scope.parent:
            self.current_scope = self.current_scope.parent

    def get(self, name: str) -> LuxValue | None:
        """Look up a variable by name."""
        # Check constants first
        if name in self.constants:
            return self.constants[name]
        return self.current_scope.get(name)

    def set(self, name: str, value: LuxValue) -> bool:
        """Set an existing variable."""
        return self.current_scope.set(name, value)

    def define(self, name: str, value: LuxValue) -> None:
        """Define a new variable in the current scope."""
        self.current_scope.define(name, value)

    def all_variables(self) -> dict[str, LuxValue]:
        """Return all currently visible variables."""
        result = dict(self.constants)
        result.update(self.current_scope.all_variables())
        return result

    def register_function(self, fn: FunctionDef) -> None:
        """Register a user-defined function."""
        self.functions[fn.name] = fn

    def lookup_function(self, name: str) -> FunctionDef | None:
        """Look up a user-defined function by name."""
        return self.functions.get(name)

    def register_struct(self, name: str, fields: list[tuple[str, str]]) -> None:
        """Register a struct type."""
        self.struct_defs[name] = fields

    def lookup_struct(self, name: str) -> list[tuple[str, str]] | None:
        """Look up struct fields by type name."""
        return self.struct_defs.get(name)
