"""Symbol table with hierarchical scopes."""

from __future__ import annotations
from dataclasses import dataclass, field
from luxc.builtins.types import LuxType


@dataclass
class Symbol:
    name: str
    type: LuxType
    kind: str  # "variable", "input", "output", "uniform_field", "push_field",
               # "constant", "sampler", "param", "builtin_position"


class Scope:
    def __init__(self, parent: Scope | None = None):
        self.parent = parent
        self.symbols: dict[str, Symbol] = {}

    def define(self, sym: Symbol) -> None:
        self.symbols[sym.name] = sym

    def lookup(self, name: str) -> Symbol | None:
        if name in self.symbols:
            return self.symbols[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None

    def child(self) -> Scope:
        return Scope(parent=self)
