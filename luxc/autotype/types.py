"""Shared data types for auto-type precision analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class Precision(Enum):
    FP16 = "fp16"
    FP32 = "fp32"
    UNKNOWN = "unknown"


# fp16 max representable value
_FP16_MAX = 65504.0


@dataclass
class Interval:
    """Tracks observed or inferred [lo, hi] range for a single scalar component."""

    lo: float = float('inf')
    hi: float = float('-inf')
    has_nan: bool = False
    has_inf: bool = False

    def update(self, value: float) -> None:
        """Update interval with an observed value."""
        if math.isnan(value):
            self.has_nan = True
            return
        if math.isinf(value):
            self.has_inf = True
            # Still track the sign direction
            if value > 0:
                self.hi = float('inf')
            else:
                self.lo = float('-inf')
            return
        self.lo = min(self.lo, value)
        self.hi = max(self.hi, value)

    def merge(self, other: Interval) -> None:
        """Merge another interval into this one (union)."""
        self.lo = min(self.lo, other.lo)
        self.hi = max(self.hi, other.hi)
        self.has_nan = self.has_nan or other.has_nan
        self.has_inf = self.has_inf or other.has_inf

    def fits_fp16(self) -> bool:
        """Check if this interval's values can be represented in fp16."""
        if self.has_nan or self.has_inf:
            return False
        if self.lo == float('inf') and self.hi == float('-inf'):
            # No data observed — not safe to assume fp16
            return False
        abs_max = max(abs(self.lo), abs(self.hi))
        return abs_max <= _FP16_MAX

    @property
    def is_empty(self) -> bool:
        return self.lo == float('inf') and self.hi == float('-inf') and not self.has_nan and not self.has_inf

    def __repr__(self) -> str:
        if self.is_empty:
            return "Interval(empty)"
        parts = [f"[{self.lo:.4g}, {self.hi:.4g}]"]
        if self.has_nan:
            parts.append("NaN")
        if self.has_inf:
            parts.append("Inf")
        return f"Interval({', '.join(parts)})"


@dataclass
class VarRange:
    """Aggregated range info for a single variable."""

    name: str
    type_name: str                  # "scalar", "vec3", etc.
    intervals: list[Interval]       # 1 for scalar, N for vecN
    assignment_count: int = 0

    def all_fit_fp16(self) -> bool:
        return all(iv.fits_fp16() for iv in self.intervals)


@dataclass
class PrecisionDecision:
    """Final precision decision for a variable."""

    precision: Precision
    confidence: float = 1.0         # 0.0..1.0
    reason: str = ""


@dataclass
class PrecisionMap:
    """Precision decisions for all variables in a stage."""

    stage_type: str
    decisions: dict[str, PrecisionDecision] = field(default_factory=dict)

    def fp16_vars(self) -> set[str]:
        """Return the set of variable names marked fp16."""
        return {
            name for name, d in self.decisions.items()
            if d.precision == Precision.FP16
        }
