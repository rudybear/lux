"""Built-in function signatures for Lux."""

from __future__ import annotations
from dataclasses import dataclass
from luxc.builtins.types import (
    LuxType, SCALAR, INT, UINT, BOOL, VOID,
    VEC2, VEC3, VEC4, MAT2, MAT3, MAT4, SAMPLER2D,
    VectorType, MatrixType, ScalarType,
)


@dataclass(frozen=True)
class FuncSig:
    name: str
    params: tuple[LuxType, ...]
    return_type: LuxType


# Helper: generate overloads for scalar + vecN float functions
def _float_overloads(name: str, arity: int) -> list[FuncSig]:
    types = [SCALAR, VEC2, VEC3, VEC4]
    sigs = []
    for t in types:
        sigs.append(FuncSig(name, tuple([t] * arity), t))
    return sigs


def _float_overloads_1(name: str) -> list[FuncSig]:
    return _float_overloads(name, 1)


def _float_overloads_2(name: str) -> list[FuncSig]:
    return _float_overloads(name, 2)


def _float_overloads_3(name: str) -> list[FuncSig]:
    return _float_overloads(name, 3)


def _build_builtins() -> dict[str, list[FuncSig]]:
    table: dict[str, list[FuncSig]] = {}

    def add(sigs: list[FuncSig]):
        for s in sigs:
            table.setdefault(s.name, []).append(s)

    # --- GLSL.std.450 math functions ---
    # 1-arg: normalize, abs, floor, ceil, fract, sqrt, sin, cos, tan,
    #        asin, acos, atan, exp, exp2, log, log2, sign, length
    for fn in ["abs", "floor", "ceil", "fract", "sqrt",
               "sin", "cos", "tan", "asin", "acos", "atan",
               "exp", "exp2", "log", "log2", "sign", "normalize"]:
        add(_float_overloads_1(fn))

    # length returns scalar
    for vt in [SCALAR, VEC2, VEC3, VEC4]:
        add([FuncSig("length", (vt,), SCALAR)])

    # 2-arg: min, max, pow, step, reflect, distance, dot, mod
    for fn in ["min", "max", "pow", "step", "reflect"]:
        add(_float_overloads_2(fn))

    # distance returns scalar
    for vt in [VEC2, VEC3, VEC4]:
        add([FuncSig("distance", (vt, vt), SCALAR)])

    # dot returns scalar
    for vt in [VEC2, VEC3, VEC4]:
        add([FuncSig("dot", (vt, vt), SCALAR)])

    # cross: vec3 only
    add([FuncSig("cross", (VEC3, VEC3), VEC3)])

    # 3-arg: mix, clamp, smoothstep, fma
    for fn in ["mix", "clamp", "smoothstep", "fma"]:
        add(_float_overloads_3(fn))

    # mix with scalar third arg
    for vt in [VEC2, VEC3, VEC4]:
        add([FuncSig("mix", (vt, vt, SCALAR), vt)])
        add([FuncSig("clamp", (vt, SCALAR, SCALAR), vt)])

    # texture sampling
    add([FuncSig("sample", (SAMPLER2D, VEC2), VEC4)])

    return table


BUILTIN_FUNCTIONS: dict[str, list[FuncSig]] = _build_builtins()


def lookup_builtin(name: str, arg_types: list[LuxType]) -> FuncSig | None:
    candidates = BUILTIN_FUNCTIONS.get(name)
    if candidates is None:
        return None
    for sig in candidates:
        if len(sig.params) != len(arg_types):
            continue
        if all(_type_matches(p, a) for p, a in zip(sig.params, arg_types)):
            return sig
    return None


def _type_matches(param: LuxType, arg: LuxType) -> bool:
    return param.name == arg.name
