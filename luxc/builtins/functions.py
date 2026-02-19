"""Built-in function signatures for Lux."""

from __future__ import annotations
from dataclasses import dataclass
from luxc.builtins.types import (
    LuxType, SCALAR, INT, UINT, BOOL, VOID,
    VEC2, VEC3, VEC4, MAT2, MAT3, MAT4, SAMPLER2D, SAMPLER_CUBE,
    ACCELERATION_STRUCTURE, STORAGE_IMAGE, IVEC2, UVEC2,
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
    for fn in ["abs", "floor", "ceil", "fract", "sqrt", "inversesqrt",
               "sin", "cos", "tan", "asin", "acos", "atan",
               "exp", "exp2", "log", "log2", "sign", "normalize"]:
        add(_float_overloads_1(fn))

    # length returns scalar
    for vt in [SCALAR, VEC2, VEC3, VEC4]:
        add([FuncSig("length", (vt,), SCALAR)])

    # 2-arg: min, max, pow, step, reflect, mod, atan2
    for fn in ["min", "max", "pow", "step", "reflect", "mod"]:
        add(_float_overloads_2(fn))

    # atan2(y, x) — 2-arg overload of atan
    add(_float_overloads_2("atan"))

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

    # refract(I, N, eta) — I and N are vecN, eta is scalar, returns vecN
    for vt in [SCALAR, VEC2, VEC3, VEC4]:
        add([FuncSig("refract", (vt, vt, SCALAR), vt)])

    # faceforward(N, I, Nref) — returns N flipped to face Nref relative to I
    for vt in [SCALAR, VEC2, VEC3, VEC4]:
        add([FuncSig("faceforward", (vt, vt, vt), vt)])

    # --- Additional GLSL.std.450 math functions ---
    # 1-arg: round, trunc, radians, degrees, sinh, cosh, tanh, asinh, acosh, atanh
    for fn in ["round", "trunc", "radians", "degrees",
               "sinh", "cosh", "tanh", "asinh", "acosh", "atanh"]:
        add(_float_overloads_1(fn))

    # --- Matrix functions ---
    # determinant(matN) -> scalar
    for mt in [MAT2, MAT3, MAT4]:
        add([FuncSig("determinant", (mt,), SCALAR)])

    # inverse(matN) -> matN
    for mt in [MAT2, MAT3, MAT4]:
        add([FuncSig("inverse", (mt,), mt)])

    # transpose(matN) -> matN
    for mt in [MAT2, MAT3, MAT4]:
        add([FuncSig("transpose", (mt,), mt)])

    # texture sampling
    add([FuncSig("sample", (SAMPLER2D, VEC2), VEC4)])
    add([FuncSig("sample", (SAMPLER_CUBE, VEC3), VEC4)])

    # texture sampling with explicit LOD
    add([FuncSig("sample_lod", (SAMPLER2D, VEC2, SCALAR), VEC4)])
    add([FuncSig("sample_lod", (SAMPLER_CUBE, VEC3, SCALAR), VEC4)])

    # --- RT instructions ---
    # trace_ray(accel, ray_flags, cull_mask, sbt_offset, sbt_stride, miss_index,
    #           origin, tmin, direction, tmax, payload_loc)
    add([FuncSig("trace_ray", (
        ACCELERATION_STRUCTURE, UINT, UINT, UINT, UINT, UINT,
        VEC3, SCALAR, VEC3, SCALAR, INT,
    ), VOID)])

    # report_intersection(hit_t, hit_kind) -> bool
    add([FuncSig("report_intersection", (SCALAR, UINT), BOOL)])

    # execute_callable(sbt_index, callable_data_loc) -> void
    add([FuncSig("execute_callable", (UINT, INT), VOID)])

    # ignore_intersection() -> void
    add([FuncSig("ignore_intersection", (), VOID)])

    # terminate_ray() -> void
    add([FuncSig("terminate_ray", (), VOID)])

    # image_store(image, coord, value) -> void
    add([FuncSig("image_store", (STORAGE_IMAGE, IVEC2, VEC4), VOID)])
    add([FuncSig("image_store", (STORAGE_IMAGE, VEC2, VEC4), VOID)])
    add([FuncSig("image_store", (STORAGE_IMAGE, UVEC2, VEC4), VOID)])

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
    if param.name == arg.name:
        return True
    # Allow scalar literals to match int/uint parameters (numeric promotion)
    if arg.name == "scalar" and param.name in ("int", "uint"):
        return True
    return False
