"""Built-in function signatures for Lux."""

from __future__ import annotations
from dataclasses import dataclass
from luxc.builtins.types import (
    LuxType, SCALAR, INT, UINT, BOOL, VOID,
    VEC2, VEC3, VEC4, MAT2, MAT3, MAT4, SAMPLER2D, SAMPLER_CUBE,
    SAMPLER_2D_ARRAY, SAMPLER_CUBE_ARRAY,
    ACCELERATION_STRUCTURE, STORAGE_IMAGE, BINDLESS_TEXTURE_ARRAY, IVEC2, UVEC2,
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

    # --- NaN/Inf detection ---
    # any_nan(vecN) -> bool, any_inf(vecN) -> bool
    for vt in [SCALAR, VEC2, VEC3, VEC4]:
        add([FuncSig("any_nan", (vt,), BOOL)])
        add([FuncSig("any_inf", (vt,), BOOL)])

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

    # bindless texture sampling (array, index, uv)
    add([FuncSig("sample_bindless", (BINDLESS_TEXTURE_ARRAY, INT, VEC2), VEC4)])
    add([FuncSig("sample_bindless", (BINDLESS_TEXTURE_ARRAY, UINT, VEC2), VEC4)])
    add([FuncSig("sample_bindless", (BINDLESS_TEXTURE_ARRAY, SCALAR, VEC2), VEC4)])
    # bindless texture sampling with explicit LOD (array, index, uv, lod)
    add([FuncSig("sample_bindless_lod", (BINDLESS_TEXTURE_ARRAY, INT, VEC2, SCALAR), VEC4)])
    add([FuncSig("sample_bindless_lod", (BINDLESS_TEXTURE_ARRAY, UINT, VEC2, SCALAR), VEC4)])
    add([FuncSig("sample_bindless_lod", (BINDLESS_TEXTURE_ARRAY, SCALAR, VEC2, SCALAR), VEC4)])

    # Shadow comparison sampling: sample_compare(shadow_tex, vec3(uv, layer), depth_ref) -> scalar
    add([FuncSig("sample_compare", (SAMPLER_2D_ARRAY, VEC3, SCALAR), SCALAR)])

    # Array texture sampling (raw depth fetch): sample_array(tex, uv, layer) -> vec4
    add([FuncSig("sample_array", (SAMPLER_2D_ARRAY, VEC2, SCALAR), VEC4)])

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

    # --- Mesh shader instructions ---
    # set_mesh_outputs(vert_count, prim_count) -> void
    add([FuncSig("set_mesh_outputs", (UINT, UINT), VOID)])

    # emit_mesh_tasks(gx, gy, gz) -> void
    add([FuncSig("emit_mesh_tasks", (UINT, UINT, UINT), VOID)])

    # --- Compute shader instructions ---
    # barrier() -> void (workgroup execution + memory barrier)
    add([FuncSig("barrier", (), VOID)])

    # --- Atomic operations ---
    # atomic_add(uint, uint) -> uint / (int, int) -> int
    add([FuncSig("atomic_add", (UINT, UINT), UINT)])
    add([FuncSig("atomic_add", (INT, INT), INT)])

    # atomic_min/max: uint and int overloads
    add([FuncSig("atomic_min", (UINT, UINT), UINT)])
    add([FuncSig("atomic_min", (INT, INT), INT)])
    add([FuncSig("atomic_max", (UINT, UINT), UINT)])
    add([FuncSig("atomic_max", (INT, INT), INT)])

    # atomic_and/or/xor: uint only
    add([FuncSig("atomic_and", (UINT, UINT), UINT)])
    add([FuncSig("atomic_or", (UINT, UINT), UINT)])
    add([FuncSig("atomic_xor", (UINT, UINT), UINT)])

    # atomic_exchange: uint and int
    add([FuncSig("atomic_exchange", (UINT, UINT), UINT)])
    add([FuncSig("atomic_exchange", (INT, INT), INT)])

    # atomic_compare_exchange(ref, comparator, value) -> old_value
    add([FuncSig("atomic_compare_exchange", (UINT, UINT, UINT), UINT)])
    add([FuncSig("atomic_compare_exchange", (INT, INT, INT), INT)])

    # atomic_load(ref) -> value
    add([FuncSig("atomic_load", (UINT,), UINT)])
    add([FuncSig("atomic_load", (INT,), INT)])

    # atomic_store(ref, value) -> void
    add([FuncSig("atomic_store", (UINT, UINT), VOID)])
    add([FuncSig("atomic_store", (INT, INT), VOID)])

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
    # Allow SemanticType to match its base type for builtin functions
    from luxc.builtins.types import SemanticType
    if isinstance(arg, SemanticType) and param.name == arg.base_type.name:
        return True
    return False
