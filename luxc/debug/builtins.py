"""Built-in function implementations for the Lux shader debugger."""

from __future__ import annotations

import math
from luxc.debug.values import LuxScalar, LuxVec, LuxMat, LuxInt, LuxBool, LuxValue


def _to_float(v: LuxValue) -> float:
    if isinstance(v, LuxScalar):
        return v.value
    if isinstance(v, LuxInt):
        return float(v.value)
    raise TypeError(f"Expected scalar, got {type(v).__name__}")


def _to_floats(v: LuxValue) -> list[float]:
    if isinstance(v, LuxVec):
        return v.components
    if isinstance(v, LuxScalar):
        return [v.value]
    if isinstance(v, LuxInt):
        return [float(v.value)]
    raise TypeError(f"Expected vec or scalar, got {type(v).__name__}")


def _apply_component_wise_1(fn, a: LuxValue) -> LuxValue:
    """Apply a 1-arg math function component-wise."""
    if isinstance(a, LuxScalar):
        return LuxScalar(fn(a.value))
    if isinstance(a, LuxVec):
        return LuxVec([fn(c) for c in a.components])
    raise TypeError(f"Cannot apply math to {type(a).__name__}")


def _apply_component_wise_2(fn, a: LuxValue, b: LuxValue) -> LuxValue:
    """Apply a 2-arg math function component-wise."""
    if isinstance(a, LuxScalar) and isinstance(b, LuxScalar):
        return LuxScalar(fn(a.value, b.value))
    if isinstance(a, LuxVec) and isinstance(b, LuxVec):
        return LuxVec([fn(x, y) for x, y in zip(a.components, b.components)])
    if isinstance(a, LuxVec) and isinstance(b, LuxScalar):
        return LuxVec([fn(x, b.value) for x in a.components])
    if isinstance(a, LuxScalar) and isinstance(b, LuxVec):
        return LuxVec([fn(a.value, y) for y in b.components])
    raise TypeError(f"Cannot apply math to {type(a).__name__} and {type(b).__name__}")


def _apply_component_wise_3(fn, a: LuxValue, b: LuxValue, c: LuxValue) -> LuxValue:
    """Apply a 3-arg math function component-wise."""
    af = _to_floats(a)
    bf = _to_floats(b)
    cf = _to_floats(c)
    size = max(len(af), len(bf), len(cf))
    if len(af) == 1:
        af = af * size
    if len(bf) == 1:
        bf = bf * size
    if len(cf) == 1:
        cf = cf * size
    result = [fn(af[i], bf[i], cf[i]) for i in range(size)]
    if size == 1:
        return LuxScalar(result[0])
    return LuxVec(result)


def _safe_div(a: float, b: float) -> float:
    if b == 0.0:
        return float('inf') if a >= 0.0 else float('-inf')
    return a / b


# --- Math builtins ---

def builtin_sin(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.sin, args[0])

def builtin_cos(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.cos, args[0])

def builtin_tan(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.tan, args[0])

def builtin_asin(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.asin, args[0])

def builtin_acos(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.acos, args[0])

def builtin_atan(args: list[LuxValue]) -> LuxValue:
    if len(args) == 2:
        return _apply_component_wise_2(math.atan2, args[0], args[1])
    return _apply_component_wise_1(math.atan, args[0])

def builtin_exp(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.exp, args[0])

def builtin_exp2(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(lambda x: 2.0 ** x, args[0])

def builtin_log(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(lambda x: math.log(x) if x > 0 else float('nan'), args[0])

def builtin_log2(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(lambda x: math.log2(x) if x > 0 else float('nan'), args[0])

def builtin_sqrt(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(lambda x: math.sqrt(x) if x >= 0 else float('nan'), args[0])

def builtin_inversesqrt(args: list[LuxValue]) -> LuxValue:
    def isqrt(x: float) -> float:
        if x <= 0:
            return float('nan')
        return 1.0 / math.sqrt(x)
    return _apply_component_wise_1(isqrt, args[0])

def builtin_abs(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(abs, args[0])

def builtin_sign(args: list[LuxValue]) -> LuxValue:
    def sign(x: float) -> float:
        if x > 0:
            return 1.0
        if x < 0:
            return -1.0
        return 0.0
    return _apply_component_wise_1(sign, args[0])

def builtin_floor(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.floor, args[0])

def builtin_ceil(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.ceil, args[0])

def builtin_fract(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(lambda x: x - math.floor(x), args[0])

def builtin_round(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(round, args[0])

def builtin_trunc(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.trunc, args[0])

def builtin_radians(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.radians, args[0])

def builtin_degrees(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.degrees, args[0])

def builtin_sinh(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.sinh, args[0])

def builtin_cosh(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.cosh, args[0])

def builtin_tanh(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.tanh, args[0])

def builtin_asinh(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.asinh, args[0])

def builtin_acosh(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.acosh, args[0])

def builtin_atanh(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_1(math.atanh, args[0])

def builtin_min(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_2(min, args[0], args[1])

def builtin_max(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_2(max, args[0], args[1])

def builtin_pow(args: list[LuxValue]) -> LuxValue:
    return _apply_component_wise_2(pow, args[0], args[1])

def builtin_step(args: list[LuxValue]) -> LuxValue:
    def step(edge: float, x: float) -> float:
        return 1.0 if x >= edge else 0.0
    return _apply_component_wise_2(step, args[0], args[1])

def builtin_clamp(args: list[LuxValue]) -> LuxValue:
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))
    return _apply_component_wise_3(clamp, args[0], args[1], args[2])

def builtin_mix(args: list[LuxValue]) -> LuxValue:
    def mix(x: float, y: float, a: float) -> float:
        return x * (1.0 - a) + y * a
    return _apply_component_wise_3(mix, args[0], args[1], args[2])

def builtin_smoothstep(args: list[LuxValue]) -> LuxValue:
    def smoothstep(edge0: float, edge1: float, x: float) -> float:
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0))) if edge1 != edge0 else 0.0
        return t * t * (3.0 - 2.0 * t)
    return _apply_component_wise_3(smoothstep, args[0], args[1], args[2])

def builtin_fma(args: list[LuxValue]) -> LuxValue:
    def fma(a: float, b: float, c: float) -> float:
        return a * b + c
    return _apply_component_wise_3(fma, args[0], args[1], args[2])


# --- Vector builtins ---

def builtin_length(args: list[LuxValue]) -> LuxValue:
    v = _to_floats(args[0])
    return LuxScalar(math.sqrt(sum(c * c for c in v)))

def builtin_distance(args: list[LuxValue]) -> LuxValue:
    a = _to_floats(args[0])
    b = _to_floats(args[1])
    return LuxScalar(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))

def builtin_normalize(args: list[LuxValue]) -> LuxValue:
    v = _to_floats(args[0])
    ln = math.sqrt(sum(c * c for c in v))
    if ln == 0.0:
        return LuxVec([float('nan')] * len(v)) if len(v) > 1 else LuxScalar(float('nan'))
    result = [c / ln for c in v]
    if len(result) == 1:
        return LuxScalar(result[0])
    return LuxVec(result)

def builtin_dot(args: list[LuxValue]) -> LuxValue:
    a = _to_floats(args[0])
    b = _to_floats(args[1])
    return LuxScalar(sum(x * y for x, y in zip(a, b)))

def builtin_cross(args: list[LuxValue]) -> LuxValue:
    a = _to_floats(args[0])
    b = _to_floats(args[1])
    return LuxVec([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])

def builtin_reflect(args: list[LuxValue]) -> LuxValue:
    i = _to_floats(args[0])
    n = _to_floats(args[1])
    d = sum(x * y for x, y in zip(n, i))
    return LuxVec([iv - 2.0 * d * nv for iv, nv in zip(i, n)])

def builtin_refract(args: list[LuxValue]) -> LuxValue:
    i = _to_floats(args[0])
    n = _to_floats(args[1])
    eta = _to_float(args[2])
    d = sum(x * y for x, y in zip(n, i))
    k = 1.0 - eta * eta * (1.0 - d * d)
    if k < 0.0:
        return LuxVec([0.0] * len(i))
    sk = math.sqrt(k)
    return LuxVec([eta * iv - (eta * d + sk) * nv for iv, nv in zip(i, n)])

def builtin_faceforward(args: list[LuxValue]) -> LuxValue:
    n = _to_floats(args[0])
    i = _to_floats(args[1])
    nref = _to_floats(args[2])
    d = sum(x * y for x, y in zip(nref, i))
    if d < 0.0:
        return LuxVec(list(n))
    return LuxVec([-c for c in n])


# --- Matrix builtins ---

def builtin_determinant(args: list[LuxValue]) -> LuxValue:
    if not isinstance(args[0], LuxMat):
        raise TypeError("determinant requires a matrix")
    m = args[0].columns
    n = len(m)
    if n == 2:
        return LuxScalar(m[0][0] * m[1][1] - m[1][0] * m[0][1])
    if n == 3:
        return LuxScalar(
            m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
            - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
            + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2])
        )
    if n == 4:
        # Cofactor expansion
        def sub(r: int, c: int) -> float:
            rows = [i for i in range(4) if i != r]
            cols = [j for j in range(4) if j != c]
            s = [[m[cols[ci]][rows[ri]] for ci in range(3)] for ri in range(3)]
            return (
                s[0][0] * (s[1][1] * s[2][2] - s[2][1] * s[1][2])
                - s[0][1] * (s[1][0] * s[2][2] - s[2][0] * s[1][2])
                + s[0][2] * (s[1][0] * s[2][1] - s[2][0] * s[1][1])
            )
        det = sum((-1) ** c * m[c][0] * sub(0, c) for c in range(4))
        return LuxScalar(det)
    raise ValueError(f"Unsupported matrix size: {n}")


def builtin_inverse(args: list[LuxValue]) -> LuxValue:
    if not isinstance(args[0], LuxMat):
        raise TypeError("inverse requires a matrix")
    m = args[0].columns
    n = len(m)
    if n == 2:
        det = m[0][0] * m[1][1] - m[1][0] * m[0][1]
        if det == 0.0:
            return LuxMat([[float('nan')] * 2 for _ in range(2)])
        inv_det = 1.0 / det
        return LuxMat([
            [m[1][1] * inv_det, -m[0][1] * inv_det],
            [-m[1][0] * inv_det, m[0][0] * inv_det],
        ])
    # For 3x3 and 4x4, use simple implementations
    # Convert column-major to row-major for easier manipulation
    rows = [[m[c][r] for c in range(n)] for r in range(n)]
    # Augmented matrix [A|I]
    aug = [row + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(rows)]
    # Gauss-Jordan elimination
    for col in range(n):
        # Find pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-30:
            return LuxMat([[float('nan')] * n for _ in range(n)])
        for j in range(2 * n):
            aug[col][j] /= pivot
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]
    # Extract inverse (row-major) and convert to column-major
    inv_rows = [aug[r][n:] for r in range(n)]
    return LuxMat([[inv_rows[r][c] for r in range(n)] for c in range(n)])


# --- Special builtins ---

def builtin_sample(args: list[LuxValue]) -> LuxValue:
    """Texture sample — uses real image data if available, else neutral grey."""
    from luxc.debug.values import LuxImage
    tex = args[0]
    if isinstance(tex, LuxImage) and len(args) >= 2:
        uv = args[1]
        if isinstance(uv, LuxVec) and uv.size >= 2:
            return tex.sample_bilinear(uv.components[0], uv.components[1])
    return LuxVec([0.8, 0.8, 0.8, 1.0])

def builtin_sample_bindless(args: list[LuxValue]) -> LuxValue:
    """Bindless texture array sample: sample_bindless(textures, index, uv)."""
    from luxc.debug.values import LuxImage
    tex_array, idx_val, uv_val = args[0], args[1], args[2]
    idx = int(_to_float(idx_val))
    if isinstance(tex_array, list) and 0 <= idx < len(tex_array):
        tex = tex_array[idx]
        if isinstance(tex, LuxImage) and isinstance(uv_val, LuxVec) and uv_val.size >= 2:
            return tex.sample_bilinear(uv_val.components[0], uv_val.components[1])
    return LuxVec([0.8, 0.8, 0.8, 1.0])

def builtin_sample_bindless_lod(args: list[LuxValue]) -> LuxValue:
    """Bindless texture array sample with LOD: sample_bindless_lod(textures, index, uv, lod)."""
    from luxc.debug.values import LuxImage
    tex_array, idx_val, uv_val = args[0], args[1], args[2]
    # LOD is ignored in CPU debugger (always mip 0)
    idx = int(_to_float(idx_val))
    if isinstance(tex_array, list) and 0 <= idx < len(tex_array):
        tex = tex_array[idx]
        if isinstance(tex, LuxImage) and isinstance(uv_val, LuxVec) and uv_val.size >= 2:
            return tex.sample_bilinear(uv_val.components[0], uv_val.components[1])
    return LuxVec([0.8, 0.8, 0.8, 1.0])

def builtin_sample_lod(args: list[LuxValue]) -> LuxValue:
    """Texture sample with LOD — uses real image data if available."""
    from luxc.debug.values import LuxImage
    tex = args[0]
    uv_or_dir = args[1] if len(args) >= 2 else None
    # LOD is ignored in CPU debugger
    if isinstance(tex, LuxImage) and isinstance(uv_or_dir, LuxVec) and uv_or_dir.size >= 2:
        return tex.sample_bilinear(uv_or_dir.components[0], uv_or_dir.components[1])
    return LuxVec([0.8, 0.8, 0.8, 1.0])

def builtin_image_size(args: list[LuxValue]) -> LuxValue:
    """Mock image size — returns 1024x1024."""
    return LuxVec([1024.0, 1024.0])

def builtin_texture_size(args: list[LuxValue]) -> LuxValue:
    """Mock texture size — returns 1024x1024."""
    return LuxVec([1024.0, 1024.0])

def builtin_texture_levels(args: list[LuxValue]) -> LuxValue:
    """Mock texture levels — returns 10."""
    return LuxInt(10)


# --- Registry ---

BUILTIN_FUNCTIONS: dict[str, callable] = {
    "sin": builtin_sin,
    "cos": builtin_cos,
    "tan": builtin_tan,
    "asin": builtin_asin,
    "acos": builtin_acos,
    "atan": builtin_atan,
    "exp": builtin_exp,
    "exp2": builtin_exp2,
    "log": builtin_log,
    "log2": builtin_log2,
    "sqrt": builtin_sqrt,
    "inversesqrt": builtin_inversesqrt,
    "abs": builtin_abs,
    "sign": builtin_sign,
    "floor": builtin_floor,
    "ceil": builtin_ceil,
    "fract": builtin_fract,
    "round": builtin_round,
    "trunc": builtin_trunc,
    "radians": builtin_radians,
    "degrees": builtin_degrees,
    "sinh": builtin_sinh,
    "cosh": builtin_cosh,
    "tanh": builtin_tanh,
    "asinh": builtin_asinh,
    "acosh": builtin_acosh,
    "atanh": builtin_atanh,
    "min": builtin_min,
    "max": builtin_max,
    "pow": builtin_pow,
    "step": builtin_step,
    "clamp": builtin_clamp,
    "mix": builtin_mix,
    "smoothstep": builtin_smoothstep,
    "fma": builtin_fma,
    "length": builtin_length,
    "distance": builtin_distance,
    "normalize": builtin_normalize,
    "dot": builtin_dot,
    "cross": builtin_cross,
    "reflect": builtin_reflect,
    "refract": builtin_refract,
    "faceforward": builtin_faceforward,
    "determinant": builtin_determinant,
    "inverse": builtin_inverse,
    "sample": builtin_sample,
    "sample_bindless": builtin_sample_bindless,
    "sample_bindless_lod": builtin_sample_bindless_lod,
    "sample_lod": builtin_sample_lod,
    "image_size": builtin_image_size,
    "texture_size": builtin_texture_size,
    "texture_levels": builtin_texture_levels,
}
