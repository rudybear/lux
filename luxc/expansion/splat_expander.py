"""Splat declaration expander for Gaussian splatting.

Takes a `splat` declaration (Gaussian splatting configuration) and a
`pipeline` declaration with `mode: gaussian_splat`, then generates
three shader stages:

  1. **Preprocess compute shader** -- transforms each Gaussian splat from
     world space to screen space, computes 2D covariance (conic), evaluates
     spherical harmonics for view-dependent color, and writes sort keys.

  2. **Render vertex shader** -- reads sorted splat indices, expands each
     visible splat into a screen-aligned quad (6 vertices, 2 triangles),
     and passes conic/color/center/offset to the fragment stage.

  3. **Render fragment shader** -- evaluates the 2D Gaussian weight for
     each fragment, applies alpha cutoff, and outputs premultiplied-alpha
     color.

The module follows the same patterns as ``surface_expander.py``.
"""

from __future__ import annotations

from luxc.parser.ast_nodes import (
    Module, StageBlock, VarDecl, UniformBlock, PushBlock, BlockField,
    SamplerDecl,
    FunctionDef, Param, LetStmt, AssignStmt, ReturnStmt, ExprStmt,
    NumberLit, BoolLit, VarRef, BinaryOp, CallExpr, ConstructorExpr,
    SwizzleAccess, UnaryOp,
    AssignTarget, IndexAccess, FieldAccess,
    StorageBufferDecl, IfStmt, DiscardStmt, ForStmt, BreakStmt,
    SplatDecl, SplatMember,
    RayPayloadDecl, HitAttributeDecl, AccelDecl, StorageImageDecl,
)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _get_splat_config(splat: SplatDecl) -> dict:
    """Extract configuration values from a SplatDecl's members.

    Supported members:
        sh_degree   -- spherical harmonics degree (0, 1, 2, or 3)
        kernel      -- splat kernel type ("ellipse" or "circle")
        color_space -- output color space ("srgb" or "linear")
        sort        -- sort strategy: "camera_distance" (default; true
                       Euclidean distance from the camera, matching the
                       ratified KHR_gaussian_splatting spec's
                       `sortingMethod: cameraDistance`) or "view_depth"
                       (raw view-space z, matching gsplat's own convention
                       -- added so the two can be measured against each
                       other; see docs/lux-4d-spec.md follow-up).
        alpha_cutoff -- (legacy name, still accepted) minimum alpha for
                       fragment discard; see `alpha_min` below, which is
                       the same runtime threshold under the reference-
                       matching name.
        dilation    -- scalar (default 0.3): added to the diagonal of the
                       projected 2D covariance before inversion (3DGS/
                       gsplat's "eps2d" antialiasing convention), with NO
                       opacity compensation for the resulting blur -- this
                       matches the reference 3DGS rasterizer / gsplat
                       exactly (earlier lux versions additionally applied a
                       `sqrt(det_orig/det)` opacity compensation gsplat's
                       default AA mode does not use).
        alpha_min   -- scalar (default 1/255 ~= 0.00392): fragments with
                       `alpha = opacity * exp(-0.5*d^2) < alpha_min` are
                       discarded (3σ quad extent, alpha clamped to <= 0.99
                       before compositing), matching 3DGS/gsplat exactly.
        motion      -- "none" (default) or "keyframes": when "keyframes", the
                       compiler emits an additional morph-apply compute stage
                       (see _build_morph_apply_stage) that blends per-keyframe
                       sparse position/rotation/SH0 deltas into the working
                       splat_pos/splat_rot/splat_sh0 buffers before preprocess
                       runs. See docs/language-reference.md's "Dynamic
                       (4D) Gaussian Splatting" section.
        motion_vectors -- bool (default False): when true, the preprocess
                       stage also projects each splat's *previous* frame
                       position with the *previous* frame's (unjittered)
                       view-projection and emits a `projected_mv` buffer;
                       the render stages carry it through into a shared
                       `out_aux` attachment's .x/.y lanes (see
                       expected_depth below for the full packed layout).
                       See docs/lux-4d-spec.md section 3 ("DLSS input
                       contract outputs") and docs/language-reference.md.
        expected_depth -- bool (default False): when true, the preprocess
                       stage emits a `projected_depth` buffer (camera-space
                       z), matching gsplat's "ED" (expected depth) mode.
                       When either motion_vectors or expected_depth is set,
                       the render stages emit ONE additional attachment
                       `out_aux` (RGBA32F, premultiplied-alpha-blended with
                       the SAME (ONE, ONE_MINUS_SRC_ALPHA) equation as
                       out_color): `(mv.x*alpha, mv.y*alpha, depth*alpha,
                       alpha)` -- lanes for a disabled feature are written
                       0.0. `.w` carries the GENUINE per-fragment alpha,
                       not a packed data value: Vulkan's fixed-function
                       SRC_ALPHA/ONE_MINUS_SRC_ALPHA blend factors read
                       "source alpha" from THIS attachment's own 4th
                       component specifically (not a shared/global value),
                       so whatever occupies `.w` directly controls how
                       every overlapping fragment decays into this
                       attachment -- an earlier version of this packing
                       put `fg*alpha` there instead (reasoning the host
                       could un-premultiply from `out_color`'s alpha
                       instead), which is a genuinely different, GPU-side
                       correctness bug, not a precision tradeoff: it made
                       `ONE_MINUS_SRC_ALPHA` evaluate to ~1 (no decay) for
                       every background/non-actor fragment, producing an
                       unbounded running SUM instead of a proper "over"
                       composite (measured 10-50x MV/depth blowups in
                       heavily-overdrawn regions). Vs. the earlier
                       two-attachment (`out_motion`/`out_depth`, each
                       RGBA32F with an always-0 `.z`) design, `out_aux` now
                       packs 3 real values with zero wasted lanes:
                       attachment count 2->1 and bytes 32B->16B whenever
                       foreground_coverage is off.
        foreground_coverage -- bool (default False): when true, the
                       preprocess stage reads a per-splat `splat_foreground`
                       input (0.0 or 1.0 -- see docs/rendering-engines.md's
                       `_FOREGROUND` attribute / morph-target fallback) and
                       the render stages composite it into a SECOND
                       attachment `out_fg` (`x = fg*alpha, y = 0, z = 0,
                       w = alpha` -- same genuine-alpha-in-`.w` requirement
                       as `out_aux` above; `out_fg` can't share `out_aux`
                       because `out_aux` already uses all 3 non-alpha lanes
                       for mv.xy/depth). Implies expected_depth: true
                       (auto-enabled if not already set on the same splat
                       block). The host divides `out_fg.x` by `out_fg.w`
                       (or, equivalently, by `out_color`'s alpha -- both are
                       the same per-fragment alpha) to recover the
                       per-pixel foreground/actor coverage fraction.
    """
    config = {
        "sh_degree": 0,
        "kernel": "ellipse",
        "color_space": "srgb",
        "sort": "camera_distance",
        "alpha_cutoff": 1.0 / 255.0,
        "dilation": 0.3,
        "alpha_min": 1.0 / 255.0,
        "motion": "none",
        "motion_vectors": False,
        "expected_depth": False,
        "foreground_coverage": False,
        # Host-side format hint for out_aux (bench/lux_perf_ablation.md
        # task 2 follow-up), reflected into the compute stage's
        # gaussian_splatting JSON section for the host to read (see
        # splat_renderer.h's getAuxFormat() comment) -- doesn't affect
        # generated SPIR-V at all, out_aux stays a plain vec4 fragment
        # output; only which VkFormat/MTLPixelFormat the HOST creates the
        # attachment as changes. "half" (DEFAULT, since the out_fg fix
        # below): out_aux RGBA16F, 8B/px -- pixel-level, out_aux's
        # half-float blend-accumulation rounding measures MV median error
        # 0.012-0.016px and depth relative median error ~5.4e-3 vs. the
        # "float" baseline (see bench/lux_perf_ablation.md), which fails
        # this repo's original tight pixel-level bars (<=0.01px /
        # <=1e-3-relative, tests/test_dlss_outputs.py's pre-existing
        # tolerances). But the MODEL-LEVEL verdict (mobiledlss's actual
        # DLSS reconstructor, trained/evaluated on lux-rendered clips)
        # is what actually matters for this option's default, and there
        # "half" is indistinguishable from "float": 35.494 vs 35.491 dB
        # PSNR on lux-rendered clips, fg bit-exact, mv 0.008px, depth 0.4%
        # -- all invisible to the reconstructor, which only ever consumes
        # `out_aux`/`out_fg` through further lossy stages (bilinear
        # resampling, a learned network) with far more slack than a
        # pixel-exact comparison allows. "float": out_aux RGBA32F, 16B/px
        # -- the original precision-safe design from task 2, still
        # selectable (examples/gaussian_splat_dlss_half.lux's inverse,
        # i.e. any splat block can set `aux_precision: float` explicitly)
        # for callers that need the tighter pixel-level bar, e.g. a
        # non-network downstream consumer that reads out_aux directly.
        #
        # NOTE: out_fg (only present when foreground_coverage is also on)
        # is DELIBERATELY NOT covered by this option -- it stays RGBA16F
        # in both "float" and "half", regardless of which is the default.
        # An earlier version of this option also shrank out_fg to RG16F
        # under "half" (12B/px total vs. this version's 16B/px), which
        # measurably broke the blend: Vulkan/Metal's fixed-function
        # SRC_ALPHA/ONE_MINUS_SRC_ALPHA blend reads "source alpha" from
        # EACH blended attachment's own 4th component, and RG16F has no
        # 4th component for it to read there, so the blend silently used
        # a constant alpha instead -- a full 0-to-1 flip on ~1% of pixels,
        # a real correctness bug, not a precision tradeoff (unlike
        # out_aux's "half", which keeps a genuine `.w` in all 4 channels
        # and only has ordinary half-float rounding error). That fix is
        # what makes flipping the default to "half" safe at all -- with
        # the old RG16F out_fg, `fg` would have been unusable (a full
        # 0-to-1 flip on ~1% of pixels, not just imprecise) for every
        # caller of the default pipeline. See splat_renderer.h's
        # getFgFormat() comment for the full story.
        "aux_precision": "half",
    }
    for m in splat.members:
        if m.name == "sh_degree":
            config["sh_degree"] = int(m.value.value)
        elif m.name in ("kernel", "color_space", "sort", "motion", "aux_precision"):
            config[m.name] = m.value.name
        elif m.name == "alpha_cutoff":
            # Legacy name; kept as an alias of alpha_min for backward
            # compatibility with existing .lux sources.
            config["alpha_cutoff"] = float(m.value.value)
            config["alpha_min"] = float(m.value.value)
        elif m.name == "alpha_min":
            config["alpha_min"] = float(m.value.value)
            config["alpha_cutoff"] = float(m.value.value)
        elif m.name == "dilation":
            config["dilation"] = float(m.value.value)
        elif m.name in ("motion_vectors", "expected_depth", "foreground_coverage"):
            if isinstance(m.value, BoolLit):
                config[m.name] = bool(m.value.value)
            else:
                # Accept true/false spelled as a bare identifier too, for
                # symmetry with the other enum-like members.
                config[m.name] = getattr(m.value, "name", "") == "true"
    if config["foreground_coverage"]:
        # Packed into out_depth's .g channel -- nowhere to write it without
        # the depth attachment also being enabled.
        config["expected_depth"] = True
    return config


# ---------------------------------------------------------------------------
# AST construction helpers
# ---------------------------------------------------------------------------

def _lit(v) -> NumberLit:
    """Create a NumberLit from a Python number or string."""
    return NumberLit(str(v))


def _uint_lit(v) -> NumberLit:
    """Create a uint NumberLit."""
    n = NumberLit(str(v))
    n.resolved_type = "uint"
    return n


def _ref(name: str) -> VarRef:
    return VarRef(name)


def _field(obj: str, name: str) -> FieldAccess:
    return FieldAccess(_ref(obj), name)


def _push_field(name: str) -> VarRef:
    """Access a push constant field directly by name."""
    return _ref(name)


def _idx(buf: str, index) -> IndexAccess:
    """Buffer index access: buf[index]."""
    idx_expr = index if not isinstance(index, str) else _ref(index)
    return IndexAccess(_ref(buf), idx_expr)


def _call(fn: str, args: list) -> CallExpr:
    return CallExpr(_ref(fn), args)


def _binop(op: str, left, right) -> BinaryOp:
    return BinaryOp(op, left, right)


def _let(name: str, ty: str, expr) -> LetStmt:
    return LetStmt(name, ty, expr)


def _assign(name: str, expr) -> AssignStmt:
    return AssignStmt(AssignTarget(_ref(name)), expr)


def _assign_idx(buf: str, index, expr) -> AssignStmt:
    return AssignStmt(AssignTarget(IndexAccess(_ref(buf), index)), expr)


def _swizzle(expr, components: str) -> SwizzleAccess:
    return SwizzleAccess(expr, components)


def _ctor(ty: str, args: list) -> ConstructorExpr:
    return ConstructorExpr(ty, args)


def _neg(expr) -> UnaryOp:
    return UnaryOp("-", expr)


def _if(cond, then_body: list, else_body: list | None = None) -> IfStmt:
    return IfStmt(cond, then_body, else_body or [])


# ---------------------------------------------------------------------------
# SH coefficient buffer names per degree
# ---------------------------------------------------------------------------

def _sh_buffer_names(sh_degree: int) -> list[str]:
    """Return the list of SH coefficient storage-buffer names.

    Degree 0: sh0 (DC term -- 1 coefficient stored as vec4 rgb+pad)
    Degree 1: sh0, sh1, sh2, sh3  (1 + 3 = 4 coefficients)
    Degree 2: sh0 .. sh8           (1 + 3 + 5 = 9 coefficients)
    Degree 3: sh0 .. sh15          (1 + 3 + 5 + 7 = 16 coefficients)

    Each buffer stores one vec4 per splat (rgb + padding).
    """
    n_coeffs = {0: 1, 1: 4, 2: 9, 3: 16}
    count = n_coeffs.get(sh_degree, 1)
    return [f"splat_sh{i}" for i in range(count)]


# ---------------------------------------------------------------------------
# Stage 0 (optional): morph-apply compute shader (motion: keyframes)
# ---------------------------------------------------------------------------
#
# Runs *before* the preprocess stage when `motion: keyframes` is set on the
# splat declaration. Blends per-keyframe sparse position/rotation/SH0 deltas
# into the working `splat_pos` / `splat_rot` / `splat_sh0` buffers -- the
# exact buffer names the preprocess stage already reads, so preprocess itself
# needs zero changes.
#
# GPU buffer layout (populated by the host loader/renderer, not luxc):
#   splat_base_pos / splat_base_rot / splat_base_sh0  -- immutable base attrs
#   morph_index            (uint) -- concatenated per-*segment* sparse index list
#   morph_delta_pos_lo/hi  (vec4) -- position deltas for the segment's low/high keyframe
#   morph_delta_rot_lo/hi  (vec4) -- rotation deltas
#   morph_delta_sh0_lo/hi  (vec4) -- SH0 (color) deltas
#   splat_pos / splat_rot / splat_sh0 (vec4) -- working buffers (also preprocess's input)
#
# A "segment" is the interval between two adjacent animation keyframes; its
# low/high delta lists are the union of the two keyframes' sparse indices,
# precomputed once at load time so each gaussian index appears at most once
# per segment (no atomics, no multi-pass reset needed -- see
# docs/language-reference.md and the loader implementations for how the CPU
# side builds `morph_index`/`morph_delta_*` and picks `segment_offset` /
# `segment_count` / `weight_lo` / `weight_hi` each frame from the current
# animation time). For the general case of more than two simultaneously
# nonzero target weights (not produced by the reference LINEAR/one-hot
# sampler), hosts fall back to a CPU-side loop over all targets -- see the
# spec's "Generic path for >2 non-zero weights" note; this GPU stage always
# handles exactly the common (>=1, <=2 active keyframes) case.
def _build_morph_apply_stage(config: dict) -> StageBlock:  # noqa: ARG001 (config kept for symmetry/future use)
    stage = StageBlock(stage_type="compute")

    # --- Immutable base attributes (read-only) ---
    stage.storage_buffers.append(StorageBufferDecl("splat_base_pos", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("splat_base_rot", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("splat_base_sh0", "vec4"))

    # --- Per-segment sparse delta lists (read-only) ---
    stage.storage_buffers.append(StorageBufferDecl("morph_index", "uint"))
    stage.storage_buffers.append(StorageBufferDecl("morph_delta_pos_lo", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("morph_delta_rot_lo", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("morph_delta_sh0_lo", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("morph_delta_pos_hi", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("morph_delta_rot_hi", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("morph_delta_sh0_hi", "vec4"))

    # --- Working buffers (write) -- same names the preprocess stage reads ---
    stage.storage_buffers.append(StorageBufferDecl("splat_pos", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("splat_rot", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("splat_sh0", "vec4"))

    pc_fields = [
        BlockField("segment_offset", "uint"),
        BlockField("segment_count", "uint"),
        BlockField("weight_lo", "scalar"),
        BlockField("weight_hi", "scalar"),
    ]
    stage.push_constants.append(PushBlock("push", pc_fields))

    body = _build_morph_apply_body()
    main_fn = FunctionDef("main", [], None, body)
    main_fn.attributes = ["workgroup_size(256)"]
    stage.functions.append(main_fn)

    return stage


def _build_morph_apply_body() -> list:
    body = []

    # let gid = global_invocation_id.x;
    body.append(_let("gid", "uint", _swizzle(_ref("global_invocation_id"), "x")))
    # if (gid >= push.segment_count) { return; }
    body.append(_if(
        _binop(">=", _ref("gid"), _push_field("segment_count")),
        [ReturnStmt(None)],
    ))

    # let entry = push.segment_offset + gid;
    body.append(_let("entry", "uint", _binop("+", _push_field("segment_offset"), _ref("gid"))))
    # let idx = morph_index[entry];
    body.append(_let("idx", "uint", _idx("morph_index", _ref("entry"))))

    # Base attributes at this gaussian.
    body.append(_let("base_pos", "vec3", _swizzle(_idx("splat_base_pos", _ref("idx")), "xyz")))
    body.append(_let("base_rot", "vec4", _idx("splat_base_rot", _ref("idx"))))
    body.append(_let("base_sh0", "vec3", _swizzle(_idx("splat_base_sh0", _ref("idx")), "xyz")))

    wlo = _push_field("weight_lo")
    whi = _push_field("weight_hi")

    # dpos = weight_lo * delta_pos_lo[entry].xyz + weight_hi * delta_pos_hi[entry].xyz;
    body.append(_let("dpos", "vec3",
        _binop("+",
               _binop("*", wlo, _swizzle(_idx("morph_delta_pos_lo", _ref("entry")), "xyz")),
               _binop("*", whi, _swizzle(_idx("morph_delta_pos_hi", _ref("entry")), "xyz")))))
    # drot = weight_lo * delta_rot_lo[entry] + weight_hi * delta_rot_hi[entry];
    body.append(_let("drot", "vec4",
        _binop("+",
               _binop("*", wlo, _idx("morph_delta_rot_lo", _ref("entry"))),
               _binop("*", whi, _idx("morph_delta_rot_hi", _ref("entry"))))))
    # dsh0 = weight_lo * delta_sh0_lo[entry].xyz + weight_hi * delta_sh0_hi[entry].xyz;
    body.append(_let("dsh0", "vec3",
        _binop("+",
               _binop("*", wlo, _swizzle(_idx("morph_delta_sh0_lo", _ref("entry")), "xyz")),
               _binop("*", whi, _swizzle(_idx("morph_delta_sh0_hi", _ref("entry")), "xyz")))))

    body.append(_let("new_pos", "vec3", _binop("+", _ref("base_pos"), _ref("dpos"))))
    body.append(_let("new_rot_raw", "vec4", _binop("+", _ref("base_rot"), _ref("drot"))))
    # Renormalize the blended quaternion (lerp + renormalize, matching the
    # reference writer's "lerp_renormalize" rotationBlend convention).
    body.append(_let("rot_len", "scalar", _call("length", [_ref("new_rot_raw")])))
    body.append(_let("new_rot", "vec4", _binop("/", _ref("new_rot_raw"), _ref("rot_len"))))
    body.append(_let("new_sh0", "vec3", _binop("+", _ref("base_sh0"), _ref("dsh0"))))

    body.append(_assign_idx("splat_pos", _ref("idx"),
        _ctor("vec4", [_ref("new_pos"), _lit("0.0")])))
    body.append(_assign_idx("splat_rot", _ref("idx"), _ref("new_rot")))
    body.append(_assign_idx("splat_sh0", _ref("idx"),
        _ctor("vec4", [_ref("new_sh0"), _lit("0.0")])))

    return body


# ---------------------------------------------------------------------------
# Stage 1: preprocess compute shader
# ---------------------------------------------------------------------------

def _build_preprocess_stage(config: dict) -> StageBlock:
    """Generate the preprocess compute stage.

    This shader runs one invocation per Gaussian splat.  It reads
    raw splat attributes (position, rotation quaternion, log-scale,
    opacity, SH coefficients), transforms them into screen-space
    parameters, and writes the results to output buffers for the
    render pass.

    Output buffers:
        projected_center -- (x_ndc, y_ndc, depth, major_extent) -- the .w
                       component is the SAME opacity-tight, screen-size-
                       clamped extent along the covariance's LARGER
                       eigenvector that used to be the (axis-aligned quad)
                       radius; kept for frustum-culling margin / debug.
        projected_axes   -- (major.x, major.y, minor.x, minor.y): the two
                       half-axis vectors of the oriented screen-space
                       quad, each `t*sqrt(lambda)*eigenvector` along the
                       projected 2D covariance's own eigenvectors (see
                       "Oriented quads" below) -- replaces the old
                       axis-aligned `radius`-only quad.
        projected_conic  -- upper triangle of inverse 2D covariance
        projected_color  -- evaluated SH color + opacity
        sort_keys        -- depth value for radix sort
        visible_count    -- atomic counter (incremented per visible splat)

    Oriented quads (perf; bench/lux_perf_ablation.md in mobiledlss,
    "Variant A"): the old vertex-stage quad was an AXIS-ALIGNED square
    sized by the covariance's single larger eigenvalue (`radius =
    t*sqrt(lambda_max)`), even though most splats are anisotropic
    (elongated) -- wasting fill-rate/overdraw on the corners of the square
    that lie outside the actual elongated ellipse. This computes the 2D
    covariance's eigenVECTORS too (eigenvalues were already computed for
    `radius`) via the standard symmetric-2x2 rotation-angle formula --
    `theta = 0.5*atan2(2*cov01, cov00-cov11)` diagonalizes the matrix, so
    `e1=(cos theta, sin theta)` is the eigenvector for the LARGER
    eigenvalue (lambda_max) and `e2=(-sin theta, cos theta)` (perpendicular
    by construction) is the eigenvector for the smaller one (lambda_min;
    guaranteed > 0 here since `det = lambda_max*lambda_min` was already
    checked `> 0` by the degenerate-ellipse cull above) -- and emits
    `major = t*sqrt(lambda_max)*e1`, `minor = t*sqrt(lambda_min)*e2`
    (`t` = the existing opacity-tight extent) instead of the scalar
    radius. The vertex stage then offsets each corner by
    `quad_x*major + quad_y*minor` (an oriented rectangle exactly
    circumscribing the covariance ellipse at extent `t`) instead of
    `vec2(quad_x,quad_y)*radius` (an axis-aligned square). The FRAGMENT
    stage is unchanged -- it evaluates the same conic/Gaussian formula
    against `frag_offset` regardless of which shape produced it, so pixel
    output is bit-identical except for fragments the smaller oriented
    quad no longer rasterizes at all (which would have been discarded by
    the existing `alpha < alpha_min` check anyway, exactly like the
    opacity-tight-extent fix above). Measured ~2.4x area-weighted overdraw
    reduction on the reference scene (mobiledlss's ablation, using the
    real per-splat covariance/opacity data with the same screen-size
    clamp applied to both variants).
    """
    stage = StageBlock(stage_type="compute")

    # --- Input storage buffers (per-splat attributes) ---
    stage.storage_buffers.append(StorageBufferDecl("splat_pos", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("splat_rot", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("splat_scale", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("splat_opacity", "scalar"))

    sh_degree = config["sh_degree"]
    for sh_name in _sh_buffer_names(sh_degree):
        stage.storage_buffers.append(StorageBufferDecl(sh_name, "vec4"))

    # --- Output storage buffers ---
    stage.storage_buffers.append(StorageBufferDecl("projected_center", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("projected_axes", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("projected_conic", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("projected_color", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("sort_keys", "uint"))
    stage.storage_buffers.append(StorageBufferDecl("sorted_indices", "uint"))
    stage.storage_buffers.append(StorageBufferDecl("visible_count", "uint"))

    # --- DLSS input-contract outputs (docs/lux-4d-spec.md section 3) ---
    if config.get("motion_vectors"):
        # Previous frame's animated world-space position (host double-buffers
        # this: it's a copy of splat_pos from the start of the previous
        # render() call -- for static splats it may simply alias splat_pos
        # since positions never change).
        stage.storage_buffers.append(StorageBufferDecl("splat_prev_pos", "vec4"))
        # Backward motion vector in pixels: uv_curr - uv_prev.
        stage.storage_buffers.append(StorageBufferDecl("projected_mv", "vec2"))
        # Per-frame camera data needed only for the (jitter-free) motion-
        # vector projection: element [0] = proj_matrix_unjittered, element
        # [1] = prev_view_proj_unjittered (the previous frame's combined
        # unjittered view-projection, pre-multiplied host-side). A 2-element
        # storage buffer rather than push-constant fields -- see the "Why
        # not push constants" note below.
        stage.storage_buffers.append(StorageBufferDecl("prev_camera_mats", "mat4"))
    if config.get("expected_depth"):
        stage.storage_buffers.append(StorageBufferDecl("projected_depth", "scalar"))
    if config.get("foreground_coverage"):
        # Per-splat is-foreground flag (0.0/1.0), passed straight through
        # to a per-pixel output buffer of the same name/shape as
        # projected_depth -- see docs/rendering-engines.md for how the host
        # populates the INPUT splat_foreground buffer (glTF _FOREGROUND
        # attribute, else "has any morph-target delta", else 0).
        stage.storage_buffers.append(StorageBufferDecl("splat_foreground", "scalar"))
        stage.storage_buffers.append(StorageBufferDecl("projected_foreground", "scalar"))

    # --- Push constants ---
    #
    # Why not push constants for proj_matrix_unjittered/prev_view_proj_unjittered
    # (docs/rendering-engines.md "Android live-rendering demo" MV
    # investigation): the base 176-byte block below plus those two mat4
    # fields (128 more bytes) totalled 304 bytes -- within the 4096-byte
    # budget desktop/iOS GPUs report (MoltenVK on Apple Silicon), but OVER
    # the 256-byte maxPushConstantsSize a real Android GPU (Mali-G715)
    # actually reports. vkCreatePipelineLayout with size > the device's
    # maxPushConstantsSize is a spec violation (VUID-VkPushConstantRange-
    # size-00298); Mali's driver accepted it anyway (no validation layers
    # on-device) and silently served undefined/recycled bytes for the
    # out-of-range tail -- which happened to be most of
    # prev_view_proj_unjittered -- corrupting every motion vector by 3-6
    # orders of magnitude while leaving everything reading only the
    # in-range bytes (color, depth) correct. Moving both fields into the
    # prev_camera_mats storage buffer above keeps this block at a portable
    # 176 bytes on every backend, well under Mali's 256 (this base block was
    # already over the Vulkan spec's guaranteed *minimum* of 128 before this
    # change and isn't addressed here -- 256+ has not been observed to be an
    # issue on any target device so far). See splat_renderer.cpp's startup
    # maxPushConstantsSize check for a guardrail against this class of bug
    # recurring.
    pc_fields = [
        BlockField("view_matrix", "mat4"),
        BlockField("proj_matrix", "mat4"),
        BlockField("cam_pos", "vec3"),
        BlockField("screen_size", "vec2"),
        BlockField("total_splats", "uint"),
        BlockField("focal_x", "scalar"),
        BlockField("focal_y", "scalar"),
        BlockField("sh_degree", "int"),
    ]
    stage.push_constants.append(PushBlock("push", pc_fields))

    # --- Main function body ---
    body = _build_preprocess_body(config)
    main_fn = FunctionDef("main", [], None, body)
    main_fn.attributes = ["workgroup_size(256)"]
    stage.functions.append(main_fn)

    return stage


def _cull_writes(config: dict | None = None) -> list:
    """Write invisible markers for culled splats (zero radius + zero opacity).

    Without this, culled splats leave uninitialized GPU memory in the projected
    output buffers, causing flickering / garbage in the render pass.
    """
    zero4 = _ctor("vec4", [_lit("0.0"), _lit("0.0"), _lit("0.0"), _lit("0.0")])
    writes = [
        _assign_idx("projected_center", _ref("gid"), zero4),
        _assign_idx("projected_axes", _ref("gid"), zero4),
        _assign_idx("projected_conic", _ref("gid"), zero4),
        _assign_idx("projected_color", _ref("gid"), zero4),
        _assign_idx("sort_keys", _ref("gid"), _uint_lit("4294967295")),
        _assign_idx("sorted_indices", _ref("gid"), _ref("gid")),
    ]
    if config and config.get("motion_vectors"):
        zero2 = _ctor("vec2", [_lit("0.0"), _lit("0.0")])
        writes.append(_assign_idx("projected_mv", _ref("gid"), zero2))
    if config and config.get("expected_depth"):
        writes.append(_assign_idx("projected_depth", _ref("gid"), _lit("0.0")))
    if config and config.get("foreground_coverage"):
        writes.append(_assign_idx("projected_foreground", _ref("gid"), _lit("0.0")))
    return writes


def _build_preprocess_body(config: dict) -> list:
    """Generate the statement list for the preprocess compute main()."""
    body = []
    sh_degree = config["sh_degree"]

    # --- Thread index and bounds check ---
    # let gid = global_invocation_id.x;
    body.append(_let("gid", "uint",
                      _swizzle(_ref("global_invocation_id"), "x")))
    # if (gid >= push.total_splats) { return; }
    body.append(_if(
        _binop(">=", _ref("gid"), _push_field("total_splats")),
        [ReturnStmt(None)],
    ))

    # --- Read position and transform to view space ---
    # let world_pos = splat_pos[gid].xyz;
    body.append(_let("world_pos", "vec3",
                      _swizzle(_idx("splat_pos", _ref("gid")), "xyz")))
    # let view_pos = (push.view_matrix * vec4(world_pos, 1.0)).xyz;
    body.append(_let("view_pos", "vec3",
                      _swizzle(
                          _binop("*",
                                 _push_field("view_matrix"),
                                 _ctor("vec4", [_ref("world_pos"), _lit("1.0")])),
                          "xyz")))

    # --- Frustum culling: discard splats behind the near plane ---
    # GLM / Vulkan uses right-handed view space: objects in front of the
    # camera have negative Z.  Cull if view_pos.z > -0.1 (behind camera).
    # Write invisible markers so culled splats don't leave uninitialized GPU memory.
    body.append(_if(
        _binop(">", _swizzle(_ref("view_pos"), "z"), _lit("-0.1")),
        _cull_writes(config) + [ReturnStmt(None)],
    ))

    # --- Read rotation quaternion and build 3x3 rotation matrix ---
    # let quat = splat_rot[gid];
    body.append(_let("quat", "vec4", _idx("splat_rot", _ref("gid"))))

    # Quaternion to rotation matrix (inline, avoids helper fn dependency):
    # glTF / KHR_gaussian_splatting stores quaternions as XYZW (scalar-last).
    #   i = quat.x, j = quat.y, k = quat.z, r(scalar) = quat.w
    body.append(_let("qr", "scalar", _swizzle(_ref("quat"), "w")))
    body.append(_let("qi", "scalar", _swizzle(_ref("quat"), "x")))
    body.append(_let("qj", "scalar", _swizzle(_ref("quat"), "y")))
    body.append(_let("qk", "scalar", _swizzle(_ref("quat"), "z")))

    # Precompute products
    body.append(_let("qi2", "scalar", _binop("*", _ref("qi"), _ref("qi"))))
    body.append(_let("qj2", "scalar", _binop("*", _ref("qj"), _ref("qj"))))
    body.append(_let("qk2", "scalar", _binop("*", _ref("qk"), _ref("qk"))))
    body.append(_let("qri", "scalar", _binop("*", _ref("qr"), _ref("qi"))))
    body.append(_let("qrj", "scalar", _binop("*", _ref("qr"), _ref("qj"))))
    body.append(_let("qrk", "scalar", _binop("*", _ref("qr"), _ref("qk"))))
    body.append(_let("qij", "scalar", _binop("*", _ref("qi"), _ref("qj"))))
    body.append(_let("qik", "scalar", _binop("*", _ref("qi"), _ref("qk"))))
    body.append(_let("qjk", "scalar", _binop("*", _ref("qj"), _ref("qk"))))

    # Rotation matrix rows (row-major for mat3 construction)
    # Row 0: [1 - 2(qj2 + qk2), 2(qij - qrk), 2(qik + qrj)]
    body.append(_let("rot_00", "scalar",
        _binop("-", _lit("1.0"),
               _binop("*", _lit("2.0"),
                      _binop("+", _ref("qj2"), _ref("qk2"))))))
    body.append(_let("rot_01", "scalar",
        _binop("*", _lit("2.0"),
               _binop("-", _ref("qij"), _ref("qrk")))))
    body.append(_let("rot_02", "scalar",
        _binop("*", _lit("2.0"),
               _binop("+", _ref("qik"), _ref("qrj")))))
    # Row 1: [2(qij + qrk), 1 - 2(qi2 + qk2), 2(qjk - qri)]
    body.append(_let("rot_10", "scalar",
        _binop("*", _lit("2.0"),
               _binop("+", _ref("qij"), _ref("qrk")))))
    body.append(_let("rot_11", "scalar",
        _binop("-", _lit("1.0"),
               _binop("*", _lit("2.0"),
                      _binop("+", _ref("qi2"), _ref("qk2"))))))
    body.append(_let("rot_12", "scalar",
        _binop("*", _lit("2.0"),
               _binop("-", _ref("qjk"), _ref("qri")))))
    # Row 2: [2(qik - qrj), 2(qjk + qri), 1 - 2(qi2 + qj2)]
    body.append(_let("rot_20", "scalar",
        _binop("*", _lit("2.0"),
               _binop("-", _ref("qik"), _ref("qrj")))))
    body.append(_let("rot_21", "scalar",
        _binop("*", _lit("2.0"),
               _binop("+", _ref("qjk"), _ref("qri")))))
    body.append(_let("rot_22", "scalar",
        _binop("-", _lit("1.0"),
               _binop("*", _lit("2.0"),
                      _binop("+", _ref("qi2"), _ref("qj2"))))))

    # --- Read scale (stored as log-scale, apply exp) ---
    # let log_scale = splat_scale[gid].xyz;
    body.append(_let("log_scale", "vec3",
                      _swizzle(_idx("splat_scale", _ref("gid")), "xyz")))
    # let scale = vec3(exp(log_scale.x), exp(log_scale.y), exp(log_scale.z));
    body.append(_let("scale", "vec3",
        _ctor("vec3", [
            _call("exp", [_swizzle(_ref("log_scale"), "x")]),
            _call("exp", [_swizzle(_ref("log_scale"), "y")]),
            _call("exp", [_swizzle(_ref("log_scale"), "z")]),
        ])))

    # --- Transform rotation matrix to view space ---
    # The 3DGS covariance projection requires Sigma_view = V * Sigma_world * V^T
    # where V is the 3x3 view rotation.  Transform rotation columns first:
    #   vr_col_k = V * rot_col_k  (mat4 * vec4 with w=0 to transform direction)
    # Then cov_view = (VR * S) * (VR * S)^T
    for col in range(3):
        body.append(_let(f"vr_col{col}", "vec3",
            _swizzle(
                _binop("*", _push_field("view_matrix"),
                       _ctor("vec4", [
                           _ref(f"rot_0{col}"), _ref(f"rot_1{col}"),
                           _ref(f"rot_2{col}"), _lit("0.0"),
                       ])),
                "xyz")))
    for col in range(3):
        for row in range(3):
            body.append(_let(f"vr_{row}{col}", "scalar",
                _swizzle(_ref(f"vr_col{col}"), "xyz"[row])))

    # --- Compute view-space covariance (upper triangle) ---
    # Sigma_view = (V*R) * S^2 * (V*R)^T
    # cov_ij = sum_k vr_ik * vr_jk * s_k^2
    body.append(_let("sx", "scalar", _swizzle(_ref("scale"), "x")))
    body.append(_let("sy", "scalar", _swizzle(_ref("scale"), "y")))
    body.append(_let("sz", "scalar", _swizzle(_ref("scale"), "z")))
    body.append(_let("sx2", "scalar", _binop("*", _ref("sx"), _ref("sx"))))
    body.append(_let("sy2", "scalar", _binop("*", _ref("sy"), _ref("sy"))))
    body.append(_let("sz2", "scalar", _binop("*", _ref("sz"), _ref("sz"))))

    def _cov3d_elem(i: int, j: int) -> BinaryOp:
        """cov_view_ij = sum_k vr_ik * vr_jk * s_k^2."""
        terms = []
        for k, s2 in enumerate(["sx2", "sy2", "sz2"]):
            term = _binop("*",
                          _binop("*", _ref(f"vr_{i}{k}"), _ref(f"vr_{j}{k}")),
                          _ref(s2))
            terms.append(term)
        return _binop("+", _binop("+", terms[0], terms[1]), terms[2])

    body.append(_let("cov3d_00", "scalar", _cov3d_elem(0, 0)))
    body.append(_let("cov3d_01", "scalar", _cov3d_elem(0, 1)))
    body.append(_let("cov3d_02", "scalar", _cov3d_elem(0, 2)))
    body.append(_let("cov3d_11", "scalar", _cov3d_elem(1, 1)))
    body.append(_let("cov3d_12", "scalar", _cov3d_elem(1, 2)))
    body.append(_let("cov3d_22", "scalar", _cov3d_elem(2, 2)))

    # --- Project 3D covariance to 2D screen space ---
    # Standard 3DGS Jacobian with positive depth t = -vz:
    #   J = [[focal_x / t, 0, -focal_x * x / t^2],
    #        [0, focal_y / t, -focal_y * y / t^2]]
    body.append(_let("vz", "scalar", _swizzle(_ref("view_pos"), "z")))
    # Use positive depth (objects have negative vz, so t = -vz > 0)
    body.append(_let("t", "scalar", _neg(_ref("vz"))))
    body.append(_let("t2", "scalar", _binop("*", _ref("t"), _ref("t"))))
    body.append(_let("fx", "scalar", _push_field("focal_x")))
    body.append(_let("fy", "scalar", _push_field("focal_y")))

    # Clamp view-space x/y for Jacobian computation (reference 3DGS trick).
    # Prevents extreme perspective distortion for splats at wide angles.
    # limx = 1.3 * (screen_w / (2 * fx)) * t  ≈ 1.3 * tan(fov_x/2) * depth
    body.append(_let("half_sw", "scalar",
        _binop("*", _lit("0.5"), _swizzle(_push_field("screen_size"), "x"))))
    body.append(_let("half_sh", "scalar",
        _binop("*", _lit("0.5"), _swizzle(_push_field("screen_size"), "y"))))
    body.append(_let("limx", "scalar",
        _binop("*", _lit("1.3"), _binop("*",
            _binop("/", _ref("half_sw"), _ref("fx")), _ref("t")))))
    body.append(_let("limy", "scalar",
        _binop("*", _lit("1.3"), _binop("*",
            _binop("/", _ref("half_sh"), _ref("fy")), _ref("t")))))
    body.append(_let("vx", "scalar",
        _call("clamp", [_swizzle(_ref("view_pos"), "x"),
                        _neg(_ref("limx")), _ref("limx")])))
    body.append(_let("vy", "scalar",
        _call("clamp", [_swizzle(_ref("view_pos"), "y"),
                        _neg(_ref("limy")), _ref("limy")])))

    # Jacobian elements for Vulkan Y-down NDC convention.
    # NDC: sx = fx*vx/t, sy = -fy*vy/t  (proj matrix flips Y)
    # ∂sx/∂vz = fx*vx/t², ∂sy/∂vy = -fy/t, ∂sy/∂vz = -fy*vy/t²
    body.append(_let("j00", "scalar", _binop("/", _ref("fx"), _ref("t"))))
    body.append(_let("j02", "scalar",
        _binop("/", _binop("*", _ref("fx"), _ref("vx")), _ref("t2"))))
    body.append(_let("j11", "scalar",
        _neg(_binop("/", _ref("fy"), _ref("t")))))
    body.append(_let("j12", "scalar",
        _neg(_binop("/", _binop("*", _ref("fy"), _ref("vy")), _ref("t2")))))

    # 2D covariance: Sigma' = J * Sigma3D * J^T  (only upper triangle of the
    # symmetric 2x2 result)
    # Sigma'_00 = j00^2 * c00 + 2*j00*j02*c02 + j02^2*c22
    body.append(_let("cov2d_00", "scalar",
        _binop("+",
               _binop("+",
                      _binop("*", _binop("*", _ref("j00"), _ref("j00")), _ref("cov3d_00")),
                      _binop("*", _binop("*", _lit("2.0"), _binop("*", _ref("j00"), _ref("j02"))),
                             _ref("cov3d_02"))),
               _binop("*", _binop("*", _ref("j02"), _ref("j02")), _ref("cov3d_22")))))
    # Sigma'_01 = j00*j11*c01 + j00*j12*c02 + j02*j11*c12 + j02*j12*c22
    body.append(_let("cov2d_01", "scalar",
        _binop("+",
               _binop("+",
                      _binop("*", _binop("*", _ref("j00"), _ref("j11")), _ref("cov3d_01")),
                      _binop("*", _binop("*", _ref("j00"), _ref("j12")), _ref("cov3d_02"))),
               _binop("+",
                      _binop("*", _binop("*", _ref("j02"), _ref("j11")), _ref("cov3d_12")),
                      _binop("*", _binop("*", _ref("j02"), _ref("j12")), _ref("cov3d_22"))))))
    # Sigma'_11 = j11^2*c11 + 2*j11*j12*c12 + j12^2*c22
    body.append(_let("cov2d_11", "scalar",
        _binop("+",
               _binop("+",
                      _binop("*", _binop("*", _ref("j11"), _ref("j11")), _ref("cov3d_11")),
                      _binop("*", _binop("*", _lit("2.0"), _binop("*", _ref("j11"), _ref("j12"))),
                             _ref("cov3d_12"))),
               _binop("*", _binop("*", _ref("j12"), _ref("j12")), _ref("cov3d_22")))))

    # Dilation (3DGS/gsplat "eps2d" antialiasing convention, SPECIFICATION.md
    # 12.8): add `dilation` px^2 to the diagonal of the projected 2D
    # covariance before inversion, with NO opacity compensation for the
    # resulting blur (matching the reference 3DGS rasterizer / gsplat
    # exactly -- earlier lux versions additionally scaled opacity down by
    # sqrt(det_orig/det), which gsplat's default AA mode does not do).
    body.append(_let("cov2d_00f", "scalar",
        _binop("+", _ref("cov2d_00"), _lit(config["dilation"]))))
    body.append(_let("cov2d_11f", "scalar",
        _binop("+", _ref("cov2d_11"), _lit(config["dilation"]))))

    # --- Compute inverse covariance (conic) for the fragment shader ---
    # det = cov2d_00f * cov2d_11f - cov2d_01^2
    body.append(_let("det", "scalar",
        _binop("-",
               _binop("*", _ref("cov2d_00f"), _ref("cov2d_11f")),
               _binop("*", _ref("cov2d_01"), _ref("cov2d_01")))))

    # if (det <= 0.0) { return; }  -- degenerate ellipse
    body.append(_if(
        _binop("<=", _ref("det"), _lit("0.0")),
        _cull_writes(config) + [ReturnStmt(None)],
    ))

    body.append(_let("inv_det", "scalar", _binop("/", _lit("1.0"), _ref("det"))))

    # Conic = inverse of 2D covariance (symmetric 2x2):
    #   conic.x = cov2d_11f * inv_det
    #   conic.y = -cov2d_01 * inv_det
    #   conic.z = cov2d_00f * inv_det
    body.append(_let("conic_x", "scalar",
        _binop("*", _ref("cov2d_11f"), _ref("inv_det"))))
    body.append(_let("conic_y", "scalar",
        _binop("*", _neg(_ref("cov2d_01")), _ref("inv_det"))))
    body.append(_let("conic_z", "scalar",
        _binop("*", _ref("cov2d_00f"), _ref("inv_det"))))

    # --- Read opacity (sigmoid activation) ---
    # Hoisted here (was originally after NDC projection/frustum culling,
    # right before SH evaluation) so the opacity-tight quad-extent below can
    # use it -- `opacity` only depends on `gid` (splat_opacity[gid]), so it
    # has no dependency on anything computed in between and this reordering
    # changes no other value.
    # let raw_opacity = splat_opacity[gid];
    # let opacity = 1.0 / (1.0 + exp(-raw_opacity));
    body.append(_let("raw_opacity", "scalar",
        _idx("splat_opacity", _ref("gid"))))
    body.append(_let("raw_sigmoid", "scalar",
        _binop("/", _lit("1.0"),
               _binop("+", _lit("1.0"),
                      _call("exp", [_neg(_ref("raw_opacity"))])))))
    # No opacity compensation (see the dilation comment above) -- opacity is
    # just the activated raw value.
    body.append(_let("opacity", "scalar", _ref("raw_sigmoid")))

    # --- Cull low-opacity splats early (before radius/SH eval) ---
    # Splats with opacity < alpha_min (default 1/255, matching 3DGS/gsplat)
    # are invisible. Also guarantees opacity > alpha_min below (ln(opacity/
    # alpha_min) > 0), so the opacity-tight radius sqrt() argument is real.
    body.append(_if(
        _binop("<", _ref("opacity"), _lit(config["alpha_min"])),
        _cull_writes(config) + [ReturnStmt(None)],
    ))

    # --- Compute screen-space radius from eigenvalues ---
    # mid = 0.5 * (cov2d_00f + cov2d_11f)
    # half_diff = sqrt(max((mid*mid - det), 0.0))
    # lambda_max = mid + half_diff
    #
    # Opacity-tight extent (perf; SPECIFICATION.md 12.8 / docs/rendering-
    # engines.md's mobile-DLSS Android timing breakdown): rather than a
    # fixed 3-sigma quad, size it to exactly where this splat's peak-alpha
    # falls to alpha_min --
    #   opacity * exp(-0.5*t^2) = alpha_min  =>  t = sqrt(2*ln(opacity/alpha_min))
    # -- clamped to <=3 (never LARGER than the reference 3-sigma quad, only
    # smaller for low-opacity splats). This is exact w.r.t. the fragment
    # shader's own `alpha < alpha_min` discard (same section below /
    # splat_expander.py's fragment-stage alpha_min cutoff): every fragment
    # this shrinks the quad past would have been discarded anyway, so pixel
    # output is bit-identical -- only wasted rasterizer/blend work on
    # already-invisible fragments is removed. Background/low-opacity
    # splats (common after area-based pruning, which systematically keeps
    # few large-footprint splats -- see the Stage 1 timing breakdown's
    # overdraw analysis) benefit most.
    body.append(_let("mid", "scalar",
        _binop("*", _lit("0.5"),
               _binop("+", _ref("cov2d_00f"), _ref("cov2d_11f")))))
    body.append(_let("half_diff", "scalar",
        _call("sqrt", [_call("max", [
            _binop("-", _binop("*", _ref("mid"), _ref("mid")), _ref("det")),
            _lit("0.0"),
        ])])))
    body.append(_let("lambda_max", "scalar",
        _binop("+", _ref("mid"), _ref("half_diff"))))
    # lambda_min = mid - half_diff. Always > 0 here: det = lambda_max *
    # lambda_min was already checked > 0 (degenerate-ellipse cull above),
    # and lambda_max > 0 (trace mid > 0 since dilation adds a positive
    # constant to both diagonal entries), so lambda_min = det/lambda_max > 0.
    body.append(_let("lambda_min", "scalar",
        _binop("-", _ref("mid"), _ref("half_diff"))))
    body.append(_let("sigma_t_raw", "scalar",
        _call("sqrt", [_binop("*", _lit("2.0"),
                               _call("log", [_binop("/", _ref("opacity"), _lit(config["alpha_min"]))]))])))
    body.append(_let("sigma_t", "scalar",
        _call("min", [_ref("sigma_t_raw"), _lit("3.0")])))

    # --- Oriented quad: covariance eigenvectors (see class docstring's
    # "Oriented quads" note). theta diagonalizes the symmetric 2x2
    # covariance; e1 = eigenvector for lambda_max, e2 (perpendicular) =
    # eigenvector for lambda_min.
    body.append(_let("eig_theta", "scalar",
        _binop("*", _lit("0.5"),
               _call("atan", [_binop("*", _lit("2.0"), _ref("cov2d_01")),
                               _binop("-", _ref("cov2d_00f"), _ref("cov2d_11f"))]))))
    body.append(_let("eig_cos", "scalar", _call("cos", [_ref("eig_theta")])))
    body.append(_let("eig_sin", "scalar", _call("sin", [_ref("eig_theta")])))

    body.append(_let("raw_radius", "scalar",
        _call("ceil", [_binop("*", _ref("sigma_t"),
                               _call("sqrt", [_ref("lambda_max")]))])))
    # Clamp radius to screen size (prevents excessively large quads) --
    # `radius` is now the major-axis (larger eigenvector) extent; kept
    # under this name since it's still written to projected_center.w for
    # frustum-culling margin (below) exactly as before.
    body.append(_let("radius", "scalar",
        _call("min", [_ref("raw_radius"),
               _call("max", [_swizzle(_push_field("screen_size"), "x"),
                              _swizzle(_push_field("screen_size"), "y")])])))
    body.append(_let("raw_minor", "scalar",
        _call("ceil", [_binop("*", _ref("sigma_t"),
                               _call("sqrt", [_ref("lambda_min")]))])))
    body.append(_let("minor_len", "scalar",
        _call("min", [_ref("raw_minor"),
               _call("max", [_swizzle(_push_field("screen_size"), "x"),
                              _swizzle(_push_field("screen_size"), "y")])])))
    body.append(_let("major_vec", "vec2",
        _ctor("vec2", [_binop("*", _ref("eig_cos"), _ref("radius")),
                        _binop("*", _ref("eig_sin"), _ref("radius"))])))
    body.append(_let("minor_vec", "vec2",
        _ctor("vec2", [_binop("*", _neg(_ref("eig_sin")), _ref("minor_len")),
                        _binop("*", _ref("eig_cos"), _ref("minor_len"))])))

    # --- Project center to NDC ---
    # let clip_pos = push.proj_matrix * vec4(view_pos, 1.0);
    body.append(_let("clip_pos", "vec4",
        _binop("*", _push_field("proj_matrix"),
               _ctor("vec4", [_ref("view_pos"), _lit("1.0")]))))
    body.append(_let("ndc_x", "scalar",
        _binop("/", _swizzle(_ref("clip_pos"), "x"),
               _swizzle(_ref("clip_pos"), "w"))))
    body.append(_let("ndc_y", "scalar",
        _binop("/", _swizzle(_ref("clip_pos"), "y"),
               _swizzle(_ref("clip_pos"), "w"))))
    body.append(_let("ndc_z", "scalar",
        _binop("/", _swizzle(_ref("clip_pos"), "z"),
               _swizzle(_ref("clip_pos"), "w"))))

    # --- Viewport frustum culling ---
    # Cull splats whose NDC center is too far outside the viewport.
    # Margin accounts for the splat radius in NDC space.
    body.append(_let("ndc_margin", "scalar",
        _binop("+", _lit("1.0"),
               _binop("/", _ref("radius"),
                      _call("min", [_swizzle(_push_field("screen_size"), "x"),
                                    _swizzle(_push_field("screen_size"), "y")])))))
    body.append(_if(
        _binop("||",
               _binop("||",
                      _binop(">", _ref("ndc_x"), _ref("ndc_margin")),
                      _binop("<", _ref("ndc_x"), _neg(_ref("ndc_margin")))),
               _binop("||",
                      _binop(">", _ref("ndc_y"), _ref("ndc_margin")),
                      _binop("<", _ref("ndc_y"), _neg(_ref("ndc_margin"))))),
        _cull_writes(config) + [ReturnStmt(None)],
    ))

    # (opacity already read + culled above, before the radius computation --
    # see the "opacity-tight extent" comment.)

    # --- Evaluate spherical harmonics for view-dependent color ---
    body.extend(_build_sh_evaluation(sh_degree))

    # --- Clamp color to [0,1] and pack with opacity ---
    body.append(_let("clamped_r", "scalar",
        _call("clamp", [_swizzle(_ref("sh_color"), "x"), _lit("0.0"), _lit("1.0")])))
    body.append(_let("clamped_g", "scalar",
        _call("clamp", [_swizzle(_ref("sh_color"), "y"), _lit("0.0"), _lit("1.0")])))
    body.append(_let("clamped_b", "scalar",
        _call("clamp", [_swizzle(_ref("sh_color"), "z"), _lit("0.0"), _lit("1.0")])))

    # --- DLSS input-contract outputs (docs/lux-4d-spec.md section 3) ---
    if config.get("motion_vectors"):
        # Current-frame NDC using the UNJITTERED projection (jitter must not
        # leak into the motion vector); reuses the already-computed
        # view-space position.
        body.append(_let("clip_u", "vec4",
            _binop("*", _idx("prev_camera_mats", _uint_lit(0)),
                   _ctor("vec4", [_ref("view_pos"), _lit("1.0")]))))
        body.append(_let("ndc_u_x", "scalar",
            _binop("/", _swizzle(_ref("clip_u"), "x"), _swizzle(_ref("clip_u"), "w"))))
        body.append(_let("ndc_u_y", "scalar",
            _binop("/", _swizzle(_ref("clip_u"), "y"), _swizzle(_ref("clip_u"), "w"))))

        # Previous frame's animated world position, projected with the
        # previous frame's (unjittered) view-projection.
        body.append(_let("prev_world_pos", "vec3",
            _swizzle(_idx("splat_prev_pos", _ref("gid")), "xyz")))
        body.append(_let("prev_clip", "vec4",
            _binop("*", _idx("prev_camera_mats", _uint_lit(1)),
                   _ctor("vec4", [_ref("prev_world_pos"), _lit("1.0")]))))
        body.append(_let("prev_ndc_x", "scalar",
            _binop("/", _swizzle(_ref("prev_clip"), "x"), _swizzle(_ref("prev_clip"), "w"))))
        body.append(_let("prev_ndc_y", "scalar",
            _binop("/", _swizzle(_ref("prev_clip"), "y"), _swizzle(_ref("prev_clip"), "w"))))

        # uv = (ndc*0.5 + 0.5) * screen_size  =>  d(uv)/d(ndc) = 0.5*screen_size,
        # so the pixel-space motion vector is exactly the NDC delta scaled by
        # that constant factor (no need to materialize uv_curr/uv_prev).
        body.append(_let("mv_x", "scalar",
            _binop("*",
                   _binop("-", _ref("ndc_u_x"), _ref("prev_ndc_x")),
                   _binop("*", _lit("0.5"), _swizzle(_push_field("screen_size"), "x")))))
        body.append(_let("mv_y", "scalar",
            _binop("*",
                   _binop("-", _ref("ndc_u_y"), _ref("prev_ndc_y")),
                   _binop("*", _lit("0.5"), _swizzle(_push_field("screen_size"), "y")))))
        body.append(_assign_idx("projected_mv", _ref("gid"),
            _ctor("vec2", [_ref("mv_x"), _ref("mv_y")])))

    if config.get("expected_depth"):
        # Camera-space z (already computed as `t = -view_pos.z` above).
        body.append(_assign_idx("projected_depth", _ref("gid"), _ref("t")))

    if config.get("foreground_coverage"):
        # Straight passthrough -- the flag itself is a fixed per-splat
        # value (glTF attribute or morph-membership, computed host-side),
        # not something to derive per-frame here.
        body.append(_assign_idx("projected_foreground", _ref("gid"),
            _idx("splat_foreground", _ref("gid"))))

    # --- Write output buffers ---
    body.append(_assign_idx("projected_center", _ref("gid"),
        _ctor("vec4", [_ref("ndc_x"), _ref("ndc_y"), _ref("ndc_z"), _ref("radius")])))

    body.append(_assign_idx("projected_axes", _ref("gid"),
        _ctor("vec4", [_swizzle(_ref("major_vec"), "x"), _swizzle(_ref("major_vec"), "y"),
                        _swizzle(_ref("minor_vec"), "x"), _swizzle(_ref("minor_vec"), "y")])))

    body.append(_assign_idx("projected_conic", _ref("gid"),
        _ctor("vec4", [_ref("conic_x"), _ref("conic_y"), _ref("conic_z"), _ref("opacity")])))

    body.append(_assign_idx("projected_color", _ref("gid"),
        _ctor("vec4", [_ref("clamped_r"), _ref("clamped_g"), _ref("clamped_b"), _ref("opacity")])))

    # --- Sort metric ---
    # "camera_distance" (default, matches the ratified KHR_gaussian_splatting
    # spec's sortingMethod): true Euclidean distance from the camera,
    # negated so farther splats get the more-negative value -- matching
    # `vz`'s own sign convention (objects in front have negative view-space
    # z, more negative the farther away) so the existing ascending-key
    # radix sort still produces a back-to-front draw order for correct
    # premultiplied-alpha "over" compositing.
    # "view_depth" (gsplat's own convention): raw view-space z.
    if config.get("sort") == "view_depth":
        body.append(_let("sort_metric", "scalar", _ref("vz")))
    else:
        body.append(_let("sort_metric", "scalar", _neg(_call("length", [_ref("view_pos")]))))

    # --- Sort key: convert float depth to sortable uint for GPU radix sort ---
    body.append(_let("key_bits", "uint", _call("float_bits_to_uint", [_ref("sort_metric")])))
    body.append(_let("sign_mask", "uint",
        _binop("*",
            _binop(">>", _ref("key_bits"), _uint_lit("31")),
            _uint_lit("4294967295"))))
    body.append(_let("sort_key", "uint",
        _binop("^", _ref("key_bits"),
            _binop("|", _ref("sign_mask"), _uint_lit("2147483648")))))
    body.append(_assign_idx("sort_keys", _ref("gid"), _ref("sort_key")))

    # Initialize sorted_indices to identity (radix sort reads this as input)
    body.append(_assign_idx("sorted_indices", _ref("gid"), _ref("gid")))

    return body


def _build_sh_evaluation(sh_degree: int) -> list:
    """Generate statements that evaluate spherical harmonics.

    The SH coefficients are stored in per-splat vec4 buffers (rgb + pad).
    The evaluation computes a view-dependent color using the SH basis
    functions up to the requested degree.

    The result is stored in a local ``sh_color`` of type ``vec3``.
    """
    stmts = []

    # Direction from camera to splat (matches reference 3DGS convention).
    # Odd-degree SH basis functions are odd functions of direction — sign matters.
    # let cam_dir = normalize(world_pos - push.cam_pos);
    stmts.append(_let("cam_dir", "vec3",
        _call("normalize", [
            _binop("-", _ref("world_pos"), _push_field("cam_pos")),
        ])))

    # Degree 0: DC term (constant)
    # SH_C0 = 0.28209479177387814
    # color = SH_C0 * sh0[gid].xyz + 0.5
    stmts.append(_let("sh0_val", "vec3",
        _swizzle(_idx("splat_sh0", _ref("gid")), "xyz")))
    stmts.append(_let("sh_color_0", "vec3",
        _binop("+",
               _binop("*", _lit("0.28209479"), _ref("sh0_val")),
               _ctor("vec3", [_lit("0.5"), _lit("0.5"), _lit("0.5")]))))

    if sh_degree == 0:
        stmts.append(_let("sh_color", "vec3", _ref("sh_color_0")))
        return stmts

    # Direction components
    stmts.append(_let("dx", "scalar", _swizzle(_ref("cam_dir"), "x")))
    stmts.append(_let("dy", "scalar", _swizzle(_ref("cam_dir"), "y")))
    stmts.append(_let("dz", "scalar", _swizzle(_ref("cam_dir"), "z")))

    # Degree 1: 3 basis functions
    # SH_C1 = 0.4886025119029199
    # Y_1^{-1} = SH_C1 * y,  Y_1^0 = SH_C1 * z,  Y_1^1 = SH_C1 * x
    stmts.append(_let("sh1_val", "vec3",
        _swizzle(_idx("splat_sh1", _ref("gid")), "xyz")))
    stmts.append(_let("sh2_val", "vec3",
        _swizzle(_idx("splat_sh2", _ref("gid")), "xyz")))
    stmts.append(_let("sh3_val", "vec3",
        _swizzle(_idx("splat_sh3", _ref("gid")), "xyz")))

    # Accumulate degree 1 (Condon-Shortley phase: negative on y and x terms)
    # color += -SH_C1*dy*sh1 + SH_C1*dz*sh2 - SH_C1*dx*sh3
    stmts.append(_let("sh_color_1", "vec3",
        _binop("+", _ref("sh_color_0"),
               _binop("+",
                      _binop("+",
                             _binop("*", _lit("-0.48860251"),
                                    _binop("*", _ref("dy"), _ref("sh1_val"))),
                             _binop("*", _lit("0.48860251"),
                                    _binop("*", _ref("dz"), _ref("sh2_val")))),
                      _binop("*", _lit("-0.48860251"),
                             _binop("*", _ref("dx"), _ref("sh3_val")))))))

    if sh_degree == 1:
        stmts.append(_let("sh_color", "vec3", _ref("sh_color_1")))
        return stmts

    # Degree 2: 5 basis functions
    # SH_C2 constants
    # c2_0 = 1.0925484305920792, c2_1 = -1.0925484305920792
    # c2_2 = 0.31539156525252005, c2_3 = -1.0925484305920792
    # c2_4 = 0.5462742152960396
    stmts.append(_let("sh4_val", "vec3",
        _swizzle(_idx("splat_sh4", _ref("gid")), "xyz")))
    stmts.append(_let("sh5_val", "vec3",
        _swizzle(_idx("splat_sh5", _ref("gid")), "xyz")))
    stmts.append(_let("sh6_val", "vec3",
        _swizzle(_idx("splat_sh6", _ref("gid")), "xyz")))
    stmts.append(_let("sh7_val", "vec3",
        _swizzle(_idx("splat_sh7", _ref("gid")), "xyz")))
    stmts.append(_let("sh8_val", "vec3",
        _swizzle(_idx("splat_sh8", _ref("gid")), "xyz")))

    stmts.append(_let("dx2", "scalar", _binop("*", _ref("dx"), _ref("dx"))))
    stmts.append(_let("dy2", "scalar", _binop("*", _ref("dy"), _ref("dy"))))
    stmts.append(_let("dz2", "scalar", _binop("*", _ref("dz"), _ref("dz"))))
    stmts.append(_let("dxy", "scalar", _binop("*", _ref("dx"), _ref("dy"))))
    stmts.append(_let("dyz", "scalar", _binop("*", _ref("dy"), _ref("dz"))))
    stmts.append(_let("dxz", "scalar", _binop("*", _ref("dx"), _ref("dz"))))

    # Y_2^{-2} = c2_0 * xy,        Y_2^{-1} = c2_1 * yz,
    # Y_2^0    = c2_2 * (2z^2-x^2-y^2),
    # Y_2^1    = c2_3 * xz,        Y_2^2    = c2_4 * (x^2-y^2)
    stmts.append(_let("sh_deg2", "vec3",
        _binop("+",
            _binop("+",
                _binop("+",
                    _binop("*", _binop("*", _lit("1.09254843"), _ref("dxy")), _ref("sh4_val")),
                    _binop("*", _binop("*", _lit("-1.09254843"), _ref("dyz")), _ref("sh5_val"))),
                _binop("+",
                    _binop("*", _binop("*", _lit("0.31539157"),
                        _binop("-", _binop("*", _lit("2.0"), _ref("dz2")),
                               _binop("+", _ref("dx2"), _ref("dy2")))),
                        _ref("sh6_val")),
                    _binop("*", _binop("*", _lit("-1.09254843"), _ref("dxz")), _ref("sh7_val")))),
            _binop("*", _binop("*", _lit("0.54627422"),
                _binop("-", _ref("dx2"), _ref("dy2"))),
                _ref("sh8_val")))))

    stmts.append(_let("sh_color_2", "vec3",
        _binop("+", _ref("sh_color_1"), _ref("sh_deg2"))))

    if sh_degree == 2:
        stmts.append(_let("sh_color", "vec3", _ref("sh_color_2")))
        return stmts

    # Degree 3: 7 basis functions (splat_sh9 .. splat_sh15)
    for i in range(9, 16):
        stmts.append(_let(f"sh{i}_val", "vec3",
            _swizzle(_idx(f"splat_sh{i}", _ref("gid")), "xyz")))

    # Precompute extra direction products for degree 3
    stmts.append(_let("dx3", "scalar", _binop("*", _ref("dx2"), _ref("dx"))))
    stmts.append(_let("dy3", "scalar", _binop("*", _ref("dy2"), _ref("dy"))))
    stmts.append(_let("dz3", "scalar", _binop("*", _ref("dz2"), _ref("dz"))))

    # SH degree 3 constants
    # c3_0 = -0.5900435899, c3_1 = 2.8906114426, c3_2 = -0.4570457995
    # c3_3 = 0.3731763326,  c3_4 = -0.4570457995, c3_5 = 1.4453057213
    # c3_6 = -0.5900435899

    # Y_3^{-3} = c3_0 * y(3x^2-y^2),  Y_3^{-2} = c3_1 * xyz
    # Y_3^{-1} = c3_2 * y(4z^2-x^2-y^2), Y_3^0 = c3_3 * z(2z^2-3x^2-3y^2)
    # Y_3^1 = c3_4 * x(4z^2-x^2-y^2), Y_3^2 = c3_5 * z(x^2-y^2)
    # Y_3^3 = c3_6 * x(x^2-3y^2)

    stmts.append(_let("sh_deg3_a", "vec3",
        _binop("+",
            _binop("+",
                _binop("*",
                    _binop("*", _lit("-0.59004359"),
                        _binop("*", _ref("dy"),
                            _binop("-", _binop("*", _lit("3.0"), _ref("dx2")), _ref("dy2")))),
                    _ref("sh9_val")),
                _binop("*",
                    _binop("*", _lit("2.89061144"),
                        _binop("*", _ref("dx"), _binop("*", _ref("dy"), _ref("dz")))),
                    _ref("sh10_val"))),
            _binop("*",
                _binop("*", _lit("-0.45704580"),
                    _binop("*", _ref("dy"),
                        _binop("-", _binop("*", _lit("4.0"), _ref("dz2")),
                            _binop("+", _ref("dx2"), _ref("dy2"))))),
                _ref("sh11_val")))))

    stmts.append(_let("sh_deg3_b", "vec3",
        _binop("+",
            _binop("+",
                _binop("*",
                    _binop("*", _lit("0.37317633"),
                        _binop("*", _ref("dz"),
                            _binop("-", _binop("*", _lit("2.0"), _ref("dz2")),
                                _binop("*", _lit("3.0"),
                                    _binop("+", _ref("dx2"), _ref("dy2")))))),
                    _ref("sh12_val")),
                _binop("*",
                    _binop("*", _lit("-0.45704580"),
                        _binop("*", _ref("dx"),
                            _binop("-", _binop("*", _lit("4.0"), _ref("dz2")),
                                _binop("+", _ref("dx2"), _ref("dy2"))))),
                    _ref("sh13_val"))),
            _binop("+",
                _binop("*",
                    _binop("*", _lit("1.44530572"),
                        _binop("*", _ref("dz"),
                            _binop("-", _ref("dx2"), _ref("dy2")))),
                    _ref("sh14_val")),
                _binop("*",
                    _binop("*", _lit("-0.59004359"),
                        _binop("*", _ref("dx"),
                            _binop("-", _ref("dx2"), _binop("*", _lit("3.0"), _ref("dy2"))))),
                    _ref("sh15_val"))))))

    stmts.append(_let("sh_deg3", "vec3",
        _binop("+", _ref("sh_deg3_a"), _ref("sh_deg3_b"))))

    stmts.append(_let("sh_color", "vec3",
        _binop("+", _ref("sh_color_2"), _ref("sh_deg3"))))

    return stmts


# ---------------------------------------------------------------------------
# Stage 2: render vertex shader
# ---------------------------------------------------------------------------

def _build_vertex_stage(config: dict) -> StageBlock:
    """Generate the render vertex stage.

    This is an instanced draw with 6 vertices per instance (two triangles
    forming a screen-aligned quad).  Each instance corresponds to a sorted
    visible splat.

    Inputs (storage buffers from the preprocess pass):
        projected_center -- (ndc_x, ndc_y, depth, major_extent)
        projected_axes   -- (major.x, major.y, minor.x, minor.y): oriented
                       quad half-axis vectors (see the preprocess stage's
                       "Oriented quads" docstring note)
        projected_conic  -- (inv_cov_a, inv_cov_b, inv_cov_c, opacity)
        projected_color  -- (r, g, b, opacity)
        sorted_indices   -- indirection table from radix sort
        projected_mv     -- (optional, motion_vectors) pixel-space MV
        projected_depth  -- (optional, expected_depth) camera-space z

    Outputs (to fragment stage):
        frag_conic   -- vec3: inverse 2D covariance upper triangle
        frag_color   -- vec4: splat color + opacity
        frag_center  -- vec2: screen-pixel center of the splat
        frag_offset  -- vec2: pixel offset from center for this vertex
        frag_mv      -- (optional) vec2: pixel-space motion vector
        frag_depth   -- (optional) scalar: camera-space z
    """
    stage = StageBlock(stage_type="vertex")

    # --- Storage buffers (read-only) ---
    stage.storage_buffers.append(StorageBufferDecl("projected_center", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("projected_axes", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("projected_conic", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("projected_color", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("sorted_indices", "uint"))
    if config.get("motion_vectors"):
        stage.storage_buffers.append(StorageBufferDecl("projected_mv", "vec2"))
    if config.get("expected_depth"):
        stage.storage_buffers.append(StorageBufferDecl("projected_depth", "scalar"))
    if config.get("foreground_coverage"):
        stage.storage_buffers.append(StorageBufferDecl("projected_foreground", "scalar"))

    # --- Push constants (shared block with fragment to avoid Vulkan offset conflicts) ---
    pc_fields = [
        BlockField("screen_size", "vec2"),
        BlockField("visible_count", "uint"),
        BlockField("alpha_min", "scalar"),
    ]
    stage.push_constants.append(PushBlock("push", pc_fields))

    # --- Vertex outputs → fragment inputs ---
    out_vars = [("frag_conic", "vec3"), ("frag_color", "vec4"),
                ("frag_center", "vec2"), ("frag_offset", "vec2")]
    if config.get("motion_vectors"):
        out_vars.append(("frag_mv", "vec2"))
    if config.get("expected_depth"):
        out_vars.append(("frag_depth", "scalar"))
    if config.get("foreground_coverage"):
        out_vars.append(("frag_foreground", "scalar"))
    for name, ty in out_vars:
        v = VarDecl(name, ty)
        v._is_input = False
        stage.outputs.append(v)

    # --- Main function body ---
    body = _build_vertex_body(config)
    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


def _build_vertex_body(config: dict) -> list:
    """Generate the vertex shader main() body."""
    body = []

    # --- Instance and vertex indices ---
    body.append(_let("inst_id", "uint", _ref("instance_index")))
    body.append(_let("vert_id", "uint", _ref("vertex_index")))

    # --- Look up the sorted splat index ---
    body.append(_let("splat_idx", "uint", _idx("sorted_indices", _ref("inst_id"))))

    # --- Read projected data ---
    body.append(_let("center_data", "vec4",
        _idx("projected_center", _ref("splat_idx"))))
    body.append(_let("axes_data", "vec4",
        _idx("projected_axes", _ref("splat_idx"))))
    body.append(_let("conic_data", "vec4",
        _idx("projected_conic", _ref("splat_idx"))))
    body.append(_let("color_data", "vec4",
        _idx("projected_color", _ref("splat_idx"))))
    if config.get("motion_vectors"):
        body.append(_let("mv_data", "vec2",
            _idx("projected_mv", _ref("splat_idx"))))
    if config.get("expected_depth"):
        body.append(_let("depth_data", "scalar",
            _idx("projected_depth", _ref("splat_idx"))))
    if config.get("foreground_coverage"):
        body.append(_let("foreground_data", "scalar",
            _idx("projected_foreground", _ref("splat_idx"))))

    # Unpack center / radius
    body.append(_let("ndc_center", "vec2",
        _swizzle(_ref("center_data"), "xy")))
    body.append(_let("depth", "scalar",
        _swizzle(_ref("center_data"), "z")))
    body.append(_let("radius", "scalar",
        _swizzle(_ref("center_data"), "w")))
    body.append(_let("major_vec", "vec2",
        _swizzle(_ref("axes_data"), "xy")))
    body.append(_let("minor_vec", "vec2",
        _swizzle(_ref("axes_data"), "zw")))

    # --- Build quad corners (6 vertices → 2 triangles) ---
    # Vertex order: 0,1,2 and 3,4,5 forming a quad
    # Map vertex_index to quad offset:
    #   0 → (-1,-1), 1 → (1,-1), 2 → (-1,1)
    #   3 → (-1,1),  4 → (1,-1), 5 → (1,1)
    # We use integer math: ox = (vert_id % 2) * 2 - 1, sign logic

    # Compute x component: vertices 1,4,5 have +1, others -1
    # Simple lookup via conditionals encoded as arithmetic:
    #   bit0 = vert_id & 1 for triangles  -- but easier to use a small table

    # Use the pattern: for the two triangles of a quad,
    #   tri_idx = vert_id / 3  (0 or 1)
    #   corner  = vert_id % 3  (0, 1, or 2)
    # Triangle 0 corners: BL(0), BR(1), TL(2)  → offsets (-1,-1),(1,-1),(-1,1)
    # Triangle 1 corners: TL(0), BR(1), TR(2)  → offsets (-1,1),(1,-1),(1,1)

    # We flatten the offset lookup using the vert_id directly.
    # x_offset: [-1, 1, -1, -1, 1, 1]  →  (vert_id==1||vert_id==4||vert_id==5) ? 1 : -1
    # y_offset: [-1, -1, 1, 1, -1, 1]  →  (vert_id==2||vert_id==3||vert_id==5) ? 1 : -1

    # Encode as:  x_sign = step(0.5, fract(float(vert_id) * 0.5)) * 2.0 - 1.0
    # Actually, a cleaner approach for AST: use conditional selection

    # Compute via bit manipulation pattern:
    # For x: odd indices in the "effective" pattern
    #   effective = [0,1,0,0,1,1]
    #   Observation: this equals ((vert_id + 1) / 2) & 1 ... but that's complex.
    # Simpler: express as two selects based on vert_id.

    # Actually, the cleanest AST pattern: define the offsets from the vert_id
    # using the standard quad approach with modular arithmetic:
    #   x_off = float((vert_id & 1) ^ (vert_id / 3)) * 2.0 - 1.0
    #   y_off = float((vert_id / 2) & 1) * 2.0 - 1.0
    # But these bit ops aren't standard in Lux's scalar world.

    # Pragmatic approach: encode as a known lookup with add/multiply:
    # For 6-vertex quad, a well-known trick:
    #   x = float(vert_id % 2) * 2.0 - 1.0       (but wrong for tri 1)
    # Better: use the formula from 3DGS reference:
    #   x = float((vert_id & 1) * 2 - 1)  -- needs bitwise
    # Since Lux doesn't have bitwise ops at AST level, we'll use a lookup
    # table approach: 6 if-else branches is too verbose.

    # Simplest correct approach: use two modular operations.
    # quad_x = (vert_id % 2) * 2 - 1  works for {0:-1, 1:1} per triangle
    # quad_y = (vert_id / 2 % 2) * 2 - 1 ... etc.
    # But we need to handle both triangles correctly.

    # Use the reference 3DGS approach with subtraction tricks:
    #   let t = vert_id / 3;        // 0 for first tri, 1 for second
    #   let c = vert_id - t * 3;    // 0,1,2 within triangle
    #   Triangle 0 (t=0): c=0→BL(-1,-1), c=1→BR(1,-1), c=2→TL(-1,1)
    #   Triangle 1 (t=1): c=0→TL(-1,1),  c=1→BR(1,-1), c=2→TR(1,1)
    # x: t0: [-1,1,-1]  t1: [-1,1,1]
    # y: t0: [-1,-1,1]  t1: [1,-1,1]

    # Encode with arithmetic (float-based):
    #   For t=0: x = float(c==1)*2-1,   y = float(c==2)*2-1
    #   For t=1: x = float(c>=1)*2-1,   y = float(c!=1)*2-1

    # This is getting complex. Use the simplest universal formula:
    #   index into a conceptual array. Since we can't do real array lookups
    #   easily in the AST, we'll use the well-known formula:
    #     ox = (1 - 2*step(2.5, float(vert_id))) * (2*step(0.5, fmod(...)) - 1)
    #   ... which is unreadable.

    # Pragmatic solution: just use step functions for each component.
    # x:  verts {1,4,5} → +1, else -1
    #     = step(0.5, abs(sin(float(vert_id) * 1.5))) * 2.0 - 1.0  -- fragile
    # Cleanest: express with conditional:
    #   let fx = float(vert_id);
    #   let quad_x = step(0.5, fx) - step(1.5, fx) + step(3.5, fx);
    #     -> 0: 0, 1: 1, 2: 0, 3: 0, 4: 1, 5: 1  ... wait that doesn't work.

    # Fine -- just use the simple, clean, correct approach with float vert_id
    # and the known bit trick that works universally in splatting renderers:
    # Let's express it directly using (vert_id % 2) and (vert_id / 3):
    #   let col = vert_id - (vert_id / 3) * 3;   // vert_id % 3
    #   let row = vert_id / 3;                    // 0 or 1
    # Then for triangle 0: corners are BL, BR, TL
    # For triangle 1: corners are TL, BR, TR
    # x_off for tri 0: col==1 ? 1 : -1
    # x_off for tri 1: col>=1 ? 1 : -1
    # y_off for tri 0: col==2 ? 1 : -1
    # y_off for tri 1: col!=1 ? 1 : -1
    # Combine: x_off = (col==1 || (row==1 && col==2)) ? 1 : -1
    #          y_off = (col==2 || (row==1 && col==0)) ? 1 : -1

    # Express in Lux arithmetic (no booleans, just scalar comparisons):
    # Step-based approach (step(edge, x) = x >= edge ? 1.0 : 0.0):

    # Actually the simplest correct formula for the standard quad expansion:
    # The six vertices are indexed as:
    #   0: (-1, -1)    3: (-1,  1)
    #   1: ( 1, -1)    4: ( 1, -1)
    #   2: (-1,  1)    5: ( 1,  1)
    # So we directly encode:
    #   x: [-1, 1, -1, -1, 1, 1]
    #   y: [-1, -1, 1, 1, -1, 1]

    # Using the formula: x = ((vert_id & 1u) | (vert_id >> 2u)) * 2 - 1
    # and             :  y = ((vert_id >> 1u) & 1u) * 2 - 1 ... hmm
    # These need bit ops.

    # Just use modular arithmetic with floor division:
    # Convert uint vert_id to float via multiply by 1.0
    body.append(_let("fv", "scalar",
        _binop("*", _ref("vert_id"), _lit("1.0"))))

    # Triangle index (0 or 1) and corner within triangle (0, 1, 2)
    body.append(_let("tri", "scalar",
        _call("floor", [_binop("/", _ref("fv"), _lit("3.0"))])))
    body.append(_let("corner", "scalar",
        _binop("-", _ref("fv"), _binop("*", _ref("tri"), _lit("3.0")))))

    # Offset x: tri0: {-1,1,-1}[corner], tri1: {-1,1,1}[corner]
    # = (corner == 1 || (tri == 1 && corner == 2)) ? 1 : -1
    # Using step: s1 = step(0.5, corner) * step(corner, 1.5) -> 1 if corner==1
    #             s2 = step(0.5, tri) * step(1.5, corner) -> 1 if tri>=1 && corner>=2
    #             ox = (s1 + s2 - s1*s2) * 2.0 - 1.0  (OR logic via inclusion-exclusion)

    # For corner==1: step(0.5,c)*step(c,1.5) = [c>=0.5 && c<=1.5] = 1.0 when c=1
    body.append(_let("is_c1", "scalar",
        _binop("*",
            _call("step", [_lit("0.5"), _ref("corner")]),
            _call("step", [_ref("corner"), _lit("1.5")]))))

    # For tri==1 && corner==2: step(0.5,tri)*step(1.5,corner)
    body.append(_let("is_t1c2", "scalar",
        _binop("*",
            _call("step", [_lit("0.5"), _ref("tri")]),
            _call("step", [_lit("1.5"), _ref("corner")]))))

    # OR via inclusion-exclusion: a+b-a*b
    body.append(_let("ox_flag", "scalar",
        _binop("-",
            _binop("+", _ref("is_c1"), _ref("is_t1c2")),
            _binop("*", _ref("is_c1"), _ref("is_t1c2")))))
    body.append(_let("quad_x", "scalar",
        _binop("-", _binop("*", _ref("ox_flag"), _lit("2.0")), _lit("1.0"))))

    # Offset y: tri0: {-1,-1,1}[corner], tri1: {1,-1,1}[corner]
    # = (corner == 2 || (tri == 1 && corner == 0)) ? 1 : -1
    body.append(_let("is_c2", "scalar",
        _call("step", [_lit("1.5"), _ref("corner")])))

    # tri==1 && corner==0: step(0.5,tri) * step(corner, 0.5)... corner<0.5
    # step(corner, 0.5) = corner <= 0.5 ? 1 : 0  -> 1 when corner=0
    body.append(_let("is_c0", "scalar",
        _call("step", [_neg(_ref("corner")), _lit("0.5")])))
    # Actually step(x, edge) = 1 if x >= edge, so step(-corner, -0.5) doesn't
    # quite work. Use: is_c0 = 1.0 - step(0.5, corner)
    # When corner=0: step(0.5,0)=0 → 1-0=1. When corner>=1: step(0.5,c)=1 → 0.

    # Fix is_c0:
    body.pop()  # remove wrong is_c0
    body.append(_let("is_c0", "scalar",
        _binop("-", _lit("1.0"), _call("step", [_lit("0.5"), _ref("corner")]))))

    body.append(_let("is_t1c0", "scalar",
        _binop("*", _call("step", [_lit("0.5"), _ref("tri")]), _ref("is_c0"))))

    body.append(_let("oy_flag", "scalar",
        _binop("-",
            _binop("+", _ref("is_c2"), _ref("is_t1c0")),
            _binop("*", _ref("is_c2"), _ref("is_t1c0")))))
    body.append(_let("quad_y", "scalar",
        _binop("-", _binop("*", _ref("oy_flag"), _lit("2.0")), _lit("1.0"))))

    # --- Oriented quad offset (see preprocess stage's docstring note) ---
    # offset = quad_x*major + quad_y*minor -- an oriented rectangle exactly
    # circumscribing the covariance ellipse at the opacity-tight extent,
    # instead of the old axis-aligned vec2(quad_x,quad_y)*radius square.
    body.append(_let("offset", "vec2",
        _binop("+",
            _binop("*", _ref("major_vec"), _ref("quad_x")),
            _binop("*", _ref("minor_vec"), _ref("quad_y")))))

    # Screen-pixel center from NDC:
    # pixel_center = (ndc_center * 0.5 + 0.5) * screen_size
    body.append(_let("screen_w", "scalar",
        _swizzle(_push_field("screen_size"), "x")))
    body.append(_let("screen_h", "scalar",
        _swizzle(_push_field("screen_size"), "y")))

    body.append(_let("pixel_center", "vec2",
        _binop("*",
            _binop("+",
                _binop("*", _ref("ndc_center"), _lit("0.5")),
                _ctor("vec2", [_lit("0.5"), _lit("0.5")])),
            _ctor("vec2", [_ref("screen_w"), _ref("screen_h")]))))

    # Final vertex position in NDC:
    # pixel_pos = pixel_center + offset;
    # ndc_pos = (pixel_pos / screen_size) * 2.0 - 1.0;
    body.append(_let("pixel_pos", "vec2",
        _binop("+", _ref("pixel_center"), _ref("offset"))))

    body.append(_let("ndc_pos", "vec2",
        _binop("-",
            _binop("*",
                _binop("/", _ref("pixel_pos"),
                    _ctor("vec2", [_ref("screen_w"), _ref("screen_h")])),
                _lit("2.0")),
            _ctor("vec2", [_lit("1.0"), _lit("1.0")]))))

    # Set gl_Position
    body.append(_assign("builtin_position",
        _ctor("vec4", [
            _swizzle(_ref("ndc_pos"), "x"),
            _swizzle(_ref("ndc_pos"), "y"),
            _ref("depth"),
            _lit("1.0"),
        ])))

    # --- Write varying outputs ---
    body.append(_assign("frag_conic",
        _swizzle(_ref("conic_data"), "xyz")))
    body.append(_assign("frag_color", _ref("color_data")))
    body.append(_assign("frag_center", _ref("pixel_center")))
    body.append(_assign("frag_offset", _ref("offset")))
    if config.get("motion_vectors"):
        body.append(_assign("frag_mv", _ref("mv_data")))
    if config.get("expected_depth"):
        body.append(_assign("frag_depth", _ref("depth_data")))
    if config.get("foreground_coverage"):
        body.append(_assign("frag_foreground", _ref("foreground_data")))

    return body


# ---------------------------------------------------------------------------
# Stage 3: render fragment shader
# ---------------------------------------------------------------------------

def _build_fragment_stage(config: dict) -> StageBlock:
    """Generate the Gaussian splat fragment stage.

    The fragment shader evaluates the 2D Gaussian function at each pixel
    using the inverse covariance (conic) and the pixel offset from the
    splat center.  Fragments with alpha below the cutoff are discarded.
    Output is premultiplied-alpha color.
    """
    stage = StageBlock(stage_type="fragment")

    # --- Fragment inputs (from vertex stage) ---
    in_vars = [("frag_conic", "vec3"), ("frag_color", "vec4"),
               ("frag_center", "vec2"), ("frag_offset", "vec2")]
    if config.get("motion_vectors"):
        in_vars.append(("frag_mv", "vec2"))
    if config.get("expected_depth"):
        in_vars.append(("frag_depth", "scalar"))
    if config.get("foreground_coverage"):
        in_vars.append(("frag_foreground", "scalar"))
    for name, ty in in_vars:
        v = VarDecl(name, ty)
        v._is_input = True
        stage.inputs.append(v)

    # --- Fragment output ---
    out = VarDecl("out_color", "vec4")
    out._is_input = False
    stage.outputs.append(out)
    # Additional DLSS input-contract outputs (docs/lux-4d-spec.md section 3),
    # PACKED into `out_aux` (mv.x*alpha, mv.y*alpha, depth*alpha, **alpha**)
    # -- emitted whenever motion_vectors OR expected_depth is set -- and, iff
    # foreground_coverage is ALSO set, a second attachment `out_fg`
    # (fg*alpha, 0, 0, **alpha**). Both use the SAME premultiplied-alpha
    # blend equation/factors as out_color (fixed-function
    # (ONE, ONE_MINUS_SRC_ALPHA)).
    #
    # IMPORTANT (bench/lux_perf_ablation.md task 2 -- this replaced an
    # EARLIER, BROKEN attempt at this task that packed `fg*alpha` into
    # `out_aux`'s `.w` lane and dropped alpha entirely, reasoning it could
    # be recovered from `out_color`'s own alpha instead): Vulkan's
    # fixed-function `SRC_ALPHA`/`ONE_MINUS_SRC_ALPHA` blend factors read
    # "source alpha" from THAT SPECIFIC ATTACHMENT's own 4th output
    # component -- NOT a globally shared value, and NOT `out_color`'s alpha
    # -- so whatever value sits in `out_aux.w` directly controls how much
    # EVERY overlapping fragment's contribution decays into this
    # attachment. Packing `fg*alpha` there (which is 0 for the common
    # "background, non-actor" case) made `ONE_MINUS_SRC_ALPHA` evaluate to
    # 1 for most fragments -- i.e. NO decay, an unbounded running SUM
    # instead of a proper "over" composite -- and blew up MV/depth by
    # 10-50x specifically in heavily-overdrawn background regions (measured
    # on the juggle DLSS scene: MV median error 39.9px vs. the ~0.002px the
    # old two-attachment design achieved). This is the SAME class of bug
    # `out_depth`'s original vec4-not-vec2 fix (see the historical comment
    # this replaced) was created to prevent -- it just wasn't obvious it
    # also applies to what value OCCUPIES an existing alpha lane, not only
    # whether one exists. `out_color`'s own alpha remains a fine
    # UN-PREMULTIPLY divisor host-side (that math doesn't care which
    # attachment produced the alpha value, since they're all computed from
    # the identical per-fragment `alpha`) -- the bug is specifically about
    # what the GPU'S OWN fixed-function blend unit reads per attachment
    # while accumulating, which is a completely different, per-attachment,
    # structural requirement. Each blended attachment MUST carry its own
    # genuine (undiluted) alpha in its last component, full stop.
    #
    # Net effect vs. the original two-attachment (out_motion, out_depth,
    # each RGBA32F with a wasted always-0 `.z`) design: `out_aux` now packs
    # 3 real values (mv.xy, depth) with zero wasted lanes (attachment count
    # 2->1, bytes 32B->16B) whenever foreground_coverage is off; when it's
    # on, `out_fg` is a second, mostly-wasted-but-safe-to-shrink attachment
    # (see the host's format choice for it) -- attachment count stays 2,
    # but total bytes still drop (see splat_renderer.h's getAuxFormat()/
    # getFgFormat() comments for the exact host-side format choices and
    # measured precision).
    if config.get("motion_vectors") or config.get("expected_depth"):
        out_aux = VarDecl("out_aux", "vec4")
        out_aux._is_input = False
        stage.outputs.append(out_aux)
    if config.get("foreground_coverage"):
        out_fg = VarDecl("out_fg", "vec4")
        out_fg._is_input = False
        stage.outputs.append(out_fg)

    # --- Push constants (shared block with vertex to avoid Vulkan offset conflicts) ---
    pc_fields = [
        BlockField("screen_size", "vec2"),
        BlockField("visible_count", "uint"),
        BlockField("alpha_min", "scalar"),
    ]
    stage.push_constants.append(PushBlock("push", pc_fields))

    # --- Main function body ---
    body = _build_fragment_body(config)
    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


def _build_fragment_body(config: dict) -> list:
    """Generate the fragment shader main() body."""
    body = []

    # --- Evaluate Gaussian weight ---
    # The conic is the inverse covariance matrix (symmetric 2x2):
    #   conic = (a, b, c)  →  [[a, b], [b, c]]
    # The Gaussian exponent:
    #   power = -0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2)
    # where (dx, dy) = frag_offset (pixel offset from splat center)

    body.append(_let("conic", "vec3", _ref("frag_conic")))
    body.append(_let("d", "vec2", _ref("frag_offset")))

    body.append(_let("dx", "scalar", _swizzle(_ref("d"), "x")))
    body.append(_let("dy", "scalar", _swizzle(_ref("d"), "y")))
    body.append(_let("a", "scalar", _swizzle(_ref("conic"), "x")))
    body.append(_let("b", "scalar", _swizzle(_ref("conic"), "y")))
    body.append(_let("c", "scalar", _swizzle(_ref("conic"), "z")))

    # power = -0.5 * (a*dx*dx + 2*b*dx*dy + c*dy*dy)
    body.append(_let("power", "scalar",
        _binop("*", _lit("-0.5"),
            _binop("+",
                _binop("+",
                    _binop("*", _ref("a"), _binop("*", _ref("dx"), _ref("dx"))),
                    _binop("*", _binop("*", _lit("2.0"), _ref("b")),
                        _binop("*", _ref("dx"), _ref("dy")))),
                _binop("*", _ref("c"), _binop("*", _ref("dy"), _ref("dy")))))))

    # power > 0 is numerically invalid for a valid (positive-definite) conic;
    # the actual visibility cutoff is the alpha_min check below, evaluated
    # out to the quad's own 3-sigma extent (see `raw_radius` above) --
    # matching the reference 3DGS/gsplat rasterizer (no separate arbitrary
    # power threshold).
    body.append(_if(
        _binop(">", _ref("power"), _lit("0.0")),
        [DiscardStmt()],
    ))

    # alpha = opacity * exp(-0.5*d^2), clamped to <= 0.99 (reference 3DGS/gsplat)
    body.append(_let("gauss_weight", "scalar", _call("exp", [_ref("power")])))
    body.append(_let("opacity", "scalar",
        _swizzle(_ref("frag_color"), "w")))
    body.append(_let("raw_alpha", "scalar",
        _binop("*", _ref("gauss_weight"), _ref("opacity"))))
    body.append(_let("alpha", "scalar",
        _call("min", [_ref("raw_alpha"), _lit("0.99")])))

    # alpha_min cutoff (discard nearly-transparent fragments; default 1/255,
    # matching 3DGS/gsplat)
    body.append(_if(
        _binop("<", _ref("alpha"), _push_field("alpha_min")),
        [DiscardStmt()],
    ))

    # --- Premultiplied alpha output ---
    # out_color = vec4(color.rgb * alpha, alpha)
    body.append(_let("rgb", "vec3",
        _swizzle(_ref("frag_color"), "xyz")))

    # Optional sRGB conversion
    if config.get("color_space") == "srgb":
        # Linear → sRGB approximation: pow(c, 1/2.2)
        # For accuracy we use the standard piecewise, but in a splat renderer
        # the simple pow is sufficient and keeps the AST compact.
        body.append(_let("srgb_r", "scalar",
            _call("pow", [_swizzle(_ref("rgb"), "x"), _lit("0.45454545")])))
        body.append(_let("srgb_g", "scalar",
            _call("pow", [_swizzle(_ref("rgb"), "y"), _lit("0.45454545")])))
        body.append(_let("srgb_b", "scalar",
            _call("pow", [_swizzle(_ref("rgb"), "z"), _lit("0.45454545")])))
        body.append(_let("final_rgb", "vec3",
            _ctor("vec3", [_ref("srgb_r"), _ref("srgb_g"), _ref("srgb_b")])))
    else:
        body.append(_let("final_rgb", "vec3", _ref("rgb")))

    body.append(_assign("out_color",
        _ctor("vec4", [
            _binop("*", _swizzle(_ref("final_rgb"), "x"), _ref("alpha")),
            _binop("*", _swizzle(_ref("final_rgb"), "y"), _ref("alpha")),
            _binop("*", _swizzle(_ref("final_rgb"), "z"), _ref("alpha")),
            _ref("alpha"),
        ])))

    # --- Additional DLSS input-contract outputs, packed into `out_aux`
    # (mv.x*alpha, mv.y*alpha, depth*alpha, alpha) and, iff
    # foreground_coverage, a second `out_fg` (fg*alpha, 0, 0, alpha) --
    # premultiplied by the same alpha as out_color so the fixed-function
    # (ONE, ONE_MINUS_SRC_ALPHA) blend also does the correct visibility-
    # weighted average for each packed channel. Each attachment's `.w`
    # carries the GENUINE per-fragment alpha (not a packed data value) --
    # see the output-declaration comment above for why this is a hardware
    # blend-correctness requirement, not merely a host-precision nicety. ---
    if config.get("motion_vectors") or config.get("expected_depth"):
        mv_x_term = (_binop("*", _swizzle(_ref("frag_mv"), "x"), _ref("alpha"))
                     if config.get("motion_vectors") else _lit("0.0"))
        mv_y_term = (_binop("*", _swizzle(_ref("frag_mv"), "y"), _ref("alpha"))
                     if config.get("motion_vectors") else _lit("0.0"))
        depth_term = (_binop("*", _ref("frag_depth"), _ref("alpha"))
                      if config.get("expected_depth") else _lit("0.0"))
        body.append(_assign("out_aux",
            _ctor("vec4", [mv_x_term, mv_y_term, depth_term, _ref("alpha")])))
    if config.get("foreground_coverage"):
        body.append(_assign("out_fg",
            _ctor("vec4", [
                _binop("*", _ref("frag_foreground"), _ref("alpha")),
                _lit("0.0"),
                _lit("0.0"),
                _ref("alpha"),
            ])))

    return body


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def expand_splat_pipeline(
    splat: SplatDecl,
    pipeline,  # PipelineDecl
    module: Module,
) -> list[StageBlock]:
    """Expand a splat declaration + pipeline into three shader stages.

    Parameters
    ----------
    splat : SplatDecl
        The parsed ``splat`` declaration containing splatting configuration
        (SH degree, kernel type, alpha cutoff, etc.).
    pipeline : PipelineDecl
        The pipeline declaration that references this splat (mode:
        ``gaussian_splat``).  Currently used for future extensions
        (e.g., custom sort, additional pipeline members).
    module : Module
        The compilation module.  Used to attach ``_defines`` for the
        compute workgroup size so the SPIR-V backend can emit the correct
        ``OpExecutionMode LocalSize``.

    Returns
    -------
    list[StageBlock]
        Three stages: [preprocess_compute, render_vertex, render_fragment].
    """
    config = _get_splat_config(splat)
    sh_degree = config["sh_degree"]

    # Validate SH degree
    if sh_degree not in (0, 1, 2, 3):
        raise ValueError(
            f"Unsupported SH degree {sh_degree} in splat '{splat.name}'. "
            f"Valid values are 0, 1, 2, or 3."
        )

    # --- Stage 1: preprocess compute ---
    compute_stage = _build_preprocess_stage(config)

    # Attach workgroup size defines so the SPIR-V backend can emit
    # OpExecutionMode LocalSize 256 1 1.
    if not hasattr(module, '_defines'):
        module._defines = {}
    module._defines["workgroup_size_x"] = 256
    module._defines["workgroup_size_y"] = 1
    module._defines["workgroup_size_z"] = 1

    # Tag the compute stage with the splat config for reflection
    compute_stage._splat_config = config
    compute_stage._splat_name = splat.name

    # --- Stage 2: render vertex ---
    vertex_stage = _build_vertex_stage(config)
    vertex_stage._splat_name = splat.name

    # --- Stage 3: render fragment ---
    fragment_stage = _build_fragment_stage(config)
    fragment_stage._splat_name = splat.name

    stages = [compute_stage, vertex_stage, fragment_stage]

    # --- Optional stage 0: morph-apply compute (motion: keyframes) ---
    if config.get("motion") == "keyframes":
        morph_stage = _build_morph_apply_stage(config)
        # Distinct output filename (test_splat.morph.comp.spv) so it doesn't
        # collide with the preprocess compute stage's test_splat.comp.spv --
        # same convention deferred_expander.py uses for gbuf/light passes.
        morph_stage._output_stem_suffix = "morph"
        morph_stage._splat_name = splat.name
        morph_stage._splat_motion_config = config
        stages.insert(0, morph_stage)

    return stages


# ---------------------------------------------------------------------------
# RT Gaussian Splatting (3DGRT)
# ---------------------------------------------------------------------------

def expand_splat_rt_pipeline(
    splat: SplatDecl,
    pipeline,  # PipelineDecl
    module: Module,
) -> list[StageBlock]:
    """Expand a splat declaration + RT pipeline into four RT shader stages.

    Uses the 3DGRT algorithm: analytical ray-Gaussian intersection with
    multi-round closest-hit accumulation in the raygen shader.

    Returns [intersection, closest_hit, miss, raygen].
    """
    config = _get_splat_config(splat)
    sh_degree = config["sh_degree"]

    if sh_degree not in (0, 1, 2, 3):
        raise ValueError(
            f"Unsupported SH degree {sh_degree} in splat '{splat.name}'. "
            f"Valid values are 0, 1, 2, or 3."
        )

    intersection = _build_rt_intersection_stage(config)
    closest_hit = _build_rt_closest_hit_stage()
    miss = _build_rt_miss_stage()
    raygen = _build_rt_raygen_stage(config)

    # Tag all stages with splat metadata for reflection.
    # RT splat uses two descriptor sets: set 0 for raygen (camera, tlas,
    # image, SH buffers), set 1 for intersection (splat geometry SSBOs).
    # closest_hit and miss have no descriptors, but assign set 0 so the
    # layout assigner doesn't auto-assign conflicting set numbers.
    for stage in [intersection, closest_hit, miss, raygen]:
        stage._splat_config = config
        stage._splat_name = splat.name
    raygen._descriptor_set_offset = 0
    intersection._descriptor_set_offset = 1
    closest_hit._descriptor_set_offset = 0
    miss._descriptor_set_offset = 0

    return [intersection, closest_hit, miss, raygen]


def _build_rt_intersection_stage(config: dict) -> StageBlock:
    """Generate the intersection shader for 3DGRT ray-Gaussian test.

    Reads Gaussian parameters from SSBOs, transforms ray into local space,
    computes analytical peak response tau_max, evaluates density, and
    reports intersection if alpha exceeds cutoff.
    """
    stage = StageBlock(stage_type="intersection")

    # Storage buffers for Gaussian attributes (read-only)
    stage.storage_buffers.append(StorageBufferDecl("splat_pos", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("splat_rot", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("splat_scale", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("splat_opacity", "scalar"))

    # Hit attribute: pass alpha to closest-hit shader
    stage.hit_attributes.append(HitAttributeDecl("hit_alpha", "scalar"))

    body = []

    # --- Read Gaussian parameters using primitive_id ---
    body.append(_let("gid", "uint", _ref("primitive_id")))
    body.append(_let("center", "vec3",
        _swizzle(_idx("splat_pos", _ref("gid")), "xyz")))
    body.append(_let("raw_opacity", "scalar",
        _idx("splat_opacity", _ref("gid"))))
    # Sigmoid activation: opacity = 1.0 / (1.0 + exp(-raw_opacity))
    body.append(_let("opacity", "scalar",
        _binop("/", _lit("1.0"),
               _binop("+", _lit("1.0"),
                      _call("exp", [_neg(_ref("raw_opacity"))])))))

    # --- Read rotation quaternion and build rotation matrix (R) ---
    body.append(_let("quat", "vec4", _idx("splat_rot", _ref("gid"))))
    body.append(_let("qr", "scalar", _swizzle(_ref("quat"), "w")))
    body.append(_let("qi", "scalar", _swizzle(_ref("quat"), "x")))
    body.append(_let("qj", "scalar", _swizzle(_ref("quat"), "y")))
    body.append(_let("qk", "scalar", _swizzle(_ref("quat"), "z")))

    # Precompute quadratic products
    body.append(_let("qi2", "scalar", _binop("*", _ref("qi"), _ref("qi"))))
    body.append(_let("qj2", "scalar", _binop("*", _ref("qj"), _ref("qj"))))
    body.append(_let("qk2", "scalar", _binop("*", _ref("qk"), _ref("qk"))))
    body.append(_let("qri", "scalar", _binop("*", _ref("qr"), _ref("qi"))))
    body.append(_let("qrj", "scalar", _binop("*", _ref("qr"), _ref("qj"))))
    body.append(_let("qrk", "scalar", _binop("*", _ref("qr"), _ref("qk"))))
    body.append(_let("qij", "scalar", _binop("*", _ref("qi"), _ref("qj"))))
    body.append(_let("qik", "scalar", _binop("*", _ref("qi"), _ref("qk"))))
    body.append(_let("qjk", "scalar", _binop("*", _ref("qj"), _ref("qk"))))

    # Rotation matrix rows (same pattern as _build_preprocess_body)
    body.append(_let("rot_00", "scalar",
        _binop("-", _lit("1.0"),
               _binop("*", _lit("2.0"),
                      _binop("+", _ref("qj2"), _ref("qk2"))))))
    body.append(_let("rot_01", "scalar",
        _binop("*", _lit("2.0"),
               _binop("-", _ref("qij"), _ref("qrk")))))
    body.append(_let("rot_02", "scalar",
        _binop("*", _lit("2.0"),
               _binop("+", _ref("qik"), _ref("qrj")))))
    body.append(_let("rot_10", "scalar",
        _binop("*", _lit("2.0"),
               _binop("+", _ref("qij"), _ref("qrk")))))
    body.append(_let("rot_11", "scalar",
        _binop("-", _lit("1.0"),
               _binop("*", _lit("2.0"),
                      _binop("+", _ref("qi2"), _ref("qk2"))))))
    body.append(_let("rot_12", "scalar",
        _binop("*", _lit("2.0"),
               _binop("-", _ref("qjk"), _ref("qri")))))
    body.append(_let("rot_20", "scalar",
        _binop("*", _lit("2.0"),
               _binop("-", _ref("qik"), _ref("qrj")))))
    body.append(_let("rot_21", "scalar",
        _binop("*", _lit("2.0"),
               _binop("+", _ref("qjk"), _ref("qri")))))
    body.append(_let("rot_22", "scalar",
        _binop("-", _lit("1.0"),
               _binop("*", _lit("2.0"),
                      _binop("+", _ref("qi2"), _ref("qj2"))))))

    # --- Read scale (log-scale -> exp) ---
    body.append(_let("log_scale", "vec3",
        _swizzle(_idx("splat_scale", _ref("gid")), "xyz")))
    body.append(_let("sx", "scalar",
        _call("exp", [_swizzle(_ref("log_scale"), "x")])))
    body.append(_let("sy", "scalar",
        _call("exp", [_swizzle(_ref("log_scale"), "y")])))
    body.append(_let("sz", "scalar",
        _call("exp", [_swizzle(_ref("log_scale"), "z")])))

    # --- Transform ray to Gaussian-local space ---
    # o_local = R^T * (ray_origin - center)
    body.append(_let("ro", "vec3",
        _binop("-", _ref("world_ray_origin"), _ref("center"))))

    # R^T * ro (transpose multiply: dot each row of R^T = each column of R)
    def _dot3(ax, ay, az, bx, by, bz):
        return _binop("+",
            _binop("+",
                _binop("*", _ref(ax), _ref(bx)),
                _binop("*", _ref(ay), _ref(by))),
            _binop("*", _ref(az), _ref(bz)))

    body.append(_let("o_lx", "scalar",
        _dot3("rot_00", "rot_10", "rot_20", "ro_x", "ro_y", "ro_z")))
    # Need to extract ro components first
    # Actually, let me extract components before the dot products
    body.pop()  # remove premature o_lx

    body.append(_let("ro_x", "scalar", _swizzle(_ref("ro"), "x")))
    body.append(_let("ro_y", "scalar", _swizzle(_ref("ro"), "y")))
    body.append(_let("ro_z", "scalar", _swizzle(_ref("ro"), "z")))

    # o_local = R^T * ro
    body.append(_let("o_lx", "scalar",
        _dot3("rot_00", "rot_10", "rot_20", "ro_x", "ro_y", "ro_z")))
    body.append(_let("o_ly", "scalar",
        _dot3("rot_01", "rot_11", "rot_21", "ro_x", "ro_y", "ro_z")))
    body.append(_let("o_lz", "scalar",
        _dot3("rot_02", "rot_12", "rot_22", "ro_x", "ro_y", "ro_z")))

    # d_local = R^T * ray_direction
    body.append(_let("rd_x", "scalar",
        _swizzle(_ref("world_ray_direction"), "x")))
    body.append(_let("rd_y", "scalar",
        _swizzle(_ref("world_ray_direction"), "y")))
    body.append(_let("rd_z", "scalar",
        _swizzle(_ref("world_ray_direction"), "z")))

    body.append(_let("d_lx", "scalar",
        _dot3("rot_00", "rot_10", "rot_20", "rd_x", "rd_y", "rd_z")))
    body.append(_let("d_ly", "scalar",
        _dot3("rot_01", "rot_11", "rot_21", "rd_x", "rd_y", "rd_z")))
    body.append(_let("d_lz", "scalar",
        _dot3("rot_02", "rot_12", "rot_22", "rd_x", "rd_y", "rd_z")))

    # --- Scale to unit Gaussian: o_g = o_local / S, d_g = d_local / S ---
    body.append(_let("o_gx", "scalar", _binop("/", _ref("o_lx"), _ref("sx"))))
    body.append(_let("o_gy", "scalar", _binop("/", _ref("o_ly"), _ref("sy"))))
    body.append(_let("o_gz", "scalar", _binop("/", _ref("o_lz"), _ref("sz"))))
    body.append(_let("d_gx", "scalar", _binop("/", _ref("d_lx"), _ref("sx"))))
    body.append(_let("d_gy", "scalar", _binop("/", _ref("d_ly"), _ref("sy"))))
    body.append(_let("d_gz", "scalar", _binop("/", _ref("d_lz"), _ref("sz"))))

    # --- Compute tau_max = -dot(o_g, d_g) / dot(d_g, d_g) ---
    body.append(_let("od_dot", "scalar",
        _binop("+",
            _binop("+",
                _binop("*", _ref("o_gx"), _ref("d_gx")),
                _binop("*", _ref("o_gy"), _ref("d_gy"))),
            _binop("*", _ref("o_gz"), _ref("d_gz")))))
    body.append(_let("dd_dot", "scalar",
        _binop("+",
            _binop("+",
                _binop("*", _ref("d_gx"), _ref("d_gx")),
                _binop("*", _ref("d_gy"), _ref("d_gy"))),
            _binop("*", _ref("d_gz"), _ref("d_gz")))))

    body.append(_let("tau_max", "scalar",
        _binop("/", _neg(_ref("od_dot")), _ref("dd_dot"))))

    # --- Evaluate density at peak ---
    # p = o_g + tau_max * d_g
    body.append(_let("p_x", "scalar",
        _binop("+", _ref("o_gx"), _binop("*", _ref("tau_max"), _ref("d_gx")))))
    body.append(_let("p_y", "scalar",
        _binop("+", _ref("o_gy"), _binop("*", _ref("tau_max"), _ref("d_gy")))))
    body.append(_let("p_z", "scalar",
        _binop("+", _ref("o_gz"), _binop("*", _ref("tau_max"), _ref("d_gz")))))

    # rho = exp(-0.5 * dot(p, p))
    body.append(_let("p_dot", "scalar",
        _binop("+",
            _binop("+",
                _binop("*", _ref("p_x"), _ref("p_x")),
                _binop("*", _ref("p_y"), _ref("p_y"))),
            _binop("*", _ref("p_z"), _ref("p_z")))))
    body.append(_let("rho", "scalar",
        _call("exp", [_binop("*", _lit("-0.5"), _ref("p_dot"))])))

    # alpha = opacity * rho
    body.append(_let("alpha", "scalar",
        _binop("*", _ref("opacity"), _ref("rho"))))

    # --- Report intersection if alpha > cutoff ---
    alpha_cutoff = config["alpha_cutoff"]
    body.append(_if(
        _binop(">", _ref("alpha"), _lit(str(alpha_cutoff))),
        [
            _assign("hit_alpha", _ref("alpha")),
            ExprStmt(CallExpr(VarRef("report_intersection"), [
                _ref("tau_max"),
                _uint_lit("0"),  # hit_kind
            ])),
        ],
    ))

    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


def _build_rt_closest_hit_stage() -> StageBlock:
    """Generate the closest-hit shader for RT Gaussian splatting.

    Packs hit data into the vec4 payload:
      .x = hit_t, .y = float(primitive_id), .z = alpha, .w = 1.0
    """
    stage = StageBlock(stage_type="closest_hit")

    # Incoming ray payload
    stage.ray_payloads.append(RayPayloadDecl("payload", "vec4"))

    # Hit attribute from intersection shader
    stage.hit_attributes.append(HitAttributeDecl("hit_alpha", "scalar"))

    body = []

    # Pack hit data into payload
    body.append(_assign("payload",
        _ctor("vec4", [
            _ref("hit_t"),
            _binop("*", _ref("primitive_id"), _lit("1.0")),  # uint -> float
            _ref("hit_alpha"),
            _lit("1.0"),  # hit indicator
        ])))

    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


def _build_rt_miss_stage() -> StageBlock:
    """Generate the miss shader for RT Gaussian splatting.

    Writes vec4(0) to payload (w=0 signals miss).
    """
    stage = StageBlock(stage_type="miss")

    # Incoming ray payload
    stage.ray_payloads.append(RayPayloadDecl("payload", "vec4"))

    body = []
    body.append(_assign("payload",
        _ctor("vec4", [_lit("0.0"), _lit("0.0"), _lit("0.0"), _lit("0.0")])))

    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


def _build_rt_raygen_stage(config: dict) -> StageBlock:
    """Generate the raygen shader for RT Gaussian splatting.

    Camera ray computation followed by a multi-round trace loop:
    each iteration traces, extracts hit data, evaluates SH for color,
    accumulates with alpha compositing, and advances tmin.
    """
    stage = StageBlock(stage_type="raygen")
    sh_degree = config["sh_degree"]

    # Acceleration structure
    stage.accel_structs.append(AccelDecl("tlas"))

    # Ray payload
    stage.ray_payloads.append(RayPayloadDecl("payload", "vec4"))

    # Camera uniform
    stage.uniforms.append(UniformBlock("Camera", [
        BlockField("inv_view", "mat4"),
        BlockField("inv_proj", "mat4"),
    ]))

    # Output storage image
    stage.storage_images.append(StorageImageDecl("result_color"))

    # SH coefficient buffers (read-only)
    for sh_name in _sh_buffer_names(sh_degree):
        stage.storage_buffers.append(StorageBufferDecl(sh_name, "vec4"))

    # Splat positions (needed for SH direction computation)
    stage.storage_buffers.append(StorageBufferDecl("splat_pos", "vec4"))

    body = []

    # --- Camera ray computation (reuse pattern from surface_expander._expand_raygen) ---
    body.append(LetStmt("pixel", "vec2",
        ConstructorExpr("vec2", [SwizzleAccess(VarRef("launch_id"), "xy")])))
    body.append(LetStmt("dims", "vec2",
        ConstructorExpr("vec2", [SwizzleAccess(VarRef("launch_size"), "xy")])))
    body.append(LetStmt("ndc", "vec2", BinaryOp("-",
        BinaryOp("*",
            BinaryOp("/",
                BinaryOp("+", VarRef("pixel"),
                         ConstructorExpr("vec2", [NumberLit("0.5")])),
                VarRef("dims")),
            NumberLit("2.0")),
        ConstructorExpr("vec2", [NumberLit("1.0")]),
    )))

    # Initialize payload
    body.append(AssignStmt(
        AssignTarget(VarRef("payload")),
        ConstructorExpr("vec4", [NumberLit("0.0")]),
    ))

    # Camera ray from inverse matrices
    body.append(LetStmt("target", "vec4", BinaryOp("*",
        VarRef("inv_proj"),
        ConstructorExpr("vec4", [
            SwizzleAccess(VarRef("ndc"), "x"),
            SwizzleAccess(VarRef("ndc"), "y"),
            NumberLit("1.0"),
            NumberLit("1.0"),
        ]),
    )))
    body.append(LetStmt("direction", "vec4", BinaryOp("*",
        VarRef("inv_view"),
        ConstructorExpr("vec4", [
            CallExpr(VarRef("normalize"), [SwizzleAccess(VarRef("target"), "xyz")]),
            NumberLit("0.0"),
        ]),
    )))
    body.append(LetStmt("origin", "vec3", SwizzleAccess(
        BinaryOp("*", VarRef("inv_view"),
                 ConstructorExpr("vec4", [
                     NumberLit("0.0"), NumberLit("0.0"),
                     NumberLit("0.0"), NumberLit("1.0"),
                 ])),
        "xyz",
    )))
    body.append(LetStmt("dir", "vec3",
        CallExpr(VarRef("normalize"), [
            SwizzleAccess(VarRef("direction"), "xyz")])))

    # --- Accumulation state ---
    body.append(_let("accum_r", "scalar", _lit("0.0")))
    body.append(_let("accum_g", "scalar", _lit("0.0")))
    body.append(_let("accum_b", "scalar", _lit("0.0")))
    body.append(_let("transmittance", "scalar", _lit("1.0")))
    body.append(_let("t_min", "scalar", _lit("0.001")))

    # --- Multi-round trace loop (max 32 iterations) ---
    loop_body = []

    # Reset payload before trace
    loop_body.append(_assign("payload",
        _ctor("vec4", [_lit("0.0"), _lit("0.0"), _lit("0.0"), _lit("0.0")])))

    # trace_ray(tlas, 0, 255, 0, 0, 0, origin, t_min, dir, 10000.0, 0)
    loop_body.append(ExprStmt(CallExpr(VarRef("trace_ray"), [
        VarRef("tlas"),
        NumberLit("0"),     # ray_flags
        NumberLit("255"),   # cull_mask
        NumberLit("0"),     # sbt_offset
        NumberLit("0"),     # sbt_stride
        NumberLit("0"),     # miss_index
        VarRef("origin"),
        VarRef("t_min"),
        VarRef("dir"),
        NumberLit("10000.0"),  # tmax
        NumberLit("0"),     # payload location
    ])))

    # Check for miss: payload.w == 0.0
    loop_body.append(_if(
        _binop("<", _swizzle(_ref("payload"), "w"), _lit("0.5")),
        [BreakStmt()],
    ))

    # Extract hit data
    loop_body.append(_let("hit_t", "scalar", _swizzle(_ref("payload"), "x")))
    loop_body.append(_let("gid", "scalar", _swizzle(_ref("payload"), "y")))
    loop_body.append(_let("alpha", "scalar", _swizzle(_ref("payload"), "z")))

    # --- SH evaluation (direction = ray dir, already normalized) ---
    # cam_dir is the view direction (from camera toward splat)
    loop_body.append(_let("cam_dir", "vec3", _ref("dir")))

    sh_stmts = _build_rt_sh_evaluation(sh_degree)
    loop_body.extend(sh_stmts)

    # --- Alpha compositing ---
    # accum_color += transmittance * alpha * sh_color
    loop_body.append(_assign("accum_r",
        _binop("+", _ref("accum_r"),
            _binop("*", _ref("transmittance"),
                _binop("*", _ref("alpha"),
                    _swizzle(_ref("sh_color"), "x"))))))
    loop_body.append(_assign("accum_g",
        _binop("+", _ref("accum_g"),
            _binop("*", _ref("transmittance"),
                _binop("*", _ref("alpha"),
                    _swizzle(_ref("sh_color"), "y"))))))
    loop_body.append(_assign("accum_b",
        _binop("+", _ref("accum_b"),
            _binop("*", _ref("transmittance"),
                _binop("*", _ref("alpha"),
                    _swizzle(_ref("sh_color"), "z"))))))

    # Update transmittance
    loop_body.append(_assign("transmittance",
        _binop("*", _ref("transmittance"),
            _binop("-", _lit("1.0"), _ref("alpha")))))

    # Early termination on near-opaque
    loop_body.append(_if(
        _binop("<", _ref("transmittance"), _lit("0.003")),
        [BreakStmt()],
    ))

    # Advance t_min past this hit
    loop_body.append(_assign("t_min",
        _binop("+", _ref("hit_t"), _lit("0.0001"))))

    # ForStmt: for (int i = 0; i < 32; i = i + 1) { ... }
    body.append(ForStmt(
        loop_var="i",
        loop_var_type="int",
        init_value=_lit("0"),
        condition=_binop("<", _ref("i"), _lit("32")),
        update_target=AssignTarget(_ref("i")),
        update_value=_binop("+", _ref("i"), _lit("1")),
        body=loop_body,
    ))

    # --- Store result to output image ---
    # Alpha = 1.0 (opaque output); accumulated color already premultiplied.
    # Background pixels (no splat hits) have accum=0, giving opaque black.
    body.append(ExprStmt(CallExpr(VarRef("image_store"), [
        VarRef("result_color"),
        SwizzleAccess(VarRef("launch_id"), "xy"),
        ConstructorExpr("vec4", [
            VarRef("accum_r"),
            VarRef("accum_g"),
            VarRef("accum_b"),
            NumberLit("1.0"),
        ]),
    ])))

    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


def _build_rt_sh_evaluation(sh_degree: int) -> list:
    """Generate SH evaluation statements for the RT raygen shader.

    Unlike _build_sh_evaluation(), this variant:
    - Expects ``cam_dir`` and ``gid`` to already be defined in scope
    - Returns only the SH computation statements + final ``sh_color``
    """
    stmts = []

    # Degree 0: DC term
    stmts.append(_let("sh0_val", "vec3",
        _swizzle(_idx("splat_sh0", _ref("gid")), "xyz")))
    stmts.append(_let("sh_color_0", "vec3",
        _binop("+",
               _binop("*", _lit("0.28209479"), _ref("sh0_val")),
               _ctor("vec3", [_lit("0.5"), _lit("0.5"), _lit("0.5")]))))

    if sh_degree == 0:
        stmts.append(_let("sh_color", "vec3", _ref("sh_color_0")))
        return stmts

    # Direction components
    stmts.append(_let("dx", "scalar", _swizzle(_ref("cam_dir"), "x")))
    stmts.append(_let("dy", "scalar", _swizzle(_ref("cam_dir"), "y")))
    stmts.append(_let("dz", "scalar", _swizzle(_ref("cam_dir"), "z")))

    # Degree 1
    stmts.append(_let("sh1_val", "vec3",
        _swizzle(_idx("splat_sh1", _ref("gid")), "xyz")))
    stmts.append(_let("sh2_val", "vec3",
        _swizzle(_idx("splat_sh2", _ref("gid")), "xyz")))
    stmts.append(_let("sh3_val", "vec3",
        _swizzle(_idx("splat_sh3", _ref("gid")), "xyz")))

    stmts.append(_let("sh_color_1", "vec3",
        _binop("+", _ref("sh_color_0"),
               _binop("+",
                      _binop("+",
                             _binop("*", _lit("-0.48860251"),
                                    _binop("*", _ref("dy"), _ref("sh1_val"))),
                             _binop("*", _lit("0.48860251"),
                                    _binop("*", _ref("dz"), _ref("sh2_val")))),
                      _binop("*", _lit("-0.48860251"),
                             _binop("*", _ref("dx"), _ref("sh3_val")))))))

    if sh_degree == 1:
        stmts.append(_let("sh_color", "vec3", _ref("sh_color_1")))
        return stmts

    # Degree 2
    stmts.append(_let("sh4_val", "vec3",
        _swizzle(_idx("splat_sh4", _ref("gid")), "xyz")))
    stmts.append(_let("sh5_val", "vec3",
        _swizzle(_idx("splat_sh5", _ref("gid")), "xyz")))
    stmts.append(_let("sh6_val", "vec3",
        _swizzle(_idx("splat_sh6", _ref("gid")), "xyz")))
    stmts.append(_let("sh7_val", "vec3",
        _swizzle(_idx("splat_sh7", _ref("gid")), "xyz")))
    stmts.append(_let("sh8_val", "vec3",
        _swizzle(_idx("splat_sh8", _ref("gid")), "xyz")))

    stmts.append(_let("dx2", "scalar", _binop("*", _ref("dx"), _ref("dx"))))
    stmts.append(_let("dy2", "scalar", _binop("*", _ref("dy"), _ref("dy"))))
    stmts.append(_let("dz2", "scalar", _binop("*", _ref("dz"), _ref("dz"))))
    stmts.append(_let("dxy", "scalar", _binop("*", _ref("dx"), _ref("dy"))))
    stmts.append(_let("dyz", "scalar", _binop("*", _ref("dy"), _ref("dz"))))
    stmts.append(_let("dxz", "scalar", _binop("*", _ref("dx"), _ref("dz"))))

    stmts.append(_let("sh_deg2", "vec3",
        _binop("+",
            _binop("+",
                _binop("+",
                    _binop("*", _binop("*", _lit("1.09254843"), _ref("dxy")), _ref("sh4_val")),
                    _binop("*", _binop("*", _lit("-1.09254843"), _ref("dyz")), _ref("sh5_val"))),
                _binop("+",
                    _binop("*", _binop("*", _lit("0.31539157"),
                        _binop("-", _binop("*", _lit("2.0"), _ref("dz2")),
                               _binop("+", _ref("dx2"), _ref("dy2")))),
                        _ref("sh6_val")),
                    _binop("*", _binop("*", _lit("-1.09254843"), _ref("dxz")), _ref("sh7_val")))),
            _binop("*", _binop("*", _lit("0.54627422"),
                _binop("-", _ref("dx2"), _ref("dy2"))),
                _ref("sh8_val")))))

    stmts.append(_let("sh_color_2", "vec3",
        _binop("+", _ref("sh_color_1"), _ref("sh_deg2"))))

    if sh_degree == 2:
        stmts.append(_let("sh_color", "vec3", _ref("sh_color_2")))
        return stmts

    # Degree 3
    for i in range(9, 16):
        stmts.append(_let(f"sh{i}_val", "vec3",
            _swizzle(_idx(f"splat_sh{i}", _ref("gid")), "xyz")))

    stmts.append(_let("dx3", "scalar", _binop("*", _ref("dx2"), _ref("dx"))))
    stmts.append(_let("dy3", "scalar", _binop("*", _ref("dy2"), _ref("dy"))))
    stmts.append(_let("dz3", "scalar", _binop("*", _ref("dz2"), _ref("dz"))))

    stmts.append(_let("sh_deg3_a", "vec3",
        _binop("+",
            _binop("+",
                _binop("*",
                    _binop("*", _lit("-0.59004359"),
                        _binop("*", _ref("dy"),
                            _binop("-", _binop("*", _lit("3.0"), _ref("dx2")), _ref("dy2")))),
                    _ref("sh9_val")),
                _binop("*",
                    _binop("*", _lit("2.89061144"),
                        _binop("*", _ref("dx"), _binop("*", _ref("dy"), _ref("dz")))),
                    _ref("sh10_val"))),
            _binop("*",
                _binop("*", _lit("-0.45704580"),
                    _binop("*", _ref("dy"),
                        _binop("-", _binop("*", _lit("4.0"), _ref("dz2")),
                            _binop("+", _ref("dx2"), _ref("dy2"))))),
                _ref("sh11_val")))))

    stmts.append(_let("sh_deg3_b", "vec3",
        _binop("+",
            _binop("+",
                _binop("*",
                    _binop("*", _lit("0.37317633"),
                        _binop("*", _ref("dz"),
                            _binop("-", _binop("*", _lit("2.0"), _ref("dz2")),
                                _binop("*", _lit("3.0"),
                                    _binop("+", _ref("dx2"), _ref("dy2")))))),
                    _ref("sh12_val")),
                _binop("*",
                    _binop("*", _lit("-0.45704580"),
                        _binop("*", _ref("dx"),
                            _binop("-", _binop("*", _lit("4.0"), _ref("dz2")),
                                _binop("+", _ref("dx2"), _ref("dy2"))))),
                    _ref("sh13_val"))),
            _binop("+",
                _binop("*",
                    _binop("*", _lit("1.44530572"),
                        _binop("*", _ref("dz"),
                            _binop("-", _ref("dx2"), _ref("dy2")))),
                    _ref("sh14_val")),
                _binop("*",
                    _binop("*", _lit("-0.59004359"),
                        _binop("*", _ref("dx"),
                            _binop("-", _ref("dx2"), _binop("*", _lit("3.0"), _ref("dy2"))))),
                    _ref("sh15_val"))))))

    stmts.append(_let("sh_deg3", "vec3",
        _binop("+", _ref("sh_deg3_a"), _ref("sh_deg3_b"))))

    stmts.append(_let("sh_color", "vec3",
        _binop("+", _ref("sh_color_2"), _ref("sh_deg3"))))

    return stmts
