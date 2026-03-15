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
    NumberLit, VarRef, BinaryOp, CallExpr, ConstructorExpr,
    SwizzleAccess, UnaryOp,
    AssignTarget, IndexAccess, FieldAccess,
    StorageBufferDecl, IfStmt, DiscardStmt,
    SplatDecl, SplatMember,
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
        sort        -- sort strategy ("camera_distance" or "depth")
        alpha_cutoff -- minimum alpha for fragment discard
    """
    config = {
        "sh_degree": 0,
        "kernel": "ellipse",
        "color_space": "srgb",
        "sort": "camera_distance",
        "alpha_cutoff": 0.004,
    }
    for m in splat.members:
        if m.name == "sh_degree":
            config["sh_degree"] = int(m.value.value)
        elif m.name in ("kernel", "color_space", "sort"):
            config[m.name] = m.value.name
        elif m.name == "alpha_cutoff":
            config["alpha_cutoff"] = float(m.value.value)
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
        projected_center -- (x_ndc, y_ndc, depth, radius)
        projected_conic  -- upper triangle of inverse 2D covariance
        projected_color  -- evaluated SH color + opacity
        sort_keys        -- depth value for radix sort
        visible_count    -- atomic counter (incremented per visible splat)
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
    stage.storage_buffers.append(StorageBufferDecl("projected_conic", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("projected_color", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("sort_keys", "uint"))
    stage.storage_buffers.append(StorageBufferDecl("sorted_indices", "uint"))
    stage.storage_buffers.append(StorageBufferDecl("visible_count", "uint"))

    # --- Push constants ---
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


def _cull_writes() -> list:
    """Write invisible markers for culled splats (zero radius + zero opacity).

    Without this, culled splats leave uninitialized GPU memory in the projected
    output buffers, causing flickering / garbage in the render pass.
    """
    zero4 = _ctor("vec4", [_lit("0.0"), _lit("0.0"), _lit("0.0"), _lit("0.0")])
    # Culled splats get max uint sort key so they sort to the end.
    # sorted_indices must also be reset to identity — otherwise stale permuted
    # values from a previous frame's sort cause visible splats to be rendered
    # twice (once correctly, once at end via stale reference → "disco" flicker).
    return [
        _assign_idx("projected_center", _ref("gid"), zero4),
        _assign_idx("projected_conic", _ref("gid"), zero4),
        _assign_idx("projected_color", _ref("gid"), zero4),
        _assign_idx("sort_keys", _ref("gid"), _uint_lit("4294967295")),
        _assign_idx("sorted_indices", _ref("gid"), _ref("gid")),
    ]


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
        _cull_writes() + [ReturnStmt(None)],
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
    # where V is the 3x3 view rotation.  Instead of computing world-space cov and
    # then transforming, we transform the rotation columns first:
    #   vr_col_k = V * rot_col_k  (mat4 * vec4 with w=0 to transform direction)
    # Then cov_view = (VR * S) * (VR * S)^T, using the same formula as cov3d
    # but with view-space rotation elements vr_ij.
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

    # --- Project view-space covariance to 2D screen space ---
    # Standard 3DGS Jacobian with positive depth t = -vz:
    #   J = [[focal_x / t, 0, -focal_x * x / t^2],
    #        [0, focal_y / t, -focal_y * y / t^2]]
    # Since cov3d is already in view space, cov2d = J * cov3d * J^T
    body.append(_let("vz", "scalar", _swizzle(_ref("view_pos"), "z")))
    body.append(_let("vx", "scalar", _swizzle(_ref("view_pos"), "x")))
    body.append(_let("vy", "scalar", _swizzle(_ref("view_pos"), "y")))
    # Use positive depth (objects have negative vz, so t = -vz > 0)
    body.append(_let("t", "scalar", _neg(_ref("vz"))))
    body.append(_let("t2", "scalar", _binop("*", _ref("t"), _ref("t"))))
    body.append(_let("fx", "scalar", _push_field("focal_x")))
    body.append(_let("fy", "scalar", _push_field("focal_y")))

    # Jacobian: maps view-space (X-right, Y-up, Z-out) to pixel-space (Y-down).
    # sx = fx * vx / t           → ∂sx/∂vx = fx/t,  ∂sx/∂vz = fx*vx/t²
    # sy = -fy * vy / t (Y-flip) → ∂sy/∂vy = -fy/t, ∂sy/∂vz = -fy*vy/t²
    # Note: reference 3DGS uses Z-forward (tz>0) convention where the signs
    # differ.  Our Z-out convention (vz<0, t=-vz) flips ∂/∂vz signs, and
    # the Y-down pixel convention flips the Y-row signs.
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

    # Determinant of the ORIGINAL 2D covariance (before low-pass filter)
    body.append(_let("det_orig", "scalar",
        _binop("-",
               _binop("*", _ref("cov2d_00"), _ref("cov2d_11")),
               _binop("*", _ref("cov2d_01"), _ref("cov2d_01")))))

    # Add low-pass filter to prevent aliasing on very small splats
    body.append(_let("cov2d_00f", "scalar",
        _binop("+", _ref("cov2d_00"), _lit("0.3"))))
    body.append(_let("cov2d_11f", "scalar",
        _binop("+", _ref("cov2d_11"), _lit("0.3"))))

    # --- Compute inverse covariance (conic) for the fragment shader ---
    # det = cov2d_00f * cov2d_11f - cov2d_01^2
    body.append(_let("det", "scalar",
        _binop("-",
               _binop("*", _ref("cov2d_00f"), _ref("cov2d_11f")),
               _binop("*", _ref("cov2d_01"), _ref("cov2d_01")))))

    # if (det <= 0.0) { return; }  -- degenerate ellipse
    body.append(_if(
        _binop("<=", _ref("det"), _lit("0.0")),
        _cull_writes() + [ReturnStmt(None)],
    ))

    # Opacity compensation: reduce opacity proportionally to how much the
    # low-pass filter enlarged the splat (matches reference 3DGS implementation).
    # compensation = sqrt(max(det_orig / det, 0.0))
    body.append(_let("compensation", "scalar",
        _call("sqrt", [_call("max", [
            _binop("/", _ref("det_orig"), _ref("det")),
            _lit("0.0"),
        ])])))

    body.append(_let("inv_det", "scalar", _binop("/", _lit("1.0"), _ref("det"))))

    # Conic = inverse of 2D covariance (symmetric 2x2).
    # Standard 2x2 inverse: [[c11, -c01], [-c01, c00]] / det
    # The Jacobian already maps to Y-down pixel space (j11 = -fy/t), so the
    # covariance is in pixel-space and the standard inverse formula applies.
    body.append(_let("conic_x", "scalar",
        _binop("*", _ref("cov2d_11f"), _ref("inv_det"))))
    body.append(_let("conic_y", "scalar",
        _binop("*", _neg(_ref("cov2d_01")), _ref("inv_det"))))
    body.append(_let("conic_z", "scalar",
        _binop("*", _ref("cov2d_00f"), _ref("inv_det"))))

    # --- Compute screen-space radius from eigenvalues ---
    # mid = 0.5 * (cov2d_00f + cov2d_11f)
    # half_diff = sqrt(max((mid*mid - det), 0.0))
    # lambda_max = mid + half_diff
    # radius = ceil(3.0 * sqrt(lambda_max))
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
    body.append(_let("raw_radius", "scalar",
        _call("ceil", [_binop("*", _lit("3.0"),
                               _call("sqrt", [_ref("lambda_max")]))])))
    # Clamp radius to prevent huge background splats from dominating
    body.append(_let("radius", "scalar",
        _call("min", [_ref("raw_radius"), _lit("256.0")])))

    # Cull splats whose projected size is excessively large (background sky/noise).
    # Splats covering more than 1/8 of the screen are almost always background.
    body.append(_if(
        _binop(">", _ref("radius"),
               _binop("*", _lit("0.25"),
                      _call("min", [_swizzle(_push_field("screen_size"), "x"),
                                    _swizzle(_push_field("screen_size"), "y")]))),
        _cull_writes() + [ReturnStmt(None)],
    ))

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
        _cull_writes() + [ReturnStmt(None)],
    ))

    # --- Read opacity (sigmoid activation) ---
    # let raw_opacity = splat_opacity[gid];
    # let opacity = 1.0 / (1.0 + exp(-raw_opacity));
    body.append(_let("raw_opacity", "scalar",
        _idx("splat_opacity", _ref("gid"))))
    body.append(_let("raw_sigmoid", "scalar",
        _binop("/", _lit("1.0"),
               _binop("+", _lit("1.0"),
                      _call("exp", [_neg(_ref("raw_opacity"))])))))
    # Apply anti-aliasing compensation: reduce opacity for splats enlarged by filter
    body.append(_let("opacity", "scalar",
        _binop("*", _ref("raw_sigmoid"), _ref("compensation"))))

    # --- Cull low-opacity splats early (before SH eval) ---
    # Splats with opacity < 1/255 after compensation are invisible.
    body.append(_if(
        _binop("<", _ref("opacity"), _lit("0.00392157")),
        _cull_writes() + [ReturnStmt(None)],
    ))

    # --- Evaluate spherical harmonics for view-dependent color ---
    body.extend(_build_sh_evaluation(sh_degree))

    # --- Clamp color to [0,1] and pack with opacity ---
    body.append(_let("clamped_r", "scalar",
        _call("clamp", [_swizzle(_ref("sh_color"), "x"), _lit("0.0"), _lit("1.0")])))
    body.append(_let("clamped_g", "scalar",
        _call("clamp", [_swizzle(_ref("sh_color"), "y"), _lit("0.0"), _lit("1.0")])))
    body.append(_let("clamped_b", "scalar",
        _call("clamp", [_swizzle(_ref("sh_color"), "z"), _lit("0.0"), _lit("1.0")])))

    # --- Write output buffers ---
    body.append(_assign_idx("projected_center", _ref("gid"),
        _ctor("vec4", [_ref("ndc_x"), _ref("ndc_y"), _ref("ndc_z"), _ref("radius")])))

    body.append(_assign_idx("projected_conic", _ref("gid"),
        _ctor("vec4", [_ref("conic_x"), _ref("conic_y"), _ref("conic_z"), _ref("opacity")])))

    body.append(_assign_idx("projected_color", _ref("gid"),
        _ctor("vec4", [_ref("clamped_r"), _ref("clamped_g"), _ref("clamped_b"), _ref("opacity")])))

    # --- Sort key: convert float depth to sortable uint for GPU radix sort ---
    # float_bits_to_uint gives the IEEE 754 bit pattern.  Negative floats have
    # reversed ordering in uint, so we flip: negative → ~bits, positive → flip
    # sign bit.  Result: uint comparison preserves float ordering (ascending).
    body.append(_let("key_bits", "uint", _call("float_bits_to_uint", [_ref("vz")])))
    # sign_mask = (key_bits >> 31) * 0xFFFFFFFF  (all-ones if negative)
    body.append(_let("sign_mask", "uint",
        _binop("*",
            _binop(">>", _ref("key_bits"), _uint_lit("31")),
            _uint_lit("4294967295"))))
    # sort_key = key_bits ^ (sign_mask | 0x80000000)
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

    # Direction from camera to splat (matches 3DGS reference convention).
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

def _build_vertex_stage() -> StageBlock:
    """Generate the render vertex stage.

    This is an instanced draw with 6 vertices per instance (two triangles
    forming a screen-aligned quad).  Each instance corresponds to a sorted
    visible splat.

    Inputs (storage buffers from the preprocess pass):
        projected_center -- (ndc_x, ndc_y, depth, radius)
        projected_conic  -- (inv_cov_a, inv_cov_b, inv_cov_c, opacity)
        projected_color  -- (r, g, b, opacity)
        sorted_indices   -- indirection table from radix sort

    Outputs (to fragment stage):
        frag_conic   -- vec3: inverse 2D covariance upper triangle
        frag_color   -- vec4: splat color + opacity
        frag_center  -- vec2: screen-pixel center of the splat
        frag_offset  -- vec2: pixel offset from center for this vertex
    """
    stage = StageBlock(stage_type="vertex")

    # --- Storage buffers (read-only) ---
    stage.storage_buffers.append(StorageBufferDecl("projected_center", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("projected_conic", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("projected_color", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("sorted_indices", "uint"))

    # --- Push constants (shared block with fragment to avoid Vulkan offset conflicts) ---
    pc_fields = [
        BlockField("screen_size", "vec2"),
        BlockField("visible_count", "uint"),
        BlockField("alpha_cutoff", "scalar"),
    ]
    stage.push_constants.append(PushBlock("push", pc_fields))

    # --- Vertex outputs → fragment inputs ---
    for name, ty in [("frag_conic", "vec3"), ("frag_color", "vec4"),
                     ("frag_center", "vec2"), ("frag_offset", "vec2")]:
        v = VarDecl(name, ty)
        v._is_input = False
        stage.outputs.append(v)

    # --- Main function body ---
    body = _build_vertex_body()
    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


def _build_vertex_body() -> list:
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
    body.append(_let("conic_data", "vec4",
        _idx("projected_conic", _ref("splat_idx"))))
    body.append(_let("color_data", "vec4",
        _idx("projected_color", _ref("splat_idx"))))

    # Unpack center / radius
    body.append(_let("ndc_center", "vec2",
        _swizzle(_ref("center_data"), "xy")))
    body.append(_let("depth", "scalar",
        _swizzle(_ref("center_data"), "z")))
    body.append(_let("radius", "scalar",
        _swizzle(_ref("center_data"), "w")))

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

    # --- Compute eigenvalues from conic to get splat extent ---
    # We already have the radius packed in center_data.w, so use that directly.
    # offset = vec2(quad_x, quad_y) * radius
    body.append(_let("offset", "vec2",
        _binop("*",
            _ctor("vec2", [_ref("quad_x"), _ref("quad_y")]),
            _ref("radius"))))

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
    for name, ty in [("frag_conic", "vec3"), ("frag_color", "vec4"),
                     ("frag_center", "vec2"), ("frag_offset", "vec2")]:
        v = VarDecl(name, ty)
        v._is_input = True
        stage.inputs.append(v)

    # --- Fragment output ---
    out = VarDecl("out_color", "vec4")
    out._is_input = False
    stage.outputs.append(out)

    # --- Push constants (shared block with vertex to avoid Vulkan offset conflicts) ---
    pc_fields = [
        BlockField("screen_size", "vec2"),
        BlockField("visible_count", "uint"),
        BlockField("alpha_cutoff", "scalar"),
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

    # Discard fragments outside the Gaussian tail (power < -4.0 is negligible)
    body.append(_if(
        _binop("<", _ref("power"), _lit("-4.0")),
        [DiscardStmt()],
    ))

    # alpha = exp(power) * opacity
    body.append(_let("gauss_weight", "scalar", _call("exp", [_ref("power")])))
    body.append(_let("opacity", "scalar",
        _swizzle(_ref("frag_color"), "w")))
    body.append(_let("alpha", "scalar",
        _binop("*", _ref("gauss_weight"), _ref("opacity"))))

    # Alpha cutoff (discard nearly-transparent fragments)
    alpha_cutoff = config["alpha_cutoff"]
    body.append(_if(
        _binop("<", _ref("alpha"), _push_field("alpha_cutoff")),
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
    vertex_stage = _build_vertex_stage()
    vertex_stage._splat_name = splat.name

    # --- Stage 3: render fragment ---
    fragment_stage = _build_fragment_stage(config)
    fragment_stage._splat_name = splat.name

    return [compute_stage, vertex_stage, fragment_stage]
