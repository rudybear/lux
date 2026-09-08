"""Reconstruction-pass expander (DLSS-style upscale/accumulate).

Runtime port of `mobiledlss.train.reconstruct` (see
`docs/lux-reconstruct-spec.md` in the mobiledlss repo). Takes a
`reconstruct { s: 2, k: 4, param_stride: 1, hidden: 8 }` declaration plus a
`pipeline { mode: reconstruct, reconstruct: X }` and emits three compute
stages -- "warp" (reproject the previous output by the motion field),
"apply" (un-multiplex the network's packed per-pixel parameters with
`pixel_shuffle`'s exact channel convention, softmax the kernel, sigmoid the
alpha, and apply the resulting per-pixel K*K kernel to the proxy colour),
and "blend" (lerp the spatial upsample with the warped history, forcing
pure-spatial where disoccluded) -- matching `warp`/`pixel_shuffle_params`+
`apply_kernel`/`blend` in the reference exactly.

Everything the reference computes is per-*target*-pixel and has no
cross-pixel dependency within a frame (unlike the splat renderer's sort),
so each pass is a single flat 1-D dispatch over `H*W` target pixels (same
convention as the splat preprocess compute stage: `global_invocation_id.x`
as a linear thread id, bounds-checked against a push constant, workgroup
size 256), decomposing to `(X, Y)` via integer div/mod. `s`, `k`,
`param_stride` and `hidden` are compile-time (baked as literals into the
generated SPIR-V, exactly like the splat block's `sh_degree`/`dilation`);
resolutions (`target_w/h`, `proxy_w/h`, `net_w/h`) and the per-frame
`jitter` are runtime push-constant fields, since one compiled pipeline is
reused across every frame/resolution of a clip.

Precision note (spec): fp16 storage is fine for colour/hidden; this
expander does all *arithmetic* in fp32 (`scalar`) regardless of how the
host stores the buffers, matching the reference's own fp32 compute.
"""

from __future__ import annotations

from luxc.parser.ast_nodes import (
    StageBlock, StorageBufferDecl, PushBlock, BlockField,
    FunctionDef, LetStmt, AssignStmt, ReturnStmt, IfStmt,
    NumberLit, VarRef, BinaryOp, CallExpr, ConstructorExpr,
    IndexAccess, AssignTarget, SwizzleAccess,
)

__all__ = ["expand_reconstruct_pipeline"]

_PI = 3.14159265358979323846


# ---------------------------------------------------------------------------
# Small AST-construction helpers (same vocabulary/style as splat_expander.py)
# ---------------------------------------------------------------------------

def _lit(v) -> NumberLit:
    return NumberLit(str(v))


def _uint_lit(v) -> NumberLit:
    n = NumberLit(str(int(v)))
    n.resolved_type = "uint"
    return n


def _ref(name: str) -> VarRef:
    return VarRef(name)


def _idx(buf, index) -> IndexAccess:
    obj = _ref(buf) if isinstance(buf, str) else buf
    idx_expr = index if not isinstance(index, str) else _ref(index)
    return IndexAccess(obj, idx_expr)


def _call(fn: str, args: list) -> CallExpr:
    return CallExpr(_ref(fn), args)


def _swizzle(expr, components: str) -> SwizzleAccess:
    return SwizzleAccess(expr, components)


def _ctor(ty: str, args: list) -> ConstructorExpr:
    return ConstructorExpr(ty, args)


def _binop(op: str, left, right) -> BinaryOp:
    return BinaryOp(op, left, right)


def _let(name: str, ty: str, expr) -> LetStmt:
    return LetStmt(name, ty, expr)


def _assign(target, expr) -> AssignStmt:
    t = _ref(target) if isinstance(target, str) else target
    return AssignStmt(AssignTarget(t), expr)


def _assign_idx(buf: str, index, expr) -> AssignStmt:
    return AssignStmt(AssignTarget(_idx(buf, index)), expr)


def _if(cond, then_body: list, else_body: list | None = None) -> IfStmt:
    return IfStmt(cond, then_body, else_body or [])


def _add(*terms):
    acc = terms[0]
    for t in terms[1:]:
        acc = _binop("+", acc, t)
    return acc


def _mul(*terms):
    acc = terms[0]
    for t in terms[1:]:
        acc = _binop("*", acc, t)
    return acc


def _tap_offsets(K: int) -> range:
    """Integer tap offsets relative to the rounded anchor pixel -- identical
    to `mobiledlss.train.reconstruct._tap_offsets` (K=4: -1, 0, 1, 2)."""
    return range(-(K // 2 - 1), K // 2 + 1)


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------

def _get_reconstruct_config(reconstruct) -> dict:
    cfg = {"s": 2, "k": 4, "param_stride": 1, "hidden": 8}
    for m in reconstruct.members:
        if isinstance(m.value, NumberLit) and m.name in cfg:
            cfg[m.name] = int(float(m.value.value))
    cfg["sp"] = cfg["s"] * cfg["param_stride"]  # un-multiplex factor
    return cfg


def _get_memory_config(reconstruct) -> dict | None:
    """Optional `memory: { channels: 8, hidden: 16 }` sub-block (explicit
    per-scene memory -- mobiledlss's docs/scene-memory-spec.md /
    `mobiledlss.train.scene_texture.SceneTexture`/`MemoryColorHead`).
    Defaults match those classes' own defaults. `None` when the reconstruct
    declaration has no memory sub-block -- the 2-way blend, byte-identical
    to the stage-1 port, is unaffected."""
    memory = getattr(reconstruct, "memory", None)
    if memory is None:
        return None
    cfg = {"channels": 8, "hidden": 16}
    for m in memory.members:
        if isinstance(m.value, NumberLit) and m.name in cfg:
            cfg[m.name] = int(float(m.value.value))
    return cfg


def _shared_push_fields() -> list[BlockField]:
    """Resolution push fields shared by all three passes (one compiled
    pipeline is reused across every clip resolution, so these can't be
    compile-time constants)."""
    return [
        BlockField("target_w", "uint"),
        BlockField("target_h", "uint"),
    ]


# ---------------------------------------------------------------------------
# Pass 1: warp history (mobiledlss.train.reconstruct.warp)
# ---------------------------------------------------------------------------

def _build_warp_stage(config: dict) -> StageBlock:
    S = config["s"]
    stage = StageBlock(stage_type="compute")
    stage.storage_buffers = [
        StorageBufferDecl("prev_color", "scalar"),
        StorageBufferDecl("mv_proxy", "scalar"),
        StorageBufferDecl("warped_color", "scalar"),
    ]
    fields = _shared_push_fields() + [
        BlockField("proxy_w", "uint"),
        BlockField("proxy_h", "uint"),
    ]
    stage.push_constants = [PushBlock("push", fields)]

    body = []
    body.append(_let("gid", "uint", _swizzle(_ref("global_invocation_id"), "x")))
    body.append(_let("total", "uint", _binop("*", _ref("target_w"), _ref("target_h"))))
    body.append(_if(_binop(">=", _ref("gid"), _ref("total")), [ReturnStmt(None)]))
    body.append(_let("Y", "uint", _binop("/", _ref("gid"), _ref("target_w"))))
    body.append(_let("X", "uint", _binop("%", _ref("gid"), _ref("target_w"))))

    # Nearest-upsampled, s-scaled proxy motion vector at this target pixel.
    body.append(_let("py", "uint", _binop("/", _ref("Y"), _uint_lit(S))))
    body.append(_let("px", "uint", _binop("/", _ref("X"), _uint_lit(S))))
    body.append(_let("mv_base", "uint",
        _mul(_add(_mul(_ref("py"), _ref("proxy_w")), _ref("px")), _uint_lit(2))))
    body.append(_let("mvx", "scalar",
        _binop("*", _idx("mv_proxy", _binop("+", _ref("mv_base"), _uint_lit(0))), _lit(float(S)))))
    body.append(_let("mvy", "scalar",
        _binop("*", _idx("mv_proxy", _binop("+", _ref("mv_base"), _uint_lit(1))), _lit(float(S)))))

    # Pixel-centre sample position (grid_sample, align_corners=False
    # semantics), then converted to a "pixel-corner" continuous coordinate
    # for the standard floor/frac bilinear decomposition.
    body.append(_let("Xf", "scalar", _ref("X")))
    body.append(_let("Yf", "scalar", _ref("Y")))
    body.append(_let("sx", "scalar", _binop("-", _binop("+", _ref("Xf"), _lit(0.5)), _ref("mvx"))))
    body.append(_let("sy", "scalar", _binop("-", _binop("+", _ref("Yf"), _lit(0.5)), _ref("mvy"))))
    body.append(_let("sxc", "scalar", _binop("-", _ref("sx"), _lit(0.5))))
    body.append(_let("syc", "scalar", _binop("-", _ref("sy"), _lit(0.5))))
    body.append(_let("x0", "scalar", _call("floor", [_ref("sxc")])))
    body.append(_let("y0", "scalar", _call("floor", [_ref("syc")])))
    body.append(_let("fx", "scalar", _binop("-", _ref("sxc"), _ref("x0"))))
    body.append(_let("fy", "scalar", _binop("-", _ref("syc"), _ref("y0"))))
    body.append(_let("target_wf", "scalar", _ref("target_w")))
    body.append(_let("target_hf", "scalar", _ref("target_h")))

    for c in range(3):
        body.append(_let(f"acc{c}", "scalar", _lit(0.0)))

    # 4-tap bilinear gather, zero-padded outside [0, size) -- grid_sample's
    # padding_mode="zeros": the tap simply doesn't contribute (weight times
    # zero), it is NOT clamped to the border and NOT renormalised.
    taps = [(0, 0, "(1.0-fx)*(1.0-fy)"), (1, 0, "fx*(1.0-fy)"),
            (0, 1, "(1.0-fx)*fy"), (1, 1, "fx*fy")]
    for tdx, tdy, _w in taps:
        tx = _ref("x0") if tdx == 0 else _binop("+", _ref("x0"), _lit(1.0))
        ty = _ref("y0") if tdy == 0 else _binop("+", _ref("y0"), _lit(1.0))
        wexpr = _mul(
            _ref("fx") if tdx == 1 else _binop("-", _lit(1.0), _ref("fx")),
            _ref("fy") if tdy == 1 else _binop("-", _lit(1.0), _ref("fy")),
        )
        txn, tyn = f"tx_{tdx}_{tdy}", f"ty_{tdx}_{tdy}"
        wn = f"w_{tdx}_{tdy}"
        body.append(_let(txn, "scalar", tx))
        body.append(_let(tyn, "scalar", ty))
        body.append(_let(wn, "scalar", wexpr))
        inbounds = _binop("&&",
            _binop("&&", _binop(">=", _ref(txn), _lit(0.0)), _binop("<", _ref(txn), _ref("target_wf"))),
            _binop("&&", _binop(">=", _ref(tyn), _lit(0.0)), _binop("<", _ref(tyn), _ref("target_hf"))))
        base = _mul(_add(_mul(_ref(tyn), _ref("target_wf")), _ref(txn)), _lit(3.0))
        basen = f"base_{tdx}_{tdy}"
        then_body = [_let(basen, "scalar", base)]
        for c in range(3):
            then_body.append(_assign(f"acc{c}", _add(_ref(f"acc{c}"),
                _mul(_ref(wn), _idx("prev_color", _binop("+", _ref(basen), _lit(float(c))))))))
        body.append(_if(inbounds, then_body))

    out_base = _mul(_ref("gid"), _uint_lit(3))
    for c in range(3):
        body.append(_assign_idx("warped_color", _binop("+", out_base, _uint_lit(c)), _ref(f"acc{c}")))

    stage.functions = [_compute_main(body)]
    return stage


def _compute_main(body: list) -> FunctionDef:
    fn = FunctionDef("main", [], None, body)
    # Task B (Mali reconstruct-pass profiling, docs/rendering-engines.md):
    # tried workgroup_size(64) (Arm's own Mali best-practices guidance
    # favors 64-128-thread compute workgroups over Desktop-GPU-sized 256
    # for kernels like these five -- simple, uniform per-pixel work, no
    # shared-memory cross-thread communication, no barriers -- reasoning
    # being Mali's fixed per-core register file is partitioned across
    # concurrently resident warps/workgroups, and a 256-thread workgroup
    # for a register-heavy kernel like apply's fully-unrolled K*K=16-tap
    # softmax+gather, ~50+ live temporaries per thread, could force lower
    # occupancy or spilling that 64 threads would avoid).
    #
    # MEASURED on device (Pixel 9 Pro XL / Mali-G715, 960x540,
    # RECON_TIMING_STAGES over ~15 live frames): workgroup_size(64) vs.
    # (256) is a no-op within noise for all five stages (apply
    # 3.4-4.0ms @256 vs. 3.7-4.6ms @64; bguv/memory/warp/blend equally
    # flat) -- occupancy/register pressure isn't this reconstruct pass's
    # bottleneck at this resolution on this GPU, so reverted to 256 rather
    # than carry the change's real cost for no benefit: the host's dispatch
    # group count (kReconWorkgroupSize in reconstruct_pass.cpp/
    # reconstruct_runner.cpp) MUST exactly match this attribute -- Vulkan
    # bakes local workgroup size into the compiled SPIR-V (unlike Metal's
    # host-controlled dispatchThreadgroups), so a mismatch here silently
    # under/over-dispatches instead of erroring (caught by
    # tests/test_lux_reconstruct.py: max|out_lux-out_ref| ~0.8 instead of
    # <1e-5 with workgroup_size(64) and the host still assuming 256).
    # Left the named kReconWorkgroupSize constant (now =256) in both host
    # files rather than reverting to a bare literal, so this coupling is
    # explicit for whoever revisits workgroup size on different hardware.
    fn.attributes = ["workgroup_size(256)"]
    return fn


# ---------------------------------------------------------------------------
# Scene-memory pass A: sphere UV (mobiledlss.datagen.camera.sphere_uv)
# ---------------------------------------------------------------------------

def _build_bguv_stage(config: dict) -> StageBlock:
    """Per-pixel sphere UV: the *far* intersection of the pixel's world-
    space view ray with a bounding sphere `(bg_sphere.xyz, bg_sphere.w)`,
    expressed as (azimuth, polar) in `[0, 1]` -- exactly
    `mobiledlss.datagen.camera.sphere_uv`. Resolution- and camera-generic
    (everything is a push constant): the host runs this same compiled
    stage twice per frame -- once at proxy resolution (feeding the
    network's own extra texture-feature input channels, dumped via
    `--dump-bg-features`) and once at target resolution (feeding the
    memory-colour decode stage below) -- mirroring
    `mobiledlss.train.train.rollout`'s own dual sampling of `SceneTexture`
    at both resolutions. Camera convention is OpenCV (+x right, +y down,
    +z forward); the host's camera bridge already handles OpenCV -> GL,
    so this stage only needs a plain camera-to-world matrix and pinhole
    intrinsics -- both convention-free once expressed as a world-space ray.
    """
    stage = StageBlock(stage_type="compute")
    stage.storage_buffers = [
        StorageBufferDecl("bguv_out", "scalar"),
    ]
    fields = [
        BlockField("width", "uint"),
        BlockField("height", "uint"),
        BlockField("k_params", "vec4"),       # fx, fy, cx, cy
        BlockField("cam_to_world", "mat4"),   # camera -> world (OpenCV convention)
        BlockField("bg_sphere", "vec4"),      # centre.xyz, radius
    ]
    stage.push_constants = [PushBlock("push", fields)]

    body = []
    body.append(_let("gid", "uint", _swizzle(_ref("global_invocation_id"), "x")))
    body.append(_let("total", "uint", _binop("*", _ref("width"), _ref("height"))))
    body.append(_if(_binop(">=", _ref("gid"), _ref("total")), [ReturnStmt(None)]))
    body.append(_let("Y", "uint", _binop("/", _ref("gid"), _ref("width"))))
    body.append(_let("X", "uint", _binop("%", _ref("gid"), _ref("width"))))
    body.append(_let("Xf", "scalar", _ref("X")))
    body.append(_let("Yf", "scalar", _ref("Y")))

    # Camera-space ray direction (pixel-centre, pinhole):
    # ((x+0.5-cx)/fx, (y+0.5-cy)/fy, 1) -- camera.sphere_uv's dirs_cam.
    body.append(_let("dcx", "scalar",
        _binop("/", _binop("-", _binop("+", _ref("Xf"), _lit(0.5)), _swizzle(_ref("k_params"), "z")),
               _swizzle(_ref("k_params"), "x"))))
    body.append(_let("dcy", "scalar",
        _binop("/", _binop("-", _binop("+", _ref("Yf"), _lit(0.5)), _swizzle(_ref("k_params"), "w")),
               _swizzle(_ref("k_params"), "y"))))

    # World-space ray direction/origin: mat4 * vec4 with w=0 transforms a
    # direction (rotation only, matches `dirs_cam @ c2w[:3,:3].T`); w=1
    # transforms the camera origin (matches `c2w[:3, 3]`).
    body.append(_let("dir_cam4", "vec4", _ctor("vec4", [_ref("dcx"), _ref("dcy"), _lit(1.0), _lit(0.0)])))
    body.append(_let("dir_world4", "vec4", _binop("*", _ref("cam_to_world"), _ref("dir_cam4"))))
    body.append(_let("dir", "vec3", _call("normalize", [_swizzle(_ref("dir_world4"), "xyz")])))
    body.append(_let("origin4", "vec4", _binop("*", _ref("cam_to_world"),
        _ctor("vec4", [_lit(0.0), _lit(0.0), _lit(0.0), _lit(1.0)]))))
    body.append(_let("o", "vec3", _swizzle(_ref("origin4"), "xyz")))

    # Far intersection with the bounding sphere (camera.sphere_uv's exact
    # geometry: b = dot(dir, oc), disc = b^2 - (dot(oc,oc) - r^2),
    # t = -b + sqrt(max(disc, 0))).
    body.append(_let("center", "vec3", _swizzle(_ref("bg_sphere"), "xyz")))
    body.append(_let("radius", "scalar", _swizzle(_ref("bg_sphere"), "w")))
    body.append(_let("oc", "vec3", _binop("-", _ref("o"), _ref("center"))))
    body.append(_let("b", "scalar", _call("dot", [_ref("dir"), _ref("oc")])))
    body.append(_let("oc_dot", "scalar", _call("dot", [_ref("oc"), _ref("oc")])))
    body.append(_let("disc", "scalar",
        _binop("-", _mul(_ref("b"), _ref("b")),
               _binop("-", _ref("oc_dot"), _mul(_ref("radius"), _ref("radius"))))))
    body.append(_let("t", "scalar",
        _add(_binop("-", _lit(0.0), _ref("b")),
             _call("sqrt", [_call("max", [_ref("disc"), _lit(0.0)])]))))
    body.append(_let("p", "vec3",
        _binop("-", _add(_ref("o"), _mul(_ref("dir"), _ref("t"))), _ref("center"))))
    body.append(_let("d", "vec3", _call("normalize", [_ref("p")])))

    body.append(_let("u", "scalar",
        _add(_binop("/", _call("atan", [_swizzle(_ref("d"), "z"), _swizzle(_ref("d"), "x")]),
                    _lit(2.0 * _PI)),
             _lit(0.5))))
    body.append(_let("v", "scalar",
        _binop("/", _call("acos", [_call("clamp", [_swizzle(_ref("d"), "y"), _lit(-1.0), _lit(1.0)])]),
               _lit(_PI))))

    out_base = _mul(_ref("gid"), _uint_lit(2))
    body.append(_assign_idx("bguv_out", _binop("+", out_base, _uint_lit(0)), _ref("u")))
    body.append(_assign_idx("bguv_out", _binop("+", out_base, _uint_lit(1)), _ref("v")))

    stage.functions = [_compute_main(body)]
    return stage


# ---------------------------------------------------------------------------
# Scene-memory pass B: texture sample + decoder MLP
# (mobiledlss.train.scene_texture.SceneTexture + MemoryColorHead)
# ---------------------------------------------------------------------------

def _build_memory_stage(config: dict, memory_config: dict) -> StageBlock:
    """Bilinear-sample the per-scene feature texture at each pixel's sphere
    UV (`u` wraps, `v` clamps -- `SceneTexture.forward`'s exact semantics,
    implemented as a direct wrap/clamp bilinear gather rather than
    `SceneTexture`'s reference implementation's one-texel-pad trick, since
    a compute storage buffer has no sampler/border-mode hardware to lean
    on -- both are exactly equivalent, see docs/language-reference.md),
    then decode through the `channels -> hidden -> 3` 1x1-conv MLP
    (`MemoryColorHead`: `LeakyReLU(0.1)` between, `sigmoid` on the output).
    `channels`/`hidden` are compile-time (baked into the SPIR-V, exactly
    like `s`/`k`/`param_stride`/`hidden` are for the main config), so the
    whole per-pixel computation -- texture gather and both MLP layers --
    is fully unrolled.

    Resolution-generic (`width`/`height` push fields): the host runs this
    same compiled stage at proxy resolution (only `bg_features_out`
    matters there -- the raw sampled feature vector fed to the network as
    an extra input channel block outside this pass, dumped via
    `--dump-bg-features`) and at target resolution (`memory_color_out`
    feeds the blend stage's 3-way blend).
    """
    C, HIDDEN = memory_config["channels"], memory_config["hidden"]

    stage = StageBlock(stage_type="compute")
    stage.storage_buffers = [
        StorageBufferDecl("bguv_in", "scalar"),
        StorageBufferDecl("bg_texture", "scalar"),
        StorageBufferDecl("mem_fc1_w", "scalar"),
        StorageBufferDecl("mem_fc1_b", "scalar"),
        StorageBufferDecl("mem_fc2_w", "scalar"),
        StorageBufferDecl("mem_fc2_b", "scalar"),
        StorageBufferDecl("bg_features_out", "scalar"),
        StorageBufferDecl("memory_color_out", "scalar"),
    ]
    fields = [
        BlockField("width", "uint"),
        BlockField("height", "uint"),
        BlockField("tex_w", "uint"),
        BlockField("tex_h", "uint"),
    ]
    stage.push_constants = [PushBlock("push", fields)]

    body = []
    body.append(_let("gid", "uint", _swizzle(_ref("global_invocation_id"), "x")))
    body.append(_let("total", "uint", _binop("*", _ref("width"), _ref("height"))))
    body.append(_if(_binop(">=", _ref("gid"), _ref("total")), [ReturnStmt(None)]))

    # u (azimuth) wraps to [0, 1) via fract (== torch.remainder(u, 1.0));
    # v (polar) clamps -- SceneTexture.forward / rollout._addressed_uv.
    body.append(_let("u_raw", "scalar", _idx("bguv_in", _mul(_ref("gid"), _uint_lit(2)))))
    body.append(_let("v_raw", "scalar",
        _idx("bguv_in", _add(_mul(_ref("gid"), _uint_lit(2)), _uint_lit(1)))))
    body.append(_let("u", "scalar", _call("fract", [_ref("u_raw")])))
    body.append(_let("v", "scalar", _call("clamp", [_ref("v_raw"), _lit(0.0), _lit(1.0)])))

    body.append(_let("tex_wf", "scalar", _ref("tex_w")))
    body.append(_let("tex_hf", "scalar", _ref("tex_h")))

    # Continuous texel-index coordinate (align_corners=False convention,
    # matching grid_sample's own px = u*W - 0.5 -- see
    # docs/language-reference.md's derivation from SceneTexture.forward's
    # one-texel-pad trick).
    body.append(_let("px", "scalar", _binop("-", _mul(_ref("u"), _ref("tex_wf")), _lit(0.5))))
    body.append(_let("py", "scalar", _binop("-", _mul(_ref("v"), _ref("tex_hf")), _lit(0.5))))
    body.append(_let("x0f", "scalar", _call("floor", [_ref("px")])))
    body.append(_let("y0f", "scalar", _call("floor", [_ref("py")])))
    body.append(_let("fx", "scalar", _binop("-", _ref("px"), _ref("x0f"))))
    body.append(_let("fy", "scalar", _binop("-", _ref("py"), _ref("y0f"))))
    body.append(_let("x1f", "scalar", _add(_ref("x0f"), _lit(1.0))))
    body.append(_let("y1f", "scalar", _add(_ref("y0f"), _lit(1.0))))

    # u wraps (OpFMod, always non-negative for a positive divisor); v clamps.
    body.append(_let("x0w", "scalar", _call("mod", [_ref("x0f"), _ref("tex_wf")])))
    body.append(_let("x1w", "scalar", _call("mod", [_ref("x1f"), _ref("tex_wf")])))
    body.append(_let("y0c", "scalar",
        _call("clamp", [_ref("y0f"), _lit(0.0), _binop("-", _ref("tex_hf"), _lit(1.0))])))
    body.append(_let("y1c", "scalar",
        _call("clamp", [_ref("y1f"), _lit(0.0), _binop("-", _ref("tex_hf"), _lit(1.0))])))

    body.append(_let("x0i", "uint", _ref("x0w")))
    body.append(_let("x1i", "uint", _ref("x1w")))
    body.append(_let("y0i", "uint", _ref("y0c")))
    body.append(_let("y1i", "uint", _ref("y1c")))

    body.append(_let("row0", "uint", _mul(_ref("y0i"), _ref("tex_w"))))
    body.append(_let("row1", "uint", _mul(_ref("y1i"), _ref("tex_w"))))
    body.append(_let("plane_stride", "uint", _mul(_ref("tex_w"), _ref("tex_h"))))

    body.append(_let("w00", "scalar", _mul(_binop("-", _lit(1.0), _ref("fx")), _binop("-", _lit(1.0), _ref("fy")))))
    body.append(_let("w01", "scalar", _mul(_ref("fx"), _binop("-", _lit(1.0), _ref("fy")))))
    body.append(_let("w10", "scalar", _mul(_binop("-", _lit(1.0), _ref("fx")), _ref("fy"))))
    body.append(_let("w11", "scalar", _mul(_ref("fx"), _ref("fy"))))

    feat_base = _mul(_ref("gid"), _uint_lit(C))
    for c in range(C):
        plane_off = _mul(_uint_lit(c), _ref("plane_stride"))
        idx00 = _add(plane_off, _add(_ref("row0"), _ref("x0i")))
        idx01 = _add(plane_off, _add(_ref("row0"), _ref("x1i")))
        idx10 = _add(plane_off, _add(_ref("row1"), _ref("x0i")))
        idx11 = _add(plane_off, _add(_ref("row1"), _ref("x1i")))
        body.append(_let(f"t00_{c}", "scalar", _idx("bg_texture", idx00)))
        body.append(_let(f"t01_{c}", "scalar", _idx("bg_texture", idx01)))
        body.append(_let(f"t10_{c}", "scalar", _idx("bg_texture", idx10)))
        body.append(_let(f"t11_{c}", "scalar", _idx("bg_texture", idx11)))
        body.append(_let(f"feat{c}", "scalar", _add(
            _mul(_ref("w00"), _ref(f"t00_{c}")),
            _mul(_ref("w01"), _ref(f"t01_{c}")),
            _mul(_ref("w10"), _ref(f"t10_{c}")),
            _mul(_ref("w11"), _ref(f"t11_{c}")),
        )))
        body.append(_assign_idx("bg_features_out", _add(feat_base, _uint_lit(c)), _ref(f"feat{c}")))

    # --- Decoder MLP: channels -> hidden (LeakyReLU 0.1) -> 3 (sigmoid). ---
    # leaky_relu(x, 0.1) == max(x, 0.1*x) for any real x (0.1 < 1, so the
    # positive branch keeps x and the negative branch keeps the smaller-
    # magnitude 0.1*x) -- avoids a per-element branch/select.
    for h in range(HIDDEN):
        terms = [_idx("mem_fc1_b", _uint_lit(h))]
        for c in range(C):
            w_idx = _uint_lit(h * C + c)
            terms.append(_mul(_idx("mem_fc1_w", w_idx), _ref(f"feat{c}")))
        body.append(_let(f"h_pre{h}", "scalar", _add(*terms)))
        body.append(_let(f"h{h}", "scalar",
            _call("max", [_ref(f"h_pre{h}"), _mul(_lit(0.1), _ref(f"h_pre{h}"))])))

    color_base = _mul(_ref("gid"), _uint_lit(3))
    for o in range(3):
        terms = [_idx("mem_fc2_b", _uint_lit(o))]
        for h in range(HIDDEN):
            w_idx = _uint_lit(o * HIDDEN + h)
            terms.append(_mul(_idx("mem_fc2_w", w_idx), _ref(f"h{h}")))
        body.append(_let(f"o_pre{o}", "scalar", _add(*terms)))
        body.append(_let(f"sig{o}", "scalar",
            _binop("/", _lit(1.0),
                   _binop("+", _lit(1.0), _call("exp", [_binop("-", _lit(0.0), _ref(f"o_pre{o}"))])))))
        body.append(_assign_idx("memory_color_out", _binop("+", color_base, _uint_lit(o)), _ref(f"sig{o}")))

    stage.functions = [_compute_main(body)]
    return stage


# ---------------------------------------------------------------------------
# Pass 2: un-multiplex params + apply kernel
# (mobiledlss.train.reconstruct.pixel_shuffle_params + apply_kernel)
# ---------------------------------------------------------------------------

def _build_apply_stage(config: dict, memory_config: dict | None = None) -> StageBlock:
    S, K, SP, HIDDEN = config["s"], config["k"], config["sp"], config["hidden"]
    # 2-way (no scene memory): 1 blend-logit channel per output pixel
    # (sigmoid "alpha"). 3-way (scene memory present): 3 channels per
    # output pixel ({spatial, history, memory}, softmax) -- selected
    # purely by blend_logits' own channel count, matching
    # `pixel_shuffle_params`'s own dispatch (no extra flag there either).
    n_blend = 3 if memory_config else 1
    total_ch = SP * SP * K * K + SP * SP * n_blend + HIDDEN
    blend_buf = "blend_out" if memory_config else "alpha_out"

    stage = StageBlock(stage_type="compute")
    stage.storage_buffers = [
        StorageBufferDecl("packed_params", "scalar"),
        StorageBufferDecl("proxy_color", "scalar"),
        StorageBufferDecl("spatial_out", "scalar"),
        StorageBufferDecl(blend_buf, "scalar"),
        StorageBufferDecl("hidden_out", "scalar"),
    ]
    fields = _shared_push_fields() + [
        BlockField("proxy_w", "uint"),
        BlockField("proxy_h", "uint"),
        BlockField("net_w", "uint"),
        BlockField("net_h", "uint"),
        BlockField("jitter_x", "scalar"),
        BlockField("jitter_y", "scalar"),
    ]
    stage.push_constants = [PushBlock("push", fields)]

    body = []
    body.append(_let("gid", "uint", _swizzle(_ref("global_invocation_id"), "x")))
    body.append(_let("total", "uint", _binop("*", _ref("target_w"), _ref("target_h"))))
    body.append(_if(_binop(">=", _ref("gid"), _ref("total")), [ReturnStmt(None)]))
    body.append(_let("Y", "uint", _binop("/", _ref("gid"), _ref("target_w"))))
    body.append(_let("X", "uint", _binop("%", _ref("gid"), _ref("target_w"))))

    # --- Un-multiplex indexing: network pixel (nu, nv) + within-block
    # offset (dx, dy), PixelShuffle convention: input channel =
    # c*sp*sp + dy*sp + dx for output channel c at block offset (dy, dx).
    body.append(_let("nu", "uint", _binop("/", _ref("X"), _uint_lit(SP))))
    body.append(_let("nv", "uint", _binop("/", _ref("Y"), _uint_lit(SP))))
    body.append(_let("bdx", "uint", _binop("%", _ref("X"), _uint_lit(SP))))
    body.append(_let("bdy", "uint", _binop("%", _ref("Y"), _uint_lit(SP))))
    body.append(_let("block_off", "uint", _add(_mul(_ref("bdy"), _uint_lit(SP)), _ref("bdx"))))
    body.append(_let("net_base", "uint",
        _mul(_add(_mul(_ref("nv"), _ref("net_w")), _ref("nu")), _uint_lit(total_ch))))

    # --- Kernel: softmax over K*K taps (numerically stable: subtract max). ---
    n_taps = K * K
    for kk in range(n_taps):
        ch = kk * SP * SP
        body.append(_let(f"klogit{kk}", "scalar",
            _idx("packed_params", _add(_ref("net_base"),
                                        _binop("+", _uint_lit(ch), _ref("block_off"))))))
    maxname = "klogit0"
    for kk in range(1, n_taps):
        newmax = f"kmax{kk}"
        body.append(_let(newmax, "scalar", _call("max", [_ref(maxname), _ref(f"klogit{kk}")])))
        maxname = newmax
    body.append(_let("kmax", "scalar", _ref(maxname)))
    for kk in range(n_taps):
        body.append(_let(f"kexp{kk}", "scalar",
            _call("exp", [_binop("-", _ref(f"klogit{kk}"), _ref("kmax"))])))
    sumname = "kexp0"
    for kk in range(1, n_taps):
        newsum = f"ksum{kk}"
        body.append(_let(newsum, "scalar", _add(_ref(sumname), _ref(f"kexp{kk}"))))
        sumname = newsum
    body.append(_let("kexpsum", "scalar", _ref(sumname)))
    for kk in range(n_taps):
        body.append(_let(f"kw{kk}", "scalar", _binop("/", _ref(f"kexp{kk}"), _ref("kexpsum"))))

    # --- Blend weight(s): 2-way sigmoid alpha, or 3-way softmax over
    # {spatial, history, memory} logits (same PixelShuffle channel
    # convention as the kernel taps: channel = kk*SP*SP + block_off). ---
    blend_ch0 = SP * SP * K * K
    if memory_config is None:
        body.append(_let("alpha_logit", "scalar",
            _idx("packed_params", _add(_ref("net_base"),
                                        _binop("+", _uint_lit(blend_ch0), _ref("block_off"))))))
        body.append(_let("alpha", "scalar",
            _binop("/", _lit(1.0),
                   _binop("+", _lit(1.0), _call("exp", [_binop("-", _lit(0.0), _ref("alpha_logit"))])))))
        body.append(_assign_idx("alpha_out", _ref("gid"), _ref("alpha")))
    else:
        for kk in range(3):
            ch = blend_ch0 + kk * SP * SP
            body.append(_let(f"blogit{kk}", "scalar",
                _idx("packed_params", _add(_ref("net_base"),
                                            _binop("+", _uint_lit(ch), _ref("block_off"))))))
        body.append(_let("bmax01", "scalar", _call("max", [_ref("blogit0"), _ref("blogit1")])))
        body.append(_let("bmax", "scalar", _call("max", [_ref("bmax01"), _ref("blogit2")])))
        for kk in range(3):
            body.append(_let(f"bexp{kk}", "scalar",
                _call("exp", [_binop("-", _ref(f"blogit{kk}"), _ref("bmax"))])))
        body.append(_let("bsum01", "scalar", _add(_ref("bexp0"), _ref("bexp1"))))
        body.append(_let("bsum", "scalar", _add(_ref("bsum01"), _ref("bexp2"))))
        blend_out_base = _mul(_ref("gid"), _uint_lit(3))
        for kk in range(3):
            body.append(_let(f"bw{kk}", "scalar", _binop("/", _ref(f"bexp{kk}"), _ref("bsum"))))
            body.append(_assign_idx("blend_out", _binop("+", blend_out_base, _uint_lit(kk)), _ref(f"bw{kk}")))

    # --- Hidden: broadcast copy, NOT pixel-shuffled (nearest upsample). ---
    hidden_ch0 = SP * SP * K * K + SP * SP * n_blend
    hidden_out_base = _mul(_ref("gid"), _uint_lit(HIDDEN))
    for h in range(HIDDEN):
        body.append(_assign_idx("hidden_out", _binop("+", hidden_out_base, _uint_lit(h)),
            _idx("packed_params", _add(_ref("net_base"), _uint_lit(hidden_ch0 + h)))))

    # --- apply_kernel: jitter-corrected proxy coordinate + nearest,
    # clamped-to-border per-tap gather (mobiledlss.train.reconstruct
    # .apply_kernel uses mode="nearest" with an exactly-integer grid
    # coordinate, which degenerates to a plain clamped index -- not a
    # bilinear resample; the *weights* softmax above supplies the
    # sub-pixel blending). ---
    body.append(_let("Xf", "scalar", _ref("X")))
    body.append(_let("Yf", "scalar", _ref("Y")))
    body.append(_let("cx", "scalar",
        _binop("-", _binop("/", _add(_binop("+", _ref("Xf"), _lit(0.5)), _ref("jitter_x")), _lit(float(S))), _lit(0.5))))
    body.append(_let("cy", "scalar",
        _binop("-", _binop("/", _add(_binop("+", _ref("Yf"), _lit(0.5)), _ref("jitter_y")), _lit(float(S))), _lit(0.5))))
    body.append(_let("anchor_x", "scalar", _call("round", [_ref("cx")])))
    body.append(_let("anchor_y", "scalar", _call("round", [_ref("cy")])))
    body.append(_let("proxy_wf", "scalar", _ref("proxy_w")))
    body.append(_let("proxy_hf", "scalar", _ref("proxy_h")))
    body.append(_let("proxy_wf_m1", "scalar", _binop("-", _ref("proxy_wf"), _lit(1.0))))
    body.append(_let("proxy_hf_m1", "scalar", _binop("-", _ref("proxy_hf"), _lit(1.0))))

    for c in range(3):
        body.append(_let(f"sacc{c}", "scalar", _lit(0.0)))

    offsets = list(_tap_offsets(K))
    kk = 0
    for dy_t in offsets:
        for dx_t in offsets:
            tx = _ref("anchor_x") if dx_t == 0 else _binop("+", _ref("anchor_x"), _lit(float(dx_t)))
            ty = _ref("anchor_y") if dy_t == 0 else _binop("+", _ref("anchor_y"), _lit(float(dy_t)))
            txc = _call("clamp", [tx, _lit(0.0), _ref("proxy_wf_m1")])
            tyc = _call("clamp", [ty, _lit(0.0), _ref("proxy_hf_m1")])
            txn, tyn = f"stx{kk}", f"sty{kk}"
            body.append(_let(txn, "scalar", txc))
            body.append(_let(tyn, "scalar", tyc))
            base = _mul(_add(_mul(_ref(tyn), _ref("proxy_wf")), _ref(txn)), _lit(3.0))
            basen = f"sbase{kk}"
            body.append(_let(basen, "scalar", base))
            for c in range(3):
                body.append(_assign(f"sacc{c}", _add(_ref(f"sacc{c}"),
                    _mul(_ref(f"kw{kk}"), _idx("proxy_color", _binop("+", _ref(basen), _lit(float(c))))))))
            kk += 1

    spatial_base = _mul(_ref("gid"), _uint_lit(3))
    for c in range(3):
        body.append(_assign_idx("spatial_out", _binop("+", spatial_base, _uint_lit(c)), _ref(f"sacc{c}")))

    stage.functions = [_compute_main(body)]
    return stage


# ---------------------------------------------------------------------------
# Pass 3: blend (mobiledlss.train.reconstruct.blend)
# ---------------------------------------------------------------------------

def _build_blend_stage(config: dict, memory_config: dict | None = None) -> StageBlock:
    stage = StageBlock(stage_type="compute")

    if memory_config is None:
        stage.storage_buffers = [
            StorageBufferDecl("spatial_out", "scalar"),
            StorageBufferDecl("warped_color", "scalar"),
            StorageBufferDecl("alpha_out", "scalar"),
            StorageBufferDecl("disocc", "scalar"),
            StorageBufferDecl("out_color", "scalar"),
        ]
        stage.push_constants = [PushBlock("push", _shared_push_fields())]

        body = []
        body.append(_let("gid", "uint", _swizzle(_ref("global_invocation_id"), "x")))
        body.append(_let("total", "uint", _binop("*", _ref("target_w"), _ref("target_h"))))
        body.append(_if(_binop(">=", _ref("gid"), _ref("total")), [ReturnStmt(None)]))

        body.append(_let("d", "scalar", _idx("disocc", _ref("gid"))))
        body.append(_let("a", "scalar", _idx("alpha_out", _ref("gid"))))
        body.append(_let("alpha_eff", "scalar", _lit(0.0)))
        body.append(_if(_binop(">", _ref("d"), _lit(0.5)),
                         [_assign("alpha_eff", _lit(1.0))],
                         [_assign("alpha_eff", _ref("a"))]))

        base = _mul(_ref("gid"), _uint_lit(3))
        for c in range(3):
            idx = _binop("+", base, _uint_lit(c))
            warped = _idx("warped_color", idx)
            spatial = _idx("spatial_out", idx)
            out_expr = _add(
                _mul(warped, _binop("-", _lit(1.0), _ref("alpha_eff"))),
                _mul(spatial, _ref("alpha_eff")),
            )
            body.append(_assign_idx("out_color", idx, out_expr))

        stage.functions = [_compute_main(body)]
        return stage

    # --- 3-way (scene memory): mobiledlss.train.reconstruct.blend3 /
    # renormalized_blend_weights -- zero the history share on disocclusion,
    # renormalise the three (already-softmaxed) weights to sum to 1, blend
    # spatial/warped/memory. ---
    stage.storage_buffers = [
        StorageBufferDecl("spatial_out", "scalar"),
        StorageBufferDecl("warped_color", "scalar"),
        StorageBufferDecl("blend_out", "scalar"),
        StorageBufferDecl("disocc", "scalar"),
        StorageBufferDecl("memory_color", "scalar"),
        StorageBufferDecl("out_color", "scalar"),
    ]
    stage.push_constants = [PushBlock("push", _shared_push_fields())]

    body = []
    body.append(_let("gid", "uint", _swizzle(_ref("global_invocation_id"), "x")))
    body.append(_let("total", "uint", _binop("*", _ref("target_w"), _ref("target_h"))))
    body.append(_if(_binop(">=", _ref("gid"), _ref("total")), [ReturnStmt(None)]))

    body.append(_let("d", "scalar", _idx("disocc", _ref("gid"))))
    bbase = _mul(_ref("gid"), _uint_lit(3))
    body.append(_let("ws0", "scalar", _idx("blend_out", _binop("+", bbase, _uint_lit(0)))))
    body.append(_let("wh0", "scalar", _idx("blend_out", _binop("+", bbase, _uint_lit(1)))))
    body.append(_let("wm0", "scalar", _idx("blend_out", _binop("+", bbase, _uint_lit(2)))))
    body.append(_let("wh", "scalar", _lit(0.0)))
    body.append(_if(_binop(">", _ref("d"), _lit(0.5)),
                     [_assign("wh", _lit(0.0))],
                     [_assign("wh", _ref("wh0"))]))
    body.append(_let("wtotal_raw", "scalar", _add(_ref("ws0"), _ref("wh"), _ref("wm0"))))
    body.append(_let("wtotal", "scalar", _call("max", [_ref("wtotal_raw"), _lit(1e-8)])))
    body.append(_let("ws", "scalar", _binop("/", _ref("ws0"), _ref("wtotal"))))
    body.append(_let("whn", "scalar", _binop("/", _ref("wh"), _ref("wtotal"))))
    body.append(_let("wmn", "scalar", _binop("/", _ref("wm0"), _ref("wtotal"))))

    base = _mul(_ref("gid"), _uint_lit(3))
    for c in range(3):
        idx = _binop("+", base, _uint_lit(c))
        out_expr = _add(
            _mul(_ref("ws"), _idx("spatial_out", idx)),
            _mul(_ref("whn"), _idx("warped_color", idx)),
            _mul(_ref("wmn"), _idx("memory_color", idx)),
        )
        body.append(_assign_idx("out_color", idx, out_expr))

    stage.functions = [_compute_main(body)]
    return stage


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def expand_reconstruct_pipeline(reconstruct, pipeline, module) -> list:
    """Expand a `reconstruct` declaration into its compute stages: always
    warp/apply/blend; additionally bguv/memory when the declaration has an
    optional `memory: { channels, hidden }` sub-block (explicit per-scene
    memory, mobiledlss's docs/scene-memory-spec.md), in which case `apply`
    predicts a 3-way ({spatial, history, memory}) blend and `blend`
    performs the renormalized 3-way mix instead of the 2-way lerp. `s`,
    `k`, `param_stride`, `hidden` (and, when present, `memory.channels`/
    `memory.hidden`) are baked in at compile time (one compiled pipeline ==
    one fixed network/memory architecture, exactly like a splat block's
    `sh_degree`); resolutions, camera and jitter are runtime push-constant
    fields."""
    config = _get_reconstruct_config(reconstruct)
    memory_config = _get_memory_config(reconstruct)

    module._defines = getattr(module, "_defines", {})
    module._defines["workgroup_size_x"] = 256  # see _compute_main()'s comment
    module._defines["workgroup_size_y"] = 1
    module._defines["workgroup_size_z"] = 1

    def _tag(stage, suffix):
        stage._output_stem_suffix = suffix
        stage._reconstruct_config = config
        stage._reconstruct_name = reconstruct.name
        return stage

    warp_stage = _tag(_build_warp_stage(config), "warp")
    apply_stage = _tag(_build_apply_stage(config, memory_config), "apply")

    stages = [warp_stage, apply_stage]

    if memory_config:
        bguv_stage = _tag(_build_bguv_stage(config), "bguv")
        memory_stage = _tag(_build_memory_stage(config, memory_config), "memory")
        for s in (bguv_stage, memory_stage):
            s._memory_config = memory_config
        stages += [bguv_stage, memory_stage]

    blend_stage = _tag(_build_blend_stage(config, memory_config), "blend")
    stages.append(blend_stage)

    return stages
