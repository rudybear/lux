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
    fn.attributes = ["workgroup_size(256)"]
    return fn


# ---------------------------------------------------------------------------
# Pass 2: un-multiplex params + apply kernel
# (mobiledlss.train.reconstruct.pixel_shuffle_params + apply_kernel)
# ---------------------------------------------------------------------------

def _build_apply_stage(config: dict) -> StageBlock:
    S, K, SP, HIDDEN = config["s"], config["k"], config["sp"], config["hidden"]
    total_ch = SP * SP * K * K + SP * SP + HIDDEN

    stage = StageBlock(stage_type="compute")
    stage.storage_buffers = [
        StorageBufferDecl("packed_params", "scalar"),
        StorageBufferDecl("proxy_color", "scalar"),
        StorageBufferDecl("spatial_out", "scalar"),
        StorageBufferDecl("alpha_out", "scalar"),
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

    # --- Alpha: sigmoid of the one alpha-logit channel for this block offset. ---
    alpha_ch = SP * SP * K * K
    body.append(_let("alpha_logit", "scalar",
        _idx("packed_params", _add(_ref("net_base"),
                                    _binop("+", _uint_lit(alpha_ch), _ref("block_off"))))))
    body.append(_let("alpha", "scalar",
        _binop("/", _lit(1.0), _binop("+", _lit(1.0), _call("exp", [_binop("-", _lit(0.0), _ref("alpha_logit"))])))))
    body.append(_assign_idx("alpha_out", _ref("gid"), _ref("alpha")))

    # --- Hidden: broadcast copy, NOT pixel-shuffled (nearest upsample). ---
    hidden_ch0 = SP * SP * K * K + SP * SP
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

def _build_blend_stage(config: dict) -> StageBlock:
    stage = StageBlock(stage_type="compute")
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def expand_reconstruct_pipeline(reconstruct, pipeline, module) -> list:
    """Expand a `reconstruct` declaration into the warp/apply/blend compute
    stages. `s`, `k`, `param_stride`, `hidden` are baked in at compile time
    (one compiled pipeline == one fixed network architecture, exactly like
    a splat block's `sh_degree`); `target_w/h`, `proxy_w/h`, `net_w/h` and
    `jitter` are runtime push-constant fields."""
    config = _get_reconstruct_config(reconstruct)

    module._defines = getattr(module, "_defines", {})
    module._defines["workgroup_size_x"] = 256
    module._defines["workgroup_size_y"] = 1
    module._defines["workgroup_size_z"] = 1

    warp_stage = _build_warp_stage(config)
    warp_stage._output_stem_suffix = "warp"
    warp_stage._reconstruct_config = config
    warp_stage._reconstruct_name = reconstruct.name

    apply_stage = _build_apply_stage(config)
    apply_stage._output_stem_suffix = "apply"
    apply_stage._reconstruct_config = config
    apply_stage._reconstruct_name = reconstruct.name

    blend_stage = _build_blend_stage(config)
    blend_stage._output_stem_suffix = "blend"
    blend_stage._reconstruct_config = config
    blend_stage._reconstruct_name = reconstruct.name

    return [warp_stage, apply_stage, blend_stage]
