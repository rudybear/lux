"""Deferred rendering pipeline expander.

Takes a `surface` declaration, `geometry` declaration, and optional
`lighting` block, then generates a 4-stage deferred pipeline:

  1. **G-buffer vertex** — geometry pass vertex shader (reuses geometry expansion)
  2. **G-buffer fragment** — writes material params to MRT outputs (3 render targets)
  3. **Lighting vertex** — fullscreen triangle (no vertex buffer)
  4. **Lighting fragment** — reads G-buffer, evaluates full lighting

The module follows the same AST-construction patterns as ``splat_expander.py``
and ``surface_expander.py``.
"""

from __future__ import annotations

from luxc.parser.ast_nodes import (
    Module, StageBlock, VarDecl, UniformBlock, BlockField,
    SamplerDecl,
    FunctionDef, LetStmt, AssignStmt, ReturnStmt,
    NumberLit, VarRef, BinaryOp, CallExpr, ConstructorExpr,
    SwizzleAccess, UnaryOp,
    AssignTarget, StorageBufferDecl,
    SurfaceDecl, GeometryDecl, LightingDecl,
)


# ---------------------------------------------------------------------------
# AST construction helpers (same pattern as splat_expander)
# ---------------------------------------------------------------------------

def _lit(v) -> NumberLit:
    return NumberLit(str(v))

def _ref(name: str) -> VarRef:
    return VarRef(name)

def _call(fn: str, args: list) -> CallExpr:
    return CallExpr(_ref(fn), args)

def _binop(op: str, left, right) -> BinaryOp:
    return BinaryOp(op, left, right)

def _let(name: str, ty: str, expr) -> LetStmt:
    return LetStmt(name, ty, expr)

def _assign(name: str, expr) -> AssignStmt:
    return AssignStmt(AssignTarget(_ref(name)), expr)

def _swizzle(expr, components: str) -> SwizzleAccess:
    return SwizzleAccess(expr, components)

def _ctor(ty: str, args: list) -> ConstructorExpr:
    return ConstructorExpr(ty, args)

def _neg(expr) -> UnaryOp:
    return UnaryOp("-", expr)


# ---------------------------------------------------------------------------
# Inline octahedron encode/decode (no stdlib dependency)
# ---------------------------------------------------------------------------

def _oct_encode_stmts(normal_var: str, result_var: str) -> list:
    """Generate AST statements for octahedron encoding of a unit normal.

    Input: `normal_var` (vec3), Output: `result_var` (vec2) in [0,1]².
    """
    stmts = []
    # let _s = abs(n.x) + abs(n.y) + abs(n.z);
    stmts.append(_let("_oct_s", "scalar", _binop("+",
        _binop("+",
            _call("abs", [_swizzle(_ref(normal_var), "x")]),
            _call("abs", [_swizzle(_ref(normal_var), "y")])),
        _call("abs", [_swizzle(_ref(normal_var), "z")]))))
    # let _ox = n.x / _s;  let _oy = n.y / _s;
    stmts.append(_let("_oct_ox", "scalar",
        _binop("/", _swizzle(_ref(normal_var), "x"), _ref("_oct_s"))))
    stmts.append(_let("_oct_oy", "scalar",
        _binop("/", _swizzle(_ref(normal_var), "y"), _ref("_oct_s"))))
    # Wrap for negative hemisphere: if n.z < 0, wrap
    # signs_x = step(0, _ox) * 2 - 1
    stmts.append(_let("_oct_sx", "scalar",
        _binop("-",
            _binop("*", _call("step", [_lit("0.0"), _ref("_oct_ox")]), _lit("2.0")),
            _lit("1.0"))))
    stmts.append(_let("_oct_sy", "scalar",
        _binop("-",
            _binop("*", _call("step", [_lit("0.0"), _ref("_oct_oy")]), _lit("2.0")),
            _lit("1.0"))))
    # wrapped_x = (1 - abs(_oy)) * sign_x
    stmts.append(_let("_oct_wx", "scalar",
        _binop("*",
            _binop("-", _lit("1.0"), _call("abs", [_ref("_oct_oy")])),
            _ref("_oct_sx"))))
    stmts.append(_let("_oct_wy", "scalar",
        _binop("*",
            _binop("-", _lit("1.0"), _call("abs", [_ref("_oct_ox")])),
            _ref("_oct_sy"))))
    # Select based on n.z >= 0
    # mix_factor = step(0, n.z)
    stmts.append(_let("_oct_fz", "scalar",
        _call("step", [_lit("0.0"), _swizzle(_ref(normal_var), "z")])))
    # final_x = mix(wrapped_x, _ox, mix_factor)
    stmts.append(_let("_oct_rx", "scalar",
        _binop("+",
            _binop("*", _ref("_oct_wx"), _binop("-", _lit("1.0"), _ref("_oct_fz"))),
            _binop("*", _ref("_oct_ox"), _ref("_oct_fz")))))
    stmts.append(_let("_oct_ry", "scalar",
        _binop("+",
            _binop("*", _ref("_oct_wy"), _binop("-", _lit("1.0"), _ref("_oct_fz"))),
            _binop("*", _ref("_oct_oy"), _ref("_oct_fz")))))
    # Map to [0,1]: result = vec2(rx, ry) * 0.5 + 0.5
    stmts.append(_let(result_var, "vec2",
        _binop("+",
            _binop("*",
                _ctor("vec2", [_ref("_oct_rx"), _ref("_oct_ry")]),
                _lit("0.5")),
            _ctor("vec2", [_lit("0.5"), _lit("0.5")]))))
    return stmts


def _oct_decode_stmts(oct_var: str, result_var: str) -> list:
    """Generate AST statements for octahedron decoding.

    Input: `oct_var` (vec2) in [0,1]², Output: `result_var` (vec3) unit normal.
    """
    stmts = []
    # Map from [0,1] to [-1,1]
    stmts.append(_let("_dec_f", "vec2",
        _binop("-",
            _binop("*", _ref(oct_var), _lit("2.0")),
            _ctor("vec2", [_lit("1.0"), _lit("1.0")]))))
    # n.z = 1 - abs(f.x) - abs(f.y)
    stmts.append(_let("_dec_nz", "scalar",
        _binop("-",
            _binop("-", _lit("1.0"),
                _call("abs", [_swizzle(_ref("_dec_f"), "x")])),
            _call("abs", [_swizzle(_ref("_dec_f"), "y")]))))
    # t = max(-nz, 0)
    stmts.append(_let("_dec_t", "scalar",
        _call("max", [_neg(_ref("_dec_nz")), _lit("0.0")])))
    # signs
    stmts.append(_let("_dec_sx", "scalar",
        _binop("-",
            _binop("*",
                _call("step", [_lit("0.0"), _swizzle(_ref("_dec_f"), "x")]),
                _lit("2.0")),
            _lit("1.0"))))
    stmts.append(_let("_dec_sy", "scalar",
        _binop("-",
            _binop("*",
                _call("step", [_lit("0.0"), _swizzle(_ref("_dec_f"), "y")]),
                _lit("2.0")),
            _lit("1.0"))))
    # adjusted
    stmts.append(_let("_dec_ax", "scalar",
        _binop("+", _swizzle(_ref("_dec_f"), "x"),
            _binop("*", _ref("_dec_sx"), _neg(_ref("_dec_t"))))))
    stmts.append(_let("_dec_ay", "scalar",
        _binop("+", _swizzle(_ref("_dec_f"), "y"),
            _binop("*", _ref("_dec_sy"), _neg(_ref("_dec_t"))))))
    stmts.append(_let(result_var, "vec3",
        _call("normalize", [
            _ctor("vec3", [_ref("_dec_ax"), _ref("_dec_ay"), _ref("_dec_nz")])])))
    return stmts


# ---------------------------------------------------------------------------
# G-buffer vertex stage
# ---------------------------------------------------------------------------

def _build_gbuffer_vertex(geometry, surface_expander_mod) -> StageBlock:
    """Build the G-buffer geometry pass vertex stage.

    Reuses _expand_geometry_to_vertex from surface_expander.
    """
    stage = surface_expander_mod._expand_geometry_to_vertex(geometry)
    stage._deferred_pass = "gbuffer"
    stage._output_stem_suffix = "gbuf"
    stage._descriptor_set_offset = 0
    return stage


# ---------------------------------------------------------------------------
# G-buffer fragment stage
# ---------------------------------------------------------------------------

def _build_gbuffer_fragment(surface, geometry, module, schedule) -> StageBlock:
    """Build the G-buffer fragment stage with MRT outputs.

    Outputs:
      gbuf_rt0 (location 0): albedo.rgb + metallic
      gbuf_rt1 (location 1): oct_normal.rg + roughness + 0.0
      gbuf_rt2 (location 2): emission.rgb + 1.0
    """
    from luxc.expansion.surface_expander import _get_layer_args, _infer_output_type

    stage = StageBlock(stage_type="fragment")

    # Fragment inputs from geometry outputs
    frag_inputs = []
    if geometry and geometry.outputs:
        for binding in geometry.outputs.bindings:
            if binding.name != "clip_pos":
                inp_type = _infer_output_type(binding.name)
                frag_inputs.append((binding.name, inp_type))
    else:
        frag_inputs = [
            ("frag_normal", "vec3"),
            ("frag_pos", "vec3"),
        ]

    for name, type_name in frag_inputs:
        v = VarDecl(name, type_name)
        v._is_input = True
        stage.inputs.append(v)

    # Add frag_uv if surface has samplers and it's not already present
    has_frag_uv = any(name == "frag_uv" for name, _ in frag_inputs)
    if surface.samplers and not has_frag_uv:
        frag_inputs.append(("frag_uv", "vec2"))
        v = VarDecl("frag_uv", "vec2")
        v._is_input = True
        stage.inputs.append(v)

    # MRT outputs (3 render targets)
    for rt_name in ("gbuf_rt0", "gbuf_rt1", "gbuf_rt2"):
        out = VarDecl(rt_name, "vec4")
        out._is_input = False
        stage.outputs.append(out)

    # Material properties uniform (from surface) — NO Light uniform
    if surface.properties:
        props = surface.properties
        stage.uniforms.append(UniformBlock(props.name, [
            BlockField(f.name, f.type_name) for f in props.fields
        ]))
        if not hasattr(stage, '_properties_defaults'):
            stage._properties_defaults = {}
        for f in props.fields:
            if f.default is not None:
                stage._properties_defaults[(props.name, f.name)] = f.default

    # Surface samplers only (no lighting samplers)
    for sam in surface.samplers:
        stage.samplers.append(SamplerDecl(sam.name, type_name=sam.type_name))

    # Generate main body
    body = _generate_gbuffer_main(surface, frag_inputs)
    stage.functions.append(FunctionDef("main", [], None, body))

    stage._deferred_pass = "gbuffer"
    stage._output_stem_suffix = "gbuf"
    stage._descriptor_set_offset = 1
    return stage


def _generate_gbuffer_main(surface, frag_inputs) -> list:
    """Generate main() body for the G-buffer fragment shader.

    Extracts material params from surface layers, encodes normal,
    writes to 3 MRT outputs.
    """
    from luxc.expansion.surface_expander import _get_layer_args

    body = []

    # Determine input variable names
    has_tangent = any(n == "world_tangent" for n, _ in frag_inputs)
    has_frag_uv = any(n == "frag_uv" for n, _ in frag_inputs)

    # UV alias
    if has_frag_uv:
        body.append(_let("uv", "vec2", _ref("frag_uv")))

    # Index layers by name
    layers_by_name = {}
    if surface.layers is not None:
        for layer in surface.layers:
            layers_by_name[layer.name] = layer

    # --- Normal ---
    if "normal_map" in layers_by_name and has_tangent:
        nmap_args = _get_layer_args(layers_by_name["normal_map"])
        map_expr = nmap_args.get("map")
        body.append(_let("normal_map_raw", "vec3", map_expr))
        body.append(_let("n", "vec3", _call("tbn_perturb_normal", [
            _ref("normal_map_raw"),
            _call("normalize", [_ref("world_normal")]),
            _call("normalize", [_ref("world_tangent")]),
            _call("normalize", [_ref("world_bitangent")]),
        ])))
    else:
        normal_var = next(
            (n for n, _ in frag_inputs if n in ("world_normal", "frag_normal")),
            "world_normal",
        )
        body.append(_let("n", "vec3", _call("normalize", [_ref(normal_var)])))

    # --- Base layer material params ---
    if "base" in layers_by_name:
        base_args = _get_layer_args(layers_by_name["base"])
        albedo_expr = base_args.get("albedo",
            _ctor("vec3", [_lit("0.5")]))
        roughness_expr = base_args.get("roughness", _lit("0.5"))
        metallic_expr = base_args.get("metallic", _lit("0.0"))
    else:
        albedo_expr = _ctor("vec3", [_lit("0.5")])
        roughness_expr = _lit("0.5")
        metallic_expr = _lit("0.0")

    body.append(_let("gbuf_albedo", "vec3", albedo_expr))
    body.append(_let("gbuf_roughness", "scalar", roughness_expr))
    body.append(_let("gbuf_metallic", "scalar", metallic_expr))

    # --- Emission ---
    if "emission" in layers_by_name:
        em_args = _get_layer_args(layers_by_name["emission"])
        em_expr = em_args.get("color", _ctor("vec3", [_lit("0.0")]))
        body.append(_let("gbuf_emission", "vec3", em_expr))
    else:
        body.append(_let("gbuf_emission", "vec3", _ctor("vec3", [_lit("0.0")])))

    # --- Octahedron encode normal ---
    body.extend(_oct_encode_stmts("n", "gbuf_oct_normal"))

    # --- Write MRT outputs ---
    # RT0: albedo.rgb + metallic
    body.append(_assign("gbuf_rt0",
        _ctor("vec4", [_ref("gbuf_albedo"), _ref("gbuf_metallic")])))
    # RT1: oct_normal.rg + roughness + 0.0
    body.append(_assign("gbuf_rt1",
        _ctor("vec4", [
            _swizzle(_ref("gbuf_oct_normal"), "x"),
            _swizzle(_ref("gbuf_oct_normal"), "y"),
            _ref("gbuf_roughness"),
            _lit("0.0")])))
    # RT2: emission.rgb + 1.0
    body.append(_assign("gbuf_rt2",
        _ctor("vec4", [_ref("gbuf_emission"), _lit("1.0")])))

    return body


# ---------------------------------------------------------------------------
# Lighting vertex stage (fullscreen triangle)
# ---------------------------------------------------------------------------

def _build_lighting_vertex() -> StageBlock:
    """Build a fullscreen triangle vertex shader.

    Generates 3 vertices covering the full screen using vertex_index.
    No vertex buffer inputs needed.
    """
    stage = StageBlock(stage_type="vertex")

    # Output: UV for fragment shader
    out = VarDecl("frag_uv", "vec2")
    out._is_input = False
    stage.outputs.append(out)

    body = []
    # let idx = vertex_index;
    # UV.x = (idx == 1) ? 2.0 : 0.0  → use float(idx & 1) * 2.0
    # UV.y = (idx == 2) ? 2.0 : 0.0  → use float(idx >> 1) * 2.0  (via idx / 2)
    # But we use fmod/floor since Lux may not have bitops.
    #
    # Standard fullscreen triangle trick:
    #   uv.x = float((idx << 1) & 2) = float(idx % 2) * 2.0
    #   uv.y = float(idx & 2)        = floor(float(idx) / 2.0) * 2.0
    #   pos  = vec4(uv * 2.0 - 1.0, 0.0, 1.0)
    #
    # Simplified with pure math:
    #   let fx = fmod(float(vertex_index) * 2.0, 4.0);  // 0, 2, 0
    #   Actually, the canonical pattern is:
    #   uv.x = (vertex_index == 1) ? 2 : 0
    #   uv.y = (vertex_index == 2) ? 2 : 0
    # Using step-based approach:
    #   vx = float(vertex_index)
    #   u = step(0.5, fmod(vx, 2.0)) * 2.0
    #   v = step(1.5, vx) * 2.0

    # let vx = scalar(vertex_index);
    body.append(_let("_fs_vx", "scalar",
        _binop("*", _ref("vertex_index"), _lit("1.0"))))

    # u = step(0.5, mod(vx, 2.0)) * 2.0
    # For vertex 0: mod(0,2)=0 → step=0 → u=0
    # For vertex 1: mod(1,2)=1 → step=1 → u=2
    # For vertex 2: mod(2,2)=0 → step=0 → u=0
    body.append(_let("_fs_u", "scalar",
        _binop("*",
            _call("step", [_lit("0.5"),
                _call("mod", [_ref("_fs_vx"), _lit("2.0")])]),
            _lit("2.0"))))

    # v = step(1.5, vx) * 2.0
    # For vertex 0: step(1.5,0)=0 → v=0
    # For vertex 1: step(1.5,1)=0 → v=0
    # For vertex 2: step(1.5,2)=1 → v=2
    body.append(_let("_fs_v", "scalar",
        _binop("*",
            _call("step", [_lit("1.5"), _ref("_fs_vx")]),
            _lit("2.0"))))

    # frag_uv = vec2(u, v)
    body.append(_assign("frag_uv",
        _ctor("vec2", [_ref("_fs_u"), _ref("_fs_v")])))

    # builtin_position = vec4(u * 2.0 - 1.0, v * 2.0 - 1.0, 0.0, 1.0)
    body.append(_assign("builtin_position",
        _ctor("vec4", [
            _binop("-", _binop("*", _ref("_fs_u"), _lit("2.0")), _lit("1.0")),
            _binop("-", _binop("*", _ref("_fs_v"), _lit("2.0")), _lit("1.0")),
            _lit("0.0"),
            _lit("1.0")])))

    stage.functions.append(FunctionDef("main", [], None, body))
    stage._deferred_pass = "lighting"
    stage._output_stem_suffix = "light"
    stage._descriptor_set_offset = 0
    return stage


# ---------------------------------------------------------------------------
# Lighting fragment stage
# ---------------------------------------------------------------------------

def _build_lighting_fragment(surface, module, schedule, lighting, bindless) -> StageBlock:
    """Build the deferred lighting pass fragment shader.

    Reads G-buffer textures, reconstructs world position from depth,
    evaluates full lighting (same as forward pass).
    """
    from luxc.expansion.surface_expander import (
        _get_layer_args, _emit_multi_light_from_lighting,
        _emit_tonemap_output, _detect_multi_light,
    )

    stage = StageBlock(stage_type="fragment")

    # Input: UV from fullscreen vertex
    v_in = VarDecl("frag_uv", "vec2")
    v_in._is_input = True
    stage.inputs.append(v_in)

    # Output: final color
    out = VarDecl("color", "vec4")
    out._is_input = False
    stage.outputs.append(out)

    # G-buffer texture samplers
    for tex_name in ("gbuf_tex0", "gbuf_tex1", "gbuf_tex2", "gbuf_depth"):
        stage.samplers.append(SamplerDecl(tex_name, type_name="sampler2d"))

    # DeferredCamera uniform (inv_view_proj + view_pos)
    stage.uniforms.append(UniformBlock("DeferredCamera", [
        BlockField("inv_view_proj", "mat4"),
        BlockField("view_pos", "vec3"),
    ]))

    # Lighting properties uniform (from lighting block)
    if lighting and lighting.properties:
        props = lighting.properties
        stage.uniforms.append(UniformBlock(props.name, [
            BlockField(f.name, f.type_name) for f in props.fields
        ]))
        if not hasattr(stage, '_properties_defaults'):
            stage._properties_defaults = {}
        for f in props.fields:
            if f.default is not None:
                stage._properties_defaults[(props.name, f.name)] = f.default

    # Storage buffers for multi-light
    if lighting and lighting.layers:
        for layer in lighting.layers:
            if layer.name == "multi_light":
                stage.storage_buffers.append(
                    StorageBufferDecl("lights", "LightData"))
                _ml_arg_names = [a.name for a in layer.args]
                if "shadow_map" in _ml_arg_names:
                    stage.storage_buffers.append(
                        StorageBufferDecl("shadow_matrices", "ShadowEntry"))
                break

    # IBL samplers from lighting block
    if lighting:
        for sam in lighting.samplers:
            stage.samplers.append(SamplerDecl(sam.name, type_name=sam.type_name))

    # Generate main body
    body = _generate_lighting_main(surface, module, schedule, lighting)
    stage.functions.append(FunctionDef("main", [], None, body))

    stage._deferred_pass = "lighting"
    stage._output_stem_suffix = "light"
    stage._deferred_config = _get_deferred_config(schedule)
    stage._descriptor_set_offset = 0
    return stage


def _get_deferred_config(schedule) -> dict:
    """Extract deferred-specific config from schedule."""
    if not schedule:
        return {}
    return {
        "normal_encoding": schedule.get("normal_encoding", "octahedron"),
        "gbuffer_precision": schedule.get("gbuffer_precision", "standard"),
        "light_culling": schedule.get("light_culling", "none"),
        "tile_size": schedule.get("tile_size", "16"),
    }


def _generate_lighting_main(surface, module, schedule, lighting) -> list:
    """Generate main() body for the deferred lighting fragment shader.

    1. Sample G-buffer textures at frag_uv
    2. Unpack albedo/metallic, normal/roughness, emission
    3. Reconstruct world position from depth via inv_view_proj
    4. Compute view direction
    5. Evaluate lighting (multi-light or single directional)
    6. Tonemap + output
    """
    from luxc.expansion.surface_expander import (
        _get_layer_args, _emit_multi_light_from_lighting,
        _emit_tonemap_output, _detect_multi_light,
    )

    body = []

    # --- Sample G-buffer textures ---
    body.append(_let("_gb0", "vec4",
        _call("sample", [_ref("gbuf_tex0"), _ref("frag_uv")])))
    body.append(_let("_gb1", "vec4",
        _call("sample", [_ref("gbuf_tex1"), _ref("frag_uv")])))
    body.append(_let("_gb2", "vec4",
        _call("sample", [_ref("gbuf_tex2"), _ref("frag_uv")])))
    body.append(_let("_gb_depth_sample", "vec4",
        _call("sample", [_ref("gbuf_depth"), _ref("frag_uv")])))
    body.append(_let("_gb_depth", "scalar",
        _swizzle(_ref("_gb_depth_sample"), "x")))

    # --- Unpack RT0: albedo + metallic ---
    body.append(_let("layer_albedo", "vec3", _swizzle(_ref("_gb0"), "xyz")))
    body.append(_let("layer_metallic", "scalar", _swizzle(_ref("_gb0"), "w")))

    # --- Unpack RT1: octahedron normal + roughness ---
    body.append(_let("_gb_oct", "vec2", _swizzle(_ref("_gb1"), "xy")))
    body.extend(_oct_decode_stmts("_gb_oct", "n"))
    body.append(_let("layer_roughness", "scalar", _swizzle(_ref("_gb1"), "z")))

    # --- Unpack RT2: emission ---
    body.append(_let("gbuf_emission", "vec3", _swizzle(_ref("_gb2"), "xyz")))

    # --- Reconstruct world position from depth ---
    # ndc = vec4(uv * 2 - 1, depth, 1)
    body.append(_let("_ndc_xy", "vec2",
        _binop("-",
            _binop("*", _ref("frag_uv"), _lit("2.0")),
            _ctor("vec2", [_lit("1.0"), _lit("1.0")]))))
    body.append(_let("_ndc", "vec4",
        _ctor("vec4", [
            _swizzle(_ref("_ndc_xy"), "x"),
            _swizzle(_ref("_ndc_xy"), "y"),
            _ref("_gb_depth"),
            _lit("1.0")])))
    # world_h = inv_view_proj * ndc
    body.append(_let("_world_h", "vec4",
        _binop("*", _ref("inv_view_proj"), _ref("_ndc"))))
    # world_pos = world_h.xyz / world_h.w
    body.append(_let("world_pos", "vec3",
        _binop("/",
            _swizzle(_ref("_world_h"), "xyz"),
            _swizzle(_ref("_world_h"), "www"))))

    # --- View direction ---
    body.append(_let("v", "vec3",
        _call("normalize", [
            _binop("-", _ref("view_pos"), _ref("world_pos"))])))

    # --- Lighting evaluation ---
    _has_multi_light = _detect_multi_light(lighting)

    if _has_multi_light:
        # Dummy l for layers that reference it
        body.append(_let("l", "vec3",
            _ctor("vec3", [_lit("0.0"), _lit("1.0"), _lit("0.0")])))
        body.append(_let("n_dot_v", "scalar",
            _call("max", [_call("dot", [_ref("n"), _ref("v")]), _lit("0.001")])))
        result_var = _emit_multi_light_from_lighting(body, lighting, "world_pos")
    else:
        # Single directional light
        lighting_by_name = {}
        if lighting and lighting.layers:
            lighting_by_name = {layer.name: layer for layer in lighting.layers}

        if lighting and "directional" in lighting_by_name:
            dir_args = _get_layer_args(lighting_by_name["directional"])
            body.append(_let("l", "vec3",
                _call("normalize", [
                    dir_args.get("direction", _ref("light_dir"))])))
            body.append(_let("light_color", "vec3",
                dir_args.get("color", _ctor("vec3", [_lit("1.0")]))))
        else:
            body.append(_let("l", "vec3",
                _call("normalize", [
                    _ctor("vec3", [_lit("0.5"), _lit("1.0"), _lit("0.3")])])))
            body.append(_let("light_color", "vec3",
                _ctor("vec3", [_lit("1.0"), _lit("0.98"), _lit("0.95")])))

        body.append(_let("n_dot_v", "scalar",
            _call("max", [_call("dot", [_ref("n"), _ref("v")]), _lit("0.001")])))

        # Direct PBR lighting
        body.append(_let("direct", "vec3",
            _call("gltf_pbr", [
                _ref("n"), _ref("v"), _ref("l"),
                _ref("layer_albedo"), _ref("layer_roughness"),
                _ref("layer_metallic")])))
        body.append(_let("direct_lit", "vec3",
            _binop("*", _ref("direct"), _ref("light_color"))))
        result_var = "direct_lit"

    # --- IBL (if present in lighting block) ---
    ibl_args = None
    if lighting and lighting.layers:
        for layer in lighting.layers:
            if layer.name == "ibl":
                ibl_args = _get_layer_args(layer)
                break

    if ibl_args is not None:
        specular_map = ibl_args.get("specular_map")
        irradiance_map = ibl_args.get("irradiance_map")
        brdf_lut_ref = ibl_args.get("brdf_lut")

        body.append(_let("r", "vec3",
            _call("reflect", [
                _binop("*", _ref("v"), _neg(_lit("1.0"))),
                _ref("n")])))
        body.append(_let("prefiltered", "vec3",
            _swizzle(
                _call("sample_lod", [
                    specular_map, _ref("r"),
                    _binop("*", _ref("layer_roughness"), _lit("8.0"))]),
                "xyz")))
        body.append(_let("irradiance", "vec3",
            _swizzle(
                _call("sample", [irradiance_map, _ref("n")]),
                "xyz")))
        body.append(_let("brdf_sample", "vec2",
            _swizzle(
                _call("sample", [
                    brdf_lut_ref,
                    _ctor("vec2", [_ref("n_dot_v"), _ref("layer_roughness")])]),
                "xy")))
        body.append(_let("ambient", "vec3",
            _call("ibl_contribution", [
                _ref("layer_albedo"), _ref("layer_roughness"),
                _ref("layer_metallic"),
                _ref("n_dot_v"), _ref("prefiltered"),
                _ref("irradiance"), _ref("brdf_sample")])))
        body.append(_let("lit_with_ibl", "vec3",
            _binop("+", _ref(result_var), _ref("ambient"))))
        result_var = "lit_with_ibl"

    # --- Add emission ---
    body.append(_let("with_emission", "vec3",
        _binop("+", _ref(result_var), _ref("gbuf_emission"))))
    result_var = "with_emission"

    # --- Tonemap ---
    output_var = _emit_tonemap_output(body, result_var, schedule)

    # --- Output ---
    body.append(_assign("color",
        _ctor("vec4", [_ref(output_var), _lit("1.0")])))

    return body


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_deferred_pipeline(
    surface: SurfaceDecl,
    geometry: GeometryDecl | None,
    module: Module,
    schedule: dict[str, str] | None = None,
    lighting: LightingDecl | None = None,
    bindless: bool = False,
) -> list[StageBlock]:
    """Expand a deferred pipeline into 4 shader stages.

    Returns: [gbuffer_vert, gbuffer_frag, lighting_vert, lighting_frag]
    """
    import luxc.expansion.surface_expander as se_mod

    stages = []

    # 1. G-buffer vertex (reuse geometry expansion)
    if geometry:
        gbuf_vert = _build_gbuffer_vertex(geometry, se_mod)
    else:
        # Default geometry: position + normal
        gbuf_vert = StageBlock(stage_type="vertex")
        for name, ty in [("position", "vec3"), ("normal", "vec3")]:
            v = VarDecl(name, ty)
            v._is_input = True
            gbuf_vert.inputs.append(v)
        for name, ty in [("frag_normal", "vec3"), ("frag_pos", "vec3")]:
            out = VarDecl(name, ty)
            out._is_input = False
            gbuf_vert.outputs.append(out)
        body = [
            _assign("builtin_position",
                _ctor("vec4", [_ref("position"), _lit("1.0")])),
            _assign("frag_normal", _ref("normal")),
            _assign("frag_pos", _ref("position")),
        ]
        gbuf_vert.functions.append(FunctionDef("main", [], None, body))
        gbuf_vert._deferred_pass = "gbuffer"
        gbuf_vert._output_stem_suffix = "gbuf"
        gbuf_vert._descriptor_set_offset = 0
    stages.append(gbuf_vert)

    # 2. G-buffer fragment
    gbuf_frag = _build_gbuffer_fragment(surface, geometry, module, schedule)
    # Tag with deferred config for reflection
    gbuf_frag._deferred_config = _get_deferred_config(schedule)
    stages.append(gbuf_frag)

    # 3. Lighting vertex (fullscreen triangle)
    light_vert = _build_lighting_vertex()
    stages.append(light_vert)

    # 4. Lighting fragment
    light_frag = _build_lighting_fragment(surface, module, schedule, lighting, bindless)
    stages.append(light_frag)

    return stages
