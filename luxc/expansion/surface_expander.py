"""Surface declaration expander.

Takes a `surface` declaration (declarative material) and generates
the corresponding fragment stage block with BRDF evaluation, light
integration, and proper uniform bindings.

If a `geometry` declaration is also present, generates the vertex
stage block with the declared transforms and output bindings.

If a `pipeline` declaration is present, ties geometry + surface
together and generates both stages.
"""

from __future__ import annotations
from luxc.parser.ast_nodes import (
    Module, StageBlock, VarDecl, UniformBlock, PushBlock, BlockField,
    SamplerDecl,
    FunctionDef, Param, LetStmt, AssignStmt, ReturnStmt, ExprStmt,
    NumberLit, VarRef, BinaryOp, CallExpr, ConstructorExpr,
    SwizzleAccess, UnaryOp,
    AssignTarget, SurfaceDecl, SurfaceSampler, LayerCall, LayerArg,
    GeometryDecl, PipelineDecl, LightingDecl,
    ScheduleDecl, EnvironmentDecl, ProceduralDecl,
    RayPayloadDecl, HitAttributeDecl, AccelDecl, StorageImageDecl,
    StorageBufferDecl, BindlessTextureArrayDecl, IndexAccess, FieldAccess,
    IfStmt, TaskPayloadDecl, PropertiesBlock,
    SplatDecl,
)
from luxc.expansion.splat_expander import expand_splat_pipeline


def _is_openpbr(module):
    """Check if module imports openpbr."""
    return any(imp.module_name == "openpbr" for imp in getattr(module, 'imports', []))


# --- Schedule strategy tables ---

_DEFAULT_STRATEGIES = {
    "fresnel": "schlick",
    "distribution": "ggx",
    "geometry_term": "smith_ggx",
    "tonemap": "none",
    # OpenPBR-specific
    "diffuse_model": "eon",
    "fuzz_model": "charlie",
    "specular_fresnel": "exact",
    "coat_fresnel": "exact",
}

_STRATEGY_FUNCTIONS = {
    ("fresnel", "schlick"): "fresnel_schlick",
    ("fresnel", "schlick_roughness"): "fresnel_schlick_roughness",
    ("distribution", "ggx"): "ggx_ndf",
    ("distribution", "ggx_fast"): "ggx_ndf_fast",
    ("geometry_term", "smith_ggx"): "smith_ggx",
    ("geometry_term", "smith_ggx_fast"): "smith_ggx_fast",
    ("tonemap", "aces"): "tonemap_aces",
    ("tonemap", "reinhard"): "tonemap_reinhard",
    ("tonemap", "none"): None,
    # OpenPBR diffuse model
    ("diffuse_model", "eon"): "openpbr_eon_diffuse",
    ("diffuse_model", "lambert"): None,  # built-in Lambert (no function needed)
    ("diffuse_model", "burley"): "burley_diffuse",
    # OpenPBR fuzz
    ("fuzz_model", "charlie"): "openpbr_fuzz_brdf",
    # OpenPBR Fresnel
    ("specular_fresnel", "exact"): "openpbr_fresnel_dielectric",
    ("specular_fresnel", "schlick"): "fresnel_schlick",
    ("coat_fresnel", "exact"): "openpbr_fresnel_dielectric",
    ("coat_fresnel", "schlick"): "fresnel_schlick",
}

_VALID_SLOTS = set(_DEFAULT_STRATEGIES.keys())


def _resolve_schedule(module: Module, name: str) -> dict[str, str]:
    """Resolve a schedule name to a strategy dict."""
    for sched in module.schedules:
        if sched.name == name:
            strategies = dict(_DEFAULT_STRATEGIES)
            for member in sched.members:
                if member.name not in _VALID_SLOTS:
                    raise ValueError(
                        f"Unknown schedule slot '{member.name}' in schedule '{name}'. "
                        f"Valid slots: {sorted(_VALID_SLOTS)}"
                    )
                strategies[member.name] = member.value
            return strategies
    raise ValueError(f"Schedule '{name}' not found")


def expand_surfaces(module: Module, pipeline_filter: str | None = None, bindless: bool = False) -> None:
    """Expand pipeline/surface/geometry declarations into stage blocks.

    If a pipeline declaration references a surface and geometry by name,
    generate vertex + fragment stages. Otherwise, if a surface exists
    without a pipeline, generate just the fragment stage.

    For RT pipelines (mode: raytrace), generates raygen, closest_hit,
    miss, and optionally any_hit/intersection stages.

    If pipeline_filter is set, only expand the named pipeline.
    """
    # Index declarations by name
    surfaces = {s.name: s for s in module.surfaces}
    geometries = {g.name: g for g in module.geometries}
    environments = {e.name: e for e in module.environments}
    procedurals = {p.name: p for p in module.procedurals}
    lightings = {l.name: l for l in module.lightings}
    splats = {s.name: s for s in getattr(module, 'splats', [])}

    for pipeline in module.pipelines:
        # Filter: skip pipelines that don't match the filter
        if pipeline_filter and pipeline.name != pipeline_filter:
            continue
        geo_name = None
        surf_name = None
        splat_name = None
        schedule_name = None
        env_name = None
        lighting_name = None
        mode = "rasterize"
        max_bounces = 1
        procedural_name = None
        use_task_shader = False
        for member in pipeline.members:
            if member.name == "geometry":
                if isinstance(member.value, VarRef):
                    geo_name = member.value.name
            elif member.name == "surface":
                if isinstance(member.value, VarRef):
                    surf_name = member.value.name
            elif member.name == "schedule":
                if isinstance(member.value, VarRef):
                    schedule_name = member.value.name
            elif member.name == "mode":
                if isinstance(member.value, VarRef):
                    mode = member.value.name
            elif member.name == "environment":
                if isinstance(member.value, VarRef):
                    env_name = member.value.name
            elif member.name == "max_bounces":
                if isinstance(member.value, NumberLit):
                    max_bounces = int(member.value.value)
            elif member.name == "procedural":
                if isinstance(member.value, VarRef):
                    procedural_name = member.value.name
            elif member.name == "lighting":
                if isinstance(member.value, VarRef):
                    lighting_name = member.value.name
            elif member.name == "splat":
                if isinstance(member.value, VarRef):
                    splat_name = member.value.name
            elif member.name == "use_task_shader":
                if isinstance(member.value, VarRef) and member.value.name == "true":
                    use_task_shader = True

        # Resolve lighting block if specified
        lighting = lightings.get(lighting_name) if lighting_name else None

        # Resolve schedule if specified
        schedule = None
        if schedule_name:
            schedule = _resolve_schedule(module, schedule_name)

        if mode == "raytrace":
            # RT pipeline expansion
            surface = surfaces.get(surf_name) if surf_name else None
            environment = environments.get(env_name) if env_name else None
            procedural = procedurals.get(procedural_name) if procedural_name else None
            stages = _expand_rt_pipeline(
                surface, environment, procedural, module, schedule, max_bounces,
                bindless=bindless, lighting=lighting,
            )
            module.stages.extend(stages)
        elif mode == "mesh_shader":
            # Mesh shader pipeline expansion
            surface = surfaces.get(surf_name) if surf_name else None
            geometry = geometries.get(geo_name) if geo_name else None
            stages = _expand_mesh_pipeline(
                surface, geometry, module, schedule, pipeline, use_task_shader,
                bindless=bindless, lighting=lighting,
            )
            module.stages.extend(stages)
        elif mode == "gaussian_splat":
            # Gaussian splatting pipeline expansion
            splat = splats.get(splat_name) if splat_name else None
            if splat is None:
                raise ValueError(
                    f"Pipeline '{pipeline.name}' has mode: gaussian_splat but no "
                    f"splat declaration found (splat: {splat_name})"
                )
            stages = expand_splat_pipeline(splat, pipeline, module)
            module.stages.extend(stages)
        elif mode == "compute":
            pass  # Compute stages are written directly, no expansion needed
        elif surf_name and surf_name in surfaces:
            surface = surfaces[surf_name]
            geometry = geometries.get(geo_name) if geo_name else None
            stages = _expand_pipeline(surface, geometry, module, schedule,
                                      bindless=bindless, lighting=lighting)
            # Tag fragment stages with set offset so uniforms don't clash
            # with vertex uniforms when combined in a pipeline
            for s in stages:
                if s.stage_type == "fragment" and geometry:
                    s._descriptor_set_offset = 1
            module.stages.extend(stages)

    # Standalone surfaces (no pipeline reference)
    referenced = set()
    for p in module.pipelines:
        for m in p.members:
            if m.name == "surface" and isinstance(m.value, VarRef):
                referenced.add(m.value.name)

    for surface in module.surfaces:
        if surface.name not in referenced:
            frag = _expand_surface_to_fragment(surface, module)
            module.stages.append(frag)


def _expand_pipeline(
    surface: SurfaceDecl,
    geometry: GeometryDecl | None,
    module: Module,
    schedule: dict[str, str] | None = None,
    bindless: bool = False,
    lighting: LightingDecl | None = None,
) -> list[StageBlock]:
    """Generate vertex and fragment stages from surface + geometry."""
    stages = []

    if geometry:
        vert = _expand_geometry_to_vertex(geometry)
        stages.append(vert)

    if bindless and surface.layers is not None:
        frag = _expand_bindless_fragment(surface, module, geometry, schedule,
                                         lighting=lighting)
    else:
        frag = _expand_surface_to_fragment(surface, module, geometry, schedule,
                                           lighting=lighting)
    stages.append(frag)
    return stages


def _expand_geometry_to_vertex(geometry: GeometryDecl) -> StageBlock:
    """Generate a vertex stage from a geometry declaration."""
    stage = StageBlock(stage_type="vertex")

    # Geometry fields become vertex inputs
    for field in geometry.fields:
        v = VarDecl(field.name, field.type_name)
        v._is_input = True
        stage.inputs.append(v)

    # Transform block becomes a uniform
    if geometry.transform:
        ub = UniformBlock(
            geometry.transform.name,
            [BlockField(f.name, f.type_name) for f in geometry.transform.fields],
        )
        stage.uniforms.append(ub)

    # Output bindings become vertex outputs + main function body
    body = []
    if geometry.outputs:
        for binding in geometry.outputs.bindings:
            if binding.name == "clip_pos":
                # clip_pos maps to builtin_position
                body.append(AssignStmt(
                    AssignTarget(VarRef("builtin_position")),
                    binding.value,
                ))
            else:
                # Regular output
                out_type = _infer_output_type(binding.name)
                v = VarDecl(binding.name, out_type)
                v._is_input = False
                stage.outputs.append(v)
                body.append(AssignStmt(
                    AssignTarget(VarRef(binding.name)),
                    binding.value,
                ))
    else:
        # Default: pass through position to builtin_position
        body.append(AssignStmt(
            AssignTarget(VarRef("builtin_position")),
            ConstructorExpr("vec4", [VarRef("position"), NumberLit("1.0")]),
        ))

    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


def _expand_surface_to_fragment(
    surface: SurfaceDecl,
    module: Module,
    geometry: GeometryDecl | None = None,
    schedule: dict[str, str] | None = None,
    lighting: LightingDecl | None = None,
) -> StageBlock:
    """Generate a fragment stage from a surface declaration.

    The surface declaration specifies material properties (brdf, normal, etc).
    The expander generates a fragment shader that:
    1. Takes interpolated inputs from the vertex stage
    2. Evaluates the BRDF for a single directional light
    3. Outputs the final color
    """
    stage = StageBlock(stage_type="fragment")

    # Standard fragment inputs (from geometry or defaults)
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

    # If the surface uses textures and frag_uv isn't already an input, add it
    has_frag_uv = any(name == "frag_uv" for name, _ in frag_inputs)
    if surface.samplers and not has_frag_uv:
        frag_inputs.append(("frag_uv", "vec2"))
        v = VarDecl("frag_uv", "vec2")
        v._is_input = True
        stage.inputs.append(v)

    # Fragment output
    out = VarDecl("color", "vec4")
    out._is_input = False
    stage.outputs.append(out)

    # Light uniform (from lighting block or legacy hardcoded)
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
    else:
        stage.uniforms.append(UniformBlock("Light", [
            BlockField("light_dir", "vec3"),
            BlockField("view_pos", "vec3"),
        ]))

    # Material properties uniform (from surface properties block)
    if surface.properties:
        props = surface.properties
        stage.uniforms.append(UniformBlock(props.name, [
            BlockField(f.name, f.type_name) for f in props.fields
        ]))
        # Store defaults for reflection emission
        if not hasattr(stage, '_properties_defaults'):
            stage._properties_defaults = {}
        for f in props.fields:
            if f.default is not None:
                stage._properties_defaults[(props.name, f.name)] = f.default

    # Add sampler declarations from the surface
    for sam in surface.samplers:
        stage.samplers.append(SamplerDecl(sam.name, type_name=sam.type_name))

    # Add sampler declarations from the lighting block
    if lighting:
        for sam in lighting.samplers:
            stage.samplers.append(SamplerDecl(sam.name, type_name=sam.type_name))

    # If multi_light layer present, add lights SSBO (and shadow_matrices if shadow_map arg)
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

    # Generate main function body - use layered path if surface has layers
    if surface.layers is not None:
        body = _generate_layered_main(surface, frag_inputs, schedule,
                                      module=module, lighting=lighting)
    else:
        body = _generate_surface_main(surface, frag_inputs, schedule)
    stage.functions.append(FunctionDef("main", [], None, body))

    return stage


def _generate_surface_main(
    surface: SurfaceDecl,
    frag_inputs: list[tuple[str, str]],
    schedule: dict[str, str] | None = None,
) -> list:
    """Generate the main() body for a surface-expanded fragment shader."""
    body = []
    # Use the surface declaration's source location for synthetic nodes
    _synth_loc = surface.loc

    # Determine which inputs we have
    has_normal = any(n in ("frag_normal", "world_normal") for n, _ in frag_inputs)
    has_pos = any(n in ("frag_pos", "world_pos") for n, _ in frag_inputs)
    normal_var = next((n for n, _ in frag_inputs if n in ("frag_normal", "world_normal")), "frag_normal")
    pos_var = next((n for n, _ in frag_inputs if n in ("frag_pos", "world_pos")), "frag_pos")

    # Normalize the surface normal
    body.append(LetStmt("n", "vec3", CallExpr(VarRef("normalize"), [VarRef(normal_var)]), loc=_synth_loc))

    # Compute view and light directions
    if has_pos:
        body.append(LetStmt("v", "vec3", CallExpr(VarRef("normalize"), [
            BinaryOp("-", VarRef("view_pos"), VarRef(pos_var))
        ])))
    else:
        body.append(LetStmt("v", "vec3", CallExpr(VarRef("normalize"), [
            ConstructorExpr("vec3", [NumberLit("0.0"), NumberLit("0.0"), NumberLit("1.0")])
        ])))

    body.append(LetStmt("l", "vec3", CallExpr(VarRef("normalize"), [VarRef("light_dir")]), loc=_synth_loc))

    # Evaluate the BRDF based on the surface members
    brdf_expr = None
    for member in surface.members:
        if member.name == "brdf":
            brdf_expr = member.value

    if brdf_expr is not None:
        # The BRDF expression is a call like `lambert(albedo)` or
        # `fresnel_blend(spec, diff)`. We evaluate it by calling
        # the appropriate function with n, v, l added.
        result_expr = _expand_brdf_call(brdf_expr, schedule)
        body.append(LetStmt("result", "vec3", result_expr))
    else:
        # Default: simple lambert with white albedo
        body.append(LetStmt("ndotl", "scalar", CallExpr(VarRef("max"), [
            CallExpr(VarRef("dot"), [VarRef("n"), VarRef("l")]),
            NumberLit("0.0"),
        ])))
        body.append(LetStmt("result", "vec3", BinaryOp("*",
            ConstructorExpr("vec3", [NumberLit("1.0")]),
            VarRef("ndotl"),
        )))

    # Add ambient illumination and exposure for visible output
    body.append(LetStmt("ambient", "vec3", BinaryOp("*",
        VarRef("result"),
        NumberLit("0.3"),
    )))
    body.append(LetStmt("lit", "vec3", BinaryOp("+",
        VarRef("result"),
        VarRef("ambient"),
    )))
    # Simple exposure boost (PBR outputs are physically correct but dark)
    body.append(LetStmt("final_color", "vec3", BinaryOp("*",
        VarRef("lit"),
        NumberLit("2.5"),
    )))

    # Apply tonemap from schedule if specified
    tonemap_strategy = schedule.get("tonemap", "none") if schedule else "none"
    if tonemap_strategy == "aces":
        body.append(LetStmt("tonemapped", "vec3",
            CallExpr(VarRef("tonemap_aces"), [VarRef("final_color")])))
        output_color_var = "tonemapped"
    elif tonemap_strategy == "reinhard":
        body.append(LetStmt("tonemapped", "vec3",
            CallExpr(VarRef("tonemap_reinhard"), [VarRef("final_color")])))
        output_color_var = "tonemapped"
    else:
        output_color_var = "final_color"

    # Output final color
    body.append(AssignStmt(
        AssignTarget(VarRef("color")),
        ConstructorExpr("vec4", [VarRef(output_color_var), NumberLit("1.0")]),
    ))

    return body


def _get_layer_args(layer: LayerCall) -> dict:
    """Extract layer arguments as a name→expr dict."""
    return {arg.name: arg.value for arg in layer.args}


# --- Custom @layer function support ---

_BUILTIN_LAYER_NAMES = frozenset({
    "base", "normal_map", "ibl", "emission", "coat", "sheen", "transmission",
})

_OPENPBR_LAYER_NAMES = frozenset({
    "base", "specular", "transmission", "subsurface",
    "coat", "fuzz", "thin_film", "emission", "normal_map",
})


def _collect_layer_functions(module: Module) -> dict:
    """Scan module functions for @layer annotations, validate, return dict[name, FunctionDef]."""
    builtin_names = _OPENPBR_LAYER_NAMES if _is_openpbr(module) else _BUILTIN_LAYER_NAMES
    layer_fns = {}
    for fn in module.functions:
        if "layer" not in fn.attributes:
            continue
        if len(fn.params) < 4:
            raise ValueError(
                f"@layer function '{fn.name}' needs ≥4 params (base, n, v, l)"
            )
        if fn.return_type != "vec3":
            raise ValueError(
                f"@layer function '{fn.name}' must return vec3"
            )
        if fn.name in builtin_names:
            raise ValueError(
                f"@layer function '{fn.name}' conflicts with built-in layer"
            )
        layer_fns[fn.name] = fn
    return layer_fns


def _emit_custom_layer(layer, fn, result_var, body, prefix, rewrite_for_rt=False):
    """Generate LetStmts + CallExpr for one custom @layer function call.

    Maps LayerArg names to function params (after the 4 implicit ones: base, n, v, l).
    Returns the new result variable name.
    """
    layer_args = _get_layer_args(layer)
    custom_params = fn.params[4:]  # skip base, n, v, l

    call_args = [VarRef(result_var), VarRef("n"), VarRef("v"), VarRef("l")]
    for p in custom_params:
        expr = layer_args.get(p.name, None)
        if expr is None:
            raise ValueError(
                f"Layer '{layer.name}': missing arg '{p.name}' "
                f"required by @layer function '{fn.name}'"
            )
        if rewrite_for_rt:
            expr = _rewrite_sample_to_lod(expr)
        var_name = f"{prefix}_{p.name}"
        body.append(LetStmt(var_name, p.type_name, expr))
        call_args.append(VarRef(var_name))

    out_var = f"{prefix}_result"
    body.append(LetStmt(out_var, "vec3", CallExpr(VarRef(fn.name), call_args)))
    return out_var


# --- Poisson disk offsets for shadow filtering ---
_POISSON_DISK_16 = [
    (-0.94201624, -0.39906216), (0.94558609, -0.76890725),
    (-0.09418410, -0.92938870), (0.34495938, 0.29387760),
    (-0.91588581, 0.45771432), (-0.81544232, -0.87912464),
    (-0.38277543, 0.27676845), (0.97484398, 0.75648379),
    (0.44323325, -0.97511554), (0.53742981, -0.47373420),
    (-0.26496911, -0.41893023), (0.79197514, 0.19090188),
    (-0.24188840, 0.99706507), (-0.81409955, 0.91437590),
    (0.19984126, 0.78641367), (0.14383161, -0.14100790),
]


def _emit_hard_shadow(body, sfx):
    """Emit single comparison sample — binary hard shadow."""
    body.append(LetStmt(f"sf{sfx}", "scalar",
        CallExpr(VarRef("sample_compare"), [
            VarRef("shadow_maps"),
            ConstructorExpr("vec3", [
                SwizzleAccess(VarRef(f"suv{sfx}"), "x"),
                SwizzleAccess(VarRef(f"suv{sfx}"), "y"),
                VarRef(f"lsi{sfx}")]),
            SwizzleAccess(VarRef(f"suv{sfx}"), "z")])))


def _emit_pcf_shadow(body, sfx):
    """Emit 4-tap PCF shadow: 4 comparison samples at +-0.5 texel offsets, averaged."""
    # texel_size = 1.0 / resolution
    body.append(LetStmt(f"stxsz{sfx}", "scalar",
        BinaryOp("/", NumberLit("1.0"), VarRef(f"sres{sfx}"))))
    offsets = [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]
    for tap_i, (ox, oy) in enumerate(offsets):
        body.append(LetStmt(f"spc{tap_i}{sfx}", "scalar",
            CallExpr(VarRef("sample_compare"), [
                VarRef("shadow_maps"),
                ConstructorExpr("vec3", [
                    BinaryOp("+", SwizzleAccess(VarRef(f"suv{sfx}"), "x"),
                        BinaryOp("*", NumberLit(str(ox)), VarRef(f"stxsz{sfx}"))),
                    BinaryOp("+", SwizzleAccess(VarRef(f"suv{sfx}"), "y"),
                        BinaryOp("*", NumberLit(str(oy)), VarRef(f"stxsz{sfx}"))),
                    VarRef(f"lsi{sfx}")]),
                SwizzleAccess(VarRef(f"suv{sfx}"), "z")])))
    # Average the 4 samples using pcf_shadow_4 stdlib function
    body.append(LetStmt(f"sf{sfx}", "scalar",
        CallExpr(VarRef("pcf_shadow_4"), [
            VarRef(f"spc0{sfx}"), VarRef(f"spc1{sfx}"),
            VarRef(f"spc2{sfx}"), VarRef(f"spc3{sfx}")])))


def _emit_pcss_shadow(body, sfx):
    """Emit PCSS shadow: 16-sample blocker search + 16-sample variable-kernel PCF."""
    # texel_size = 1.0 / resolution
    body.append(LetStmt(f"stxsz{sfx}", "scalar",
        BinaryOp("/", NumberLit("1.0"), VarRef(f"sres{sfx}"))))

    # Search radius for blocker search (light_size * receiver_depth / resolution)
    body.append(LetStmt(f"ssrad{sfx}", "scalar",
        BinaryOp("*", VarRef(f"slsz{sfx}"),
            BinaryOp("*", SwizzleAccess(VarRef(f"suv{sfx}"), "z"),
                BinaryOp("/", NumberLit("1.0"), VarRef(f"sres{sfx}"))))))

    # --- Blocker search: 16 raw depth samples ---
    for tap_i, (ox, oy) in enumerate(_POISSON_DISK_16):
        body.append(LetStmt(f"sbd{tap_i}{sfx}", "vec4",
            CallExpr(VarRef("sample_array"), [
                VarRef("shadow_maps"),
                ConstructorExpr("vec2", [
                    BinaryOp("+", SwizzleAccess(VarRef(f"suv{sfx}"), "x"),
                        BinaryOp("*", NumberLit(str(ox)), VarRef(f"ssrad{sfx}"))),
                    BinaryOp("+", SwizzleAccess(VarRef(f"suv{sfx}"), "y"),
                        BinaryOp("*", NumberLit(str(oy)), VarRef(f"ssrad{sfx}")))]),
                VarRef(f"lsi{sfx}")])))

    # Average blockers using stdlib (16-sample version)
    body.append(LetStmt(f"sbavg{sfx}", "vec2",
        CallExpr(VarRef("pcss_average_blockers_16"), [
            *[SwizzleAccess(VarRef(f"sbd{j}{sfx}"), "x") for j in range(16)],
            SwizzleAccess(VarRef(f"suv{sfx}"), "z")])))

    # Penumbra size
    body.append(LetStmt(f"spen{sfx}", "scalar",
        CallExpr(VarRef("pcss_penumbra_size"), [
            SwizzleAccess(VarRef(f"sbavg{sfx}"), "x"),
            SwizzleAccess(VarRef(f"suv{sfx}"), "z"),
            VarRef(f"slsz{sfx}")])))

    # Filter radius = max(penumbra, 1 texel) / resolution
    body.append(LetStmt(f"sfrad{sfx}", "scalar",
        CallExpr(VarRef("max"), [
            VarRef(f"spen{sfx}"),
            VarRef(f"stxsz{sfx}")])))

    # --- Variable-kernel PCF: 16 comparison samples ---
    for tap_i, (ox, oy) in enumerate(_POISSON_DISK_16):
        body.append(LetStmt(f"spcs{tap_i}{sfx}", "scalar",
            CallExpr(VarRef("sample_compare"), [
                VarRef("shadow_maps"),
                ConstructorExpr("vec3", [
                    BinaryOp("+", SwizzleAccess(VarRef(f"suv{sfx}"), "x"),
                        BinaryOp("*", NumberLit(str(ox)), VarRef(f"sfrad{sfx}"))),
                    BinaryOp("+", SwizzleAccess(VarRef(f"suv{sfx}"), "y"),
                        BinaryOp("*", NumberLit(str(oy)), VarRef(f"sfrad{sfx}"))),
                    VarRef(f"lsi{sfx}")]),
                SwizzleAccess(VarRef(f"suv{sfx}"), "z")])))

    # Average the 16 PCF taps
    body.append(LetStmt(f"spcss_raw{sfx}", "scalar",
        CallExpr(VarRef("pcf_shadow_16"), [
            *[VarRef(f"spcs{j}{sfx}") for j in range(16)]])))

    # If no blockers found (blocker count == 0), fully lit
    body.append(LetStmt(f"sf{sfx}", "scalar",
        BinaryOp("+",
            BinaryOp("*", VarRef(f"spcss_raw{sfx}"),
                SwizzleAccess(VarRef(f"sbavg{sfx}"), "y")),
            BinaryOp("*", NumberLit("1.0"),
                BinaryOp("-", NumberLit("1.0"),
                    SwizzleAccess(VarRef(f"sbavg{sfx}"), "y"))))))


# --- Multi-light unrolling helper ---

def _emit_multi_light_loop(
    body: list,
    max_lights: int,
    pos_var: str,      # "world_pos" or "hit_pos"
    normal_var: str,   # "n"
    view_var: str,     # "v"
    albedo_var: str,   # "layer_albedo"
    roughness_var: str, # "layer_roughness"
    metallic_var: str,  # "layer_metallic"
    light_count_var: str = "_ml_light_count",
    has_shadows: bool = False,
    shadow_filter: str = "pcf",
) -> str:
    """Emit unrolled multi-light evaluation.

    Generates max_lights iterations, each guarded by if (i < light_count).
    Each iteration:
    1. Loads light data from SSBO: lights[i].field
    2. Calls evaluate_light_direction() -> l_i
    3. Calls gltf_pbr(n, v, l_i, albedo, roughness, metallic) -> brdf_i
    4. Calls evaluate_light() -> radiance_i
    5. Accumulates: total_direct += brdf_i * radiance_i

    Returns the result variable name ("total_direct").
    """
    # Initialize accumulator
    body.append(LetStmt("total_direct", "vec3",
        ConstructorExpr("vec3", [NumberLit("0.0")])))

    for i in range(max_lights):
        idx = str(i)
        sfx = f"_{i}"
        guard_body = []

        # Load light fields from SSBO: lights[i].field_name
        # Uses FieldAccess(IndexAccess(VarRef("lights"), NumberLit(i)), field)
        def _light_ref(field, _idx=idx):
            return FieldAccess(
                IndexAccess(VarRef("lights"), NumberLit(_idx)),
                field
            )

        guard_body.append(LetStmt(f"lt{sfx}", "scalar", _light_ref("light_type")))
        guard_body.append(LetStmt(f"li{sfx}", "scalar", _light_ref("intensity")))
        guard_body.append(LetStmt(f"lr{sfx}", "scalar", _light_ref("range")))
        guard_body.append(LetStmt(f"lic{sfx}", "scalar", _light_ref("inner_cone")))
        guard_body.append(LetStmt(f"lp{sfx}", "vec3", _light_ref("position")))
        guard_body.append(LetStmt(f"loc{sfx}", "scalar", _light_ref("outer_cone")))
        guard_body.append(LetStmt(f"ld{sfx}", "vec3", _light_ref("direction")))
        guard_body.append(LetStmt(f"lsi{sfx}", "scalar", _light_ref("shadow_index")))
        guard_body.append(LetStmt(f"lc{sfx}", "vec3", _light_ref("color")))

        # Compute light direction using stdlib helper
        guard_body.append(LetStmt(f"l{sfx}", "vec3",
            CallExpr(VarRef("evaluate_light_direction"), [
                VarRef(f"lt{sfx}"),
                VarRef(f"lp{sfx}"),
                VarRef(f"ld{sfx}"),
                VarRef(pos_var),
            ])))

        # Evaluate BRDF
        guard_body.append(LetStmt(f"brdf{sfx}", "vec3",
            CallExpr(VarRef("gltf_pbr"), [
                VarRef(normal_var), VarRef(view_var), VarRef(f"l{sfx}"),
                VarRef(albedo_var), VarRef(roughness_var),
                VarRef(metallic_var),
            ])))

        # Evaluate light radiance (attenuation etc.)
        guard_body.append(LetStmt(f"radiance{sfx}", "vec3",
            CallExpr(VarRef("evaluate_light"), [
                VarRef(f"lt{sfx}"),
                VarRef(f"lp{sfx}"),
                VarRef(f"ld{sfx}"),
                VarRef(f"lc{sfx}"),
                VarRef(f"li{sfx}"),
                VarRef(f"lr{sfx}"),
                VarRef(f"lic{sfx}"),
                VarRef(f"loc{sfx}"),
                VarRef(pos_var),
                VarRef(normal_var),
            ])))

        if has_shadows:
            # 1. Init shadow factor to 1.0 (fully lit)
            guard_body.append(LetStmt(f"shadow{sfx}", "scalar", NumberLit("1.0")))

            # 2. Build shadow evaluation body (guarded by shadow_index >= 0)
            shadow_body = []
            # Load ShadowEntry fields from SSBO
            def _shadow_field(field, _idx=idx):
                return FieldAccess(
                    IndexAccess(VarRef("shadow_matrices"),
                                VarRef(f"lsi{sfx}")),
                    field)
            shadow_body.append(LetStmt(f"svp{sfx}", "mat4", _shadow_field("view_projection")))
            shadow_body.append(LetStmt(f"sbias{sfx}", "scalar", _shadow_field("bias")))
            shadow_body.append(LetStmt(f"snbias{sfx}", "scalar", _shadow_field("normal_bias")))
            shadow_body.append(LetStmt(f"sres{sfx}", "scalar", _shadow_field("resolution")))
            shadow_body.append(LetStmt(f"slsz{sfx}", "scalar", _shadow_field("light_size")))

            # Normal-offset bias
            shadow_body.append(LetStmt(f"soffpos{sfx}", "vec3",
                CallExpr(VarRef("normal_offset_world"), [
                    VarRef(pos_var), VarRef(normal_var),
                    VarRef(f"snbias{sfx}"),
                    CallExpr(VarRef("max"), [
                        CallExpr(VarRef("dot"), [VarRef(normal_var), VarRef(f"l{sfx}")]),
                        NumberLit("0.0")])])))

            # Project to shadow UV space
            shadow_body.append(LetStmt(f"suv{sfx}", "vec3",
                CallExpr(VarRef("compute_shadow_uv"), [
                    VarRef(f"soffpos{sfx}"), VarRef(f"svp{sfx}"),
                    VarRef(f"sbias{sfx}")])))

            # Emit filter-specific sampling code
            if shadow_filter == "hard":
                _emit_hard_shadow(shadow_body, sfx)
            elif shadow_filter == "pcf":
                _emit_pcf_shadow(shadow_body, sfx)
            elif shadow_filter == "pcss":
                _emit_pcss_shadow(shadow_body, sfx)
            else:
                _emit_pcf_shadow(shadow_body, sfx)  # default to PCF

            shadow_body.append(AssignStmt(
                AssignTarget(VarRef(f"shadow{sfx}")), VarRef(f"sf{sfx}")))

            # 3. Guard: if (shadow_index >= 0.0) { evaluate shadow }
            guard_body.append(IfStmt(
                BinaryOp(">=", VarRef(f"lsi{sfx}"), NumberLit("0.0")),
                shadow_body, []))

            # 4. Accumulate with shadow factor
            guard_body.append(AssignStmt(
                AssignTarget(VarRef("total_direct")),
                BinaryOp("+", VarRef("total_direct"),
                    BinaryOp("*",
                        BinaryOp("*", VarRef(f"brdf{sfx}"), VarRef(f"radiance{sfx}")),
                        VarRef(f"shadow{sfx}")))))
        else:
            # Original: no shadow attenuation
            guard_body.append(AssignStmt(
                AssignTarget(VarRef("total_direct")),
                BinaryOp("+", VarRef("total_direct"),
                    BinaryOp("*", VarRef(f"brdf{sfx}"), VarRef(f"radiance{sfx}"))),
            ))

        # Guard: if (i < light_count)
        body.append(IfStmt(
            BinaryOp("<", NumberLit(f"{i}.0"),
                      BinaryOp("+", VarRef(light_count_var), NumberLit("0.0"))),
            guard_body, []))

    return "total_direct"


# --- Shared helper functions for code deduplication ---


def _emit_multi_light_from_lighting(body, lighting, pos_var):
    """Extract multi-light args from lighting block and emit unrolled loop.

    Assumes layer_albedo, layer_roughness, layer_metallic, n, v are in scope.
    Returns result_var name ("total_direct"), or None if no multi_light layer.
    """
    if not (lighting and lighting.layers):
        return None
    ml_layer = None
    for layer in lighting.layers:
        if layer.name == "multi_light":
            ml_layer = layer
            break
    if ml_layer is None:
        return None

    ml_args = _get_layer_args(ml_layer)
    max_lights = 16
    if "max_lights" in ml_args and isinstance(ml_args["max_lights"], NumberLit):
        max_lights = int(float(ml_args["max_lights"].value))
    _has_shadow_map = "shadow_map" in ml_args
    _shadow_filter = "pcf"
    if "shadow_filter" in ml_args:
        sf_val = ml_args["shadow_filter"]
        if isinstance(sf_val, VarRef):
            _shadow_filter = sf_val.name
    count_expr = ml_args.get("count", NumberLit(str(max_lights)))
    body.append(LetStmt("_ml_light_count", "scalar",
        BinaryOp("*", count_expr, NumberLit("1.0"))))
    return _emit_multi_light_loop(
        body, max_lights,
        pos_var=pos_var, normal_var="n", view_var="v",
        albedo_var="layer_albedo", roughness_var="layer_roughness",
        metallic_var="layer_metallic",
        has_shadows=_has_shadow_map,
        shadow_filter=_shadow_filter,
    )


def _detect_multi_light(lighting):
    """Return True if lighting block has a multi_light layer."""
    if not (lighting and lighting.layers):
        return False
    return any(layer.name == "multi_light" for layer in lighting.layers)


def _emit_barycentric_interpolation(body, index_offset_expr=None):
    """Emit barycentric interpolation for RT closest-hit shaders.

    Generates index lookups, barycentric weights, and interpolated
    hit_pos, n (normal), and uv from storage buffers.

    If index_offset_expr is given (bindless RT), base = offset + primitive_id * 3.
    Otherwise (non-bindless RT), base = primitive_id * 3.
    """
    if index_offset_expr is not None:
        body.append(LetStmt("base", "scalar",
            BinaryOp("+",
                BinaryOp("*", index_offset_expr, NumberLit("1.0")),
                BinaryOp("*", VarRef("primitive_id"), NumberLit("3.0")))))
    else:
        body.append(LetStmt("base", "scalar",
            BinaryOp("*", VarRef("primitive_id"), NumberLit("3.0"))))

    body.append(LetStmt("i0", "uint", IndexAccess(VarRef("indices"), VarRef("base"))))
    body.append(LetStmt("i1", "uint", IndexAccess(VarRef("indices"),
        BinaryOp("+", VarRef("base"), NumberLit("1.0")))))
    body.append(LetStmt("i2", "uint", IndexAccess(VarRef("indices"),
        BinaryOp("+", VarRef("base"), NumberLit("2.0")))))

    body.append(LetStmt("b", "vec2", VarRef("bary")))
    body.append(LetStmt("bw", "scalar", BinaryOp("-",
        BinaryOp("-", NumberLit("1.0"), SwizzleAccess(VarRef("b"), "x")),
        SwizzleAccess(VarRef("b"), "y"))))

    # Interpolate position
    body.append(LetStmt("p0", "vec4", IndexAccess(VarRef("positions"), VarRef("i0"))))
    body.append(LetStmt("p1", "vec4", IndexAccess(VarRef("positions"), VarRef("i1"))))
    body.append(LetStmt("p2", "vec4", IndexAccess(VarRef("positions"), VarRef("i2"))))
    body.append(LetStmt("hit_pos", "vec3", BinaryOp("+",
        BinaryOp("+",
            BinaryOp("*", SwizzleAccess(VarRef("p0"), "xyz"), VarRef("bw")),
            BinaryOp("*", SwizzleAccess(VarRef("p1"), "xyz"),
                SwizzleAccess(VarRef("b"), "x"))),
        BinaryOp("*", SwizzleAccess(VarRef("p2"), "xyz"),
            SwizzleAccess(VarRef("b"), "y")))))

    # Interpolate normal
    body.append(LetStmt("n0", "vec4", IndexAccess(VarRef("normals"), VarRef("i0"))))
    body.append(LetStmt("n1", "vec4", IndexAccess(VarRef("normals"), VarRef("i1"))))
    body.append(LetStmt("n2", "vec4", IndexAccess(VarRef("normals"), VarRef("i2"))))
    body.append(LetStmt("n", "vec3", CallExpr(VarRef("normalize"), [
        BinaryOp("+",
            BinaryOp("+",
                BinaryOp("*", SwizzleAccess(VarRef("n0"), "xyz"), VarRef("bw")),
                BinaryOp("*", SwizzleAccess(VarRef("n1"), "xyz"),
                    SwizzleAccess(VarRef("b"), "x"))),
            BinaryOp("*", SwizzleAccess(VarRef("n2"), "xyz"),
                SwizzleAccess(VarRef("b"), "y")))])))

    # Interpolate UV
    body.append(LetStmt("uv0", "vec2", IndexAccess(VarRef("tex_coords"), VarRef("i0"))))
    body.append(LetStmt("uv1", "vec2", IndexAccess(VarRef("tex_coords"), VarRef("i1"))))
    body.append(LetStmt("uv2", "vec2", IndexAccess(VarRef("tex_coords"), VarRef("i2"))))
    body.append(LetStmt("uv", "vec2", BinaryOp("+",
        BinaryOp("+",
            BinaryOp("*", VarRef("uv0"), VarRef("bw")),
            BinaryOp("*", VarRef("uv1"), SwizzleAccess(VarRef("b"), "x"))),
        BinaryOp("*", VarRef("uv2"), SwizzleAccess(VarRef("b"), "y")))))


def _emit_tonemap_output(body, result_var, schedule):
    """Emit tonemap selection + sRGB conversion. Returns output variable name."""
    tonemap_strategy = schedule.get("tonemap", "none") if schedule else "none"
    if tonemap_strategy == "aces":
        body.append(LetStmt("tonemapped", "vec3",
            CallExpr(VarRef("tonemap_aces"), [VarRef(result_var)])))
        body.append(LetStmt("final_color", "vec3",
            CallExpr(VarRef("linear_to_srgb"), [VarRef("tonemapped")])))
        return "final_color"
    elif tonemap_strategy == "reinhard":
        body.append(LetStmt("tonemapped", "vec3",
            CallExpr(VarRef("tonemap_reinhard"), [VarRef(result_var)])))
        body.append(LetStmt("final_color", "vec3",
            CallExpr(VarRef("linear_to_srgb"), [VarRef("tonemapped")])))
        return "final_color"
    else:
        return result_var


def _generate_openpbr_main(
    surface: SurfaceDecl,
    frag_inputs: list[tuple[str, str]],
    schedule: dict[str, str] | None = None,
    module: Module | None = None,
    lighting: LightingDecl | None = None,
) -> list:
    """Generate main() body for an OpenPBR layered fragment shader.

    Mirrors _generate_layered_main() but emits calls to openpbr_direct() and
    openpbr_compose() instead of gltf_pbr() and compose_pbr_layers().  Activated
    when the module contains ``import openpbr;``.
    """
    body = []

    # Index layers by name
    layers_by_name = {}
    for layer in surface.layers:
        layers_by_name[layer.name] = layer

    # Determine which inputs we have
    has_pos = any(n in ("world_pos", "frag_pos") for n, _ in frag_inputs)
    pos_var = next(
        (n for n, _ in frag_inputs if n in ("world_pos", "frag_pos")),
        "world_pos",
    )
    has_tangent = any(n == "world_tangent" for n, _ in frag_inputs)

    # Create UV alias so layer expressions using `uv` resolve correctly
    _has_frag_uv = any(n == "frag_uv" for n, _ in frag_inputs)
    if _has_frag_uv:
        body.append(LetStmt("uv", "vec2", VarRef("frag_uv")))

    # --- Normal setup ---
    if "normal_map" in layers_by_name and has_tangent:
        nmap_args = _get_layer_args(layers_by_name["normal_map"])
        map_expr = nmap_args.get("map")
        body.append(LetStmt("normal_map_raw", "vec3", map_expr))
        body.append(LetStmt("n", "vec3", CallExpr(VarRef("tbn_perturb_normal"), [
            VarRef("normal_map_raw"),
            CallExpr(VarRef("normalize"), [VarRef("world_normal")]),
            CallExpr(VarRef("normalize"), [VarRef("world_tangent")]),
            CallExpr(VarRef("normalize"), [VarRef("world_bitangent")]),
        ])))
    else:
        normal_var = next(
            (n for n, _ in frag_inputs if n in ("world_normal", "frag_normal")),
            "world_normal",
        )
        body.append(LetStmt("n", "vec3",
            CallExpr(VarRef("normalize"), [VarRef(normal_var)])))

    # --- View and light directions ---
    if has_pos:
        body.append(LetStmt("v", "vec3", CallExpr(VarRef("normalize"), [
            BinaryOp("-", VarRef("view_pos"), VarRef(pos_var)),
        ])))
    else:
        body.append(LetStmt("v", "vec3", CallExpr(VarRef("normalize"), [
            ConstructorExpr("vec3", [NumberLit("0.0"), NumberLit("0.0"), NumberLit("1.0")]),
        ])))

    # Detect multi_light vs directional lighting mode
    _has_multi_light = _detect_multi_light(lighting)
    lighting_by_name = {}
    if lighting and lighting.layers:
        lighting_by_name = {layer.name: layer for layer in lighting.layers}

    # Light direction: from lighting block (directional/multi_light) or legacy uniform
    if _has_multi_light:
        body.append(LetStmt("l", "vec3",
            ConstructorExpr("vec3", [NumberLit("0.0"), NumberLit("1.0"), NumberLit("0.0")])))
    elif lighting and lighting.layers:
        if "directional" in lighting_by_name:
            dir_args = _get_layer_args(lighting_by_name["directional"])
            body.append(LetStmt("l", "vec3",
                CallExpr(VarRef("normalize"), [
                    dir_args.get("direction", VarRef("light_dir"))])))
            body.append(LetStmt("light_color", "vec3",
                dir_args.get("color",
                    ConstructorExpr("vec3", [NumberLit("1.0")]))))
        else:
            body.append(LetStmt("l", "vec3",
                CallExpr(VarRef("normalize"), [VarRef("light_dir")])))
            body.append(LetStmt("light_color", "vec3",
                ConstructorExpr("vec3", [
                    NumberLit("1.0"), NumberLit("0.98"), NumberLit("0.95")])))
    else:
        body.append(LetStmt("l", "vec3",
            CallExpr(VarRef("normalize"), [VarRef("light_dir")])))
        body.append(LetStmt("light_color", "vec3",
            ConstructorExpr("vec3", [
                NumberLit("1.0"), NumberLit("0.98"), NumberLit("0.95")])))

    body.append(LetStmt("n_dot_v", "scalar", CallExpr(VarRef("max"), [
        CallExpr(VarRef("dot"), [VarRef("n"), VarRef("v")]),
        NumberLit("0.001"),
    ])))

    # ----------------------------------------------------------------
    # Extract OpenPBR layer parameters
    # ----------------------------------------------------------------

    # --- base layer ---
    base_args = _get_layer_args(layers_by_name["base"]) if "base" in layers_by_name else {}
    base_color_expr = base_args.get("color", ConstructorExpr("vec3", [NumberLit("0.8")]))
    base_metalness_expr = base_args.get("metalness", NumberLit("0.0"))
    base_diffuse_roughness_expr = base_args.get("diffuse_roughness", NumberLit("0.0"))
    base_weight_expr = base_args.get("weight", NumberLit("1.0"))

    body.append(LetStmt("base_color", "vec3", base_color_expr))
    body.append(LetStmt("base_metalness", "scalar", base_metalness_expr))
    body.append(LetStmt("base_diffuse_roughness", "scalar", base_diffuse_roughness_expr))
    body.append(LetStmt("base_weight", "scalar", base_weight_expr))

    # --- specular layer ---
    spec_args = _get_layer_args(layers_by_name["specular"]) if "specular" in layers_by_name else {}
    body.append(LetStmt("specular_weight", "scalar",
        spec_args.get("weight", NumberLit("1.0"))))
    body.append(LetStmt("specular_color", "vec3",
        spec_args.get("color", ConstructorExpr("vec3", [NumberLit("1.0")]))))
    body.append(LetStmt("specular_roughness", "scalar",
        spec_args.get("roughness", NumberLit("0.3"))))
    body.append(LetStmt("specular_ior", "scalar",
        spec_args.get("ior", NumberLit("1.5"))))

    # --- coat layer ---
    coat_args = _get_layer_args(layers_by_name["coat"]) if "coat" in layers_by_name else {}
    body.append(LetStmt("coat_weight", "scalar",
        coat_args.get("weight", NumberLit("0.0"))))
    body.append(LetStmt("coat_color", "vec3",
        coat_args.get("color", ConstructorExpr("vec3", [NumberLit("1.0")]))))
    body.append(LetStmt("coat_roughness", "scalar",
        coat_args.get("roughness", NumberLit("0.0"))))
    body.append(LetStmt("coat_ior", "scalar",
        coat_args.get("ior", NumberLit("1.6"))))
    body.append(LetStmt("coat_darkening", "scalar",
        coat_args.get("darkening", NumberLit("1.0"))))

    # --- fuzz layer ---
    fuzz_args = _get_layer_args(layers_by_name["fuzz"]) if "fuzz" in layers_by_name else {}
    body.append(LetStmt("fuzz_weight", "scalar",
        fuzz_args.get("weight", NumberLit("0.0"))))
    body.append(LetStmt("fuzz_color", "vec3",
        fuzz_args.get("color", ConstructorExpr("vec3", [NumberLit("1.0")]))))
    body.append(LetStmt("fuzz_roughness", "scalar",
        fuzz_args.get("roughness", NumberLit("0.5"))))

    # --- thin_film layer ---
    tf_args = _get_layer_args(layers_by_name["thin_film"]) if "thin_film" in layers_by_name else {}
    body.append(LetStmt("thin_film_weight", "scalar",
        tf_args.get("weight", NumberLit("0.0"))))
    body.append(LetStmt("thin_film_thickness", "scalar",
        tf_args.get("thickness", NumberLit("0.5"))))
    body.append(LetStmt("thin_film_ior", "scalar",
        tf_args.get("ior", NumberLit("1.4"))))

    # --- transmission layer ---
    trans_args = _get_layer_args(layers_by_name["transmission"]) if "transmission" in layers_by_name else {}
    body.append(LetStmt("trans_weight", "scalar",
        trans_args.get("weight", NumberLit("0.0"))))
    body.append(LetStmt("trans_color", "vec3",
        trans_args.get("color", ConstructorExpr("vec3", [NumberLit("1.0")]))))
    body.append(LetStmt("trans_depth", "scalar",
        trans_args.get("depth", NumberLit("0.0"))))

    # --- emission layer ---
    em_args = _get_layer_args(layers_by_name["emission"]) if "emission" in layers_by_name else {}
    body.append(LetStmt("emission_luminance", "scalar",
        em_args.get("luminance", NumberLit("0.0"))))
    body.append(LetStmt("emission_color", "vec3",
        em_args.get("color", ConstructorExpr("vec3", [NumberLit("1.0")]))))

    # ----------------------------------------------------------------
    # Also expose base params as layer_albedo/layer_roughness/layer_metallic
    # for multi-light loop compatibility (gltf_pbr call inside the loop)
    # ----------------------------------------------------------------
    body.append(LetStmt("layer_albedo", "vec3", VarRef("base_color")))
    body.append(LetStmt("layer_roughness", "scalar", VarRef("specular_roughness")))
    body.append(LetStmt("layer_metallic", "scalar", VarRef("base_metalness")))

    # ----------------------------------------------------------------
    # Direct lighting
    # ----------------------------------------------------------------
    # Determine which OpenPBR direct function to use based on schedule
    _use_fast_direct = False
    if schedule:
        _use_fast_direct = (
            schedule.get("diffuse_model") == "lambert"
            or schedule.get("specular_fresnel") == "schlick"
        )
    _direct_fn = "openpbr_direct_fast" if _use_fast_direct else "openpbr_direct"

    result_var = "result_zero"
    body.append(LetStmt(result_var, "vec3",
        ConstructorExpr("vec3", [NumberLit("0.0")])))

    if _has_multi_light:
        result_var = _emit_multi_light_from_lighting(
            body, lighting, pos_var)
    else:
        # Single directional light: call openpbr_direct or openpbr_direct_fast
        body.append(LetStmt("direct_raw", "vec3", CallExpr(VarRef(_direct_fn), [
            VarRef("n"), VarRef("v"), VarRef("l"),
            VarRef("base_color"), VarRef("base_metalness"),
            VarRef("base_diffuse_roughness"), VarRef("base_weight"),
            VarRef("specular_weight"), VarRef("specular_color"),
            VarRef("specular_roughness"), VarRef("specular_ior"),
            VarRef("coat_weight"), VarRef("coat_roughness"), VarRef("coat_ior"),
            VarRef("thin_film_weight"), VarRef("thin_film_thickness"), VarRef("thin_film_ior"),
        ])))

        # Light color tint
        body.append(LetStmt("direct_lit", "vec3", BinaryOp("*",
            VarRef("direct_raw"), VarRef("light_color"),
        )))
        result_var = "direct_lit"

    # ----------------------------------------------------------------
    # IBL sampling
    # ----------------------------------------------------------------
    ibl_args = None
    if lighting and lighting.layers:
        _lighting_by_name = {layer.name: layer for layer in lighting.layers}
        if "ibl" in _lighting_by_name:
            ibl_args = _get_layer_args(_lighting_by_name["ibl"])
    if ibl_args is None and "ibl" in layers_by_name:
        ibl_args = _get_layer_args(layers_by_name["ibl"])
    _has_ibl = ibl_args is not None

    body.append(LetStmt("bl_ambient", "vec3",
        ConstructorExpr("vec3", [NumberLit("0.0")])))
    body.append(LetStmt("bl_coat_ibl", "vec3",
        ConstructorExpr("vec3", [NumberLit("0.0")])))

    if _has_ibl:
        specular_map = ibl_args.get("specular_map")
        irradiance_map = ibl_args.get("irradiance_map")
        brdf_lut_ref = ibl_args.get("brdf_lut")

        body.append(LetStmt("r", "vec3", CallExpr(VarRef("reflect"), [
            BinaryOp("*", VarRef("v"), UnaryOp("-", NumberLit("1.0"))),
            VarRef("n"),
        ])))
        body.append(LetStmt("prefiltered", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                specular_map, VarRef("r"),
                BinaryOp("*", VarRef("specular_roughness"),
                         NumberLit("8.0")),
            ]), "xyz")))
        body.append(LetStmt("irradiance", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample"), [irradiance_map, VarRef("n")]),
            "xyz")))
        body.append(LetStmt("brdf_sample", "vec2", SwizzleAccess(
            CallExpr(VarRef("sample"), [
                brdf_lut_ref,
                ConstructorExpr("vec2", [
                    VarRef("n_dot_v"), VarRef("specular_roughness"),
                ]),
            ]), "xy")))

        body.append(AssignStmt(AssignTarget(VarRef("bl_ambient")),
            CallExpr(VarRef("ibl_contribution"), [
                VarRef("base_color"), VarRef("specular_roughness"),
                VarRef("base_metalness"),
                VarRef("n_dot_v"), VarRef("prefiltered"),
                VarRef("irradiance"), VarRef("brdf_sample"),
            ])))

        # Coat IBL contribution (if coat layer is active)
        if "coat" in layers_by_name:
            body.append(LetStmt("prefiltered_coat", "vec3", SwizzleAccess(
                CallExpr(VarRef("sample_lod"), [
                    specular_map, VarRef("r"),
                    BinaryOp("*", VarRef("coat_roughness"),
                             NumberLit("8.0")),
                ]), "xyz")))
            body.append(AssignStmt(
                AssignTarget(VarRef("bl_coat_ibl")),
                CallExpr(VarRef("coat_ibl"), [
                    VarRef("coat_weight"),
                    VarRef("coat_roughness"),
                    VarRef("n"), VarRef("v"),
                    VarRef("prefiltered_coat"),
                ])))

    # ----------------------------------------------------------------
    # Compose via openpbr_compose or openpbr_compose_fast
    # ----------------------------------------------------------------
    # Determine which compose function to use based on schedule
    _use_fast_compose = False
    if schedule:
        _use_fast_compose = (
            schedule.get("diffuse_model") == "lambert"
            or schedule.get("specular_fresnel") == "schlick"
        )
    _compose_fn = "openpbr_compose_fast" if _use_fast_compose else "openpbr_compose"

    # Custom @layer functions
    _custom_layer_list = []
    if module is not None:
        layer_fns = _collect_layer_functions(module)
        _custom_layer_list = [
            (layer, layer_fns[layer.name])
            for layer in surface.layers if layer.name in layer_fns
        ]
    _has_custom_layers = len(_custom_layer_list) > 0

    def _emit_openpbr_compose(emission_lum_expr, emission_col_expr):
        return LetStmt("composed", "vec3", CallExpr(
            VarRef(_compose_fn), [
                VarRef(result_var),
                VarRef("n"), VarRef("v"), VarRef("l"),
                VarRef("base_color"), VarRef("base_metalness"), VarRef("base_weight"),
                VarRef("specular_weight"), VarRef("specular_color"),
                VarRef("specular_roughness"), VarRef("specular_ior"),
                VarRef("trans_weight"), VarRef("trans_color"), VarRef("trans_depth"),
                VarRef("coat_weight"), VarRef("coat_color"),
                VarRef("coat_roughness"), VarRef("coat_ior"), VarRef("coat_darkening"),
                VarRef("fuzz_weight"), VarRef("fuzz_color"), VarRef("fuzz_roughness"),
                VarRef("thin_film_weight"), VarRef("thin_film_thickness"), VarRef("thin_film_ior"),
                emission_lum_expr, emission_col_expr,
                VarRef("bl_ambient"), VarRef("bl_coat_ibl"),
            ]))

    if _has_custom_layers:
        # Compose WITHOUT emission (custom layers run before emission)
        body.append(_emit_openpbr_compose(
            NumberLit("0.0"),
            ConstructorExpr("vec3", [NumberLit("1.0")])))
        result_var = "composed"
        for layer, fn in _custom_layer_list:
            result_var = _emit_custom_layer(
                layer, fn, result_var, body,
                prefix=f"custom_{layer.name}", rewrite_for_rt=False)
        body.append(LetStmt("hdr", "vec3",
            BinaryOp("+", VarRef(result_var),
                BinaryOp("*", VarRef("emission_color"), VarRef("emission_luminance")))))
        result_var = "hdr"
    else:
        body.append(_emit_openpbr_compose(
            VarRef("emission_luminance"), VarRef("emission_color")))
        result_var = "composed"

    # --- Tonemap + gamma ---
    output_var = _emit_tonemap_output(body, result_var, schedule)

    # Output final color
    body.append(AssignStmt(
        AssignTarget(VarRef("color")),
        ConstructorExpr("vec4", [VarRef(output_var), NumberLit("1.0")]),
    ))

    return body


def _generate_layered_main(
    surface: SurfaceDecl,
    frag_inputs: list[tuple[str, str]],
    schedule: dict[str, str] | None = None,
    module: Module | None = None,
    lighting: LightingDecl | None = None,
) -> list:
    """Generate main() body for a layered surface-expanded fragment shader.

    Processes layer declarations to generate PBR lighting code equivalent to
    hand-written gltf_pbr.lux. Supports base, normal_map, ibl, and emission layers.
    """
    # OpenPBR dispatch
    if module and _is_openpbr(module):
        return _generate_openpbr_main(surface, frag_inputs, schedule, module, lighting)

    body = []

    # Index layers by name
    layers_by_name = {}
    for layer in surface.layers:
        layers_by_name[layer.name] = layer

    # Determine which inputs we have
    has_pos = any(n in ("world_pos", "frag_pos") for n, _ in frag_inputs)
    pos_var = next(
        (n for n, _ in frag_inputs if n in ("world_pos", "frag_pos")),
        "world_pos",
    )
    has_tangent = any(n == "world_tangent" for n, _ in frag_inputs)

    # Create UV alias so layer expressions using `uv` resolve correctly
    _has_frag_uv_layered = any(n == "frag_uv" for n, _ in frag_inputs)
    if _has_frag_uv_layered:
        body.append(LetStmt("uv", "vec2", VarRef("frag_uv")))

    # --- Normal setup ---
    if "normal_map" in layers_by_name and has_tangent:
        nmap_args = _get_layer_args(layers_by_name["normal_map"])
        map_expr = nmap_args.get("map")
        body.append(LetStmt("normal_map_raw", "vec3", map_expr))
        body.append(LetStmt("n", "vec3", CallExpr(VarRef("tbn_perturb_normal"), [
            VarRef("normal_map_raw"),
            CallExpr(VarRef("normalize"), [VarRef("world_normal")]),
            CallExpr(VarRef("normalize"), [VarRef("world_tangent")]),
            CallExpr(VarRef("normalize"), [VarRef("world_bitangent")]),
        ])))
    else:
        normal_var = next(
            (n for n, _ in frag_inputs if n in ("world_normal", "frag_normal")),
            "world_normal",
        )
        body.append(LetStmt("n", "vec3",
            CallExpr(VarRef("normalize"), [VarRef(normal_var)])))

    # --- View and light directions ---
    if has_pos:
        body.append(LetStmt("v", "vec3", CallExpr(VarRef("normalize"), [
            BinaryOp("-", VarRef("view_pos"), VarRef(pos_var)),
        ])))
    else:
        body.append(LetStmt("v", "vec3", CallExpr(VarRef("normalize"), [
            ConstructorExpr("vec3", [NumberLit("0.0"), NumberLit("0.0"), NumberLit("1.0")]),
        ])))
    # Detect multi_light vs directional lighting mode
    _has_multi_light = _detect_multi_light(lighting)
    lighting_by_name = {}
    if lighting and lighting.layers:
        lighting_by_name = {layer.name: layer for layer in lighting.layers}

    # Light direction: from lighting block (directional/multi_light) or legacy uniform
    if _has_multi_light:
        # Multi-light path: l will be computed per-light inside unrolled loop.
        # We still need a dummy `l` for layers that reference it (coat, sheen, etc.)
        # It will be overwritten per-iteration, but downstream layers use the last value.
        body.append(LetStmt("l", "vec3",
            ConstructorExpr("vec3", [NumberLit("0.0"), NumberLit("1.0"), NumberLit("0.0")])))
    elif lighting and lighting.layers:
        if "directional" in lighting_by_name:
            dir_args = _get_layer_args(lighting_by_name["directional"])
            body.append(LetStmt("l", "vec3",
                CallExpr(VarRef("normalize"), [
                    dir_args.get("direction", VarRef("light_dir"))])))
            body.append(LetStmt("light_color", "vec3",
                dir_args.get("color",
                    ConstructorExpr("vec3", [NumberLit("1.0")]))))
        else:
            body.append(LetStmt("l", "vec3",
                CallExpr(VarRef("normalize"), [VarRef("light_dir")])))
            body.append(LetStmt("light_color", "vec3",
                ConstructorExpr("vec3", [
                    NumberLit("1.0"), NumberLit("0.98"), NumberLit("0.95")])))
    else:
        body.append(LetStmt("l", "vec3",
            CallExpr(VarRef("normalize"), [VarRef("light_dir")])))
        body.append(LetStmt("light_color", "vec3",
            ConstructorExpr("vec3", [
                NumberLit("1.0"), NumberLit("0.98"), NumberLit("0.95")])))
    body.append(LetStmt("n_dot_v", "scalar", CallExpr(VarRef("max"), [
        CallExpr(VarRef("dot"), [VarRef("n"), VarRef("v")]),
        NumberLit("0.001"),
    ])))

    # --- Base layer: direct lighting ---
    result_var = "result_zero"
    body.append(LetStmt(result_var, "vec3",
        ConstructorExpr("vec3", [NumberLit("0.0")])))

    if "base" in layers_by_name:
        base_args = _get_layer_args(layers_by_name["base"])
        albedo_expr = base_args.get("albedo",
            ConstructorExpr("vec3", [NumberLit("0.5")]))
        roughness_expr = base_args.get("roughness", NumberLit("0.5"))
        metallic_expr = base_args.get("metallic", NumberLit("0.0"))

        body.append(LetStmt("layer_albedo", "vec3", albedo_expr))
        body.append(LetStmt("layer_roughness", "scalar", roughness_expr))
        body.append(LetStmt("layer_metallic", "scalar", metallic_expr))

        if _has_multi_light:
            result_var = _emit_multi_light_from_lighting(
                body, lighting, pos_var)
        else:
            # Single directional light path
            # Direct lighting via gltf_pbr (includes N-dot-L)
            body.append(LetStmt("direct", "vec3", CallExpr(VarRef("gltf_pbr"), [
                VarRef("n"), VarRef("v"), VarRef("l"),
                VarRef("layer_albedo"), VarRef("layer_roughness"),
                VarRef("layer_metallic"),
            ])))
            # Light color tint
            body.append(LetStmt("direct_lit", "vec3", BinaryOp("*",
                VarRef("direct"), VarRef("light_color"),
            )))
            result_var = "direct_lit"

    # --- Detect which optional layers are present ---
    _has_transmission = "transmission" in layers_by_name
    _has_sheen = "sheen" in layers_by_name
    _has_coat = "coat" in layers_by_name
    _has_emission = "emission" in layers_by_name

    # IBL detection (check lighting block first, then surface)
    ibl_args = None
    if lighting and lighting.layers:
        _lighting_by_name = {layer.name: layer for layer in lighting.layers}
        if "ibl" in _lighting_by_name:
            ibl_args = _get_layer_args(_lighting_by_name["ibl"])
    if ibl_args is None and "ibl" in layers_by_name:
        ibl_args = _get_layer_args(layers_by_name["ibl"])
    _has_ibl = ibl_args is not None

    # Custom @layer functions
    _custom_layer_list = []
    if module is not None:
        layer_fns = _collect_layer_functions(module)
        _custom_layer_list = [
            (layer, layer_fns[layer.name])
            for layer in surface.layers if layer.name in layer_fns
        ]
    _has_custom_layers = len(_custom_layer_list) > 0

    # Use compose_pbr_layers when any compositing-specific layer is active
    # (transmission, sheen, coat, IBL — these require stdlib/compositing.lux).
    # Emission is simple additive and handled inline without compose.
    _use_compose = (_has_transmission or _has_sheen or _has_coat or _has_ibl)

    if _use_compose:
        # --- Initialize optional layer params (defaults = layer disabled) ---
        body.append(LetStmt("bl_trans_factor", "scalar", NumberLit("0.0")))
        body.append(LetStmt("bl_trans_ior", "scalar", NumberLit("1.5")))
        body.append(LetStmt("bl_trans_thickness", "scalar", NumberLit("0.0")))
        body.append(LetStmt("bl_trans_atten_color", "vec3",
            ConstructorExpr("vec3", [NumberLit("1.0")])))
        body.append(LetStmt("bl_trans_atten_dist", "scalar",
            NumberLit("1000000.0")))
        body.append(LetStmt("bl_sheen_color", "vec3",
            ConstructorExpr("vec3", [NumberLit("0.0")])))
        body.append(LetStmt("bl_sheen_roughness", "scalar", NumberLit("0.0")))
        body.append(LetStmt("bl_coat_factor", "scalar", NumberLit("0.0")))
        body.append(LetStmt("bl_coat_roughness", "scalar", NumberLit("0.0")))
        body.append(LetStmt("bl_emission", "vec3",
            ConstructorExpr("vec3", [NumberLit("0.0")])))

        # --- Load transmission params from layer args ---
        if _has_transmission:
            trans_args = _get_layer_args(layers_by_name["transmission"])
            body.append(AssignStmt(AssignTarget(VarRef("bl_trans_factor")),
                trans_args.get("factor", NumberLit("0.0"))))
            body.append(AssignStmt(AssignTarget(VarRef("bl_trans_ior")),
                trans_args.get("ior", NumberLit("1.5"))))
            body.append(AssignStmt(AssignTarget(VarRef("bl_trans_thickness")),
                trans_args.get("thickness", NumberLit("0.0"))))
            body.append(AssignStmt(AssignTarget(VarRef("bl_trans_atten_color")),
                trans_args.get("attenuation_color",
                    ConstructorExpr("vec3", [NumberLit("1.0")]))))
            body.append(AssignStmt(AssignTarget(VarRef("bl_trans_atten_dist")),
                trans_args.get("attenuation_distance",
                    NumberLit("1000000.0"))))

        # --- Load sheen params from layer args ---
        if _has_sheen:
            sheen_args = _get_layer_args(layers_by_name["sheen"])
            body.append(AssignStmt(AssignTarget(VarRef("bl_sheen_color")),
                sheen_args.get("color",
                    ConstructorExpr("vec3", [NumberLit("0.0")]))))
            body.append(AssignStmt(AssignTarget(VarRef("bl_sheen_roughness")),
                sheen_args.get("roughness", NumberLit("0.5"))))

        # --- Load coat params from layer args ---
        if _has_coat:
            coat_args = _get_layer_args(layers_by_name["coat"])
            body.append(AssignStmt(AssignTarget(VarRef("bl_coat_factor")),
                coat_args.get("factor", NumberLit("1.0"))))
            body.append(AssignStmt(AssignTarget(VarRef("bl_coat_roughness")),
                coat_args.get("roughness", NumberLit("0.0"))))

        # --- Load emission from layer args ---
        if _has_emission:
            em_args = _get_layer_args(layers_by_name["emission"])
            body.append(AssignStmt(AssignTarget(VarRef("bl_emission")),
                em_args.get("color",
                    ConstructorExpr("vec3", [NumberLit("0.0")]))))

        # --- IBL sampling (path-specific: uses layer arg texture expressions) ---
        body.append(LetStmt("bl_ambient", "vec3",
            ConstructorExpr("vec3", [NumberLit("0.0")])))
        body.append(LetStmt("bl_coat_ibl", "vec3",
            ConstructorExpr("vec3", [NumberLit("0.0")])))

        if _has_ibl:
            specular_map = ibl_args.get("specular_map")
            irradiance_map = ibl_args.get("irradiance_map")
            brdf_lut_ref = ibl_args.get("brdf_lut")

            body.append(LetStmt("r", "vec3", CallExpr(VarRef("reflect"), [
                BinaryOp("*", VarRef("v"), UnaryOp("-", NumberLit("1.0"))),
                VarRef("n"),
            ])))
            body.append(LetStmt("prefiltered", "vec3", SwizzleAccess(
                CallExpr(VarRef("sample_lod"), [
                    specular_map, VarRef("r"),
                    BinaryOp("*", VarRef("layer_roughness"),
                             NumberLit("8.0")),
                ]), "xyz")))
            body.append(LetStmt("irradiance", "vec3", SwizzleAccess(
                CallExpr(VarRef("sample"), [irradiance_map, VarRef("n")]),
                "xyz")))
            body.append(LetStmt("brdf_sample", "vec2", SwizzleAccess(
                CallExpr(VarRef("sample"), [
                    brdf_lut_ref,
                    ConstructorExpr("vec2", [
                        VarRef("n_dot_v"), VarRef("layer_roughness"),
                    ]),
                ]), "xy")))

            body.append(AssignStmt(AssignTarget(VarRef("bl_ambient")),
                CallExpr(VarRef("ibl_contribution"), [
                    VarRef("layer_albedo"), VarRef("layer_roughness"),
                    VarRef("layer_metallic"),
                    VarRef("n_dot_v"), VarRef("prefiltered"),
                    VarRef("irradiance"), VarRef("brdf_sample"),
                ])))

            # Coat IBL contribution (if coat layer present)
            if _has_coat:
                body.append(LetStmt("prefiltered_coat", "vec3", SwizzleAccess(
                    CallExpr(VarRef("sample_lod"), [
                        specular_map, VarRef("r"),
                        BinaryOp("*", VarRef("bl_coat_roughness"),
                                 NumberLit("8.0")),
                    ]), "xyz")))
                body.append(AssignStmt(
                    AssignTarget(VarRef("bl_coat_ibl")),
                    CallExpr(VarRef("coat_ibl"), [
                        VarRef("bl_coat_factor"),
                        VarRef("bl_coat_roughness"),
                        VarRef("n"), VarRef("v"),
                        VarRef("prefiltered_coat"),
                    ])))

        # --- Compose via compose_pbr_layers (stdlib) ---
        def _emit_compose_call(emission_expr):
            return LetStmt("composed", "vec3", CallExpr(
                VarRef("compose_pbr_layers"), [
                    VarRef(result_var),
                    VarRef("n"), VarRef("v"), VarRef("l"),
                    VarRef("layer_albedo"), VarRef("layer_roughness"),
                    VarRef("bl_trans_factor"), VarRef("bl_trans_ior"),
                    VarRef("bl_trans_thickness"),
                    VarRef("bl_trans_atten_color"),
                    VarRef("bl_trans_atten_dist"),
                    VarRef("bl_ambient"),
                    VarRef("bl_sheen_color"), VarRef("bl_sheen_roughness"),
                    VarRef("bl_coat_factor"), VarRef("bl_coat_roughness"),
                    VarRef("bl_coat_ibl"),
                    emission_expr,
                ]))

        if _has_custom_layers:
            # Compose WITHOUT emission (custom layers run before emission)
            body.append(_emit_compose_call(
                ConstructorExpr("vec3", [NumberLit("0.0")])))
            result_var = "composed"
            for layer, fn in _custom_layer_list:
                result_var = _emit_custom_layer(
                    layer, fn, result_var, body,
                    prefix=f"custom_{layer.name}", rewrite_for_rt=False)
            body.append(LetStmt("hdr", "vec3",
                BinaryOp("+", VarRef(result_var), VarRef("bl_emission"))))
            result_var = "hdr"
        else:
            body.append(_emit_compose_call(VarRef("bl_emission")))
            result_var = "composed"

    else:
        # No compositing layers — apply custom @layer functions + emission inline
        if _has_custom_layers:
            for layer, fn in _custom_layer_list:
                result_var = _emit_custom_layer(
                    layer, fn, result_var, body,
                    prefix=f"custom_{layer.name}", rewrite_for_rt=False)
        if _has_emission:
            em_args = _get_layer_args(layers_by_name["emission"])
            em_expr = em_args.get("color",
                ConstructorExpr("vec3", [NumberLit("0.0")]))
            body.append(LetStmt("emission_color", "vec3", em_expr))
            body.append(LetStmt("hdr", "vec3",
                BinaryOp("+", VarRef(result_var), VarRef("emission_color"))))
            result_var = "hdr"

    # --- Tonemap + gamma ---
    output_var = _emit_tonemap_output(body, result_var, schedule)

    # Output final color
    body.append(AssignStmt(
        AssignTarget(VarRef("color")),
        ConstructorExpr("vec4", [VarRef(output_var), NumberLit("1.0")]),
    ))

    return body


def _expand_brdf_call(expr, schedule: dict[str, str] | None = None) -> any:
    """Expand a BRDF expression from surface declaration into concrete calls.

    Surface BRDF expressions like `lambert(albedo: vec3(0.8))` get expanded
    to full evaluation calls: `lambert_brdf(vec3(0.8), max(dot(n, l), 0.0))`.

    If a schedule is provided, selects fast/alternative BRDF implementations
    based on the strategy choices (distribution, geometry_term, fresnel).
    """
    if isinstance(expr, CallExpr) and isinstance(expr.func, VarRef):
        fname = expr.func.name
        if fname == "lambert":
            # lambert(albedo) -> lambert_brdf(albedo, max(dot(n, l), 0.0))
            albedo = expr.args[0] if expr.args else ConstructorExpr("vec3", [NumberLit("1.0")])
            return CallExpr(VarRef("lambert_brdf"), [
                albedo,
                CallExpr(VarRef("max"), [
                    CallExpr(VarRef("dot"), [VarRef("n"), VarRef("l")]),
                    NumberLit("0.0"),
                ]),
            ])
        elif fname == "microfacet_ggx":
            roughness = expr.args[0] if len(expr.args) > 0 else NumberLit("0.5")
            f0 = expr.args[1] if len(expr.args) > 1 else ConstructorExpr("vec3", [NumberLit("0.04")])
            # Choose variant based on schedule
            if schedule:
                dist = schedule.get("distribution", "ggx")
                geom = schedule.get("geometry_term", "smith_ggx")
                fresnel = schedule.get("fresnel", "schlick")
                if dist == "ggx_fast" or geom == "smith_ggx_fast":
                    return CallExpr(VarRef("microfacet_brdf_fast"), [
                        VarRef("n"), VarRef("v"), VarRef("l"),
                        roughness, f0,
                    ])
                if fresnel == "schlick_roughness":
                    return CallExpr(VarRef("microfacet_brdf_roughness"), [
                        VarRef("n"), VarRef("v"), VarRef("l"),
                        roughness, f0,
                    ])
            return CallExpr(VarRef("microfacet_brdf"), [
                VarRef("n"), VarRef("v"), VarRef("l"),
                roughness, f0,
            ])
        elif fname == "pbr":
            albedo = expr.args[0] if len(expr.args) > 0 else ConstructorExpr("vec3", [NumberLit("0.5")])
            roughness = expr.args[1] if len(expr.args) > 1 else NumberLit("0.5")
            metallic = expr.args[2] if len(expr.args) > 2 else NumberLit("0.0")
            # Choose variant based on schedule
            if schedule:
                dist = schedule.get("distribution", "ggx")
                geom = schedule.get("geometry_term", "smith_ggx")
                if dist == "ggx_fast" or geom == "smith_ggx_fast":
                    return CallExpr(VarRef("pbr_brdf_fast"), [
                        VarRef("n"), VarRef("v"), VarRef("l"),
                        albedo, roughness, metallic,
                    ])
            return CallExpr(VarRef("pbr_brdf"), [
                VarRef("n"), VarRef("v"), VarRef("l"),
                albedo, roughness, metallic,
            ])

    # Default: pass through as-is (it's a custom expression)
    return expr


def _infer_output_type(name: str) -> str:
    """Infer the type of a geometry output from its name."""
    type_hints = {
        "world_pos": "vec3",
        "world_normal": "vec3",
        "world_tangent": "vec3",
        "world_bitangent": "vec3",
        "frag_pos": "vec3",
        "frag_normal": "vec3",
        "frag_uv": "vec2",
        "uv": "vec2",
    }
    return type_hints.get(name, "vec3")


# =========================================================================
# RT Pipeline Expansion
# =========================================================================

def _expand_rt_pipeline(
    surface: SurfaceDecl | None,
    environment: EnvironmentDecl | None,
    procedural: ProceduralDecl | None,
    module: Module,
    schedule: dict[str, str] | None = None,
    max_bounces: int = 1,
    bindless: bool = False,
    lighting: LightingDecl | None = None,
) -> list[StageBlock]:
    """Generate ray tracing stages from pipeline declarations."""
    stages = []

    # 1. Ray generation shader
    raygen = _expand_raygen(surface, environment, max_bounces)
    stages.append(raygen)

    # 2. Closest-hit shader from surface
    if surface:
        if bindless and surface.layers is not None:
            chit = _expand_bindless_closest_hit(surface, module, schedule,
                                                lighting=lighting)
        else:
            chit = _expand_surface_to_closest_hit(surface, module, schedule,
                                                  lighting=lighting)
        stages.append(chit)

        # 3. Any-hit shader if surface has opacity
        has_opacity = any(m.name == "opacity" for m in surface.members)
        if has_opacity:
            ahit = _expand_surface_to_any_hit(surface)
            stages.append(ahit)

    # 4. Miss shader from environment
    if environment:
        miss = _expand_environment_to_miss(environment)
        stages.append(miss)

    # 5. Intersection shader from procedural
    if procedural:
        isect = _expand_procedural_to_intersection(procedural)
        stages.append(isect)

    return stages


def _expand_raygen(
    surface: SurfaceDecl | None,
    environment: EnvironmentDecl | None,
    max_bounces: int = 1,
) -> StageBlock:
    """Generate a ray generation shader.

    Creates a simple primary ray caster that:
    - Computes ray from launch_id/launch_size (camera model)
    - Calls trace_ray
    - Stores result from payload into output image
    """
    stage = StageBlock(stage_type="raygen")

    # Acceleration structure binding
    stage.accel_structs.append(AccelDecl("tlas"))

    # Ray payload: color result
    stage.ray_payloads.append(RayPayloadDecl("payload", "vec4"))

    # Camera uniform (matches engine layout: 2 x mat4, 128 bytes)
    stage.uniforms.append(UniformBlock("Camera", [
        BlockField("inv_view", "mat4"),
        BlockField("inv_proj", "mat4"),
    ]))

    # Output storage image (RT shaders cannot use Output storage class)
    stage.storage_images.append(StorageImageDecl("result_color"))

    # Main body: compute ray and trace (matches hand-written gltf_pbr_rt.lux)
    body = []

    # let pixel: vec2 = vec2(launch_id.xy);
    body.append(LetStmt("pixel", "vec2",
        ConstructorExpr("vec2", [SwizzleAccess(VarRef("launch_id"), "xy")])))

    # let dims: vec2 = vec2(launch_size.xy);
    body.append(LetStmt("dims", "vec2",
        ConstructorExpr("vec2", [SwizzleAccess(VarRef("launch_size"), "xy")])))

    # let ndc: vec2 = (pixel + vec2(0.5)) / dims * 2.0 - vec2(1.0);
    body.append(LetStmt("ndc", "vec2", BinaryOp("-",
        BinaryOp("*",
            BinaryOp("/",
                BinaryOp("+", VarRef("pixel"),
                         ConstructorExpr("vec2", [NumberLit("0.5")])),
                VarRef("dims")),
            NumberLit("2.0")),
        ConstructorExpr("vec2", [NumberLit("1.0")]),
    )))

    # Initialize payload to zero
    body.append(AssignStmt(
        AssignTarget(VarRef("payload")),
        ConstructorExpr("vec4", [NumberLit("0.0")]),
    ))

    # let target: vec4 = inv_proj * vec4(ndc.x, ndc.y, 1.0, 1.0);
    body.append(LetStmt("target", "vec4", BinaryOp("*",
        VarRef("inv_proj"),
        ConstructorExpr("vec4", [
            SwizzleAccess(VarRef("ndc"), "x"),
            SwizzleAccess(VarRef("ndc"), "y"),
            NumberLit("1.0"),
            NumberLit("1.0"),
        ]),
    )))

    # let direction: vec4 = inv_view * vec4(normalize(target.xyz), 0.0);
    body.append(LetStmt("direction", "vec4", BinaryOp("*",
        VarRef("inv_view"),
        ConstructorExpr("vec4", [
            CallExpr(VarRef("normalize"), [SwizzleAccess(VarRef("target"), "xyz")]),
            NumberLit("0.0"),
        ]),
    )))

    # let origin: vec3 = (inv_view * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    body.append(LetStmt("origin", "vec3", SwizzleAccess(
        BinaryOp("*", VarRef("inv_view"),
                 ConstructorExpr("vec4", [
                     NumberLit("0.0"), NumberLit("0.0"),
                     NumberLit("0.0"), NumberLit("1.0"),
                 ])),
        "xyz",
    )))

    # let dir: vec3 = normalize(direction.xyz);
    body.append(LetStmt("dir", "vec3",
        CallExpr(VarRef("normalize"), [
            SwizzleAccess(VarRef("direction"), "xyz")])))

    # trace_ray(tlas, 0, 255, 0, 0, 0, origin, 0.001, dir, 1000.0, 0);
    body.append(ExprStmt(CallExpr(VarRef("trace_ray"), [
        VarRef("tlas"),
        NumberLit("0"),    # ray_flags
        NumberLit("255"),  # cull_mask
        NumberLit("0"),    # sbt_offset
        NumberLit("0"),    # sbt_stride
        NumberLit("0"),    # miss_index
        VarRef("origin"),
        NumberLit("0.001"),  # tmin
        VarRef("dir"),
        NumberLit("1000.0"),  # tmax
        NumberLit("0"),    # payload location
    ])))

    # image_store(result_color, launch_id.xy, payload);
    body.append(ExprStmt(CallExpr(VarRef("image_store"), [
        VarRef("result_color"),
        SwizzleAccess(VarRef("launch_id"), "xy"),
        VarRef("payload"),
    ])))

    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


def _expand_surface_to_closest_hit(
    surface: SurfaceDecl,
    module: Module,
    schedule: dict[str, str] | None = None,
    lighting: LightingDecl | None = None,
) -> StageBlock:
    """Generate a closest-hit shader from a surface declaration.

    For surfaces with `layers`, generates full barycentric interpolation,
    storage buffer reads, texture sampling (sample_lod), and layered
    PBR evaluation matching the hand-written gltf_pbr_rt.lux output.

    For surfaces with `brdf:`, generates the simpler existing path.
    """
    if surface.layers is not None:
        return _expand_layered_closest_hit(surface, module, schedule,
                                           lighting=lighting)

    stage = StageBlock(stage_type="closest_hit")

    # Incoming ray payload
    stage.ray_payloads.append(RayPayloadDecl("payload", "vec4"))

    # Hit attributes (barycentric coordinates from triangle intersection)
    stage.hit_attributes.append(HitAttributeDecl("attribs", "vec2"))

    # Main body: evaluate BRDF
    body = []

    # Compute hit point from ray origin + direction * hit_t
    body.append(LetStmt("hit_pos", "vec3", BinaryOp("+",
        VarRef("world_ray_origin"),
        BinaryOp("*", VarRef("world_ray_direction"), VarRef("hit_t")),
    )))

    # Use a default normal pointing up (in a real implementation,
    # this would be interpolated from vertex normals using barycentrics)
    body.append(LetStmt("n", "vec3", ConstructorExpr("vec3", [
        NumberLit("0.0"), NumberLit("1.0"), NumberLit("0.0"),
    ])))

    # View direction: from hit point back to ray origin
    body.append(LetStmt("v", "vec3", CallExpr(VarRef("normalize"), [
        BinaryOp("-", VarRef("world_ray_origin"), VarRef("hit_pos")),
    ])))

    # Light direction (default: overhead directional)
    body.append(LetStmt("l", "vec3", CallExpr(VarRef("normalize"), [
        ConstructorExpr("vec3", [NumberLit("1.0"), NumberLit("0.8"), NumberLit("0.6")]),
    ])))

    # Evaluate the BRDF from the surface declaration
    brdf_expr = None
    for member in surface.members:
        if member.name == "brdf":
            brdf_expr = member.value

    if brdf_expr is not None:
        result_expr = _expand_brdf_call(brdf_expr, schedule)
        body.append(LetStmt("result", "vec3", result_expr))
    else:
        # Default: simple lambert
        body.append(LetStmt("ndotl", "scalar", CallExpr(VarRef("max"), [
            CallExpr(VarRef("dot"), [VarRef("n"), VarRef("l")]),
            NumberLit("0.0"),
        ])))
        body.append(LetStmt("result", "vec3", BinaryOp("*",
            ConstructorExpr("vec3", [NumberLit("1.0")]),
            VarRef("ndotl"),
        )))

    # Write result to payload
    body.append(AssignStmt(
        AssignTarget(VarRef("payload")),
        ConstructorExpr("vec4", [VarRef("result"), NumberLit("1.0")]),
    ))

    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


def _rewrite_sample_to_lod(expr):
    """Rewrite sample(tex, uv) → sample_lod(tex, uv, 0.0) in an AST expression.

    In RT shaders there are no implicit derivatives, so all texture sampling
    must use explicit LOD. This recursively walks the expression tree.
    """
    import copy

    if isinstance(expr, CallExpr):
        # Rewrite sample() calls (but not sample_lod which is already explicit)
        if isinstance(expr.func, VarRef) and expr.func.name == "sample":
            new_args = [_rewrite_sample_to_lod(a) for a in expr.args]
            new_args.append(NumberLit("0.0"))
            return CallExpr(VarRef("sample_lod"), new_args, loc=expr.loc)
        else:
            return CallExpr(
                _rewrite_sample_to_lod(expr.func),
                [_rewrite_sample_to_lod(a) for a in expr.args],
                loc=expr.loc,
            )
    elif isinstance(expr, BinaryOp):
        return BinaryOp(
            expr.op,
            _rewrite_sample_to_lod(expr.left),
            _rewrite_sample_to_lod(expr.right),
            loc=expr.loc,
        )
    elif isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, _rewrite_sample_to_lod(expr.operand), loc=expr.loc)
    elif isinstance(expr, ConstructorExpr):
        return ConstructorExpr(
            expr.type_name,
            [_rewrite_sample_to_lod(a) for a in expr.args],
            loc=expr.loc,
        )
    elif isinstance(expr, SwizzleAccess):
        return SwizzleAccess(
            _rewrite_sample_to_lod(expr.object), expr.components, loc=expr.loc,
        )
    elif isinstance(expr, FieldAccess):
        return FieldAccess(
            _rewrite_sample_to_lod(expr.object), expr.field, loc=expr.loc,
        )
    elif isinstance(expr, IndexAccess):
        return IndexAccess(
            _rewrite_sample_to_lod(expr.object),
            _rewrite_sample_to_lod(expr.index),
            loc=expr.loc,
        )
    # Leaf nodes: VarRef, NumberLit, etc. — no rewriting needed
    return expr


def _rewrite_refs(expr, name_map: dict[str, str]):
    """Deep-copy an AST expression, replacing VarRef names per name_map.

    Used by mesh shader output expansion so that cross-binding references
    (e.g. world_bitangent = cross(world_normal, world_tangent)) resolve to
    the local scalar temporaries rather than the per-vertex output arrays.
    """
    if isinstance(expr, VarRef):
        if expr.name in name_map:
            return VarRef(name_map[expr.name], loc=expr.loc)
        return expr
    elif isinstance(expr, CallExpr):
        return CallExpr(
            _rewrite_refs(expr.func, name_map),
            [_rewrite_refs(a, name_map) for a in expr.args],
            loc=expr.loc,
        )
    elif isinstance(expr, BinaryOp):
        return BinaryOp(
            expr.op,
            _rewrite_refs(expr.left, name_map),
            _rewrite_refs(expr.right, name_map),
            loc=expr.loc,
        )
    elif isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, _rewrite_refs(expr.operand, name_map), loc=expr.loc)
    elif isinstance(expr, ConstructorExpr):
        return ConstructorExpr(
            expr.type_name,
            [_rewrite_refs(a, name_map) for a in expr.args],
            loc=expr.loc,
        )
    elif isinstance(expr, SwizzleAccess):
        return SwizzleAccess(
            _rewrite_refs(expr.object, name_map), expr.components, loc=expr.loc,
        )
    elif isinstance(expr, FieldAccess):
        return FieldAccess(
            _rewrite_refs(expr.object, name_map), expr.field, loc=expr.loc,
        )
    elif isinstance(expr, IndexAccess):
        return IndexAccess(
            _rewrite_refs(expr.object, name_map),
            _rewrite_refs(expr.index, name_map),
            loc=expr.loc,
        )
    # Leaf nodes: NumberLit, etc. — no rewriting needed
    return expr


def _expand_layered_closest_hit(
    surface: SurfaceDecl,
    module: Module,
    schedule: dict[str, str] | None = None,
    lighting: LightingDecl | None = None,
) -> StageBlock:
    """Generate a closest-hit shader from a layered surface declaration.

    Produces output matching the hand-written gltf_pbr_rt.lux:
    - Storage buffers for vertex data (positions, normals, tex_coords, indices)
    - Barycentric interpolation to reconstruct hit-point attributes
    - sample_lod() for all texture sampling (no derivatives in RT)
    - Layered PBR evaluation with IBL and energy conservation
    """
    stage = StageBlock(stage_type="closest_hit")

    # Incoming ray payload
    stage.ray_payloads.append(RayPayloadDecl("payload", "vec4"))

    # Hit attributes (barycentrics)
    stage.hit_attributes.append(HitAttributeDecl("bary", "vec2"))

    # Storage buffers for vertex data (SoA layout)
    stage.storage_buffers.append(StorageBufferDecl("positions", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("normals", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("tex_coords", "vec2"))
    stage.storage_buffers.append(StorageBufferDecl("indices", "uint"))

    # Light uniform (from lighting block or legacy hardcoded)
    if lighting and lighting.properties:
        lprops = lighting.properties
        stage.uniforms.append(UniformBlock(lprops.name, [
            BlockField(f.name, f.type_name) for f in lprops.fields
        ]))
        if not hasattr(stage, '_properties_defaults'):
            stage._properties_defaults = {}
        for f in lprops.fields:
            if f.default is not None:
                stage._properties_defaults[(lprops.name, f.name)] = f.default
    else:
        stage.uniforms.append(UniformBlock("Light", [
            BlockField("light_dir", "vec3"),
            BlockField("view_pos", "vec3"),
        ]))

    # Material properties uniform (from surface properties block)
    if surface.properties:
        props = surface.properties
        stage.uniforms.append(UniformBlock(props.name, [
            BlockField(f.name, f.type_name) for f in props.fields
        ]))
        # Store defaults for reflection emission
        if not hasattr(stage, '_properties_defaults'):
            stage._properties_defaults = {}
        for f in props.fields:
            if f.default is not None:
                stage._properties_defaults[(props.name, f.name)] = f.default

    # Sampler declarations from the surface
    for sam in surface.samplers:
        stage.samplers.append(SamplerDecl(sam.name, type_name=sam.type_name))

    # Add sampler declarations from the lighting block
    if lighting:
        for sam in lighting.samplers:
            stage.samplers.append(SamplerDecl(sam.name, type_name=sam.type_name))

    # If multi_light layer present, add lights SSBO (and shadow_matrices if shadow_map arg)
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

    # --- Generate main body ---
    body = []

    # Barycentric interpolation preamble (matches gltf_pbr_rt.lux)
    _emit_barycentric_interpolation(body)

    # Index layers
    layers_by_name = {}
    for layer in surface.layers:
        layers_by_name[layer.name] = layer

    # --- View and light directions (RT-specific) ---
    body.append(LetStmt("v", "vec3", CallExpr(VarRef("normalize"), [
        UnaryOp("-", VarRef("world_ray_direction"))])))

    # Detect multi_light vs directional lighting mode
    _rt_has_multi_light = _detect_multi_light(lighting)
    rt_lighting_by_name = {}
    if lighting and lighting.layers:
        rt_lighting_by_name = {layer.name: layer for layer in lighting.layers}

    # Light direction: from lighting block (directional/multi_light) or legacy uniform
    if _rt_has_multi_light:
        # Multi-light path: l will be computed per-light inside unrolled loop.
        # Dummy l for downstream layers (coat, sheen, etc.)
        body.append(LetStmt("l", "vec3",
            ConstructorExpr("vec3", [NumberLit("0.0"), NumberLit("1.0"), NumberLit("0.0")])))
    elif lighting and lighting.layers:
        if "directional" in rt_lighting_by_name:
            dir_args = _get_layer_args(rt_lighting_by_name["directional"])
            body.append(LetStmt("l", "vec3",
                CallExpr(VarRef("normalize"), [
                    dir_args.get("direction", VarRef("light_dir"))])))
            body.append(LetStmt("light_color", "vec3",
                dir_args.get("color",
                    ConstructorExpr("vec3", [NumberLit("1.0")]))))
        else:
            body.append(LetStmt("l", "vec3",
                CallExpr(VarRef("normalize"), [VarRef("light_dir")])))
            body.append(LetStmt("light_color", "vec3",
                ConstructorExpr("vec3", [
                    NumberLit("1.0"), NumberLit("0.98"), NumberLit("0.95")])))
    else:
        body.append(LetStmt("l", "vec3",
            CallExpr(VarRef("normalize"), [VarRef("light_dir")])))
        body.append(LetStmt("light_color", "vec3",
            ConstructorExpr("vec3", [
                NumberLit("1.0"), NumberLit("0.98"), NumberLit("0.95")])))
    body.append(LetStmt("n_dot_v", "scalar", CallExpr(VarRef("max"), [
        CallExpr(VarRef("dot"), [VarRef("n"), VarRef("v")]),
        NumberLit("0.001"),
    ])))

    # --- Base layer: direct lighting ---
    # Rewrite sample->sample_lod for all layer expressions in RT
    result_var = "result_zero"
    body.append(LetStmt(result_var, "vec3",
        ConstructorExpr("vec3", [NumberLit("0.0")])))

    if "base" in layers_by_name:
        base_args = _get_layer_args(layers_by_name["base"])
        albedo_expr = _rewrite_sample_to_lod(
            base_args.get("albedo", ConstructorExpr("vec3", [NumberLit("0.5")])))
        roughness_expr = _rewrite_sample_to_lod(
            base_args.get("roughness", NumberLit("0.5")))
        metallic_expr = _rewrite_sample_to_lod(
            base_args.get("metallic", NumberLit("0.0")))

        body.append(LetStmt("layer_albedo", "vec3", albedo_expr))
        body.append(LetStmt("layer_roughness", "scalar", roughness_expr))
        body.append(LetStmt("layer_metallic", "scalar", metallic_expr))

        if _rt_has_multi_light:
            result_var = _emit_multi_light_from_lighting(
                body, lighting, "hit_pos")
        else:
            # Single directional light path
            body.append(LetStmt("direct", "vec3", CallExpr(VarRef("gltf_pbr"), [
                VarRef("n"), VarRef("v"), VarRef("l"),
                VarRef("layer_albedo"), VarRef("layer_roughness"),
                VarRef("layer_metallic"),
            ])))
            body.append(LetStmt("direct_lit", "vec3", BinaryOp("*",
                VarRef("direct"), VarRef("light_color"),
            )))
            result_var = "direct_lit"

    # --- Transmission layer (replaces diffuse proportionally) ---
    if "transmission" in layers_by_name:
        trans_args = _get_layer_args(layers_by_name["transmission"])
        factor_expr = _rewrite_sample_to_lod(
            trans_args.get("factor", NumberLit("0.0")))
        ior_expr = trans_args.get("ior", NumberLit("1.5"))
        thickness_expr = trans_args.get("thickness", NumberLit("0.0"))
        atten_color_expr = trans_args.get("attenuation_color",
            ConstructorExpr("vec3", [NumberLit("1.0")]))
        atten_dist_expr = trans_args.get("attenuation_distance",
            NumberLit("1000000.0"))

        body.append(LetStmt("trans_factor", "scalar", factor_expr))
        body.append(LetStmt("trans_ior", "scalar", ior_expr))
        body.append(LetStmt("trans_thickness", "scalar", thickness_expr))
        body.append(LetStmt("trans_atten_color", "vec3", atten_color_expr))
        body.append(LetStmt("trans_atten_dist", "scalar", atten_dist_expr))

        body.append(LetStmt("transmitted", "vec3", CallExpr(
            VarRef("transmission_replace"), [
                VarRef(result_var), VarRef("layer_albedo"),
                VarRef("layer_roughness"),
                VarRef("trans_factor"), VarRef("trans_ior"),
                VarRef("n"), VarRef("v"), VarRef("l"),
                VarRef("trans_thickness"), VarRef("trans_atten_color"),
                VarRef("trans_atten_dist"),
            ])))
        result_var = "transmitted"

    # --- IBL layer (check lighting block first, then surface) ---
    rt_ibl_args = None
    if lighting and lighting.layers:
        _rt_lighting_by_name = {layer.name: layer for layer in lighting.layers}
        if "ibl" in _rt_lighting_by_name:
            rt_ibl_args = _get_layer_args(_rt_lighting_by_name["ibl"])
    if rt_ibl_args is None and "ibl" in layers_by_name:
        rt_ibl_args = _get_layer_args(layers_by_name["ibl"])

    if rt_ibl_args is not None:
        specular_map = rt_ibl_args.get("specular_map")
        irradiance_map = rt_ibl_args.get("irradiance_map")
        brdf_lut_ref = rt_ibl_args.get("brdf_lut")

        body.append(LetStmt("r", "vec3", CallExpr(VarRef("reflect"), [
            BinaryOp("*", VarRef("v"), UnaryOp("-", NumberLit("1.0"))),
            VarRef("n"),
        ])))

        # RT: all sampling uses sample_lod
        body.append(LetStmt("prefiltered", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                specular_map, VarRef("r"),
                BinaryOp("*", VarRef("layer_roughness"), NumberLit("8.0")),
            ]), "xyz")))
        body.append(LetStmt("irradiance", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                irradiance_map, VarRef("n"), NumberLit("0.0"),
            ]), "xyz")))
        body.append(LetStmt("brdf_sample", "vec2", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                brdf_lut_ref,
                ConstructorExpr("vec2", [
                    VarRef("n_dot_v"), VarRef("layer_roughness"),
                ]),
                NumberLit("0.0"),
            ]), "xy")))

        # Delegate math to compositing.lux::ibl_contribution()
        body.append(LetStmt("ambient", "vec3", CallExpr(
            VarRef("ibl_contribution"), [
                VarRef("layer_albedo"), VarRef("layer_roughness"),
                VarRef("layer_metallic"),
                VarRef("n_dot_v"), VarRef("prefiltered"), VarRef("irradiance"),
                VarRef("brdf_sample"),
            ])))

        # Coat IBL contribution (if coat layer present)
        if "coat" in layers_by_name:
            coat_args = _get_layer_args(layers_by_name["coat"])
            coat_factor_expr = _rewrite_sample_to_lod(
                coat_args.get("factor", NumberLit("1.0")))
            coat_rough_expr = _rewrite_sample_to_lod(
                coat_args.get("roughness", NumberLit("0.0")))
            body.append(LetStmt("coat_ibl_factor", "scalar", coat_factor_expr))
            body.append(LetStmt("coat_ibl_roughness", "scalar",
                                coat_rough_expr))
            body.append(LetStmt("prefiltered_coat", "vec3", SwizzleAccess(
                CallExpr(VarRef("sample_lod"), [
                    specular_map, VarRef("r"),
                    BinaryOp("*", VarRef("coat_ibl_roughness"),
                             NumberLit("8.0")),
                ]),
                "xyz",
            )))
            body.append(LetStmt("coat_ibl_contrib", "vec3", CallExpr(
                VarRef("coat_ibl"), [
                    VarRef("coat_ibl_factor"), VarRef("coat_ibl_roughness"),
                    VarRef("n"), VarRef("v"), VarRef("prefiltered_coat"),
                ])))
            body.append(LetStmt("ambient_with_coat", "vec3",
                BinaryOp("+", VarRef("ambient"),
                         VarRef("coat_ibl_contrib"))))
            body.append(LetStmt("hdr_partial", "vec3",
                BinaryOp("+", VarRef(result_var),
                         VarRef("ambient_with_coat"))))
        else:
            body.append(LetStmt("hdr_partial", "vec3",
                BinaryOp("+", VarRef(result_var), VarRef("ambient"))))
        result_var = "hdr_partial"

    # --- Sheen layer (additive with energy conservation) ---
    if "sheen" in layers_by_name:
        sheen_args = _get_layer_args(layers_by_name["sheen"])
        color_expr = _rewrite_sample_to_lod(
            sheen_args.get("color",
                ConstructorExpr("vec3", [NumberLit("0.0")])))
        rough_expr = _rewrite_sample_to_lod(
            sheen_args.get("roughness", NumberLit("0.5")))

        body.append(LetStmt("sheen_color_val", "vec3", color_expr))
        body.append(LetStmt("sheen_roughness_val", "scalar", rough_expr))

        body.append(LetStmt("sheened", "vec3", CallExpr(
            VarRef("sheen_over"), [
                VarRef(result_var), VarRef("n"), VarRef("v"), VarRef("l"),
                VarRef("sheen_color_val"), VarRef("sheen_roughness_val"),
            ])))
        result_var = "sheened"

    # --- Coat layer (outermost, attenuates everything + adds specular) ---
    if "coat" in layers_by_name:
        coat_args = _get_layer_args(layers_by_name["coat"])
        factor_expr = _rewrite_sample_to_lod(
            coat_args.get("factor", NumberLit("1.0")))
        rough_expr = _rewrite_sample_to_lod(
            coat_args.get("roughness", NumberLit("0.0")))
        coat_normal = coat_args.get("normal", None)

        body.append(LetStmt("coat_factor", "scalar", factor_expr))
        body.append(LetStmt("coat_roughness", "scalar", rough_expr))
        coat_n = VarRef("coat_n") if coat_normal else VarRef("n")
        if coat_normal:
            body.append(LetStmt("coat_n", "vec3",
                                _rewrite_sample_to_lod(coat_normal)))

        body.append(LetStmt("coated", "vec3", CallExpr(
            VarRef("coat_over"), [
                VarRef(result_var), coat_n, VarRef("v"), VarRef("l"),
                VarRef("coat_factor"), VarRef("coat_roughness"),
            ])))
        result_var = "coated"

    # --- Custom @layer functions (declaration order, after coat, before emission) ---
    layer_fns = _collect_layer_functions(module)
    for layer in surface.layers:
        if layer.name in layer_fns:
            result_var = _emit_custom_layer(
                layer, layer_fns[layer.name], result_var, body,
                prefix=f"custom_{layer.name}", rewrite_for_rt=True)

    # --- Emission layer ---
    if "emission" in layers_by_name:
        em_args = _get_layer_args(layers_by_name["emission"])
        em_expr = _rewrite_sample_to_lod(
            em_args.get("color", ConstructorExpr("vec3", [NumberLit("0.0")])))
        body.append(LetStmt("emission_color", "vec3", em_expr))
        body.append(LetStmt("hdr", "vec3",
            BinaryOp("+", VarRef(result_var), VarRef("emission_color"))))
        result_var = "hdr"

    # --- Tonemap + gamma ---
    output_var = _emit_tonemap_output(body, result_var, schedule)

    # Write to payload
    body.append(AssignStmt(
        AssignTarget(VarRef("payload")),
        ConstructorExpr("vec4", [VarRef(output_var), NumberLit("1.0")]),
    ))

    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


def _expand_surface_to_any_hit(surface: SurfaceDecl) -> StageBlock:
    """Generate an any-hit shader from a surface with opacity.

    Reads the opacity value and calls ignore_intersection if transparent.
    """
    stage = StageBlock(stage_type="any_hit")

    # Hit attributes
    stage.hit_attributes.append(HitAttributeDecl("attribs", "vec2"))

    body = []

    # Find opacity expression
    opacity_expr = None
    for member in surface.members:
        if member.name == "opacity":
            opacity_expr = member.value

    if opacity_expr is not None:
        body.append(LetStmt("alpha", "scalar", opacity_expr))
    else:
        body.append(LetStmt("alpha", "scalar", NumberLit("1.0")))

    # if (alpha < 0.5) { ignore_intersection(); }
    from luxc.parser.ast_nodes import IfStmt
    body.append(IfStmt(
        BinaryOp("<", VarRef("alpha"), NumberLit("0.5")),
        [ExprStmt(CallExpr(VarRef("ignore_intersection"), []))],
        [],
    ))

    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


def _expand_environment_to_miss(environment: EnvironmentDecl) -> StageBlock:
    """Generate a miss shader from an environment declaration.

    The miss shader sets the payload color based on the environment definition,
    typically using the incoming ray direction for sky/environment mapping.
    """
    stage = StageBlock(stage_type="miss")

    # Incoming ray payload
    stage.ray_payloads.append(RayPayloadDecl("payload", "vec4"))

    # Add samplers from environment
    for sam in environment.samplers:
        stage.samplers.append(SamplerDecl(sam.name, type_name=sam.type_name))

    body = []

    # Find color expression
    color_expr = None
    for member in environment.members:
        if member.name == "color":
            color_expr = member.value

    if color_expr is not None:
        body.append(LetStmt("env_color", "vec3", color_expr))
    else:
        # Default sky gradient
        body.append(LetStmt("t", "scalar", BinaryOp("*",
            BinaryOp("+", SwizzleAccess(VarRef("world_ray_direction"), "y"), NumberLit("1.0")),
            NumberLit("0.5"),
        )))
        body.append(LetStmt("env_color", "vec3", CallExpr(VarRef("mix"), [
            ConstructorExpr("vec3", [NumberLit("1.0")]),
            ConstructorExpr("vec3", [NumberLit("0.5"), NumberLit("0.7"), NumberLit("1.0")]),
            VarRef("t"),
        ])))

    # Write to payload
    body.append(AssignStmt(
        AssignTarget(VarRef("payload")),
        ConstructorExpr("vec4", [VarRef("env_color"), NumberLit("1.0")]),
    ))

    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


def _expand_procedural_to_intersection(procedural: ProceduralDecl) -> StageBlock:
    """Generate an intersection shader from a procedural declaration.

    Evaluates the SDF expression to find ray-surface intersection using
    sphere tracing (ray marching). Reports intersection when close enough.
    """
    stage = StageBlock(stage_type="intersection")

    # Hit attributes for barycentric-like data
    stage.hit_attributes.append(HitAttributeDecl("attribs", "vec2"))

    body = []

    # Find SDF expression
    sdf_expr = None
    for member in procedural.members:
        if member.name == "sdf":
            sdf_expr = member.value

    if sdf_expr is None:
        # Default: unit sphere
        sdf_expr = CallExpr(VarRef("sdf_sphere"), [NumberLit("1.0")])

    # Simple ray marching loop unrolled as chained let statements
    # (Lux has no loops — we unroll a fixed number of march steps)
    max_steps = 64
    threshold = 0.001

    # Initialize march state
    body.append(LetStmt("t", "scalar", VarRef("ray_tmin")))
    body.append(LetStmt("hit_found", "scalar", NumberLit("0.0")))

    # For each step, compute position, evaluate SDF, advance
    # We unroll a small number of steps (8) as a demonstration
    # (Full 64 steps would bloat the AST; in practice the user writes
    # their own intersection shader for complex procedurals)
    for i in range(8):
        sfx = f"_{i}"
        body.append(LetStmt(f"p{sfx}", "vec3", BinaryOp("+",
            VarRef("world_ray_origin"),
            BinaryOp("*", VarRef("world_ray_direction"), VarRef("t")),
        )))

        # Evaluate SDF at the march point
        # We wrap the sdf_expr to pass the current position
        if isinstance(sdf_expr, CallExpr) and isinstance(sdf_expr.func, VarRef):
            sdf_call = CallExpr(sdf_expr.func, [VarRef(f"p{sfx}")] + list(sdf_expr.args))
        else:
            sdf_call = sdf_expr

        body.append(LetStmt(f"d{sfx}", "scalar", sdf_call))

        # Advance: t = t + d
        body.append(AssignStmt(
            AssignTarget(VarRef("t")),
            BinaryOp("+", VarRef("t"), VarRef(f"d{sfx}")),
        ))

    # After marching, check if we hit (last distance < threshold)
    from luxc.parser.ast_nodes import IfStmt
    body.append(IfStmt(
        BinaryOp("<", VarRef("d_7"), NumberLit(str(threshold))),
        [
            # Set attribs to zero (no barycentric for procedural)
            AssignStmt(
                AssignTarget(VarRef("attribs")),
                ConstructorExpr("vec2", [NumberLit("0.0")]),
            ),
            # Report intersection
            ExprStmt(CallExpr(VarRef("report_intersection"), [
                VarRef("t"),
                NumberLit("0"),  # hit_kind
            ])),
        ],
        [],
    ))

    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


# =========================================================================
# Mesh Shader Pipeline Expansion
# =========================================================================

def _expand_mesh_pipeline(
    surface: SurfaceDecl | None,
    geometry: GeometryDecl | None,
    module: Module,
    schedule: dict[str, str] | None = None,
    pipeline: PipelineDecl | None = None,
    use_task_shader: bool = False,
    bindless: bool = False,
    lighting: LightingDecl | None = None,
) -> list[StageBlock]:
    """Generate mesh + fragment stages (and optionally task) from pipeline declarations.

    The mesh shader replaces the traditional vertex stage with a compute-like shader
    that reads from storage buffers and outputs vertices + triangle indices per workgroup.
    The fragment stage is identical to rasterization (reuses _expand_surface_to_fragment).
    """
    stages = []
    defines = getattr(module, '_defines', {})

    # 1. Optional task shader (amplification)
    if use_task_shader:
        task = _expand_task_shader(defines)
        task._descriptor_set_offset = 0
        stages.append(task)

    # 2. Mesh shader (bindless adds material_index to push constants)
    mesh = _expand_geometry_to_mesh(geometry, module, defines, bindless=bindless)
    mesh._descriptor_set_offset = 1 if use_task_shader else 0
    stages.append(mesh)

    # 3. Fragment shader (bindless or standard)
    if surface:
        if bindless and surface.layers is not None:
            frag = _expand_bindless_fragment(surface, module, geometry, schedule,
                                             lighting=lighting)
        else:
            frag = _expand_surface_to_fragment(surface, module, geometry, schedule,
                                               lighting=lighting)
        frag._descriptor_set_offset = (2 if use_task_shader else 1)
        stages.append(frag)

    return stages


def _expand_geometry_to_mesh(
    geometry: GeometryDecl | None,
    module: Module,
    defines: dict[str, int],
    bindless: bool = False,
) -> StageBlock:
    """Generate a mesh stage from a geometry declaration.

    The mesh shader reads meshlet data from storage buffers and outputs
    vertices + triangle indices. Each workgroup processes one meshlet.
    """
    stage = StageBlock(stage_type="mesh")

    max_verts = defines.get('max_vertices', 64)
    max_prims = defines.get('max_primitives', 124)

    # Storage buffers for meshlet data
    stage.storage_buffers.append(StorageBufferDecl("meshlet_descriptors", "uvec4"))
    stage.storage_buffers.append(StorageBufferDecl("meshlet_vertices", "uint"))
    stage.storage_buffers.append(StorageBufferDecl("meshlet_triangles", "uint"))

    # Storage buffers for vertex data (SoA, same as RT)
    stage.storage_buffers.append(StorageBufferDecl("positions", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("normals", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("tex_coords", "vec2"))

    # Optional tangent buffer (present when geometry has a tangent field)
    if geometry and any(f.name == "tangent" for f in geometry.fields):
        stage.storage_buffers.append(StorageBufferDecl("tangents", "vec4"))

    # Push constant for meshlet offset (+ material_index in bindless mode)
    pc_fields = [BlockField("meshletOffset", "uint")]
    if bindless:
        pc_fields.append(BlockField("material_index", "uint"))
    stage.push_constants.append(PushBlock("mesh_params", pc_fields))

    # Transform uniform (from geometry declaration)
    if geometry and geometry.transform:
        ub = UniformBlock(
            geometry.transform.name,
            [BlockField(f.name, f.type_name) for f in geometry.transform.fields],
        )
        stage.uniforms.append(ub)

    # Per-vertex outputs (excluding clip_pos which maps to gl_Position)
    if geometry and geometry.outputs:
        for binding in geometry.outputs.bindings:
            if binding.name != "clip_pos":
                out_type = _infer_output_type(binding.name)
                v = VarDecl(binding.name, out_type)
                v._is_input = False
                stage.outputs.append(v)

    # Generate main function body
    body = _generate_mesh_main(geometry, defines)
    stage.functions.append(FunctionDef("main", [], None, body))

    return stage


def _generate_mesh_main(
    geometry: GeometryDecl | None,
    defines: dict[str, int],
) -> list:
    """Generate mesh shader main() body with unrolled iterations.

    Each workgroup has `workgroup_size` threads but may need to process up to
    `max_vertices` vertices and `max_primitives` triangles.  When these limits
    exceed workgroup_size, we unroll: each thread handles multiple vertices /
    triangles at stride = workgroup_size.

    Pattern per iteration i (offset = i * workgroup_size):
      let vid = tid + offset;
      if (vid < vert_count) { process vertex vid }

      let trid = tid + offset;
      if (trid < tri_count) { write triangle trid }
    """
    body = []

    workgroup_size = defines.get('workgroup_size', 32)
    max_verts = defines.get('max_vertices', 64)
    max_prims = defines.get('max_primitives', 124)

    vert_iters = (max_verts + workgroup_size - 1) // workgroup_size
    tri_iters = (max_prims + workgroup_size - 1) // workgroup_size

    # --- Common setup ---
    body.append(LetStmt("meshlet_id", "scalar",
        BinaryOp("+",
                 SwizzleAccess(VarRef("workgroup_id"), "x"),
                 VarRef("meshletOffset"))))
    body.append(LetStmt("tid", "uint", VarRef("local_invocation_index")))

    body.append(LetStmt("desc", "uvec4",
        IndexAccess(VarRef("meshlet_descriptors"), VarRef("meshlet_id"))))

    body.append(LetStmt("vert_offset", "scalar", SwizzleAccess(VarRef("desc"), "x")))
    body.append(LetStmt("vert_count", "scalar", SwizzleAccess(VarRef("desc"), "y")))
    body.append(LetStmt("tri_offset", "scalar", SwizzleAccess(VarRef("desc"), "z")))
    body.append(LetStmt("tri_count", "scalar", SwizzleAccess(VarRef("desc"), "w")))

    body.append(ExprStmt(CallExpr(VarRef("set_mesh_outputs"), [
        VarRef("vert_count"),
        VarRef("tri_count"),
    ])))

    # --- Vertex processing (unrolled) ---
    for i in range(vert_iters):
        if i == 0:
            vid = "tid"
        else:
            vid = f"vid_{i}"
            body.append(LetStmt(vid, "scalar",
                BinaryOp("+", VarRef("tid"), NumberLit(str(i * workgroup_size)))))

        vbody = _make_vertex_iteration(geometry, vid)
        body.append(IfStmt(
            BinaryOp("<", VarRef(vid), VarRef("vert_count")),
            vbody,
            [],
        ))

    # --- Triangle index writing (unrolled) ---
    for i in range(tri_iters):
        if i == 0:
            trid = "tid"
        else:
            trid = f"trid_{i}"
            body.append(LetStmt(trid, "scalar",
                BinaryOp("+", VarRef("tid"), NumberLit(str(i * workgroup_size)))))

        tbody = _make_tri_iteration(trid)
        body.append(IfStmt(
            BinaryOp("<", VarRef(trid), VarRef("tri_count")),
            tbody,
            [],
        ))

    return body


def _make_vertex_iteration(
    geometry: GeometryDecl | None,
    vid: str,
) -> list:
    """Generate one iteration of vertex processing for invocation variable *vid*."""
    vbody = []

    # Read vertex index and position from storage buffers
    vbody.append(LetStmt("global_idx", "uint",
        IndexAccess(VarRef("meshlet_vertices"),
            BinaryOp("+", VarRef("vert_offset"), VarRef(vid)))))
    vbody.append(LetStmt("pos", "vec4",
        IndexAccess(VarRef("positions"), VarRef("global_idx"))))

    # Bind geometry fields from storage buffers (position, normal, uv)
    if geometry and geometry.fields:
        for field in geometry.fields:
            if field.name == "position":
                vbody.append(LetStmt("position", "vec3",
                    SwizzleAccess(VarRef("pos"), "xyz")))
            elif field.name == "normal":
                vbody.append(LetStmt("normal_raw", "vec4",
                    IndexAccess(VarRef("normals"), VarRef("global_idx"))))
                vbody.append(LetStmt("normal", "vec3",
                    SwizzleAccess(VarRef("normal_raw"), "xyz")))
            elif field.name == "uv":
                vbody.append(LetStmt("uv", "vec2",
                    IndexAccess(VarRef("tex_coords"), VarRef("global_idx"))))
            elif field.name == "tangent":
                vbody.append(LetStmt("tangent", "vec4",
                    IndexAccess(VarRef("tangents"), VarRef("global_idx"))))

    # Write geometry outputs — two-pass to handle cross-binding references.
    # E.g. world_bitangent = cross(world_normal, world_tangent) needs the
    # earlier bindings to resolve to scalar locals, not the output arrays.
    if geometry and geometry.outputs:
        temp_map = {}  # binding.name -> local temp name
        for binding in geometry.outputs.bindings:
            if binding.name == "clip_pos":
                rewritten = _rewrite_refs(binding.value, temp_map)
                vbody.append(AssignStmt(
                    AssignTarget(IndexAccess(VarRef("gl_MeshVerticesEXT"), VarRef(vid))),
                    rewritten,
                ))
            else:
                out_type = _infer_output_type(binding.name)
                local = f"_local_{binding.name}"
                rewritten = _rewrite_refs(binding.value, temp_map)
                vbody.append(LetStmt(local, out_type, rewritten))
                vbody.append(AssignStmt(
                    AssignTarget(IndexAccess(VarRef(binding.name), VarRef(vid))),
                    VarRef(local),
                ))
                temp_map[binding.name] = local
    else:
        vbody.append(AssignStmt(
            AssignTarget(IndexAccess(VarRef("gl_MeshVerticesEXT"), VarRef(vid))),
            ConstructorExpr("vec4", [
                SwizzleAccess(VarRef("pos"), "xyz"),
                NumberLit("1.0"),
            ]),
        ))

    return vbody


def _make_tri_iteration(trid: str) -> list:
    """Generate one iteration of triangle index writing for invocation variable *trid*."""
    tbody = []

    tbody.append(LetStmt("idx_base", "scalar",
        BinaryOp("*", VarRef(trid), NumberLit("3"))))

    tbody.append(LetStmt("t0", "uint",
        IndexAccess(VarRef("meshlet_triangles"),
            BinaryOp("+", VarRef("tri_offset"), VarRef("idx_base")))))
    tbody.append(LetStmt("t1", "uint",
        IndexAccess(VarRef("meshlet_triangles"),
            BinaryOp("+", VarRef("tri_offset"),
                BinaryOp("+", VarRef("idx_base"), NumberLit("1"))))))
    tbody.append(LetStmt("t2", "uint",
        IndexAccess(VarRef("meshlet_triangles"),
            BinaryOp("+", VarRef("tri_offset"),
                BinaryOp("+", VarRef("idx_base"), NumberLit("2"))))))

    tbody.append(AssignStmt(
        AssignTarget(IndexAccess(VarRef("gl_PrimitiveTriangleIndicesEXT"), VarRef(trid))),
        ConstructorExpr("uvec3", [VarRef("t0"), VarRef("t1"), VarRef("t2")]),
    ))

    return tbody


def _expand_task_shader(defines: dict[str, int]) -> StageBlock:
    """Generate a basic task (amplification) shader.

    This is a passthrough task shader that dispatches mesh shader workgroups 1:1.
    Future versions can add frustum/occlusion culling here.
    """
    stage = StageBlock(stage_type="task")

    # Task payload to pass meshlet ID to mesh shader
    stage.task_payloads.append(TaskPayloadDecl("task_data", "uint"))

    # Simple passthrough: each task workgroup dispatches one mesh workgroup
    body = []

    # task_data = workgroup_id.x;
    body.append(AssignStmt(
        AssignTarget(VarRef("task_data")),
        SwizzleAccess(VarRef("workgroup_id"), "x"),
    ))

    # emit_mesh_tasks(1, 1, 1);
    body.append(ExprStmt(CallExpr(VarRef("emit_mesh_tasks"), [
        NumberLit("1"), NumberLit("1"), NumberLit("1"),
    ])))

    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


# =========================================================================
# Bindless Descriptor Expansion (uber-shaders)
# =========================================================================

# Feature flag bits for BindlessMaterialData.material_flags
_FLAG_NORMAL_MAP   = 0x1
_FLAG_CLEARCOAT    = 0x2
_FLAG_SHEEN        = 0x4
_FLAG_EMISSION     = 0x8
_FLAG_TRANSMISSION = 0x10
_FLAG_FUZZ         = 0x20
_FLAG_THIN_FILM    = 0x40
_FLAG_SUBSURFACE   = 0x80
_FLAG_ANISOTROPY   = 0x100
_FLAG_OPENPBR      = 0x200

# Texture slot names in declaration order (matches BindlessMaterialData struct)
_BINDLESS_TEX_SLOTS = [
    "base_color_tex",
    "normal_tex",
    "metallic_roughness_tex",
    "occlusion_tex",
    "emissive_tex",
    "clearcoat_tex",
    "clearcoat_roughness_tex",
    "sheen_color_tex",
    "transmission_tex",
]

# Corresponding index field names in the materials SSBO struct
_BINDLESS_TEX_INDEX_FIELDS = [
    "base_color_tex_index",
    "normal_tex_index",
    "metallic_roughness_tex_index",
    "occlusion_tex_index",
    "emissive_tex_index",
    "clearcoat_tex_index",
    "clearcoat_roughness_tex_index",
    "sheen_color_tex_index",
    "transmission_tex_index",
]

# Map surface property names to BindlessMaterialData SSBO field names
_PROPERTY_TO_BINDLESS = {
    "base_color": "baseColorFactor",
    "roughness": "roughnessFactor",
    "metallic": "metallicFactor",
    "emissive": "emissiveFactor",
    "emission_strength": "emissionStrength",
    "ior": "ior",
    "clearcoat": "clearcoatFactor",
    "clearcoat_roughness": "clearcoatRoughness",
    "transmission": "transmissionFactor",
    "sheen_color": "sheenColorFactor",
    "sheen_roughness": "sheenRoughness",
}

# Map OpenPBR surface property names to SSBO field names
_OPENPBR_PROPERTY_TO_BINDLESS = {
    "base_color": "baseColorFactor",       # reuse existing core fields
    "roughness": "roughnessFactor",
    "metallic": "metallicFactor",
    # OpenPBR-specific fields (stored in extended region)
    "base_diffuse_roughness": "baseDiffuseRoughness",
    "base_weight": "baseWeight",
    "specular_weight": "specularWeight",
    "specular_ior": "specularIor",
    "coat_ior": "coatIor",
    "coat_darkening": "coatDarkening",
    "fuzz_weight": "fuzzWeight",
    "fuzz_roughness": "fuzzRoughness",
    "thin_film_weight": "thinFilmWeight",
    "thin_film_thickness": "thinFilmThickness",
    "thin_film_ior": "thinFilmIor",
}

# Extra struct fields appended to BindlessMaterialData when OpenPBR is active.
# Must maintain std430 alignment: vec3 fields are padded to 16 bytes.
_OPENPBR_EXTRA_FIELDS = [
    ("baseDiffuseRoughness", "scalar"),
    ("baseWeight", "scalar"),
    ("specularWeight", "scalar"),
    ("specularIor", "scalar"),
    ("coatIor", "scalar"),
    ("coatDarkening", "scalar"),
    ("fuzzWeight", "scalar"),
    ("fuzzRoughness", "scalar"),
    ("thinFilmWeight", "scalar"),
    ("thinFilmThickness", "scalar"),
    ("thinFilmIor", "scalar"),
    ("_pad_openpbr_0", "scalar"),       # align to 16 bytes
    ("specularColor", "vec3"),
    ("_pad_openpbr_1", "scalar"),
    ("coatColor", "vec3"),
    ("_pad_openpbr_2", "scalar"),
    ("fuzzColor", "vec3"),
    ("_pad_openpbr_3", "scalar"),
    ("transmissionColor", "vec3"),
    ("_pad_openpbr_4", "scalar"),
    ("transmissionDepth", "scalar"),
    ("emissionLuminance", "scalar"),
    ("_pad_openpbr_end_0", "scalar"),
    ("_pad_openpbr_end_1", "scalar"),
]


def _build_bindless_material_struct_fields(extra_properties=None, openpbr=False) -> list[BlockField]:
    """Return the fields for BindlessMaterialData (std430 SSBO layout).

    If extra_properties is provided (list of (name, type) tuples for properties
    NOT in _PROPERTY_TO_BINDLESS), they are appended after the core 128-byte struct
    with proper std430 alignment padding.

    If openpbr is True, the OpenPBR extended fields (_OPENPBR_EXTRA_FIELDS) are
    appended after the core struct (before any extra_properties).
    """
    fields = [
        # PBR factors
        BlockField("baseColorFactor", "vec4"),
        BlockField("emissiveFactor", "vec3"),
        BlockField("metallicFactor", "scalar"),
        BlockField("roughnessFactor", "scalar"),
        BlockField("emissionStrength", "scalar"),
        BlockField("ior", "scalar"),
        BlockField("clearcoatFactor", "scalar"),
        BlockField("clearcoatRoughness", "scalar"),
        BlockField("transmissionFactor", "scalar"),
        BlockField("sheenRoughness", "scalar"),
        BlockField("_pad0", "scalar"),  # padding to align sheenColorFactor
        BlockField("sheenColorFactor", "vec3"),
        BlockField("_pad1", "scalar"),  # padding after vec3
        # Texture indices (int, -1 = no texture)
        BlockField("base_color_tex_index", "int"),
        BlockField("normal_tex_index", "int"),
        BlockField("metallic_roughness_tex_index", "int"),
        BlockField("occlusion_tex_index", "int"),
        BlockField("emissive_tex_index", "int"),
        BlockField("clearcoat_tex_index", "int"),
        BlockField("clearcoat_roughness_tex_index", "int"),
        BlockField("sheen_color_tex_index", "int"),
        BlockField("transmission_tex_index", "int"),
        # Feature flags
        BlockField("material_flags", "uint"),
        # Per-geometry index offset for RT barycentric interpolation (float for type compat)
        BlockField("index_offset", "scalar"),
        BlockField("_pad3", "uint"),
    ]

    # Append OpenPBR extended fields when active
    if openpbr:
        for fname, ftype in _OPENPBR_EXTRA_FIELDS:
            fields.append(BlockField(fname, ftype))

    if extra_properties:
        # Add padding if needed for alignment after the core struct
        # Core struct ends at 128 bytes (32 x 4-byte fields)
        for name, type_name in extra_properties:
            # vec3 fields need 16-byte alignment in std430
            if type_name == "vec3":
                fields.append(BlockField(f"_pad_{name}", "scalar"))
            fields.append(BlockField(name, type_name))
        # Pad to 16-byte alignment
        # Count total field slots to check alignment
        total_slots = len(fields)
        if total_slots % 4 != 0:
            for i in range(4 - (total_slots % 4)):
                fields.append(BlockField(f"_pad_end_{i}", "scalar"))

    return fields


def _emit_bindless_layer_body(
    surface: SurfaceDecl,
    mat_idx_expr,
    schedule: dict[str, str] | None = None,
    is_rt: bool = False,
    lighting: LightingDecl | None = None,
    pos_var: str = "world_pos",
    properties=None,
) -> tuple[list, str]:
    """Generate shared uber-shader body for bindless mode.

    Works for both raster/mesh fragment and RT closest-hit.
    Uses mat_idx_expr to index into materials[] SSBO and
    sample_bindless/sample_bindless_lod for texture array access.

    Returns (body_stmts, output_var_name).
    """
    body = []

    # Determine type from the source expression
    # geometry_index is int, push constant material_index is uint
    mat_idx_type = "int" if is_rt else "uint"
    body.append(LetStmt("mat_idx", mat_idx_type, mat_idx_expr))

    # --- Load PBR factors from materials SSBO ---
    def _mat_field(local_name, field_name, lux_type):
        """Generate: let <local_name>: <type> = materials[mat_idx].<field_name>;"""
        return LetStmt(local_name, lux_type,
            FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")), field_name))

    body.append(_mat_field("mat_baseColor", "baseColorFactor", "vec4"))
    body.append(_mat_field("mat_emissiveFactor", "emissiveFactor", "vec3"))
    body.append(_mat_field("mat_metallic", "metallicFactor", "scalar"))
    body.append(_mat_field("mat_roughness", "roughnessFactor", "scalar"))
    body.append(_mat_field("mat_emissionStrength", "emissionStrength", "scalar"))
    body.append(_mat_field("mat_flags", "material_flags", "uint"))

    # --- Load custom properties from SSBO ---
    if properties and hasattr(properties, 'fields') and properties.fields:
        for prop in properties.fields:
            prop_name = prop.name
            ssbo_field = _PROPERTY_TO_BINDLESS.get(prop_name)
            if ssbo_field:
                # Property maps to existing SSBO field — already loaded above
                continue
            # Custom property — load from extended SSBO
            body.append(_mat_field(prop_name, prop_name, prop.type_name))

    # --- Texture index loads ---
    body.append(_mat_field("tex_baseColor", "base_color_tex_index", "int"))
    body.append(_mat_field("tex_normal", "normal_tex_index", "int"))
    body.append(_mat_field("tex_mr", "metallic_roughness_tex_index", "int"))
    body.append(_mat_field("tex_occlusion", "occlusion_tex_index", "int"))
    body.append(_mat_field("tex_emissive", "emissive_tex_index", "int"))
    body.append(_mat_field("tex_clearcoat", "clearcoat_tex_index", "int"))
    body.append(_mat_field("tex_clearcoatRough", "clearcoat_roughness_tex_index", "int"))
    body.append(_mat_field("tex_sheenColor", "sheen_color_tex_index", "int"))
    body.append(_mat_field("tex_transmission", "transmission_tex_index", "int"))

    # Use the appropriate sample function
    sample_fn = "sample_bindless_lod" if is_rt else "sample_bindless"

    def _sample_tex(tex_idx_var, result_var, result_type="vec4"):
        """Generate: if (tex_idx >= 0) result = sample_bindless(textures, tex_idx, uv);"""
        if is_rt:
            sample_call = CallExpr(VarRef(sample_fn), [
                VarRef("textures"), VarRef(tex_idx_var), VarRef("uv"), NumberLit("0.0")])
        else:
            sample_call = CallExpr(VarRef(sample_fn), [
                VarRef("textures"), VarRef(tex_idx_var), VarRef("uv")])
        sample_call.resolved_type = "vec4"
        return IfStmt(
            BinaryOp(">=", VarRef(tex_idx_var), NumberLit("0")),
            [AssignStmt(AssignTarget(VarRef(result_var)), sample_call)],
            [],
        )

    # --- Base color: srgb_to_linear(texture) * factor (per glTF spec) ---
    body.append(LetStmt("albedo", "vec4", VarRef("mat_baseColor")))
    body.append(LetStmt("bc_tex_sample", "vec4",
        ConstructorExpr("vec4", [NumberLit("1.0")])))
    body.append(_sample_tex("tex_baseColor", "bc_tex_sample"))
    body.append(IfStmt(
        BinaryOp(">=", VarRef("tex_baseColor"), NumberLit("0")),
        [
            # glTF base color textures are sRGB-encoded; convert to linear before PBR math
            LetStmt("bc_linear", "vec3",
                CallExpr(VarRef("srgb_to_linear"), [
                    SwizzleAccess(VarRef("bc_tex_sample"), "xyz")])),
            AssignStmt(AssignTarget(VarRef("albedo")),
                ConstructorExpr("vec4", [
                    BinaryOp("*", SwizzleAccess(VarRef("mat_baseColor"), "xyz"),
                        VarRef("bc_linear")),
                    SwizzleAccess(VarRef("mat_baseColor"), "w"),
                ])),
        ],
        [],
    ))
    body.append(LetStmt("layer_albedo", "vec3", SwizzleAccess(VarRef("albedo"), "xyz")))

    # --- Metallic/Roughness: factor * texture (B=metallic, G=roughness) ---
    body.append(LetStmt("layer_roughness", "scalar", VarRef("mat_roughness")))
    body.append(LetStmt("layer_metallic", "scalar", VarRef("mat_metallic")))
    body.append(LetStmt("mr_sample", "vec4",
        ConstructorExpr("vec4", [NumberLit("1.0")])))
    body.append(_sample_tex("tex_mr", "mr_sample"))
    body.append(IfStmt(
        BinaryOp(">=", VarRef("tex_mr"), NumberLit("0")),
        [
            AssignStmt(AssignTarget(VarRef("layer_roughness")),
                BinaryOp("*", VarRef("mat_roughness"),
                    SwizzleAccess(VarRef("mr_sample"), "y"))),
            AssignStmt(AssignTarget(VarRef("layer_metallic")),
                BinaryOp("*", VarRef("mat_metallic"),
                    SwizzleAccess(VarRef("mr_sample"), "z"))),
        ],
        [],
    ))

    # --- Direct lighting ---
    _bl_has_multi_light = _detect_multi_light(lighting)

    if _bl_has_multi_light:
        result_var = _emit_multi_light_from_lighting(
            body, lighting, pos_var)
    else:
        body.append(LetStmt("direct", "vec3", CallExpr(VarRef("gltf_pbr"), [
            VarRef("n"), VarRef("v"), VarRef("l"),
            VarRef("layer_albedo"), VarRef("layer_roughness"),
            VarRef("layer_metallic"),
        ])))
        body.append(LetStmt("direct_lit", "vec3", BinaryOp("*",
            VarRef("direct"),
            ConstructorExpr("vec3", [
                NumberLit("1.0"), NumberLit("0.98"), NumberLit("0.95"),
            ]),
        )))
        result_var = "direct_lit"

    # --- Initialize optional layer params (defaults = layer disabled) ---
    body.append(LetStmt("bl_trans_factor", "scalar", NumberLit("0.0")))
    body.append(LetStmt("bl_trans_ior", "scalar", NumberLit("1.5")))
    body.append(LetStmt("bl_sheen_color", "vec3",
        ConstructorExpr("vec3", [NumberLit("0.0")])))
    body.append(LetStmt("bl_sheen_roughness", "scalar", NumberLit("0.0")))
    body.append(LetStmt("bl_coat_factor", "scalar", NumberLit("0.0")))
    body.append(LetStmt("bl_coat_roughness", "scalar", NumberLit("0.0")))
    body.append(LetStmt("bl_emission", "vec3",
        ConstructorExpr("vec3", [NumberLit("0.0")])))

    # --- Transmission param loading (flag-gated) ---
    body.append(IfStmt(
        BinaryOp("!=",
            BinaryOp("&", VarRef("mat_flags"),
                NumberLit(str(_FLAG_TRANSMISSION))),
            NumberLit("0")),
        [
            AssignStmt(AssignTarget(VarRef("bl_trans_factor")),
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "transmissionFactor")),
            AssignStmt(AssignTarget(VarRef("bl_trans_ior")),
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "ior")),
        ],
        [],
    ))

    # --- Sheen param loading (flag-gated) ---
    body.append(IfStmt(
        BinaryOp("!=",
            BinaryOp("&", VarRef("mat_flags"), NumberLit(str(_FLAG_SHEEN))),
            NumberLit("0")),
        [
            LetStmt("sheen_color_loaded", "vec3",
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "sheenColorFactor")),
            LetStmt("sheen_tex_sample", "vec4",
                ConstructorExpr("vec4", [NumberLit("1.0")])),
            _sample_tex("tex_sheenColor", "sheen_tex_sample"),
            # glTF sheen color textures are sRGB-encoded
            AssignStmt(AssignTarget(VarRef("bl_sheen_color")),
                BinaryOp("*", VarRef("sheen_color_loaded"),
                    CallExpr(VarRef("srgb_to_linear"), [
                        SwizzleAccess(VarRef("sheen_tex_sample"), "xyz")]))),
            AssignStmt(AssignTarget(VarRef("bl_sheen_roughness")),
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "sheenRoughness")),
        ],
        [],
    ))

    # --- Clearcoat param loading (flag-gated) ---
    body.append(IfStmt(
        BinaryOp("!=",
            BinaryOp("&", VarRef("mat_flags"), NumberLit(str(_FLAG_CLEARCOAT))),
            NumberLit("0")),
        [
            LetStmt("coat_factor_loaded", "scalar",
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "clearcoatFactor")),
            LetStmt("coat_roughness_loaded", "scalar",
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "clearcoatRoughness")),
            LetStmt("coat_tex_sample", "vec4",
                ConstructorExpr("vec4", [NumberLit("1.0")])),
            _sample_tex("tex_clearcoat", "coat_tex_sample"),
            LetStmt("coat_rough_sample", "vec4",
                ConstructorExpr("vec4", [NumberLit("1.0")])),
            _sample_tex("tex_clearcoatRough", "coat_rough_sample"),
            AssignStmt(AssignTarget(VarRef("bl_coat_factor")),
                BinaryOp("*", VarRef("coat_factor_loaded"),
                    SwizzleAccess(VarRef("coat_tex_sample"), "x"))),
            AssignStmt(AssignTarget(VarRef("bl_coat_roughness")),
                BinaryOp("*", VarRef("coat_roughness_loaded"),
                    SwizzleAccess(VarRef("coat_rough_sample"), "y"))),
        ],
        [],
    ))

    # --- Emission param loading (flag-gated) ---
    body.append(IfStmt(
        BinaryOp("!=",
            BinaryOp("&", VarRef("mat_flags"), NumberLit(str(_FLAG_EMISSION))),
            NumberLit("0")),
        [
            LetStmt("em_color_loaded", "vec3", BinaryOp("*",
                VarRef("mat_emissiveFactor"), VarRef("mat_emissionStrength"))),
            LetStmt("em_tex_sample", "vec4",
                ConstructorExpr("vec4", [NumberLit("1.0")])),
            _sample_tex("tex_emissive", "em_tex_sample"),
            # glTF emissive textures are sRGB-encoded
            AssignStmt(AssignTarget(VarRef("bl_emission")),
                BinaryOp("*", VarRef("em_color_loaded"),
                    CallExpr(VarRef("srgb_to_linear"), [
                        SwizzleAccess(VarRef("em_tex_sample"), "xyz")]))),
        ],
        [],
    ))

    # --- IBL sampling ---
    body.append(LetStmt("r", "vec3", CallExpr(VarRef("reflect"), [
        BinaryOp("*", VarRef("v"), UnaryOp("-", NumberLit("1.0"))),
        VarRef("n"),
    ])))

    if is_rt:
        body.append(LetStmt("prefiltered", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                VarRef("env_specular"), VarRef("r"),
                BinaryOp("*", VarRef("layer_roughness"), NumberLit("8.0")),
            ]), "xyz")))
        body.append(LetStmt("irradiance", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                VarRef("env_irradiance"), VarRef("n"), NumberLit("0.0"),
            ]), "xyz")))
        body.append(LetStmt("brdf_sample", "vec2", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                VarRef("brdf_lut"),
                ConstructorExpr("vec2", [VarRef("n_dot_v"), VarRef("layer_roughness")]),
                NumberLit("0.0"),
            ]), "xy")))
    else:
        body.append(LetStmt("prefiltered", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                VarRef("env_specular"), VarRef("r"),
                BinaryOp("*", VarRef("layer_roughness"), NumberLit("8.0")),
            ]), "xyz")))
        body.append(LetStmt("irradiance", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                VarRef("env_irradiance"), VarRef("n"), NumberLit("0.0"),
            ]), "xyz")))
        body.append(LetStmt("brdf_sample", "vec2", SwizzleAccess(
            CallExpr(VarRef("sample"), [
                VarRef("brdf_lut"),
                ConstructorExpr("vec2", [VarRef("n_dot_v"), VarRef("layer_roughness")]),
            ]), "xy")))

    body.append(LetStmt("ambient", "vec3", CallExpr(
        VarRef("ibl_contribution"), [
            VarRef("layer_albedo"), VarRef("layer_roughness"),
            VarRef("layer_metallic"),
            VarRef("n_dot_v"), VarRef("prefiltered"), VarRef("irradiance"),
            VarRef("brdf_sample"),
        ])))

    # --- Coat IBL (flag-gated: only when clearcoat is active) ---
    body.append(LetStmt("bl_coat_ibl", "vec3",
        ConstructorExpr("vec3", [NumberLit("0.0")])))
    coat_ibl_body = []
    if is_rt:
        coat_ibl_body.append(LetStmt("prefiltered_coat", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                VarRef("env_specular"), VarRef("r"),
                BinaryOp("*", VarRef("bl_coat_roughness"), NumberLit("8.0")),
            ]), "xyz")))
    else:
        coat_ibl_body.append(LetStmt("prefiltered_coat", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                VarRef("env_specular"), VarRef("r"),
                BinaryOp("*", VarRef("bl_coat_roughness"), NumberLit("8.0")),
            ]), "xyz")))
    coat_ibl_body.append(AssignStmt(AssignTarget(VarRef("bl_coat_ibl")),
        CallExpr(VarRef("coat_ibl"), [
            VarRef("bl_coat_factor"), VarRef("bl_coat_roughness"),
            VarRef("n"), VarRef("v"), VarRef("prefiltered_coat"),
        ])))
    body.append(IfStmt(
        BinaryOp("!=",
            BinaryOp("&", VarRef("mat_flags"), NumberLit(str(_FLAG_CLEARCOAT))),
            NumberLit("0")),
        coat_ibl_body, []))

    # --- Unified composition via compose_pbr_layers (stdlib) ---
    body.append(LetStmt("composed", "vec3", CallExpr(
        VarRef("compose_pbr_layers"), [
            VarRef(result_var),
            VarRef("n"), VarRef("v"), VarRef("l"),
            VarRef("layer_albedo"), VarRef("layer_roughness"),
            # Transmission
            VarRef("bl_trans_factor"), VarRef("bl_trans_ior"),
            NumberLit("0.0"),
            ConstructorExpr("vec3", [NumberLit("1.0")]),
            NumberLit("1000000.0"),
            # IBL
            VarRef("ambient"),
            # Sheen
            VarRef("bl_sheen_color"), VarRef("bl_sheen_roughness"),
            # Coat
            VarRef("bl_coat_factor"), VarRef("bl_coat_roughness"),
            # Coat IBL (now properly computed)
            VarRef("bl_coat_ibl"),
            # Emission
            VarRef("bl_emission"),
        ])))
    result_var = "composed"

    # --- Tonemap + output ---
    output_var = _emit_tonemap_output(body, result_var, schedule)

    return body, output_var


def _emit_openpbr_bindless_body(
    surface: SurfaceDecl,
    mat_idx_expr,
    schedule: dict[str, str] | None = None,
    is_rt: bool = False,
    lighting: LightingDecl | None = None,
    pos_var: str = "world_pos",
) -> tuple[list, str]:
    """Generate shared OpenPBR uber-shader body for bindless mode.

    Mirrors _emit_bindless_layer_body() but loads OpenPBR-specific fields
    from the extended SSBO region and calls openpbr_direct() / openpbr_compose()
    instead of gltf_pbr() / compose_pbr_layers().

    Returns (body_stmts, output_var_name).
    """
    body = []

    # Determine type from the source expression
    mat_idx_type = "int" if is_rt else "uint"
    body.append(LetStmt("mat_idx", mat_idx_type, mat_idx_expr))

    # --- Load core PBR factors from materials SSBO ---
    def _mat_field(local_name, field_name, lux_type):
        """Generate: let <local_name>: <type> = materials[mat_idx].<field_name>;"""
        return LetStmt(local_name, lux_type,
            FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")), field_name))

    body.append(_mat_field("mat_baseColor", "baseColorFactor", "vec4"))
    body.append(_mat_field("mat_emissiveFactor", "emissiveFactor", "vec3"))
    body.append(_mat_field("mat_metallic", "metallicFactor", "scalar"))
    body.append(_mat_field("mat_roughness", "roughnessFactor", "scalar"))
    body.append(_mat_field("mat_flags", "material_flags", "uint"))

    # --- Load OpenPBR-specific fields from extended SSBO region ---
    body.append(_mat_field("base_diffuse_roughness", "baseDiffuseRoughness", "scalar"))
    body.append(_mat_field("base_weight", "baseWeight", "scalar"))
    body.append(_mat_field("specular_weight", "specularWeight", "scalar"))
    body.append(_mat_field("specular_ior", "specularIor", "scalar"))
    body.append(_mat_field("specular_color", "specularColor", "vec3"))
    body.append(_mat_field("coat_ior", "coatIor", "scalar"))
    body.append(_mat_field("coat_darkening", "coatDarkening", "scalar"))
    body.append(_mat_field("coat_color", "coatColor", "vec3"))
    body.append(_mat_field("trans_color", "transmissionColor", "vec3"))
    body.append(_mat_field("trans_depth", "transmissionDepth", "scalar"))
    body.append(_mat_field("emission_luminance", "emissionLuminance", "scalar"))

    # --- Texture index loads ---
    body.append(_mat_field("tex_baseColor", "base_color_tex_index", "int"))
    body.append(_mat_field("tex_normal", "normal_tex_index", "int"))
    body.append(_mat_field("tex_mr", "metallic_roughness_tex_index", "int"))
    body.append(_mat_field("tex_occlusion", "occlusion_tex_index", "int"))
    body.append(_mat_field("tex_emissive", "emissive_tex_index", "int"))
    body.append(_mat_field("tex_clearcoat", "clearcoat_tex_index", "int"))
    body.append(_mat_field("tex_clearcoatRough", "clearcoat_roughness_tex_index", "int"))
    body.append(_mat_field("tex_sheenColor", "sheen_color_tex_index", "int"))
    body.append(_mat_field("tex_transmission", "transmission_tex_index", "int"))

    # Use the appropriate sample function
    sample_fn = "sample_bindless_lod" if is_rt else "sample_bindless"

    def _sample_tex(tex_idx_var, result_var, result_type="vec4"):
        """Generate: if (tex_idx >= 0) result = sample_bindless(textures, tex_idx, uv);"""
        if is_rt:
            sample_call = CallExpr(VarRef(sample_fn), [
                VarRef("textures"), VarRef(tex_idx_var), VarRef("uv"), NumberLit("0.0")])
        else:
            sample_call = CallExpr(VarRef(sample_fn), [
                VarRef("textures"), VarRef(tex_idx_var), VarRef("uv")])
        sample_call.resolved_type = "vec4"
        return IfStmt(
            BinaryOp(">=", VarRef(tex_idx_var), NumberLit("0")),
            [AssignStmt(AssignTarget(VarRef(result_var)), sample_call)],
            [],
        )

    # --- Base color: srgb_to_linear(texture) * factor ---
    body.append(LetStmt("base_color", "vec3", SwizzleAccess(VarRef("mat_baseColor"), "xyz")))
    body.append(LetStmt("bc_tex_sample", "vec4",
        ConstructorExpr("vec4", [NumberLit("1.0")])))
    body.append(_sample_tex("tex_baseColor", "bc_tex_sample"))
    body.append(IfStmt(
        BinaryOp(">=", VarRef("tex_baseColor"), NumberLit("0")),
        [
            LetStmt("bc_linear", "vec3",
                CallExpr(VarRef("srgb_to_linear"), [
                    SwizzleAccess(VarRef("bc_tex_sample"), "xyz")])),
            AssignStmt(AssignTarget(VarRef("base_color")),
                BinaryOp("*", SwizzleAccess(VarRef("mat_baseColor"), "xyz"),
                    VarRef("bc_linear"))),
        ],
        [],
    ))

    # --- Metallic/Roughness: factor * texture (B=metallic, G=roughness) ---
    body.append(LetStmt("specular_roughness", "scalar", VarRef("mat_roughness")))
    body.append(LetStmt("base_metalness", "scalar", VarRef("mat_metallic")))
    body.append(LetStmt("mr_sample", "vec4",
        ConstructorExpr("vec4", [NumberLit("1.0")])))
    body.append(_sample_tex("tex_mr", "mr_sample"))
    body.append(IfStmt(
        BinaryOp(">=", VarRef("tex_mr"), NumberLit("0")),
        [
            AssignStmt(AssignTarget(VarRef("specular_roughness")),
                BinaryOp("*", VarRef("mat_roughness"),
                    SwizzleAccess(VarRef("mr_sample"), "y"))),
            AssignStmt(AssignTarget(VarRef("base_metalness")),
                BinaryOp("*", VarRef("mat_metallic"),
                    SwizzleAccess(VarRef("mr_sample"), "z"))),
        ],
        [],
    ))

    # --- Expose aliases for multi-light loop compatibility ---
    body.append(LetStmt("layer_albedo", "vec3", VarRef("base_color")))
    body.append(LetStmt("layer_roughness", "scalar", VarRef("specular_roughness")))
    body.append(LetStmt("layer_metallic", "scalar", VarRef("base_metalness")))

    # --- Direct lighting ---
    _bl_has_multi_light = _detect_multi_light(lighting)

    if _bl_has_multi_light:
        result_var = _emit_multi_light_from_lighting(
            body, lighting, pos_var)
    else:
        body.append(LetStmt("direct_raw", "vec3", CallExpr(VarRef("openpbr_direct"), [
            VarRef("n"), VarRef("v"), VarRef("l"),
            VarRef("base_color"), VarRef("base_metalness"),
            VarRef("base_diffuse_roughness"), VarRef("base_weight"),
            VarRef("specular_weight"), VarRef("specular_color"),
            VarRef("specular_roughness"), VarRef("specular_ior"),
            # Coat params (loaded conditionally below, defaults safe for non-coat)
            NumberLit("0.0"), VarRef("specular_roughness"), NumberLit("1.6"),
            # Thin film params (defaults)
            NumberLit("0.0"), NumberLit("0.5"), NumberLit("1.4"),
        ])))
        body.append(LetStmt("direct_lit", "vec3", BinaryOp("*",
            VarRef("direct_raw"),
            ConstructorExpr("vec3", [
                NumberLit("1.0"), NumberLit("0.98"), NumberLit("0.95"),
            ]),
        )))
        result_var = "direct_lit"

    # --- Initialize optional OpenPBR layer params (defaults = layer disabled) ---
    body.append(LetStmt("bl_coat_weight", "scalar", NumberLit("0.0")))
    body.append(LetStmt("bl_coat_color", "vec3",
        ConstructorExpr("vec3", [NumberLit("1.0")])))
    body.append(LetStmt("bl_coat_roughness", "scalar", NumberLit("0.0")))
    body.append(LetStmt("bl_coat_ior", "scalar", NumberLit("1.6")))
    body.append(LetStmt("bl_coat_darkening", "scalar", NumberLit("1.0")))
    body.append(LetStmt("bl_trans_weight", "scalar", NumberLit("0.0")))
    body.append(LetStmt("bl_trans_color", "vec3",
        ConstructorExpr("vec3", [NumberLit("1.0")])))
    body.append(LetStmt("bl_trans_depth", "scalar", NumberLit("0.0")))
    body.append(LetStmt("bl_fuzz_weight", "scalar", NumberLit("0.0")))
    body.append(LetStmt("bl_fuzz_color", "vec3",
        ConstructorExpr("vec3", [NumberLit("1.0")])))
    body.append(LetStmt("bl_fuzz_roughness", "scalar", NumberLit("0.5")))
    body.append(LetStmt("bl_thin_film_weight", "scalar", NumberLit("0.0")))
    body.append(LetStmt("bl_thin_film_thickness", "scalar", NumberLit("0.5")))
    body.append(LetStmt("bl_thin_film_ior", "scalar", NumberLit("1.4")))
    body.append(LetStmt("bl_emission_luminance", "scalar", NumberLit("0.0")))
    body.append(LetStmt("bl_emission_color", "vec3",
        ConstructorExpr("vec3", [NumberLit("1.0")])))

    # --- Coat param loading (flag-gated) ---
    body.append(IfStmt(
        BinaryOp("!=",
            BinaryOp("&", VarRef("mat_flags"), NumberLit(str(_FLAG_CLEARCOAT))),
            NumberLit("0")),
        [
            AssignStmt(AssignTarget(VarRef("bl_coat_weight")),
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "clearcoatFactor")),
            AssignStmt(AssignTarget(VarRef("bl_coat_roughness")),
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "clearcoatRoughness")),
            AssignStmt(AssignTarget(VarRef("bl_coat_ior")), VarRef("coat_ior")),
            AssignStmt(AssignTarget(VarRef("bl_coat_darkening")), VarRef("coat_darkening")),
            AssignStmt(AssignTarget(VarRef("bl_coat_color")), VarRef("coat_color")),
        ],
        [],
    ))

    # --- Transmission param loading (flag-gated) ---
    body.append(IfStmt(
        BinaryOp("!=",
            BinaryOp("&", VarRef("mat_flags"), NumberLit(str(_FLAG_TRANSMISSION))),
            NumberLit("0")),
        [
            AssignStmt(AssignTarget(VarRef("bl_trans_weight")),
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "transmissionFactor")),
            AssignStmt(AssignTarget(VarRef("bl_trans_color")), VarRef("trans_color")),
            AssignStmt(AssignTarget(VarRef("bl_trans_depth")), VarRef("trans_depth")),
        ],
        [],
    ))

    # --- Fuzz param loading (flag-gated) ---
    body.append(IfStmt(
        BinaryOp("!=",
            BinaryOp("&", VarRef("mat_flags"), NumberLit(str(_FLAG_FUZZ))),
            NumberLit("0")),
        [
            AssignStmt(AssignTarget(VarRef("bl_fuzz_weight")),
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "fuzzWeight")),
            AssignStmt(AssignTarget(VarRef("bl_fuzz_roughness")),
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "fuzzRoughness")),
            AssignStmt(AssignTarget(VarRef("bl_fuzz_color")),
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "fuzzColor")),
        ],
        [],
    ))

    # --- Thin film param loading (flag-gated) ---
    body.append(IfStmt(
        BinaryOp("!=",
            BinaryOp("&", VarRef("mat_flags"), NumberLit(str(_FLAG_THIN_FILM))),
            NumberLit("0")),
        [
            AssignStmt(AssignTarget(VarRef("bl_thin_film_weight")),
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "thinFilmWeight")),
            AssignStmt(AssignTarget(VarRef("bl_thin_film_thickness")),
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "thinFilmThickness")),
            AssignStmt(AssignTarget(VarRef("bl_thin_film_ior")),
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "thinFilmIor")),
        ],
        [],
    ))

    # --- Emission param loading (flag-gated) ---
    body.append(IfStmt(
        BinaryOp("!=",
            BinaryOp("&", VarRef("mat_flags"), NumberLit(str(_FLAG_EMISSION))),
            NumberLit("0")),
        [
            AssignStmt(AssignTarget(VarRef("bl_emission_luminance")),
                VarRef("emission_luminance")),
            AssignStmt(AssignTarget(VarRef("bl_emission_color")),
                VarRef("mat_emissiveFactor")),
        ],
        [],
    ))

    # --- IBL sampling ---
    body.append(LetStmt("r", "vec3", CallExpr(VarRef("reflect"), [
        BinaryOp("*", VarRef("v"), UnaryOp("-", NumberLit("1.0"))),
        VarRef("n"),
    ])))

    if is_rt:
        body.append(LetStmt("prefiltered", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                VarRef("env_specular"), VarRef("r"),
                BinaryOp("*", VarRef("specular_roughness"), NumberLit("8.0")),
            ]), "xyz")))
        body.append(LetStmt("irradiance", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                VarRef("env_irradiance"), VarRef("n"), NumberLit("0.0"),
            ]), "xyz")))
        body.append(LetStmt("brdf_sample", "vec2", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                VarRef("brdf_lut"),
                ConstructorExpr("vec2", [VarRef("n_dot_v"), VarRef("specular_roughness")]),
                NumberLit("0.0"),
            ]), "xy")))
    else:
        body.append(LetStmt("prefiltered", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                VarRef("env_specular"), VarRef("r"),
                BinaryOp("*", VarRef("specular_roughness"), NumberLit("8.0")),
            ]), "xyz")))
        body.append(LetStmt("irradiance", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                VarRef("env_irradiance"), VarRef("n"), NumberLit("0.0"),
            ]), "xyz")))
        body.append(LetStmt("brdf_sample", "vec2", SwizzleAccess(
            CallExpr(VarRef("sample"), [
                VarRef("brdf_lut"),
                ConstructorExpr("vec2", [VarRef("n_dot_v"), VarRef("specular_roughness")]),
            ]), "xy")))

    body.append(LetStmt("bl_ambient", "vec3", CallExpr(
        VarRef("ibl_contribution"), [
            VarRef("base_color"), VarRef("specular_roughness"),
            VarRef("base_metalness"),
            VarRef("n_dot_v"), VarRef("prefiltered"), VarRef("irradiance"),
            VarRef("brdf_sample"),
        ])))

    # --- Coat IBL (flag-gated) ---
    body.append(LetStmt("bl_coat_ibl", "vec3",
        ConstructorExpr("vec3", [NumberLit("0.0")])))
    coat_ibl_body = []
    if is_rt:
        coat_ibl_body.append(LetStmt("prefiltered_coat", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                VarRef("env_specular"), VarRef("r"),
                BinaryOp("*", VarRef("bl_coat_roughness"), NumberLit("8.0")),
            ]), "xyz")))
    else:
        coat_ibl_body.append(LetStmt("prefiltered_coat", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                VarRef("env_specular"), VarRef("r"),
                BinaryOp("*", VarRef("bl_coat_roughness"), NumberLit("8.0")),
            ]), "xyz")))
    coat_ibl_body.append(AssignStmt(AssignTarget(VarRef("bl_coat_ibl")),
        CallExpr(VarRef("coat_ibl"), [
            VarRef("bl_coat_weight"), VarRef("bl_coat_roughness"),
            VarRef("n"), VarRef("v"), VarRef("prefiltered_coat"),
        ])))
    body.append(IfStmt(
        BinaryOp("!=",
            BinaryOp("&", VarRef("mat_flags"), NumberLit(str(_FLAG_CLEARCOAT))),
            NumberLit("0")),
        coat_ibl_body, []))

    # --- Unified composition via openpbr_compose ---
    body.append(LetStmt("composed", "vec3", CallExpr(
        VarRef("openpbr_compose"), [
            VarRef(result_var),
            VarRef("n"), VarRef("v"), VarRef("l"),
            VarRef("base_color"), VarRef("base_metalness"), VarRef("base_weight"),
            VarRef("specular_weight"), VarRef("specular_color"),
            VarRef("specular_roughness"), VarRef("specular_ior"),
            # Transmission
            VarRef("bl_trans_weight"), VarRef("bl_trans_color"), VarRef("bl_trans_depth"),
            # Coat
            VarRef("bl_coat_weight"), VarRef("bl_coat_color"),
            VarRef("bl_coat_roughness"), VarRef("bl_coat_ior"), VarRef("bl_coat_darkening"),
            # Fuzz
            VarRef("bl_fuzz_weight"), VarRef("bl_fuzz_color"), VarRef("bl_fuzz_roughness"),
            # Thin film
            VarRef("bl_thin_film_weight"), VarRef("bl_thin_film_thickness"),
            VarRef("bl_thin_film_ior"),
            # Emission
            VarRef("bl_emission_luminance"), VarRef("bl_emission_color"),
            # IBL
            VarRef("bl_ambient"), VarRef("bl_coat_ibl"),
        ])))
    result_var = "composed"

    # --- Tonemap + output ---
    output_var = _emit_tonemap_output(body, result_var, schedule)

    return body, output_var


def _expand_bindless_fragment(
    surface: SurfaceDecl,
    module: Module,
    geometry: GeometryDecl | None = None,
    schedule: dict[str, str] | None = None,
    lighting: LightingDecl | None = None,
) -> StageBlock:
    """Generate a bindless uber-shader fragment stage.

    Uses materials SSBO + bindless texture array instead of per-material
    descriptor sets. Material index comes from push constant.
    """
    stage = StageBlock(stage_type="fragment")

    # Standard fragment inputs (from geometry or defaults)
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

    # Ensure frag_uv input exists
    has_frag_uv = any(name == "frag_uv" for name, _ in frag_inputs)
    if not has_frag_uv:
        frag_inputs.append(("frag_uv", "vec2"))
        v = VarDecl("frag_uv", "vec2")
        v._is_input = True
        stage.inputs.append(v)

    # Fragment output
    out = VarDecl("color", "vec4")
    out._is_input = False
    stage.outputs.append(out)

    # --- Uniforms: Light (from lighting block or legacy hardcoded) ---
    if lighting and lighting.properties:
        lprops = lighting.properties
        stage.uniforms.append(UniformBlock(lprops.name, [
            BlockField(f.name, f.type_name) for f in lprops.fields
        ]))
        if not hasattr(stage, '_properties_defaults'):
            stage._properties_defaults = {}
        for f in lprops.fields:
            if f.default is not None:
                stage._properties_defaults[(lprops.name, f.name)] = f.default
    else:
        stage.uniforms.append(UniformBlock("Light", [
            BlockField("light_dir", "vec3"),
            BlockField("view_pos", "vec3"),
        ]))

    # --- Materials SSBO (replaces per-material UBO) ---
    stage.storage_buffers.append(StorageBufferDecl(
        "materials", "BindlessMaterialData"))

    # If multi_light layer present, add lights SSBO (and shadow_matrices if shadow_map arg)
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

    # --- IBL samplers (kept as regular samplers — scene-global, not per-material) ---
    # Check lighting block first, then fall back to surface samplers
    if lighting:
        for sam in lighting.samplers:
            stage.samplers.append(SamplerDecl(sam.name, type_name=sam.type_name))
    else:
        for sam in surface.samplers:
            if sam.type_name == "samplerCube" or sam.name in ("brdf_lut",):
                stage.samplers.append(SamplerDecl(sam.name, type_name=sam.type_name))

    # --- Bindless texture array ---
    stage.bindless_texture_arrays.append(BindlessTextureArrayDecl("textures"))

    # --- Push constants: model + material_index ---
    stage.push_constants.append(PushBlock("PushConstants", [
        BlockField("model", "mat4"),
        BlockField("material_index", "uint"),
    ]))

    # --- Generate main body ---
    body = []

    # UV alias
    body.append(LetStmt("uv", "vec2", VarRef("frag_uv")))

    # Determine normal/position variables
    has_pos = any(n in ("world_pos", "frag_pos") for n, _ in frag_inputs)
    pos_var = next(
        (n for n, _ in frag_inputs if n in ("world_pos", "frag_pos")),
        "world_pos",
    )
    normal_var = next(
        (n for n, _ in frag_inputs if n in ("frag_normal", "world_normal")),
        "frag_normal",
    )

    # Detect multi_light mode for bindless raster
    _bl_raster_has_multi_light = _detect_multi_light(lighting)

    # n, v, l setup
    body.append(LetStmt("n", "vec3", CallExpr(VarRef("normalize"), [VarRef(normal_var)])))
    if has_pos:
        body.append(LetStmt("v", "vec3", CallExpr(VarRef("normalize"), [
            BinaryOp("-", VarRef("view_pos"), VarRef(pos_var))])))
    else:
        body.append(LetStmt("v", "vec3", CallExpr(VarRef("normalize"), [
            ConstructorExpr("vec3", [NumberLit("0.0"), NumberLit("0.0"), NumberLit("1.0")])])))
    if _bl_raster_has_multi_light:
        # Multi-light: dummy l for downstream layers (coat, sheen, etc.)
        body.append(LetStmt("l", "vec3",
            ConstructorExpr("vec3", [NumberLit("0.0"), NumberLit("1.0"), NumberLit("0.0")])))
    else:
        body.append(LetStmt("l", "vec3", CallExpr(VarRef("normalize"), [VarRef("light_dir")])))
    body.append(LetStmt("n_dot_v", "scalar", CallExpr(VarRef("max"), [
        CallExpr(VarRef("dot"), [VarRef("n"), VarRef("v")]),
        NumberLit("0.001"),
    ])))

    # Detect OpenPBR mode
    _use_openpbr = _is_openpbr(module)

    # Collect custom properties for extended struct
    _bl_frag_props = getattr(surface, 'properties', None)
    extra_props = []
    if _bl_frag_props and hasattr(_bl_frag_props, 'fields') and _bl_frag_props.fields:
        prop_map = _OPENPBR_PROPERTY_TO_BINDLESS if _use_openpbr else _PROPERTY_TO_BINDLESS
        for prop in _bl_frag_props.fields:
            if prop.name not in prop_map:
                extra_props.append((prop.name, prop.type_name))

    # Build struct with extra fields (OpenPBR appends its own extended fields)
    if _use_openpbr:
        stage._bindless_extra_properties = list(_OPENPBR_EXTRA_FIELDS) + extra_props
        stage._openpbr = True
    elif extra_props:
        stage._bindless_extra_properties = extra_props

    # Bindless uber-shader body (material_index from push constant)
    if _use_openpbr:
        uber_body, output_var = _emit_openpbr_bindless_body(
            surface, VarRef("material_index"), schedule, is_rt=False,
            lighting=lighting, pos_var=pos_var)
    else:
        uber_body, output_var = _emit_bindless_layer_body(
            surface, VarRef("material_index"), schedule, is_rt=False,
            lighting=lighting, pos_var=pos_var,
            properties=getattr(surface, 'properties', None))
    body.extend(uber_body)

    # Output final color
    body.append(AssignStmt(
        AssignTarget(VarRef("color")),
        ConstructorExpr("vec4", [VarRef(output_var), NumberLit("1.0")]),
    ))

    stage.functions.append(FunctionDef("main", [], None, body))
    return stage


def _expand_bindless_closest_hit(
    surface: SurfaceDecl,
    module: Module,
    schedule: dict[str, str] | None = None,
    lighting: LightingDecl | None = None,
) -> StageBlock:
    """Generate a bindless uber-shader closest-hit stage.

    Uses materials SSBO + bindless texture array. Material index
    comes from gl_GeometryIndexEXT (set per BLAS geometry).
    """
    stage = StageBlock(stage_type="closest_hit")

    # Incoming ray payload
    stage.ray_payloads.append(RayPayloadDecl("payload", "vec4"))

    # Hit attributes (barycentrics)
    stage.hit_attributes.append(HitAttributeDecl("bary", "vec2"))

    # Storage buffers for vertex data (SoA layout)
    stage.storage_buffers.append(StorageBufferDecl("positions", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("normals", "vec4"))
    stage.storage_buffers.append(StorageBufferDecl("tex_coords", "vec2"))
    stage.storage_buffers.append(StorageBufferDecl("indices", "uint"))

    # Light properties uniform (from lighting block — needed for multi_light count)
    if lighting and lighting.properties:
        lprops = lighting.properties
        stage.uniforms.append(UniformBlock(lprops.name, [
            BlockField(f.name, f.type_name) for f in lprops.fields
        ]))
        if not hasattr(stage, '_properties_defaults'):
            stage._properties_defaults = {}
        for f in lprops.fields:
            if f.default is not None:
                stage._properties_defaults[(lprops.name, f.name)] = f.default

    # Materials SSBO
    stage.storage_buffers.append(StorageBufferDecl(
        "materials", "BindlessMaterialData"))

    # If multi_light layer present, add lights SSBO (and shadow_matrices if shadow_map arg)
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

    # IBL samplers (scene-global) — from lighting block or surface
    if lighting:
        for sam in lighting.samplers:
            stage.samplers.append(SamplerDecl(sam.name, type_name=sam.type_name))
    else:
        for sam in surface.samplers:
            if sam.type_name == "samplerCube" or sam.name in ("brdf_lut",):
                stage.samplers.append(SamplerDecl(sam.name, type_name=sam.type_name))

    # Bindless texture array
    stage.bindless_texture_arrays.append(BindlessTextureArrayDecl("textures"))

    # --- Generate main body ---
    body = []

    # Load per-geometry index offset from materials SSBO (indexed by geometry_index)
    body.append(LetStmt("geo_index_offset", "scalar",
        FieldAccess(IndexAccess(VarRef("materials"), VarRef("geometry_index")),
            "index_offset")))

    # Barycentric interpolation preamble — offset primitive_id by geometry's index_offset
    _emit_barycentric_interpolation(body, VarRef("geo_index_offset"))

    # Detect multi_light mode for bindless RT
    _bl_rt_has_multi_light = _detect_multi_light(lighting)

    # View and light directions (RT-specific)
    body.append(LetStmt("v", "vec3", CallExpr(VarRef("normalize"), [
        UnaryOp("-", VarRef("world_ray_direction"))])))
    if _bl_rt_has_multi_light:
        # Multi-light: dummy l for downstream layers (coat, sheen, etc.)
        body.append(LetStmt("l", "vec3",
            ConstructorExpr("vec3", [NumberLit("0.0"), NumberLit("1.0"), NumberLit("0.0")])))
    else:
        body.append(LetStmt("l", "vec3",
            CallExpr(VarRef("normalize"), [
                ConstructorExpr("vec3", [
                    NumberLit("1.0"), NumberLit("0.8"), NumberLit("0.6")])])))
    body.append(LetStmt("n_dot_v", "scalar", CallExpr(VarRef("max"), [
        CallExpr(VarRef("dot"), [VarRef("n"), VarRef("v")]),
        NumberLit("0.001"),
    ])))

    # Detect OpenPBR mode
    _use_openpbr = _is_openpbr(module)

    # Collect custom properties for extended struct
    _bl_rt_props = getattr(surface, 'properties', None)
    extra_props = []
    if _bl_rt_props and hasattr(_bl_rt_props, 'fields') and _bl_rt_props.fields:
        prop_map = _OPENPBR_PROPERTY_TO_BINDLESS if _use_openpbr else _PROPERTY_TO_BINDLESS
        for prop in _bl_rt_props.fields:
            if prop.name not in prop_map:
                extra_props.append((prop.name, prop.type_name))

    # Build struct with extra fields (OpenPBR appends its own extended fields)
    if _use_openpbr:
        stage._bindless_extra_properties = list(_OPENPBR_EXTRA_FIELDS) + extra_props
        stage._openpbr = True
    elif extra_props:
        stage._bindless_extra_properties = extra_props

    # Bindless uber-shader body (material index from gl_GeometryIndexEXT)
    if _use_openpbr:
        uber_body, output_var = _emit_openpbr_bindless_body(
            surface, VarRef("geometry_index"), schedule, is_rt=True,
            lighting=lighting, pos_var="hit_pos")
    else:
        uber_body, output_var = _emit_bindless_layer_body(
            surface, VarRef("geometry_index"), schedule, is_rt=True,
            lighting=lighting, pos_var="hit_pos",
            properties=getattr(surface, 'properties', None))
    body.extend(uber_body)

    # Write to payload
    body.append(AssignStmt(
        AssignTarget(VarRef("payload")),
        ConstructorExpr("vec4", [VarRef(output_var), NumberLit("1.0")]),
    ))

    stage.functions.append(FunctionDef("main", [], None, body))
    return stage
