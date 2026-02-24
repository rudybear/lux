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
    GeometryDecl, PipelineDecl,
    ScheduleDecl, EnvironmentDecl, ProceduralDecl,
    RayPayloadDecl, HitAttributeDecl, AccelDecl, StorageImageDecl,
    StorageBufferDecl, BindlessTextureArrayDecl, IndexAccess, FieldAccess,
    IfStmt, TaskPayloadDecl, PropertiesBlock,
)


# --- Schedule strategy tables ---

_DEFAULT_STRATEGIES = {
    "fresnel": "schlick",
    "distribution": "ggx",
    "geometry_term": "smith_ggx",
    "tonemap": "none",
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

    for pipeline in module.pipelines:
        # Filter: skip pipelines that don't match the filter
        if pipeline_filter and pipeline.name != pipeline_filter:
            continue
        geo_name = None
        surf_name = None
        schedule_name = None
        env_name = None
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
            elif member.name == "use_task_shader":
                if isinstance(member.value, VarRef) and member.value.name == "true":
                    use_task_shader = True

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
                bindless=bindless,
            )
            module.stages.extend(stages)
        elif mode == "mesh_shader":
            # Mesh shader pipeline expansion
            surface = surfaces.get(surf_name) if surf_name else None
            geometry = geometries.get(geo_name) if geo_name else None
            stages = _expand_mesh_pipeline(
                surface, geometry, module, schedule, pipeline, use_task_shader,
                bindless=bindless,
            )
            module.stages.extend(stages)
        elif surf_name and surf_name in surfaces:
            surface = surfaces[surf_name]
            geometry = geometries.get(geo_name) if geo_name else None
            stages = _expand_pipeline(surface, geometry, module, schedule,
                                      bindless=bindless)
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
) -> list[StageBlock]:
    """Generate vertex and fragment stages from surface + geometry."""
    stages = []

    if geometry:
        vert = _expand_geometry_to_vertex(geometry)
        stages.append(vert)

    if bindless and surface.layers is not None:
        frag = _expand_bindless_fragment(surface, module, geometry, schedule)
    else:
        frag = _expand_surface_to_fragment(surface, module, geometry, schedule)
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

    # Light uniform (use uniform block for wgpu compatibility)
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

    # Generate main function body - use layered path if surface has layers
    if surface.layers is not None:
        body = _generate_layered_main(surface, frag_inputs, schedule, module=module)
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


def _collect_layer_functions(module: Module) -> dict:
    """Scan module functions for @layer annotations, validate, return dict[name, FunctionDef]."""
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
        if fn.name in _BUILTIN_LAYER_NAMES:
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


def _generate_layered_main(
    surface: SurfaceDecl,
    frag_inputs: list[tuple[str, str]],
    schedule: dict[str, str] | None = None,
    module: Module | None = None,
) -> list:
    """Generate main() body for a layered surface-expanded fragment shader.

    Processes layer declarations to generate PBR lighting code equivalent to
    hand-written gltf_pbr.lux. Supports base, normal_map, ibl, and emission layers.
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
    body.append(LetStmt("l", "vec3",
        CallExpr(VarRef("normalize"), [VarRef("light_dir")])))
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

        # Direct lighting via gltf_pbr (includes N·L)
        body.append(LetStmt("direct", "vec3", CallExpr(VarRef("gltf_pbr"), [
            VarRef("n"), VarRef("v"), VarRef("l"),
            VarRef("layer_albedo"), VarRef("layer_roughness"),
            VarRef("layer_metallic"),
        ])))
        # Light color tint
        body.append(LetStmt("direct_lit", "vec3", BinaryOp("*",
            VarRef("direct"),
            ConstructorExpr("vec3", [
                NumberLit("1.0"), NumberLit("0.98"), NumberLit("0.95"),
            ]),
        )))
        result_var = "direct_lit"

    # --- Transmission layer (replaces diffuse proportionally) ---
    if "transmission" in layers_by_name:
        trans_args = _get_layer_args(layers_by_name["transmission"])
        factor_expr = trans_args.get("factor", NumberLit("0.0"))
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

    # --- IBL layer: image-based lighting ---
    if "ibl" in layers_by_name:
        ibl_args = _get_layer_args(layers_by_name["ibl"])
        specular_map = ibl_args.get("specular_map")
        irradiance_map = ibl_args.get("irradiance_map")
        brdf_lut_ref = ibl_args.get("brdf_lut")

        # Reflection vector
        body.append(LetStmt("r", "vec3", CallExpr(VarRef("reflect"), [
            BinaryOp("*", VarRef("v"), UnaryOp("-", NumberLit("1.0"))),
            VarRef("n"),
        ])))

        # Sample IBL textures (surface expander's job)
        body.append(LetStmt("prefiltered", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample_lod"), [
                specular_map, VarRef("r"),
                BinaryOp("*", VarRef("layer_roughness"), NumberLit("8.0")),
            ]),
            "xyz",
        )))
        body.append(LetStmt("irradiance", "vec3", SwizzleAccess(
            CallExpr(VarRef("sample"), [irradiance_map, VarRef("n")]),
            "xyz",
        )))
        body.append(LetStmt("brdf_sample", "vec2", SwizzleAccess(
            CallExpr(VarRef("sample"), [
                brdf_lut_ref,
                ConstructorExpr("vec2", [
                    VarRef("n_dot_v"), VarRef("layer_roughness"),
                ]),
            ]),
            "xy",
        )))

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
            coat_factor_expr = coat_args.get("factor", NumberLit("1.0"))
            coat_rough_expr = coat_args.get("roughness", NumberLit("0.0"))
            body.append(LetStmt("coat_ibl_factor", "scalar", coat_factor_expr))
            body.append(LetStmt("coat_ibl_roughness", "scalar", coat_rough_expr))
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
                BinaryOp("+", VarRef("ambient"), VarRef("coat_ibl_contrib"))))
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
        color_expr = sheen_args.get("color",
            ConstructorExpr("vec3", [NumberLit("0.0")]))
        rough_expr = sheen_args.get("roughness", NumberLit("0.5"))

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
        factor_expr = coat_args.get("factor", NumberLit("1.0"))
        rough_expr = coat_args.get("roughness", NumberLit("0.0"))
        coat_normal = coat_args.get("normal", None)

        body.append(LetStmt("coat_factor", "scalar", factor_expr))
        body.append(LetStmt("coat_roughness", "scalar", rough_expr))
        coat_n = VarRef("coat_n") if coat_normal else VarRef("n")
        if coat_normal:
            body.append(LetStmt("coat_n", "vec3", coat_normal))

        body.append(LetStmt("coated", "vec3", CallExpr(
            VarRef("coat_over"), [
                VarRef(result_var), coat_n, VarRef("v"), VarRef("l"),
                VarRef("coat_factor"), VarRef("coat_roughness"),
            ])))
        result_var = "coated"

    # --- Custom @layer functions (declaration order, after coat, before emission) ---
    if module is not None:
        layer_fns = _collect_layer_functions(module)
        for layer in surface.layers:
            if layer.name in layer_fns:
                result_var = _emit_custom_layer(
                    layer, layer_fns[layer.name], result_var, body,
                    prefix=f"custom_{layer.name}", rewrite_for_rt=False)

    # --- Emission layer (additive) ---
    if "emission" in layers_by_name:
        em_args = _get_layer_args(layers_by_name["emission"])
        em_expr = em_args.get("color",
            ConstructorExpr("vec3", [NumberLit("0.0")]))
        body.append(LetStmt("emission_color", "vec3", em_expr))
        body.append(LetStmt("hdr", "vec3",
            BinaryOp("+", VarRef(result_var), VarRef("emission_color"))))
        result_var = "hdr"

    # --- Tonemap + gamma ---
    tonemap_strategy = schedule.get("tonemap", "none") if schedule else "none"
    if tonemap_strategy == "aces":
        body.append(LetStmt("tonemapped", "vec3",
            CallExpr(VarRef("tonemap_aces"), [VarRef(result_var)])))
        body.append(LetStmt("final_color", "vec3",
            CallExpr(VarRef("linear_to_srgb"), [VarRef("tonemapped")])))
        output_var = "final_color"
    elif tonemap_strategy == "reinhard":
        body.append(LetStmt("tonemapped", "vec3",
            CallExpr(VarRef("tonemap_reinhard"), [VarRef(result_var)])))
        body.append(LetStmt("final_color", "vec3",
            CallExpr(VarRef("linear_to_srgb"), [VarRef("tonemapped")])))
        output_var = "final_color"
    else:
        output_var = result_var

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
) -> list[StageBlock]:
    """Generate ray tracing stages from pipeline declarations."""
    stages = []

    # 1. Ray generation shader
    raygen = _expand_raygen(surface, environment, max_bounces)
    stages.append(raygen)

    # 2. Closest-hit shader from surface
    if surface:
        if bindless and surface.layers is not None:
            chit = _expand_bindless_closest_hit(surface, module, schedule)
        else:
            chit = _expand_surface_to_closest_hit(surface, module, schedule)
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
) -> StageBlock:
    """Generate a closest-hit shader from a surface declaration.

    For surfaces with `layers`, generates full barycentric interpolation,
    storage buffer reads, texture sampling (sample_lod), and layered
    PBR evaluation matching the hand-written gltf_pbr_rt.lux output.

    For surfaces with `brdf:`, generates the simpler existing path.
    """
    if surface.layers is not None:
        return _expand_layered_closest_hit(surface, module, schedule)

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

    # Light uniform
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

    # --- Generate main body ---
    body = []

    # Barycentric interpolation preamble (matches gltf_pbr_rt.lux)
    # let base: scalar = primitive_id * 3.0;
    body.append(LetStmt("base", "scalar",
        BinaryOp("*", VarRef("primitive_id"), NumberLit("3.0"))))

    # Index lookups
    body.append(LetStmt("i0", "uint", IndexAccess(VarRef("indices"), VarRef("base"))))
    body.append(LetStmt("i1", "uint", IndexAccess(VarRef("indices"),
        BinaryOp("+", VarRef("base"), NumberLit("1.0")))))
    body.append(LetStmt("i2", "uint", IndexAccess(VarRef("indices"),
        BinaryOp("+", VarRef("base"), NumberLit("2.0")))))

    # Barycentric weights
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

    # Index layers
    layers_by_name = {}
    for layer in surface.layers:
        layers_by_name[layer.name] = layer

    # --- View and light directions (RT-specific) ---
    body.append(LetStmt("v", "vec3", CallExpr(VarRef("normalize"), [
        UnaryOp("-", VarRef("world_ray_direction"))])))
    body.append(LetStmt("l", "vec3",
        CallExpr(VarRef("normalize"), [VarRef("light_dir")])))
    body.append(LetStmt("n_dot_v", "scalar", CallExpr(VarRef("max"), [
        CallExpr(VarRef("dot"), [VarRef("n"), VarRef("v")]),
        NumberLit("0.001"),
    ])))

    # --- Base layer: direct lighting ---
    # Rewrite sample→sample_lod for all layer expressions in RT
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

    # --- IBL layer ---
    if "ibl" in layers_by_name:
        ibl_args = _get_layer_args(layers_by_name["ibl"])
        specular_map = ibl_args.get("specular_map")
        irradiance_map = ibl_args.get("irradiance_map")
        brdf_lut_ref = ibl_args.get("brdf_lut")

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
    tonemap_strategy = schedule.get("tonemap", "none") if schedule else "none"
    if tonemap_strategy == "aces":
        body.append(LetStmt("tonemapped", "vec3",
            CallExpr(VarRef("tonemap_aces"), [VarRef(result_var)])))
        body.append(LetStmt("final_color", "vec3",
            CallExpr(VarRef("linear_to_srgb"), [VarRef("tonemapped")])))
        output_var = "final_color"
    elif tonemap_strategy == "reinhard":
        body.append(LetStmt("tonemapped", "vec3",
            CallExpr(VarRef("tonemap_reinhard"), [VarRef(result_var)])))
        body.append(LetStmt("final_color", "vec3",
            CallExpr(VarRef("linear_to_srgb"), [VarRef("tonemapped")])))
        output_var = "final_color"
    else:
        output_var = result_var

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
            frag = _expand_bindless_fragment(surface, module, geometry, schedule)
        else:
            frag = _expand_surface_to_fragment(surface, module, geometry, schedule)
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


def _build_bindless_material_struct_fields() -> list[BlockField]:
    """Return the fields for BindlessMaterialData (std430 SSBO layout)."""
    return [
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


def _emit_bindless_layer_body(
    surface: SurfaceDecl,
    mat_idx_expr,
    schedule: dict[str, str] | None = None,
    is_rt: bool = False,
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

    # --- Transmission (flag check) ---
    body.append(LetStmt("result_pre_ibl", "vec3", VarRef(result_var)))
    body.append(IfStmt(
        BinaryOp("!=",
            BinaryOp("&", VarRef("mat_flags"),
                NumberLit(str(_FLAG_TRANSMISSION))),
            NumberLit("0")),
        [
            LetStmt("trans_factor", "scalar",
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "transmissionFactor")),
            LetStmt("trans_ior", "scalar",
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "ior")),
            LetStmt("transmitted", "vec3", CallExpr(
                VarRef("transmission_replace"), [
                    VarRef("result_pre_ibl"), VarRef("layer_albedo"),
                    VarRef("layer_roughness"),
                    VarRef("trans_factor"), VarRef("trans_ior"),
                    VarRef("n"), VarRef("v"), VarRef("l"),
                    NumberLit("0.0"),
                    ConstructorExpr("vec3", [NumberLit("1.0")]),
                    NumberLit("1000000.0"),
                ])),
            AssignStmt(AssignTarget(VarRef("result_pre_ibl")), VarRef("transmitted")),
        ],
        [],
    ))
    result_var = "result_pre_ibl"

    # --- IBL ---
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

    body.append(LetStmt("hdr_partial", "vec3",
        BinaryOp("+", VarRef(result_var), VarRef("ambient"))))
    result_var = "hdr_partial"

    # --- Sheen (flag check) ---
    body.append(IfStmt(
        BinaryOp("!=",
            BinaryOp("&", VarRef("mat_flags"), NumberLit(str(_FLAG_SHEEN))),
            NumberLit("0")),
        [
            LetStmt("sheen_color_val", "vec3",
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "sheenColorFactor")),
            LetStmt("sheen_roughness_val", "scalar",
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "sheenRoughness")),
            LetStmt("sheen_tex_sample", "vec4",
                ConstructorExpr("vec4", [NumberLit("1.0")])),
            _sample_tex("tex_sheenColor", "sheen_tex_sample"),
            # glTF sheen color textures are sRGB-encoded
            AssignStmt(AssignTarget(VarRef("sheen_color_val")),
                BinaryOp("*", VarRef("sheen_color_val"),
                    CallExpr(VarRef("srgb_to_linear"), [
                        SwizzleAccess(VarRef("sheen_tex_sample"), "xyz")]))),
            LetStmt("sheened", "vec3", CallExpr(
                VarRef("sheen_over"), [
                    VarRef(result_var), VarRef("n"), VarRef("v"), VarRef("l"),
                    VarRef("sheen_color_val"), VarRef("sheen_roughness_val"),
                ])),
            AssignStmt(AssignTarget(VarRef(result_var)), VarRef("sheened")),
        ],
        [],
    ))

    # --- Clearcoat (flag check) ---
    body.append(IfStmt(
        BinaryOp("!=",
            BinaryOp("&", VarRef("mat_flags"), NumberLit(str(_FLAG_CLEARCOAT))),
            NumberLit("0")),
        [
            LetStmt("coat_factor", "scalar",
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "clearcoatFactor")),
            LetStmt("coat_roughness", "scalar",
                FieldAccess(IndexAccess(VarRef("materials"), VarRef("mat_idx")),
                    "clearcoatRoughness")),
            LetStmt("coat_tex_sample", "vec4",
                ConstructorExpr("vec4", [NumberLit("1.0")])),
            _sample_tex("tex_clearcoat", "coat_tex_sample"),
            AssignStmt(AssignTarget(VarRef("coat_factor")),
                BinaryOp("*", VarRef("coat_factor"),
                    SwizzleAccess(VarRef("coat_tex_sample"), "x"))),
            LetStmt("coat_rough_sample", "vec4",
                ConstructorExpr("vec4", [NumberLit("1.0")])),
            _sample_tex("tex_clearcoatRough", "coat_rough_sample"),
            AssignStmt(AssignTarget(VarRef("coat_roughness")),
                BinaryOp("*", VarRef("coat_roughness"),
                    SwizzleAccess(VarRef("coat_rough_sample"), "y"))),
            LetStmt("coated", "vec3", CallExpr(
                VarRef("coat_over"), [
                    VarRef(result_var), VarRef("n"), VarRef("v"), VarRef("l"),
                    VarRef("coat_factor"), VarRef("coat_roughness"),
                ])),
            AssignStmt(AssignTarget(VarRef(result_var)), VarRef("coated")),
        ],
        [],
    ))

    # --- Emission (flag check) ---
    body.append(IfStmt(
        BinaryOp("!=",
            BinaryOp("&", VarRef("mat_flags"), NumberLit(str(_FLAG_EMISSION))),
            NumberLit("0")),
        [
            LetStmt("em_color", "vec3", BinaryOp("*",
                VarRef("mat_emissiveFactor"), VarRef("mat_emissionStrength"))),
            LetStmt("em_tex_sample", "vec4",
                ConstructorExpr("vec4", [NumberLit("1.0")])),
            _sample_tex("tex_emissive", "em_tex_sample"),
            # glTF emissive textures are sRGB-encoded
            AssignStmt(AssignTarget(VarRef("em_color")),
                BinaryOp("*", VarRef("em_color"),
                    CallExpr(VarRef("srgb_to_linear"), [
                        SwizzleAccess(VarRef("em_tex_sample"), "xyz")]))),
            AssignStmt(AssignTarget(VarRef(result_var)),
                BinaryOp("+", VarRef(result_var), VarRef("em_color"))),
        ],
        [],
    ))

    # --- Tonemap + output ---
    tonemap_strategy = schedule.get("tonemap", "none") if schedule else "none"
    if tonemap_strategy == "aces":
        body.append(LetStmt("tonemapped", "vec3",
            CallExpr(VarRef("tonemap_aces"), [VarRef(result_var)])))
        body.append(LetStmt("final_color", "vec3",
            CallExpr(VarRef("linear_to_srgb"), [VarRef("tonemapped")])))
        output_var = "final_color"
    elif tonemap_strategy == "reinhard":
        body.append(LetStmt("tonemapped", "vec3",
            CallExpr(VarRef("tonemap_reinhard"), [VarRef(result_var)])))
        body.append(LetStmt("final_color", "vec3",
            CallExpr(VarRef("linear_to_srgb"), [VarRef("tonemapped")])))
        output_var = "final_color"
    else:
        output_var = result_var

    return body, output_var


def _expand_bindless_fragment(
    surface: SurfaceDecl,
    module: Module,
    geometry: GeometryDecl | None = None,
    schedule: dict[str, str] | None = None,
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

    # --- Uniforms: Light ---
    stage.uniforms.append(UniformBlock("Light", [
        BlockField("light_dir", "vec3"),
        BlockField("view_pos", "vec3"),
    ]))

    # --- Materials SSBO (replaces per-material UBO) ---
    stage.storage_buffers.append(StorageBufferDecl(
        "materials", "BindlessMaterialData"))

    # --- IBL samplers (kept as regular samplers — scene-global, not per-material) ---
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

    # n, v, l setup
    body.append(LetStmt("n", "vec3", CallExpr(VarRef("normalize"), [VarRef(normal_var)])))
    if has_pos:
        body.append(LetStmt("v", "vec3", CallExpr(VarRef("normalize"), [
            BinaryOp("-", VarRef("view_pos"), VarRef(pos_var))])))
    else:
        body.append(LetStmt("v", "vec3", CallExpr(VarRef("normalize"), [
            ConstructorExpr("vec3", [NumberLit("0.0"), NumberLit("0.0"), NumberLit("1.0")])])))
    body.append(LetStmt("l", "vec3", CallExpr(VarRef("normalize"), [VarRef("light_dir")])))
    body.append(LetStmt("n_dot_v", "scalar", CallExpr(VarRef("max"), [
        CallExpr(VarRef("dot"), [VarRef("n"), VarRef("v")]),
        NumberLit("0.001"),
    ])))

    # Bindless uber-shader body (material_index from push constant)
    uber_body, output_var = _emit_bindless_layer_body(
        surface, VarRef("material_index"), schedule, is_rt=False)
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

    # Materials SSBO
    stage.storage_buffers.append(StorageBufferDecl(
        "materials", "BindlessMaterialData"))

    # IBL samplers (scene-global)
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
    body.append(LetStmt("base", "scalar",
        BinaryOp("+",
            BinaryOp("*", VarRef("geo_index_offset"), NumberLit("1.0")),
            BinaryOp("*", VarRef("primitive_id"), NumberLit("3.0")))))
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

    # View and light directions (RT-specific)
    body.append(LetStmt("v", "vec3", CallExpr(VarRef("normalize"), [
        UnaryOp("-", VarRef("world_ray_direction"))])))
    body.append(LetStmt("l", "vec3",
        CallExpr(VarRef("normalize"), [
            ConstructorExpr("vec3", [
                NumberLit("1.0"), NumberLit("0.8"), NumberLit("0.6")])])))
    body.append(LetStmt("n_dot_v", "scalar", CallExpr(VarRef("max"), [
        CallExpr(VarRef("dot"), [VarRef("n"), VarRef("v")]),
        NumberLit("0.001"),
    ])))

    # Bindless uber-shader body (material index from gl_GeometryIndexEXT)
    uber_body, output_var = _emit_bindless_layer_body(
        surface, VarRef("geometry_index"), schedule, is_rt=True)
    body.extend(uber_body)

    # Write to payload
    body.append(AssignStmt(
        AssignTarget(VarRef("payload")),
        ConstructorExpr("vec4", [VarRef(output_var), NumberLit("1.0")]),
    ))

    stage.functions.append(FunctionDef("main", [], None, body))
    return stage
