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
    RayPayloadDecl, HitAttributeDecl, AccelDecl,
    StorageBufferDecl, IndexAccess, FieldAccess,
    IfStmt,
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


def expand_surfaces(module: Module, pipeline_filter: str | None = None) -> None:
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
                surface, environment, procedural, module, schedule, max_bounces
            )
            module.stages.extend(stages)
        elif surf_name and surf_name in surfaces:
            surface = surfaces[surf_name]
            geometry = geometries.get(geo_name) if geo_name else None
            stages = _expand_pipeline(surface, geometry, module, schedule)
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
) -> list[StageBlock]:
    """Generate vertex and fragment stages from surface + geometry."""
    stages = []

    if geometry:
        vert = _expand_geometry_to_vertex(geometry)
        stages.append(vert)

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

    # Add sampler declarations from the surface
    for sam in surface.samplers:
        stage.samplers.append(SamplerDecl(sam.name, type_name=sam.type_name))

    # Generate main function body - use layered path if surface has layers
    if surface.layers is not None:
        body = _generate_layered_main(surface, frag_inputs, schedule)
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


def _generate_layered_main(
    surface: SurfaceDecl,
    frag_inputs: list[tuple[str, str]],
    schedule: dict[str, str] | None = None,
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

    # --- IBL layer: image-based lighting ---
    if "ibl" in layers_by_name:
        ibl_args = _get_layer_args(layers_by_name["ibl"])
        specular_map = ibl_args.get("specular_map")
        irradiance_map = ibl_args.get("irradiance_map")
        brdf_lut_ref = ibl_args.get("brdf_lut")

        # F0 = mix(0.04, albedo, metallic)
        body.append(LetStmt("f0", "vec3", CallExpr(VarRef("mix"), [
            ConstructorExpr("vec3", [NumberLit("0.04")]),
            VarRef("layer_albedo"),
            VarRef("layer_metallic"),
        ])))

        # Reflection vector
        body.append(LetStmt("r", "vec3", CallExpr(VarRef("reflect"), [
            BinaryOp("*", VarRef("v"), UnaryOp("-", NumberLit("1.0"))),
            VarRef("n"),
        ])))

        # Sample IBL textures
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

        # Roughness-dependent Fresnel (Fdez-Aguera 2019)
        body.append(LetStmt("fr", "vec3", BinaryOp("-",
            CallExpr(VarRef("max"), [
                ConstructorExpr("vec3", [
                    BinaryOp("-", NumberLit("1.0"), VarRef("layer_roughness")),
                ]),
                VarRef("f0"),
            ]),
            VarRef("f0"),
        )))
        body.append(LetStmt("k_s", "vec3", BinaryOp("+",
            VarRef("f0"),
            BinaryOp("*",
                VarRef("fr"),
                CallExpr(VarRef("pow"), [
                    CallExpr(VarRef("clamp"), [
                        BinaryOp("-", NumberLit("1.0"), VarRef("n_dot_v")),
                        NumberLit("0.0"),
                        NumberLit("1.0"),
                    ]),
                    NumberLit("5.0"),
                ]),
            ),
        )))

        # Multi-scattering energy compensation (Fdez-Aguera 2019)
        body.append(LetStmt("e_ss", "scalar", BinaryOp("+",
            SwizzleAccess(VarRef("brdf_sample"), "x"),
            SwizzleAccess(VarRef("brdf_sample"), "y"),
        )))
        body.append(LetStmt("f_single", "vec3", BinaryOp("+",
            BinaryOp("*", VarRef("k_s"),
                SwizzleAccess(VarRef("brdf_sample"), "x")),
            ConstructorExpr("vec3", [
                SwizzleAccess(VarRef("brdf_sample"), "y"),
            ]),
        )))
        body.append(LetStmt("f_avg", "vec3", BinaryOp("+",
            VarRef("f0"),
            BinaryOp("*",
                BinaryOp("-",
                    ConstructorExpr("vec3", [NumberLit("1.0")]),
                    VarRef("f0")),
                BinaryOp("/", NumberLit("1.0"), NumberLit("21.0")),
            ),
        )))
        body.append(LetStmt("f_ms", "vec3", BinaryOp("/",
            BinaryOp("*", VarRef("f_single"), VarRef("f_avg")),
            BinaryOp("-",
                ConstructorExpr("vec3", [NumberLit("1.0")]),
                BinaryOp("*", VarRef("f_avg"),
                    BinaryOp("-", NumberLit("1.0"), VarRef("e_ss"))),
            ),
        )))
        body.append(LetStmt("f_total", "vec3", BinaryOp("+",
            VarRef("f_single"),
            BinaryOp("*", VarRef("f_ms"),
                BinaryOp("-", NumberLit("1.0"), VarRef("e_ss"))),
        )))
        body.append(LetStmt("ibl_spec", "vec3",
            BinaryOp("*", VarRef("f_total"), VarRef("prefiltered"))))

        # Energy-conserving diffuse
        body.append(LetStmt("kd", "vec3", BinaryOp("*",
            BinaryOp("-",
                ConstructorExpr("vec3", [NumberLit("1.0")]),
                VarRef("f_total")),
            BinaryOp("-", NumberLit("1.0"), VarRef("layer_metallic")),
        )))
        body.append(LetStmt("ibl_diff", "vec3", BinaryOp("*",
            BinaryOp("*", VarRef("kd"), VarRef("layer_albedo")),
            VarRef("irradiance"),
        )))

        body.append(LetStmt("ambient", "vec3",
            BinaryOp("+", VarRef("ibl_diff"), VarRef("ibl_spec"))))

        # Combine direct + ambient
        body.append(LetStmt("hdr_partial", "vec3",
            BinaryOp("+", VarRef(result_var), VarRef("ambient"))))
        result_var = "hdr_partial"

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
) -> list[StageBlock]:
    """Generate ray tracing stages from pipeline declarations."""
    stages = []

    # 1. Ray generation shader
    raygen = _expand_raygen(surface, environment, max_bounces)
    stages.append(raygen)

    # 2. Closest-hit shader from surface
    if surface:
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

    # Camera uniform
    stage.uniforms.append(UniformBlock("Camera", [
        BlockField("camera_pos", "vec3"),
        BlockField("camera_inv_view", "mat4"),
        BlockField("camera_inv_proj", "mat4"),
    ]))

    # Output image (out variable for now; in real RT this would be a storage image)
    out = VarDecl("result_color", "vec4")
    out._is_input = False
    stage.outputs.append(out)

    # Main body: compute ray and trace
    body = []

    # Compute normalized pixel coordinates from launch_id / launch_size
    # let pixel: vec2 = vec2(launch_id.x, launch_id.y) + vec2(0.5);
    body.append(LetStmt("pixel", "vec2", BinaryOp("+",
        ConstructorExpr("vec2", [
            SwizzleAccess(VarRef("launch_id"), "x"),
            SwizzleAccess(VarRef("launch_id"), "y"),
        ]),
        ConstructorExpr("vec2", [NumberLit("0.5")]),
    )))

    # let uv: vec2 = pixel / vec2(launch_size.x, launch_size.y);
    body.append(LetStmt("uv", "vec2", BinaryOp("/",
        VarRef("pixel"),
        ConstructorExpr("vec2", [
            SwizzleAccess(VarRef("launch_size"), "x"),
            SwizzleAccess(VarRef("launch_size"), "y"),
        ]),
    )))

    # let ndc: vec2 = uv * 2.0 - vec2(1.0);
    body.append(LetStmt("ndc", "vec2", BinaryOp("-",
        BinaryOp("*", VarRef("uv"), NumberLit("2.0")),
        ConstructorExpr("vec2", [NumberLit("1.0")]),
    )))

    # Initialize payload to zero
    body.append(AssignStmt(
        AssignTarget(VarRef("payload")),
        ConstructorExpr("vec4", [NumberLit("0.0")]),
    ))

    # Compute ray origin and direction using camera matrices
    # let target: vec4 = camera_inv_proj * vec4(ndc.x, ndc.y, 1.0, 1.0);
    body.append(LetStmt("target", "vec4", BinaryOp("*",
        VarRef("camera_inv_proj"),
        ConstructorExpr("vec4", [
            SwizzleAccess(VarRef("ndc"), "x"),
            SwizzleAccess(VarRef("ndc"), "y"),
            NumberLit("1.0"),
            NumberLit("1.0"),
        ]),
    )))

    # let ray_dir: vec4 = camera_inv_view * vec4(normalize(target.xyz), 0.0);
    body.append(LetStmt("ray_dir", "vec4", BinaryOp("*",
        VarRef("camera_inv_view"),
        ConstructorExpr("vec4", [
            CallExpr(VarRef("normalize"), [SwizzleAccess(VarRef("target"), "xyz")]),
            NumberLit("0.0"),
        ]),
    )))

    # trace_ray(tlas, 0xff, 0xff, 0, 0, 0, camera_pos, 0.001, ray_dir.xyz, 10000.0, 0);
    body.append(ExprStmt(CallExpr(VarRef("trace_ray"), [
        VarRef("tlas"),
        NumberLit("0"),    # ray_flags
        NumberLit("255"),  # cull_mask
        NumberLit("0"),    # sbt_offset
        NumberLit("0"),    # sbt_stride
        NumberLit("0"),    # miss_index
        VarRef("camera_pos"),
        NumberLit("0.001"),  # tmin
        SwizzleAccess(VarRef("ray_dir"), "xyz"),
        NumberLit("10000.0"),  # tmax
        NumberLit("0"),    # payload location
    ])))

    # result_color = payload;
    body.append(AssignStmt(
        AssignTarget(VarRef("result_color")),
        VarRef("payload"),
    ))

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

    # --- IBL layer ---
    if "ibl" in layers_by_name:
        ibl_args = _get_layer_args(layers_by_name["ibl"])
        specular_map = ibl_args.get("specular_map")
        irradiance_map = ibl_args.get("irradiance_map")
        brdf_lut_ref = ibl_args.get("brdf_lut")

        body.append(LetStmt("f0", "vec3", CallExpr(VarRef("mix"), [
            ConstructorExpr("vec3", [NumberLit("0.04")]),
            VarRef("layer_albedo"), VarRef("layer_metallic"),
        ])))
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

        # Roughness-dependent Fresnel
        body.append(LetStmt("fr", "vec3", BinaryOp("-",
            CallExpr(VarRef("max"), [
                ConstructorExpr("vec3", [
                    BinaryOp("-", NumberLit("1.0"), VarRef("layer_roughness")),
                ]),
                VarRef("f0"),
            ]),
            VarRef("f0"),
        )))
        body.append(LetStmt("k_s", "vec3", BinaryOp("+",
            VarRef("f0"),
            BinaryOp("*", VarRef("fr"),
                CallExpr(VarRef("pow"), [
                    CallExpr(VarRef("clamp"), [
                        BinaryOp("-", NumberLit("1.0"), VarRef("n_dot_v")),
                        NumberLit("0.0"), NumberLit("1.0"),
                    ]),
                    NumberLit("5.0"),
                ])),
        )))

        # Multi-scattering energy compensation
        body.append(LetStmt("e_ss", "scalar", BinaryOp("+",
            SwizzleAccess(VarRef("brdf_sample"), "x"),
            SwizzleAccess(VarRef("brdf_sample"), "y"))))
        body.append(LetStmt("f_single", "vec3", BinaryOp("+",
            BinaryOp("*", VarRef("k_s"),
                SwizzleAccess(VarRef("brdf_sample"), "x")),
            ConstructorExpr("vec3", [
                SwizzleAccess(VarRef("brdf_sample"), "y")]))))
        body.append(LetStmt("f_avg", "vec3", BinaryOp("+",
            VarRef("f0"),
            BinaryOp("*",
                BinaryOp("-",
                    ConstructorExpr("vec3", [NumberLit("1.0")]),
                    VarRef("f0")),
                BinaryOp("/", NumberLit("1.0"), NumberLit("21.0"))))))
        body.append(LetStmt("f_ms", "vec3", BinaryOp("/",
            BinaryOp("*", VarRef("f_single"), VarRef("f_avg")),
            BinaryOp("-",
                ConstructorExpr("vec3", [NumberLit("1.0")]),
                BinaryOp("*", VarRef("f_avg"),
                    BinaryOp("-", NumberLit("1.0"), VarRef("e_ss")))))))
        body.append(LetStmt("f_total", "vec3", BinaryOp("+",
            VarRef("f_single"),
            BinaryOp("*", VarRef("f_ms"),
                BinaryOp("-", NumberLit("1.0"), VarRef("e_ss"))))))
        body.append(LetStmt("ibl_spec", "vec3",
            BinaryOp("*", VarRef("f_total"), VarRef("prefiltered"))))

        # Energy-conserving diffuse
        body.append(LetStmt("kd", "vec3", BinaryOp("*",
            BinaryOp("-",
                ConstructorExpr("vec3", [NumberLit("1.0")]),
                VarRef("f_total")),
            BinaryOp("-", NumberLit("1.0"), VarRef("layer_metallic")))))
        body.append(LetStmt("ibl_diff", "vec3", BinaryOp("*",
            BinaryOp("*", VarRef("kd"), VarRef("layer_albedo")),
            VarRef("irradiance"))))
        body.append(LetStmt("ambient", "vec3",
            BinaryOp("+", VarRef("ibl_diff"), VarRef("ibl_spec"))))
        body.append(LetStmt("hdr_partial", "vec3",
            BinaryOp("+", VarRef(result_var), VarRef("ambient"))))
        result_var = "hdr_partial"

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
