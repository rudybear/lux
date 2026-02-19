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
    AssignTarget, SurfaceDecl, GeometryDecl, PipelineDecl,
    ScheduleDecl, EnvironmentDecl, ProceduralDecl,
    RayPayloadDecl, HitAttributeDecl, AccelDecl,
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


def expand_surfaces(module: Module) -> None:
    """Expand pipeline/surface/geometry declarations into stage blocks.

    If a pipeline declaration references a surface and geometry by name,
    generate vertex + fragment stages. Otherwise, if a surface exists
    without a pipeline, generate just the fragment stage.

    For RT pipelines (mode: raytrace), generates raygen, closest_hit,
    miss, and optionally any_hit/intersection stages.
    """
    # Index declarations by name
    surfaces = {s.name: s for s in module.surfaces}
    geometries = {g.name: g for g in module.geometries}
    environments = {e.name: e for e in module.environments}
    procedurals = {p.name: p for p in module.procedurals}

    for pipeline in module.pipelines:
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
    for sam_name in surface.samplers:
        stage.samplers.append(SamplerDecl(sam_name))

    # Generate main function body
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

    The closest-hit shader:
    - Receives incoming ray payload
    - Evaluates BRDF using hit point data (builtins: world_ray_origin, hit_t, etc.)
    - Writes color to the payload
    """
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
    for sam_name in environment.samplers:
        stage.samplers.append(SamplerDecl(sam_name))

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
