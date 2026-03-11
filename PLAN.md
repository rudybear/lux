# Lux v0.2 — From Shader Language to Rendering Specification Language

## The Problem With v0.1

We built GLSL-with-nicer-syntax. That's not the vision. The vision is:

> **The rendering equation should be expressible directly. BRDFs are composable
> first-class objects. The compiler — not the programmer — decides approximation
> strategies and GPU scheduling.**

The gap we're filling:

```
Current world:
    Rendering Equation (math on paper)
        → Human translates (error-prone, lossy)
    GLSL/HLSL (API-oriented boilerplate)
        → Compiler
    SPIR-V

Lux world:
    Rendering Equation (math)
        → LLM translates (natural — LLMs are good at math)
    Lux (mathematical rendering specification)
        → Smart compiler (handles approximations, scheduling, GPU tricks)
    SPIR-V
```

Nobody has built an AI-first, mathematically-oriented rendering specification
language. That's our gap. Not "better GLSL" — **expressible optics**.

---

## Phase 0: Shader Playground ✅ COMPLETE

Proved the backend works end-to-end by rendering actual pixels via `wgpu-py`.

- **0.1** — Headless render test harness (`playground/render_harness.py`)
- **0.2** — Interactive preview window (`playground/preview.py`)
- **0.3** — Reference comparison tool (`playground/compare.py`)
- Mesh loading, orbit camera, live reload all working

---

## Phase 1: The Mathematical Core ✅ COMPLETE

Lux is now a declarative rendering specification language with first-class
radiometric types, BRDFs, and surface/geometry/pipeline declarations.

- **1.1** — Radiometric types: `Radiance`, `Reflectance`, `Direction`, `Normal`, `Irradiance` (type aliases with semantic meaning, zero runtime cost)
- **1.2** — BRDF as first-class type with composable math objects
- **1.3** — `surface` declaration: declarative material specification → fragment shader
- **1.4** — `geometry` declaration: declarative vertex transforms → vertex shader
- **1.5** — `pipeline` declaration: ties geometry + surface + lighting + tonemap together

---

## Phase 2: Standard Library ✅ COMPLETE

Written in Lux itself, fully inlineable.

- **2.1** — `stdlib/brdf.lux`: Lambert, GGX NDF, Smith GGX, Fresnel Schlick, PBR BRDF (13 functions)
- **2.2** — `stdlib/sdf.lux`: Sphere, box, cylinder, torus, CSG ops, smooth ops, transforms, normals, raymarching (18 functions)
- **2.3** — `stdlib/noise.lux`: Perlin 2D/3D, simplex 2D/3D, Voronoi, FBM, turbulence, cellular, curl, domain warp (13 functions)
- **2.4** — `stdlib/color.lux`: sRGB conversion, tonemapping (Reinhard, ACES), luminance (5 functions)

**Total: 76 functions, 177 overloaded signatures, 19 types, 5 type aliases, 3 constants.**

---

## Phase 3: Schedule Separation & Compiler Intelligence ✅ COMPLETE

- **3.1** — Constant folding pass (arithmetic, built-in function evaluation)
- **3.2** — Import system with `_resolve_imports()` (monolithic SPIR-V output)
- **3.3** — Surface expansion (`expand_surfaces()`) generates fragment shaders from declarative specs

---

## Phase 4: Differentiable Rendering + AI Training Pipeline ✅ COMPLETE

### 4.1 — Forward-Mode Autodiff ✅

- `@differentiable` annotation on functions
- Symbolic AST differentiation (forward-mode, chain rule throughout)
- Generates gradient functions: `fn f(x) → fn f_d_x(x)` for each scalar parameter
- 20+ differentiation rules: arithmetic, trig, exp/log, sqrt, pow, dot, length, normalize, mix, clamp, smoothstep, constructors, swizzle, ternary
- Zero-expression optimization to prevent expression explosion
- Pipeline: parse → expand_surfaces → **autodiff_expand** → type_check → constant_fold → generate_spirv

### 4.2 — GLSL-to-Lux Transpiler ✅

- GLSL subset grammar (`luxc/grammar/glsl_subset.lark`)
- Lightweight GLSL AST → Lux AST conversion
- Type mapping (`float→scalar`, `sampler2D→sampler2d`), function mapping (`texture()→sample()`)
- Unsupported constructs (loops, `++`/`--`) flagged as comments
- CLI: `luxc --transpile input.glsl -o output.lux`

### 4.3 — AI Shader Generation ✅

- System prompt builder with grammar reference, builtins, stdlib summary, examples, constraints
- Claude API integration with compilation verification
- CLI: `luxc --ai "frosted glass" -o shader.lux`

### 4.4 — Training Data Pipeline ✅

- Batch GLSL→Lux→JSONL corpus generator (`tools/generate_training_data.py`)
- Optional Claude API descriptions, optional compilation verification
- Valid JSONL output with source, transpiled Lux, warnings, descriptions

---

## Phase 5: glTF PBR Compliance & MaterialX Parity

Gap analysis documented in [`ANALYSIS.md`](ANALYSIS.md).

### P5.1 — Critical Built-in Gaps ✅ COMPLETE

| Item | Effort | Status |
|------|--------|--------|
| `refract(I, N, eta)` built-in → GLSL.std.450 Refract | Tiny | ✅ |
| 2-arg `atan(y, x)` built-in → GLSL.std.450 Atan2 | Tiny | ✅ |
| `inversesqrt(x)` built-in → GLSL.std.450 InverseSqrt | Tiny | ✅ |
| `mod(x, y)` built-in → OpFMod | Tiny | ✅ |

### P5.2 — stdlib Expansions ✅ COMPLETE

**New BRDF functions (`stdlib/brdf.lux`):** ✅

| Function | Purpose | Status |
|----------|---------|--------|
| `v_ggx_correlated(NdotL, NdotV, alpha)` | Height-correlated Smith G (glTF spec compliance) | ✅ |
| `clearcoat_brdf(n, v, l, clearcoat, roughness)` | glTF clearcoat extension | ✅ |
| `charlie_ndf(roughness, NdotH)` | Sheen distribution (glTF sheen) | ✅ |
| `sheen_brdf(color, roughness, NdotH, NdotL, NdotV)` | Complete sheen evaluation | ✅ |
| `sheen_visibility(NdotL, NdotV)` | Sheen visibility term | ✅ |
| `anisotropic_ggx_ndf(NdotH, TdotH, BdotH, at, ab)` | Anisotropic GGX (glTF anisotropy) | ✅ |
| `anisotropic_v_ggx(...)` | Anisotropic visibility | ✅ |
| `oren_nayar_diffuse(albedo, roughness, NdotL, NdotV)` | Better diffuse model | ✅ |
| `burley_diffuse(albedo, roughness, NdotL, NdotV, VdotH)` | Disney diffuse | ✅ |
| `conductor_fresnel(f0, f82, VdotH)` | Metals with complex IOR (Lazanyi) | ✅ |
| `volume_attenuation(dist, attColor, attDist)` | Beer-Lambert absorption | ✅ |
| `ior_to_f0(ior)` | IOR to Fresnel F0 conversion | ✅ |
| `gltf_pbr(n, v, l, albedo, roughness, metallic)` | Full glTF PBR uber-shader | ✅ |

**New color module (`stdlib/colorspace.lux`):** ✅

| Function | Purpose | Status |
|----------|---------|--------|
| `rgb_to_hsv(c)` / `hsv_to_rgb(c)` | HSV color space | ✅ |
| `contrast(c, pivot, amount)` | Artistic contrast control | ✅ |
| `saturate_color(c, amount)` | Saturation adjustment | ✅ |
| `hue_shift(c, shift)` | Hue rotation | ✅ |
| `brightness(c, amount)` | Brightness scaling | ✅ |
| `gamma_correct(c, gamma)` | Gamma correction | ✅ |

**Normal mapping:**

| Item | Purpose |
|------|---------|
| TBN matrix construction from tangent + bitangent | Essential for textured content |
| Tangent/bitangent as stage inputs | Geometry pipeline support |

### P5.3 — Advanced Material Models ✅ COMPLETE

**New BRDF functions (`stdlib/brdf.lux`):** ✅

| Function | Purpose | Status |
|----------|---------|--------|
| `transmission_btdf(n, v, l, roughness, ior)` | Thin-surface microfacet BTDF (glTF transmission) | ✅ |
| `transmission_color(base_color, btdf, factor)` | Transmission color tinting | ✅ |
| `diffuse_transmission(albedo, n_dot_l)` | Back-face Lambert (glTF diffuse_transmission) | ✅ |
| `volumetric_btdf(n, v, l, roughness, eta_i, eta_o)` | Walter 2007 volumetric BTDF with Jacobian | ✅ |
| `iridescence_fresnel(outside_ior, film_ior, base_f0, thickness, cos_theta)` | Belcour & Barla 2017 thin-film interference | ✅ |
| `iridescence_sensitivity(opd, shift)` | CIE XYZ spectral evaluation via Gaussians | ✅ |
| `iridescence_f0_to_ior(f0)` / `iridescence_ior_to_f0(n_t, n_i)` | IOR↔F0 conversion helpers | ✅ |
| `dispersion_ior(base_ior, dispersion)` | Per-channel IOR via Abbe number | ✅ |
| `dispersion_f0(base_ior, dispersion)` | Per-channel F0 from dispersed IOR | ✅ |
| `dispersion_refract(v, n, base_ior, dispersion)` | Per-channel refraction | ✅ |

**New texture module (`stdlib/texture.lux`):** ✅

| Function | Purpose | Status |
|----------|---------|--------|
| `tbn_perturb_normal(sample, n, t, b)` | Normal mapping via TBN matrix | ✅ |
| `tbn_from_tangent(normal, tangent_vec4)` | Bitangent from tangent (w=handedness) | ✅ |
| `unpack_normal(encoded)` | Decode [0,1]→[-1,1] normal map | ✅ |
| `unpack_normal_strength(encoded, strength)` | Normal map with strength control | ✅ |
| `triplanar_weights(normal, sharpness)` | Triplanar blend weights | ✅ |
| `triplanar_uv_x/y/z(world_pos)` | Triplanar UV projection | ✅ |
| `triplanar_blend(x, y, z, weights)` | Triplanar color blending | ✅ |
| `triplanar_blend_scalar(x, y, z, weights)` | Triplanar scalar blending | ✅ |
| `parallax_offset(height, scale, view_ts)` | Simple parallax mapping | ✅ |
| `rotate_uv(uv, angle, center)` | UV rotation | ✅ |
| `tile_uv(uv, scale)` | UV tiling | ✅ |

### glTF Extension Coverage — ALL COMPLETE ✅

| Extension | Status | Implementation |
|-----------|--------|----------------|
| **Core PBR** | ✅ | `gltf_pbr()` with height-correlated Smith (`v_ggx_correlated`) |
| **ior** | ✅ | `ior_to_f0()` |
| **specular** | ✅ | Expressible via F0/F90 modification |
| **emissive_strength** | ✅ | Scalar multiply (no special function needed) |
| **unlit** | ✅ | Bypass BRDF (already expressible) |
| **occlusion** | ✅ | `mix(1.0, ao, strength)` |
| **clearcoat** | ✅ | `clearcoat_brdf()` |
| **sheen** | ✅ | `charlie_ndf()` + `sheen_brdf()` |
| **anisotropy** | ✅ | `anisotropic_ggx_ndf()` + `anisotropic_v_ggx()` + `atan()` 2-arg |
| **transmission** | ✅ | `transmission_btdf()` + `transmission_color()` |
| **volume** | ✅ | `volume_attenuation()` + `volumetric_btdf()` |
| **iridescence** | ✅ | `iridescence_fresnel()` (Belcour & Barla 2017) |
| **dispersion** | ✅ | `dispersion_ior()` + `dispersion_f0()` + `dispersion_refract()` |
| **diffuse_transmission** | ✅ | `diffuse_transmission()` |
| **normal mapping** | ✅ | `tbn_perturb_normal()` + `tbn_from_tangent()` + `unpack_normal()` |

---

## Phase 5.5: Layered Surface Shader System ✅ COMPLETE

A declarative layer system that unifies rasterization and ray tracing from a single `surface` declaration. Instead of writing separate shaders for each pipeline mode, authors declare composable **layers** that the compiler expands to the appropriate shader stages.

### P5.5.1 — `layers [...]` Syntax for Surfaces ✅

The `surface` declaration now accepts a `layers` field with an ordered list of built-in layer names:

```lux
surface GltfPBR {
    params {
        base_color: vec4,
        metallic: scalar,
        roughness: scalar,
        normal_map: sampler2d,
        ibl_diffuse: samplerCube,
        ibl_specular: samplerCube,
        brdf_lut: sampler2d,
        emission: vec3,
    },
    layers [base, normal_map, ibl, emission],
}
```

Built-in layers:

| Layer | Purpose |
|-------|---------|
| `base` | Core metallic-roughness PBR (GGX NDF, height-correlated Smith G, Schlick Fresnel) |
| `normal_map` | TBN-space normal perturbation from a `sampler2d` parameter |
| `ibl` | Image-based lighting via diffuse/specular `samplerCube` + BRDF LUT |
| `emission` | Additive emissive term |

### P5.5.2 — Energy Conservation with Albedo-Scaling ✅

Layers compose with energy conservation: each layer's contribution is scaled by the remaining energy budget. The `base` layer computes a specular+diffuse split using the Fresnel term, and subsequent layers (IBL, emission) are blended so total energy does not exceed 1.0 per channel.

### P5.5.3 — RT/Raster Unification ✅

A single `surface` declaration with `layers` compiles to **both** rasterization fragment shaders and ray tracing closest-hit shaders. The compiler determines the output path from the `pipeline` declaration's `mode` field or the `--pipeline` CLI parameter:

- `--pipeline raster` (default) — generates vertex + fragment shaders
- `--pipeline raytrace` — generates ray generation + closest-hit + miss shaders

### P5.5.4 — `samplerCube` Support in Surface Declarations ✅

The type system now includes `samplerCube` as a first-class sampler type for cubemap textures, used by IBL layers for diffuse irradiance and pre-filtered specular environment maps. Surface `params` blocks accept `samplerCube` parameters alongside existing `sampler2d` types.

### P5.5.5 — `--pipeline` CLI Parameter ✅

Pipeline selection is exposed via the command line:

```bash
luxc shader.lux --pipeline raster    # default: rasterization output
luxc shader.lux --pipeline raytrace  # ray tracing output
```

This allows the same `.lux` source to target different backends without modifying the shader code.

---

## Phase 5.6: Compile-Time Features & Shader Permutations (Completed)

General-purpose compile-time feature system for shader permutation generation.

**Syntax:**
- `features { has_normal_map: bool, has_clearcoat: bool }` — module-level flag declarations
- `if feature_expr` suffix on any comma-separated declaration item
- Module-level `if expr { ... }` blocks for grouping
- Boolean expressions: `&&`, `||`, `!`, parentheses

**Implementation:**
- Grammar: `features_decl`, `conditional_block`, `feature_expr` rules in `lux.lark`
- AST: `FeaturesDecl`, `ConditionalBlock`, `FeatureRef/And/Or/Not` nodes; `condition` field on items
- Evaluator: `luxc/features/evaluator.py` — expression evaluation + AST stripping
- Compiler: `strip_features()` runs after parse, before import resolution
- CLI: `--features`, `--all-permutations`, `--list-features`
- Reflection: `features` dict and `feature_suffix` in JSON output
- Permutation manifest: `*.manifest.json` with all combinations

**Status:** Implemented and verified. All 379 tests pass. 16 permutations of gltf_pbr_layered.lux compile successfully.

---

## Phase 6: Coat & Sheen Layers

Extends the layered surface system with additional physically-based layers.

| Layer | Purpose | glTF Extension |
|-------|---------|----------------|
| `coat` | Clearcoat: second GGX lobe with independent roughness and optional normal | `KHR_materials_clearcoat` |
| `sheen` | Fabric sheen via Charlie NDF + fitted visibility | `KHR_materials_sheen` |

The stdlib already contains `clearcoat_brdf()` and `sheen_brdf()` / `charlie_ndf()`; this phase wires them into the layer system so they compose automatically with energy conservation.

---

## Phase 7: Transmission Layer

| Layer | Purpose | glTF Extension |
|-------|---------|----------------|
| `transmission` | Microfacet BTDF for glass, liquids, thin surfaces | `KHR_materials_transmission` + `KHR_materials_volume` |

Requires LOD-aware background sampling for raster mode and recursive tracing for RT mode. The stdlib already provides `transmission_btdf()`, `transmission_color()`, and `volume_attenuation()`.

---

## Phase 8: `@layer` Custom Functions ✅ COMPLETE

User-defined layers via `@layer` annotation. Custom layer functions receive the accumulated color, geometric vectors (n, v, l), and user-defined parameters, returning an updated color. The compiler validates signatures (≥4 params, returns `vec3`, no name collision with built-in layers) and inserts them in declaration order after built-in layers, before emission — in both raster and RT paths.

```lux
@layer
fn cartoon(base: vec3, n: vec3, v: vec3, l: vec3,
           bands: scalar, rim_power: scalar, rim_color: vec3) -> vec3 {
    let n_dot_l: scalar = max(dot(n, l), 0.0);
    let quantized: scalar = floor(n_dot_l * bands + 0.5) / bands;
    let cel: vec3 = base * quantized;
    let n_dot_v: scalar = max(dot(n, v), 0.0);
    let rim: scalar = pow(1.0 - n_dot_v, rim_power);
    return cel + rim_color * rim;
}

surface ToonSurface {
    sampler2d albedo_tex,
    layers [
        base(albedo: sample(albedo_tex, uv).xyz, roughness: 0.8, metallic: 0.0),
        cartoon(bands: 4.0, rim_power: 3.0, rim_color: vec3(0.3, 0.3, 0.5)),
    ]
}
```

Implementation: `_collect_layer_functions()` and `_emit_custom_layer()` helpers in `surface_expander.py`. Validated and rendered across all three engines (Python, C++, Rust) in both raster and RT modes.

New files: `luxc/stdlib/toon.lux`, `examples/cartoon_toon.lux`, `tests/test_custom_layers.py` (15 tests). Total: 379 tests pass.

---

## Phase 9: Deferred Pipeline Mode

A third pipeline mode alongside raster and raytrace:

```bash
luxc shader.lux --pipeline deferred
```

The compiler expands `surface` declarations into a G-buffer write pass and a deferred lighting pass. Layers that depend on lighting (IBL, emission) are deferred to the second pass; geometry-only layers (base params, normal_map) are written in the first pass.

---

## Phase 10: Ray Tracing Pipeline (Full)

The `surface` declaration is a natural fit for ray tracing — the BRDF math is identical, only data sourcing and output mechanism change.

### Architecture: Surface → RT Stage Mapping

| Lux Concept | Rasterization (Today) | Ray Tracing (Future) |
|---|---|---|
| `surface { brdf: ... }` | → Fragment shader | → **Closest-hit** shader |
| `geometry { ... }` | → Vertex shader | → Vertex fetch via SBT + barycentrics |
| `pipeline { ... }` | → Rasterization draw | → **Ray generation** shader |
| `environment { ... }` (new) | — | → **Miss** shader |
| `surface { opacity: ... }` | → Alpha blend | → **Any-hit** shader |
| SDF stdlib | Fragment raymarching | → **Intersection** shader |

### P10.1 — RT Stage Types & Grammar

| Item | Effort |
|------|--------|
| New `EnvironmentDecl` AST node | Medium |
| Extend `PipelineDecl` with `mode: raytrace` | Medium |
| RT built-in variables (launch_id, ray_origin, ray_direction, hit_t, etc.) | Medium |
| `acceleration_structure` type | Small |

### P10.2 — RT SPIR-V Codegen

| Item | Effort |
|------|--------|
| 6 new execution models (RayGen, ClosestHit, AnyHit, Miss, Intersection, Callable) | Large |
| RT storage classes (RayPayloadKHR, HitAttributeKHR, etc.) | Medium |
| RT instructions (OpTraceRayKHR, OpReportIntersectionKHR, etc.) | Large |
| SPV_KHR_ray_tracing extension + capability | Small |

### P10.3 — RT Surface Expansion

| Item | Effort |
|------|--------|
| `_expand_surface_to_closest_hit()` | Medium |
| `_expand_surface_to_any_hit()` | Medium |
| `_expand_environment_to_miss()` | Medium |
| `_expand_procedural_to_intersection()` (SDF → intersection shader) | Medium |
| Callable shader dispatch for multi-material | Medium |

### Proposed Syntax

```lux
// Surface is UNCHANGED — same BRDF, works for both raster and RT
surface CopperMetal {
    brdf: pbr(vec3(0.95, 0.64, 0.54), 0.3, 0.9),
}

// New: environment for miss shader
environment HDRISky {
    color: sample(env_cubemap, ray_direction),
}

// Pipeline gets a mode switch
pipeline PathTracer {
    mode: raytrace,
    surface: CopperMetal,
    environment: HDRISky,
    max_bounces: 4,
}

// SDF primitives become intersection shaders
procedural MetaBalls {
    sdf: sdf_smooth_union(sdf_sphere(0.5), sdf_sphere(0.3), 0.2),
    surface: ChromeSurface,
}
```

**Key insight**: Every BRDF function we add in P5 pays off twice — it works in both rasterization and ray tracing. The `surface` abstraction decouples material math from rendering strategy.

---

### Phase 11: Metal Backend via SPIR-V Cross-Compilation

Create a Metal shading language backend by cross-compiling from SPIR-V using SPIRV-Cross. Implementation target: macOS agent with Xcode/Metal toolchain.

- Integrate SPIRV-Cross as a post-compilation step (SPIR-V → MSL)
- Handle Metal-specific descriptor set mapping (argument buffers, texture indices)
- Metal Shading Language differences: no separate sampler/texture in older Metal, different coordinate systems
- RT support via Metal ray tracing API (intersection functions vs Vulkan closest_hit)
- Test with Metal Performance Shaders for IBL preprocessing
- Target: separate Mac agent for CI/testing since Metal requires macOS

### Phase 12: Official glTF PBR Extensions in Engine Materials

Extend the three engine runtimes (Python/wgpu, C++/Vulkan, Rust/ash) to detect and bind glTF material extensions at load time:

- Parse `KHR_materials_clearcoat`, `KHR_materials_sheen`, `KHR_materials_transmission`, `KHR_materials_volume`, `KHR_materials_ior`, `KHR_materials_specular`, `KHR_materials_iridescence`, `KHR_materials_emissive_strength`, `KHR_materials_unlit` from glTF JSON
- Map detected extensions to Lux feature flags for permutation selection
- Load extension-specific textures (clearcoat normal, sheen color LUT, transmission texture)
- Automatic pipeline permutation selection: engine reads manifest JSON, matches asset extensions → picks correct compiled shader variant
- Validation: render Khronos sample models with all extensions enabled

### Phase 13: Mesh Shader Support ✅ COMPLETE

Mesh shader pipeline mode alongside rasterization and ray tracing. Meshlet-based geometry processing with GPU-driven culling, task/mesh shader stages, and compile-time `--define` parameters for hardware-adaptive meshlet sizing.

- New execution models: `task` (amplification) and `mesh` shader stages
- SPIR-V codegen for `OpEmitMeshTasksEXT`, `OpSetMeshOutputsEXT`
- Geometry declaration expansion to mesh shaders (meshlet-based vertex processing)
- Surface declaration works unchanged -- only the geometry stage changes
- Pipeline mode: `mode: mesh_shader` in pipeline declaration
- `--define key=value` compile-time parameter system for hardware-adaptive limits (`max_vertices`, `max_primitives`, `workgroup_size`)
- New built-in variables: `local_invocation_id`, `local_invocation_index`, `workgroup_id`, `num_workgroups`, `global_invocation_id`
- New built-in functions: `set_mesh_outputs(uint, uint)`, `emit_mesh_tasks(uint, uint, uint)`
- New stage declarations: `mesh_output` (per-vertex/per-primitive outputs), `task_payload` (task-to-mesh data)
- Requires `SPV_EXT_mesh_shader` capability and `VK_EXT_mesh_shader` in engines
- Supported in C++ and Rust engines only (Python/wgpu does not support mesh shaders)
- Data-driven approach: engine queries hardware, builds meshlets, compiles shader with matching limits
- Meshlet generation utility for glTF assets (offline preprocessing step)

### Phase 14: Gaussian Splatting Representation

Discuss and design how Gaussian splats should be represented in Lux, then implement:

- Research: investigate how 3D Gaussian splatting rendering pipelines work (point-based rendering, alpha blending, spherical harmonics for view-dependent color)
- Design: determine whether Gaussian splats fit as a new `geometry` type (e.g., `mode: gaussian_splat`), a new pipeline mode, or a new declaration type
- Key technical decisions: SH coefficient storage, covariance matrix representation, tile-based sorting/rasterization approach
- Integration with existing features system for optional SH bands, opacity culling, etc.
- Implementation: splat sorting compute shader, tile-based rasterizer, SH evaluation
- Target: real-time rendering of pre-trained Gaussian splat scenes (.ply format)

### Phase 15: BRDF & Layer Visualization ✅ COMPLETE

GPU-rendered visualization of BRDF functions as fullscreen Lux fragment shaders. Each shader plots BRDF functions using grid cells, `smoothstep` anti-aliased curves, and `fract`/`floor` cell selection. A Python tool compiles and renders them into a composite report.

**New visualization shaders (`examples/`):**

| Shader | Content |
|--------|---------|
| `viz_transfer_functions.lux` | 2x3 grid: Fresnel (Schlick), GGX NDF, Smith G, Charlie NDF, Lambert vs Burley, conductor Fresnel |
| `viz_brdf_polar.lux` | 2x2 polar lobe plots: GGX specular, Lambert diffuse, sheen, PBR composite |
| `viz_param_sweep.lux` | Viridis heatmaps: roughness × metallic, roughness × NdotV |
| `viz_furnace_test.lux` | White furnace test: hemisphere integration with 16 Fibonacci samples (energy ≤ 1.0 = green, > 1.0 = red) |
| `viz_layer_energy.lux` | Stacked area chart: per-layer energy (diffuse, specular, coat, sheen) vs viewing angle |

**New tools:**

| File | Purpose |
|------|---------|
| `tools/visualize_brdf.py` | CLI: compile all viz shaders, render to PNG, composite into report |
| `tests/test_brdf_visualization.py` | 10 tests: compilation + pixel validation for all 5 shaders |

**CLI usage:**
```bash
python -m tools.visualize_brdf --composite          # Full report
python -m tools.visualize_brdf --shader transfer     # Single shader
python -m tools.visualize_brdf --skip-compile         # Use cached SPIR-V
```

### Phase 16: AI Features for Lux

Design and implement AI-powered authoring capabilities for the Lux ecosystem — from material generation to intelligent shader assistance.

#### 16.1 — Image/Video-to-Shader Generation

- **Material capture**: Given a photograph of a real-world material (wood grain, brushed metal, wet concrete), use a vision model to estimate PBR parameters (albedo, roughness, metallic, normal characteristics) and generate a complete `.lux` surface declaration
- **Video-to-animation**: Analyze video of dynamic materials (flowing water, flickering fire) to generate time-varying Lux shaders with appropriate noise/animation parameters
- **Reference matching**: Given a target screenshot or reference image, iteratively refine a Lux shader until the rendered output approximates the reference (differentiable rendering + AI optimization loop)

#### 16.2 — Prompt-Based Material Generation

- **Natural language → Lux**: Extend the existing `--ai` CLI to handle complex material descriptions: "weathered copper with verdigris patina and rain droplets" → complete Lux file with appropriate layers, textures, and parameters
- **Style transfer prompts**: "Make this material look more worn / more glossy / more alien" → AI suggests parameter modifications or additional layers
- **Scene-aware generation**: "Generate materials for a medieval tavern scene" → batch generation of complementary materials (wood, stone, metal, fabric, candle wax)

#### 16.3 — AI Skills & Markdown Specification

- **Skill system**: Define reusable AI capabilities as markdown skill files (`.md`) that teach Claude how to work with specific Lux patterns:
  - `skills/pbr-authoring.md` — PBR material authoring best practices, common parameter ranges, physically plausible constraints
  - `skills/layer-composition.md` — How to compose layers for specific visual effects (wet surfaces, car paint, skin, fabric)
  - `skills/optimization.md` — Performance-aware shader authoring, LOD strategies, approximation selection
  - `skills/debugging.md` — How to diagnose rendering artifacts, common BRDF mistakes, energy conservation violations
- **Context-aware assistance**: AI understands the full Lux grammar, stdlib, and current project state to provide targeted suggestions when authoring shaders
- **Validation & critique**: AI reviews a `.lux` file and identifies potential issues: non-physical parameter combinations, energy conservation violations, missing layers for realism, performance bottlenecks

#### 16.4 — Training Data & Fine-Tuning Pipeline

- **Synthetic dataset expansion**: Generate thousands of Lux shader variants with corresponding rendered images for vision-language model training
- **BRDF parameter estimation dataset**: Pairs of (material photo, ground-truth PBR parameters) for training the image-to-shader pipeline
- **Evaluation benchmark**: Standardized test suite of material descriptions → expected shader quality metrics (PSNR against reference, parameter accuracy, energy conservation)

### Phase 17: Light & Shadow Management System

A comprehensive, declarative light and shadow system that integrates with Lux's surface/pipeline/schedule architecture. Light sources become first-class language constructs; the compiler automatically generates the required shadow passes, culling structures, and volumetric effects.

#### 17.1 — Declarative Light Types in Lux

Define lights as named declarations with physically-based parameters:

```lux
light SunLight {
    type: directional,
    direction: vec3(0.5, -1.0, 0.3),
    color: vec3(1.0, 0.95, 0.9),
    intensity: 100000.0,  // lux
    shadow: cascaded { cascades: 4, split_lambda: 0.75 },
}

light TorchLight {
    type: point,
    position: vec3(2.0, 3.0, 1.0),
    color: vec3(1.0, 0.6, 0.2),
    intensity: 800.0,  // lumens
    radius: 10.0,
    shadow: omnidirectional { resolution: 512 },
}

light Flashlight {
    type: spot,
    position: vec3(0.0, 1.0, 0.0),
    direction: vec3(0.0, 0.0, -1.0),
    color: vec3(1.0, 1.0, 1.0),
    intensity: 1200.0,
    inner_cone: 0.35,  // radians
    outer_cone: 0.55,
    shadow: perspective { resolution: 1024, filter: pcss },
}

light PanelLight {
    type: area,
    shape: rect { width: 2.0, height: 1.0 },
    position: vec3(0.0, 4.0, 0.0),
    normal: vec3(0.0, -1.0, 0.0),
    color: vec3(1.0, 1.0, 1.0),
    intensity: 5000.0,
}
```

Supported light types:

| Type | Description | Shadow Strategy |
|------|-------------|-----------------|
| `directional` | Parallel rays (sun, moon) | Cascaded shadow maps (CSM) |
| `point` | Omnidirectional emitter | Cubemap shadow maps |
| `spot` | Cone-shaped emitter with inner/outer angles | Perspective shadow maps |
| `area` (rect) | Rectangular area emitter | RT shadows or LTC approximation |
| `area` (disk) | Disk-shaped area emitter | RT shadows or LTC approximation |
| `area` (sphere) | Spherical area emitter | RT shadows or LTC approximation |
| `ies` | Measured photometric profile | Inherits from point/spot shadow |

IES profile lights reference an external photometric data file:

```lux
light StudioFixture {
    type: ies,
    profile: "fixture.ies",
    position: vec3(0.0, 3.0, 0.0),
    intensity: 2400.0,
}
```

#### 17.2 — Shadow Mapping System

The compiler generates the appropriate shadow passes based on each light's `shadow` configuration. Shadow maps are allocated and managed automatically by the schedule system.

**Cascaded Shadow Maps (CSM) for directional lights:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cascades` | Number of cascade splits (1–8) | 4 |
| `split_lambda` | Log/uniform split interpolation factor | 0.75 |
| `resolution` | Per-cascade shadow map resolution | 2048 |
| `stabilize` | Texel-snapping to eliminate shimmering | true |

**Omnidirectional cubemap shadows for point lights:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `resolution` | Per-face resolution | 512 |
| `near` / `far` | Depth range | 0.1 / light radius |

**Perspective shadow maps for spot lights:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `resolution` | Shadow map resolution | 1024 |
| `near` / `far` | Depth range | auto from cone |

#### 17.3 — Shadow Filtering

Each shadow configuration accepts a `filter` parameter selecting the filtering technique:

| Filter | Description | Quality | Cost |
|--------|-------------|---------|------|
| `hard` | Single-sample depth test | Lowest | Cheapest |
| `pcf` | Percentage-closer filtering (NxN kernel) | Medium | Low |
| `pcss` | Percentage-closer soft shadows (variable penumbra) | High | Medium |
| `vsm` | Variance shadow maps (two-moment, Chebyshev) | Medium | Low (filterable) |
| `msm` | Moment shadow maps (4-moment, hamburger) | High | Medium |

Syntax:

```lux
shadow: cascaded {
    cascades: 4,
    filter: pcss { light_size: 0.02, blocker_samples: 16, pcf_samples: 32 },
}
```

#### 17.4 — Shadow Bias Strategies

Bias configuration prevents shadow acne and peter-panning artifacts:

```lux
shadow: perspective {
    bias: {
        constant: 0.005,
        slope_scale: 1.5,
        normal_offset: 0.02,
        receiver_plane: true,
    },
}
```

| Strategy | Description |
|----------|-------------|
| `constant` | Fixed depth offset added to shadow comparison |
| `slope_scale` | Bias scaled by the surface slope relative to light |
| `normal_offset` | Offsets sample position along the surface normal |
| `receiver_plane` | Computes per-fragment optimal bias from receiver plane (Holländer 2011) |

#### 17.5 — Light Culling

For scenes with many lights, the compiler generates tiled or clustered light culling compute passes:

```lux
schedule LightCull {
    method: clustered {
        tile_size: 16,
        depth_slices: 24,
        max_lights_per_cluster: 128,
    },
}
```

| Method | Description | Use Case |
|--------|-------------|----------|
| `tiled` | 2D screen-space tiles, light-AABB intersection | Medium light counts (< 256) |
| `clustered` | 3D froxel grid (tiles + depth slices), light-AABB test | High light counts (1000+) |

The culling pass outputs a per-tile/cluster light index list consumed by the lighting pass. Integrates with the deferred pipeline (Phase 9) for maximum efficiency.

#### 17.6 — Volumetric Lighting & God Rays

Volumetric light scattering for atmospheric effects:

```lux
light SunLight {
    type: directional,
    volumetric: {
        density: 0.02,
        scattering: 0.7,   // Henyey-Greenstein g parameter
        samples: 64,
        max_distance: 100.0,
    },
}
```

Implementation approaches:

| Technique | Description | Pipeline Mode |
|-----------|-------------|---------------|
| Ray-marched froxel volumes | 3D froxel grid with temporal reprojection | Raster / Deferred |
| Screen-space ray marching | Per-pixel rays through shadow map | Raster |
| RT volumetric scattering | Native ray tracing with medium sampling | Ray Tracing |

#### 17.7 — Soft Shadows with Contact Hardening

Physically-based soft shadows where penumbra width varies with distance from the occluder (contact hardening). This is the default behavior of PCSS but is also exposed as an explicit parameter for other techniques:

```lux
shadow: cascaded {
    filter: pcss {
        light_size: 0.04,
        contact_hardening: true,
        min_penumbra: 0.001,
        max_penumbra: 0.1,
    },
}
```

For area lights in ray tracing mode, soft shadows are computed natively by sampling the light's surface area — no shadow maps required.

#### 17.8 — Integration with Surface/Pipeline/Schedule System

Lights are referenced in `pipeline` declarations and the compiler generates all required shadow passes:

```lux
pipeline ForwardPBR {
    mode: raster,
    geometry: SceneGeometry,
    surface: GltfPBR,
    lights: [SunLight, TorchLight, PanelLight],
    light_cull: LightCull,
    tonemap: aces,
}
```

Compiler responsibilities:

| Step | Description |
|------|-------------|
| **Shadow pass generation** | For each shadowed light, emit a depth-only render pass with appropriate projection (ortho for CSM, perspective for spot, cube for point) |
| **Shadow atlas allocation** | Pack multiple shadow maps into a single atlas texture, generate UV remap metadata |
| **Light uniform buffer layout** | Auto-generate a structured light UBO/SSBO with position, direction, color, shadow matrix per light |
| **Culling compute dispatch** | If `light_cull` is specified, emit the tiled/clustered culling compute pass before the main lighting pass |
| **Volumetric pass insertion** | If any light has `volumetric` config, insert the volumetric scattering pass into the schedule |
| **Shadow sampler binding** | Auto-bind shadow map samplers with comparison mode and appropriate filtering |

The schedule system (Phase 3) ensures correct pass ordering: shadow passes execute before the main color pass, culling before lighting, and volumetric compositing after the main pass.

#### 17.9 — Light Probes & Light Propagation Volumes

Indirect lighting approximations for global illumination:

```lux
schedule IndirectLighting {
    method: light_probes {
        probe_grid: ivec3(8, 4, 8),
        probe_spacing: 2.0,
        sh_bands: 2,       // L2 spherical harmonics
        update: per_frame,  // or on_demand
    },
}

schedule IndirectLighting {
    method: lpv {
        grid_resolution: ivec3(32, 32, 32),
        propagation_steps: 8,
        occlusion: true,
    },
}
```

| Technique | Description | Quality | Cost |
|-----------|-------------|---------|------|
| **Irradiance probes** | SH-encoded irradiance at grid positions | Medium | Low |
| **Reflection probes** | Filtered cubemaps for specular IBL at discrete positions | Medium | Medium |
| **Light propagation volumes (LPV)** | Real-time GI via SH propagation on a 3D grid (Kaplanyan 2009) | Medium | Medium |
| **DDGI** (future) | Dynamic diffuse GI with ray-traced probe updates | High | High |

Probes and LPV integrate with the existing IBL layer — when probe data is available, the IBL layer blends between baked cubemaps and real-time probe interpolation.

---

## Implementation Priority

| Priority | What | Status |
|----------|------|--------|
| **P0** | Shader playground (Phase 0) | ✅ Complete |
| **P1** | Radiometric types (1.1) | ✅ Complete |
| **P1** | BRDF type + stdlib (1.2, 2.1) | ✅ Complete |
| **P1** | `surface` declaration (1.3) | ✅ Complete |
| **P2** | `geometry` + `pipeline` (1.4, 1.5) | ✅ Complete |
| **P2** | SDF + noise + color stdlib (2.2–2.4) | ✅ Complete |
| **P2** | Import system | ✅ Complete |
| **P3** | Constant folding + compiler intelligence | ✅ Complete |
| **P4** | Autodiff (forward-mode) | ✅ Complete |
| **P4** | GLSL-to-Lux transpiler | ✅ Complete |
| **P4** | AI generation pipeline | ✅ Complete |
| **P4** | Training data pipeline | ✅ Complete |
| **P5.1** | Critical built-in gaps (refract, atan2, inversesqrt, mod) | ✅ Complete |
| **P5.2** | stdlib expansions (clearcoat, sheen, anisotropy, diffuse models, color) | ✅ Complete |
| **P5.3** | Advanced materials (transmission, iridescence, dispersion, texture) | ✅ Complete |
| **P5.5** | Layered surface system (`layers [...]`, energy conservation, RT/raster unification, `samplerCube`, `--pipeline` CLI) | ✅ Complete |
| **P5.6** | Compile-time features & shader permutations (`features {}`, `--all-permutations`, manifest generation) | ✅ Complete |
| **P6** | Coat & sheen layers (clearcoat, sheen as composable layers) | ✅ Complete |
| **P7** | Transmission layer (microfacet BTDF, volume attenuation) | ✅ Complete |
| **P8** | `@layer` custom functions (user-defined layer extensibility) | ✅ Complete |
| **P9** | Deferred pipeline mode (`--pipeline deferred`, G-buffer pass) | Planned |
| **P10** | Ray tracing pipeline — full (RT stages, SPIR-V codegen, surface→RT expansion) | ✅ Complete |
| **P11** | Metal backend via SPIR-V cross-compilation (SPIRV-Cross → MSL) | ✅ Complete |
| **P12** | Official glTF PBR extensions in engine materials (auto-detect, permutation selection) | ✅ Complete |
| **P13** | Mesh shader support (`task`/`mesh` stages, meshlet-based geometry, `--define` compile-time parameters) | ✅ Complete |
| **P14** | Gaussian splatting representation (splat sorting, tile-based rasterizer, SH evaluation) | Planned |
| **P15** | BRDF & layer visualization (lobe plots, transfer function graphs, energy conservation tests) | ✅ Complete |
| **P16** | AI features for Lux (image-to-shader, prompt-based generation, AI skills, training pipeline, critique, style transfer, batch generation, video-to-animation, reference matching, benchmark) | ✅ Complete |
| **P17.1** | Lighting block (`lighting` declarations, `directional()` + `ibl()` layers, IBL migration from surface to lighting, backward compat) | ✅ Complete |
| **P17.2** | Multi-light + shadows (`multi_light()` layer, compile-time unrolled N-light loop, `LightData` + `ShadowEntry` SSBOs, `sampler2DArray` + `samplerCubeArray` types, `shadow.lux` stdlib, shadow map infrastructure in all engines) | ✅ Complete |
| **P17.3** | Shadow sampling & validation (shadow stdlib, `+shadows` permutation, PCF4 filtering, procedural lighttest scene, shadow direction fix) | ✅ Complete |
| **P17.4+** | Advanced shadow & light management (CSM, cubemap/perspective shadows, PCSS/VSM/MSM filtering, tiled/clustered culling, volumetric lighting, light probes, LPV) | Planned |
| **P18** | Material property pipeline (`properties` block, Material UBO, `Material.field` access, engine wiring) | ✅ Complete (18.1) |
| **P19** | Linux support (build scripts, path handling, CI) | Planned |
| **P20** | Validation & debugging (debug_print, assert, @[debug], semantic types, NaN analysis, OpMemberName, debug utils labels) | ✅ Complete |
| **P22** | Loops & control flow (`for`/`while` loops, `break`/`continue`, `@[unroll]`, native integer arithmetic) | ✅ Complete |
| **P23** | GPU compute shaders (compute stage, RW SSBOs, storage images, barriers, shared memory, atomics, dispatch) | ✅ Complete (23.1 + 23.2) |
| **P24** | Shader hot-reload (`--watch` file watcher, live recompile, sentinel protocol for engine hot-swap) | ✅ Complete |
| **P25** | Performance optimization (`-O` spirv-opt integration, algebraic identity folding) | ✅ Complete (25.1) |

---

### Phase 18: Material Property Pipeline ✅ COMPLETE (18.1)

#### 18.1 — Material Uniforms from glTF Properties ✅

Implemented `properties` block syntax in surface declarations with full compiler and engine support:

- **`properties` block syntax**: `properties Material { field: type = default, ... }` inside `surface` declarations
- **Compiler pipeline**: grammar (`surface_properties`, `properties_field` rules), AST (`PropertiesField`, `PropertiesBlock` nodes), tree builder, surface expander (emits UBO for fragment + closest-hit + mesh stages), SPIR-V builder (`FieldAccess(VarRef(block_name), field)` via `OpAccessChain`), reflection
- **`Material.field` qualified access**: layer expressions reference properties via qualified names (e.g., `Material.roughness_factor`)
- **Reflection JSON**: emits default values from properties field initializers
- **Engine integration**: all three engines (Python/wgpu, C++/Vulkan, Rust/ash) wire glTF material properties to Material UBO
- **MaterialUBOData struct**: 80 bytes, std140 layout, shared across all three engines
- **Updated `gltf_pbr_layered.lux`**: properties Material block with 11 PBR fields, layer expressions use `Material.field`
- **Tests**: 10 new tests (parse, AST, compilation, reflection offsets/defaults, RT, backward compat), all pass; 435 total tests passing

#### 18.2 — Bindless Descriptors for Multi-Material Scenes

Use `VK_EXT_descriptor_indexing` (descriptor binding without update after bind) to support scenes with many materials and textures efficiently:

- Single large descriptor array for all textures in the scene
- Material index passed via push constant or instance data
- Material properties stored in an SSBO (structured buffer of material structs)
- Per-draw-call: bind material index → shader indexes into texture array + material SSBO
- Eliminates per-material descriptor set switching (major performance win for complex scenes)

This is the foundation for GPU-driven rendering where draw calls are batched and materials are selected per-primitive via indirection.

---

### Phase 19: Linux Support

Add first-class Linux build and run support for the C++ and Rust engines.

| Item | Description |
|------|-------------|
| Shell scripts | `compile_mesh.sh`, `run_mesh_headless_cpp.sh`, etc. (Linux equivalents of `.bat` files) |
| CMake paths | Platform-agnostic path handling in CMakeLists.txt |
| Vulkan SDK detection | Find Vulkan headers/libs on Linux (`find_package(Vulkan)` already works) |
| GLFW on Linux | Ensure X11/Wayland support via system GLFW or bundled |
| CI pipeline | GitHub Actions Linux build + test matrix |
| Asset paths | Forward-slash path normalization throughout engine code |

---

### Phase 20: Validation & Debugging ✅ COMPLETE

First-class debug instrumentation in the Lux language and validation infrastructure improvements across the compiler and engines. All debug features compile to zero instructions in release builds.

#### 20.1 — Fix ALL spirv-val Errors ✅

- All shader configurations pass `spirv-val` without `--no-validate`
- Removed `--no-validate` from all batch files and test scripts
- Created `validate_all.py` full compilation matrix script

#### 20.2 — Runtime Validation Control ✅

- `--validation` CLI flag for C++ and Rust engines (force Vulkan validation layers in release)

#### 20.3 — OpMemberName Emission ✅

- SPIR-V `OpMemberName` for uniform blocks, push constants, `gl_PerVertex`, and bindless material structs
- Enables readable struct field names in RenderDoc and spirv-cross

#### 20.4 — VK_EXT_debug_utils Labels ✅

- RenderDoc markers in all renderers (C++ and Rust): "Raster Pass" (green), "RT Trace" (red), "Mesh Dispatch" (blue)
- Function pointer loading + helper methods in VulkanContext

#### 20.5 — `debug_print` Statement ✅

- Syntax: `debug_print("roughness={} metallic={}", roughness, metallic);`
- SPIR-V: `NonSemantic.DebugPrintf` extension + `SPV_KHR_non_semantic_info`
- Stripped to zero instructions in release builds

#### 20.6 — `assert` Statement ✅

- Syntax: `assert(roughness >= 0.0, "roughness out of range");`
- SPIR-V: conditional branch → debugPrintf on failure, continues (no shader kill)
- Stripped to zero instructions in release builds

#### 20.7 — `@[debug]` Blocks ✅

- Syntax: `@[debug] { debug_print(...); assert(...); }`
- Entire block stripped at compile time in release (not just skipped — zero instructions)

#### 20.8 — Semantic Type Wrappers ✅

- Syntax: `type strict WorldPos = vec3;` — prevents mixing coordinate spaces at compile time
- `SemanticType` class wrapping base type; `WorldPos` and `ViewPos` are NOT interchangeable
- Builtins (`normalize`, `dot`, `cross`, etc.) accept semantic types transparently
- Zero SPIR-V overhead — purely compiler-side check

#### 20.9 — Debug Visualization Stdlib ✅

- New `luxc/stdlib/debug.lux`: `debug_normal`, `debug_depth`, `debug_heatmap`, `debug_index_color`, `debug_uv_checker`
- New builtins: `any_nan(x)` → `OpIsNan` + `OpAny`, `any_inf(x)` → `OpIsInf` + `OpAny`

#### 20.10 — Static NaN/Division Warnings ✅

- `--warn-nan` CLI flag for static analysis
- Detects unguarded division, `sqrt` of negative, `normalize` of zero-length, `pow` with negative base, `log`/`log2` of non-positive
- Emits Python warnings (not errors — doesn't break compilation)

#### 20.11 — Tests ✅

- 26 new tests across `test_debug_features.py`, `test_codegen.py`, `test_e2e.py`
- 461 total tests passing (zero regressions)

---

### Phase 18: Material Property Pipeline + Bindless Descriptors ✅ COMPLETE (18.1)

A `properties` block in `surface` declarations — an abstract data source for runtime material parameters. The compiler generates a UBO for standard mode; the same syntax supports SSBO+push-constant for bindless mode (18.2, future).

**New syntax:**
```lux
surface GltfPBR {
    sampler2d base_color_tex,
    properties Material {
        base_color_factor: vec4 = vec4(1.0, 1.0, 1.0, 1.0),
        roughness_factor: scalar = 1.0,
        metallic_factor: scalar = 1.0,
        ior: scalar = 1.5,
    },
    layers [
        base(albedo: sample(base_color_tex, uv).xyz * Material.base_color_factor.xyz,
             roughness: Material.roughness_factor,
             metallic: Material.metallic_factor),
    ]
}
```

**18.1 deliverables (standard UBO mode):**
- Grammar: `surface_properties` and `properties_field` rules in `lux.lark`
- AST: `PropertiesField`, `PropertiesBlock` nodes; `SurfaceDecl.properties` field
- Type system: `UniformBlockType` for qualified field access (`Material.field`)
- Surface expander: emits UBO from properties block (fragment + closest-hit + mesh)
- SPIR-V builder: handles `FieldAccess(VarRef(block_name), field)` → `OpAccessChain`
- Reflection: emits default values from properties field initializers
- `gltf_pbr_layered.lux`: properties Material block with 11 PBR fields, layer expressions use `Material.field`
- All three engines: MaterialUBOData struct (80 bytes std140), buffer creation, descriptor binding by name
- 10 new tests (parse, AST, compilation, reflection offsets/defaults, RT, backward compat)
- 435 total tests passing (zero regressions)

**18.2 (future): Bindless mode** — same `properties` syntax, `bindless: true` in pipeline → SSBO[material_idx] + texture arrays + push constant.

---

### Phase 17.1: Lighting Block + IBL Migration ✅ COMPLETE

Separated illumination configuration from material response by adding a new `lighting` top-level block to the language.

**New syntax:**
```lux
lighting SceneLighting {
    samplerCube env_specular,
    samplerCube env_irradiance,
    sampler2d brdf_lut,

    properties Light {
        light_dir: vec3 = vec3(0.0, -1.0, 0.0),
        view_pos: vec3 = vec3(0.0, 0.0, 3.0),
    },

    layers [
        directional(direction: Light.light_dir,
                    color: vec3(1.0, 0.98, 0.95)),
        ibl(specular_map: env_specular, irradiance_map: env_irradiance,
            brdf_lut: brdf_lut),
    ]
}

pipeline GltfForward {
    geometry: StandardMesh,
    surface: GltfPBR,
    lighting: SceneLighting,
    schedule: HighQuality,
}
```

**Deliverables:**
- **Grammar**: `lighting_decl` rule in `lux.lark` (reuses `surface_item`)
- **AST**: `LightingDecl` dataclass with name, members, samplers, layers, properties
- **Parser**: `lighting_decl()` transformer, routing in `start()`
- **Feature evaluator**: conditional sampler/layer stripping for lighting blocks
- **Surface expander**: lighting detection in all expansion paths (raster, RT, mesh, bindless); `directional()` layer replaces hardcoded light direction/color; `ibl()` layer migrated from surface to lighting with backward-compatible fallback
- **IBL migration**: env_specular, env_irradiance, brdf_lut moved from surface to lighting block; engines unaffected (same sampler names in reflection JSON)
- **Backward compatibility**: pipelines without `lighting:` member use legacy hardcoded Light UBO; `ibl()` in surface layers still works as fallback
- **Updated `gltf_pbr_layered.lux`**: lighting SceneLighting block, all three pipelines reference it
- **Tests**: 21 new tests (parsing, feature stripping, expansion, cross-block interaction); 497 total tests passing
- **Zero engine changes**: Light UBO field names and IBL sampler names unchanged in reflection JSON

**Future (P17.4+):** Full declarative light types, advanced shadow filtering (PCSS, VSM, MSM), tiled/clustered light culling, volumetric lighting, light probes.

---

### Phase 17.2: Multi-Light + Shadows ✅ COMPLETE

Extended the lighting system with runtime N-light evaluation and shadow mapping infrastructure.

**Compiler changes:**
- **`multi_light()` lighting layer**: compile-time unrolled N-light evaluation loop (default 16 iterations), each guarded by `if (i < light_count)`, reading from `LightData` SSBO (64 bytes/light)
- **`_emit_multi_light_loop()`**: shared helper across all 6 expansion paths (raster, RT, mesh × non-bindless, bindless)
- **`LightData` SSBO struct**: light_type, intensity, range, inner_cone, position, outer_cone, direction, shadow_index, color, _pad
- **`ShadowEntry` SSBO struct**: mat4 viewProjection, bias, normalBias, resolution, _pad (80 bytes)
- **`sampler2DArray` and `samplerCubeArray`** type support in grammar + SPIR-V codegen
- **`evaluate_light_direction()`** stdlib function: branchless directional/point/spot selection
- **`shadow.lux` stdlib**: `sample_shadow_basic`, `sample_shadow_pcf4`, `select_cascade`, `compute_shadow_uv`
- Backward compatible: `directional()` layer still works unchanged

**Engine changes (C++, Rust, Python):**
- `SceneLight` struct with type (directional/point/spot), position, direction, color, intensity, range, cone angles, castsShadow, shadowIndex
- `populateLightsFromGltf()` / `populate_lights_from_gltf()`: extract lights from `KHR_lights_punctual` with world-space transforms
- `packLightsBuffer()` / `pack_lights_buffer()`: GPU buffer packing (std430, 64 bytes/light)
- Shadow map infrastructure: 1024x1024 depth texture arrays (up to 8 layers), comparison samplers, depth-only render passes
- Shadow matrix computation: orthographic for directional, perspective for spot lights
- Reflection-driven detection: engines auto-detect multi-light and shadow support from shader reflection JSON
- Full descriptor binding for `shadow_maps` (sampler2DArray) and `shadow_matrices` (SSBO) across all render paths

**New files:**
- `luxc/stdlib/shadow.lux` — shadow sampling and cascade selection
- `examples/multi_light_demo.lux` — standalone multi-light demo
- Updated `examples/gltf_pbr_layered.lux` with `multi_light()` layers and `has_shadows` feature
- `tests/test_multi_light.py` — 40 new tests covering parsing, AST unrolling, SSBO generation, sampler arrays, backward compat, all 6 pipeline paths, shadow args
- 542 total tests passing (zero regressions)

---

### Phase 17.3: Shadow Sampling & Validation ✅ COMPLETE

Extended the shadow infrastructure with sampling functions, shader permutation support, and a procedural test scene for visual validation.

**Compiler changes:**
- **`shadow.lux` stdlib**: `sample_shadow_basic()` (hard shadows), `sample_shadow_pcf4()` (4-tap PCF), `select_cascade()`, `compute_shadow_uv()` — all consume `ShadowEntry` SSBO + `sampler2DArray`
- **`+shadows` permutation**: `has_shadows` feature flag gates shadow sampling in `gltf_pbr_layered.lux`; manifest generates both `base` and `base+shadows` SPIR-V variants
- **SceneLightUBO layout fix**: reordered to match std140 padding rules for multi-light rendering

**Engine changes (C++, Rust):**
- Shadow permutation selection: engines detect `castsShadow` on any light → select `+shadows` pipeline variant from manifest
- Light direction convention fix: shaders use direction-toward-light (`dot(N, dir) > 0` = lit); shadow matrix uses `lookAt(center + lightDir, center, up)` to match
- `groupMaterialsByFeatures()` ordering: `castsShadow` must be set BEFORE feature grouping to ensure correct permutation selection
- Depth buffer fix: `m_needsDepth` extended to recognize procedural scene types ("lighttest")

**Procedural lighttest scene:**
- Ground plane (10×10), floating cube, red/green light-marker spheres — all using `GltfVertex` (48-byte: pos + normal + uv + tangent)
- Geometry generators: `generatePlaneGltf()`, `generateCubeGltf()`, `generateSphereGltf()` in `scene.cpp`
- `loadProceduralTestScene()` in SceneManager: 4 materials, 4 meshes, 4 nodes, programmatic `GltfScene`
- `setupTestLights()`: directional sun (shadow-casting) + red/green point lights
- `--scene lighttest` CLI routing, `--no-ibl` flag, auto-camera at (6,4,6) looking at (0,1,0)
- Visual validation: colored light tinting visible, cube shadow clearly falls on ground plane

**Future (P17.4+):** Full declarative light types, advanced shadow filtering (PCSS, VSM, MSM), tiled/clustered light culling, volumetric lighting, light probes.

---

### Phase 22: OpenPBR Material Model Integration [DONE]

**Goal:** Add first-class support for the [OpenPBR Surface](https://github.com/adobe/openpbr-bsdf) material model (Adobe/ASWF, v1.1) as an alternative to the existing glTF PBR composition. Activated via `import openpbr;` — no grammar changes, no new keywords.

**Status:** All phases complete (P22.1–P22.8). Fixed transitive import resolution bug in `compiler.py` — imports now recurse before merging. P22.5 adds bindless OpenPBR uber-shader with extended SSBO struct, feature flags, and `_emit_openpbr_bindless_body()`. P22.6 adds schedule strategies with 4 new slots (`diffuse_model`, `fuzz_model`, `specular_fresnel`, `coat_fresnel`) and fast variants (`openpbr_direct_fast`, `openpbr_compose_fast`) for mobile/low-end targets. 35 new tests (1084 total), 4 ASWF reference examples, 17 stdlib functions.

**Why:** OpenPBR is becoming the industry standard material interchange format (USD, MaterialX, Blender, Maya, Substance). It defines a 36-parameter physically-based model with 9 component lobes, proper energy conservation via albedo-scaling, and features Lux currently lacks: coat darkening, F82-tint metals, energy-preserving Oren-Nayar, specular weight IOR modulation, fuzz (Zeltner LTC), subsurface scattering, and thin-film iridescence. Lux already implements many of the required BRDF primitives (`conductor_fresnel`, `iridescence_fresnel`, `anisotropic_ggx_ndf`, `oren_nayar_diffuse`, `volumetric_btdf`, `dispersion_ior`) but none are wired into the layer system.

**Design principle:** `import openpbr;` extends the set of recognized layer names, layer parameter schemas, and the composition function. The expander detects this import and dispatches to OpenPBR-specific code. Existing `import brdf;` surfaces work identically — zero breaking changes.

**User-facing syntax:**

```lux
import openpbr;

surface CarPaint {
    sampler2d color_tex,

    layers [
        base(color: sample(color_tex, uv).xyz, metalness: 0.0,
             diffuse_roughness: 0.2),
        specular(weight: 1.0, ior: 1.5, roughness: 0.3,
                 anisotropy: 0.0),
        coat(weight: 1.0, color: vec3(1.0, 0.98, 0.95),
             roughness: 0.05, ior: 1.6, darkening: 1.0),
        fuzz(weight: 0.0, color: vec3(1.0), roughness: 0.5),
        thin_film(weight: 0.3, thickness: 0.5, ior: 1.4),
        emission(luminance: 0.0, color: vec3(1.0)),
    ]
}

lighting SceneLighting {
    samplerCube env_specular,
    samplerCube env_irradiance,
    sampler2d brdf_lut,
    layers [
        directional(direction: vec3(0.0, -1.0, 0.0), color: vec3(1.0)),
        ibl(specular_map: env_specular, irradiance_map: env_irradiance,
            brdf_lut: brdf_lut),
    ]
}

pipeline CarPaintForward {
    geometry: StandardMesh,
    surface: CarPaint,
    lighting: SceneLighting,
}
```

---

#### OpenPBR Composition Tree (from the spec)

The full physical model, reduced to a linear lobe mixture:

```
f_PBR     = mix(f_transparent, f_surface, opacity)
f_surface = F*f_fuzz + (1 - F*E[f_fuzz]) * f_coated_base
f_coated  = C*f_coat + (1 - C*E[f_coat]) * T_coat * f_base_darkened
f_base    = mix(f_dielectric, f_conductor, metalness)
f_dielec  = f_spec_refl + (1 - E[f_spec_refl]) * f_dielec_transmission
f_dielec_T = mix(mix(f_diffuse, f_SSS, S), f_spec_trans, T)
```

Where `E[f]` = directional albedo (hemispherical reflectance) of lobe f, ensuring energy conservation at every layer boundary.

---

#### P22.1: `stdlib/openpbr.lux` — Core Math Functions

**File:** `luxc/stdlib/openpbr.lux`
**Imports:** `brdf`, `compositing`, `ibl`
**Functions to implement (15 new, from OpenPBR GLSL reference):**

| Function | Source | Formula |
|---|---|---|
| `openpbr_f82_tint(cos, f0, f82) -> vec3` | metal_brdf.glsl | Lazanyi-Szirmay-Kalos: `schlick - cos*(1-cos)^6*(1-f82)*schlick_bar/denom` clamped to [0,1]. Uses `mu_bar = 1/7`. Already have `conductor_fresnel()` in brdf.lux — verify equivalence and delegate or replace. |
| `openpbr_specular_ior_modulated(eta, spec_weight) -> scalar` | openpbr.frag.glsl | `F_s = ((eta-1)/(eta+1))^2`, `eps = sgn(eta-1)*sqrt(spec_weight*F_s)`, `eta' = (1+eps)/(1-eps)`. Maps specular_weight [0,∞) to physical IOR. |
| `openpbr_dielectric_fresnel(cos, eta, spec_weight) -> scalar` | openpbr.frag.glsl | Full dielectric Fresnel with `specular_weight` modulation and TIR handling. Delegates to exact Fresnel (not Schlick). |
| `openpbr_coat_ior_ratio(spec_ior, coat_ior, coat_weight) -> scalar` | openpbr.frag.glsl | `eta_sc = spec_ior/coat_ior; if eta_sc<1 eta_sc=1/eta_sc; mix(spec_ior, eta_sc, coat_weight)`. Substrate IOR seen through coat. |
| `openpbr_coat_roughening(base_r, coat_r, coat_weight) -> scalar` | OpenPBR spec §4.6 | `mix(base_r, min(1, (base_r^4 + 2*coat_r^4)^0.25), coat_weight)`. Coat broadens substrate lobes. |
| `openpbr_coat_absorption(coat_color, coat_weight, cos_theta) -> vec3` | openpbr.frag.glsl | `mix(vec3(1), pow(coat_color, vec3(1/cos_theta)), coat_weight)`. View-dependent coat tinting. |
| `openpbr_fresnel_avg(eta) -> scalar` | openpbr.frag.glsl | d'Eon exact hemispherical Fresnel average. 15-line formula with `f_perp`, `f_para` terms. Needed for coat darkening. |
| `openpbr_coat_darkening(base_albedo, coat_ior, base_roughness, coat_weight, coat_darkening, spec_weight, spec_ior, metalness, cos_theta) -> vec3` | openpbr.frag.glsl | `K = mix(K_s, K_r, roughness)`, `Delta = (1-K)/(1-E_base*K)`, `mix(vec3(1), Delta, coat_weight*coat_darkening)`. Internal reflection compensation. |
| `openpbr_eon_diffuse(albedo, roughness, n_dot_l, n_dot_v, v_dot_l) -> vec3` | openpbr.frag.glsl | Energy-preserving Oren-Nayar (Fujii): single-scatter `f_ss` + multi-scatter `f_ms` with compensation term. Replaces Lambert. |
| `openpbr_eon_albedo(cos, roughness) -> scalar` | openpbr.frag.glsl | `E_FON_approx`: directional albedo of EON diffuse. Needed for energy conservation. |
| `openpbr_specular_albedo(cos, roughness, f0) -> scalar` | Karis fit | Approximate directional albedo of GGX specular. Needed for diffuse weighting. |
| `openpbr_coat_albedo(cos, coat_roughness, coat_ior) -> scalar` | openpbr.frag.glsl | Approximate directional albedo of coat GGX. Used for substrate attenuation. |
| `openpbr_fuzz_albedo(cos, fuzz_roughness) -> scalar` | openpbr.frag.glsl | Approximate directional albedo of fuzz lobe. Used for substrate attenuation. |
| `openpbr_fuzz_brdf(n, v, l, fuzz_color, fuzz_roughness) -> vec3` | fuzz_brdf.glsl | Zeltner 2022 LTC microflake model, or Charlie NDF as real-time approximation (config via schedule). |
| `openpbr_compose(direct_lit, n, v, l, <all OpenPBR params>, ibl_ambient, coat_ibl) -> vec3` | New | Top-level composition function — implements the full tree above. ~60 lines. |

**Key design decisions:**

1. `openpbr_compose()` takes ~30 parameters (OpenPBR's 36 minus geometry/normals + IBL inputs). Each subsystem is disabled by its weight parameter (0.0 = off). Zero-weight paths compile to no-ops after constant folding.

2. `openpbr_fuzz_brdf` uses Charlie NDF (reuses `charlie_ndf` from brdf.lux) as real-time approximation. Full Zeltner LTC available via schedule: `fuzz_model: zeltner`.

3. `openpbr_eon_diffuse` replaces Lambert. The existing `oren_nayar_diffuse()` in brdf.lux is the qualitative (no energy compensation) version — the EON model adds multi-scatter compensation for full energy preservation.

4. All albedo functions (`openpbr_*_albedo`) are analytical approximations (polynomial fits from the OpenPBR reference), not LUT lookups. This is the standard approach for real-time — the pathtracer uses exact integration.

**Estimated size:** ~350 lines of Lux code.

---

#### P22.2: Wire Existing Unwired Functions

Six functions in `brdf.lux` are already implemented but not used anywhere. Wire them:

| Function | Wire Into | How |
|---|---|---|
| `conductor_fresnel(f0, f82, cos)` | `openpbr_compose` metal path | Called when `metalness > 0` instead of Schlick |
| `iridescence_fresnel(...)` | `openpbr_compose` thin_film path | Called when `thin_film_weight > 0`, modulates base Fresnel |
| `anisotropic_ggx_ndf(...)` + `anisotropic_v_ggx(...)` | `openpbr_compose` specular/coat paths | Called when `anisotropy > 0` instead of isotropic GGX |
| `oren_nayar_diffuse(...)` | Superseded by `openpbr_eon_diffuse` | EON is the improved version; keep old for glTF compat |
| `volumetric_btdf(...)` | `openpbr_compose` transmission path | Called when `transmission_weight > 0` and not thin-walled |
| `dispersion_ior(...)` / `dispersion_f0(...)` | `openpbr_compose` transmission path | Per-channel IOR when dispersion > 0 |

**No changes to brdf.lux** — the existing functions stay as-is. `openpbr.lux` calls them directly.

---

#### P22.3: Expander — OpenPBR Layer Dispatch

**File:** `luxc/expansion/surface_expander.py`

**Detection** (in `expand_surfaces`, ~line 72):

```python
_use_openpbr = any(imp.module_name == "openpbr" for imp in module.imports)
```

**New builtin layer names** (registered when OpenPBR detected):

```python
_OPENPBR_LAYER_NAMES = frozenset({
    "base", "specular", "transmission", "subsurface",
    "coat", "fuzz", "thin_film", "emission", "normal_map",
})
```

When `_use_openpbr`, validation uses `_OPENPBR_LAYER_NAMES` instead of `_BUILTIN_LAYER_NAMES`.

**Layer parameter schemas** (used for extraction in `_generate_layered_main`):

| Layer | OpenPBR Parameters | Defaults |
|---|---|---|
| `base` | `color: vec3`, `metalness: scalar`, `diffuse_roughness: scalar`, `weight: scalar` | `(0.8,0.8,0.8)`, `0.0`, `0.0`, `1.0` |
| `specular` | `weight: scalar`, `color: vec3`, `roughness: scalar`, `anisotropy: scalar`, `ior: scalar` | `1.0`, `(1,1,1)`, `0.3`, `0.0`, `1.5` |
| `transmission` | `weight: scalar`, `color: vec3`, `depth: scalar`, `scatter: vec3`, `scatter_anisotropy: scalar`, `dispersion_scale: scalar`, `dispersion_abbe: scalar` | `0.0`, `(1,1,1)`, `0.0`, `(0,0,0)`, `0.0`, `0.0`, `20.0` |
| `subsurface` | `weight: scalar`, `color: vec3`, `radius: scalar`, `radius_scale: vec3`, `anisotropy: scalar` | `0.0`, `(0.8,0.8,0.8)`, `1.0`, `(1,0.5,0.25)`, `0.0` |
| `coat` | `weight: scalar`, `color: vec3`, `roughness: scalar`, `anisotropy: scalar`, `ior: scalar`, `darkening: scalar` | `0.0`, `(1,1,1)`, `0.0`, `0.0`, `1.6`, `1.0` |
| `fuzz` | `weight: scalar`, `color: vec3`, `roughness: scalar` | `0.0`, `(1,1,1)`, `0.5` |
| `thin_film` | `weight: scalar`, `thickness: scalar`, `ior: scalar` | `0.0`, `0.5`, `1.4` |
| `emission` | `luminance: scalar`, `color: vec3` | `0.0`, `(1,1,1)` |
| `normal_map` | `map: vec3` | (required) |

**New function: `_generate_openpbr_main()`** (~200 lines, mirrors `_generate_layered_main` structure):

1. UV alias, normal setup, view/light direction — identical to existing path
2. Extract all layer parameters using `_get_layer_args()` with OpenPBR defaults
3. Emit `let` statements for all ~30 OpenPBR params
4. Call `openpbr_direct(n, v, l, <base+specular params>)` for direct lighting
5. Multi-light dispatch via existing `_emit_multi_light_from_lighting()` helper (P21)
6. IBL sampling — same as existing path (reuses `ibl_contribution`)
7. Call `openpbr_compose(<all params>, ibl_ambient, coat_ibl)` for final composition
8. Tonemap + output — reuses `_emit_tonemap_output()`

**Dispatch point** (in `_generate_layered_main`, line ~953, or separate function):

```python
if _use_openpbr:
    return _generate_openpbr_main(surface, frag_inputs, schedule, module, lighting)
else:
    # existing glTF path unchanged
    ...
```

**Custom `@layer` functions:** Still work in OpenPBR mode. Applied after `openpbr_compose` but before emission, receiving `(composed_color, n, v, l, ...args)`.

---

#### P22.4: OpenPBR Direct Lighting Function

**File:** `luxc/stdlib/openpbr.lux`

```
fn openpbr_direct(
    n: vec3, v: vec3, l: vec3,
    base_color: vec3, base_metalness: scalar, base_diffuse_roughness: scalar,
    base_weight: scalar,
    specular_weight: scalar, specular_color: vec3, specular_roughness: scalar,
    specular_ior: scalar, specular_anisotropy: scalar,
    coat_weight: scalar, coat_roughness: scalar, coat_ior: scalar,
    thin_film_weight: scalar, thin_film_thickness: scalar, thin_film_ior: scalar
) -> vec3
```

Replaces `gltf_pbr()` for the OpenPBR path. Differences from `gltf_pbr`:
- Metal Fresnel uses `openpbr_f82_tint(cos, base_weight*base_color, specular_color)` instead of Schlick
- Dielectric Fresnel uses `openpbr_dielectric_fresnel(cos, eta_modulated, specular_weight)` with exact Fresnel
- Diffuse uses `openpbr_eon_diffuse()` instead of Lambert
- Specular weight modulates F0 via `openpbr_specular_ior_modulated()`
- Optional anisotropic NDF when `specular_anisotropy > 0`
- Optional thin-film Fresnel modulation when `thin_film_weight > 0`
- Coat contribution via `openpbr_coat_roughening()` + `openpbr_coat_darkening()` + `openpbr_coat_absorption()`

---

#### P22.5: Bindless OpenPBR Support

**File:** `luxc/expansion/surface_expander.py`

**Extended BindlessMaterialData struct** (OpenPBR mode):

Add fields to the SSBO after existing glTF fields:

```python
_OPENPBR_EXTRA_FIELDS = [
    ("baseDiffuseRoughness", "scalar"),
    ("baseWeight", "scalar"),
    ("specularWeight", "scalar"),
    ("specularIor", "scalar"),
    ("specularAnisotropy", "scalar"),
    ("coatIor", "scalar"),
    ("coatDarkening", "scalar"),
    ("fuzzWeight", "scalar"),
    ("fuzzRoughness", "scalar"),
    ("subsurfaceWeight", "scalar"),
    ("subsurfaceRadius", "scalar"),
    ("thinFilmWeight", "scalar"),
    ("thinFilmThickness", "scalar"),
    ("thinFilmIor", "scalar"),
    ("specularColor", "vec3"),
    ("coatColor", "vec3"),
    ("fuzzColor", "vec3"),
    ("subsurfaceColor", "vec3"),
    ("subsurfaceRadiusScale", "vec3"),
    ("transmissionScatter", "vec3"),
    ("transmissionDepth", "scalar"),
    ("transmissionScatterAniso", "scalar"),
    ("dispersionScale", "scalar"),
    ("dispersionAbbe", "scalar"),
]
```

**Extended flags:**
```python
_FLAG_FUZZ = 0x20
_FLAG_THIN_FILM = 0x40
_FLAG_SUBSURFACE = 0x80
_FLAG_ANISOTROPY = 0x100
_FLAG_OPENPBR = 0x200  # master flag: use OpenPBR composition
```

**New function: `_emit_openpbr_bindless_body()`** — mirrors `_emit_bindless_layer_body()` but loads OpenPBR-specific fields and calls `openpbr_compose()` instead of `compose_pbr_layers()`.

**Reflection JSON:** Extended `bindless.struct_fields` section includes all OpenPBR fields + `openpbr: true` flag for engines.

---

#### P22.6: OpenPBR Schedule Strategies

**File:** `luxc/expansion/surface_expander.py`

New schedule slots available when OpenPBR is active:

```python
_OPENPBR_STRATEGIES = {
    "diffuse_model": "eon",            # eon | lambert | burley
    "fuzz_model": "charlie",           # charlie | zeltner
    "specular_fresnel": "exact",       # exact | schlick
    "coat_fresnel": "exact",           # exact | schlick
    "thin_film_model": "airy",         # airy | none
    "subsurface_model": "diffusion",   # diffusion | wrap_diffuse
}
```

```lux
schedule OpenPBRDesktop {
    diffuse_model: eon,
    fuzz_model: charlie,
    thin_film_model: airy,
    tonemap: aces,
}

schedule OpenPBRMobile {
    diffuse_model: lambert,
    fuzz_model: charlie,
    specular_fresnel: schlick,
    coat_fresnel: schlick,
    thin_film_model: none,
    tonemap: reinhard,
}
```

This allows quality scaling — full OpenPBR on desktop, simplified on mobile — same material, different schedule. Matches Lux's existing algorithm/schedule separation philosophy.

---

#### P22.7: Tests

**File:** `tests/test_openpbr.py`

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestOpenPBRParsing` | 5 | Surfaces with OpenPBR layer names parse correctly; specular, fuzz, thin_film, subsurface layers accepted when `import openpbr` present |
| `TestOpenPBRExpansion` | 8 | AST inspection: `openpbr_compose` called (not `compose_pbr_layers`), correct params passed, disabled layers get zero defaults, coat darkening params present |
| `TestOpenPBRCompilation` | 8 | SPIR-V compilation succeeds: minimal surface, all-layers surface, metal surface, coated-glass, fuzz-only, thin-film, bindless, with schedule |
| `TestOpenPBRMath` | 6 | Stdlib unit tests: `openpbr_f82_tint` matches reference values, `openpbr_specular_ior_modulated(1.5, 1.0) == 1.5`, coat darkening with `coat_darkening=0` returns identity, EON diffuse energy <= 1.0 |
| `TestOpenPBRRegression` | 4 | Existing glTF shaders still compile identically (no import openpbr = no behavior change) |
| `TestOpenPBRBindless` | 3 | Bindless mode with OpenPBR: extended SSBO struct, correct flags, reflection JSON has `openpbr: true` |

~34 tests total.

---

#### P22.8: Documentation & Examples

**New files:**
- `examples/openpbr_carpaint.lux` — car paint: metallic base + clearcoat + thin-film iridescence
- `examples/openpbr_velvet.lux` — velvet: dielectric base + fuzz layer
- `examples/openpbr_glass.lux` — glass: transmission + volume + dispersion
- `docs/openpbr.md` — OpenPBR parameter reference, comparison with glTF layers, migration guide

**Updated files:**
- `README.md` — add OpenPBR to features, stdlib table (15th module)
- `docs/language-reference.md` — add OpenPBR layer table, `import openpbr` section
- `docs/project-structure.md` — add `openpbr.lux`, `test_openpbr.py`, examples

---

#### Implementation Order and Dependencies

```
P22.1: stdlib/openpbr.lux (core math)
  ↓
P22.2: wire existing brdf.lux functions (verify, delegate)
  ↓
P22.4: openpbr_direct() + openpbr_compose() (full composition)
  ↓
P22.3: surface_expander.py (layer dispatch, _generate_openpbr_main)
  ↓
P22.6: schedule strategies (quality tiers)
  ↓
P22.5: bindless support (extended SSBO, reflection)
  ↓
P22.7: tests (34 tests)
  ↓
P22.8: docs + examples
```

P22.1–P22.4 are the core — once those work, a basic `import openpbr;` surface compiles to SPIR-V. P22.5–P22.8 are extensions.

---

#### OpenPBR ↔ glTF Parameter Mapping (for migration/compatibility)

| OpenPBR | glTF Equivalent | Notes |
|---|---|---|
| `base_color` | `baseColorFactor.rgb` | Direct |
| `base_metalness` | `metallicFactor` | Direct |
| `specular_roughness` | `roughnessFactor` | Direct |
| `specular_ior` | `KHR_materials_ior.ior` | Default 1.5 both |
| `specular_weight` | `KHR_materials_specular.specularFactor` | Direct |
| `specular_color` | `KHR_materials_specular.specularColorFactor` | Direct |
| `coat_weight` | `KHR_materials_clearcoat.clearcoatFactor` | Direct |
| `coat_roughness` | `KHR_materials_clearcoat.clearcoatRoughnessFactor` | Direct |
| `coat_color` | — | glTF has no coat color |
| `coat_ior` | — | glTF assumes 1.5 |
| `coat_darkening` | — | glTF has no darkening control |
| `fuzz_weight` / `fuzz_color` | `KHR_materials_sheen.sheenColorFactor` | Different BRDF model (LTC vs Charlie) |
| `fuzz_roughness` | `KHR_materials_sheen.sheenRoughnessFactor` | Direct |
| `transmission_weight` | `KHR_materials_transmission.transmissionFactor` | Direct |
| `transmission_color/depth` | `KHR_materials_volume.attenuationColor/Distance` | Direct |
| `thin_film_*` | `KHR_materials_iridescence.*` | Direct mapping |
| `subsurface_*` | — | No glTF equivalent |
| `base_diffuse_roughness` | — | glTF uses Lambert (roughness=0) |
| `emission_luminance` | `emissiveFactor * KHR_materials_emissive_strength` | Conversion needed |

---

#### Estimated Scope

| Component | Lines | Effort |
|---|---|---|
| `stdlib/openpbr.lux` (15 functions) | ~350 | Medium |
| `surface_expander.py` changes | ~250 | Medium |
| `tests/test_openpbr.py` (34 tests) | ~500 | Medium |
| Examples (3 .lux files) | ~150 | Small |
| Documentation | ~200 | Small |
| **Total** | **~1450** | |

### Phase 21: Shared Stdlib Refactoring [DONE]

The bindless uber-shader's PBR orchestration logic (`_emit_bindless_layer_body()` in `surface_expander.py`, ~340 lines) duplicates the same material pipeline that the non-bindless path (`_generate_layered_main()`, ~330 lines) builds declaratively from `.lux` surface layers. While the math functions (`gltf_pbr`, `ibl_contribution`, `tonemap_aces`, etc.) are correctly in stdlib, the orchestration — geometric preamble, direct lighting dispatch, IBL sampling, optional layer composition, tonemapping — is hardcoded in the expander with two divergent implementations.

**Goal:** Unify both paths so the stdlib owns the full PBR pipeline, the expander shares Python-side helpers, and a known rendering gap (coat IBL in bindless) is fixed.

**Current state of duplication (from deep research):**

| Component | Non-bindless (`_generate_layered_main`) | Bindless (`_emit_bindless_layer_body`) | Already shared? |
|-----------|----------------------------------------|----------------------------------------|-----------------|
| n/v/l/n_dot_v geometric setup | Lines 937-1001 | Lines 3091-3108 + external caller | **No** — duplicated |
| Multi-light detection + arg extraction | Lines 965-1044 | Lines 2746-2778 | **Partially** — both call `_emit_multi_light_loop()` but duplicate detection/setup |
| Single-light fallback (`gltf_pbr` + tint) | Lines 1046-1057 | Lines 2780-2791 | **No** — duplicated |
| IBL sampling (reflect, sample, ibl_contribution) | Lines 1096-1137 | Lines 2893-2937 | **No** — duplicated |
| Coat IBL computation | Lines 1139-1162 (proper) | Line 2957: `vec3(0.0)` placeholder | **BUG** — bindless is broken |
| Optional layer composition | Lines 1059-1222 (manual chaining) | Lines 2793-2960 (via `compose_pbr_layers`) | **No** — non-bindless duplicates stdlib |
| Tonemap + output | Line 1225 | Line 2964 | **Yes** — `_emit_tonemap_output()` |
| `compose_pbr_layers()` stdlib | Not called (manual chaining) | Called at line 2940 | **Gap** — non-bindless should use it |

**Texture sampling is path-specific and CANNOT be unified:**
- Non-bindless: `sample(albedo_tex, uv)` — texture bound per-draw
- Bindless: `sample_bindless(textures, tex_idx, uv)` — SSBO index + texture array
- sRGB conversion: bindless does explicit `srgb_to_linear()`, non-bindless assumes decoded textures

The multi-light loop is compile-time unrolled (N iterations with unique variable suffixes) and must remain in the Python expander.

---

#### P21.1: Non-bindless Uses `compose_pbr_layers()`

**Problem:** `_generate_layered_main()` manually chains `transmission_replace()` → `+= ambient` → `sheen_over()` → `coat_over()` → `+= emission` (lines 1059-1222), duplicating what `compose_pbr_layers()` already does. The bindless path correctly calls `compose_pbr_layers()` (line 2940).

**Change:** Replace the manual chaining in `_generate_layered_main()` with:
1. Collect optional layer values into standardized variables (same names as bindless: `bl_trans_factor`, `bl_sheen_color`, etc.)
2. Set disabled layers to zero/default (factor=0.0, color=vec3(0.0))
3. Call `compose_pbr_layers()` with the collected values
4. Remove ~80 lines of manual `transmission_replace` → `sheen_over` → `coat_over` → emission chaining

**Diff sketch** (Python AST in `surface_expander.py`):
```python
# BEFORE (lines 1059-1222): ~160 lines of manual chaining
if "transmission" in layers_by_name:
    # ... 20 lines: extract args, call transmission_replace ...
if ibl_args is not None:
    # ... 50 lines: sample IBL, call ibl_contribution, coat IBL ...
if "sheen" in layers_by_name:
    # ... 15 lines: extract args, call sheen_over ...
if "coat" in layers_by_name:
    # ... 18 lines: extract args, call coat_over ...
if "emission" in layers_by_name:
    # ... 8 lines: add emission ...

# AFTER: ~40 lines — collect values, single compose call
# Initialize defaults (disabled = zero)
body.append(LetStmt("bl_trans_factor", "scalar", NumberLit("0.0")))
# ... (same pattern as bindless lines 2793-2802)
# Conditionally load from layer args
if "transmission" in layers_by_name:
    # ... extract factor/ior from layer args, assign to bl_trans_* vars
# Sample IBL → ambient (shared helper, see P21.2)
# Compute coat IBL if both coat + ibl present
# Call compose_pbr_layers() — exactly as bindless does at line 2940
body.append(LetStmt("composed", "vec3", CallExpr(VarRef("compose_pbr_layers"), [...])))
```

**Files:** `luxc/expansion/surface_expander.py` (modify `_generate_layered_main()`)
**Net effect:** ~80 lines removed, non-bindless and bindless share identical composition logic
**Risk:** Low — mathematical equivalence guaranteed (stdlib functions are the same)

---

#### P21.2: Extract Shared Expander Helpers

**Problem:** Both paths duplicate Python AST-generation code for geometric setup, direct lighting dispatch, and IBL sampling.

**New private functions in `surface_expander.py`:**

**`_emit_geometric_preamble(body, frag_inputs, lighting, pos_var)`** → returns `(result_var_for_l)`
- Normalize `n` from world_normal (with TBN option for non-bindless)
- Compute `v` from `view_pos - pos` (or fallback `vec3(0,0,1)`)
- Detect multi_light from lighting block
- Emit `l` (dummy for multi-light, directional otherwise)
- Emit `n_dot_v = max(dot(n, v), 0.001)`
- Replaces: lines 937-1001 (non-bindless) and 3091-3108 (bindless fragment caller)

**`_emit_direct_lighting(body, lighting, pos_var)`** → returns `result_var`
- Detect multi_light layer, extract max_lights, shadow args
- Emit `_ml_light_count` if multi-light
- Call `_emit_multi_light_loop()` or emit single-light `gltf_pbr()` + tint
- Replaces: lines 1003-1057 (non-bindless) and 2744-2791 (bindless)

**`_emit_ibl_sampling(body, is_bindless, is_rt, ibl_args=None)`** → appends `ambient` variable
- Emit reflection vector `r`
- Sample IBL textures (specular, irradiance, BRDF LUT)
  - Bindless: uses `env_specular`/`env_irradiance`/`brdf_lut` uniform names
  - Non-bindless: uses `ibl_args` expressions (from lighting/surface layer)
  - RT: all `sample_lod` calls; raster: `sample_lod` for cubemaps, `sample` for 2D BRDF LUT
- Call `ibl_contribution()`
- Replaces: lines 1096-1137 (non-bindless) and 2893-2937 (bindless)

**`_emit_coat_ibl(body, coat_factor_var, coat_roughness_var, specular_map, is_bindless)`** → appends `coat_ibl_contrib` variable
- Emit `prefiltered_coat = sample_lod(specular, r, coat_rough * 8.0)`
- Emit `coat_ibl_contrib = coat_ibl(factor, roughness, n, v, prefiltered_coat)`
- Used by both non-bindless (already works) and bindless (new — fixes P21.3)

**Files:** `luxc/expansion/surface_expander.py`
**Net effect:** Each helper called 2-3x (raster, RT, mesh × bindless/non-bindless). Estimated ~200 lines of duplication removed.

---

#### P21.3: Fix Coat IBL Gap in Bindless

**Problem:** `_emit_bindless_layer_body()` line 2957 passes `vec3(0.0)` for `coat_ibl_val` to `compose_pbr_layers()`. The non-bindless path correctly computes `coat_ibl()` (lines 1139-1162). This means bindless scenes with clearcoat materials are missing coat IBL specular reflection — a visible rendering quality gap.

**Fix:** Inside the clearcoat flag-gated block (lines 2845-2871), after loading coat factor/roughness from SSBO:
1. Sample `prefiltered_coat = sample_lod(env_specular, r, coat_roughness * 8.0)`
2. Compute `coat_ibl_contrib = coat_ibl(coat_factor, coat_roughness, n, v, prefiltered_coat)`
3. Replace the `vec3(0.0)` placeholder at line 2957 with `coat_ibl_contrib`

Or, with P21.2's `_emit_coat_ibl()` helper, this becomes a single function call.

**Files:** `luxc/expansion/surface_expander.py` (modify `_emit_bindless_layer_body()`)
**Net effect:** Correct coat IBL in all bindless pipelines (raster, RT, mesh)

---

#### P21.4: Create `stdlib/pbr_pipeline.lux`

**New stdlib file** with a single convenience function that wraps `ibl_contribution()` + `compose_pbr_layers()`:

```lux
import compositing;
import ibl;

fn pbr_shade(
    direct_lit: vec3,
    n: vec3, v: vec3, l: vec3,
    albedo: vec3, roughness: scalar, metallic: scalar,
    // Pre-sampled IBL textures
    prefiltered: vec3, irradiance: vec3, brdf_sample: vec2,
    // Optional layers (factor=0 / color=vec3(0) to disable)
    trans_factor: scalar, trans_ior: scalar,
    trans_thickness: scalar, trans_atten_color: vec3, trans_atten_dist: scalar,
    sheen_color: vec3, sheen_roughness: scalar,
    coat_factor: scalar, coat_roughness: scalar,
    coat_ibl_contrib: vec3,
    emission: vec3
) -> vec3 {
    let n_dot_v: scalar = max(dot(n, v), 0.001);
    let ambient: vec3 = ibl_contribution(
        albedo, roughness, metallic, n_dot_v,
        prefiltered, irradiance, brdf_sample);
    return compose_pbr_layers(
        direct_lit, n, v, l, albedo, roughness,
        trans_factor, trans_ior, trans_thickness, trans_atten_color, trans_atten_dist,
        ambient,
        sheen_color, sheen_roughness,
        coat_factor, coat_roughness,
        coat_ibl_contrib,
        emission);
}
```

Both `_generate_layered_main()` and `_emit_bindless_layer_body()` emit a single `CallExpr(VarRef("pbr_shade"), [...])` after direct lighting + IBL sampling, replacing separate `ibl_contribution()` + `compose_pbr_layers()` calls.

**Files:** `luxc/stdlib/pbr_pipeline.lux` (new), `luxc/expansion/surface_expander.py` (both paths)
**Net effect:** Both paths converge to a single stdlib entry point for post-sampling PBR composition

---

#### P21.5: Backward Compatibility & Validation

**What must NOT change:**
- Existing `.lux` files with explicit `layers [...]` syntax (tested by `test_surface_expander.py`, `test_gltf_extensions.py`, `test_material_properties.py`)
- Bindless pipeline expansion (tested by existing tests + engine renders)
- Reflection JSON layout (engines parse it)
- SPIR-V output for all non-modified shaders (binary diff check)

**Validation strategy:**
1. `pytest tests/` → 1036+ tests pass (zero regressions)
2. Add ~15 new tests in `tests/test_stdlib_refactoring.py`:
   - Non-bindless layered shader calls `compose_pbr_layers` (AST inspection)
   - Non-bindless with all optional layers (transmission + sheen + coat + emission + IBL) produces identical SPIR-V
   - Bindless coat IBL: new SPIR-V includes `coat_ibl()` call when clearcoat flag set
   - `pbr_shade()` stdlib function compiles and inlines correctly
   - Shared helper functions produce same AST as old inline code
3. Visual regression: render `examples/gltf_pbr_layered.lux` and `examples/advanced_materials.lux` headless, compare PNG output (pixel-perfect or PSNR > 40dB)

---

#### Implementation Order

| Step | Description | Estimated lines changed | Risk |
|------|-------------|------------------------|------|
| P21.1 | Non-bindless → `compose_pbr_layers()` | -80 / +40 in `surface_expander.py` | Low: mathematical identity |
| P21.3 | Fix coat IBL in bindless | +20 in `surface_expander.py` | Low: additive fix |
| P21.2 | Extract 4 shared helpers | -200 / +120 in `surface_expander.py` | Medium: refactor of central code |
| P21.4 | `stdlib/pbr_pipeline.lux` + wire up | +30 new file, ±20 in expander | Low: thin wrapper |
| P21.5 | Tests + validation | +80 in new test file | None |

**Total estimated impact:** ~180 net lines removed from `surface_expander.py` (3266 → ~3086), 1 new stdlib file (30 lines), 1 new test file (80 lines). The expander's two 330-line PBR functions each shrink to ~120 lines of path-specific data loading + shared helper calls.

---

## What Makes This Different From Everything Else

| Language | What it is | What Lux is |
|----------|-----------|-------------|
| GLSL/HLSL | Imperative GPU programming | Mathematical rendering specification |
| WGSL | Safer GLSL for the web | Physics-aware composition |
| Slang | Extensible shading + autodiff | Declarative BRDFs, not imperative |
| OSL | Closure-based (offline only) | Closure-inspired but real-time SPIR-V |
| MDL | Declarative but NVIDIA-only | Open, compiles to standard SPIR-V |
| MaterialX | Interchange format, not a language | A language with a smart compiler |

**The one-liner:** Lux is what happens when you take OSL's philosophy
(describe scattering, not pixels), MDL's declarative materials, and Halide's
algorithm/schedule separation — and compile it all to real-time Vulkan SPIR-V.

---

### Phase 23: Interactive Scene Editor (C++ & Rust) [DONE — P23.1-P23.2]

**Goal:** Build a full interactive application with UI for both C++ and Rust engines — load glTF scenes, select shaders/pipelines, pick and transform meshes, and exercise all current Lux features from a single app.

**Why:** Current viewers are single-pipeline launchers. A real scene editor lets users explore Lux's full feature set (PBR, OpenPBR, splats, mesh shaders, compute, RT) in one session without relaunching.

**Core features:**

1. **Scene browser / glTF loader**
   - File dialog or drag-and-drop to load `.glb`/`.gltf` files
   - Scene tree panel showing nodes, meshes, materials, cameras, lights
   - Multi-model support (load several glTFs into one scene)

2. **Shader / pipeline selector**
   - List compiled `.spv` pipelines from a directory (auto-discover from reflection JSON)
   - Hot-swap shaders on selected meshes
   - Pipeline modes: raster, raytrace, mesh_shader, gaussian_splat
   - Display active pipeline info (inputs, uniforms, descriptor sets from reflection JSON)

3. **Mesh selection & basic transforms**
   - Click-to-select mesh in viewport (ray picking or ID pass)
   - Gizmo overlay for translate / rotate / scale (ImGuizmo or custom)
   - Transform hierarchy (parent-child, world vs local)
   - Selection outline or highlight

4. **Material editor panel**
   - Read surface properties from reflection JSON
   - Sliders for scalar parameters, color pickers for vec3
   - Support for `properties` UBO — live update uniform values
   - OpenPBR layer parameter editing (base, specular, coat, fuzz, thin_film, etc.)

5. **Lighting controls**
   - Add/remove/edit directional, point, spot lights
   - IBL environment selector (load HDR/EXR, switch cubemaps)
   - Shadow map visualization toggle

6. **Rendering feature toggles**
   - Wireframe, normal visualization, depth view, UV checker
   - Debug mode toggle (`-g`, `--debug-print`)
   - Auto-type precision overlay (show fp16 vs fp32 regions)
   - Tonemap selector (none, Reinhard, ACES)

7. **Gaussian splat integration**
   - Load `.glb` with `KHR_gaussian_splatting` extension
   - Splat + mesh coexistence in the same viewport
   - Splat debug views (SH degree, covariance ellipsoids)

**UI framework:**
- C++: Dear ImGui (already widely used in Vulkan apps) + GLFW
- Rust: egui (native integration with winit/ash) or imgui-rs

**Architecture:**
- Scene graph abstraction layer shared between renderers
- Reflection-driven pipeline binding (existing `reflected_pipeline.cpp/rs` as base)
- Command pattern for undo/redo on transforms
- Serialization: save/load scene state as JSON

**Implementation phases:**
1. P23.1: ImGui integration + basic panels (scene tree, properties) — C++ first
2. P23.2: Pipeline hot-swap + material editor
3. P23.3: Mesh picking + transform gizmos
4. P23.4: Lighting editor + IBL selector
5. P23.5: Gaussian splat integration in editor
6. P23.6: Rust port (egui or imgui-rs)

---

### Phase 24: KHR_gaussian_splatting Conformance Test Assets [DONE]

**Goal:** Integrate the official Khronos `KHR_gaussian_splatting` test suite into Lux's test infrastructure, ensuring our splat renderer handles all edge cases from the upcoming ratification process.

**Source:** [KhronosGroup/glTF#2562](https://github.com/KhronosGroup/glTF/issues/2562) — conformance test data collection.

**Test data:** `gltf-splat-examples-2026-02-17.zip` containing glTF test scenes.

**Test categories (from the Khronos issue):**

| Test | Description | What It Validates |
|------|-------------|-------------------|
| Rotation tests | Splats rotated around x/y/z axes | Coordinate system, quaternion handling |
| Mesh-splat mixing | glTF with both mesh primitives and splats | Multi-primitive-type rendering |
| Depth/sorting | Front-to-back disc-shaped splats | Depth sorting correctness |
| Scaling | Carefully positioned splats for scale validation | Linear vs log-space scale interpretation |
| Spherical harmonics | 6 rotated primitives with directional color response | SH evaluation from multiple viewpoints |
| Mixed SH degrees | Primitives with SH degree 2 vs 3 in same file | Per-primitive SH degree handling |

**Important:** Opacity accessor stores alpha values [0,1], NOT log-space opacity (corrected in latest test data).

**Implementation:**

1. P24.1: Download and extract test assets to `tests/assets/khr_splat_conformance/`
2. P24.2: Create `tests/test_khr_splat_conformance.py` — load each test glTF, compile splat pipeline, render headless, validate:
   - Rotation: compare screenshots against reference at multiple camera angles
   - Mesh-splat: verify both mesh and splat primitives render
   - Sorting: verify correct front-to-back ordering (no Z-fighting artifacts)
   - SH: verify color changes with viewpoint rotation
   - Mixed SH: verify different SH degrees coexist
3. P24.3: Fix any renderer issues discovered by conformance tests
4. P24.4: Add conformance test results to documentation

---

### Phase 25: PLY-to-glTF Gaussian Splat Conversion Tool [DONE]

**Goal:** Add a robust PLY → glTF conversion pipeline for Gaussian splats as a first-class Lux tool, supporting the `KHR_gaussian_splatting` extension.

**Why:** Most 3DGS training outputs (3DGS, gsplat, nerfstudio) produce `.ply` files. Lux's splat renderer expects glTF with `KHR_gaussian_splatting`. We already have `tools/ply_to_gltf.py` but want a more robust solution.

**Reference implementation:** [NorbertNopper-Huawei/3dgs_ply2gltf](https://github.com/NorbertNopper-Huawei/3dgs_ply2gltf) — C++ converter with coordinate system conversion and round-trip verification.

**Key features:**

1. **Coordinate system conversion**
   - Input: right-handed Z-up (3DGS training convention)
   - Output: right-handed Y-up (glTF convention)
   - `-90°` rotation around X axis, with quaternion adjustment
   - Flag: `--convert` (enabled by default, `--no-convert` to skip)

2. **SH degree detection**
   - Auto-detect from PLY header (count of `f_rest_*` properties)
   - Degree 0: 0 coefficients (DC only via `f_dc_*`)
   - Degree 1: 9 coefficients (3 per band)
   - Degree 2: 24 coefficients
   - Degree 3: 45 coefficients

3. **Opacity handling**
   - Input: log-space opacity from training (`sigmoid(x)` to get alpha)
   - Output: linear alpha [0,1] in glTF accessor (per Khronos spec correction)

4. **Scale handling**
   - Input: log-space scale from training (`exp(x)` to get actual scale)
   - Output: linear scale in glTF accessor

5. **Round-trip verification** (`--verify`)
   - Convert PLY → glTF → PLY
   - Compare numerical equality of all attributes
   - Report any precision loss

6. **Batch processing** (`--batch <dir>`)
   - Convert all `.ply` files in a directory
   - Progress bar, parallel processing

**Implementation options:**

- **Option A:** Enhance existing `tools/ply_to_gltf.py` (Python, numpy — already handles basic case)
- **Option B:** Integrate the C++ `3dgs_ply2gltf` as a submodule + build dependency
- **Option C:** Port the C++ logic to Python for zero-compile dependency

**Recommended: Option A** — extend `tools/ply_to_gltf.py` with coordinate conversion, opacity/scale transforms, SH degree detection, verification. Python-only, no build dependency.

**CLI:**
```bash
# Basic conversion
python -m tools.ply_to_gltf input.ply -o output.glb

# With coordinate conversion (default)
python -m tools.ply_to_gltf input.ply --convert -o output.glb

# Round-trip verification
python -m tools.ply_to_gltf input.ply --verify -o output.glb

# Batch
python -m tools.ply_to_gltf --batch ./splats/ -o ./gltf_out/
```

**Tests:** `tests/test_ply_to_gltf.py` — PLY parsing, coordinate conversion, opacity transform, SH degree detection, round-trip verification, POSITION min/max.
