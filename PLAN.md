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
| **P11** | Metal backend via SPIR-V cross-compilation (SPIRV-Cross → MSL) | Planned |
| **P12** | Official glTF PBR extensions in engine materials (auto-detect, permutation selection) | ✅ Complete |
| **P13** | Mesh shader support (`task`/`mesh` stages, meshlet-based geometry, `--define` compile-time parameters) | ✅ Complete |
| **P14** | Gaussian splatting representation (splat sorting, tile-based rasterizer, SH evaluation) | Planned |
| **P15** | BRDF & layer visualization (lobe plots, transfer function graphs, energy conservation tests) | ✅ Complete |
| **P16** | AI features for Lux (image-to-shader, prompt-based generation, AI skills, training pipeline) | Planned |
| **P17** | Light & shadow management (declarative lights, CSM/cubemap/perspective shadows, PCF/PCSS/VSM/MSM filtering, tiled/clustered culling, volumetric lighting, light probes, LPV) | Planned |

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
