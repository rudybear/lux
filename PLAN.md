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

## Phase 6: Ray Tracing Pipeline

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

### P6.1 — RT Stage Types & Grammar

| Item | Effort |
|------|--------|
| New `EnvironmentDecl` AST node | Medium |
| Extend `PipelineDecl` with `mode: raytrace` | Medium |
| RT built-in variables (launch_id, ray_origin, ray_direction, hit_t, etc.) | Medium |
| `acceleration_structure` type | Small |

### P6.2 — RT SPIR-V Codegen

| Item | Effort |
|------|--------|
| 6 new execution models (RayGen, ClosestHit, AnyHit, Miss, Intersection, Callable) | Large |
| RT storage classes (RayPayloadKHR, HitAttributeKHR, etc.) | Medium |
| RT instructions (OpTraceRayKHR, OpReportIntersectionKHR, etc.) | Large |
| SPV_KHR_ray_tracing extension + capability | Small |

### P6.3 — RT Surface Expansion

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
| **P6** | Ray tracing pipeline (RT stages, SPIR-V codegen, surface→RT expansion) | ✅ Complete |

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
