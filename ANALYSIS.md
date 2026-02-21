# Gap Analysis: Lux vs MaterialX / glTF PBR / Ray Tracing

*Generated 2026-02-18*

---

## Current Inventory

**76 unique functions** (177 overloaded signatures):

| Category | Functions | Signatures |
|----------|-----------|------------|
| Built-in math/vector | 27 | 128 |
| stdlib/brdf | 13 | 13 |
| stdlib/color | 5 | 5 |
| stdlib/noise | 13 | 13 |
| stdlib/sdf | 18 | 18 |
| **Total** | **76** | **177** |

**19 built-in types**, **5 type aliases** (Radiance, Reflectance, Direction, Normal, Irradiance), **3 constants** (PI, INV_PI, EPSILON).

---

## Part 1: Built-in Math — Lux vs MaterialX

### Covered (no gaps)

- Trig: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- Exponential: `exp`, `exp2`, `log`, `log2`, `sqrt`, `pow`
- Common: `abs`, `floor`, `ceil`, `fract`, `sign`, `min`, `max`, `clamp`
- Vector: `normalize`, `length`, `distance`, `dot`, `cross`, `reflect`
- Interpolation: `mix`, `smoothstep`, `step`
- Constructors & swizzle: full coverage

### Previously Missing Built-ins — NOW RESOLVED

| Function | MaterialX | glTF PBR | Status | Notes |
|---|---|---|---|---|
| `atan2(y, x)` | `atan2` | Anisotropy rotation | ✅ | 2-arg `atan(y,x)` overload added |
| `refract(I, N, eta)` | `refract` | Volume, dispersion | ✅ | GLSL.std.450 Refract |
| `mod(x, y)` | `modulo` | — | ✅ | OpFMod |
| `inversesqrt(x)` | — | Implicit in V_GGX | ✅ | GLSL.std.450 InverseSqrt |
| `round(x)` | `round` | — | **LOW** | Nice-to-have. Not yet added. |
| `determinant(m)` | `determinant` | — | **LOW** | Matrix math. Not yet added. |
| `inverse(m)` | `invertmatrix` | — | **LOW** | Matrix math. Not yet added. |
| `transpose(m)` | `transpose` | — | **LOW** | Matrix math. Not yet added. |

**Verdict: All critical and medium-priority gaps resolved.** Only low-priority matrix math and `round()` remain.

---

## Part 2: glTF PBR — Feature Gap Analysis

### Core PBR (metallic-roughness) — FULLY ADDRESSED

Our `pbr_brdf()` implements the full Cook-Torrance model:
- GGX NDF (D term) via `ggx_ndf()`
- Smith GGX geometry (G term) via `smith_ggx()` and `v_ggx_correlated()` (height-correlated, glTF spec-compliant)
- Schlick Fresnel (F term) via `fresnel_schlick()`
- Lambert diffuse via `lambert_brdf()`
- Metallic/dielectric blend via `pbr_brdf()` and `gltf_pbr()`

**Layered Surface System**: The `layers [base, normal_map, ibl, emission]` syntax now provides a unified, declarative path for glTF PBR rendering. A single `surface` declaration with these layers compiles to both rasterization fragment shaders and ray tracing closest-hit shaders, with automatic energy conservation via albedo-scaling between layers. The `samplerCube` type enables IBL (image-based lighting) via diffuse irradiance and pre-filtered specular cubemaps directly in surface parameter blocks.

**Previous discrepancy resolved**: The height-correlated Smith G term is now available via `v_ggx_correlated()` and is used by the `base` layer, achieving full glTF spec compliance.

### glTF Extensions — ALL COMPLETE

| Extension | Status | Implementation |
|---|---|---|
| **clearcoat** | ✅ **COMPLETE** | `clearcoat_brdf()` in stdlib; `coat` layer planned (Phase 6) |
| **transmission** | ✅ **COMPLETE** | `transmission_btdf()` + `transmission_color()` in stdlib; `transmission` layer planned (Phase 7) |
| **volume** | ✅ **COMPLETE** | `volume_attenuation()` + `volumetric_btdf()` in stdlib |
| **ior** | ✅ **COMPLETE** | `ior_to_f0()` in stdlib |
| **specular** | ✅ **COMPLETE** | Expressible via F0/F90 modification |
| **sheen** | ✅ **COMPLETE** | `charlie_ndf()` + `sheen_brdf()` in stdlib; `sheen` layer planned (Phase 6) |
| **iridescence** | ✅ **COMPLETE** | `iridescence_fresnel()` (Belcour & Barla 2017) in stdlib |
| **anisotropy** | ✅ **COMPLETE** | `anisotropic_ggx_ndf()` + `anisotropic_v_ggx()` + 2-arg `atan()` |
| **emissive_strength** | ✅ **COMPLETE** | Scalar multiply; `emission` layer in layered surface system |
| **dispersion** | ✅ **COMPLETE** | `dispersion_ior()` + `dispersion_f0()` + `dispersion_refract()` in stdlib |
| **diffuse_transmission** | ✅ **COMPLETE** | `diffuse_transmission()` in stdlib |
| **unlit** | ✅ **COMPLETE** | Bypass BRDF (already expressible) |
| **Normal mapping** | ✅ **COMPLETE** | `tbn_perturb_normal()` + `tbn_from_tangent()` + `unpack_normal()`; `normal_map` layer in layered surface system |
| **Occlusion** | ✅ **COMPLETE** | `mix(1.0, ao, strength)` |
| **IBL** | ✅ **COMPLETE** | `ibl` layer with `samplerCube` support for diffuse irradiance + pre-filtered specular |

### Priority Tiers for glTF Compliance

**Tier 1 — Core compliance** (needed for any glTF viewer): ✅ ALL COMPLETE
- `refract()` built-in ✅
- `atan2()` built-in (2-arg `atan` overload) ✅
- Height-correlated Smith G (`v_ggx_correlated()`) ✅
- Normal mapping support (tangent/bitangent inputs) ✅
- **Layered surface system** (`layers [base, normal_map, ibl, emission]`) ✅ — unified glTF PBR path

**Tier 2 — Common extensions** (most glTF content uses these): ✅ STDLIB COMPLETE, LAYER INTEGRATION PLANNED
- Clearcoat BRDF library ✅ — `clearcoat_brdf()` in stdlib; `coat` layer planned (Phase 6)
- Sheen (Charlie NDF) library ✅ — `sheen_brdf()` in stdlib; `sheen` layer planned (Phase 6)
- Transmission + Volume library ✅ — `transmission_btdf()` in stdlib; `transmission` layer planned (Phase 7)
- Anisotropy (anisotropic GGX) library ✅

**Tier 3 — Advanced** (specialty content): ✅ STDLIB COMPLETE
- Iridescence library ✅
- Dispersion (per-channel refraction) ✅
- Diffuse transmission ✅

#### Per-Asset Permutation Selection (Implemented)

The compile-time features system (`features { ... }` + `if` guards) enables per-asset shader permutation selection. Each glTF model's material extensions can be mapped to a specific compiled shader variant:

- `--features has_normal_map,has_emission` compiles a variant with only normal mapping and emission
- `--all-permutations` generates all 2^N combinations with a manifest for runtime selection
- Reflection JSON includes `features` dict and `feature_suffix` for engine-side lookup
- No dead code from unused extensions — each permutation contains only the enabled material layers

---

## Part 3: MaterialX — Where We're Far Behind

MaterialX has **~180 node types** vs our **76 functions**. Most of the gap is in categories we don't need to compete on directly:

### Significant Gaps

| Category | MaterialX | Lux | Gap Size |
|---|---|---|---|
| **BSDF models** | 8 distinct lobes (Oren-Nayar, Burley, translucent, dielectric, conductor, generalized Schlick, subsurface, sheen, hair) | GGX, Lambert, Oren-Nayar, Burley, Charlie (sheen), clearcoat, transmission, conductor, iridescence | **Small** — subsurface and hair remain |
| **Uber-shaders** | 5 (Standard Surface, OpenPBR, glTF PBR, Disney, UsdPreviewSurface) | `gltf_pbr()` + layered surface system (`layers [base, normal_map, ibl, emission]`) | **Small** — glTF PBR fully covered; Standard Surface and OpenPBR remain |
| **Color ops** | HSV/HSL conversion, contrast, saturation, hue adjust, color correct, remap | HSV, contrast, saturation, hue shift, brightness, gamma, sRGB, luminance, tonemap | **Small** — HSL and color correct remain |
| **Compositing** | 17 blend/composite modes (Porter-Duff, screen, overlay, etc.) | None | **Medium** |
| **Texture** | Triplanar projection, hex tiling, normal map transform | `sample()` only | **Medium** |
| **Geometric** | Position, normal, tangent, bitangent, texcoord, vertex color access | Via stage inputs only | Architectural |
| **Procedural patterns** | Checkerboard, circles, hexagons, grids, ramps, 10+ patterns | Noise only | **Low** |
| **Volume** | Absorption VDF, anisotropic scattering (Henyey-Greenstein) | None | **Medium** |
| **Conditionals** | ifgreater, ifequal, switch(10) | Ternary only | **None** — ternary covers this |

### MaterialX Functions — Status

**glTF PBR uber-shader (`stdlib/brdf.lux`):** ✅ COMPLETE
- `gltf_pbr(n, v, l, albedo, roughness, metallic)` ✅
- `clearcoat_brdf(n, v, l, clearcoat, roughness)` ✅
- `sheen_brdf(color, roughness, NdotH, NdotL, NdotV)` ✅
- `transmission_btdf(n, v, l, roughness, ior)` ✅
- `volume_attenuation(dist, attColor, attDist)` ✅
- **Layered surface system** (`layers [base, normal_map, ibl, emission]`) ✅ — declarative composition with energy conservation

**Color space module (`stdlib/colorspace.lux`):** ✅ COMPLETE
- `rgb_to_hsv(c)` / `hsv_to_rgb(c)` ✅
- `contrast(c, pivot, amount)` ✅
- `saturate_color(c, amount)` ✅
- `hue_shift(c, shift)` ✅
- `brightness(c, amount)` ✅
- `gamma_correct(c, gamma)` ✅

**BRDF functions (`stdlib/brdf.lux`):** ✅ COMPLETE
- `oren_nayar_diffuse(albedo, roughness, NdotL, NdotV)` ✅
- `burley_diffuse(albedo, roughness, NdotL, NdotV, VdotH)` ✅
- `charlie_ndf(roughness, NdotH)` ✅
- `v_ggx_correlated(NdotL, NdotV, alpha)` ✅
- `anisotropic_ggx_ndf(NdotH, TdotH, BdotH, at, ab)` ✅
- `anisotropic_v_ggx(NdotL, NdotV, TdotV, BdotV, TdotL, BdotL, at, ab)` ✅
- `iridescence_fresnel(outside_ior, film_ior, base_f0, thickness, cos_theta)` ✅
- `conductor_fresnel(f0, f82, VdotH)` ✅

**Remaining MaterialX gaps** (not yet addressed):
- Subsurface scattering BSSRDF (Burley SSS, random walk)
- Hair BSDF (Marschner / d'Eon model)
- Standard Surface and OpenPBR uber-shaders
- HSL color space, color correct node
- Porter-Duff compositing (17 blend modes)
- Procedural patterns (checkerboard, circles, hexagons, grids, ramps)

---

## Part 4: Ray Tracing Pipeline — Natural Extension of `surface`

The `surface` declaration is a perfect fit for ray tracing because **the BRDF math is identical**. Only data sourcing and output mechanism change.

### Current → Ray Tracing Mapping

| Lux Concept | Rasterization (Today) | Ray Tracing (Future) |
|---|---|---|
| `surface { brdf: ... }` | → Fragment shader | → **Closest-hit** shader (same BRDF eval) |
| `geometry { ... }` | → Vertex shader | → Vertex data fetch via SBT + barycentrics |
| `pipeline { ... }` | → Rasterization draw | → **Ray generation** shader (camera rays) |
| (new) `environment { ... }` | — | → **Miss** shader (sky/background) |
| `surface { opacity: ... }` | → Alpha blend | → **Any-hit** shader (alpha test) |
| SDF stdlib | Fragment-only raymarching | → **Intersection** shader (procedural geometry!) |

### Shader Stages Required

| Stage | SPIR-V Execution Model | Purpose |
|---|---|---|
| Ray Generation | `RayGenerationKHR` | Entry point — launches rays, processes results |
| Closest Hit | `ClosestHitKHR` | Invoked at nearest intersection — **evaluates surface BRDF** |
| Any Hit | `AnyHitKHR` | Every candidate intersection — alpha testing, transparency |
| Miss | `MissKHR` | Ray hits nothing — returns environment/sky |
| Intersection | `IntersectionKHR` | Custom intersection test — **SDF raymarching** |
| Callable | `CallableKHR` | Subroutine — modular BRDF evaluation for multi-material |

### Built-in Variables Needed (by stage)

**All stages**: `launch_id: uvec3`, `launch_size: uvec3`

**Hit stages** (intersection, any-hit, closest-hit):
- `world_ray_origin: vec3`, `world_ray_direction: vec3`
- `object_ray_origin: vec3`, `object_ray_direction: vec3`
- `ray_tmin: scalar`, `ray_tmax: scalar`
- `primitive_id: int`, `instance_id: int`, `instance_custom_index: int`
- `geometry_index: int`
- `object_to_world: mat4x3`, `world_to_object: mat4x3`
- `incoming_ray_flags: uint`

**Any-hit + closest-hit only**: `hit_t: scalar`, `hit_kind: uint`

**Miss**: ray origin, direction, tmin, tmax, flags

### Built-in Functions/Statements Needed

| Function | SPIR-V | Available In |
|---|---|---|
| `trace_ray(accel, flags, mask, sbt_offset, sbt_stride, miss_idx, origin, tmin, dir, tmax, payload)` | `OpTraceRayKHR` | raygen, closest-hit, miss |
| `report_intersection(hitT, hitKind) -> bool` | `OpReportIntersectionKHR` | intersection |
| `execute_callable(sbt_index, data)` | `OpExecuteCallableKHR` | raygen, closest-hit, miss, callable |
| `ignore_intersection` (statement) | `OpIgnoreIntersectionKHR` | any-hit |
| `terminate_ray` (statement) | `OpTerminateRayKHR` | any-hit |

### New Types Needed

- `acceleration_structure` → `OpTypeAccelerationStructureKHR`
- Payload structs (user-defined, matched by location between caller/callee)
- Hit attribute structs

### SPIR-V Requirements

- `OpCapability RayTracingKHR` (4479)
- `OpExtension "SPV_KHR_ray_tracing"`
- SPIR-V version 1.4 minimum
- 6 new storage classes: `RayPayloadKHR`, `IncomingRayPayloadKHR`, `HitAttributeKHR`, `CallableDataKHR`, `IncomingCallableDataKHR`, `ShaderRecordBufferKHR`
- 5 new instructions + `OpTypeAccelerationStructureKHR`

### Proposed Lux Syntax

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
    mode: raytrace,        // NEW — switches codegen target
    surface: CopperMetal,  // → closest-hit shader
    environment: HDRISky,  // → miss shader
    max_bounces: 4,
}

// SDF primitives become intersection shaders
procedural MetaBalls {
    sdf: sdf_smooth_union(sdf_sphere(0.5), sdf_sphere(0.3), 0.2),
    surface: ChromeSurface,
}
```

### Compiler Architecture Changes

1. **AST**: New `EnvironmentDecl`, extend `StageBlock.stage_type` to RT stages, extend `PipelineDecl` with `mode`
2. **Surface expander**: `_expand_surface_to_closest_hit()`, `_expand_surface_to_any_hit()`, `_expand_environment_to_miss()`, `_expand_procedural_to_intersection()`
3. **SPIR-V codegen**: RT execution models, storage classes, built-in decorations, RT instructions as block terminators
4. **Built-in types**: `acceleration_structure`
5. **Built-in functions**: `trace_ray`, `report_intersection`, `execute_callable`

**Key architectural insight**: Every BRDF function we add now pays off twice — it works in both rasterization and ray tracing. The `surface` abstraction decouples material math from rendering strategy.

---

## Summary: Priority Roadmap

### ✅ COMPLETE — P5.1: Built-in gaps

| Item | Effort | Impact | Status |
|---|---|---|---|
| Add `refract(I, N, eta)` built-in | Tiny | Unlocks transmission, volume, dispersion | ✅ |
| Add 2-arg `atan(y, x)` built-in | Tiny | Unlocks anisotropy | ✅ |
| Add `inversesqrt(x)` built-in | Tiny | Common in PBR | ✅ |
| Height-correlated Smith G in brdf.lux | Small | glTF spec compliance | ✅ |

### ✅ COMPLETE — P5.2: stdlib expansions

| Item | Effort | Impact | Status |
|---|---|---|---|
| Clearcoat library | Small | glTF clearcoat extension | ✅ |
| Sheen library (Charlie NDF) | Medium | glTF sheen extension | ✅ |
| Anisotropic GGX library | Medium | glTF anisotropy extension | ✅ |
| Burley/Oren-Nayar diffuse | Small | Better diffuse models | ✅ |
| Color space module (HSV, contrast) | Small | Artistic shading | ✅ |
| Normal mapping helpers (TBN) | Small | Essential for textured content | ✅ |

### ✅ COMPLETE — P5.3: Advanced material models

| Item | Effort | Impact | Status |
|---|---|---|---|
| Volume/transmission library | Medium | Glass, liquids | ✅ |
| Iridescence library | Large | Thin-film effects | ✅ |
| `gltf_pbr()` uber-shader | Medium | Full glTF material model | ✅ |
| Triplanar projection | Small | Common texture technique | ✅ |

### ✅ COMPLETE — P5.5: Layered surface shader system

| Item | Effort | Impact | Status |
|---|---|---|---|
| `layers [...]` syntax for surfaces | Medium | Declarative layer composition | ✅ |
| Energy conservation with albedo-scaling | Medium | Physically correct layer blending | ✅ |
| RT/Raster unification from one surface | Large | Single source for both pipelines | ✅ |
| Built-in layers: base, normal_map, ibl, emission | Medium | glTF PBR unified rendering path | ✅ |
| `samplerCube` support in surface declarations | Small | IBL cubemap textures | ✅ |
| `--pipeline` CLI parameter | Small | Pipeline selection without code changes | ✅ |

### Next — P6: Coat & sheen layers

| Item | Effort | Impact |
|---|---|---|
| `coat` layer (clearcoat as composable layer) | Medium | glTF clearcoat via layer system |
| `sheen` layer (fabric sheen as composable layer) | Medium | glTF sheen via layer system |

### Future — P7–P9: Extended layer system

| Item | Phase | Impact |
|---|---|---|
| `transmission` layer (BTDF + volume) | P7 | Glass, liquids via layer system |
| `@layer` custom functions | P8 | User-extensible layer system |
| Deferred pipeline mode (`--pipeline deferred`) | P9 | G-buffer rendering path |

### ✅ COMPLETE — P10: Ray tracing pipeline (full)

| Item | Effort | Impact | Status |
|---|---|---|---|
| RT stage types in grammar/AST | Large | Foundation for RT | ✅ |
| RT SPIR-V codegen (6 execution models) | Large | RT shader output | ✅ |
| `environment` declaration | Medium | Miss shader generation | ✅ |
| RT surface expansion | Medium | Closest-hit from `surface` | ✅ |
| SDF → intersection shader | Medium | Procedural RT geometry | ✅ |
| Callable shader dispatch | Medium | Multi-material RT | ✅ |
