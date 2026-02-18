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

### Missing Built-ins

| Function | MaterialX | glTF PBR | Priority | Notes |
|---|---|---|---|---|
| `atan2(y, x)` | `atan2` | Anisotropy rotation | **HIGH** | GLSL has `atan(y,x)` overload. We only have 1-arg `atan`. Needed for anisotropy. |
| `refract(I, N, eta)` | `refract` | Volume, dispersion | **HIGH** | GLSL.std.450 #72. Essential for glass/transmission. |
| `mod(x, y)` | `modulo` | — | **MEDIUM** | We have `%` operator but no function form. MaterialX has it. |
| `round(x)` | `round` | — | **LOW** | MaterialX added v1.38.9. Nice-to-have. |
| `inversesqrt(x)` | — | Implicit in V_GGX | **MEDIUM** | GLSL.std.450 #32. Common in PBR for `1.0/sqrt(x)`. |
| `determinant(m)` | `determinant` | — | **LOW** | Matrix math. |
| `inverse(m)` | `invertmatrix` | — | **LOW** | Matrix math. |
| `transpose(m)` | `transpose` | — | **LOW** | Matrix math. |

**Verdict: 2 critical gaps** (`atan2` / 2-arg `atan`, `refract`), 1 medium (`inversesqrt`). The rest are nice-to-haves.

---

## Part 2: glTF PBR — Feature Gap Analysis

### Core PBR (metallic-roughness) — HAVE THIS

Our `pbr_brdf()` implements the full Cook-Torrance model:
- GGX NDF (D term) via `ggx_ndf()`
- Smith GGX geometry (G term) via `smith_ggx()`
- Schlick Fresnel (F term) via `fresnel_schlick()`
- Lambert diffuse via `lambert_brdf()`
- Metallic/dielectric blend via `pbr_brdf()`

**One discrepancy**: glTF specifies the **height-correlated** Smith G term (`V_GGX`), while our `smith_ggx()` uses the **separable** approximation. Technically not spec-compliant.

### glTF Extensions — What's Missing

| Extension | Status | What We'd Need |
|---|---|---|
| **clearcoat** | **MISSING** | Second GGX lobe with separate roughness + optional normal map. Math-wise we have everything — just need a `clearcoat_brdf()` function. |
| **transmission** | **MISSING** | Microfacet BTDF + background sampling with LOD. Needs: `refract()` built-in, `textureLod()` / mip-level sampling. |
| **volume** | **MISSING** | Beer-Lambert attenuation: `exp(-sigma * d)`. We have `exp()`, just need the library function. Also needs `refract()`. |
| **ior** | **TRIVIAL** | Just arithmetic: `((ior-1)/(ior+1))^2`. No new built-ins needed. |
| **specular** | **TRIVIAL** | Modifies F0/F90 with artistic controls. No new built-ins. |
| **sheen** | **MISSING** | Charlie NDF (different from GGX), fitted visibility term, LUT sampling. Needs new distribution function + `exp()` in fitted curve. |
| **iridescence** | **MISSING** | Thin-film interference. Most math-heavy extension. Needs spectral evaluation with `cos()` at optical frequencies, XYZ→RGB matrix multiply, multi-order Fresnel summation. We have all the built-ins but need a substantial library function. |
| **anisotropy** | **MISSING** | Anisotropic GGX NDF + Smith. Needs: `atan2()` (for rotation), tangent/bitangent handling, modified D and V functions. |
| **emissive_strength** | **TRIVIAL** | Just a scalar multiplier. No built-ins needed. |
| **dispersion** | **MISSING** | Per-channel IOR refraction. Needs `refract()` built-in (3× per pixel). |
| **diffuse_transmission** | **EASY** | Back-face Lambert. Same math as `lambert_brdf()` with flipped normal. |
| **unlit** | **TRIVIAL** | Bypass BRDF entirely. Already expressible. |
| **Normal mapping** | **MISSING** | TBN matrix construction from tangent + bitangent. Needs: `cross()` (have it), but also needs tangent/bitangent as stage inputs and mat3 construction. |
| **Occlusion** | **EASY** | Just `mix(1.0, ao, strength)`. We have everything. |

### Priority Tiers for glTF Compliance

**Tier 1 — Core compliance** (needed for any glTF viewer):
- `refract()` built-in
- `atan2()` built-in (or 2-arg `atan` overload)
- Height-correlated Smith G (`v_ggx_correlated()`)
- Normal mapping support (tangent/bitangent inputs)

**Tier 2 — Common extensions** (most glTF content uses these):
- Clearcoat BRDF library
- Sheen (Charlie NDF) library
- Transmission + Volume library
- Anisotropy (anisotropic GGX) library

**Tier 3 — Advanced** (specialty content):
- Iridescence library
- Dispersion (needs per-channel refraction)
- Diffuse transmission

---

## Part 3: MaterialX — Where We're Far Behind

MaterialX has **~180 node types** vs our **76 functions**. Most of the gap is in categories we don't need to compete on directly:

### Significant Gaps

| Category | MaterialX | Lux | Gap Size |
|---|---|---|---|
| **BSDF models** | 8 distinct lobes (Oren-Nayar, Burley, translucent, dielectric, conductor, generalized Schlick, subsurface, sheen, hair) | GGX + Lambert only | **Large** |
| **Uber-shaders** | 5 (Standard Surface, OpenPBR, glTF PBR, Disney, UsdPreviewSurface) | `pbr_brdf` only | **Medium** |
| **Color ops** | HSV/HSL conversion, contrast, saturation, hue adjust, color correct, remap | sRGB + luminance + tonemap only | **Medium** |
| **Compositing** | 17 blend/composite modes (Porter-Duff, screen, overlay, etc.) | None | **Medium** |
| **Texture** | Triplanar projection, hex tiling, normal map transform | `sample()` only | **Medium** |
| **Geometric** | Position, normal, tangent, bitangent, texcoord, vertex color access | Via stage inputs only | Architectural |
| **Procedural patterns** | Checkerboard, circles, hexagons, grids, ramps, 10+ patterns | Noise only | **Low** |
| **Volume** | Absorption VDF, anisotropic scattering (Henyey-Greenstein) | None | **Medium** |
| **Conditionals** | ifgreater, ifequal, switch(10) | Ternary only | **None** — ternary covers this |

### MaterialX Functions Worth Adding

**New stdlib module: `stdlib/material.lux`** (glTF PBR uber-shader):
- `gltf_pbr(base_color, metallic, roughness, normal, ...)` — the full model
- `clearcoat_layer(base, clearcoat, ccRoughness, ccNormal)`
- `sheen_layer(base, sheenColor, sheenRoughness)`
- `transmission_btdf(roughness, ior, baseColor)`
- `volume_attenuation(distance, attenuationColor, attenuationDist)`

**New stdlib module: `stdlib/colorspace.lux`**:
- `rgb_to_hsv(c)` / `hsv_to_rgb(c)`
- `contrast(c, pivot, amount)`
- `saturate_color(c, amount)`

**New BRDF functions for `stdlib/brdf.lux`**:
- `oren_nayar_diffuse(albedo, roughness, NdotL, NdotV)`
- `burley_diffuse(albedo, roughness, NdotL, NdotV, VdotH)`
- `charlie_ndf(roughness, NdotH)` — for sheen
- `v_ggx_correlated(NdotL, NdotV, alpha)` — height-correlated Smith
- `anisotropic_ggx_ndf(NdotH, TdotH, BdotH, at, ab)`
- `anisotropic_v_ggx(NdotL, NdotV, TdotV, BdotV, TdotL, BdotL, at, ab)`
- `iridescence_fresnel(outsideIOR, iridescenceIOR, baseIOR, thickness, cosTheta)`
- `conductor_fresnel(f0, f82, VdotH)` — for metals with complex IOR

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

### Immediate (P5.1 — built-in gaps)

| Item | Effort | Impact |
|---|---|---|
| Add `refract(I, N, eta)` built-in | Tiny | Unlocks transmission, volume, dispersion |
| Add 2-arg `atan(y, x)` built-in | Tiny | Unlocks anisotropy |
| Add `inversesqrt(x)` built-in | Tiny | Common in PBR |
| Height-correlated Smith G in brdf.lux | Small | glTF spec compliance |

### Near-term (P5.2 — stdlib expansions)

| Item | Effort | Impact |
|---|---|---|
| Clearcoat library | Small | glTF clearcoat extension |
| Sheen library (Charlie NDF) | Medium | glTF sheen extension |
| Anisotropic GGX library | Medium | glTF anisotropy extension |
| Burley/Oren-Nayar diffuse | Small | Better diffuse models |
| Color space module (HSV, contrast) | Small | Artistic shading |
| Normal mapping helpers (TBN) | Small | Essential for textured content |

### Medium-term (P5.3 — advanced material models)

| Item | Effort | Impact |
|---|---|---|
| Volume/transmission library | Medium | Glass, liquids |
| Iridescence library | Large | Thin-film effects |
| `gltf_pbr()` uber-shader | Medium | Full glTF material model |
| Triplanar projection | Small | Common texture technique |

### Long-term (P6 — ray tracing pipeline)

| Item | Effort | Impact |
|---|---|---|
| RT stage types in grammar/AST | Large | Foundation for RT |
| RT SPIR-V codegen (6 execution models) | Large | RT shader output |
| `environment` declaration | Medium | Miss shader generation |
| RT surface expansion | Medium | Closest-hit from `surface` |
| SDF → intersection shader | Medium | Procedural RT geometry |
| Callable shader dispatch | Medium | Multi-material RT |
