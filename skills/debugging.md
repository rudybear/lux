# Diagnosing Shader Artifacts

## When to Apply
Use when reviewing or critiquing Lux shaders for correctness issues,
or when a generated shader produces visual artifacts.

## Key Rules
- **NaN/Inf propagation**: Any NaN in a shader output turns the pixel
  black or produces flickering. Common sources:
  - `sqrt(x)` where x < 0 — clamp input: `sqrt(max(x, 0.0))`
  - `normalize(v)` where v = vec3(0) — check length first
  - `pow(0.0, 0.0)` — undefined in SPIR-V
  - Division by zero in roughness calculations
- **Energy gain (too bright)**: Material appears brighter than light source.
  Causes:
  - Albedo > 1.0 (non-physical)
  - Missing energy-conservation in custom BRDF
  - Additive coat + sheen without clamping
  - HDR emission values bleeding into reflection
- **Shadow acne**: Self-shadowing Z-fighting artifacts.
  Fix: increase bias in `compute_shadow_uv()` or use `normal_offset_world()`
- **Normal map artifacts**: Inverted normals, blue-tint seams.
  Fix: ensure `unpack_normal()` is applied, check tangent space
  construction.
- **Moire / aliasing**: Procedural patterns at high frequency.
  Fix: use `fwidth()`-based antialiasing or reduce noise frequency.
- **Dark edges on metals**: Missing Fresnel at grazing angles.
  Fix: use `fresnel_schlick()` or the standard `pbr_brdf()`.

## Patterns

### Safe sqrt
```lux
fn safe_sqrt(x: scalar) -> scalar {
    return sqrt(max(x, 0.0));
}
```

### Safe normalize
```lux
fn safe_normalize(v: vec3) -> vec3 {
    let len: scalar = length(v);
    return len > 0.001 ? v * (1.0 / len) : vec3(0.0, 1.0, 0.0);
}
```

### Energy conservation check
```lux
// After compositing all layers, total reflected energy should <= 1.0
// base_weight + coat_weight + sheen_weight <= 1.0
// If coat_factor = 0.5, base is implicitly scaled by (1 - 0.5)
```

### Debug visualization
```lux
import debug;

// Visualize normals as RGB
let dbg: vec3 = debug.debug_normal(world_normal);

// Visualize depth
let dbg_depth: vec3 = debug.debug_depth(gl_depth, 0.1, 100.0);

// Heatmap a scalar value (0=blue, 0.5=green, 1=red)
let dbg_heat: vec3 = debug.debug_heatmap(roughness);
```

## Anti-Patterns
- **`normalize(cross(a, b))` without zero check** — If a and b are
  parallel, cross product is zero vector → NaN from normalize.
- **`pow(base, exp)` with negative base** — SPIR-V `OpFPow` is undefined
  for negative base. Use `abs(base)` or guard.
- **`1.0 / roughness` without clamp** — Roughness = 0.0 → division by
  zero. Clamp to `max(roughness, 0.001)`.
- **Raw normal map sample without unpacking** — Normal maps store
  [0,1] values that must be remapped to [-1,1] with `unpack_normal()`.
- **Mixing world-space and tangent-space normals** — Always transform
  to the same space before BRDF evaluation.

## Reference
- Khronos SPIR-V spec: OpFPow, OpFDiv behaviour
- "GPU Gems" Ch. 8: Debugging Techniques
- Lux `debug` stdlib module for visualization helpers
