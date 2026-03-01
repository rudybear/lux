# Performance-Aware Authoring

## When to Apply
Use when generating shaders that must run efficiently on mobile GPUs, in VR
(90+ fps requirement), or when the user mentions performance constraints.

## Key Rules
- **Minimize ALU in fragment shader**: Each additional layer (coat, sheen,
  transmission) adds a full BRDF evaluation per fragment. Three layers ≈ 3x
  cost of a single `base()`.
- **Feature flags for optional layers**: Guard expensive layers with
  `[if feature]` so the engine can compile stripped variants:
  ```lux
  features { has_coat: bool, has_normal: bool, }
  ```
- **Procedural vs. texture**: Procedural noise (`fbm`, `voronoi`) is
  compute-heavy. For static surfaces, prefer texture lookups.
  For animated/infinite surfaces, procedural is appropriate.
- **LOD strategy**: Provide simpler surface declarations for distant
  objects. Use `features` to strip `coat`, `sheen`, `normal_map` at
  lower detail levels.
- **Half-precision awareness**: Lux compiles to SPIR-V scalar (32-bit).
  The GPU driver may use 16-bit where possible, but complex `fbm` chains
  can lose precision in half-float mode.

## Patterns

### Feature-gated layers for LOD
```lux
features {
    has_coat: bool,
    has_normal_map: bool,
}

surface OptimizedMetal {
    sampler2d normal_tex [if has_normal_map],
    properties MatParams {
        reflectance: vec3 = vec3(0.91, 0.92, 0.92),
        roughness: scalar = 0.3,
    },
    layers [
        base(
            albedo: MatParams.reflectance,
            roughness: MatParams.roughness,
            metallic: 1.0
        ),
        normal_map(map: sample(normal_tex, uv).xyz) [if has_normal_map],
        coat(factor: 0.5, roughness: 0.05) [if has_coat],
    ]
}
```

### Cheap procedural variation (hash instead of fbm)
```lux
import noise;

fn cheap_variation(p: vec2) -> scalar {
    // Single hash call instead of multi-octave fbm
    return noise.hash21(p) * 0.1 + 0.9;
}
```

### Texture-based variation (fastest)
```lux
surface TexturedWood {
    sampler2d albedo_map,
    sampler2d roughness_map,
    properties MatParams {
        tint: vec3 = vec3(1.0, 1.0, 1.0),
    },
    layers [
        base(
            albedo: sample(albedo_map, uv).rgb * MatParams.tint,
            roughness: sample(roughness_map, uv).r,
            metallic: 0.0
        ),
    ]
}
```

## Anti-Patterns
- **`fbm3d_6` in fragment shader** — 6-octave 3D noise is very expensive.
  Use `fbm2d_4` or `value_noise2d` when possible.
- **Unguarded `coat + sheen + transmission`** — All three together is rare
  and extremely expensive. Gate with features.
- **Procedural noise for static props** — Bake to texture instead.
- **Multiple `sample()` calls for same texture** — Cache the result in a
  `let` variable.

## Reference
- "Real-Time Rendering" Ch. 9: Performance
- ARM Mali GPU Best Practices Guide
