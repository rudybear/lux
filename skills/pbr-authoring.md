# PBR Authoring Best Practices

## When to Apply
Use this knowledge when creating physically-based materials, choosing material
parameters, or building surface declarations with layers.

## Key Rules
- **Energy conservation**: reflected + transmitted + absorbed = 1.0. Never exceed.
- **Metallic is binary**: Real-world metals have metallic = 1.0, dielectrics = 0.0.
  Values between 0 and 1 are only for transitions/blending (e.g., rust on metal).
- **Albedo constraints**: Dielectric albedo should stay in [0.02, 0.95] range.
  No real-world material is pure black (0.0) or pure white (1.0).
  Metals store reflectance (F0) in the albedo channel.
- **Roughness mapping**: Mirror = 0.0, Polished = 0.1, Brushed = 0.3,
  Satin = 0.5, Rough = 0.6, Matte = 0.9
- **IOR typical ranges**: Glass 1.5, Water 1.33, Diamond 2.42, Plastic 1.46-1.55,
  Gemstones 1.6-2.4, Air 1.0
- **Linear colour space**: All albedo values must be in linear sRGB. Convert
  from sRGB with `srgb_to_linear()` if needed.

## Patterns

### Basic dielectric (wood, plastic, ceramic)
```lux
surface WoodFloor {
    properties MatParams {
        base_color: vec3 = vec3(0.43, 0.24, 0.11),
        roughness: scalar = 0.65,
    },
    layers [
        base(
            albedo: MatParams.base_color,
            roughness: MatParams.roughness,
            metallic: 0.0
        ),
    ]
}
```

### Basic metal
```lux
surface BrushedGold {
    properties MatParams {
        reflectance: vec3 = vec3(1.0, 0.77, 0.34),
        roughness: scalar = 0.3,
    },
    layers [
        base(
            albedo: MatParams.reflectance,
            roughness: MatParams.roughness,
            metallic: 1.0
        ),
    ]
}
```

### Coated metal (car paint, lacquer)
```lux
surface CarPaint {
    properties MatParams {
        base_color: vec3 = vec3(0.7, 0.04, 0.02),
        roughness: scalar = 0.35,
        coat_strength: scalar = 1.0,
        coat_roughness: scalar = 0.05,
    },
    layers [
        base(
            albedo: MatParams.base_color,
            roughness: MatParams.roughness,
            metallic: 0.0
        ),
        coat(factor: MatParams.coat_strength, roughness: MatParams.coat_roughness),
    ]
}
```

### Transmissive (glass, water)
```lux
surface ClearGlass {
    properties MatParams {
        tint: vec3 = vec3(1.0, 1.0, 1.0),
        roughness: scalar = 0.0,
        ior_value: scalar = 1.52,
    },
    layers [
        base(
            albedo: MatParams.tint,
            roughness: MatParams.roughness,
            metallic: 0.0
        ),
        transmission(factor: 1.0, ior: MatParams.ior_value),
    ]
}
```

## Anti-Patterns
- **`metallic: 0.5`** — Not physically meaningful. Use 0.0 or 1.0.
- **`albedo: vec3(0.0)`** — Nothing in reality is pure black. Use >= 0.02.
- **`albedo: vec3(1.0)` on a dielectric** — Violates energy conservation.
  Maximum ~0.95.
- **`roughness: 0.0` on fabric/cloth** — Fabric is always rough (>= 0.7).
  Use `sheen` layer instead.
- **Mixing sRGB and linear values** — Always work in linear space.

## Reference
- glTF metallic-roughness spec: https://registry.khronos.org/glTF/specs/2.0/
- Physically Based (CC0 data): https://physicallybased.info/
- "PBR Guide" by Allegorithmic/Adobe
