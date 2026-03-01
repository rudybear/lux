# Layer Composition Patterns

## When to Apply
Use when building multi-layered materials that combine base, coat, sheen,
transmission, emission, and normal mapping layers.

## Key Rules
- **Layer order matters**: Lux evaluates layers in declaration order.
  The standard physically-correct ordering is:
  1. `base()` — always first, defines the primary BRDF
  2. `normal_map()` — perturbs the surface normal for all subsequent layers
  3. `transmission()` — replaces base reflection with refraction
  4. `sheen()` — adds energy on top at grazing angles (fabric)
  5. `coat()` — applies a clear-coat dielectric layer over everything
  6. `emission()` — adds emitted light (does not interact with other layers)

- **One base layer required**: Every surface must have exactly one `base()`.
- **Coat and sheen are additive**: They add energy on top of the base,
  so total reflected energy increases. Use moderate factors (0.3-0.8)
  unless modelling a true clear coat.
- **Transmission replaces base**: The `factor` controls the blend
  between reflected (base) and transmitted light.

## Patterns

### Coat-over-metal (chrome, lacquered brass)
```lux
surface LacqueredBrass {
    properties MatParams {
        reflectance: vec3 = vec3(0.91, 0.78, 0.42),
        roughness: scalar = 0.15,
        coat_factor: scalar = 0.8,
        coat_roughness: scalar = 0.03,
    },
    layers [
        base(
            albedo: MatParams.reflectance,
            roughness: MatParams.roughness,
            metallic: 1.0
        ),
        coat(factor: MatParams.coat_factor, roughness: MatParams.coat_roughness),
    ]
}
```

### Sheen-on-fabric (velvet, silk)
```lux
surface Velvet {
    properties MatParams {
        color: vec3 = vec3(0.35, 0.05, 0.08),
        roughness: scalar = 0.9,
        sheen_tint: vec3 = vec3(0.5, 0.3, 0.35),
        sheen_roughness: scalar = 0.6,
    },
    layers [
        base(
            albedo: MatParams.color,
            roughness: MatParams.roughness,
            metallic: 0.0
        ),
        sheen(color: MatParams.sheen_tint, roughness: MatParams.sheen_roughness),
    ]
}
```

### Transmission glass (window, bottle)
```lux
surface StainedGlass {
    properties MatParams {
        tint: vec3 = vec3(0.2, 0.6, 0.3),
        roughness: scalar = 0.02,
        ior_value: scalar = 1.52,
        transmission_factor: scalar = 0.9,
    },
    layers [
        base(
            albedo: MatParams.tint,
            roughness: MatParams.roughness,
            metallic: 0.0
        ),
        transmission(factor: MatParams.transmission_factor, ior: MatParams.ior_value),
    ]
}
```

### Emissive (neon, lava, LED)
```lux
surface NeonSign {
    properties MatParams {
        base_color: vec3 = vec3(0.02, 0.02, 0.02),
        glow_color: vec3 = vec3(2.0, 0.1, 4.0),
    },
    layers [
        base(
            albedo: MatParams.base_color,
            roughness: 0.8,
            metallic: 0.0
        ),
        emission(color: MatParams.glow_color),
    ]
}
```

### Full layered material (coated, normal-mapped)
```lux
surface ComplexCar {
    sampler2d normal_tex [if has_normal_map],
    properties MatParams {
        base_color: vec3 = vec3(0.02, 0.15, 0.45),
        roughness: scalar = 0.3,
        coat_factor: scalar = 1.0,
        coat_roughness: scalar = 0.04,
    },
    layers [
        base(
            albedo: MatParams.base_color,
            roughness: MatParams.roughness,
            metallic: 0.0
        ),
        normal_map(map: sample(normal_tex, uv).xyz) [if has_normal_map],
        coat(factor: MatParams.coat_factor, roughness: MatParams.coat_roughness),
    ]
}
```

## Anti-Patterns
- **Coat on fabric** — Real fabric doesn't have a clear coat. Use `sheen` instead.
- **`sheen` on metal** — Sheen models fibre scattering; metals don't have this.
- **Multiple `base()` layers** — Only one base is supported.
- **`transmission(factor: 1.0)` with `coat()`** — Fully transmissive + coat
  gives odd results. Reduce transmission factor or remove coat.
- **Emission as sole layer** — Always provide a `base()` even for emissive
  objects, as the base handles ambient/indirect light.

## Reference
- glTF KHR_materials_clearcoat
- glTF KHR_materials_sheen
- glTF KHR_materials_transmission
