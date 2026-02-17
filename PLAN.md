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

## Phase 0: Shader Playground (Prove v0.1 Works)

Before evolving the language, prove the current backend works end-to-end by
rendering actual pixels.

### 0.1 — Headless Render Test Harness

Python script using `wgpu-py` (which accepts SPIR-V bytes directly via
`create_shader_module(code=spv_bytes)`). Renders to an offscreen texture,
saves PNG.

```
playground/
    render_harness.py      # wgpu-py offscreen renderer
    preview.py             # GLFW windowed interactive preview
    mesh.py                # Simple mesh loading (hardcoded triangle, OBJ)
    compare.py             # Compare Lux output vs glslangValidator output
```

**Test plan:**
1. Compile `hello_triangle.lux` → `.vert.spv` + `.frag.spv`
2. Load both into wgpu-py render pipeline
3. Render a colored triangle to 512x512 texture
4. Save as PNG — visual proof the shaders work
5. Compare against equivalent GLSL compiled with `glslangValidator`

**Dependencies:** `pip install wgpu glfw Pillow`

### 0.2 — Interactive Preview Window

GLFW window with live reload: edit `.lux` file, compiler re-runs, pipeline
rebuilds, frame updates. Supports orbit camera for mesh viewing.

### 0.3 — Reference Comparison Tool

Compile equivalent GLSL with `glslangValidator -V`, disassemble both with
`spirv-dis`, diff the structure. Automated structural equivalence checking.

---

## Phase 1: The Mathematical Core (The Real Language)

This is where Lux stops being "another shader language" and becomes what it
was meant to be.

### 1.1 — Radiometric Types

First-class physical quantity types. Not just `vec3` — **what kind of vec3**.

```lux
// Declare physical quantity types
type Radiance    = vec3  // W / (m² · sr)
type Irradiance  = vec3  // W / m²
type Reflectance = vec3  // dimensionless [0,1]
type Direction   = vec3  // unit vector (normalized)
type Normal      = vec3  // unit vector (normalized)

// The compiler enforces physical correctness
let light: Radiance = vec3(1.0, 0.9, 0.8);
let surface: Reflectance = vec3(0.5);
let result: Radiance = light * surface;      // OK: Radiance * Reflectance → Radiance
let wrong: Radiance = light + surface;       // ERROR: cannot add Radiance + Reflectance
```

**Implementation:** Type aliases with semantic tags. The type checker tracks
physical dimensions and rejects nonsensical operations. Compiles to the same
SPIR-V vec3 — zero runtime cost.

### 1.2 — BRDF as a First-Class Type

BRDFs are not functions you write — they're composable mathematical objects.

```lux
// A BRDF is a type: takes incident + outgoing directions, returns reflectance
type BRDF = (wi: Direction, wo: Direction, n: Normal) -> Reflectance

// Built-in BRDFs from the standard library
let diffuse: BRDF = lambert(albedo: vec3(0.8, 0.2, 0.1));

let specular: BRDF = microfacet_ggx(
    roughness: 0.3,
    f0: vec3(0.95, 0.64, 0.54),  // copper
);

// Compose BRDFs with mathematical operators
let material: BRDF = diffuse + specular;              // additive blend
let coated: BRDF = fresnel_blend(specular, diffuse);  // Fresnel-weighted
let layered: BRDF = layer(coat: specular, base: diffuse, ior: 1.5);
```

**Key insight from OSL:** Surface shaders produce a symbolic *radiance closure*
— a mathematical description of scattering — not a color. The renderer's
integrator evaluates the closure. We apply this at compile time: the compiler
expands BRDF compositions into concrete SPIR-V math.

### 1.3 — The `surface` Declaration

Declarative material specification. The compiler generates the fragment shader.

```lux
surface CopperMetal {
    brdf: fresnel_blend(
        reflection: microfacet_ggx(roughness: 0.3, f0: vec3(0.95, 0.64, 0.54)),
        base: lambert(albedo: vec3(0.01)),
    ),
    normal: sample(normal_tex, uv),
    occlusion: sample(ao_tex, uv).r,
}

surface Fabric {
    brdf: lambert(albedo: sample(diffuse_tex, uv).rgb)
        + sheen(roughness: 0.8, tint: vec3(1.0)),
    normal: sample(normal_tex, uv),
}
```

**What the compiler does:**
1. Expands `surface` → full fragment shader with proper light integration
2. Generates the `evaluate(wi, wo, n)` call for each BRDF term
3. Handles energy conservation automatically
4. Emits `max(dot, epsilon)` guards — the programmer never writes these
5. Handles the diffuse/specular Fresnel balance correctly

### 1.4 — The `geometry` Declaration

Declarative vertex transform specification.

```lux
geometry StandardMesh {
    position: vec3,
    normal: vec3,
    tangent: vec4,
    uv: vec2,

    transform: MVP {
        model: mat4,
        view: mat4,
        projection: mat4,
    },

    outputs {
        world_pos: (model * vec4(position, 1.0)).xyz,
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        frag_uv: uv,
        clip_pos: projection * view * model * vec4(position, 1.0),
    }
}
```

**What the compiler does:**
1. Generates vertex shader with proper `gl_PerVertex` output
2. Auto-assigns locations
3. Matches `geometry` outputs to `surface` inputs by type/name
4. Handles TBN matrix construction when `normal` and `tangent` are present

### 1.5 — The `pipeline` Declaration (Tying It Together)

```lux
pipeline PBRForward {
    geometry: StandardMesh,
    surface: CopperMetal,

    lighting: point_lights(max: 4),  // or: directional, environment_map, etc.
    tonemap: aces,
    output: srgb,
}
```

This single declaration generates **both** vertex and fragment SPIR-V modules,
with all the plumbing (locations, bindings, uniform blocks) handled automatically.

---

## Phase 2: Standard Library (`stdlib/`)

Written in Lux itself. The compiler can inline everything.

### 2.1 — `stdlib/brdf.lux` — BRDF Library

```lux
// Exact implementations of standard BRDFs
fn lambert(albedo: Reflectance) -> BRDF { ... }
fn microfacet_ggx(roughness: scalar, f0: Reflectance) -> BRDF { ... }
fn microfacet_beckmann(roughness: scalar, f0: Reflectance) -> BRDF { ... }
fn sheen(roughness: scalar, tint: Reflectance) -> BRDF { ... }

// BRDF combinators
fn fresnel_blend(reflection: BRDF, base: BRDF) -> BRDF { ... }
fn layer(coat: BRDF, base: BRDF, ior: scalar) -> BRDF { ... }
fn mix_brdf(a: BRDF, b: BRDF, factor: scalar) -> BRDF { ... }

// The NDF, masking, and Fresnel building blocks
fn ggx_ndf(roughness: scalar, n: Normal, h: Direction) -> scalar { ... }
fn smith_masking(roughness: scalar, ndotl: scalar, ndotv: scalar) -> scalar { ... }
fn fresnel_schlick(f0: Reflectance, cos_theta: scalar) -> Reflectance { ... }
fn fresnel_exact(ior: scalar, cos_theta: scalar) -> scalar { ... }
```

### 2.2 — `stdlib/sdf.lux` — Signed Distance Fields

```lux
type SDF = (p: vec3) -> scalar

fn sdf_sphere(radius: scalar) -> SDF { ... }
fn sdf_box(half_extents: vec3) -> SDF { ... }
fn sdf_cylinder(radius: scalar, height: scalar) -> SDF { ... }
fn sdf_torus(major: scalar, minor: scalar) -> SDF { ... }

// CSG operators — compose SDFs mathematically
fn sdf_union(a: SDF, b: SDF) -> SDF { ... }
fn sdf_intersection(a: SDF, b: SDF) -> SDF { ... }
fn sdf_subtraction(a: SDF, b: SDF) -> SDF { ... }
fn sdf_smooth_union(a: SDF, b: SDF, k: scalar) -> SDF { ... }

// Transforms
fn sdf_translate(s: SDF, offset: vec3) -> SDF { ... }
fn sdf_rotate(s: SDF, rotation: mat3) -> SDF { ... }
fn sdf_scale(s: SDF, factor: scalar) -> SDF { ... }

// Utilities
fn sdf_normal(s: SDF, p: vec3) -> Normal { ... }       // gradient-based normal
fn sdf_raymarch(s: SDF, origin: vec3, dir: Direction) -> scalar { ... }
```

### 2.3 — `stdlib/noise.lux` — Noise Functions

```lux
fn perlin2d(p: vec2) -> scalar { ... }
fn perlin3d(p: vec3) -> scalar { ... }
fn simplex2d(p: vec2) -> scalar { ... }
fn simplex3d(p: vec3) -> scalar { ... }
fn voronoi2d(p: vec2) -> vec2 { ... }   // returns (cell_dist, edge_dist)
fn fbm(p: vec3, octaves: int, lacunarity: scalar, gain: scalar) -> scalar { ... }
fn turbulence(p: vec3, octaves: int) -> scalar { ... }
```

### 2.4 — `stdlib/color.lux` — Color Science

```lux
fn linear_to_srgb(c: vec3) -> vec3 { ... }
fn srgb_to_linear(c: vec3) -> vec3 { ... }
fn tonemap_reinhard(hdr: Radiance) -> vec3 { ... }
fn tonemap_aces(hdr: Radiance) -> vec3 { ... }
fn luminance(c: vec3) -> scalar { ... }
```

---

## Phase 3: Schedule Separation (Halide-Inspired)

The algorithm says *what* to compute. The schedule says *how*.

### 3.1 — Approximation Schedules

```lux
// The math is exact
surface GlassBall {
    brdf: microfacet_ggx(roughness: 0.1, f0: fresnel_exact(ior: 1.5)),
}

// The schedule chooses approximations for real-time
schedule GlassBall {
    fresnel: schlick,              // use Schlick instead of exact Fresnel
    ggx_visibility: fast_smith,    // height-correlated Smith approximation
    precision: mediump where safe, // use half precision where it won't cause artifacts
}

// Different schedule for offline/reference rendering
schedule GlassBall::reference {
    fresnel: exact,
    ggx_visibility: full_smith,
    precision: highp,
}
```

### 3.2 — Compiler-Driven Optimization

The compiler, knowing the mathematical structure, can:

- **Constant-fold** uniform expressions (roughness² computed once, not per-pixel)
- **Merge identical BRDF evaluations** across surface layers
- **Approximate** based on schedule (Schlick vs exact Fresnel)
- **Dead-code-eliminate** BRDF terms that contribute nothing (e.g., specular on
  a perfectly rough surface)
- **Choose precision** per-operation (half for color, full for positions)

---

## Phase 4: Import System and Modules

```lux
import brdf;
import sdf;
import noise;

surface ProceduralRock {
    let base_color: Reflectance = mix(
        vec3(0.4, 0.35, 0.3),
        vec3(0.6, 0.55, 0.45),
        noise.fbm(world_pos * 2.0, 6, 2.0, 0.5)
    );

    brdf: brdf.lambert(albedo: base_color)
        + brdf.microfacet_ggx(roughness: 0.7, f0: vec3(0.04)),
    normal: perturb_normal(noise.simplex3d(world_pos * 5.0)),
}
```

**Implementation:** Resolve imports at parse time, merge ASTs, inline everything.
No dynamic linking — Lux compiles to monolithic SPIR-V.

---

## Phase 5: Differentiable Rendering (Future)

```lux
@differentiable
fn material_loss(params: MaterialParams, target: Image) -> scalar {
    let rendered = render(scene, params);
    return mse(rendered, target);
}

// Compiler auto-generates gradient code for backpropagation
let grads = backward(material_loss, current_params);
let new_params = current_params - learning_rate * grads;
```

This is the endgame: materials as differentiable programs. Useful for:
- Inverse rendering (fit material params from photos)
- Neural material models
- Procedural texture optimization

---

## Phase 6: AI Training Pipeline

### 6.1 — GLSL-to-Lux Transpiler

Automated conversion of existing GLSL shaders to Lux syntax. Target: the
Shadertoy corpus (thousands of shaders) → training data.

### 6.2 — Fine-Tuned Model

Create a Lux-specialized model (LoRA or small fine-tune) that:
- Understands radiometric types natively
- Composes BRDFs declaratively
- Writes `surface` + `pipeline` declarations, not imperative code
- Explains its rendering math in terms of optics, not API calls

### 6.3 — Ship With Integrated AI

`luxc --ai "frosted glass with subsurface scattering"` → generates a
complete `surface` declaration with appropriate BRDF composition.

---

## Implementation Priority

| Priority | What | Why |
|----------|------|-----|
| **P0** | Shader playground (Phase 0) | Prove v0.1 works, builds confidence |
| **P1** | Radiometric types (1.1) | Zero-cost, huge semantic value |
| **P1** | BRDF type + stdlib (1.2, 2.1) | This IS the language's identity |
| **P1** | `surface` declaration (1.3) | The killer feature — declarative materials |
| **P2** | `geometry` + `pipeline` (1.4, 1.5) | Full declarative pipeline |
| **P2** | SDF + noise stdlib (2.2, 2.3) | Enables procedural content |
| **P2** | Import system (Phase 4) | Composability |
| **P3** | Schedule separation (Phase 3) | Optimization without changing math |
| **P3** | IR layer + constant folding | Enables compiler intelligence |
| **P4** | Differentiable rendering (Phase 5) | Research frontier |
| **P4** | AI training pipeline (Phase 6) | The long game |

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
