# Language Reference

### Types

| Lux Type | SPIR-V | Notes |
|----------|--------|-------|
| `scalar` | `OpTypeFloat 32` | Always f32 |
| `int` / `uint` | `OpTypeInt 32` | Signed / unsigned |
| `bool` | `OpTypeBool` | |
| `vec2` / `vec3` / `vec4` | `OpTypeVector` | Float vectors |
| `ivec2/3/4` / `uvec2/3/4` | `OpTypeVector` | Integer vectors |
| `mat2` / `mat3` / `mat4` | `OpTypeMatrix` | Square column-major matrices |
| `mat4x3` | `OpTypeMatrix` | Non-square: 4 columns of 3-component vectors (used by RT `object_to_world`) |
| `sampler2d` | `OpTypeSampledImage` | Combined image sampler |
| `samplerCube` | `OpTypeSampledImage (Cube)` | Cubemap image sampler |
| `sampler2DArray` | `OpTypeSampledImage (2D, Arrayed)` | 2D texture array sampler (shadow maps) |
| `samplerCubeArray` | `OpTypeSampledImage (Cube, Arrayed)` | Cubemap array sampler (point light shadows) |
| `storage_image` | `OpTypeImage (sampled=0)` | Read/write storage image (see [Storage Images](#storage-images)) |
| `acceleration_structure` | `OpTypeAccelerationStructureKHR` | RT top-level acceleration structure |
| `type strict Foo = vec3` | Same as base type | Compile-time type safety, zero SPIR-V cost |

### Stage Blocks

```
vertex {
    in position: vec3;          // auto location=0
    in normal: vec3;            // auto location=1
    out frag_normal: vec3;      // auto location=0

    uniform MVP {               // auto set=0, binding=0
        model: mat4,
        view: mat4,
        projection: mat4,
    }

    push Camera { view_pos: vec3 }

    fn main() {
        builtin_position = projection * view * model * vec4(position, 1.0);
    }
}

fragment {
    in frag_normal: vec3;
    out color: vec4;

    sampler2d albedo_tex;       // auto set=0, binding=0

    fn main() {
        color = vec4(normalize(frag_normal), 1.0);
    }
}
```

### Declarative Materials

Lux v0.2 introduces high-level material declarations that expand to shader stages:

```
import brdf;

// Geometry: vertex layout + transform + outputs
geometry StandardMesh {
    position: vec3, normal: vec3, uv: vec2,
    transform: MVP { model: mat4, view: mat4, projection: mat4, }
    outputs {
        world_pos: (model * vec4(position, 1.0)).xyz,
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        frag_uv: uv,
        clip_pos: projection * view * model * vec4(position, 1.0),
    }
}

// Surface: material properties + BRDF
surface TexturedPBR {
    sampler2d albedo_tex,
    brdf: pbr(sample(albedo_tex, frag_uv).xyz, 0.5, 0.0),
}

// Pipeline: wires geometry to surface, compiler generates stages
pipeline PBRForward {
    geometry: StandardMesh,
    surface: TexturedPBR,
}
```

### Layered Surfaces

For complex PBR pipelines, use `layers [...]` instead of `brdf:` — one surface generates both forward and RT shaders. Illumination is declared separately in a `lighting` block:

```
import brdf;
import color;
import ibl;

surface GltfPBR {
    sampler2d base_color_tex,

    layers [
        base(albedo: srgb_to_linear(sample(base_color_tex, uv).xyz),
             roughness: sample(mr_tex, uv).y, metallic: sample(mr_tex, uv).z),
        normal_map(map: sample(normal_tex, uv).xyz),
        emission(color: srgb_to_linear(sample(emissive_tex, uv).xyz)),
    ]
}

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

pipeline GltfRT {
    mode: raytrace,
    surface: GltfPBR,
    lighting: SceneLighting,
    environment: HDRSky,
    schedule: HighQuality,
}
```

The `lighting` block separates illumination configuration (light sources, IBL samplers) from material response (surface layers). Pipelines reference both via `surface:` and `lighting:`. Pipelines without a `lighting:` member fall back to legacy hardcoded behavior for backward compatibility.

Layers are listed bottom-to-top (base first, outermost last). The compiler generates energy-conserving evaluation with `sample()` auto-rewritten to `sample_lod()` for RT.

#### Custom Layers with `@layer`

Define custom layers as annotated functions. The first 4 parameters (base color, normal, view, light) are provided automatically; remaining parameters come from layer arguments:

```
import toon;

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

Custom layers are validated at compile time (signature, return type, no name collision with built-in layers) and inserted in declaration order after built-in layers, before emission.

Built-in layer types:

| Layer | Block | Purpose | Parameters |
|-------|-------|---------|------------|
| `base` | surface | PBR direct lighting | albedo, roughness, metallic |
| `normal_map` | surface | TBN normal perturbation | map |
| `emission` | surface | Additive emission | color |
| `coat` | surface | Clearcoat | factor, roughness |
| `sheen` | surface | Sheen/fuzz | color, roughness |
| `transmission` | surface | Volumetric transmission | factor, ior, thickness |
| `directional` | lighting | Directional light source | direction, color |
| `ibl` | lighting | Image-based lighting | specular_map, irradiance_map, brdf_lut |
| `multi_light` | lighting | N-light evaluation with shadows | (reads from LightData + ShadowEntry SSBOs) |
| *custom* | surface | User-defined `@layer` function | function-specific |

### Compile-Time Features

Declare boolean feature flags and use `if` guards to conditionally include declarations:

```
features {
    has_normal_map: bool,
    has_clearcoat: bool,
}

geometry StandardMesh {
    position: vec3,
    normal: vec3,
    tangent: vec4 if has_normal_map,
    // ...
}

surface GltfPBR {
    sampler2d normal_tex if has_normal_map,
    sampler2d clearcoat_tex if has_clearcoat,

    layers [
        base(albedo: ..., roughness: ..., metallic: ...),
        normal_map(map: sample(normal_tex, uv).xyz) if has_normal_map,
    ]
}
```

Guards work on any comma-separated declaration item: surface samplers, layers, geometry fields, output bindings, schedule members, and pipeline members.

Module-level `if` blocks group multiple declarations:

```
if has_clearcoat {
    import clearcoat;
}
```

Feature expressions support `&&`, `||`, `!`, and parentheses:

```
sheen(color: ...) if (has_sheen && !has_clearcoat),
```

Features are resolved at compile time — disabled items are stripped before expansion. The generated SPIR-V contains no dead code.

### Material Property Pipeline

The `properties` block declares an abstract data source for runtime material parameters inside a `surface` declaration. Instead of hardcoding BRDF inputs, you declare typed fields with defaults -- the compiler generates a UBO (std140 layout) and reflection JSON so engines can fill in values at runtime from glTF materials or any other source.

```
import brdf;
import color;

surface GltfPBR {
    sampler2d base_color_tex,
    sampler2d metallic_roughness_tex,

    properties Material {
        base_color_factor: vec4 = vec4(1.0, 1.0, 1.0, 1.0),
        emissive_factor: vec3 = vec3(0.0),
        metallic_factor: scalar = 1.0,
        roughness_factor: scalar = 1.0,
        emissive_strength: scalar = 1.0,
        ior: scalar = 1.5,
        clearcoat_factor: scalar = 0.0,
        clearcoat_roughness_factor: scalar = 0.0,
        sheen_roughness_factor: scalar = 0.0,
        transmission_factor: scalar = 0.0,
        sheen_color_factor: vec3 = vec3(0.0),
    },

    layers [
        base(albedo: srgb_to_linear(sample(base_color_tex, uv).xyz)
                     * Material.base_color_factor.xyz,
             roughness: sample(metallic_roughness_tex, uv).y
                        * Material.roughness_factor,
             metallic: sample(metallic_roughness_tex, uv).z
                        * Material.metallic_factor),
    ]
}
```

Fields are accessed with qualified syntax (`Material.roughness_factor`) and can appear anywhere in layer expressions -- multiplied with texture samples, used directly, or composed with other fields. Swizzling works as expected (`Material.base_color_factor.xyz`).

The compiler generates:

- A **UBO** with std140 layout placed in the fragment shader (and closest-hit / mesh stages for RT and mesh pipelines)
- **Reflection JSON** (`*.lux.json`) that lists each field's name, type, byte offset, and default value -- engines read this to build the buffer without hardcoding struct layouts

All three engines (Python/wgpu, C++/Vulkan, Rust/ash) use the reflection JSON to wire glTF material properties into the generated UBO automatically. When loading a glTF model, each engine reads the material's `pbrMetallicRoughness`, clearcoat, sheen, transmission, and emissive parameters and fills the corresponding UBO fields. Fields not present in the glTF material fall back to the defaults declared in the `properties` block.

### Semantic Types

Prevent mixing coordinate spaces at compile time with zero runtime cost:

```
type strict WorldPos = vec3;
type strict ViewPos = vec3;
type strict WorldNormal = vec3;

fn transform(p: WorldPos) -> ViewPos {
    return (view * vec4(p, 1.0)).xyz;
}

// Compile error: cannot pass ViewPos where WorldPos is expected
let v: ViewPos = transform(my_view_pos);  // error!
```

Builtins like `normalize`, `dot`, `length` accept semantic types transparently — they operate on the underlying base type.

### Algorithm/Schedule Separation

Decouple *what* to render from *how* to render it:

```
surface CopperMetal {
    brdf: pbr(vec3(0.95, 0.64, 0.54), 0.3, 0.9),
}

schedule HighQuality {
    fresnel: schlick,
    distribution: ggx,
    geometry_term: smith_ggx,
    tonemap: aces,
}

schedule Mobile {
    distribution: ggx_fast,
    geometry_term: smith_ggx_fast,
    tonemap: reinhard,
}

pipeline DesktopForward {
    geometry: StandardMesh,
    surface: CopperMetal,
    schedule: HighQuality,
}
```

### Ray Tracing

Declarative RT pipelines expand surfaces to raygen + closest_hit + miss stages:

```
import brdf;

surface CopperMetal {
    brdf: pbr(vec3(0.95, 0.64, 0.54), 0.3, 0.9),
}

environment GradientSky {
    color: mix(vec3(1.0), vec3(0.5, 0.7, 1.0), 0.5),
}

pipeline PathTracer {
    mode: raytrace,
    surface: CopperMetal,
    environment: GradientSky,
    max_bounces: 1,
}
```

Or write RT stages manually with full control (see [Ray Tracing Stages](#ray-tracing-stages-manual) below).

#### RT Built-in Variables

| Variable | Type | SPIR-V BuiltIn | Valid Stages |
|----------|------|----------------|--------------|
| `launch_id` | `uvec3` | `LaunchIdKHR` | raygen |
| `launch_size` | `uvec3` | `LaunchSizeKHR` | raygen |
| `world_ray_origin` | `vec3` | `WorldRayOriginKHR` | closest_hit, any_hit, miss, intersection |
| `world_ray_direction` | `vec3` | `WorldRayDirectionKHR` | closest_hit, any_hit, miss, intersection |
| `ray_tmin` | `scalar` | `RayTminKHR` | closest_hit, any_hit, miss, intersection |
| `ray_tmax` | `scalar` | `RayTmaxKHR` | closest_hit, any_hit, miss, intersection |
| `hit_t` | `scalar` | `RayTmaxKHR` | closest_hit, any_hit |
| `instance_id` | `int` | `InstanceCustomIndexKHR` | closest_hit, any_hit, intersection |
| `primitive_id` | `int` | `PrimitiveId` | closest_hit, any_hit, intersection |
| `hit_kind` | `uint` | `HitKindKHR` | closest_hit, any_hit |
| `object_to_world` | `mat4x3` | `ObjectToWorldKHR` | closest_hit, any_hit, intersection |
| `world_to_object` | `mat4x3` | `WorldToObjectKHR` | closest_hit, any_hit, intersection |
| `incoming_ray_flags` | `uint` | `IncomingRayFlagsKHR` | closest_hit, any_hit, miss, intersection |
| `geometry_index` | `int` | `RayGeometryIndexKHR` | closest_hit, any_hit, intersection |

### Mesh Shaders

Mesh shaders provide a third rendering pipeline mode alongside rasterization and ray tracing. Instead of the traditional vertex/index pipeline, mesh shaders operate on meshlets -- small clusters of triangles processed by workgroups -- enabling GPU-driven rendering with fine-grained culling.

The surface declaration works unchanged; only the geometry processing stage changes:

```
pipeline GltfMesh {
    mode: mesh_shader,
    geometry: StandardMesh,
    surface: GltfPBR,
    schedule: HighQuality,
}
```

The `GltfMesh` pipeline lives alongside `GltfForward` and `GltfRT` in the same `gltf_pbr_layered.lux` file. Compile with `--define` to set hardware-matched limits:

```bash
luxc gltf_pbr_layered.lux --pipeline GltfMesh --features has_emission --define max_vertices=64 --define max_primitives=124 --define workgroup_size=32
```

The approach is data-driven: at runtime the engine queries hardware capabilities (`maxMeshOutputVertices`, `maxMeshOutputPrimitives`, `maxMeshWorkGroupSize`), builds meshlets that respect those limits, and compiles the shader with matching `--define` parameters. This ensures optimal meshlet sizing across different GPU vendors without source changes.

Mesh shaders require `VK_EXT_mesh_shader` and are supported in the C++ and Rust engines only -- Python/wgpu does not expose mesh shader support.

### Metal / MSL Backend

The Metal backend provides a native macOS renderer that runs the same Lux-compiled SPIR-V shaders as the Vulkan engines, transpiled to MSL at runtime via [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross).

**Architecture:**

```
  .lux source
    → luxc compiler → .spv (SPIR-V binary)
    → SPIRV-Cross (CompilerMSL) → MSL source
    → MTL::Device::newLibrary() → MTLLibrary → MTLFunction
    → MTL::RenderPipelineState / MTL::MeshRenderPipelineState
```

The transpilation happens once at pipeline creation. SPIRV-Cross maps Vulkan descriptor bindings (set, binding) to Metal buffer/texture/sampler indices, and the transpiler records this mapping so renderers bind resources at the correct Metal indices.

**Key implementation details:**

| Aspect | Approach |
|--------|----------|
| Windowing | GLFW + ObjC++ bridge (`metal_bridge.mm`) attaches a `CAMetalLayer` to the GLFW `NSWindow` |
| Shader transpilation | SPIRV-Cross `CompilerMSL` with MSL 3.0, discrete bindings, `force_native_arrays` |
| Push constants | Detected via SPIR-V binary scan (OpVariable + PushConstant storage class), mapped to `[[buffer(N)]]` |
| Vertex data | Buffer index 0 reserved for vertex stage-in; UBOs/SSBOs assigned sequentially |
| Textures | Auto-assigned by SPIRV-Cross; combined image-samplers produce matching texture + sampler indices |
| Depth | `MTL::PixelFormatDepth32Float`, compare less, write enabled |
| Coordinate system | Vulkan Y-flip projection matrix reused as-is; `MTL::WindingClockwise` compensates inverted winding |
| Mesh shaders | Metal 3 `MeshRenderPipelineDescriptor` with `[[mesh]]` + `[[fragment]]` functions; one threadgroup per meshlet |
| SoA vertex buffers | Positions and normals uploaded as `vec4` (16-byte stride) to match std430 SSBO layout |

**Requirements:**

- macOS 13+ (Ventura) with Metal 3 for mesh shaders, macOS 12+ for raster-only
- Apple Silicon or AMD GPU with Metal support
- Metal-cpp headers vendored in `playground_cpp/deps/metal-cpp/` ([download from Apple](https://developer.apple.com/metal/cpp/))
- SPIRV-Cross (fetched automatically by CMake via FetchContent)

### Storage Images

Storage images provide read/write image access in compute and raygen shaders. Declared with `storage_image` followed by a name and format:

```
compute {
    storage_image output_img: rgba8;

    fn main() {
        let gid: uvec3 = global_invocation_id;
        let color: vec4 = vec4(1.0, 0.0, 0.0, 1.0);
        image_store(output_img, gid.xy, color);
    }
}
```

Supported formats: `rgba8`, `rgba16f`, `rgba32f`, `rg16f`, `rg32f`, `r16f`, `r32f`, `r32i`, `r32ui`, `r11g11b10f`.

### Gaussian Splatting

First-class 3D Gaussian splatting via the `splat` declaration. One block generates a complete 3-stage pipeline: compute preprocess (projection, covariance, SH evaluation, sort keys), instanced vertex shader (sorted quad rendering), and fragment shader (2D Gaussian evaluation with alpha compositing).

```
splat GaussianCloud {
    sh_degree: 0,        // SH bands: 0 (DC only), 1, 2, or 3
    kernel: ellipse,     // Gaussian kernel shape
    color_space: srgb,   // Output color space
    sort: camera_distance,  // Sorting method
    alpha_cutoff: 0.004, // Minimum alpha threshold (OpKill below this)
}

pipeline SplatViewer {
    mode: gaussian_splat,
    splat: GaussianCloud,
}
```

The compiler generates:
- **Compute shader** (workgroup 256): transforms splat positions to screen space, computes 3D→2D covariance matrices via Jacobian projection, evaluates spherical harmonics for view-dependent color, and writes sort keys for depth ordering
- **Vertex shader**: reads sorted splat data and emits instanced screen-space quads (6 vertices per splat) sized by the 2D Gaussian radius
- **Fragment shader**: evaluates the 2D Gaussian kernel, applies alpha cutoff with `discard` (OpKill), and outputs premultiplied alpha for back-to-front compositing

Push constants provide camera matrices, screen dimensions, focal lengths, splat count, and SH degree. SSBOs carry per-splat input data (positions, rotations, scales, opacities, SH coefficients) and intermediate projected data.

SH degrees 0–3 are supported (1/4/9/16 coefficient buffers). The stdlib module `gaussian.lux` provides quaternion-to-rotation, 3D/2D covariance, Gaussian evaluation, and quad radius helpers.

Reflection JSON includes a `gaussian_splatting` section with SH degree, kernel type, color space, and input/output buffer lists for engine integration.

All three rendering engines (C++/Vulkan, Rust/ash, Python/numpy) support Gaussian splat rendering with CPU-side depth sorting and instanced draw.

### Bindless Rendering

The `--bindless` flag enables uber-shaders with runtime descriptor arrays, eliminating per-material descriptor switching:

```bash
luxc gltf_pbr_layered.lux --pipeline GltfForward --bindless
```

In bindless mode, texture samplers become runtime-indexed arrays with `nonuniformEXT` decoration. Materials are stored in an SSBO, and per-geometry `gl_GeometryIndexEXT` selects the material in RT pipelines. Requires descriptor indexing support (C++ and Rust engines only).

Texture sampling uses `sample_bindless(tex_array, index, uv)` and `sample_bindless_lod(tex_array, index, uv, lod)` instead of the standard `sample()`.

### `@binding(N)` Annotation

Override automatic binding assignment with an explicit binding number:

```
fragment {
    @binding(3)
    sampler2d my_texture;
    // ...
}
```

The compiler normally assigns bindings by declaration order. `@binding(N)` overrides this for specific resources when engine integration requires fixed binding slots.

### Debug Instrumentation

Lux provides first-class debug features that compile to zero instructions in release builds. Use `--debug` to enable:

```lux
import debug;

// Semantic types: prevent mixing coordinate spaces at compile time
type strict WorldPos = vec3;
type strict WorldNormal = vec3;

fragment {
    in world_normal: vec3;
    out color: vec4;

    push Material { roughness: scalar, metallic: scalar }

    fn main() {
        let n: WorldNormal = normalize(world_normal);

        // Runtime assertions (prints failure, continues — no shader kill)
        assert(roughness >= 0.0, "roughness must be non-negative");
        assert(roughness <= 1.0, "roughness exceeds valid range");

        // Runtime value inspection (visible in Vulkan validation layer output)
        debug_print("roughness={} metallic={}", roughness, metallic);

        // Entire block stripped in release — zero instructions, not just skipped
        @[debug] {
            debug_print("normal=({}, {}, {})", n.x, n.y, n.z);
            assert(!any_nan(n), "NaN in normal!");

            // Stdlib debug visualization helpers
            let viz: vec3 = debug_normal(n);
            let heat: vec3 = debug_heatmap(roughness);
        }

        color = vec4(n * 0.5 + vec3(0.5), 1.0);
    }
}
```

```bash
# Debug mode: all instrumentation active
luxc debug_features_demo.lux --debug -o shadercache/

# Release mode: debug_print, assert, @[debug] blocks all stripped
luxc debug_features_demo.lux -o shadercache/

# Keep debug_print/assert in release (no full debug overhead)
luxc debug_features_demo.lux --debug-print -o shadercache/

# Assert kills fragment invocation on failure
luxc debug_features_demo.lux --debug --assert-kill -o shadercache/

# Static analysis: warns about unguarded division, sqrt, normalize, pow, log
luxc debug_features_demo.lux --warn-nan -o shadercache/
```

### CPU Shader Debugger

Step through shader code on the CPU with no GPU required. Inspect every variable, detect NaN/Inf sources, and simulate different pixels with custom inputs.

```bash
# Interactive debugging (gdb-style REPL)
python -m luxc examples/debug_playground.lux --debug-run --stage fragment

# Batch mode: detect NaN/Inf and report exact source line
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch --check-nan
# {
#   "status": "completed",
#   "nan_detected": true,
#   "nan_events": [{"line": 55, "variable": "dir", "operation": "let", ...}],
#   "output": {"type": "vec4", "value": [0.403, 0.381, 0.345, 1.0]}
# }

# Trace every intermediate value
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch --dump-vars

# Breakpoint inspection — dump full scope at specific lines
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch --break 100 --break 113 --dump-at-break

# Debug a specific pixel (auto-computes uv, position, normal from coords)
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --pixel 960,540 --batch --check-nan

# Quick inline overrides (no JSON file needed)
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --set roughness=0.01 --set metallic=1.0 --batch

# Custom inputs from JSON file
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --input examples/debug_playground_inputs.json --batch --check-nan
```

Interactive session:
```
$ python -m luxc examples/debug_playground.lux --debug-run --stage fragment

lux-debug> start
Stopped at line 87
 >   87 |         let albedo: vec3 = sample(albedo_tex, uv).rgb;
lux-debug> step
  + albedo = vec3(0.800, 0.800, 0.800) (vec3)
Stopped at line 88
 >   88 |         let n: vec3 = normalize(world_normal);
lux-debug> step
  + n = vec3(0.000, 0.000, 1.000) (vec3)
lux-debug> break 113
lux-debug> continue
Hit breakpoint 1 at line 113
lux-debug> print d
  d = 0.141471 (scalar)
lux-debug> locals
  albedo = vec3(0.800, ...) roughness = 0.500 n_dot_l = 0.333 ...
lux-debug> continue
Output: vec4(0.403, 0.381, 0.345, 1.000)
```

Commands: `start`, `run`, `step`, `next`, `continue`, `finish`, `break <line>`, `delete <id>`, `print <var>`, `locals`, `list`, `source`, `output`, `quit`

See [debugger-guide.md](debugger-guide.md) for the full reference and [debug-session-transcript.md](debug-session-transcript.md) for a complete debugging walkthrough.

### Auto-Type Precision Optimization

Automatically classify variables as fp16-safe or fp32-required:

```bash
# Analyze only — print report, no code changes
luxc shader.lux --auto-type=report

# Emit OpDecorate RelaxedPrecision on fp16-safe variables
luxc shader.lux --auto-type=relaxed
```

Three-signal analysis architecture:
1. **Dynamic range tracing** — runs the AST interpreter with 60+ diverse inputs to profile actual value ranges
2. **Static interval analysis** — forward dataflow propagation of [lo, hi] intervals through 30+ expression types
3. **Name/usage heuristics** — pattern matching (e.g., `position` -> fp32, `roughness` -> fp16)

Conservative-union-with-veto logic: any doubt defaults to fp32. Source: `luxc/autotype/`.

### Ray Tracing Stages (Manual)

Or write RT stages manually with full control:

```
raygen {
    acceleration_structure tlas;
    ray_payload payload: vec4;

    fn main() {
        let origin: vec3 = vec3(0.0, 0.0, 2.0);
        let direction: vec3 = normalize(vec3(0.0, 0.0, -1.0));
        trace_ray(tlas, 0, 255, 0, 0, 0, origin, 0.001, direction, 1000.0, 0);
    }
}

closest_hit {
    ray_payload payload: vec4;
    hit_attribute attribs: vec2;

    fn main() {
        let shade: scalar = 1.0 / (1.0 + hit_t * hit_t * 0.1);
        payload = vec4(shade, shade, shade, 1.0);
    }
}

miss {
    ray_payload payload: vec4;

    fn main() {
        let t: scalar = world_ray_direction.y * 0.5 + 0.5;
        payload = vec4(mix(vec3(1.0), vec3(0.5, 0.7, 1.0), vec3(t)), 1.0);
    }
}
```

Or write mesh/task stages manually:

```
task {
    task_payload payload: MeshletPayload;

    fn main() {
        // Frustum cull meshlets on the task shader side
        payload.meshlet_id = workgroup_id.x;
        emit_mesh_tasks(1, 1, 1);
    }
}

mesh {
    mesh_output vertices: 64;
    mesh_output primitives: 126;

    fn main() {
        set_mesh_outputs(3, 1);
        // Emit a single triangle from meshlet data
    }
}
```

### Automatic Differentiation

Mark functions with `@differentiable` to auto-generate gradient functions:

```
@differentiable
fn energy(x: scalar) -> scalar {
    return x * x + sin(x);
}

fragment {
    in param: scalar;
    out color: vec4;
    fn main() {
        let val: scalar = energy(param);
        let grad: scalar = energy_d_x(param);  // auto-generated
        color = vec4(val, grad, 0.0, 1.0);
    }
}
```

The compiler generates `energy_d_x` using forward-mode differentiation rules for all supported operations.

### Standard Library

Import modules with `import <name>;` — functions are inlined at the call site.

| Module | Functions | Description |
|--------|-----------|-------------|
| `brdf` | 30+ | Fresnel (Schlick, conductor), NDF (GGX, Charlie, anisotropic), Geometry (Smith GGX, height-correlated), Diffuse (Lambert, Oren-Nayar, Burley), Composite (PBR, glTF PBR), Clearcoat, Sheen, Transmission BTDF, Volumetric refraction (Walter 2007), Iridescence (Belcour 2017), Dispersion (Abbe number), Volume attenuation |
| `ibl` | 8 | Specular/diffuse IBL contributions, Fresnel-roughness, GGX importance sampling, Hammersley sequence, combined glTF PBR+IBL with multi-scattering energy compensation |
| `sdf` | 18 | Sphere, box, round box, plane, torus, cylinder, capsule, union, intersection, subtraction, smooth union/subtraction, translate, scale, repeat, round, onion, elongate |
| `noise` | 13 | Hash functions (2D/3D), value noise, gradient/Perlin noise, FBM (4/6 octaves, 2D/3D, loop-unrolled), Voronoi 2D |
| `color` | 5 | linear-to-sRGB, sRGB-to-linear, luminance, Reinhard tonemap, ACES tonemap |
| `colorspace` | 8 | RGB-to-HSV, HSV-to-RGB, contrast, saturation, hue shift, brightness, gamma correction |
| `texture` | 11 | TBN normal perturbation, normal unpacking, triplanar projection (weights, UVs, blending), parallax offset, UV rotation, UV tiling |
| `lighting` | 7 | Distance/spot attenuation, evaluate directional/point/spot lights, unified light evaluation, branchless light direction selection |
| `shadow` | 4 | Basic shadow sampling, PCF4 shadow filtering, cascade selection, shadow UV computation |
| `toon` | 1 | Cartoon cel-shading with quantized NdotL + rim lighting (`@layer` function) |
| `compositing` | 2 | IBL multi-scattering (Fdez-Aguera 2019), layer compositing helpers |
| `gaussian` | 6 | SH constants (degrees 0–3), quaternion-to-rotation, 3D/2D covariance, Gaussian 2D eval, quad radius |
| `debug` | 5 | Normal visualization, depth grayscale, scalar heatmap, index coloring, UV checkerboard |

### Built-in Functions

**GLSL.std.450** (44 builtins + 2 aliases): `normalize`, `reflect`, `refract`, `pow`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` (2-arg atan), `exp`, `exp2`, `log`, `log2`, `sqrt`, `inversesqrt`, `abs`, `sign`, `floor`, `ceil`, `fract`, `round`, `trunc`, `min`, `max`, `clamp`, `mix`, `step`, `smoothstep`, `length`, `distance`, `cross`, `fma`, `mod`, `radians`, `degrees`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `faceforward`, `determinant`, `inverse`

**Native SPIR-V**: `dot` (OpDot), `any_nan` (OpIsNan + OpAny), `any_inf` (OpIsInf + OpAny)

**Texture sampling**:
| Function | SPIR-V | Description |
|----------|--------|-------------|
| `sample(tex, uv)` | `OpImageSampleImplicitLod` | Standard texture sample |
| `sample_lod(tex, uv, lod)` | `OpImageSampleExplicitLod` | Explicit mip level (IBL, RT) |
| `sample_grad(tex, uv, ddx, ddy)` | `OpImageSampleExplicitLod` | Explicit gradients |
| `sample_compare(tex, coords, ref)` | `OpImageSampleDrefExplicitLod` | Shadow/depth comparison |
| `sample_array(tex, uv, layer)` | `OpImageSampleExplicitLod` | Array layer sampling |
| `sample_bindless(array, index, uv)` | `OpImageSampleImplicitLod` | Bindless descriptor array |
| `sample_bindless_lod(array, index, uv, lod)` | `OpImageSampleExplicitLod` | Bindless with explicit LOD |

**Image query**:
| Function | SPIR-V | Returns | Description |
|----------|--------|---------|-------------|
| `texture_size(tex, lod)` | `OpImageQuerySizeLod` | `ivec2` | Texture dimensions at mip level |
| `texture_levels(tex)` | `OpImageQueryLevels` | `int` | Number of mipmap levels |
| `image_size(img)` | `OpImageQuerySize` | `ivec2` | Storage image dimensions |

**Ray tracing**: `trace_ray`, `report_intersection`, `execute_callable`, `ignore_intersection`, `terminate_ray`

**Mesh shader**: `set_mesh_outputs(vert_count, prim_count)`, `emit_mesh_tasks(gx, gy, gz)`

**Compute**: `barrier()`, `atomic_add`, `atomic_min`, `atomic_max`, `atomic_and`, `atomic_or`, `atomic_xor`, `atomic_exchange`, `atomic_compare_exchange`, `atomic_load`, `atomic_store`, `image_store`

### Compute Built-in Variables

| Variable | Type | SPIR-V BuiltIn | Valid Stages |
|----------|------|----------------|--------------|
| `local_invocation_id` | `uvec3` | `LocalInvocationId` | compute, mesh, task |
| `local_invocation_index` | `uint` | `LocalInvocationIndex` | compute, mesh, task |
| `workgroup_id` | `uvec3` | `WorkgroupId` | compute, mesh, task |
| `num_workgroups` | `uvec3` | `NumWorkgroups` | compute, mesh, task |
| `global_invocation_id` | `uvec3` | `GlobalInvocationId` | compute, mesh, task |

### User-Defined Functions

```
fn fresnel_schlick(cos_theta: scalar, f0: vec3) -> vec3 {
    return f0 + (vec3(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}
```

Module-level functions can be called from any stage block. They are inlined at the call site.

### Constants

```
const PI: scalar = 3.14159265;
```

### Imports

```
import brdf;        // loads luxc/stdlib/brdf.lux
import noise;       // loads luxc/stdlib/noise.lux
```

The compiler searches `luxc/stdlib/` first, then the source file's directory.
