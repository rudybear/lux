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
    sh_degree: 0,          // SH bands: 0 (DC only), 1, 2, or 3
    kernel: ellipse,       // Gaussian kernel shape
    color_space: srgb,     // Output color space
    sort: camera_distance, // camera_distance (default) | view_depth
    dilation: 0.3,         // 3DGS/gsplat "eps2d" antialiasing convention
    alpha_min: 0.00392,    // ~= 1/255, minimum alpha threshold (OpKill below this)
}

pipeline SplatViewer {
    mode: gaussian_splat,
    splat: GaussianCloud,
}
```

The compiler generates:
- **Compute shader** (workgroup 256): transforms splat positions to screen space, computes 3D→2D covariance matrices via Jacobian projection, evaluates spherical harmonics for view-dependent color, and writes sort keys for depth ordering. Covariance dilation (`dilation`, default `0.3`) is added to the diagonal before inversion with **no opacity compensation** (matching the reference 3DGS rasterizer / gsplat exactly — earlier lux versions scaled opacity by `sqrt(det_orig/det)` to compensate for the dilation blur, which gsplat's default AA mode does not do).
- **Vertex shader**: reads sorted splat data and emits instanced screen-space quads (6 vertices per splat) sized to the Gaussian's 3σ extent (`radius = ceil(3*sqrt(lambda_max))`)
- **Fragment shader**: evaluates the 2D Gaussian kernel (`alpha = min(0.99, opacity*exp(-0.5*d²))`, discarding only on the trivial `power > 0` validity check in between — i.e. evaluated out to the full 3σ quad, not an arbitrary tighter cutoff), discards fragments below `alpha_min` (the reference-matching name for what was previously `alpha_cutoff` — still accepted as a legacy alias), and outputs premultiplied alpha for back-to-front compositing

`sort: camera_distance` (default, matches the ratified `KHR_gaussian_splatting` spec's `sortingMethod: cameraDistance`) sorts by true Euclidean distance from the camera; `sort: view_depth` (gsplat's own convention) sorts by raw view-space `z` instead. `camera_distance` is the `KHR_gaussian_splatting` spec default; `view_depth` is what the 3DGS reference rasterizer and gsplat use — pick `view_depth` when the scene or a downstream network was trained/validated against those renderers, as `examples/gaussian_splat_dlss.lux` does (DLSS-style reconstruction reads its proxy colour/motion/depth). (Metal's hand-written splat pipeline has no compiled `.lux` config to read this option from — it mirrors the same choice as a host-level `--sort camera_distance|view_depth` CLI flag instead, which you must pass explicitly to match a `sort: view_depth` pipeline in headless Metal renders; see `docs/rendering-engines.md`.)

**Note**: the antialiasing/alpha changes above are a real behavior change from lux's previous defaults (no compensation, wider fragment-tail evaluation, `1/255` instead of `0.004` for the threshold) — recompiled splat pipelines render measurably differently (there is no committed golden-image regression fixture in this repo to regenerate).

Push constants provide camera matrices, screen dimensions, focal lengths, splat count, and SH degree. SSBOs carry per-splat input data (positions, rotations, scales, opacities, SH coefficients) and intermediate projected data.

SH degrees 0–3 are supported (1/4/9/16 coefficient buffers). The stdlib module `gaussian.lux` provides quaternion-to-rotation, 3D/2D covariance, Gaussian evaluation, and quad radius helpers.

Reflection JSON includes a `gaussian_splatting` section with SH degree, kernel type, color space, and input/output buffer lists for engine integration.

All three rendering engines (C++/Vulkan, Rust/ash, Python/numpy) support Gaussian splat rendering with CPU-side depth sorting and instanced draw.

#### Dynamic (4D) Gaussian Splatting

Add `motion: keyframes` to a `splat` block to animate the cloud via glTF morph
targets (see `docs/lux-4d-spec.md` for the full file-format spec):

```
splat DynamicGaussianCloud {
    sh_degree: 0,
    motion: keyframes,   // optional: emits a morph-apply compute pre-pass
}

pipeline DynamicSplatViewer {
    mode: gaussian_splat,
    splat: DynamicGaussianCloud,
}
```

When `motion: keyframes` is set, the compiler emits a **4th stage** — a
morph-apply compute shader that runs *before* preprocess and writes animated
position/rotation/SH-degree-0 (color) values into the same `splat_pos` /
`splat_rot` / `splat_sh0` buffers preprocess already reads, so preprocess
itself is completely unchanged. It blends up to two active keyframes'
sparse per-gaussian deltas (`base + weight_lo * delta_lo + weight_hi *
delta_hi`, renormalizing the resulting quaternion), so its cost is
proportional to the number of gaussians that actually move between the two
keyframes, not the total splat count. The host (loader + renderer) is
responsible for: parsing the glTF `targets`/`animation`/`weights`/extras,
uploading the immutable `splat_base_pos/rot/sh0` buffers and the
concatenated per-segment `morph_index`/`morph_delta_*` sparse delta buffers
once at load time, and setting the `segment_offset`/`segment_count`/
`weight_lo`/`weight_hi` push constants each frame from the current
animation time (`--time`/`--frame` in the playgrounds). Scenes with more
than two simultaneously non-zero target weights (not produced by the
reference one-hot `LINEAR` sampler) fall back to a CPU-side loop over all
targets — correctness over speed, since this case is not expected in
practice.

Compile with `python -m luxc examples/gaussian_splat_dynamic.lux`, which
writes `gaussian_splat_dynamic.morph.comp.spv` alongside the usual
`.comp.spv`/`.vert.spv`/`.frag.spv`. Reflection JSON for the morph stage
carries a `gaussian_splatting_morph` section (`role: "morph_apply"`,
`base_buffers`, `delta_buffers`, `output_buffers`); the preprocess stage's
own `gaussian_splatting` section also reports `motion: "keyframes"` (or
`"none"` for static splats).

#### DLSS Input-Contract Outputs (motion vectors + expected depth)

Add `motion_vectors: true` and/or `expected_depth: true` to a `splat` block
to make the pipeline additionally emit the two auxiliary render targets a
temporal upscaler (DLSS/FSR/XeSS-style) or a training/eval harness expects,
alongside `out_color` (see `docs/lux-4d-spec.md` section 3):

```
splat DlssGaussianCloud {
    sh_degree: 0,
    motion: keyframes,      // optional, independent of the two flags below
    motion_vectors: true,   // packs mv.xy into the shared "out_aux" fragment output
    expected_depth: true,   // packs depth into the shared "out_aux" fragment output
}

pipeline DlssSplatViewer {
    mode: gaussian_splat,
    splat: DlssGaussianCloud,
}
```

Both flags default to `false`; with both off, the generated shaders are
byte-for-byte identical to the plain 3-stage pipeline (no regression for
existing splat assets). `motion_vectors` and `expected_depth` are
independent of each other and of `motion: keyframes` — a purely static
splat cloud under camera motion alone can still emit motion vectors.

**`motion_vectors: true`** adds:
- an input buffer `splat_prev_pos` (vec4): the previous frame's animated
  world-space position per splat. The host double-buffers this — for
  static splats it may simply alias `splat_pos` (positions never change,
  so `mv` reduces to the camera-motion term only); for `motion: keyframes`
  splats the host copies `splat_pos` into it once per frame, *after* the
  morph-apply stage has written the current frame's positions and *before*
  the next frame's dispatch.
- two extra preprocess push-constant fields: `proj_matrix_unjittered` (the
  current frame's projection matrix *without* any `--jitter` offset baked
  in — `proj_matrix` itself may be jittered) and
  `prev_view_proj_unjittered` (the previous frame's combined, unjittered
  view-projection matrix). Jitter must never leak into the motion vector,
  so both the current and previous projections used for `projected_mv` are
  always the unjittered ones, even when the color/depth render itself uses
  a jittered `proj_matrix`.
- an output buffer `projected_mv` (vec2, pixels): backward motion vector,
  `mv = uv_curr - uv_prev`, such that `uv_prev = uv - mv` recovers the
  previous frame's pixel location of the same surface point. `uv` uses a
  top-left origin with +y down, matching the existing `pixel_center`
  convention in the vertex stage. Carried through as `frag_mv` (vertex →
  fragment) and written into the shared `out_aux` fragment output's `.xy`
  lanes (`x = frag_mv.x * alpha`, `y = frag_mv.y * alpha` — see the packed
  `out_aux` layout below) — i.e. **premultiplied by the same per-fragment
  alpha as `out_color`**, so the engine can reuse the identical
  `(ONE, ONE_MINUS_SRC_ALPHA)` blend state for the aux attachment and get
  the correct visibility-weighted average of overlapping splats for free.
  `out_aux.w` carries the genuine per-fragment alpha (see the packed
  `out_aux` note below for why this can't be recovered from `out_color`'s
  alpha instead), which the host divides by to un-premultiply `.xy`. On
  the very first rendered
  frame (no previous camera/position data yet), the host is expected to
  set `prev_view_proj_unjittered` equal to the current frame's unjittered
  view-projection and alias/copy `splat_prev_pos` to the current
  positions, which makes `mv` evaluate to (0, 0) exactly.

  The C++ playgrounds (Vulkan and Metal) apply this same "prev == curr"
  convention to every *single headless render* by default (there being no
  real previous frame to double-buffer from), which makes `--camera-json`/
  `--camera-json-prev` renders reflect the camera delta but *not* a
  `motion: keyframes` actor's own animation. `--time-prev <SECONDS>` /
  `--frame-prev <N>` overrides this for validation/tooling purposes: the
  morph animation is evaluated a second time, at this earlier time, into
  `splat_prev_pos`, so a single headless render's `mv` reflects real actor
  motion between two arbitrary times (`mv = projection(pos(time), cam) -
  projection(pos(time_prev), cam_prev)`). Combine with `--camera-json-prev`
  to also get a real camera delta in the same render. See
  `docs/rendering-engines.md`'s CLI flags table.

**`expected_depth: true`** adds an output buffer `projected_depth`
(scalar): the camera-space depth `t = -view_pos.z` already computed by the
preprocess stage's Jacobian projection (gsplat's `"ED"` — expected depth
— mode). Carried through as `frag_depth` and written into the shared
`out_aux` fragment output's `.z` lane (`z = frag_depth * alpha`), again
using the same premultiplied-alpha blend as color, so overlapping splats
contribute an alpha-weighted average depth and pixels with no splats at
all end up at exactly `0.0`.

**Packed `out_aux` fragment output**: when EITHER `motion_vectors` or
`expected_depth` is set, the fragment stage emits ONE additional `vec4`
attachment `out_aux`: `(x = frag_mv.x * alpha, y = frag_mv.y * alpha,
z = frag_depth * alpha, w = alpha)` — lanes for a disabled feature are
written `0.0`. `.w` is `vec4`'s reason for being here, and it MUST hold
the genuine per-fragment alpha, not a repurposed data channel: Vulkan's
fixed-function alpha blend factors (`SRC_ALPHA`/`ONE_MINUS_SRC_ALPHA`)
read "source alpha" from the 4th component of THIS attachment's own
fragment output specifically (not a shared/global value, and not
`out_color`'s alpha) — so whatever occupies `.w` directly controls how
every overlapping fragment decays into this attachment. An earlier
version of this design packed `frag_foreground * alpha` into `.w` instead
(reasoning the host could un-premultiply everything from `out_color`'s own
alpha afterward, since that math is agnostic to which attachment produced
the value) — that is a genuinely different, GPU-side correctness bug, not
a precision tradeoff: `ONE_MINUS_SRC_ALPHA` evaluates to ~1 (no decay) for
every fragment where `foreground` is 0 (the common "background" case),
turning the blend into an unbounded running SUM instead of a proper
"over" composite — measured as a 10-50x MV/depth error in heavily
overdrawn background regions on a real scene. This is the same class of
bug the historical vec2-vs-vec4 `out_depth` fix (below) guards against; it
just wasn't obvious it also applies to what VALUE occupies an existing
alpha lane, not only whether one exists at all.

Vs. the earlier two-attachment (`out_motion`, `out_depth`, each RGBA32F
with an always-`0.0` `.z`) design, `out_aux` packs 3 real values (mv.xy,
depth) with ZERO wasted lanes: attachment count 2→1 and bytes 32B→16B
whenever `foreground_coverage` is off (see below for when it's on).
`out_aux` stays `RGBA32F`, not `RGBA16F`: downgrading it to half-float
(even with the `.w`-alpha bug fixed) reproduces the same class of
catastrophic error, because premultiplied mv/depth values — unlike
`out_color`'s own `[0, 1]`-bounded channels — are not magnitude-bounded,
so repeated half-float blend-accumulation steps compound rounding error
badly over the many overlapping low-alpha fragments a real scene's
background can have.

`out_aux` is appended *after* `out_color` in declaration order (so
`out_color` is always location 0, `out_aux` takes location 1), and the
`frag_mv`/`frag_depth` vertex-stage varyings feeding it are appended after
the existing four, exactly as before — only the final fragment-output
packing changed, not the preprocess/vertex stages. Reflection JSON's
`gaussian_splatting` section (on the preprocess stage) still reports
independent `"motion_vectors"` and `"expected_depth"` booleans (each
buffer/push-constant addition is unaffected by the packing change), and
the buffer/push changes are visible in the usual
`descriptor_sets`/`push_constants` sections — see
`examples/gaussian_splat_dlss.lux` for a compiling example with both
flags and `motion: keyframes` all enabled together. See
`tests/test_dlss_outputs.py`'s
`TestExpectedDepth::test_overlapping_splats_depth_matches_over_compositing`.

**`aux_precision: float | half`** (optional, default `float`): a host-side format
hint for `out_aux`/`out_fg`, reflected into `gaussian_splatting.aux_precision` —
does NOT change the compiled shader at all (identical `vec4` outputs either way),
only which pixel format the host allocates the attachment as. `float` (default) is
the precision-safe design above (RGBA32F `out_aux`, RGBA16F `out_fg`). `half`
(RGBA16F `out_aux`, RG16F `out_fg`, `examples/gaussian_splat_dlss_half.lux`) was
built and measured, and **fails** the DLSS-consumer parity gate on both Vulkan and
Metal (MV/depth precision and, more sharply, `out_fg`'s 2-channel format not
carrying a real alpha component for the blend hardware to read) — see
`bench/lux_perf_ablation.md` for the numbers. Kept as a real, opt-in-only compiler
option, not a default.

Sub-pixel jitter (`--jitter jx jy` in the playgrounds), the previous-frame
double buffering, and the `--output-aux`/`--camera-json` headless dump and
camera-bridge flags are all host (playground) responsibilities, not
compiler ones — see `docs/lux-4d-spec.md` sections 3-4 for the full
runtime design.

**`foreground_coverage: true`** adds a per-pixel actor/foreground coverage
mask — a visibility-weighted composite of a per-splat is-foreground flag,
computed exactly like `mv`/`depth` above. It implies `expected_depth:
true` (auto-enabled even if not written explicitly). Composited into a
SECOND attachment `out_fg` (`vec4`: `x = frag_foreground * alpha`, `y`/`z`
unused, `w = alpha`) — it can't share `out_aux`'s lanes: `out_aux` already
uses all 3 non-alpha lanes for mv.xy/depth, and `.w` must independently
stay genuine alpha in EVERY blended attachment (see the `out_aux` note
above — the exact bug that note describes is what happens if you try to
squeeze `foreground` into `out_aux.w` instead of giving it its own
attachment). `out_fg` is safely `RGBA16F` (not `RGBA32F`, unlike
`out_aux`): `foreground` is bounded to `[0, 1]` just like alpha itself, so
it doesn't have the unbounded-magnitude precision problem `out_aux`'s
mv/depth values do — measured with no precision regression vs. `RGBA32F`
on the reference scene. The host divides `out_fg.x` by `out_fg.w` to
recover the coverage value in `[0, 1]`. This adds an input storage buffer
`splat_foreground` (scalar, one 0.0/1.0 value per splat)
and an output buffer `projected_foreground` (scalar); the preprocess
stage does no per-frame computation on it beyond a straight
buffer-to-buffer passthrough, since the flag is a static per-splat
property, not something derived from the current camera or animation
time. A `frag_foreground` varying carries it from vertex to fragment
stage. No new push-constant fields.

Populating `splat_foreground` on the host is a loader/renderer
responsibility: the C++ playground's `gltf_loader.cpp` reads the custom
`_FOREGROUND` glTF vertex attribute (`SCALAR`, `UNSIGNED_BYTE`,
normalized — 255 means foreground) when the splat primitive has one;
otherwise it falls back to "this gaussian has any morph-target delta"
(the union of `dynamics.targets[*].indices` across all keyframes, i.e. a
gaussian counts as foreground iff it ever moves); otherwise every splat
defaults to `0.0` (non-foreground). See `docs/rendering-engines.md` for
the host-side buffer wiring and the `--output-aux` `_fg.npy` output.

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
| `compositing` | 2 | IBL multi-scattering (Fdez-Aguera 2019), unified `compose_pbr_layers` (transmission, sheen, coat, IBL, emission) |
| `pbr_pipeline` | 1 | `pbr_shade()` — single-call PBR orchestration (direct lighting + IBL + all optional layers) |
| `gaussian` | 6 | SH constants (degrees 0–3), quaternion-to-rotation, 3D/2D covariance, Gaussian 2D eval, quad radius |
| `openpbr` | 18 | OpenPBR Surface v1.1: F82-tint conductor Fresnel, energy-preserving Oren-Nayar (EON), coat darkening/absorption/roughening, fuzz BRDF, specular IOR modulation, thin-film, direct lighting, full composition (9 layers), fast variants for mobile/low-end |
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
import openpbr;     // enables OpenPBR Surface v1.1 material model
```

The compiler searches `luxc/stdlib/` first, then the source file's directory.
Imports are transitive: `import openpbr;` also brings in `brdf` and `compositing`.

### OpenPBR Material Model

`import openpbr;` activates the OpenPBR Surface v1.1 material model (Adobe/ASWF standard).
The surface expander automatically detects the import and generates physically-based composition
using energy-conserving layer stacking instead of the default glTF PBR path.

**Supported layers**: `base`, `specular`, `coat`, `fuzz`, `thin_film`, `transmission`, `emission`, `subsurface`, `normal_map`

```
import openpbr;

surface CarPaint {
    sampler2d albedo_tex,
    layers [
        base(color: sample(albedo_tex, uv).xyz, metalness: 0.9, diffuse_roughness: 0.0),
        specular(weight: 1.0, color: vec3(1.0), roughness: 0.15, ior: 1.5),
        coat(weight: 1.0, roughness: 0.02, ior: 1.6, color: vec3(1.0), darkening: 1.0),
        thin_film(weight: 0.6, thickness: 0.5, ior: 1.4),
        emission(luminance: 0.0, color: vec3(1.0)),
    ]
}
```

**Key features:**
- F82-tint metal Fresnel (Lazanyi-Szirmay-Kalos conductor model)
- Energy-preserving Oren-Nayar diffuse (EON/Fujii 2024)
- Coat darkening with internal reflection compensation
- Specular weight IOR modulation
- Fuzz BRDF for velvet/fabric
- Thin-film iridescence
- Transmission with volume absorption
- Bindless uber-shader support (`--bindless` with extended SSBO: `_FLAG_OPENPBR`, fuzz/thin_film/subsurface/anisotropy flags)
- Schedule-based quality tiers: `diffuse_model` (eon/lambert/burley), `fuzz_model` (charlie), `specular_fresnel` (exact/schlick), `coat_fresnel` (exact/schlick)
- Fast variants: `openpbr_direct_fast` (Lambert + Schlick) and `openpbr_compose_fast` (no fuzz, simplified coat) for mobile/low-end

**Schedule example (quality tiers):**

```
schedule OpenPBRMobile {
    diffuse_model: lambert,
    specular_fresnel: schlick,
    coat_fresnel: schlick,
    tonemap: aces,
}

pipeline MobilePBR {
    geometry: StandardMesh,
    surface: CarPaint,
    schedule: OpenPBRMobile,
}
```

## Reconstruction Pass (DLSS-style Upscale/Accumulate)

A runtime port of `mobiledlss.train.reconstruct` (see `docs/lux-reconstruct-spec.md`
in the mobiledlss repo, and `SPECIFICATION.md` section 12.9 for the compute-stage
details). Declares the fixed shape of a temporal-accumulation upscaler and expands
to three compute stages -- `warp`, `apply` (un-multiplex the network's packed
per-pixel parameters + apply the resulting kernel, fused), `blend`:

```
reconstruct DlssReconstruct {
    s: 2,             // upscale factor (proxy -> target resolution ratio)
    k: 4,             // taps per axis of the predicted kernel (K*K taps total)
    param_stride: 1,  // network's own param_stride (see mobiledlss.train.model.ParamPredUNet)
    hidden: 8,        // recurrent hidden-state channel count
}

pipeline ReconstructPass {
    mode: reconstruct,
    reconstruct: DlssReconstruct,
}
```

`s`, `k`, `param_stride`, `hidden` are compile-time (one compiled pipeline is one
fixed network architecture); the three stages take runtime push-constant fields
for resolution (`target_w/h`, `proxy_w/h`, `net_w/h`) and per-frame jitter, so the
same compiled pipeline is reused across every clip/frame. Only the 2-way blend
(no scene-memory `blend3`) is covered, unless a `memory` sub-block is present
(below).

This is stage 1 of the port: the packed per-pixel network parameters
(`kernel_logits`/`blend_logits`/`hidden_raw`, concatenated exactly as
`ParamPredUNet.forward_packed` produces them) are supplied by the host as a
`packed_params` input, not computed by a network running inside lux -- see
`docs/lux-reconstruct-spec.md`'s stage 2 scope note.

### Scene-memory path (`memory` sub-block)

An optional `memory: { channels, hidden }` sub-block ports mobiledlss's explicit
per-scene memory (`docs/scene-memory-spec.md` in the mobiledlss repo --
`mobiledlss.train.scene_texture.SceneTexture` + `MemoryColorHead`): a learnable
per-scene feature texture, addressed by the sphere UV of each pixel's view ray,
decoded into an RGB "memory colour" that gives the blend a third option --
unlike the spatial upsample or warped history (both convex combinations of the
proxy image), the memory colour can be an arbitrary per-pixel colour, exactly
what an aggressively-pruned background needs (measured on mobiledlss:
28.4 -> 36.7 dB PSNR recovered at 80% background pruning, `m_juggle_p0.8`).

```
reconstruct DlssReconstructMem {
    s: 2,
    k: 4,
    param_stride: 1,
    hidden: 8,
    memory: {
        channels: 8,   // scene-texture feature channel count
        hidden: 16,    // MemoryColorHead decoder hidden width
    },
}

pipeline ReconstructMemPass {
    mode: reconstruct,
    reconstruct: DlssReconstructMem,
}
```

`memory.channels`/`memory.hidden` are compile-time, exactly like the parent
block's fields. When `memory` is present, luxc emits two additional compute
stages and changes the shape of `apply`/`blend`:

- **`bguv`**: per-pixel sphere UV (`mobiledlss.datagen.camera.sphere_uv` --
  the *far* intersection of the pixel's world-space view ray with a bounding
  sphere `bg_sphere` (`vec4`: centre.xyz, radius), `u = atan2(d.z, d.x)/2pi +
  0.5`, `v = acos(d.y)/pi`). Resolution- and camera-generic (`width`/`height`,
  pinhole intrinsics `k_params` (`fx,fy,cx,cy`), and a `cam_to_world` `mat4`
  are all push constants), so the host runs the *same* compiled stage twice
  per frame -- once at proxy resolution (feeding the network's own extra
  input-channel block, dumped via `--dump-bg-features`, see below) and once
  at target resolution (feeding `memory`) -- mirroring
  `mobiledlss.train.train.rollout`'s own dual sampling of `SceneTexture`. The
  camera-to-world matrix is convention-free once expressed as a world-space
  ray -- lux's camera bridge already handles the OpenCV -> GL conversion
  upstream of this push field.
- **`memory`**: bilinearly samples the `channels x tex_h x tex_w` feature
  texture at each pixel's sphere UV (`u` wraps around the seam, `v` clamps at
  the poles) and decodes the sampled feature vector through the
  `channels -> hidden -> 3` decoder MLP (`LeakyReLU(0.1)`, then `sigmoid`) --
  `MemoryColorHead`'s exact architecture. Writes both `bg_features_out` (the
  raw sampled feature, before decoding -- what `--dump-bg-features` saves at
  proxy resolution for the network host to consume) and `memory_color_out`
  (the decoded RGB, used by `blend` at target resolution).
- **`apply`** now predicts a 3-way blend: `blend_logits` selects 2-way vs.
  3-way purely by its own channel count (`s*s` vs. `3*s*s`, matching
  `pixel_shuffle_params`), softmax over `{spatial, history, memory}` instead
  of a single sigmoid alpha, written to a 3-channel `blend_out` buffer instead
  of `alpha_out`.
- **`blend`** performs the 3-way `blend3`/`renormalized_blend_weights` mix
  instead of the 2-way lerp: the history share is zeroed on disocclusion, the
  three (already-softmaxed) weights are renormalised to sum to 1, then
  `out = w_s*spatial + w_h*warped + w_m*memory`.

The network itself (proxy-resolution feature sampling feeding
`ParamPredUNet`'s extra input-channel block) still runs outside this pass --
see the stage-2 scope note above -- but `--dump-bg-features <DIR>` runs the
`bguv`+`memory` stages standalone at proxy resolution and writes the sampled
features as `.npy` so the network host can consume them.

Both playgrounds can run these stages standalone, driven entirely by a
directory of dumped `.npy` inputs (produced offline, e.g. by mobiledlss's
`tools/reconstruct_reference_dump.py`) instead of any splat rendering:

```
lux-playground --reconstruct-dump <DIR> [--reconstruct-out <DIR>] \
               [--reconstruct-pipeline examples/reconstruct]
lux-playground --dump-bg-features <DIR> [--reconstruct-pipeline examples/reconstruct_mem]
```

writing `out_f{t}.npy` / `hidden_f{t}.npy` per frame into the output directory
(default: the dump directory itself); when the compiled pipeline has the
memory path, the dump directory may additionally carry `bg_sphere.npy` (`[4]`,
centre+radius), `texture.npy` (`[C, H, W]`), `memory_head.npz` (`fc1_w, fc1_b,
fc2_w, fc2_b`) and per-frame `k_params_f{t}.npy`/`cam_to_world_f{t}.npy`, and
`out_f{t}.npy` reflects the memory path. See `docs/rendering-engines.md`'s CLI
table for the dump directory's full expected file layout.
