"""System prompt builder for AI shader generation.

Covers the full Lux language surface: traditional vertex/fragment stages,
modern declarative syntax (surface/pipeline/geometry/lighting), all 12
stdlib modules, and compile-time features.
"""

from __future__ import annotations
from pathlib import Path

from luxc.ai.materials import build_material_reference


_EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def build_system_prompt(
    mode: str = "general",
    skills: list[str] | None = None,
    extra_context: str = "",
) -> str:
    """Construct the system prompt for Claude to generate Lux shaders.

    Parameters
    ----------
    mode:
        Prompt specialisation.  Only ``"general"`` is meaningful today;
        the parameter exists so future callers can request focused prompts
        without changing the signature.
    skills:
        Optional list of skill names to load and inject between the stdlib
        reference and constraints sections.
    extra_context:
        Optional additional context to insert after skills.
    """
    sections = [
        _HEADER,
        _GRAMMAR_REFERENCE,
        _BUILTIN_REFERENCE,
        _STDLIB_REFERENCE,
        build_material_reference(),
    ]

    # Inject skill context if requested
    if skills:
        from luxc.ai.skills import build_skill_context
        skill_text = build_skill_context(skills)
        if skill_text:
            sections.append(skill_text)

    if extra_context:
        sections.append(extra_context)

    sections.extend([
        _CONSTRAINTS,
        _examples_section(),
        _FOOTER,
    ])
    return "\n\n".join(sections)


def build_material_extraction_prompt() -> str:
    """Build a specialised prompt for P16.2 image-to-material extraction.

    The prompt instructs the model to analyse a photograph, extract PBR
    material properties, and emit a complete ``surface`` declaration with
    ``properties`` and ``layers [...]``.
    """
    base = "\n\n".join([
        _HEADER,
        _GRAMMAR_REFERENCE,
        _BUILTIN_REFERENCE,
        _STDLIB_REFERENCE,
        build_material_reference(),
        _CONSTRAINTS,
    ])
    return base + "\n\n" + _MATERIAL_EXTRACTION_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Prompt sections
# ---------------------------------------------------------------------------

_HEADER = """\
You are a shader programming expert. You write shaders in the Lux shader \
language, a math-first language that compiles to SPIR-V for Vulkan.

Lux supports TWO authoring styles:

1. **Traditional stage blocks** — explicit `vertex { ... }` and \
`fragment { ... }` blocks where you control every input, output, and \
operation.
2. **Declarative syntax (preferred for PBR materials)** — high-level \
`surface`, `geometry`, `pipeline`, and `lighting` declarations. The \
compiler generates the vertex and fragment stages automatically from the \
combination of these declarations.

When the user asks for a PBR or physically-based material, prefer the \
declarative surface/pipeline style. For low-level effects, traditional \
vertex/fragment stages are still appropriate.

Your output should be a complete, compilable Lux program."""

_GRAMMAR_REFERENCE = """\
## Lux Language Reference

### Types
- `scalar` (equivalent to GLSL float — always use `scalar`, never `float`)
- `int`, `uint`, `bool`
- `vec2`, `vec3`, `vec4` (float vectors)
- `ivec2`, `ivec3`, `ivec4` (integer vectors)
- `uvec2`, `uvec3`, `uvec4` (unsigned integer vectors)
- `mat2`, `mat3`, `mat4` (matrices)
- `sampler2d` (combined image sampler)
- `samplerCube` (cube-map sampler)
- `sampler2DArray` (array texture sampler)
- `void`

### Top-level Declarations
- Constants: `const NAME: type = value;`
- Type aliases: `type Alias = target_type;`
- Imports: `import module_name;` (available: brdf, color, noise, sdf, \
compositing, ibl, lighting, shadow, texture, toon, colorspace, debug)

### Traditional Stage Blocks
```
vertex {
    in name: type;
    out name: type;
    uniform BlockName { field: type, ... }
    push BlockName { field: type, ... }
    sampler2d name;
    fn main() { ... }
}

fragment {
    in name: type;
    out color: vec4;
    uniform BlockName { field: type, ... }
    push BlockName { field: type, ... }
    sampler2d name;
    fn main() { ... }
}
```

### Declarative Surface Declaration
```
surface Name {
    sampler2d tex_name [if feature],
    properties BlockName { field: type = default, ... },
    layers [
        base(albedo: expr, roughness: expr, metallic: expr),
        normal_map(map: expr) [if feature],
        transmission(factor: expr, ior: expr) [if feature],
        sheen(color: expr, roughness: expr) [if feature],
        coat(factor: expr, roughness: expr) [if feature],
        emission(color: expr) [if feature],
    ]
}
```

### Geometry Declaration
```
geometry Name {
    field: type [if feature],
    transform: UniformBlockName { field: type, ... }
    outputs {
        binding_name: expr [if feature],
    }
}
```

### Pipeline Declaration
```
pipeline Name {
    mode: raytrace | mesh_shader,    // optional, default is rasterize
    geometry: GeometryName,
    surface: SurfaceName,
    lighting: LightingName,          // optional
    schedule: ScheduleName,          // optional
    environment: EnvironmentName,    // optional (raytrace only)
}
```

### Lighting Declaration
```
lighting Name {
    sampler2DArray shadow_maps [if feature],
    samplerCube env_specular,
    properties BlockName { field: type = default, ... },
    layers [
        multi_light(count: expr, shadow_map: name, shadow_filter: pcf, \
max_lights: N) [if feature],
        ibl(specular_map: name, irradiance_map: name, brdf_lut: name),
    ]
}
```

### Feature Flags
```
features {
    has_feature_name: bool,
}
```

### Schedule Declaration
```
schedule Name {
    tonemap: aces | reinhard,
}
```

### Environment Declaration
```
environment Name {
    color: expr,
}
```

### Properties Block
```
properties Name {
    field: type = default_value,
}
```

### Functions
```
fn name(param: type, ...) -> return_type {
    ...
}
```

### Statements
- `let name: type = expr;` (variable declaration)
- `name = expr;` (assignment)
- `return expr;`
- `if (condition) { ... } else { ... }`

### Expressions
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `&&`, `||`, `!`
- Ternary: `condition ? then : else`
- Constructors: `vec3(1.0, 2.0, 3.0)`, `vec3(0.5)` (splat)
- Swizzle: `v.xyz`, `v.rg`
- Function calls: `sin(x)`, `dot(a, b)`
- Texture sampling: `sample(sampler_name, uv)`

### Special Variables
- `builtin_position` — write in vertex shader (equivalent to gl_Position)"""

_BUILTIN_REFERENCE = """\
## Built-in Functions

### Math (scalar and vecN overloads)
- 1-arg: `abs`, `floor`, `ceil`, `fract`, `sqrt`, `inversesqrt`, `sin`, \
`cos`, `tan`, `asin`, `acos`, `atan`, `exp`, `exp2`, `log`, `log2`, \
`sign`, `normalize`
- 2-arg: `min`, `max`, `pow`, `step`, `reflect`, `mod`
- 3-arg: `mix`, `clamp`, `smoothstep`, `fma`, `refract`

### Vector-specific
- `length(v)` -> scalar
- `distance(a, b)` -> scalar
- `dot(a, b)` -> scalar
- `cross(a, b)` -> vec3
- `normalize(v)` -> same type

### Texture
- `sample(sampler, uv)` -> vec4
- `sample_lod(sampler, uv, lod)` -> vec4 (explicit LOD sampling)
- For `samplerCube`, pass a direction `vec3` instead of `vec2` uv."""

_STDLIB_REFERENCE = """\
## Standard Library Modules

### brdf (import brdf;)
- `fresnel_schlick(cos_theta: scalar, f0: vec3) -> vec3`
- `ggx_ndf(n_dot_h: scalar, roughness: scalar) -> scalar`
- `smith_ggx(n_dot_v: scalar, n_dot_l: scalar, roughness: scalar) -> scalar`
- `lambert_brdf(albedo: vec3, n_dot_l: scalar) -> vec3`
- `microfacet_brdf(n: vec3, v: vec3, l: vec3, roughness: scalar, \
f0: vec3) -> vec3`
- `pbr_brdf(n: vec3, v: vec3, l: vec3, albedo: vec3, roughness: scalar, \
metallic: scalar) -> vec3`
- `gltf_pbr(n, v, l, albedo, roughness, metallic) -> vec3` — glTF-spec \
PBR with height-correlated Smith
- `clearcoat_brdf(n, v, l, clearcoat, clearcoat_roughness) -> scalar`
- `sheen_brdf(sheen_color, roughness, n_dot_h, n_dot_l, \
n_dot_v) -> vec3` — Charlie NDF
- `transmission_btdf(n, v, l, roughness, ior) -> scalar` — thin-surface \
transmission
- `iridescence_fresnel(outside_ior, film_ior, base_f0, thickness, \
cos_theta) -> vec3`
- `conductor_fresnel(f0, f82, v_dot_h) -> vec3` — metals with complex IOR
- `ior_to_f0(ior) -> scalar`
- `volume_attenuation(distance, atten_color, atten_dist) -> vec3`
- Type aliases: `Radiance`, `Reflectance`, `Direction`, `Normal`, \
`Irradiance` (all = vec3)

### color (import color;)
- `linear_to_srgb(c: vec3) -> vec3`
- `srgb_to_linear(c: vec3) -> vec3`
- `luminance(c: vec3) -> scalar`
- `tonemap_reinhard(hdr: vec3) -> vec3`
- `tonemap_aces(hdr: vec3) -> vec3`

### noise (import noise;)
- `hash21(p: vec2) -> scalar`, `hash22(p: vec2) -> vec2`
- `hash31(p: vec3) -> scalar`, `hash33(p: vec3) -> vec3`
- `value_noise2d(p: vec2) -> scalar`, `value_noise3d(p: vec3) -> scalar`
- `gradient_noise2d(p: vec2) -> scalar`, \
`gradient_noise3d(p: vec3) -> scalar`
- `fbm2d_4(p: vec2, lacunarity: scalar, gain: scalar) -> scalar` \
(4 octaves, 2D)
- `fbm2d_6(p: vec2, lacunarity: scalar, gain: scalar) -> scalar` \
(6 octaves, 2D)
- `fbm3d_4(p: vec3, lacunarity: scalar, gain: scalar) -> scalar` \
(4 octaves, 3D)
- `fbm3d_6(p: vec3, lacunarity: scalar, gain: scalar) -> scalar` \
(6 octaves, 3D)
- `voronoi2d(p: vec2) -> vec2` — returns (cell_distance, edge_distance)

### sdf (import sdf;)
- SDF primitives: `sdf_sphere`, `sdf_box`, `sdf_cylinder`, etc.
- CSG: `sdf_union`, `sdf_intersection`, `sdf_difference`, \
`sdf_smooth_union`
- Transforms: `sdf_translate`, `sdf_scale`

### compositing (import compositing;)
- `ibl_contribution(albedo, roughness, metallic, n_dot_v, prefiltered, \
irradiance, brdf_sample) -> vec3`
- `coat_over(base, n, v, l, coat_factor, coat_roughness) -> vec3`
- `coat_ibl(coat_factor, coat_roughness, n, v, \
prefiltered_coat) -> vec3`
- `sheen_over(base, n, v, l, sheen_color, sheen_roughness) -> vec3`
- `transmission_replace(base, albedo, roughness, trans_factor, ior, n, \
v, l, thickness, atten_color, atten_dist) -> vec3`
- `compose_pbr_layers(...)` — unified PBR composition in \
glTF-standard order

### ibl (import ibl;)
- `ibl_specular_contribution(n_dot_v, roughness, f0, \
brdf_lut_sample) -> vec3`
- `ibl_diffuse_contribution(irradiance, albedo, metallic, f0, \
n_dot_v) -> vec3`
- `sh_irradiance_l0(...)`, `sh_irradiance_l1(...)`, \
`sh_irradiance_l2(...)` — spherical harmonics
- `gltf_pbr_ibl(n, v, l, albedo, roughness, metallic, prefiltered, \
irradiance, brdf_sample, ao, light_color) -> vec3`

### lighting (import lighting;)
- Constants: `MAX_LIGHTS`, `LIGHT_DIRECTIONAL`, `LIGHT_POINT`, \
`LIGHT_SPOT`, `LIGHT_AREA`
- `distance_attenuation(dist, range) -> scalar`
- `spot_attenuation(cos_angle, inner_cone, outer_cone) -> scalar`
- `evaluate_directional_light(direction, color, intensity, \
surface_normal) -> vec3`
- `evaluate_point_light(light_pos, color, intensity, range, \
surface_pos, surface_normal) -> vec3`
- `evaluate_spot_light(light_pos, light_dir, color, intensity, range, \
inner_cone, outer_cone, surface_pos, surface_normal) -> vec3`
- `evaluate_light(light_type, light_pos, light_dir, color, intensity, \
range, inner_cone, outer_cone, surface_pos, surface_normal) -> vec3`

### shadow (import shadow;)
- `sample_shadow_basic(shadow_depth, fragment_depth, bias) -> scalar`
- `sample_shadow_pcf4(d00, d10, d01, d11, fragment_depth, \
bias) -> scalar`
- `select_cascade(view_depth, split0..3) -> scalar`
- `compute_shadow_uv(world_pos, shadow_vp, bias) -> vec3`
- `normal_offset_world(pos, normal, normal_bias, n_dot_l) -> vec3`
- `slope_scale_bias(n_dot_l, constant_bias, slope_scale) -> scalar`
- PCF kernels: `pcf_shadow_4(...)`, `pcf_shadow_9(...)`, \
`pcf_shadow_16(...)` — unrolled averaging

### texture (import texture;)
- `tbn_perturb_normal(normal_sample, world_normal, world_tangent, \
world_bitangent) -> vec3`
- `unpack_normal(encoded) -> vec3`
- `unpack_normal_strength(encoded, strength) -> vec3`
- `triplanar_weights(normal, sharpness) -> vec3`
- `triplanar_uv_x(world_pos) -> vec2`, \
`triplanar_uv_y(world_pos) -> vec2`, \
`triplanar_uv_z(world_pos) -> vec2`
- `triplanar_blend(sx, sy, sz, weights) -> vec3`
- `parallax_offset(height, scale, view_dir_ts) -> vec2`
- `rotate_uv(uv, angle, center) -> vec2`
- `tile_uv(uv, scale) -> vec2`
- `transform_uv(uv, st, rot) -> vec2`

### toon (import toon;)
- `@layer cartoon(base, n, v, l, bands, rim_power, \
rim_color) -> vec3` — cel-shading with rim lighting

### colorspace (import colorspace;)
- `rgb_to_hsv(c) -> vec3`, `hsv_to_rgb(c) -> vec3`
- `contrast(c, pivot, amount) -> vec3`
- `saturate_color(c, amount) -> vec3`
- `hue_shift(c, shift) -> vec3`
- `brightness(c, amount) -> vec3`
- `gamma_correct(c, gamma) -> vec3`

### debug (import debug;)
- `debug_normal(n) -> vec3`
- `debug_depth(z, near, far) -> vec3`
- `debug_heatmap(v) -> vec3`
- `debug_index_color(idx) -> vec3`
- `debug_uv_checker(uv, scale) -> vec3`"""

_CONSTRAINTS = """\
## Critical Constraints

1. **No loops** — Lux does not support `for`, `while`, or any loop \
construct. Use explicit unrolling or recursive function calls instead.
2. **No arrays** — Array types are not supported.
3. **Use `scalar` not `float`** — The float type is called `scalar` in Lux.
4. **Explicit types required** — All variables must have type annotations: \
`let x: scalar = 1.0;` (not `let x = 1.0;`)
5. **No type inference** — Function parameters and return types must be \
annotated.
6. **Semicolons required** — All statements end with `;`
7. **Fragment output** — Fragment shaders must output `color: vec4`.
8. **Vertex position** — Use `builtin_position = vec4(...)` instead of \
`gl_Position`.
9. **One file** — Vertex and fragment stages go in the same file.
10. **Prefer `surface` declarations for PBR** — When creating PBR or \
physically-based materials, use the declarative `surface` declaration with \
`layers [...]` rather than manual vertex/fragment blocks.
11. **Compiler-generated stages** — When using declarative syntax \
(`surface`, `geometry`, `pipeline`), the compiler generates vertex and \
fragment stages automatically. You do NOT need vertex/fragment blocks.
12. **`properties` blocks create uniform buffers** — A `properties` block \
inside a surface or lighting declaration is compiled into a GPU uniform \
buffer automatically.
13. **Feature flags enable compile-time conditionals** — Declare features \
in a `features { ... }` block, then guard samplers, layers, or geometry \
fields with `[if feature_name]` for compile-time conditional compilation."""

_FOOTER = """\
Generate a complete, compilable Lux shader program based on the user's \
description. Output ONLY the Lux code, with no markdown fences or \
explanatory text. When the task calls for a PBR material, prefer the \
declarative surface/pipeline syntax."""

_MATERIAL_EXTRACTION_INSTRUCTIONS = """\
## Material Extraction Task

You are analysing a photograph to extract PBR material properties and \
produce a Lux `surface` declaration.

### Extraction Steps

1. **Base colour** — Identify the dominant RGB colour of the material in \
linear colour space. Express as a `vec3`.
2. **Roughness** — Estimate surface roughness on a 0.0 (mirror) to 1.0 \
(fully diffuse) scale.
3. **Metallic** — Estimate metalness: 0.0 for dielectrics (wood, fabric, \
plastic), 1.0 for bare metals (gold, copper, steel).
4. **Optional layers** — Determine whether any of these layers apply:
   - `coat` — for lacquered, glossy, or clear-coated surfaces (e.g. car \
paint, varnished wood). Estimate `factor` (0.0-1.0) and \
`roughness` (0.0-1.0).
   - `sheen` — for fabric, velvet, or cloth. Estimate `color` (vec3) and \
`roughness` (0.0-1.0).
   - `transmission` — for glass, water, or translucent materials. \
Estimate `factor` (0.0-1.0) and `ior` (index of refraction, \
e.g. 1.5 for glass).
   - `emission` — for glowing or self-illuminated areas. Estimate \
`color` (vec3, HDR values allowed).
5. **Procedural noise** (optional) — If the surface is non-uniform (e.g. \
natural stone, worn metal, fabric weave), suggest procedural noise \
parameters to vary base colour, roughness, or normal across the surface.

### Output Format

Emit a complete Lux program containing:
- Any necessary `import` statements (e.g. `import noise;`)
- A `properties` block with the extracted parameters as fields with \
defaults
- A `surface` declaration referencing the properties block and listing \
the appropriate `layers [...]`

Output ONLY the Lux code, with no markdown fences or explanatory text."""


def build_critique_prompt() -> str:
    """Build a specialised prompt for AI critique / validation mode.

    Automatically loads the ``debugging`` and ``pbr-authoring`` skills
    and instructs the AI to output a structured JSON list of issues.
    """
    base = build_system_prompt(skills=["debugging", "pbr-authoring"])
    return base + "\n\n" + _CRITIQUE_INSTRUCTIONS


def build_style_transfer_prompt() -> str:
    """Build a prompt for style-transfer / material modification.

    Loads ``layer-composition`` and ``pbr-authoring`` skills and instructs
    the AI to receive existing source + modification instruction and output
    a complete modified Lux program.
    """
    base = build_system_prompt(skills=["layer-composition", "pbr-authoring"])
    return base + "\n\n" + _STYLE_TRANSFER_INSTRUCTIONS


def build_batch_planning_prompt() -> str:
    """Build a prompt for batch material planning.

    Instructs the AI to generate a JSON list of material plans
    (name + description) for a scene description.
    """
    base = "\n\n".join([
        _HEADER,
        build_material_reference(),
    ])
    return base + "\n\n" + _BATCH_PLANNING_INSTRUCTIONS


def build_video_analysis_prompt() -> str:
    """Build a prompt for video-to-animation shader generation.

    Instructs the AI to generate an animated Lux shader using time-based
    noise and procedural patterns based on observed motion.
    """
    base = build_system_prompt(skills=["pbr-authoring"])
    return base + "\n\n" + _VIDEO_ANIMATION_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Specialised instruction blocks
# ---------------------------------------------------------------------------

_CRITIQUE_INSTRUCTIONS = """\
## Critique Task

You are reviewing a Lux shader program for correctness, physical plausibility,
and performance. Analyse the provided source code and output a JSON object
with the following structure:

```json
{
    "issues": [
        {
            "severity": "error|warning|info",
            "category": "physics|energy|performance|style|correctness",
            "message": "Description of the issue",
            "line": 10,
            "suggestion": "How to fix it"
        }
    ],
    "summary": "Brief overall assessment"
}
```

Check for:
- **correctness**: Syntax errors, type mismatches, undefined variables
- **physics**: Non-physical parameter values (albedo > 1.0, metallic 0.5)
- **energy**: Energy conservation violations, excessive brightness
- **performance**: Expensive operations, unguarded layers, heavy noise
- **style**: Naming conventions, code organisation, missing properties

Output ONLY the JSON object, no markdown fences or explanatory text."""

_STYLE_TRANSFER_INSTRUCTIONS = """\
## Material Modification Task

You will receive an existing Lux shader program and an instruction describing
how to modify it (e.g., "make it more weathered", "add clear coat",
"change to dark copper").

Rules:
1. Preserve the overall structure (surface name, properties block layout)
2. Modify only what the instruction requests
3. Keep all parameter values physically plausible
4. If adding a new layer, follow the correct layer ordering
5. Output the COMPLETE modified Lux program (not just the changes)

Output ONLY the Lux code, with no markdown fences or explanatory text."""

_BATCH_PLANNING_INSTRUCTIONS = """\
## Batch Material Planning Task

You will receive a scene description (e.g., "medieval tavern",
"sci-fi spacecraft interior"). Generate a JSON list of complementary
materials that would be needed for this scene.

Output format:
```json
[
    {"name": "tavern_wood_floor", "description": "Dark aged oak wood flooring with visible grain"},
    {"name": "tavern_stone_wall", "description": "Rough grey stone wall material"},
    ...
]
```

Rules:
- Each material should have a unique, descriptive snake_case name
- Descriptions should be specific enough to generate a complete PBR material
- Include a variety of material types (metals, dielectrics, fabrics, etc.)
- Materials should be complementary and coherent for the scene
- If no count is specified, generate 4-8 materials

Output ONLY the JSON array, no markdown fences or explanatory text."""

_VIDEO_ANIMATION_INSTRUCTIONS = """\
## Animated Shader Task

You will receive a motion description derived from video analysis.
Generate an animated Lux shader that captures the observed motion pattern.

Rules:
1. Use `import noise;` for procedural animation
2. Accept a `time` uniform for animation (add it to a properties or uniform block)
3. Use time-based noise (e.g., `noise.fbm2d_4(uv + vec2(time * 0.5, 0.0), 2.0, 0.5)`)
4. Match the described motion speed, direction, and pattern
5. Output a complete, compilable surface declaration
6. Keep parameters physically plausible

Output ONLY the Lux code, with no markdown fences or explanatory text."""


def _examples_section() -> str:
    """Load example shaders to include in the prompt."""
    examples = []
    for name in [
        "hello_triangle.lux",
        "pbr_basic.lux",
        "pbr_surface.lux",
        "cartoon_toon.lux",
    ]:
        path = _EXAMPLES_DIR / name
        if path.exists():
            src = path.read_text(encoding="utf-8")
            examples.append(f"### {name}\n```\n{src.strip()}\n```")

    if not examples:
        return "## Examples\n(no examples available)"
    return "## Examples\n\n" + "\n\n".join(examples)
