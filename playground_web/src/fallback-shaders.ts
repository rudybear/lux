/**
 * Embedded WGSL fallback shaders for when pre-compiled shaders aren't available.
 *
 * Provides a minimal PBR-ish shader that reads from a uniform buffer with
 * the standard Lux camera layout and a push-emulated uniform for material params.
 */

/** Minimal vertex shader: transforms position by view/proj matrices. */
export const FALLBACK_VERT_WGSL = /* wgsl */`
struct CameraUniforms {
  view: mat4x4f,
  proj: mat4x4f,
  eye: vec3f,
  _pad0: f32,
  resolution: vec2f,
  time: f32,
  exposure: f32,
};

struct PushParams {
  light_dir: vec3f,
  metallic: f32,
  view_pos: vec3f,
  roughness: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> push: PushParams;

struct VertexInput {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
};

struct VertexOutput {
  @builtin(position) clip_position: vec4f,
  @location(0) world_pos: vec3f,
  @location(1) world_normal: vec3f,
  @location(2) uv: vec2f,
};

@vertex
fn main(in: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  let world_pos = vec4f(in.position, 1.0);
  out.clip_position = camera.proj * camera.view * world_pos;
  out.world_pos = in.position;
  out.world_normal = in.normal;
  out.uv = in.uv;
  return out;
}
`;

/** Minimal PBR fragment shader with directional light. */
export const FALLBACK_FRAG_WGSL = /* wgsl */`
struct CameraUniforms {
  view: mat4x4f,
  proj: mat4x4f,
  eye: vec3f,
  _pad0: f32,
  resolution: vec2f,
  time: f32,
  exposure: f32,
};

struct PushParams {
  light_dir: vec3f,
  metallic: f32,
  view_pos: vec3f,
  roughness: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> push: PushParams;

struct FragmentInput {
  @location(0) world_pos: vec3f,
  @location(1) world_normal: vec3f,
  @location(2) uv: vec2f,
};

const PI: f32 = 3.14159265359;

fn ggx_distribution(ndoth: f32, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let d = ndoth * ndoth * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

fn schlick_fresnel(cos_theta: f32, f0: vec3f) -> vec3f {
  return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

fn smith_ggx(ndotv: f32, ndotl: f32, roughness: f32) -> f32 {
  let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
  let g1 = ndotv / (ndotv * (1.0 - k) + k);
  let g2 = ndotl / (ndotl * (1.0 - k) + k);
  return g1 * g2;
}

@fragment
fn main(in: FragmentInput) -> @location(0) vec4f {
  let n = normalize(in.world_normal);
  let v = normalize(camera.eye - in.world_pos);
  let l = normalize(push.light_dir);
  let h = normalize(v + l);

  let ndotl = max(dot(n, l), 0.0);
  let ndotv = max(dot(n, v), 0.001);
  let ndoth = max(dot(n, h), 0.0);
  let vdoth = max(dot(v, h), 0.0);

  // Albedo from UV checkerboard
  let checker = step(0.5, fract(in.uv.x * 4.0)) * step(0.5, fract(in.uv.y * 4.0))
              + step(0.5, 1.0 - fract(in.uv.x * 4.0)) * step(0.5, 1.0 - fract(in.uv.y * 4.0));
  let base_color = mix(vec3f(0.8, 0.2, 0.2), vec3f(0.9, 0.9, 0.9), checker);

  let f0 = mix(vec3f(0.04), base_color, push.metallic);
  let diffuse_color = base_color * (1.0 - push.metallic);

  // Cook-Torrance BRDF
  let d = ggx_distribution(ndoth, push.roughness);
  let f = schlick_fresnel(vdoth, f0);
  let g = smith_ggx(ndotv, ndotl, push.roughness);

  let specular = (d * f * g) / max(4.0 * ndotv * ndotl, 0.001);
  let diffuse = diffuse_color / PI;

  let color = (diffuse + specular) * ndotl * 3.0;

  // Ambient
  let ambient = base_color * 0.03;

  // Tone mapping (Reinhard)
  var hdr = (color + ambient) * camera.exposure;
  hdr = hdr / (hdr + 1.0);

  // Gamma
  let gamma_color = pow(hdr, vec3f(1.0 / 2.2));

  return vec4f(gamma_color, 1.0);
}
`;

/** Reflection metadata for the fallback vertex shader. */
export const FALLBACK_VERT_REFLECTION = {
  version: 1,
  source: 'fallback',
  stage: 'vertex',
  execution_model: 'Vertex',
  inputs: [
    { name: 'position', type: 'vec3', location: 0 },
    { name: 'normal', type: 'vec3', location: 1 },
    { name: 'uv', type: 'vec2', location: 2 },
  ],
  outputs: [
    { name: 'world_pos', type: 'vec3', location: 0 },
    { name: 'world_normal', type: 'vec3', location: 1 },
    { name: 'uv', type: 'vec2', location: 2 },
  ],
  descriptor_sets: {
    '0': [
      {
        binding: 0,
        type: 'uniform_buffer' as const,
        name: 'CameraUniforms',
        fields: [
          { name: 'view', type: 'mat4', offset: 0, size: 64 },
          { name: 'proj', type: 'mat4', offset: 64, size: 64 },
          { name: 'eye', type: 'vec3', offset: 128, size: 12 },
          { name: 'resolution', type: 'vec2', offset: 144, size: 8 },
          { name: 'time', type: 'scalar', offset: 152, size: 4 },
          { name: 'exposure', type: 'scalar', offset: 156, size: 4 },
        ],
        size: 160,
        stage_flags: ['vertex'],
      },
    ],
  },
  push_constants: [],
  push_emulated: {
    set: 1,
    binding: 0,
    fields: [
      { name: 'light_dir', type: 'vec3', offset: 0, size: 12 },
      { name: 'metallic', type: 'scalar', offset: 12, size: 4 },
      { name: 'view_pos', type: 'vec3', offset: 16, size: 12 },
      { name: 'roughness', type: 'scalar', offset: 28, size: 4 },
    ],
    size: 32,
  },
  vertex_attributes: [
    { location: 0, type: 'vec3', name: 'position', format: 'R32G32B32_SFLOAT', offset: 0 },
    { location: 1, type: 'vec3', name: 'normal', format: 'R32G32B32_SFLOAT', offset: 12 },
    { location: 2, type: 'vec2', name: 'uv', format: 'R32G32_SFLOAT', offset: 24 },
  ],
  vertex_stride: 32,
};

/** Reflection metadata for the fallback fragment shader. */
export const FALLBACK_FRAG_REFLECTION = {
  version: 1,
  source: 'fallback',
  stage: 'fragment',
  execution_model: 'Fragment',
  inputs: [
    { name: 'world_pos', type: 'vec3', location: 0 },
    { name: 'world_normal', type: 'vec3', location: 1 },
    { name: 'uv', type: 'vec2', location: 2 },
  ],
  outputs: [
    { name: 'color', type: 'vec4', location: 0 },
  ],
  descriptor_sets: {
    '0': [
      {
        binding: 0,
        type: 'uniform_buffer' as const,
        name: 'CameraUniforms',
        fields: [
          { name: 'view', type: 'mat4', offset: 0, size: 64 },
          { name: 'proj', type: 'mat4', offset: 64, size: 64 },
          { name: 'eye', type: 'vec3', offset: 128, size: 12 },
          { name: 'resolution', type: 'vec2', offset: 144, size: 8 },
          { name: 'time', type: 'scalar', offset: 152, size: 4 },
          { name: 'exposure', type: 'scalar', offset: 156, size: 4 },
        ],
        size: 160,
        stage_flags: ['fragment'],
      },
    ],
  },
  push_constants: [],
  push_emulated: {
    set: 1,
    binding: 0,
    fields: [
      { name: 'light_dir', type: 'vec3', offset: 0, size: 12 },
      { name: 'metallic', type: 'scalar', offset: 12, size: 4 },
      { name: 'view_pos', type: 'vec3', offset: 16, size: 12 },
      { name: 'roughness', type: 'scalar', offset: 28, size: 4 },
    ],
    size: 32,
  },
  vertex_attributes: [],
  vertex_stride: 0,
};
