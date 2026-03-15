struct Material {
    base_color_factor: vec4<f32>,
    emissive_factor: vec3<f32>,
    metallic_factor: f32,
    roughness_factor: f32,
    emissive_strength: f32,
    ior: f32,
    clearcoat_factor: f32,
    clearcoat_roughness_factor: f32,
    sheen_roughness_factor: f32,
    transmission_factor: f32,
    _pad_sheen: f32,
    sheen_color_factor: vec3<f32>,
    base_color_uv_st: vec4<f32>,
    normal_uv_st: vec4<f32>,
    mr_uv_st: vec4<f32>,
    base_color_uv_rot: f32,
    normal_uv_rot: f32,
    mr_uv_rot: f32,
}

struct FragmentOutput {
    @location(0) member: vec4<f32>,
    @location(1) member_1: vec4<f32>,
    @location(2) member_2: vec4<f32>,
}

var<private> world_pos_1: vec3<f32>;
var<private> world_normal_1: vec3<f32>;
var<private> world_tangent_1: vec3<f32>;
var<private> world_bitangent_1: vec3<f32>;
var<private> frag_uv_1: vec2<f32>;
var<private> gbuf_rt0_: vec4<f32>;
var<private> gbuf_rt1_: vec4<f32>;
var<private> gbuf_rt2_: vec4<f32>;
@group(1) @binding(0) 
var<uniform> global: Material;
@group(1) @binding(1) 
var base_color_tex_u002e_sampler: sampler;
@group(1) @binding(2) 
var base_color_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(3) 
var normal_tex_u002e_sampler: sampler;
@group(1) @binding(4) 
var normal_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(5) 
var metallic_roughness_tex_u002e_sampler: sampler;
@group(1) @binding(6) 
var metallic_roughness_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(7) 
var emissive_tex_u002e_sampler: sampler;
@group(1) @binding(8) 
var emissive_tex_u002e_texture: texture_2d<f32>;

fn main_1() {
    let _e36 = frag_uv_1;
    let _e38 = global.normal_uv_st;
    let _e40 = global.normal_uv_rot;
    let _e41 = cos(_e40);
    let _e42 = sin(_e40);
    let _e62 = textureSample(normal_tex_u002e_texture, normal_tex_u002e_sampler, ((vec2<f32>(((_e36.x * _e41) - (_e36.y * _e42)), ((_e36.x * _e42) + (_e36.y * _e41))) * vec2<f32>(_e38.z, _e38.w)) + vec2<f32>(_e38.x, _e38.y)));
    let _e64 = world_normal_1;
    let _e66 = world_tangent_1;
    let _e68 = world_bitangent_1;
    let _e71 = ((_e62.xyz * 2f) - vec3<f32>(1f, 1f, 1f));
    let _e80 = normalize((((normalize(_e66) * _e71.x) + (normalize(_e68) * _e71.y)) + (normalize(_e64) * _e71.z)));
    let _e82 = global.base_color_uv_st;
    let _e84 = global.base_color_uv_rot;
    let _e85 = cos(_e84);
    let _e86 = sin(_e84);
    let _e106 = textureSample(base_color_tex_u002e_texture, base_color_tex_u002e_sampler, ((vec2<f32>(((_e36.x * _e85) - (_e36.y * _e86)), ((_e36.x * _e86) + (_e36.y * _e85))) * vec2<f32>(_e82.z, _e82.w)) + vec2<f32>(_e82.x, _e82.y)));
    let _e110 = global.base_color_factor;
    let _e112 = (pow(_e106.xyz, vec3<f32>(2.2f, 2.2f, 2.2f)) * _e110.xyz);
    let _e114 = global.mr_uv_st;
    let _e116 = global.mr_uv_rot;
    let _e117 = cos(_e116);
    let _e118 = sin(_e116);
    let _e138 = textureSample(metallic_roughness_tex_u002e_texture, metallic_roughness_tex_u002e_sampler, ((vec2<f32>(((_e36.x * _e117) - (_e36.y * _e118)), ((_e36.x * _e118) + (_e36.y * _e117))) * vec2<f32>(_e114.z, _e114.w)) + vec2<f32>(_e114.x, _e114.y)));
    let _e141 = global.roughness_factor;
    let _e143 = cos(_e116);
    let _e144 = sin(_e116);
    let _e164 = textureSample(metallic_roughness_tex_u002e_texture, metallic_roughness_tex_u002e_sampler, ((vec2<f32>(((_e36.x * _e143) - (_e36.y * _e144)), ((_e36.x * _e144) + (_e36.y * _e143))) * vec2<f32>(_e114.z, _e114.w)) + vec2<f32>(_e114.x, _e114.y)));
    let _e167 = global.metallic_factor;
    let _e169 = textureSample(emissive_tex_u002e_texture, emissive_tex_u002e_sampler, _e36);
    let _e173 = global.emissive_factor;
    let _e176 = global.emissive_strength;
    let _e177 = ((pow(_e169.xyz, vec3<f32>(2.2f, 2.2f, 2.2f)) * _e173) * _e176);
    let _e185 = ((abs(_e80.x) + abs(_e80.y)) + abs(_e80.z));
    let _e187 = (_e80.x / _e185);
    let _e189 = (_e80.y / _e185);
    let _e203 = step(0f, _e80.z);
    let _e204 = (1f - _e203);
    let _e213 = ((vec2<f32>(((((1f - abs(_e189)) * ((step(0f, _e187) * 2f) - 1f)) * _e204) + (_e187 * _e203)), ((((1f - abs(_e187)) * ((step(0f, _e189) * 2f) - 1f)) * _e204) + (_e189 * _e203))) * 0.5f) + vec2<f32>(0.5f, 0.5f));
    gbuf_rt0_ = vec4<f32>(_e112.x, _e112.y, _e112.z, (_e164.z * _e167));
    gbuf_rt1_ = vec4<f32>(_e213.x, _e213.y, (_e138.y * _e141), 0f);
    gbuf_rt2_ = vec4<f32>(_e177.x, _e177.y, _e177.z, 1f);
    return;
}

@fragment 
fn main(@location(0) world_pos: vec3<f32>, @location(1) world_normal: vec3<f32>, @location(2) world_tangent: vec3<f32>, @location(3) world_bitangent: vec3<f32>, @location(4) frag_uv: vec2<f32>) -> FragmentOutput {
    world_pos_1 = world_pos;
    world_normal_1 = world_normal;
    world_tangent_1 = world_tangent;
    world_bitangent_1 = world_bitangent;
    frag_uv_1 = frag_uv;
    main_1();
    let _e13 = gbuf_rt0_;
    let _e14 = gbuf_rt1_;
    let _e15 = gbuf_rt2_;
    return FragmentOutput(_e13, _e14, _e15);
}
