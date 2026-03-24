struct Light {
    light_dir: vec3<f32>,
    view_pos: vec3<f32>,
    light_color: vec3<f32>,
}

struct Material {
    base_color_factor: vec4<f32>,
    emissive_factor: vec3<f32>,
    metallic_factor: f32,
    roughness_factor: f32,
    emissive_strength: f32,
}

var<private> world_pos_1: vec3<f32>;
var<private> world_normal_1: vec3<f32>;
var<private> frag_uv_1: vec2<f32>;
var<private> world_tangent_1: vec3<f32>;
var<private> world_bitangent_1: vec3<f32>;
var<private> color: vec4<f32>;
@group(1) @binding(0) 
var<uniform> global: Light;
@group(1) @binding(1) 
var<uniform> global_1: Material;
@group(1) @binding(2) 
var base_color_tex_u002e_sampler: sampler;
@group(1) @binding(3) 
var base_color_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(4) 
var normal_tex_u002e_sampler: sampler;
@group(1) @binding(5) 
var normal_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(6) 
var metallic_roughness_tex_u002e_sampler: sampler;
@group(1) @binding(7) 
var metallic_roughness_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(8) 
var occlusion_tex_u002e_sampler: sampler;
@group(1) @binding(9) 
var occlusion_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(10) 
var emissive_tex_u002e_sampler: sampler;
@group(1) @binding(11) 
var emissive_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(12) 
var env_specular_u002e_sampler: sampler;
@group(1) @binding(13) 
var env_specular_u002e_texture: texture_cube<f32>;
@group(1) @binding(14) 
var env_irradiance_u002e_sampler: sampler;
@group(1) @binding(15) 
var env_irradiance_u002e_texture: texture_cube<f32>;
@group(1) @binding(16) 
var brdf_lut_u002e_sampler: sampler;
@group(1) @binding(17) 
var brdf_lut_u002e_texture: texture_2d<f32>;

fn main_1() {
    let _e53 = frag_uv_1;
    let _e54 = textureSample(base_color_tex_u002e_texture, base_color_tex_u002e_sampler, _e53);
    let _e56 = global_1.base_color_factor;
    let _e57 = (_e54 * _e56);
    let _e59 = pow(_e57.xyz, vec3<f32>(2.2f, 2.2f, 2.2f));
    let _e60 = frag_uv_1;
    let _e61 = textureSample(metallic_roughness_tex_u002e_texture, metallic_roughness_tex_u002e_sampler, _e60);
    let _e64 = global_1.roughness_factor;
    let _e65 = (_e61.y * _e64);
    let _e68 = global_1.metallic_factor;
    let _e69 = (_e61.z * _e68);
    let _e70 = frag_uv_1;
    let _e71 = textureSample(occlusion_tex_u002e_texture, occlusion_tex_u002e_sampler, _e70);
    let _e73 = frag_uv_1;
    let _e74 = textureSample(emissive_tex_u002e_texture, emissive_tex_u002e_sampler, _e73);
    let _e78 = global_1.emissive_factor;
    let _e81 = global_1.emissive_strength;
    let _e83 = frag_uv_1;
    let _e84 = textureSample(normal_tex_u002e_texture, normal_tex_u002e_sampler, _e83);
    let _e86 = world_normal_1;
    let _e88 = world_tangent_1;
    let _e90 = world_bitangent_1;
    let _e93 = ((_e84.xyz * 2f) - vec3<f32>(1f, 1f, 1f));
    let _e102 = normalize((((normalize(_e88) * _e93.x) + (normalize(_e90) * _e93.y)) + (normalize(_e86) * _e93.z)));
    let _e104 = global.view_pos;
    let _e105 = world_pos_1;
    let _e107 = normalize((_e104 - _e105));
    let _e109 = global.light_dir;
    let _e110 = normalize(_e109);
    let _e112 = max(dot(_e102, _e107), 0.001f);
    let _e114 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e59, vec3(_e69));
    let _e116 = normalize((_e107 + _e110));
    let _e118 = max(dot(_e102, _e116), 0f);
    let _e120 = max(dot(_e102, _e107), 0.00001f);
    let _e122 = max(dot(_e102, _e110), 0.00001f);
    let _e125 = (_e65 * _e65);
    let _e126 = (_e125 * _e125);
    let _e130 = (((_e118 * _e118) * (_e126 - 1f)) + 1f);
    let _e137 = (((_e65 * _e65) * _e65) * _e65);
    let _e138 = (1f - _e137);
    let _e157 = (_e114 + ((vec3<f32>(1f, 1f, 1f) - _e114) * pow(clamp((1f - max(dot(_e107, _e116), 0f)), 0f, 1f), 5f)));
    let _e168 = global.light_color;
    let _e171 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e59, vec3(_e69));
    let _e175 = textureSampleLevel(env_specular_u002e_texture, env_specular_u002e_sampler, reflect((_e107 * -1f), _e102), (_e65 * 8f));
    let _e177 = textureSample(env_irradiance_u002e_texture, env_irradiance_u002e_sampler, _e102);
    let _e180 = textureSample(brdf_lut_u002e_texture, brdf_lut_u002e_sampler, vec2<f32>(_e112, _e65));
    let _e181 = _e180.xy;
    let _e198 = (((_e171 + ((max(vec3((1f - _e65)), _e171) - _e171) * pow(clamp((1f - _e112), 0f, 1f), 5f))) * _e181.x) + vec3(_e181.y));
    let _e201 = (_e171 + ((vec3<f32>(1f, 1f, 1f) - _e171) * 0.04761905f));
    let _e202 = (1f - (_e181.x + _e181.y));
    let _e208 = (_e198 + (((_e198 * _e201) / (vec3<f32>(1f, 1f, 1f) - (_e201 * _e202))) * _e202));
    let _e218 = (((((((((vec3<f32>(1f, 1f, 1f) - _e157) * (1f - _e69)) * _e59) * 0.31830987f) + ((_e157 * ((_e126 * 0.31830987f) / ((_e130 * _e130) + 0.00001f))) * (0.5f / (((_e122 * sqrt((((_e120 * _e120) * _e138) + _e137))) + (_e120 * sqrt((((_e122 * _e122) * _e138) + _e137)))) + 0.00001f)))) * _e122) * _e168) + ((((((vec3<f32>(1f, 1f, 1f) - _e208) * (1f - _e69)) * _e59) * _e177.xyz) + (_e208 * _e175.xyz)) * _e71.x)) + ((pow(_e74.xyz, vec3<f32>(2.2f, 2.2f, 2.2f)) * _e78) * _e81));
    let _e231 = pow(clamp(((_e218 * ((_e218 * 2.51f) + vec3(0.03f))) / ((_e218 * ((_e218 * 2.43f) + vec3(0.59f))) + vec3(0.14f))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f)), vec3<f32>(0.45454547f, 0.45454547f, 0.45454547f));
    color = vec4<f32>(_e231.x, _e231.y, _e231.z, _e57.w);
    return;
}

@fragment 
fn main(@location(0) world_pos: vec3<f32>, @location(1) world_normal: vec3<f32>, @location(2) frag_uv: vec2<f32>, @location(3) world_tangent: vec3<f32>, @location(4) world_bitangent: vec3<f32>) -> @location(0) vec4<f32> {
    world_pos_1 = world_pos;
    world_normal_1 = world_normal;
    frag_uv_1 = frag_uv;
    world_tangent_1 = world_tangent;
    world_bitangent_1 = world_bitangent;
    main_1();
    let _e11 = color;
    return _e11;
}
