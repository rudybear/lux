struct Light {
    light_dir: vec3<f32>,
    view_pos: vec3<f32>,
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
var occlusion_tex_u002e_sampler: sampler;
@group(1) @binding(8) 
var occlusion_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(9) 
var emissive_tex_u002e_sampler: sampler;
@group(1) @binding(10) 
var emissive_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(11) 
var env_specular_u002e_sampler: sampler;
@group(1) @binding(12) 
var env_specular_u002e_texture: texture_cube<f32>;
@group(1) @binding(13) 
var env_irradiance_u002e_sampler: sampler;
@group(1) @binding(14) 
var env_irradiance_u002e_texture: texture_cube<f32>;
@group(1) @binding(15) 
var brdf_lut_u002e_sampler: sampler;
@group(1) @binding(16) 
var brdf_lut_u002e_texture: texture_2d<f32>;

fn main_1() {
    let _e52 = frag_uv_1;
    let _e53 = textureSample(base_color_tex_u002e_texture, base_color_tex_u002e_sampler, _e52);
    let _e55 = pow(_e53.xyz, vec3<f32>(2.2f, 2.2f, 2.2f));
    let _e56 = frag_uv_1;
    let _e57 = textureSample(metallic_roughness_tex_u002e_texture, metallic_roughness_tex_u002e_sampler, _e56);
    let _e60 = frag_uv_1;
    let _e61 = textureSample(occlusion_tex_u002e_texture, occlusion_tex_u002e_sampler, _e60);
    let _e63 = frag_uv_1;
    let _e64 = textureSample(emissive_tex_u002e_texture, emissive_tex_u002e_sampler, _e63);
    let _e67 = frag_uv_1;
    let _e68 = textureSample(normal_tex_u002e_texture, normal_tex_u002e_sampler, _e67);
    let _e70 = world_normal_1;
    let _e72 = world_tangent_1;
    let _e74 = world_bitangent_1;
    let _e77 = ((_e68.xyz * 2f) - vec3<f32>(1f, 1f, 1f));
    let _e86 = normalize((((normalize(_e72) * _e77.x) + (normalize(_e74) * _e77.y)) + (normalize(_e70) * _e77.z)));
    let _e88 = global.view_pos;
    let _e89 = world_pos_1;
    let _e91 = normalize((_e88 - _e89));
    let _e93 = global.light_dir;
    let _e94 = normalize(_e93);
    let _e96 = max(dot(_e86, _e91), 0.001f);
    let _e98 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e55, vec3(_e57.z));
    let _e100 = normalize((_e91 + _e94));
    let _e102 = max(dot(_e86, _e100), 0f);
    let _e104 = max(dot(_e86, _e91), 0.00001f);
    let _e106 = max(dot(_e86, _e94), 0.00001f);
    let _e109 = (_e57.y * _e57.y);
    let _e110 = (_e109 * _e109);
    let _e114 = (((_e102 * _e102) * (_e110 - 1f)) + 1f);
    let _e121 = (((_e57.y * _e57.y) * _e57.y) * _e57.y);
    let _e122 = (1f - _e121);
    let _e141 = (_e98 + ((vec3<f32>(1f, 1f, 1f) - _e98) * pow(clamp((1f - max(dot(_e91, _e100), 0f)), 0f, 1f), 5f)));
    let _e153 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e55, vec3(_e57.z));
    let _e157 = textureSampleLevel(env_specular_u002e_texture, env_specular_u002e_sampler, reflect((_e91 * -1f), _e86), (_e57.y * 8f));
    let _e159 = textureSample(env_irradiance_u002e_texture, env_irradiance_u002e_sampler, _e86);
    let _e162 = textureSample(brdf_lut_u002e_texture, brdf_lut_u002e_sampler, vec2<f32>(_e96, _e57.y));
    let _e163 = _e162.xy;
    let _e180 = (((_e153 + ((max(vec3((1f - _e57.y)), _e153) - _e153) * pow(clamp((1f - _e96), 0f, 1f), 5f))) * _e163.x) + vec3(_e163.y));
    let _e183 = (_e153 + ((vec3<f32>(1f, 1f, 1f) - _e153) * 0.04761905f));
    let _e184 = (1f - (_e163.x + _e163.y));
    let _e190 = (_e180 + (((_e180 * _e183) / (vec3<f32>(1f, 1f, 1f) - (_e183 * _e184))) * _e184));
    let _e200 = (((((((((vec3<f32>(1f, 1f, 1f) - _e141) * (1f - _e57.z)) * _e55) * 0.31830987f) + ((_e141 * ((_e110 * 0.31830987f) / ((_e114 * _e114) + 0.00001f))) * (0.5f / (((_e106 * sqrt((((_e104 * _e104) * _e122) + _e121))) + (_e104 * sqrt((((_e106 * _e106) * _e122) + _e121)))) + 0.00001f)))) * _e106) * vec3<f32>(1f, 0.98f, 0.95f)) + ((((((vec3<f32>(1f, 1f, 1f) - _e190) * (1f - _e57.z)) * _e55) * _e159.xyz) + (_e190 * _e157.xyz)) * _e61.x)) + pow(_e64.xyz, vec3<f32>(2.2f, 2.2f, 2.2f)));
    let _e213 = pow(clamp(((_e200 * ((_e200 * 2.51f) + vec3(0.03f))) / ((_e200 * ((_e200 * 2.43f) + vec3(0.59f))) + vec3(0.14f))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f)), vec3<f32>(0.45454547f, 0.45454547f, 0.45454547f));
    color = vec4<f32>(_e213.x, _e213.y, _e213.z, _e53.w);
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
