struct Material {
    roughness: f32,
    metallic: f32,
}

var<private> world_pos_1: vec3<f32>;
var<private> world_normal_1: vec3<f32>;
var<private> frag_uv_1: vec2<f32>;
var<private> color: vec4<f32>;
@group(2) @binding(0) 
var<uniform> global: Material;
@group(1) @binding(0) 
var albedo_tex_u002e_sampler: sampler;
@group(1) @binding(1) 
var albedo_tex_u002e_texture: texture_2d<f32>;

fn main_1() {
    let _e25 = frag_uv_1;
    let _e26 = textureSample(albedo_tex_u002e_texture, albedo_tex_u002e_sampler, _e25);
    let _e27 = _e26.xyz;
    let _e28 = world_normal_1;
    let _e29 = normalize(_e28);
    let _e30 = normalize(vec3<f32>(0f, 0f, 1f));
    let _e32 = max(dot(_e29, _e30), 0.001f);
    let _e34 = global.metallic;
    let _e36 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e27, vec3(_e34));
    let _e38 = global.metallic;
    let _e39 = (1f - _e38);
    let _e41 = normalize(vec3<f32>(1f, 1f, 1f));
    let _e43 = max(dot(_e29, _e41), 0f);
    let _e45 = normalize((_e41 + _e30));
    let _e47 = max(dot(_e29, _e45), 0f);
    let _e49 = global.roughness;
    let _e51 = global.roughness;
    let _e52 = (_e49 * _e51);
    let _e53 = (_e52 * _e52);
    let _e54 = (_e53 * _e53);
    let _e58 = (((_e47 * _e47) * (_e54 - 1f)) + 1f);
    let _e63 = (_e52 + 1f);
    let _e65 = ((_e63 * _e63) / 8f);
    let _e71 = (_e52 + 1f);
    let _e73 = ((_e71 * _e71) / 8f);
    let _e87 = (_e36 + ((vec3<f32>(1f, 1f, 1f) - _e36) * pow(clamp((1f - max(dot(_e45, _e30), 0f)), 0f, 1f), 5f)));
    let _e102 = (((((((vec3<f32>(1f, 1f, 1f) - _e87) * _e39) * (_e27 * _e39)) / vec3(3.14159f)) + (((_e87 * ((_e54 * 0.31830987f) / ((_e58 * _e58) + 0.00001f))) * ((_e32 / (((_e32 * (1f - _e65)) + _e65) + 0.00001f)) * (_e43 / (((_e43 * (1f - _e73)) + _e73) + 0.00001f)))) / vec3((((4f * _e32) * _e43) + 0.0001f)))) * vec3<f32>(3f, 3f, 3f)) * _e43);
    color = vec4<f32>(_e102.x, _e102.y, _e102.z, 1f);
    return;
}

@fragment 
fn main(@location(0) world_pos: vec3<f32>, @location(1) world_normal: vec3<f32>, @location(2) frag_uv: vec2<f32>) -> @location(0) vec4<f32> {
    world_pos_1 = world_pos;
    world_normal_1 = world_normal;
    frag_uv_1 = frag_uv;
    main_1();
    let _e7 = color;
    return _e7;
}
