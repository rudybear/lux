struct Material {
    roughness: f32,
    metallic: f32,
    exposure: f32,
}

var<private> world_normal_1: vec3<f32>;
var<private> world_position_1: vec3<f32>;
var<private> uv_1: vec2<f32>;
var<private> color: vec4<f32>;
@group(1) @binding(0) 
var<uniform> global: Material;
@group(0) @binding(0) 
var albedo_tex_u002e_sampler: sampler;
@group(0) @binding(1) 
var albedo_tex_u002e_texture: texture_2d<f32>;

fn main_1() {
    let _e39 = uv_1;
    let _e40 = textureSample(albedo_tex_u002e_texture, albedo_tex_u002e_sampler, _e39);
    let _e41 = _e40.xyz;
    let _e42 = world_normal_1;
    let _e43 = normalize(_e42);
    let _e44 = world_position_1;
    let _e46 = normalize((vec3<f32>(0f, 0f, 5f) - _e44));
    let _e48 = global.metallic;
    let _e50 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e41, vec3(_e48));
    let _e52 = global.metallic;
    let _e53 = (1f - _e52);
    let _e55 = normalize(vec3<f32>(1f, 1f, 0.5f));
    let _e57 = max(dot(_e43, _e55), 0f);
    let _e59 = max(dot(_e43, _e46), 0.001f);
    let _e61 = normalize((_e55 + _e46));
    let _e63 = max(dot(_e43, _e61), 0f);
    let _e67 = global.roughness;
    let _e68 = (_e67 * _e67);
    let _e69 = (_e68 * _e68);
    let _e73 = (((_e63 * _e63) * (_e69 - 1f)) + 1f);
    let _e78 = global.roughness;
    let _e79 = (_e78 + 1f);
    let _e81 = ((_e79 * _e79) / 8f);
    let _e82 = (1f - _e81);
    let _e95 = (_e50 + ((vec3<f32>(1f, 1f, 1f) - _e50) * pow(clamp((1f - max(dot(_e46, _e61), 0f)), 0f, 1f), 5f)));
    let _e118 = global.exposure;
    let _e119 = ((((((((((vec3<f32>(1f, 1f, 1f) - _e95) * _e53) * (_e41 * _e53)) / vec3(3.1415927f)) + (((_e95 * (_e69 / ((3.1415927f * _e73) * _e73))) * ((_e59 / ((_e59 * _e82) + _e81)) * (_e57 / ((_e57 * _e82) + _e81)))) / vec3((((4f * _e59) * _e57) + 0.0001f)))) * vec3<f32>(3f, 2.8f, 2.5f)) * _e57) + (vec3<f32>(0.1f, 0.15f, 0.2f) * pow((1f - _e59), 4f))) + (vec3<f32>(0.03f, 0.03f, 0.03f) * _e41)) * _e118);
    let _e131 = clamp(((_e119 * ((_e119 * 2.51f) + vec3(0.03f))) / ((_e119 * ((_e119 * 2.43f) + vec3(0.59f))) + vec3(0.14f))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f));
    color = vec4<f32>(_e131.x, _e131.y, _e131.z, 1f);
    return;
}

@fragment 
fn main(@location(0) world_normal: vec3<f32>, @location(1) world_position: vec3<f32>, @location(2) uv: vec2<f32>) -> @location(0) vec4<f32> {
    world_normal_1 = world_normal;
    world_position_1 = world_position;
    uv_1 = uv;
    main_1();
    let _e7 = color;
    return _e7;
}
