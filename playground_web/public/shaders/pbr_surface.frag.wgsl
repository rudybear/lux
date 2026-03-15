struct Light {
    light_dir: vec3<f32>,
    view_pos: vec3<f32>,
}

var<private> world_pos_1: vec3<f32>;
var<private> world_normal_1: vec3<f32>;
var<private> frag_uv_1: vec2<f32>;
var<private> color: vec4<f32>;
@group(1) @binding(0) 
var<uniform> global: Light;
@group(1) @binding(1) 
var albedo_tex_u002e_sampler: sampler;
@group(1) @binding(2) 
var albedo_tex_u002e_texture: texture_2d<f32>;

fn main_1() {
    let _e22 = world_normal_1;
    let _e23 = normalize(_e22);
    let _e25 = global.view_pos;
    let _e26 = world_pos_1;
    let _e28 = normalize((_e25 - _e26));
    let _e30 = global.light_dir;
    let _e31 = normalize(_e30);
    let _e32 = frag_uv_1;
    let _e33 = textureSample(albedo_tex_u002e_texture, albedo_tex_u002e_sampler, _e32);
    let _e34 = _e33.xyz;
    let _e36 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e34, vec3(0f));
    let _e55 = normalize((_e28 + _e31));
    let _e57 = max(dot(_e23, _e55), 0f);
    let _e59 = max(dot(_e23, _e28), 0f);
    let _e61 = max(dot(_e23, _e31), 0f);
    let _e64 = (0.5f * 0.5f);
    let _e65 = (_e64 * _e64);
    let _e69 = (((_e57 * _e57) * (_e65 - 1f)) + 1f);
    let _e74 = (0.5f + 1f);
    let _e76 = ((_e74 * _e74) / 8f);
    let _e82 = (0.5f + 1f);
    let _e84 = ((_e82 * _e82) / 8f);
    let _e105 = ((((((vec3<f32>(1f, 1f, 1f) - (_e36 + ((vec3<f32>(1f, 1f, 1f) - _e36) * pow(clamp((1f - max(dot(normalize((_e28 + _e31)), _e28), 0f)), 0f, 1f), 5f)))) * (1f - 0f)) * _e34) * 0.31830987f) + ((((_e36 + ((vec3<f32>(1f, 1f, 1f) - _e36) * pow(clamp((1f - max(dot(_e55, _e28), 0f)), 0f, 1f), 5f))) * ((_e65 * 0.31830987f) / ((_e69 * _e69) + 0.00001f))) * ((_e59 / (((_e59 * (1f - _e76)) + _e76) + 0.00001f)) * (_e61 / (((_e61 * (1f - _e84)) + _e84) + 0.00001f)))) / vec3((((4f * _e59) * _e61) + 0.00001f)))) * max(dot(_e23, _e31), 0f));
    let _e108 = ((_e105 + (_e105 * 0.3f)) * 2.5f);
    color = vec4<f32>(_e108.x, _e108.y, _e108.z, 1f);
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
