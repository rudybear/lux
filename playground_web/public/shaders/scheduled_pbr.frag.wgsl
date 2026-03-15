struct Light {
    light_dir: vec3<f32>,
    view_pos: vec3<f32>,
}

var<private> world_pos_1: vec3<f32>;
var<private> world_normal_1: vec3<f32>;
var<private> color: vec4<f32>;
@group(1) @binding(0) 
var<uniform> global: Light;

fn main_1() {
    let _e29 = world_normal_1;
    let _e30 = normalize(_e29);
    let _e32 = global.view_pos;
    let _e33 = world_pos_1;
    let _e35 = normalize((_e32 - _e33));
    let _e37 = global.light_dir;
    let _e38 = normalize(_e37);
    let _e40 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), vec3<f32>(0.95f, 0.64f, 0.54f), vec3(0.9f));
    let _e59 = normalize((_e35 + _e38));
    let _e61 = max(dot(_e30, _e59), 0f);
    let _e63 = max(dot(_e30, _e35), 0f);
    let _e65 = max(dot(_e30, _e38), 0f);
    let _e68 = (0.3f * 0.3f);
    let _e69 = (_e68 * _e68);
    let _e73 = (((_e61 * _e61) * (_e69 - 1f)) + 1f);
    let _e78 = (0.3f + 1f);
    let _e80 = ((_e78 * _e78) / 8f);
    let _e86 = (0.3f + 1f);
    let _e88 = ((_e86 * _e86) / 8f);
    let _e109 = ((((((vec3<f32>(1f, 1f, 1f) - (_e40 + ((vec3<f32>(1f, 1f, 1f) - _e40) * pow(clamp((1f - max(dot(normalize((_e35 + _e38)), _e35), 0f)), 0f, 1f), 5f)))) * (1f - 0.9f)) * vec3<f32>(0.95f, 0.64f, 0.54f)) * 0.31830987f) + ((((_e40 + ((vec3<f32>(1f, 1f, 1f) - _e40) * pow(clamp((1f - max(dot(_e59, _e35), 0f)), 0f, 1f), 5f))) * ((_e69 * 0.31830987f) / ((_e73 * _e73) + 0.00001f))) * ((_e63 / (((_e63 * (1f - _e80)) + _e80) + 0.00001f)) * (_e65 / (((_e65 * (1f - _e88)) + _e88) + 0.00001f)))) / vec3((((4f * _e63) * _e65) + 0.00001f)))) * max(dot(_e30, _e38), 0f));
    let _e112 = ((_e109 + (_e109 * 0.3f)) * 2.5f);
    let _e124 = clamp(((_e112 * ((_e112 * 2.51f) + vec3(0.03f))) / ((_e112 * ((_e112 * 2.43f) + vec3(0.59f))) + vec3(0.14f))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f));
    color = vec4<f32>(_e124.x, _e124.y, _e124.z, 1f);
    return;
}

@fragment 
fn main(@location(0) world_pos: vec3<f32>, @location(1) world_normal: vec3<f32>) -> @location(0) vec4<f32> {
    world_pos_1 = world_pos;
    world_normal_1 = world_normal;
    main_1();
    let _e5 = color;
    return _e5;
}
