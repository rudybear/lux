var<private> uv_1: vec2<f32>;
var<private> color: vec4<f32>;

fn main_1() {
    let _e24 = uv_1;
    let _e26 = ((_e24 - vec2<f32>(0.5f, 0.5f)) * 4f);
    let _e29 = vec3<f32>(_e26.x, _e26.y, 0f);
    let _e31 = (length(_e29) - 0.8f);
    let _e34 = (abs((_e29 - vec3<f32>(1.2f, 0f, 0f))) - vec3<f32>(0.5f, 0.5f, 0.5f));
    let _e43 = (length(max(_e34, vec3<f32>(0f, 0f, 0f))) + min(max(_e34.x, max(_e34.y, _e34.z)), 0f));
    let _e44 = (_e29 - vec3<f32>(-1.2f, 0f, 0f));
    let _e51 = (length(vec2<f32>((length(_e44.xz) - 0.5f), _e44.y)) - 0.2f);
    let _e56 = clamp((0.5f + ((0.5f * (_e43 - _e31)) / 0.3f)), 0f, 1f);
    let _e61 = (mix(_e43, _e31, _e56) - ((0.3f * _e56) * (1f - _e56)));
    let _e66 = clamp((0.5f + ((0.5f * (_e51 - _e61)) / 0.3f)), 0f, 1f);
    let _e72 = ((mix(_e51, _e61, _e66) - ((0.3f * _e66) * (1f - _e66))) - 0.02f);
    let _e73 = smoothstep(0.01f, 0f, _e72);
    let _e81 = (((vec3<f32>(0.9f, 0.4f, 0.1f) * _e73) + (vec3<f32>(0.1f, 0.2f, 0.4f) * (1f - _e73))) + (vec3<f32>(1f, 1f, 1f) * smoothstep(0.02f, 0f, abs(_e72))));
    color = vec4<f32>(_e81.x, _e81.y, _e81.z, 1f);
    return;
}

@fragment 
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    uv_1 = uv;
    main_1();
    let _e3 = color;
    return _e3;
}
