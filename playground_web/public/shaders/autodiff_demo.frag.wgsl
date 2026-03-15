var<private> uv_1: vec2<f32>;
var<private> color: vec4<f32>;

fn main_1() {
    let _e16 = uv_1;
    let _e19 = ((_e16.x * 4f) - 2f);
    let _e34 = ((((sin((_e19 * 6.28318f)) * 0.5f) + ((_e19 * _e19) * 0.3f)) * 0.3f) + 0.5f);
    let _e36 = (((((cos((_e19 * 6.28318f)) * 6.28318f) * 0.5f) + ((_e19 + _e19) * 0.3f)) * 0.1f) + 0.5f);
    let _e37 = uv_1;
    let _e40 = select(0f, 1f, (_e37.y > 0.5f));
    let _e49 = uv_1;
    let _e52 = uv_1;
    let _e61 = (((vec3<f32>(clamp(_e34, 0f, 1f), clamp((_e34 * 0.6f), 0f, 1f), 0.2f) * _e40) + (vec3<f32>(0.2f, clamp((_e36 * 0.7f), 0f, 1f), clamp(_e36, 0f, 1f)) * (1f - _e40))) + (vec3<f32>(1f, 1f, 1f) * (smoothstep(0.49f, 0.5f, _e49.y) - smoothstep(0.5f, 0.51f, _e52.y))));
    color = vec4<f32>(_e61.x, _e61.y, _e61.z, 1f);
    return;
}

@fragment 
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    uv_1 = uv;
    main_1();
    let _e3 = color;
    return _e3;
}
