var<private> uv_1: vec2<f32>;
var<private> color: vec4<f32>;

fn main_1() {
    let _e31 = uv_1;
    let _e34 = uv_1;
    let _e36 = (_e34.y * 3f);
    let _e37 = floor(_e36);
    let _e38 = fract(_e36);
    let _e42 = vec2<f32>(((_e31 * 2f) - vec2<f32>(1f, 1f)).x, ((_e38 * 2f) - 1f));
    let _e43 = length(_e42);
    let _e52 = normalize(vec3<f32>(_e42.x, _e42.y, sqrt(max((0.49f - (_e43 * _e43)), 0f))));
    let _e55 = vec3<f32>(_e42.x, _e42.y, 0f);
    let _e61 = (vec3<f32>(0.3f, 0.3f, 1f) - _e55);
    let _e62 = length(_e61);
    let _e68 = (_e62 * _e62);
    let _e71 = (_e68 / max((5f * 5f), 0.0001f));
    let _e74 = clamp((1f - (_e71 * _e71)), 0f, 1f);
    let _e81 = (vec3<f32>(0f, 0f, 2f) - _e55);
    let _e82 = length(_e81);
    let _e85 = (_e81 / vec3(max(_e82, 0.0001f)));
    let _e90 = (_e82 * _e82);
    let _e93 = (_e90 / max((5f * 5f), 0.0001f));
    let _e96 = clamp((1f - (_e93 * _e93)), 0f, 1f);
    let _e104 = clamp(((dot(normalize(vec3<f32>(0f, 0f, -1f)), _e85) - 0.7f) / max((0.9f - 0.7f), 0.0001f)), 0f, 1f);
    let _e131 = (((vec3<f32>(0.05f, 0.05f, 0.08f) + (((((vec3<f32>(1f, 0.95f, 0.9f) * 1.5f) * max(dot(_e52, normalize(vec3<f32>(0.5f, 0.7f, 1f))), 0f)) * select(0f, 1f, (_e37 < 0.5f))) + ((((vec3<f32>(0.9f, 0.7f, 0.4f) * 3f) * max(dot(_e52, (_e61 / vec3(max(_e62, 0.0001f)))), 0f)) * ((_e74 * _e74) / max(_e68, 0.0001f))) * select(0f, 1f, ((_e37 > 0.5f) && (_e37 < 1.5f))))) + (((((vec3<f32>(0.4f, 0.8f, 1f) * 4f) * max(dot(_e52, _e85), 0f)) * ((_e96 * _e96) / max(_e90, 0.0001f))) * (_e104 * _e104)) * select(0f, 1f, (_e37 > 1.5f))))) * step(_e43, 0.7f)) * (smoothstep(0f, 0.01f, abs(_e38)) * smoothstep(0f, 0.01f, abs((_e38 - 1f)))));
    color = vec4<f32>(_e131.x, _e131.y, _e131.z, 1f);
    return;
}

@fragment 
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    uv_1 = uv;
    main_1();
    let _e3 = color;
    return _e3;
}
