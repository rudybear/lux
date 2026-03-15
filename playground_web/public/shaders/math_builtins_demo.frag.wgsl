var<private> uv_1: vec2<f32>;
var<private> color: vec4<f32>;

fn main_1() {
    let _e31 = uv_1;
    let _e33 = (_e31.x * 4f);
    let _e34 = uv_1;
    let _e36 = (_e34.y * 2f);
    let _e38 = fract(vec2<f32>(_e33, _e36));
    let _e42 = ((floor(_e36) * 4f) + floor(_e33));
    let _e45 = ((_e38.x * 6f) - 3f);
    let _e52 = (_e45 * 0.5f);
    let _e71 = (_e38.y - 0.5f);
    let _e122 = ((((((((((round(_e45) * 0.15f) + 0.5f) * select(0f, 1f, (_e42 < 0.5f))) + (((trunc(_e45) * 0.15f) + 0.5f) * select(0f, 1f, ((_e42 > 0.5f) && (_e42 < 1.5f))))) + (((sinh(_e52) * 0.1f) + 0.5f) * select(0f, 1f, ((_e42 > 1.5f) && (_e42 < 2.5f))))) + ((((cosh(_e52) - 1f) * 0.15f) + 0.2f) * select(0f, 1f, ((_e42 > 2.5f) && (_e42 < 3.5f))))) + (((tanh(_e45) * 0.4f) + 0.5f) * select(0f, 1f, ((_e42 > 3.5f) && (_e42 < 4.5f))))) + (((radians((_e45 * 60f)) * 0.1f) + 0.5f) * select(0f, 1f, ((_e42 > 4.5f) && (_e42 < 5.5f))))) + (((degrees(_e52) * 0.005f) + 0.5f) * select(0f, 1f, ((_e42 > 5.5f) && (_e42 < 6.5f))))) + (((faceForward(vec3<f32>(0f, 1f, 0f), normalize(vec3<f32>((_e38.x - 0.5f), _e71, 0.5f)), vec3<f32>(0f, 0f, 1f)).y * 0.5f) + 0.5f) * select(0f, 1f, (_e42 > 6.5f))));
    let _e127 = smoothstep(0.02f, 0.005f, abs((_e38.y - clamp(_e122, 0f, 1f))));
    let _e129 = select(0.6f, 0.1f, (_e122 > 0.5f));
    let _e142 = (smoothstep(0.01f, 0.003f, abs(_e71)) * 0.2f);
    let _e160 = ((vec3<f32>(((0.1f + (_e127 * select(0.2f, 0.9f, (_e129 < 0.3f)))) + _e142), ((0.1f + (_e127 * 0.4f)) + (_e142 * 0.5f)), (0.15f + (_e127 * select(0.2f, 0.8f, (_e129 > 0.3f))))) * (smoothstep(0f, 0.015f, _e38.x) * smoothstep(0f, 0.015f, (1f - _e38.x)))) * (smoothstep(0f, 0.015f, _e38.y) * smoothstep(0f, 0.015f, (1f - _e38.y))));
    color = vec4<f32>(_e160.x, _e160.y, _e160.z, 1f);
    return;
}

@fragment 
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    uv_1 = uv;
    main_1();
    let _e3 = color;
    return _e3;
}
