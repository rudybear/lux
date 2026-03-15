var<private> uv_1: vec2<f32>;
var<private> color: vec4<f32>;

fn main_1() {
    let _e67 = uv_1;
    let _e69 = (_e67.x * 2f);
    let _e70 = uv_1;
    let _e72 = (_e70.y * 3f);
    let _e74 = fract(vec2<f32>(_e69, _e72));
    let _e75 = floor(_e69);
    let _e76 = floor(_e72);
    let _e117 = (0.1f * 0.1f);
    let _e118 = (_e117 * _e117);
    let _e122 = (((_e74.x * _e74.x) * (_e118 - 1f)) + 1f);
    let _e129 = (0.3f * 0.3f);
    let _e130 = (_e129 * _e129);
    let _e134 = (((_e74.x * _e74.x) * (_e130 - 1f)) + 1f);
    let _e141 = (0.7f * 0.7f);
    let _e142 = (_e141 * _e141);
    let _e146 = (((_e74.x * _e74.x) * (_e142 - 1f)) + 1f);
    let _e173 = (((0.1f * 0.1f) * 0.1f) * 0.1f);
    let _e174 = (1f - _e173);
    let _e192 = (((0.3f * 0.3f) * 0.3f) * 0.3f);
    let _e193 = (1f - _e192);
    let _e211 = (((0.7f * 0.7f) * 0.7f) * 0.7f);
    let _e212 = (1f - _e211);
    let _e247 = (1f / (0.3f * 0.3f));
    let _e259 = (1f / (0.5f * 0.5f));
    let _e271 = (1f / (0.8f * 0.8f));
    let _e306 = ((0.5f + (((2f * 0.5f) * 0.7f) * 0.7f)) - 1f);
    let _e335 = clamp(_e74.x, 0f, 1f);
    let _e336 = (1f - _e335);
    let _e347 = clamp(_e74.x, 0f, 1f);
    let _e348 = (1f - _e347);
    let _e371 = (_e75 < 0.5f);
    let _e372 = (_e76 < 0.5f);
    let _e375 = (_e75 > 0.5f);
    let _e378 = (_e76 > 0.5f);
    let _e379 = (_e76 < 1.5f);
    let _e386 = (_e76 > 1.5f);
    let _e421 = ((((((((((((vec3<f32>(0.06f, 0.06f, 0.08f) + (vec3<f32>(0f, 0.8f, 0.9f) * smoothstep(0.015f, 0.005f, abs((_e74.y - (vec3<f32>(0.04f, 0.04f, 0.04f) + ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(0.04f, 0.04f, 0.04f)) * pow(clamp((1f - _e74.x), 0f, 1f), 5f))).x))))) + (vec3<f32>(0.2f, 0.8f, 0.2f) * smoothstep(0.015f, 0.005f, abs((_e74.y - (vec3<f32>(0.5f, 0.5f, 0.5f) + ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(0.5f, 0.5f, 0.5f)) * pow(clamp((1f - _e74.x), 0f, 1f), 5f))).x))))) + (vec3<f32>(0.9f, 0.8f, 0.2f) * smoothstep(0.015f, 0.005f, abs((_e74.y - (vec3<f32>(0.96f, 0.96f, 0.96f) + ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(0.96f, 0.96f, 0.96f)) * pow(clamp((1f - _e74.x), 0f, 1f), 5f))).x))))) * select(0f, 1f, (_e371 && _e372))) + ((((vec3<f32>(0.06f, 0.06f, 0.08f) + (vec3<f32>(1f, 0.2f, 0.1f) * smoothstep(0.015f, 0.005f, abs((_e74.y - clamp((((_e118 * 0.31830987f) / ((_e122 * _e122) + 0.00001f)) * 0.05f), 0f, 1f)))))) + (vec3<f32>(0.9f, 0.4f, 0.2f) * smoothstep(0.015f, 0.005f, abs((_e74.y - clamp((((_e130 * 0.31830987f) / ((_e134 * _e134) + 0.00001f)) * 0.15f), 0f, 1f)))))) + (vec3<f32>(0.8f, 0.5f, 0.3f) * smoothstep(0.015f, 0.005f, abs((_e74.y - clamp((((_e142 * 0.31830987f) / ((_e146 * _e146) + 0.00001f)) * 0.5f), 0f, 1f)))))) * select(0f, 1f, (_e375 && _e372)))) + ((((vec3<f32>(0.06f, 0.06f, 0.08f) + (vec3<f32>(0.2f, 0.4f, 1f) * smoothstep(0.015f, 0.005f, abs((_e74.y - clamp(((0.5f / (((_e74.x * sqrt((((0.5f * 0.5f) * _e174) + _e173))) + (0.5f * sqrt((((_e74.x * _e74.x) * _e174) + _e173)))) + 0.00001f)) * 2f), 0f, 1f)))))) + (vec3<f32>(0.3f, 0.5f, 0.9f) * smoothstep(0.015f, 0.005f, abs((_e74.y - clamp(((0.5f / (((_e74.x * sqrt((((0.5f * 0.5f) * _e193) + _e192))) + (0.5f * sqrt((((_e74.x * _e74.x) * _e193) + _e192)))) + 0.00001f)) * 2f), 0f, 1f)))))) + (vec3<f32>(0.4f, 0.6f, 0.8f) * smoothstep(0.015f, 0.005f, abs((_e74.y - clamp(((0.5f / (((_e74.x * sqrt((((0.5f * 0.5f) * _e212) + _e211))) + (0.5f * sqrt((((_e74.x * _e74.x) * _e212) + _e211)))) + 0.00001f)) * 2f), 0f, 1f)))))) * select(0f, 1f, ((_e371 && _e378) && _e379)))) + ((((vec3<f32>(0.06f, 0.06f, 0.08f) + (vec3<f32>(0.9f, 0.2f, 0.8f) * smoothstep(0.015f, 0.005f, abs((_e74.y - clamp(((((2f + _e247) * pow(max((1f - (_e74.x * _e74.x)), 0f), (_e247 * 0.5f))) / 6.2831855f) * 0.3f), 0f, 1f)))))) + (vec3<f32>(0.8f, 0.3f, 0.7f) * smoothstep(0.015f, 0.005f, abs((_e74.y - clamp(((((2f + _e259) * pow(max((1f - (_e74.x * _e74.x)), 0f), (_e259 * 0.5f))) / 6.2831855f) * 0.3f), 0f, 1f)))))) + (vec3<f32>(0.7f, 0.4f, 0.6f) * smoothstep(0.015f, 0.005f, abs((_e74.y - clamp(((((2f + _e271) * pow(max((1f - (_e74.x * _e74.x)), 0f), (_e271 * 0.5f))) / 6.2831855f) * 0.3f), 0f, 1f)))))) * select(0f, 1f, ((_e375 && _e378) && _e379)))) + (((vec3<f32>(0.06f, 0.06f, 0.08f) + (vec3<f32>(0.2f, 0.9f, 0.3f) * smoothstep(0.015f, 0.005f, abs((_e74.y - clamp((_e74.x * 0.31830987f), 0f, 1f)))))) + (vec3<f32>(0.9f, 0.9f, 0.2f) * smoothstep(0.015f, 0.005f, abs((_e74.y - clamp((((((vec3<f32>(1f, 1f, 1f) * 0.31830987f) * (1f + (_e306 * pow((1f - _e74.x), 5f)))) * (1f + (_e306 * pow((1f - 0.5f), 5f)))) * max(_e74.x, 0f)).x * 3.1415927f), 0f, 1f)))))) * select(0f, 1f, (_e371 && _e386)))) + (((vec3<f32>(0.06f, 0.06f, 0.08f) + (vec3<f32>(1f, 0.85f, 0.3f) * smoothstep(0.015f, 0.005f, abs((_e74.y - ((vec3<f32>(1f, 0.71f, 0.29f) + ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(1f, 0.71f, 0.29f)) * pow(_e336, 5f))) - ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(1f, 0.87f, 0.61f)) * (_e335 * pow(_e336, 6f)))).x))))) + (vec3<f32>(1f, 0.6f, 0.3f) * smoothstep(0.015f, 0.005f, abs((_e74.y - ((vec3<f32>(0.95f, 0.64f, 0.54f) + ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(0.95f, 0.64f, 0.54f)) * pow(_e348, 5f))) - ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(1f, 0.78f, 0.7f)) * (_e347 * pow(_e348, 6f)))).x))))) * select(0f, 1f, (_e375 && _e386)))) + (vec3<f32>(0.15f, 0.15f, 0.1f) * smoothstep(0.008f, 0.003f, abs((_e74.y - 0.5f))))) * (smoothstep(0f, 0.015f, _e74.x) * smoothstep(0f, 0.015f, (1f - _e74.x)))) * (smoothstep(0f, 0.015f, _e74.y) * smoothstep(0f, 0.015f, (1f - _e74.y))));
    color = vec4<f32>(_e421.x, _e421.y, _e421.z, 1f);
    return;
}

@fragment 
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    uv_1 = uv;
    main_1();
    let _e3 = color;
    return _e3;
}
