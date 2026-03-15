var<private> uv_1: vec2<f32>;
var<private> color: vec4<f32>;

fn main_1() {
    let _e46 = uv_1;
    let _e49 = ((_e46.x * 0.99f) + 0.01f);
    let _e55 = normalize(vec3<f32>(sqrt(max((1f - (_e49 * _e49)), 0f)), 0f, _e49));
    let _e56 = normalize(vec3<f32>(0.707f, 0f, 0.707f));
    let _e58 = normalize((_e55 + _e56));
    let _e60 = max(dot(vec3<f32>(0f, 0f, 1f), _e56), 0f);
    let _e62 = max(dot(vec3<f32>(0f, 0f, 1f), _e55), 0f);
    let _e64 = max(dot(vec3<f32>(0f, 0f, 1f), _e58), 0f);
    let _e68 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), vec3<f32>(0.8f, 0.3f, 0.2f), vec3(0f));
    let _e82 = (dot((((((vec3<f32>(1f, 1f, 1f) - (_e68 + ((vec3<f32>(1f, 1f, 1f) - _e68) * pow(clamp((1f - max(dot(_e55, _e58), 0f)), 0f, 1f), 5f)))) * (1f - 0f)) * vec3<f32>(0.8f, 0.3f, 0.2f)) * 0.31830987f) * _e60), vec3<f32>(0.2126f, 0.7152f, 0.0722f)) * 2.5f);
    let _e84 = normalize((_e55 + _e56));
    let _e86 = max(dot(vec3<f32>(0f, 0f, 1f), _e84), 0f);
    let _e88 = max(dot(vec3<f32>(0f, 0f, 1f), _e55), 0f);
    let _e90 = max(dot(vec3<f32>(0f, 0f, 1f), _e56), 0f);
    let _e93 = (0.3f * 0.3f);
    let _e94 = (_e93 * _e93);
    let _e98 = (((_e86 * _e86) * (_e94 - 1f)) + 1f);
    let _e103 = (0.3f + 1f);
    let _e105 = ((_e103 * _e103) / 8f);
    let _e111 = (0.3f + 1f);
    let _e113 = ((_e111 * _e111) / 8f);
    let _e137 = normalize((_e55 + _e56));
    let _e139 = max(dot(vec3<f32>(0f, 0f, 1f), _e137), 0f);
    let _e141 = max(dot(vec3<f32>(0f, 0f, 1f), _e56), 0f);
    let _e143 = max(dot(vec3<f32>(0f, 0f, 1f), _e55), 0f);
    let _e146 = (0.1f * 0.1f);
    let _e147 = (_e146 * _e146);
    let _e151 = (((_e139 * _e139) * (_e147 - 1f)) + 1f);
    let _e158 = (((0.1f * 0.1f) * 0.1f) * 0.1f);
    let _e159 = (1f - _e158);
    let _e184 = (1f / (0.5f * 0.5f));
    let _e205 = clamp(_e82, 0f, 1f);
    let _e206 = (_e82 + (dot((((((_e68 + ((vec3<f32>(1f, 1f, 1f) - _e68) * pow(clamp((1f - max(dot(_e84, _e55), 0f)), 0f, 1f), 5f))) * ((_e94 * 0.31830987f) / ((_e98 * _e98) + 0.00001f))) * ((_e88 / (((_e88 * (1f - _e105)) + _e105) + 0.00001f)) * (_e90 / (((_e90 * (1f - _e113)) + _e113) + 0.00001f)))) / vec3((((4f * _e88) * _e90) + 0.00001f))) * _e60), vec3<f32>(0.2126f, 0.7152f, 0.0722f)) * 2.5f));
    let _e207 = clamp(_e206, 0f, 1f);
    let _e208 = (_e206 + (((((0.5f * ((_e147 * 0.31830987f) / ((_e151 * _e151) + 0.00001f))) * (0.5f / (((_e141 * sqrt((((_e143 * _e143) * _e159) + _e158))) + (_e143 * sqrt((((_e141 * _e141) * _e159) + _e158)))) + 0.00001f))) * (0.04f + ((1f - 0.04f) * pow((1f - max(dot(_e55, _e137), 0f)), 5f)))) * _e141) * 2.5f));
    let _e209 = clamp(_e208, 0f, 1f);
    let _e211 = clamp((_e208 + (dot((((vec3<f32>(0.5f, 0.3f, 0.8f) * (((2f + _e184) * pow(max((1f - (_e64 * _e64)), 0f), (_e184 * 0.5f))) / 6.2831855f)) * (1f / ((4f * ((_e60 + _e62) - (_e60 * _e62))) + 0.00001f))) * max(_e60, 0f)), vec3<f32>(0.2126f, 0.7152f, 0.0722f)) * 2.5f)), 0f, 1f);
    let _e212 = uv_1;
    let _e248 = ((select(select(select(select(vec3<f32>(0.04f, 0.04f, 0.06f), (vec3<f32>(0.8f, 0.3f, 0.9f) * 0.6f), (_e212.y < _e211)), (vec3<f32>(0.85f, 0.85f, 0.9f) * 0.6f), (_e212.y < _e209)), (vec3<f32>(0.1f, 0.8f, 0.9f) * 0.6f), (_e212.y < _e207)), (vec3<f32>(0.9f, 0.5f, 0.15f) * 0.6f), (_e212.y < _e205)) + (vec3<f32>(0.9f, 0.9f, 0.9f) * max(max(smoothstep(0.012f, 0.004f, abs((_e212.y - _e205))), smoothstep(0.012f, 0.004f, abs((_e212.y - _e207)))), max(smoothstep(0.012f, 0.004f, abs((_e212.y - _e209))), smoothstep(0.012f, 0.004f, abs((_e212.y - _e211))))))) + vec3((smoothstep(0.008f, 0.002f, abs((_e212.y - 1f))) * 0.3f)));
    color = vec4<f32>(_e248.x, _e248.y, _e248.z, 1f);
    return;
}

@fragment 
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    uv_1 = uv;
    main_1();
    let _e3 = color;
    return _e3;
}
