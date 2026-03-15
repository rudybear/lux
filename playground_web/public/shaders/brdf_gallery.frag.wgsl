var<private> uv_1: vec2<f32>;
var<private> color: vec4<f32>;

fn main_1() {
    let _e29 = uv_1;
    let _e33 = uv_1;
    let _e38 = normalize(vec3<f32>(((_e29.x * 0.5f) - 0.25f), ((_e33.y * 0.5f) - 0.25f), 1f));
    let _e39 = uv_1;
    let _e44 = normalize(vec3<f32>(((_e39.x * 2f) - 1f), 0.5f, 0.8f));
    let _e46 = max(dot(vec3<f32>(0f, 0f, 1f), _e44), 0f);
    let _e47 = uv_1;
    let _e49 = (_e47.y * 4f);
    let _e54 = normalize((_e38 + _e44));
    let _e56 = max(dot(vec3<f32>(0f, 0f, 1f), _e54), 0f);
    let _e58 = max(dot(vec3<f32>(0f, 0f, 1f), _e38), 0f);
    let _e60 = max(dot(vec3<f32>(0f, 0f, 1f), _e44), 0f);
    let _e63 = (0.3f * 0.3f);
    let _e64 = (_e63 * _e63);
    let _e68 = (((_e56 * _e56) * (_e64 - 1f)) + 1f);
    let _e73 = (0.3f + 1f);
    let _e75 = ((_e73 * _e73) / 8f);
    let _e81 = (0.3f + 1f);
    let _e83 = ((_e81 * _e81) / 8f);
    let _e110 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), vec3<f32>(0.95f, 0.64f, 0.54f), vec3(0.9f));
    let _e129 = normalize((_e38 + _e44));
    let _e131 = max(dot(vec3<f32>(0f, 0f, 1f), _e129), 0f);
    let _e133 = max(dot(vec3<f32>(0f, 0f, 1f), _e38), 0f);
    let _e135 = max(dot(vec3<f32>(0f, 0f, 1f), _e44), 0f);
    let _e138 = (0.3f * 0.3f);
    let _e139 = (_e138 * _e138);
    let _e143 = (((_e131 * _e131) * (_e139 - 1f)) + 1f);
    let _e148 = (0.3f + 1f);
    let _e150 = ((_e148 * _e148) / 8f);
    let _e156 = (0.3f + 1f);
    let _e158 = ((_e156 * _e156) / 8f);
    let _e184 = normalize((_e38 + _e44));
    let _e186 = max(dot(vec3<f32>(0f, 0f, 1f), _e184), 0f);
    let _e188 = max(dot(vec3<f32>(0f, 0f, 1f), _e44), 0f);
    let _e190 = max(dot(vec3<f32>(0f, 0f, 1f), _e38), 0f);
    let _e193 = (0.1f * 0.1f);
    let _e194 = (_e193 * _e193);
    let _e198 = (((_e186 * _e186) * (_e194 - 1f)) + 1f);
    let _e205 = (((0.1f * 0.1f) * 0.1f) * 0.1f);
    let _e206 = (1f - _e205);
    let _e236 = select(select(select((((vec3<f32>(0.3f, 0.3f, 0.8f) * 0.31830987f) * max(_e46, 0f)) + vec3(((((1f * ((_e194 * 0.31830987f) / ((_e198 * _e198) + 0.00001f))) * (0.5f / (((_e188 * sqrt((((_e190 * _e190) * _e206) + _e205))) + (_e190 * sqrt((((_e188 * _e188) * _e206) + _e205)))) + 0.00001f))) * (0.04f + ((1f - 0.04f) * pow((1f - max(dot(_e38, _e184), 0f)), 5f)))) * _e188))), ((((((vec3<f32>(1f, 1f, 1f) - (_e110 + ((vec3<f32>(1f, 1f, 1f) - _e110) * pow(clamp((1f - max(dot(normalize((_e38 + _e44)), _e38), 0f)), 0f, 1f), 5f)))) * (1f - 0.9f)) * vec3<f32>(0.95f, 0.64f, 0.54f)) * 0.31830987f) + ((((_e110 + ((vec3<f32>(1f, 1f, 1f) - _e110) * pow(clamp((1f - max(dot(_e129, _e38), 0f)), 0f, 1f), 5f))) * ((_e139 * 0.31830987f) / ((_e143 * _e143) + 0.00001f))) * ((_e133 / (((_e133 * (1f - _e150)) + _e150) + 0.00001f)) * (_e135 / (((_e135 * (1f - _e158)) + _e158) + 0.00001f)))) / vec3((((4f * _e133) * _e135) + 0.00001f)))) * max(dot(vec3<f32>(0f, 0f, 1f), _e44), 0f)), (_e49 < 3f)), ((((((vec3<f32>(0.04f, 0.04f, 0.04f) + ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(0.04f, 0.04f, 0.04f)) * pow(clamp((1f - max(dot(_e54, _e38), 0f)), 0f, 1f), 5f))) * ((_e64 * 0.31830987f) / ((_e68 * _e68) + 0.00001f))) * ((_e58 / (((_e58 * (1f - _e75)) + _e75) + 0.00001f)) * (_e60 / (((_e60 * (1f - _e83)) + _e83) + 0.00001f)))) / vec3((((4f * _e58) * _e60) + 0.00001f))) * _e46) + (((vec3<f32>(0.2f, 0.8f, 0.2f) * 0.31830987f) * max(_e46, 0f)) * 0.5f)), (_e49 < 2f)), ((vec3<f32>(0.8f, 0.2f, 0.2f) * 0.31830987f) * max(_e46, 0f)), (_e49 < 1f));
    color = vec4<f32>(_e236.x, _e236.y, _e236.z, 1f);
    return;
}

@fragment 
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    uv_1 = uv;
    main_1();
    let _e3 = color;
    return _e3;
}
