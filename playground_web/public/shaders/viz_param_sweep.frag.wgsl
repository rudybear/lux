var<private> uv_1: vec2<f32>;
var<private> color: vec4<f32>;

fn main_1() {
    let _e50 = uv_1;
    let _e52 = (_e50.x * 2f);
    let _e53 = uv_1;
    let _e56 = fract(vec2<f32>(_e52, _e53.y));
    let _e58 = normalize(vec3<f32>(0f, 0f, 1f));
    let _e59 = normalize(vec3<f32>(0.5f, 0.5f, 0.8f));
    let _e62 = ((_e56.x * 0.95f) + 0.05f);
    let _e65 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), vec3<f32>(0.8f, 0.3f, 0.2f), vec3(_e56.y));
    let _e67 = normalize((_e58 + _e59));
    let _e69 = max(dot(vec3<f32>(0f, 0f, 1f), _e67), 0f);
    let _e71 = max(dot(vec3<f32>(0f, 0f, 1f), _e58), 0.00001f);
    let _e73 = max(dot(vec3<f32>(0f, 0f, 1f), _e59), 0.00001f);
    let _e76 = (_e62 * _e62);
    let _e77 = (_e76 * _e76);
    let _e81 = (((_e69 * _e69) * (_e77 - 1f)) + 1f);
    let _e88 = (((_e62 * _e62) * _e62) * _e62);
    let _e89 = (1f - _e88);
    let _e108 = (_e65 + ((vec3<f32>(1f, 1f, 1f) - _e65) * pow(clamp((1f - max(dot(_e58, _e67), 0f)), 0f, 1f), 5f)));
    let _e122 = (clamp(clamp((dot(((((((vec3<f32>(1f, 1f, 1f) - _e108) * (1f - _e56.y)) * vec3<f32>(0.8f, 0.3f, 0.2f)) * 0.31830987f) + ((_e108 * ((_e77 * 0.31830987f) / ((_e81 * _e81) + 0.00001f))) * (0.5f / (((_e73 * sqrt((((_e71 * _e71) * _e89) + _e88))) + (_e71 * sqrt((((_e73 * _e73) * _e89) + _e88)))) + 0.00001f)))) * _e73), vec3<f32>(0.2126f, 0.7152f, 0.0722f)) * 3f), 0f, 1f), 0f, 1f) * 4f);
    let _e146 = ((_e56.y * 0.99f) + 0.01f);
    let _e151 = normalize(vec3<f32>(sqrt((1f - (_e146 * _e146))), 0f, _e146));
    let _e152 = normalize(vec3<f32>(0.3f, 0.3f, 0.8f));
    let _e154 = normalize((_e151 + _e152));
    let _e156 = max(dot(vec3<f32>(0f, 0f, 1f), _e154), 0f);
    let _e158 = max(dot(vec3<f32>(0f, 0f, 1f), _e151), 0f);
    let _e160 = max(dot(vec3<f32>(0f, 0f, 1f), _e152), 0f);
    let _e163 = (_e62 * _e62);
    let _e164 = (_e163 * _e163);
    let _e168 = (((_e156 * _e156) * (_e164 - 1f)) + 1f);
    let _e173 = (_e62 + 1f);
    let _e175 = ((_e173 * _e173) / 8f);
    let _e181 = (_e62 + 1f);
    let _e183 = ((_e181 * _e181) / 8f);
    let _e207 = (clamp(clamp((dot(((((vec3<f32>(0.04f, 0.04f, 0.04f) + ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(0.04f, 0.04f, 0.04f)) * pow(clamp((1f - max(dot(_e154, _e151), 0f)), 0f, 1f), 5f))) * ((_e164 * 0.31830987f) / ((_e168 * _e168) + 0.00001f))) * ((_e158 / (((_e158 * (1f - _e175)) + _e175) + 0.00001f)) * (_e160 / (((_e160 * (1f - _e183)) + _e183) + 0.00001f)))) / vec3((((4f * _e158) * _e160) + 0.00001f))), vec3<f32>(0.2126f, 0.7152f, 0.0722f)) * 5f), 0f, 1f), 0f, 1f) * 4f);
    let _e244 = ((select(select(select(select(mix(vec3<f32>(0.544f, 0.774f, 0.247f), vec3<f32>(0.993f, 0.906f, 0.144f), vec3(clamp((_e207 - 3f), 0f, 1f))), mix(vec3<f32>(0.127f, 0.566f, 0.551f), vec3<f32>(0.544f, 0.774f, 0.247f), vec3(clamp((_e207 - 2f), 0f, 1f))), (_e207 < 3f)), mix(vec3<f32>(0.283f, 0.14f, 0.458f), vec3<f32>(0.127f, 0.566f, 0.551f), vec3(clamp((_e207 - 1f), 0f, 1f))), (_e207 < 2f)), mix(vec3<f32>(0.267f, 0.004f, 0.329f), vec3<f32>(0.283f, 0.14f, 0.458f), vec3(clamp(_e207, 0f, 1f))), (_e207 < 1f)), select(select(select(mix(vec3<f32>(0.544f, 0.774f, 0.247f), vec3<f32>(0.993f, 0.906f, 0.144f), vec3(clamp((_e122 - 3f), 0f, 1f))), mix(vec3<f32>(0.127f, 0.566f, 0.551f), vec3<f32>(0.544f, 0.774f, 0.247f), vec3(clamp((_e122 - 2f), 0f, 1f))), (_e122 < 3f)), mix(vec3<f32>(0.283f, 0.14f, 0.458f), vec3<f32>(0.127f, 0.566f, 0.551f), vec3(clamp((_e122 - 1f), 0f, 1f))), (_e122 < 2f)), mix(vec3<f32>(0.267f, 0.004f, 0.329f), vec3<f32>(0.283f, 0.14f, 0.458f), vec3(clamp(_e122, 0f, 1f))), (_e122 < 1f)), (floor(_e52) < 0.5f)) * (smoothstep(0f, 0.01f, _e56.x) * smoothstep(0f, 0.01f, (1f - _e56.x)))) * (smoothstep(0f, 0.01f, _e56.y) * smoothstep(0f, 0.01f, (1f - _e56.y))));
    color = vec4<f32>(_e244.x, _e244.y, _e244.z, 1f);
    return;
}

@fragment 
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    uv_1 = uv;
    main_1();
    let _e3 = color;
    return _e3;
}
