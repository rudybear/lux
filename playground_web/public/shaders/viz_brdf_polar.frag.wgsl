var<private> uv_1: vec2<f32>;
var<private> color: vec4<f32>;

fn main_1() {
    let _e49 = uv_1;
    let _e51 = (_e49.x * 2f);
    let _e52 = uv_1;
    let _e54 = (_e52.y * 2f);
    let _e56 = fract(vec2<f32>(_e51, _e54));
    let _e57 = floor(_e51);
    let _e58 = floor(_e54);
    let _e61 = ((_e56.x * 2f) - 1f);
    let _e64 = ((_e56.y * 2f) - 1f);
    let _e68 = sqrt(((_e61 * _e61) + (_e64 * _e64)));
    let _e69 = atan2(_e61, _e64);
    let _e70 = normalize(vec3<f32>(0.707f, 0f, 0.707f));
    let _e72 = cos(_e69);
    let _e75 = normalize(vec3<f32>(sin(_e69), 0f, max(_e72, 0.001f)));
    let _e77 = normalize((_e70 + _e75));
    let _e79 = max(dot(vec3<f32>(0f, 0f, 1f), _e77), 0f);
    let _e81 = max(dot(vec3<f32>(0f, 0f, 1f), _e70), 0f);
    let _e83 = max(dot(vec3<f32>(0f, 0f, 1f), _e75), 0f);
    let _e86 = (0.2f * 0.2f);
    let _e87 = (_e86 * _e86);
    let _e91 = (((_e79 * _e79) * (_e87 - 1f)) + 1f);
    let _e96 = (0.2f + 1f);
    let _e98 = ((_e96 * _e96) / 8f);
    let _e104 = (0.2f + 1f);
    let _e106 = ((_e104 * _e104) / 8f);
    let _e128 = clamp((length(((((vec3<f32>(0.04f, 0.04f, 0.04f) + ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(0.04f, 0.04f, 0.04f)) * pow(clamp((1f - max(dot(_e77, _e70), 0f)), 0f, 1f), 5f))) * ((_e87 * 0.31830987f) / ((_e91 * _e91) + 0.00001f))) * ((_e81 / (((_e81 * (1f - _e98)) + _e98) + 0.00001f)) * (_e83 / (((_e83 * (1f - _e106)) + _e106) + 0.00001f)))) / vec3((((4f * _e81) * _e83) + 0.00001f)))) * 2f), 0f, 0.95f);
    let _e129 = (_e72 > 0f);
    let _e133 = select(0f, 1f, _e129);
    let _e143 = max(dot(vec3<f32>(0f, 0f, 1f), _e75), 0f);
    let _e149 = clamp((length(((vec3<f32>(1f, 1f, 1f) * 0.31830987f) * max(_e143, 0f))) * 3f), 0f, 0.95f);
    let _e164 = max(dot(vec3<f32>(0f, 0f, 1f), normalize((_e70 + _e75))), 0f);
    let _e166 = max(dot(vec3<f32>(0f, 0f, 1f), _e70), 0f);
    let _e168 = (1f / (0.5f * 0.5f));
    let _e189 = clamp((length((((vec3<f32>(1f, 1f, 1f) * (((2f + _e168) * pow(max((1f - (_e164 * _e164)), 0f), (_e168 * 0.5f))) / 6.2831855f)) * (1f / ((4f * ((_e143 + _e166) - (_e143 * _e166))) + 0.00001f))) * max(_e143, 0f))) * 4f), 0f, 0.95f);
    let _e202 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), vec3<f32>(0.8f, 0.4f, 0.2f), vec3(0f));
    let _e221 = normalize((_e70 + _e75));
    let _e223 = max(dot(vec3<f32>(0f, 0f, 1f), _e221), 0f);
    let _e225 = max(dot(vec3<f32>(0f, 0f, 1f), _e70), 0f);
    let _e227 = max(dot(vec3<f32>(0f, 0f, 1f), _e75), 0f);
    let _e230 = (0.3f * 0.3f);
    let _e231 = (_e230 * _e230);
    let _e235 = (((_e223 * _e223) * (_e231 - 1f)) + 1f);
    let _e240 = (0.3f + 1f);
    let _e242 = ((_e240 * _e240) / 8f);
    let _e248 = (0.3f + 1f);
    let _e250 = ((_e248 * _e248) / 8f);
    let _e274 = clamp((length(((((((vec3<f32>(1f, 1f, 1f) - (_e202 + ((vec3<f32>(1f, 1f, 1f) - _e202) * pow(clamp((1f - max(dot(normalize((_e70 + _e75)), _e70), 0f)), 0f, 1f), 5f)))) * (1f - 0f)) * vec3<f32>(0.8f, 0.4f, 0.2f)) * 0.31830987f) + ((((_e202 + ((vec3<f32>(1f, 1f, 1f) - _e202) * pow(clamp((1f - max(dot(_e221, _e70), 0f)), 0f, 1f), 5f))) * ((_e231 * 0.31830987f) / ((_e235 * _e235) + 0.00001f))) * ((_e225 / (((_e225 * (1f - _e242)) + _e242) + 0.00001f)) * (_e227 / (((_e227 * (1f - _e250)) + _e250) + 0.00001f)))) / vec3((((4f * _e225) * _e227) + 0.00001f)))) * max(dot(vec3<f32>(0f, 0f, 1f), _e75), 0f))) * 3f), 0f, 0.95f);
    let _e291 = (_e64 > 0f);
    let _e314 = (_e57 < 0.5f);
    let _e315 = (_e58 < 0.5f);
    let _e318 = (_e57 > 0.5f);
    let _e321 = (_e58 > 0.5f);
    let _e347 = (((((((((vec3<f32>(0.05f, 0.05f, 0.07f) + (vec3<f32>(0.2f, 0.7f, 1f) * select(0f, 0.4f, ((_e68 < _e128) && _e129)))) + (vec3<f32>(0.4f, 0.9f, 1f) * (smoothstep(0.02f, 0.005f, abs((_e68 - _e128))) * _e133))) * select(0f, 1f, (_e314 && _e315))) + (((vec3<f32>(0.05f, 0.05f, 0.07f) + (vec3<f32>(0.2f, 0.9f, 0.3f) * select(0f, 0.4f, ((_e68 < _e149) && _e129)))) + (vec3<f32>(0.4f, 1f, 0.5f) * (smoothstep(0.02f, 0.005f, abs((_e68 - _e149))) * _e133))) * select(0f, 1f, (_e318 && _e315)))) + (((vec3<f32>(0.05f, 0.05f, 0.07f) + (vec3<f32>(0.9f, 0.3f, 0.8f) * select(0f, 0.4f, ((_e68 < _e189) && _e129)))) + (vec3<f32>(1f, 0.5f, 0.9f) * (smoothstep(0.02f, 0.005f, abs((_e68 - _e189))) * _e133))) * select(0f, 1f, (_e314 && _e321)))) + (((vec3<f32>(0.05f, 0.05f, 0.07f) + (vec3<f32>(0.9f, 0.6f, 0.2f) * select(0f, 0.4f, ((_e68 < _e274) && _e129)))) + (vec3<f32>(1f, 0.8f, 0.4f) * (smoothstep(0.02f, 0.005f, abs((_e68 - _e274))) * _e133))) * select(0f, 1f, (_e318 && _e321)))) + vec3(((((smoothstep(0.01f, 0.003f, abs((_e68 - 0.95f))) * select(0f, 0.3f, _e129)) + (smoothstep(0.008f, 0.002f, abs(_e61)) * select(0f, 0.25f, _e291))) + (smoothstep(0.008f, 0.002f, abs(_e64)) * 0.15f)) + (smoothstep(0.01f, 0.003f, abs(((_e61 * 0.707f) - (_e64 * 0.707f)))) * select(0f, 0.4f, (((_e61 > 0f) && _e291) && (_e68 < 0.6f))))))) * (smoothstep(0f, 0.015f, _e56.x) * smoothstep(0f, 0.015f, (1f - _e56.x)))) * (smoothstep(0f, 0.015f, _e56.y) * smoothstep(0f, 0.015f, (1f - _e56.y))));
    color = vec4<f32>(_e347.x, _e347.y, _e347.z, 1f);
    return;
}

@fragment 
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    uv_1 = uv;
    main_1();
    let _e3 = color;
    return _e3;
}
