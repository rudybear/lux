struct Light {
    light_dir: vec3<f32>,
    view_pos: vec3<f32>,
}

var<private> world_pos_1: vec3<f32>;
var<private> world_normal_1: vec3<f32>;
var<private> frag_uv_1: vec2<f32>;
var<private> color: vec4<f32>;
@group(1) @binding(0) 
var<uniform> global: Light;

fn main_1() {
    var local: f32;
    var local_1: vec3<f32>;

    let _e55 = world_normal_1;
    let _e56 = normalize(_e55);
    let _e58 = global.view_pos;
    let _e59 = world_pos_1;
    let _e61 = normalize((_e58 - _e59));
    let _e62 = normalize(vec3<f32>(0.5f, -0.7f, 0.5f));
    let _e64 = normalize((_e61 + _e62));
    let _e66 = max(dot(_e56, _e64), 0f);
    let _e68 = max(dot(_e56, _e61), 0.001f);
    let _e70 = max(dot(_e56, _e62), 0f);
    let _e72 = max(dot(_e61, _e64), 0f);
    let _e73 = max(0.2f, 0.045f);
    let _e74 = (_e73 * _e73);
    let _e75 = (_e74 * _e74);
    let _e79 = (((_e66 * _e66) * (_e75 - 1f)) + 1f);
    let _e86 = (((_e73 * _e73) * _e73) * _e73);
    let _e87 = (1f - _e86);
    let _e101 = (vec3<f32>(0.912f, 0.914f, 0.92f) * 1f);
    let _e102 = clamp(_e72, 0f, 1f);
    let _e103 = (1f - _e102);
    let _e118 = ((1.5f - 1f) / (1.5f + 1f));
    let _e125 = (sign((1.5f - 1f)) * sqrt(clamp((1f * (_e118 * _e118)), 0f, 1f)));
    let _e129 = ((1f + _e125) / max((1f - _e125), 0.0001f));
    let _e130 = clamp(_e72, 0f, 1f);
    let _e134 = ((1f - (_e130 * _e130)) / (_e129 * _e129));
    let _e137 = sqrt((1f - min(_e134, 0.9999f)));
    let _e138 = (_e129 * _e137);
    let _e141 = ((_e130 - _e138) / (_e130 + _e138));
    let _e142 = (_e129 * _e130);
    let _e145 = ((_e142 - _e137) / (_e142 + _e137));
    let _e152 = (((_e75 * 0.31830987f) / ((_e79 * _e79) + 0.00001f)) * (0.5f / (((_e70 * sqrt((((_e68 * _e68) * _e87) + _e86))) + (_e68 * sqrt((((_e70 * _e70) * _e87) + _e86)))) + 0.00001f)));
    let _e163 = (max(dot(_e61, _e62), 0f) - (_e68 * _e70));
    local = _e163;
    if (_e163 > 0f) {
        local = (_e163 / max(_e70, _e68));
    }
    let _e169 = (1f / (1f + (0.2877934f * 0f)));
    let _e172 = local;
    let _e176 = (1f - _e70);
    let _e177 = (_e176 * _e176);
    let _e192 = (1f - _e68);
    let _e193 = (_e192 * _e192);
    let _e210 = (_e169 * (1f + (0.07248821f * 0f)));
    let _e211 = (1f - _e210);
    let _e233 = ((1.5f - 1f) / (1.5f + 1f));
    let _e240 = (sign((1.5f - 1f)) * sqrt(clamp((1f * (_e233 * _e233)), 0f, 1f)));
    let _e244 = ((1f + _e240) / max((1f - _e240), 0.0001f));
    let _e247 = ((_e244 - 1f) / (_e244 + 1f));
    let _e249 = (0.2f * 0.2f);
    let _e269 = max(dot(_e56, _e61), 0.001f);
    local_1 = (((mix(vec3((_e152 * mix((0.5f * ((_e141 * _e141) + (_e145 * _e145))), 1f, step(1f, _e134)))), (vec3(_e152) * (clamp(((_e101 + ((vec3<f32>(1f, 1f, 1f) - _e101) * pow(_e103, 5f))) - ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(0.97f, 0.979f, 0.988f)) * (_e102 * pow(_e103, 6f)))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f)) * min(1f, 1f))), vec3(1f)) * _e70) + (((((((_e101 * 0.31830987f) * _e169) * (1f + (0f * _e172))) + (((((((_e101 * _e101) * _e210) / (vec3<f32>(1f, 1f, 1f) - (_e101 * max(_e211, 0f)))) * 0.31830987f) * max((1f - ((1f + (0f * (((0.05710853f * (1f + _e176)) + ((-0.33218145f + (0.071443f * _e177)) * _e177)) + ((0.49188188f * _e176) * _e177)))) / (1f + (0.2877934f * 0f)))), 0f)) * max((1f - ((1f + (0f * (((0.05710853f * (1f + _e192)) + ((-0.33218145f + (0.071443f * _e193)) * _e193)) + ((0.49188188f * _e192) * _e193)))) / (1f + (0.2877934f * 0f)))), 0f)) / vec3(max(_e211, 0.001f)))) * max(_e70, 0f)) * (1f - 1f)) * (1f - clamp((((_e247 * _e247) * (1f - ((_e249 * max((1f - _e68), _e68)) * 0.31830987f))) + ((_e249 * max(0f, (0.75f - _e68))) * 0.31830987f)), 0f, 1f)))) * vec3<f32>(3f, 2.94f, 2.85f));
    if (0f > 0f) {
        let _e279 = local_1;
        local_1 = mix(_e279, ((vec3<f32>(0.912f, 0.914f, 0.92f) * exp(((log(vec3<f32>(1f, 1f, 1f)) / vec3(max(0f, 0.001f))) * 0f))) * 0f), vec3(0f));
    }
    let _e282 = local_1;
    local_1 = (_e282 + vec3<f32>(0f, 0f, 0f));
    if (0f > 0f) {
        let _e287 = ((1.5f - 1f) / (1.5f + 1f));
        let _e295 = ((1.6f - 1f) / (1.6f + 1f));
        let _e296 = (_e295 * _e295);
        let _e309 = clamp(_e269, 0f, 1f);
        let _e313 = ((1f - (_e309 * _e309)) / (1.6f * 1.6f));
        let _e316 = sqrt((1f - min(_e313, 0.9999f)));
        let _e317 = (1.6f * _e316);
        let _e320 = ((_e309 - _e317) / (_e309 + _e317));
        let _e321 = (1.6f * _e309);
        let _e324 = ((_e321 - _e316) / (_e321 + _e316));
        let _e331 = mix(mix((0.5f * ((_e320 * _e320) + (_e324 * _e324))), 1f, step(1f, _e313)), (1f - ((1f - (_e296 + ((1f - _e296) * ((0.0477f + (0.8965f * _e296)) + ((0.0582f * _e296) * _e296))))) / (1.6f * 1.6f))), 0.2f);
        let _e349 = normalize((_e61 + _e62));
        let _e351 = max(dot(_e56, _e349), 0f);
        let _e353 = max(dot(_e56, _e61), 0.001f);
        let _e355 = max(dot(_e56, _e62), 0f);
        let _e358 = max(0f, 0.045f);
        let _e359 = (_e358 * _e358);
        let _e360 = (_e359 * _e359);
        let _e364 = (((_e351 * _e351) * (_e360 - 1f)) + 1f);
        let _e371 = (((_e358 * _e358) * _e358) * _e358);
        let _e372 = (1f - _e371);
        let _e386 = clamp(max(dot(_e61, _e349), 0f), 0f, 1f);
        let _e390 = ((1f - (_e386 * _e386)) / (1.6f * 1.6f));
        let _e393 = sqrt((1f - min(_e390, 0.9999f)));
        let _e394 = (1.6f * _e393);
        let _e397 = ((_e386 - _e394) / (_e386 + _e394));
        let _e398 = (1.6f * _e386);
        let _e401 = ((_e398 - _e393) / (_e398 + _e393));
        let _e414 = ((1.6f - 1f) / (1.6f + 1f));
        let _e416 = (0f * 0f);
        let _e431 = local_1;
        local_1 = (((((_e431 * mix(vec3<f32>(1f, 1f, 1f), (vec3(max((1f - _e331), 0f)) / max((vec3<f32>(1f, 1f, 1f) - ((mix((vec3<f32>(0.912f, 0.914f, 0.92f) * 1f), vec3<f32>(0.97f, 0.979f, 0.988f), vec3(1f)) * (_e287 * _e287)) * _e331)), vec3<f32>(0.001f, 0.001f, 0.001f))), vec3((0f * 1f)))) * mix(vec3<f32>(1f, 1f, 1f), pow(vec3<f32>(1f, 1f, 1f), vec3((1f / max(_e269, 0.01f)))), vec3(0f))) * (1f - (0f * clamp((((_e414 * _e414) * (1f - ((_e416 * max((1f - _e269), _e269)) * 0.31830987f))) + ((_e416 * max(0f, (0.75f - _e269))) * 0.31830987f)), 0f, 1f)))) + vec3(((((0f * ((_e360 * 0.31830987f) / ((_e364 * _e364) + 0.00001f))) * (0.5f / (((_e355 * sqrt((((_e353 * _e353) * _e372) + _e371))) + (_e353 * sqrt((((_e355 * _e355) * _e372) + _e371)))) + 0.00001f))) * mix((0.5f * ((_e397 * _e397) + (_e401 * _e401))), 1f, step(1f, _e390))) * _e355))) + vec3<f32>(0f, 0f, 0f));
    }
    if (0f > 0f) {
        let _e442 = max(dot(_e56, normalize((_e61 + _e62))), 0f);
        let _e444 = max(dot(_e56, _e62), 0f);
        let _e446 = max(dot(_e56, _e61), 0.001f);
        let _e447 = max(0.5f, 0.045f);
        let _e449 = (1f / (_e447 * _e447));
        let _e475 = local_1;
        local_1 = ((_e475 * (1f - ((0f * max(vec3<f32>(1f, 1f, 1f).x, max(vec3<f32>(1f, 1f, 1f).y, vec3<f32>(1f, 1f, 1f).z))) * 0.3f))) + ((((vec3<f32>(1f, 1f, 1f) * (((2f + _e449) * pow(max((1f - (_e442 * _e442)), 0f), (_e449 * 0.5f))) / 6.2831855f)) * (1f / ((4f * ((_e444 + _e446) - (_e444 * _e446))) + 0.00001f))) * max(_e444, 0f)) * 0f));
    }
    if (0f > 0f) {
        let _e489 = local_1;
        local_1 = (_e489 + (((vec3<f32>(1f, 1f, 1f) * 0f) * 0.01f) * mix(vec3<f32>(1f, 1f, 1f), pow(vec3<f32>(1f, 1f, 1f), vec3((1f / max(_e269, 0.01f)))), vec3(0f))));
    }
    let _e492 = local_1;
    color = vec4<f32>(_e492.x, _e492.y, _e492.z, 1f);
    return;
}

@fragment 
fn main(@location(0) world_pos: vec3<f32>, @location(1) world_normal: vec3<f32>, @location(2) frag_uv: vec2<f32>) -> @location(0) vec4<f32> {
    world_pos_1 = world_pos;
    world_normal_1 = world_normal;
    frag_uv_1 = frag_uv;
    main_1();
    let _e7 = color;
    return _e7;
}
