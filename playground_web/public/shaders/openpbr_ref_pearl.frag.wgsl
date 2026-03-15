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

    let _e51 = world_normal_1;
    let _e52 = normalize(_e51);
    let _e54 = global.view_pos;
    let _e55 = world_pos_1;
    let _e57 = normalize((_e54 - _e55));
    let _e58 = normalize(vec3<f32>(0.5f, -0.7f, 0.5f));
    let _e60 = normalize((_e57 + _e58));
    let _e62 = max(dot(_e52, _e60), 0f);
    let _e64 = max(dot(_e52, _e57), 0.001f);
    let _e66 = max(dot(_e52, _e58), 0f);
    let _e68 = max(dot(_e57, _e60), 0f);
    let _e69 = max(0.35f, 0.045f);
    let _e70 = (_e69 * _e69);
    let _e71 = (_e70 * _e70);
    let _e75 = (((_e62 * _e62) * (_e71 - 1f)) + 1f);
    let _e82 = (((_e69 * _e69) * _e69) * _e69);
    let _e83 = (1f - _e82);
    let _e97 = (vec3<f32>(0.8f, 0.75f, 0.7f) * 1f);
    let _e98 = clamp(_e68, 0f, 1f);
    let _e99 = (1f - _e98);
    let _e114 = ((1.5f - 1f) / (1.5f + 1f));
    let _e121 = (sign((1.5f - 1f)) * sqrt(clamp((1f * (_e114 * _e114)), 0f, 1f)));
    let _e125 = ((1f + _e121) / max((1f - _e121), 0.0001f));
    let _e126 = clamp(_e68, 0f, 1f);
    let _e130 = ((1f - (_e126 * _e126)) / (_e125 * _e125));
    let _e133 = sqrt((1f - min(_e130, 0.9999f)));
    let _e134 = (_e125 * _e133);
    let _e137 = ((_e126 - _e134) / (_e126 + _e134));
    let _e138 = (_e125 * _e126);
    let _e141 = ((_e138 - _e133) / (_e138 + _e133));
    let _e148 = (((_e71 * 0.31830987f) / ((_e75 * _e75) + 0.00001f)) * (0.5f / (((_e66 * sqrt((((_e64 * _e64) * _e83) + _e82))) + (_e64 * sqrt((((_e66 * _e66) * _e83) + _e82)))) + 0.00001f)));
    let _e159 = (max(dot(_e57, _e58), 0f) - (_e64 * _e66));
    local = _e159;
    if (_e159 > 0f) {
        local = (_e159 / max(_e66, _e64));
    }
    let _e165 = (1f / (1f + (0.2877934f * 0f)));
    let _e168 = local;
    let _e172 = (1f - _e66);
    let _e173 = (_e172 * _e172);
    let _e188 = (1f - _e64);
    let _e189 = (_e188 * _e188);
    let _e206 = (_e165 * (1f + (0.07248821f * 0f)));
    let _e207 = (1f - _e206);
    let _e229 = ((1.5f - 1f) / (1.5f + 1f));
    let _e236 = (sign((1.5f - 1f)) * sqrt(clamp((1f * (_e229 * _e229)), 0f, 1f)));
    let _e240 = ((1f + _e236) / max((1f - _e236), 0.0001f));
    let _e243 = ((_e240 - 1f) / (_e240 + 1f));
    let _e245 = (0.35f * 0.35f);
    let _e265 = max(dot(_e52, _e57), 0.001f);
    local_1 = (((mix(vec3((_e148 * mix((0.5f * ((_e137 * _e137) + (_e141 * _e141))), 1f, step(1f, _e130)))), (vec3(_e148) * (clamp(((_e97 + ((vec3<f32>(1f, 1f, 1f) - _e97) * pow(_e99, 5f))) - ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(1f, 1f, 1f)) * (_e98 * pow(_e99, 6f)))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f)) * min(1f, 1f))), vec3(0f)) * _e66) + (((((((_e97 * 0.31830987f) * _e165) * (1f + (0f * _e168))) + (((((((_e97 * _e97) * _e206) / (vec3<f32>(1f, 1f, 1f) - (_e97 * max(_e207, 0f)))) * 0.31830987f) * max((1f - ((1f + (0f * (((0.05710853f * (1f + _e172)) + ((-0.33218145f + (0.071443f * _e173)) * _e173)) + ((0.49188188f * _e172) * _e173)))) / (1f + (0.2877934f * 0f)))), 0f)) * max((1f - ((1f + (0f * (((0.05710853f * (1f + _e188)) + ((-0.33218145f + (0.071443f * _e189)) * _e189)) + ((0.49188188f * _e188) * _e189)))) / (1f + (0.2877934f * 0f)))), 0f)) / vec3(max(_e207, 0.001f)))) * max(_e66, 0f)) * (1f - 0f)) * (1f - clamp((((_e243 * _e243) * (1f - ((_e245 * max((1f - _e64), _e64)) * 0.31830987f))) + ((_e245 * max(0f, (0.75f - _e64))) * 0.31830987f)), 0f, 1f)))) * vec3<f32>(3f, 2.94f, 2.85f));
    if (0f > 0f) {
        let _e275 = local_1;
        local_1 = mix(_e275, ((vec3<f32>(0.8f, 0.75f, 0.7f) * exp(((log(vec3<f32>(1f, 1f, 1f)) / vec3(max(0f, 0.001f))) * 0f))) * 0f), vec3(0f));
    }
    let _e278 = local_1;
    local_1 = (_e278 + vec3<f32>(0f, 0f, 0f));
    if (1f > 0f) {
        let _e283 = ((1.5f - 1f) / (1.5f + 1f));
        let _e291 = ((1.68f - 1f) / (1.68f + 1f));
        let _e292 = (_e291 * _e291);
        let _e305 = clamp(_e265, 0f, 1f);
        let _e309 = ((1f - (_e305 * _e305)) / (1.68f * 1.68f));
        let _e312 = sqrt((1f - min(_e309, 0.9999f)));
        let _e313 = (1.68f * _e312);
        let _e316 = ((_e305 - _e313) / (_e305 + _e313));
        let _e317 = (1.68f * _e305);
        let _e320 = ((_e317 - _e312) / (_e317 + _e312));
        let _e327 = mix(mix((0.5f * ((_e316 * _e316) + (_e320 * _e320))), 1f, step(1f, _e309)), (1f - ((1f - (_e292 + ((1f - _e292) * ((0.0477f + (0.8965f * _e292)) + ((0.0582f * _e292) * _e292))))) / (1.68f * 1.68f))), 0.35f);
        let _e345 = normalize((_e57 + _e58));
        let _e347 = max(dot(_e52, _e345), 0f);
        let _e349 = max(dot(_e52, _e57), 0.001f);
        let _e351 = max(dot(_e52, _e58), 0f);
        let _e354 = max(0.15f, 0.045f);
        let _e355 = (_e354 * _e354);
        let _e356 = (_e355 * _e355);
        let _e360 = (((_e347 * _e347) * (_e356 - 1f)) + 1f);
        let _e367 = (((_e354 * _e354) * _e354) * _e354);
        let _e368 = (1f - _e367);
        let _e382 = clamp(max(dot(_e57, _e345), 0f), 0f, 1f);
        let _e386 = ((1f - (_e382 * _e382)) / (1.68f * 1.68f));
        let _e389 = sqrt((1f - min(_e386, 0.9999f)));
        let _e390 = (1.68f * _e389);
        let _e393 = ((_e382 - _e390) / (_e382 + _e390));
        let _e394 = (1.68f * _e382);
        let _e397 = ((_e394 - _e389) / (_e394 + _e389));
        let _e410 = ((1.68f - 1f) / (1.68f + 1f));
        let _e412 = (0.15f * 0.15f);
        let _e427 = local_1;
        local_1 = (((((_e427 * mix(vec3<f32>(1f, 1f, 1f), (vec3(max((1f - _e327), 0f)) / max((vec3<f32>(1f, 1f, 1f) - ((mix((vec3<f32>(0.8f, 0.75f, 0.7f) * 1f), vec3<f32>(1f, 1f, 1f), vec3(0f)) * (_e283 * _e283)) * _e327)), vec3<f32>(0.001f, 0.001f, 0.001f))), vec3((1f * 1f)))) * mix(vec3<f32>(1f, 1f, 1f), pow(vec3<f32>(1f, 1f, 1f), vec3((1f / max(_e265, 0.01f)))), vec3(1f))) * (1f - (1f * clamp((((_e410 * _e410) * (1f - ((_e412 * max((1f - _e265), _e265)) * 0.31830987f))) + ((_e412 * max(0f, (0.75f - _e265))) * 0.31830987f)), 0f, 1f)))) + vec3(((((1f * ((_e356 * 0.31830987f) / ((_e360 * _e360) + 0.00001f))) * (0.5f / (((_e351 * sqrt((((_e349 * _e349) * _e368) + _e367))) + (_e349 * sqrt((((_e351 * _e351) * _e368) + _e367)))) + 0.00001f))) * mix((0.5f * ((_e393 * _e393) + (_e397 * _e397))), 1f, step(1f, _e386))) * _e351))) + vec3<f32>(0f, 0f, 0f));
    }
    if (0f > 0f) {
        let _e438 = max(dot(_e52, normalize((_e57 + _e58))), 0f);
        let _e440 = max(dot(_e52, _e58), 0f);
        let _e442 = max(dot(_e52, _e57), 0.001f);
        let _e443 = max(0.5f, 0.045f);
        let _e445 = (1f / (_e443 * _e443));
        let _e471 = local_1;
        local_1 = ((_e471 * (1f - ((0f * max(vec3<f32>(1f, 1f, 1f).x, max(vec3<f32>(1f, 1f, 1f).y, vec3<f32>(1f, 1f, 1f).z))) * 0.3f))) + ((((vec3<f32>(1f, 1f, 1f) * (((2f + _e445) * pow(max((1f - (_e438 * _e438)), 0f), (_e445 * 0.5f))) / 6.2831855f)) * (1f / ((4f * ((_e440 + _e442) - (_e440 * _e442))) + 0.00001f))) * max(_e440, 0f)) * 0f));
    }
    if (0f > 0f) {
        let _e485 = local_1;
        local_1 = (_e485 + (((vec3<f32>(1f, 1f, 1f) * 0f) * 0.01f) * mix(vec3<f32>(1f, 1f, 1f), pow(vec3<f32>(1f, 1f, 1f), vec3((1f / max(_e265, 0.01f)))), vec3(1f))));
    }
    let _e488 = local_1;
    color = vec4<f32>(_e488.x, _e488.y, _e488.z, 1f);
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
