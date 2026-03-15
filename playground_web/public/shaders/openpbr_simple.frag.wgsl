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

    let _e49 = world_normal_1;
    let _e50 = normalize(_e49);
    let _e52 = global.view_pos;
    let _e53 = world_pos_1;
    let _e55 = normalize((_e52 - _e53));
    let _e56 = normalize(vec3<f32>(0.5f, -0.8f, 0.3f));
    let _e58 = normalize((_e55 + _e56));
    let _e60 = max(dot(_e50, _e58), 0f);
    let _e62 = max(dot(_e50, _e55), 0.001f);
    let _e64 = max(dot(_e50, _e56), 0f);
    let _e66 = max(dot(_e55, _e58), 0f);
    let _e67 = max(0.3f, 0.045f);
    let _e68 = (_e67 * _e67);
    let _e69 = (_e68 * _e68);
    let _e73 = (((_e60 * _e60) * (_e69 - 1f)) + 1f);
    let _e80 = (((_e67 * _e67) * _e67) * _e67);
    let _e81 = (1f - _e80);
    let _e95 = (vec3<f32>(0.8f, 0.2f, 0.1f) * 1f);
    let _e96 = clamp(_e66, 0f, 1f);
    let _e97 = (1f - _e96);
    let _e112 = ((1.5f - 1f) / (1.5f + 1f));
    let _e119 = (sign((1.5f - 1f)) * sqrt(clamp((1f * (_e112 * _e112)), 0f, 1f)));
    let _e123 = ((1f + _e119) / max((1f - _e119), 0.0001f));
    let _e124 = clamp(_e66, 0f, 1f);
    let _e128 = ((1f - (_e124 * _e124)) / (_e123 * _e123));
    let _e131 = sqrt((1f - min(_e128, 0.9999f)));
    let _e132 = (_e123 * _e131);
    let _e135 = ((_e124 - _e132) / (_e124 + _e132));
    let _e136 = (_e123 * _e124);
    let _e139 = ((_e136 - _e131) / (_e136 + _e131));
    let _e146 = (((_e69 * 0.31830987f) / ((_e73 * _e73) + 0.00001f)) * (0.5f / (((_e64 * sqrt((((_e62 * _e62) * _e81) + _e80))) + (_e62 * sqrt((((_e64 * _e64) * _e81) + _e80)))) + 0.00001f)));
    let _e157 = (max(dot(_e55, _e56), 0f) - (_e62 * _e64));
    local = _e157;
    if (_e157 > 0f) {
        local = (_e157 / max(_e64, _e62));
    }
    let _e163 = (1f / (1f + (0.2877934f * 0.2f)));
    let _e166 = local;
    let _e170 = (1f - _e64);
    let _e171 = (_e170 * _e170);
    let _e186 = (1f - _e62);
    let _e187 = (_e186 * _e186);
    let _e204 = (_e163 * (1f + (0.07248821f * 0.2f)));
    let _e205 = (1f - _e204);
    let _e227 = ((1.5f - 1f) / (1.5f + 1f));
    let _e234 = (sign((1.5f - 1f)) * sqrt(clamp((1f * (_e227 * _e227)), 0f, 1f)));
    let _e238 = ((1f + _e234) / max((1f - _e234), 0.0001f));
    let _e241 = ((_e238 - 1f) / (_e238 + 1f));
    let _e243 = (0.3f * 0.3f);
    let _e263 = max(dot(_e50, _e55), 0.001f);
    local_1 = (((mix(vec3((_e146 * mix((0.5f * ((_e135 * _e135) + (_e139 * _e139))), 1f, step(1f, _e128)))), (vec3(_e146) * (clamp(((_e95 + ((vec3<f32>(1f, 1f, 1f) - _e95) * pow(_e97, 5f))) - ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(1f, 1f, 1f)) * (_e96 * pow(_e97, 6f)))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f)) * min(1f, 1f))), vec3(0f)) * _e64) + (((((((_e95 * 0.31830987f) * _e163) * (1f + (0.2f * _e166))) + (((((((_e95 * _e95) * _e204) / (vec3<f32>(1f, 1f, 1f) - (_e95 * max(_e205, 0f)))) * 0.31830987f) * max((1f - ((1f + (0.2f * (((0.05710853f * (1f + _e170)) + ((-0.33218145f + (0.071443f * _e171)) * _e171)) + ((0.49188188f * _e170) * _e171)))) / (1f + (0.2877934f * 0.2f)))), 0f)) * max((1f - ((1f + (0.2f * (((0.05710853f * (1f + _e186)) + ((-0.33218145f + (0.071443f * _e187)) * _e187)) + ((0.49188188f * _e186) * _e187)))) / (1f + (0.2877934f * 0.2f)))), 0f)) / vec3(max(_e205, 0.001f)))) * max(_e64, 0f)) * (1f - 0f)) * (1f - clamp((((_e241 * _e241) * (1f - ((_e243 * max((1f - _e62), _e62)) * 0.31830987f))) + ((_e243 * max(0f, (0.75f - _e62))) * 0.31830987f)), 0f, 1f)))) * vec3<f32>(1f, 0.98f, 0.95f));
    if (0f > 0f) {
        let _e273 = local_1;
        local_1 = mix(_e273, ((vec3<f32>(0.8f, 0.2f, 0.1f) * exp(((log(vec3<f32>(1f, 1f, 1f)) / vec3(max(0f, 0.001f))) * 0f))) * 0f), vec3(0f));
    }
    let _e276 = local_1;
    local_1 = (_e276 + vec3<f32>(0f, 0f, 0f));
    if (0.5f > 0f) {
        let _e281 = ((1.5f - 1f) / (1.5f + 1f));
        let _e289 = ((1.6f - 1f) / (1.6f + 1f));
        let _e290 = (_e289 * _e289);
        let _e303 = clamp(_e263, 0f, 1f);
        let _e307 = ((1f - (_e303 * _e303)) / (1.6f * 1.6f));
        let _e310 = sqrt((1f - min(_e307, 0.9999f)));
        let _e311 = (1.6f * _e310);
        let _e314 = ((_e303 - _e311) / (_e303 + _e311));
        let _e315 = (1.6f * _e303);
        let _e318 = ((_e315 - _e310) / (_e315 + _e310));
        let _e325 = mix(mix((0.5f * ((_e314 * _e314) + (_e318 * _e318))), 1f, step(1f, _e307)), (1f - ((1f - (_e290 + ((1f - _e290) * ((0.0477f + (0.8965f * _e290)) + ((0.0582f * _e290) * _e290))))) / (1.6f * 1.6f))), 0.3f);
        let _e343 = normalize((_e55 + _e56));
        let _e345 = max(dot(_e50, _e343), 0f);
        let _e347 = max(dot(_e50, _e55), 0.001f);
        let _e349 = max(dot(_e50, _e56), 0f);
        let _e352 = max(0.1f, 0.045f);
        let _e353 = (_e352 * _e352);
        let _e354 = (_e353 * _e353);
        let _e358 = (((_e345 * _e345) * (_e354 - 1f)) + 1f);
        let _e365 = (((_e352 * _e352) * _e352) * _e352);
        let _e366 = (1f - _e365);
        let _e380 = clamp(max(dot(_e55, _e343), 0f), 0f, 1f);
        let _e384 = ((1f - (_e380 * _e380)) / (1.6f * 1.6f));
        let _e387 = sqrt((1f - min(_e384, 0.9999f)));
        let _e388 = (1.6f * _e387);
        let _e391 = ((_e380 - _e388) / (_e380 + _e388));
        let _e392 = (1.6f * _e380);
        let _e395 = ((_e392 - _e387) / (_e392 + _e387));
        let _e408 = ((1.6f - 1f) / (1.6f + 1f));
        let _e410 = (0.1f * 0.1f);
        let _e425 = local_1;
        local_1 = (((((_e425 * mix(vec3<f32>(1f, 1f, 1f), (vec3(max((1f - _e325), 0f)) / max((vec3<f32>(1f, 1f, 1f) - ((mix((vec3<f32>(0.8f, 0.2f, 0.1f) * 1f), vec3<f32>(1f, 1f, 1f), vec3(0f)) * (_e281 * _e281)) * _e325)), vec3<f32>(0.001f, 0.001f, 0.001f))), vec3((0.5f * 0.8f)))) * mix(vec3<f32>(1f, 1f, 1f), pow(vec3<f32>(1f, 1f, 1f), vec3((1f / max(_e263, 0.01f)))), vec3(0.5f))) * (1f - (0.5f * clamp((((_e408 * _e408) * (1f - ((_e410 * max((1f - _e263), _e263)) * 0.31830987f))) + ((_e410 * max(0f, (0.75f - _e263))) * 0.31830987f)), 0f, 1f)))) + vec3(((((0.5f * ((_e354 * 0.31830987f) / ((_e358 * _e358) + 0.00001f))) * (0.5f / (((_e349 * sqrt((((_e347 * _e347) * _e366) + _e365))) + (_e347 * sqrt((((_e349 * _e349) * _e366) + _e365)))) + 0.00001f))) * mix((0.5f * ((_e391 * _e391) + (_e395 * _e395))), 1f, step(1f, _e384))) * _e349))) + vec3<f32>(0f, 0f, 0f));
    }
    if (0f > 0f) {
        let _e436 = max(dot(_e50, normalize((_e55 + _e56))), 0f);
        let _e438 = max(dot(_e50, _e56), 0f);
        let _e440 = max(dot(_e50, _e55), 0.001f);
        let _e441 = max(0.5f, 0.045f);
        let _e443 = (1f / (_e441 * _e441));
        let _e469 = local_1;
        local_1 = ((_e469 * (1f - ((0f * max(vec3<f32>(1f, 1f, 1f).x, max(vec3<f32>(1f, 1f, 1f).y, vec3<f32>(1f, 1f, 1f).z))) * 0.3f))) + ((((vec3<f32>(1f, 1f, 1f) * (((2f + _e443) * pow(max((1f - (_e436 * _e436)), 0f), (_e443 * 0.5f))) / 6.2831855f)) * (1f / ((4f * ((_e438 + _e440) - (_e438 * _e440))) + 0.00001f))) * max(_e438, 0f)) * 0f));
    }
    if (0f > 0f) {
        let _e483 = local_1;
        local_1 = (_e483 + (((vec3<f32>(1f, 1f, 1f) * 0f) * 0.01f) * mix(vec3<f32>(1f, 1f, 1f), pow(vec3<f32>(1f, 1f, 1f), vec3((1f / max(_e263, 0.01f)))), vec3(0.5f))));
    }
    let _e486 = local_1;
    color = vec4<f32>(_e486.x, _e486.y, _e486.z, 1f);
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
