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

    let _e50 = world_normal_1;
    let _e51 = normalize(_e50);
    let _e53 = global.view_pos;
    let _e54 = world_pos_1;
    let _e56 = normalize((_e53 - _e54));
    let _e57 = normalize(vec3<f32>(0.5f, -0.7f, 0.5f));
    let _e59 = normalize((_e56 + _e57));
    let _e61 = max(dot(_e51, _e59), 0f);
    let _e63 = max(dot(_e51, _e56), 0.001f);
    let _e65 = max(dot(_e51, _e57), 0f);
    let _e67 = max(dot(_e56, _e59), 0f);
    let _e68 = max(0.3f, 0.045f);
    let _e69 = (_e68 * _e68);
    let _e70 = (_e69 * _e69);
    let _e74 = (((_e61 * _e61) * (_e70 - 1f)) + 1f);
    let _e81 = (((_e68 * _e68) * _e68) * _e68);
    let _e82 = (1f - _e81);
    let _e96 = (vec3<f32>(0.1f, 0.6f, 0.9f) * 1f);
    let _e97 = clamp(_e67, 0f, 1f);
    let _e98 = (1f - _e97);
    let _e113 = ((1.6f - 1f) / (1.6f + 1f));
    let _e120 = (sign((1.6f - 1f)) * sqrt(clamp((1f * (_e113 * _e113)), 0f, 1f)));
    let _e124 = ((1f + _e120) / max((1f - _e120), 0.0001f));
    let _e125 = clamp(_e67, 0f, 1f);
    let _e129 = ((1f - (_e125 * _e125)) / (_e124 * _e124));
    let _e132 = sqrt((1f - min(_e129, 0.9999f)));
    let _e133 = (_e124 * _e132);
    let _e136 = ((_e125 - _e133) / (_e125 + _e133));
    let _e137 = (_e124 * _e125);
    let _e140 = ((_e137 - _e132) / (_e137 + _e132));
    let _e147 = (((_e70 * 0.31830987f) / ((_e74 * _e74) + 0.00001f)) * (0.5f / (((_e65 * sqrt((((_e63 * _e63) * _e82) + _e81))) + (_e63 * sqrt((((_e65 * _e65) * _e82) + _e81)))) + 0.00001f)));
    let _e158 = (max(dot(_e56, _e57), 0f) - (_e63 * _e65));
    local = _e158;
    if (_e158 > 0f) {
        local = (_e158 / max(_e65, _e63));
    }
    let _e164 = (1f / (1f + (0.2877934f * 0f)));
    let _e167 = local;
    let _e171 = (1f - _e65);
    let _e172 = (_e171 * _e171);
    let _e187 = (1f - _e63);
    let _e188 = (_e187 * _e187);
    let _e205 = (_e164 * (1f + (0.07248821f * 0f)));
    let _e206 = (1f - _e205);
    let _e228 = ((1.6f - 1f) / (1.6f + 1f));
    let _e235 = (sign((1.6f - 1f)) * sqrt(clamp((1f * (_e228 * _e228)), 0f, 1f)));
    let _e239 = ((1f + _e235) / max((1f - _e235), 0.0001f));
    let _e242 = ((_e239 - 1f) / (_e239 + 1f));
    let _e244 = (0.3f * 0.3f);
    let _e264 = max(dot(_e51, _e56), 0.001f);
    local_1 = (((mix(vec3((_e147 * mix((0.5f * ((_e136 * _e136) + (_e140 * _e140))), 1f, step(1f, _e129)))), (vec3(_e147) * (clamp(((_e96 + ((vec3<f32>(1f, 1f, 1f) - _e96) * pow(_e98, 5f))) - ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(1f, 1f, 1f)) * (_e97 * pow(_e98, 6f)))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f)) * min(1f, 1f))), vec3(0f)) * _e65) + (((((((_e96 * 0.31830987f) * _e164) * (1f + (0f * _e167))) + (((((((_e96 * _e96) * _e205) / (vec3<f32>(1f, 1f, 1f) - (_e96 * max(_e206, 0f)))) * 0.31830987f) * max((1f - ((1f + (0f * (((0.05710853f * (1f + _e171)) + ((-0.33218145f + (0.071443f * _e172)) * _e172)) + ((0.49188188f * _e171) * _e172)))) / (1f + (0.2877934f * 0f)))), 0f)) * max((1f - ((1f + (0f * (((0.05710853f * (1f + _e187)) + ((-0.33218145f + (0.071443f * _e188)) * _e188)) + ((0.49188188f * _e187) * _e188)))) / (1f + (0.2877934f * 0f)))), 0f)) / vec3(max(_e206, 0.001f)))) * max(_e65, 0f)) * (1f - 0f)) * (1f - clamp((((_e242 * _e242) * (1f - ((_e244 * max((1f - _e63), _e63)) * 0.31830987f))) + ((_e244 * max(0f, (0.75f - _e63))) * 0.31830987f)), 0f, 1f)))) * vec3<f32>(3f, 2.94f, 2.85f));
    if (0f > 0f) {
        let _e274 = local_1;
        local_1 = mix(_e274, ((vec3<f32>(0.1f, 0.6f, 0.9f) * exp(((log(vec3<f32>(1f, 1f, 1f)) / vec3(max(0f, 0.001f))) * 0f))) * 0f), vec3(0f));
    }
    let _e277 = local_1;
    local_1 = (_e277 + vec3<f32>(0f, 0f, 0f));
    if (1f > 0f) {
        let _e282 = ((1.6f - 1f) / (1.6f + 1f));
        let _e290 = ((1.6f - 1f) / (1.6f + 1f));
        let _e291 = (_e290 * _e290);
        let _e304 = clamp(_e264, 0f, 1f);
        let _e308 = ((1f - (_e304 * _e304)) / (1.6f * 1.6f));
        let _e311 = sqrt((1f - min(_e308, 0.9999f)));
        let _e312 = (1.6f * _e311);
        let _e315 = ((_e304 - _e312) / (_e304 + _e312));
        let _e316 = (1.6f * _e304);
        let _e319 = ((_e316 - _e311) / (_e316 + _e311));
        let _e326 = mix(mix((0.5f * ((_e315 * _e315) + (_e319 * _e319))), 1f, step(1f, _e308)), (1f - ((1f - (_e291 + ((1f - _e291) * ((0.0477f + (0.8965f * _e291)) + ((0.0582f * _e291) * _e291))))) / (1.6f * 1.6f))), 0.3f);
        let _e344 = normalize((_e56 + _e57));
        let _e346 = max(dot(_e51, _e344), 0f);
        let _e348 = max(dot(_e51, _e56), 0.001f);
        let _e350 = max(dot(_e51, _e57), 0f);
        let _e353 = max(0.02f, 0.045f);
        let _e354 = (_e353 * _e353);
        let _e355 = (_e354 * _e354);
        let _e359 = (((_e346 * _e346) * (_e355 - 1f)) + 1f);
        let _e366 = (((_e353 * _e353) * _e353) * _e353);
        let _e367 = (1f - _e366);
        let _e381 = clamp(max(dot(_e56, _e344), 0f), 0f, 1f);
        let _e385 = ((1f - (_e381 * _e381)) / (1.6f * 1.6f));
        let _e388 = sqrt((1f - min(_e385, 0.9999f)));
        let _e389 = (1.6f * _e388);
        let _e392 = ((_e381 - _e389) / (_e381 + _e389));
        let _e393 = (1.6f * _e381);
        let _e396 = ((_e393 - _e388) / (_e393 + _e388));
        let _e409 = ((1.6f - 1f) / (1.6f + 1f));
        let _e411 = (0.02f * 0.02f);
        let _e426 = local_1;
        local_1 = (((((_e426 * mix(vec3<f32>(1f, 1f, 1f), (vec3(max((1f - _e326), 0f)) / max((vec3<f32>(1f, 1f, 1f) - ((mix((vec3<f32>(0.1f, 0.6f, 0.9f) * 1f), vec3<f32>(1f, 1f, 1f), vec3(0f)) * (_e282 * _e282)) * _e326)), vec3<f32>(0.001f, 0.001f, 0.001f))), vec3((1f * 1f)))) * mix(vec3<f32>(1f, 1f, 1f), pow(vec3<f32>(1f, 1f, 1f), vec3((1f / max(_e264, 0.01f)))), vec3(1f))) * (1f - (1f * clamp((((_e409 * _e409) * (1f - ((_e411 * max((1f - _e264), _e264)) * 0.31830987f))) + ((_e411 * max(0f, (0.75f - _e264))) * 0.31830987f)), 0f, 1f)))) + vec3(((((1f * ((_e355 * 0.31830987f) / ((_e359 * _e359) + 0.00001f))) * (0.5f / (((_e350 * sqrt((((_e348 * _e348) * _e367) + _e366))) + (_e348 * sqrt((((_e350 * _e350) * _e367) + _e366)))) + 0.00001f))) * mix((0.5f * ((_e392 * _e392) + (_e396 * _e396))), 1f, step(1f, _e385))) * _e350))) + vec3<f32>(0f, 0f, 0f));
    }
    if (0f > 0f) {
        let _e437 = max(dot(_e51, normalize((_e56 + _e57))), 0f);
        let _e439 = max(dot(_e51, _e57), 0f);
        let _e441 = max(dot(_e51, _e56), 0.001f);
        let _e442 = max(0.5f, 0.045f);
        let _e444 = (1f / (_e442 * _e442));
        let _e470 = local_1;
        local_1 = ((_e470 * (1f - ((0f * max(vec3<f32>(1f, 1f, 1f).x, max(vec3<f32>(1f, 1f, 1f).y, vec3<f32>(1f, 1f, 1f).z))) * 0.3f))) + ((((vec3<f32>(1f, 1f, 1f) * (((2f + _e444) * pow(max((1f - (_e437 * _e437)), 0f), (_e444 * 0.5f))) / 6.2831855f)) * (1f / ((4f * ((_e439 + _e441) - (_e439 * _e441))) + 0.00001f))) * max(_e439, 0f)) * 0f));
    }
    if (0f > 0f) {
        let _e484 = local_1;
        local_1 = (_e484 + (((vec3<f32>(1f, 1f, 1f) * 0f) * 0.01f) * mix(vec3<f32>(1f, 1f, 1f), pow(vec3<f32>(1f, 1f, 1f), vec3((1f / max(_e264, 0.01f)))), vec3(1f))));
    }
    let _e487 = local_1;
    color = vec4<f32>(_e487.x, _e487.y, _e487.z, 1f);
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
