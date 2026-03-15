struct Light {
    light_dir: vec3<f32>,
    view_pos: vec3<f32>,
}

struct Glass {
    base_tint: vec3<f32>,
    volume_color: vec3<f32>,
    volume_depth: f32,
    glass_ior: f32,
}

var<private> world_pos_1: vec3<f32>;
var<private> world_normal_1: vec3<f32>;
var<private> frag_uv_1: vec2<f32>;
var<private> color: vec4<f32>;
@group(1) @binding(0) 
var<uniform> global: Light;
@group(1) @binding(1) 
var<uniform> global_1: Glass;
@group(1) @binding(2) 
var env_specular_u002e_sampler: sampler;
@group(1) @binding(3) 
var env_specular_u002e_texture: texture_cube<f32>;
@group(1) @binding(4) 
var env_irradiance_u002e_sampler: sampler;
@group(1) @binding(5) 
var env_irradiance_u002e_texture: texture_cube<f32>;
@group(1) @binding(6) 
var brdf_lut_u002e_sampler: sampler;
@group(1) @binding(7) 
var brdf_lut_u002e_texture: texture_2d<f32>;

fn main_1() {
    var local: vec3<f32>;
    var local_1: f32;
    var local_2: vec3<f32>;

    let _e59 = world_normal_1;
    let _e60 = normalize(_e59);
    let _e62 = global.view_pos;
    let _e63 = world_pos_1;
    let _e65 = normalize((_e62 - _e63));
    let _e67 = global.light_dir;
    let _e68 = normalize(_e67);
    let _e70 = max(dot(_e60, _e65), 0.001f);
    let _e72 = global_1.base_tint;
    let _e74 = global_1.glass_ior;
    let _e76 = global_1.volume_color;
    let _e78 = global_1.volume_depth;
    let _e80 = normalize((_e65 + _e68));
    let _e82 = max(dot(_e60, _e80), 0f);
    let _e84 = max(dot(_e60, _e65), 0.001f);
    let _e86 = max(dot(_e60, _e68), 0f);
    let _e88 = max(dot(_e65, _e80), 0f);
    let _e89 = max(0.02f, 0.045f);
    let _e90 = (_e89 * _e89);
    let _e91 = (_e90 * _e90);
    let _e95 = (((_e82 * _e82) * (_e91 - 1f)) + 1f);
    let _e102 = (((_e89 * _e89) * _e89) * _e89);
    let _e103 = (1f - _e102);
    let _e117 = (_e72 * 1f);
    let _e118 = clamp(_e88, 0f, 1f);
    let _e119 = (1f - _e118);
    let _e134 = ((_e74 - 1f) / (_e74 + 1f));
    let _e141 = (sign((_e74 - 1f)) * sqrt(clamp((1f * (_e134 * _e134)), 0f, 1f)));
    let _e145 = ((1f + _e141) / max((1f - _e141), 0.0001f));
    let _e146 = clamp(_e88, 0f, 1f);
    let _e150 = ((1f - (_e146 * _e146)) / (_e145 * _e145));
    let _e153 = sqrt((1f - min(_e150, 0.9999f)));
    let _e154 = (_e145 * _e153);
    let _e157 = ((_e146 - _e154) / (_e146 + _e154));
    let _e158 = (_e145 * _e146);
    let _e161 = ((_e158 - _e153) / (_e158 + _e153));
    let _e168 = (((_e91 * 0.31830987f) / ((_e95 * _e95) + 0.00001f)) * (0.5f / (((_e86 * sqrt((((_e84 * _e84) * _e103) + _e102))) + (_e84 * sqrt((((_e86 * _e86) * _e103) + _e102)))) + 0.00001f)));
    let _e179 = (max(dot(_e65, _e68), 0f) - (_e84 * _e86));
    local_1 = _e179;
    if (_e179 > 0f) {
        local_1 = (_e179 / max(_e86, _e84));
    }
    let _e185 = (1f / (1f + (0.2877934f * 0f)));
    let _e188 = local_1;
    let _e192 = (1f - _e86);
    let _e193 = (_e192 * _e192);
    let _e208 = (1f - _e84);
    let _e209 = (_e208 * _e208);
    let _e226 = (_e185 * (1f + (0.07248821f * 0f)));
    let _e227 = (1f - _e226);
    let _e249 = ((_e74 - 1f) / (_e74 + 1f));
    let _e256 = (sign((_e74 - 1f)) * sqrt(clamp((1f * (_e249 * _e249)), 0f, 1f)));
    let _e260 = ((1f + _e256) / max((1f - _e256), 0.0001f));
    let _e263 = ((_e260 - 1f) / (_e260 + 1f));
    let _e265 = (0.02f * 0.02f);
    local = vec3<f32>(0f, 0f, 0f);
    let _e287 = textureSampleLevel(env_specular_u002e_texture, env_specular_u002e_sampler, reflect((_e65 * -1f), _e60), (0.02f * 8f));
    let _e289 = textureSample(env_irradiance_u002e_texture, env_irradiance_u002e_sampler, _e60);
    let _e292 = textureSample(brdf_lut_u002e_texture, brdf_lut_u002e_sampler, vec2<f32>(_e70, 0.02f));
    let _e293 = _e292.xy;
    let _e295 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e72, vec3(0f));
    let _e312 = (((_e295 + ((max(vec3((1f - 0.02f)), _e295) - _e295) * pow(clamp((1f - _e70), 0f, 1f), 5f))) * _e293.x) + vec3(_e293.y));
    let _e315 = (_e295 + ((vec3<f32>(1f, 1f, 1f) - _e295) * 0.04761905f));
    let _e316 = (1f - (_e293.x + _e293.y));
    let _e322 = (_e312 + (((_e312 * _e315) / (vec3<f32>(1f, 1f, 1f) - (_e315 * _e316))) * _e316));
    local = (((((vec3<f32>(1f, 1f, 1f) - _e322) * (1f - 0f)) * _e72) * _e289.xyz) + (_e322 * _e287.xyz));
    let _e330 = local;
    let _e332 = max(dot(_e60, _e65), 0.001f);
    local_2 = (((mix(vec3((_e168 * mix((0.5f * ((_e157 * _e157) + (_e161 * _e161))), 1f, step(1f, _e150)))), (vec3(_e168) * (clamp(((_e117 + ((vec3<f32>(1f, 1f, 1f) - _e117) * pow(_e119, 5f))) - ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(1f, 1f, 1f)) * (_e118 * pow(_e119, 6f)))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f)) * min(1f, 1f))), vec3(0f)) * _e86) + (((((((_e117 * 0.31830987f) * _e185) * (1f + (0f * _e188))) + (((((((_e117 * _e117) * _e226) / (vec3<f32>(1f, 1f, 1f) - (_e117 * max(_e227, 0f)))) * 0.31830987f) * max((1f - ((1f + (0f * (((0.05710853f * (1f + _e192)) + ((-0.33218145f + (0.071443f * _e193)) * _e193)) + ((0.49188188f * _e192) * _e193)))) / (1f + (0.2877934f * 0f)))), 0f)) * max((1f - ((1f + (0f * (((0.05710853f * (1f + _e208)) + ((-0.33218145f + (0.071443f * _e209)) * _e209)) + ((0.49188188f * _e208) * _e209)))) / (1f + (0.2877934f * 0f)))), 0f)) / vec3(max(_e227, 0.001f)))) * max(_e86, 0f)) * (1f - 0f)) * (1f - clamp((((_e263 * _e263) * (1f - ((_e265 * max((1f - _e84), _e84)) * 0.31830987f))) + ((_e265 * max(0f, (0.75f - _e84))) * 0.31830987f)), 0f, 1f)))) * vec3<f32>(1f, 0.98f, 0.95f));
    if (0.95f > 0f) {
        let _e342 = local_2;
        local_2 = mix(_e342, ((_e72 * exp(((log(_e76) / vec3(max(_e78, 0.001f))) * _e78))) * 0.95f), vec3(0.95f));
    }
    let _e345 = local_2;
    local_2 = (_e345 + _e330);
    if (0f > 0f) {
        let _e350 = ((_e74 - 1f) / (_e74 + 1f));
        let _e358 = ((1.6f - 1f) / (1.6f + 1f));
        let _e359 = (_e358 * _e358);
        let _e372 = clamp(_e332, 0f, 1f);
        let _e376 = ((1f - (_e372 * _e372)) / (1.6f * 1.6f));
        let _e379 = sqrt((1f - min(_e376, 0.9999f)));
        let _e380 = (1.6f * _e379);
        let _e383 = ((_e372 - _e380) / (_e372 + _e380));
        let _e384 = (1.6f * _e372);
        let _e387 = ((_e384 - _e379) / (_e384 + _e379));
        let _e394 = mix(mix((0.5f * ((_e383 * _e383) + (_e387 * _e387))), 1f, step(1f, _e376)), (1f - ((1f - (_e359 + ((1f - _e359) * ((0.0477f + (0.8965f * _e359)) + ((0.0582f * _e359) * _e359))))) / (1.6f * 1.6f))), 0.02f);
        let _e412 = normalize((_e65 + _e68));
        let _e414 = max(dot(_e60, _e412), 0f);
        let _e416 = max(dot(_e60, _e65), 0.001f);
        let _e418 = max(dot(_e60, _e68), 0f);
        let _e421 = max(0f, 0.045f);
        let _e422 = (_e421 * _e421);
        let _e423 = (_e422 * _e422);
        let _e427 = (((_e414 * _e414) * (_e423 - 1f)) + 1f);
        let _e434 = (((_e421 * _e421) * _e421) * _e421);
        let _e435 = (1f - _e434);
        let _e449 = clamp(max(dot(_e65, _e412), 0f), 0f, 1f);
        let _e453 = ((1f - (_e449 * _e449)) / (1.6f * 1.6f));
        let _e456 = sqrt((1f - min(_e453, 0.9999f)));
        let _e457 = (1.6f * _e456);
        let _e460 = ((_e449 - _e457) / (_e449 + _e457));
        let _e461 = (1.6f * _e449);
        let _e464 = ((_e461 - _e456) / (_e461 + _e456));
        let _e477 = ((1.6f - 1f) / (1.6f + 1f));
        let _e479 = (0f * 0f);
        let _e494 = local_2;
        local_2 = (((((_e494 * mix(vec3<f32>(1f, 1f, 1f), (vec3(max((1f - _e394), 0f)) / max((vec3<f32>(1f, 1f, 1f) - ((mix((_e72 * 1f), vec3<f32>(1f, 1f, 1f), vec3(0f)) * (_e350 * _e350)) * _e394)), vec3<f32>(0.001f, 0.001f, 0.001f))), vec3((0f * 1f)))) * mix(vec3<f32>(1f, 1f, 1f), pow(vec3<f32>(1f, 1f, 1f), vec3((1f / max(_e332, 0.01f)))), vec3(0f))) * (1f - (0f * clamp((((_e477 * _e477) * (1f - ((_e479 * max((1f - _e332), _e332)) * 0.31830987f))) + ((_e479 * max(0f, (0.75f - _e332))) * 0.31830987f)), 0f, 1f)))) + vec3(((((0f * ((_e423 * 0.31830987f) / ((_e427 * _e427) + 0.00001f))) * (0.5f / (((_e418 * sqrt((((_e416 * _e416) * _e435) + _e434))) + (_e416 * sqrt((((_e418 * _e418) * _e435) + _e434)))) + 0.00001f))) * mix((0.5f * ((_e460 * _e460) + (_e464 * _e464))), 1f, step(1f, _e453))) * _e418))) + vec3<f32>(0f, 0f, 0f));
    }
    if (0f > 0f) {
        let _e505 = max(dot(_e60, normalize((_e65 + _e68))), 0f);
        let _e507 = max(dot(_e60, _e68), 0f);
        let _e509 = max(dot(_e60, _e65), 0.001f);
        let _e510 = max(0.5f, 0.045f);
        let _e512 = (1f / (_e510 * _e510));
        let _e538 = local_2;
        local_2 = ((_e538 * (1f - ((0f * max(vec3<f32>(1f, 1f, 1f).x, max(vec3<f32>(1f, 1f, 1f).y, vec3<f32>(1f, 1f, 1f).z))) * 0.3f))) + ((((vec3<f32>(1f, 1f, 1f) * (((2f + _e512) * pow(max((1f - (_e505 * _e505)), 0f), (_e512 * 0.5f))) / 6.2831855f)) * (1f / ((4f * ((_e507 + _e509) - (_e507 * _e509))) + 0.00001f))) * max(_e507, 0f)) * 0f));
    }
    if (0f > 0f) {
        let _e552 = local_2;
        local_2 = (_e552 + (((vec3<f32>(0f, 0f, 0f) * 0f) * 0.01f) * mix(vec3<f32>(1f, 1f, 1f), pow(vec3<f32>(1f, 1f, 1f), vec3((1f / max(_e332, 0.01f)))), vec3(0f))));
    }
    let _e555 = local_2;
    color = vec4<f32>(_e555.x, _e555.y, _e555.z, 1f);
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
