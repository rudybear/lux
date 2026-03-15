struct Light {
    light_dir: vec3<f32>,
    view_pos: vec3<f32>,
}

struct Paint {
    base_color: vec3<f32>,
    flake_intensity: f32,
}

var<private> world_pos_1: vec3<f32>;
var<private> world_normal_1: vec3<f32>;
var<private> frag_uv_1: vec2<f32>;
var<private> color: vec4<f32>;
@group(1) @binding(0) 
var<uniform> global: Light;
@group(1) @binding(1) 
var<uniform> global_1: Paint;
@group(1) @binding(2) 
var flake_noise_tex_u002e_sampler: sampler;
@group(1) @binding(3) 
var flake_noise_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(4) 
var env_specular_u002e_sampler: sampler;
@group(1) @binding(5) 
var env_specular_u002e_texture: texture_cube<f32>;
@group(1) @binding(6) 
var env_irradiance_u002e_sampler: sampler;
@group(1) @binding(7) 
var env_irradiance_u002e_texture: texture_cube<f32>;
@group(1) @binding(8) 
var brdf_lut_u002e_sampler: sampler;
@group(1) @binding(9) 
var brdf_lut_u002e_texture: texture_2d<f32>;

fn main_1() {
    var local: vec3<f32>;
    var local_1: vec3<f32>;
    var local_2: f32;
    var local_3: vec3<f32>;

    let _e64 = frag_uv_1;
    let _e65 = world_normal_1;
    let _e66 = normalize(_e65);
    let _e68 = global.view_pos;
    let _e69 = world_pos_1;
    let _e71 = normalize((_e68 - _e69));
    let _e73 = global.light_dir;
    let _e74 = normalize(_e73);
    let _e76 = max(dot(_e66, _e71), 0.001f);
    let _e78 = global_1.base_color;
    let _e80 = textureSample(flake_noise_tex_u002e_texture, flake_noise_tex_u002e_sampler, (_e64 * 8f));
    let _e83 = global_1.flake_intensity;
    let _e85 = (_e78 + (_e80.xyz * _e83));
    let _e87 = normalize((_e71 + _e74));
    let _e89 = max(dot(_e66, _e87), 0f);
    let _e91 = max(dot(_e66, _e71), 0.001f);
    let _e93 = max(dot(_e66, _e74), 0f);
    let _e95 = max(dot(_e71, _e87), 0f);
    let _e96 = max(0.25f, 0.045f);
    let _e97 = (_e96 * _e96);
    let _e98 = (_e97 * _e97);
    let _e102 = (((_e89 * _e89) * (_e98 - 1f)) + 1f);
    let _e109 = (((_e96 * _e96) * _e96) * _e96);
    let _e110 = (1f - _e109);
    let _e124 = (_e85 * 1f);
    let _e125 = clamp(_e95, 0f, 1f);
    let _e126 = (1f - _e125);
    let _e141 = ((1.5f - 1f) / (1.5f + 1f));
    let _e148 = (sign((1.5f - 1f)) * sqrt(clamp((1f * (_e141 * _e141)), 0f, 1f)));
    let _e152 = ((1f + _e148) / max((1f - _e148), 0.0001f));
    let _e153 = clamp(_e95, 0f, 1f);
    let _e157 = ((1f - (_e153 * _e153)) / (_e152 * _e152));
    let _e160 = sqrt((1f - min(_e157, 0.9999f)));
    let _e161 = (_e152 * _e160);
    let _e164 = ((_e153 - _e161) / (_e153 + _e161));
    let _e165 = (_e152 * _e153);
    let _e168 = ((_e165 - _e160) / (_e165 + _e160));
    let _e175 = (((_e98 * 0.31830987f) / ((_e102 * _e102) + 0.00001f)) * (0.5f / (((_e93 * sqrt((((_e91 * _e91) * _e110) + _e109))) + (_e91 * sqrt((((_e93 * _e93) * _e110) + _e109)))) + 0.00001f)));
    let _e186 = (max(dot(_e71, _e74), 0f) - (_e91 * _e93));
    local_2 = _e186;
    if (_e186 > 0f) {
        local_2 = (_e186 / max(_e93, _e91));
    }
    let _e192 = (1f / (1f + (0.2877934f * 0.3f)));
    let _e195 = local_2;
    let _e199 = (1f - _e93);
    let _e200 = (_e199 * _e199);
    let _e215 = (1f - _e91);
    let _e216 = (_e215 * _e215);
    let _e233 = (_e192 * (1f + (0.07248821f * 0.3f)));
    let _e234 = (1f - _e233);
    let _e256 = ((1.5f - 1f) / (1.5f + 1f));
    let _e263 = (sign((1.5f - 1f)) * sqrt(clamp((1f * (_e256 * _e256)), 0f, 1f)));
    let _e267 = ((1f + _e263) / max((1f - _e263), 0.0001f));
    let _e270 = ((_e267 - 1f) / (_e267 + 1f));
    let _e272 = (0.25f * 0.25f);
    local = vec3<f32>(0f, 0f, 0f);
    local_1 = vec3<f32>(0f, 0f, 0f);
    let _e292 = reflect((_e71 * -1f), _e66);
    let _e294 = textureSampleLevel(env_specular_u002e_texture, env_specular_u002e_sampler, _e292, (0.25f * 8f));
    let _e296 = textureSample(env_irradiance_u002e_texture, env_irradiance_u002e_sampler, _e66);
    let _e299 = textureSample(brdf_lut_u002e_texture, brdf_lut_u002e_sampler, vec2<f32>(_e76, 0.25f));
    let _e300 = _e299.xy;
    let _e302 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e85, vec3(0.8f));
    let _e319 = (((_e302 + ((max(vec3((1f - 0.25f)), _e302) - _e302) * pow(clamp((1f - _e76), 0f, 1f), 5f))) * _e300.x) + vec3(_e300.y));
    let _e322 = (_e302 + ((vec3<f32>(1f, 1f, 1f) - _e302) * 0.04761905f));
    let _e323 = (1f - (_e300.x + _e300.y));
    let _e329 = (_e319 + (((_e319 * _e322) / (vec3<f32>(1f, 1f, 1f) - (_e322 * _e323))) * _e323));
    local = (((((vec3<f32>(1f, 1f, 1f) - _e329) * (1f - 0.8f)) * _e85) * _e296.xyz) + (_e329 * _e294.xyz));
    let _e338 = textureSampleLevel(env_specular_u002e_texture, env_specular_u002e_sampler, _e292, (0.05f * 8f));
    local_1 = (_e338.xyz * ((vec3<f32>(0.04f, 0.04f, 0.04f) + ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(0.04f, 0.04f, 0.04f)) * pow(clamp((1f - max(dot(_e66, _e71), 0.001f)), 0f, 1f), 5f))).x * 1f));
    let _e351 = local;
    let _e352 = local_1;
    let _e354 = max(dot(_e66, _e71), 0.001f);
    local_3 = (((mix(vec3((_e175 * mix((0.5f * ((_e164 * _e164) + (_e168 * _e168))), 1f, step(1f, _e157)))), (vec3(_e175) * (clamp(((_e124 + ((vec3<f32>(1f, 1f, 1f) - _e124) * pow(_e126, 5f))) - ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(1f, 1f, 1f)) * (_e125 * pow(_e126, 6f)))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f)) * min(1f, 1f))), vec3(0.8f)) * _e93) + (((((((_e124 * 0.31830987f) * _e192) * (1f + (0.3f * _e195))) + (((((((_e124 * _e124) * _e233) / (vec3<f32>(1f, 1f, 1f) - (_e124 * max(_e234, 0f)))) * 0.31830987f) * max((1f - ((1f + (0.3f * (((0.05710853f * (1f + _e199)) + ((-0.33218145f + (0.071443f * _e200)) * _e200)) + ((0.49188188f * _e199) * _e200)))) / (1f + (0.2877934f * 0.3f)))), 0f)) * max((1f - ((1f + (0.3f * (((0.05710853f * (1f + _e215)) + ((-0.33218145f + (0.071443f * _e216)) * _e216)) + ((0.49188188f * _e215) * _e216)))) / (1f + (0.2877934f * 0.3f)))), 0f)) / vec3(max(_e234, 0.001f)))) * max(_e93, 0f)) * (1f - 0.8f)) * (1f - clamp((((_e270 * _e270) * (1f - ((_e272 * max((1f - _e91), _e91)) * 0.31830987f))) + ((_e272 * max(0f, (0.75f - _e91))) * 0.31830987f)), 0f, 1f)))) * vec3<f32>(1f, 0.98f, 0.95f));
    if (0f > 0f) {
        let _e364 = local_3;
        local_3 = mix(_e364, ((_e85 * exp(((log(vec3<f32>(1f, 1f, 1f)) / vec3(max(0f, 0.001f))) * 0f))) * 0f), vec3(0f));
    }
    let _e367 = local_3;
    local_3 = (_e367 + _e351);
    if (1f > 0f) {
        let _e372 = ((1.5f - 1f) / (1.5f + 1f));
        let _e380 = ((1.6f - 1f) / (1.6f + 1f));
        let _e381 = (_e380 * _e380);
        let _e394 = clamp(_e354, 0f, 1f);
        let _e398 = ((1f - (_e394 * _e394)) / (1.6f * 1.6f));
        let _e401 = sqrt((1f - min(_e398, 0.9999f)));
        let _e402 = (1.6f * _e401);
        let _e405 = ((_e394 - _e402) / (_e394 + _e402));
        let _e406 = (1.6f * _e394);
        let _e409 = ((_e406 - _e401) / (_e406 + _e401));
        let _e416 = mix(mix((0.5f * ((_e405 * _e405) + (_e409 * _e409))), 1f, step(1f, _e398)), (1f - ((1f - (_e381 + ((1f - _e381) * ((0.0477f + (0.8965f * _e381)) + ((0.0582f * _e381) * _e381))))) / (1.6f * 1.6f))), 0.25f);
        let _e434 = normalize((_e71 + _e74));
        let _e436 = max(dot(_e66, _e434), 0f);
        let _e438 = max(dot(_e66, _e71), 0.001f);
        let _e440 = max(dot(_e66, _e74), 0f);
        let _e443 = max(0.05f, 0.045f);
        let _e444 = (_e443 * _e443);
        let _e445 = (_e444 * _e444);
        let _e449 = (((_e436 * _e436) * (_e445 - 1f)) + 1f);
        let _e456 = (((_e443 * _e443) * _e443) * _e443);
        let _e457 = (1f - _e456);
        let _e471 = clamp(max(dot(_e71, _e434), 0f), 0f, 1f);
        let _e475 = ((1f - (_e471 * _e471)) / (1.6f * 1.6f));
        let _e478 = sqrt((1f - min(_e475, 0.9999f)));
        let _e479 = (1.6f * _e478);
        let _e482 = ((_e471 - _e479) / (_e471 + _e479));
        let _e483 = (1.6f * _e471);
        let _e486 = ((_e483 - _e478) / (_e483 + _e478));
        let _e499 = ((1.6f - 1f) / (1.6f + 1f));
        let _e501 = (0.05f * 0.05f);
        let _e516 = local_3;
        local_3 = (((((_e516 * mix(vec3<f32>(1f, 1f, 1f), (vec3(max((1f - _e416), 0f)) / max((vec3<f32>(1f, 1f, 1f) - ((mix((_e85 * 1f), vec3<f32>(1f, 1f, 1f), vec3(0.8f)) * (_e372 * _e372)) * _e416)), vec3<f32>(0.001f, 0.001f, 0.001f))), vec3((1f * 1f)))) * mix(vec3<f32>(1f, 1f, 1f), pow(vec3<f32>(1f, 1f, 1f), vec3((1f / max(_e354, 0.01f)))), vec3(1f))) * (1f - (1f * clamp((((_e499 * _e499) * (1f - ((_e501 * max((1f - _e354), _e354)) * 0.31830987f))) + ((_e501 * max(0f, (0.75f - _e354))) * 0.31830987f)), 0f, 1f)))) + vec3(((((1f * ((_e445 * 0.31830987f) / ((_e449 * _e449) + 0.00001f))) * (0.5f / (((_e440 * sqrt((((_e438 * _e438) * _e457) + _e456))) + (_e438 * sqrt((((_e440 * _e440) * _e457) + _e456)))) + 0.00001f))) * mix((0.5f * ((_e482 * _e482) + (_e486 * _e486))), 1f, step(1f, _e475))) * _e440))) + _e352);
    }
    if (0f > 0f) {
        let _e527 = max(dot(_e66, normalize((_e71 + _e74))), 0f);
        let _e529 = max(dot(_e66, _e74), 0f);
        let _e531 = max(dot(_e66, _e71), 0.001f);
        let _e532 = max(0.5f, 0.045f);
        let _e534 = (1f / (_e532 * _e532));
        let _e560 = local_3;
        local_3 = ((_e560 * (1f - ((0f * max(vec3<f32>(1f, 1f, 1f).x, max(vec3<f32>(1f, 1f, 1f).y, vec3<f32>(1f, 1f, 1f).z))) * 0.3f))) + ((((vec3<f32>(1f, 1f, 1f) * (((2f + _e534) * pow(max((1f - (_e527 * _e527)), 0f), (_e534 * 0.5f))) / 6.2831855f)) * (1f / ((4f * ((_e529 + _e531) - (_e529 * _e531))) + 0.00001f))) * max(_e529, 0f)) * 0f));
    }
    if (0f > 0f) {
        let _e574 = local_3;
        local_3 = (_e574 + (((vec3<f32>(1f, 1f, 1f) * 0f) * 0.01f) * mix(vec3<f32>(1f, 1f, 1f), pow(vec3<f32>(1f, 1f, 1f), vec3((1f / max(_e354, 0.01f)))), vec3(1f))));
    }
    let _e577 = local_3;
    color = vec4<f32>(_e577.x, _e577.y, _e577.z, 1f);
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
