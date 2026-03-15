struct Light {
    light_dir: vec3<f32>,
    view_pos: vec3<f32>,
}

struct Fabric {
    base_color: vec3<f32>,
    fuzz_color: vec3<f32>,
    weave_scale: f32,
}

var<private> world_pos_1: vec3<f32>;
var<private> world_normal_1: vec3<f32>;
var<private> frag_uv_1: vec2<f32>;
var<private> color: vec4<f32>;
@group(1) @binding(0) 
var<uniform> global: Light;
@group(1) @binding(1) 
var<uniform> global_1: Fabric;
@group(1) @binding(2) 
var weave_tex_u002e_sampler: sampler;
@group(1) @binding(3) 
var weave_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(4) 
var normal_tex_u002e_sampler: sampler;
@group(1) @binding(5) 
var normal_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(6) 
var env_specular_u002e_sampler: sampler;
@group(1) @binding(7) 
var env_specular_u002e_texture: texture_cube<f32>;
@group(1) @binding(8) 
var env_irradiance_u002e_sampler: sampler;
@group(1) @binding(9) 
var env_irradiance_u002e_texture: texture_cube<f32>;
@group(1) @binding(10) 
var brdf_lut_u002e_sampler: sampler;
@group(1) @binding(11) 
var brdf_lut_u002e_texture: texture_2d<f32>;

fn main_1() {
    var local: vec3<f32>;
    var local_1: f32;
    var local_2: vec3<f32>;

    let _e65 = frag_uv_1;
    let _e66 = world_normal_1;
    let _e67 = normalize(_e66);
    let _e69 = global.view_pos;
    let _e70 = world_pos_1;
    let _e72 = normalize((_e69 - _e70));
    let _e74 = global.light_dir;
    let _e75 = normalize(_e74);
    let _e77 = max(dot(_e67, _e72), 0.001f);
    let _e79 = global_1.base_color;
    let _e81 = global_1.weave_scale;
    let _e83 = textureSample(weave_tex_u002e_texture, weave_tex_u002e_sampler, (_e65 * _e81));
    let _e85 = (_e79 * _e83.xyz);
    let _e87 = global_1.fuzz_color;
    let _e89 = normalize((_e72 + _e75));
    let _e91 = max(dot(_e67, _e89), 0f);
    let _e93 = max(dot(_e67, _e72), 0.001f);
    let _e95 = max(dot(_e67, _e75), 0f);
    let _e97 = max(dot(_e72, _e89), 0f);
    let _e98 = max(0.6f, 0.045f);
    let _e99 = (_e98 * _e98);
    let _e100 = (_e99 * _e99);
    let _e104 = (((_e91 * _e91) * (_e100 - 1f)) + 1f);
    let _e111 = (((_e98 * _e98) * _e98) * _e98);
    let _e112 = (1f - _e111);
    let _e126 = (_e85 * 1f);
    let _e127 = clamp(_e97, 0f, 1f);
    let _e128 = (1f - _e127);
    let _e143 = ((1.5f - 1f) / (1.5f + 1f));
    let _e150 = (sign((1.5f - 1f)) * sqrt(clamp((0.3f * (_e143 * _e143)), 0f, 1f)));
    let _e154 = ((1f + _e150) / max((1f - _e150), 0.0001f));
    let _e155 = clamp(_e97, 0f, 1f);
    let _e159 = ((1f - (_e155 * _e155)) / (_e154 * _e154));
    let _e162 = sqrt((1f - min(_e159, 0.9999f)));
    let _e163 = (_e154 * _e162);
    let _e166 = ((_e155 - _e163) / (_e155 + _e163));
    let _e167 = (_e154 * _e155);
    let _e170 = ((_e167 - _e162) / (_e167 + _e162));
    let _e177 = (((_e100 * 0.31830987f) / ((_e104 * _e104) + 0.00001f)) * (0.5f / (((_e95 * sqrt((((_e93 * _e93) * _e112) + _e111))) + (_e93 * sqrt((((_e95 * _e95) * _e112) + _e111)))) + 0.00001f)));
    let _e188 = (max(dot(_e72, _e75), 0f) - (_e93 * _e95));
    local_1 = _e188;
    if (_e188 > 0f) {
        local_1 = (_e188 / max(_e95, _e93));
    }
    let _e194 = (1f / (1f + (0.2877934f * 0.5f)));
    let _e197 = local_1;
    let _e201 = (1f - _e95);
    let _e202 = (_e201 * _e201);
    let _e217 = (1f - _e93);
    let _e218 = (_e217 * _e217);
    let _e235 = (_e194 * (1f + (0.07248821f * 0.5f)));
    let _e236 = (1f - _e235);
    let _e258 = ((1.5f - 1f) / (1.5f + 1f));
    let _e265 = (sign((1.5f - 1f)) * sqrt(clamp((0.3f * (_e258 * _e258)), 0f, 1f)));
    let _e269 = ((1f + _e265) / max((1f - _e265), 0.0001f));
    let _e272 = ((_e269 - 1f) / (_e269 + 1f));
    let _e274 = (0.6f * 0.6f);
    local = vec3<f32>(0f, 0f, 0f);
    let _e296 = textureSampleLevel(env_specular_u002e_texture, env_specular_u002e_sampler, reflect((_e72 * -1f), _e67), (0.6f * 8f));
    let _e298 = textureSample(env_irradiance_u002e_texture, env_irradiance_u002e_sampler, _e67);
    let _e301 = textureSample(brdf_lut_u002e_texture, brdf_lut_u002e_sampler, vec2<f32>(_e77, 0.6f));
    let _e302 = _e301.xy;
    let _e304 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e85, vec3(0f));
    let _e321 = (((_e304 + ((max(vec3((1f - 0.6f)), _e304) - _e304) * pow(clamp((1f - _e77), 0f, 1f), 5f))) * _e302.x) + vec3(_e302.y));
    let _e324 = (_e304 + ((vec3<f32>(1f, 1f, 1f) - _e304) * 0.04761905f));
    let _e325 = (1f - (_e302.x + _e302.y));
    let _e331 = (_e321 + (((_e321 * _e324) / (vec3<f32>(1f, 1f, 1f) - (_e324 * _e325))) * _e325));
    local = (((((vec3<f32>(1f, 1f, 1f) - _e331) * (1f - 0f)) * _e85) * _e298.xyz) + (_e331 * _e296.xyz));
    let _e339 = local;
    let _e341 = max(dot(_e67, _e72), 0.001f);
    local_2 = (((mix(vec3((_e177 * mix((0.5f * ((_e166 * _e166) + (_e170 * _e170))), 1f, step(1f, _e159)))), (vec3(_e177) * (clamp(((_e126 + ((vec3<f32>(1f, 1f, 1f) - _e126) * pow(_e128, 5f))) - ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(1f, 1f, 1f)) * (_e127 * pow(_e128, 6f)))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f)) * min(0.3f, 1f))), vec3(0f)) * _e95) + (((((((_e126 * 0.31830987f) * _e194) * (1f + (0.5f * _e197))) + (((((((_e126 * _e126) * _e235) / (vec3<f32>(1f, 1f, 1f) - (_e126 * max(_e236, 0f)))) * 0.31830987f) * max((1f - ((1f + (0.5f * (((0.05710853f * (1f + _e201)) + ((-0.33218145f + (0.071443f * _e202)) * _e202)) + ((0.49188188f * _e201) * _e202)))) / (1f + (0.2877934f * 0.5f)))), 0f)) * max((1f - ((1f + (0.5f * (((0.05710853f * (1f + _e217)) + ((-0.33218145f + (0.071443f * _e218)) * _e218)) + ((0.49188188f * _e217) * _e218)))) / (1f + (0.2877934f * 0.5f)))), 0f)) / vec3(max(_e236, 0.001f)))) * max(_e95, 0f)) * (1f - 0f)) * (1f - clamp((((_e272 * _e272) * (1f - ((_e274 * max((1f - _e93), _e93)) * 0.31830987f))) + ((_e274 * max(0f, (0.75f - _e93))) * 0.31830987f)), 0f, 1f)))) * vec3<f32>(1f, 0.95f, 0.9f));
    if (0f > 0f) {
        let _e351 = local_2;
        local_2 = mix(_e351, ((_e85 * exp(((log(vec3<f32>(1f, 1f, 1f)) / vec3(max(0f, 0.001f))) * 0f))) * 0f), vec3(0f));
    }
    let _e354 = local_2;
    local_2 = (_e354 + _e339);
    if (0f > 0f) {
        let _e359 = ((1.5f - 1f) / (1.5f + 1f));
        let _e367 = ((1.6f - 1f) / (1.6f + 1f));
        let _e368 = (_e367 * _e367);
        let _e381 = clamp(_e341, 0f, 1f);
        let _e385 = ((1f - (_e381 * _e381)) / (1.6f * 1.6f));
        let _e388 = sqrt((1f - min(_e385, 0.9999f)));
        let _e389 = (1.6f * _e388);
        let _e392 = ((_e381 - _e389) / (_e381 + _e389));
        let _e393 = (1.6f * _e381);
        let _e396 = ((_e393 - _e388) / (_e393 + _e388));
        let _e403 = mix(mix((0.5f * ((_e392 * _e392) + (_e396 * _e396))), 1f, step(1f, _e385)), (1f - ((1f - (_e368 + ((1f - _e368) * ((0.0477f + (0.8965f * _e368)) + ((0.0582f * _e368) * _e368))))) / (1.6f * 1.6f))), 0.6f);
        let _e421 = normalize((_e72 + _e75));
        let _e423 = max(dot(_e67, _e421), 0f);
        let _e425 = max(dot(_e67, _e72), 0.001f);
        let _e427 = max(dot(_e67, _e75), 0f);
        let _e430 = max(0f, 0.045f);
        let _e431 = (_e430 * _e430);
        let _e432 = (_e431 * _e431);
        let _e436 = (((_e423 * _e423) * (_e432 - 1f)) + 1f);
        let _e443 = (((_e430 * _e430) * _e430) * _e430);
        let _e444 = (1f - _e443);
        let _e458 = clamp(max(dot(_e72, _e421), 0f), 0f, 1f);
        let _e462 = ((1f - (_e458 * _e458)) / (1.6f * 1.6f));
        let _e465 = sqrt((1f - min(_e462, 0.9999f)));
        let _e466 = (1.6f * _e465);
        let _e469 = ((_e458 - _e466) / (_e458 + _e466));
        let _e470 = (1.6f * _e458);
        let _e473 = ((_e470 - _e465) / (_e470 + _e465));
        let _e486 = ((1.6f - 1f) / (1.6f + 1f));
        let _e488 = (0f * 0f);
        let _e503 = local_2;
        local_2 = (((((_e503 * mix(vec3<f32>(1f, 1f, 1f), (vec3(max((1f - _e403), 0f)) / max((vec3<f32>(1f, 1f, 1f) - ((mix((_e85 * 1f), vec3<f32>(1f, 1f, 1f), vec3(0f)) * (_e359 * _e359)) * _e403)), vec3<f32>(0.001f, 0.001f, 0.001f))), vec3((0f * 1f)))) * mix(vec3<f32>(1f, 1f, 1f), pow(vec3<f32>(1f, 1f, 1f), vec3((1f / max(_e341, 0.01f)))), vec3(0f))) * (1f - (0f * clamp((((_e486 * _e486) * (1f - ((_e488 * max((1f - _e341), _e341)) * 0.31830987f))) + ((_e488 * max(0f, (0.75f - _e341))) * 0.31830987f)), 0f, 1f)))) + vec3(((((0f * ((_e432 * 0.31830987f) / ((_e436 * _e436) + 0.00001f))) * (0.5f / (((_e427 * sqrt((((_e425 * _e425) * _e444) + _e443))) + (_e425 * sqrt((((_e427 * _e427) * _e444) + _e443)))) + 0.00001f))) * mix((0.5f * ((_e469 * _e469) + (_e473 * _e473))), 1f, step(1f, _e462))) * _e427))) + vec3<f32>(0f, 0f, 0f));
    }
    if (0.8f > 0f) {
        let _e514 = max(dot(_e67, normalize((_e72 + _e75))), 0f);
        let _e516 = max(dot(_e67, _e75), 0f);
        let _e518 = max(dot(_e67, _e72), 0.001f);
        let _e519 = max(0.7f, 0.045f);
        let _e521 = (1f / (_e519 * _e519));
        let _e547 = local_2;
        local_2 = ((_e547 * (1f - ((0.8f * max(_e87.x, max(_e87.y, _e87.z))) * 0.3f))) + ((((_e87 * (((2f + _e521) * pow(max((1f - (_e514 * _e514)), 0f), (_e521 * 0.5f))) / 6.2831855f)) * (1f / ((4f * ((_e516 + _e518) - (_e516 * _e518))) + 0.00001f))) * max(_e516, 0f)) * 0.8f));
    }
    if (0f > 0f) {
        let _e561 = local_2;
        local_2 = (_e561 + (((vec3<f32>(1f, 1f, 1f) * 0f) * 0.01f) * mix(vec3<f32>(1f, 1f, 1f), pow(vec3<f32>(1f, 1f, 1f), vec3((1f / max(_e341, 0.01f)))), vec3(0f))));
    }
    let _e564 = local_2;
    color = vec4<f32>(_e564.x, _e564.y, _e564.z, 1f);
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
