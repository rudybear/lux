struct DeferredCamera {
    inv_view_proj: mat4x4<f32>,
    view_pos: vec3<f32>,
}

struct SceneLight {
    view_pos: vec3<f32>,
    light_count: i32,
}

struct type_11 {
    member: f32,
    member_1: f32,
    member_2: f32,
    member_3: f32,
    member_4: vec3<f32>,
    member_5: f32,
    member_6: vec3<f32>,
    member_7: f32,
    member_8: vec3<f32>,
    member_9: f32,
}

struct type_13 {
    member: array<type_11>,
}

var<private> frag_uv_1: vec2<f32>;
var<private> color: vec4<f32>;
@group(0) @binding(0) 
var<uniform> global: DeferredCamera;
@group(0) @binding(1) 
var<uniform> global_1: SceneLight;
@group(0) @binding(2) 
var gbuf_tex0_u002e_sampler: sampler;
@group(0) @binding(3) 
var gbuf_tex0_u002e_texture: texture_2d<f32>;
@group(0) @binding(4) 
var gbuf_tex1_u002e_sampler: sampler;
@group(0) @binding(5) 
var gbuf_tex1_u002e_texture: texture_2d<f32>;
@group(0) @binding(6) 
var gbuf_tex2_u002e_sampler: sampler;
@group(0) @binding(7) 
var gbuf_tex2_u002e_texture: texture_2d<f32>;
@group(0) @binding(8) 
var gbuf_depth_u002e_sampler: sampler;
@group(0) @binding(9) 
var gbuf_depth_u002e_texture: texture_2d<f32>;
@group(0) @binding(10) 
var env_specular_u002e_sampler: sampler;
@group(0) @binding(11) 
var env_specular_u002e_texture: texture_cube<f32>;
@group(0) @binding(12) 
var env_irradiance_u002e_sampler: sampler;
@group(0) @binding(13) 
var env_irradiance_u002e_texture: texture_cube<f32>;
@group(0) @binding(14) 
var brdf_lut_u002e_sampler: sampler;
@group(0) @binding(15) 
var brdf_lut_u002e_texture: texture_2d<f32>;
@group(0) @binding(16) 
var<storage> lights: type_13;

fn main_1() {
    var local: vec3<f32>;

    let _e64 = frag_uv_1;
    let _e65 = textureSample(gbuf_tex0_u002e_texture, gbuf_tex0_u002e_sampler, _e64);
    let _e66 = frag_uv_1;
    let _e67 = textureSample(gbuf_tex1_u002e_texture, gbuf_tex1_u002e_sampler, _e66);
    let _e68 = frag_uv_1;
    let _e69 = textureSample(gbuf_tex2_u002e_texture, gbuf_tex2_u002e_sampler, _e68);
    let _e70 = frag_uv_1;
    let _e71 = textureSample(gbuf_depth_u002e_texture, gbuf_depth_u002e_sampler, _e70);
    let _e73 = _e65.xyz;
    let _e77 = ((_e67.xy * 2f) - vec2<f32>(1f, 1f));
    let _e83 = ((1f - abs(_e77.x)) - abs(_e77.y));
    let _e94 = -(max(-(_e83), 0f));
    let _e102 = normalize(vec3<f32>((_e77.x + (((step(0f, _e77.x) * 2f) - 1f) * _e94)), (_e77.y + (((step(0f, _e77.y) * 2f) - 1f) * _e94)), _e83));
    let _e105 = frag_uv_1;
    let _e107 = ((_e105 * 2f) - vec2<f32>(1f, 1f));
    let _e112 = global.inv_view_proj;
    let _e113 = (_e112 * vec4<f32>(_e107.x, _e107.y, _e71.x, 1f));
    let _e116 = (_e113.xyz / _e113.www);
    let _e118 = global_1.view_pos;
    let _e120 = normalize((_e118 - _e116));
    let _e122 = max(dot(_e102, _e120), 0.001f);
    let _e124 = global_1.light_count;
    let _e125 = f32(_e124);
    local = vec3<f32>(0f, 0f, 0f);
    if (0f < _e125) {
        let _e131 = lights.member[i32(0f)].member;
        let _e136 = lights.member[i32(0f)].member_1;
        let _e141 = lights.member[i32(0f)].member_2;
        let _e146 = lights.member[i32(0f)].member_3;
        let _e151 = lights.member[i32(0f)].member_4;
        let _e156 = lights.member[i32(0f)].member_5;
        let _e161 = lights.member[i32(0f)].member_6;
        let _e166 = lights.member[i32(0f)].member_8;
        let _e171 = select(0f, 1f, (_e131 < 0.5f));
        let _e175 = ((normalize(_e161) * _e171) + (normalize((_e151 - _e116)) * (1f - _e171)));
        let _e177 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e179 = normalize((_e120 + _e175));
        let _e181 = max(dot(_e102, _e179), 0f);
        let _e183 = max(dot(_e102, _e120), 0.00001f);
        let _e185 = max(dot(_e102, _e175), 0.00001f);
        let _e188 = (_e67.z * _e67.z);
        let _e189 = (_e188 * _e188);
        let _e193 = (((_e181 * _e181) * (_e189 - 1f)) + 1f);
        let _e200 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e201 = (1f - _e200);
        let _e220 = (_e177 + ((vec3<f32>(1f, 1f, 1f) - _e177) * pow(clamp((1f - max(dot(_e120, _e179), 0f)), 0f, 1f), 5f)));
        let _e243 = (_e151 - _e116);
        let _e244 = length(_e243);
        let _e250 = (_e244 * _e244);
        let _e253 = (_e250 / max((_e141 * _e141), 0.0001f));
        let _e256 = clamp((1f - (_e253 * _e253)), 0f, 1f);
        let _e263 = (_e151 - _e116);
        let _e264 = length(_e263);
        let _e267 = (_e263 / vec3(max(_e264, 0.0001f)));
        let _e272 = (_e264 * _e264);
        let _e275 = (_e272 / max((_e141 * _e141), 0.0001f));
        let _e278 = clamp((1f - (_e275 * _e275)), 0f, 1f);
        let _e286 = clamp(((dot(normalize(_e161), _e267) - _e156) / max((_e146 - _e156), 0.0001f)), 0f, 1f);
        let _e297 = local;
        local = (_e297 + (((((((vec3<f32>(1f, 1f, 1f) - _e220) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e220 * ((_e189 * 0.31830987f) / ((_e193 * _e193) + 0.00001f))) * (0.5f / (((_e185 * sqrt((((_e183 * _e183) * _e201) + _e200))) + (_e183 * sqrt((((_e185 * _e185) * _e201) + _e200)))) + 0.00001f)))) * _e185) * (((((_e166 * _e136) * max(dot(_e102, normalize(_e161)), 0f)) * select(0f, 1f, (_e131 < 0.5f))) + ((((_e166 * _e136) * max(dot(_e102, (_e243 / vec3(max(_e244, 0.0001f)))), 0f)) * ((_e256 * _e256) / max(_e250, 0.0001f))) * select(0f, 1f, ((_e131 > 0.5f) && (_e131 < 1.5f))))) + (((((_e166 * _e136) * max(dot(_e102, _e267), 0f)) * ((_e278 * _e278) / max(_e272, 0.0001f))) * (_e286 * _e286)) * select(0f, 1f, (_e131 > 1.5f))))));
    }
    if (1f < _e125) {
        let _e305 = lights.member[i32(1f)].member;
        let _e310 = lights.member[i32(1f)].member_1;
        let _e315 = lights.member[i32(1f)].member_2;
        let _e320 = lights.member[i32(1f)].member_3;
        let _e325 = lights.member[i32(1f)].member_4;
        let _e330 = lights.member[i32(1f)].member_5;
        let _e335 = lights.member[i32(1f)].member_6;
        let _e340 = lights.member[i32(1f)].member_8;
        let _e345 = select(0f, 1f, (_e305 < 0.5f));
        let _e349 = ((normalize(_e335) * _e345) + (normalize((_e325 - _e116)) * (1f - _e345)));
        let _e351 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e353 = normalize((_e120 + _e349));
        let _e355 = max(dot(_e102, _e353), 0f);
        let _e357 = max(dot(_e102, _e120), 0.00001f);
        let _e359 = max(dot(_e102, _e349), 0.00001f);
        let _e362 = (_e67.z * _e67.z);
        let _e363 = (_e362 * _e362);
        let _e367 = (((_e355 * _e355) * (_e363 - 1f)) + 1f);
        let _e374 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e375 = (1f - _e374);
        let _e394 = (_e351 + ((vec3<f32>(1f, 1f, 1f) - _e351) * pow(clamp((1f - max(dot(_e120, _e353), 0f)), 0f, 1f), 5f)));
        let _e417 = (_e325 - _e116);
        let _e418 = length(_e417);
        let _e424 = (_e418 * _e418);
        let _e427 = (_e424 / max((_e315 * _e315), 0.0001f));
        let _e430 = clamp((1f - (_e427 * _e427)), 0f, 1f);
        let _e437 = (_e325 - _e116);
        let _e438 = length(_e437);
        let _e441 = (_e437 / vec3(max(_e438, 0.0001f)));
        let _e446 = (_e438 * _e438);
        let _e449 = (_e446 / max((_e315 * _e315), 0.0001f));
        let _e452 = clamp((1f - (_e449 * _e449)), 0f, 1f);
        let _e460 = clamp(((dot(normalize(_e335), _e441) - _e330) / max((_e320 - _e330), 0.0001f)), 0f, 1f);
        let _e471 = local;
        local = (_e471 + (((((((vec3<f32>(1f, 1f, 1f) - _e394) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e394 * ((_e363 * 0.31830987f) / ((_e367 * _e367) + 0.00001f))) * (0.5f / (((_e359 * sqrt((((_e357 * _e357) * _e375) + _e374))) + (_e357 * sqrt((((_e359 * _e359) * _e375) + _e374)))) + 0.00001f)))) * _e359) * (((((_e340 * _e310) * max(dot(_e102, normalize(_e335)), 0f)) * select(0f, 1f, (_e305 < 0.5f))) + ((((_e340 * _e310) * max(dot(_e102, (_e417 / vec3(max(_e418, 0.0001f)))), 0f)) * ((_e430 * _e430) / max(_e424, 0.0001f))) * select(0f, 1f, ((_e305 > 0.5f) && (_e305 < 1.5f))))) + (((((_e340 * _e310) * max(dot(_e102, _e441), 0f)) * ((_e452 * _e452) / max(_e446, 0.0001f))) * (_e460 * _e460)) * select(0f, 1f, (_e305 > 1.5f))))));
    }
    if (2f < _e125) {
        let _e479 = lights.member[i32(2f)].member;
        let _e484 = lights.member[i32(2f)].member_1;
        let _e489 = lights.member[i32(2f)].member_2;
        let _e494 = lights.member[i32(2f)].member_3;
        let _e499 = lights.member[i32(2f)].member_4;
        let _e504 = lights.member[i32(2f)].member_5;
        let _e509 = lights.member[i32(2f)].member_6;
        let _e514 = lights.member[i32(2f)].member_8;
        let _e519 = select(0f, 1f, (_e479 < 0.5f));
        let _e523 = ((normalize(_e509) * _e519) + (normalize((_e499 - _e116)) * (1f - _e519)));
        let _e525 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e527 = normalize((_e120 + _e523));
        let _e529 = max(dot(_e102, _e527), 0f);
        let _e531 = max(dot(_e102, _e120), 0.00001f);
        let _e533 = max(dot(_e102, _e523), 0.00001f);
        let _e536 = (_e67.z * _e67.z);
        let _e537 = (_e536 * _e536);
        let _e541 = (((_e529 * _e529) * (_e537 - 1f)) + 1f);
        let _e548 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e549 = (1f - _e548);
        let _e568 = (_e525 + ((vec3<f32>(1f, 1f, 1f) - _e525) * pow(clamp((1f - max(dot(_e120, _e527), 0f)), 0f, 1f), 5f)));
        let _e591 = (_e499 - _e116);
        let _e592 = length(_e591);
        let _e598 = (_e592 * _e592);
        let _e601 = (_e598 / max((_e489 * _e489), 0.0001f));
        let _e604 = clamp((1f - (_e601 * _e601)), 0f, 1f);
        let _e611 = (_e499 - _e116);
        let _e612 = length(_e611);
        let _e615 = (_e611 / vec3(max(_e612, 0.0001f)));
        let _e620 = (_e612 * _e612);
        let _e623 = (_e620 / max((_e489 * _e489), 0.0001f));
        let _e626 = clamp((1f - (_e623 * _e623)), 0f, 1f);
        let _e634 = clamp(((dot(normalize(_e509), _e615) - _e504) / max((_e494 - _e504), 0.0001f)), 0f, 1f);
        let _e645 = local;
        local = (_e645 + (((((((vec3<f32>(1f, 1f, 1f) - _e568) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e568 * ((_e537 * 0.31830987f) / ((_e541 * _e541) + 0.00001f))) * (0.5f / (((_e533 * sqrt((((_e531 * _e531) * _e549) + _e548))) + (_e531 * sqrt((((_e533 * _e533) * _e549) + _e548)))) + 0.00001f)))) * _e533) * (((((_e514 * _e484) * max(dot(_e102, normalize(_e509)), 0f)) * select(0f, 1f, (_e479 < 0.5f))) + ((((_e514 * _e484) * max(dot(_e102, (_e591 / vec3(max(_e592, 0.0001f)))), 0f)) * ((_e604 * _e604) / max(_e598, 0.0001f))) * select(0f, 1f, ((_e479 > 0.5f) && (_e479 < 1.5f))))) + (((((_e514 * _e484) * max(dot(_e102, _e615), 0f)) * ((_e626 * _e626) / max(_e620, 0.0001f))) * (_e634 * _e634)) * select(0f, 1f, (_e479 > 1.5f))))));
    }
    if (3f < _e125) {
        let _e653 = lights.member[i32(3f)].member;
        let _e658 = lights.member[i32(3f)].member_1;
        let _e663 = lights.member[i32(3f)].member_2;
        let _e668 = lights.member[i32(3f)].member_3;
        let _e673 = lights.member[i32(3f)].member_4;
        let _e678 = lights.member[i32(3f)].member_5;
        let _e683 = lights.member[i32(3f)].member_6;
        let _e688 = lights.member[i32(3f)].member_8;
        let _e693 = select(0f, 1f, (_e653 < 0.5f));
        let _e697 = ((normalize(_e683) * _e693) + (normalize((_e673 - _e116)) * (1f - _e693)));
        let _e699 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e701 = normalize((_e120 + _e697));
        let _e703 = max(dot(_e102, _e701), 0f);
        let _e705 = max(dot(_e102, _e120), 0.00001f);
        let _e707 = max(dot(_e102, _e697), 0.00001f);
        let _e710 = (_e67.z * _e67.z);
        let _e711 = (_e710 * _e710);
        let _e715 = (((_e703 * _e703) * (_e711 - 1f)) + 1f);
        let _e722 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e723 = (1f - _e722);
        let _e742 = (_e699 + ((vec3<f32>(1f, 1f, 1f) - _e699) * pow(clamp((1f - max(dot(_e120, _e701), 0f)), 0f, 1f), 5f)));
        let _e765 = (_e673 - _e116);
        let _e766 = length(_e765);
        let _e772 = (_e766 * _e766);
        let _e775 = (_e772 / max((_e663 * _e663), 0.0001f));
        let _e778 = clamp((1f - (_e775 * _e775)), 0f, 1f);
        let _e785 = (_e673 - _e116);
        let _e786 = length(_e785);
        let _e789 = (_e785 / vec3(max(_e786, 0.0001f)));
        let _e794 = (_e786 * _e786);
        let _e797 = (_e794 / max((_e663 * _e663), 0.0001f));
        let _e800 = clamp((1f - (_e797 * _e797)), 0f, 1f);
        let _e808 = clamp(((dot(normalize(_e683), _e789) - _e678) / max((_e668 - _e678), 0.0001f)), 0f, 1f);
        let _e819 = local;
        local = (_e819 + (((((((vec3<f32>(1f, 1f, 1f) - _e742) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e742 * ((_e711 * 0.31830987f) / ((_e715 * _e715) + 0.00001f))) * (0.5f / (((_e707 * sqrt((((_e705 * _e705) * _e723) + _e722))) + (_e705 * sqrt((((_e707 * _e707) * _e723) + _e722)))) + 0.00001f)))) * _e707) * (((((_e688 * _e658) * max(dot(_e102, normalize(_e683)), 0f)) * select(0f, 1f, (_e653 < 0.5f))) + ((((_e688 * _e658) * max(dot(_e102, (_e765 / vec3(max(_e766, 0.0001f)))), 0f)) * ((_e778 * _e778) / max(_e772, 0.0001f))) * select(0f, 1f, ((_e653 > 0.5f) && (_e653 < 1.5f))))) + (((((_e688 * _e658) * max(dot(_e102, _e789), 0f)) * ((_e800 * _e800) / max(_e794, 0.0001f))) * (_e808 * _e808)) * select(0f, 1f, (_e653 > 1.5f))))));
    }
    if (4f < _e125) {
        let _e827 = lights.member[i32(4f)].member;
        let _e832 = lights.member[i32(4f)].member_1;
        let _e837 = lights.member[i32(4f)].member_2;
        let _e842 = lights.member[i32(4f)].member_3;
        let _e847 = lights.member[i32(4f)].member_4;
        let _e852 = lights.member[i32(4f)].member_5;
        let _e857 = lights.member[i32(4f)].member_6;
        let _e862 = lights.member[i32(4f)].member_8;
        let _e867 = select(0f, 1f, (_e827 < 0.5f));
        let _e871 = ((normalize(_e857) * _e867) + (normalize((_e847 - _e116)) * (1f - _e867)));
        let _e873 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e875 = normalize((_e120 + _e871));
        let _e877 = max(dot(_e102, _e875), 0f);
        let _e879 = max(dot(_e102, _e120), 0.00001f);
        let _e881 = max(dot(_e102, _e871), 0.00001f);
        let _e884 = (_e67.z * _e67.z);
        let _e885 = (_e884 * _e884);
        let _e889 = (((_e877 * _e877) * (_e885 - 1f)) + 1f);
        let _e896 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e897 = (1f - _e896);
        let _e916 = (_e873 + ((vec3<f32>(1f, 1f, 1f) - _e873) * pow(clamp((1f - max(dot(_e120, _e875), 0f)), 0f, 1f), 5f)));
        let _e939 = (_e847 - _e116);
        let _e940 = length(_e939);
        let _e946 = (_e940 * _e940);
        let _e949 = (_e946 / max((_e837 * _e837), 0.0001f));
        let _e952 = clamp((1f - (_e949 * _e949)), 0f, 1f);
        let _e959 = (_e847 - _e116);
        let _e960 = length(_e959);
        let _e963 = (_e959 / vec3(max(_e960, 0.0001f)));
        let _e968 = (_e960 * _e960);
        let _e971 = (_e968 / max((_e837 * _e837), 0.0001f));
        let _e974 = clamp((1f - (_e971 * _e971)), 0f, 1f);
        let _e982 = clamp(((dot(normalize(_e857), _e963) - _e852) / max((_e842 - _e852), 0.0001f)), 0f, 1f);
        let _e993 = local;
        local = (_e993 + (((((((vec3<f32>(1f, 1f, 1f) - _e916) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e916 * ((_e885 * 0.31830987f) / ((_e889 * _e889) + 0.00001f))) * (0.5f / (((_e881 * sqrt((((_e879 * _e879) * _e897) + _e896))) + (_e879 * sqrt((((_e881 * _e881) * _e897) + _e896)))) + 0.00001f)))) * _e881) * (((((_e862 * _e832) * max(dot(_e102, normalize(_e857)), 0f)) * select(0f, 1f, (_e827 < 0.5f))) + ((((_e862 * _e832) * max(dot(_e102, (_e939 / vec3(max(_e940, 0.0001f)))), 0f)) * ((_e952 * _e952) / max(_e946, 0.0001f))) * select(0f, 1f, ((_e827 > 0.5f) && (_e827 < 1.5f))))) + (((((_e862 * _e832) * max(dot(_e102, _e963), 0f)) * ((_e974 * _e974) / max(_e968, 0.0001f))) * (_e982 * _e982)) * select(0f, 1f, (_e827 > 1.5f))))));
    }
    if (5f < _e125) {
        let _e1001 = lights.member[i32(5f)].member;
        let _e1006 = lights.member[i32(5f)].member_1;
        let _e1011 = lights.member[i32(5f)].member_2;
        let _e1016 = lights.member[i32(5f)].member_3;
        let _e1021 = lights.member[i32(5f)].member_4;
        let _e1026 = lights.member[i32(5f)].member_5;
        let _e1031 = lights.member[i32(5f)].member_6;
        let _e1036 = lights.member[i32(5f)].member_8;
        let _e1041 = select(0f, 1f, (_e1001 < 0.5f));
        let _e1045 = ((normalize(_e1031) * _e1041) + (normalize((_e1021 - _e116)) * (1f - _e1041)));
        let _e1047 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e1049 = normalize((_e120 + _e1045));
        let _e1051 = max(dot(_e102, _e1049), 0f);
        let _e1053 = max(dot(_e102, _e120), 0.00001f);
        let _e1055 = max(dot(_e102, _e1045), 0.00001f);
        let _e1058 = (_e67.z * _e67.z);
        let _e1059 = (_e1058 * _e1058);
        let _e1063 = (((_e1051 * _e1051) * (_e1059 - 1f)) + 1f);
        let _e1070 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e1071 = (1f - _e1070);
        let _e1090 = (_e1047 + ((vec3<f32>(1f, 1f, 1f) - _e1047) * pow(clamp((1f - max(dot(_e120, _e1049), 0f)), 0f, 1f), 5f)));
        let _e1113 = (_e1021 - _e116);
        let _e1114 = length(_e1113);
        let _e1120 = (_e1114 * _e1114);
        let _e1123 = (_e1120 / max((_e1011 * _e1011), 0.0001f));
        let _e1126 = clamp((1f - (_e1123 * _e1123)), 0f, 1f);
        let _e1133 = (_e1021 - _e116);
        let _e1134 = length(_e1133);
        let _e1137 = (_e1133 / vec3(max(_e1134, 0.0001f)));
        let _e1142 = (_e1134 * _e1134);
        let _e1145 = (_e1142 / max((_e1011 * _e1011), 0.0001f));
        let _e1148 = clamp((1f - (_e1145 * _e1145)), 0f, 1f);
        let _e1156 = clamp(((dot(normalize(_e1031), _e1137) - _e1026) / max((_e1016 - _e1026), 0.0001f)), 0f, 1f);
        let _e1167 = local;
        local = (_e1167 + (((((((vec3<f32>(1f, 1f, 1f) - _e1090) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e1090 * ((_e1059 * 0.31830987f) / ((_e1063 * _e1063) + 0.00001f))) * (0.5f / (((_e1055 * sqrt((((_e1053 * _e1053) * _e1071) + _e1070))) + (_e1053 * sqrt((((_e1055 * _e1055) * _e1071) + _e1070)))) + 0.00001f)))) * _e1055) * (((((_e1036 * _e1006) * max(dot(_e102, normalize(_e1031)), 0f)) * select(0f, 1f, (_e1001 < 0.5f))) + ((((_e1036 * _e1006) * max(dot(_e102, (_e1113 / vec3(max(_e1114, 0.0001f)))), 0f)) * ((_e1126 * _e1126) / max(_e1120, 0.0001f))) * select(0f, 1f, ((_e1001 > 0.5f) && (_e1001 < 1.5f))))) + (((((_e1036 * _e1006) * max(dot(_e102, _e1137), 0f)) * ((_e1148 * _e1148) / max(_e1142, 0.0001f))) * (_e1156 * _e1156)) * select(0f, 1f, (_e1001 > 1.5f))))));
    }
    if (6f < _e125) {
        let _e1175 = lights.member[i32(6f)].member;
        let _e1180 = lights.member[i32(6f)].member_1;
        let _e1185 = lights.member[i32(6f)].member_2;
        let _e1190 = lights.member[i32(6f)].member_3;
        let _e1195 = lights.member[i32(6f)].member_4;
        let _e1200 = lights.member[i32(6f)].member_5;
        let _e1205 = lights.member[i32(6f)].member_6;
        let _e1210 = lights.member[i32(6f)].member_8;
        let _e1215 = select(0f, 1f, (_e1175 < 0.5f));
        let _e1219 = ((normalize(_e1205) * _e1215) + (normalize((_e1195 - _e116)) * (1f - _e1215)));
        let _e1221 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e1223 = normalize((_e120 + _e1219));
        let _e1225 = max(dot(_e102, _e1223), 0f);
        let _e1227 = max(dot(_e102, _e120), 0.00001f);
        let _e1229 = max(dot(_e102, _e1219), 0.00001f);
        let _e1232 = (_e67.z * _e67.z);
        let _e1233 = (_e1232 * _e1232);
        let _e1237 = (((_e1225 * _e1225) * (_e1233 - 1f)) + 1f);
        let _e1244 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e1245 = (1f - _e1244);
        let _e1264 = (_e1221 + ((vec3<f32>(1f, 1f, 1f) - _e1221) * pow(clamp((1f - max(dot(_e120, _e1223), 0f)), 0f, 1f), 5f)));
        let _e1287 = (_e1195 - _e116);
        let _e1288 = length(_e1287);
        let _e1294 = (_e1288 * _e1288);
        let _e1297 = (_e1294 / max((_e1185 * _e1185), 0.0001f));
        let _e1300 = clamp((1f - (_e1297 * _e1297)), 0f, 1f);
        let _e1307 = (_e1195 - _e116);
        let _e1308 = length(_e1307);
        let _e1311 = (_e1307 / vec3(max(_e1308, 0.0001f)));
        let _e1316 = (_e1308 * _e1308);
        let _e1319 = (_e1316 / max((_e1185 * _e1185), 0.0001f));
        let _e1322 = clamp((1f - (_e1319 * _e1319)), 0f, 1f);
        let _e1330 = clamp(((dot(normalize(_e1205), _e1311) - _e1200) / max((_e1190 - _e1200), 0.0001f)), 0f, 1f);
        let _e1341 = local;
        local = (_e1341 + (((((((vec3<f32>(1f, 1f, 1f) - _e1264) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e1264 * ((_e1233 * 0.31830987f) / ((_e1237 * _e1237) + 0.00001f))) * (0.5f / (((_e1229 * sqrt((((_e1227 * _e1227) * _e1245) + _e1244))) + (_e1227 * sqrt((((_e1229 * _e1229) * _e1245) + _e1244)))) + 0.00001f)))) * _e1229) * (((((_e1210 * _e1180) * max(dot(_e102, normalize(_e1205)), 0f)) * select(0f, 1f, (_e1175 < 0.5f))) + ((((_e1210 * _e1180) * max(dot(_e102, (_e1287 / vec3(max(_e1288, 0.0001f)))), 0f)) * ((_e1300 * _e1300) / max(_e1294, 0.0001f))) * select(0f, 1f, ((_e1175 > 0.5f) && (_e1175 < 1.5f))))) + (((((_e1210 * _e1180) * max(dot(_e102, _e1311), 0f)) * ((_e1322 * _e1322) / max(_e1316, 0.0001f))) * (_e1330 * _e1330)) * select(0f, 1f, (_e1175 > 1.5f))))));
    }
    if (7f < _e125) {
        let _e1349 = lights.member[i32(7f)].member;
        let _e1354 = lights.member[i32(7f)].member_1;
        let _e1359 = lights.member[i32(7f)].member_2;
        let _e1364 = lights.member[i32(7f)].member_3;
        let _e1369 = lights.member[i32(7f)].member_4;
        let _e1374 = lights.member[i32(7f)].member_5;
        let _e1379 = lights.member[i32(7f)].member_6;
        let _e1384 = lights.member[i32(7f)].member_8;
        let _e1389 = select(0f, 1f, (_e1349 < 0.5f));
        let _e1393 = ((normalize(_e1379) * _e1389) + (normalize((_e1369 - _e116)) * (1f - _e1389)));
        let _e1395 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e1397 = normalize((_e120 + _e1393));
        let _e1399 = max(dot(_e102, _e1397), 0f);
        let _e1401 = max(dot(_e102, _e120), 0.00001f);
        let _e1403 = max(dot(_e102, _e1393), 0.00001f);
        let _e1406 = (_e67.z * _e67.z);
        let _e1407 = (_e1406 * _e1406);
        let _e1411 = (((_e1399 * _e1399) * (_e1407 - 1f)) + 1f);
        let _e1418 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e1419 = (1f - _e1418);
        let _e1438 = (_e1395 + ((vec3<f32>(1f, 1f, 1f) - _e1395) * pow(clamp((1f - max(dot(_e120, _e1397), 0f)), 0f, 1f), 5f)));
        let _e1461 = (_e1369 - _e116);
        let _e1462 = length(_e1461);
        let _e1468 = (_e1462 * _e1462);
        let _e1471 = (_e1468 / max((_e1359 * _e1359), 0.0001f));
        let _e1474 = clamp((1f - (_e1471 * _e1471)), 0f, 1f);
        let _e1481 = (_e1369 - _e116);
        let _e1482 = length(_e1481);
        let _e1485 = (_e1481 / vec3(max(_e1482, 0.0001f)));
        let _e1490 = (_e1482 * _e1482);
        let _e1493 = (_e1490 / max((_e1359 * _e1359), 0.0001f));
        let _e1496 = clamp((1f - (_e1493 * _e1493)), 0f, 1f);
        let _e1504 = clamp(((dot(normalize(_e1379), _e1485) - _e1374) / max((_e1364 - _e1374), 0.0001f)), 0f, 1f);
        let _e1515 = local;
        local = (_e1515 + (((((((vec3<f32>(1f, 1f, 1f) - _e1438) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e1438 * ((_e1407 * 0.31830987f) / ((_e1411 * _e1411) + 0.00001f))) * (0.5f / (((_e1403 * sqrt((((_e1401 * _e1401) * _e1419) + _e1418))) + (_e1401 * sqrt((((_e1403 * _e1403) * _e1419) + _e1418)))) + 0.00001f)))) * _e1403) * (((((_e1384 * _e1354) * max(dot(_e102, normalize(_e1379)), 0f)) * select(0f, 1f, (_e1349 < 0.5f))) + ((((_e1384 * _e1354) * max(dot(_e102, (_e1461 / vec3(max(_e1462, 0.0001f)))), 0f)) * ((_e1474 * _e1474) / max(_e1468, 0.0001f))) * select(0f, 1f, ((_e1349 > 0.5f) && (_e1349 < 1.5f))))) + (((((_e1384 * _e1354) * max(dot(_e102, _e1485), 0f)) * ((_e1496 * _e1496) / max(_e1490, 0.0001f))) * (_e1504 * _e1504)) * select(0f, 1f, (_e1349 > 1.5f))))));
    }
    if (8f < _e125) {
        let _e1523 = lights.member[i32(8f)].member;
        let _e1528 = lights.member[i32(8f)].member_1;
        let _e1533 = lights.member[i32(8f)].member_2;
        let _e1538 = lights.member[i32(8f)].member_3;
        let _e1543 = lights.member[i32(8f)].member_4;
        let _e1548 = lights.member[i32(8f)].member_5;
        let _e1553 = lights.member[i32(8f)].member_6;
        let _e1558 = lights.member[i32(8f)].member_8;
        let _e1563 = select(0f, 1f, (_e1523 < 0.5f));
        let _e1567 = ((normalize(_e1553) * _e1563) + (normalize((_e1543 - _e116)) * (1f - _e1563)));
        let _e1569 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e1571 = normalize((_e120 + _e1567));
        let _e1573 = max(dot(_e102, _e1571), 0f);
        let _e1575 = max(dot(_e102, _e120), 0.00001f);
        let _e1577 = max(dot(_e102, _e1567), 0.00001f);
        let _e1580 = (_e67.z * _e67.z);
        let _e1581 = (_e1580 * _e1580);
        let _e1585 = (((_e1573 * _e1573) * (_e1581 - 1f)) + 1f);
        let _e1592 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e1593 = (1f - _e1592);
        let _e1612 = (_e1569 + ((vec3<f32>(1f, 1f, 1f) - _e1569) * pow(clamp((1f - max(dot(_e120, _e1571), 0f)), 0f, 1f), 5f)));
        let _e1635 = (_e1543 - _e116);
        let _e1636 = length(_e1635);
        let _e1642 = (_e1636 * _e1636);
        let _e1645 = (_e1642 / max((_e1533 * _e1533), 0.0001f));
        let _e1648 = clamp((1f - (_e1645 * _e1645)), 0f, 1f);
        let _e1655 = (_e1543 - _e116);
        let _e1656 = length(_e1655);
        let _e1659 = (_e1655 / vec3(max(_e1656, 0.0001f)));
        let _e1664 = (_e1656 * _e1656);
        let _e1667 = (_e1664 / max((_e1533 * _e1533), 0.0001f));
        let _e1670 = clamp((1f - (_e1667 * _e1667)), 0f, 1f);
        let _e1678 = clamp(((dot(normalize(_e1553), _e1659) - _e1548) / max((_e1538 - _e1548), 0.0001f)), 0f, 1f);
        let _e1689 = local;
        local = (_e1689 + (((((((vec3<f32>(1f, 1f, 1f) - _e1612) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e1612 * ((_e1581 * 0.31830987f) / ((_e1585 * _e1585) + 0.00001f))) * (0.5f / (((_e1577 * sqrt((((_e1575 * _e1575) * _e1593) + _e1592))) + (_e1575 * sqrt((((_e1577 * _e1577) * _e1593) + _e1592)))) + 0.00001f)))) * _e1577) * (((((_e1558 * _e1528) * max(dot(_e102, normalize(_e1553)), 0f)) * select(0f, 1f, (_e1523 < 0.5f))) + ((((_e1558 * _e1528) * max(dot(_e102, (_e1635 / vec3(max(_e1636, 0.0001f)))), 0f)) * ((_e1648 * _e1648) / max(_e1642, 0.0001f))) * select(0f, 1f, ((_e1523 > 0.5f) && (_e1523 < 1.5f))))) + (((((_e1558 * _e1528) * max(dot(_e102, _e1659), 0f)) * ((_e1670 * _e1670) / max(_e1664, 0.0001f))) * (_e1678 * _e1678)) * select(0f, 1f, (_e1523 > 1.5f))))));
    }
    if (9f < _e125) {
        let _e1697 = lights.member[i32(9f)].member;
        let _e1702 = lights.member[i32(9f)].member_1;
        let _e1707 = lights.member[i32(9f)].member_2;
        let _e1712 = lights.member[i32(9f)].member_3;
        let _e1717 = lights.member[i32(9f)].member_4;
        let _e1722 = lights.member[i32(9f)].member_5;
        let _e1727 = lights.member[i32(9f)].member_6;
        let _e1732 = lights.member[i32(9f)].member_8;
        let _e1737 = select(0f, 1f, (_e1697 < 0.5f));
        let _e1741 = ((normalize(_e1727) * _e1737) + (normalize((_e1717 - _e116)) * (1f - _e1737)));
        let _e1743 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e1745 = normalize((_e120 + _e1741));
        let _e1747 = max(dot(_e102, _e1745), 0f);
        let _e1749 = max(dot(_e102, _e120), 0.00001f);
        let _e1751 = max(dot(_e102, _e1741), 0.00001f);
        let _e1754 = (_e67.z * _e67.z);
        let _e1755 = (_e1754 * _e1754);
        let _e1759 = (((_e1747 * _e1747) * (_e1755 - 1f)) + 1f);
        let _e1766 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e1767 = (1f - _e1766);
        let _e1786 = (_e1743 + ((vec3<f32>(1f, 1f, 1f) - _e1743) * pow(clamp((1f - max(dot(_e120, _e1745), 0f)), 0f, 1f), 5f)));
        let _e1809 = (_e1717 - _e116);
        let _e1810 = length(_e1809);
        let _e1816 = (_e1810 * _e1810);
        let _e1819 = (_e1816 / max((_e1707 * _e1707), 0.0001f));
        let _e1822 = clamp((1f - (_e1819 * _e1819)), 0f, 1f);
        let _e1829 = (_e1717 - _e116);
        let _e1830 = length(_e1829);
        let _e1833 = (_e1829 / vec3(max(_e1830, 0.0001f)));
        let _e1838 = (_e1830 * _e1830);
        let _e1841 = (_e1838 / max((_e1707 * _e1707), 0.0001f));
        let _e1844 = clamp((1f - (_e1841 * _e1841)), 0f, 1f);
        let _e1852 = clamp(((dot(normalize(_e1727), _e1833) - _e1722) / max((_e1712 - _e1722), 0.0001f)), 0f, 1f);
        let _e1863 = local;
        local = (_e1863 + (((((((vec3<f32>(1f, 1f, 1f) - _e1786) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e1786 * ((_e1755 * 0.31830987f) / ((_e1759 * _e1759) + 0.00001f))) * (0.5f / (((_e1751 * sqrt((((_e1749 * _e1749) * _e1767) + _e1766))) + (_e1749 * sqrt((((_e1751 * _e1751) * _e1767) + _e1766)))) + 0.00001f)))) * _e1751) * (((((_e1732 * _e1702) * max(dot(_e102, normalize(_e1727)), 0f)) * select(0f, 1f, (_e1697 < 0.5f))) + ((((_e1732 * _e1702) * max(dot(_e102, (_e1809 / vec3(max(_e1810, 0.0001f)))), 0f)) * ((_e1822 * _e1822) / max(_e1816, 0.0001f))) * select(0f, 1f, ((_e1697 > 0.5f) && (_e1697 < 1.5f))))) + (((((_e1732 * _e1702) * max(dot(_e102, _e1833), 0f)) * ((_e1844 * _e1844) / max(_e1838, 0.0001f))) * (_e1852 * _e1852)) * select(0f, 1f, (_e1697 > 1.5f))))));
    }
    if (10f < _e125) {
        let _e1871 = lights.member[i32(10f)].member;
        let _e1876 = lights.member[i32(10f)].member_1;
        let _e1881 = lights.member[i32(10f)].member_2;
        let _e1886 = lights.member[i32(10f)].member_3;
        let _e1891 = lights.member[i32(10f)].member_4;
        let _e1896 = lights.member[i32(10f)].member_5;
        let _e1901 = lights.member[i32(10f)].member_6;
        let _e1906 = lights.member[i32(10f)].member_8;
        let _e1911 = select(0f, 1f, (_e1871 < 0.5f));
        let _e1915 = ((normalize(_e1901) * _e1911) + (normalize((_e1891 - _e116)) * (1f - _e1911)));
        let _e1917 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e1919 = normalize((_e120 + _e1915));
        let _e1921 = max(dot(_e102, _e1919), 0f);
        let _e1923 = max(dot(_e102, _e120), 0.00001f);
        let _e1925 = max(dot(_e102, _e1915), 0.00001f);
        let _e1928 = (_e67.z * _e67.z);
        let _e1929 = (_e1928 * _e1928);
        let _e1933 = (((_e1921 * _e1921) * (_e1929 - 1f)) + 1f);
        let _e1940 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e1941 = (1f - _e1940);
        let _e1960 = (_e1917 + ((vec3<f32>(1f, 1f, 1f) - _e1917) * pow(clamp((1f - max(dot(_e120, _e1919), 0f)), 0f, 1f), 5f)));
        let _e1983 = (_e1891 - _e116);
        let _e1984 = length(_e1983);
        let _e1990 = (_e1984 * _e1984);
        let _e1993 = (_e1990 / max((_e1881 * _e1881), 0.0001f));
        let _e1996 = clamp((1f - (_e1993 * _e1993)), 0f, 1f);
        let _e2003 = (_e1891 - _e116);
        let _e2004 = length(_e2003);
        let _e2007 = (_e2003 / vec3(max(_e2004, 0.0001f)));
        let _e2012 = (_e2004 * _e2004);
        let _e2015 = (_e2012 / max((_e1881 * _e1881), 0.0001f));
        let _e2018 = clamp((1f - (_e2015 * _e2015)), 0f, 1f);
        let _e2026 = clamp(((dot(normalize(_e1901), _e2007) - _e1896) / max((_e1886 - _e1896), 0.0001f)), 0f, 1f);
        let _e2037 = local;
        local = (_e2037 + (((((((vec3<f32>(1f, 1f, 1f) - _e1960) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e1960 * ((_e1929 * 0.31830987f) / ((_e1933 * _e1933) + 0.00001f))) * (0.5f / (((_e1925 * sqrt((((_e1923 * _e1923) * _e1941) + _e1940))) + (_e1923 * sqrt((((_e1925 * _e1925) * _e1941) + _e1940)))) + 0.00001f)))) * _e1925) * (((((_e1906 * _e1876) * max(dot(_e102, normalize(_e1901)), 0f)) * select(0f, 1f, (_e1871 < 0.5f))) + ((((_e1906 * _e1876) * max(dot(_e102, (_e1983 / vec3(max(_e1984, 0.0001f)))), 0f)) * ((_e1996 * _e1996) / max(_e1990, 0.0001f))) * select(0f, 1f, ((_e1871 > 0.5f) && (_e1871 < 1.5f))))) + (((((_e1906 * _e1876) * max(dot(_e102, _e2007), 0f)) * ((_e2018 * _e2018) / max(_e2012, 0.0001f))) * (_e2026 * _e2026)) * select(0f, 1f, (_e1871 > 1.5f))))));
    }
    if (11f < _e125) {
        let _e2045 = lights.member[i32(11f)].member;
        let _e2050 = lights.member[i32(11f)].member_1;
        let _e2055 = lights.member[i32(11f)].member_2;
        let _e2060 = lights.member[i32(11f)].member_3;
        let _e2065 = lights.member[i32(11f)].member_4;
        let _e2070 = lights.member[i32(11f)].member_5;
        let _e2075 = lights.member[i32(11f)].member_6;
        let _e2080 = lights.member[i32(11f)].member_8;
        let _e2085 = select(0f, 1f, (_e2045 < 0.5f));
        let _e2089 = ((normalize(_e2075) * _e2085) + (normalize((_e2065 - _e116)) * (1f - _e2085)));
        let _e2091 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e2093 = normalize((_e120 + _e2089));
        let _e2095 = max(dot(_e102, _e2093), 0f);
        let _e2097 = max(dot(_e102, _e120), 0.00001f);
        let _e2099 = max(dot(_e102, _e2089), 0.00001f);
        let _e2102 = (_e67.z * _e67.z);
        let _e2103 = (_e2102 * _e2102);
        let _e2107 = (((_e2095 * _e2095) * (_e2103 - 1f)) + 1f);
        let _e2114 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e2115 = (1f - _e2114);
        let _e2134 = (_e2091 + ((vec3<f32>(1f, 1f, 1f) - _e2091) * pow(clamp((1f - max(dot(_e120, _e2093), 0f)), 0f, 1f), 5f)));
        let _e2157 = (_e2065 - _e116);
        let _e2158 = length(_e2157);
        let _e2164 = (_e2158 * _e2158);
        let _e2167 = (_e2164 / max((_e2055 * _e2055), 0.0001f));
        let _e2170 = clamp((1f - (_e2167 * _e2167)), 0f, 1f);
        let _e2177 = (_e2065 - _e116);
        let _e2178 = length(_e2177);
        let _e2181 = (_e2177 / vec3(max(_e2178, 0.0001f)));
        let _e2186 = (_e2178 * _e2178);
        let _e2189 = (_e2186 / max((_e2055 * _e2055), 0.0001f));
        let _e2192 = clamp((1f - (_e2189 * _e2189)), 0f, 1f);
        let _e2200 = clamp(((dot(normalize(_e2075), _e2181) - _e2070) / max((_e2060 - _e2070), 0.0001f)), 0f, 1f);
        let _e2211 = local;
        local = (_e2211 + (((((((vec3<f32>(1f, 1f, 1f) - _e2134) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e2134 * ((_e2103 * 0.31830987f) / ((_e2107 * _e2107) + 0.00001f))) * (0.5f / (((_e2099 * sqrt((((_e2097 * _e2097) * _e2115) + _e2114))) + (_e2097 * sqrt((((_e2099 * _e2099) * _e2115) + _e2114)))) + 0.00001f)))) * _e2099) * (((((_e2080 * _e2050) * max(dot(_e102, normalize(_e2075)), 0f)) * select(0f, 1f, (_e2045 < 0.5f))) + ((((_e2080 * _e2050) * max(dot(_e102, (_e2157 / vec3(max(_e2158, 0.0001f)))), 0f)) * ((_e2170 * _e2170) / max(_e2164, 0.0001f))) * select(0f, 1f, ((_e2045 > 0.5f) && (_e2045 < 1.5f))))) + (((((_e2080 * _e2050) * max(dot(_e102, _e2181), 0f)) * ((_e2192 * _e2192) / max(_e2186, 0.0001f))) * (_e2200 * _e2200)) * select(0f, 1f, (_e2045 > 1.5f))))));
    }
    if (12f < _e125) {
        let _e2219 = lights.member[i32(12f)].member;
        let _e2224 = lights.member[i32(12f)].member_1;
        let _e2229 = lights.member[i32(12f)].member_2;
        let _e2234 = lights.member[i32(12f)].member_3;
        let _e2239 = lights.member[i32(12f)].member_4;
        let _e2244 = lights.member[i32(12f)].member_5;
        let _e2249 = lights.member[i32(12f)].member_6;
        let _e2254 = lights.member[i32(12f)].member_8;
        let _e2259 = select(0f, 1f, (_e2219 < 0.5f));
        let _e2263 = ((normalize(_e2249) * _e2259) + (normalize((_e2239 - _e116)) * (1f - _e2259)));
        let _e2265 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e2267 = normalize((_e120 + _e2263));
        let _e2269 = max(dot(_e102, _e2267), 0f);
        let _e2271 = max(dot(_e102, _e120), 0.00001f);
        let _e2273 = max(dot(_e102, _e2263), 0.00001f);
        let _e2276 = (_e67.z * _e67.z);
        let _e2277 = (_e2276 * _e2276);
        let _e2281 = (((_e2269 * _e2269) * (_e2277 - 1f)) + 1f);
        let _e2288 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e2289 = (1f - _e2288);
        let _e2308 = (_e2265 + ((vec3<f32>(1f, 1f, 1f) - _e2265) * pow(clamp((1f - max(dot(_e120, _e2267), 0f)), 0f, 1f), 5f)));
        let _e2331 = (_e2239 - _e116);
        let _e2332 = length(_e2331);
        let _e2338 = (_e2332 * _e2332);
        let _e2341 = (_e2338 / max((_e2229 * _e2229), 0.0001f));
        let _e2344 = clamp((1f - (_e2341 * _e2341)), 0f, 1f);
        let _e2351 = (_e2239 - _e116);
        let _e2352 = length(_e2351);
        let _e2355 = (_e2351 / vec3(max(_e2352, 0.0001f)));
        let _e2360 = (_e2352 * _e2352);
        let _e2363 = (_e2360 / max((_e2229 * _e2229), 0.0001f));
        let _e2366 = clamp((1f - (_e2363 * _e2363)), 0f, 1f);
        let _e2374 = clamp(((dot(normalize(_e2249), _e2355) - _e2244) / max((_e2234 - _e2244), 0.0001f)), 0f, 1f);
        let _e2385 = local;
        local = (_e2385 + (((((((vec3<f32>(1f, 1f, 1f) - _e2308) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e2308 * ((_e2277 * 0.31830987f) / ((_e2281 * _e2281) + 0.00001f))) * (0.5f / (((_e2273 * sqrt((((_e2271 * _e2271) * _e2289) + _e2288))) + (_e2271 * sqrt((((_e2273 * _e2273) * _e2289) + _e2288)))) + 0.00001f)))) * _e2273) * (((((_e2254 * _e2224) * max(dot(_e102, normalize(_e2249)), 0f)) * select(0f, 1f, (_e2219 < 0.5f))) + ((((_e2254 * _e2224) * max(dot(_e102, (_e2331 / vec3(max(_e2332, 0.0001f)))), 0f)) * ((_e2344 * _e2344) / max(_e2338, 0.0001f))) * select(0f, 1f, ((_e2219 > 0.5f) && (_e2219 < 1.5f))))) + (((((_e2254 * _e2224) * max(dot(_e102, _e2355), 0f)) * ((_e2366 * _e2366) / max(_e2360, 0.0001f))) * (_e2374 * _e2374)) * select(0f, 1f, (_e2219 > 1.5f))))));
    }
    if (13f < _e125) {
        let _e2393 = lights.member[i32(13f)].member;
        let _e2398 = lights.member[i32(13f)].member_1;
        let _e2403 = lights.member[i32(13f)].member_2;
        let _e2408 = lights.member[i32(13f)].member_3;
        let _e2413 = lights.member[i32(13f)].member_4;
        let _e2418 = lights.member[i32(13f)].member_5;
        let _e2423 = lights.member[i32(13f)].member_6;
        let _e2428 = lights.member[i32(13f)].member_8;
        let _e2433 = select(0f, 1f, (_e2393 < 0.5f));
        let _e2437 = ((normalize(_e2423) * _e2433) + (normalize((_e2413 - _e116)) * (1f - _e2433)));
        let _e2439 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e2441 = normalize((_e120 + _e2437));
        let _e2443 = max(dot(_e102, _e2441), 0f);
        let _e2445 = max(dot(_e102, _e120), 0.00001f);
        let _e2447 = max(dot(_e102, _e2437), 0.00001f);
        let _e2450 = (_e67.z * _e67.z);
        let _e2451 = (_e2450 * _e2450);
        let _e2455 = (((_e2443 * _e2443) * (_e2451 - 1f)) + 1f);
        let _e2462 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e2463 = (1f - _e2462);
        let _e2482 = (_e2439 + ((vec3<f32>(1f, 1f, 1f) - _e2439) * pow(clamp((1f - max(dot(_e120, _e2441), 0f)), 0f, 1f), 5f)));
        let _e2505 = (_e2413 - _e116);
        let _e2506 = length(_e2505);
        let _e2512 = (_e2506 * _e2506);
        let _e2515 = (_e2512 / max((_e2403 * _e2403), 0.0001f));
        let _e2518 = clamp((1f - (_e2515 * _e2515)), 0f, 1f);
        let _e2525 = (_e2413 - _e116);
        let _e2526 = length(_e2525);
        let _e2529 = (_e2525 / vec3(max(_e2526, 0.0001f)));
        let _e2534 = (_e2526 * _e2526);
        let _e2537 = (_e2534 / max((_e2403 * _e2403), 0.0001f));
        let _e2540 = clamp((1f - (_e2537 * _e2537)), 0f, 1f);
        let _e2548 = clamp(((dot(normalize(_e2423), _e2529) - _e2418) / max((_e2408 - _e2418), 0.0001f)), 0f, 1f);
        let _e2559 = local;
        local = (_e2559 + (((((((vec3<f32>(1f, 1f, 1f) - _e2482) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e2482 * ((_e2451 * 0.31830987f) / ((_e2455 * _e2455) + 0.00001f))) * (0.5f / (((_e2447 * sqrt((((_e2445 * _e2445) * _e2463) + _e2462))) + (_e2445 * sqrt((((_e2447 * _e2447) * _e2463) + _e2462)))) + 0.00001f)))) * _e2447) * (((((_e2428 * _e2398) * max(dot(_e102, normalize(_e2423)), 0f)) * select(0f, 1f, (_e2393 < 0.5f))) + ((((_e2428 * _e2398) * max(dot(_e102, (_e2505 / vec3(max(_e2506, 0.0001f)))), 0f)) * ((_e2518 * _e2518) / max(_e2512, 0.0001f))) * select(0f, 1f, ((_e2393 > 0.5f) && (_e2393 < 1.5f))))) + (((((_e2428 * _e2398) * max(dot(_e102, _e2529), 0f)) * ((_e2540 * _e2540) / max(_e2534, 0.0001f))) * (_e2548 * _e2548)) * select(0f, 1f, (_e2393 > 1.5f))))));
    }
    if (14f < _e125) {
        let _e2567 = lights.member[i32(14f)].member;
        let _e2572 = lights.member[i32(14f)].member_1;
        let _e2577 = lights.member[i32(14f)].member_2;
        let _e2582 = lights.member[i32(14f)].member_3;
        let _e2587 = lights.member[i32(14f)].member_4;
        let _e2592 = lights.member[i32(14f)].member_5;
        let _e2597 = lights.member[i32(14f)].member_6;
        let _e2602 = lights.member[i32(14f)].member_8;
        let _e2607 = select(0f, 1f, (_e2567 < 0.5f));
        let _e2611 = ((normalize(_e2597) * _e2607) + (normalize((_e2587 - _e116)) * (1f - _e2607)));
        let _e2613 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e2615 = normalize((_e120 + _e2611));
        let _e2617 = max(dot(_e102, _e2615), 0f);
        let _e2619 = max(dot(_e102, _e120), 0.00001f);
        let _e2621 = max(dot(_e102, _e2611), 0.00001f);
        let _e2624 = (_e67.z * _e67.z);
        let _e2625 = (_e2624 * _e2624);
        let _e2629 = (((_e2617 * _e2617) * (_e2625 - 1f)) + 1f);
        let _e2636 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e2637 = (1f - _e2636);
        let _e2656 = (_e2613 + ((vec3<f32>(1f, 1f, 1f) - _e2613) * pow(clamp((1f - max(dot(_e120, _e2615), 0f)), 0f, 1f), 5f)));
        let _e2679 = (_e2587 - _e116);
        let _e2680 = length(_e2679);
        let _e2686 = (_e2680 * _e2680);
        let _e2689 = (_e2686 / max((_e2577 * _e2577), 0.0001f));
        let _e2692 = clamp((1f - (_e2689 * _e2689)), 0f, 1f);
        let _e2699 = (_e2587 - _e116);
        let _e2700 = length(_e2699);
        let _e2703 = (_e2699 / vec3(max(_e2700, 0.0001f)));
        let _e2708 = (_e2700 * _e2700);
        let _e2711 = (_e2708 / max((_e2577 * _e2577), 0.0001f));
        let _e2714 = clamp((1f - (_e2711 * _e2711)), 0f, 1f);
        let _e2722 = clamp(((dot(normalize(_e2597), _e2703) - _e2592) / max((_e2582 - _e2592), 0.0001f)), 0f, 1f);
        let _e2733 = local;
        local = (_e2733 + (((((((vec3<f32>(1f, 1f, 1f) - _e2656) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e2656 * ((_e2625 * 0.31830987f) / ((_e2629 * _e2629) + 0.00001f))) * (0.5f / (((_e2621 * sqrt((((_e2619 * _e2619) * _e2637) + _e2636))) + (_e2619 * sqrt((((_e2621 * _e2621) * _e2637) + _e2636)))) + 0.00001f)))) * _e2621) * (((((_e2602 * _e2572) * max(dot(_e102, normalize(_e2597)), 0f)) * select(0f, 1f, (_e2567 < 0.5f))) + ((((_e2602 * _e2572) * max(dot(_e102, (_e2679 / vec3(max(_e2680, 0.0001f)))), 0f)) * ((_e2692 * _e2692) / max(_e2686, 0.0001f))) * select(0f, 1f, ((_e2567 > 0.5f) && (_e2567 < 1.5f))))) + (((((_e2602 * _e2572) * max(dot(_e102, _e2703), 0f)) * ((_e2714 * _e2714) / max(_e2708, 0.0001f))) * (_e2722 * _e2722)) * select(0f, 1f, (_e2567 > 1.5f))))));
    }
    if (15f < _e125) {
        let _e2741 = lights.member[i32(15f)].member;
        let _e2746 = lights.member[i32(15f)].member_1;
        let _e2751 = lights.member[i32(15f)].member_2;
        let _e2756 = lights.member[i32(15f)].member_3;
        let _e2761 = lights.member[i32(15f)].member_4;
        let _e2766 = lights.member[i32(15f)].member_5;
        let _e2771 = lights.member[i32(15f)].member_6;
        let _e2776 = lights.member[i32(15f)].member_8;
        let _e2781 = select(0f, 1f, (_e2741 < 0.5f));
        let _e2785 = ((normalize(_e2771) * _e2781) + (normalize((_e2761 - _e116)) * (1f - _e2781)));
        let _e2787 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
        let _e2789 = normalize((_e120 + _e2785));
        let _e2791 = max(dot(_e102, _e2789), 0f);
        let _e2793 = max(dot(_e102, _e120), 0.00001f);
        let _e2795 = max(dot(_e102, _e2785), 0.00001f);
        let _e2798 = (_e67.z * _e67.z);
        let _e2799 = (_e2798 * _e2798);
        let _e2803 = (((_e2791 * _e2791) * (_e2799 - 1f)) + 1f);
        let _e2810 = (((_e67.z * _e67.z) * _e67.z) * _e67.z);
        let _e2811 = (1f - _e2810);
        let _e2830 = (_e2787 + ((vec3<f32>(1f, 1f, 1f) - _e2787) * pow(clamp((1f - max(dot(_e120, _e2789), 0f)), 0f, 1f), 5f)));
        let _e2853 = (_e2761 - _e116);
        let _e2854 = length(_e2853);
        let _e2860 = (_e2854 * _e2854);
        let _e2863 = (_e2860 / max((_e2751 * _e2751), 0.0001f));
        let _e2866 = clamp((1f - (_e2863 * _e2863)), 0f, 1f);
        let _e2873 = (_e2761 - _e116);
        let _e2874 = length(_e2873);
        let _e2877 = (_e2873 / vec3(max(_e2874, 0.0001f)));
        let _e2882 = (_e2874 * _e2874);
        let _e2885 = (_e2882 / max((_e2751 * _e2751), 0.0001f));
        let _e2888 = clamp((1f - (_e2885 * _e2885)), 0f, 1f);
        let _e2896 = clamp(((dot(normalize(_e2771), _e2877) - _e2766) / max((_e2756 - _e2766), 0.0001f)), 0f, 1f);
        let _e2907 = local;
        local = (_e2907 + (((((((vec3<f32>(1f, 1f, 1f) - _e2830) * (1f - _e65.w)) * _e73) * 0.31830987f) + ((_e2830 * ((_e2799 * 0.31830987f) / ((_e2803 * _e2803) + 0.00001f))) * (0.5f / (((_e2795 * sqrt((((_e2793 * _e2793) * _e2811) + _e2810))) + (_e2793 * sqrt((((_e2795 * _e2795) * _e2811) + _e2810)))) + 0.00001f)))) * _e2795) * (((((_e2776 * _e2746) * max(dot(_e102, normalize(_e2771)), 0f)) * select(0f, 1f, (_e2741 < 0.5f))) + ((((_e2776 * _e2746) * max(dot(_e102, (_e2853 / vec3(max(_e2854, 0.0001f)))), 0f)) * ((_e2866 * _e2866) / max(_e2860, 0.0001f))) * select(0f, 1f, ((_e2741 > 0.5f) && (_e2741 < 1.5f))))) + (((((_e2776 * _e2746) * max(dot(_e102, _e2877), 0f)) * ((_e2888 * _e2888) / max(_e2882, 0.0001f))) * (_e2896 * _e2896)) * select(0f, 1f, (_e2741 > 1.5f))))));
    }
    let _e2913 = textureSampleLevel(env_specular_u002e_texture, env_specular_u002e_sampler, reflect((_e120 * -1f), _e102), (_e67.z * 8f));
    let _e2915 = textureSample(env_irradiance_u002e_texture, env_irradiance_u002e_sampler, _e102);
    let _e2918 = textureSample(brdf_lut_u002e_texture, brdf_lut_u002e_sampler, vec2<f32>(_e122, _e67.z));
    let _e2919 = _e2918.xy;
    let _e2921 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e73, vec3(_e65.w));
    let _e2938 = (((_e2921 + ((max(vec3((1f - _e67.z)), _e2921) - _e2921) * pow(clamp((1f - _e122), 0f, 1f), 5f))) * _e2919.x) + vec3(_e2919.y));
    let _e2941 = (_e2921 + ((vec3<f32>(1f, 1f, 1f) - _e2921) * 0.04761905f));
    let _e2942 = (1f - (_e2919.x + _e2919.y));
    let _e2948 = (_e2938 + (((_e2938 * _e2941) / (vec3<f32>(1f, 1f, 1f) - (_e2941 * _e2942))) * _e2942));
    let _e2956 = local;
    let _e2958 = ((_e2956 + (((((vec3<f32>(1f, 1f, 1f) - _e2948) * (1f - _e65.w)) * _e73) * _e2915.xyz) + (_e2948 * _e2913.xyz))) + _e69.xyz);
    let _e2971 = pow(clamp(((_e2958 * ((_e2958 * 2.51f) + vec3(0.03f))) / ((_e2958 * ((_e2958 * 2.43f) + vec3(0.59f))) + vec3(0.14f))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f)), vec3<f32>(0.45454547f, 0.45454547f, 0.45454547f));
    color = vec4<f32>(_e2971.x, _e2971.y, _e2971.z, 1f);
    return;
}

@fragment 
fn main(@location(0) frag_uv: vec2<f32>) -> @location(0) vec4<f32> {
    frag_uv_1 = frag_uv;
    main_1();
    let _e3 = color;
    return _e3;
}
