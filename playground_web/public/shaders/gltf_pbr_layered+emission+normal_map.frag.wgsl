struct SceneLight {
    view_pos: vec3<f32>,
    light_count: i32,
}

struct Material {
    base_color_factor: vec4<f32>,
    emissive_factor: vec3<f32>,
    metallic_factor: f32,
    roughness_factor: f32,
    emissive_strength: f32,
    ior: f32,
    clearcoat_factor: f32,
    clearcoat_roughness_factor: f32,
    sheen_roughness_factor: f32,
    transmission_factor: f32,
    sheen_color_factor: vec3<f32>,
    base_color_uv_st: vec4<f32>,
    normal_uv_st: vec4<f32>,
    mr_uv_st: vec4<f32>,
    base_color_uv_rot: f32,
    normal_uv_rot: f32,
    mr_uv_rot: f32,
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

var<private> world_pos_1: vec3<f32>;
var<private> world_normal_1: vec3<f32>;
var<private> frag_uv_1: vec2<f32>;
var<private> color: vec4<f32>;
@group(1) @binding(0) 
var<uniform> global: SceneLight;
@group(1) @binding(1) 
var<uniform> global_1: Material;
@group(1) @binding(2) 
var base_color_tex_u002e_sampler: sampler;
@group(1) @binding(3) 
var base_color_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(4) 
var metallic_roughness_tex_u002e_sampler: sampler;
@group(1) @binding(5) 
var metallic_roughness_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(6) 
var occlusion_tex_u002e_sampler: sampler;
@group(1) @binding(7) 
var occlusion_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(8) 
var env_specular_u002e_sampler: sampler;
@group(1) @binding(9) 
var env_specular_u002e_texture: texture_cube<f32>;
@group(1) @binding(10) 
var env_irradiance_u002e_sampler: sampler;
@group(1) @binding(11) 
var env_irradiance_u002e_texture: texture_cube<f32>;
@group(1) @binding(12) 
var brdf_lut_u002e_sampler: sampler;
@group(1) @binding(13) 
var brdf_lut_u002e_texture: texture_2d<f32>;
@group(1) @binding(14) 
var<storage> lights: type_13;

fn main_1() {
    var local: vec3<f32>;
    var local_1: vec3<f32>;

    let _e76 = frag_uv_1;
    let _e77 = world_normal_1;
    let _e78 = normalize(_e77);
    let _e80 = global.view_pos;
    let _e81 = world_pos_1;
    let _e83 = normalize((_e80 - _e81));
    let _e85 = max(dot(_e78, _e83), 0.001f);
    let _e87 = global_1.base_color_uv_st;
    let _e89 = global_1.base_color_uv_rot;
    let _e90 = cos(_e89);
    let _e91 = sin(_e89);
    let _e111 = textureSample(base_color_tex_u002e_texture, base_color_tex_u002e_sampler, ((vec2<f32>(((_e76.x * _e90) - (_e76.y * _e91)), ((_e76.x * _e91) + (_e76.y * _e90))) * vec2<f32>(_e87.z, _e87.w)) + vec2<f32>(_e87.x, _e87.y)));
    let _e115 = global_1.base_color_factor;
    let _e117 = (pow(_e111.xyz, vec3<f32>(2.2f, 2.2f, 2.2f)) * _e115.xyz);
    let _e119 = global_1.mr_uv_st;
    let _e121 = global_1.mr_uv_rot;
    let _e122 = cos(_e121);
    let _e123 = sin(_e121);
    let _e143 = textureSample(metallic_roughness_tex_u002e_texture, metallic_roughness_tex_u002e_sampler, ((vec2<f32>(((_e76.x * _e122) - (_e76.y * _e123)), ((_e76.x * _e123) + (_e76.y * _e122))) * vec2<f32>(_e119.z, _e119.w)) + vec2<f32>(_e119.x, _e119.y)));
    let _e146 = global_1.roughness_factor;
    let _e147 = (_e143.y * _e146);
    let _e148 = cos(_e121);
    let _e149 = sin(_e121);
    let _e169 = textureSample(metallic_roughness_tex_u002e_texture, metallic_roughness_tex_u002e_sampler, ((vec2<f32>(((_e76.x * _e148) - (_e76.y * _e149)), ((_e76.x * _e149) + (_e76.y * _e148))) * vec2<f32>(_e119.z, _e119.w)) + vec2<f32>(_e119.x, _e119.y)));
    let _e172 = global_1.metallic_factor;
    let _e173 = (_e169.z * _e172);
    let _e175 = global.light_count;
    let _e176 = f32(_e175);
    local = vec3<f32>(0f, 0f, 0f);
    if (0f < _e176) {
        let _e182 = lights.member[i32(0f)].member;
        let _e187 = lights.member[i32(0f)].member_1;
        let _e192 = lights.member[i32(0f)].member_2;
        let _e197 = lights.member[i32(0f)].member_3;
        let _e202 = lights.member[i32(0f)].member_4;
        let _e207 = lights.member[i32(0f)].member_5;
        let _e212 = lights.member[i32(0f)].member_6;
        let _e217 = lights.member[i32(0f)].member_8;
        let _e218 = world_pos_1;
        let _e223 = select(0f, 1f, (_e182 < 0.5f));
        let _e227 = ((normalize(_e212) * _e223) + (normalize((_e202 - _e218)) * (1f - _e223)));
        let _e229 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e231 = normalize((_e83 + _e227));
        let _e233 = max(dot(_e78, _e231), 0f);
        let _e235 = max(dot(_e78, _e83), 0.00001f);
        let _e237 = max(dot(_e78, _e227), 0.00001f);
        let _e240 = (_e147 * _e147);
        let _e241 = (_e240 * _e240);
        let _e245 = (((_e233 * _e233) * (_e241 - 1f)) + 1f);
        let _e252 = (((_e147 * _e147) * _e147) * _e147);
        let _e253 = (1f - _e252);
        let _e272 = (_e229 + ((vec3<f32>(1f, 1f, 1f) - _e229) * pow(clamp((1f - max(dot(_e83, _e231), 0f)), 0f, 1f), 5f)));
        let _e282 = world_pos_1;
        let _e296 = (_e202 - _e282);
        let _e297 = length(_e296);
        let _e303 = (_e297 * _e297);
        let _e306 = (_e303 / max((_e192 * _e192), 0.0001f));
        let _e309 = clamp((1f - (_e306 * _e306)), 0f, 1f);
        let _e316 = (_e202 - _e282);
        let _e317 = length(_e316);
        let _e320 = (_e316 / vec3(max(_e317, 0.0001f)));
        let _e325 = (_e317 * _e317);
        let _e328 = (_e325 / max((_e192 * _e192), 0.0001f));
        let _e331 = clamp((1f - (_e328 * _e328)), 0f, 1f);
        let _e339 = clamp(((dot(normalize(_e212), _e320) - _e207) / max((_e197 - _e207), 0.0001f)), 0f, 1f);
        let _e350 = local;
        local = (_e350 + (((((((vec3<f32>(1f, 1f, 1f) - _e272) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e272 * ((_e241 * 0.31830987f) / ((_e245 * _e245) + 0.00001f))) * (0.5f / (((_e237 * sqrt((((_e235 * _e235) * _e253) + _e252))) + (_e235 * sqrt((((_e237 * _e237) * _e253) + _e252)))) + 0.00001f)))) * _e237) * (((((_e217 * _e187) * max(dot(_e78, normalize(_e212)), 0f)) * select(0f, 1f, (_e182 < 0.5f))) + ((((_e217 * _e187) * max(dot(_e78, (_e296 / vec3(max(_e297, 0.0001f)))), 0f)) * ((_e309 * _e309) / max(_e303, 0.0001f))) * select(0f, 1f, ((_e182 > 0.5f) && (_e182 < 1.5f))))) + (((((_e217 * _e187) * max(dot(_e78, _e320), 0f)) * ((_e331 * _e331) / max(_e325, 0.0001f))) * (_e339 * _e339)) * select(0f, 1f, (_e182 > 1.5f))))));
    }
    if (1f < _e176) {
        let _e358 = lights.member[i32(1f)].member;
        let _e363 = lights.member[i32(1f)].member_1;
        let _e368 = lights.member[i32(1f)].member_2;
        let _e373 = lights.member[i32(1f)].member_3;
        let _e378 = lights.member[i32(1f)].member_4;
        let _e383 = lights.member[i32(1f)].member_5;
        let _e388 = lights.member[i32(1f)].member_6;
        let _e393 = lights.member[i32(1f)].member_8;
        let _e394 = world_pos_1;
        let _e399 = select(0f, 1f, (_e358 < 0.5f));
        let _e403 = ((normalize(_e388) * _e399) + (normalize((_e378 - _e394)) * (1f - _e399)));
        let _e405 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e407 = normalize((_e83 + _e403));
        let _e409 = max(dot(_e78, _e407), 0f);
        let _e411 = max(dot(_e78, _e83), 0.00001f);
        let _e413 = max(dot(_e78, _e403), 0.00001f);
        let _e416 = (_e147 * _e147);
        let _e417 = (_e416 * _e416);
        let _e421 = (((_e409 * _e409) * (_e417 - 1f)) + 1f);
        let _e428 = (((_e147 * _e147) * _e147) * _e147);
        let _e429 = (1f - _e428);
        let _e448 = (_e405 + ((vec3<f32>(1f, 1f, 1f) - _e405) * pow(clamp((1f - max(dot(_e83, _e407), 0f)), 0f, 1f), 5f)));
        let _e458 = world_pos_1;
        let _e472 = (_e378 - _e458);
        let _e473 = length(_e472);
        let _e479 = (_e473 * _e473);
        let _e482 = (_e479 / max((_e368 * _e368), 0.0001f));
        let _e485 = clamp((1f - (_e482 * _e482)), 0f, 1f);
        let _e492 = (_e378 - _e458);
        let _e493 = length(_e492);
        let _e496 = (_e492 / vec3(max(_e493, 0.0001f)));
        let _e501 = (_e493 * _e493);
        let _e504 = (_e501 / max((_e368 * _e368), 0.0001f));
        let _e507 = clamp((1f - (_e504 * _e504)), 0f, 1f);
        let _e515 = clamp(((dot(normalize(_e388), _e496) - _e383) / max((_e373 - _e383), 0.0001f)), 0f, 1f);
        let _e526 = local;
        local = (_e526 + (((((((vec3<f32>(1f, 1f, 1f) - _e448) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e448 * ((_e417 * 0.31830987f) / ((_e421 * _e421) + 0.00001f))) * (0.5f / (((_e413 * sqrt((((_e411 * _e411) * _e429) + _e428))) + (_e411 * sqrt((((_e413 * _e413) * _e429) + _e428)))) + 0.00001f)))) * _e413) * (((((_e393 * _e363) * max(dot(_e78, normalize(_e388)), 0f)) * select(0f, 1f, (_e358 < 0.5f))) + ((((_e393 * _e363) * max(dot(_e78, (_e472 / vec3(max(_e473, 0.0001f)))), 0f)) * ((_e485 * _e485) / max(_e479, 0.0001f))) * select(0f, 1f, ((_e358 > 0.5f) && (_e358 < 1.5f))))) + (((((_e393 * _e363) * max(dot(_e78, _e496), 0f)) * ((_e507 * _e507) / max(_e501, 0.0001f))) * (_e515 * _e515)) * select(0f, 1f, (_e358 > 1.5f))))));
    }
    if (2f < _e176) {
        let _e534 = lights.member[i32(2f)].member;
        let _e539 = lights.member[i32(2f)].member_1;
        let _e544 = lights.member[i32(2f)].member_2;
        let _e549 = lights.member[i32(2f)].member_3;
        let _e554 = lights.member[i32(2f)].member_4;
        let _e559 = lights.member[i32(2f)].member_5;
        let _e564 = lights.member[i32(2f)].member_6;
        let _e569 = lights.member[i32(2f)].member_8;
        let _e570 = world_pos_1;
        let _e575 = select(0f, 1f, (_e534 < 0.5f));
        let _e579 = ((normalize(_e564) * _e575) + (normalize((_e554 - _e570)) * (1f - _e575)));
        let _e581 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e583 = normalize((_e83 + _e579));
        let _e585 = max(dot(_e78, _e583), 0f);
        let _e587 = max(dot(_e78, _e83), 0.00001f);
        let _e589 = max(dot(_e78, _e579), 0.00001f);
        let _e592 = (_e147 * _e147);
        let _e593 = (_e592 * _e592);
        let _e597 = (((_e585 * _e585) * (_e593 - 1f)) + 1f);
        let _e604 = (((_e147 * _e147) * _e147) * _e147);
        let _e605 = (1f - _e604);
        let _e624 = (_e581 + ((vec3<f32>(1f, 1f, 1f) - _e581) * pow(clamp((1f - max(dot(_e83, _e583), 0f)), 0f, 1f), 5f)));
        let _e634 = world_pos_1;
        let _e648 = (_e554 - _e634);
        let _e649 = length(_e648);
        let _e655 = (_e649 * _e649);
        let _e658 = (_e655 / max((_e544 * _e544), 0.0001f));
        let _e661 = clamp((1f - (_e658 * _e658)), 0f, 1f);
        let _e668 = (_e554 - _e634);
        let _e669 = length(_e668);
        let _e672 = (_e668 / vec3(max(_e669, 0.0001f)));
        let _e677 = (_e669 * _e669);
        let _e680 = (_e677 / max((_e544 * _e544), 0.0001f));
        let _e683 = clamp((1f - (_e680 * _e680)), 0f, 1f);
        let _e691 = clamp(((dot(normalize(_e564), _e672) - _e559) / max((_e549 - _e559), 0.0001f)), 0f, 1f);
        let _e702 = local;
        local = (_e702 + (((((((vec3<f32>(1f, 1f, 1f) - _e624) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e624 * ((_e593 * 0.31830987f) / ((_e597 * _e597) + 0.00001f))) * (0.5f / (((_e589 * sqrt((((_e587 * _e587) * _e605) + _e604))) + (_e587 * sqrt((((_e589 * _e589) * _e605) + _e604)))) + 0.00001f)))) * _e589) * (((((_e569 * _e539) * max(dot(_e78, normalize(_e564)), 0f)) * select(0f, 1f, (_e534 < 0.5f))) + ((((_e569 * _e539) * max(dot(_e78, (_e648 / vec3(max(_e649, 0.0001f)))), 0f)) * ((_e661 * _e661) / max(_e655, 0.0001f))) * select(0f, 1f, ((_e534 > 0.5f) && (_e534 < 1.5f))))) + (((((_e569 * _e539) * max(dot(_e78, _e672), 0f)) * ((_e683 * _e683) / max(_e677, 0.0001f))) * (_e691 * _e691)) * select(0f, 1f, (_e534 > 1.5f))))));
    }
    if (3f < _e176) {
        let _e710 = lights.member[i32(3f)].member;
        let _e715 = lights.member[i32(3f)].member_1;
        let _e720 = lights.member[i32(3f)].member_2;
        let _e725 = lights.member[i32(3f)].member_3;
        let _e730 = lights.member[i32(3f)].member_4;
        let _e735 = lights.member[i32(3f)].member_5;
        let _e740 = lights.member[i32(3f)].member_6;
        let _e745 = lights.member[i32(3f)].member_8;
        let _e746 = world_pos_1;
        let _e751 = select(0f, 1f, (_e710 < 0.5f));
        let _e755 = ((normalize(_e740) * _e751) + (normalize((_e730 - _e746)) * (1f - _e751)));
        let _e757 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e759 = normalize((_e83 + _e755));
        let _e761 = max(dot(_e78, _e759), 0f);
        let _e763 = max(dot(_e78, _e83), 0.00001f);
        let _e765 = max(dot(_e78, _e755), 0.00001f);
        let _e768 = (_e147 * _e147);
        let _e769 = (_e768 * _e768);
        let _e773 = (((_e761 * _e761) * (_e769 - 1f)) + 1f);
        let _e780 = (((_e147 * _e147) * _e147) * _e147);
        let _e781 = (1f - _e780);
        let _e800 = (_e757 + ((vec3<f32>(1f, 1f, 1f) - _e757) * pow(clamp((1f - max(dot(_e83, _e759), 0f)), 0f, 1f), 5f)));
        let _e810 = world_pos_1;
        let _e824 = (_e730 - _e810);
        let _e825 = length(_e824);
        let _e831 = (_e825 * _e825);
        let _e834 = (_e831 / max((_e720 * _e720), 0.0001f));
        let _e837 = clamp((1f - (_e834 * _e834)), 0f, 1f);
        let _e844 = (_e730 - _e810);
        let _e845 = length(_e844);
        let _e848 = (_e844 / vec3(max(_e845, 0.0001f)));
        let _e853 = (_e845 * _e845);
        let _e856 = (_e853 / max((_e720 * _e720), 0.0001f));
        let _e859 = clamp((1f - (_e856 * _e856)), 0f, 1f);
        let _e867 = clamp(((dot(normalize(_e740), _e848) - _e735) / max((_e725 - _e735), 0.0001f)), 0f, 1f);
        let _e878 = local;
        local = (_e878 + (((((((vec3<f32>(1f, 1f, 1f) - _e800) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e800 * ((_e769 * 0.31830987f) / ((_e773 * _e773) + 0.00001f))) * (0.5f / (((_e765 * sqrt((((_e763 * _e763) * _e781) + _e780))) + (_e763 * sqrt((((_e765 * _e765) * _e781) + _e780)))) + 0.00001f)))) * _e765) * (((((_e745 * _e715) * max(dot(_e78, normalize(_e740)), 0f)) * select(0f, 1f, (_e710 < 0.5f))) + ((((_e745 * _e715) * max(dot(_e78, (_e824 / vec3(max(_e825, 0.0001f)))), 0f)) * ((_e837 * _e837) / max(_e831, 0.0001f))) * select(0f, 1f, ((_e710 > 0.5f) && (_e710 < 1.5f))))) + (((((_e745 * _e715) * max(dot(_e78, _e848), 0f)) * ((_e859 * _e859) / max(_e853, 0.0001f))) * (_e867 * _e867)) * select(0f, 1f, (_e710 > 1.5f))))));
    }
    if (4f < _e176) {
        let _e886 = lights.member[i32(4f)].member;
        let _e891 = lights.member[i32(4f)].member_1;
        let _e896 = lights.member[i32(4f)].member_2;
        let _e901 = lights.member[i32(4f)].member_3;
        let _e906 = lights.member[i32(4f)].member_4;
        let _e911 = lights.member[i32(4f)].member_5;
        let _e916 = lights.member[i32(4f)].member_6;
        let _e921 = lights.member[i32(4f)].member_8;
        let _e922 = world_pos_1;
        let _e927 = select(0f, 1f, (_e886 < 0.5f));
        let _e931 = ((normalize(_e916) * _e927) + (normalize((_e906 - _e922)) * (1f - _e927)));
        let _e933 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e935 = normalize((_e83 + _e931));
        let _e937 = max(dot(_e78, _e935), 0f);
        let _e939 = max(dot(_e78, _e83), 0.00001f);
        let _e941 = max(dot(_e78, _e931), 0.00001f);
        let _e944 = (_e147 * _e147);
        let _e945 = (_e944 * _e944);
        let _e949 = (((_e937 * _e937) * (_e945 - 1f)) + 1f);
        let _e956 = (((_e147 * _e147) * _e147) * _e147);
        let _e957 = (1f - _e956);
        let _e976 = (_e933 + ((vec3<f32>(1f, 1f, 1f) - _e933) * pow(clamp((1f - max(dot(_e83, _e935), 0f)), 0f, 1f), 5f)));
        let _e986 = world_pos_1;
        let _e1000 = (_e906 - _e986);
        let _e1001 = length(_e1000);
        let _e1007 = (_e1001 * _e1001);
        let _e1010 = (_e1007 / max((_e896 * _e896), 0.0001f));
        let _e1013 = clamp((1f - (_e1010 * _e1010)), 0f, 1f);
        let _e1020 = (_e906 - _e986);
        let _e1021 = length(_e1020);
        let _e1024 = (_e1020 / vec3(max(_e1021, 0.0001f)));
        let _e1029 = (_e1021 * _e1021);
        let _e1032 = (_e1029 / max((_e896 * _e896), 0.0001f));
        let _e1035 = clamp((1f - (_e1032 * _e1032)), 0f, 1f);
        let _e1043 = clamp(((dot(normalize(_e916), _e1024) - _e911) / max((_e901 - _e911), 0.0001f)), 0f, 1f);
        let _e1054 = local;
        local = (_e1054 + (((((((vec3<f32>(1f, 1f, 1f) - _e976) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e976 * ((_e945 * 0.31830987f) / ((_e949 * _e949) + 0.00001f))) * (0.5f / (((_e941 * sqrt((((_e939 * _e939) * _e957) + _e956))) + (_e939 * sqrt((((_e941 * _e941) * _e957) + _e956)))) + 0.00001f)))) * _e941) * (((((_e921 * _e891) * max(dot(_e78, normalize(_e916)), 0f)) * select(0f, 1f, (_e886 < 0.5f))) + ((((_e921 * _e891) * max(dot(_e78, (_e1000 / vec3(max(_e1001, 0.0001f)))), 0f)) * ((_e1013 * _e1013) / max(_e1007, 0.0001f))) * select(0f, 1f, ((_e886 > 0.5f) && (_e886 < 1.5f))))) + (((((_e921 * _e891) * max(dot(_e78, _e1024), 0f)) * ((_e1035 * _e1035) / max(_e1029, 0.0001f))) * (_e1043 * _e1043)) * select(0f, 1f, (_e886 > 1.5f))))));
    }
    if (5f < _e176) {
        let _e1062 = lights.member[i32(5f)].member;
        let _e1067 = lights.member[i32(5f)].member_1;
        let _e1072 = lights.member[i32(5f)].member_2;
        let _e1077 = lights.member[i32(5f)].member_3;
        let _e1082 = lights.member[i32(5f)].member_4;
        let _e1087 = lights.member[i32(5f)].member_5;
        let _e1092 = lights.member[i32(5f)].member_6;
        let _e1097 = lights.member[i32(5f)].member_8;
        let _e1098 = world_pos_1;
        let _e1103 = select(0f, 1f, (_e1062 < 0.5f));
        let _e1107 = ((normalize(_e1092) * _e1103) + (normalize((_e1082 - _e1098)) * (1f - _e1103)));
        let _e1109 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e1111 = normalize((_e83 + _e1107));
        let _e1113 = max(dot(_e78, _e1111), 0f);
        let _e1115 = max(dot(_e78, _e83), 0.00001f);
        let _e1117 = max(dot(_e78, _e1107), 0.00001f);
        let _e1120 = (_e147 * _e147);
        let _e1121 = (_e1120 * _e1120);
        let _e1125 = (((_e1113 * _e1113) * (_e1121 - 1f)) + 1f);
        let _e1132 = (((_e147 * _e147) * _e147) * _e147);
        let _e1133 = (1f - _e1132);
        let _e1152 = (_e1109 + ((vec3<f32>(1f, 1f, 1f) - _e1109) * pow(clamp((1f - max(dot(_e83, _e1111), 0f)), 0f, 1f), 5f)));
        let _e1162 = world_pos_1;
        let _e1176 = (_e1082 - _e1162);
        let _e1177 = length(_e1176);
        let _e1183 = (_e1177 * _e1177);
        let _e1186 = (_e1183 / max((_e1072 * _e1072), 0.0001f));
        let _e1189 = clamp((1f - (_e1186 * _e1186)), 0f, 1f);
        let _e1196 = (_e1082 - _e1162);
        let _e1197 = length(_e1196);
        let _e1200 = (_e1196 / vec3(max(_e1197, 0.0001f)));
        let _e1205 = (_e1197 * _e1197);
        let _e1208 = (_e1205 / max((_e1072 * _e1072), 0.0001f));
        let _e1211 = clamp((1f - (_e1208 * _e1208)), 0f, 1f);
        let _e1219 = clamp(((dot(normalize(_e1092), _e1200) - _e1087) / max((_e1077 - _e1087), 0.0001f)), 0f, 1f);
        let _e1230 = local;
        local = (_e1230 + (((((((vec3<f32>(1f, 1f, 1f) - _e1152) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e1152 * ((_e1121 * 0.31830987f) / ((_e1125 * _e1125) + 0.00001f))) * (0.5f / (((_e1117 * sqrt((((_e1115 * _e1115) * _e1133) + _e1132))) + (_e1115 * sqrt((((_e1117 * _e1117) * _e1133) + _e1132)))) + 0.00001f)))) * _e1117) * (((((_e1097 * _e1067) * max(dot(_e78, normalize(_e1092)), 0f)) * select(0f, 1f, (_e1062 < 0.5f))) + ((((_e1097 * _e1067) * max(dot(_e78, (_e1176 / vec3(max(_e1177, 0.0001f)))), 0f)) * ((_e1189 * _e1189) / max(_e1183, 0.0001f))) * select(0f, 1f, ((_e1062 > 0.5f) && (_e1062 < 1.5f))))) + (((((_e1097 * _e1067) * max(dot(_e78, _e1200), 0f)) * ((_e1211 * _e1211) / max(_e1205, 0.0001f))) * (_e1219 * _e1219)) * select(0f, 1f, (_e1062 > 1.5f))))));
    }
    if (6f < _e176) {
        let _e1238 = lights.member[i32(6f)].member;
        let _e1243 = lights.member[i32(6f)].member_1;
        let _e1248 = lights.member[i32(6f)].member_2;
        let _e1253 = lights.member[i32(6f)].member_3;
        let _e1258 = lights.member[i32(6f)].member_4;
        let _e1263 = lights.member[i32(6f)].member_5;
        let _e1268 = lights.member[i32(6f)].member_6;
        let _e1273 = lights.member[i32(6f)].member_8;
        let _e1274 = world_pos_1;
        let _e1279 = select(0f, 1f, (_e1238 < 0.5f));
        let _e1283 = ((normalize(_e1268) * _e1279) + (normalize((_e1258 - _e1274)) * (1f - _e1279)));
        let _e1285 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e1287 = normalize((_e83 + _e1283));
        let _e1289 = max(dot(_e78, _e1287), 0f);
        let _e1291 = max(dot(_e78, _e83), 0.00001f);
        let _e1293 = max(dot(_e78, _e1283), 0.00001f);
        let _e1296 = (_e147 * _e147);
        let _e1297 = (_e1296 * _e1296);
        let _e1301 = (((_e1289 * _e1289) * (_e1297 - 1f)) + 1f);
        let _e1308 = (((_e147 * _e147) * _e147) * _e147);
        let _e1309 = (1f - _e1308);
        let _e1328 = (_e1285 + ((vec3<f32>(1f, 1f, 1f) - _e1285) * pow(clamp((1f - max(dot(_e83, _e1287), 0f)), 0f, 1f), 5f)));
        let _e1338 = world_pos_1;
        let _e1352 = (_e1258 - _e1338);
        let _e1353 = length(_e1352);
        let _e1359 = (_e1353 * _e1353);
        let _e1362 = (_e1359 / max((_e1248 * _e1248), 0.0001f));
        let _e1365 = clamp((1f - (_e1362 * _e1362)), 0f, 1f);
        let _e1372 = (_e1258 - _e1338);
        let _e1373 = length(_e1372);
        let _e1376 = (_e1372 / vec3(max(_e1373, 0.0001f)));
        let _e1381 = (_e1373 * _e1373);
        let _e1384 = (_e1381 / max((_e1248 * _e1248), 0.0001f));
        let _e1387 = clamp((1f - (_e1384 * _e1384)), 0f, 1f);
        let _e1395 = clamp(((dot(normalize(_e1268), _e1376) - _e1263) / max((_e1253 - _e1263), 0.0001f)), 0f, 1f);
        let _e1406 = local;
        local = (_e1406 + (((((((vec3<f32>(1f, 1f, 1f) - _e1328) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e1328 * ((_e1297 * 0.31830987f) / ((_e1301 * _e1301) + 0.00001f))) * (0.5f / (((_e1293 * sqrt((((_e1291 * _e1291) * _e1309) + _e1308))) + (_e1291 * sqrt((((_e1293 * _e1293) * _e1309) + _e1308)))) + 0.00001f)))) * _e1293) * (((((_e1273 * _e1243) * max(dot(_e78, normalize(_e1268)), 0f)) * select(0f, 1f, (_e1238 < 0.5f))) + ((((_e1273 * _e1243) * max(dot(_e78, (_e1352 / vec3(max(_e1353, 0.0001f)))), 0f)) * ((_e1365 * _e1365) / max(_e1359, 0.0001f))) * select(0f, 1f, ((_e1238 > 0.5f) && (_e1238 < 1.5f))))) + (((((_e1273 * _e1243) * max(dot(_e78, _e1376), 0f)) * ((_e1387 * _e1387) / max(_e1381, 0.0001f))) * (_e1395 * _e1395)) * select(0f, 1f, (_e1238 > 1.5f))))));
    }
    if (7f < _e176) {
        let _e1414 = lights.member[i32(7f)].member;
        let _e1419 = lights.member[i32(7f)].member_1;
        let _e1424 = lights.member[i32(7f)].member_2;
        let _e1429 = lights.member[i32(7f)].member_3;
        let _e1434 = lights.member[i32(7f)].member_4;
        let _e1439 = lights.member[i32(7f)].member_5;
        let _e1444 = lights.member[i32(7f)].member_6;
        let _e1449 = lights.member[i32(7f)].member_8;
        let _e1450 = world_pos_1;
        let _e1455 = select(0f, 1f, (_e1414 < 0.5f));
        let _e1459 = ((normalize(_e1444) * _e1455) + (normalize((_e1434 - _e1450)) * (1f - _e1455)));
        let _e1461 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e1463 = normalize((_e83 + _e1459));
        let _e1465 = max(dot(_e78, _e1463), 0f);
        let _e1467 = max(dot(_e78, _e83), 0.00001f);
        let _e1469 = max(dot(_e78, _e1459), 0.00001f);
        let _e1472 = (_e147 * _e147);
        let _e1473 = (_e1472 * _e1472);
        let _e1477 = (((_e1465 * _e1465) * (_e1473 - 1f)) + 1f);
        let _e1484 = (((_e147 * _e147) * _e147) * _e147);
        let _e1485 = (1f - _e1484);
        let _e1504 = (_e1461 + ((vec3<f32>(1f, 1f, 1f) - _e1461) * pow(clamp((1f - max(dot(_e83, _e1463), 0f)), 0f, 1f), 5f)));
        let _e1514 = world_pos_1;
        let _e1528 = (_e1434 - _e1514);
        let _e1529 = length(_e1528);
        let _e1535 = (_e1529 * _e1529);
        let _e1538 = (_e1535 / max((_e1424 * _e1424), 0.0001f));
        let _e1541 = clamp((1f - (_e1538 * _e1538)), 0f, 1f);
        let _e1548 = (_e1434 - _e1514);
        let _e1549 = length(_e1548);
        let _e1552 = (_e1548 / vec3(max(_e1549, 0.0001f)));
        let _e1557 = (_e1549 * _e1549);
        let _e1560 = (_e1557 / max((_e1424 * _e1424), 0.0001f));
        let _e1563 = clamp((1f - (_e1560 * _e1560)), 0f, 1f);
        let _e1571 = clamp(((dot(normalize(_e1444), _e1552) - _e1439) / max((_e1429 - _e1439), 0.0001f)), 0f, 1f);
        let _e1582 = local;
        local = (_e1582 + (((((((vec3<f32>(1f, 1f, 1f) - _e1504) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e1504 * ((_e1473 * 0.31830987f) / ((_e1477 * _e1477) + 0.00001f))) * (0.5f / (((_e1469 * sqrt((((_e1467 * _e1467) * _e1485) + _e1484))) + (_e1467 * sqrt((((_e1469 * _e1469) * _e1485) + _e1484)))) + 0.00001f)))) * _e1469) * (((((_e1449 * _e1419) * max(dot(_e78, normalize(_e1444)), 0f)) * select(0f, 1f, (_e1414 < 0.5f))) + ((((_e1449 * _e1419) * max(dot(_e78, (_e1528 / vec3(max(_e1529, 0.0001f)))), 0f)) * ((_e1541 * _e1541) / max(_e1535, 0.0001f))) * select(0f, 1f, ((_e1414 > 0.5f) && (_e1414 < 1.5f))))) + (((((_e1449 * _e1419) * max(dot(_e78, _e1552), 0f)) * ((_e1563 * _e1563) / max(_e1557, 0.0001f))) * (_e1571 * _e1571)) * select(0f, 1f, (_e1414 > 1.5f))))));
    }
    if (8f < _e176) {
        let _e1590 = lights.member[i32(8f)].member;
        let _e1595 = lights.member[i32(8f)].member_1;
        let _e1600 = lights.member[i32(8f)].member_2;
        let _e1605 = lights.member[i32(8f)].member_3;
        let _e1610 = lights.member[i32(8f)].member_4;
        let _e1615 = lights.member[i32(8f)].member_5;
        let _e1620 = lights.member[i32(8f)].member_6;
        let _e1625 = lights.member[i32(8f)].member_8;
        let _e1626 = world_pos_1;
        let _e1631 = select(0f, 1f, (_e1590 < 0.5f));
        let _e1635 = ((normalize(_e1620) * _e1631) + (normalize((_e1610 - _e1626)) * (1f - _e1631)));
        let _e1637 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e1639 = normalize((_e83 + _e1635));
        let _e1641 = max(dot(_e78, _e1639), 0f);
        let _e1643 = max(dot(_e78, _e83), 0.00001f);
        let _e1645 = max(dot(_e78, _e1635), 0.00001f);
        let _e1648 = (_e147 * _e147);
        let _e1649 = (_e1648 * _e1648);
        let _e1653 = (((_e1641 * _e1641) * (_e1649 - 1f)) + 1f);
        let _e1660 = (((_e147 * _e147) * _e147) * _e147);
        let _e1661 = (1f - _e1660);
        let _e1680 = (_e1637 + ((vec3<f32>(1f, 1f, 1f) - _e1637) * pow(clamp((1f - max(dot(_e83, _e1639), 0f)), 0f, 1f), 5f)));
        let _e1690 = world_pos_1;
        let _e1704 = (_e1610 - _e1690);
        let _e1705 = length(_e1704);
        let _e1711 = (_e1705 * _e1705);
        let _e1714 = (_e1711 / max((_e1600 * _e1600), 0.0001f));
        let _e1717 = clamp((1f - (_e1714 * _e1714)), 0f, 1f);
        let _e1724 = (_e1610 - _e1690);
        let _e1725 = length(_e1724);
        let _e1728 = (_e1724 / vec3(max(_e1725, 0.0001f)));
        let _e1733 = (_e1725 * _e1725);
        let _e1736 = (_e1733 / max((_e1600 * _e1600), 0.0001f));
        let _e1739 = clamp((1f - (_e1736 * _e1736)), 0f, 1f);
        let _e1747 = clamp(((dot(normalize(_e1620), _e1728) - _e1615) / max((_e1605 - _e1615), 0.0001f)), 0f, 1f);
        let _e1758 = local;
        local = (_e1758 + (((((((vec3<f32>(1f, 1f, 1f) - _e1680) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e1680 * ((_e1649 * 0.31830987f) / ((_e1653 * _e1653) + 0.00001f))) * (0.5f / (((_e1645 * sqrt((((_e1643 * _e1643) * _e1661) + _e1660))) + (_e1643 * sqrt((((_e1645 * _e1645) * _e1661) + _e1660)))) + 0.00001f)))) * _e1645) * (((((_e1625 * _e1595) * max(dot(_e78, normalize(_e1620)), 0f)) * select(0f, 1f, (_e1590 < 0.5f))) + ((((_e1625 * _e1595) * max(dot(_e78, (_e1704 / vec3(max(_e1705, 0.0001f)))), 0f)) * ((_e1717 * _e1717) / max(_e1711, 0.0001f))) * select(0f, 1f, ((_e1590 > 0.5f) && (_e1590 < 1.5f))))) + (((((_e1625 * _e1595) * max(dot(_e78, _e1728), 0f)) * ((_e1739 * _e1739) / max(_e1733, 0.0001f))) * (_e1747 * _e1747)) * select(0f, 1f, (_e1590 > 1.5f))))));
    }
    if (9f < _e176) {
        let _e1766 = lights.member[i32(9f)].member;
        let _e1771 = lights.member[i32(9f)].member_1;
        let _e1776 = lights.member[i32(9f)].member_2;
        let _e1781 = lights.member[i32(9f)].member_3;
        let _e1786 = lights.member[i32(9f)].member_4;
        let _e1791 = lights.member[i32(9f)].member_5;
        let _e1796 = lights.member[i32(9f)].member_6;
        let _e1801 = lights.member[i32(9f)].member_8;
        let _e1802 = world_pos_1;
        let _e1807 = select(0f, 1f, (_e1766 < 0.5f));
        let _e1811 = ((normalize(_e1796) * _e1807) + (normalize((_e1786 - _e1802)) * (1f - _e1807)));
        let _e1813 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e1815 = normalize((_e83 + _e1811));
        let _e1817 = max(dot(_e78, _e1815), 0f);
        let _e1819 = max(dot(_e78, _e83), 0.00001f);
        let _e1821 = max(dot(_e78, _e1811), 0.00001f);
        let _e1824 = (_e147 * _e147);
        let _e1825 = (_e1824 * _e1824);
        let _e1829 = (((_e1817 * _e1817) * (_e1825 - 1f)) + 1f);
        let _e1836 = (((_e147 * _e147) * _e147) * _e147);
        let _e1837 = (1f - _e1836);
        let _e1856 = (_e1813 + ((vec3<f32>(1f, 1f, 1f) - _e1813) * pow(clamp((1f - max(dot(_e83, _e1815), 0f)), 0f, 1f), 5f)));
        let _e1866 = world_pos_1;
        let _e1880 = (_e1786 - _e1866);
        let _e1881 = length(_e1880);
        let _e1887 = (_e1881 * _e1881);
        let _e1890 = (_e1887 / max((_e1776 * _e1776), 0.0001f));
        let _e1893 = clamp((1f - (_e1890 * _e1890)), 0f, 1f);
        let _e1900 = (_e1786 - _e1866);
        let _e1901 = length(_e1900);
        let _e1904 = (_e1900 / vec3(max(_e1901, 0.0001f)));
        let _e1909 = (_e1901 * _e1901);
        let _e1912 = (_e1909 / max((_e1776 * _e1776), 0.0001f));
        let _e1915 = clamp((1f - (_e1912 * _e1912)), 0f, 1f);
        let _e1923 = clamp(((dot(normalize(_e1796), _e1904) - _e1791) / max((_e1781 - _e1791), 0.0001f)), 0f, 1f);
        let _e1934 = local;
        local = (_e1934 + (((((((vec3<f32>(1f, 1f, 1f) - _e1856) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e1856 * ((_e1825 * 0.31830987f) / ((_e1829 * _e1829) + 0.00001f))) * (0.5f / (((_e1821 * sqrt((((_e1819 * _e1819) * _e1837) + _e1836))) + (_e1819 * sqrt((((_e1821 * _e1821) * _e1837) + _e1836)))) + 0.00001f)))) * _e1821) * (((((_e1801 * _e1771) * max(dot(_e78, normalize(_e1796)), 0f)) * select(0f, 1f, (_e1766 < 0.5f))) + ((((_e1801 * _e1771) * max(dot(_e78, (_e1880 / vec3(max(_e1881, 0.0001f)))), 0f)) * ((_e1893 * _e1893) / max(_e1887, 0.0001f))) * select(0f, 1f, ((_e1766 > 0.5f) && (_e1766 < 1.5f))))) + (((((_e1801 * _e1771) * max(dot(_e78, _e1904), 0f)) * ((_e1915 * _e1915) / max(_e1909, 0.0001f))) * (_e1923 * _e1923)) * select(0f, 1f, (_e1766 > 1.5f))))));
    }
    if (10f < _e176) {
        let _e1942 = lights.member[i32(10f)].member;
        let _e1947 = lights.member[i32(10f)].member_1;
        let _e1952 = lights.member[i32(10f)].member_2;
        let _e1957 = lights.member[i32(10f)].member_3;
        let _e1962 = lights.member[i32(10f)].member_4;
        let _e1967 = lights.member[i32(10f)].member_5;
        let _e1972 = lights.member[i32(10f)].member_6;
        let _e1977 = lights.member[i32(10f)].member_8;
        let _e1978 = world_pos_1;
        let _e1983 = select(0f, 1f, (_e1942 < 0.5f));
        let _e1987 = ((normalize(_e1972) * _e1983) + (normalize((_e1962 - _e1978)) * (1f - _e1983)));
        let _e1989 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e1991 = normalize((_e83 + _e1987));
        let _e1993 = max(dot(_e78, _e1991), 0f);
        let _e1995 = max(dot(_e78, _e83), 0.00001f);
        let _e1997 = max(dot(_e78, _e1987), 0.00001f);
        let _e2000 = (_e147 * _e147);
        let _e2001 = (_e2000 * _e2000);
        let _e2005 = (((_e1993 * _e1993) * (_e2001 - 1f)) + 1f);
        let _e2012 = (((_e147 * _e147) * _e147) * _e147);
        let _e2013 = (1f - _e2012);
        let _e2032 = (_e1989 + ((vec3<f32>(1f, 1f, 1f) - _e1989) * pow(clamp((1f - max(dot(_e83, _e1991), 0f)), 0f, 1f), 5f)));
        let _e2042 = world_pos_1;
        let _e2056 = (_e1962 - _e2042);
        let _e2057 = length(_e2056);
        let _e2063 = (_e2057 * _e2057);
        let _e2066 = (_e2063 / max((_e1952 * _e1952), 0.0001f));
        let _e2069 = clamp((1f - (_e2066 * _e2066)), 0f, 1f);
        let _e2076 = (_e1962 - _e2042);
        let _e2077 = length(_e2076);
        let _e2080 = (_e2076 / vec3(max(_e2077, 0.0001f)));
        let _e2085 = (_e2077 * _e2077);
        let _e2088 = (_e2085 / max((_e1952 * _e1952), 0.0001f));
        let _e2091 = clamp((1f - (_e2088 * _e2088)), 0f, 1f);
        let _e2099 = clamp(((dot(normalize(_e1972), _e2080) - _e1967) / max((_e1957 - _e1967), 0.0001f)), 0f, 1f);
        let _e2110 = local;
        local = (_e2110 + (((((((vec3<f32>(1f, 1f, 1f) - _e2032) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e2032 * ((_e2001 * 0.31830987f) / ((_e2005 * _e2005) + 0.00001f))) * (0.5f / (((_e1997 * sqrt((((_e1995 * _e1995) * _e2013) + _e2012))) + (_e1995 * sqrt((((_e1997 * _e1997) * _e2013) + _e2012)))) + 0.00001f)))) * _e1997) * (((((_e1977 * _e1947) * max(dot(_e78, normalize(_e1972)), 0f)) * select(0f, 1f, (_e1942 < 0.5f))) + ((((_e1977 * _e1947) * max(dot(_e78, (_e2056 / vec3(max(_e2057, 0.0001f)))), 0f)) * ((_e2069 * _e2069) / max(_e2063, 0.0001f))) * select(0f, 1f, ((_e1942 > 0.5f) && (_e1942 < 1.5f))))) + (((((_e1977 * _e1947) * max(dot(_e78, _e2080), 0f)) * ((_e2091 * _e2091) / max(_e2085, 0.0001f))) * (_e2099 * _e2099)) * select(0f, 1f, (_e1942 > 1.5f))))));
    }
    if (11f < _e176) {
        let _e2118 = lights.member[i32(11f)].member;
        let _e2123 = lights.member[i32(11f)].member_1;
        let _e2128 = lights.member[i32(11f)].member_2;
        let _e2133 = lights.member[i32(11f)].member_3;
        let _e2138 = lights.member[i32(11f)].member_4;
        let _e2143 = lights.member[i32(11f)].member_5;
        let _e2148 = lights.member[i32(11f)].member_6;
        let _e2153 = lights.member[i32(11f)].member_8;
        let _e2154 = world_pos_1;
        let _e2159 = select(0f, 1f, (_e2118 < 0.5f));
        let _e2163 = ((normalize(_e2148) * _e2159) + (normalize((_e2138 - _e2154)) * (1f - _e2159)));
        let _e2165 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e2167 = normalize((_e83 + _e2163));
        let _e2169 = max(dot(_e78, _e2167), 0f);
        let _e2171 = max(dot(_e78, _e83), 0.00001f);
        let _e2173 = max(dot(_e78, _e2163), 0.00001f);
        let _e2176 = (_e147 * _e147);
        let _e2177 = (_e2176 * _e2176);
        let _e2181 = (((_e2169 * _e2169) * (_e2177 - 1f)) + 1f);
        let _e2188 = (((_e147 * _e147) * _e147) * _e147);
        let _e2189 = (1f - _e2188);
        let _e2208 = (_e2165 + ((vec3<f32>(1f, 1f, 1f) - _e2165) * pow(clamp((1f - max(dot(_e83, _e2167), 0f)), 0f, 1f), 5f)));
        let _e2218 = world_pos_1;
        let _e2232 = (_e2138 - _e2218);
        let _e2233 = length(_e2232);
        let _e2239 = (_e2233 * _e2233);
        let _e2242 = (_e2239 / max((_e2128 * _e2128), 0.0001f));
        let _e2245 = clamp((1f - (_e2242 * _e2242)), 0f, 1f);
        let _e2252 = (_e2138 - _e2218);
        let _e2253 = length(_e2252);
        let _e2256 = (_e2252 / vec3(max(_e2253, 0.0001f)));
        let _e2261 = (_e2253 * _e2253);
        let _e2264 = (_e2261 / max((_e2128 * _e2128), 0.0001f));
        let _e2267 = clamp((1f - (_e2264 * _e2264)), 0f, 1f);
        let _e2275 = clamp(((dot(normalize(_e2148), _e2256) - _e2143) / max((_e2133 - _e2143), 0.0001f)), 0f, 1f);
        let _e2286 = local;
        local = (_e2286 + (((((((vec3<f32>(1f, 1f, 1f) - _e2208) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e2208 * ((_e2177 * 0.31830987f) / ((_e2181 * _e2181) + 0.00001f))) * (0.5f / (((_e2173 * sqrt((((_e2171 * _e2171) * _e2189) + _e2188))) + (_e2171 * sqrt((((_e2173 * _e2173) * _e2189) + _e2188)))) + 0.00001f)))) * _e2173) * (((((_e2153 * _e2123) * max(dot(_e78, normalize(_e2148)), 0f)) * select(0f, 1f, (_e2118 < 0.5f))) + ((((_e2153 * _e2123) * max(dot(_e78, (_e2232 / vec3(max(_e2233, 0.0001f)))), 0f)) * ((_e2245 * _e2245) / max(_e2239, 0.0001f))) * select(0f, 1f, ((_e2118 > 0.5f) && (_e2118 < 1.5f))))) + (((((_e2153 * _e2123) * max(dot(_e78, _e2256), 0f)) * ((_e2267 * _e2267) / max(_e2261, 0.0001f))) * (_e2275 * _e2275)) * select(0f, 1f, (_e2118 > 1.5f))))));
    }
    if (12f < _e176) {
        let _e2294 = lights.member[i32(12f)].member;
        let _e2299 = lights.member[i32(12f)].member_1;
        let _e2304 = lights.member[i32(12f)].member_2;
        let _e2309 = lights.member[i32(12f)].member_3;
        let _e2314 = lights.member[i32(12f)].member_4;
        let _e2319 = lights.member[i32(12f)].member_5;
        let _e2324 = lights.member[i32(12f)].member_6;
        let _e2329 = lights.member[i32(12f)].member_8;
        let _e2330 = world_pos_1;
        let _e2335 = select(0f, 1f, (_e2294 < 0.5f));
        let _e2339 = ((normalize(_e2324) * _e2335) + (normalize((_e2314 - _e2330)) * (1f - _e2335)));
        let _e2341 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e2343 = normalize((_e83 + _e2339));
        let _e2345 = max(dot(_e78, _e2343), 0f);
        let _e2347 = max(dot(_e78, _e83), 0.00001f);
        let _e2349 = max(dot(_e78, _e2339), 0.00001f);
        let _e2352 = (_e147 * _e147);
        let _e2353 = (_e2352 * _e2352);
        let _e2357 = (((_e2345 * _e2345) * (_e2353 - 1f)) + 1f);
        let _e2364 = (((_e147 * _e147) * _e147) * _e147);
        let _e2365 = (1f - _e2364);
        let _e2384 = (_e2341 + ((vec3<f32>(1f, 1f, 1f) - _e2341) * pow(clamp((1f - max(dot(_e83, _e2343), 0f)), 0f, 1f), 5f)));
        let _e2394 = world_pos_1;
        let _e2408 = (_e2314 - _e2394);
        let _e2409 = length(_e2408);
        let _e2415 = (_e2409 * _e2409);
        let _e2418 = (_e2415 / max((_e2304 * _e2304), 0.0001f));
        let _e2421 = clamp((1f - (_e2418 * _e2418)), 0f, 1f);
        let _e2428 = (_e2314 - _e2394);
        let _e2429 = length(_e2428);
        let _e2432 = (_e2428 / vec3(max(_e2429, 0.0001f)));
        let _e2437 = (_e2429 * _e2429);
        let _e2440 = (_e2437 / max((_e2304 * _e2304), 0.0001f));
        let _e2443 = clamp((1f - (_e2440 * _e2440)), 0f, 1f);
        let _e2451 = clamp(((dot(normalize(_e2324), _e2432) - _e2319) / max((_e2309 - _e2319), 0.0001f)), 0f, 1f);
        let _e2462 = local;
        local = (_e2462 + (((((((vec3<f32>(1f, 1f, 1f) - _e2384) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e2384 * ((_e2353 * 0.31830987f) / ((_e2357 * _e2357) + 0.00001f))) * (0.5f / (((_e2349 * sqrt((((_e2347 * _e2347) * _e2365) + _e2364))) + (_e2347 * sqrt((((_e2349 * _e2349) * _e2365) + _e2364)))) + 0.00001f)))) * _e2349) * (((((_e2329 * _e2299) * max(dot(_e78, normalize(_e2324)), 0f)) * select(0f, 1f, (_e2294 < 0.5f))) + ((((_e2329 * _e2299) * max(dot(_e78, (_e2408 / vec3(max(_e2409, 0.0001f)))), 0f)) * ((_e2421 * _e2421) / max(_e2415, 0.0001f))) * select(0f, 1f, ((_e2294 > 0.5f) && (_e2294 < 1.5f))))) + (((((_e2329 * _e2299) * max(dot(_e78, _e2432), 0f)) * ((_e2443 * _e2443) / max(_e2437, 0.0001f))) * (_e2451 * _e2451)) * select(0f, 1f, (_e2294 > 1.5f))))));
    }
    if (13f < _e176) {
        let _e2470 = lights.member[i32(13f)].member;
        let _e2475 = lights.member[i32(13f)].member_1;
        let _e2480 = lights.member[i32(13f)].member_2;
        let _e2485 = lights.member[i32(13f)].member_3;
        let _e2490 = lights.member[i32(13f)].member_4;
        let _e2495 = lights.member[i32(13f)].member_5;
        let _e2500 = lights.member[i32(13f)].member_6;
        let _e2505 = lights.member[i32(13f)].member_8;
        let _e2506 = world_pos_1;
        let _e2511 = select(0f, 1f, (_e2470 < 0.5f));
        let _e2515 = ((normalize(_e2500) * _e2511) + (normalize((_e2490 - _e2506)) * (1f - _e2511)));
        let _e2517 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e2519 = normalize((_e83 + _e2515));
        let _e2521 = max(dot(_e78, _e2519), 0f);
        let _e2523 = max(dot(_e78, _e83), 0.00001f);
        let _e2525 = max(dot(_e78, _e2515), 0.00001f);
        let _e2528 = (_e147 * _e147);
        let _e2529 = (_e2528 * _e2528);
        let _e2533 = (((_e2521 * _e2521) * (_e2529 - 1f)) + 1f);
        let _e2540 = (((_e147 * _e147) * _e147) * _e147);
        let _e2541 = (1f - _e2540);
        let _e2560 = (_e2517 + ((vec3<f32>(1f, 1f, 1f) - _e2517) * pow(clamp((1f - max(dot(_e83, _e2519), 0f)), 0f, 1f), 5f)));
        let _e2570 = world_pos_1;
        let _e2584 = (_e2490 - _e2570);
        let _e2585 = length(_e2584);
        let _e2591 = (_e2585 * _e2585);
        let _e2594 = (_e2591 / max((_e2480 * _e2480), 0.0001f));
        let _e2597 = clamp((1f - (_e2594 * _e2594)), 0f, 1f);
        let _e2604 = (_e2490 - _e2570);
        let _e2605 = length(_e2604);
        let _e2608 = (_e2604 / vec3(max(_e2605, 0.0001f)));
        let _e2613 = (_e2605 * _e2605);
        let _e2616 = (_e2613 / max((_e2480 * _e2480), 0.0001f));
        let _e2619 = clamp((1f - (_e2616 * _e2616)), 0f, 1f);
        let _e2627 = clamp(((dot(normalize(_e2500), _e2608) - _e2495) / max((_e2485 - _e2495), 0.0001f)), 0f, 1f);
        let _e2638 = local;
        local = (_e2638 + (((((((vec3<f32>(1f, 1f, 1f) - _e2560) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e2560 * ((_e2529 * 0.31830987f) / ((_e2533 * _e2533) + 0.00001f))) * (0.5f / (((_e2525 * sqrt((((_e2523 * _e2523) * _e2541) + _e2540))) + (_e2523 * sqrt((((_e2525 * _e2525) * _e2541) + _e2540)))) + 0.00001f)))) * _e2525) * (((((_e2505 * _e2475) * max(dot(_e78, normalize(_e2500)), 0f)) * select(0f, 1f, (_e2470 < 0.5f))) + ((((_e2505 * _e2475) * max(dot(_e78, (_e2584 / vec3(max(_e2585, 0.0001f)))), 0f)) * ((_e2597 * _e2597) / max(_e2591, 0.0001f))) * select(0f, 1f, ((_e2470 > 0.5f) && (_e2470 < 1.5f))))) + (((((_e2505 * _e2475) * max(dot(_e78, _e2608), 0f)) * ((_e2619 * _e2619) / max(_e2613, 0.0001f))) * (_e2627 * _e2627)) * select(0f, 1f, (_e2470 > 1.5f))))));
    }
    if (14f < _e176) {
        let _e2646 = lights.member[i32(14f)].member;
        let _e2651 = lights.member[i32(14f)].member_1;
        let _e2656 = lights.member[i32(14f)].member_2;
        let _e2661 = lights.member[i32(14f)].member_3;
        let _e2666 = lights.member[i32(14f)].member_4;
        let _e2671 = lights.member[i32(14f)].member_5;
        let _e2676 = lights.member[i32(14f)].member_6;
        let _e2681 = lights.member[i32(14f)].member_8;
        let _e2682 = world_pos_1;
        let _e2687 = select(0f, 1f, (_e2646 < 0.5f));
        let _e2691 = ((normalize(_e2676) * _e2687) + (normalize((_e2666 - _e2682)) * (1f - _e2687)));
        let _e2693 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e2695 = normalize((_e83 + _e2691));
        let _e2697 = max(dot(_e78, _e2695), 0f);
        let _e2699 = max(dot(_e78, _e83), 0.00001f);
        let _e2701 = max(dot(_e78, _e2691), 0.00001f);
        let _e2704 = (_e147 * _e147);
        let _e2705 = (_e2704 * _e2704);
        let _e2709 = (((_e2697 * _e2697) * (_e2705 - 1f)) + 1f);
        let _e2716 = (((_e147 * _e147) * _e147) * _e147);
        let _e2717 = (1f - _e2716);
        let _e2736 = (_e2693 + ((vec3<f32>(1f, 1f, 1f) - _e2693) * pow(clamp((1f - max(dot(_e83, _e2695), 0f)), 0f, 1f), 5f)));
        let _e2746 = world_pos_1;
        let _e2760 = (_e2666 - _e2746);
        let _e2761 = length(_e2760);
        let _e2767 = (_e2761 * _e2761);
        let _e2770 = (_e2767 / max((_e2656 * _e2656), 0.0001f));
        let _e2773 = clamp((1f - (_e2770 * _e2770)), 0f, 1f);
        let _e2780 = (_e2666 - _e2746);
        let _e2781 = length(_e2780);
        let _e2784 = (_e2780 / vec3(max(_e2781, 0.0001f)));
        let _e2789 = (_e2781 * _e2781);
        let _e2792 = (_e2789 / max((_e2656 * _e2656), 0.0001f));
        let _e2795 = clamp((1f - (_e2792 * _e2792)), 0f, 1f);
        let _e2803 = clamp(((dot(normalize(_e2676), _e2784) - _e2671) / max((_e2661 - _e2671), 0.0001f)), 0f, 1f);
        let _e2814 = local;
        local = (_e2814 + (((((((vec3<f32>(1f, 1f, 1f) - _e2736) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e2736 * ((_e2705 * 0.31830987f) / ((_e2709 * _e2709) + 0.00001f))) * (0.5f / (((_e2701 * sqrt((((_e2699 * _e2699) * _e2717) + _e2716))) + (_e2699 * sqrt((((_e2701 * _e2701) * _e2717) + _e2716)))) + 0.00001f)))) * _e2701) * (((((_e2681 * _e2651) * max(dot(_e78, normalize(_e2676)), 0f)) * select(0f, 1f, (_e2646 < 0.5f))) + ((((_e2681 * _e2651) * max(dot(_e78, (_e2760 / vec3(max(_e2761, 0.0001f)))), 0f)) * ((_e2773 * _e2773) / max(_e2767, 0.0001f))) * select(0f, 1f, ((_e2646 > 0.5f) && (_e2646 < 1.5f))))) + (((((_e2681 * _e2651) * max(dot(_e78, _e2784), 0f)) * ((_e2795 * _e2795) / max(_e2789, 0.0001f))) * (_e2803 * _e2803)) * select(0f, 1f, (_e2646 > 1.5f))))));
    }
    if (15f < _e176) {
        let _e2822 = lights.member[i32(15f)].member;
        let _e2827 = lights.member[i32(15f)].member_1;
        let _e2832 = lights.member[i32(15f)].member_2;
        let _e2837 = lights.member[i32(15f)].member_3;
        let _e2842 = lights.member[i32(15f)].member_4;
        let _e2847 = lights.member[i32(15f)].member_5;
        let _e2852 = lights.member[i32(15f)].member_6;
        let _e2857 = lights.member[i32(15f)].member_8;
        let _e2858 = world_pos_1;
        let _e2863 = select(0f, 1f, (_e2822 < 0.5f));
        let _e2867 = ((normalize(_e2852) * _e2863) + (normalize((_e2842 - _e2858)) * (1f - _e2863)));
        let _e2869 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
        let _e2871 = normalize((_e83 + _e2867));
        let _e2873 = max(dot(_e78, _e2871), 0f);
        let _e2875 = max(dot(_e78, _e83), 0.00001f);
        let _e2877 = max(dot(_e78, _e2867), 0.00001f);
        let _e2880 = (_e147 * _e147);
        let _e2881 = (_e2880 * _e2880);
        let _e2885 = (((_e2873 * _e2873) * (_e2881 - 1f)) + 1f);
        let _e2892 = (((_e147 * _e147) * _e147) * _e147);
        let _e2893 = (1f - _e2892);
        let _e2912 = (_e2869 + ((vec3<f32>(1f, 1f, 1f) - _e2869) * pow(clamp((1f - max(dot(_e83, _e2871), 0f)), 0f, 1f), 5f)));
        let _e2922 = world_pos_1;
        let _e2936 = (_e2842 - _e2922);
        let _e2937 = length(_e2936);
        let _e2943 = (_e2937 * _e2937);
        let _e2946 = (_e2943 / max((_e2832 * _e2832), 0.0001f));
        let _e2949 = clamp((1f - (_e2946 * _e2946)), 0f, 1f);
        let _e2956 = (_e2842 - _e2922);
        let _e2957 = length(_e2956);
        let _e2960 = (_e2956 / vec3(max(_e2957, 0.0001f)));
        let _e2965 = (_e2957 * _e2957);
        let _e2968 = (_e2965 / max((_e2832 * _e2832), 0.0001f));
        let _e2971 = clamp((1f - (_e2968 * _e2968)), 0f, 1f);
        let _e2979 = clamp(((dot(normalize(_e2852), _e2960) - _e2847) / max((_e2837 - _e2847), 0.0001f)), 0f, 1f);
        let _e2990 = local;
        local = (_e2990 + (((((((vec3<f32>(1f, 1f, 1f) - _e2912) * (1f - _e173)) * _e117) * 0.31830987f) + ((_e2912 * ((_e2881 * 0.31830987f) / ((_e2885 * _e2885) + 0.00001f))) * (0.5f / (((_e2877 * sqrt((((_e2875 * _e2875) * _e2893) + _e2892))) + (_e2875 * sqrt((((_e2877 * _e2877) * _e2893) + _e2892)))) + 0.00001f)))) * _e2877) * (((((_e2857 * _e2827) * max(dot(_e78, normalize(_e2852)), 0f)) * select(0f, 1f, (_e2822 < 0.5f))) + ((((_e2857 * _e2827) * max(dot(_e78, (_e2936 / vec3(max(_e2937, 0.0001f)))), 0f)) * ((_e2949 * _e2949) / max(_e2943, 0.0001f))) * select(0f, 1f, ((_e2822 > 0.5f) && (_e2822 < 1.5f))))) + (((((_e2857 * _e2827) * max(dot(_e78, _e2960), 0f)) * ((_e2971 * _e2971) / max(_e2965, 0.0001f))) * (_e2979 * _e2979)) * select(0f, 1f, (_e2822 > 1.5f))))));
    }
    local_1 = vec3<f32>(0f, 0f, 0f);
    let _e2996 = textureSampleLevel(env_specular_u002e_texture, env_specular_u002e_sampler, reflect((_e83 * -1f), _e78), (_e147 * 8f));
    let _e2998 = textureSample(env_irradiance_u002e_texture, env_irradiance_u002e_sampler, _e78);
    let _e3001 = textureSample(brdf_lut_u002e_texture, brdf_lut_u002e_sampler, vec2<f32>(_e85, _e147));
    let _e3002 = _e3001.xy;
    let _e3004 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e117, vec3(_e173));
    let _e3021 = (((_e3004 + ((max(vec3((1f - _e147)), _e3004) - _e3004) * pow(clamp((1f - _e85), 0f, 1f), 5f))) * _e3002.x) + vec3(_e3002.y));
    let _e3024 = (_e3004 + ((vec3<f32>(1f, 1f, 1f) - _e3004) * 0.04761905f));
    let _e3025 = (1f - (_e3002.x + _e3002.y));
    let _e3031 = (_e3021 + (((_e3021 * _e3024) / (vec3<f32>(1f, 1f, 1f) - (_e3024 * _e3025))) * _e3025));
    local_1 = (((((vec3<f32>(1f, 1f, 1f) - _e3031) * (1f - _e173)) * _e117) * _e2998.xyz) + (_e3031 * _e2996.xyz));
    let _e3039 = local;
    let _e3040 = local_1;
    let _e3044 = max(dot(_e78, normalize((_e83 + vec3<f32>(0f, 1f, 0f)))), 0f);
    let _e3046 = max(dot(_e78, _e83), 0.00001f);
    let _e3048 = max(dot(_e78, vec3<f32>(0f, 1f, 0f)), 0.00001f);
    let _e3049 = (_e147 * _e147);
    let _e3050 = (_e3049 * _e3049);
    let _e3054 = (((_e3044 * _e3044) * (_e3050 - 1f)) + 1f);
    let _e3061 = (((_e147 * _e147) * _e147) * _e147);
    let _e3062 = (1f - _e3061);
    let _e3089 = max(0f, 0.045f);
    let _e3093 = max(dot(_e78, normalize((_e83 + vec3<f32>(0f, 1f, 0f)))), 0.001f);
    let _e3095 = max(dot(_e78, vec3<f32>(0f, 1f, 0f)), 0f);
    let _e3097 = max(dot(_e78, _e83), 0.001f);
    let _e3099 = (1f / (_e3089 * _e3089));
    let _e3129 = max(0f, 0.045f);
    let _e3133 = normalize((_e83 + vec3<f32>(0f, 1f, 0f)));
    let _e3135 = max(dot(_e78, _e3133), 0f);
    let _e3137 = max(dot(_e78, vec3<f32>(0f, 1f, 0f)), 0f);
    let _e3139 = max(dot(_e78, _e83), 0f);
    let _e3142 = (_e3129 * _e3129);
    let _e3143 = (_e3142 * _e3142);
    let _e3147 = (((_e3135 * _e3135) * (_e3143 - 1f)) + 1f);
    let _e3154 = (((_e3129 * _e3129) * _e3129) * _e3129);
    let _e3155 = (1f - _e3154);
    let _e3190 = (((((((mix(_e3039, (((_e117 * (((_e3050 * 0.31830987f) / ((_e3054 * _e3054) + 0.00001f)) * (0.5f / (((_e3048 * sqrt((((_e3046 * _e3046) * _e3062) + _e3061))) + (_e3046 * sqrt((((_e3048 * _e3048) * _e3062) + _e3061)))) + 0.00001f)))) * 0f) * exp(((log(vec3<f32>(1f, 1f, 1f)) / vec3(1000000f)) * 0f))), vec3(0f)) + _e3040) + vec3<f32>(0f, 0f, 0f)) * (1f - (max(vec3<f32>(0f, 0f, 0f).x, max(vec3<f32>(0f, 0f, 0f).y, vec3<f32>(0f, 0f, 0f).z)) * (0.09f + (0.42f * _e3089))))) + (((vec3<f32>(0f, 0f, 0f) * (((2f + _e3099) * pow(max((1f - (_e3093 * _e3093)), 0f), (_e3099 * 0.5f))) / 6.2831855f)) * (1f / ((4f * ((_e3095 + _e3097) - (_e3095 * _e3097))) + 0.00001f))) * max(_e3095, 0f))) * (1f - ((vec3<f32>(0.04f, 0.04f, 0.04f) + ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(0.04f, 0.04f, 0.04f)) * pow(clamp((1f - max(dot(_e78, _e83), 0.001f)), 0f, 1f), 5f))).x * 0f))) + vec3(((((0f * ((_e3143 * 0.31830987f) / ((_e3147 * _e3147) + 0.00001f))) * (0.5f / (((_e3137 * sqrt((((_e3139 * _e3139) * _e3155) + _e3154))) + (_e3139 * sqrt((((_e3137 * _e3137) * _e3155) + _e3154)))) + 0.00001f))) * (0.04f + ((1f - 0.04f) * pow((1f - max(dot(_e83, _e3133), 0f)), 5f)))) * _e3137))) + vec3<f32>(0f, 0f, 0f));
    let _e3203 = pow(clamp(((_e3190 * ((_e3190 * 2.51f) + vec3(0.03f))) / ((_e3190 * ((_e3190 * 2.43f) + vec3(0.59f))) + vec3(0.14f))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f)), vec3<f32>(0.45454547f, 0.45454547f, 0.45454547f));
    color = vec4<f32>(_e3203.x, _e3203.y, _e3203.z, 1f);
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
