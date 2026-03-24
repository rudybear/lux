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
var emissive_tex_u002e_sampler: sampler;
@group(1) @binding(9) 
var emissive_tex_u002e_texture: texture_2d<f32>;
@group(1) @binding(10) 
var env_specular_u002e_sampler: sampler;
@group(1) @binding(11) 
var env_specular_u002e_texture: texture_cube<f32>;
@group(1) @binding(12) 
var env_irradiance_u002e_sampler: sampler;
@group(1) @binding(13) 
var env_irradiance_u002e_texture: texture_cube<f32>;
@group(1) @binding(14) 
var brdf_lut_u002e_sampler: sampler;
@group(1) @binding(15) 
var brdf_lut_u002e_texture: texture_2d<f32>;
@group(1) @binding(16) 
var<storage> lights: type_13;

fn main_1() {
    var local: vec3<f32>;
    var local_1: vec3<f32>;
    var local_2: vec3<f32>;

    let _e79 = frag_uv_1;
    let _e80 = world_normal_1;
    let _e81 = normalize(_e80);
    let _e83 = global.view_pos;
    let _e84 = world_pos_1;
    let _e86 = normalize((_e83 - _e84));
    let _e88 = max(dot(_e81, _e86), 0.001f);
    let _e90 = global_1.base_color_uv_st;
    let _e92 = global_1.base_color_uv_rot;
    let _e93 = cos(_e92);
    let _e94 = sin(_e92);
    let _e114 = textureSample(base_color_tex_u002e_texture, base_color_tex_u002e_sampler, ((vec2<f32>(((_e79.x * _e93) - (_e79.y * _e94)), ((_e79.x * _e94) + (_e79.y * _e93))) * vec2<f32>(_e90.z, _e90.w)) + vec2<f32>(_e90.x, _e90.y)));
    let _e118 = global_1.base_color_factor;
    let _e120 = (pow(_e114.xyz, vec3<f32>(2.2f, 2.2f, 2.2f)) * _e118.xyz);
    let _e122 = global_1.mr_uv_st;
    let _e124 = global_1.mr_uv_rot;
    let _e125 = cos(_e124);
    let _e126 = sin(_e124);
    let _e146 = textureSample(metallic_roughness_tex_u002e_texture, metallic_roughness_tex_u002e_sampler, ((vec2<f32>(((_e79.x * _e125) - (_e79.y * _e126)), ((_e79.x * _e126) + (_e79.y * _e125))) * vec2<f32>(_e122.z, _e122.w)) + vec2<f32>(_e122.x, _e122.y)));
    let _e149 = global_1.roughness_factor;
    let _e150 = (_e146.y * _e149);
    let _e151 = cos(_e124);
    let _e152 = sin(_e124);
    let _e172 = textureSample(metallic_roughness_tex_u002e_texture, metallic_roughness_tex_u002e_sampler, ((vec2<f32>(((_e79.x * _e151) - (_e79.y * _e152)), ((_e79.x * _e152) + (_e79.y * _e151))) * vec2<f32>(_e122.z, _e122.w)) + vec2<f32>(_e122.x, _e122.y)));
    let _e175 = global_1.metallic_factor;
    let _e176 = (_e172.z * _e175);
    let _e178 = global.light_count;
    let _e179 = f32(_e178);
    local = vec3<f32>(0f, 0f, 0f);
    if (0f < _e179) {
        let _e185 = lights.member[i32(0f)].member;
        let _e190 = lights.member[i32(0f)].member_1;
        let _e195 = lights.member[i32(0f)].member_2;
        let _e200 = lights.member[i32(0f)].member_3;
        let _e205 = lights.member[i32(0f)].member_4;
        let _e210 = lights.member[i32(0f)].member_5;
        let _e215 = lights.member[i32(0f)].member_6;
        let _e220 = lights.member[i32(0f)].member_8;
        let _e221 = world_pos_1;
        let _e226 = select(0f, 1f, (_e185 < 0.5f));
        let _e230 = ((normalize(_e215) * _e226) + (normalize((_e205 - _e221)) * (1f - _e226)));
        let _e232 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e234 = normalize((_e86 + _e230));
        let _e236 = max(dot(_e81, _e234), 0f);
        let _e238 = max(dot(_e81, _e86), 0.00001f);
        let _e240 = max(dot(_e81, _e230), 0.00001f);
        let _e243 = (_e150 * _e150);
        let _e244 = (_e243 * _e243);
        let _e248 = (((_e236 * _e236) * (_e244 - 1f)) + 1f);
        let _e255 = (((_e150 * _e150) * _e150) * _e150);
        let _e256 = (1f - _e255);
        let _e275 = (_e232 + ((vec3<f32>(1f, 1f, 1f) - _e232) * pow(clamp((1f - max(dot(_e86, _e234), 0f)), 0f, 1f), 5f)));
        let _e285 = world_pos_1;
        let _e299 = (_e205 - _e285);
        let _e300 = length(_e299);
        let _e306 = (_e300 * _e300);
        let _e309 = (_e306 / max((_e195 * _e195), 0.0001f));
        let _e312 = clamp((1f - (_e309 * _e309)), 0f, 1f);
        let _e319 = (_e205 - _e285);
        let _e320 = length(_e319);
        let _e323 = (_e319 / vec3(max(_e320, 0.0001f)));
        let _e328 = (_e320 * _e320);
        let _e331 = (_e328 / max((_e195 * _e195), 0.0001f));
        let _e334 = clamp((1f - (_e331 * _e331)), 0f, 1f);
        let _e342 = clamp(((dot(normalize(_e215), _e323) - _e210) / max((_e200 - _e210), 0.0001f)), 0f, 1f);
        let _e353 = local;
        local = (_e353 + (((((((vec3<f32>(1f, 1f, 1f) - _e275) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e275 * ((_e244 * 0.31830987f) / ((_e248 * _e248) + 0.00001f))) * (0.5f / (((_e240 * sqrt((((_e238 * _e238) * _e256) + _e255))) + (_e238 * sqrt((((_e240 * _e240) * _e256) + _e255)))) + 0.00001f)))) * _e240) * (((((_e220 * _e190) * max(dot(_e81, normalize(_e215)), 0f)) * select(0f, 1f, (_e185 < 0.5f))) + ((((_e220 * _e190) * max(dot(_e81, (_e299 / vec3(max(_e300, 0.0001f)))), 0f)) * ((_e312 * _e312) / max(_e306, 0.0001f))) * select(0f, 1f, ((_e185 > 0.5f) && (_e185 < 1.5f))))) + (((((_e220 * _e190) * max(dot(_e81, _e323), 0f)) * ((_e334 * _e334) / max(_e328, 0.0001f))) * (_e342 * _e342)) * select(0f, 1f, (_e185 > 1.5f))))));
    }
    if (1f < _e179) {
        let _e361 = lights.member[i32(1f)].member;
        let _e366 = lights.member[i32(1f)].member_1;
        let _e371 = lights.member[i32(1f)].member_2;
        let _e376 = lights.member[i32(1f)].member_3;
        let _e381 = lights.member[i32(1f)].member_4;
        let _e386 = lights.member[i32(1f)].member_5;
        let _e391 = lights.member[i32(1f)].member_6;
        let _e396 = lights.member[i32(1f)].member_8;
        let _e397 = world_pos_1;
        let _e402 = select(0f, 1f, (_e361 < 0.5f));
        let _e406 = ((normalize(_e391) * _e402) + (normalize((_e381 - _e397)) * (1f - _e402)));
        let _e408 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e410 = normalize((_e86 + _e406));
        let _e412 = max(dot(_e81, _e410), 0f);
        let _e414 = max(dot(_e81, _e86), 0.00001f);
        let _e416 = max(dot(_e81, _e406), 0.00001f);
        let _e419 = (_e150 * _e150);
        let _e420 = (_e419 * _e419);
        let _e424 = (((_e412 * _e412) * (_e420 - 1f)) + 1f);
        let _e431 = (((_e150 * _e150) * _e150) * _e150);
        let _e432 = (1f - _e431);
        let _e451 = (_e408 + ((vec3<f32>(1f, 1f, 1f) - _e408) * pow(clamp((1f - max(dot(_e86, _e410), 0f)), 0f, 1f), 5f)));
        let _e461 = world_pos_1;
        let _e475 = (_e381 - _e461);
        let _e476 = length(_e475);
        let _e482 = (_e476 * _e476);
        let _e485 = (_e482 / max((_e371 * _e371), 0.0001f));
        let _e488 = clamp((1f - (_e485 * _e485)), 0f, 1f);
        let _e495 = (_e381 - _e461);
        let _e496 = length(_e495);
        let _e499 = (_e495 / vec3(max(_e496, 0.0001f)));
        let _e504 = (_e496 * _e496);
        let _e507 = (_e504 / max((_e371 * _e371), 0.0001f));
        let _e510 = clamp((1f - (_e507 * _e507)), 0f, 1f);
        let _e518 = clamp(((dot(normalize(_e391), _e499) - _e386) / max((_e376 - _e386), 0.0001f)), 0f, 1f);
        let _e529 = local;
        local = (_e529 + (((((((vec3<f32>(1f, 1f, 1f) - _e451) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e451 * ((_e420 * 0.31830987f) / ((_e424 * _e424) + 0.00001f))) * (0.5f / (((_e416 * sqrt((((_e414 * _e414) * _e432) + _e431))) + (_e414 * sqrt((((_e416 * _e416) * _e432) + _e431)))) + 0.00001f)))) * _e416) * (((((_e396 * _e366) * max(dot(_e81, normalize(_e391)), 0f)) * select(0f, 1f, (_e361 < 0.5f))) + ((((_e396 * _e366) * max(dot(_e81, (_e475 / vec3(max(_e476, 0.0001f)))), 0f)) * ((_e488 * _e488) / max(_e482, 0.0001f))) * select(0f, 1f, ((_e361 > 0.5f) && (_e361 < 1.5f))))) + (((((_e396 * _e366) * max(dot(_e81, _e499), 0f)) * ((_e510 * _e510) / max(_e504, 0.0001f))) * (_e518 * _e518)) * select(0f, 1f, (_e361 > 1.5f))))));
    }
    if (2f < _e179) {
        let _e537 = lights.member[i32(2f)].member;
        let _e542 = lights.member[i32(2f)].member_1;
        let _e547 = lights.member[i32(2f)].member_2;
        let _e552 = lights.member[i32(2f)].member_3;
        let _e557 = lights.member[i32(2f)].member_4;
        let _e562 = lights.member[i32(2f)].member_5;
        let _e567 = lights.member[i32(2f)].member_6;
        let _e572 = lights.member[i32(2f)].member_8;
        let _e573 = world_pos_1;
        let _e578 = select(0f, 1f, (_e537 < 0.5f));
        let _e582 = ((normalize(_e567) * _e578) + (normalize((_e557 - _e573)) * (1f - _e578)));
        let _e584 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e586 = normalize((_e86 + _e582));
        let _e588 = max(dot(_e81, _e586), 0f);
        let _e590 = max(dot(_e81, _e86), 0.00001f);
        let _e592 = max(dot(_e81, _e582), 0.00001f);
        let _e595 = (_e150 * _e150);
        let _e596 = (_e595 * _e595);
        let _e600 = (((_e588 * _e588) * (_e596 - 1f)) + 1f);
        let _e607 = (((_e150 * _e150) * _e150) * _e150);
        let _e608 = (1f - _e607);
        let _e627 = (_e584 + ((vec3<f32>(1f, 1f, 1f) - _e584) * pow(clamp((1f - max(dot(_e86, _e586), 0f)), 0f, 1f), 5f)));
        let _e637 = world_pos_1;
        let _e651 = (_e557 - _e637);
        let _e652 = length(_e651);
        let _e658 = (_e652 * _e652);
        let _e661 = (_e658 / max((_e547 * _e547), 0.0001f));
        let _e664 = clamp((1f - (_e661 * _e661)), 0f, 1f);
        let _e671 = (_e557 - _e637);
        let _e672 = length(_e671);
        let _e675 = (_e671 / vec3(max(_e672, 0.0001f)));
        let _e680 = (_e672 * _e672);
        let _e683 = (_e680 / max((_e547 * _e547), 0.0001f));
        let _e686 = clamp((1f - (_e683 * _e683)), 0f, 1f);
        let _e694 = clamp(((dot(normalize(_e567), _e675) - _e562) / max((_e552 - _e562), 0.0001f)), 0f, 1f);
        let _e705 = local;
        local = (_e705 + (((((((vec3<f32>(1f, 1f, 1f) - _e627) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e627 * ((_e596 * 0.31830987f) / ((_e600 * _e600) + 0.00001f))) * (0.5f / (((_e592 * sqrt((((_e590 * _e590) * _e608) + _e607))) + (_e590 * sqrt((((_e592 * _e592) * _e608) + _e607)))) + 0.00001f)))) * _e592) * (((((_e572 * _e542) * max(dot(_e81, normalize(_e567)), 0f)) * select(0f, 1f, (_e537 < 0.5f))) + ((((_e572 * _e542) * max(dot(_e81, (_e651 / vec3(max(_e652, 0.0001f)))), 0f)) * ((_e664 * _e664) / max(_e658, 0.0001f))) * select(0f, 1f, ((_e537 > 0.5f) && (_e537 < 1.5f))))) + (((((_e572 * _e542) * max(dot(_e81, _e675), 0f)) * ((_e686 * _e686) / max(_e680, 0.0001f))) * (_e694 * _e694)) * select(0f, 1f, (_e537 > 1.5f))))));
    }
    if (3f < _e179) {
        let _e713 = lights.member[i32(3f)].member;
        let _e718 = lights.member[i32(3f)].member_1;
        let _e723 = lights.member[i32(3f)].member_2;
        let _e728 = lights.member[i32(3f)].member_3;
        let _e733 = lights.member[i32(3f)].member_4;
        let _e738 = lights.member[i32(3f)].member_5;
        let _e743 = lights.member[i32(3f)].member_6;
        let _e748 = lights.member[i32(3f)].member_8;
        let _e749 = world_pos_1;
        let _e754 = select(0f, 1f, (_e713 < 0.5f));
        let _e758 = ((normalize(_e743) * _e754) + (normalize((_e733 - _e749)) * (1f - _e754)));
        let _e760 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e762 = normalize((_e86 + _e758));
        let _e764 = max(dot(_e81, _e762), 0f);
        let _e766 = max(dot(_e81, _e86), 0.00001f);
        let _e768 = max(dot(_e81, _e758), 0.00001f);
        let _e771 = (_e150 * _e150);
        let _e772 = (_e771 * _e771);
        let _e776 = (((_e764 * _e764) * (_e772 - 1f)) + 1f);
        let _e783 = (((_e150 * _e150) * _e150) * _e150);
        let _e784 = (1f - _e783);
        let _e803 = (_e760 + ((vec3<f32>(1f, 1f, 1f) - _e760) * pow(clamp((1f - max(dot(_e86, _e762), 0f)), 0f, 1f), 5f)));
        let _e813 = world_pos_1;
        let _e827 = (_e733 - _e813);
        let _e828 = length(_e827);
        let _e834 = (_e828 * _e828);
        let _e837 = (_e834 / max((_e723 * _e723), 0.0001f));
        let _e840 = clamp((1f - (_e837 * _e837)), 0f, 1f);
        let _e847 = (_e733 - _e813);
        let _e848 = length(_e847);
        let _e851 = (_e847 / vec3(max(_e848, 0.0001f)));
        let _e856 = (_e848 * _e848);
        let _e859 = (_e856 / max((_e723 * _e723), 0.0001f));
        let _e862 = clamp((1f - (_e859 * _e859)), 0f, 1f);
        let _e870 = clamp(((dot(normalize(_e743), _e851) - _e738) / max((_e728 - _e738), 0.0001f)), 0f, 1f);
        let _e881 = local;
        local = (_e881 + (((((((vec3<f32>(1f, 1f, 1f) - _e803) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e803 * ((_e772 * 0.31830987f) / ((_e776 * _e776) + 0.00001f))) * (0.5f / (((_e768 * sqrt((((_e766 * _e766) * _e784) + _e783))) + (_e766 * sqrt((((_e768 * _e768) * _e784) + _e783)))) + 0.00001f)))) * _e768) * (((((_e748 * _e718) * max(dot(_e81, normalize(_e743)), 0f)) * select(0f, 1f, (_e713 < 0.5f))) + ((((_e748 * _e718) * max(dot(_e81, (_e827 / vec3(max(_e828, 0.0001f)))), 0f)) * ((_e840 * _e840) / max(_e834, 0.0001f))) * select(0f, 1f, ((_e713 > 0.5f) && (_e713 < 1.5f))))) + (((((_e748 * _e718) * max(dot(_e81, _e851), 0f)) * ((_e862 * _e862) / max(_e856, 0.0001f))) * (_e870 * _e870)) * select(0f, 1f, (_e713 > 1.5f))))));
    }
    if (4f < _e179) {
        let _e889 = lights.member[i32(4f)].member;
        let _e894 = lights.member[i32(4f)].member_1;
        let _e899 = lights.member[i32(4f)].member_2;
        let _e904 = lights.member[i32(4f)].member_3;
        let _e909 = lights.member[i32(4f)].member_4;
        let _e914 = lights.member[i32(4f)].member_5;
        let _e919 = lights.member[i32(4f)].member_6;
        let _e924 = lights.member[i32(4f)].member_8;
        let _e925 = world_pos_1;
        let _e930 = select(0f, 1f, (_e889 < 0.5f));
        let _e934 = ((normalize(_e919) * _e930) + (normalize((_e909 - _e925)) * (1f - _e930)));
        let _e936 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e938 = normalize((_e86 + _e934));
        let _e940 = max(dot(_e81, _e938), 0f);
        let _e942 = max(dot(_e81, _e86), 0.00001f);
        let _e944 = max(dot(_e81, _e934), 0.00001f);
        let _e947 = (_e150 * _e150);
        let _e948 = (_e947 * _e947);
        let _e952 = (((_e940 * _e940) * (_e948 - 1f)) + 1f);
        let _e959 = (((_e150 * _e150) * _e150) * _e150);
        let _e960 = (1f - _e959);
        let _e979 = (_e936 + ((vec3<f32>(1f, 1f, 1f) - _e936) * pow(clamp((1f - max(dot(_e86, _e938), 0f)), 0f, 1f), 5f)));
        let _e989 = world_pos_1;
        let _e1003 = (_e909 - _e989);
        let _e1004 = length(_e1003);
        let _e1010 = (_e1004 * _e1004);
        let _e1013 = (_e1010 / max((_e899 * _e899), 0.0001f));
        let _e1016 = clamp((1f - (_e1013 * _e1013)), 0f, 1f);
        let _e1023 = (_e909 - _e989);
        let _e1024 = length(_e1023);
        let _e1027 = (_e1023 / vec3(max(_e1024, 0.0001f)));
        let _e1032 = (_e1024 * _e1024);
        let _e1035 = (_e1032 / max((_e899 * _e899), 0.0001f));
        let _e1038 = clamp((1f - (_e1035 * _e1035)), 0f, 1f);
        let _e1046 = clamp(((dot(normalize(_e919), _e1027) - _e914) / max((_e904 - _e914), 0.0001f)), 0f, 1f);
        let _e1057 = local;
        local = (_e1057 + (((((((vec3<f32>(1f, 1f, 1f) - _e979) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e979 * ((_e948 * 0.31830987f) / ((_e952 * _e952) + 0.00001f))) * (0.5f / (((_e944 * sqrt((((_e942 * _e942) * _e960) + _e959))) + (_e942 * sqrt((((_e944 * _e944) * _e960) + _e959)))) + 0.00001f)))) * _e944) * (((((_e924 * _e894) * max(dot(_e81, normalize(_e919)), 0f)) * select(0f, 1f, (_e889 < 0.5f))) + ((((_e924 * _e894) * max(dot(_e81, (_e1003 / vec3(max(_e1004, 0.0001f)))), 0f)) * ((_e1016 * _e1016) / max(_e1010, 0.0001f))) * select(0f, 1f, ((_e889 > 0.5f) && (_e889 < 1.5f))))) + (((((_e924 * _e894) * max(dot(_e81, _e1027), 0f)) * ((_e1038 * _e1038) / max(_e1032, 0.0001f))) * (_e1046 * _e1046)) * select(0f, 1f, (_e889 > 1.5f))))));
    }
    if (5f < _e179) {
        let _e1065 = lights.member[i32(5f)].member;
        let _e1070 = lights.member[i32(5f)].member_1;
        let _e1075 = lights.member[i32(5f)].member_2;
        let _e1080 = lights.member[i32(5f)].member_3;
        let _e1085 = lights.member[i32(5f)].member_4;
        let _e1090 = lights.member[i32(5f)].member_5;
        let _e1095 = lights.member[i32(5f)].member_6;
        let _e1100 = lights.member[i32(5f)].member_8;
        let _e1101 = world_pos_1;
        let _e1106 = select(0f, 1f, (_e1065 < 0.5f));
        let _e1110 = ((normalize(_e1095) * _e1106) + (normalize((_e1085 - _e1101)) * (1f - _e1106)));
        let _e1112 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e1114 = normalize((_e86 + _e1110));
        let _e1116 = max(dot(_e81, _e1114), 0f);
        let _e1118 = max(dot(_e81, _e86), 0.00001f);
        let _e1120 = max(dot(_e81, _e1110), 0.00001f);
        let _e1123 = (_e150 * _e150);
        let _e1124 = (_e1123 * _e1123);
        let _e1128 = (((_e1116 * _e1116) * (_e1124 - 1f)) + 1f);
        let _e1135 = (((_e150 * _e150) * _e150) * _e150);
        let _e1136 = (1f - _e1135);
        let _e1155 = (_e1112 + ((vec3<f32>(1f, 1f, 1f) - _e1112) * pow(clamp((1f - max(dot(_e86, _e1114), 0f)), 0f, 1f), 5f)));
        let _e1165 = world_pos_1;
        let _e1179 = (_e1085 - _e1165);
        let _e1180 = length(_e1179);
        let _e1186 = (_e1180 * _e1180);
        let _e1189 = (_e1186 / max((_e1075 * _e1075), 0.0001f));
        let _e1192 = clamp((1f - (_e1189 * _e1189)), 0f, 1f);
        let _e1199 = (_e1085 - _e1165);
        let _e1200 = length(_e1199);
        let _e1203 = (_e1199 / vec3(max(_e1200, 0.0001f)));
        let _e1208 = (_e1200 * _e1200);
        let _e1211 = (_e1208 / max((_e1075 * _e1075), 0.0001f));
        let _e1214 = clamp((1f - (_e1211 * _e1211)), 0f, 1f);
        let _e1222 = clamp(((dot(normalize(_e1095), _e1203) - _e1090) / max((_e1080 - _e1090), 0.0001f)), 0f, 1f);
        let _e1233 = local;
        local = (_e1233 + (((((((vec3<f32>(1f, 1f, 1f) - _e1155) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e1155 * ((_e1124 * 0.31830987f) / ((_e1128 * _e1128) + 0.00001f))) * (0.5f / (((_e1120 * sqrt((((_e1118 * _e1118) * _e1136) + _e1135))) + (_e1118 * sqrt((((_e1120 * _e1120) * _e1136) + _e1135)))) + 0.00001f)))) * _e1120) * (((((_e1100 * _e1070) * max(dot(_e81, normalize(_e1095)), 0f)) * select(0f, 1f, (_e1065 < 0.5f))) + ((((_e1100 * _e1070) * max(dot(_e81, (_e1179 / vec3(max(_e1180, 0.0001f)))), 0f)) * ((_e1192 * _e1192) / max(_e1186, 0.0001f))) * select(0f, 1f, ((_e1065 > 0.5f) && (_e1065 < 1.5f))))) + (((((_e1100 * _e1070) * max(dot(_e81, _e1203), 0f)) * ((_e1214 * _e1214) / max(_e1208, 0.0001f))) * (_e1222 * _e1222)) * select(0f, 1f, (_e1065 > 1.5f))))));
    }
    if (6f < _e179) {
        let _e1241 = lights.member[i32(6f)].member;
        let _e1246 = lights.member[i32(6f)].member_1;
        let _e1251 = lights.member[i32(6f)].member_2;
        let _e1256 = lights.member[i32(6f)].member_3;
        let _e1261 = lights.member[i32(6f)].member_4;
        let _e1266 = lights.member[i32(6f)].member_5;
        let _e1271 = lights.member[i32(6f)].member_6;
        let _e1276 = lights.member[i32(6f)].member_8;
        let _e1277 = world_pos_1;
        let _e1282 = select(0f, 1f, (_e1241 < 0.5f));
        let _e1286 = ((normalize(_e1271) * _e1282) + (normalize((_e1261 - _e1277)) * (1f - _e1282)));
        let _e1288 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e1290 = normalize((_e86 + _e1286));
        let _e1292 = max(dot(_e81, _e1290), 0f);
        let _e1294 = max(dot(_e81, _e86), 0.00001f);
        let _e1296 = max(dot(_e81, _e1286), 0.00001f);
        let _e1299 = (_e150 * _e150);
        let _e1300 = (_e1299 * _e1299);
        let _e1304 = (((_e1292 * _e1292) * (_e1300 - 1f)) + 1f);
        let _e1311 = (((_e150 * _e150) * _e150) * _e150);
        let _e1312 = (1f - _e1311);
        let _e1331 = (_e1288 + ((vec3<f32>(1f, 1f, 1f) - _e1288) * pow(clamp((1f - max(dot(_e86, _e1290), 0f)), 0f, 1f), 5f)));
        let _e1341 = world_pos_1;
        let _e1355 = (_e1261 - _e1341);
        let _e1356 = length(_e1355);
        let _e1362 = (_e1356 * _e1356);
        let _e1365 = (_e1362 / max((_e1251 * _e1251), 0.0001f));
        let _e1368 = clamp((1f - (_e1365 * _e1365)), 0f, 1f);
        let _e1375 = (_e1261 - _e1341);
        let _e1376 = length(_e1375);
        let _e1379 = (_e1375 / vec3(max(_e1376, 0.0001f)));
        let _e1384 = (_e1376 * _e1376);
        let _e1387 = (_e1384 / max((_e1251 * _e1251), 0.0001f));
        let _e1390 = clamp((1f - (_e1387 * _e1387)), 0f, 1f);
        let _e1398 = clamp(((dot(normalize(_e1271), _e1379) - _e1266) / max((_e1256 - _e1266), 0.0001f)), 0f, 1f);
        let _e1409 = local;
        local = (_e1409 + (((((((vec3<f32>(1f, 1f, 1f) - _e1331) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e1331 * ((_e1300 * 0.31830987f) / ((_e1304 * _e1304) + 0.00001f))) * (0.5f / (((_e1296 * sqrt((((_e1294 * _e1294) * _e1312) + _e1311))) + (_e1294 * sqrt((((_e1296 * _e1296) * _e1312) + _e1311)))) + 0.00001f)))) * _e1296) * (((((_e1276 * _e1246) * max(dot(_e81, normalize(_e1271)), 0f)) * select(0f, 1f, (_e1241 < 0.5f))) + ((((_e1276 * _e1246) * max(dot(_e81, (_e1355 / vec3(max(_e1356, 0.0001f)))), 0f)) * ((_e1368 * _e1368) / max(_e1362, 0.0001f))) * select(0f, 1f, ((_e1241 > 0.5f) && (_e1241 < 1.5f))))) + (((((_e1276 * _e1246) * max(dot(_e81, _e1379), 0f)) * ((_e1390 * _e1390) / max(_e1384, 0.0001f))) * (_e1398 * _e1398)) * select(0f, 1f, (_e1241 > 1.5f))))));
    }
    if (7f < _e179) {
        let _e1417 = lights.member[i32(7f)].member;
        let _e1422 = lights.member[i32(7f)].member_1;
        let _e1427 = lights.member[i32(7f)].member_2;
        let _e1432 = lights.member[i32(7f)].member_3;
        let _e1437 = lights.member[i32(7f)].member_4;
        let _e1442 = lights.member[i32(7f)].member_5;
        let _e1447 = lights.member[i32(7f)].member_6;
        let _e1452 = lights.member[i32(7f)].member_8;
        let _e1453 = world_pos_1;
        let _e1458 = select(0f, 1f, (_e1417 < 0.5f));
        let _e1462 = ((normalize(_e1447) * _e1458) + (normalize((_e1437 - _e1453)) * (1f - _e1458)));
        let _e1464 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e1466 = normalize((_e86 + _e1462));
        let _e1468 = max(dot(_e81, _e1466), 0f);
        let _e1470 = max(dot(_e81, _e86), 0.00001f);
        let _e1472 = max(dot(_e81, _e1462), 0.00001f);
        let _e1475 = (_e150 * _e150);
        let _e1476 = (_e1475 * _e1475);
        let _e1480 = (((_e1468 * _e1468) * (_e1476 - 1f)) + 1f);
        let _e1487 = (((_e150 * _e150) * _e150) * _e150);
        let _e1488 = (1f - _e1487);
        let _e1507 = (_e1464 + ((vec3<f32>(1f, 1f, 1f) - _e1464) * pow(clamp((1f - max(dot(_e86, _e1466), 0f)), 0f, 1f), 5f)));
        let _e1517 = world_pos_1;
        let _e1531 = (_e1437 - _e1517);
        let _e1532 = length(_e1531);
        let _e1538 = (_e1532 * _e1532);
        let _e1541 = (_e1538 / max((_e1427 * _e1427), 0.0001f));
        let _e1544 = clamp((1f - (_e1541 * _e1541)), 0f, 1f);
        let _e1551 = (_e1437 - _e1517);
        let _e1552 = length(_e1551);
        let _e1555 = (_e1551 / vec3(max(_e1552, 0.0001f)));
        let _e1560 = (_e1552 * _e1552);
        let _e1563 = (_e1560 / max((_e1427 * _e1427), 0.0001f));
        let _e1566 = clamp((1f - (_e1563 * _e1563)), 0f, 1f);
        let _e1574 = clamp(((dot(normalize(_e1447), _e1555) - _e1442) / max((_e1432 - _e1442), 0.0001f)), 0f, 1f);
        let _e1585 = local;
        local = (_e1585 + (((((((vec3<f32>(1f, 1f, 1f) - _e1507) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e1507 * ((_e1476 * 0.31830987f) / ((_e1480 * _e1480) + 0.00001f))) * (0.5f / (((_e1472 * sqrt((((_e1470 * _e1470) * _e1488) + _e1487))) + (_e1470 * sqrt((((_e1472 * _e1472) * _e1488) + _e1487)))) + 0.00001f)))) * _e1472) * (((((_e1452 * _e1422) * max(dot(_e81, normalize(_e1447)), 0f)) * select(0f, 1f, (_e1417 < 0.5f))) + ((((_e1452 * _e1422) * max(dot(_e81, (_e1531 / vec3(max(_e1532, 0.0001f)))), 0f)) * ((_e1544 * _e1544) / max(_e1538, 0.0001f))) * select(0f, 1f, ((_e1417 > 0.5f) && (_e1417 < 1.5f))))) + (((((_e1452 * _e1422) * max(dot(_e81, _e1555), 0f)) * ((_e1566 * _e1566) / max(_e1560, 0.0001f))) * (_e1574 * _e1574)) * select(0f, 1f, (_e1417 > 1.5f))))));
    }
    if (8f < _e179) {
        let _e1593 = lights.member[i32(8f)].member;
        let _e1598 = lights.member[i32(8f)].member_1;
        let _e1603 = lights.member[i32(8f)].member_2;
        let _e1608 = lights.member[i32(8f)].member_3;
        let _e1613 = lights.member[i32(8f)].member_4;
        let _e1618 = lights.member[i32(8f)].member_5;
        let _e1623 = lights.member[i32(8f)].member_6;
        let _e1628 = lights.member[i32(8f)].member_8;
        let _e1629 = world_pos_1;
        let _e1634 = select(0f, 1f, (_e1593 < 0.5f));
        let _e1638 = ((normalize(_e1623) * _e1634) + (normalize((_e1613 - _e1629)) * (1f - _e1634)));
        let _e1640 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e1642 = normalize((_e86 + _e1638));
        let _e1644 = max(dot(_e81, _e1642), 0f);
        let _e1646 = max(dot(_e81, _e86), 0.00001f);
        let _e1648 = max(dot(_e81, _e1638), 0.00001f);
        let _e1651 = (_e150 * _e150);
        let _e1652 = (_e1651 * _e1651);
        let _e1656 = (((_e1644 * _e1644) * (_e1652 - 1f)) + 1f);
        let _e1663 = (((_e150 * _e150) * _e150) * _e150);
        let _e1664 = (1f - _e1663);
        let _e1683 = (_e1640 + ((vec3<f32>(1f, 1f, 1f) - _e1640) * pow(clamp((1f - max(dot(_e86, _e1642), 0f)), 0f, 1f), 5f)));
        let _e1693 = world_pos_1;
        let _e1707 = (_e1613 - _e1693);
        let _e1708 = length(_e1707);
        let _e1714 = (_e1708 * _e1708);
        let _e1717 = (_e1714 / max((_e1603 * _e1603), 0.0001f));
        let _e1720 = clamp((1f - (_e1717 * _e1717)), 0f, 1f);
        let _e1727 = (_e1613 - _e1693);
        let _e1728 = length(_e1727);
        let _e1731 = (_e1727 / vec3(max(_e1728, 0.0001f)));
        let _e1736 = (_e1728 * _e1728);
        let _e1739 = (_e1736 / max((_e1603 * _e1603), 0.0001f));
        let _e1742 = clamp((1f - (_e1739 * _e1739)), 0f, 1f);
        let _e1750 = clamp(((dot(normalize(_e1623), _e1731) - _e1618) / max((_e1608 - _e1618), 0.0001f)), 0f, 1f);
        let _e1761 = local;
        local = (_e1761 + (((((((vec3<f32>(1f, 1f, 1f) - _e1683) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e1683 * ((_e1652 * 0.31830987f) / ((_e1656 * _e1656) + 0.00001f))) * (0.5f / (((_e1648 * sqrt((((_e1646 * _e1646) * _e1664) + _e1663))) + (_e1646 * sqrt((((_e1648 * _e1648) * _e1664) + _e1663)))) + 0.00001f)))) * _e1648) * (((((_e1628 * _e1598) * max(dot(_e81, normalize(_e1623)), 0f)) * select(0f, 1f, (_e1593 < 0.5f))) + ((((_e1628 * _e1598) * max(dot(_e81, (_e1707 / vec3(max(_e1708, 0.0001f)))), 0f)) * ((_e1720 * _e1720) / max(_e1714, 0.0001f))) * select(0f, 1f, ((_e1593 > 0.5f) && (_e1593 < 1.5f))))) + (((((_e1628 * _e1598) * max(dot(_e81, _e1731), 0f)) * ((_e1742 * _e1742) / max(_e1736, 0.0001f))) * (_e1750 * _e1750)) * select(0f, 1f, (_e1593 > 1.5f))))));
    }
    if (9f < _e179) {
        let _e1769 = lights.member[i32(9f)].member;
        let _e1774 = lights.member[i32(9f)].member_1;
        let _e1779 = lights.member[i32(9f)].member_2;
        let _e1784 = lights.member[i32(9f)].member_3;
        let _e1789 = lights.member[i32(9f)].member_4;
        let _e1794 = lights.member[i32(9f)].member_5;
        let _e1799 = lights.member[i32(9f)].member_6;
        let _e1804 = lights.member[i32(9f)].member_8;
        let _e1805 = world_pos_1;
        let _e1810 = select(0f, 1f, (_e1769 < 0.5f));
        let _e1814 = ((normalize(_e1799) * _e1810) + (normalize((_e1789 - _e1805)) * (1f - _e1810)));
        let _e1816 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e1818 = normalize((_e86 + _e1814));
        let _e1820 = max(dot(_e81, _e1818), 0f);
        let _e1822 = max(dot(_e81, _e86), 0.00001f);
        let _e1824 = max(dot(_e81, _e1814), 0.00001f);
        let _e1827 = (_e150 * _e150);
        let _e1828 = (_e1827 * _e1827);
        let _e1832 = (((_e1820 * _e1820) * (_e1828 - 1f)) + 1f);
        let _e1839 = (((_e150 * _e150) * _e150) * _e150);
        let _e1840 = (1f - _e1839);
        let _e1859 = (_e1816 + ((vec3<f32>(1f, 1f, 1f) - _e1816) * pow(clamp((1f - max(dot(_e86, _e1818), 0f)), 0f, 1f), 5f)));
        let _e1869 = world_pos_1;
        let _e1883 = (_e1789 - _e1869);
        let _e1884 = length(_e1883);
        let _e1890 = (_e1884 * _e1884);
        let _e1893 = (_e1890 / max((_e1779 * _e1779), 0.0001f));
        let _e1896 = clamp((1f - (_e1893 * _e1893)), 0f, 1f);
        let _e1903 = (_e1789 - _e1869);
        let _e1904 = length(_e1903);
        let _e1907 = (_e1903 / vec3(max(_e1904, 0.0001f)));
        let _e1912 = (_e1904 * _e1904);
        let _e1915 = (_e1912 / max((_e1779 * _e1779), 0.0001f));
        let _e1918 = clamp((1f - (_e1915 * _e1915)), 0f, 1f);
        let _e1926 = clamp(((dot(normalize(_e1799), _e1907) - _e1794) / max((_e1784 - _e1794), 0.0001f)), 0f, 1f);
        let _e1937 = local;
        local = (_e1937 + (((((((vec3<f32>(1f, 1f, 1f) - _e1859) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e1859 * ((_e1828 * 0.31830987f) / ((_e1832 * _e1832) + 0.00001f))) * (0.5f / (((_e1824 * sqrt((((_e1822 * _e1822) * _e1840) + _e1839))) + (_e1822 * sqrt((((_e1824 * _e1824) * _e1840) + _e1839)))) + 0.00001f)))) * _e1824) * (((((_e1804 * _e1774) * max(dot(_e81, normalize(_e1799)), 0f)) * select(0f, 1f, (_e1769 < 0.5f))) + ((((_e1804 * _e1774) * max(dot(_e81, (_e1883 / vec3(max(_e1884, 0.0001f)))), 0f)) * ((_e1896 * _e1896) / max(_e1890, 0.0001f))) * select(0f, 1f, ((_e1769 > 0.5f) && (_e1769 < 1.5f))))) + (((((_e1804 * _e1774) * max(dot(_e81, _e1907), 0f)) * ((_e1918 * _e1918) / max(_e1912, 0.0001f))) * (_e1926 * _e1926)) * select(0f, 1f, (_e1769 > 1.5f))))));
    }
    if (10f < _e179) {
        let _e1945 = lights.member[i32(10f)].member;
        let _e1950 = lights.member[i32(10f)].member_1;
        let _e1955 = lights.member[i32(10f)].member_2;
        let _e1960 = lights.member[i32(10f)].member_3;
        let _e1965 = lights.member[i32(10f)].member_4;
        let _e1970 = lights.member[i32(10f)].member_5;
        let _e1975 = lights.member[i32(10f)].member_6;
        let _e1980 = lights.member[i32(10f)].member_8;
        let _e1981 = world_pos_1;
        let _e1986 = select(0f, 1f, (_e1945 < 0.5f));
        let _e1990 = ((normalize(_e1975) * _e1986) + (normalize((_e1965 - _e1981)) * (1f - _e1986)));
        let _e1992 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e1994 = normalize((_e86 + _e1990));
        let _e1996 = max(dot(_e81, _e1994), 0f);
        let _e1998 = max(dot(_e81, _e86), 0.00001f);
        let _e2000 = max(dot(_e81, _e1990), 0.00001f);
        let _e2003 = (_e150 * _e150);
        let _e2004 = (_e2003 * _e2003);
        let _e2008 = (((_e1996 * _e1996) * (_e2004 - 1f)) + 1f);
        let _e2015 = (((_e150 * _e150) * _e150) * _e150);
        let _e2016 = (1f - _e2015);
        let _e2035 = (_e1992 + ((vec3<f32>(1f, 1f, 1f) - _e1992) * pow(clamp((1f - max(dot(_e86, _e1994), 0f)), 0f, 1f), 5f)));
        let _e2045 = world_pos_1;
        let _e2059 = (_e1965 - _e2045);
        let _e2060 = length(_e2059);
        let _e2066 = (_e2060 * _e2060);
        let _e2069 = (_e2066 / max((_e1955 * _e1955), 0.0001f));
        let _e2072 = clamp((1f - (_e2069 * _e2069)), 0f, 1f);
        let _e2079 = (_e1965 - _e2045);
        let _e2080 = length(_e2079);
        let _e2083 = (_e2079 / vec3(max(_e2080, 0.0001f)));
        let _e2088 = (_e2080 * _e2080);
        let _e2091 = (_e2088 / max((_e1955 * _e1955), 0.0001f));
        let _e2094 = clamp((1f - (_e2091 * _e2091)), 0f, 1f);
        let _e2102 = clamp(((dot(normalize(_e1975), _e2083) - _e1970) / max((_e1960 - _e1970), 0.0001f)), 0f, 1f);
        let _e2113 = local;
        local = (_e2113 + (((((((vec3<f32>(1f, 1f, 1f) - _e2035) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e2035 * ((_e2004 * 0.31830987f) / ((_e2008 * _e2008) + 0.00001f))) * (0.5f / (((_e2000 * sqrt((((_e1998 * _e1998) * _e2016) + _e2015))) + (_e1998 * sqrt((((_e2000 * _e2000) * _e2016) + _e2015)))) + 0.00001f)))) * _e2000) * (((((_e1980 * _e1950) * max(dot(_e81, normalize(_e1975)), 0f)) * select(0f, 1f, (_e1945 < 0.5f))) + ((((_e1980 * _e1950) * max(dot(_e81, (_e2059 / vec3(max(_e2060, 0.0001f)))), 0f)) * ((_e2072 * _e2072) / max(_e2066, 0.0001f))) * select(0f, 1f, ((_e1945 > 0.5f) && (_e1945 < 1.5f))))) + (((((_e1980 * _e1950) * max(dot(_e81, _e2083), 0f)) * ((_e2094 * _e2094) / max(_e2088, 0.0001f))) * (_e2102 * _e2102)) * select(0f, 1f, (_e1945 > 1.5f))))));
    }
    if (11f < _e179) {
        let _e2121 = lights.member[i32(11f)].member;
        let _e2126 = lights.member[i32(11f)].member_1;
        let _e2131 = lights.member[i32(11f)].member_2;
        let _e2136 = lights.member[i32(11f)].member_3;
        let _e2141 = lights.member[i32(11f)].member_4;
        let _e2146 = lights.member[i32(11f)].member_5;
        let _e2151 = lights.member[i32(11f)].member_6;
        let _e2156 = lights.member[i32(11f)].member_8;
        let _e2157 = world_pos_1;
        let _e2162 = select(0f, 1f, (_e2121 < 0.5f));
        let _e2166 = ((normalize(_e2151) * _e2162) + (normalize((_e2141 - _e2157)) * (1f - _e2162)));
        let _e2168 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e2170 = normalize((_e86 + _e2166));
        let _e2172 = max(dot(_e81, _e2170), 0f);
        let _e2174 = max(dot(_e81, _e86), 0.00001f);
        let _e2176 = max(dot(_e81, _e2166), 0.00001f);
        let _e2179 = (_e150 * _e150);
        let _e2180 = (_e2179 * _e2179);
        let _e2184 = (((_e2172 * _e2172) * (_e2180 - 1f)) + 1f);
        let _e2191 = (((_e150 * _e150) * _e150) * _e150);
        let _e2192 = (1f - _e2191);
        let _e2211 = (_e2168 + ((vec3<f32>(1f, 1f, 1f) - _e2168) * pow(clamp((1f - max(dot(_e86, _e2170), 0f)), 0f, 1f), 5f)));
        let _e2221 = world_pos_1;
        let _e2235 = (_e2141 - _e2221);
        let _e2236 = length(_e2235);
        let _e2242 = (_e2236 * _e2236);
        let _e2245 = (_e2242 / max((_e2131 * _e2131), 0.0001f));
        let _e2248 = clamp((1f - (_e2245 * _e2245)), 0f, 1f);
        let _e2255 = (_e2141 - _e2221);
        let _e2256 = length(_e2255);
        let _e2259 = (_e2255 / vec3(max(_e2256, 0.0001f)));
        let _e2264 = (_e2256 * _e2256);
        let _e2267 = (_e2264 / max((_e2131 * _e2131), 0.0001f));
        let _e2270 = clamp((1f - (_e2267 * _e2267)), 0f, 1f);
        let _e2278 = clamp(((dot(normalize(_e2151), _e2259) - _e2146) / max((_e2136 - _e2146), 0.0001f)), 0f, 1f);
        let _e2289 = local;
        local = (_e2289 + (((((((vec3<f32>(1f, 1f, 1f) - _e2211) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e2211 * ((_e2180 * 0.31830987f) / ((_e2184 * _e2184) + 0.00001f))) * (0.5f / (((_e2176 * sqrt((((_e2174 * _e2174) * _e2192) + _e2191))) + (_e2174 * sqrt((((_e2176 * _e2176) * _e2192) + _e2191)))) + 0.00001f)))) * _e2176) * (((((_e2156 * _e2126) * max(dot(_e81, normalize(_e2151)), 0f)) * select(0f, 1f, (_e2121 < 0.5f))) + ((((_e2156 * _e2126) * max(dot(_e81, (_e2235 / vec3(max(_e2236, 0.0001f)))), 0f)) * ((_e2248 * _e2248) / max(_e2242, 0.0001f))) * select(0f, 1f, ((_e2121 > 0.5f) && (_e2121 < 1.5f))))) + (((((_e2156 * _e2126) * max(dot(_e81, _e2259), 0f)) * ((_e2270 * _e2270) / max(_e2264, 0.0001f))) * (_e2278 * _e2278)) * select(0f, 1f, (_e2121 > 1.5f))))));
    }
    if (12f < _e179) {
        let _e2297 = lights.member[i32(12f)].member;
        let _e2302 = lights.member[i32(12f)].member_1;
        let _e2307 = lights.member[i32(12f)].member_2;
        let _e2312 = lights.member[i32(12f)].member_3;
        let _e2317 = lights.member[i32(12f)].member_4;
        let _e2322 = lights.member[i32(12f)].member_5;
        let _e2327 = lights.member[i32(12f)].member_6;
        let _e2332 = lights.member[i32(12f)].member_8;
        let _e2333 = world_pos_1;
        let _e2338 = select(0f, 1f, (_e2297 < 0.5f));
        let _e2342 = ((normalize(_e2327) * _e2338) + (normalize((_e2317 - _e2333)) * (1f - _e2338)));
        let _e2344 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e2346 = normalize((_e86 + _e2342));
        let _e2348 = max(dot(_e81, _e2346), 0f);
        let _e2350 = max(dot(_e81, _e86), 0.00001f);
        let _e2352 = max(dot(_e81, _e2342), 0.00001f);
        let _e2355 = (_e150 * _e150);
        let _e2356 = (_e2355 * _e2355);
        let _e2360 = (((_e2348 * _e2348) * (_e2356 - 1f)) + 1f);
        let _e2367 = (((_e150 * _e150) * _e150) * _e150);
        let _e2368 = (1f - _e2367);
        let _e2387 = (_e2344 + ((vec3<f32>(1f, 1f, 1f) - _e2344) * pow(clamp((1f - max(dot(_e86, _e2346), 0f)), 0f, 1f), 5f)));
        let _e2397 = world_pos_1;
        let _e2411 = (_e2317 - _e2397);
        let _e2412 = length(_e2411);
        let _e2418 = (_e2412 * _e2412);
        let _e2421 = (_e2418 / max((_e2307 * _e2307), 0.0001f));
        let _e2424 = clamp((1f - (_e2421 * _e2421)), 0f, 1f);
        let _e2431 = (_e2317 - _e2397);
        let _e2432 = length(_e2431);
        let _e2435 = (_e2431 / vec3(max(_e2432, 0.0001f)));
        let _e2440 = (_e2432 * _e2432);
        let _e2443 = (_e2440 / max((_e2307 * _e2307), 0.0001f));
        let _e2446 = clamp((1f - (_e2443 * _e2443)), 0f, 1f);
        let _e2454 = clamp(((dot(normalize(_e2327), _e2435) - _e2322) / max((_e2312 - _e2322), 0.0001f)), 0f, 1f);
        let _e2465 = local;
        local = (_e2465 + (((((((vec3<f32>(1f, 1f, 1f) - _e2387) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e2387 * ((_e2356 * 0.31830987f) / ((_e2360 * _e2360) + 0.00001f))) * (0.5f / (((_e2352 * sqrt((((_e2350 * _e2350) * _e2368) + _e2367))) + (_e2350 * sqrt((((_e2352 * _e2352) * _e2368) + _e2367)))) + 0.00001f)))) * _e2352) * (((((_e2332 * _e2302) * max(dot(_e81, normalize(_e2327)), 0f)) * select(0f, 1f, (_e2297 < 0.5f))) + ((((_e2332 * _e2302) * max(dot(_e81, (_e2411 / vec3(max(_e2412, 0.0001f)))), 0f)) * ((_e2424 * _e2424) / max(_e2418, 0.0001f))) * select(0f, 1f, ((_e2297 > 0.5f) && (_e2297 < 1.5f))))) + (((((_e2332 * _e2302) * max(dot(_e81, _e2435), 0f)) * ((_e2446 * _e2446) / max(_e2440, 0.0001f))) * (_e2454 * _e2454)) * select(0f, 1f, (_e2297 > 1.5f))))));
    }
    if (13f < _e179) {
        let _e2473 = lights.member[i32(13f)].member;
        let _e2478 = lights.member[i32(13f)].member_1;
        let _e2483 = lights.member[i32(13f)].member_2;
        let _e2488 = lights.member[i32(13f)].member_3;
        let _e2493 = lights.member[i32(13f)].member_4;
        let _e2498 = lights.member[i32(13f)].member_5;
        let _e2503 = lights.member[i32(13f)].member_6;
        let _e2508 = lights.member[i32(13f)].member_8;
        let _e2509 = world_pos_1;
        let _e2514 = select(0f, 1f, (_e2473 < 0.5f));
        let _e2518 = ((normalize(_e2503) * _e2514) + (normalize((_e2493 - _e2509)) * (1f - _e2514)));
        let _e2520 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e2522 = normalize((_e86 + _e2518));
        let _e2524 = max(dot(_e81, _e2522), 0f);
        let _e2526 = max(dot(_e81, _e86), 0.00001f);
        let _e2528 = max(dot(_e81, _e2518), 0.00001f);
        let _e2531 = (_e150 * _e150);
        let _e2532 = (_e2531 * _e2531);
        let _e2536 = (((_e2524 * _e2524) * (_e2532 - 1f)) + 1f);
        let _e2543 = (((_e150 * _e150) * _e150) * _e150);
        let _e2544 = (1f - _e2543);
        let _e2563 = (_e2520 + ((vec3<f32>(1f, 1f, 1f) - _e2520) * pow(clamp((1f - max(dot(_e86, _e2522), 0f)), 0f, 1f), 5f)));
        let _e2573 = world_pos_1;
        let _e2587 = (_e2493 - _e2573);
        let _e2588 = length(_e2587);
        let _e2594 = (_e2588 * _e2588);
        let _e2597 = (_e2594 / max((_e2483 * _e2483), 0.0001f));
        let _e2600 = clamp((1f - (_e2597 * _e2597)), 0f, 1f);
        let _e2607 = (_e2493 - _e2573);
        let _e2608 = length(_e2607);
        let _e2611 = (_e2607 / vec3(max(_e2608, 0.0001f)));
        let _e2616 = (_e2608 * _e2608);
        let _e2619 = (_e2616 / max((_e2483 * _e2483), 0.0001f));
        let _e2622 = clamp((1f - (_e2619 * _e2619)), 0f, 1f);
        let _e2630 = clamp(((dot(normalize(_e2503), _e2611) - _e2498) / max((_e2488 - _e2498), 0.0001f)), 0f, 1f);
        let _e2641 = local;
        local = (_e2641 + (((((((vec3<f32>(1f, 1f, 1f) - _e2563) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e2563 * ((_e2532 * 0.31830987f) / ((_e2536 * _e2536) + 0.00001f))) * (0.5f / (((_e2528 * sqrt((((_e2526 * _e2526) * _e2544) + _e2543))) + (_e2526 * sqrt((((_e2528 * _e2528) * _e2544) + _e2543)))) + 0.00001f)))) * _e2528) * (((((_e2508 * _e2478) * max(dot(_e81, normalize(_e2503)), 0f)) * select(0f, 1f, (_e2473 < 0.5f))) + ((((_e2508 * _e2478) * max(dot(_e81, (_e2587 / vec3(max(_e2588, 0.0001f)))), 0f)) * ((_e2600 * _e2600) / max(_e2594, 0.0001f))) * select(0f, 1f, ((_e2473 > 0.5f) && (_e2473 < 1.5f))))) + (((((_e2508 * _e2478) * max(dot(_e81, _e2611), 0f)) * ((_e2622 * _e2622) / max(_e2616, 0.0001f))) * (_e2630 * _e2630)) * select(0f, 1f, (_e2473 > 1.5f))))));
    }
    if (14f < _e179) {
        let _e2649 = lights.member[i32(14f)].member;
        let _e2654 = lights.member[i32(14f)].member_1;
        let _e2659 = lights.member[i32(14f)].member_2;
        let _e2664 = lights.member[i32(14f)].member_3;
        let _e2669 = lights.member[i32(14f)].member_4;
        let _e2674 = lights.member[i32(14f)].member_5;
        let _e2679 = lights.member[i32(14f)].member_6;
        let _e2684 = lights.member[i32(14f)].member_8;
        let _e2685 = world_pos_1;
        let _e2690 = select(0f, 1f, (_e2649 < 0.5f));
        let _e2694 = ((normalize(_e2679) * _e2690) + (normalize((_e2669 - _e2685)) * (1f - _e2690)));
        let _e2696 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e2698 = normalize((_e86 + _e2694));
        let _e2700 = max(dot(_e81, _e2698), 0f);
        let _e2702 = max(dot(_e81, _e86), 0.00001f);
        let _e2704 = max(dot(_e81, _e2694), 0.00001f);
        let _e2707 = (_e150 * _e150);
        let _e2708 = (_e2707 * _e2707);
        let _e2712 = (((_e2700 * _e2700) * (_e2708 - 1f)) + 1f);
        let _e2719 = (((_e150 * _e150) * _e150) * _e150);
        let _e2720 = (1f - _e2719);
        let _e2739 = (_e2696 + ((vec3<f32>(1f, 1f, 1f) - _e2696) * pow(clamp((1f - max(dot(_e86, _e2698), 0f)), 0f, 1f), 5f)));
        let _e2749 = world_pos_1;
        let _e2763 = (_e2669 - _e2749);
        let _e2764 = length(_e2763);
        let _e2770 = (_e2764 * _e2764);
        let _e2773 = (_e2770 / max((_e2659 * _e2659), 0.0001f));
        let _e2776 = clamp((1f - (_e2773 * _e2773)), 0f, 1f);
        let _e2783 = (_e2669 - _e2749);
        let _e2784 = length(_e2783);
        let _e2787 = (_e2783 / vec3(max(_e2784, 0.0001f)));
        let _e2792 = (_e2784 * _e2784);
        let _e2795 = (_e2792 / max((_e2659 * _e2659), 0.0001f));
        let _e2798 = clamp((1f - (_e2795 * _e2795)), 0f, 1f);
        let _e2806 = clamp(((dot(normalize(_e2679), _e2787) - _e2674) / max((_e2664 - _e2674), 0.0001f)), 0f, 1f);
        let _e2817 = local;
        local = (_e2817 + (((((((vec3<f32>(1f, 1f, 1f) - _e2739) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e2739 * ((_e2708 * 0.31830987f) / ((_e2712 * _e2712) + 0.00001f))) * (0.5f / (((_e2704 * sqrt((((_e2702 * _e2702) * _e2720) + _e2719))) + (_e2702 * sqrt((((_e2704 * _e2704) * _e2720) + _e2719)))) + 0.00001f)))) * _e2704) * (((((_e2684 * _e2654) * max(dot(_e81, normalize(_e2679)), 0f)) * select(0f, 1f, (_e2649 < 0.5f))) + ((((_e2684 * _e2654) * max(dot(_e81, (_e2763 / vec3(max(_e2764, 0.0001f)))), 0f)) * ((_e2776 * _e2776) / max(_e2770, 0.0001f))) * select(0f, 1f, ((_e2649 > 0.5f) && (_e2649 < 1.5f))))) + (((((_e2684 * _e2654) * max(dot(_e81, _e2787), 0f)) * ((_e2798 * _e2798) / max(_e2792, 0.0001f))) * (_e2806 * _e2806)) * select(0f, 1f, (_e2649 > 1.5f))))));
    }
    if (15f < _e179) {
        let _e2825 = lights.member[i32(15f)].member;
        let _e2830 = lights.member[i32(15f)].member_1;
        let _e2835 = lights.member[i32(15f)].member_2;
        let _e2840 = lights.member[i32(15f)].member_3;
        let _e2845 = lights.member[i32(15f)].member_4;
        let _e2850 = lights.member[i32(15f)].member_5;
        let _e2855 = lights.member[i32(15f)].member_6;
        let _e2860 = lights.member[i32(15f)].member_8;
        let _e2861 = world_pos_1;
        let _e2866 = select(0f, 1f, (_e2825 < 0.5f));
        let _e2870 = ((normalize(_e2855) * _e2866) + (normalize((_e2845 - _e2861)) * (1f - _e2866)));
        let _e2872 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
        let _e2874 = normalize((_e86 + _e2870));
        let _e2876 = max(dot(_e81, _e2874), 0f);
        let _e2878 = max(dot(_e81, _e86), 0.00001f);
        let _e2880 = max(dot(_e81, _e2870), 0.00001f);
        let _e2883 = (_e150 * _e150);
        let _e2884 = (_e2883 * _e2883);
        let _e2888 = (((_e2876 * _e2876) * (_e2884 - 1f)) + 1f);
        let _e2895 = (((_e150 * _e150) * _e150) * _e150);
        let _e2896 = (1f - _e2895);
        let _e2915 = (_e2872 + ((vec3<f32>(1f, 1f, 1f) - _e2872) * pow(clamp((1f - max(dot(_e86, _e2874), 0f)), 0f, 1f), 5f)));
        let _e2925 = world_pos_1;
        let _e2939 = (_e2845 - _e2925);
        let _e2940 = length(_e2939);
        let _e2946 = (_e2940 * _e2940);
        let _e2949 = (_e2946 / max((_e2835 * _e2835), 0.0001f));
        let _e2952 = clamp((1f - (_e2949 * _e2949)), 0f, 1f);
        let _e2959 = (_e2845 - _e2925);
        let _e2960 = length(_e2959);
        let _e2963 = (_e2959 / vec3(max(_e2960, 0.0001f)));
        let _e2968 = (_e2960 * _e2960);
        let _e2971 = (_e2968 / max((_e2835 * _e2835), 0.0001f));
        let _e2974 = clamp((1f - (_e2971 * _e2971)), 0f, 1f);
        let _e2982 = clamp(((dot(normalize(_e2855), _e2963) - _e2850) / max((_e2840 - _e2850), 0.0001f)), 0f, 1f);
        let _e2993 = local;
        local = (_e2993 + (((((((vec3<f32>(1f, 1f, 1f) - _e2915) * (1f - _e176)) * _e120) * 0.31830987f) + ((_e2915 * ((_e2884 * 0.31830987f) / ((_e2888 * _e2888) + 0.00001f))) * (0.5f / (((_e2880 * sqrt((((_e2878 * _e2878) * _e2896) + _e2895))) + (_e2878 * sqrt((((_e2880 * _e2880) * _e2896) + _e2895)))) + 0.00001f)))) * _e2880) * (((((_e2860 * _e2830) * max(dot(_e81, normalize(_e2855)), 0f)) * select(0f, 1f, (_e2825 < 0.5f))) + ((((_e2860 * _e2830) * max(dot(_e81, (_e2939 / vec3(max(_e2940, 0.0001f)))), 0f)) * ((_e2952 * _e2952) / max(_e2946, 0.0001f))) * select(0f, 1f, ((_e2825 > 0.5f) && (_e2825 < 1.5f))))) + (((((_e2860 * _e2830) * max(dot(_e81, _e2963), 0f)) * ((_e2974 * _e2974) / max(_e2968, 0.0001f))) * (_e2982 * _e2982)) * select(0f, 1f, (_e2825 > 1.5f))))));
    }
    local_1 = vec3<f32>(0f, 0f, 0f);
    let _e2996 = textureSample(emissive_tex_u002e_texture, emissive_tex_u002e_sampler, _e79);
    let _e3000 = global_1.emissive_factor;
    let _e3003 = global_1.emissive_strength;
    local_1 = ((pow(_e2996.xyz, vec3<f32>(2.2f, 2.2f, 2.2f)) * _e3000) * _e3003);
    local_2 = vec3<f32>(0f, 0f, 0f);
    let _e3008 = textureSampleLevel(env_specular_u002e_texture, env_specular_u002e_sampler, reflect((_e86 * -1f), _e81), (_e150 * 8f));
    let _e3010 = textureSample(env_irradiance_u002e_texture, env_irradiance_u002e_sampler, _e81);
    let _e3013 = textureSample(brdf_lut_u002e_texture, brdf_lut_u002e_sampler, vec2<f32>(_e88, _e150));
    let _e3014 = _e3013.xy;
    let _e3016 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e120, vec3(_e176));
    let _e3033 = (((_e3016 + ((max(vec3((1f - _e150)), _e3016) - _e3016) * pow(clamp((1f - _e88), 0f, 1f), 5f))) * _e3014.x) + vec3(_e3014.y));
    let _e3036 = (_e3016 + ((vec3<f32>(1f, 1f, 1f) - _e3016) * 0.04761905f));
    let _e3037 = (1f - (_e3014.x + _e3014.y));
    let _e3043 = (_e3033 + (((_e3033 * _e3036) / (vec3<f32>(1f, 1f, 1f) - (_e3036 * _e3037))) * _e3037));
    local_2 = (((((vec3<f32>(1f, 1f, 1f) - _e3043) * (1f - _e176)) * _e120) * _e3010.xyz) + (_e3043 * _e3008.xyz));
    let _e3051 = local;
    let _e3052 = local_2;
    let _e3053 = local_1;
    let _e3057 = max(dot(_e81, normalize((_e86 + vec3<f32>(0f, 1f, 0f)))), 0f);
    let _e3059 = max(dot(_e81, _e86), 0.00001f);
    let _e3061 = max(dot(_e81, vec3<f32>(0f, 1f, 0f)), 0.00001f);
    let _e3062 = (_e150 * _e150);
    let _e3063 = (_e3062 * _e3062);
    let _e3067 = (((_e3057 * _e3057) * (_e3063 - 1f)) + 1f);
    let _e3074 = (((_e150 * _e150) * _e150) * _e150);
    let _e3075 = (1f - _e3074);
    let _e3102 = max(0f, 0.045f);
    let _e3106 = max(dot(_e81, normalize((_e86 + vec3<f32>(0f, 1f, 0f)))), 0.001f);
    let _e3108 = max(dot(_e81, vec3<f32>(0f, 1f, 0f)), 0f);
    let _e3110 = max(dot(_e81, _e86), 0.001f);
    let _e3112 = (1f / (_e3102 * _e3102));
    let _e3142 = max(0f, 0.045f);
    let _e3146 = normalize((_e86 + vec3<f32>(0f, 1f, 0f)));
    let _e3148 = max(dot(_e81, _e3146), 0f);
    let _e3150 = max(dot(_e81, vec3<f32>(0f, 1f, 0f)), 0f);
    let _e3152 = max(dot(_e81, _e86), 0f);
    let _e3155 = (_e3142 * _e3142);
    let _e3156 = (_e3155 * _e3155);
    let _e3160 = (((_e3148 * _e3148) * (_e3156 - 1f)) + 1f);
    let _e3167 = (((_e3142 * _e3142) * _e3142) * _e3142);
    let _e3168 = (1f - _e3167);
    let _e3203 = (((((((mix(_e3051, (((_e120 * (((_e3063 * 0.31830987f) / ((_e3067 * _e3067) + 0.00001f)) * (0.5f / (((_e3061 * sqrt((((_e3059 * _e3059) * _e3075) + _e3074))) + (_e3059 * sqrt((((_e3061 * _e3061) * _e3075) + _e3074)))) + 0.00001f)))) * 0f) * exp(((log(vec3<f32>(1f, 1f, 1f)) / vec3(1000000f)) * 0f))), vec3(0f)) + _e3052) + vec3<f32>(0f, 0f, 0f)) * (1f - (max(vec3<f32>(0f, 0f, 0f).x, max(vec3<f32>(0f, 0f, 0f).y, vec3<f32>(0f, 0f, 0f).z)) * (0.09f + (0.42f * _e3102))))) + (((vec3<f32>(0f, 0f, 0f) * (((2f + _e3112) * pow(max((1f - (_e3106 * _e3106)), 0f), (_e3112 * 0.5f))) / 6.2831855f)) * (1f / ((4f * ((_e3108 + _e3110) - (_e3108 * _e3110))) + 0.00001f))) * max(_e3108, 0f))) * (1f - ((vec3<f32>(0.04f, 0.04f, 0.04f) + ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(0.04f, 0.04f, 0.04f)) * pow(clamp((1f - max(dot(_e81, _e86), 0.001f)), 0f, 1f), 5f))).x * 0f))) + vec3(((((0f * ((_e3156 * 0.31830987f) / ((_e3160 * _e3160) + 0.00001f))) * (0.5f / (((_e3150 * sqrt((((_e3152 * _e3152) * _e3168) + _e3167))) + (_e3152 * sqrt((((_e3150 * _e3150) * _e3168) + _e3167)))) + 0.00001f))) * (0.04f + ((1f - 0.04f) * pow((1f - max(dot(_e86, _e3146), 0f)), 5f)))) * _e3150))) + _e3053);
    let _e3216 = pow(clamp(((_e3203 * ((_e3203 * 2.51f) + vec3(0.03f))) / ((_e3203 * ((_e3203 * 2.43f) + vec3(0.59f))) + vec3(0.14f))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f)), vec3<f32>(0.45454547f, 0.45454547f, 0.45454547f));
    color = vec4<f32>(_e3216.x, _e3216.y, _e3216.z, 1f);
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
