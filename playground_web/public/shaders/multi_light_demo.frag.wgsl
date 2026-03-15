struct SceneLight {
    view_pos: vec3<f32>,
    light_count: i32,
}

struct Material {
    base_color_factor: vec4<f32>,
    metallic_factor: f32,
    roughness_factor: f32,
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
@group(1) @binding(12) 
var<storage> lights: type_13;

fn main_1() {
    var local: vec3<f32>;
    var local_1: vec3<f32>;

    let _e70 = frag_uv_1;
    let _e71 = world_normal_1;
    let _e72 = normalize(_e71);
    let _e74 = global.view_pos;
    let _e75 = world_pos_1;
    let _e77 = normalize((_e74 - _e75));
    let _e79 = max(dot(_e72, _e77), 0.001f);
    let _e80 = textureSample(base_color_tex_u002e_texture, base_color_tex_u002e_sampler, _e70);
    let _e84 = global_1.base_color_factor;
    let _e86 = (pow(_e80.xyz, vec3<f32>(2.2f, 2.2f, 2.2f)) * _e84.xyz);
    let _e87 = textureSample(metallic_roughness_tex_u002e_texture, metallic_roughness_tex_u002e_sampler, _e70);
    let _e90 = global_1.roughness_factor;
    let _e91 = (_e87.y * _e90);
    let _e92 = textureSample(metallic_roughness_tex_u002e_texture, metallic_roughness_tex_u002e_sampler, _e70);
    let _e95 = global_1.metallic_factor;
    let _e96 = (_e92.z * _e95);
    let _e98 = global.light_count;
    let _e99 = f32(_e98);
    local = vec3<f32>(0f, 0f, 0f);
    if (0f < _e99) {
        let _e105 = lights.member[i32(0f)].member;
        let _e110 = lights.member[i32(0f)].member_1;
        let _e115 = lights.member[i32(0f)].member_2;
        let _e120 = lights.member[i32(0f)].member_3;
        let _e125 = lights.member[i32(0f)].member_4;
        let _e130 = lights.member[i32(0f)].member_5;
        let _e135 = lights.member[i32(0f)].member_6;
        let _e140 = lights.member[i32(0f)].member_8;
        let _e141 = world_pos_1;
        let _e146 = select(0f, 1f, (_e105 < 0.5f));
        let _e150 = ((normalize(_e135) * _e146) + (normalize((_e125 - _e141)) * (1f - _e146)));
        let _e152 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e154 = normalize((_e77 + _e150));
        let _e156 = max(dot(_e72, _e154), 0f);
        let _e158 = max(dot(_e72, _e77), 0.00001f);
        let _e160 = max(dot(_e72, _e150), 0.00001f);
        let _e163 = (_e91 * _e91);
        let _e164 = (_e163 * _e163);
        let _e168 = (((_e156 * _e156) * (_e164 - 1f)) + 1f);
        let _e175 = (((_e91 * _e91) * _e91) * _e91);
        let _e176 = (1f - _e175);
        let _e195 = (_e152 + ((vec3<f32>(1f, 1f, 1f) - _e152) * pow(clamp((1f - max(dot(_e77, _e154), 0f)), 0f, 1f), 5f)));
        let _e205 = world_pos_1;
        let _e219 = (_e125 - _e205);
        let _e220 = length(_e219);
        let _e226 = (_e220 * _e220);
        let _e229 = (_e226 / max((_e115 * _e115), 0.0001f));
        let _e232 = clamp((1f - (_e229 * _e229)), 0f, 1f);
        let _e239 = (_e125 - _e205);
        let _e240 = length(_e239);
        let _e243 = (_e239 / vec3(max(_e240, 0.0001f)));
        let _e248 = (_e240 * _e240);
        let _e251 = (_e248 / max((_e115 * _e115), 0.0001f));
        let _e254 = clamp((1f - (_e251 * _e251)), 0f, 1f);
        let _e262 = clamp(((dot(normalize(_e135), _e243) - _e130) / max((_e120 - _e130), 0.0001f)), 0f, 1f);
        let _e273 = local;
        local = (_e273 + (((((((vec3<f32>(1f, 1f, 1f) - _e195) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e195 * ((_e164 * 0.31830987f) / ((_e168 * _e168) + 0.00001f))) * (0.5f / (((_e160 * sqrt((((_e158 * _e158) * _e176) + _e175))) + (_e158 * sqrt((((_e160 * _e160) * _e176) + _e175)))) + 0.00001f)))) * _e160) * (((((_e140 * _e110) * max(dot(_e72, normalize(_e135)), 0f)) * select(0f, 1f, (_e105 < 0.5f))) + ((((_e140 * _e110) * max(dot(_e72, (_e219 / vec3(max(_e220, 0.0001f)))), 0f)) * ((_e232 * _e232) / max(_e226, 0.0001f))) * select(0f, 1f, ((_e105 > 0.5f) && (_e105 < 1.5f))))) + (((((_e140 * _e110) * max(dot(_e72, _e243), 0f)) * ((_e254 * _e254) / max(_e248, 0.0001f))) * (_e262 * _e262)) * select(0f, 1f, (_e105 > 1.5f))))));
    }
    if (1f < _e99) {
        let _e281 = lights.member[i32(1f)].member;
        let _e286 = lights.member[i32(1f)].member_1;
        let _e291 = lights.member[i32(1f)].member_2;
        let _e296 = lights.member[i32(1f)].member_3;
        let _e301 = lights.member[i32(1f)].member_4;
        let _e306 = lights.member[i32(1f)].member_5;
        let _e311 = lights.member[i32(1f)].member_6;
        let _e316 = lights.member[i32(1f)].member_8;
        let _e317 = world_pos_1;
        let _e322 = select(0f, 1f, (_e281 < 0.5f));
        let _e326 = ((normalize(_e311) * _e322) + (normalize((_e301 - _e317)) * (1f - _e322)));
        let _e328 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e330 = normalize((_e77 + _e326));
        let _e332 = max(dot(_e72, _e330), 0f);
        let _e334 = max(dot(_e72, _e77), 0.00001f);
        let _e336 = max(dot(_e72, _e326), 0.00001f);
        let _e339 = (_e91 * _e91);
        let _e340 = (_e339 * _e339);
        let _e344 = (((_e332 * _e332) * (_e340 - 1f)) + 1f);
        let _e351 = (((_e91 * _e91) * _e91) * _e91);
        let _e352 = (1f - _e351);
        let _e371 = (_e328 + ((vec3<f32>(1f, 1f, 1f) - _e328) * pow(clamp((1f - max(dot(_e77, _e330), 0f)), 0f, 1f), 5f)));
        let _e381 = world_pos_1;
        let _e395 = (_e301 - _e381);
        let _e396 = length(_e395);
        let _e402 = (_e396 * _e396);
        let _e405 = (_e402 / max((_e291 * _e291), 0.0001f));
        let _e408 = clamp((1f - (_e405 * _e405)), 0f, 1f);
        let _e415 = (_e301 - _e381);
        let _e416 = length(_e415);
        let _e419 = (_e415 / vec3(max(_e416, 0.0001f)));
        let _e424 = (_e416 * _e416);
        let _e427 = (_e424 / max((_e291 * _e291), 0.0001f));
        let _e430 = clamp((1f - (_e427 * _e427)), 0f, 1f);
        let _e438 = clamp(((dot(normalize(_e311), _e419) - _e306) / max((_e296 - _e306), 0.0001f)), 0f, 1f);
        let _e449 = local;
        local = (_e449 + (((((((vec3<f32>(1f, 1f, 1f) - _e371) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e371 * ((_e340 * 0.31830987f) / ((_e344 * _e344) + 0.00001f))) * (0.5f / (((_e336 * sqrt((((_e334 * _e334) * _e352) + _e351))) + (_e334 * sqrt((((_e336 * _e336) * _e352) + _e351)))) + 0.00001f)))) * _e336) * (((((_e316 * _e286) * max(dot(_e72, normalize(_e311)), 0f)) * select(0f, 1f, (_e281 < 0.5f))) + ((((_e316 * _e286) * max(dot(_e72, (_e395 / vec3(max(_e396, 0.0001f)))), 0f)) * ((_e408 * _e408) / max(_e402, 0.0001f))) * select(0f, 1f, ((_e281 > 0.5f) && (_e281 < 1.5f))))) + (((((_e316 * _e286) * max(dot(_e72, _e419), 0f)) * ((_e430 * _e430) / max(_e424, 0.0001f))) * (_e438 * _e438)) * select(0f, 1f, (_e281 > 1.5f))))));
    }
    if (2f < _e99) {
        let _e457 = lights.member[i32(2f)].member;
        let _e462 = lights.member[i32(2f)].member_1;
        let _e467 = lights.member[i32(2f)].member_2;
        let _e472 = lights.member[i32(2f)].member_3;
        let _e477 = lights.member[i32(2f)].member_4;
        let _e482 = lights.member[i32(2f)].member_5;
        let _e487 = lights.member[i32(2f)].member_6;
        let _e492 = lights.member[i32(2f)].member_8;
        let _e493 = world_pos_1;
        let _e498 = select(0f, 1f, (_e457 < 0.5f));
        let _e502 = ((normalize(_e487) * _e498) + (normalize((_e477 - _e493)) * (1f - _e498)));
        let _e504 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e506 = normalize((_e77 + _e502));
        let _e508 = max(dot(_e72, _e506), 0f);
        let _e510 = max(dot(_e72, _e77), 0.00001f);
        let _e512 = max(dot(_e72, _e502), 0.00001f);
        let _e515 = (_e91 * _e91);
        let _e516 = (_e515 * _e515);
        let _e520 = (((_e508 * _e508) * (_e516 - 1f)) + 1f);
        let _e527 = (((_e91 * _e91) * _e91) * _e91);
        let _e528 = (1f - _e527);
        let _e547 = (_e504 + ((vec3<f32>(1f, 1f, 1f) - _e504) * pow(clamp((1f - max(dot(_e77, _e506), 0f)), 0f, 1f), 5f)));
        let _e557 = world_pos_1;
        let _e571 = (_e477 - _e557);
        let _e572 = length(_e571);
        let _e578 = (_e572 * _e572);
        let _e581 = (_e578 / max((_e467 * _e467), 0.0001f));
        let _e584 = clamp((1f - (_e581 * _e581)), 0f, 1f);
        let _e591 = (_e477 - _e557);
        let _e592 = length(_e591);
        let _e595 = (_e591 / vec3(max(_e592, 0.0001f)));
        let _e600 = (_e592 * _e592);
        let _e603 = (_e600 / max((_e467 * _e467), 0.0001f));
        let _e606 = clamp((1f - (_e603 * _e603)), 0f, 1f);
        let _e614 = clamp(((dot(normalize(_e487), _e595) - _e482) / max((_e472 - _e482), 0.0001f)), 0f, 1f);
        let _e625 = local;
        local = (_e625 + (((((((vec3<f32>(1f, 1f, 1f) - _e547) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e547 * ((_e516 * 0.31830987f) / ((_e520 * _e520) + 0.00001f))) * (0.5f / (((_e512 * sqrt((((_e510 * _e510) * _e528) + _e527))) + (_e510 * sqrt((((_e512 * _e512) * _e528) + _e527)))) + 0.00001f)))) * _e512) * (((((_e492 * _e462) * max(dot(_e72, normalize(_e487)), 0f)) * select(0f, 1f, (_e457 < 0.5f))) + ((((_e492 * _e462) * max(dot(_e72, (_e571 / vec3(max(_e572, 0.0001f)))), 0f)) * ((_e584 * _e584) / max(_e578, 0.0001f))) * select(0f, 1f, ((_e457 > 0.5f) && (_e457 < 1.5f))))) + (((((_e492 * _e462) * max(dot(_e72, _e595), 0f)) * ((_e606 * _e606) / max(_e600, 0.0001f))) * (_e614 * _e614)) * select(0f, 1f, (_e457 > 1.5f))))));
    }
    if (3f < _e99) {
        let _e633 = lights.member[i32(3f)].member;
        let _e638 = lights.member[i32(3f)].member_1;
        let _e643 = lights.member[i32(3f)].member_2;
        let _e648 = lights.member[i32(3f)].member_3;
        let _e653 = lights.member[i32(3f)].member_4;
        let _e658 = lights.member[i32(3f)].member_5;
        let _e663 = lights.member[i32(3f)].member_6;
        let _e668 = lights.member[i32(3f)].member_8;
        let _e669 = world_pos_1;
        let _e674 = select(0f, 1f, (_e633 < 0.5f));
        let _e678 = ((normalize(_e663) * _e674) + (normalize((_e653 - _e669)) * (1f - _e674)));
        let _e680 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e682 = normalize((_e77 + _e678));
        let _e684 = max(dot(_e72, _e682), 0f);
        let _e686 = max(dot(_e72, _e77), 0.00001f);
        let _e688 = max(dot(_e72, _e678), 0.00001f);
        let _e691 = (_e91 * _e91);
        let _e692 = (_e691 * _e691);
        let _e696 = (((_e684 * _e684) * (_e692 - 1f)) + 1f);
        let _e703 = (((_e91 * _e91) * _e91) * _e91);
        let _e704 = (1f - _e703);
        let _e723 = (_e680 + ((vec3<f32>(1f, 1f, 1f) - _e680) * pow(clamp((1f - max(dot(_e77, _e682), 0f)), 0f, 1f), 5f)));
        let _e733 = world_pos_1;
        let _e747 = (_e653 - _e733);
        let _e748 = length(_e747);
        let _e754 = (_e748 * _e748);
        let _e757 = (_e754 / max((_e643 * _e643), 0.0001f));
        let _e760 = clamp((1f - (_e757 * _e757)), 0f, 1f);
        let _e767 = (_e653 - _e733);
        let _e768 = length(_e767);
        let _e771 = (_e767 / vec3(max(_e768, 0.0001f)));
        let _e776 = (_e768 * _e768);
        let _e779 = (_e776 / max((_e643 * _e643), 0.0001f));
        let _e782 = clamp((1f - (_e779 * _e779)), 0f, 1f);
        let _e790 = clamp(((dot(normalize(_e663), _e771) - _e658) / max((_e648 - _e658), 0.0001f)), 0f, 1f);
        let _e801 = local;
        local = (_e801 + (((((((vec3<f32>(1f, 1f, 1f) - _e723) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e723 * ((_e692 * 0.31830987f) / ((_e696 * _e696) + 0.00001f))) * (0.5f / (((_e688 * sqrt((((_e686 * _e686) * _e704) + _e703))) + (_e686 * sqrt((((_e688 * _e688) * _e704) + _e703)))) + 0.00001f)))) * _e688) * (((((_e668 * _e638) * max(dot(_e72, normalize(_e663)), 0f)) * select(0f, 1f, (_e633 < 0.5f))) + ((((_e668 * _e638) * max(dot(_e72, (_e747 / vec3(max(_e748, 0.0001f)))), 0f)) * ((_e760 * _e760) / max(_e754, 0.0001f))) * select(0f, 1f, ((_e633 > 0.5f) && (_e633 < 1.5f))))) + (((((_e668 * _e638) * max(dot(_e72, _e771), 0f)) * ((_e782 * _e782) / max(_e776, 0.0001f))) * (_e790 * _e790)) * select(0f, 1f, (_e633 > 1.5f))))));
    }
    if (4f < _e99) {
        let _e809 = lights.member[i32(4f)].member;
        let _e814 = lights.member[i32(4f)].member_1;
        let _e819 = lights.member[i32(4f)].member_2;
        let _e824 = lights.member[i32(4f)].member_3;
        let _e829 = lights.member[i32(4f)].member_4;
        let _e834 = lights.member[i32(4f)].member_5;
        let _e839 = lights.member[i32(4f)].member_6;
        let _e844 = lights.member[i32(4f)].member_8;
        let _e845 = world_pos_1;
        let _e850 = select(0f, 1f, (_e809 < 0.5f));
        let _e854 = ((normalize(_e839) * _e850) + (normalize((_e829 - _e845)) * (1f - _e850)));
        let _e856 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e858 = normalize((_e77 + _e854));
        let _e860 = max(dot(_e72, _e858), 0f);
        let _e862 = max(dot(_e72, _e77), 0.00001f);
        let _e864 = max(dot(_e72, _e854), 0.00001f);
        let _e867 = (_e91 * _e91);
        let _e868 = (_e867 * _e867);
        let _e872 = (((_e860 * _e860) * (_e868 - 1f)) + 1f);
        let _e879 = (((_e91 * _e91) * _e91) * _e91);
        let _e880 = (1f - _e879);
        let _e899 = (_e856 + ((vec3<f32>(1f, 1f, 1f) - _e856) * pow(clamp((1f - max(dot(_e77, _e858), 0f)), 0f, 1f), 5f)));
        let _e909 = world_pos_1;
        let _e923 = (_e829 - _e909);
        let _e924 = length(_e923);
        let _e930 = (_e924 * _e924);
        let _e933 = (_e930 / max((_e819 * _e819), 0.0001f));
        let _e936 = clamp((1f - (_e933 * _e933)), 0f, 1f);
        let _e943 = (_e829 - _e909);
        let _e944 = length(_e943);
        let _e947 = (_e943 / vec3(max(_e944, 0.0001f)));
        let _e952 = (_e944 * _e944);
        let _e955 = (_e952 / max((_e819 * _e819), 0.0001f));
        let _e958 = clamp((1f - (_e955 * _e955)), 0f, 1f);
        let _e966 = clamp(((dot(normalize(_e839), _e947) - _e834) / max((_e824 - _e834), 0.0001f)), 0f, 1f);
        let _e977 = local;
        local = (_e977 + (((((((vec3<f32>(1f, 1f, 1f) - _e899) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e899 * ((_e868 * 0.31830987f) / ((_e872 * _e872) + 0.00001f))) * (0.5f / (((_e864 * sqrt((((_e862 * _e862) * _e880) + _e879))) + (_e862 * sqrt((((_e864 * _e864) * _e880) + _e879)))) + 0.00001f)))) * _e864) * (((((_e844 * _e814) * max(dot(_e72, normalize(_e839)), 0f)) * select(0f, 1f, (_e809 < 0.5f))) + ((((_e844 * _e814) * max(dot(_e72, (_e923 / vec3(max(_e924, 0.0001f)))), 0f)) * ((_e936 * _e936) / max(_e930, 0.0001f))) * select(0f, 1f, ((_e809 > 0.5f) && (_e809 < 1.5f))))) + (((((_e844 * _e814) * max(dot(_e72, _e947), 0f)) * ((_e958 * _e958) / max(_e952, 0.0001f))) * (_e966 * _e966)) * select(0f, 1f, (_e809 > 1.5f))))));
    }
    if (5f < _e99) {
        let _e985 = lights.member[i32(5f)].member;
        let _e990 = lights.member[i32(5f)].member_1;
        let _e995 = lights.member[i32(5f)].member_2;
        let _e1000 = lights.member[i32(5f)].member_3;
        let _e1005 = lights.member[i32(5f)].member_4;
        let _e1010 = lights.member[i32(5f)].member_5;
        let _e1015 = lights.member[i32(5f)].member_6;
        let _e1020 = lights.member[i32(5f)].member_8;
        let _e1021 = world_pos_1;
        let _e1026 = select(0f, 1f, (_e985 < 0.5f));
        let _e1030 = ((normalize(_e1015) * _e1026) + (normalize((_e1005 - _e1021)) * (1f - _e1026)));
        let _e1032 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e1034 = normalize((_e77 + _e1030));
        let _e1036 = max(dot(_e72, _e1034), 0f);
        let _e1038 = max(dot(_e72, _e77), 0.00001f);
        let _e1040 = max(dot(_e72, _e1030), 0.00001f);
        let _e1043 = (_e91 * _e91);
        let _e1044 = (_e1043 * _e1043);
        let _e1048 = (((_e1036 * _e1036) * (_e1044 - 1f)) + 1f);
        let _e1055 = (((_e91 * _e91) * _e91) * _e91);
        let _e1056 = (1f - _e1055);
        let _e1075 = (_e1032 + ((vec3<f32>(1f, 1f, 1f) - _e1032) * pow(clamp((1f - max(dot(_e77, _e1034), 0f)), 0f, 1f), 5f)));
        let _e1085 = world_pos_1;
        let _e1099 = (_e1005 - _e1085);
        let _e1100 = length(_e1099);
        let _e1106 = (_e1100 * _e1100);
        let _e1109 = (_e1106 / max((_e995 * _e995), 0.0001f));
        let _e1112 = clamp((1f - (_e1109 * _e1109)), 0f, 1f);
        let _e1119 = (_e1005 - _e1085);
        let _e1120 = length(_e1119);
        let _e1123 = (_e1119 / vec3(max(_e1120, 0.0001f)));
        let _e1128 = (_e1120 * _e1120);
        let _e1131 = (_e1128 / max((_e995 * _e995), 0.0001f));
        let _e1134 = clamp((1f - (_e1131 * _e1131)), 0f, 1f);
        let _e1142 = clamp(((dot(normalize(_e1015), _e1123) - _e1010) / max((_e1000 - _e1010), 0.0001f)), 0f, 1f);
        let _e1153 = local;
        local = (_e1153 + (((((((vec3<f32>(1f, 1f, 1f) - _e1075) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e1075 * ((_e1044 * 0.31830987f) / ((_e1048 * _e1048) + 0.00001f))) * (0.5f / (((_e1040 * sqrt((((_e1038 * _e1038) * _e1056) + _e1055))) + (_e1038 * sqrt((((_e1040 * _e1040) * _e1056) + _e1055)))) + 0.00001f)))) * _e1040) * (((((_e1020 * _e990) * max(dot(_e72, normalize(_e1015)), 0f)) * select(0f, 1f, (_e985 < 0.5f))) + ((((_e1020 * _e990) * max(dot(_e72, (_e1099 / vec3(max(_e1100, 0.0001f)))), 0f)) * ((_e1112 * _e1112) / max(_e1106, 0.0001f))) * select(0f, 1f, ((_e985 > 0.5f) && (_e985 < 1.5f))))) + (((((_e1020 * _e990) * max(dot(_e72, _e1123), 0f)) * ((_e1134 * _e1134) / max(_e1128, 0.0001f))) * (_e1142 * _e1142)) * select(0f, 1f, (_e985 > 1.5f))))));
    }
    if (6f < _e99) {
        let _e1161 = lights.member[i32(6f)].member;
        let _e1166 = lights.member[i32(6f)].member_1;
        let _e1171 = lights.member[i32(6f)].member_2;
        let _e1176 = lights.member[i32(6f)].member_3;
        let _e1181 = lights.member[i32(6f)].member_4;
        let _e1186 = lights.member[i32(6f)].member_5;
        let _e1191 = lights.member[i32(6f)].member_6;
        let _e1196 = lights.member[i32(6f)].member_8;
        let _e1197 = world_pos_1;
        let _e1202 = select(0f, 1f, (_e1161 < 0.5f));
        let _e1206 = ((normalize(_e1191) * _e1202) + (normalize((_e1181 - _e1197)) * (1f - _e1202)));
        let _e1208 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e1210 = normalize((_e77 + _e1206));
        let _e1212 = max(dot(_e72, _e1210), 0f);
        let _e1214 = max(dot(_e72, _e77), 0.00001f);
        let _e1216 = max(dot(_e72, _e1206), 0.00001f);
        let _e1219 = (_e91 * _e91);
        let _e1220 = (_e1219 * _e1219);
        let _e1224 = (((_e1212 * _e1212) * (_e1220 - 1f)) + 1f);
        let _e1231 = (((_e91 * _e91) * _e91) * _e91);
        let _e1232 = (1f - _e1231);
        let _e1251 = (_e1208 + ((vec3<f32>(1f, 1f, 1f) - _e1208) * pow(clamp((1f - max(dot(_e77, _e1210), 0f)), 0f, 1f), 5f)));
        let _e1261 = world_pos_1;
        let _e1275 = (_e1181 - _e1261);
        let _e1276 = length(_e1275);
        let _e1282 = (_e1276 * _e1276);
        let _e1285 = (_e1282 / max((_e1171 * _e1171), 0.0001f));
        let _e1288 = clamp((1f - (_e1285 * _e1285)), 0f, 1f);
        let _e1295 = (_e1181 - _e1261);
        let _e1296 = length(_e1295);
        let _e1299 = (_e1295 / vec3(max(_e1296, 0.0001f)));
        let _e1304 = (_e1296 * _e1296);
        let _e1307 = (_e1304 / max((_e1171 * _e1171), 0.0001f));
        let _e1310 = clamp((1f - (_e1307 * _e1307)), 0f, 1f);
        let _e1318 = clamp(((dot(normalize(_e1191), _e1299) - _e1186) / max((_e1176 - _e1186), 0.0001f)), 0f, 1f);
        let _e1329 = local;
        local = (_e1329 + (((((((vec3<f32>(1f, 1f, 1f) - _e1251) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e1251 * ((_e1220 * 0.31830987f) / ((_e1224 * _e1224) + 0.00001f))) * (0.5f / (((_e1216 * sqrt((((_e1214 * _e1214) * _e1232) + _e1231))) + (_e1214 * sqrt((((_e1216 * _e1216) * _e1232) + _e1231)))) + 0.00001f)))) * _e1216) * (((((_e1196 * _e1166) * max(dot(_e72, normalize(_e1191)), 0f)) * select(0f, 1f, (_e1161 < 0.5f))) + ((((_e1196 * _e1166) * max(dot(_e72, (_e1275 / vec3(max(_e1276, 0.0001f)))), 0f)) * ((_e1288 * _e1288) / max(_e1282, 0.0001f))) * select(0f, 1f, ((_e1161 > 0.5f) && (_e1161 < 1.5f))))) + (((((_e1196 * _e1166) * max(dot(_e72, _e1299), 0f)) * ((_e1310 * _e1310) / max(_e1304, 0.0001f))) * (_e1318 * _e1318)) * select(0f, 1f, (_e1161 > 1.5f))))));
    }
    if (7f < _e99) {
        let _e1337 = lights.member[i32(7f)].member;
        let _e1342 = lights.member[i32(7f)].member_1;
        let _e1347 = lights.member[i32(7f)].member_2;
        let _e1352 = lights.member[i32(7f)].member_3;
        let _e1357 = lights.member[i32(7f)].member_4;
        let _e1362 = lights.member[i32(7f)].member_5;
        let _e1367 = lights.member[i32(7f)].member_6;
        let _e1372 = lights.member[i32(7f)].member_8;
        let _e1373 = world_pos_1;
        let _e1378 = select(0f, 1f, (_e1337 < 0.5f));
        let _e1382 = ((normalize(_e1367) * _e1378) + (normalize((_e1357 - _e1373)) * (1f - _e1378)));
        let _e1384 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e1386 = normalize((_e77 + _e1382));
        let _e1388 = max(dot(_e72, _e1386), 0f);
        let _e1390 = max(dot(_e72, _e77), 0.00001f);
        let _e1392 = max(dot(_e72, _e1382), 0.00001f);
        let _e1395 = (_e91 * _e91);
        let _e1396 = (_e1395 * _e1395);
        let _e1400 = (((_e1388 * _e1388) * (_e1396 - 1f)) + 1f);
        let _e1407 = (((_e91 * _e91) * _e91) * _e91);
        let _e1408 = (1f - _e1407);
        let _e1427 = (_e1384 + ((vec3<f32>(1f, 1f, 1f) - _e1384) * pow(clamp((1f - max(dot(_e77, _e1386), 0f)), 0f, 1f), 5f)));
        let _e1437 = world_pos_1;
        let _e1451 = (_e1357 - _e1437);
        let _e1452 = length(_e1451);
        let _e1458 = (_e1452 * _e1452);
        let _e1461 = (_e1458 / max((_e1347 * _e1347), 0.0001f));
        let _e1464 = clamp((1f - (_e1461 * _e1461)), 0f, 1f);
        let _e1471 = (_e1357 - _e1437);
        let _e1472 = length(_e1471);
        let _e1475 = (_e1471 / vec3(max(_e1472, 0.0001f)));
        let _e1480 = (_e1472 * _e1472);
        let _e1483 = (_e1480 / max((_e1347 * _e1347), 0.0001f));
        let _e1486 = clamp((1f - (_e1483 * _e1483)), 0f, 1f);
        let _e1494 = clamp(((dot(normalize(_e1367), _e1475) - _e1362) / max((_e1352 - _e1362), 0.0001f)), 0f, 1f);
        let _e1505 = local;
        local = (_e1505 + (((((((vec3<f32>(1f, 1f, 1f) - _e1427) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e1427 * ((_e1396 * 0.31830987f) / ((_e1400 * _e1400) + 0.00001f))) * (0.5f / (((_e1392 * sqrt((((_e1390 * _e1390) * _e1408) + _e1407))) + (_e1390 * sqrt((((_e1392 * _e1392) * _e1408) + _e1407)))) + 0.00001f)))) * _e1392) * (((((_e1372 * _e1342) * max(dot(_e72, normalize(_e1367)), 0f)) * select(0f, 1f, (_e1337 < 0.5f))) + ((((_e1372 * _e1342) * max(dot(_e72, (_e1451 / vec3(max(_e1452, 0.0001f)))), 0f)) * ((_e1464 * _e1464) / max(_e1458, 0.0001f))) * select(0f, 1f, ((_e1337 > 0.5f) && (_e1337 < 1.5f))))) + (((((_e1372 * _e1342) * max(dot(_e72, _e1475), 0f)) * ((_e1486 * _e1486) / max(_e1480, 0.0001f))) * (_e1494 * _e1494)) * select(0f, 1f, (_e1337 > 1.5f))))));
    }
    if (8f < _e99) {
        let _e1513 = lights.member[i32(8f)].member;
        let _e1518 = lights.member[i32(8f)].member_1;
        let _e1523 = lights.member[i32(8f)].member_2;
        let _e1528 = lights.member[i32(8f)].member_3;
        let _e1533 = lights.member[i32(8f)].member_4;
        let _e1538 = lights.member[i32(8f)].member_5;
        let _e1543 = lights.member[i32(8f)].member_6;
        let _e1548 = lights.member[i32(8f)].member_8;
        let _e1549 = world_pos_1;
        let _e1554 = select(0f, 1f, (_e1513 < 0.5f));
        let _e1558 = ((normalize(_e1543) * _e1554) + (normalize((_e1533 - _e1549)) * (1f - _e1554)));
        let _e1560 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e1562 = normalize((_e77 + _e1558));
        let _e1564 = max(dot(_e72, _e1562), 0f);
        let _e1566 = max(dot(_e72, _e77), 0.00001f);
        let _e1568 = max(dot(_e72, _e1558), 0.00001f);
        let _e1571 = (_e91 * _e91);
        let _e1572 = (_e1571 * _e1571);
        let _e1576 = (((_e1564 * _e1564) * (_e1572 - 1f)) + 1f);
        let _e1583 = (((_e91 * _e91) * _e91) * _e91);
        let _e1584 = (1f - _e1583);
        let _e1603 = (_e1560 + ((vec3<f32>(1f, 1f, 1f) - _e1560) * pow(clamp((1f - max(dot(_e77, _e1562), 0f)), 0f, 1f), 5f)));
        let _e1613 = world_pos_1;
        let _e1627 = (_e1533 - _e1613);
        let _e1628 = length(_e1627);
        let _e1634 = (_e1628 * _e1628);
        let _e1637 = (_e1634 / max((_e1523 * _e1523), 0.0001f));
        let _e1640 = clamp((1f - (_e1637 * _e1637)), 0f, 1f);
        let _e1647 = (_e1533 - _e1613);
        let _e1648 = length(_e1647);
        let _e1651 = (_e1647 / vec3(max(_e1648, 0.0001f)));
        let _e1656 = (_e1648 * _e1648);
        let _e1659 = (_e1656 / max((_e1523 * _e1523), 0.0001f));
        let _e1662 = clamp((1f - (_e1659 * _e1659)), 0f, 1f);
        let _e1670 = clamp(((dot(normalize(_e1543), _e1651) - _e1538) / max((_e1528 - _e1538), 0.0001f)), 0f, 1f);
        let _e1681 = local;
        local = (_e1681 + (((((((vec3<f32>(1f, 1f, 1f) - _e1603) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e1603 * ((_e1572 * 0.31830987f) / ((_e1576 * _e1576) + 0.00001f))) * (0.5f / (((_e1568 * sqrt((((_e1566 * _e1566) * _e1584) + _e1583))) + (_e1566 * sqrt((((_e1568 * _e1568) * _e1584) + _e1583)))) + 0.00001f)))) * _e1568) * (((((_e1548 * _e1518) * max(dot(_e72, normalize(_e1543)), 0f)) * select(0f, 1f, (_e1513 < 0.5f))) + ((((_e1548 * _e1518) * max(dot(_e72, (_e1627 / vec3(max(_e1628, 0.0001f)))), 0f)) * ((_e1640 * _e1640) / max(_e1634, 0.0001f))) * select(0f, 1f, ((_e1513 > 0.5f) && (_e1513 < 1.5f))))) + (((((_e1548 * _e1518) * max(dot(_e72, _e1651), 0f)) * ((_e1662 * _e1662) / max(_e1656, 0.0001f))) * (_e1670 * _e1670)) * select(0f, 1f, (_e1513 > 1.5f))))));
    }
    if (9f < _e99) {
        let _e1689 = lights.member[i32(9f)].member;
        let _e1694 = lights.member[i32(9f)].member_1;
        let _e1699 = lights.member[i32(9f)].member_2;
        let _e1704 = lights.member[i32(9f)].member_3;
        let _e1709 = lights.member[i32(9f)].member_4;
        let _e1714 = lights.member[i32(9f)].member_5;
        let _e1719 = lights.member[i32(9f)].member_6;
        let _e1724 = lights.member[i32(9f)].member_8;
        let _e1725 = world_pos_1;
        let _e1730 = select(0f, 1f, (_e1689 < 0.5f));
        let _e1734 = ((normalize(_e1719) * _e1730) + (normalize((_e1709 - _e1725)) * (1f - _e1730)));
        let _e1736 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e1738 = normalize((_e77 + _e1734));
        let _e1740 = max(dot(_e72, _e1738), 0f);
        let _e1742 = max(dot(_e72, _e77), 0.00001f);
        let _e1744 = max(dot(_e72, _e1734), 0.00001f);
        let _e1747 = (_e91 * _e91);
        let _e1748 = (_e1747 * _e1747);
        let _e1752 = (((_e1740 * _e1740) * (_e1748 - 1f)) + 1f);
        let _e1759 = (((_e91 * _e91) * _e91) * _e91);
        let _e1760 = (1f - _e1759);
        let _e1779 = (_e1736 + ((vec3<f32>(1f, 1f, 1f) - _e1736) * pow(clamp((1f - max(dot(_e77, _e1738), 0f)), 0f, 1f), 5f)));
        let _e1789 = world_pos_1;
        let _e1803 = (_e1709 - _e1789);
        let _e1804 = length(_e1803);
        let _e1810 = (_e1804 * _e1804);
        let _e1813 = (_e1810 / max((_e1699 * _e1699), 0.0001f));
        let _e1816 = clamp((1f - (_e1813 * _e1813)), 0f, 1f);
        let _e1823 = (_e1709 - _e1789);
        let _e1824 = length(_e1823);
        let _e1827 = (_e1823 / vec3(max(_e1824, 0.0001f)));
        let _e1832 = (_e1824 * _e1824);
        let _e1835 = (_e1832 / max((_e1699 * _e1699), 0.0001f));
        let _e1838 = clamp((1f - (_e1835 * _e1835)), 0f, 1f);
        let _e1846 = clamp(((dot(normalize(_e1719), _e1827) - _e1714) / max((_e1704 - _e1714), 0.0001f)), 0f, 1f);
        let _e1857 = local;
        local = (_e1857 + (((((((vec3<f32>(1f, 1f, 1f) - _e1779) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e1779 * ((_e1748 * 0.31830987f) / ((_e1752 * _e1752) + 0.00001f))) * (0.5f / (((_e1744 * sqrt((((_e1742 * _e1742) * _e1760) + _e1759))) + (_e1742 * sqrt((((_e1744 * _e1744) * _e1760) + _e1759)))) + 0.00001f)))) * _e1744) * (((((_e1724 * _e1694) * max(dot(_e72, normalize(_e1719)), 0f)) * select(0f, 1f, (_e1689 < 0.5f))) + ((((_e1724 * _e1694) * max(dot(_e72, (_e1803 / vec3(max(_e1804, 0.0001f)))), 0f)) * ((_e1816 * _e1816) / max(_e1810, 0.0001f))) * select(0f, 1f, ((_e1689 > 0.5f) && (_e1689 < 1.5f))))) + (((((_e1724 * _e1694) * max(dot(_e72, _e1827), 0f)) * ((_e1838 * _e1838) / max(_e1832, 0.0001f))) * (_e1846 * _e1846)) * select(0f, 1f, (_e1689 > 1.5f))))));
    }
    if (10f < _e99) {
        let _e1865 = lights.member[i32(10f)].member;
        let _e1870 = lights.member[i32(10f)].member_1;
        let _e1875 = lights.member[i32(10f)].member_2;
        let _e1880 = lights.member[i32(10f)].member_3;
        let _e1885 = lights.member[i32(10f)].member_4;
        let _e1890 = lights.member[i32(10f)].member_5;
        let _e1895 = lights.member[i32(10f)].member_6;
        let _e1900 = lights.member[i32(10f)].member_8;
        let _e1901 = world_pos_1;
        let _e1906 = select(0f, 1f, (_e1865 < 0.5f));
        let _e1910 = ((normalize(_e1895) * _e1906) + (normalize((_e1885 - _e1901)) * (1f - _e1906)));
        let _e1912 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e1914 = normalize((_e77 + _e1910));
        let _e1916 = max(dot(_e72, _e1914), 0f);
        let _e1918 = max(dot(_e72, _e77), 0.00001f);
        let _e1920 = max(dot(_e72, _e1910), 0.00001f);
        let _e1923 = (_e91 * _e91);
        let _e1924 = (_e1923 * _e1923);
        let _e1928 = (((_e1916 * _e1916) * (_e1924 - 1f)) + 1f);
        let _e1935 = (((_e91 * _e91) * _e91) * _e91);
        let _e1936 = (1f - _e1935);
        let _e1955 = (_e1912 + ((vec3<f32>(1f, 1f, 1f) - _e1912) * pow(clamp((1f - max(dot(_e77, _e1914), 0f)), 0f, 1f), 5f)));
        let _e1965 = world_pos_1;
        let _e1979 = (_e1885 - _e1965);
        let _e1980 = length(_e1979);
        let _e1986 = (_e1980 * _e1980);
        let _e1989 = (_e1986 / max((_e1875 * _e1875), 0.0001f));
        let _e1992 = clamp((1f - (_e1989 * _e1989)), 0f, 1f);
        let _e1999 = (_e1885 - _e1965);
        let _e2000 = length(_e1999);
        let _e2003 = (_e1999 / vec3(max(_e2000, 0.0001f)));
        let _e2008 = (_e2000 * _e2000);
        let _e2011 = (_e2008 / max((_e1875 * _e1875), 0.0001f));
        let _e2014 = clamp((1f - (_e2011 * _e2011)), 0f, 1f);
        let _e2022 = clamp(((dot(normalize(_e1895), _e2003) - _e1890) / max((_e1880 - _e1890), 0.0001f)), 0f, 1f);
        let _e2033 = local;
        local = (_e2033 + (((((((vec3<f32>(1f, 1f, 1f) - _e1955) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e1955 * ((_e1924 * 0.31830987f) / ((_e1928 * _e1928) + 0.00001f))) * (0.5f / (((_e1920 * sqrt((((_e1918 * _e1918) * _e1936) + _e1935))) + (_e1918 * sqrt((((_e1920 * _e1920) * _e1936) + _e1935)))) + 0.00001f)))) * _e1920) * (((((_e1900 * _e1870) * max(dot(_e72, normalize(_e1895)), 0f)) * select(0f, 1f, (_e1865 < 0.5f))) + ((((_e1900 * _e1870) * max(dot(_e72, (_e1979 / vec3(max(_e1980, 0.0001f)))), 0f)) * ((_e1992 * _e1992) / max(_e1986, 0.0001f))) * select(0f, 1f, ((_e1865 > 0.5f) && (_e1865 < 1.5f))))) + (((((_e1900 * _e1870) * max(dot(_e72, _e2003), 0f)) * ((_e2014 * _e2014) / max(_e2008, 0.0001f))) * (_e2022 * _e2022)) * select(0f, 1f, (_e1865 > 1.5f))))));
    }
    if (11f < _e99) {
        let _e2041 = lights.member[i32(11f)].member;
        let _e2046 = lights.member[i32(11f)].member_1;
        let _e2051 = lights.member[i32(11f)].member_2;
        let _e2056 = lights.member[i32(11f)].member_3;
        let _e2061 = lights.member[i32(11f)].member_4;
        let _e2066 = lights.member[i32(11f)].member_5;
        let _e2071 = lights.member[i32(11f)].member_6;
        let _e2076 = lights.member[i32(11f)].member_8;
        let _e2077 = world_pos_1;
        let _e2082 = select(0f, 1f, (_e2041 < 0.5f));
        let _e2086 = ((normalize(_e2071) * _e2082) + (normalize((_e2061 - _e2077)) * (1f - _e2082)));
        let _e2088 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e2090 = normalize((_e77 + _e2086));
        let _e2092 = max(dot(_e72, _e2090), 0f);
        let _e2094 = max(dot(_e72, _e77), 0.00001f);
        let _e2096 = max(dot(_e72, _e2086), 0.00001f);
        let _e2099 = (_e91 * _e91);
        let _e2100 = (_e2099 * _e2099);
        let _e2104 = (((_e2092 * _e2092) * (_e2100 - 1f)) + 1f);
        let _e2111 = (((_e91 * _e91) * _e91) * _e91);
        let _e2112 = (1f - _e2111);
        let _e2131 = (_e2088 + ((vec3<f32>(1f, 1f, 1f) - _e2088) * pow(clamp((1f - max(dot(_e77, _e2090), 0f)), 0f, 1f), 5f)));
        let _e2141 = world_pos_1;
        let _e2155 = (_e2061 - _e2141);
        let _e2156 = length(_e2155);
        let _e2162 = (_e2156 * _e2156);
        let _e2165 = (_e2162 / max((_e2051 * _e2051), 0.0001f));
        let _e2168 = clamp((1f - (_e2165 * _e2165)), 0f, 1f);
        let _e2175 = (_e2061 - _e2141);
        let _e2176 = length(_e2175);
        let _e2179 = (_e2175 / vec3(max(_e2176, 0.0001f)));
        let _e2184 = (_e2176 * _e2176);
        let _e2187 = (_e2184 / max((_e2051 * _e2051), 0.0001f));
        let _e2190 = clamp((1f - (_e2187 * _e2187)), 0f, 1f);
        let _e2198 = clamp(((dot(normalize(_e2071), _e2179) - _e2066) / max((_e2056 - _e2066), 0.0001f)), 0f, 1f);
        let _e2209 = local;
        local = (_e2209 + (((((((vec3<f32>(1f, 1f, 1f) - _e2131) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e2131 * ((_e2100 * 0.31830987f) / ((_e2104 * _e2104) + 0.00001f))) * (0.5f / (((_e2096 * sqrt((((_e2094 * _e2094) * _e2112) + _e2111))) + (_e2094 * sqrt((((_e2096 * _e2096) * _e2112) + _e2111)))) + 0.00001f)))) * _e2096) * (((((_e2076 * _e2046) * max(dot(_e72, normalize(_e2071)), 0f)) * select(0f, 1f, (_e2041 < 0.5f))) + ((((_e2076 * _e2046) * max(dot(_e72, (_e2155 / vec3(max(_e2156, 0.0001f)))), 0f)) * ((_e2168 * _e2168) / max(_e2162, 0.0001f))) * select(0f, 1f, ((_e2041 > 0.5f) && (_e2041 < 1.5f))))) + (((((_e2076 * _e2046) * max(dot(_e72, _e2179), 0f)) * ((_e2190 * _e2190) / max(_e2184, 0.0001f))) * (_e2198 * _e2198)) * select(0f, 1f, (_e2041 > 1.5f))))));
    }
    if (12f < _e99) {
        let _e2217 = lights.member[i32(12f)].member;
        let _e2222 = lights.member[i32(12f)].member_1;
        let _e2227 = lights.member[i32(12f)].member_2;
        let _e2232 = lights.member[i32(12f)].member_3;
        let _e2237 = lights.member[i32(12f)].member_4;
        let _e2242 = lights.member[i32(12f)].member_5;
        let _e2247 = lights.member[i32(12f)].member_6;
        let _e2252 = lights.member[i32(12f)].member_8;
        let _e2253 = world_pos_1;
        let _e2258 = select(0f, 1f, (_e2217 < 0.5f));
        let _e2262 = ((normalize(_e2247) * _e2258) + (normalize((_e2237 - _e2253)) * (1f - _e2258)));
        let _e2264 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e2266 = normalize((_e77 + _e2262));
        let _e2268 = max(dot(_e72, _e2266), 0f);
        let _e2270 = max(dot(_e72, _e77), 0.00001f);
        let _e2272 = max(dot(_e72, _e2262), 0.00001f);
        let _e2275 = (_e91 * _e91);
        let _e2276 = (_e2275 * _e2275);
        let _e2280 = (((_e2268 * _e2268) * (_e2276 - 1f)) + 1f);
        let _e2287 = (((_e91 * _e91) * _e91) * _e91);
        let _e2288 = (1f - _e2287);
        let _e2307 = (_e2264 + ((vec3<f32>(1f, 1f, 1f) - _e2264) * pow(clamp((1f - max(dot(_e77, _e2266), 0f)), 0f, 1f), 5f)));
        let _e2317 = world_pos_1;
        let _e2331 = (_e2237 - _e2317);
        let _e2332 = length(_e2331);
        let _e2338 = (_e2332 * _e2332);
        let _e2341 = (_e2338 / max((_e2227 * _e2227), 0.0001f));
        let _e2344 = clamp((1f - (_e2341 * _e2341)), 0f, 1f);
        let _e2351 = (_e2237 - _e2317);
        let _e2352 = length(_e2351);
        let _e2355 = (_e2351 / vec3(max(_e2352, 0.0001f)));
        let _e2360 = (_e2352 * _e2352);
        let _e2363 = (_e2360 / max((_e2227 * _e2227), 0.0001f));
        let _e2366 = clamp((1f - (_e2363 * _e2363)), 0f, 1f);
        let _e2374 = clamp(((dot(normalize(_e2247), _e2355) - _e2242) / max((_e2232 - _e2242), 0.0001f)), 0f, 1f);
        let _e2385 = local;
        local = (_e2385 + (((((((vec3<f32>(1f, 1f, 1f) - _e2307) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e2307 * ((_e2276 * 0.31830987f) / ((_e2280 * _e2280) + 0.00001f))) * (0.5f / (((_e2272 * sqrt((((_e2270 * _e2270) * _e2288) + _e2287))) + (_e2270 * sqrt((((_e2272 * _e2272) * _e2288) + _e2287)))) + 0.00001f)))) * _e2272) * (((((_e2252 * _e2222) * max(dot(_e72, normalize(_e2247)), 0f)) * select(0f, 1f, (_e2217 < 0.5f))) + ((((_e2252 * _e2222) * max(dot(_e72, (_e2331 / vec3(max(_e2332, 0.0001f)))), 0f)) * ((_e2344 * _e2344) / max(_e2338, 0.0001f))) * select(0f, 1f, ((_e2217 > 0.5f) && (_e2217 < 1.5f))))) + (((((_e2252 * _e2222) * max(dot(_e72, _e2355), 0f)) * ((_e2366 * _e2366) / max(_e2360, 0.0001f))) * (_e2374 * _e2374)) * select(0f, 1f, (_e2217 > 1.5f))))));
    }
    if (13f < _e99) {
        let _e2393 = lights.member[i32(13f)].member;
        let _e2398 = lights.member[i32(13f)].member_1;
        let _e2403 = lights.member[i32(13f)].member_2;
        let _e2408 = lights.member[i32(13f)].member_3;
        let _e2413 = lights.member[i32(13f)].member_4;
        let _e2418 = lights.member[i32(13f)].member_5;
        let _e2423 = lights.member[i32(13f)].member_6;
        let _e2428 = lights.member[i32(13f)].member_8;
        let _e2429 = world_pos_1;
        let _e2434 = select(0f, 1f, (_e2393 < 0.5f));
        let _e2438 = ((normalize(_e2423) * _e2434) + (normalize((_e2413 - _e2429)) * (1f - _e2434)));
        let _e2440 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e2442 = normalize((_e77 + _e2438));
        let _e2444 = max(dot(_e72, _e2442), 0f);
        let _e2446 = max(dot(_e72, _e77), 0.00001f);
        let _e2448 = max(dot(_e72, _e2438), 0.00001f);
        let _e2451 = (_e91 * _e91);
        let _e2452 = (_e2451 * _e2451);
        let _e2456 = (((_e2444 * _e2444) * (_e2452 - 1f)) + 1f);
        let _e2463 = (((_e91 * _e91) * _e91) * _e91);
        let _e2464 = (1f - _e2463);
        let _e2483 = (_e2440 + ((vec3<f32>(1f, 1f, 1f) - _e2440) * pow(clamp((1f - max(dot(_e77, _e2442), 0f)), 0f, 1f), 5f)));
        let _e2493 = world_pos_1;
        let _e2507 = (_e2413 - _e2493);
        let _e2508 = length(_e2507);
        let _e2514 = (_e2508 * _e2508);
        let _e2517 = (_e2514 / max((_e2403 * _e2403), 0.0001f));
        let _e2520 = clamp((1f - (_e2517 * _e2517)), 0f, 1f);
        let _e2527 = (_e2413 - _e2493);
        let _e2528 = length(_e2527);
        let _e2531 = (_e2527 / vec3(max(_e2528, 0.0001f)));
        let _e2536 = (_e2528 * _e2528);
        let _e2539 = (_e2536 / max((_e2403 * _e2403), 0.0001f));
        let _e2542 = clamp((1f - (_e2539 * _e2539)), 0f, 1f);
        let _e2550 = clamp(((dot(normalize(_e2423), _e2531) - _e2418) / max((_e2408 - _e2418), 0.0001f)), 0f, 1f);
        let _e2561 = local;
        local = (_e2561 + (((((((vec3<f32>(1f, 1f, 1f) - _e2483) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e2483 * ((_e2452 * 0.31830987f) / ((_e2456 * _e2456) + 0.00001f))) * (0.5f / (((_e2448 * sqrt((((_e2446 * _e2446) * _e2464) + _e2463))) + (_e2446 * sqrt((((_e2448 * _e2448) * _e2464) + _e2463)))) + 0.00001f)))) * _e2448) * (((((_e2428 * _e2398) * max(dot(_e72, normalize(_e2423)), 0f)) * select(0f, 1f, (_e2393 < 0.5f))) + ((((_e2428 * _e2398) * max(dot(_e72, (_e2507 / vec3(max(_e2508, 0.0001f)))), 0f)) * ((_e2520 * _e2520) / max(_e2514, 0.0001f))) * select(0f, 1f, ((_e2393 > 0.5f) && (_e2393 < 1.5f))))) + (((((_e2428 * _e2398) * max(dot(_e72, _e2531), 0f)) * ((_e2542 * _e2542) / max(_e2536, 0.0001f))) * (_e2550 * _e2550)) * select(0f, 1f, (_e2393 > 1.5f))))));
    }
    if (14f < _e99) {
        let _e2569 = lights.member[i32(14f)].member;
        let _e2574 = lights.member[i32(14f)].member_1;
        let _e2579 = lights.member[i32(14f)].member_2;
        let _e2584 = lights.member[i32(14f)].member_3;
        let _e2589 = lights.member[i32(14f)].member_4;
        let _e2594 = lights.member[i32(14f)].member_5;
        let _e2599 = lights.member[i32(14f)].member_6;
        let _e2604 = lights.member[i32(14f)].member_8;
        let _e2605 = world_pos_1;
        let _e2610 = select(0f, 1f, (_e2569 < 0.5f));
        let _e2614 = ((normalize(_e2599) * _e2610) + (normalize((_e2589 - _e2605)) * (1f - _e2610)));
        let _e2616 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e2618 = normalize((_e77 + _e2614));
        let _e2620 = max(dot(_e72, _e2618), 0f);
        let _e2622 = max(dot(_e72, _e77), 0.00001f);
        let _e2624 = max(dot(_e72, _e2614), 0.00001f);
        let _e2627 = (_e91 * _e91);
        let _e2628 = (_e2627 * _e2627);
        let _e2632 = (((_e2620 * _e2620) * (_e2628 - 1f)) + 1f);
        let _e2639 = (((_e91 * _e91) * _e91) * _e91);
        let _e2640 = (1f - _e2639);
        let _e2659 = (_e2616 + ((vec3<f32>(1f, 1f, 1f) - _e2616) * pow(clamp((1f - max(dot(_e77, _e2618), 0f)), 0f, 1f), 5f)));
        let _e2669 = world_pos_1;
        let _e2683 = (_e2589 - _e2669);
        let _e2684 = length(_e2683);
        let _e2690 = (_e2684 * _e2684);
        let _e2693 = (_e2690 / max((_e2579 * _e2579), 0.0001f));
        let _e2696 = clamp((1f - (_e2693 * _e2693)), 0f, 1f);
        let _e2703 = (_e2589 - _e2669);
        let _e2704 = length(_e2703);
        let _e2707 = (_e2703 / vec3(max(_e2704, 0.0001f)));
        let _e2712 = (_e2704 * _e2704);
        let _e2715 = (_e2712 / max((_e2579 * _e2579), 0.0001f));
        let _e2718 = clamp((1f - (_e2715 * _e2715)), 0f, 1f);
        let _e2726 = clamp(((dot(normalize(_e2599), _e2707) - _e2594) / max((_e2584 - _e2594), 0.0001f)), 0f, 1f);
        let _e2737 = local;
        local = (_e2737 + (((((((vec3<f32>(1f, 1f, 1f) - _e2659) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e2659 * ((_e2628 * 0.31830987f) / ((_e2632 * _e2632) + 0.00001f))) * (0.5f / (((_e2624 * sqrt((((_e2622 * _e2622) * _e2640) + _e2639))) + (_e2622 * sqrt((((_e2624 * _e2624) * _e2640) + _e2639)))) + 0.00001f)))) * _e2624) * (((((_e2604 * _e2574) * max(dot(_e72, normalize(_e2599)), 0f)) * select(0f, 1f, (_e2569 < 0.5f))) + ((((_e2604 * _e2574) * max(dot(_e72, (_e2683 / vec3(max(_e2684, 0.0001f)))), 0f)) * ((_e2696 * _e2696) / max(_e2690, 0.0001f))) * select(0f, 1f, ((_e2569 > 0.5f) && (_e2569 < 1.5f))))) + (((((_e2604 * _e2574) * max(dot(_e72, _e2707), 0f)) * ((_e2718 * _e2718) / max(_e2712, 0.0001f))) * (_e2726 * _e2726)) * select(0f, 1f, (_e2569 > 1.5f))))));
    }
    if (15f < _e99) {
        let _e2745 = lights.member[i32(15f)].member;
        let _e2750 = lights.member[i32(15f)].member_1;
        let _e2755 = lights.member[i32(15f)].member_2;
        let _e2760 = lights.member[i32(15f)].member_3;
        let _e2765 = lights.member[i32(15f)].member_4;
        let _e2770 = lights.member[i32(15f)].member_5;
        let _e2775 = lights.member[i32(15f)].member_6;
        let _e2780 = lights.member[i32(15f)].member_8;
        let _e2781 = world_pos_1;
        let _e2786 = select(0f, 1f, (_e2745 < 0.5f));
        let _e2790 = ((normalize(_e2775) * _e2786) + (normalize((_e2765 - _e2781)) * (1f - _e2786)));
        let _e2792 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
        let _e2794 = normalize((_e77 + _e2790));
        let _e2796 = max(dot(_e72, _e2794), 0f);
        let _e2798 = max(dot(_e72, _e77), 0.00001f);
        let _e2800 = max(dot(_e72, _e2790), 0.00001f);
        let _e2803 = (_e91 * _e91);
        let _e2804 = (_e2803 * _e2803);
        let _e2808 = (((_e2796 * _e2796) * (_e2804 - 1f)) + 1f);
        let _e2815 = (((_e91 * _e91) * _e91) * _e91);
        let _e2816 = (1f - _e2815);
        let _e2835 = (_e2792 + ((vec3<f32>(1f, 1f, 1f) - _e2792) * pow(clamp((1f - max(dot(_e77, _e2794), 0f)), 0f, 1f), 5f)));
        let _e2845 = world_pos_1;
        let _e2859 = (_e2765 - _e2845);
        let _e2860 = length(_e2859);
        let _e2866 = (_e2860 * _e2860);
        let _e2869 = (_e2866 / max((_e2755 * _e2755), 0.0001f));
        let _e2872 = clamp((1f - (_e2869 * _e2869)), 0f, 1f);
        let _e2879 = (_e2765 - _e2845);
        let _e2880 = length(_e2879);
        let _e2883 = (_e2879 / vec3(max(_e2880, 0.0001f)));
        let _e2888 = (_e2880 * _e2880);
        let _e2891 = (_e2888 / max((_e2755 * _e2755), 0.0001f));
        let _e2894 = clamp((1f - (_e2891 * _e2891)), 0f, 1f);
        let _e2902 = clamp(((dot(normalize(_e2775), _e2883) - _e2770) / max((_e2760 - _e2770), 0.0001f)), 0f, 1f);
        let _e2913 = local;
        local = (_e2913 + (((((((vec3<f32>(1f, 1f, 1f) - _e2835) * (1f - _e96)) * _e86) * 0.31830987f) + ((_e2835 * ((_e2804 * 0.31830987f) / ((_e2808 * _e2808) + 0.00001f))) * (0.5f / (((_e2800 * sqrt((((_e2798 * _e2798) * _e2816) + _e2815))) + (_e2798 * sqrt((((_e2800 * _e2800) * _e2816) + _e2815)))) + 0.00001f)))) * _e2800) * (((((_e2780 * _e2750) * max(dot(_e72, normalize(_e2775)), 0f)) * select(0f, 1f, (_e2745 < 0.5f))) + ((((_e2780 * _e2750) * max(dot(_e72, (_e2859 / vec3(max(_e2860, 0.0001f)))), 0f)) * ((_e2872 * _e2872) / max(_e2866, 0.0001f))) * select(0f, 1f, ((_e2745 > 0.5f) && (_e2745 < 1.5f))))) + (((((_e2780 * _e2750) * max(dot(_e72, _e2883), 0f)) * ((_e2894 * _e2894) / max(_e2888, 0.0001f))) * (_e2902 * _e2902)) * select(0f, 1f, (_e2745 > 1.5f))))));
    }
    local_1 = vec3<f32>(0f, 0f, 0f);
    let _e2919 = textureSampleLevel(env_specular_u002e_texture, env_specular_u002e_sampler, reflect((_e77 * -1f), _e72), (_e91 * 8f));
    let _e2921 = textureSample(env_irradiance_u002e_texture, env_irradiance_u002e_sampler, _e72);
    let _e2924 = textureSample(brdf_lut_u002e_texture, brdf_lut_u002e_sampler, vec2<f32>(_e79, _e91));
    let _e2925 = _e2924.xy;
    let _e2927 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), _e86, vec3(_e96));
    let _e2944 = (((_e2927 + ((max(vec3((1f - _e91)), _e2927) - _e2927) * pow(clamp((1f - _e79), 0f, 1f), 5f))) * _e2925.x) + vec3(_e2925.y));
    let _e2947 = (_e2927 + ((vec3<f32>(1f, 1f, 1f) - _e2927) * 0.04761905f));
    let _e2948 = (1f - (_e2925.x + _e2925.y));
    let _e2954 = (_e2944 + (((_e2944 * _e2947) / (vec3<f32>(1f, 1f, 1f) - (_e2947 * _e2948))) * _e2948));
    local_1 = (((((vec3<f32>(1f, 1f, 1f) - _e2954) * (1f - _e96)) * _e86) * _e2921.xyz) + (_e2954 * _e2919.xyz));
    let _e2962 = local;
    let _e2963 = local_1;
    let _e2967 = max(dot(_e72, normalize((_e77 + vec3<f32>(0f, 1f, 0f)))), 0f);
    let _e2969 = max(dot(_e72, _e77), 0.00001f);
    let _e2971 = max(dot(_e72, vec3<f32>(0f, 1f, 0f)), 0.00001f);
    let _e2972 = (_e91 * _e91);
    let _e2973 = (_e2972 * _e2972);
    let _e2977 = (((_e2967 * _e2967) * (_e2973 - 1f)) + 1f);
    let _e2984 = (((_e91 * _e91) * _e91) * _e91);
    let _e2985 = (1f - _e2984);
    let _e3012 = max(0f, 0.045f);
    let _e3016 = max(dot(_e72, normalize((_e77 + vec3<f32>(0f, 1f, 0f)))), 0.001f);
    let _e3018 = max(dot(_e72, vec3<f32>(0f, 1f, 0f)), 0f);
    let _e3020 = max(dot(_e72, _e77), 0.001f);
    let _e3022 = (1f / (_e3012 * _e3012));
    let _e3052 = max(0f, 0.045f);
    let _e3056 = normalize((_e77 + vec3<f32>(0f, 1f, 0f)));
    let _e3058 = max(dot(_e72, _e3056), 0f);
    let _e3060 = max(dot(_e72, vec3<f32>(0f, 1f, 0f)), 0f);
    let _e3062 = max(dot(_e72, _e77), 0f);
    let _e3065 = (_e3052 * _e3052);
    let _e3066 = (_e3065 * _e3065);
    let _e3070 = (((_e3058 * _e3058) * (_e3066 - 1f)) + 1f);
    let _e3077 = (((_e3052 * _e3052) * _e3052) * _e3052);
    let _e3078 = (1f - _e3077);
    let _e3113 = (((((((mix(_e2962, (((_e86 * (((_e2973 * 0.31830987f) / ((_e2977 * _e2977) + 0.00001f)) * (0.5f / (((_e2971 * sqrt((((_e2969 * _e2969) * _e2985) + _e2984))) + (_e2969 * sqrt((((_e2971 * _e2971) * _e2985) + _e2984)))) + 0.00001f)))) * 0f) * exp(((log(vec3<f32>(1f, 1f, 1f)) / vec3(1000000f)) * 0f))), vec3(0f)) + _e2963) + vec3<f32>(0f, 0f, 0f)) * (1f - (max(vec3<f32>(0f, 0f, 0f).x, max(vec3<f32>(0f, 0f, 0f).y, vec3<f32>(0f, 0f, 0f).z)) * (0.09f + (0.42f * _e3012))))) + (((vec3<f32>(0f, 0f, 0f) * (((2f + _e3022) * pow(max((1f - (_e3016 * _e3016)), 0f), (_e3022 * 0.5f))) / 6.2831855f)) * (1f / ((4f * ((_e3018 + _e3020) - (_e3018 * _e3020))) + 0.00001f))) * max(_e3018, 0f))) * (1f - ((vec3<f32>(0.04f, 0.04f, 0.04f) + ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(0.04f, 0.04f, 0.04f)) * pow(clamp((1f - max(dot(_e72, _e77), 0.001f)), 0f, 1f), 5f))).x * 0f))) + vec3(((((0f * ((_e3066 * 0.31830987f) / ((_e3070 * _e3070) + 0.00001f))) * (0.5f / (((_e3060 * sqrt((((_e3062 * _e3062) * _e3078) + _e3077))) + (_e3062 * sqrt((((_e3060 * _e3060) * _e3078) + _e3077)))) + 0.00001f))) * (0.04f + ((1f - 0.04f) * pow((1f - max(dot(_e77, _e3056), 0f)), 5f)))) * _e3060))) + vec3<f32>(0f, 0f, 0f));
    let _e3126 = pow(clamp(((_e3113 * ((_e3113 * 2.51f) + vec3(0.03f))) / ((_e3113 * ((_e3113 * 2.43f) + vec3(0.59f))) + vec3(0.14f))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f)), vec3<f32>(0.45454547f, 0.45454547f, 0.45454547f));
    color = vec4<f32>(_e3126.x, _e3126.y, _e3126.z, 1f);
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
