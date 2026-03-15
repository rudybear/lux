struct Lighting {
    light_dir: vec3<f32>,
    view_pos: vec3<f32>,
}

var<private> frag_normal_1: vec3<f32>;
var<private> frag_pos_1: vec3<f32>;
var<private> frag_uv_1: vec2<f32>;
var<private> color: vec4<f32>;
@group(2) @binding(0) 
var<uniform> global: Lighting;
@group(1) @binding(0) 
var albedo_tex_u002e_sampler: sampler;
@group(1) @binding(1) 
var albedo_tex_u002e_texture: texture_2d<f32>;

fn main_1() {
    let _e16 = frag_uv_1;
    let _e17 = textureSample(albedo_tex_u002e_texture, albedo_tex_u002e_sampler, _e16);
    let _e18 = frag_normal_1;
    let _e19 = normalize(_e18);
    let _e21 = global.light_dir;
    let _e22 = normalize(_e21);
    let _e24 = global.view_pos;
    let _e25 = frag_pos_1;
    let _e27 = normalize((_e24 - _e25));
    let _e29 = normalize((_e22 + _e27));
    let _e46 = ((_e17.xyz * max(dot(_e19, _e22), 0f)) + (vec3(pow(max(dot(_e19, _e29), 0f), 64f)) * (vec3<f32>(0.04f, 0.04f, 0.04f) + ((vec3<f32>(1f, 1f, 1f) - vec3<f32>(0.04f, 0.04f, 0.04f)) * pow((1f - max(dot(_e29, _e27), 0f)), 5f)))));
    color = vec4<f32>(_e46.x, _e46.y, _e46.z, 1f);
    return;
}

@fragment 
fn main(@location(0) frag_normal: vec3<f32>, @location(1) frag_pos: vec3<f32>, @location(2) frag_uv: vec2<f32>) -> @location(0) vec4<f32> {
    frag_normal_1 = frag_normal;
    frag_pos_1 = frag_pos;
    frag_uv_1 = frag_uv;
    main_1();
    let _e7 = color;
    return _e7;
}
