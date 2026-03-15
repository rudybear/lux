struct push {
    screen_size: vec2<f32>,
    visible_count: u32,
    alpha_cutoff: f32,
}

var<private> frag_conic_1: vec3<f32>;
var<private> frag_color_1: vec4<f32>;
var<private> frag_center_1: vec2<f32>;
var<private> frag_offset_1: vec2<f32>;
var<private> out_color: vec4<f32>;
@group(2) @binding(0) 
var<uniform> global: push;

fn main_1() {
    let _e10 = frag_conic_1;
    let _e11 = frag_offset_1;
    let _e26 = (-0.5f * (((_e10.x * (_e11.x * _e11.x)) + ((2f * _e10.y) * (_e11.x * _e11.y))) + (_e10.z * (_e11.y * _e11.y))));
    if (_e26 < -4f) {
        discard;
    }
    let _e29 = frag_color_1;
    let _e31 = (exp(_e26) * _e29.w);
    let _e33 = global.alpha_cutoff;
    if (_e31 < _e33) {
        discard;
    }
    let _e35 = frag_color_1;
    let _e36 = _e35.xyz;
    out_color = vec4<f32>((_e36.x * _e31), (_e36.y * _e31), (_e36.z * _e31), _e31);
    return;
}

@fragment 
fn main(@location(0) frag_conic: vec3<f32>, @location(1) frag_color: vec4<f32>, @location(2) frag_center: vec2<f32>, @location(3) frag_offset: vec2<f32>) -> @location(0) vec4<f32> {
    frag_conic_1 = frag_conic;
    frag_color_1 = frag_color;
    frag_center_1 = frag_center;
    frag_offset_1 = frag_offset;
    main_1();
    let _e9 = out_color;
    return _e9;
}
