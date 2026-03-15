var<private> frag_color_1: vec3<f32>;
var<private> color: vec4<f32>;

fn main_1() {
    let _e3 = frag_color_1;
    color = vec4<f32>(_e3.x, _e3.y, _e3.z, 1f);
    return;
}

@fragment 
fn main(@location(0) frag_color: vec3<f32>) -> @location(0) vec4<f32> {
    frag_color_1 = frag_color;
    main_1();
    let _e3 = color;
    return _e3;
}
