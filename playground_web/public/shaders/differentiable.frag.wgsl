var<private> param_1: f32;
var<private> color: vec4<f32>;

fn main_1() {
    let _e4 = param_1;
    let _e8 = param_1;
    color = vec4<f32>(((_e4 * _e4) + sin(_e4)), ((_e8 + _e8) + cos(_e8)), 0f, 1f);
    return;
}

@fragment 
fn main(@location(0) param: f32) -> @location(0) vec4<f32> {
    param_1 = param;
    main_1();
    let _e3 = color;
    return _e3;
}
