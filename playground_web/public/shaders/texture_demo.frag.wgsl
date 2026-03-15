var<private> uv_1: vec2<f32>;
var<private> color: vec4<f32>;

fn main_1() {
    let _e15 = uv_1;
    let _e18 = (_e15.x * vec2<f32>(4f, 4f).x);
    let _e25 = (_e15.y * vec2<f32>(4f, 4f).y);
    let _e31 = uv_1;
    let _e33 = (_e31.x * 1.5708f);
    let _e34 = cos(_e33);
    let _e35 = sin(_e33);
    let _e36 = (vec2<f32>((_e18 - (floor((_e18 / 1f)) * 1f)), (_e25 - (floor((_e25 / 1f)) * 1f))) - vec2<f32>(0.5f, 0.5f));
    let _e48 = (vec2<f32>(((_e36.x * _e34) - (_e36.y * _e35)), ((_e36.x * _e35) + (_e36.y * _e34))) + vec2<f32>(0.5f, 0.5f));
    let _e57 = (smoothstep(0.02f, 0.05f, abs((_e48.x - 0.5f))) * smoothstep(0.02f, 0.05f, abs((_e48.y - 0.5f))));
    let _e58 = uv_1;
    let _e62 = uv_1;
    let _e67 = normalize(vec3<f32>(sin((_e58.x * 6.28318f)), cos((_e62.y * 6.28318f)), 0.5f));
    let _e77 = vec3<f32>(pow(abs(_e67.x), 4f), pow(abs(_e67.y), 4f), pow(abs(_e67.z), 4f));
    let _e104 = (((_e77 / vec3((((_e77.x + _e77.y) + _e77.z) + 0.00001f))) * _e57) + vec3((((1f - _e57) * ((normalize(((vec3<f32>((0.5f + (_e48.x * 0.5f)), (0.5f + (_e48.y * 0.5f)), 1f) * 2f) - vec3<f32>(1f, 1f, 1f))).z * 0.5f) + 0.5f)) * 0.3f)));
    color = vec4<f32>(_e104.x, _e104.y, _e104.z, 1f);
    return;
}

@fragment 
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    uv_1 = uv;
    main_1();
    let _e3 = color;
    return _e3;
}
