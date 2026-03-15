var<private> uv_1: vec2<f32>;
var<private> color: vec4<f32>;

fn main_1() {
    let _e38 = uv_1;
    let _e40 = fract((_e38 * 2f));
    let _e41 = uv_1;
    let _e45 = uv_1;
    let _e50 = ((_e40 * 2f) - vec2<f32>(1f, 1f));
    let _e51 = length(_e50);
    let _e60 = normalize(vec3<f32>(_e50.x, _e50.y, sqrt(max((0.64f - (_e51 * _e51)), 0f))));
    let _e62 = max(dot(_e60, vec3<f32>(0f, 0f, 1f)), 0.001f);
    let _e64 = select(0.8f, 0.2f, (floor((_e41.x * 2f)) < 0.5f));
    let _e66 = select(1f, 0f, (floor((_e45.y * 2f)) < 0.5f));
    let _e68 = mix(vec3<f32>(0.04f, 0.04f, 0.04f), vec3<f32>(0.8f, 0.3f, 0.2f), vec3(_e66));
    let _e80 = vec2<f32>(clamp((1f - ((_e64 * max((1f - _e62), _e62)) * 0.31830987f)), 0f, 1f), clamp(((_e64 * max(0f, (0.75f - _e62))) * 0.31830987f), 0f, 1f));
    let _e110 = (((_e68 * _e80.x) + vec3(_e80.y)) + ((((vec3<f32>(1f, 1f, 1f) - (_e68 + ((vec3<f32>(1f, 1f, 1f) - _e68) * pow(clamp((1f - _e62), 0f, 1f), 5f)))) * (1f - _e66)) * vec3<f32>(0.8f, 0.3f, 0.2f)) * ((vec3<f32>(0.8f, 0.9f, 1f) * 0.282095f) + ((((vec3<f32>(0.1f, 0.1f, 0.2f) * 0.488603f) * _e60.y) + ((vec3<f32>(0.2f, 0.15f, 0.1f) * 0.488603f) * _e60.z)) + ((vec3<f32>(0.15f, 0.1f, 0.2f) * 0.488603f) * _e60.x)))));
    let _e138 = (((pow(clamp(((_e110 * ((_e110 * 2.51f) + vec3(0.03f))) / ((_e110 * ((_e110 * 2.43f) + vec3(0.59f))) + vec3(0.14f))), vec3<f32>(0f, 0f, 0f), vec3<f32>(1f, 1f, 1f)), vec3<f32>(0.45454547f, 0.45454547f, 0.45454547f)) * step(_e51, 0.8f)) * (smoothstep(0f, 0.02f, _e40.x) * smoothstep(0f, 0.02f, (1f - _e40.x)))) * (smoothstep(0f, 0.02f, _e40.y) * smoothstep(0f, 0.02f, (1f - _e40.y))));
    color = vec4<f32>(_e138.x, _e138.y, _e138.z, 1f);
    return;
}

@fragment 
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    uv_1 = uv;
    main_1();
    let _e3 = color;
    return _e3;
}
