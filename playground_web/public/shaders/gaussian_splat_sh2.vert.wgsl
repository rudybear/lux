struct gl_PerVertex {
    @builtin(position) gl_Position: vec4<f32>,
    gl_PointSize: f32,
    gl_ClipDistance: array<f32, 1>,
    gl_CullDistance: array<f32, 1>,
}

struct push {
    screen_size: vec2<f32>,
    visible_count: u32,
    alpha_cutoff: f32,
}

struct type_13 {
    member: array<vec4<f32>>,
}

struct type_16 {
    member: array<u32>,
}

struct VertexOutput {
    @location(0) member: vec3<f32>,
    @location(1) member_1: vec4<f32>,
    @location(2) member_2: vec2<f32>,
    @location(3) member_3: vec2<f32>,
    @builtin(position) gl_Position: vec4<f32>,
}

var<private> frag_conic: vec3<f32>;
var<private> frag_color: vec4<f32>;
var<private> frag_center: vec2<f32>;
var<private> frag_offset: vec2<f32>;
var<private> unnamed: gl_PerVertex = gl_PerVertex(vec4<f32>(0f, 0f, 0f, 1f), 1f, array<f32, 1>(), array<f32, 1>());
var<private> vertex_index_1: u32;
var<private> instance_index_1: u32;
@group(1) @binding(0) 
var<uniform> global: push;
@group(0) @binding(0) 
var<storage> projected_center: type_13;
@group(0) @binding(1) 
var<storage> projected_conic: type_13;
@group(0) @binding(2) 
var<storage> projected_color: type_13;
@group(0) @binding(3) 
var<storage> sorted_indices: type_16;

fn main_1() {
    let _e21 = instance_index_1;
    let _e22 = vertex_index_1;
    let _e25 = sorted_indices.member[_e21];
    let _e28 = projected_center.member[_e25];
    let _e31 = projected_conic.member[_e25];
    let _e34 = projected_color.member[_e25];
    let _e38 = f32(_e22);
    let _e40 = floor((_e38 / 3f));
    let _e42 = (_e38 - (_e40 * 3f));
    let _e43 = step(0.5f, _e42);
    let _e45 = (_e43 * step(_e42, 1.5f));
    let _e46 = step(0.5f, _e40);
    let _e47 = step(1.5f, _e42);
    let _e48 = (_e46 * _e47);
    let _e55 = (_e46 * (1f - _e43));
    let _e62 = (vec2<f32>(((((_e45 + _e48) - (_e45 * _e48)) * 2f) - 1f), ((((_e47 + _e55) - (_e47 * _e55)) * 2f) - 1f)) * _e28.w);
    let _e64 = global.screen_size;
    let _e67 = global.screen_size;
    let _e69 = vec2<f32>(_e64.x, _e67.y);
    let _e72 = (((_e28.xy * 0.5f) + vec2<f32>(0.5f, 0.5f)) * _e69);
    let _e76 = ((((_e72 + _e62) / _e69) * 2f) - vec2<f32>(1f, 1f));
    unnamed.gl_Position = vec4<f32>(_e76.x, _e76.y, _e28.z, 1f);
    frag_conic = _e31.xyz;
    frag_color = _e34;
    frag_center = _e72;
    frag_offset = _e62;
    return;
}

@vertex 
fn main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    vertex_index_1 = vertex_index;
    instance_index_1 = instance_index;
    main_1();
    let _e11 = unnamed.gl_Position.y;
    unnamed.gl_Position.y = -(_e11);
    let _e13 = frag_conic;
    let _e14 = frag_color;
    let _e15 = frag_center;
    let _e16 = frag_offset;
    let _e17 = unnamed.gl_Position;
    return VertexOutput(_e13, _e14, _e15, _e16, _e17);
}
