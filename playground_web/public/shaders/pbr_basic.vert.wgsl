struct gl_PerVertex {
    @builtin(position) gl_Position: vec4<f32>,
    gl_PointSize: f32,
    gl_ClipDistance: array<f32, 1>,
    gl_CullDistance: array<f32, 1>,
}

struct MVP {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
}

struct VertexOutput {
    @location(0) member: vec3<f32>,
    @location(1) member_1: vec3<f32>,
    @location(2) member_2: vec2<f32>,
    @builtin(position) gl_Position: vec4<f32>,
}

var<private> position_1: vec3<f32>;
var<private> normal_1: vec3<f32>;
var<private> uv_1: vec2<f32>;
var<private> frag_normal: vec3<f32>;
var<private> frag_pos: vec3<f32>;
var<private> frag_uv: vec2<f32>;
var<private> unnamed: gl_PerVertex = gl_PerVertex(vec4<f32>(0f, 0f, 0f, 1f), 1f, array<f32, 1>(), array<f32, 1>());
var<private> vertex_index_1: u32;
var<private> instance_index_1: u32;
@group(0) @binding(0) 
var<uniform> global: MVP;

fn main_1() {
    let _e17 = global.model;
    let _e18 = position_1;
    let _e23 = (_e17 * vec4<f32>(_e18.x, _e18.y, _e18.z, 1f));
    frag_pos = _e23.xyz;
    let _e26 = global.model;
    let _e27 = normal_1;
    frag_normal = normalize((_e26 * vec4<f32>(_e27.x, _e27.y, _e27.z, 0f)).xyz);
    let _e35 = uv_1;
    frag_uv = _e35;
    let _e37 = global.projection;
    let _e39 = global.view;
    unnamed.gl_Position = ((_e37 * _e39) * _e23);
    return;
}

@vertex 
fn main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, @builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    position_1 = position;
    normal_1 = normal;
    uv_1 = uv;
    vertex_index_1 = vertex_index;
    instance_index_1 = instance_index;
    main_1();
    let _e16 = unnamed.gl_Position.y;
    unnamed.gl_Position.y = -(_e16);
    let _e18 = frag_normal;
    let _e19 = frag_pos;
    let _e20 = frag_uv;
    let _e21 = unnamed.gl_Position;
    return VertexOutput(_e18, _e19, _e20, _e21);
}
