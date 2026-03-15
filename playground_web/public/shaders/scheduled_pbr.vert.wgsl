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
    @builtin(position) gl_Position: vec4<f32>,
}

var<private> position_1: vec3<f32>;
var<private> normal_1: vec3<f32>;
var<private> world_pos: vec3<f32>;
var<private> world_normal: vec3<f32>;
var<private> unnamed: gl_PerVertex = gl_PerVertex(vec4<f32>(0f, 0f, 0f, 1f), 1f, array<f32, 1>(), array<f32, 1>());
var<private> vertex_index_1: u32;
var<private> instance_index_1: u32;
@group(0) @binding(0) 
var<uniform> global: MVP;

fn main_1() {
    let _e14 = position_1;
    let _e18 = vec4<f32>(_e14.x, _e14.y, _e14.z, 1f);
    let _e20 = global.model;
    world_pos = (_e20 * _e18).xyz;
    let _e24 = global.model;
    let _e25 = normal_1;
    world_normal = normalize((_e24 * vec4<f32>(_e25.x, _e25.y, _e25.z, 0f)).xyz);
    let _e34 = global.projection;
    let _e36 = global.view;
    let _e39 = global.model;
    unnamed.gl_Position = (((_e34 * _e36) * _e39) * _e18);
    return;
}

@vertex 
fn main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    position_1 = position;
    normal_1 = normal;
    vertex_index_1 = vertex_index;
    instance_index_1 = instance_index;
    main_1();
    let _e13 = unnamed.gl_Position.y;
    unnamed.gl_Position.y = -(_e13);
    let _e15 = world_pos;
    let _e16 = world_normal;
    let _e17 = unnamed.gl_Position;
    return VertexOutput(_e15, _e16, _e17);
}
