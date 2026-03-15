struct gl_PerVertex {
    @builtin(position) gl_Position: vec4<f32>,
    gl_PointSize: f32,
    gl_ClipDistance: array<f32, 1>,
    gl_CullDistance: array<f32, 1>,
}

struct VertexOutput {
    @location(0) member: vec3<f32>,
    @builtin(position) gl_Position: vec4<f32>,
}

var<private> position_1: vec3<f32>;
var<private> color_1: vec3<f32>;
var<private> frag_color: vec3<f32>;
var<private> unnamed: gl_PerVertex = gl_PerVertex(vec4<f32>(0f, 0f, 0f, 1f), 1f, array<f32, 1>(), array<f32, 1>());
var<private> vertex_index_1: u32;
var<private> instance_index_1: u32;

fn main_1() {
    let _e9 = color_1;
    frag_color = _e9;
    let _e10 = position_1;
    unnamed.gl_Position = vec4<f32>(_e10.x, _e10.y, _e10.z, 1f);
    return;
}

@vertex 
fn main(@location(0) position: vec3<f32>, @location(1) color: vec3<f32>, @builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    position_1 = position;
    color_1 = color;
    vertex_index_1 = vertex_index;
    instance_index_1 = instance_index;
    main_1();
    let _e12 = unnamed.gl_Position.y;
    unnamed.gl_Position.y = -(_e12);
    let _e14 = frag_color;
    let _e15 = unnamed.gl_Position;
    return VertexOutput(_e14, _e15);
}
