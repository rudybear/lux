struct gl_PerVertex {
    @builtin(position) gl_Position: vec4<f32>,
    gl_PointSize: f32,
    gl_ClipDistance: array<f32, 1>,
    gl_CullDistance: array<f32, 1>,
}

struct VertexOutput {
    @location(0) member: vec2<f32>,
    @builtin(position) gl_Position: vec4<f32>,
}

var<private> frag_uv: vec2<f32>;
var<private> unnamed: gl_PerVertex = gl_PerVertex(vec4<f32>(0f, 0f, 0f, 1f), 1f, array<f32, 1>(), array<f32, 1>());
var<private> vertex_index_1: u32;
var<private> instance_index_1: u32;

fn main_1() {
    let _e11 = vertex_index_1;
    let _e12 = f32(_e11);
    let _e18 = (step(0.5f, (_e12 - (floor((_e12 / 2f)) * 2f))) * 2f);
    let _e20 = (step(1.5f, _e12) * 2f);
    frag_uv = vec2<f32>(_e18, _e20);
    unnamed.gl_Position = vec4<f32>(((_e18 * 2f) - 1f), ((_e20 * 2f) - 1f), 0f, 1f);
    return;
}

@vertex 
fn main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    vertex_index_1 = vertex_index;
    instance_index_1 = instance_index;
    main_1();
    let _e8 = unnamed.gl_Position.y;
    unnamed.gl_Position.y = -(_e8);
    let _e10 = frag_uv;
    let _e11 = unnamed.gl_Position;
    return VertexOutput(_e10, _e11);
}
