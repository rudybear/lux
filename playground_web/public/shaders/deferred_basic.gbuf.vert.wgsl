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
    @location(2) member_2: vec3<f32>,
    @location(3) member_3: vec3<f32>,
    @location(4) member_4: vec2<f32>,
    @builtin(position) gl_Position: vec4<f32>,
}

var<private> position_1: vec3<f32>;
var<private> normal_1: vec3<f32>;
var<private> uv_1: vec2<f32>;
var<private> tangent_1: vec4<f32>;
var<private> world_pos: vec3<f32>;
var<private> world_normal: vec3<f32>;
var<private> world_tangent: vec3<f32>;
var<private> world_bitangent: vec3<f32>;
var<private> frag_uv: vec2<f32>;
var<private> unnamed: gl_PerVertex = gl_PerVertex(vec4<f32>(0f, 0f, 0f, 1f), 1f, array<f32, 1>(), array<f32, 1>());
var<private> vertex_index_1: u32;
var<private> instance_index_1: u32;
@group(0) @binding(0) 
var<uniform> global: MVP;

fn main_1() {
    let _e19 = position_1;
    let _e23 = vec4<f32>(_e19.x, _e19.y, _e19.z, 1f);
    let _e25 = global.model;
    world_pos = (_e25 * _e23).xyz;
    let _e29 = global.model;
    let _e30 = normal_1;
    world_normal = normalize((_e29 * vec4<f32>(_e30.x, _e30.y, _e30.z, 0f)).xyz);
    let _e39 = global.model;
    let _e40 = tangent_1;
    let _e41 = _e40.xyz;
    world_tangent = normalize((_e39 * vec4<f32>(_e41.x, _e41.y, _e41.z, 0f)).xyz);
    let _e49 = world_normal;
    let _e50 = world_tangent;
    let _e52 = tangent_1;
    world_bitangent = (cross(_e49, _e50) * _e52.w);
    let _e55 = uv_1;
    frag_uv = _e55;
    let _e57 = global.projection;
    let _e59 = global.view;
    let _e62 = global.model;
    unnamed.gl_Position = (((_e57 * _e59) * _e62) * _e23);
    return;
}

@vertex 
fn main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, @location(3) tangent: vec4<f32>, @builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    position_1 = position;
    normal_1 = normal;
    uv_1 = uv;
    tangent_1 = tangent;
    vertex_index_1 = vertex_index;
    instance_index_1 = instance_index;
    main_1();
    let _e20 = unnamed.gl_Position.y;
    unnamed.gl_Position.y = -(_e20);
    let _e22 = world_pos;
    let _e23 = world_normal;
    let _e24 = world_tangent;
    let _e25 = world_bitangent;
    let _e26 = frag_uv;
    let _e27 = unnamed.gl_Position;
    return VertexOutput(_e22, _e23, _e24, _e25, _e26, _e27);
}
