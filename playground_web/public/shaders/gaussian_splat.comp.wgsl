struct push {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    cam_pos: vec3<f32>,
    screen_size: vec2<f32>,
    total_splats: u32,
    focal_x: f32,
    focal_y: f32,
    sh_degree: i32,
}

struct type_9 {
    member: array<vec4<f32>>,
}

struct type_12 {
    member: array<f32>,
}

struct type_15 {
    member: array<u32>,
}

@group(1) @binding(0) 
var<uniform> global: push;
@group(0) @binding(0) 
var<storage, read_write> splat_pos: type_9;
@group(0) @binding(1) 
var<storage, read_write> splat_rot: type_9;
@group(0) @binding(2) 
var<storage, read_write> splat_scale: type_9;
@group(0) @binding(3) 
var<storage, read_write> splat_opacity: type_12;
@group(0) @binding(4) 
var<storage, read_write> splat_sh0_: type_9;
@group(0) @binding(5) 
var<storage, read_write> projected_center: type_9;
@group(0) @binding(6) 
var<storage, read_write> projected_conic: type_9;
@group(0) @binding(7) 
var<storage, read_write> projected_color: type_9;
@group(0) @binding(8) 
var<storage, read_write> sort_keys: type_15;
@group(0) @binding(9) 
var<storage, read_write> sorted_indices: type_15;
@group(0) @binding(10) 
var<storage, read_write> visible_count: type_15;
var<private> local_invocation_id_1: vec3<u32>;
var<private> local_invocation_index_1: u32;
var<private> workgroup_id_1: vec3<u32>;
var<private> num_workgroups_1: vec3<u32>;
var<private> global_invocation_id_1: vec3<u32>;

fn main_1() {
    let _e40 = global_invocation_id_1;
    let _e43 = u32(f32(_e40.x));
    let _e45 = global.total_splats;
    if (_e43 >= _e45) {
        return;
    }
    let _e49 = splat_pos.member[_e43];
    let _e50 = _e49.xyz;
    let _e52 = global.view_matrix;
    let _e58 = (_e52 * vec4<f32>(_e50.x, _e50.y, _e50.z, 1f)).xyz;
    if (_e58.z > -0.1f) {
        projected_center.member[_e43] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e43] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e43] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e43] = 4294967295u;
        sorted_indices.member[_e43] = _e43;
        return;
    }
    let _e73 = splat_rot.member[_e43];
    let _e78 = (_e73.x * _e73.x);
    let _e79 = (_e73.y * _e73.y);
    let _e80 = (_e73.z * _e73.z);
    let _e81 = (_e73.w * _e73.x);
    let _e82 = (_e73.w * _e73.y);
    let _e83 = (_e73.w * _e73.z);
    let _e84 = (_e73.x * _e73.y);
    let _e85 = (_e73.x * _e73.z);
    let _e86 = (_e73.y * _e73.z);
    let _e110 = splat_scale.member[_e43];
    let _e111 = _e110.xyz;
    let _e118 = vec3<f32>(exp(_e111.x), exp(_e111.y), exp(_e111.z));
    let _e120 = global.view_matrix;
    let _e123 = (_e120 * vec4<f32>((1f - (2f * (_e79 + _e80))), (2f * (_e84 + _e83)), (2f * (_e85 - _e82)), 0f)).xyz;
    let _e125 = global.view_matrix;
    let _e128 = (_e125 * vec4<f32>((2f * (_e84 - _e83)), (1f - (2f * (_e78 + _e80))), (2f * (_e86 + _e81)), 0f)).xyz;
    let _e130 = global.view_matrix;
    let _e133 = (_e130 * vec4<f32>((2f * (_e85 + _e82)), (2f * (_e86 - _e81)), (1f - (2f * (_e78 + _e79))), 0f)).xyz;
    let _e146 = (_e118.x * _e118.x);
    let _e147 = (_e118.y * _e118.y);
    let _e148 = (_e118.z * _e118.z);
    let _e172 = ((((_e123.x * _e123.z) * _e146) + ((_e128.x * _e128.z) * _e147)) + ((_e133.x * _e133.z) * _e148));
    let _e188 = ((((_e123.y * _e123.z) * _e146) + ((_e128.y * _e128.z) * _e147)) + ((_e133.y * _e133.z) * _e148));
    let _e196 = ((((_e123.z * _e123.z) * _e146) + ((_e128.z * _e128.z) * _e147)) + ((_e133.z * _e133.z) * _e148));
    let _e200 = -(_e58.z);
    let _e201 = (_e200 * _e200);
    let _e203 = global.focal_x;
    let _e205 = global.focal_y;
    let _e206 = (_e203 / _e200);
    let _e208 = ((_e203 * _e58.x) / _e201);
    let _e210 = -((_e205 / _e200));
    let _e213 = -(((_e205 * _e58.y) / _e201));
    let _e222 = ((((_e206 * _e206) * ((((_e123.x * _e123.x) * _e146) + ((_e128.x * _e128.x) * _e147)) + ((_e133.x * _e133.x) * _e148))) + ((2f * (_e206 * _e208)) * _e172)) + ((_e208 * _e208) * _e196));
    let _e233 = ((((_e206 * _e210) * ((((_e123.x * _e123.y) * _e146) + ((_e128.x * _e128.y) * _e147)) + ((_e133.x * _e133.y) * _e148))) + ((_e206 * _e213) * _e172)) + (((_e208 * _e210) * _e188) + ((_e208 * _e213) * _e196)));
    let _e242 = ((((_e210 * _e210) * ((((_e123.y * _e123.y) * _e146) + ((_e128.y * _e128.y) * _e147)) + ((_e133.y * _e133.y) * _e148))) + ((2f * (_e210 * _e213)) * _e188)) + ((_e213 * _e213) * _e196));
    let _e243 = (_e233 * _e233);
    let _e246 = (_e222 + 0.3f);
    let _e247 = (_e242 + 0.3f);
    let _e249 = ((_e246 * _e247) - _e243);
    if (_e249 <= 0f) {
        projected_center.member[_e43] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e43] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e43] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e43] = 4294967295u;
        sorted_indices.member[_e43] = _e43;
        return;
    }
    let _e264 = (1f / _e249);
    let _e270 = (0.5f * (_e246 + _e247));
    let _e279 = min(ceil((3f * sqrt((_e270 + sqrt(max(((_e270 * _e270) - _e249), 0f)))))), 256f);
    let _e281 = global.screen_size;
    let _e284 = global.screen_size;
    let _e286 = min(_e281.x, _e284.y);
    if (_e279 > (0.25f * _e286)) {
        projected_center.member[_e43] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e43] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e43] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e43] = 4294967295u;
        sorted_indices.member[_e43] = _e43;
        return;
    }
    let _e300 = global.proj_matrix;
    let _e305 = (_e300 * vec4<f32>(_e58.x, _e58.y, _e58.z, 1f));
    let _e308 = (_e305.x / _e305.w);
    let _e311 = (_e305.y / _e305.w);
    let _e316 = (1f + (_e279 / _e286));
    let _e317 = -(_e316);
    if (((_e308 > _e316) || (_e308 < _e317)) || ((_e311 > _e316) || (_e311 < _e317))) {
        projected_center.member[_e43] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e43] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e43] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e43] = 4294967295u;
        sorted_indices.member[_e43] = _e43;
        return;
    }
    let _e337 = splat_opacity.member[_e43];
    let _e342 = ((1f / (1f + exp(-(_e337)))) * sqrt(max((((_e222 * _e242) - _e243) / _e249), 0f)));
    if (_e342 < 0.00392157f) {
        projected_center.member[_e43] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e43] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e43] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e43] = 4294967295u;
        sorted_indices.member[_e43] = _e43;
        return;
    }
    let _e356 = splat_sh0_.member[_e43];
    let _e359 = ((_e356.xyz * 0.28209478f) + vec3<f32>(0.5f, 0.5f, 0.5f));
    projected_center.member[_e43] = vec4<f32>(_e308, _e311, (_e305.z / _e305.w), _e279);
    projected_conic.member[_e43] = vec4<f32>((_e247 * _e264), (-(_e233) * _e264), (_e246 * _e264), _e342);
    projected_color.member[_e43] = vec4<f32>(clamp(_e359.x, 0f, 1f), clamp(_e359.y, 0f, 1f), clamp(_e359.z, 0f, 1f), _e342);
    let _e375 = bitcast<u32>(_e58.z);
    sort_keys.member[_e43] = (_e375 ^ (((_e375 >> bitcast<u32>(31u)) * 4294967295u) | 2147483648u));
    sorted_indices.member[_e43] = _e43;
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn main(@builtin(local_invocation_id) local_invocation_id: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32, @builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>, @builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    local_invocation_id_1 = local_invocation_id;
    local_invocation_index_1 = local_invocation_index;
    workgroup_id_1 = workgroup_id;
    num_workgroups_1 = num_workgroups;
    global_invocation_id_1 = global_invocation_id;
    main_1();
}
