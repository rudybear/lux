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
var<storage, read_write> splat_sh1_: type_9;
@group(0) @binding(6) 
var<storage, read_write> splat_sh2_: type_9;
@group(0) @binding(7) 
var<storage, read_write> splat_sh3_: type_9;
@group(0) @binding(8) 
var<storage, read_write> splat_sh4_: type_9;
@group(0) @binding(9) 
var<storage, read_write> splat_sh5_: type_9;
@group(0) @binding(10) 
var<storage, read_write> splat_sh6_: type_9;
@group(0) @binding(11) 
var<storage, read_write> splat_sh7_: type_9;
@group(0) @binding(12) 
var<storage, read_write> splat_sh8_: type_9;
@group(0) @binding(13) 
var<storage, read_write> projected_center: type_9;
@group(0) @binding(14) 
var<storage, read_write> projected_conic: type_9;
@group(0) @binding(15) 
var<storage, read_write> projected_color: type_9;
@group(0) @binding(16) 
var<storage, read_write> sort_keys: type_15;
@group(0) @binding(17) 
var<storage, read_write> sorted_indices: type_15;
@group(0) @binding(18) 
var<storage, read_write> visible_count: type_15;
var<private> local_invocation_id_1: vec3<u32>;
var<private> local_invocation_index_1: u32;
var<private> workgroup_id_1: vec3<u32>;
var<private> num_workgroups_1: vec3<u32>;
var<private> global_invocation_id_1: vec3<u32>;

fn main_1() {
    let _e55 = global_invocation_id_1;
    let _e58 = u32(f32(_e55.x));
    let _e60 = global.total_splats;
    if (_e58 >= _e60) {
        return;
    }
    let _e64 = splat_pos.member[_e58];
    let _e65 = _e64.xyz;
    let _e67 = global.view_matrix;
    let _e73 = (_e67 * vec4<f32>(_e65.x, _e65.y, _e65.z, 1f)).xyz;
    if (_e73.z > -0.1f) {
        projected_center.member[_e58] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e58] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e58] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e58] = 4294967295u;
        sorted_indices.member[_e58] = _e58;
        return;
    }
    let _e88 = splat_rot.member[_e58];
    let _e93 = (_e88.x * _e88.x);
    let _e94 = (_e88.y * _e88.y);
    let _e95 = (_e88.z * _e88.z);
    let _e96 = (_e88.w * _e88.x);
    let _e97 = (_e88.w * _e88.y);
    let _e98 = (_e88.w * _e88.z);
    let _e99 = (_e88.x * _e88.y);
    let _e100 = (_e88.x * _e88.z);
    let _e101 = (_e88.y * _e88.z);
    let _e125 = splat_scale.member[_e58];
    let _e126 = _e125.xyz;
    let _e133 = vec3<f32>(exp(_e126.x), exp(_e126.y), exp(_e126.z));
    let _e135 = global.view_matrix;
    let _e138 = (_e135 * vec4<f32>((1f - (2f * (_e94 + _e95))), (2f * (_e99 + _e98)), (2f * (_e100 - _e97)), 0f)).xyz;
    let _e140 = global.view_matrix;
    let _e143 = (_e140 * vec4<f32>((2f * (_e99 - _e98)), (1f - (2f * (_e93 + _e95))), (2f * (_e101 + _e96)), 0f)).xyz;
    let _e145 = global.view_matrix;
    let _e148 = (_e145 * vec4<f32>((2f * (_e100 + _e97)), (2f * (_e101 - _e96)), (1f - (2f * (_e93 + _e94))), 0f)).xyz;
    let _e161 = (_e133.x * _e133.x);
    let _e162 = (_e133.y * _e133.y);
    let _e163 = (_e133.z * _e133.z);
    let _e187 = ((((_e138.x * _e138.z) * _e161) + ((_e143.x * _e143.z) * _e162)) + ((_e148.x * _e148.z) * _e163));
    let _e203 = ((((_e138.y * _e138.z) * _e161) + ((_e143.y * _e143.z) * _e162)) + ((_e148.y * _e148.z) * _e163));
    let _e211 = ((((_e138.z * _e138.z) * _e161) + ((_e143.z * _e143.z) * _e162)) + ((_e148.z * _e148.z) * _e163));
    let _e215 = -(_e73.z);
    let _e216 = (_e215 * _e215);
    let _e218 = global.focal_x;
    let _e220 = global.focal_y;
    let _e221 = (_e218 / _e215);
    let _e223 = ((_e218 * _e73.x) / _e216);
    let _e225 = -((_e220 / _e215));
    let _e228 = -(((_e220 * _e73.y) / _e216));
    let _e237 = ((((_e221 * _e221) * ((((_e138.x * _e138.x) * _e161) + ((_e143.x * _e143.x) * _e162)) + ((_e148.x * _e148.x) * _e163))) + ((2f * (_e221 * _e223)) * _e187)) + ((_e223 * _e223) * _e211));
    let _e248 = ((((_e221 * _e225) * ((((_e138.x * _e138.y) * _e161) + ((_e143.x * _e143.y) * _e162)) + ((_e148.x * _e148.y) * _e163))) + ((_e221 * _e228) * _e187)) + (((_e223 * _e225) * _e203) + ((_e223 * _e228) * _e211)));
    let _e257 = ((((_e225 * _e225) * ((((_e138.y * _e138.y) * _e161) + ((_e143.y * _e143.y) * _e162)) + ((_e148.y * _e148.y) * _e163))) + ((2f * (_e225 * _e228)) * _e203)) + ((_e228 * _e228) * _e211));
    let _e258 = (_e248 * _e248);
    let _e261 = (_e237 + 0.3f);
    let _e262 = (_e257 + 0.3f);
    let _e264 = ((_e261 * _e262) - _e258);
    if (_e264 <= 0f) {
        projected_center.member[_e58] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e58] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e58] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e58] = 4294967295u;
        sorted_indices.member[_e58] = _e58;
        return;
    }
    let _e279 = (1f / _e264);
    let _e285 = (0.5f * (_e261 + _e262));
    let _e294 = min(ceil((3f * sqrt((_e285 + sqrt(max(((_e285 * _e285) - _e264), 0f)))))), 256f);
    let _e296 = global.screen_size;
    let _e299 = global.screen_size;
    let _e301 = min(_e296.x, _e299.y);
    if (_e294 > (0.25f * _e301)) {
        projected_center.member[_e58] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e58] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e58] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e58] = 4294967295u;
        sorted_indices.member[_e58] = _e58;
        return;
    }
    let _e315 = global.proj_matrix;
    let _e320 = (_e315 * vec4<f32>(_e73.x, _e73.y, _e73.z, 1f));
    let _e323 = (_e320.x / _e320.w);
    let _e326 = (_e320.y / _e320.w);
    let _e331 = (1f + (_e294 / _e301));
    let _e332 = -(_e331);
    if (((_e323 > _e331) || (_e323 < _e332)) || ((_e326 > _e331) || (_e326 < _e332))) {
        projected_center.member[_e58] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e58] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e58] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e58] = 4294967295u;
        sorted_indices.member[_e58] = _e58;
        return;
    }
    let _e352 = splat_opacity.member[_e58];
    let _e357 = ((1f / (1f + exp(-(_e352)))) * sqrt(max((((_e237 * _e257) - _e258) / _e264), 0f)));
    if (_e357 < 0.00392157f) {
        projected_center.member[_e58] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e58] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e58] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e58] = 4294967295u;
        sorted_indices.member[_e58] = _e58;
        return;
    }
    let _e370 = global.cam_pos;
    let _e372 = normalize((_e65 - _e370));
    let _e375 = splat_sh0_.member[_e58];
    let _e384 = splat_sh1_.member[_e58];
    let _e388 = splat_sh2_.member[_e58];
    let _e392 = splat_sh3_.member[_e58];
    let _e405 = splat_sh4_.member[_e58];
    let _e409 = splat_sh5_.member[_e58];
    let _e413 = splat_sh6_.member[_e58];
    let _e417 = splat_sh7_.member[_e58];
    let _e421 = splat_sh8_.member[_e58];
    let _e423 = (_e372.x * _e372.x);
    let _e424 = (_e372.y * _e372.y);
    let _e447 = ((((_e375.xyz * 0.28209478f) + vec3<f32>(0.5f, 0.5f, 0.5f)) + ((((_e384.xyz * _e372.y) * -0.48860252f) + ((_e388.xyz * _e372.z) * 0.48860252f)) + ((_e392.xyz * _e372.x) * -0.48860252f))) + ((((_e405.xyz * (1.0925485f * (_e372.x * _e372.y))) + (_e409.xyz * (-1.0925485f * (_e372.y * _e372.z)))) + ((_e413.xyz * (0.31539157f * ((2f * (_e372.z * _e372.z)) - (_e423 + _e424)))) + (_e417.xyz * (-1.0925485f * (_e372.x * _e372.z))))) + (_e421.xyz * (0.54627424f * (_e423 - _e424)))));
    projected_center.member[_e58] = vec4<f32>(_e323, _e326, (_e320.z / _e320.w), _e294);
    projected_conic.member[_e58] = vec4<f32>((_e262 * _e279), (-(_e248) * _e279), (_e261 * _e279), _e357);
    projected_color.member[_e58] = vec4<f32>(clamp(_e447.x, 0f, 1f), clamp(_e447.y, 0f, 1f), clamp(_e447.z, 0f, 1f), _e357);
    let _e463 = bitcast<u32>(_e73.z);
    sort_keys.member[_e58] = (_e463 ^ (((_e463 >> bitcast<u32>(31u)) * 4294967295u) | 2147483648u));
    sorted_indices.member[_e58] = _e58;
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
