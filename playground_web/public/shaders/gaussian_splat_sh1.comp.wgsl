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
var<storage, read_write> projected_center: type_9;
@group(0) @binding(9) 
var<storage, read_write> projected_conic: type_9;
@group(0) @binding(10) 
var<storage, read_write> projected_color: type_9;
@group(0) @binding(11) 
var<storage, read_write> sort_keys: type_15;
@group(0) @binding(12) 
var<storage, read_write> sorted_indices: type_15;
@group(0) @binding(13) 
var<storage, read_write> visible_count: type_15;
var<private> local_invocation_id_1: vec3<u32>;
var<private> local_invocation_index_1: u32;
var<private> workgroup_id_1: vec3<u32>;
var<private> num_workgroups_1: vec3<u32>;
var<private> global_invocation_id_1: vec3<u32>;

fn main_1() {
    let _e46 = global_invocation_id_1;
    let _e49 = u32(f32(_e46.x));
    let _e51 = global.total_splats;
    if (_e49 >= _e51) {
        return;
    }
    let _e55 = splat_pos.member[_e49];
    let _e56 = _e55.xyz;
    let _e58 = global.view_matrix;
    let _e64 = (_e58 * vec4<f32>(_e56.x, _e56.y, _e56.z, 1f)).xyz;
    if (_e64.z > -0.1f) {
        projected_center.member[_e49] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e49] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e49] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e49] = 4294967295u;
        sorted_indices.member[_e49] = _e49;
        return;
    }
    let _e79 = splat_rot.member[_e49];
    let _e84 = (_e79.x * _e79.x);
    let _e85 = (_e79.y * _e79.y);
    let _e86 = (_e79.z * _e79.z);
    let _e87 = (_e79.w * _e79.x);
    let _e88 = (_e79.w * _e79.y);
    let _e89 = (_e79.w * _e79.z);
    let _e90 = (_e79.x * _e79.y);
    let _e91 = (_e79.x * _e79.z);
    let _e92 = (_e79.y * _e79.z);
    let _e116 = splat_scale.member[_e49];
    let _e117 = _e116.xyz;
    let _e124 = vec3<f32>(exp(_e117.x), exp(_e117.y), exp(_e117.z));
    let _e126 = global.view_matrix;
    let _e129 = (_e126 * vec4<f32>((1f - (2f * (_e85 + _e86))), (2f * (_e90 + _e89)), (2f * (_e91 - _e88)), 0f)).xyz;
    let _e131 = global.view_matrix;
    let _e134 = (_e131 * vec4<f32>((2f * (_e90 - _e89)), (1f - (2f * (_e84 + _e86))), (2f * (_e92 + _e87)), 0f)).xyz;
    let _e136 = global.view_matrix;
    let _e139 = (_e136 * vec4<f32>((2f * (_e91 + _e88)), (2f * (_e92 - _e87)), (1f - (2f * (_e84 + _e85))), 0f)).xyz;
    let _e152 = (_e124.x * _e124.x);
    let _e153 = (_e124.y * _e124.y);
    let _e154 = (_e124.z * _e124.z);
    let _e178 = ((((_e129.x * _e129.z) * _e152) + ((_e134.x * _e134.z) * _e153)) + ((_e139.x * _e139.z) * _e154));
    let _e194 = ((((_e129.y * _e129.z) * _e152) + ((_e134.y * _e134.z) * _e153)) + ((_e139.y * _e139.z) * _e154));
    let _e202 = ((((_e129.z * _e129.z) * _e152) + ((_e134.z * _e134.z) * _e153)) + ((_e139.z * _e139.z) * _e154));
    let _e206 = -(_e64.z);
    let _e207 = (_e206 * _e206);
    let _e209 = global.focal_x;
    let _e211 = global.focal_y;
    let _e212 = (_e209 / _e206);
    let _e214 = ((_e209 * _e64.x) / _e207);
    let _e216 = -((_e211 / _e206));
    let _e219 = -(((_e211 * _e64.y) / _e207));
    let _e228 = ((((_e212 * _e212) * ((((_e129.x * _e129.x) * _e152) + ((_e134.x * _e134.x) * _e153)) + ((_e139.x * _e139.x) * _e154))) + ((2f * (_e212 * _e214)) * _e178)) + ((_e214 * _e214) * _e202));
    let _e239 = ((((_e212 * _e216) * ((((_e129.x * _e129.y) * _e152) + ((_e134.x * _e134.y) * _e153)) + ((_e139.x * _e139.y) * _e154))) + ((_e212 * _e219) * _e178)) + (((_e214 * _e216) * _e194) + ((_e214 * _e219) * _e202)));
    let _e248 = ((((_e216 * _e216) * ((((_e129.y * _e129.y) * _e152) + ((_e134.y * _e134.y) * _e153)) + ((_e139.y * _e139.y) * _e154))) + ((2f * (_e216 * _e219)) * _e194)) + ((_e219 * _e219) * _e202));
    let _e249 = (_e239 * _e239);
    let _e252 = (_e228 + 0.3f);
    let _e253 = (_e248 + 0.3f);
    let _e255 = ((_e252 * _e253) - _e249);
    if (_e255 <= 0f) {
        projected_center.member[_e49] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e49] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e49] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e49] = 4294967295u;
        sorted_indices.member[_e49] = _e49;
        return;
    }
    let _e270 = (1f / _e255);
    let _e276 = (0.5f * (_e252 + _e253));
    let _e285 = min(ceil((3f * sqrt((_e276 + sqrt(max(((_e276 * _e276) - _e255), 0f)))))), 256f);
    let _e287 = global.screen_size;
    let _e290 = global.screen_size;
    let _e292 = min(_e287.x, _e290.y);
    if (_e285 > (0.25f * _e292)) {
        projected_center.member[_e49] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e49] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e49] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e49] = 4294967295u;
        sorted_indices.member[_e49] = _e49;
        return;
    }
    let _e306 = global.proj_matrix;
    let _e311 = (_e306 * vec4<f32>(_e64.x, _e64.y, _e64.z, 1f));
    let _e314 = (_e311.x / _e311.w);
    let _e317 = (_e311.y / _e311.w);
    let _e322 = (1f + (_e285 / _e292));
    let _e323 = -(_e322);
    if (((_e314 > _e322) || (_e314 < _e323)) || ((_e317 > _e322) || (_e317 < _e323))) {
        projected_center.member[_e49] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e49] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e49] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e49] = 4294967295u;
        sorted_indices.member[_e49] = _e49;
        return;
    }
    let _e343 = splat_opacity.member[_e49];
    let _e348 = ((1f / (1f + exp(-(_e343)))) * sqrt(max((((_e228 * _e248) - _e249) / _e255), 0f)));
    if (_e348 < 0.00392157f) {
        projected_center.member[_e49] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e49] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e49] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e49] = 4294967295u;
        sorted_indices.member[_e49] = _e49;
        return;
    }
    let _e361 = global.cam_pos;
    let _e363 = normalize((_e56 - _e361));
    let _e366 = splat_sh0_.member[_e49];
    let _e375 = splat_sh1_.member[_e49];
    let _e379 = splat_sh2_.member[_e49];
    let _e383 = splat_sh3_.member[_e49];
    let _e393 = (((_e366.xyz * 0.28209478f) + vec3<f32>(0.5f, 0.5f, 0.5f)) + ((((_e375.xyz * _e363.y) * -0.48860252f) + ((_e379.xyz * _e363.z) * 0.48860252f)) + ((_e383.xyz * _e363.x) * -0.48860252f)));
    projected_center.member[_e49] = vec4<f32>(_e314, _e317, (_e311.z / _e311.w), _e285);
    projected_conic.member[_e49] = vec4<f32>((_e253 * _e270), (-(_e239) * _e270), (_e252 * _e270), _e348);
    projected_color.member[_e49] = vec4<f32>(clamp(_e393.x, 0f, 1f), clamp(_e393.y, 0f, 1f), clamp(_e393.z, 0f, 1f), _e348);
    let _e409 = bitcast<u32>(_e64.z);
    sort_keys.member[_e49] = (_e409 ^ (((_e409 >> bitcast<u32>(31u)) * 4294967295u) | 2147483648u));
    sorted_indices.member[_e49] = _e49;
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
