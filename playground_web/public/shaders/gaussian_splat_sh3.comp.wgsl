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
var<storage, read_write> splat_sh9_: type_9;
@group(0) @binding(14) 
var<storage, read_write> splat_sh10_: type_9;
@group(0) @binding(15) 
var<storage, read_write> splat_sh11_: type_9;
@group(0) @binding(16) 
var<storage, read_write> splat_sh12_: type_9;
@group(0) @binding(17) 
var<storage, read_write> splat_sh13_: type_9;
@group(0) @binding(18) 
var<storage, read_write> splat_sh14_: type_9;
@group(0) @binding(19) 
var<storage, read_write> splat_sh15_: type_9;
@group(0) @binding(20) 
var<storage, read_write> projected_center: type_9;
@group(0) @binding(21) 
var<storage, read_write> projected_conic: type_9;
@group(0) @binding(22) 
var<storage, read_write> projected_color: type_9;
@group(0) @binding(23) 
var<storage, read_write> sort_keys: type_15;
@group(0) @binding(24) 
var<storage, read_write> sorted_indices: type_15;
@group(0) @binding(25) 
var<storage, read_write> visible_count: type_15;
var<private> local_invocation_id_1: vec3<u32>;
var<private> local_invocation_index_1: u32;
var<private> workgroup_id_1: vec3<u32>;
var<private> num_workgroups_1: vec3<u32>;
var<private> global_invocation_id_1: vec3<u32>;

fn main_1() {
    let _e68 = global_invocation_id_1;
    let _e71 = u32(f32(_e68.x));
    let _e73 = global.total_splats;
    if (_e71 >= _e73) {
        return;
    }
    let _e77 = splat_pos.member[_e71];
    let _e78 = _e77.xyz;
    let _e80 = global.view_matrix;
    let _e86 = (_e80 * vec4<f32>(_e78.x, _e78.y, _e78.z, 1f)).xyz;
    if (_e86.z > -0.1f) {
        projected_center.member[_e71] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e71] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e71] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e71] = 4294967295u;
        sorted_indices.member[_e71] = _e71;
        return;
    }
    let _e101 = splat_rot.member[_e71];
    let _e106 = (_e101.x * _e101.x);
    let _e107 = (_e101.y * _e101.y);
    let _e108 = (_e101.z * _e101.z);
    let _e109 = (_e101.w * _e101.x);
    let _e110 = (_e101.w * _e101.y);
    let _e111 = (_e101.w * _e101.z);
    let _e112 = (_e101.x * _e101.y);
    let _e113 = (_e101.x * _e101.z);
    let _e114 = (_e101.y * _e101.z);
    let _e138 = splat_scale.member[_e71];
    let _e139 = _e138.xyz;
    let _e146 = vec3<f32>(exp(_e139.x), exp(_e139.y), exp(_e139.z));
    let _e148 = global.view_matrix;
    let _e151 = (_e148 * vec4<f32>((1f - (2f * (_e107 + _e108))), (2f * (_e112 + _e111)), (2f * (_e113 - _e110)), 0f)).xyz;
    let _e153 = global.view_matrix;
    let _e156 = (_e153 * vec4<f32>((2f * (_e112 - _e111)), (1f - (2f * (_e106 + _e108))), (2f * (_e114 + _e109)), 0f)).xyz;
    let _e158 = global.view_matrix;
    let _e161 = (_e158 * vec4<f32>((2f * (_e113 + _e110)), (2f * (_e114 - _e109)), (1f - (2f * (_e106 + _e107))), 0f)).xyz;
    let _e174 = (_e146.x * _e146.x);
    let _e175 = (_e146.y * _e146.y);
    let _e176 = (_e146.z * _e146.z);
    let _e200 = ((((_e151.x * _e151.z) * _e174) + ((_e156.x * _e156.z) * _e175)) + ((_e161.x * _e161.z) * _e176));
    let _e216 = ((((_e151.y * _e151.z) * _e174) + ((_e156.y * _e156.z) * _e175)) + ((_e161.y * _e161.z) * _e176));
    let _e224 = ((((_e151.z * _e151.z) * _e174) + ((_e156.z * _e156.z) * _e175)) + ((_e161.z * _e161.z) * _e176));
    let _e228 = -(_e86.z);
    let _e229 = (_e228 * _e228);
    let _e231 = global.focal_x;
    let _e233 = global.focal_y;
    let _e234 = (_e231 / _e228);
    let _e236 = ((_e231 * _e86.x) / _e229);
    let _e238 = -((_e233 / _e228));
    let _e241 = -(((_e233 * _e86.y) / _e229));
    let _e250 = ((((_e234 * _e234) * ((((_e151.x * _e151.x) * _e174) + ((_e156.x * _e156.x) * _e175)) + ((_e161.x * _e161.x) * _e176))) + ((2f * (_e234 * _e236)) * _e200)) + ((_e236 * _e236) * _e224));
    let _e261 = ((((_e234 * _e238) * ((((_e151.x * _e151.y) * _e174) + ((_e156.x * _e156.y) * _e175)) + ((_e161.x * _e161.y) * _e176))) + ((_e234 * _e241) * _e200)) + (((_e236 * _e238) * _e216) + ((_e236 * _e241) * _e224)));
    let _e270 = ((((_e238 * _e238) * ((((_e151.y * _e151.y) * _e174) + ((_e156.y * _e156.y) * _e175)) + ((_e161.y * _e161.y) * _e176))) + ((2f * (_e238 * _e241)) * _e216)) + ((_e241 * _e241) * _e224));
    let _e271 = (_e261 * _e261);
    let _e274 = (_e250 + 0.3f);
    let _e275 = (_e270 + 0.3f);
    let _e277 = ((_e274 * _e275) - _e271);
    if (_e277 <= 0f) {
        projected_center.member[_e71] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e71] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e71] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e71] = 4294967295u;
        sorted_indices.member[_e71] = _e71;
        return;
    }
    let _e292 = (1f / _e277);
    let _e298 = (0.5f * (_e274 + _e275));
    let _e307 = min(ceil((3f * sqrt((_e298 + sqrt(max(((_e298 * _e298) - _e277), 0f)))))), 256f);
    let _e309 = global.screen_size;
    let _e312 = global.screen_size;
    let _e314 = min(_e309.x, _e312.y);
    if (_e307 > (0.25f * _e314)) {
        projected_center.member[_e71] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e71] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e71] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e71] = 4294967295u;
        sorted_indices.member[_e71] = _e71;
        return;
    }
    let _e328 = global.proj_matrix;
    let _e333 = (_e328 * vec4<f32>(_e86.x, _e86.y, _e86.z, 1f));
    let _e336 = (_e333.x / _e333.w);
    let _e339 = (_e333.y / _e333.w);
    let _e344 = (1f + (_e307 / _e314));
    let _e345 = -(_e344);
    if (((_e336 > _e344) || (_e336 < _e345)) || ((_e339 > _e344) || (_e339 < _e345))) {
        projected_center.member[_e71] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e71] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e71] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e71] = 4294967295u;
        sorted_indices.member[_e71] = _e71;
        return;
    }
    let _e365 = splat_opacity.member[_e71];
    let _e370 = ((1f / (1f + exp(-(_e365)))) * sqrt(max((((_e250 * _e270) - _e271) / _e277), 0f)));
    if (_e370 < 0.00392157f) {
        projected_center.member[_e71] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_conic.member[_e71] = vec4<f32>(0f, 0f, 0f, 0f);
        projected_color.member[_e71] = vec4<f32>(0f, 0f, 0f, 0f);
        sort_keys.member[_e71] = 4294967295u;
        sorted_indices.member[_e71] = _e71;
        return;
    }
    let _e383 = global.cam_pos;
    let _e385 = normalize((_e78 - _e383));
    let _e388 = splat_sh0_.member[_e71];
    let _e397 = splat_sh1_.member[_e71];
    let _e401 = splat_sh2_.member[_e71];
    let _e405 = splat_sh3_.member[_e71];
    let _e418 = splat_sh4_.member[_e71];
    let _e422 = splat_sh5_.member[_e71];
    let _e426 = splat_sh6_.member[_e71];
    let _e430 = splat_sh7_.member[_e71];
    let _e434 = splat_sh8_.member[_e71];
    let _e436 = (_e385.x * _e385.x);
    let _e437 = (_e385.y * _e385.y);
    let _e438 = (_e385.z * _e385.z);
    let _e440 = (_e385.y * _e385.z);
    let _e442 = (2f * _e438);
    let _e443 = (_e436 + _e437);
    let _e444 = (_e436 - _e437);
    let _e463 = splat_sh9_.member[_e71];
    let _e467 = splat_sh10_.member[_e71];
    let _e471 = splat_sh11_.member[_e71];
    let _e475 = splat_sh12_.member[_e71];
    let _e479 = splat_sh13_.member[_e71];
    let _e483 = splat_sh14_.member[_e71];
    let _e487 = splat_sh15_.member[_e71];
    let _e490 = ((4f * _e438) - _e443);
    let _e524 = (((((_e388.xyz * 0.28209478f) + vec3<f32>(0.5f, 0.5f, 0.5f)) + ((((_e397.xyz * _e385.y) * -0.48860252f) + ((_e401.xyz * _e385.z) * 0.48860252f)) + ((_e405.xyz * _e385.x) * -0.48860252f))) + ((((_e418.xyz * (1.0925485f * (_e385.x * _e385.y))) + (_e422.xyz * (-1.0925485f * _e440))) + ((_e426.xyz * (0.31539157f * (_e442 - _e443))) + (_e430.xyz * (-1.0925485f * (_e385.x * _e385.z))))) + (_e434.xyz * (0.54627424f * _e444)))) + ((((_e463.xyz * (-0.5900436f * (_e385.y * ((3f * _e436) - _e437)))) + (_e467.xyz * (2.8906114f * (_e385.x * _e440)))) + (_e471.xyz * (-0.4570458f * (_e385.y * _e490)))) + (((_e475.xyz * (0.37317634f * (_e385.z * (_e442 - (3f * _e443))))) + (_e479.xyz * (-0.4570458f * (_e385.x * _e490)))) + ((_e483.xyz * (1.4453057f * (_e385.z * _e444))) + (_e487.xyz * (-0.5900436f * (_e385.x * (_e436 - (3f * _e437)))))))));
    projected_center.member[_e71] = vec4<f32>(_e336, _e339, (_e333.z / _e333.w), _e307);
    projected_conic.member[_e71] = vec4<f32>((_e275 * _e292), (-(_e261) * _e292), (_e274 * _e292), _e370);
    projected_color.member[_e71] = vec4<f32>(clamp(_e524.x, 0f, 1f), clamp(_e524.y, 0f, 1f), clamp(_e524.z, 0f, 1f), _e370);
    let _e540 = bitcast<u32>(_e86.z);
    sort_keys.member[_e71] = (_e540 ^ (((_e540 >> bitcast<u32>(31u)) * 4294967295u) | 2147483648u));
    sorted_indices.member[_e71] = _e71;
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
