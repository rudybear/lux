struct type_2 {
    member: array<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> input_data: type_2;
@group(0) @binding(1) 
var<storage, read_write> output_data: type_2;
var<private> local_invocation_id_1: vec3<u32>;
var<private> local_invocation_index_1: u32;
var<private> workgroup_id_1: vec3<u32>;
var<private> num_workgroups_1: vec3<u32>;
var<private> global_invocation_id_1: vec3<u32>;
var<workgroup> scratch: array<u32, 256>;

fn main_1() {
    let _e21 = local_invocation_index_1;
    let _e22 = global_invocation_id_1;
    let _e28 = input_data.member[i32(f32(_e22.x))];
    scratch[_e21] = _e28;
    workgroupBarrier();
    if (_e21 < 128u) {
        let _e32 = scratch[_e21];
        let _e35 = scratch[(_e21 + 128u)];
        scratch[_e21] = (_e32 + _e35);
    }
    workgroupBarrier();
    if (_e21 < 64u) {
        let _e40 = scratch[_e21];
        let _e43 = scratch[(_e21 + 64u)];
        scratch[_e21] = (_e40 + _e43);
    }
    workgroupBarrier();
    if (_e21 < 32u) {
        let _e48 = scratch[_e21];
        let _e51 = scratch[(_e21 + 32u)];
        scratch[_e21] = (_e48 + _e51);
    }
    workgroupBarrier();
    if (_e21 < 16u) {
        let _e56 = scratch[_e21];
        let _e59 = scratch[(_e21 + 16u)];
        scratch[_e21] = (_e56 + _e59);
    }
    workgroupBarrier();
    if (_e21 < 8u) {
        let _e64 = scratch[_e21];
        let _e67 = scratch[(_e21 + 8u)];
        scratch[_e21] = (_e64 + _e67);
    }
    workgroupBarrier();
    if (_e21 < 4u) {
        let _e72 = scratch[_e21];
        let _e75 = scratch[(_e21 + 4u)];
        scratch[_e21] = (_e72 + _e75);
    }
    workgroupBarrier();
    if (_e21 < 2u) {
        let _e80 = scratch[_e21];
        let _e83 = scratch[(_e21 + 2u)];
        scratch[_e21] = (_e80 + _e83);
    }
    workgroupBarrier();
    let _e86 = (_e21 < 1u);
    if _e86 {
        let _e88 = scratch[_e21];
        let _e91 = scratch[(_e21 + 1u)];
        scratch[_e21] = (_e88 + _e91);
    }
    workgroupBarrier();
    if _e86 {
        let _e94 = workgroup_id_1;
        let _e97 = scratch[i32(0f)];
        output_data.member[i32(f32(_e94.x))] = _e97;
    }
    return;
}

@compute @workgroup_size(64, 1, 1) 
fn main(@builtin(local_invocation_id) local_invocation_id: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32, @builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>, @builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    local_invocation_id_1 = local_invocation_id;
    local_invocation_index_1 = local_invocation_index;
    workgroup_id_1 = workgroup_id;
    num_workgroups_1 = num_workgroups;
    global_invocation_id_1 = global_invocation_id;
    main_1();
}
