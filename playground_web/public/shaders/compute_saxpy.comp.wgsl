struct params {
    a: f32,
    count: f32,
}

struct type_3 {
    member: array<f32>,
}

@group(1) @binding(0) 
var<uniform> global: params;
@group(0) @binding(0) 
var<storage, read_write> x: type_3;
@group(0) @binding(1) 
var<storage, read_write> y: type_3;
@group(0) @binding(2) 
var<storage, read_write> result: type_3;
var<private> local_invocation_id_1: vec3<u32>;
var<private> local_invocation_index_1: u32;
var<private> workgroup_id_1: vec3<u32>;
var<private> num_workgroups_1: vec3<u32>;
var<private> global_invocation_id_1: vec3<u32>;

fn main_1() {
    let _e11 = global_invocation_id_1;
    let _e13 = f32(_e11.x);
    let _e15 = global.count;
    if (_e13 < _e15) {
        let _e18 = global.a;
        let _e22 = x.member[i32(_e13)];
        let _e27 = y.member[i32(_e13)];
        result.member[i32(_e13)] = ((_e18 * _e22) + _e27);
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
