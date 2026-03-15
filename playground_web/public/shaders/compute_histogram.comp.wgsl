struct type_2 {
    member: array<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> input_data: type_2;
@group(0) @binding(1) 
var<storage, read_write> output_histogram: type_2;
var<private> local_invocation_id_1: vec3<u32>;
var<private> local_invocation_index_1: u32;
var<private> workgroup_id_1: vec3<u32>;
var<private> num_workgroups_1: vec3<u32>;
var<private> global_invocation_id_1: vec3<u32>;
var<workgroup> histogram: array<u32, 256>;

fn main_1() {
    workgroupBarrier();
    workgroupBarrier();
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
