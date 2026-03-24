struct params {
    num_elements: u32,
    bit_offset: u32,
}

struct type_3 {
    member: array<u32>,
}

@group(1) @binding(0) 
var<uniform> global: params;
@group(0) @binding(0) 
var<storage, read_write> keys_in: type_3;
@group(0) @binding(1) 
var<storage, read_write> histograms: type_3;
var<private> local_invocation_id_1: vec3<u32>;
var<private> local_invocation_index_1: u32;
var<private> workgroup_id_1: vec3<u32>;
var<private> num_workgroups_1: vec3<u32>;
var<private> global_invocation_id_1: vec3<u32>;
var<workgroup> s_histogram: array<u32, 256>;

fn main_1() {
    var local: u32;

    let _e19 = local_invocation_id_1;
    let _e22 = u32(f32(_e19.x));
    let _e23 = workgroup_id_1;
    let _e26 = u32(f32(_e23.x));
    let _e27 = num_workgroups_1;
    s_histogram[_e22] = 0u;
    workgroupBarrier();
    local = 0u;
    loop {
        let _e33 = local;
        if (_e33 < 15u) {
            let _e36 = local;
            let _e40 = global.num_elements;
            if ((((_e26 * 3840u) + _e22) + (_e36 * 256u)) < _e40) {
            }
            continue;
        } else {
            break;
        }
        continuing {
            let _e42 = local;
            local = (_e42 + 1u);
        }
    }
    workgroupBarrier();
    let _e45 = s_histogram[_e22];
    histograms.member[((_e22 * u32(f32(_e27.x))) + _e26)] = _e45;
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
