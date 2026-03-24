struct PushConstants {
    num_elements: u32,
    bit_offset: u32,
}

struct KeysIn {
    keys_in: array<u32>,
}

struct Histograms {
    histograms: array<u32>,
}

var<private> gl_LocalInvocationID_1: vec3<u32>;
var<private> gl_WorkGroupID_1: vec3<u32>;
var<private> gl_NumWorkGroups_1: vec3<u32>;
var<workgroup> s_histogram: array<atomic<u32>, 256>;
@group(1) @binding(0) var<uniform> unnamed: PushConstants;
@group(0) @binding(0) 
var<storage, read_write> unnamed_1: KeysIn;
@group(0) @binding(1) 
var<storage, read_write> unnamed_2: Histograms;

fn main_1() {
    var local_id: u32;
    var group_id: u32;
    var num_groups: u32;
    var tile_start: u32;
    var i: u32;
    var idx: u32;
    var key: u32;
    var digit: u32;

    let _e27 = gl_LocalInvocationID_1[0u];
    local_id = _e27;
    let _e29 = gl_WorkGroupID_1[0u];
    group_id = _e29;
    let _e31 = gl_NumWorkGroups_1[0u];
    num_groups = _e31;
    let _e32 = local_id;
    atomicStore((&s_histogram[_e32]), 0u);
    workgroupBarrier();
    let _e34 = group_id;
    tile_start = (_e34 * 3840u);
    i = 0u;
    loop {
        let _e36 = i;
        if (_e36 < 15u) {
            let _e38 = tile_start;
            let _e39 = local_id;
            let _e41 = i;
            idx = ((_e38 + _e39) + (_e41 * 256u));
            let _e44 = idx;
            let _e46 = unnamed.num_elements;
            if (_e44 < _e46) {
                let _e48 = idx;
                let _e51 = unnamed_1.keys_in[_e48];
                key = _e51;
                let _e52 = key;
                let _e54 = unnamed.bit_offset;
                digit = ((_e52 >> bitcast<u32>(_e54)) & 255u);
                let _e58 = digit;
                let _e60 = atomicAdd((&s_histogram[_e58]), 1u);
            }
            continue;
        } else {
            break;
        }
        continuing {
            let _e61 = i;
            i = (_e61 + bitcast<u32>(1i));
        }
    }
    workgroupBarrier();
    let _e64 = local_id;
    let _e65 = num_groups;
    let _e67 = group_id;
    let _e69 = local_id;
    let _e71 = atomicLoad((&s_histogram[_e69]));
    unnamed_2.histograms[((_e64 * _e65) + _e67)] = _e71;
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn main(@builtin(local_invocation_id) gl_LocalInvocationID: vec3<u32>, @builtin(workgroup_id) gl_WorkGroupID: vec3<u32>, @builtin(num_workgroups) gl_NumWorkGroups: vec3<u32>) {
    gl_LocalInvocationID_1 = gl_LocalInvocationID;
    gl_WorkGroupID_1 = gl_WorkGroupID;
    gl_NumWorkGroups_1 = gl_NumWorkGroups;
    main_1();
}
