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
var<storage, read_write> keys_out: type_3;
@group(0) @binding(2) 
var<storage, read_write> vals_in: type_3;
@group(0) @binding(3) 
var<storage, read_write> vals_out: type_3;
@group(0) @binding(4) 
var<storage, read_write> histograms: type_3;
var<private> local_invocation_id_1: vec3<u32>;
var<private> local_invocation_index_1: u32;
var<private> workgroup_id_1: vec3<u32>;
var<private> num_workgroups_1: vec3<u32>;
var<private> global_invocation_id_1: vec3<u32>;
var<workgroup> s_keys: array<u32, 3840>;
var<workgroup> s_global_offset: array<u32, 256>;
var<workgroup> s_row_offset: array<u32, 256>;
var<workgroup> s_digits: array<u32, 256>;

fn main_1() {
    var local: u32;
    var local_1: u32;
    var local_2: u32;
    var local_3: u32;
    var local_4: u32;
    var local_5: u32;
    var local_6: u32;
    var local_7: u32;
    var local_8: u32;
    var local_9: u32;

    let _e39 = local_invocation_id_1;
    let _e42 = u32(f32(_e39.x));
    let _e43 = workgroup_id_1;
    let _e46 = u32(f32(_e43.x));
    let _e47 = num_workgroups_1;
    let _e51 = (_e46 * 3840u);
    let _e52 = (_e51 + 3840u);
    local = _e52;
    let _e54 = global.num_elements;
    if (_e52 > _e54) {
        let _e57 = global.num_elements;
        local = _e57;
    }
    let _e58 = local;
    let _e59 = (_e58 - _e51);
    let _e60 = local_2;
    let _e61 = (_e60 < 15u);
    let _e62 = local_2;
    let _e63 = (_e62 + 1u);
    local_2 = 0u;
    loop {
        if _e61 {
            let _e64 = local_2;
            let _e66 = (_e42 + (_e64 * 256u));
            if (_e66 < _e59) {
                let _e71 = keys_in.member[(_e51 + _e66)];
                s_keys[_e66] = _e71;
            }
            continue;
        } else {
            break;
        }
        continuing {
            local_2 = _e63;
        }
    }
    let _e77 = histograms.member[((_e42 * u32(f32(_e47.x))) + _e46)];
    s_global_offset[_e42] = _e77;
    s_row_offset[_e42] = 0u;
    workgroupBarrier();
    local_2 = 0u;
    loop {
        if _e61 {
            let _e80 = local_2;
            let _e81 = (_e80 * 256u);
            let _e82 = (_e42 + _e81);
            local_3 = 4294967295u;
            local_4 = 0u;
            let _e84 = (_e82 < _e59);
            if _e84 {
                let _e86 = s_keys[_e82];
                local_4 = _e86;
                let _e87 = local_4;
                let _e89 = global.bit_offset;
                local_3 = ((_e87 >> bitcast<u32>(_e89)) & 255u);
            }
            let _e93 = local_3;
            s_digits[_e42] = _e93;
            workgroupBarrier();
            if _e84 {
                local_5 = 0u;
                local_9 = 0u;
                loop {
                    let _e95 = local_9;
                    if (_e95 < _e42) {
                        let _e97 = local_9;
                        let _e99 = s_digits[_e97];
                        let _e100 = local_3;
                        if (_e99 == _e100) {
                            let _e102 = local_5;
                            local_5 = (_e102 + 1u);
                        }
                        continue;
                    } else {
                        break;
                    }
                    continuing {
                        let _e104 = local_9;
                        local_9 = (_e104 + 1u);
                    }
                }
                let _e106 = local_3;
                let _e108 = s_row_offset[_e106];
                let _e109 = local_5;
                let _e111 = local_3;
                let _e113 = s_global_offset[_e111];
                let _e114 = (_e113 + (_e108 + _e109));
                let _e115 = local_4;
                keys_out.member[_e114] = _e115;
                let _e120 = vals_in.member[(_e51 + _e82)];
                vals_out.member[_e114] = _e120;
            }
            workgroupBarrier();
            local_7 = 0u;
            if (_e81 < _e59) {
                let _e124 = (_e59 - _e81);
                if (_e124 > 256u) {
                    local_7 = 256u;
                    local_7 = _e124;
                }
            }
            local_8 = 0u;
            local_9 = 0u;
            loop {
                let _e126 = local_9;
                let _e127 = local_7;
                if (_e126 < _e127) {
                    let _e129 = local_9;
                    let _e131 = s_digits[_e129];
                    if (_e131 == _e42) {
                        let _e133 = local_8;
                        local_8 = (_e133 + 1u);
                    }
                    continue;
                } else {
                    break;
                }
                continuing {
                    let _e135 = local_9;
                    local_9 = (_e135 + 1u);
                }
            }
            let _e138 = s_row_offset[_e42];
            let _e139 = local_8;
            s_row_offset[_e42] = (_e138 + _e139);
            workgroupBarrier();
            continue;
        } else {
            break;
        }
        continuing {
            local_2 = _e63;
        }
    }
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
