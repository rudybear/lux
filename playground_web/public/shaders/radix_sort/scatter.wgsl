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

struct KeysOut {
    keys_out: array<u32>,
}

struct ValsOut {
    vals_out: array<u32>,
}

struct ValsIn {
    vals_in: array<u32>,
}

var<private> gl_LocalInvocationID_1: vec3<u32>;
var<private> gl_WorkGroupID_1: vec3<u32>;
var<private> gl_NumWorkGroups_1: vec3<u32>;
@group(1) @binding(0) var<uniform> unnamed: PushConstants;
var<workgroup> s_keys: array<u32, 3840>;
@group(0) @binding(0) 
var<storage, read_write> unnamed_1: KeysIn;
var<workgroup> s_global_offset: array<u32, 256>;
@group(0) @binding(4) 
var<storage, read_write> unnamed_2: Histograms;
var<workgroup> s_row_offset: array<u32, 256>;
var<workgroup> s_digits: array<u32, 256>;
@group(0) @binding(1) 
var<storage, read_write> unnamed_3: KeysOut;
@group(0) @binding(3) 
var<storage, read_write> unnamed_4: ValsOut;
@group(0) @binding(2) 
var<storage, read_write> unnamed_5: ValsIn;
var<workgroup> s_row_count: array<u32, 256>;

fn main_1() {
    var local_id: u32;
    var group_id: u32;
    var num_groups: u32;
    var tile_start: u32;
    var tile_end: u32;
    var tile_count: u32;
    var i: u32;
    var local_idx: u32;
    var global_idx: u32;
    var i_1: u32;
    var local_idx_1: u32;
    var global_idx_1: u32;
    var valid: bool;
    var digit: u32;
    var key: u32;
    var rank_in_row: u32;
    var j: u32;
    var local_rank: u32;
    var out_idx: u32;
    var row_valid: u32;
    var local: u32;
    var cnt: u32;
    var j_1: u32;

    let _e50 = gl_LocalInvocationID_1[0u];
    local_id = _e50;
    let _e52 = gl_WorkGroupID_1[0u];
    group_id = _e52;
    let _e54 = gl_NumWorkGroups_1[0u];
    num_groups = _e54;
    let _e55 = group_id;
    tile_start = (_e55 * 3840u);
    let _e57 = tile_start;
    let _e60 = unnamed.num_elements;
    tile_end = min((_e57 + 3840u), _e60);
    let _e62 = tile_end;
    let _e63 = tile_start;
    tile_count = (_e62 - _e63);
    i = 0u;
    loop {
        let _e65 = i;
        if (_e65 < 15u) {
            let _e67 = local_id;
            let _e68 = i;
            local_idx = (_e67 + (_e68 * 256u));
            let _e71 = tile_start;
            let _e72 = local_idx;
            global_idx = (_e71 + _e72);
            let _e74 = local_idx;
            let _e75 = tile_count;
            if (_e74 < _e75) {
                let _e77 = local_idx;
                let _e78 = global_idx;
                let _e81 = unnamed_1.keys_in[_e78];
                s_keys[_e77] = _e81;
            }
            continue;
        } else {
            break;
        }
        continuing {
            let _e83 = i;
            i = (_e83 + bitcast<u32>(1i));
        }
    }
    let _e86 = local_id;
    let _e87 = local_id;
    let _e88 = num_groups;
    let _e90 = group_id;
    let _e94 = unnamed_2.histograms[((_e87 * _e88) + _e90)];
    s_global_offset[_e86] = _e94;
    let _e96 = local_id;
    s_row_offset[_e96] = 0u;
    workgroupBarrier();
    i_1 = 0u;
    loop {
        let _e98 = i_1;
        if (_e98 < 15u) {
            let _e100 = local_id;
            let _e101 = i_1;
            local_idx_1 = (_e100 + (_e101 * 256u));
            let _e104 = tile_start;
            let _e105 = local_idx_1;
            global_idx_1 = (_e104 + _e105);
            let _e107 = local_idx_1;
            let _e108 = tile_count;
            valid = (_e107 < _e108);
            digit = 4294967295u;
            key = 0u;
            let _e110 = valid;
            if _e110 {
                let _e111 = local_idx_1;
                let _e113 = s_keys[_e111];
                key = _e113;
                let _e114 = key;
                let _e116 = unnamed.bit_offset;
                digit = ((_e114 >> bitcast<u32>(_e116)) & 255u);
            }
            let _e120 = local_id;
            let _e121 = digit;
            s_digits[_e120] = _e121;
            workgroupBarrier();
            rank_in_row = 0u;
            let _e123 = valid;
            if _e123 {
                j = 0u;
                loop {
                    let _e124 = j;
                    let _e125 = local_id;
                    if (_e124 < _e125) {
                        let _e127 = j;
                        let _e129 = s_digits[_e127];
                        let _e130 = digit;
                        if (_e129 == _e130) {
                            let _e132 = rank_in_row;
                            rank_in_row = (_e132 + bitcast<u32>(1i));
                        }
                        continue;
                    } else {
                        break;
                    }
                    continuing {
                        let _e135 = j;
                        j = (_e135 + bitcast<u32>(1i));
                    }
                }
                let _e138 = digit;
                let _e140 = s_row_offset[_e138];
                let _e141 = rank_in_row;
                local_rank = (_e140 + _e141);
                let _e143 = digit;
                let _e145 = s_global_offset[_e143];
                let _e146 = local_rank;
                out_idx = (_e145 + _e146);
                let _e148 = out_idx;
                let _e149 = key;
                unnamed_3.keys_out[_e148] = _e149;
                let _e152 = out_idx;
                let _e153 = global_idx_1;
                let _e156 = unnamed_5.vals_in[_e153];
                unnamed_4.vals_out[_e152] = _e156;
            }
            workgroupBarrier();
            let _e159 = i_1;
            let _e161 = tile_count;
            if ((_e159 * 256u) < _e161) {
                let _e163 = tile_count;
                let _e164 = i_1;
                local = min((_e163 - (_e164 * 256u)), 256u);
            } else {
                local = 0u;
            }
            let _e168 = local;
            row_valid = _e168;
            cnt = 0u;
            j_1 = 0u;
            loop {
                let _e169 = j_1;
                let _e170 = row_valid;
                if (_e169 < _e170) {
                    let _e172 = j_1;
                    let _e174 = s_digits[_e172];
                    let _e175 = local_id;
                    if (_e174 == _e175) {
                        let _e177 = cnt;
                        cnt = (_e177 + bitcast<u32>(1i));
                    }
                    continue;
                } else {
                    break;
                }
                continuing {
                    let _e180 = j_1;
                    j_1 = (_e180 + bitcast<u32>(1i));
                }
            }
            let _e183 = local_id;
            let _e184 = cnt;
            let _e186 = s_row_offset[_e183];
            s_row_offset[_e183] = (_e186 + _e184);
            workgroupBarrier();
            continue;
        } else {
            break;
        }
        continuing {
            let _e189 = i_1;
            i_1 = (_e189 + bitcast<u32>(1i));
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn main(@builtin(local_invocation_id) gl_LocalInvocationID: vec3<u32>, @builtin(workgroup_id) gl_WorkGroupID: vec3<u32>, @builtin(num_workgroups) gl_NumWorkGroups: vec3<u32>) {
    gl_LocalInvocationID_1 = gl_LocalInvocationID;
    gl_WorkGroupID_1 = gl_WorkGroupID;
    gl_NumWorkGroups_1 = gl_NumWorkGroups;
    main_1();
}
