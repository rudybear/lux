struct params {
    num_entries: u32,
    pass_id: u32,
}

struct type_3 {
    member: array<u32>,
}

@group(1) @binding(0) 
var<uniform> global: params;
@group(0) @binding(0) 
var<storage, read_write> histograms: type_3;
@group(0) @binding(1) 
var<storage, read_write> partition_sums: type_3;
var<private> local_invocation_id_1: vec3<u32>;
var<private> local_invocation_index_1: u32;
var<private> workgroup_id_1: vec3<u32>;
var<private> num_workgroups_1: vec3<u32>;
var<private> global_invocation_id_1: vec3<u32>;
var<workgroup> s_data: array<u32, 2048>;

fn main_1() {
    var local: u32;
    var local_1: u32;
    var local_2: u32;
    var local_3: u32;
    var local_4: u32;
    var local_5: u32;
    var local_6: u32;

    let _e32 = local_invocation_id_1;
    let _e35 = u32(f32(_e32.x));
    let _e36 = workgroup_id_1;
    let _e39 = u32(f32(_e36.x));
    let _e41 = global.pass_id;
    if (_e41 == 0u) {
        let _e44 = ((_e39 * 2048u) + _e35);
        let _e45 = (_e44 + 1024u);
        let _e47 = global.num_entries;
        let _e48 = (_e44 < _e47);
        if _e48 {
            let _e51 = histograms.member[_e44];
            s_data[_e35] = _e51;
            s_data[_e35] = 0u;
        }
        let _e55 = global.num_entries;
        let _e56 = (_e45 < _e55);
        if _e56 {
            let _e59 = histograms.member[_e45];
            s_data[(_e35 + 1024u)] = _e59;
            s_data[(_e35 + 1024u)] = 0u;
        }
        workgroupBarrier();
        local = 1u;
        local_1 = 1024u;
        loop {
            let _e64 = local_1;
            if (_e64 > 0u) {
                workgroupBarrier();
                let _e66 = local_1;
                if (_e35 < _e66) {
                    let _e68 = (2u * _e35);
                    let _e69 = local;
                    let _e73 = local;
                    let _e76 = ((_e73 * (_e68 + 2u)) - 1u);
                    let _e78 = s_data[_e76];
                    let _e80 = s_data[((_e69 * (_e68 + 1u)) - 1u)];
                    s_data[_e76] = (_e78 + _e80);
                }
                let _e83 = local;
                local = (_e83 << bitcast<u32>(1u));
                let _e86 = local_1;
                local_1 = (_e86 >> bitcast<u32>(1u));
                continue;
            } else {
                break;
            }
        }
        workgroupBarrier();
        let _e91 = s_data[i32(2047f)];
        workgroupBarrier();
        let _e92 = (_e35 == 0u);
        if _e92 {
            s_data[i32(2047f)] = 0u;
        }
        local_2 = 1u;
        loop {
            let _e95 = local_2;
            if (_e95 < 2048u) {
                let _e97 = local;
                local = (_e97 >> bitcast<u32>(1u));
                workgroupBarrier();
                let _e100 = local_2;
                if (_e35 < _e100) {
                    let _e102 = (2u * _e35);
                    let _e103 = local;
                    let _e106 = ((_e103 * (_e102 + 1u)) - 1u);
                    let _e107 = local;
                    let _e110 = ((_e107 * (_e102 + 2u)) - 1u);
                    let _e112 = s_data[_e106];
                    let _e114 = s_data[_e110];
                    s_data[_e106] = _e114;
                    s_data[_e110] = (_e114 + _e112);
                }
                let _e118 = local_2;
                local_2 = (_e118 << bitcast<u32>(1u));
                continue;
            } else {
                break;
            }
        }
        workgroupBarrier();
        if _e92 {
            partition_sums.member[_e39] = _e91;
        }
        if _e48 {
            let _e124 = s_data[_e35];
            histograms.member[_e44] = _e124;
        }
        if _e56 {
            let _e129 = s_data[(_e35 + 1024u)];
            histograms.member[_e45] = _e129;
        }
    }
    let _e133 = global.pass_id;
    if (_e133 == 1u) {
        let _e136 = global.num_entries;
        local_3 = _e136;
        let _e137 = local_3;
        if (_e137 < 2u) {
            local_3 = 2u;
        }
        let _e139 = local_3;
        local_3 = (_e139 - 1u);
        let _e141 = local_3;
        let _e142 = local_3;
        local_3 = (_e141 | (_e142 >> bitcast<u32>(1u)));
        let _e146 = local_3;
        let _e147 = local_3;
        local_3 = (_e146 | (_e147 >> bitcast<u32>(2u)));
        let _e151 = local_3;
        let _e152 = local_3;
        local_3 = (_e151 | (_e152 >> bitcast<u32>(4u)));
        let _e156 = local_3;
        let _e157 = local_3;
        local_3 = (_e156 | (_e157 >> bitcast<u32>(8u)));
        let _e161 = local_3;
        let _e162 = local_3;
        local_3 = (_e161 | (_e162 >> bitcast<u32>(16u)));
        let _e166 = local_3;
        let _e167 = (_e166 + 1u);
        let _e168 = (_e35 < _e136);
        if _e168 {
            let _e171 = partition_sums.member[_e35];
            s_data[_e35] = _e171;
            if (_e35 < _e167) {
                s_data[_e35] = 0u;
            }
        }
        if ((_e35 + 1024u) < _e167) {
            if ((_e35 + 1024u) < _e136) {
                let _e182 = partition_sums.member[(_e35 + 1024u)];
                s_data[(_e35 + 1024u)] = _e182;
                s_data[(_e35 + 1024u)] = 0u;
            }
        }
        workgroupBarrier();
        local_4 = 1u;
        local_5 = (_e167 >> bitcast<u32>(1u));
        loop {
            let _e189 = local_5;
            if (_e189 > 0u) {
                workgroupBarrier();
                let _e191 = local_5;
                if (_e35 < _e191) {
                    let _e193 = (2u * _e35);
                    let _e194 = local_4;
                    let _e198 = local_4;
                    let _e201 = ((_e198 * (_e193 + 2u)) - 1u);
                    let _e203 = s_data[_e201];
                    let _e205 = s_data[((_e194 * (_e193 + 1u)) - 1u)];
                    s_data[_e201] = (_e203 + _e205);
                }
                let _e208 = local_4;
                local_4 = (_e208 << bitcast<u32>(1u));
                let _e211 = local_5;
                local_5 = (_e211 >> bitcast<u32>(1u));
                continue;
            } else {
                break;
            }
        }
        workgroupBarrier();
        workgroupBarrier();
        if (_e35 == 0u) {
            s_data[(_e167 - 1u)] = 0u;
        }
        local_6 = 1u;
        loop {
            let _e217 = local_6;
            if (_e217 < _e167) {
                let _e219 = local_4;
                local_4 = (_e219 >> bitcast<u32>(1u));
                workgroupBarrier();
                let _e222 = local_6;
                if (_e35 < _e222) {
                    let _e224 = (2u * _e35);
                    let _e225 = local_4;
                    let _e228 = ((_e225 * (_e224 + 1u)) - 1u);
                    let _e229 = local_4;
                    let _e232 = ((_e229 * (_e224 + 2u)) - 1u);
                    let _e234 = s_data[_e228];
                    let _e236 = s_data[_e232];
                    s_data[_e228] = _e236;
                    s_data[_e232] = (_e236 + _e234);
                }
                let _e240 = local_6;
                local_6 = (_e240 << bitcast<u32>(1u));
                continue;
            } else {
                break;
            }
        }
        workgroupBarrier();
        if _e168 {
            let _e244 = s_data[_e35];
            partition_sums.member[_e35] = _e244;
        }
    }
    let _e248 = global.pass_id;
    if (_e248 == 2u) {
        let _e253 = partition_sums.member[_e39];
        if (_e253 != 0u) {
            let _e255 = ((_e39 * 2048u) + _e35);
            let _e256 = (_e255 + 1024u);
            let _e258 = global.num_entries;
            if (_e255 < _e258) {
                let _e262 = histograms.member[_e255];
                histograms.member[_e255] = (_e262 + _e253);
            }
            let _e267 = global.num_entries;
            if (_e256 < _e267) {
                let _e271 = histograms.member[_e256];
                histograms.member[_e256] = (_e271 + _e253);
            }
        }
    }
    return;
}

@compute @workgroup_size(1024, 1, 1) 
fn main(@builtin(local_invocation_id) local_invocation_id: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32, @builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>, @builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    local_invocation_id_1 = local_invocation_id;
    local_invocation_index_1 = local_invocation_index;
    workgroup_id_1 = workgroup_id;
    num_workgroups_1 = num_workgroups;
    global_invocation_id_1 = global_invocation_id;
    main_1();
}
