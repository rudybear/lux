struct PushConstants {
    num_entries: u32,
    pass_id: u32,
}

struct Histograms {
    histograms: array<u32>,
}

struct PartitionSums {
    partition_sums: array<u32>,
}

var<workgroup> s_data: array<u32, 2048>;
var<private> gl_LocalInvocationID_1: vec3<u32>;
var<private> gl_WorkGroupID_1: vec3<u32>;
@group(1) @binding(0) var<uniform> unnamed: PushConstants;
@group(0) @binding(0) 
var<storage, read_write> unnamed_1: Histograms;
@group(0) @binding(1) 
var<storage, read_write> unnamed_2: PartitionSums;

fn next_pow2_u0028_u1_u003b(v: ptr<function, u32>) -> u32 {
    let _e19 = (*v);
    (*v) = max(_e19, 2u);
    let _e21 = (*v);
    (*v) = (_e21 - bitcast<u32>(1i));
    let _e24 = (*v);
    let _e27 = (*v);
    (*v) = (_e27 | (_e24 >> bitcast<u32>(1u)));
    let _e29 = (*v);
    let _e32 = (*v);
    (*v) = (_e32 | (_e29 >> bitcast<u32>(2u)));
    let _e34 = (*v);
    let _e37 = (*v);
    (*v) = (_e37 | (_e34 >> bitcast<u32>(4u)));
    let _e39 = (*v);
    let _e42 = (*v);
    (*v) = (_e42 | (_e39 >> bitcast<u32>(8u)));
    let _e44 = (*v);
    let _e47 = (*v);
    (*v) = (_e47 | (_e44 >> bitcast<u32>(16u)));
    let _e49 = (*v);
    (*v) = (_e49 + bitcast<u32>(1i));
    let _e52 = (*v);
    return _e52;
}

fn blelloch_exclusive_scan_u0028_u1_u003b_u1_u003b(tid: ptr<function, u32>, n: ptr<function, u32>) -> u32 {
    var offset: u32;
    var d: u32;
    var ai: u32;
    var bi: u32;
    var total: u32;
    var d_1: u32;
    var ai_1: u32;
    var bi_1: u32;
    var temp: u32;

    offset = 1u;
    let _e29 = (*n);
    d = (_e29 >> bitcast<u32>(1u));
    loop {
        let _e32 = d;
        if (_e32 > 0u) {
            workgroupBarrier();
            let _e34 = (*tid);
            let _e35 = d;
            if (_e34 < _e35) {
                let _e37 = offset;
                let _e38 = (*tid);
                ai = ((_e37 * ((2u * _e38) + 1u)) - 1u);
                let _e43 = offset;
                let _e44 = (*tid);
                bi = ((_e43 * ((2u * _e44) + 2u)) - 1u);
                let _e49 = bi;
                let _e50 = ai;
                let _e52 = s_data[_e50];
                let _e54 = s_data[_e49];
                s_data[_e49] = (_e54 + _e52);
            }
            let _e57 = offset;
            offset = (_e57 << bitcast<u32>(1u));
            continue;
        } else {
            break;
        }
        continuing {
            let _e60 = d;
            d = (_e60 >> bitcast<u32>(1u));
        }
    }
    workgroupBarrier();
    let _e63 = (*n);
    let _e66 = s_data[(_e63 - 1u)];
    total = _e66;
    workgroupBarrier();
    let _e67 = (*tid);
    if (_e67 == 0u) {
        let _e69 = (*n);
        s_data[(_e69 - 1u)] = 0u;
    }
    d_1 = 1u;
    loop {
        let _e72 = d_1;
        let _e73 = (*n);
        if (_e72 < _e73) {
            let _e75 = offset;
            offset = (_e75 >> bitcast<u32>(1u));
            workgroupBarrier();
            let _e78 = (*tid);
            let _e79 = d_1;
            if (_e78 < _e79) {
                let _e81 = offset;
                let _e82 = (*tid);
                ai_1 = ((_e81 * ((2u * _e82) + 1u)) - 1u);
                let _e87 = offset;
                let _e88 = (*tid);
                bi_1 = ((_e87 * ((2u * _e88) + 2u)) - 1u);
                let _e93 = ai_1;
                let _e95 = s_data[_e93];
                temp = _e95;
                let _e96 = ai_1;
                let _e97 = bi_1;
                let _e99 = s_data[_e97];
                s_data[_e96] = _e99;
                let _e101 = bi_1;
                let _e102 = temp;
                let _e104 = s_data[_e101];
                s_data[_e101] = (_e104 + _e102);
            }
            continue;
        } else {
            break;
        }
        continuing {
            let _e107 = d_1;
            d_1 = (_e107 << bitcast<u32>(1u));
        }
    }
    workgroupBarrier();
    let _e110 = total;
    return _e110;
}

fn main_1() {
    var tid_1: u32;
    var group_id: u32;
    var base: u32;
    var idx0_: u32;
    var idx1_: u32;
    var local: u32;
    var local_1: u32;
    var total_1: u32;
    var param: u32;
    var param_1: u32;
    var num_partitions: u32;
    var n_1: u32;
    var param_2: u32;
    var local_2: u32;
    var param_3: u32;
    var param_4: u32;
    var base_1: u32;
    var spine_val: u32;
    var idx0_1: u32;
    var idx1_1: u32;

    let _e39 = gl_LocalInvocationID_1[0u];
    tid_1 = _e39;
    let _e41 = gl_WorkGroupID_1[0u];
    group_id = _e41;
    let _e43 = unnamed.pass_id;
    if (_e43 == 0u) {
        let _e45 = group_id;
        base = (_e45 * 2048u);
        let _e47 = base;
        let _e48 = tid_1;
        idx0_ = (_e47 + _e48);
        let _e50 = base;
        let _e51 = tid_1;
        idx1_ = ((_e50 + _e51) + 1024u);
        let _e54 = tid_1;
        let _e55 = idx0_;
        let _e57 = unnamed.num_entries;
        if (_e55 < _e57) {
            let _e59 = idx0_;
            let _e62 = unnamed_1.histograms[_e59];
            local = _e62;
        } else {
            local = 0u;
        }
        let _e63 = local;
        s_data[_e54] = _e63;
        let _e65 = tid_1;
        let _e67 = idx1_;
        let _e69 = unnamed.num_entries;
        if (_e67 < _e69) {
            let _e71 = idx1_;
            let _e74 = unnamed_1.histograms[_e71];
            local_1 = _e74;
        } else {
            local_1 = 0u;
        }
        let _e75 = local_1;
        s_data[(_e65 + 1024u)] = _e75;
        workgroupBarrier();
        let _e77 = tid_1;
        param = _e77;
        param_1 = 2048u;
        let _e78 = blelloch_exclusive_scan_u0028_u1_u003b_u1_u003b((&param), (&param_1));
        total_1 = _e78;
        let _e79 = tid_1;
        if (_e79 == 0u) {
            let _e81 = group_id;
            let _e82 = total_1;
            unnamed_2.partition_sums[_e81] = _e82;
        }
        let _e85 = idx0_;
        let _e87 = unnamed.num_entries;
        if (_e85 < _e87) {
            let _e89 = idx0_;
            let _e90 = tid_1;
            let _e92 = s_data[_e90];
            unnamed_1.histograms[_e89] = _e92;
        }
        let _e95 = idx1_;
        let _e97 = unnamed.num_entries;
        if (_e95 < _e97) {
            let _e99 = idx1_;
            let _e100 = tid_1;
            let _e103 = s_data[(_e100 + 1024u)];
            unnamed_1.histograms[_e99] = _e103;
        }
    } else {
        let _e107 = unnamed.pass_id;
        if (_e107 == 1u) {
            let _e110 = unnamed.num_entries;
            num_partitions = _e110;
            let _e111 = num_partitions;
            param_2 = _e111;
            let _e112 = next_pow2_u0028_u1_u003b((&param_2));
            n_1 = _e112;
            let _e113 = tid_1;
            let _e114 = num_partitions;
            if (_e113 < _e114) {
                let _e116 = tid_1;
                let _e117 = tid_1;
                let _e120 = unnamed_2.partition_sums[_e117];
                s_data[_e116] = _e120;
            } else {
                let _e122 = tid_1;
                let _e123 = n_1;
                if (_e122 < _e123) {
                    let _e125 = tid_1;
                    s_data[_e125] = 0u;
                }
            }
            let _e127 = tid_1;
            let _e129 = n_1;
            if ((_e127 + 1024u) < _e129) {
                let _e131 = tid_1;
                let _e133 = tid_1;
                let _e135 = num_partitions;
                if ((_e133 + 1024u) < _e135) {
                    let _e137 = tid_1;
                    let _e141 = unnamed_2.partition_sums[(_e137 + 1024u)];
                    local_2 = _e141;
                } else {
                    local_2 = 0u;
                }
                let _e142 = local_2;
                s_data[(_e131 + 1024u)] = _e142;
            }
            workgroupBarrier();
            let _e144 = tid_1;
            param_3 = _e144;
            let _e145 = n_1;
            param_4 = _e145;
            let _e146 = blelloch_exclusive_scan_u0028_u1_u003b_u1_u003b((&param_3), (&param_4));
            let _e147 = tid_1;
            let _e148 = num_partitions;
            if (_e147 < _e148) {
                let _e150 = tid_1;
                let _e151 = tid_1;
                let _e153 = s_data[_e151];
                unnamed_2.partition_sums[_e150] = _e153;
            }
        } else {
            let _e156 = group_id;
            base_1 = (_e156 * 2048u);
            let _e158 = group_id;
            let _e161 = unnamed_2.partition_sums[_e158];
            spine_val = _e161;
            let _e162 = spine_val;
            if (_e162 != 0u) {
                let _e164 = base_1;
                let _e165 = tid_1;
                idx0_1 = (_e164 + _e165);
                let _e167 = base_1;
                let _e168 = tid_1;
                idx1_1 = ((_e167 + _e168) + 1024u);
                let _e171 = idx0_1;
                let _e173 = unnamed.num_entries;
                if (_e171 < _e173) {
                    let _e175 = idx0_1;
                    let _e176 = spine_val;
                    let _e179 = unnamed_1.histograms[_e175];
                    unnamed_1.histograms[_e175] = (_e179 + _e176);
                }
                let _e183 = idx1_1;
                let _e185 = unnamed.num_entries;
                if (_e183 < _e185) {
                    let _e187 = idx1_1;
                    let _e188 = spine_val;
                    let _e191 = unnamed_1.histograms[_e187];
                    unnamed_1.histograms[_e187] = (_e191 + _e188);
                }
            }
        }
    }
    return;
}

@compute @workgroup_size(1024, 1, 1) 
fn main(@builtin(local_invocation_id) gl_LocalInvocationID: vec3<u32>, @builtin(workgroup_id) gl_WorkGroupID: vec3<u32>) {
    gl_LocalInvocationID_1 = gl_LocalInvocationID;
    gl_WorkGroupID_1 = gl_WorkGroupID;
    main_1();
}
