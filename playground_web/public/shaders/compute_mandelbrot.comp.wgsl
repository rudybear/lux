@group(0) @binding(0) 
var output_img: texture_storage_2d<rgba8unorm,write>;
var<private> local_invocation_id_1: vec3<u32>;
var<private> local_invocation_index_1: u32;
var<private> workgroup_id_1: vec3<u32>;
var<private> num_workgroups_1: vec3<u32>;
var<private> global_invocation_id_1: vec3<u32>;

fn main_1() {
    var local: f32;
    var local_1: f32;
    var local_2: i32;

    let _e22 = global_invocation_id_1;
    let _e31 = (-0.5f + ((((f32(_e22.x) / 512f) - 0.5f) * 3f) / 1f));
    let _e35 = ((((f32(_e22.y) / 512f) - 0.5f) * 3f) / 1f);
    local = _e31;
    local_1 = _e35;
    local_2 = 0i;
    loop {
        let _e36 = local_2;
        if (_e36 < 64i) {
            let _e38 = local;
            let _e39 = local;
            let _e40 = (_e38 * _e39);
            let _e41 = local_1;
            let _e42 = local_1;
            let _e43 = (_e41 * _e42);
            if ((_e40 + _e43) > 4f) {
                break;
            }
            let _e48 = local;
            let _e50 = local_1;
            local_1 = (((2f * _e48) * _e50) + _e35);
            local = ((_e40 - _e43) + _e31);
            continue;
        } else {
            break;
        }
        continuing {
            let _e53 = local_2;
            local_2 = (_e53 + 1i);
        }
    }
    let _e55 = local;
    let _e56 = local;
    let _e58 = local_1;
    let _e59 = local_1;
    let _e61 = ((_e55 * _e56) + (_e58 * _e59));
    let _e64 = clamp((sqrt(_e61) / 4f), 0f, 1f);
    let _e65 = step(4f, _e61);
    textureStore(output_img, vec2<i32>(vec2<f32>(_e22.xy)), vec4<f32>((_e65 * smoothstep(0f, 0.8f, _e64)), ((_e65 * _e64) * _e64), (_e65 * clamp(((_e64 * 0.5f) + 0.3f), 0f, 1f)), 1f));
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
