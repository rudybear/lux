@group(0) @binding(0) 
var output_img: texture_storage_2d<rgba8unorm,write>;
var<private> local_invocation_id_1: vec3<u32>;
var<private> local_invocation_index_1: u32;
var<private> workgroup_id_1: vec3<u32>;
var<private> num_workgroups_1: vec3<u32>;
var<private> global_invocation_id_1: vec3<u32>;

fn main_1() {
    let _e15 = global_invocation_id_1;
    let _e22 = ((f32(_e15.x) / 512f) - 0.5f);
    let _e23 = ((f32(_e15.y) / 512f) - 0.5f);
    let _e27 = sqrt(((_e22 * _e22) + (_e23 * _e23)));
    let _e35 = (((atan2(_e23, _e22) / 6.283f) + 0.5f) * 6.283f);
    let _e53 = clamp((((sin((_e27 * 40f)) * 0.5f) + 0.5f) * (1f - (_e27 * 1.5f))), 0f, 1f);
    textureStore(output_img, vec2<i32>(vec2<f32>(_e15.xy)), vec4<f32>((clamp(((sin(_e35) * 0.5f) + 0.5f), 0f, 1f) * _e53), (clamp(((sin((_e35 + 2.094f)) * 0.5f) + 0.5f), 0f, 1f) * _e53), (clamp(((sin((_e35 + 4.189f)) * 0.5f) + 0.5f), 0f, 1f) * _e53), 1f));
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
