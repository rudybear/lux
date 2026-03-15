struct params {
    brightness: f32,
}

@group(1) @binding(0) 
var<uniform> global: params;
@group(0) @binding(0) 
var output_img: texture_storage_2d<rgba8unorm,write>;
var<private> local_invocation_id_1: vec3<u32>;
var<private> local_invocation_index_1: u32;
var<private> workgroup_id_1: vec3<u32>;
var<private> num_workgroups_1: vec3<u32>;
var<private> global_invocation_id_1: vec3<u32>;

fn main_1() {
    let _e10 = global_invocation_id_1;
    let _e14 = global.brightness;
    let _e16 = global.brightness;
    let _e18 = global.brightness;
    textureStore(output_img, vec2<i32>(vec2<f32>(_e10.xy)), vec4<f32>(_e14, _e16, _e18, 1f));
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
