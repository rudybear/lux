//! Two-pass deferred shading renderer.
//!
//! Pass 1 (G-buffer): Renders scene geometry into 3 RGBA16F color attachments
//! (albedo+metallic, oct_normal+roughness, emission+occlusion) plus D32 depth.
//!
//! Pass 2 (Lighting): Fullscreen triangle reads G-buffer textures, evaluates
//! lighting, and writes the final color to an RGBA8 output image.

use ash::vk;
use log::info;
use std::collections::HashMap;
use std::path::Path;

use crate::camera::DefaultCamera;
use crate::reflected_pipeline::{self, ReflectionData};
use crate::scene_manager::{self, GpuBuffer, GpuImage, IblAssets};
use crate::spv_loader;
use crate::vulkan_context::VulkanContext;

// ===========================================================================
// Local helpers wrapping scene_manager functions
// ===========================================================================

/// Delegate to scene_manager::create_texture_image with FRAGMENT_SHADER dst_stage.
fn create_texture_image(
    ctx: &mut VulkanContext,
    width: u32,
    height: u32,
    pixels: &[u8],
    name: &str,
) -> Result<GpuImage, String> {
    scene_manager::create_texture_image(ctx, width, height, pixels, name, vk::PipelineStageFlags::FRAGMENT_SHADER)
}

/// Delegate to scene_manager::load_ibl_assets with FRAGMENT_SHADER dst_stage.
fn load_ibl_assets(ctx: &mut VulkanContext, ibl_name: &str) -> IblAssets {
    scene_manager::load_ibl_assets(ctx, vk::PipelineStageFlags::FRAGMENT_SHADER, ibl_name)
}

/// Upload per-material textures from a glTF scene, returning a Vec of texture maps.
fn upload_per_material_textures(
    ctx: &mut VulkanContext,
    scene: &crate::gltf_loader::GltfScene,
) -> Result<Vec<HashMap<String, GpuImage>>, String> {
    let mut result = Vec::new();
    for (mat_idx, mat) in scene.materials.iter().enumerate() {
        let mut tex_map: HashMap<String, GpuImage> = HashMap::new();
        let tex_slots: Vec<(&str, Option<&crate::gltf_loader::TextureImage>)> = vec![
            ("base_color_tex", mat.base_color_image.as_ref()),
            ("normal_tex", mat.normal_image.as_ref()),
            ("metallic_roughness_tex", mat.metallic_roughness_image.as_ref()),
            ("occlusion_tex", mat.occlusion_image.as_ref()),
            ("emissive_tex", mat.emissive_image.as_ref()),
            ("clearcoat_tex", mat.clearcoat_image.as_ref()),
            ("clearcoat_roughness_tex", mat.clearcoat_roughness_image.as_ref()),
            ("sheen_color_tex", mat.sheen_color_image.as_ref()),
            ("transmission_tex", mat.transmission_image.as_ref()),
        ];
        for (name, img_opt) in tex_slots {
            if let Some(tex_data) = img_opt {
                let label = format!("deferred_{}[{}]", name, mat_idx);
                let gpu_img = create_texture_image(ctx, tex_data.width, tex_data.height, &tex_data.pixels, &label)?;
                tex_map.insert(name.to_string(), gpu_img);
            }
        }
        result.push(tex_map);
    }
    Ok(result)
}

// ===========================================================================
// Material UBO (144 bytes, std140 layout — matches mesh_renderer/C++ version)
// ===========================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct MaterialUboData {
    base_color_factor: [f32; 4],          // offset  0, size 16
    emissive_factor: [f32; 3],            // offset 16, size 12
    metallic_factor: f32,                 // offset 28, size  4
    roughness_factor: f32,                // offset 32, size  4
    emissive_strength: f32,               // offset 36, size  4
    ior: f32,                             // offset 40, size  4
    clearcoat_factor: f32,                // offset 44, size  4
    clearcoat_roughness_factor: f32,      // offset 48, size  4
    sheen_roughness_factor: f32,          // offset 52, size  4
    transmission_factor: f32,             // offset 56, size  4
    _pad_before_sheen: f32,              // offset 60, size  4
    sheen_color_factor: [f32; 3],         // offset 64, size 12
    _pad0: f32,                           // offset 76, size  4
    // KHR_texture_transform
    base_color_uv_st: [f32; 4],          // offset 80
    normal_uv_st: [f32; 4],              // offset 96
    mr_uv_st: [f32; 4],                  // offset 112
    base_color_uv_rot: f32,              // offset 128
    normal_uv_rot: f32,                  // offset 132
    mr_uv_rot: f32,                      // offset 136
    _pad1: f32,                          // offset 140
}
unsafe impl bytemuck::Pod for MaterialUboData {}
unsafe impl bytemuck::Zeroable for MaterialUboData {}

impl MaterialUboData {
    fn default_values() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            emissive_factor: [0.0, 0.0, 0.0],
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            emissive_strength: 1.0,
            ior: 1.5,
            clearcoat_factor: 0.0,
            clearcoat_roughness_factor: 0.0,
            sheen_roughness_factor: 0.0,
            transmission_factor: 0.0,
            _pad_before_sheen: 0.0,
            sheen_color_factor: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            base_color_uv_st: [0.0, 0.0, 1.0, 1.0],
            normal_uv_st: [0.0, 0.0, 1.0, 1.0],
            mr_uv_st: [0.0, 0.0, 1.0, 1.0],
            base_color_uv_rot: 0.0,
            normal_uv_rot: 0.0,
            mr_uv_rot: 0.0,
            _pad1: 0.0,
        }
    }

    fn from_gltf_material(mat: &crate::gltf_loader::GltfMaterial) -> Self {
        Self {
            base_color_factor: mat.base_color,
            metallic_factor: mat.metallic,
            roughness_factor: mat.roughness,
            emissive_factor: mat.emissive,
            emissive_strength: mat.emissive_strength,
            ior: mat.ior,
            clearcoat_factor: mat.clearcoat_factor,
            clearcoat_roughness_factor: mat.clearcoat_roughness_factor,
            sheen_color_factor: mat.sheen_color_factor,
            sheen_roughness_factor: mat.sheen_roughness_factor,
            transmission_factor: mat.transmission_factor,
            _pad_before_sheen: 0.0,
            _pad0: 0.0,
            base_color_uv_st: [
                mat.base_color_uv_xform.offset[0], mat.base_color_uv_xform.offset[1],
                mat.base_color_uv_xform.scale[0], mat.base_color_uv_xform.scale[1],
            ],
            normal_uv_st: [
                mat.normal_uv_xform.offset[0], mat.normal_uv_xform.offset[1],
                mat.normal_uv_xform.scale[0], mat.normal_uv_xform.scale[1],
            ],
            mr_uv_st: [
                mat.metallic_roughness_uv_xform.offset[0], mat.metallic_roughness_uv_xform.offset[1],
                mat.metallic_roughness_uv_xform.scale[0], mat.metallic_roughness_uv_xform.scale[1],
            ],
            base_color_uv_rot: mat.base_color_uv_xform.rotation,
            normal_uv_rot: mat.normal_uv_xform.rotation,
            mr_uv_rot: mat.metallic_roughness_uv_xform.rotation,
            _pad1: 0.0,
        }
    }
}

// ===========================================================================
// DeferredCamera UBO (80 bytes, std140)
// ===========================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct DeferredCameraData {
    inv_view_proj: [f32; 16], // offset  0, size 64 (mat4)
    view_pos: [f32; 3],      // offset 64, size 12
    _pad: f32,                // offset 76, size  4
}
unsafe impl bytemuck::Pod for DeferredCameraData {}
unsafe impl bytemuck::Zeroable for DeferredCameraData {}

// ===========================================================================
// MVP UBO (192 bytes = 3 x mat4)
// ===========================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct MvpData {
    model: [f32; 16],
    view: [f32; 16],
    projection: [f32; 16],
}
unsafe impl bytemuck::Pod for MvpData {}
unsafe impl bytemuck::Zeroable for MvpData {}

// ===========================================================================
// Light UBO (32 bytes)
// ===========================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct LightData {
    light_dir: [f32; 3],
    _pad0: f32,
    view_pos: [f32; 3],
    _pad1: f32,
}
unsafe impl bytemuck::Pod for LightData {}
unsafe impl bytemuck::Zeroable for LightData {}

// ===========================================================================
// SceneLight UBO (16 bytes)
// ===========================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct SceneLightUboData {
    view_pos: [f32; 3],
    light_count: i32,
}
unsafe impl bytemuck::Pod for SceneLightUboData {}
unsafe impl bytemuck::Zeroable for SceneLightUboData {}

// ===========================================================================
// G-buffer texture names that the lighting pass shader may reference
// ===========================================================================

fn is_gbuffer_binding(name: &str) -> bool {
    matches!(
        name,
        "gbuf_tex0" | "gbuffer_albedo" | "gbuf_rt0"
            | "gbuf_tex1" | "gbuffer_normal" | "gbuf_rt1"
            | "gbuf_tex2" | "gbuffer_material" | "gbuf_rt2"
            | "gbuf_depth" | "gbuffer_depth"
    )
}

// ===========================================================================
// Deferred renderer
// ===========================================================================

pub struct DeferredRenderer {
    // Dimensions
    width: u32,
    height: u32,

    // --- G-buffer resources ---
    gbuf_rt0: GpuImage,
    gbuf_rt1: GpuImage,
    gbuf_rt2: GpuImage,
    gbuf_depth: GpuImage,
    gbuf_sampler: vk::Sampler,
    gbuf_render_pass: vk::RenderPass,
    gbuf_framebuffer: vk::Framebuffer,

    // G-buffer pipeline
    gbuf_vert_module: vk::ShaderModule,
    gbuf_frag_module: vk::ShaderModule,
    gbuf_pipeline_layout: vk::PipelineLayout,
    gbuf_pipeline: vk::Pipeline,

    // G-buffer descriptor resources
    gbuf_set_layouts: HashMap<u32, vk::DescriptorSetLayout>,
    gbuf_desc_pool: vk::DescriptorPool,
    gbuf_vert_desc_set: vk::DescriptorSet,
    gbuf_per_material_desc_sets: Vec<vk::DescriptorSet>,
    gbuf_per_material_buffers: Vec<GpuBuffer>,

    // --- Lighting pass resources ---
    light_output_image: GpuImage,
    light_render_pass: vk::RenderPass,
    light_framebuffer: vk::Framebuffer,

    // Lighting pipeline
    light_vert_module: vk::ShaderModule,
    light_frag_module: vk::ShaderModule,
    light_pipeline_layout: vk::PipelineLayout,
    light_pipeline: vk::Pipeline,

    // Lighting descriptor resources
    light_set_layouts: HashMap<u32, vk::DescriptorSetLayout>,
    light_desc_pool: vk::DescriptorPool,
    light_desc_sets: HashMap<u32, vk::DescriptorSet>,

    // --- Uniform buffers ---
    mvp_buffer: GpuBuffer,
    light_buffer: GpuBuffer,
    deferred_camera_buffer: GpuBuffer,

    // Multi-light resources
    lights_ssbo: Option<GpuBuffer>,
    scene_light_ubo: Option<GpuBuffer>,
    has_multi_light: bool,

    // Geometry
    vbo: GpuBuffer,
    ibo: GpuBuffer,
    num_indices: u32,
    draw_ranges: Vec<crate::gltf_loader::DrawRange>,

    // Per-material textures
    per_material_textures: Vec<HashMap<String, GpuImage>>,
    texture_sampler: vk::Sampler,
    default_white_texture: GpuImage,
    default_black_texture: GpuImage,
    default_normal_texture: GpuImage,

    // IBL
    ibl_assets: IblAssets,

    // Push constants (G-buffer pass)
    push_constant_data: Vec<u8>,
    push_constant_stage_flags: vk::ShaderStageFlags,
    push_constant_size: u32,

    // Auto-camera state
    has_scene_bounds: bool,
    auto_eye: glam::Vec3,
    auto_target: glam::Vec3,
    auto_up: glam::Vec3,
    auto_far: f32,
}

impl DeferredRenderer {
    /// Create a new deferred renderer.
    ///
    /// Loads 4 shader stages (gbuf.vert, gbuf.frag, light.vert, light.frag),
    /// sets up G-buffer and lighting render passes, pipelines, and descriptors.
    pub fn new(
        ctx: &mut VulkanContext,
        scene_source: &str,
        pipeline_base: &str,
        width: u32,
        height: u32,
        ibl_name: &str,
        demo_lights: bool,
        sponza_lights: bool,
    ) -> Result<Self, String> {
        let device = ctx.device.clone();
        let aspect = width as f32 / height as f32;

        // ---------------------------------------------------------------
        // 1. Load glTF scene
        // ---------------------------------------------------------------
        let is_gltf = scene_source.ends_with(".glb") || scene_source.ends_with(".gltf");
        let mut gltf_scene = if is_gltf {
            Some(crate::gltf_loader::load_gltf(Path::new(scene_source))?)
        } else {
            None
        };

        // ---------------------------------------------------------------
        // 2. Scene manager: lights
        // ---------------------------------------------------------------
        let mut scene_mgr = scene_manager::SceneManager::new();
        if let Some(ref scene) = gltf_scene {
            scene_mgr.populate_lights_from_gltf(&scene.lights);
        }
        if demo_lights {
            let lights = vec![
                scene_manager::SceneLight {
                    light_type: 0,
                    direction: glam::Vec3::new(0.6, -0.8, 0.4).normalize(),
                    color: glam::Vec3::new(1.0, 0.95, 0.85),
                    intensity: 1.2,
                    casts_shadow: true,
                    ..Default::default()
                },
                scene_manager::SceneLight {
                    light_type: 1,
                    position: glam::Vec3::new(-2.0, 1.0, 1.0),
                    color: glam::Vec3::new(0.3, 0.5, 1.0),
                    intensity: 3.0,
                    range: 10.0,
                    ..Default::default()
                },
                scene_manager::SceneLight {
                    light_type: 2,
                    position: glam::Vec3::new(2.5, 2.0, 1.5),
                    direction: glam::Vec3::new(-1.0, -1.0, -0.5).normalize(),
                    color: glam::Vec3::new(1.0, 0.3, 0.2),
                    intensity: 5.0,
                    range: 15.0,
                    inner_cone_angle: 0.2,
                    outer_cone_angle: 0.5,
                    casts_shadow: true,
                    ..Default::default()
                },
            ];
            scene_mgr.override_lights(lights);
        } else if sponza_lights {
            let lights = vec![
                scene_manager::SceneLight {
                    light_type: 0,
                    direction: glam::Vec3::new(0.5, -0.7, 0.3).normalize(),
                    color: glam::Vec3::new(1.0, 0.95, 0.85),
                    intensity: 5.0,
                    casts_shadow: true,
                    ..Default::default()
                },
                scene_manager::SceneLight {
                    light_type: 2,
                    position: glam::Vec3::new(0.0, 600.0, 0.0),
                    direction: glam::Vec3::new(0.0, -1.0, 0.0),
                    color: glam::Vec3::new(1.0, 0.7, 0.3),
                    intensity: 500000.0,
                    range: 3000.0,
                    inner_cone_angle: 0.3,
                    outer_cone_angle: 0.7,
                    casts_shadow: true,
                    ..Default::default()
                },
                scene_manager::SceneLight {
                    light_type: 1,
                    position: glam::Vec3::new(-500.0, 400.0, -300.0),
                    color: glam::Vec3::new(0.3, 0.5, 1.0),
                    intensity: 100000.0,
                    range: 2000.0,
                    ..Default::default()
                },
            ];
            scene_mgr.override_lights(lights);
        }

        // ---------------------------------------------------------------
        // 3. Auto-camera from scene bounds
        // ---------------------------------------------------------------
        let mut has_scene_bounds = false;
        let mut auto_eye = DefaultCamera::EYE;
        let mut auto_target = DefaultCamera::TARGET;
        let mut auto_up = DefaultCamera::UP;
        let mut auto_far = DefaultCamera::FAR;

        let view = crate::camera::look_at(auto_eye, auto_target, auto_up);
        let proj = crate::camera::perspective(DefaultCamera::FOV_Y_DEG.to_radians(), aspect, DefaultCamera::NEAR, auto_far);

        // ---------------------------------------------------------------
        // 4. Load 4 SPIR-V shader modules
        // ---------------------------------------------------------------
        let gbuf_vert_path = format!("{}.gbuf.vert.spv", pipeline_base);
        let gbuf_frag_path = format!("{}.gbuf.frag.spv", pipeline_base);
        let light_vert_path = format!("{}.light.vert.spv", pipeline_base);
        let light_frag_path = format!("{}.light.frag.spv", pipeline_base);

        let gbuf_vert_code = spv_loader::load_spirv(Path::new(&gbuf_vert_path))?;
        let gbuf_frag_code = spv_loader::load_spirv(Path::new(&gbuf_frag_path))?;
        let light_vert_code = spv_loader::load_spirv(Path::new(&light_vert_path))?;
        let light_frag_code = spv_loader::load_spirv(Path::new(&light_frag_path))?;

        let gbuf_vert_module = spv_loader::create_shader_module(&device, &gbuf_vert_code)?;
        let gbuf_frag_module = spv_loader::create_shader_module(&device, &gbuf_frag_code)?;
        let light_vert_module = spv_loader::create_shader_module(&device, &light_vert_code)?;
        let light_frag_module = spv_loader::create_shader_module(&device, &light_frag_code)?;

        info!("Loaded 4 deferred shader modules");

        // ---------------------------------------------------------------
        // 5. Load reflection JSON
        // ---------------------------------------------------------------
        let gbuf_vert_json = format!("{}.gbuf.vert.json", pipeline_base);
        let gbuf_frag_json = format!("{}.gbuf.frag.json", pipeline_base);
        let light_vert_json = format!("{}.light.vert.json", pipeline_base);
        let light_frag_json = format!("{}.light.frag.json", pipeline_base);

        let gbuf_vert_refl = reflected_pipeline::load_reflection(Path::new(&gbuf_vert_json))?;
        let gbuf_frag_refl = reflected_pipeline::load_reflection(Path::new(&gbuf_frag_json))?;
        let light_vert_refl = reflected_pipeline::load_reflection(Path::new(&light_vert_json))?;
        let light_frag_refl = reflected_pipeline::load_reflection(Path::new(&light_frag_json))?;

        // ---------------------------------------------------------------
        // 6. Upload geometry and compute auto-camera
        // ---------------------------------------------------------------
        let (vbo, ibo, num_indices, draw_ranges) = if let Some(ref mut scene) = gltf_scene {
            if scene.meshes.is_empty() {
                return Err(format!("No meshes found in glTF file: {}", scene_source));
            }
            let draw_items = crate::gltf_loader::flatten_scene(scene);
            if draw_items.is_empty() {
                return Err(format!("No draw items in glTF scene: {}", scene_source));
            }

            // Manually merge draw items into vertex + index buffers (same pattern as raster_renderer)
            let mut vdata: Vec<f32> = Vec::new();
            let mut all_indices: Vec<u32> = Vec::new();
            let mut vertex_offset: u32 = 0;
            let mut vertex_positions: Vec<[f32; 3]> = Vec::new();
            let mut merged_ranges: Vec<crate::gltf_loader::DrawRange> = Vec::new();
            let mut index_offset: u32 = 0;

            for item in &draw_items {
                let gmesh = &scene.meshes[item.mesh_index];
                let world = item.world_transform;
                let upper3x3 = glam::Mat3::from_mat4(world);
                let normal_mat = upper3x3.inverse().transpose();

                for gv in &gmesh.vertices {
                    let pos = world.transform_point3(glam::Vec3::from(gv.position));
                    let norm = (normal_mat * glam::Vec3::from(gv.normal)).normalize();
                    vertex_positions.push([pos.x, pos.y, pos.z]);
                    vdata.push(pos.x); vdata.push(pos.y); vdata.push(pos.z);
                    vdata.push(norm.x); vdata.push(norm.y); vdata.push(norm.z);
                    vdata.push(gv.uv[0]); vdata.push(gv.uv[1]);

                    // Tangent (4 floats) — always emit for deferred PBR pipeline (stride=48)
                    if gmesh.has_tangents {
                        let tang3 = (upper3x3 * glam::Vec3::new(gv.tangent[0], gv.tangent[1], gv.tangent[2])).normalize();
                        vdata.push(tang3.x); vdata.push(tang3.y); vdata.push(tang3.z);
                        vdata.push(gv.tangent[3]);
                    } else {
                        vdata.push(1.0); vdata.push(0.0); vdata.push(0.0); vdata.push(1.0);
                    }
                }

                for &idx in &gmesh.indices {
                    all_indices.push(idx + vertex_offset);
                }
                vertex_offset += gmesh.vertices.len() as u32;

                merged_ranges.push(crate::gltf_loader::DrawRange {
                    index_offset,
                    index_count: gmesh.indices.len() as u32,
                    material_index: item.material_index,
                });
                index_offset += gmesh.indices.len() as u32;
            }

            has_scene_bounds = true;
            let (eye, target, up, far) = scene_manager::compute_auto_camera_from_draw_items(
                &vertex_positions, &draw_items,
            );
            auto_eye = eye;
            auto_target = target;
            auto_up = up;
            auto_far = far;

            let vertex_bytes: &[u8] = bytemuck::cast_slice(&vdata);
            let vbo = scene_manager::create_buffer_with_data(
                &device, ctx.allocator_mut(),
                vertex_bytes,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                "deferred_vbo",
            )?;
            let ibo = scene_manager::create_buffer_with_data(
                &device, ctx.allocator_mut(),
                bytemuck::cast_slice(&all_indices),
                vk::BufferUsageFlags::INDEX_BUFFER,
                "deferred_ibo",
            )?;
            (vbo, ibo, all_indices.len() as u32, merged_ranges)
        } else {
            // Fallback: unit sphere
            let (vertices, indices) = crate::scene::generate_sphere(64, 64);
            let vbo = scene_manager::create_buffer_with_data(
                &device, ctx.allocator_mut(),
                bytemuck::cast_slice(&vertices),
                vk::BufferUsageFlags::VERTEX_BUFFER,
                "deferred_vbo",
            )?;
            let ibo = scene_manager::create_buffer_with_data(
                &device, ctx.allocator_mut(),
                bytemuck::cast_slice(&indices),
                vk::BufferUsageFlags::INDEX_BUFFER,
                "deferred_ibo",
            )?;
            (vbo, ibo, indices.len() as u32, Vec::new())
        };

        // ---------------------------------------------------------------
        // 7. Upload per-material textures + create sampler + defaults
        // ---------------------------------------------------------------
        let per_material_textures = if let Some(ref scene) = gltf_scene {
            upload_per_material_textures(ctx, scene)?
        } else {
            Vec::new()
        };

        // Texture sampler for material textures (LINEAR, REPEAT)
        let texture_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .min_lod(0.0)
                    .max_lod(0.0),
                None,
            ).map_err(|e| format!("Failed to create texture sampler: {:?}", e))?
        };

        // Default fallback textures
        let default_white_texture = create_texture_image(ctx, 1, 1, &[255u8, 255, 255, 255], "deferred_default_white")?;
        let default_black_texture = create_texture_image(ctx, 1, 1, &[0u8, 0, 0, 255], "deferred_default_black")?;
        let default_normal_texture = create_texture_image(ctx, 1, 1, &[128u8, 128, 255, 255], "deferred_default_normal")?;

        // ---------------------------------------------------------------
        // 8. Load IBL textures
        // ---------------------------------------------------------------
        let ibl_assets = load_ibl_assets(ctx, ibl_name);

        // ---------------------------------------------------------------
        // 9. Create MVP UBO
        // ---------------------------------------------------------------
        let mvp_data = MvpData {
            model: glam::Mat4::IDENTITY.to_cols_array(),
            view: view.to_cols_array(),
            projection: proj.to_cols_array(),
        };
        let mvp_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            bytemuck::bytes_of(&mvp_data),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "deferred_mvp",
        )?;

        // ---------------------------------------------------------------
        // 10. Create Light UBO (32 bytes)
        // ---------------------------------------------------------------
        let light_dir = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
        let light_data = LightData {
            light_dir: light_dir.into(),
            _pad0: 0.0,
            view_pos: auto_eye.into(),
            _pad1: 0.0,
        };
        let light_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            bytemuck::bytes_of(&light_data),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "deferred_light",
        )?;

        // ---------------------------------------------------------------
        // 11. Create DeferredCamera UBO (80 bytes)
        // ---------------------------------------------------------------
        let vp = proj * view;
        let inv_vp = vp.inverse();
        let cam_data = DeferredCameraData {
            inv_view_proj: inv_vp.to_cols_array(),
            view_pos: auto_eye.into(),
            _pad: 0.0,
        };
        let deferred_camera_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            bytemuck::bytes_of(&cam_data),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "deferred_camera",
        )?;

        // ---------------------------------------------------------------
        // 12. Detect multi-light and create SSBO + SceneLight UBO
        // ---------------------------------------------------------------
        let has_multi_light = Self::check_has_multi_light(&gbuf_frag_refl, &light_frag_refl);

        let (lights_ssbo, scene_light_ubo) = if has_multi_light {
            let light_buf = scene_mgr.pack_lights_buffer();
            let ssbo_data: Vec<u8> = if light_buf.is_empty() {
                vec![0u8; 64]
            } else {
                bytemuck::cast_slice(&light_buf).to_vec()
            };
            let ssbo = scene_manager::create_buffer_with_data(
                &device, ctx.allocator_mut(),
                &ssbo_data,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                "deferred_lights_ssbo",
            )?;

            let sl_data = SceneLightUboData {
                view_pos: auto_eye.into(),
                light_count: scene_mgr.lights.len() as i32,
            };
            let sl_ubo = scene_manager::create_buffer_with_data(
                &device, ctx.allocator_mut(),
                bytemuck::bytes_of(&sl_data),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                "deferred_scene_light",
            )?;

            info!("Multi-light SSBO created: {} light(s), {} bytes",
                scene_mgr.lights.len(), ssbo_data.len());

            (Some(ssbo), Some(sl_ubo))
        } else {
            (None, None)
        };

        // ---------------------------------------------------------------
        // 13. Create G-buffer images (3 color + depth)
        // ---------------------------------------------------------------
        let allocator = ctx.allocator_mut();

        let gbuf_rt0 = scene_manager::create_offscreen_image(
            &device, allocator, width, height,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "gbuf_rt0",
        )?;

        let gbuf_rt1 = scene_manager::create_offscreen_image(
            &device, allocator, width, height,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "gbuf_rt1",
        )?;

        let gbuf_rt2 = scene_manager::create_offscreen_image(
            &device, allocator, width, height,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "gbuf_rt2",
        )?;

        let gbuf_depth = scene_manager::create_offscreen_image(
            &device, allocator, width, height,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::DEPTH,
            "gbuf_depth",
        )?;

        // G-buffer sampler (linear filter for lighting pass reads)
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .min_lod(0.0)
            .max_lod(0.0);
        let gbuf_sampler = unsafe {
            device.create_sampler(&sampler_info, None)
                .map_err(|e| format!("Failed to create G-buffer sampler: {:?}", e))?
        };

        // ---------------------------------------------------------------
        // 14. Create G-buffer render pass
        // ---------------------------------------------------------------
        let gbuf_render_pass = unsafe {
            Self::create_gbuffer_render_pass(&device)?
        };

        // ---------------------------------------------------------------
        // 15. Create G-buffer framebuffer
        // ---------------------------------------------------------------
        let gbuf_framebuffer = unsafe {
            let attachments = [gbuf_rt0.view, gbuf_rt1.view, gbuf_rt2.view, gbuf_depth.view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(gbuf_render_pass)
                .attachments(&attachments)
                .width(width)
                .height(height)
                .layers(1);
            device.create_framebuffer(&fb_info, None)
                .map_err(|e| format!("Failed to create G-buffer framebuffer: {:?}", e))?
        };

        // ---------------------------------------------------------------
        // 16. Create G-buffer descriptor sets (reflection-driven)
        // ---------------------------------------------------------------
        let num_materials = if let Some(ref scene) = gltf_scene {
            scene.materials.len().max(1)
        } else {
            1
        };

        let gbuf_merged = reflected_pipeline::merge_descriptor_sets(&[&gbuf_vert_refl, &gbuf_frag_refl]);
        let gbuf_set_layouts = unsafe {
            reflected_pipeline::create_descriptor_set_layouts_from_merged(&device, &gbuf_merged)?
        };

        // Identify vertex and fragment set indices
        let (vert_set_idx, frag_set_idx) = Self::find_gbuf_set_indices(&gbuf_merged);

        // Compute pool sizes — fragment set multiplied by material count
        let (mut pool_sizes, _) = reflected_pipeline::compute_pool_sizes(&gbuf_merged);
        // Increase counts for per-material descriptors
        for ps in &mut pool_sizes {
            ps.descriptor_count *= num_materials as u32;
        }
        let max_sets = 1 + num_materials as u32;

        let gbuf_desc_pool = unsafe {
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(max_sets)
                .pool_sizes(&pool_sizes);
            device.create_descriptor_pool(&pool_info, None)
                .map_err(|e| format!("Failed to create G-buffer descriptor pool: {:?}", e))?
        };

        // Allocate vertex descriptor set (set 0)
        let gbuf_vert_desc_set = if let Some(&layout) = gbuf_set_layouts.get(&vert_set_idx) {
            unsafe {
                let layouts = [layout];
                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(gbuf_desc_pool)
                    .set_layouts(&layouts);
                device.allocate_descriptor_sets(&alloc_info)
                    .map_err(|e| format!("Failed to allocate G-buffer vertex desc set: {:?}", e))?[0]
            }
        } else {
            vk::DescriptorSet::null()
        };

        // Write vertex descriptor set
        Self::write_gbuf_vertex_descriptors(
            &device,
            gbuf_vert_desc_set,
            vert_set_idx,
            &gbuf_merged,
            &mvp_buffer,
            &light_buffer,
            has_multi_light,
            lights_ssbo.as_ref(),
            scene_light_ubo.as_ref(),
        );

        // Allocate per-material fragment descriptor sets
        let mut gbuf_per_material_desc_sets = Vec::with_capacity(num_materials);
        let mut gbuf_per_material_buffers = Vec::with_capacity(num_materials);

        if let Some(&frag_layout) = gbuf_set_layouts.get(&frag_set_idx) {
            for mi in 0..num_materials {
                // Create material UBO for this material
                let mat_data = if let Some(ref scene) = gltf_scene {
                    if mi < scene.materials.len() {
                        MaterialUboData::from_gltf_material(&scene.materials[mi])
                    } else {
                        MaterialUboData::default_values()
                    }
                } else {
                    MaterialUboData::default_values()
                };

                let mat_buf = scene_manager::create_buffer_with_data(
                    &device, ctx.allocator_mut(),
                    bytemuck::bytes_of(&mat_data),
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    &format!("deferred_material_{}", mi),
                )?;

                // Allocate descriptor set for this material
                let ds = unsafe {
                    let layouts = [frag_layout];
                    let alloc_info = vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(gbuf_desc_pool)
                        .set_layouts(&layouts);
                    device.allocate_descriptor_sets(&alloc_info)
                        .map_err(|e| format!("Failed to allocate G-buffer material desc set: {:?}", e))?[0]
                };

                // Write descriptors for this material
                Self::write_gbuf_material_descriptors(
                    &device,
                    ds,
                    frag_set_idx,
                    &gbuf_merged,
                    &mat_buf,
                    &light_buffer,
                    has_multi_light,
                    lights_ssbo.as_ref(),
                    scene_light_ubo.as_ref(),
                    &ibl_assets,
                    if mi < per_material_textures.len() { Some(&per_material_textures[mi]) } else { None },
                    texture_sampler,
                    &default_white_texture,
                    &default_black_texture,
                    &default_normal_texture,
                );

                gbuf_per_material_desc_sets.push(ds);
                gbuf_per_material_buffers.push(mat_buf);
            }
        }

        info!("G-buffer descriptors created: {} material(s)", num_materials);

        // ---------------------------------------------------------------
        // 17. Create G-buffer pipeline
        // ---------------------------------------------------------------
        let gbuf_pipeline_layout = unsafe {
            let push_ranges = Self::collect_push_constant_ranges(&gbuf_vert_refl, &gbuf_frag_refl);
            reflected_pipeline::create_pipeline_layout_from_merged(
                &device, &gbuf_set_layouts, &push_ranges,
            )?
        };

        let vertex_stride = if gltf_scene.is_some() { 48u32 } else { 0 };
        let gbuf_pipeline = unsafe {
            Self::create_gbuffer_pipeline(
                &device, gbuf_render_pass, gbuf_pipeline_layout,
                gbuf_vert_module, gbuf_frag_module,
                &gbuf_vert_refl, vertex_stride, width, height,
            )?
        };

        // ---------------------------------------------------------------
        // 18. Create lighting output image (RGBA8)
        // ---------------------------------------------------------------
        let light_output_image = scene_manager::create_offscreen_image(
            &device, ctx.allocator_mut(), width, height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
            "deferred_light_output",
        )?;

        // ---------------------------------------------------------------
        // 19. Create lighting render pass
        // ---------------------------------------------------------------
        let light_render_pass = unsafe {
            Self::create_lighting_render_pass(&device)?
        };

        // ---------------------------------------------------------------
        // 20. Create lighting framebuffer
        // ---------------------------------------------------------------
        let light_framebuffer = unsafe {
            let attachments = [light_output_image.view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(light_render_pass)
                .attachments(&attachments)
                .width(width)
                .height(height)
                .layers(1);
            device.create_framebuffer(&fb_info, None)
                .map_err(|e| format!("Failed to create lighting framebuffer: {:?}", e))?
        };

        // ---------------------------------------------------------------
        // 21. Create lighting descriptor sets (reflection-driven)
        // ---------------------------------------------------------------
        let light_merged = reflected_pipeline::merge_descriptor_sets(&[&light_vert_refl, &light_frag_refl]);
        let light_set_layouts = unsafe {
            reflected_pipeline::create_descriptor_set_layouts_from_merged(&device, &light_merged)?
        };

        let (light_pool_sizes, light_total_sets) = reflected_pipeline::compute_pool_sizes(&light_merged);
        let light_max_sets = light_total_sets.max(1);

        let light_desc_pool = unsafe {
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(light_max_sets)
                .pool_sizes(&light_pool_sizes);
            device.create_descriptor_pool(&pool_info, None)
                .map_err(|e| format!("Failed to create lighting descriptor pool: {:?}", e))?
        };

        // Allocate all lighting descriptor sets
        let mut light_desc_sets: HashMap<u32, vk::DescriptorSet> = HashMap::new();
        let max_light_set = light_set_layouts.keys().copied().max().unwrap_or(0);
        for set_idx in 0..=max_light_set {
            if let Some(&layout) = light_set_layouts.get(&set_idx) {
                let ds = unsafe {
                    let layouts = [layout];
                    let alloc_info = vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(light_desc_pool)
                        .set_layouts(&layouts);
                    device.allocate_descriptor_sets(&alloc_info)
                        .map_err(|e| format!("Failed to allocate lighting desc set {}: {:?}", set_idx, e))?[0]
                };
                light_desc_sets.insert(set_idx, ds);
            }
        }

        // Write lighting descriptors
        Self::write_lighting_descriptors(
            &device,
            &light_desc_sets,
            &light_merged,
            &deferred_camera_buffer,
            &mvp_buffer,
            &light_buffer,
            has_multi_light,
            lights_ssbo.as_ref(),
            scene_light_ubo.as_ref(),
            gbuf_sampler,
            gbuf_rt0.view,
            gbuf_rt1.view,
            gbuf_rt2.view,
            gbuf_depth.view,
            &ibl_assets,
            texture_sampler,
            &default_white_texture,
            &default_black_texture,
            &default_normal_texture,
        );

        info!("Lighting descriptors created");

        // ---------------------------------------------------------------
        // 22. Create lighting pipeline
        // ---------------------------------------------------------------
        let light_pipeline_layout = unsafe {
            let push_ranges = Self::collect_push_constant_ranges(&light_vert_refl, &light_frag_refl);
            reflected_pipeline::create_pipeline_layout_from_merged(
                &device, &light_set_layouts, &push_ranges,
            )?
        };

        let light_pipeline = unsafe {
            Self::create_lighting_pipeline(
                &device, light_render_pass, light_pipeline_layout,
                light_vert_module, light_frag_module, width, height,
            )?
        };

        // ---------------------------------------------------------------
        // 23. Setup push constants for G-buffer pass
        // ---------------------------------------------------------------
        let (push_constant_data, push_constant_stage_flags, push_constant_size) =
            Self::build_push_constant_data(
                &gbuf_vert_refl, &gbuf_frag_refl,
                light_dir, auto_eye,
            );

        info!("Deferred renderer initialized: {} ({}x{})", pipeline_base, width, height);

        Ok(Self {
            width,
            height,

            gbuf_rt0,
            gbuf_rt1,
            gbuf_rt2,
            gbuf_depth,
            gbuf_sampler,
            gbuf_render_pass,
            gbuf_framebuffer,

            gbuf_vert_module,
            gbuf_frag_module,
            gbuf_pipeline_layout,
            gbuf_pipeline,

            gbuf_set_layouts,
            gbuf_desc_pool,
            gbuf_vert_desc_set,
            gbuf_per_material_desc_sets,
            gbuf_per_material_buffers,

            light_output_image,
            light_render_pass,
            light_framebuffer,

            light_vert_module,
            light_frag_module,
            light_pipeline_layout,
            light_pipeline,

            light_set_layouts,
            light_desc_pool,
            light_desc_sets,

            mvp_buffer,
            light_buffer,
            deferred_camera_buffer,

            lights_ssbo,
            scene_light_ubo,
            has_multi_light,

            vbo,
            ibo,
            num_indices,
            draw_ranges,

            per_material_textures,
            texture_sampler,
            default_white_texture,
            default_black_texture,
            default_normal_texture,
            ibl_assets,

            push_constant_data,
            push_constant_stage_flags,
            push_constant_size,

            has_scene_bounds,
            auto_eye,
            auto_target,
            auto_up,
            auto_far,
        })
    }

    // ===================================================================
    // Internal helpers
    // ===================================================================

    /// Check if any reflection data contains a "lights" storage buffer binding.
    fn check_has_multi_light(frag1: &ReflectionData, frag2: &ReflectionData) -> bool {
        let check = |refl: &ReflectionData| -> bool {
            for bindings in refl.descriptor_sets.values() {
                for b in bindings {
                    if b.name == "lights" && b.binding_type == "storage_buffer" {
                        return true;
                    }
                }
            }
            false
        };
        check(frag1) || check(frag2)
    }

    /// Find the vertex and fragment set indices from merged bindings.
    fn find_gbuf_set_indices(merged: &HashMap<u32, Vec<reflected_pipeline::BindingInfo>>) -> (u32, u32) {
        let mut vert_set = 0u32;
        let mut frag_set = 1u32;

        for (&set_idx, bindings) in merged {
            for b in bindings {
                if b.name == "MVP" {
                    vert_set = set_idx;
                }
                if b.name == "Material" {
                    frag_set = set_idx;
                }
            }
        }

        (vert_set, frag_set)
    }

    /// Collect push constant ranges from reflection data.
    fn collect_push_constant_ranges(
        vert_refl: &ReflectionData,
        frag_refl: &ReflectionData,
    ) -> Vec<vk::PushConstantRange> {
        let mut ranges = Vec::new();
        for pc in &vert_refl.push_constants {
            if pc.size > 0 {
                ranges.push(
                    vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::VERTEX)
                        .offset(0)
                        .size(pc.size),
                );
            }
        }
        for pc in &frag_refl.push_constants {
            if pc.size > 0 {
                ranges.push(
                    vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                        .offset(0)
                        .size(pc.size),
                );
            }
        }
        ranges
    }

    /// Build push constant data buffer from G-buffer reflection.
    fn build_push_constant_data(
        vert_refl: &ReflectionData,
        frag_refl: &ReflectionData,
        light_dir: glam::Vec3,
        view_pos: glam::Vec3,
    ) -> (Vec<u8>, vk::ShaderStageFlags, u32) {
        let mut data = Vec::new();
        let mut stage_flags = vk::ShaderStageFlags::empty();
        let mut size = 0u32;

        let process = |refl: &ReflectionData, data: &mut Vec<u8>, stage_flags: &mut vk::ShaderStageFlags, size: &mut u32| {
            for pc in &refl.push_constants {
                if pc.size <= 0 { continue; }
                let new_size = (*size).max(pc.size);
                data.resize(new_size as usize, 0);
                *size = new_size;
                *stage_flags |= if refl.stage == "vertex" {
                    vk::ShaderStageFlags::VERTEX
                } else {
                    vk::ShaderStageFlags::FRAGMENT
                };
                for f in &pc.fields {
                    let offset = f.offset as usize;
                    if f.name == "light_dir" && offset + 12 <= *size as usize {
                        let bytes: [f32; 3] = light_dir.into();
                        data[offset..offset + 12].copy_from_slice(bytemuck::bytes_of(&bytes));
                    } else if f.name == "view_pos" && offset + 12 <= *size as usize {
                        let bytes: [f32; 3] = view_pos.into();
                        data[offset..offset + 12].copy_from_slice(bytemuck::bytes_of(&bytes));
                    }
                }
            }
        };

        process(vert_refl, &mut data, &mut stage_flags, &mut size);
        process(frag_refl, &mut data, &mut stage_flags, &mut size);

        (data, stage_flags, size)
    }

    // ===================================================================
    // G-buffer render pass creation
    // ===================================================================

    unsafe fn create_gbuffer_render_pass(device: &ash::Device) -> Result<vk::RenderPass, String> {
        let attachments = [
            // RT0: RGBA16F
            vk::AttachmentDescription::default()
                .format(vk::Format::R16G16B16A16_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            // RT1: RGBA16F
            vk::AttachmentDescription::default()
                .format(vk::Format::R16G16B16A16_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            // RT2: RGBA16F
            vk::AttachmentDescription::default()
                .format(vk::Format::R16G16B16A16_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            // Depth: D32_SFLOAT
            vk::AttachmentDescription::default()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        ];

        let color_refs = [
            vk::AttachmentReference::default()
                .attachment(0).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            vk::AttachmentReference::default()
                .attachment(1).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            vk::AttachmentReference::default()
                .attachment(2).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        ];

        let depth_ref = vk::AttachmentReference::default()
            .attachment(3)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs)
            .depth_stencil_attachment(&depth_ref);

        let dependencies = [
            // EXTERNAL -> 0
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                )
                .dst_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                )
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ),
            // 0 -> EXTERNAL
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                )
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .src_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::SHADER_READ),
        ];

        let rp_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);

        device.create_render_pass(&rp_info, None)
            .map_err(|e| format!("Failed to create G-buffer render pass: {:?}", e))
    }

    // ===================================================================
    // Lighting render pass creation
    // ===================================================================

    unsafe fn create_lighting_render_pass(device: &ash::Device) -> Result<vk::RenderPass, String> {
        let attachment = vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref));

        let dependencies = [
            // EXTERNAL -> 0
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
            // 0 -> EXTERNAL
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
        ];

        let rp_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&attachment))
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);

        device.create_render_pass(&rp_info, None)
            .map_err(|e| format!("Failed to create lighting render pass: {:?}", e))
    }

    // ===================================================================
    // G-buffer pipeline creation
    // ===================================================================

    unsafe fn create_gbuffer_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        pipeline_layout: vk::PipelineLayout,
        vert_module: vk::ShaderModule,
        frag_module: vk::ShaderModule,
        vert_refl: &ReflectionData,
        vertex_stride_override: u32,
        width: u32,
        height: u32,
    ) -> Result<vk::Pipeline, String> {
        let entry_name = c"main";

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(entry_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(entry_name),
        ];

        let (binding_descs, attr_descs) = if vertex_stride_override > 0 {
            reflected_pipeline::create_reflected_vertex_input_with_stride(vert_refl, vertex_stride_override)
        } else {
            reflected_pipeline::create_reflected_vertex_input(vert_refl)
        };

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descs)
            .vertex_attribute_descriptions(&attr_descs);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);

        // 3 color blend attachments (all blend disabled for G-buffer raw writes)
        let color_blend_attachments = [
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false),
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false),
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false),
        ];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&color_blend_attachments);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blending)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipelines = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_pipes, e)| format!("Failed to create G-buffer pipeline: {:?}", e))?;

        Ok(pipelines[0])
    }

    // ===================================================================
    // Lighting pipeline creation (fullscreen triangle)
    // ===================================================================

    unsafe fn create_lighting_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        pipeline_layout: vk::PipelineLayout,
        vert_module: vk::ShaderModule,
        frag_module: vk::ShaderModule,
        width: u32,
        height: u32,
    ) -> Result<vk::Pipeline, String> {
        let entry_name = c"main";

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(entry_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(entry_name),
        ];

        // No vertex input (fullscreen triangle generated in shader)
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipelines = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_pipes, e)| format!("Failed to create lighting pipeline: {:?}", e))?;

        Ok(pipelines[0])
    }

    // ===================================================================
    // Descriptor set writing helpers
    // ===================================================================

    fn write_gbuf_vertex_descriptors(
        device: &ash::Device,
        desc_set: vk::DescriptorSet,
        vert_set_idx: u32,
        merged: &HashMap<u32, Vec<reflected_pipeline::BindingInfo>>,
        mvp_buffer: &GpuBuffer,
        light_buffer: &GpuBuffer,
        has_multi_light: bool,
        lights_ssbo: Option<&GpuBuffer>,
        scene_light_ubo: Option<&GpuBuffer>,
    ) {
        if desc_set == vk::DescriptorSet::null() {
            return;
        }

        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();

        if let Some(bindings) = merged.get(&vert_set_idx) {
            for b in bindings {
                if b.binding_type == "uniform_buffer" {
                    let buf_info = if b.name == "MVP" {
                        vk::DescriptorBufferInfo::default()
                            .buffer(mvp_buffer.buffer)
                            .range(if b.size > 0 { b.size as u64 } else { 192 })
                    } else if b.name == "Light" {
                        vk::DescriptorBufferInfo::default()
                            .buffer(light_buffer.buffer)
                            .range(if b.size > 0 { b.size as u64 } else { 32 })
                    } else if b.name == "SceneLight" && has_multi_light {
                        if let Some(sl) = scene_light_ubo {
                            vk::DescriptorBufferInfo::default()
                                .buffer(sl.buffer)
                                .range(if b.size > 0 { b.size as u64 } else { 32 })
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    };
                    buffer_infos.push(buf_info);
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(desc_set)
                            .dst_binding(b.binding)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER),
                    );
                } else if b.binding_type == "storage_buffer" && b.name == "lights" && has_multi_light {
                    if let Some(ssbo) = lights_ssbo {
                        buffer_infos.push(
                            vk::DescriptorBufferInfo::default()
                                .buffer(ssbo.buffer)
                                .range(vk::WHOLE_SIZE),
                        );
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(desc_set)
                                .dst_binding(b.binding)
                                .descriptor_count(1)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER),
                        );
                    }
                }
            }
        }

        if !writes.is_empty() {
            // Patch buffer info pointers
            let mut final_writes = Vec::with_capacity(writes.len());
            for (i, w) in writes.into_iter().enumerate() {
                final_writes.push(w.buffer_info(std::slice::from_ref(&buffer_infos[i])));
            }
            unsafe {
                device.update_descriptor_sets(&final_writes, &[]);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn write_gbuf_material_descriptors(
        device: &ash::Device,
        desc_set: vk::DescriptorSet,
        frag_set_idx: u32,
        merged: &HashMap<u32, Vec<reflected_pipeline::BindingInfo>>,
        material_buffer: &GpuBuffer,
        light_buffer: &GpuBuffer,
        has_multi_light: bool,
        lights_ssbo: Option<&GpuBuffer>,
        scene_light_ubo: Option<&GpuBuffer>,
        ibl: &IblAssets,
        per_mat_textures: Option<&HashMap<String, GpuImage>>,
        texture_sampler: vk::Sampler,
        default_white: &GpuImage,
        default_black: &GpuImage,
        default_normal: &GpuImage,
    ) {
        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
        let mut writes: Vec<(vk::WriteDescriptorSet, bool)> = Vec::new(); // (write, is_image)

        if let Some(bindings) = merged.get(&frag_set_idx) {
            for b in bindings {
                if b.binding_type == "uniform_buffer" {
                    let buf_info = if b.name == "Material" {
                        vk::DescriptorBufferInfo::default()
                            .buffer(material_buffer.buffer)
                            .range(std::mem::size_of::<MaterialUboData>() as u64)
                    } else if b.name == "Light" {
                        vk::DescriptorBufferInfo::default()
                            .buffer(light_buffer.buffer)
                            .range(if b.size > 0 { b.size as u64 } else { 32 })
                    } else if b.name == "SceneLight" && has_multi_light {
                        if let Some(sl) = scene_light_ubo {
                            vk::DescriptorBufferInfo::default()
                                .buffer(sl.buffer)
                                .range(if b.size > 0 { b.size as u64 } else { 32 })
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    };
                    buffer_infos.push(buf_info);
                    writes.push((
                        vk::WriteDescriptorSet::default()
                            .dst_set(desc_set)
                            .dst_binding(b.binding)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER),
                        false,
                    ));
                } else if b.binding_type == "storage_buffer" && b.name == "lights" && has_multi_light {
                    if let Some(ssbo) = lights_ssbo {
                        buffer_infos.push(
                            vk::DescriptorBufferInfo::default()
                                .buffer(ssbo.buffer)
                                .range(vk::WHOLE_SIZE),
                        );
                        writes.push((
                            vk::WriteDescriptorSet::default()
                                .dst_set(desc_set)
                                .dst_binding(b.binding)
                                .descriptor_count(1)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER),
                            false,
                        ));
                    }
                } else if b.binding_type == "sampler" {
                    let sampler = Self::find_sampler_for_binding(&b.name, ibl, per_mat_textures, texture_sampler);
                    image_infos.push(
                        vk::DescriptorImageInfo::default()
                            .sampler(sampler),
                    );
                    writes.push((
                        vk::WriteDescriptorSet::default()
                            .dst_set(desc_set)
                            .dst_binding(b.binding)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::SAMPLER),
                        true,
                    ));
                } else if b.binding_type == "sampled_image" || b.binding_type == "sampled_cube_image" {
                    let (view, layout) = Self::find_image_for_binding(&b.name, ibl, per_mat_textures, default_white, default_black, default_normal);
                    image_infos.push(
                        vk::DescriptorImageInfo::default()
                            .image_view(view)
                            .image_layout(layout),
                    );
                    writes.push((
                        vk::WriteDescriptorSet::default()
                            .dst_set(desc_set)
                            .dst_binding(b.binding)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE),
                        true,
                    ));
                }
            }
        }

        if !writes.is_empty() {
            let mut buf_idx = 0usize;
            let mut img_idx = 0usize;
            let mut final_writes = Vec::with_capacity(writes.len());
            for (w, is_image) in writes {
                if is_image {
                    final_writes.push(w.image_info(std::slice::from_ref(&image_infos[img_idx])));
                    img_idx += 1;
                } else {
                    final_writes.push(w.buffer_info(std::slice::from_ref(&buffer_infos[buf_idx])));
                    buf_idx += 1;
                }
            }
            unsafe {
                device.update_descriptor_sets(&final_writes, &[]);
            }
        }
    }

    fn find_sampler_for_binding(
        name: &str,
        ibl: &IblAssets,
        per_mat_textures: Option<&HashMap<String, GpuImage>>,
        texture_sampler: vk::Sampler,
    ) -> vk::Sampler {
        // Check IBL assets (they carry their own samplers)
        if let Some(ref tex) = ibl.env_specular {
            if name == "env_specular_sampler" || name == "env_specular" {
                return tex.sampler;
            }
        }
        if let Some(ref tex) = ibl.env_irradiance {
            if name == "env_irradiance_sampler" || name == "env_irradiance" {
                return tex.sampler;
            }
        }
        if let Some(ref tex) = ibl.brdf_lut {
            if name == "brdf_lut_sampler" || name == "brdf_lut" || name == "brdfLUT" {
                return tex.sampler;
            }
        }
        // All material textures use the dedicated texture sampler (LINEAR, REPEAT)
        texture_sampler
    }

    fn find_image_for_binding(
        name: &str,
        ibl: &IblAssets,
        per_mat_textures: Option<&HashMap<String, GpuImage>>,
        default_white: &GpuImage,
        default_black: &GpuImage,
        default_normal: &GpuImage,
    ) -> (vk::ImageView, vk::ImageLayout) {
        // Check IBL
        if let Some(ref tex) = ibl.env_specular {
            if name == "env_specular" || name == "env_specular_tex" {
                return (tex.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            }
        }
        if let Some(ref tex) = ibl.env_irradiance {
            if name == "env_irradiance" || name == "env_irradiance_tex" {
                return (tex.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            }
        }
        if let Some(ref tex) = ibl.brdf_lut {
            if name == "brdf_lut" || name == "brdfLUT" || name == "brdf_lut_tex" {
                return (tex.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            }
        }
        // Check per-material textures
        if let Some(textures) = per_mat_textures {
            if let Some(img) = textures.get(name) {
                return (img.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            }
        }
        // Use appropriate default texture based on binding name
        let default_view = if name.contains("normal") {
            default_normal.view
        } else if name.contains("emissive") || name.contains("metallic_roughness") || name.contains("occlusion") {
            default_black.view
        } else {
            default_white.view
        };
        (default_view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
    }

    #[allow(clippy::too_many_arguments)]
    fn write_lighting_descriptors(
        device: &ash::Device,
        light_desc_sets: &HashMap<u32, vk::DescriptorSet>,
        merged: &HashMap<u32, Vec<reflected_pipeline::BindingInfo>>,
        deferred_camera_buffer: &GpuBuffer,
        mvp_buffer: &GpuBuffer,
        light_buffer: &GpuBuffer,
        has_multi_light: bool,
        lights_ssbo: Option<&GpuBuffer>,
        scene_light_ubo: Option<&GpuBuffer>,
        gbuf_sampler: vk::Sampler,
        gbuf_rt0_view: vk::ImageView,
        gbuf_rt1_view: vk::ImageView,
        gbuf_rt2_view: vk::ImageView,
        gbuf_depth_view: vk::ImageView,
        ibl: &IblAssets,
        texture_sampler: vk::Sampler,
        default_white: &GpuImage,
        default_black: &GpuImage,
        default_normal: &GpuImage,
    ) {
        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
        let mut writes: Vec<(vk::WriteDescriptorSet, bool)> = Vec::new(); // (write, is_image)

        let get_gbuf_view = |name: &str| -> Option<(vk::ImageView, vk::ImageLayout)> {
            match name {
                "gbuf_tex0" | "gbuffer_albedo" | "gbuf_rt0" =>
                    Some((gbuf_rt0_view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)),
                "gbuf_tex1" | "gbuffer_normal" | "gbuf_rt1" =>
                    Some((gbuf_rt1_view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)),
                "gbuf_tex2" | "gbuffer_material" | "gbuf_rt2" =>
                    Some((gbuf_rt2_view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)),
                "gbuf_depth" | "gbuffer_depth" =>
                    Some((gbuf_depth_view, vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)),
                _ => None,
            }
        };

        for (&set_idx, bindings) in merged {
            let ds = match light_desc_sets.get(&set_idx) {
                Some(&ds) => ds,
                None => continue,
            };

            for b in bindings {
                if b.binding_type == "uniform_buffer" {
                    let buf_info = if b.name == "DeferredCamera" {
                        vk::DescriptorBufferInfo::default()
                            .buffer(deferred_camera_buffer.buffer)
                            .range(if b.size > 0 { b.size as u64 } else { 80 })
                    } else if b.name == "MVP" {
                        vk::DescriptorBufferInfo::default()
                            .buffer(mvp_buffer.buffer)
                            .range(if b.size > 0 { b.size as u64 } else { 192 })
                    } else if b.name == "Light" {
                        vk::DescriptorBufferInfo::default()
                            .buffer(light_buffer.buffer)
                            .range(if b.size > 0 { b.size as u64 } else { 32 })
                    } else if b.name == "SceneLight" && has_multi_light {
                        if let Some(sl) = scene_light_ubo {
                            vk::DescriptorBufferInfo::default()
                                .buffer(sl.buffer)
                                .range(if b.size > 0 { b.size as u64 } else { 32 })
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    };
                    buffer_infos.push(buf_info);
                    writes.push((
                        vk::WriteDescriptorSet::default()
                            .dst_set(ds)
                            .dst_binding(b.binding)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER),
                        false,
                    ));
                } else if b.binding_type == "storage_buffer" && b.name == "lights" && has_multi_light {
                    if let Some(ssbo) = lights_ssbo {
                        buffer_infos.push(
                            vk::DescriptorBufferInfo::default()
                                .buffer(ssbo.buffer)
                                .range(vk::WHOLE_SIZE),
                        );
                        writes.push((
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds)
                                .dst_binding(b.binding)
                                .descriptor_count(1)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER),
                            false,
                        ));
                    }
                } else if b.binding_type == "sampler" {
                    let sampler = if is_gbuffer_binding(&b.name) {
                        gbuf_sampler
                    } else {
                        Self::find_sampler_for_binding(&b.name, ibl, None, texture_sampler)
                    };
                    image_infos.push(vk::DescriptorImageInfo::default().sampler(sampler));
                    writes.push((
                        vk::WriteDescriptorSet::default()
                            .dst_set(ds)
                            .dst_binding(b.binding)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::SAMPLER),
                        true,
                    ));
                } else if b.binding_type == "sampled_image" || b.binding_type == "sampled_cube_image" {
                    let (view, layout) = if let Some(gbuf) = get_gbuf_view(&b.name) {
                        gbuf
                    } else {
                        Self::find_image_for_binding(&b.name, ibl, None, default_white, default_black, default_normal)
                    };
                    image_infos.push(
                        vk::DescriptorImageInfo::default()
                            .image_view(view)
                            .image_layout(layout),
                    );
                    writes.push((
                        vk::WriteDescriptorSet::default()
                            .dst_set(ds)
                            .dst_binding(b.binding)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE),
                        true,
                    ));
                } else if b.binding_type == "combined_image_sampler" {
                    let (view, layout) = if let Some(gbuf) = get_gbuf_view(&b.name) {
                        gbuf
                    } else {
                        Self::find_image_for_binding(&b.name, ibl, None, default_white, default_black, default_normal)
                    };
                    let sampler = if is_gbuffer_binding(&b.name) {
                        gbuf_sampler
                    } else {
                        Self::find_sampler_for_binding(&b.name, ibl, None, texture_sampler)
                    };
                    image_infos.push(
                        vk::DescriptorImageInfo::default()
                            .sampler(sampler)
                            .image_view(view)
                            .image_layout(layout),
                    );
                    writes.push((
                        vk::WriteDescriptorSet::default()
                            .dst_set(ds)
                            .dst_binding(b.binding)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                        true,
                    ));
                }
            }
        }

        if !writes.is_empty() {
            let mut buf_idx = 0usize;
            let mut img_idx = 0usize;
            let mut final_writes = Vec::with_capacity(writes.len());
            for (w, is_image) in writes {
                if is_image {
                    final_writes.push(w.image_info(std::slice::from_ref(&image_infos[img_idx])));
                    img_idx += 1;
                } else {
                    final_writes.push(w.buffer_info(std::slice::from_ref(&buffer_infos[buf_idx])));
                    buf_idx += 1;
                }
            }
            unsafe {
                device.update_descriptor_sets(&final_writes, &[]);
            }
        }
    }

    // ===================================================================
    // Record G-buffer commands
    // ===================================================================

    fn record_gbuffer_commands(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.gbuf_pipeline);

            // Push constants
            if self.push_constant_size > 0 && !self.push_constant_stage_flags.is_empty() {
                device.cmd_push_constants(
                    cmd,
                    self.gbuf_pipeline_layout,
                    self.push_constant_stage_flags,
                    0,
                    &self.push_constant_data,
                );
            }

            // Bind vertex descriptor set (set 0)
            if self.gbuf_vert_desc_set != vk::DescriptorSet::null() {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.gbuf_pipeline_layout,
                    0, &[self.gbuf_vert_desc_set], &[],
                );
            }

            // Bind vertex/index buffers
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.vbo.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, self.ibo.buffer, 0, vk::IndexType::UINT32);

            // Draw: per-material draw ranges or single draw
            if !self.draw_ranges.is_empty() && !self.gbuf_per_material_desc_sets.is_empty() {
                let mut current_mat: i32 = -1;
                for range in &self.draw_ranges {
                    if range.material_index as i32 != current_mat {
                        current_mat = range.material_index as i32;
                        let mat_idx = (current_mat as usize).min(self.gbuf_per_material_desc_sets.len() - 1);
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.gbuf_pipeline_layout,
                            1, &[self.gbuf_per_material_desc_sets[mat_idx]], &[],
                        );
                    }
                    device.cmd_draw_indexed(cmd, range.index_count, 1, range.index_offset, 0, 0);
                }
            } else {
                if !self.gbuf_per_material_desc_sets.is_empty() {
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.gbuf_pipeline_layout,
                        1, &[self.gbuf_per_material_desc_sets[0]], &[],
                    );
                }
                device.cmd_draw_indexed(cmd, self.num_indices, 1, 0, 0, 0);
            }
        }
    }

    // ===================================================================
    // Blit to swapchain
    // ===================================================================

    pub fn cmd_blit_to_swapchain(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        swapchain_image: vk::Image,
        swapchain_extent: vk::Extent2D,
    ) {
        unsafe {
            // Transition swapchain image: UNDEFINED -> TRANSFER_DST
            let barrier = vk::ImageMemoryBarrier::default()
                .image(swapchain_image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier],
            );

            // Blit light output -> swapchain
            let blit_region = vk::ImageBlit::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: self.width as i32,
                        y: self.height as i32,
                        z: 1,
                    },
                ])
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: swapchain_extent.width as i32,
                        y: swapchain_extent.height as i32,
                        z: 1,
                    },
                ]);

            device.cmd_blit_image(
                cmd,
                self.light_output_image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit_region],
                vk::Filter::LINEAR,
            );

            // Transition swapchain image: TRANSFER_DST -> PRESENT_SRC
            let barrier = vk::ImageMemoryBarrier::default()
                .image(swapchain_image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier],
            );
        }
    }

    // ===================================================================
    // Cleanup
    // ===================================================================

    pub fn cleanup(&mut self, ctx: &mut VulkanContext) {
        unsafe {
            let _ = ctx.device.device_wait_idle();
        }

        let device = ctx.device.clone();

        // Lighting pipeline
        unsafe {
            device.destroy_pipeline(self.light_pipeline, None);
            device.destroy_pipeline_layout(self.light_pipeline_layout, None);
            device.destroy_framebuffer(self.light_framebuffer, None);
            device.destroy_render_pass(self.light_render_pass, None);
            device.destroy_shader_module(self.light_vert_module, None);
            device.destroy_shader_module(self.light_frag_module, None);
        }

        // Lighting output image
        self.light_output_image.destroy(&device, ctx.allocator_mut());

        // Lighting descriptors
        unsafe {
            device.destroy_descriptor_pool(self.light_desc_pool, None);
        }
        for (_, layout) in self.light_set_layouts.drain() {
            unsafe { device.destroy_descriptor_set_layout(layout, None); }
        }

        // G-buffer pipeline
        unsafe {
            device.destroy_pipeline(self.gbuf_pipeline, None);
            device.destroy_pipeline_layout(self.gbuf_pipeline_layout, None);
            device.destroy_framebuffer(self.gbuf_framebuffer, None);
            device.destroy_render_pass(self.gbuf_render_pass, None);
            device.destroy_shader_module(self.gbuf_vert_module, None);
            device.destroy_shader_module(self.gbuf_frag_module, None);
        }

        // G-buffer images
        self.gbuf_rt0.destroy(&device, ctx.allocator_mut());
        self.gbuf_rt1.destroy(&device, ctx.allocator_mut());
        self.gbuf_rt2.destroy(&device, ctx.allocator_mut());
        self.gbuf_depth.destroy(&device, ctx.allocator_mut());

        // G-buffer sampler
        unsafe { device.destroy_sampler(self.gbuf_sampler, None); }

        // G-buffer descriptors
        for buf in &mut self.gbuf_per_material_buffers {
            buf.destroy(&device, ctx.allocator_mut());
        }
        self.gbuf_per_material_buffers.clear();
        self.gbuf_per_material_desc_sets.clear();

        unsafe { device.destroy_descriptor_pool(self.gbuf_desc_pool, None); }
        for (_, layout) in self.gbuf_set_layouts.drain() {
            unsafe { device.destroy_descriptor_set_layout(layout, None); }
        }

        // Uniform buffers
        self.mvp_buffer.destroy(&device, ctx.allocator_mut());
        self.light_buffer.destroy(&device, ctx.allocator_mut());
        self.deferred_camera_buffer.destroy(&device, ctx.allocator_mut());

        // Multi-light
        if let Some(ref mut ssbo) = self.lights_ssbo {
            ssbo.destroy(&device, ctx.allocator_mut());
        }
        if let Some(ref mut ubo) = self.scene_light_ubo {
            ubo.destroy(&device, ctx.allocator_mut());
        }

        // Geometry
        self.vbo.destroy(&device, ctx.allocator_mut());
        self.ibo.destroy(&device, ctx.allocator_mut());

        // Per-material textures
        for textures in &mut self.per_material_textures {
            for (_, img) in textures.drain() {
                let mut img = img;
                img.destroy(&device, ctx.allocator_mut());
            }
        }

        // Default textures and sampler
        unsafe { device.destroy_sampler(self.texture_sampler, None); }
        self.default_white_texture.destroy(&device, ctx.allocator_mut());
        self.default_black_texture.destroy(&device, ctx.allocator_mut());
        self.default_normal_texture.destroy(&device, ctx.allocator_mut());

        // IBL
        self.ibl_assets.destroy(&device, ctx.allocator_mut());
    }
}

// ===========================================================================
// Renderer trait implementation
// ===========================================================================

impl scene_manager::Renderer for DeferredRenderer {
    fn render(&mut self, ctx: &VulkanContext) -> Result<vk::CommandBuffer, String> {
        let cmd = ctx.begin_single_commands()?;
        let device = &ctx.device;

        // === Pass 1: G-buffer ===
        {
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
                },
            ];

            let rp_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.gbuf_render_pass)
                .framebuffer(self.gbuf_framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: self.width, height: self.height },
                })
                .clear_values(&clear_values);

            unsafe {
                device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            }
            self.record_gbuffer_commands(device, cmd);
            unsafe {
                device.cmd_end_render_pass(cmd);
            }
        }

        // === Transition G-buffer images: COLOR_ATTACHMENT -> SHADER_READ_ONLY ===
        {
            let color_images = [self.gbuf_rt0.image, self.gbuf_rt1.image, self.gbuf_rt2.image];
            let mut barriers: Vec<vk::ImageMemoryBarrier> = Vec::with_capacity(4);

            for &img in &color_images {
                barriers.push(
                    vk::ImageMemoryBarrier::default()
                        .image(img)
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                );
            }

            // Depth barrier
            barriers.push(
                vk::ImageMemoryBarrier::default()
                    .image(self.gbuf_depth.image)
                    .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .level_count(1)
                            .layer_count(1),
                    ),
            );

            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[], &[], &barriers,
                );
            }
        }

        // === Pass 2: Lighting ===
        {
            let clear_value = vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
            };

            let rp_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.light_render_pass)
                .framebuffer(self.light_framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: self.width, height: self.height },
                })
                .clear_values(std::slice::from_ref(&clear_value));

            unsafe {
                device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);

                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.light_pipeline);

                // Bind all lighting descriptor sets
                let max_set = self.light_desc_sets.keys().copied().max().unwrap_or(0);
                let mut all_sets = Vec::with_capacity((max_set + 1) as usize);
                for i in 0..=max_set {
                    if let Some(&ds) = self.light_desc_sets.get(&i) {
                        all_sets.push(ds);
                    }
                }
                if !all_sets.is_empty() {
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.light_pipeline_layout,
                        0, &all_sets, &[],
                    );
                }

                // Draw fullscreen triangle (3 vertices, no vertex buffer)
                device.cmd_draw(cmd, 3, 1, 0, 0);

                device.cmd_end_render_pass(cmd);
            }
        }

        Ok(cmd)
    }

    fn update_camera(
        &mut self,
        eye: glam::Vec3,
        target: glam::Vec3,
        up: glam::Vec3,
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) {
        let view = crate::camera::look_at(eye, target, up);
        let proj = crate::camera::perspective(fov_y, aspect, near, far);

        // Update MVP buffer
        let mvp = MvpData {
            model: glam::Mat4::IDENTITY.to_cols_array(),
            view: view.to_cols_array(),
            projection: proj.to_cols_array(),
        };
        if let Some(ref mut alloc) = self.mvp_buffer.allocation {
            if let Some(mapped) = alloc.mapped_slice_mut() {
                let bytes = bytemuck::bytes_of(&mvp);
                mapped[..bytes.len()].copy_from_slice(bytes);
            }
        }

        // Update Light buffer (preserve light_dir, update view_pos)
        if let Some(ref mut alloc) = self.light_buffer.allocation {
            if let Some(mapped) = alloc.mapped_slice_mut() {
                // Read existing light_dir from offset 0..12
                let mut light_data: LightData = *bytemuck::from_bytes(&mapped[..std::mem::size_of::<LightData>()]);
                light_data.view_pos = eye.into();
                let bytes = bytemuck::bytes_of(&light_data);
                mapped[..bytes.len()].copy_from_slice(bytes);
            }
        }

        // Update DeferredCamera buffer
        let vp = proj * view;
        let inv_vp = vp.inverse();
        let cam_data = DeferredCameraData {
            inv_view_proj: inv_vp.to_cols_array(),
            view_pos: eye.into(),
            _pad: 0.0,
        };
        if let Some(ref mut alloc) = self.deferred_camera_buffer.allocation {
            if let Some(mapped) = alloc.mapped_slice_mut() {
                let bytes = bytemuck::bytes_of(&cam_data);
                mapped[..bytes.len()].copy_from_slice(bytes);
            }
        }

        // Update SceneLight UBO viewPos
        if self.has_multi_light {
            if let Some(ref mut ubo) = self.scene_light_ubo {
                if let Some(ref mut alloc) = ubo.allocation {
                    if let Some(mapped) = alloc.mapped_slice_mut() {
                        let mut sl: SceneLightUboData = *bytemuck::from_bytes(&mapped[..std::mem::size_of::<SceneLightUboData>()]);
                        sl.view_pos = eye.into();
                        let bytes = bytemuck::bytes_of(&sl);
                        mapped[..bytes.len()].copy_from_slice(bytes);
                    }
                }
            }
        }
    }

    fn blit_to_swapchain(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        swap_image: vk::Image,
        extent: vk::Extent2D,
    ) {
        self.cmd_blit_to_swapchain(device, cmd, swap_image, extent);
    }

    fn output_image(&self) -> vk::Image {
        self.light_output_image.image
    }

    fn output_format(&self) -> vk::Format {
        vk::Format::R8G8B8A8_UNORM
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn has_scene_bounds(&self) -> bool {
        self.has_scene_bounds
    }

    fn auto_eye(&self) -> glam::Vec3 {
        self.auto_eye
    }

    fn auto_target(&self) -> glam::Vec3 {
        self.auto_target
    }

    fn auto_up(&self) -> glam::Vec3 {
        self.auto_up
    }

    fn auto_far(&self) -> f32 {
        self.auto_far
    }

    fn wait_stage(&self) -> vk::PipelineStageFlags {
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
    }

    fn destroy(&mut self, ctx: &mut VulkanContext) {
        self.cleanup(ctx);
    }
}
