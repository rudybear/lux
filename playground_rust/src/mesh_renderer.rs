//! Mesh shader renderer: builds meshlets, creates mesh+fragment pipeline, dispatches mesh tasks.
//!
//! Follows the raster_renderer pattern for render pass/framebuffer/depth buffer
//! and the rt_renderer pattern for SoA storage buffers.

use ash::vk;
use bytemuck;
use gpu_allocator::vulkan::Allocator;
use log::info;
use std::collections::HashMap;
use std::path::Path;

use crate::camera::DefaultCamera;
use crate::meshlet;
use crate::reflected_pipeline;
use crate::scene;
use crate::scene_manager::{self, GpuBuffer, GpuImage, IblAssets};
use crate::spv_loader;
use crate::vulkan_context::VulkanContext;

/// Mesh shader renderer state (persistent, suitable for interactive multi-frame rendering).
pub struct MeshShaderRenderer {
    // Pipeline
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    ds_layouts: HashMap<u32, vk::DescriptorSetLayout>,
    descriptor_sets: Vec<vk::DescriptorSet>,

    // Shader modules
    mesh_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,

    // Render target (offscreen)
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    color_image: GpuImage,
    depth_image: GpuImage,
    width: u32,
    height: u32,

    // Meshlet data
    meshlet_count: u32,

    // SoA storage buffers (meshlet descriptors, vertices, triangles, positions, normals, tex_coords, tangents)
    meshlet_desc_buffer: GpuBuffer,
    meshlet_verts_buffer: GpuBuffer,
    meshlet_tris_buffer: GpuBuffer,
    positions_buffer: GpuBuffer,
    normals_buffer: GpuBuffer,
    tex_coords_buffer: GpuBuffer,
    tangents_buffer: GpuBuffer,

    // Uniforms
    mvp_buffer: GpuBuffer,
    light_buffer: GpuBuffer,

    // Textures
    sampler: vk::Sampler,
    default_texture: GpuImage,
    default_black_texture: GpuImage,
    default_normal_texture: GpuImage,
    texture_images: HashMap<String, GpuImage>,
    ibl_assets: IblAssets,

    // Auto-camera
    pub auto_eye: glam::Vec3,
    pub auto_target: glam::Vec3,
    pub auto_up: glam::Vec3,
    pub auto_far: f32,
    pub has_scene_bounds: bool,
}

/// Delegate to scene_manager::create_buffer_with_data.
fn create_buffer_with_data(
    device: &ash::Device,
    allocator: &mut Allocator,
    data: &[u8],
    usage: vk::BufferUsageFlags,
    name: &str,
) -> Result<GpuBuffer, String> {
    scene_manager::create_buffer_with_data(device, allocator, data, usage, name)
}

/// Delegate to scene_manager::create_offscreen_image.
fn create_offscreen_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
    name: &str,
) -> Result<GpuImage, String> {
    scene_manager::create_offscreen_image(
        device, allocator, width, height, format, usage, aspect, name,
    )
}

/// Delegate to scene_manager::create_texture_image with FRAGMENT_SHADER dst_stage.
fn create_texture_image(
    ctx: &mut VulkanContext,
    width: u32,
    height: u32,
    pixels: &[u8],
    name: &str,
) -> Result<GpuImage, String> {
    scene_manager::create_texture_image(
        ctx,
        width,
        height,
        pixels,
        name,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
    )
}

/// Create a 1x1 white RGBA default texture image.
fn create_default_white_texture(ctx: &mut VulkanContext) -> Result<GpuImage, String> {
    let white_pixel: [u8; 4] = [255, 255, 255, 255];
    create_texture_image(ctx, 1, 1, &white_pixel, "default_white_1x1")
}

/// Delegate to scene_manager::load_ibl_assets with FRAGMENT_SHADER dst_stage.
fn load_ibl_assets(ctx: &mut VulkanContext, ibl_name: &str) -> IblAssets {
    scene_manager::load_ibl_assets(ctx, vk::PipelineStageFlags::FRAGMENT_SHADER, ibl_name)
}

impl MeshShaderRenderer {
    /// Create a new mesh shader renderer.
    ///
    /// Loads reflection, builds meshlets from scene geometry, uploads storage buffers,
    /// creates render pass/framebuffer/pipeline, and writes descriptors.
    pub fn new(
        ctx: &mut VulkanContext,
        scene_source: &str,
        pipeline_base: &str,
        width: u32,
        height: u32,
        ibl_name: &str,
    ) -> Result<Self, String> {
        let device = ctx.device.clone();

        // --- Phase 0: Load reflection JSON ---
        let mesh_json_path = format!("{}.mesh.json", pipeline_base);
        let frag_json_path = format!("{}.frag.json", pipeline_base);

        let mesh_refl = reflected_pipeline::load_reflection(Path::new(&mesh_json_path))?;
        let frag_refl = reflected_pipeline::load_reflection(Path::new(&frag_json_path))?;

        info!(
            "Mesh shader reflection loaded: mesh={} frag={} descriptor_sets: mesh={:?}, frag={:?}",
            mesh_json_path,
            frag_json_path,
            mesh_refl.descriptor_sets.keys().collect::<Vec<_>>(),
            frag_refl.descriptor_sets.keys().collect::<Vec<_>>(),
        );

        // Get mesh output info from reflection (for meshlet building limits)
        let mesh_output = mesh_refl.mesh_output.as_ref();
        let max_verts = mesh_output.map(|m| m.max_vertices).unwrap_or(64);
        let max_prims = mesh_output.map(|m| m.max_primitives).unwrap_or(124);
        info!(
            "Mesh output limits from reflection: max_vertices={}, max_primitives={}",
            max_verts, max_prims
        );

        // Merge descriptor sets from both stages
        let merged = reflected_pipeline::merge_descriptor_sets(&[&mesh_refl, &frag_refl]);

        // Create descriptor set layouts
        let ds_layouts = unsafe {
            reflected_pipeline::create_descriptor_set_layouts_from_merged(&device, &merged)?
        };

        let (pool_sizes, max_sets) = reflected_pipeline::compute_pool_sizes(&merged);

        info!(
            "Mesh shader reflection-driven descriptors: {} sets, {} pool size entries",
            max_sets,
            pool_sizes.len()
        );

        // Create descriptor pool
        let descriptor_pool = unsafe {
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(max_sets)
                .pool_sizes(&pool_sizes);
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| format!("Failed to create mesh shader descriptor pool: {:?}", e))?
        };

        // Allocate descriptor sets
        let max_set_idx = ds_layouts.keys().copied().max().unwrap_or(0);
        let mut ordered_layouts: Vec<vk::DescriptorSetLayout> = Vec::new();
        for i in 0..=max_set_idx {
            if let Some(&layout) = ds_layouts.get(&i) {
                ordered_layouts.push(layout);
            }
        }

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&ordered_layouts);

        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| format!("Failed to allocate mesh shader descriptor sets: {:?}", e))?
        };

        let mut ds_map: HashMap<u32, vk::DescriptorSet> = HashMap::new();
        {
            let mut ds_idx = 0;
            for i in 0..=max_set_idx {
                if ds_layouts.contains_key(&i) {
                    ds_map.insert(i, descriptor_sets[ds_idx]);
                    ds_idx += 1;
                }
            }
        }

        // Pipeline layout (no push constants for mesh shader by default)
        let pipeline_layout = unsafe {
            reflected_pipeline::create_pipeline_layout_from_merged(&device, &ds_layouts, &[])?
        };

        // --- Phase 1: Load scene geometry ---
        let is_gltf = scene_source.ends_with(".glb") || scene_source.ends_with(".gltf");

        let mut gltf_scene = if is_gltf {
            Some(crate::gltf_loader::load_gltf(Path::new(scene_source))?)
        } else {
            None
        };

        let mut auto_eye = glam::Vec3::new(0.0, 0.0, 3.0);
        let mut auto_target = glam::Vec3::ZERO;
        let mut auto_up = glam::Vec3::new(0.0, 1.0, 0.0);
        let mut auto_far = 100.0f32;
        let mut has_scene_bounds = false;

        let mut tangent_data_raw: Vec<[f32; 4]> = Vec::new();
        let (vertices, indices): (Vec<scene::PbrVertex>, Vec<u32>) =
            if let Some(ref mut gltf_s) = gltf_scene {
                if gltf_s.meshes.is_empty() {
                    return Err(format!("No meshes found in glTF file: {}", scene_source));
                }
                let draw_items = crate::gltf_loader::flatten_scene(gltf_s);
                if draw_items.is_empty() {
                    return Err(format!("No draw items in glTF scene: {}", scene_source));
                }

                let mut all_verts: Vec<scene::PbrVertex> = Vec::new();
                let mut all_indices: Vec<u32> = Vec::new();
                let mut vertex_offset: u32 = 0;
                let mut vertex_positions: Vec<[f32; 3]> = Vec::new();

                for item in &draw_items {
                    let mesh = &gltf_s.meshes[item.mesh_index];
                    let world = item.world_transform;
                    let normal_mat =
                        glam::Mat3::from_mat4(world).inverse().transpose();

                    for v in &mesh.vertices {
                        let pos =
                            world.transform_point3(glam::Vec3::from(v.position));
                        let norm =
                            (normal_mat * glam::Vec3::from(v.normal)).normalize();
                        vertex_positions.push([pos.x, pos.y, pos.z]);
                        all_verts.push(scene::PbrVertex {
                            position: pos.into(),
                            normal: norm.into(),
                            uv: v.uv,
                        });
                        let tan3 = (normal_mat * glam::Vec3::from_slice(&v.tangent[..3])).normalize();
                        tangent_data_raw.push([tan3.x, tan3.y, tan3.z, v.tangent[3]]);
                    }
                    for &idx in &mesh.indices {
                        all_indices.push(idx + vertex_offset);
                    }
                    vertex_offset += mesh.vertices.len() as u32;
                }

                // Compute auto-camera
                has_scene_bounds = true;
                let (eye, target, up, far) =
                    scene_manager::compute_auto_camera_from_draw_items(
                        &vertex_positions,
                        &draw_items,
                    );
                auto_eye = eye;
                auto_target = target;
                auto_up = up;
                auto_far = far;

                (all_verts, all_indices)
            } else {
                scene::generate_sphere(32, 32)
            };
        if tangent_data_raw.is_empty() {
            tangent_data_raw = vec![[1.0, 0.0, 0.0, 1.0]; vertices.len()];
        }

        // --- Phase 2: Build meshlets ---
        let meshlet_result = meshlet::build_meshlets(
            &indices,
            vertices.len() as u32,
            max_verts,
            max_prims,
        );

        let meshlet_count = meshlet_result.meshlets.len() as u32;
        info!(
            "Built {} meshlets from {} triangles ({} vertices)",
            meshlet_count,
            indices.len() / 3,
            vertices.len()
        );

        // --- Phase 3: Upload storage buffers ---
        // Meshlet descriptors (vec4 each)
        let meshlet_desc_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            bytemuck::cast_slice(&meshlet_result.meshlets),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "mesh_meshlet_descriptors",
        )?;

        // Meshlet vertex indices
        let meshlet_verts_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            bytemuck::cast_slice(&meshlet_result.meshlet_vertices),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "mesh_meshlet_vertices",
        )?;

        // Meshlet triangle indices
        let meshlet_tris_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            bytemuck::cast_slice(&meshlet_result.meshlet_triangles),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "mesh_meshlet_triangles",
        )?;

        // SoA vertex attribute buffers (same pattern as rt_renderer)
        let positions_data: Vec<[f32; 4]> = vertices
            .iter()
            .map(|v| [v.position[0], v.position[1], v.position[2], 1.0])
            .collect();
        let normals_data: Vec<[f32; 4]> = vertices
            .iter()
            .map(|v| [v.normal[0], v.normal[1], v.normal[2], 0.0])
            .collect();
        let tex_coords_data: Vec<[f32; 2]> = vertices.iter().map(|v| v.uv).collect();

        let positions_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            bytemuck::cast_slice(&positions_data),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "mesh_positions",
        )?;
        let normals_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            bytemuck::cast_slice(&normals_data),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "mesh_normals",
        )?;
        let tex_coords_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            bytemuck::cast_slice(&tex_coords_data),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "mesh_tex_coords",
        )?;

        let tangents_data: Vec<[f32; 4]> = tangent_data_raw;
        let tangents_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            bytemuck::cast_slice(&tangents_data),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "mesh_tangents",
        )?;

        info!(
            "Created mesh SoA storage buffers: positions={}B, normals={}B, texCoords={}B, tangents={}B, meshlets={}B",
            positions_data.len() * 16,
            normals_data.len() * 16,
            tex_coords_data.len() * 8,
            tangents_data.len() * 16,
            meshlet_result.meshlets.len() * 16
        );

        // --- Phase 4: Uniform buffers ---
        let aspect = width as f32 / height as f32;
        let (model, view, proj) = if has_scene_bounds {
            (
                glam::Mat4::IDENTITY,
                crate::camera::look_at(auto_eye, auto_target, auto_up),
                crate::camera::perspective(45.0f32.to_radians(), aspect, 0.1, auto_far),
            )
        } else {
            (
                DefaultCamera::model(),
                DefaultCamera::view(),
                DefaultCamera::projection(aspect),
            )
        };

        let mut mvp_data = [0u8; 192];
        mvp_data[0..64].copy_from_slice(bytemuck::cast_slice(model.as_ref()));
        mvp_data[64..128].copy_from_slice(bytemuck::cast_slice(view.as_ref()));
        mvp_data[128..192].copy_from_slice(bytemuck::cast_slice(proj.as_ref()));

        let mvp_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            &mvp_data,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "mesh_mvp",
        )?;

        let light_dir = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
        let camera_pos = if has_scene_bounds {
            auto_eye
        } else {
            DefaultCamera::EYE
        };
        let mut light_data = [0u8; 32];
        light_data[0..12].copy_from_slice(bytemuck::cast_slice(light_dir.as_ref()));
        light_data[12..16].copy_from_slice(&0.0f32.to_le_bytes());
        light_data[16..28].copy_from_slice(bytemuck::cast_slice(camera_pos.as_ref()));
        light_data[28..32].copy_from_slice(&0.0f32.to_le_bytes());

        let light_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            &light_data,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "mesh_light",
        )?;

        // --- Phase 5: Textures ---
        let sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::REPEAT)
                        .address_mode_v(vk::SamplerAddressMode::REPEAT)
                        .address_mode_w(vk::SamplerAddressMode::REPEAT),
                    None,
                )
                .map_err(|e| format!("Failed to create sampler: {:?}", e))?
        };

        let gltf_material = gltf_scene.as_ref().and_then(|s| {
            if !s.materials.is_empty() {
                Some(&s.materials[0])
            } else {
                None
            }
        });

        let default_texture = create_default_white_texture(ctx)?;
        let default_black_texture =
            create_texture_image(ctx, 1, 1, &[0u8, 0, 0, 255], "default_black")?;
        let default_normal_texture =
            create_texture_image(ctx, 1, 1, &[128u8, 128, 255, 255], "default_normal")?;

        let mut texture_images: HashMap<String, GpuImage> = HashMap::new();
        let texture_names = [
            "base_color_tex",
            "normal_tex",
            "metallic_roughness_tex",
            "occlusion_tex",
            "emissive_tex",
            "albedo_tex",
        ];
        for tex_name in &texture_names {
            let tex_image_opt = match *tex_name {
                "base_color_tex" | "albedo_tex" => {
                    gltf_material.and_then(|m| m.base_color_image.as_ref())
                }
                "normal_tex" => gltf_material.and_then(|m| m.normal_image.as_ref()),
                "metallic_roughness_tex" => {
                    gltf_material.and_then(|m| m.metallic_roughness_image.as_ref())
                }
                "occlusion_tex" => gltf_material.and_then(|m| m.occlusion_image.as_ref()),
                "emissive_tex" => gltf_material.and_then(|m| m.emissive_image.as_ref()),
                _ => None,
            };
            if let Some(tex_data) = tex_image_opt {
                let gpu_img = create_texture_image(
                    ctx,
                    tex_data.width,
                    tex_data.height,
                    &tex_data.pixels,
                    tex_name,
                )?;
                texture_images.insert(tex_name.to_string(), gpu_img);
            }
        }

        if gltf_material.is_none() {
            let tex_pixels = scene::generate_procedural_texture(512);
            let proc_texture =
                create_texture_image(ctx, 512, 512, &tex_pixels, "procedural_albedo")?;
            texture_images.insert("albedo_tex".to_string(), proc_texture);
            let proc_texture2 =
                create_texture_image(ctx, 512, 512, &tex_pixels, "procedural_base_color")?;
            texture_images.insert("base_color_tex".to_string(), proc_texture2);
        }

        let ibl_assets = load_ibl_assets(ctx, ibl_name);

        // --- Phase 6: Write descriptor sets from reflection ---
        // Build SSBO map by name -> (buffer, size)
        let ssbo_map: HashMap<String, (vk::Buffer, u64)> = [
            (
                "meshlet_descriptors".to_string(),
                (
                    meshlet_desc_buffer.buffer,
                    meshlet_result.meshlets.len() as u64 * 16,
                ),
            ),
            (
                "meshlet_vertices".to_string(),
                (
                    meshlet_verts_buffer.buffer,
                    meshlet_result.meshlet_vertices.len() as u64 * 4,
                ),
            ),
            (
                "meshlet_triangles".to_string(),
                (
                    meshlet_tris_buffer.buffer,
                    meshlet_result.meshlet_triangles.len() as u64 * 4,
                ),
            ),
            (
                "positions".to_string(),
                (positions_buffer.buffer, positions_data.len() as u64 * 16),
            ),
            (
                "normals".to_string(),
                (normals_buffer.buffer, normals_data.len() as u64 * 16),
            ),
            (
                "tex_coords".to_string(),
                (tex_coords_buffer.buffer, tex_coords_data.len() as u64 * 8),
            ),
            (
                "tangents".to_string(),
                (tangents_buffer.buffer, tangents_data.len() as u64 * 16),
            ),
        ]
        .into_iter()
        .collect();

        // Write descriptors (same pattern as raster_renderer)
        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();
        let total_bindings: usize = merged.values().map(|v| v.len()).sum();
        buffer_infos.reserve(total_bindings);
        image_infos.reserve(total_bindings);

        for (&set_idx, bindings) in &merged {
            let ds = match ds_map.get(&set_idx) {
                Some(&ds) => ds,
                None => continue,
            };
            for binding in bindings {
                let vk_type =
                    reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);
                match vk_type {
                    vk::DescriptorType::UNIFORM_BUFFER => {
                        let (buffer, range) = match binding.name.as_str() {
                            "MVP" => (mvp_buffer.buffer, binding.size as u64),
                            "Light" => (light_buffer.buffer, binding.size as u64),
                            _ => continue,
                        };
                        let idx = buffer_infos.len();
                        buffer_infos.push(
                            vk::DescriptorBufferInfo::default()
                                .buffer(buffer)
                                .offset(0)
                                .range(range),
                        );
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds)
                                .dst_binding(binding.binding)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .buffer_info(unsafe {
                                    std::slice::from_raw_parts(
                                        &buffer_infos[idx] as *const _,
                                        1,
                                    )
                                }),
                        );
                    }
                    vk::DescriptorType::STORAGE_BUFFER => {
                        let (buffer, size) = match ssbo_map.get(&binding.name) {
                            Some(&(buf, sz)) => (buf, sz),
                            None => {
                                info!(
                                    "Unknown SSBO name '{}' at set={} binding={}, skipping",
                                    binding.name, set_idx, binding.binding
                                );
                                continue;
                            }
                        };
                        let idx = buffer_infos.len();
                        buffer_infos.push(
                            vk::DescriptorBufferInfo::default()
                                .buffer(buffer)
                                .offset(0)
                                .range(size),
                        );
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds)
                                .dst_binding(binding.binding)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                .buffer_info(unsafe {
                                    std::slice::from_raw_parts(
                                        &buffer_infos[idx] as *const _,
                                        1,
                                    )
                                }),
                        );
                    }
                    vk::DescriptorType::SAMPLER => {
                        let actual_sampler = ibl_assets
                            .sampler_for_binding(&binding.name)
                            .unwrap_or(sampler);
                        let idx = image_infos.len();
                        image_infos
                            .push(vk::DescriptorImageInfo::default().sampler(actual_sampler));
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds)
                                .dst_binding(binding.binding)
                                .descriptor_type(vk::DescriptorType::SAMPLER)
                                .image_info(unsafe {
                                    std::slice::from_raw_parts(
                                        &image_infos[idx] as *const _,
                                        1,
                                    )
                                }),
                        );
                    }
                    vk::DescriptorType::SAMPLED_IMAGE => {
                        let is_cube =
                            reflected_pipeline::is_cube_image_binding(&binding.binding_type);
                        let view = if is_cube {
                            ibl_assets
                                .view_for_binding(&binding.name)
                                .unwrap_or(default_texture.view)
                        } else if binding.name == "brdf_lut" {
                            ibl_assets
                                .view_for_binding(&binding.name)
                                .unwrap_or(default_texture.view)
                        } else {
                            let fallback = match binding.name.as_str() {
                                "emissive_tex" => default_black_texture.view,
                                "normal_tex" => default_normal_texture.view,
                                _ => default_texture.view,
                            };
                            texture_images
                                .get(&binding.name)
                                .map(|img| img.view)
                                .unwrap_or(fallback)
                        };
                        let idx = image_infos.len();
                        image_infos.push(
                            vk::DescriptorImageInfo::default()
                                .image_view(view)
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                        );
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds)
                                .dst_binding(binding.binding)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .image_info(unsafe {
                                    std::slice::from_raw_parts(
                                        &image_infos[idx] as *const _,
                                        1,
                                    )
                                }),
                        );
                    }
                    _ => {
                        info!(
                            "Unhandled descriptor type {:?} at set={} binding={} name='{}'",
                            vk_type, set_idx, binding.binding, binding.name
                        );
                    }
                }
            }
        }

        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }

        // --- Phase 7: Render pass, framebuffer, pipeline ---
        let color_image = create_offscreen_image(
            &device,
            ctx.allocator_mut(),
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
            "mesh_color",
        )?;

        let depth_image = create_offscreen_image(
            &device,
            ctx.allocator_mut(),
            width,
            height,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
            "mesh_depth",
        )?;

        let attachments = [
            vk::AttachmentDescription::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            vk::AttachmentDescription::default()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        ];

        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref))
            .depth_stencil_attachment(&depth_ref);

        let render_pass = unsafe {
            device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&attachments)
                        .subpasses(std::slice::from_ref(&subpass)),
                    None,
                )
                .map_err(|e| format!("Failed to create mesh shader render pass: {:?}", e))?
        };

        let fb_attachments = [color_image.view, depth_image.view];
        let framebuffer = unsafe {
            device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass)
                        .attachments(&fb_attachments)
                        .width(width)
                        .height(height)
                        .layers(1),
                    None,
                )
                .map_err(|e| format!("Failed to create mesh shader framebuffer: {:?}", e))?
        };

        // Load shader modules
        let mesh_path = format!("{}.mesh.spv", pipeline_base);
        let frag_path = format!("{}.frag.spv", pipeline_base);
        let mesh_code = spv_loader::load_spirv(Path::new(&mesh_path))?;
        let frag_code = spv_loader::load_spirv(Path::new(&frag_path))?;
        let mesh_module = spv_loader::create_shader_module(&device, &mesh_code)?;
        let frag_module = spv_loader::create_shader_module(&device, &frag_code)?;

        // Create graphics pipeline with mesh + fragment stages (no vertex input)
        let entry_name = c"main";
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MESH_EXT)
                .module(mesh_module)
                .name(entry_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(entry_name),
        ];

        let viewport = vk::Viewport::default()
            .width(width as f32)
            .height(height as f32)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(vk::Extent2D { width, height });
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));

        // Mesh shader pipelines do NOT have vertex_input_state or input_assembly_state.
        // They only need viewport, rasterization, multisample, depth_stencil, color_blend.
        let pipeline = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&shader_stages)
                        .viewport_state(&viewport_state)
                        .rasterization_state(&rasterizer)
                        .multisample_state(&multisampling)
                        .depth_stencil_state(&depth_stencil)
                        .color_blend_state(&color_blending)
                        .layout(pipeline_layout)
                        .render_pass(render_pass)
                        .subpass(0)],
                    None,
                )
                .map_err(|e| format!("Failed to create mesh shader pipeline: {:?}", e))?[0]
        };

        info!(
            "MeshShaderRenderer initialized: {}x{}, {} meshlets",
            width, height, meshlet_count
        );

        Ok(MeshShaderRenderer {
            pipeline,
            pipeline_layout,
            descriptor_pool,
            ds_layouts,
            descriptor_sets,
            mesh_module,
            frag_module,
            render_pass,
            framebuffer,
            color_image,
            depth_image,
            width,
            height,
            meshlet_count,
            meshlet_desc_buffer,
            meshlet_verts_buffer,
            meshlet_tris_buffer,
            positions_buffer,
            normals_buffer,
            tex_coords_buffer,
            tangents_buffer,
            mvp_buffer,
            light_buffer,
            sampler,
            default_texture,
            default_black_texture,
            default_normal_texture,
            texture_images,
            ibl_assets,
            auto_eye,
            auto_target,
            auto_up,
            auto_far,
            has_scene_bounds,
        })
    }

    /// Update camera uniforms (MVP + light view_pos).
    pub fn update_camera_internal(
        &mut self,
        eye: glam::Vec3,
        target: glam::Vec3,
        up: glam::Vec3,
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) {
        let model = glam::Mat4::IDENTITY;
        let view = crate::camera::look_at(eye, target, up);
        let proj = crate::camera::perspective(fov_y, aspect, near, far);

        let mut mvp_data = [0u8; 192];
        mvp_data[0..64].copy_from_slice(bytemuck::cast_slice(model.as_ref()));
        mvp_data[64..128].copy_from_slice(bytemuck::cast_slice(view.as_ref()));
        mvp_data[128..192].copy_from_slice(bytemuck::cast_slice(proj.as_ref()));

        if let Some(ref mut alloc) = self.mvp_buffer.allocation {
            if let Some(mapped) = alloc.mapped_slice_mut() {
                mapped[..192].copy_from_slice(&mvp_data);
            }
        }

        // Update view_pos in light buffer (offset 16, size 12)
        if let Some(ref mut alloc) = self.light_buffer.allocation {
            if let Some(mapped) = alloc.mapped_slice_mut() {
                let bytes = bytemuck::cast_slice::<f32, u8>(eye.as_ref());
                mapped[16..28].copy_from_slice(bytes);
            }
        }
    }

    /// Record mesh shader rendering commands.
    pub fn render_frame(
        &self,
        ctx: &VulkanContext,
    ) -> Result<vk::CommandBuffer, String> {
        let device = &ctx.device;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe {
            device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| format!("Failed to allocate mesh command buffer: {:?}", e))?[0]
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| format!("Failed to begin mesh command buffer: {:?}", e))?;
        }

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.05, 0.05, 0.08, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: self.width,
                    height: self.height,
                },
            })
            .clear_values(&clear_values);

        let ms_loader = ctx
            .mesh_shader_loader
            .as_ref()
            .ok_or("Mesh shader loader not available")?;

        unsafe {
            device.cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &self.descriptor_sets,
                &[],
            );

            // Dispatch mesh shader workgroups: one workgroup per meshlet
            ms_loader.cmd_draw_mesh_tasks(cmd, self.meshlet_count, 1, 1);

            device.cmd_end_render_pass(cmd);
        }

        Ok(cmd)
    }

    /// Blit the offscreen color image to a swapchain image.
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
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            // Blit offscreen -> swapchain
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
                self.color_image.image,
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
                &[],
                &[],
                &[barrier],
            );
        }
    }

    /// Clean up all GPU resources.
    pub fn cleanup(&mut self, ctx: &mut VulkanContext) {
        let device = ctx.device.clone();

        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_render_pass(self.render_pass, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            for (_, layout) in &self.ds_layouts {
                device.destroy_descriptor_set_layout(*layout, None);
            }
            device.destroy_sampler(self.sampler, None);
            device.destroy_shader_module(self.mesh_module, None);
            device.destroy_shader_module(self.frag_module, None);
        }

        self.color_image.destroy(&device, ctx.allocator_mut());
        self.depth_image.destroy(&device, ctx.allocator_mut());
        self.default_texture.destroy(&device, ctx.allocator_mut());
        self.default_black_texture
            .destroy(&device, ctx.allocator_mut());
        self.default_normal_texture
            .destroy(&device, ctx.allocator_mut());
        for (_, mut tex) in self.texture_images.drain() {
            tex.destroy(&device, ctx.allocator_mut());
        }
        self.ibl_assets.destroy(&device, ctx.allocator_mut());
        self.mvp_buffer.destroy(&device, ctx.allocator_mut());
        self.light_buffer.destroy(&device, ctx.allocator_mut());
        self.meshlet_desc_buffer
            .destroy(&device, ctx.allocator_mut());
        self.meshlet_verts_buffer
            .destroy(&device, ctx.allocator_mut());
        self.meshlet_tris_buffer
            .destroy(&device, ctx.allocator_mut());
        self.positions_buffer.destroy(&device, ctx.allocator_mut());
        self.normals_buffer.destroy(&device, ctx.allocator_mut());
        self.tex_coords_buffer
            .destroy(&device, ctx.allocator_mut());
        self.tangents_buffer
            .destroy(&device, ctx.allocator_mut());
    }
}

// ===========================================================================
// Renderer trait implementation for MeshShaderRenderer
// ===========================================================================

impl scene_manager::Renderer for MeshShaderRenderer {
    fn render(&mut self, ctx: &VulkanContext) -> Result<vk::CommandBuffer, String> {
        self.render_frame(ctx)
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
        self.update_camera_internal(eye, target, up, fov_y, aspect, near, far);
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
        self.color_image.image
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
