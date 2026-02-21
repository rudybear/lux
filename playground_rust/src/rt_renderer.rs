//! Ray tracing renderer: builds acceleration structures, creates RT pipeline, dispatches rays.

use ash::vk;
use bytemuck;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use log::info;
use std::collections::HashMap;
use std::path::Path;

use crate::camera::DefaultCamera;
use crate::reflected_pipeline;
use crate::scene;
use crate::scene_manager::{self, GpuBuffer};
use crate::screenshot;
use crate::spv_loader;
use crate::vulkan_context::VulkanContext;

/// Ray tracing renderer state.
pub struct RTRenderer {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layouts: HashMap<u32, vk::DescriptorSetLayout>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    blas: vk::AccelerationStructureKHR,
    blas_buffer: vk::Buffer,
    blas_allocation: Option<Allocation>,

    tlas: vk::AccelerationStructureKHR,
    tlas_buffer: vk::Buffer,
    tlas_allocation: Option<Allocation>,

    sbt_buffer: vk::Buffer,
    sbt_allocation: Option<Allocation>,
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,

    pub storage_image: vk::Image,
    storage_image_view: vk::ImageView,
    storage_image_allocation: Option<Allocation>,

    camera_buffer: vk::Buffer,
    camera_allocation: Option<Allocation>,

    // SoA storage buffers for closest_hit vertex interpolation
    positions_buffer: GpuBuffer,
    normals_buffer: GpuBuffer,
    tex_coords_buffer: GpuBuffer,
    index_storage_buffer: GpuBuffer,

    // Texture images for RT (image, allocation, view, sampler)
    texture_images: Vec<(vk::Image, Option<Allocation>, vk::ImageView, vk::Sampler)>,

    pub width: u32,
    pub height: u32,

    // Auto-camera from scene bounds (for interactive orbit init)
    pub has_scene_bounds: bool,
    pub auto_eye: glam::Vec3,
    pub auto_target: glam::Vec3,
    pub auto_up: glam::Vec3,
    pub auto_far: f32,
}

/// Render using the ray tracing pipeline and save the result as a PNG.
pub fn render_rt(
    ctx: &mut VulkanContext,
    shader_base: &str,
    scene_source: &str,
    width: u32,
    height: u32,
    output_path: &Path,
) -> Result<(), String> {
    info!("RT render: {}x{}, scene='{}'", width, height, scene_source);

    if !ctx.supports_rt() {
        return Err("Ray tracing is not supported on this GPU".to_string());
    }

    let mut renderer = RTRenderer::new(ctx, shader_base, scene_source, width, height)?;

    // Record command buffer
    let device_clone = ctx.device.clone();
    let cmd = ctx.begin_single_commands()?;
    renderer.render_internal(&device_clone, ctx.rt_pipeline_loader.as_ref().unwrap(), cmd);

    // Copy storage image to staging buffer for readback
    let mut staging =
        screenshot::StagingBuffer::new(&device_clone, ctx.allocator_mut(), width, height)?;

    screenshot::cmd_copy_image_to_buffer(
        &device_clone,
        cmd,
        renderer.storage_image,
        staging.buffer,
        width,
        height,
    );

    ctx.end_single_commands(cmd)?;

    let pixels = staging.read_pixels(width, height)?;
    screenshot::save_png(&pixels, width, height, output_path)?;

    info!("Saved RT render to {:?}", output_path);

    // Cleanup
    staging.destroy(&device_clone, ctx.allocator_mut());
    renderer.destroy_internal(ctx);

    Ok(())
}

impl RTRenderer {
    /// Create a new RT renderer: builds AS, creates pipeline, SBT, descriptors.
    ///
    /// Descriptor set layouts are driven by reflection JSON sidecar files.
    /// If `scene_source` ends with `.glb` or `.gltf`, the BLAS is built from
    /// the glTF mesh; otherwise a procedural sphere is used.
    pub fn new(
        ctx: &mut VulkanContext,
        shader_base: &str,
        scene_source: &str,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        // Verify RT support is available
        if ctx.accel_struct_loader.is_none() {
            return Err("Acceleration structure loader not available".to_string());
        }
        if ctx.rt_pipeline_loader.is_none() {
            return Err("RT pipeline loader not available".to_string());
        }
        if ctx.rt_properties.is_none() {
            return Err("RT properties not available".to_string());
        }

        // Clone device to avoid borrow conflicts with allocator_mut()
        let device = ctx.device.clone();

        // --- 0. Load reflection JSON for all RT stages ---
        let rgen_json_path = format!("{}.rgen.json", shader_base);
        let rchit_json_path = format!("{}.rchit.json", shader_base);
        let rmiss_json_path = format!("{}.rmiss.json", shader_base);

        let mut reflections: Vec<reflected_pipeline::ReflectionData> = Vec::new();
        for json_path in &[&rgen_json_path, &rchit_json_path, &rmiss_json_path] {
            if Path::new(json_path).exists() {
                let refl = reflected_pipeline::load_reflection(Path::new(json_path))?;
                info!("Loaded RT reflection: {} (sets: {:?})", json_path,
                    refl.descriptor_sets.keys().collect::<Vec<_>>());
                reflections.push(refl);
            } else {
                info!("RT reflection file not found (optional): {}", json_path);
            }
        }

        // Merge descriptor sets from all RT stages
        let refl_refs: Vec<&reflected_pipeline::ReflectionData> = reflections.iter().collect();
        let merged = reflected_pipeline::merge_descriptor_sets(&refl_refs);

        // Create descriptor set layouts from reflection
        let descriptor_set_layouts = unsafe {
            reflected_pipeline::create_descriptor_set_layouts_from_merged(&device, &merged)?
        };

        // Compute pool sizes
        let (pool_sizes, max_sets) = reflected_pipeline::compute_pool_sizes(&merged);

        info!("RT reflection-driven descriptors: {} sets, {} pool size entries", max_sets, pool_sizes.len());

        // Create pipeline layout from reflection
        let pipeline_layout = unsafe {
            reflected_pipeline::create_pipeline_layout_from_merged(&device, &descriptor_set_layouts, &[])?
        };

        // --- 1. Create storage image (RGBA8_UNORM, STORAGE | TRANSFER_SRC) ---
        let (storage_image, storage_image_view, storage_image_allocation) =
            create_storage_image(&device, ctx.allocator_mut(), width, height)?;

        // --- 2. Create camera UBO (128 bytes: inverse_view + inverse_proj) ---
        // Auto-camera will be set after loading vertices (below); use default for now
        let (camera_buffer, mut camera_allocation) =
            create_camera_ubo(&device, ctx.allocator_mut(), width, height, None)?;

        // --- 3. Build BLAS from scene geometry ---
        let is_gltf = scene_source.ends_with(".glb") || scene_source.ends_with(".gltf");

        let mut gltf_scene_opt: Option<crate::gltf_loader::GltfScene> = if is_gltf {
            info!("Loading glTF mesh for RT BLAS: {}", scene_source);
            let gs = crate::gltf_loader::load_gltf(Path::new(scene_source))?;
            if gs.meshes.is_empty() {
                return Err(format!("No meshes found in glTF file: {}", scene_source));
            }
            Some(gs)
        } else {
            None
        };

        let (vertices, indices): (Vec<scene::PbrVertex>, Vec<u32>) = if let Some(ref mut gltf_scene) = gltf_scene_opt {
            // Flatten scene to get draw items with world transforms (matches raster renderer)
            let draw_items = crate::gltf_loader::flatten_scene(gltf_scene);
            let mut all_verts: Vec<scene::PbrVertex> = Vec::new();
            let mut all_indices: Vec<u32> = Vec::new();
            let mut vertex_offset: u32 = 0;
            for item in &draw_items {
                let mesh = &gltf_scene.meshes[item.mesh_index];
                let world = item.world_transform;
                let normal_mat = glam::Mat3::from_mat4(world).inverse().transpose();
                for v in &mesh.vertices {
                    let pos = world.transform_point3(glam::Vec3::from(v.position));
                    let norm = (normal_mat * glam::Vec3::from(v.normal)).normalize();
                    all_verts.push(scene::PbrVertex {
                        position: pos.into(),
                        normal: norm.into(),
                        uv: v.uv,
                    });
                }
                for &idx in &mesh.indices {
                    all_indices.push(idx + vertex_offset);
                }
                vertex_offset += mesh.vertices.len() as u32;
            }
            (all_verts, all_indices)
        } else {
            scene::generate_sphere(32, 32)
        };

        let (blas, blas_buffer, blas_allocation, blas_device_address) =
            build_blas(ctx, &vertices, &indices)?;

        // Create SoA storage buffers for closest_hit vertex interpolation
        let positions_data: Vec<[f32; 4]> = vertices.iter()
            .map(|v| [v.position[0], v.position[1], v.position[2], 1.0])
            .collect();
        let normals_data: Vec<[f32; 4]> = vertices.iter()
            .map(|v| [v.normal[0], v.normal[1], v.normal[2], 0.0])
            .collect();
        let tex_coords_data: Vec<[f32; 2]> = vertices.iter()
            .map(|v| v.uv)
            .collect();

        let positions_buffer = create_device_address_buffer(
            &device, ctx.allocator_mut(),
            bytemuck::cast_slice(&positions_data),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "rt_positions",
        )?;
        let normals_buffer = create_device_address_buffer(
            &device, ctx.allocator_mut(),
            bytemuck::cast_slice(&normals_data),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "rt_normals",
        )?;
        let tex_coords_buffer = create_device_address_buffer(
            &device, ctx.allocator_mut(),
            bytemuck::cast_slice(&tex_coords_data),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "rt_tex_coords",
        )?;
        let index_storage_buffer = create_device_address_buffer(
            &device, ctx.allocator_mut(),
            bytemuck::cast_slice(&indices),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "rt_indices",
        )?;

        info!(
            "Created SoA storage buffers: positions={}B, normals={}B, texCoords={}B, indices={}B",
            positions_data.len() * 16, normals_data.len() * 16,
            tex_coords_data.len() * 8, indices.len() * 4
        );

        // Upload glTF textures and create default textures
        let mut texture_images: Vec<(vk::Image, Option<Allocation>, vk::ImageView, vk::Sampler)> = Vec::new();
        let mut texture_map: HashMap<String, (vk::ImageView, vk::Sampler)> = HashMap::new();

        // Create shared sampler for textures
        let tex_sampler = {
            let sampler_create = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT);
            unsafe {
                device
                    .create_sampler(&sampler_create, None)
                    .map_err(|e| format!("Failed to create RT texture sampler: {:?}", e))?
            }
        };

        // Create 1x1 white default texture (for base_color, metallic_roughness, occlusion)
        let white_pixel: [u8; 4] = [255, 255, 255, 255];
        let default_white = create_rt_texture_image(ctx, 1, 1, &white_pixel, "rt_default_white")?;
        let default_white_view = default_white.1;
        texture_images.push((default_white.0, Some(default_white.2), default_white.1, tex_sampler));

        // Create 1x1 black default texture (for emissive)
        let black_pixel: [u8; 4] = [0, 0, 0, 255];
        let default_black = create_rt_texture_image(ctx, 1, 1, &black_pixel, "rt_default_black")?;
        let default_black_view = default_black.1;
        texture_images.push((default_black.0, Some(default_black.2), default_black.1, tex_sampler));

        // Upload glTF material textures
        if let Some(ref gltf_scene) = gltf_scene_opt {
            let gltf_material = if !gltf_scene.materials.is_empty() {
                Some(&gltf_scene.materials[0])
            } else {
                None
            };

            let tex_slots: &[(&str, Box<dyn Fn(&crate::gltf_loader::GltfMaterial) -> &Option<crate::gltf_loader::TextureImage>>)] = &[
                ("base_color_tex", Box::new(|m: &crate::gltf_loader::GltfMaterial| &m.base_color_image)),
                ("metallic_roughness_tex", Box::new(|m: &crate::gltf_loader::GltfMaterial| &m.metallic_roughness_image)),
                ("occlusion_tex", Box::new(|m: &crate::gltf_loader::GltfMaterial| &m.occlusion_image)),
                ("emissive_tex", Box::new(|m: &crate::gltf_loader::GltfMaterial| &m.emissive_image)),
            ];

            for (tex_name, accessor) in tex_slots {
                let tex_data_opt = gltf_material.and_then(|m| accessor(m).as_ref());

                if let Some(tex_data) = tex_data_opt {
                    info!(
                        "RT: Uploading texture '{}': {}x{}",
                        tex_name, tex_data.width, tex_data.height
                    );
                    let gpu_tex = create_rt_texture_image(
                        ctx,
                        tex_data.width,
                        tex_data.height,
                        &tex_data.pixels,
                        tex_name,
                    )?;
                    let view = gpu_tex.1;
                    texture_images.push((gpu_tex.0, Some(gpu_tex.2), gpu_tex.1, tex_sampler));
                    texture_map.insert(tex_name.to_string(), (view, tex_sampler));
                } else {
                    // Use appropriate default
                    let fallback_view = if *tex_name == "emissive_tex" {
                        default_black_view
                    } else {
                        default_white_view
                    };
                    texture_map.insert(tex_name.to_string(), (fallback_view, tex_sampler));
                }
            }
        } else {
            // Non-glTF scene: all textures get defaults
            texture_map.insert("base_color_tex".to_string(), (default_white_view, tex_sampler));
            texture_map.insert("metallic_roughness_tex".to_string(), (default_white_view, tex_sampler));
            texture_map.insert("occlusion_tex".to_string(), (default_white_view, tex_sampler));
            texture_map.insert("emissive_tex".to_string(), (default_black_view, tex_sampler));
        }

        // Load IBL assets (cubemaps + BRDF LUT)
        load_rt_ibl_assets(ctx, &mut texture_map, &mut texture_images);

        // Compute scene bounds and update camera UBO with auto-camera (shared function)
        // Get draw_items from glTF scene for node-transform camera
        let draw_items: Vec<crate::gltf_loader::DrawItem> = if let Some(ref mut gltf_s) = gltf_scene_opt {
            crate::gltf_loader::flatten_scene(gltf_s)
        } else {
            Vec::new()
        };

        let vertex_positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
        let (saved_eye, saved_target, saved_up, saved_far) = scene_manager::compute_auto_camera_from_draw_items(
            &vertex_positions,
            &draw_items,
        );
        {
            // Overwrite the camera UBO with auto-camera data
            let aspect = width as f32 / height as f32;
            let fov_y = DefaultCamera::FOV_Y_DEG.to_radians();
            let view = crate::camera::look_at(saved_eye, saved_target, saved_up);
            let proj = crate::camera::perspective(fov_y, aspect, DefaultCamera::NEAR, saved_far);
            let inv_view = view.inverse();
            let inv_proj = proj.inverse();

            let mut ubo_data = [0u8; 128];
            ubo_data[0..64].copy_from_slice(bytemuck::cast_slice(inv_view.as_ref()));
            ubo_data[64..128].copy_from_slice(bytemuck::cast_slice(inv_proj.as_ref()));

            if let Some(mapped) = camera_allocation.mapped_slice_mut() {
                mapped[..128].copy_from_slice(&ubo_data);
            }
        }

        // --- 4. Build TLAS with single instance ---
        let (tlas, tlas_buffer, tlas_allocation) =
            build_tlas(ctx, blas_device_address)?;

        // --- 5. Load shader modules ---
        let rgen_path = format!("{}.rgen.spv", shader_base);
        let rchit_path = format!("{}.rchit.spv", shader_base);
        let rmiss_path = format!("{}.rmiss.spv", shader_base);

        let rgen_code = spv_loader::load_spirv(Path::new(&rgen_path))?;
        let rchit_code = spv_loader::load_spirv(Path::new(&rchit_path))?;
        let rmiss_code = spv_loader::load_spirv(Path::new(&rmiss_path))?;

        let rgen_module = spv_loader::create_shader_module(&device, &rgen_code)?;
        let rchit_module = spv_loader::create_shader_module(&device, &rchit_code)?;
        let rmiss_module = spv_loader::create_shader_module(&device, &rmiss_code)?;

        // --- 6. Create RT pipeline ---
        let rt_pipeline_loader = ctx.rt_pipeline_loader.as_ref().unwrap();
        let pipeline = create_rt_pipeline(
            &device,
            rt_pipeline_loader,
            rgen_module,
            rmiss_module,
            rchit_module,
            pipeline_layout,
        )?;

        // Destroy shader modules (no longer needed after pipeline creation)
        unsafe {
            device.destroy_shader_module(rgen_module, None);
            device.destroy_shader_module(rchit_module, None);
            device.destroy_shader_module(rmiss_module, None);
        }

        // --- 7. Create SBT ---
        let rt_props = ctx.rt_properties.as_ref().unwrap().clone();
        let rt_pipeline_loader_clone = ctx.rt_pipeline_loader.clone().unwrap();
        let (sbt_buffer, sbt_allocation, raygen_region, miss_region, hit_region) =
            create_sbt(
                &device,
                ctx.allocator_mut(),
                &rt_pipeline_loader_clone,
                &rt_props,
                pipeline,
            )?;

        // --- 8. Create descriptor pool and allocate sets from reflection ---
        let descriptor_pool = unsafe {
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(max_sets)
                .pool_sizes(&pool_sizes);
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| format!("Failed to create RT descriptor pool: {:?}", e))?
        };

        let max_set_idx = descriptor_set_layouts.keys().copied().max().unwrap_or(0);
        let mut ordered_layouts: Vec<vk::DescriptorSetLayout> = Vec::new();
        for i in 0..=max_set_idx {
            if let Some(&layout) = descriptor_set_layouts.get(&i) {
                ordered_layouts.push(layout);
            }
        }

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&ordered_layouts);

        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| format!("Failed to allocate RT descriptor sets: {:?}", e))?
        };

        // Build map: set_index -> descriptor_set
        let mut ds_map: HashMap<u32, vk::DescriptorSet> = HashMap::new();
        {
            let mut ds_idx = 0;
            for i in 0..=max_set_idx {
                if descriptor_set_layouts.contains_key(&i) {
                    ds_map.insert(i, descriptor_sets[ds_idx]);
                    ds_idx += 1;
                }
            }
        }

        // --- 9. Write descriptors from reflection data ---
        let ssbo_map: HashMap<String, (vk::Buffer, u64)> = [
            ("positions".to_string(), (positions_buffer.buffer, positions_data.len() as u64 * 16)),
            ("normals".to_string(), (normals_buffer.buffer, normals_data.len() as u64 * 16)),
            ("tex_coords".to_string(), (tex_coords_buffer.buffer, tex_coords_data.len() as u64 * 8)),
            ("indices".to_string(), (index_storage_buffer.buffer, indices.len() as u64 * 4)),
        ].into_iter().collect();

        write_rt_descriptors(
            &device,
            &merged,
            &ds_map,
            tlas,
            storage_image_view,
            camera_buffer,
            &ssbo_map,
            &texture_map,
        )?;

        info!("RT renderer initialized successfully (reflection-driven)");

        Ok(RTRenderer {
            pipeline,
            pipeline_layout,
            descriptor_set_layouts,
            descriptor_pool,
            descriptor_sets,
            blas,
            blas_buffer,
            blas_allocation: Some(blas_allocation),
            tlas,
            tlas_buffer,
            tlas_allocation: Some(tlas_allocation),
            sbt_buffer,
            sbt_allocation: Some(sbt_allocation),
            raygen_region,
            miss_region,
            hit_region,
            storage_image,
            storage_image_view,
            storage_image_allocation: Some(storage_image_allocation),
            camera_buffer,
            camera_allocation: Some(camera_allocation),
            positions_buffer,
            normals_buffer,
            tex_coords_buffer,
            index_storage_buffer,
            texture_images,
            width,
            height,
            has_scene_bounds: true,
            auto_eye: saved_eye,
            auto_target: saved_target,
            auto_up: saved_up,
            auto_far: saved_far,
        })
    }

    /// Update the RT camera UBO from orbit camera parameters.
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
        let view = crate::camera::look_at(eye, target, up);
        let proj = crate::camera::perspective(fov_y, aspect, near, far);
        let inv_view = view.inverse();
        let inv_proj = proj.inverse();

        let mut ubo_data = [0u8; 128];
        ubo_data[0..64].copy_from_slice(bytemuck::cast_slice(inv_view.as_ref()));
        ubo_data[64..128].copy_from_slice(bytemuck::cast_slice(inv_proj.as_ref()));

        if let Some(alloc) = &mut self.camera_allocation {
            if let Some(mapped) = alloc.mapped_slice_mut() {
                mapped[..128].copy_from_slice(&ubo_data);
            }
        }
    }

    /// Record ray tracing commands into the given command buffer.
    ///
    /// The storage image is transitioned to GENERAL for the trace, then to
    /// TRANSFER_SRC_OPTIMAL for readback.
    pub fn render_internal(
        &self,
        device: &ash::Device,
        rt_loader: &ash::khr::ray_tracing_pipeline::Device,
        cmd: vk::CommandBuffer,
    ) {
        // Transition storage image UNDEFINED -> GENERAL
        screenshot::cmd_transition_image(
            device,
            cmd,
            self.storage_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        );

        // Bind RT pipeline and descriptor sets
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_layout,
                0,
                &self.descriptor_sets,
                &[],
            );
        }

        // Dispatch rays
        let callable_region = vk::StridedDeviceAddressRegionKHR::default();

        unsafe {
            rt_loader.cmd_trace_rays(
                cmd,
                &self.raygen_region,
                &self.miss_region,
                &self.hit_region,
                &callable_region,
                self.width,
                self.height,
                1,
            );
        }

        // Transition storage image GENERAL -> TRANSFER_SRC_OPTIMAL
        screenshot::cmd_transition_image(
            device,
            cmd,
            self.storage_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::PipelineStageFlags::TRANSFER,
        );
    }

    /// Destroy all resources owned by this renderer.
    pub fn destroy_internal(&mut self, ctx: &mut VulkanContext) {
        unsafe {
            ctx.device.destroy_pipeline(self.pipeline, None);
            ctx.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            ctx.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for (_, layout) in &self.descriptor_set_layouts {
                ctx.device.destroy_descriptor_set_layout(*layout, None);
            }
        }

        // Destroy acceleration structures
        if let Some(as_loader) = ctx.accel_struct_loader.as_ref() {
            unsafe {
                as_loader.destroy_acceleration_structure(self.blas, None);
                as_loader.destroy_acceleration_structure(self.tlas, None);
            }
        }

        // Free BLAS buffer
        if let Some(alloc) = self.blas_allocation.take() {
            let _ = ctx.allocator_mut().free(alloc);
        }
        unsafe {
            ctx.device.destroy_buffer(self.blas_buffer, None);
        }

        // Free TLAS buffer
        if let Some(alloc) = self.tlas_allocation.take() {
            let _ = ctx.allocator_mut().free(alloc);
        }
        unsafe {
            ctx.device.destroy_buffer(self.tlas_buffer, None);
        }

        // Free SBT buffer
        if let Some(alloc) = self.sbt_allocation.take() {
            let _ = ctx.allocator_mut().free(alloc);
        }
        unsafe {
            ctx.device.destroy_buffer(self.sbt_buffer, None);
        }

        // Destroy storage image
        unsafe {
            ctx.device
                .destroy_image_view(self.storage_image_view, None);
        }
        if let Some(alloc) = self.storage_image_allocation.take() {
            let _ = ctx.allocator_mut().free(alloc);
        }
        unsafe {
            ctx.device.destroy_image(self.storage_image, None);
        }

        // Free camera buffer
        if let Some(alloc) = self.camera_allocation.take() {
            let _ = ctx.allocator_mut().free(alloc);
        }
        unsafe {
            ctx.device.destroy_buffer(self.camera_buffer, None);
        }

        // Free SoA storage buffers
        {
            let dev = ctx.device.clone();
            self.positions_buffer.destroy(&dev, ctx.allocator_mut());
            self.normals_buffer.destroy(&dev, ctx.allocator_mut());
            self.tex_coords_buffer.destroy(&dev, ctx.allocator_mut());
            self.index_storage_buffer.destroy(&dev, ctx.allocator_mut());
        }

        // Free texture images and samplers
        {
            // Collect unique samplers to avoid double-destroy
            let mut destroyed_samplers = std::collections::HashSet::new();
            for (image, alloc, view, sampler) in self.texture_images.drain(..) {
                unsafe {
                    ctx.device.destroy_image_view(view, None);
                }
                if let Some(a) = alloc {
                    let _ = ctx.allocator_mut().free(a);
                }
                unsafe {
                    ctx.device.destroy_image(image, None);
                    if destroyed_samplers.insert(sampler) {
                        ctx.device.destroy_sampler(sampler, None);
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Renderer trait implementation
// ===========================================================================

impl scene_manager::Renderer for RTRenderer {
    fn render(&mut self, ctx: &VulkanContext) -> Result<vk::CommandBuffer, String> {
        let device = &ctx.device;

        // Allocate a command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe {
            device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| format!("Failed to allocate RT command buffer: {:?}", e))?[0]
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| format!("Failed to begin RT command buffer: {:?}", e))?;
        }

        // Record trace commands using the internal method
        let rt_loader = ctx.rt_pipeline_loader.as_ref()
            .ok_or("RT pipeline loader not available")?;
        self.render_internal(device, rt_loader, cmd);

        // Return the command buffer still recording (so blit_to_swapchain can be appended)
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
        self.update_camera_internal(eye, target, up, fov_y, aspect, near, far);
    }

    fn blit_to_swapchain(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        swap_image: vk::Image,
        extent: vk::Extent2D,
    ) {
        unsafe {
            // Transition swapchain image: UNDEFINED -> TRANSFER_DST
            let barrier = vk::ImageMemoryBarrier::default()
                .image(swap_image)
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
                &[], &[], &[barrier],
            );

            // Blit storage image -> swapchain (storage_image is already in TRANSFER_SRC_OPTIMAL)
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
                        x: extent.width as i32,
                        y: extent.height as i32,
                        z: 1,
                    },
                ]);

            device.cmd_blit_image(
                cmd,
                self.storage_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swap_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit_region],
                vk::Filter::LINEAR,
            );

            // Transition swapchain image: TRANSFER_DST -> PRESENT_SRC
            let barrier = vk::ImageMemoryBarrier::default()
                .image(swap_image)
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

    fn output_image(&self) -> vk::Image {
        self.storage_image
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
        vk::PipelineStageFlags::TRANSFER
    }

    fn destroy(&mut self, ctx: &mut VulkanContext) {
        self.destroy_internal(ctx);
    }
}

// ---------------------------------------------------------------------------
// Private helper functions
// ---------------------------------------------------------------------------

/// Create RGBA8_UNORM storage image for RT output.
fn create_storage_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    width: u32,
    height: u32,
) -> Result<(vk::Image, vk::ImageView, Allocation), String> {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe {
        device
            .create_image(&image_info, None)
            .map_err(|e| format!("Failed to create RT storage image: {:?}", e))?
    };

    let requirements = unsafe { device.get_image_memory_requirements(image) };

    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name: "rt_storage_image",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate RT storage image memory: {:?}", e))?;

    unsafe {
        device
            .bind_image_memory(image, allocation.memory(), allocation.offset())
            .map_err(|e| format!("Failed to bind RT storage image memory: {:?}", e))?;
    }

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .components(vk::ComponentMapping::default())
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );

    let view = unsafe {
        device
            .create_image_view(&view_info, None)
            .map_err(|e| format!("Failed to create RT storage image view: {:?}", e))?
    };

    Ok((image, view, allocation))
}

/// Create camera UBO with inverse view and inverse projection matrices (128 bytes).
fn create_camera_ubo(
    device: &ash::Device,
    allocator: &mut Allocator,
    width: u32,
    height: u32,
    auto_cam: Option<(glam::Vec3, glam::Vec3, glam::Vec3, f32)>,
) -> Result<(vk::Buffer, Allocation), String> {
    let aspect = width as f32 / height as f32;
    let (view, proj) = if let Some((eye, target, up, far)) = auto_cam {
        (
            crate::camera::look_at(eye, target, up),
            crate::camera::perspective(DefaultCamera::FOV_Y_DEG.to_radians(), aspect, DefaultCamera::NEAR, far),
        )
    } else {
        (DefaultCamera::view(), DefaultCamera::projection(aspect))
    };
    let inv_view = view.inverse();
    let inv_proj = proj.inverse();

    let mut ubo_data = [0u8; 128];
    ubo_data[0..64].copy_from_slice(bytemuck::cast_slice(inv_view.as_ref()));
    ubo_data[64..128].copy_from_slice(bytemuck::cast_slice(inv_proj.as_ref()));

    let buffer_info = vk::BufferCreateInfo::default()
        .size(128)
        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        device
            .create_buffer(&buffer_info, None)
            .map_err(|e| format!("Failed to create camera UBO: {:?}", e))?
    };

    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let mut allocation = allocator
        .allocate(&AllocationCreateDesc {
            name: "rt_camera_ubo",
            requirements,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate camera UBO memory: {:?}", e))?;

    unsafe {
        device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            .map_err(|e| format!("Failed to bind camera UBO memory: {:?}", e))?;
    }

    // Copy data
    if let Some(mapped) = allocation.mapped_slice_mut() {
        mapped[..128].copy_from_slice(&ubo_data);
    } else {
        return Err("Camera UBO is not host-visible".to_string());
    }

    Ok((buffer, allocation))
}

/// Build a Bottom-Level Acceleration Structure from sphere mesh triangles.
///
/// Returns (BLAS handle, BLAS buffer, BLAS allocation, BLAS device address).
fn build_blas(
    ctx: &mut VulkanContext,
    vertices: &[scene::PbrVertex],
    indices: &[u32],
) -> Result<(vk::AccelerationStructureKHR, vk::Buffer, Allocation, u64), String> {
    let device = ctx.device.clone();
    let as_loader = ctx
        .accel_struct_loader
        .clone()
        .ok_or("Acceleration structure loader not available")?;

    // Upload vertex and index data to GPU buffers with device address support
    let vertex_data: &[u8] = bytemuck::cast_slice(vertices);
    let index_data: &[u8] = bytemuck::cast_slice(indices);

    let mut vertex_buffer = create_device_address_buffer(
        &device,
        ctx.allocator_mut(),
        vertex_data,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        "blas_vertices",
    )?;

    let mut index_buffer = create_device_address_buffer(
        &device,
        ctx.allocator_mut(),
        index_data,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        "blas_indices",
    )?;

    let vertex_buffer_addr = get_buffer_device_address(&device, vertex_buffer.buffer);
    let index_buffer_addr = get_buffer_device_address(&device, index_buffer.buffer);

    // Geometry description
    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: vertex_buffer_addr,
        })
        .vertex_stride(std::mem::size_of::<scene::PbrVertex>() as u64)
        .max_vertex(vertices.len() as u32 - 1)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: index_buffer_addr,
        });

    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles,
        });

    let primitive_count = indices.len() as u32 / 3;

    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(std::slice::from_ref(&geometry));

    // Query build sizes
    let mut build_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[primitive_count],
            &mut build_sizes,
        );
    }

    info!(
        "BLAS sizes: as={}, scratch={}, build={}",
        build_sizes.acceleration_structure_size,
        build_sizes.build_scratch_size,
        build_sizes.update_scratch_size
    );

    // Allocate BLAS buffer
    let blas_buffer_info = vk::BufferCreateInfo::default()
        .size(build_sizes.acceleration_structure_size)
        .usage(
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let blas_buffer = unsafe {
        device.create_buffer(&blas_buffer_info, None)
            .map_err(|e| format!("Failed to create BLAS buffer: {:?}", e))?
    };

    let blas_reqs = unsafe { device.get_buffer_memory_requirements(blas_buffer) };

    let blas_allocation = ctx
        .allocator_mut()
        .allocate(&AllocationCreateDesc {
            name: "blas_buffer",
            requirements: blas_reqs,
            location: MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate BLAS buffer memory: {:?}", e))?;

    unsafe {
        device
            .bind_buffer_memory(blas_buffer, blas_allocation.memory(), blas_allocation.offset())
            .map_err(|e| format!("Failed to bind BLAS buffer memory: {:?}", e))?;
    }

    // Create acceleration structure
    let as_create_info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(blas_buffer)
        .offset(0)
        .size(build_sizes.acceleration_structure_size)
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

    let blas = unsafe {
        as_loader
            .create_acceleration_structure(&as_create_info, None)
            .map_err(|e| format!("Failed to create BLAS: {:?}", e))?
    };

    // Allocate scratch buffer
    let mut scratch_buffer = create_device_address_buffer_empty(
        &device,
        ctx.allocator_mut(),
        build_sizes.build_scratch_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        "blas_scratch",
    )?;

    let scratch_addr = get_buffer_device_address(&device, scratch_buffer.buffer);

    // Build BLAS in a one-shot command buffer
    let build_info_final = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .dst_acceleration_structure(blas)
        .geometries(std::slice::from_ref(&geometry))
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        });

    let build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
        .primitive_count(primitive_count)
        .primitive_offset(0)
        .first_vertex(0)
        .transform_offset(0);

    let cmd = ctx.begin_single_commands()?;
    unsafe {
        as_loader.cmd_build_acceleration_structures(
            cmd,
            &[build_info_final],
            &[std::slice::from_ref(&build_range)],
        );
    }
    ctx.end_single_commands(cmd)?;

    // Get BLAS device address
    let blas_addr_info =
        vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(blas);
    let blas_device_address = unsafe {
        as_loader.get_acceleration_structure_device_address(&blas_addr_info)
    };

    info!("BLAS built, device address: 0x{:016X}", blas_device_address);

    // Free scratch and input buffers
    scratch_buffer.destroy(&device, ctx.allocator_mut());
    vertex_buffer.destroy(&device, ctx.allocator_mut());
    index_buffer.destroy(&device, ctx.allocator_mut());

    Ok((blas, blas_buffer, blas_allocation, blas_device_address))
}

/// Build a Top-Level Acceleration Structure with a single instance.
fn build_tlas(
    ctx: &mut VulkanContext,
    blas_device_address: u64,
) -> Result<(vk::AccelerationStructureKHR, vk::Buffer, Allocation), String> {
    let device = ctx.device.clone();
    let as_loader = ctx
        .accel_struct_loader
        .clone()
        .ok_or("Acceleration structure loader not available")?;

    // Create instance data (identity transform, mask 0xFF)
    let transform = vk::TransformMatrixKHR {
        matrix: [
            1.0, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, 0.0, // row 2
        ],
    };

    let instance = vk::AccelerationStructureInstanceKHR {
        transform,
        instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0x01), // TRIANGLE_FACING_CULL_DISABLE
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: blas_device_address,
        },
    };

    let instance_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(&instance as *const _ as *const u8, 64) };

    let mut instance_buffer = create_device_address_buffer(
        &device,
        ctx.allocator_mut(),
        instance_bytes,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        "tlas_instances",
    )?;

    let instance_addr = get_buffer_device_address(&device, instance_buffer.buffer);

    // TLAS geometry (instances)
    let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
        .array_of_pointers(false)
        .data(vk::DeviceOrHostAddressConstKHR {
            device_address: instance_addr,
        });

    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: instances_data,
        });

    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(std::slice::from_ref(&geometry));

    let instance_count = 1u32;

    // Query sizes
    let mut build_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[instance_count],
            &mut build_sizes,
        );
    }

    info!(
        "TLAS sizes: as={}, scratch={}",
        build_sizes.acceleration_structure_size, build_sizes.build_scratch_size
    );

    // Allocate TLAS buffer
    let tlas_buffer_info = vk::BufferCreateInfo::default()
        .size(build_sizes.acceleration_structure_size)
        .usage(
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let tlas_buffer = unsafe {
        device.create_buffer(&tlas_buffer_info, None)
            .map_err(|e| format!("Failed to create TLAS buffer: {:?}", e))?
    };

    let tlas_reqs = unsafe { device.get_buffer_memory_requirements(tlas_buffer) };

    let tlas_allocation = ctx
        .allocator_mut()
        .allocate(&AllocationCreateDesc {
            name: "tlas_buffer",
            requirements: tlas_reqs,
            location: MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate TLAS buffer memory: {:?}", e))?;

    unsafe {
        device
            .bind_buffer_memory(tlas_buffer, tlas_allocation.memory(), tlas_allocation.offset())
            .map_err(|e| format!("Failed to bind TLAS buffer memory: {:?}", e))?;
    }

    // Create acceleration structure
    let as_create_info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(tlas_buffer)
        .offset(0)
        .size(build_sizes.acceleration_structure_size)
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

    let tlas = unsafe {
        as_loader
            .create_acceleration_structure(&as_create_info, None)
            .map_err(|e| format!("Failed to create TLAS: {:?}", e))?
    };

    // Allocate scratch buffer
    let mut scratch_buffer = create_device_address_buffer_empty(
        &device,
        ctx.allocator_mut(),
        build_sizes.build_scratch_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        "tlas_scratch",
    )?;

    let scratch_addr = get_buffer_device_address(&device, scratch_buffer.buffer);

    // Build TLAS
    let build_info_final = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .dst_acceleration_structure(tlas)
        .geometries(std::slice::from_ref(&geometry))
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        });

    let build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
        .primitive_count(instance_count)
        .primitive_offset(0)
        .first_vertex(0)
        .transform_offset(0);

    let cmd = ctx.begin_single_commands()?;

    // Memory barrier: ensure BLAS build is complete before TLAS build
    let memory_barrier = vk::MemoryBarrier::default()
        .src_access_mask(
            vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
                | vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        )
        .dst_access_mask(
            vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
                | vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        );

    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::DependencyFlags::empty(),
            &[memory_barrier],
            &[],
            &[],
        );

        as_loader.cmd_build_acceleration_structures(
            cmd,
            &[build_info_final],
            &[std::slice::from_ref(&build_range)],
        );
    }

    ctx.end_single_commands(cmd)?;

    info!("TLAS built successfully");

    // Free scratch and instance buffers
    scratch_buffer.destroy(&device, ctx.allocator_mut());
    instance_buffer.destroy(&device, ctx.allocator_mut());

    Ok((tlas, tlas_buffer, tlas_allocation))
}

/// Create the RT pipeline with raygen, miss, and closest-hit shader groups.
fn create_rt_pipeline(
    _device: &ash::Device,
    rt_loader: &ash::khr::ray_tracing_pipeline::Device,
    rgen_module: vk::ShaderModule,
    rmiss_module: vk::ShaderModule,
    rchit_module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
) -> Result<vk::Pipeline, String> {
    let entry_name = c"main";

    let shader_stages = [
        // Index 0: raygen
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
            .module(rgen_module)
            .name(entry_name),
        // Index 1: miss
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .module(rmiss_module)
            .name(entry_name),
        // Index 2: closest hit
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(rchit_module)
            .name(entry_name),
    ];

    let shader_groups = [
        // Group 0: raygen (general)
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(0)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
        // Group 1: miss (general)
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(1)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
        // Group 2: closest hit (triangles)
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(2)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
    ];

    let pipeline_info = vk::RayTracingPipelineCreateInfoKHR::default()
        .stages(&shader_stages)
        .groups(&shader_groups)
        .max_pipeline_ray_recursion_depth(1)
        .layout(pipeline_layout);

    let pipeline = unsafe {
        rt_loader
            .create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
            .map_err(|e| format!("Failed to create RT pipeline: {:?}", e))?[0]
    };

    info!("RT pipeline created");
    Ok(pipeline)
}

/// Create the Shader Binding Table (SBT) with properly aligned regions.
fn create_sbt(
    device: &ash::Device,
    allocator: &mut Allocator,
    rt_loader: &ash::khr::ray_tracing_pipeline::Device,
    rt_props: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    pipeline: vk::Pipeline,
) -> Result<
    (
        vk::Buffer,
        Allocation,
        vk::StridedDeviceAddressRegionKHR,
        vk::StridedDeviceAddressRegionKHR,
        vk::StridedDeviceAddressRegionKHR,
    ),
    String,
> {
    let handle_size = rt_props.shader_group_handle_size as u64;
    let handle_alignment = rt_props.shader_group_handle_alignment as u64;
    let base_alignment = rt_props.shader_group_base_alignment as u64;

    // Aligned handle size (stride within a group)
    let handle_size_aligned = align_up(handle_size, handle_alignment);

    // Each region is aligned to base_alignment and contains one handle
    let raygen_size = align_up(handle_size_aligned, base_alignment);
    let miss_size = align_up(handle_size_aligned, base_alignment);
    let hit_size = align_up(handle_size_aligned, base_alignment);

    let sbt_total_size = raygen_size + miss_size + hit_size;

    // Get shader group handles
    let group_count = 3u32;
    let handle_data_size = (handle_size as u32 * group_count) as usize;
    let handles = unsafe {
        rt_loader
            .get_ray_tracing_shader_group_handles(pipeline, 0, group_count, handle_data_size)
            .map_err(|e| format!("Failed to get RT shader group handles: {:?}", e))?
    };

    // Create SBT buffer
    let buffer_info = vk::BufferCreateInfo::default()
        .size(sbt_total_size)
        .usage(
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let sbt_buffer = unsafe {
        device
            .create_buffer(&buffer_info, None)
            .map_err(|e| format!("Failed to create SBT buffer: {:?}", e))?
    };

    let requirements = unsafe { device.get_buffer_memory_requirements(sbt_buffer) };

    let mut sbt_allocation = allocator
        .allocate(&AllocationCreateDesc {
            name: "rt_sbt",
            requirements,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate SBT memory: {:?}", e))?;

    unsafe {
        device
            .bind_buffer_memory(sbt_buffer, sbt_allocation.memory(), sbt_allocation.offset())
            .map_err(|e| format!("Failed to bind SBT buffer memory: {:?}", e))?;
    }

    // Map and copy handles at aligned offsets
    if let Some(mapped) = sbt_allocation.mapped_slice_mut() {
        // Zero out
        for byte in mapped[..sbt_total_size as usize].iter_mut() {
            *byte = 0;
        }

        let hs = handle_size as usize;

        // Raygen at offset 0
        mapped[0..hs].copy_from_slice(&handles[0..hs]);

        // Miss at offset raygen_size
        let miss_offset = raygen_size as usize;
        mapped[miss_offset..miss_offset + hs].copy_from_slice(&handles[hs..hs * 2]);

        // Hit at offset raygen_size + miss_size
        let hit_offset = (raygen_size + miss_size) as usize;
        mapped[hit_offset..hit_offset + hs].copy_from_slice(&handles[hs * 2..hs * 3]);
    } else {
        return Err("SBT buffer is not host-visible".to_string());
    }

    let sbt_base_addr = get_buffer_device_address(device, sbt_buffer);

    let raygen_region = vk::StridedDeviceAddressRegionKHR {
        device_address: sbt_base_addr,
        stride: handle_size_aligned,
        size: raygen_size,
    };

    let miss_region = vk::StridedDeviceAddressRegionKHR {
        device_address: sbt_base_addr + raygen_size,
        stride: handle_size_aligned,
        size: miss_size,
    };

    let hit_region = vk::StridedDeviceAddressRegionKHR {
        device_address: sbt_base_addr + raygen_size + miss_size,
        stride: handle_size_aligned,
        size: hit_size,
    };

    info!(
        "SBT created: raygen=0x{:X}, miss=0x{:X}, hit=0x{:X}, total={}",
        raygen_region.device_address,
        miss_region.device_address,
        hit_region.device_address,
        sbt_total_size
    );

    Ok((sbt_buffer, sbt_allocation, raygen_region, miss_region, hit_region))
}

/// Write RT descriptor sets from reflection data.
///
/// Matches resource names from the reflection JSON to actual Vulkan resources:
/// - "Camera" / "Light" uniform_buffer -> camera_buffer (128 bytes)
/// - "tlas" acceleration_structure -> tlas
/// - "output_image" storage_image -> storage_image_view
/// - sampler / sampled_image -> texture_map entries
fn write_rt_descriptors(
    device: &ash::Device,
    merged: &HashMap<u32, Vec<reflected_pipeline::BindingInfo>>,
    ds_map: &HashMap<u32, vk::DescriptorSet>,
    tlas: vk::AccelerationStructureKHR,
    storage_image_view: vk::ImageView,
    camera_buffer: vk::Buffer,
    ssbo_map: &HashMap<String, (vk::Buffer, u64)>,
    texture_map: &HashMap<String, (vk::ImageView, vk::Sampler)>,
) -> Result<(), String> {
    // We need to hold the AS write info alive across the update_descriptor_sets call.
    // Use a Vec of all writes, but handle acceleration structure specially since
    // it needs push_next.

    // First pass: collect non-AS writes
    let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
    let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
    let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();

    // Track AS binding for special handling
    let mut as_bindings: Vec<(u32, u32)> = Vec::new(); // (set_idx, binding)

    // Pre-allocate to keep pointers stable
    let total_bindings: usize = merged.values().map(|v| v.len()).sum();
    buffer_infos.reserve(total_bindings);
    image_infos.reserve(total_bindings);

    for (&set_idx, bindings) in merged {
        let ds = match ds_map.get(&set_idx) {
            Some(&ds) => ds,
            None => continue,
        };

        for binding in bindings {
            let vk_type = reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);

            match vk_type {
                vk::DescriptorType::UNIFORM_BUFFER => {
                    let range = if binding.size > 0 { binding.size as u64 } else { 128 };
                    let buf_info_idx = buffer_infos.len();
                    buffer_infos.push(
                        vk::DescriptorBufferInfo::default()
                            .buffer(camera_buffer)
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
                                    &buffer_infos[buf_info_idx] as *const _,
                                    1,
                                )
                            }),
                    );
                }
                vk::DescriptorType::STORAGE_IMAGE => {
                    let img_info_idx = image_infos.len();
                    image_infos.push(
                        vk::DescriptorImageInfo::default()
                            .image_view(storage_image_view)
                            .image_layout(vk::ImageLayout::GENERAL),
                    );
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(ds)
                            .dst_binding(binding.binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(unsafe {
                                std::slice::from_raw_parts(
                                    &image_infos[img_info_idx] as *const _,
                                    1,
                                )
                            }),
                    );
                }
                vk::DescriptorType::STORAGE_BUFFER => {
                    if let Some(&(buf, size)) = ssbo_map.get(&binding.name) {
                        let buf_info_idx = buffer_infos.len();
                        buffer_infos.push(
                            vk::DescriptorBufferInfo::default()
                                .buffer(buf)
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
                                        &buffer_infos[buf_info_idx] as *const _,
                                        1,
                                    )
                                }),
                        );
                    } else {
                        info!("Unknown storage buffer name: '{}'", binding.name);
                    }
                }
                vk::DescriptorType::SAMPLER => {
                    // Look up the sampler from texture_map by name (strip _sampler suffix if present)
                    let tex_name = binding.name.trim_end_matches("_sampler").to_string();
                    let actual_sampler = texture_map
                        .get(&tex_name)
                        .or_else(|| texture_map.get(&binding.name))
                        .map(|t| t.1)
                        .unwrap_or_else(|| {
                            // Fall back to any sampler in the map
                            texture_map.values().next().map(|t| t.1).unwrap_or(vk::Sampler::null())
                        });

                    let img_info_idx = image_infos.len();
                    image_infos.push(
                        vk::DescriptorImageInfo::default().sampler(actual_sampler),
                    );

                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(ds)
                            .dst_binding(binding.binding)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .image_info(unsafe {
                                std::slice::from_raw_parts(
                                    &image_infos[img_info_idx] as *const _,
                                    1,
                                )
                            }),
                    );
                }
                vk::DescriptorType::SAMPLED_IMAGE => {
                    // Look up the image view from texture_map by name (strip _image suffix if present)
                    let tex_name = binding.name.trim_end_matches("_image").to_string();
                    let view = texture_map
                        .get(&tex_name)
                        .or_else(|| texture_map.get(&binding.name))
                        .map(|t| t.0)
                        .unwrap_or_else(|| {
                            // Fall back to first texture view in the map
                            texture_map.values().next().map(|t| t.0).unwrap_or(vk::ImageView::null())
                        });

                    let img_info_idx = image_infos.len();
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
                                    &image_infos[img_info_idx] as *const _,
                                    1,
                                )
                            }),
                    );
                }
                vk::DescriptorType::ACCELERATION_STRUCTURE_KHR => {
                    as_bindings.push((set_idx, binding.binding));
                }
                _ => {
                    info!(
                        "Unhandled RT descriptor type {:?} at set={} binding={} name='{}'",
                        vk_type, set_idx, binding.binding, binding.name
                    );
                }
            }
        }
    }

    // Write non-AS descriptors first
    if !writes.is_empty() {
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    // Write acceleration structure descriptors (needs push_next, done separately)
    for (set_idx, binding_idx) in &as_bindings {
        let ds = match ds_map.get(set_idx) {
            Some(&ds) => ds,
            None => continue,
        };

        let tlas_array = [tlas];
        let mut as_write_info =
            vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(&tlas_array);

        let as_write = vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(*binding_idx)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .push_next(&mut as_write_info);

        unsafe {
            device.update_descriptor_sets(&[as_write], &[]);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// IBL cubemap utilities
// ---------------------------------------------------------------------------

/// Create a cubemap image for IBL (Float16 RGBA, 6 layers, optional mipmaps).
fn create_rt_cubemap_image(
    ctx: &mut VulkanContext,
    face_size: u32,
    mip_count: u32,
    data: &[u8],
    name: &str,
) -> Result<(vk::Image, vk::ImageView, Allocation, vk::Sampler), String> {
    let device = ctx.device.clone();

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R16G16B16A16_SFLOAT)
        .extent(vk::Extent3D { width: face_size, height: face_size, depth: 1 })
        .mip_levels(mip_count)
        .array_layers(6)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);

    let image = unsafe {
        device.create_image(&image_info, None)
            .map_err(|e| format!("Failed to create cubemap '{}': {:?}", name, e))?
    };

    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let allocation = ctx.allocator_mut()
        .allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate cubemap '{}': {:?}", name, e))?;

    unsafe {
        device.bind_image_memory(image, allocation.memory(), allocation.offset())
            .map_err(|e| format!("Failed to bind cubemap '{}': {:?}", name, e))?;
    }

    // Staging buffer
    let staging_info = vk::BufferCreateInfo::default()
        .size(data.len() as u64)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC);
    let staging_buffer = unsafe {
        device.create_buffer(&staging_info, None)
            .map_err(|e| format!("Failed to create staging for '{}': {:?}", name, e))?
    };
    let staging_reqs = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
    let mut staging_alloc = ctx.allocator_mut()
        .allocate(&AllocationCreateDesc {
            name: "ibl_staging",
            requirements: staging_reqs,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("staging alloc failed: {:?}", e))?;
    unsafe {
        device.bind_buffer_memory(staging_buffer, staging_alloc.memory(), staging_alloc.offset())
            .map_err(|e| format!("staging bind failed: {:?}", e))?;
    }
    if let Some(mapped) = staging_alloc.mapped_slice_mut() {
        mapped[..data.len()].copy_from_slice(data);
    }

    // Transfer
    let cmd = ctx.begin_single_commands()?;

    screenshot::cmd_transition_image_layers(
        &device, cmd, image,
        vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER,
        mip_count, 6,
    );

    // Copy each mip level, each face
    let mut offset: u64 = 0;
    let mut regions = Vec::new();
    for mip in 0..mip_count {
        let mip_size = (face_size >> mip).max(1);
        let face_bytes = (mip_size as u64) * (mip_size as u64) * 8; // 4 channels * 2 bytes
        for face in 0..6u32 {
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(offset)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(mip)
                            .base_array_layer(face)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D { width: mip_size, height: mip_size, depth: 1 }),
            );
            offset += face_bytes;
        }
    }
    unsafe {
        device.cmd_copy_buffer_to_image(cmd, staging_buffer, image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions);
    }

    screenshot::cmd_transition_image_layers(
        &device, cmd, image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        mip_count, 6,
    );

    ctx.end_single_commands(cmd)?;

    let _ = ctx.allocator_mut().free(staging_alloc);
    unsafe { device.destroy_buffer(staging_buffer, None); }

    // Cube image view
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::CUBE)
        .format(vk::Format::R16G16B16A16_SFLOAT)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(mip_count)
                .base_array_layer(0)
                .layer_count(6),
        );
    let view = unsafe {
        device.create_image_view(&view_info, None)
            .map_err(|e| format!("Failed to create cubemap view '{}': {:?}", name, e))?
    };

    // Sampler with mipmap support
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .max_lod(mip_count as f32);
    let sampler = unsafe {
        device.create_sampler(&sampler_info, None)
            .map_err(|e| format!("Failed to create cubemap sampler '{}': {:?}", name, e))?
    };

    Ok((image, view, allocation, sampler))
}

/// Load IBL assets (specular, irradiance cubemaps + BRDF LUT) and add to texture_map.
fn load_rt_ibl_assets(
    ctx: &mut VulkanContext,
    texture_map: &mut HashMap<String, (vk::ImageView, vk::Sampler)>,
    texture_images: &mut Vec<(vk::Image, Option<Allocation>, vk::ImageView, vk::Sampler)>,
) {
    let ibl_base = std::path::Path::new("assets/ibl");
    if !ibl_base.exists() {
        info!("No IBL assets directory, skipping RT IBL");
        return;
    }

    // Find IBL directory: prefer "pisa" then "neutral"
    let preferred = ["pisa", "neutral"];
    let ibl_dir = preferred.iter()
        .map(|n| ibl_base.join(n))
        .find(|p| p.join("manifest.json").exists());

    let ibl_dir = match ibl_dir {
        Some(d) => d,
        None => {
            info!("No IBL manifest found");
            return;
        }
    };

    info!("Loading RT IBL assets from {:?}", ibl_dir);

    // Parse manifest
    let manifest_str = match std::fs::read_to_string(ibl_dir.join("manifest.json")) {
        Ok(s) => s,
        Err(_) => { info!("Failed to read IBL manifest"); return; }
    };
    let manifest: serde_json::Value = match serde_json::from_str(&manifest_str) {
        Ok(v) => v,
        Err(_) => { info!("Failed to parse IBL manifest"); return; }
    };

    let spec_face_size = manifest["specular"]["face_size"].as_u64().unwrap_or(256) as u32;
    let spec_mip_count = manifest["specular"]["mip_count"].as_u64().unwrap_or(9) as u32;
    let irr_face_size = manifest["irradiance"]["face_size"].as_u64().unwrap_or(32) as u32;
    let brdf_size = manifest["brdf_lut"]["size"].as_u64().unwrap_or(512) as u32;

    // Load specular cubemap
    let spec_path = ibl_dir.join("specular.bin");
    if spec_path.exists() {
        if let Ok(data) = std::fs::read(&spec_path) {
            match create_rt_cubemap_image(ctx, spec_face_size, spec_mip_count, &data, "rt_ibl_specular") {
                Ok((image, view, alloc, sampler)) => {
                    texture_map.insert("env_specular".to_string(), (view, sampler));
                    texture_images.push((image, Some(alloc), view, sampler));
                    info!("Loaded RT IBL specular: {}x{}, {} mips", spec_face_size, spec_face_size, spec_mip_count);
                }
                Err(e) => info!("Failed to load RT IBL specular: {}", e),
            }
        }
    }

    // Load irradiance cubemap
    let irr_path = ibl_dir.join("irradiance.bin");
    if irr_path.exists() {
        if let Ok(data) = std::fs::read(&irr_path) {
            match create_rt_cubemap_image(ctx, irr_face_size, 1, &data, "rt_ibl_irradiance") {
                Ok((image, view, alloc, sampler)) => {
                    texture_map.insert("env_irradiance".to_string(), (view, sampler));
                    texture_images.push((image, Some(alloc), view, sampler));
                    info!("Loaded RT IBL irradiance: {}x{}", irr_face_size, irr_face_size);
                }
                Err(e) => info!("Failed to load RT IBL irradiance: {}", e),
            }
        }
    }

    // Load BRDF LUT (2D, RG16F padded to RGBA16F)
    let brdf_path = ibl_dir.join("brdf_lut.bin");
    if brdf_path.exists() {
        if let Ok(raw_data) = std::fs::read(&brdf_path) {
            let total_pixels = (brdf_size as usize) * (brdf_size as usize);
            let raw_u16: &[u16] = bytemuck::cast_slice(&raw_data);

            let rgba_data: Vec<u16> = if raw_u16.len() == total_pixels * 2 {
                let mut rgba = vec![0u16; total_pixels * 4];
                for p in 0..total_pixels {
                    rgba[p * 4 + 0] = raw_u16[p * 2 + 0];
                    rgba[p * 4 + 1] = raw_u16[p * 2 + 1];
                    rgba[p * 4 + 2] = 0;
                    rgba[p * 4 + 3] = 0x3C00; // 1.0 in half-float
                }
                rgba
            } else {
                raw_u16.to_vec()
            };

            let rgba_bytes: &[u8] = bytemuck::cast_slice(&rgba_data);

            let device = ctx.device.clone();
            let format = vk::Format::R16G16B16A16_SFLOAT;
            let data_size = rgba_bytes.len() as u64;

            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width: brdf_size, height: brdf_size, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            let image = match unsafe { device.create_image(&image_info, None) } {
                Ok(i) => i,
                Err(e) => { info!("Failed to create BRDF LUT image: {:?}", e); return; }
            };

            let requirements = unsafe { device.get_image_memory_requirements(image) };
            let allocation = match ctx.allocator_mut().allocate(&AllocationCreateDesc {
                name: "rt_brdf_lut",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            }) {
                Ok(a) => a,
                Err(e) => { info!("Failed to allocate BRDF LUT: {:?}", e); return; }
            };

            if unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset()) }.is_err() {
                return;
            }

            // Staging
            let staging_info = vk::BufferCreateInfo::default()
                .size(data_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC);
            let staging_buffer = match unsafe { device.create_buffer(&staging_info, None) } {
                Ok(b) => b,
                Err(_) => return,
            };
            let staging_reqs = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
            let mut staging_alloc = match ctx.allocator_mut().allocate(&AllocationCreateDesc {
                name: "brdf_staging",
                requirements: staging_reqs,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            }) {
                Ok(a) => a,
                Err(_) => return,
            };
            let _ = unsafe { device.bind_buffer_memory(staging_buffer, staging_alloc.memory(), staging_alloc.offset()) };
            if let Some(mapped) = staging_alloc.mapped_slice_mut() {
                mapped[..rgba_bytes.len()].copy_from_slice(rgba_bytes);
            }

            let cmd = match ctx.begin_single_commands() {
                Ok(c) => c,
                Err(_) => return,
            };

            screenshot::cmd_transition_image(
                &device, cmd, image,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER,
            );

            let region = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D { width: brdf_size, height: brdf_size, depth: 1 });

            unsafe {
                device.cmd_copy_buffer_to_image(cmd, staging_buffer, image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
            }

            screenshot::cmd_transition_image(
                &device, cmd, image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            );

            let _ = ctx.end_single_commands(cmd);
            let _ = ctx.allocator_mut().free(staging_alloc);
            unsafe { device.destroy_buffer(staging_buffer, None); }

            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            let view = match unsafe { device.create_image_view(&view_info, None) } {
                Ok(v) => v,
                Err(_) => return,
            };

            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR);
            let sampler = match unsafe { device.create_sampler(&sampler_info, None) } {
                Ok(s) => s,
                Err(_) => return,
            };

            texture_map.insert("brdf_lut".to_string(), (view, sampler));
            texture_images.push((image, Some(allocation), view, sampler));
            info!("Loaded RT IBL BRDF LUT: {}x{}", brdf_size, brdf_size);
        }
    }
}

// ---------------------------------------------------------------------------
// Texture utilities
// ---------------------------------------------------------------------------

/// Create a 2D RGBA8 texture image, upload pixels via staging buffer, and return
/// (vk::Image, vk::ImageView, Allocation).
///
/// The image is transitioned to SHADER_READ_ONLY_OPTIMAL for sampling.
fn create_rt_texture_image(
    ctx: &mut VulkanContext,
    width: u32,
    height: u32,
    pixels: &[u8],
    name: &str,
) -> Result<(vk::Image, vk::ImageView, Allocation), String> {
    let format = vk::Format::R8G8B8A8_UNORM;
    let data_size = (width * height * 4) as u64;
    let device = ctx.device.clone();

    // Create staging buffer
    let staging_info = vk::BufferCreateInfo::default()
        .size(data_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let staging_buffer = unsafe {
        device
            .create_buffer(&staging_info, None)
            .map_err(|e| format!("Failed to create RT texture staging buffer: {:?}", e))?
    };

    let staging_reqs = unsafe { device.get_buffer_memory_requirements(staging_buffer) };

    let mut staging_alloc = ctx
        .allocator_mut()
        .allocate(&AllocationCreateDesc {
            name: "rt_texture_staging",
            requirements: staging_reqs,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate RT texture staging memory: {:?}", e))?;

    unsafe {
        device
            .bind_buffer_memory(staging_buffer, staging_alloc.memory(), staging_alloc.offset())
            .map_err(|e| format!("Failed to bind RT texture staging memory: {:?}", e))?;
    }

    if let Some(mapped) = staging_alloc.mapped_slice_mut() {
        mapped[..pixels.len()].copy_from_slice(pixels);
    }

    // Create GPU image
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe {
        device
            .create_image(&image_info, None)
            .map_err(|e| format!("Failed to create RT texture image '{}': {:?}", name, e))?
    };

    let requirements = unsafe { device.get_image_memory_requirements(image) };

    let allocation = ctx
        .allocator_mut()
        .allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate RT texture image memory '{}': {:?}", name, e))?;

    unsafe {
        device
            .bind_image_memory(image, allocation.memory(), allocation.offset())
            .map_err(|e| format!("Failed to bind RT texture image memory '{}': {:?}", name, e))?;
    }

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(vk::ComponentMapping::default())
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );

    let view = unsafe {
        device
            .create_image_view(&view_info, None)
            .map_err(|e| format!("Failed to create RT texture image view '{}': {:?}", name, e))?
    };

    // Copy staging -> image via command buffer
    let cmd = ctx.begin_single_commands()?;

    screenshot::cmd_transition_image(
        &ctx.device,
        cmd,
        image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );

    let region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D::default())
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    unsafe {
        ctx.device.cmd_copy_buffer_to_image(
            cmd,
            staging_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }

    screenshot::cmd_transition_image(
        &ctx.device,
        cmd,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
    );

    ctx.end_single_commands(cmd)?;

    // Cleanup staging
    let _ = ctx.allocator_mut().free(staging_alloc);
    unsafe {
        ctx.device.destroy_buffer(staging_buffer, None);
    }

    Ok((image, view, allocation))
}

// ---------------------------------------------------------------------------
// Buffer utilities
// ---------------------------------------------------------------------------

/// Create a GPU buffer with device address support and upload data.
fn create_device_address_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    data: &[u8],
    usage: vk::BufferUsageFlags,
    name: &str,
) -> Result<GpuBuffer, String> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(data.len() as u64)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        device
            .create_buffer(&buffer_info, None)
            .map_err(|e| format!("Failed to create buffer '{}': {:?}", name, e))?
    };

    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let mut allocation = allocator
        .allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate memory for '{}': {:?}", name, e))?;

    unsafe {
        device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            .map_err(|e| format!("Failed to bind buffer memory '{}': {:?}", name, e))?;
    }

    if let Some(mapped) = allocation.mapped_slice_mut() {
        mapped[..data.len()].copy_from_slice(data);
    } else {
        return Err(format!("Buffer '{}' is not host-visible", name));
    }

    Ok(GpuBuffer {
        buffer,
        allocation: Some(allocation),
    })
}

/// Create an empty GPU buffer with device address support (for scratch buffers).
fn create_device_address_buffer_empty(
    device: &ash::Device,
    allocator: &mut Allocator,
    size: u64,
    usage: vk::BufferUsageFlags,
    name: &str,
) -> Result<GpuBuffer, String> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        device
            .create_buffer(&buffer_info, None)
            .map_err(|e| format!("Failed to create buffer '{}': {:?}", name, e))?
    };

    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate memory for '{}': {:?}", name, e))?;

    unsafe {
        device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            .map_err(|e| format!("Failed to bind buffer memory '{}': {:?}", name, e))?;
    }

    Ok(GpuBuffer {
        buffer,
        allocation: Some(allocation),
    })
}

/// Get the device address of a buffer.
fn get_buffer_device_address(device: &ash::Device, buffer: vk::Buffer) -> u64 {
    let info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    unsafe { device.get_buffer_device_address(&info) }
}

/// Align a value up to the given alignment (must be power of two).
fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}
