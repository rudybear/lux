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
use crate::screenshot;
use crate::spv_loader;
use crate::vulkan_context::VulkanContext;

/// GPU buffer with its allocation (mirrors raster_renderer pattern).
struct GpuBuffer {
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
}

impl GpuBuffer {
    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(alloc) = self.allocation.take() {
            let _ = allocator.free(alloc);
        }
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
    }
}

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

    storage_image: vk::Image,
    storage_image_view: vk::ImageView,
    storage_image_allocation: Option<Allocation>,

    camera_buffer: vk::Buffer,
    camera_allocation: Option<Allocation>,

    width: u32,
    height: u32,
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
    renderer.render(&device_clone, ctx.rt_pipeline_loader.as_ref().unwrap(), cmd);

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
    renderer.destroy(ctx);

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
        let (camera_buffer, camera_allocation) =
            create_camera_ubo(&device, ctx.allocator_mut(), width, height)?;

        // --- 3. Build BLAS from scene geometry ---
        let is_gltf = scene_source.ends_with(".glb") || scene_source.ends_with(".gltf");

        let (blas, blas_buffer, blas_allocation, blas_device_address) = if is_gltf {
            info!("Loading glTF mesh for RT BLAS: {}", scene_source);
            let gltf_scene = crate::gltf_loader::load_gltf(Path::new(scene_source))?;
            if gltf_scene.meshes.is_empty() {
                return Err(format!("No meshes found in glTF file: {}", scene_source));
            }
            let mesh = &gltf_scene.meshes[0];
            // Convert GltfVertex to PbrVertex for BLAS (position is at offset 0 in both)
            let verts: Vec<scene::PbrVertex> = mesh.vertices.iter().map(|v| {
                scene::PbrVertex {
                    position: v.position,
                    normal: v.normal,
                    uv: v.uv,
                }
            }).collect();
            build_blas(ctx, &verts, &mesh.indices)?
        } else {
            let (sphere_verts, sphere_indices) = scene::generate_sphere(32, 32);
            build_blas(ctx, &sphere_verts, &sphere_indices)?
        };

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
        write_rt_descriptors(
            &device,
            &merged,
            &ds_map,
            tlas,
            storage_image_view,
            camera_buffer,
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
            width,
            height,
        })
    }

    /// Record ray tracing commands into the given command buffer.
    ///
    /// The storage image is transitioned to GENERAL for the trace, then to
    /// TRANSFER_SRC_OPTIMAL for readback.
    pub fn render(
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
    pub fn destroy(&mut self, ctx: &mut VulkanContext) {
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
) -> Result<(vk::Buffer, Allocation), String> {
    let view = DefaultCamera::view();
    let proj = DefaultCamera::projection(width as f32 / height as f32);
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
fn write_rt_descriptors(
    device: &ash::Device,
    merged: &HashMap<u32, Vec<reflected_pipeline::BindingInfo>>,
    ds_map: &HashMap<u32, vk::DescriptorSet>,
    tlas: vk::AccelerationStructureKHR,
    storage_image_view: vk::ImageView,
    camera_buffer: vk::Buffer,
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
