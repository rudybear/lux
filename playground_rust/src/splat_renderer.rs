//! Gaussian splat renderer using ash/Vulkan.
//!
//! Implements the `Renderer` trait for unified interactive/headless rendering.
//! Pipeline: compute projection → CPU depth sort → instanced quad draw.

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use log::info;
use std::path::Path;

use crate::gltf_loader;
use crate::scene_manager::{self, GpuBuffer, GpuImage, Renderer};
use crate::spv_loader;
use crate::vulkan_context::VulkanContext;

// --------------------------------------------------------------------------
// Push constant structs
// --------------------------------------------------------------------------

/// Compute push constants: 176 bytes.
/// Layout: view(0) + proj(64) + cam_pos(128) + pad(140) + screen_size(144)
///         + total_splats(152) + focal_x(156) + focal_y(160) + sh_degree(164) + pad(168)
#[repr(C)]
#[derive(Copy, Clone)]
struct ComputePush {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    cam_pos: [f32; 3],
    _pad0: f32,
    screen_w: f32,
    screen_h: f32,
    num_splats: u32,
    focal_x: f32,
    focal_y: f32,
    sh_degree: i32,
    _pad1: [f32; 2],
}
unsafe impl bytemuck::Pod for ComputePush {}
unsafe impl bytemuck::Zeroable for ComputePush {}

/// Render push constants: 16 bytes (shared between vertex and fragment).
/// Layout: screen_size(0) + visible_count(8) + alpha_cutoff(12)
#[repr(C)]
#[derive(Copy, Clone)]
struct RenderPush {
    screen_width: f32,
    screen_height: f32,
    visible_count: u32,
    alpha_cutoff: f32,
}
unsafe impl bytemuck::Pod for RenderPush {}
unsafe impl bytemuck::Zeroable for RenderPush {}

// --------------------------------------------------------------------------
// GaussianSplatRenderer
// --------------------------------------------------------------------------

pub struct GaussianSplatRenderer {
    // Pipelines
    compute_pipeline: vk::Pipeline,
    compute_layout: vk::PipelineLayout,
    render_pipeline: vk::Pipeline,
    render_layout: vk::PipelineLayout,

    // Offscreen render targets
    color_image: GpuImage,
    depth_image: GpuImage,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,

    // Render pass variants for hybrid compositing (LOAD instead of CLEAR)
    render_pass_load: vk::RenderPass,       // color=LOAD, depth=CLEAR
    framebuffer_load: vk::Framebuffer,
    render_pass_load_depth: vk::RenderPass, // color=LOAD, depth=LOAD
    framebuffer_load_depth: vk::Framebuffer,
    // Graphics pipeline for LOAD render pass (compatible render pass required)
    render_pipeline_load: vk::Pipeline,
    render_pipeline_load_depth: vk::Pipeline,

    // Hybrid compositing state
    has_background: bool,
    has_background_depth: bool,

    // Descriptor pools/sets
    descriptor_pool: vk::DescriptorPool,
    compute_set_layout: vk::DescriptorSetLayout,
    render_set_layout: vk::DescriptorSetLayout,
    compute_desc_set: vk::DescriptorSet,
    render_desc_set: vk::DescriptorSet,

    // GPU buffers (input)
    pos_buffer: GpuBuffer,
    rot_buffer: GpuBuffer,
    scale_buffer: GpuBuffer,
    opacity_buffer: GpuBuffer,
    sh_buffers: Vec<GpuBuffer>,

    // GPU buffers (projected output)
    proj_center_buffer: GpuBuffer,
    proj_conic_buffer: GpuBuffer,
    proj_color_buffer: GpuBuffer,

    // Sort buffers
    sort_keys_buffer: GpuBuffer,
    sorted_indices_buffer: GpuBuffer,
    visible_count_buffer: GpuBuffer,

    // Camera state
    view_matrix: glam::Mat4,
    proj_matrix: glam::Mat4,
    cam_pos: glam::Vec3,
    focal_x: f32,
    focal_y: f32,

    // Auto-camera
    auto_eye: glam::Vec3,
    auto_target: glam::Vec3,
    auto_up: glam::Vec3,
    auto_far: f32,
    has_scene_bounds: bool,

    // Dimensions
    width: u32,
    height: u32,
    num_splats: u32,

    // SH degree
    sh_degree: u32,

    // Cached host positions for CPU sort
    host_positions: Vec<[f32; 4]>,
}

impl GaussianSplatRenderer {
    pub fn new(
        ctx: &mut VulkanContext,
        scene_source: &str,
        pipeline_base: &str,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        info!("Loading glTF scene for splat rendering: {}", scene_source);
        let device = ctx.device.clone();
        let gltf_scene = gltf_loader::load_gltf(Path::new(scene_source))?;
        if !gltf_scene.splat_data.has_splats {
            return Err("Scene has no gaussian splat data".to_string());
        }
        let splat_data = &gltf_scene.splat_data;
        let num_splats = splat_data.num_splats;
        info!("Loaded {} gaussian splats (SH degree {})", num_splats, splat_data.sh_degree);

        // --- Create offscreen render targets ---
        let color_image = scene_manager::create_offscreen_image(
            &device,
            ctx.allocator_mut(),
            width, height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageAspectFlags::COLOR,
            "splat_color",
        )?;

        let depth_image = scene_manager::create_offscreen_image(
            &device,
            ctx.allocator_mut(),
            width, height,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageAspectFlags::DEPTH,
            "splat_depth",
        )?;

        // --- Render passes ---
        let render_pass = Self::create_render_pass(&device)?;
        let render_pass_load = Self::create_render_pass_load(&device)?;
        let render_pass_load_depth = Self::create_render_pass_load_depth(&device)?;

        // --- Framebuffers (one per render pass variant, same attachments) ---
        let create_fb = |rp: vk::RenderPass| -> Result<vk::Framebuffer, String> {
            let views = [color_image.view, depth_image.view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(rp)
                .attachments(&views)
                .width(width)
                .height(height)
                .layers(1);
            unsafe {
                device.create_framebuffer(&fb_info, None)
                    .map_err(|e| format!("Failed to create framebuffer: {:?}", e))
            }
        };
        let framebuffer = create_fb(render_pass)?;
        let framebuffer_load = create_fb(render_pass_load)?;
        let framebuffer_load_depth = create_fb(render_pass_load_depth)?;

        // --- Descriptor set layouts ---
        // Compute bindings: 4 input (pos,rot,scale,opacity) + N SH buffers + 5 output
        let num_sh_buffers = splat_data.sh_coefficients.len().max(1) as u32;
        let num_compute_bindings = 4 + num_sh_buffers + 5;
        let compute_set_layout = Self::create_compute_set_layout(&device, num_compute_bindings)?;
        let render_set_layout = Self::create_render_set_layout(&device)?;

        // --- Pipeline layouts ---
        let compute_layout = {
            let push_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<ComputePush>() as u32);
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&compute_set_layout))
                .push_constant_ranges(std::slice::from_ref(&push_range));
            unsafe {
                device.create_pipeline_layout(&layout_info, None)
                    .map_err(|e| format!("Failed to create compute layout: {:?}", e))?
            }
        };

        let render_layout = {
            let push_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .offset(0)
                .size(std::mem::size_of::<RenderPush>() as u32);
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&render_set_layout))
                .push_constant_ranges(std::slice::from_ref(&push_range));
            unsafe {
                device.create_pipeline_layout(&layout_info, None)
                    .map_err(|e| format!("Failed to create render layout: {:?}", e))?
            }
        };

        // --- Load shaders and create pipelines ---
        let comp_spv = spv_loader::load_spirv(Path::new(&format!("{}.comp.spv", pipeline_base)))?;
        let comp_module = spv_loader::create_shader_module(&device, &comp_spv)?;

        let compute_pipeline = {
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(comp_module)
                .name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(compute_layout);
            let pipeline = unsafe {
                device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None)
                    .map_err(|e| format!("Failed to create compute pipeline: {:?}", e.1))?[0]
            };
            unsafe { device.destroy_shader_module(comp_module, None); }
            pipeline
        };

        let vert_spv = spv_loader::load_spirv(Path::new(&format!("{}.vert.spv", pipeline_base)))?;
        let frag_spv = spv_loader::load_spirv(Path::new(&format!("{}.frag.spv", pipeline_base)))?;
        let vert_module = spv_loader::create_shader_module(&device, &vert_spv)?;
        let frag_module = spv_loader::create_shader_module(&device, &frag_spv)?;

        let render_pipeline = Self::create_render_pipeline(
            &device, render_pass, render_layout, vert_module, frag_module, width, height,
        )?;
        let render_pipeline_load = Self::create_render_pipeline(
            &device, render_pass_load, render_layout, vert_module, frag_module, width, height,
        )?;
        let render_pipeline_load_depth = Self::create_render_pipeline(
            &device, render_pass_load_depth, render_layout, vert_module, frag_module, width, height,
        )?;

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        // --- Descriptor pool ---
        // Need enough descriptors for compute (num_compute_bindings) + render (4)
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(num_compute_bindings + 4);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2)
            .pool_sizes(std::slice::from_ref(&pool_size));
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(&pool_info, None)
                .map_err(|e| format!("Failed to create descriptor pool: {:?}", e))?
        };

        let layouts = [compute_set_layout, render_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let sets = unsafe {
            device.allocate_descriptor_sets(&alloc_info)
                .map_err(|e| format!("Failed to allocate descriptor sets: {:?}", e))?
        };
        let compute_desc_set = sets[0];
        let render_desc_set = sets[1];

        // --- Upload splat data to GPU buffers ---
        let n = num_splats as usize;

        // Positions (already vec4)
        let pos_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            bytemuck::cast_slice(&splat_data.positions),
            vk::BufferUsageFlags::STORAGE_BUFFER, "splat_positions",
        )?;

        // Rotations (already vec4)
        let rot_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            bytemuck::cast_slice(&splat_data.rotations),
            vk::BufferUsageFlags::STORAGE_BUFFER, "splat_rotations",
        )?;

        // Scales (already vec4)
        let scale_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            bytemuck::cast_slice(&splat_data.scales),
            vk::BufferUsageFlags::STORAGE_BUFFER, "splat_scales",
        )?;

        // Opacities
        let opacity_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            bytemuck::cast_slice(&splat_data.opacities),
            vk::BufferUsageFlags::STORAGE_BUFFER, "splat_opacities",
        )?;

        // SH coefficient buffers
        let mut sh_buffers = Vec::new();
        for (i, coeffs) in splat_data.sh_coefficients.iter().enumerate() {
            let data: &[u8] = if coeffs.is_empty() {
                &[0u8; 16] // dummy
            } else {
                bytemuck::cast_slice(coeffs)
            };
            let buf = scene_manager::create_buffer_with_data(
                &device, ctx.allocator_mut(),
                data, vk::BufferUsageFlags::STORAGE_BUFFER,
                &format!("splat_sh_{}", i),
            )?;
            sh_buffers.push(buf);
        }

        // Projected output buffers (GPU-only, use dummy initial data)
        let proj_center_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            &vec![0u8; n * 4 * 4],
            vk::BufferUsageFlags::STORAGE_BUFFER, "proj_centers",
        )?;
        let proj_conic_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            &vec![0u8; n * 4 * 4],
            vk::BufferUsageFlags::STORAGE_BUFFER, "proj_conics",
        )?;
        let proj_color_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            &vec![0u8; n * 4 * 4],
            vk::BufferUsageFlags::STORAGE_BUFFER, "proj_colors",
        )?;

        // Sort buffers
        let sort_keys_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            &vec![0u8; n * 4],
            vk::BufferUsageFlags::STORAGE_BUFFER, "sort_keys",
        )?;
        let sorted_indices_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            &vec![0u8; n * 4],
            vk::BufferUsageFlags::STORAGE_BUFFER, "sorted_indices",
        )?;
        let visible_count_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            &[0u8; 4],
            vk::BufferUsageFlags::STORAGE_BUFFER, "visible_count",
        )?;

        // --- Update descriptor sets ---
        let write_ssbo = |set: vk::DescriptorSet, binding: u32, buffer: vk::Buffer, size: u64| {
            let buf_info = vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .offset(0)
                .range(size);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info));
            unsafe { device.update_descriptor_sets(&[write], &[]); }
        };

        // Compute set bindings: 0-3 input, 4..4+N SH, then output buffers
        write_ssbo(compute_desc_set, 0, pos_buffer.buffer, (n * 4 * 4) as u64);
        write_ssbo(compute_desc_set, 1, rot_buffer.buffer, (n * 4 * 4) as u64);
        write_ssbo(compute_desc_set, 2, scale_buffer.buffer, (n * 4 * 4) as u64);
        write_ssbo(compute_desc_set, 3, opacity_buffer.buffer, (n * 4) as u64);

        // Bind ALL SH coefficient buffers at bindings 4..4+num_sh_buffers
        for (i, sh_buf) in sh_buffers.iter().enumerate() {
            let sh_size = if i < splat_data.sh_coefficients.len() {
                (splat_data.sh_coefficients[i].len() * 16) as u64
            } else {
                16u64
            };
            write_ssbo(compute_desc_set, 4 + i as u32, sh_buf.buffer, sh_size.max(16));
        }
        if sh_buffers.is_empty() {
            // Bind dummy SH buffer
            write_ssbo(compute_desc_set, 4, pos_buffer.buffer, 16);
        }

        // Output buffers start after SH buffers
        let out_base = 4 + num_sh_buffers;
        write_ssbo(compute_desc_set, out_base, proj_center_buffer.buffer, (n * 4 * 4) as u64);
        write_ssbo(compute_desc_set, out_base + 1, proj_conic_buffer.buffer, (n * 4 * 4) as u64);
        write_ssbo(compute_desc_set, out_base + 2, proj_color_buffer.buffer, (n * 4 * 4) as u64);
        write_ssbo(compute_desc_set, out_base + 3, sort_keys_buffer.buffer, (n * 4) as u64);
        write_ssbo(compute_desc_set, out_base + 4, visible_count_buffer.buffer, 4);

        // Render set bindings 0-3
        write_ssbo(render_desc_set, 0, proj_center_buffer.buffer, (n * 4 * 4) as u64);
        write_ssbo(render_desc_set, 1, proj_conic_buffer.buffer, (n * 4 * 4) as u64);
        write_ssbo(render_desc_set, 2, proj_color_buffer.buffer, (n * 4 * 4) as u64);
        write_ssbo(render_desc_set, 3, sorted_indices_buffer.buffer, (n * 4) as u64);

        // --- Compute bounding box and auto-camera ---
        let (auto_eye, auto_target, auto_up, auto_far, has_scene_bounds) = {
            let mut min_b = glam::Vec3::splat(f32::MAX);
            let mut max_b = glam::Vec3::splat(f32::MIN);
            for pos in &splat_data.positions {
                let p = glam::Vec3::new(pos[0], pos[1], pos[2]);
                min_b = min_b.min(p);
                max_b = max_b.max(p);
            }
            let center = (min_b + max_b) * 0.5;
            let radius = (max_b - min_b).length() * 0.5;
            let r = if radius < 0.001 { 1.0 } else { radius };
            let eye = center + glam::Vec3::new(0.0, r * 0.5, r * 2.5);
            (eye, center, glam::Vec3::Y, r * 10.0, true)
        };

        // Default camera
        let aspect = width as f32 / height as f32;
        let fov_y = 45.0f32.to_radians();
        let view_matrix = glam::Mat4::look_at_rh(auto_eye, auto_target, auto_up);
        let mut proj_matrix = glam::Mat4::perspective_rh(fov_y, aspect, 0.01, auto_far);
        proj_matrix.y_axis.y *= -1.0; // Vulkan Y-flip

        // Focal lengths: fy = h/(2*tan(fov_y/2)), fx = fy for square pixels
        let focal_y = 0.5 * height as f32 / (fov_y * 0.5).tan();
        let focal_x = focal_y;

        // Cache host positions for CPU sort
        let host_positions = splat_data.positions.clone();

        info!("GaussianSplatRenderer initialized: {} splats, {}x{}", num_splats, width, height);

        Ok(Self {
            compute_pipeline,
            compute_layout,
            render_pipeline,
            render_layout,
            color_image,
            depth_image,
            render_pass,
            framebuffer,
            render_pass_load,
            framebuffer_load,
            render_pass_load_depth,
            framebuffer_load_depth,
            render_pipeline_load,
            render_pipeline_load_depth,
            has_background: false,
            has_background_depth: false,
            descriptor_pool,
            compute_set_layout,
            render_set_layout,
            compute_desc_set,
            render_desc_set,
            pos_buffer,
            rot_buffer,
            scale_buffer,
            opacity_buffer,
            sh_buffers,
            proj_center_buffer,
            proj_conic_buffer,
            proj_color_buffer,
            sort_keys_buffer,
            sorted_indices_buffer,
            visible_count_buffer,
            view_matrix,
            proj_matrix,
            cam_pos: auto_eye,
            focal_x,
            focal_y,
            auto_eye,
            auto_target,
            auto_up,
            auto_far,
            has_scene_bounds,
            width,
            height,
            num_splats,
            sh_degree: splat_data.sh_degree,
            host_positions,
        })
    }

    // --- Helper: create render pass ---
    fn create_render_pass(device: &ash::Device) -> Result<vk::RenderPass, String> {
        let attachments = [
            // Color
            vk::AttachmentDescription::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            // Depth
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

        let dependencies = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
        ];

        let rp_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);

        unsafe {
            device.create_render_pass(&rp_info, None)
                .map_err(|e| format!("Failed to create render pass: {:?}", e))
        }
    }

    // --- Helper: create render pass (LOAD color, CLEAR depth) ---
    // Used for hybrid compositing: background color was blitted in, splats render on top.
    fn create_render_pass_load(device: &ash::Device) -> Result<vk::RenderPass, String> {
        let attachments = [
            // Color: LOAD existing contents (background was blitted in)
            vk::AttachmentDescription::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            // Depth: still CLEAR (splats have their own depth)
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

        let dependencies = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
        ];

        let rp_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);

        unsafe {
            device.create_render_pass(&rp_info, None)
                .map_err(|e| format!("Failed to create render pass (load): {:?}", e))
        }
    }

    // --- Helper: create render pass (LOAD both color AND depth) ---
    // Used for full hybrid compositing: both raster color and depth are loaded,
    // so splats depth-test against mesh geometry for proper occlusion.
    fn create_render_pass_load_depth(device: &ash::Device) -> Result<vk::RenderPass, String> {
        let attachments = [
            // Color: LOAD existing contents
            vk::AttachmentDescription::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            // Depth: LOAD existing depth from raster pass
            vk::AttachmentDescription::default()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
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

        let dependencies = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
        ];

        let rp_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);

        unsafe {
            device.create_render_pass(&rp_info, None)
                .map_err(|e| format!("Failed to create render pass (load depth): {:?}", e))
        }
    }

    // --- Helper: create compute descriptor set layout ---
    fn create_compute_set_layout(device: &ash::Device, num_bindings: u32) -> Result<vk::DescriptorSetLayout, String> {
        let bindings: Vec<_> = (0..num_bindings)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        unsafe {
            device.create_descriptor_set_layout(&layout_info, None)
                .map_err(|e| format!("Failed to create compute set layout: {:?}", e))
        }
    }

    // --- Helper: create render descriptor set layout ---
    fn create_render_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout, String> {
        let bindings: Vec<_> = (0..4u32)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            })
            .collect();
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        unsafe {
            device.create_descriptor_set_layout(&layout_info, None)
                .map_err(|e| format!("Failed to create render set layout: {:?}", e))
        }
    }

    // --- Helper: create graphics pipeline ---
    fn create_render_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        layout: vk::PipelineLayout,
        vert_module: vk::ShaderModule,
        frag_module: vk::ShaderModule,
        width: u32,
        height: u32,
    ) -> Result<vk::Pipeline, String> {
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(c"main"),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport = vk::Viewport::default()
            .width(width as f32)
            .height(height as f32)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default()
            .extent(vk::Extent2D { width, height });
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
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)  // premultiplied alpha
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&blend_attachment));

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blending)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipeline = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| format!("Failed to create render pipeline: {:?}", e.1))?[0]
        };

        Ok(pipeline)
    }

    /// CPU depth sort: argsort positions by view-space depth, upload sorted indices.
    fn cpu_sort(&mut self, ctx: &VulkanContext) {
        if self.num_splats == 0 { return; }

        let vm = self.view_matrix;
        let n = self.num_splats as usize;

        // Compute view-space depth for each splat
        let mut depths: Vec<f32> = Vec::with_capacity(n);
        for pos in &self.host_positions {
            // View matrix row 2 (column-major): [2],[6],[10],[14] mapped to glam
            let depth = vm.col(0).z * pos[0] + vm.col(1).z * pos[1] + vm.col(2).z * pos[2] + vm.col(3).z;
            depths.push(depth);
        }

        // Argsort back-to-front: most negative Z (farthest) first
        let mut indices: Vec<u32> = (0..n as u32).collect();
        indices.sort_unstable_by(|&a, &b| {
            depths[a as usize]
                .partial_cmp(&depths[b as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Upload sorted indices
        if let Some(ref mut alloc) = self.sorted_indices_buffer.allocation {
            if let Some(mapped) = alloc.mapped_slice_mut() {
                let bytes = bytemuck::cast_slice::<u32, u8>(&indices);
                mapped[..bytes.len()].copy_from_slice(bytes);
            }
        }
    }

    /// Access to the depth image (for hybrid compositing).
    pub fn depth_image_handle(&self) -> vk::Image {
        self.depth_image.image
    }

    /// Preload a background color image into the splat color target.
    /// The subsequent `render()` call will use LOAD instead of CLEAR so splats
    /// are composited on top of the background.
    pub fn preload_background(
        &mut self,
        ctx: &VulkanContext,
        src_image: vk::Image,
        src_width: u32,
        src_height: u32,
    ) -> Result<(), String> {
        let device = &ctx.device;
        let cmd = ctx.begin_single_commands()?;

        unsafe {
            // Transition splat color image: UNDEFINED -> TRANSFER_DST
            let barrier = vk::ImageMemoryBarrier::default()
                .image(self.color_image.image)
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

            // Blit source image -> splat color image
            let blit_region = vk::ImageBlit::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: src_width as i32,
                        y: src_height as i32,
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
                        x: self.width as i32,
                        y: self.height as i32,
                        z: 1,
                    },
                ]);
            device.cmd_blit_image(
                cmd,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.color_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit_region],
                vk::Filter::LINEAR,
            );

            // Transition splat color image: TRANSFER_DST -> COLOR_ATTACHMENT_OPTIMAL
            // (the LOAD render pass expects this layout as initialLayout)
            let barrier = vk::ImageMemoryBarrier::default()
                .image(self.color_image.image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                )
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier],
            );
        }

        ctx.end_single_commands(cmd)?;
        self.has_background = true;
        info!(
            "Preloaded background image into splat color target ({}x{} -> {}x{})",
            src_width, src_height, self.width, self.height
        );
        Ok(())
    }

    /// Preload depth from raster pass into splat depth buffer.
    /// Splats will depth-test against mesh geometry so occluded splats are hidden.
    pub fn preload_depth(
        &mut self,
        ctx: &VulkanContext,
        src_depth_image: vk::Image,
        src_width: u32,
        src_height: u32,
    ) -> Result<(), String> {
        let device = &ctx.device;
        let cmd = ctx.begin_single_commands()?;

        unsafe {
            // Transition raster depth: DEPTH_STENCIL_ATTACHMENT_OPTIMAL -> TRANSFER_SRC
            let barrier_src = vk::ImageMemoryBarrier::default()
                .image(src_depth_image)
                .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1),
                );

            // Transition splat depth: UNDEFINED -> TRANSFER_DST
            let barrier_dst = vk::ImageMemoryBarrier::default()
                .image(self.depth_image.image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1),
                );

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS | vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier_src, barrier_dst],
            );

            // Copy or blit depth image
            if src_width == self.width && src_height == self.height {
                let copy_region = vk::ImageCopy::default()
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .layer_count(1),
                    )
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .layer_count(1),
                    )
                    .extent(vk::Extent3D {
                        width: src_width,
                        height: src_height,
                        depth: 1,
                    });
                device.cmd_copy_image(
                    cmd,
                    src_depth_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.depth_image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy_region],
                );
            } else {
                // Blit with nearest filter for depth (no linear interpolation on depth)
                let blit_region = vk::ImageBlit::default()
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .layer_count(1),
                    )
                    .src_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: src_width as i32,
                            y: src_height as i32,
                            z: 1,
                        },
                    ])
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .layer_count(1),
                    )
                    .dst_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: self.width as i32,
                            y: self.height as i32,
                            z: 1,
                        },
                    ]);
                device.cmd_blit_image(
                    cmd,
                    src_depth_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.depth_image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit_region],
                    vk::Filter::NEAREST,
                );
            }

            // Transition splat depth: TRANSFER_DST -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            let depth_barrier = vk::ImageMemoryBarrier::default()
                .image(self.depth_image.image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[], &[], &[depth_barrier],
            );
        }

        ctx.end_single_commands(cmd)?;
        self.has_background_depth = true;
        info!(
            "Preloaded raster depth into splat depth buffer ({}x{} -> {}x{})",
            src_width, src_height, self.width, self.height
        );
        Ok(())
    }

    /// Blit offscreen to swapchain image.
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
                        .level_count(1).layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier],
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
                        .level_count(1).layer_count(1),
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

    pub fn cleanup(&mut self, ctx: &mut VulkanContext) {
        let device = ctx.device.clone();

        unsafe { let _ = device.device_wait_idle(); }

        unsafe {
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_framebuffer(self.framebuffer_load, None);
            device.destroy_framebuffer(self.framebuffer_load_depth, None);
            device.destroy_render_pass(self.render_pass, None);
            device.destroy_render_pass(self.render_pass_load, None);
            device.destroy_render_pass(self.render_pass_load_depth, None);
            device.destroy_pipeline(self.compute_pipeline, None);
            device.destroy_pipeline(self.render_pipeline, None);
            device.destroy_pipeline(self.render_pipeline_load, None);
            device.destroy_pipeline(self.render_pipeline_load_depth, None);
            device.destroy_pipeline_layout(self.compute_layout, None);
            device.destroy_pipeline_layout(self.render_layout, None);
            device.destroy_descriptor_set_layout(self.compute_set_layout, None);
            device.destroy_descriptor_set_layout(self.render_set_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
        }

        self.color_image.destroy(&device, ctx.allocator_mut());
        self.depth_image.destroy(&device, ctx.allocator_mut());
        self.pos_buffer.destroy(&device, ctx.allocator_mut());
        self.rot_buffer.destroy(&device, ctx.allocator_mut());
        self.scale_buffer.destroy(&device, ctx.allocator_mut());
        self.opacity_buffer.destroy(&device, ctx.allocator_mut());
        for buf in &mut self.sh_buffers {
            buf.destroy(&device, ctx.allocator_mut());
        }
        self.proj_center_buffer.destroy(&device, ctx.allocator_mut());
        self.proj_conic_buffer.destroy(&device, ctx.allocator_mut());
        self.proj_color_buffer.destroy(&device, ctx.allocator_mut());
        self.sort_keys_buffer.destroy(&device, ctx.allocator_mut());
        self.sorted_indices_buffer.destroy(&device, ctx.allocator_mut());
        self.visible_count_buffer.destroy(&device, ctx.allocator_mut());
    }
}

// --------------------------------------------------------------------------
// Renderer trait implementation
// --------------------------------------------------------------------------

impl Renderer for GaussianSplatRenderer {
    fn render(&mut self, ctx: &VulkanContext) -> Result<vk::CommandBuffer, String> {
        if self.num_splats == 0 {
            let cmd = ctx.begin_single_commands()?;
            return Ok(cmd);
        }

        // CPU sort before recording commands
        self.cpu_sort(ctx);

        let cmd = ctx.begin_single_commands()?;

        unsafe {
            // --- Compute dispatch ---
            let push = ComputePush {
                view: self.view_matrix.to_cols_array_2d(),
                proj: self.proj_matrix.to_cols_array_2d(),
                cam_pos: self.cam_pos.to_array(),
                _pad0: 0.0,
                screen_w: self.width as f32,
                screen_h: self.height as f32,
                num_splats: self.num_splats,
                focal_x: self.focal_x,
                focal_y: self.focal_y,
                sh_degree: self.sh_degree as i32,
                _pad1: [0.0; 2],
            };

            ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.compute_pipeline);
            ctx.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, self.compute_layout,
                0, &[self.compute_desc_set], &[],
            );
            ctx.device.cmd_push_constants(
                cmd, self.compute_layout, vk::ShaderStageFlags::COMPUTE,
                0, bytemuck::bytes_of(&push),
            );

            let groups = (self.num_splats + 255) / 256;
            ctx.device.cmd_dispatch(cmd, groups, 1, 1);

            // Barrier: compute -> vertex/fragment
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier], &[], &[],
            );

            // --- Render pass ---
            // Use LOAD render pass when a background has been blitted in (hybrid compositing)
            let clear_values = [
                vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } },
                vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
            ];

            let (rp, fb, pipeline) = if self.has_background_depth {
                // Full hybrid: both color and depth loaded from raster pass
                (self.render_pass_load_depth, self.framebuffer_load_depth, self.render_pipeline_load_depth)
            } else if self.has_background {
                // Color-only compositing: color loaded, depth cleared
                (self.render_pass_load, self.framebuffer_load, self.render_pipeline_load)
            } else {
                (self.render_pass, self.framebuffer, self.render_pipeline)
            };

            // Reset background flags after use (one-shot per render call)
            self.has_background = false;
            self.has_background_depth = false;

            let rp_begin = vk::RenderPassBeginInfo::default()
                .render_pass(rp)
                .framebuffer(fb)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D { width: self.width, height: self.height },
                })
                .clear_values(&clear_values);

            ctx.device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);

            ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            ctx.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::GRAPHICS, self.render_layout,
                0, &[self.render_desc_set], &[],
            );

            let render_push = RenderPush {
                screen_width: self.width as f32,
                screen_height: self.height as f32,
                visible_count: self.num_splats,
                alpha_cutoff: 0.004,
            };
            ctx.device.cmd_push_constants(
                cmd, self.render_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0, bytemuck::bytes_of(&render_push),
            );

            // 6 vertices (quad) x numSplats instances
            ctx.device.cmd_draw(cmd, 6, self.num_splats, 0, 0);

            ctx.device.cmd_end_render_pass(cmd);
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
        self.cam_pos = eye;
        self.view_matrix = glam::Mat4::look_at_rh(eye, target, up);
        self.proj_matrix = glam::Mat4::perspective_rh(fov_y, aspect, near, far);
        self.proj_matrix.y_axis.y *= -1.0; // Vulkan Y-flip

        // Focal lengths: fy = h/(2*tan(fov_y/2)), fx = fy for square pixels
        self.focal_y = 0.5 * self.height as f32 / (fov_y * 0.5).tan();
        self.focal_x = self.focal_y;
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

    fn output_image(&self) -> vk::Image { self.color_image.image }
    fn output_format(&self) -> vk::Format { vk::Format::R8G8B8A8_UNORM }
    fn width(&self) -> u32 { self.width }
    fn height(&self) -> u32 { self.height }
    fn has_scene_bounds(&self) -> bool { self.has_scene_bounds }
    fn auto_eye(&self) -> glam::Vec3 { self.auto_eye }
    fn auto_target(&self) -> glam::Vec3 { self.auto_target }
    fn auto_up(&self) -> glam::Vec3 { self.auto_up }
    fn auto_far(&self) -> f32 { self.auto_far }

    fn wait_stage(&self) -> vk::PipelineStageFlags {
        vk::PipelineStageFlags::COMPUTE_SHADER
    }

    fn destroy(&mut self, ctx: &mut VulkanContext) {
        self.cleanup(ctx);
    }
}
