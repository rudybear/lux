//! Gaussian splat renderer using ash/Vulkan.
//!
//! Implements the `Renderer` trait for unified interactive/headless rendering.
//! Pipeline: compute projection → GPU radix sort → instanced quad draw.

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

/// GPU radix sort push constants: 8 bytes.
/// For histogram/scatter: num_elements + bit_offset.
/// For prefix_sum: num_entries + pass_id.
#[repr(C)]
#[derive(Copy, Clone)]
struct SortPush {
    num_elements: u32,
    bit_offset: u32,
}
unsafe impl bytemuck::Pod for SortPush {}
unsafe impl bytemuck::Zeroable for SortPush {}

/// Radix sort tile size: 256 threads * 15 elements per thread.
const SORT_TILE_SIZE: u32 = 3840;
/// Prefix sum block size: 1024 threads * 2 elements per thread.
const PREFIX_SUM_BLOCK_SIZE: u32 = 2048;

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

    // Sort buffers (buffer A = primary)
    sort_keys_buffer: GpuBuffer,
    sorted_indices_buffer: GpuBuffer,
    visible_count_buffer: GpuBuffer,

    // GPU radix sort resources
    sort_keys_b_buffer: GpuBuffer,      // ping-pong buffer B for keys
    sort_vals_b_buffer: GpuBuffer,      // ping-pong buffer B for values (indices)
    histogram_buffer: GpuBuffer,        // 256 * num_wg entries
    partition_sums_buffer: GpuBuffer,   // ceil(256 * num_wg / 2048) entries

    // Sort pipelines (3 compute stages)
    sort_histogram_pipeline: vk::Pipeline,
    sort_prefix_sum_pipeline: vk::Pipeline,
    sort_scatter_pipeline: vk::Pipeline,

    // Sort pipeline layouts (per-shader, each with its own descriptor set layout)
    sort_histogram_layout: vk::PipelineLayout,
    sort_prefix_sum_layout: vk::PipelineLayout,
    sort_scatter_layout: vk::PipelineLayout,

    // Sort descriptor set layouts
    sort_histogram_set_layout: vk::DescriptorSetLayout,
    sort_prefix_sum_set_layout: vk::DescriptorSetLayout,
    sort_scatter_set_layout: vk::DescriptorSetLayout,

    // Sort descriptor sets: [0]=A->B, [1]=B->A for histogram and scatter; [0] only for prefix_sum
    sort_histogram_desc_sets: [vk::DescriptorSet; 2],
    sort_prefix_sum_desc_set: vk::DescriptorSet,
    sort_scatter_desc_sets: [vk::DescriptorSet; 2],

    // Precomputed sort workgroup count
    sort_num_wg: u32,

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
        // Compute bindings: 4 input (pos,rot,scale,opacity) + N SH buffers + 6 output
        // Output: proj_center, proj_conic, proj_color, sort_keys, sorted_indices, visible_count
        let num_sh_buffers = splat_data.sh_coefficients.len().max(1) as u32;
        let num_compute_bindings = 4 + num_sh_buffers + 6;
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

        // --- GPU Radix Sort pipelines ---
        // Load sort shader SPIR-V from shaders/radix_sort/ relative to working directory
        let sort_shader_dir = Path::new("shaders/radix_sort");
        let hist_spv = spv_loader::load_spirv(&sort_shader_dir.join("histogram.comp.spv"))?;
        let prefix_spv = spv_loader::load_spirv(&sort_shader_dir.join("prefix_sum.comp.spv"))?;
        let scatter_spv = spv_loader::load_spirv(&sort_shader_dir.join("scatter.comp.spv"))?;
        let hist_module = spv_loader::create_shader_module(&device, &hist_spv)?;
        let prefix_module = spv_loader::create_shader_module(&device, &prefix_spv)?;
        let scatter_module = spv_loader::create_shader_module(&device, &scatter_spv)?;

        // Sort descriptor set layouts (matching GLSL binding declarations)
        // histogram.comp: binding 0 = keys_in, binding 1 = histograms
        let sort_histogram_set_layout = Self::create_sort_set_layout(&device, 2)?;
        // prefix_sum.comp: binding 0 = histograms, binding 1 = partition_sums
        let sort_prefix_sum_set_layout = Self::create_sort_set_layout(&device, 2)?;
        // scatter.comp: binding 0-4 = keys_in, keys_out, vals_in, vals_out, histograms
        let sort_scatter_set_layout = Self::create_sort_set_layout(&device, 5)?;

        // Sort pipeline layouts (push constant: SortPush = 8 bytes)
        let sort_push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<SortPush>() as u32);

        let sort_histogram_layout = {
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&sort_histogram_set_layout))
                .push_constant_ranges(std::slice::from_ref(&sort_push_range));
            unsafe {
                device.create_pipeline_layout(&layout_info, None)
                    .map_err(|e| format!("Failed to create sort histogram layout: {:?}", e))?
            }
        };
        let sort_prefix_sum_layout = {
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&sort_prefix_sum_set_layout))
                .push_constant_ranges(std::slice::from_ref(&sort_push_range));
            unsafe {
                device.create_pipeline_layout(&layout_info, None)
                    .map_err(|e| format!("Failed to create sort prefix_sum layout: {:?}", e))?
            }
        };
        let sort_scatter_layout = {
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&sort_scatter_set_layout))
                .push_constant_ranges(std::slice::from_ref(&sort_push_range));
            unsafe {
                device.create_pipeline_layout(&layout_info, None)
                    .map_err(|e| format!("Failed to create sort scatter layout: {:?}", e))?
            }
        };

        // Sort compute pipelines
        let create_sort_pipeline = |module: vk::ShaderModule, layout: vk::PipelineLayout, name: &str| -> Result<vk::Pipeline, String> {
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            unsafe {
                device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None)
                    .map_err(|e| format!("Failed to create sort {} pipeline: {:?}", name, e.1))
                    .map(|p| p[0])
            }
        };

        let sort_histogram_pipeline = create_sort_pipeline(hist_module, sort_histogram_layout, "histogram")?;
        let sort_prefix_sum_pipeline = create_sort_pipeline(prefix_module, sort_prefix_sum_layout, "prefix_sum")?;
        let sort_scatter_pipeline = create_sort_pipeline(scatter_module, sort_scatter_layout, "scatter")?;

        unsafe {
            device.destroy_shader_module(hist_module, None);
            device.destroy_shader_module(prefix_module, None);
            device.destroy_shader_module(scatter_module, None);
        }

        // --- Descriptor pool ---
        // Need enough descriptors for:
        //   compute (num_compute_bindings) + render (4)
        //   + sort histogram (2 sets * 2 bindings = 4)
        //   + sort prefix_sum (1 set * 2 bindings = 2)
        //   + sort scatter (2 sets * 5 bindings = 10)
        let total_sort_descriptors: u32 = 4 + 2 + 10;
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(num_compute_bindings + 4 + total_sort_descriptors);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2 + 5)  // compute, render, + 2 histogram + 1 prefix_sum + 2 scatter
            .pool_sizes(std::slice::from_ref(&pool_size));
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(&pool_info, None)
                .map_err(|e| format!("Failed to create descriptor pool: {:?}", e))?
        };

        // Allocate compute + render descriptor sets
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

        // Allocate sort descriptor sets
        let sort_layouts = [
            sort_histogram_set_layout,  // A->B
            sort_histogram_set_layout,  // B->A
            sort_prefix_sum_set_layout, // single
            sort_scatter_set_layout,    // A->B
            sort_scatter_set_layout,    // B->A
        ];
        let sort_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&sort_layouts);
        let sort_sets = unsafe {
            device.allocate_descriptor_sets(&sort_alloc_info)
                .map_err(|e| format!("Failed to allocate sort descriptor sets: {:?}", e))?
        };
        let sort_histogram_desc_sets = [sort_sets[0], sort_sets[1]];
        let sort_prefix_sum_desc_set = sort_sets[2];
        let sort_scatter_desc_sets = [sort_sets[3], sort_sets[4]];

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

        // Sort buffers (buffer A = primary, written by compute shader)
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

        // Ping-pong sort buffers (buffer B)
        let sort_keys_b_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            &vec![0u8; n * 4],
            vk::BufferUsageFlags::STORAGE_BUFFER, "sort_keys_b",
        )?;
        let sort_vals_b_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            &vec![0u8; n * 4],
            vk::BufferUsageFlags::STORAGE_BUFFER, "sort_vals_b",
        )?;

        // Radix sort histogram and partition buffers
        let sort_num_wg = (num_splats + SORT_TILE_SIZE - 1) / SORT_TILE_SIZE;
        let histogram_size = (256 * sort_num_wg as usize * 4).max(4);
        let histogram_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            &vec![0u8; histogram_size],
            vk::BufferUsageFlags::STORAGE_BUFFER, "sort_histograms",
        )?;
        let total_histogram_entries = 256 * sort_num_wg;
        let num_partitions = (total_histogram_entries + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE;
        let partition_sums_size = (num_partitions as usize * 4).max(4);
        let partition_sums_buffer = scene_manager::create_buffer_with_data(
            &device, ctx.allocator_mut(),
            &vec![0u8; partition_sums_size],
            vk::BufferUsageFlags::STORAGE_BUFFER, "sort_partition_sums",
        )?;

        info!("GPU radix sort: {} splats, {} workgroups, {} histogram entries, {} partitions",
              num_splats, sort_num_wg, total_histogram_entries, num_partitions);

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
        // Order: proj_center, proj_conic, proj_color, sort_keys, sorted_indices, visible_count
        let out_base = 4 + num_sh_buffers;
        write_ssbo(compute_desc_set, out_base, proj_center_buffer.buffer, (n * 4 * 4) as u64);
        write_ssbo(compute_desc_set, out_base + 1, proj_conic_buffer.buffer, (n * 4 * 4) as u64);
        write_ssbo(compute_desc_set, out_base + 2, proj_color_buffer.buffer, (n * 4 * 4) as u64);
        write_ssbo(compute_desc_set, out_base + 3, sort_keys_buffer.buffer, (n * 4) as u64);
        write_ssbo(compute_desc_set, out_base + 4, sorted_indices_buffer.buffer, (n * 4) as u64);
        write_ssbo(compute_desc_set, out_base + 5, visible_count_buffer.buffer, 4);

        // Render set bindings 0-3
        write_ssbo(render_desc_set, 0, proj_center_buffer.buffer, (n * 4 * 4) as u64);
        write_ssbo(render_desc_set, 1, proj_conic_buffer.buffer, (n * 4 * 4) as u64);
        write_ssbo(render_desc_set, 2, proj_color_buffer.buffer, (n * 4 * 4) as u64);
        write_ssbo(render_desc_set, 3, sorted_indices_buffer.buffer, (n * 4) as u64);

        // --- Sort descriptor sets ---
        let sort_buf_size = (n * 4) as u64;
        let hist_buf_size = histogram_size as u64;
        let part_buf_size = partition_sums_size as u64;

        // Histogram descriptor sets: binding 0 = keys_in, binding 1 = histograms
        // [0] = A->B direction (reads from keys A)
        write_ssbo(sort_histogram_desc_sets[0], 0, sort_keys_buffer.buffer, sort_buf_size);
        write_ssbo(sort_histogram_desc_sets[0], 1, histogram_buffer.buffer, hist_buf_size);
        // [1] = B->A direction (reads from keys B)
        write_ssbo(sort_histogram_desc_sets[1], 0, sort_keys_b_buffer.buffer, sort_buf_size);
        write_ssbo(sort_histogram_desc_sets[1], 1, histogram_buffer.buffer, hist_buf_size);

        // Prefix sum descriptor set: binding 0 = histograms, binding 1 = partition_sums
        write_ssbo(sort_prefix_sum_desc_set, 0, histogram_buffer.buffer, hist_buf_size);
        write_ssbo(sort_prefix_sum_desc_set, 1, partition_sums_buffer.buffer, part_buf_size);

        // Scatter descriptor sets: binding 0=keys_in, 1=keys_out, 2=vals_in, 3=vals_out, 4=histograms
        // [0] = A->B direction
        write_ssbo(sort_scatter_desc_sets[0], 0, sort_keys_buffer.buffer, sort_buf_size);
        write_ssbo(sort_scatter_desc_sets[0], 1, sort_keys_b_buffer.buffer, sort_buf_size);
        write_ssbo(sort_scatter_desc_sets[0], 2, sorted_indices_buffer.buffer, sort_buf_size);
        write_ssbo(sort_scatter_desc_sets[0], 3, sort_vals_b_buffer.buffer, sort_buf_size);
        write_ssbo(sort_scatter_desc_sets[0], 4, histogram_buffer.buffer, hist_buf_size);
        // [1] = B->A direction
        write_ssbo(sort_scatter_desc_sets[1], 0, sort_keys_b_buffer.buffer, sort_buf_size);
        write_ssbo(sort_scatter_desc_sets[1], 1, sort_keys_buffer.buffer, sort_buf_size);
        write_ssbo(sort_scatter_desc_sets[1], 2, sort_vals_b_buffer.buffer, sort_buf_size);
        write_ssbo(sort_scatter_desc_sets[1], 3, sorted_indices_buffer.buffer, sort_buf_size);
        write_ssbo(sort_scatter_desc_sets[1], 4, histogram_buffer.buffer, hist_buf_size);

        // --- Compute bounding box and auto-camera ---
        // Use median center and IQR (P25-P75) radius for tight framing.
        // Real 3DGS scenes have extreme outlier background splats that make
        // P5-P95 or full bbox camera placement far too distant.
        let (auto_eye, auto_target, auto_up, auto_far, has_scene_bounds) = {
            let n_splats = splat_data.positions.len();
            if n_splats > 100 {
                let mut xs: Vec<f32> = splat_data.positions.iter().map(|p| p[0]).collect();
                let mut ys: Vec<f32> = splat_data.positions.iter().map(|p| p[1]).collect();
                let mut zs: Vec<f32> = splat_data.positions.iter().map(|p| p[2]).collect();
                xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                ys.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                zs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                // IQR (interquartile range) for robust scene extent
                let p25 = n_splats * 25 / 100;
                let p75 = n_splats * 75 / 100;
                let center = glam::Vec3::new(
                    xs[n_splats / 2], ys[n_splats / 2], zs[n_splats / 2],
                );
                let extent = glam::Vec3::new(
                    xs[p75] - xs[p25], ys[p75] - ys[p25], zs[p75] - zs[p25],
                );
                // Use max IQR axis as base radius for camera placement
                let max_iqr = extent.x.max(extent.y).max(extent.z);
                let r = if max_iqr < 0.001 { 1.0 } else { max_iqr };

                // Full bbox for far plane
                let full_radius = glam::Vec3::new(
                    xs[n_splats - 1] - xs[0], ys[n_splats - 1] - ys[0], zs[n_splats - 1] - zs[0],
                ).length() * 0.5;

                // Detect the "up" axis: the shortest IQR extent is likely the vertical axis.
                // Use negative direction for Y (COLMAP convention: Y points down).
                let (up_vec, up_idx) = if extent.y <= extent.x && extent.y <= extent.z {
                    (glam::Vec3::NEG_Y, 1usize)
                } else if extent.z <= extent.x && extent.z <= extent.y {
                    (glam::Vec3::Z, 2usize)
                } else {
                    (glam::Vec3::X, 0usize)
                };

                // For 3DGS scenes, place camera at eye-level near training distance.
                // Training cameras are typically 3-6× the object radius from center.
                // Use the up-axis extent as height reference (don't elevate too much).
                let up_extent = [extent.x, extent.y, extent.z][up_idx];
                let ground_r = r; // largest IQR extent = ground plane radius
                let cam_dist = ground_r * 0.4; // close to match training distance

                // Build eye offset: on the two ground axes, slightly elevated in up
                let mut off = [0.0f32; 3];
                let ground_axes: Vec<usize> = (0..3).filter(|&i| i != up_idx).collect();
                off[ground_axes[0]] = cam_dist * 0.7;
                off[ground_axes[1]] = cam_dist * 0.7;
                off[up_idx] = up_extent * 0.1; // slight elevation
                let eye = center + glam::Vec3::new(off[0], off[1], off[2]);

                let up_name = ["X", "Y", "Z"][up_idx];
                info!("AUTO-CAM: center=({:.2},{:.2},{:.2}) extent=({:.2},{:.2},{:.2}) r={:.2} up={} eye=({:.2},{:.2},{:.2})",
                    center.x, center.y, center.z, extent.x, extent.y, extent.z, r,
                    up_name, eye.x, eye.y, eye.z);
                (eye, center, up_vec, full_radius * 5.0, true)
            } else {
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
            }
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

        info!("GaussianSplatRenderer initialized: {} splats, {}x{} (GPU radix sort)", num_splats, width, height);

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
            sort_keys_b_buffer,
            sort_vals_b_buffer,
            histogram_buffer,
            partition_sums_buffer,
            sort_histogram_pipeline,
            sort_prefix_sum_pipeline,
            sort_scatter_pipeline,
            sort_histogram_layout,
            sort_prefix_sum_layout,
            sort_scatter_layout,
            sort_histogram_set_layout,
            sort_prefix_sum_set_layout,
            sort_scatter_set_layout,
            sort_histogram_desc_sets,
            sort_prefix_sum_desc_set,
            sort_scatter_desc_sets,
            sort_num_wg,
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

    // --- Helper: create sort descriptor set layout (all COMPUTE SSBOs) ---
    fn create_sort_set_layout(device: &ash::Device, num_bindings: u32) -> Result<vk::DescriptorSetLayout, String> {
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
                .map_err(|e| format!("Failed to create sort set layout: {:?}", e))
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

    /// Read back and print projected buffer values after GPU compute (debug helper).
    /// Call AFTER the command buffer containing the compute dispatch has been
    /// submitted and the fence has been waited on (GPU is idle).
    #[allow(dead_code)]
    pub fn debug_dump_projected(&self) {
        let n = self.num_splats as usize;
        let entries = 20.min(n);

        // --- visible_count ---
        if let Some(ref alloc) = self.visible_count_buffer.allocation {
            if let Some(mapped) = alloc.mapped_slice() {
                if mapped.len() >= 4 {
                    let count = u32::from_le_bytes([mapped[0], mapped[1], mapped[2], mapped[3]]);
                    info!("DEBUG visible_count = {} (num_splats = {})", count, self.num_splats);
                }
            } else {
                info!("DEBUG visible_count_buffer: not mapped");
            }
        }

        // --- sort_keys (first entries) ---
        if let Some(ref alloc) = self.sort_keys_buffer.allocation {
            if let Some(mapped) = alloc.mapped_slice() {
                let floats: &[f32] = bytemuck::cast_slice(&mapped[..entries * 4]);
                info!("DEBUG sort_keys[0..{}]: {:?}", entries, &floats[..entries]);
            }
        }

        // --- sorted_indices (first entries) ---
        if let Some(ref alloc) = self.sorted_indices_buffer.allocation {
            if let Some(mapped) = alloc.mapped_slice() {
                let indices: &[u32] = bytemuck::cast_slice(&mapped[..entries * 4]);
                info!("DEBUG sorted_indices[0..{}]: {:?}", entries, &indices[..entries]);
            }
        }

        // --- proj_center (vec4 per splat) ---
        if let Some(ref alloc) = self.proj_center_buffer.allocation {
            if let Some(mapped) = alloc.mapped_slice() {
                let floats: &[f32] = bytemuck::cast_slice(&mapped[..entries * 16]);
                for i in 0..entries {
                    let base = i * 4;
                    info!(
                        "DEBUG proj_center[{}]: ({:.4}, {:.4}, {:.4}, {:.4})",
                        i, floats[base], floats[base + 1], floats[base + 2], floats[base + 3]
                    );
                }
            } else {
                info!("DEBUG proj_center_buffer: not mapped");
            }
        }

        // --- proj_conic (vec4 per splat) ---
        if let Some(ref alloc) = self.proj_conic_buffer.allocation {
            if let Some(mapped) = alloc.mapped_slice() {
                let floats: &[f32] = bytemuck::cast_slice(&mapped[..entries * 16]);
                for i in 0..entries {
                    let base = i * 4;
                    info!(
                        "DEBUG proj_conic[{}]: ({:.6}, {:.6}, {:.6}, {:.6})",
                        i, floats[base], floats[base + 1], floats[base + 2], floats[base + 3]
                    );
                }
            } else {
                info!("DEBUG proj_conic_buffer: not mapped");
            }
        }

        // --- proj_color (vec4 per splat) ---
        if let Some(ref alloc) = self.proj_color_buffer.allocation {
            if let Some(mapped) = alloc.mapped_slice() {
                let floats: &[f32] = bytemuck::cast_slice(&mapped[..entries * 16]);
                for i in 0..entries {
                    let base = i * 4;
                    info!(
                        "DEBUG proj_color[{}]: ({:.4}, {:.4}, {:.4}, {:.4})",
                        i, floats[base], floats[base + 1], floats[base + 2], floats[base + 3]
                    );
                }
            } else {
                info!("DEBUG proj_color_buffer: not mapped");
            }
        }

        // --- Summary: count non-zero projected centers ---
        if let Some(ref alloc) = self.proj_center_buffer.allocation {
            if let Some(mapped) = alloc.mapped_slice() {
                let max_check = n.min(mapped.len() / 16);
                let floats: &[f32] = bytemuck::cast_slice(&mapped[..max_check * 16]);
                let mut nonzero = 0u32;
                let mut nan_count = 0u32;
                let mut inf_count = 0u32;
                let mut offscreen = 0u32;
                for i in 0..max_check {
                    let x = floats[i * 4];
                    let y = floats[i * 4 + 1];
                    if x.is_nan() || y.is_nan() {
                        nan_count += 1;
                    } else if x.is_infinite() || y.is_infinite() {
                        inf_count += 1;
                    } else if x != 0.0 || y != 0.0 {
                        nonzero += 1;
                        if x < 0.0 || x > self.width as f32 || y < 0.0 || y > self.height as f32 {
                            offscreen += 1;
                        }
                    }
                }
                info!(
                    "DEBUG proj_center summary ({} checked): {} nonzero, {} NaN, {} Inf, {} offscreen",
                    max_check, nonzero, nan_count, inf_count, offscreen
                );
            }
        }

        // --- Also dump push constant values used ---
        info!("DEBUG camera: eye=({:.3}, {:.3}, {:.3})", self.cam_pos.x, self.cam_pos.y, self.cam_pos.z);
        info!("DEBUG screen: {}x{}, focal=({:.1}, {:.1}), sh_degree={}",
              self.width, self.height, self.focal_x, self.focal_y, self.sh_degree);
        info!("DEBUG view_matrix row0: {:?}", self.view_matrix.row(0));
        info!("DEBUG view_matrix row1: {:?}", self.view_matrix.row(1));
        info!("DEBUG view_matrix row2: {:?}", self.view_matrix.row(2));
        info!("DEBUG view_matrix row3: {:?}", self.view_matrix.row(3));
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

            // Sort pipelines and layouts
            device.destroy_pipeline(self.sort_histogram_pipeline, None);
            device.destroy_pipeline(self.sort_prefix_sum_pipeline, None);
            device.destroy_pipeline(self.sort_scatter_pipeline, None);
            device.destroy_pipeline_layout(self.sort_histogram_layout, None);
            device.destroy_pipeline_layout(self.sort_prefix_sum_layout, None);
            device.destroy_pipeline_layout(self.sort_scatter_layout, None);
            device.destroy_descriptor_set_layout(self.sort_histogram_set_layout, None);
            device.destroy_descriptor_set_layout(self.sort_prefix_sum_set_layout, None);
            device.destroy_descriptor_set_layout(self.sort_scatter_set_layout, None);

            // Descriptor pool (frees all descriptor sets)
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
        self.sort_keys_b_buffer.destroy(&device, ctx.allocator_mut());
        self.sort_vals_b_buffer.destroy(&device, ctx.allocator_mut());
        self.histogram_buffer.destroy(&device, ctx.allocator_mut());
        self.partition_sums_buffer.destroy(&device, ctx.allocator_mut());
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

        let cmd = ctx.begin_single_commands()?;

        unsafe {
            // --- Compute dispatch (projection + sort key generation) ---
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

            // Barrier: compute writes -> sort reads
            let compute_to_sort_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[compute_to_sort_barrier], &[], &[],
            );

            // --- GPU Radix Sort (4 passes, 8 bits per pass = 32-bit keys) ---
            let num_elements = self.num_splats;
            let num_wg = self.sort_num_wg;
            let total_histogram = 256 * num_wg;
            let num_parts = (total_histogram + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE;

            let sort_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);

            for pass in 0..4u32 {
                let bit_offset = pass * 8;
                let ping = (pass % 2) as usize;  // 0 = A->B, 1 = B->A

                // --- Phase 1: Histogram ---
                ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.sort_histogram_pipeline);
                ctx.device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE, self.sort_histogram_layout,
                    0, &[self.sort_histogram_desc_sets[ping]], &[],
                );
                let hist_push = SortPush { num_elements, bit_offset };
                ctx.device.cmd_push_constants(
                    cmd, self.sort_histogram_layout, vk::ShaderStageFlags::COMPUTE,
                    0, bytemuck::bytes_of(&hist_push),
                );
                ctx.device.cmd_dispatch(cmd, num_wg, 1, 1);

                // Barrier: histogram write -> prefix sum read
                ctx.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[sort_barrier], &[], &[],
                );

                // --- Phase 2: Prefix Sum (3 sub-passes) ---
                ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.sort_prefix_sum_pipeline);
                ctx.device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE, self.sort_prefix_sum_layout,
                    0, &[self.sort_prefix_sum_desc_set], &[],
                );

                // Sub-pass 0: Local scan
                let ps_push_0 = SortPush { num_elements: total_histogram, bit_offset: 0 };
                ctx.device.cmd_push_constants(
                    cmd, self.sort_prefix_sum_layout, vk::ShaderStageFlags::COMPUTE,
                    0, bytemuck::bytes_of(&ps_push_0),
                );
                ctx.device.cmd_dispatch(cmd, num_parts, 1, 1);

                ctx.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[sort_barrier], &[], &[],
                );

                // Sub-pass 1: Spine scan
                let ps_push_1 = SortPush { num_elements: num_parts, bit_offset: 1 };
                ctx.device.cmd_push_constants(
                    cmd, self.sort_prefix_sum_layout, vk::ShaderStageFlags::COMPUTE,
                    0, bytemuck::bytes_of(&ps_push_1),
                );
                ctx.device.cmd_dispatch(cmd, 1, 1, 1);

                ctx.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[sort_barrier], &[], &[],
                );

                // Sub-pass 2: Propagate
                let ps_push_2 = SortPush { num_elements: total_histogram, bit_offset: 2 };
                ctx.device.cmd_push_constants(
                    cmd, self.sort_prefix_sum_layout, vk::ShaderStageFlags::COMPUTE,
                    0, bytemuck::bytes_of(&ps_push_2),
                );
                ctx.device.cmd_dispatch(cmd, num_parts, 1, 1);

                ctx.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[sort_barrier], &[], &[],
                );

                // --- Phase 3: Scatter ---
                ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.sort_scatter_pipeline);
                ctx.device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE, self.sort_scatter_layout,
                    0, &[self.sort_scatter_desc_sets[ping]], &[],
                );
                let scatter_push = SortPush { num_elements, bit_offset };
                ctx.device.cmd_push_constants(
                    cmd, self.sort_scatter_layout, vk::ShaderStageFlags::COMPUTE,
                    0, bytemuck::bytes_of(&scatter_push),
                );
                ctx.device.cmd_dispatch(cmd, num_wg, 1, 1);

                // Barrier between sort passes (and final pass -> render)
                ctx.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[sort_barrier], &[], &[],
                );
            }

            // After 4 passes (even count), sorted results are in buffer A
            // (sort_keys_buffer, sorted_indices_buffer) which is what render reads.

            // Barrier: sort compute -> vertex/fragment shader reads
            let sort_to_render_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[sort_to_render_barrier], &[], &[],
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
                alpha_cutoff: 1.0 / 255.0,
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
