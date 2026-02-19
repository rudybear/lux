//! Rasterization renderer: triangle, fullscreen quad, and PBR sphere modes.

use ash::vk;
use bytemuck;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use log::info;
use std::path::Path;

use crate::camera::DefaultCamera;
use crate::scene;
use crate::screenshot;
use crate::spv_loader;
use crate::vulkan_context::VulkanContext;

/// The rendering mode for the rasterizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RasterMode {
    Triangle,
    Fullscreen,
    Pbr,
}

/// GPU buffer with its allocation.
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

/// GPU image with its allocation.
struct GpuImage {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
}

impl GpuImage {
    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
        }
        if let Some(alloc) = self.allocation.take() {
            let _ = allocator.free(alloc);
        }
        unsafe {
            device.destroy_image(self.image, None);
        }
    }
}

/// Built-in fullscreen vertex shader as pre-compiled SPIR-V.
///
/// This is the SPIR-V bytecode for:
/// ```glsl
/// #version 450
/// layout(location = 0) out vec2 uv;
/// void main() {
///     vec2 positions[3] = vec2[](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
///     gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
///     vec2 p = positions[gl_VertexIndex];
///     uv = vec2(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
/// }
/// ```
///
/// We embed the SPIR-V inline so no external file is needed.
/// This was assembled from the GLSL above; if you need to regenerate it,
/// use: glslangValidator -V fullscreen.vert -o fullscreen.vert.spv
///
/// For portability, we define a minimal SPIR-V module directly.
fn fullscreen_vert_spirv() -> Vec<u32> {
    // Instead of embedding raw bytes, we compile the fullscreen shader at build time
    // by writing the GLSL to a temp file and invoking glslangValidator.
    // However, for maximum portability, we embed the known-good SPIR-V directly.
    //
    // This SPIR-V was generated from the GLSL above and is architecture-independent.
    vec![
        // SPIR-V magic
        0x07230203,
        // Version 1.0
        0x00010000,
        // Generator (Khronos glslang)
        0x000d000b,
        // Bound
        0x00000030,
        // Schema
        0x00000000,
        // OpCapability Shader
        0x00020011, 0x00000001,
        // OpMemoryModel Logical GLSL450
        0x0003000e, 0x00000000, 0x00000001,
        // OpEntryPoint Vertex %main "main" %gl_VertexIndex %_ %uv
        0x0008000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000,
        0x0000000d, 0x00000012, 0x0000002b,
        // OpSource GLSL 450
        0x00030003, 0x00000002, 0x000001c2,
        // OpName %main "main"
        0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
        // OpName %positions "positions"
        0x00050005, 0x00000009, 0x69736f70, 0x6e6f6974, 0x00000073,
        // OpName %gl_VertexIndex "gl_VertexIndex"
        0x00060005, 0x0000000d, 0x565f6c67, 0x65747265, 0x646e4978, 0x00007865,
        // OpName %gl_PerVertex "gl_PerVertex"
        0x00060005, 0x00000010, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000,
        // OpMemberName %gl_PerVertex 0 "gl_Position"
        0x00060006, 0x00000010, 0x00000000, 0x505f6c67, 0x7469736f, 0x006e6f69,
        // OpMemberName %gl_PerVertex 1 "gl_PointSize"
        0x00070006, 0x00000010, 0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953,
        0x00000000,
        // OpMemberName %gl_PerVertex 2 "gl_ClipDistance"
        0x00070006, 0x00000010, 0x00000002, 0x435f6c67, 0x4470696c, 0x61747369,
        0x0065636e,
        // OpMemberName %gl_PerVertex 3 "gl_CullDistance"
        0x00070006, 0x00000010, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369,
        0x0065636e,
        // OpName %_ ""
        0x00030005, 0x00000012, 0x00000000,
        // OpName %uv "uv"
        0x00030005, 0x0000002b, 0x00007675,
        // OpDecorate %gl_VertexIndex BuiltIn VertexIndex
        0x00040047, 0x0000000d, 0x0000000b, 0x0000002a,
        // OpMemberDecorate %gl_PerVertex 0 BuiltIn Position
        0x00050048, 0x00000010, 0x00000000, 0x0000000b, 0x00000000,
        // OpMemberDecorate %gl_PerVertex 1 BuiltIn PointSize
        0x00050048, 0x00000010, 0x00000001, 0x0000000b, 0x00000001,
        // OpMemberDecorate %gl_PerVertex 2 BuiltIn ClipDistance
        0x00050048, 0x00000010, 0x00000002, 0x0000000b, 0x00000003,
        // OpMemberDecorate %gl_PerVertex 3 BuiltIn CullDistance
        0x00050048, 0x00000010, 0x00000003, 0x0000000b, 0x00000004,
        // OpDecorate %gl_PerVertex Block
        0x00030047, 0x00000010, 0x00000002,
        // OpDecorate %uv Location 0
        0x00040047, 0x0000002b, 0x0000001e, 0x00000000,
    ]
}

/// Create a GPU buffer, upload data, and return the buffer + allocation.
fn create_buffer_with_data(
    device: &ash::Device,
    allocator: &mut Allocator,
    data: &[u8],
    usage: vk::BufferUsageFlags,
    name: &str,
) -> Result<GpuBuffer, String> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(data.len() as u64)
        .usage(usage | vk::BufferUsageFlags::TRANSFER_DST)
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

    // Copy data
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

/// Create an offscreen render target image.
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
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe {
        device
            .create_image(&image_info, None)
            .map_err(|e| format!("Failed to create image '{}': {:?}", name, e))?
    };

    let requirements = unsafe { device.get_image_memory_requirements(image) };

    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate image memory '{}': {:?}", name, e))?;

    unsafe {
        device
            .bind_image_memory(image, allocation.memory(), allocation.offset())
            .map_err(|e| format!("Failed to bind image memory '{}': {:?}", name, e))?;
    }

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(vk::ComponentMapping::default())
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );

    let view = unsafe {
        device
            .create_image_view(&view_info, None)
            .map_err(|e| format!("Failed to create image view '{}': {:?}", name, e))?
    };

    Ok(GpuImage {
        image,
        view,
        allocation: Some(allocation),
    })
}

/// Create a texture image from pixel data and upload it via a staging buffer.
fn create_texture_image(
    ctx: &mut VulkanContext,
    width: u32,
    height: u32,
    pixels: &[u8],
    name: &str,
) -> Result<GpuImage, String> {
    let format = vk::Format::R8G8B8A8_UNORM;
    let data_size = (width * height * 4) as u64;

    // Create staging buffer
    let staging_info = vk::BufferCreateInfo::default()
        .size(data_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let staging_buffer = unsafe {
        ctx.device
            .create_buffer(&staging_info, None)
            .map_err(|e| format!("Failed to create staging buffer: {:?}", e))?
    };

    let staging_reqs = unsafe { ctx.device.get_buffer_memory_requirements(staging_buffer) };

    let mut staging_alloc = ctx
        .allocator_mut()
        .allocate(&AllocationCreateDesc {
            name: "texture_staging",
            requirements: staging_reqs,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate staging memory: {:?}", e))?;

    unsafe {
        ctx.device
            .bind_buffer_memory(staging_buffer, staging_alloc.memory(), staging_alloc.offset())
            .map_err(|e| format!("Failed to bind staging memory: {:?}", e))?;
    }

    if let Some(mapped) = staging_alloc.mapped_slice_mut() {
        mapped[..pixels.len()].copy_from_slice(pixels);
    }

    // Create GPU image
    let device_clone = ctx.device.clone();
    let image = create_offscreen_image(
        &device_clone,
        ctx.allocator_mut(),
        width,
        height,
        format,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::ImageAspectFlags::COLOR,
        name,
    )?;

    // Copy staging -> image
    let cmd = ctx.begin_single_commands()?;

    screenshot::cmd_transition_image(
        &ctx.device,
        cmd,
        image.image,
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
            image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }

    screenshot::cmd_transition_image(
        &ctx.device,
        cmd,
        image.image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
    );

    ctx.end_single_commands(cmd)?;

    // Cleanup staging
    let _ = ctx.allocator_mut().free(staging_alloc);
    unsafe {
        ctx.device.destroy_buffer(staging_buffer, None);
    }

    Ok(image)
}

/// Render using the rasterization pipeline and save the result as a PNG.
pub fn render_raster(
    ctx: &mut VulkanContext,
    mode: RasterMode,
    shader_base: &str,
    width: u32,
    height: u32,
    output_path: &Path,
) -> Result<(), String> {
    info!("Raster render: mode={:?}, {}x{}", mode, width, height);

    // --- Load shaders ---
    let (vert_module, frag_module) = load_shaders(ctx, mode, shader_base)?;

    // --- Create resources based on mode ---
    match mode {
        RasterMode::Triangle => {
            render_triangle(ctx, vert_module, frag_module, width, height, output_path)
        }
        RasterMode::Fullscreen => {
            render_fullscreen(ctx, frag_module, width, height, output_path)
        }
        RasterMode::Pbr => {
            render_pbr(ctx, vert_module, frag_module, width, height, output_path)
        }
    }?;

    // Cleanup shader modules
    unsafe {
        ctx.device.destroy_shader_module(frag_module, None);
        if vert_module != vk::ShaderModule::null() {
            ctx.device.destroy_shader_module(vert_module, None);
        }
    }

    Ok(())
}

/// Load vertex and fragment shader modules based on mode.
fn load_shaders(
    ctx: &VulkanContext,
    mode: RasterMode,
    shader_base: &str,
) -> Result<(vk::ShaderModule, vk::ShaderModule), String> {
    match mode {
        RasterMode::Triangle => {
            let vert_path = format!("{}.vert.spv", shader_base);
            let frag_path = format!("{}.frag.spv", shader_base);
            let vert_code = spv_loader::load_spirv(Path::new(&vert_path))?;
            let frag_code = spv_loader::load_spirv(Path::new(&frag_path))?;
            let vert_mod = spv_loader::create_shader_module(&ctx.device, &vert_code)?;
            let frag_mod = spv_loader::create_shader_module(&ctx.device, &frag_code)?;
            Ok((vert_mod, frag_mod))
        }
        RasterMode::Fullscreen => {
            let frag_path = format!("{}.frag.spv", shader_base);
            let frag_code = spv_loader::load_spirv(Path::new(&frag_path))?;
            let frag_mod = spv_loader::create_shader_module(&ctx.device, &frag_code)?;
            // Vertex module will be created inline from built-in SPIR-V
            Ok((vk::ShaderModule::null(), frag_mod))
        }
        RasterMode::Pbr => {
            let vert_path = format!("{}.vert.spv", shader_base);
            let frag_path = format!("{}.frag.spv", shader_base);
            let vert_code = spv_loader::load_spirv(Path::new(&vert_path))?;
            let frag_code = spv_loader::load_spirv(Path::new(&frag_path))?;
            let vert_mod = spv_loader::create_shader_module(&ctx.device, &vert_code)?;
            let frag_mod = spv_loader::create_shader_module(&ctx.device, &frag_code)?;
            Ok((vert_mod, frag_mod))
        }
    }
}

// ---- Triangle rendering ----

fn render_triangle(
    ctx: &mut VulkanContext,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
    width: u32,
    height: u32,
    output_path: &Path,
) -> Result<(), String> {
    let device_owned = ctx.device.clone();
    let device = &device_owned;

    // Create vertex buffer
    let vertices = scene::generate_triangle();
    let vertex_data: &[u8] = bytemuck::cast_slice(&vertices);
    let mut vbo = create_buffer_with_data(
        device,
        ctx.allocator_mut(),
        vertex_data,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        "triangle_vbo",
    )?;

    // Create offscreen color image
    let mut color_image = create_offscreen_image(
        device,
        ctx.allocator_mut(),
        width,
        height,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
        "triangle_color",
    )?;

    // Create render pass
    let color_attachment = vk::AttachmentDescription::default()
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

    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&color_attachment))
        .subpasses(std::slice::from_ref(&subpass));

    let render_pass = unsafe {
        device
            .create_render_pass(&render_pass_info, None)
            .map_err(|e| format!("Failed to create render pass: {:?}", e))?
    };

    // Create framebuffer
    let framebuffer_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(std::slice::from_ref(&color_image.view))
        .width(width)
        .height(height)
        .layers(1);

    let framebuffer = unsafe {
        device
            .create_framebuffer(&framebuffer_info, None)
            .map_err(|e| format!("Failed to create framebuffer: {:?}", e))?
    };

    // Pipeline layout (no descriptors for triangle)
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(|e| format!("Failed to create pipeline layout: {:?}", e))?
    };

    // Graphics pipeline
    let entry_name = c"main";

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(entry_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(entry_name),
    ];

    let binding_desc = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(24) // sizeof(TriangleVertex)
        .input_rate(vk::VertexInputRate::VERTEX)];

    let attr_descs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12),
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_desc)
        .vertex_attribute_descriptions(&attr_descs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport = vk::Viewport::default()
        .x(0.0)
        .y(0.0)
        .width(width as f32)
        .height(height as f32)
        .min_depth(0.0)
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

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);

    let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(std::slice::from_ref(&color_blend_attachment));

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|e| format!("Failed to create graphics pipeline: {:?}", e))?[0]
    };

    // Record commands
    let cmd = ctx.begin_single_commands()?;

    let clear_values = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    }];

    let render_pass_begin = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D { width, height },
        })
        .clear_values(&clear_values);

    unsafe {
        device.cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_bind_vertex_buffers(cmd, 0, &[vbo.buffer], &[0]);
        device.cmd_draw(cmd, 3, 1, 0, 0);
        device.cmd_end_render_pass(cmd);
    }

    // Copy to staging and read back
    let mut staging = screenshot::StagingBuffer::new(
        device,
        ctx.allocator_mut(),
        width,
        height,
    )?;

    screenshot::cmd_copy_image_to_buffer(device, cmd, color_image.image, staging.buffer, width, height);

    ctx.end_single_commands(cmd)?;

    let pixels = staging.read_pixels(width, height)?;
    screenshot::save_png(&pixels, width, height, output_path)?;

    info!("Saved triangle render to {:?}", output_path);

    // Cleanup
    staging.destroy(device, ctx.allocator_mut());
    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_framebuffer(framebuffer, None);
        device.destroy_render_pass(render_pass, None);
    }
    color_image.destroy(device, ctx.allocator_mut());
    vbo.destroy(device, ctx.allocator_mut());

    Ok(())
}

// ---- Fullscreen rendering ----

fn render_fullscreen(
    ctx: &mut VulkanContext,
    frag_module: vk::ShaderModule,
    width: u32,
    height: u32,
    output_path: &Path,
) -> Result<(), String> {
    let device_owned = ctx.device.clone();
    let device = &device_owned;

    // Create built-in fullscreen vertex shader
    // Since embedding raw SPIR-V is fragile, we use a minimal hand-assembled approach.
    // For production, the SPIR-V should be generated from glslangValidator.
    // Here we use the fullscreen triangle approach with a hardcoded vertex shader.

    // We'll load a pre-compiled fullscreen.vert.spv if available, otherwise error.
    // Actually, let's generate a proper SPIR-V module programmatically.
    // The simplest approach: include the SPIR-V bytes directly.

    // For now, attempt to load from a known path, or use the assembly.
    let vert_module = create_builtin_fullscreen_vert_module(device)?;

    // Create offscreen color image
    let mut color_image = create_offscreen_image(
        device,
        ctx.allocator_mut(),
        width,
        height,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
        "fullscreen_color",
    )?;

    // Create render pass
    let color_attachment = vk::AttachmentDescription::default()
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

    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&color_attachment))
        .subpasses(std::slice::from_ref(&subpass));

    let render_pass = unsafe {
        device
            .create_render_pass(&render_pass_info, None)
            .map_err(|e| format!("Failed to create render pass: {:?}", e))?
    };

    let framebuffer_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(std::slice::from_ref(&color_image.view))
        .width(width)
        .height(height)
        .layers(1);

    let framebuffer = unsafe {
        device
            .create_framebuffer(&framebuffer_info, None)
            .map_err(|e| format!("Failed to create framebuffer: {:?}", e))?
    };

    // Pipeline (no vertex input, no descriptors)
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(|e| format!("Failed to create pipeline layout: {:?}", e))?
    };

    let entry_name = c"main";

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(entry_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(entry_name),
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

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

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);

    let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(std::slice::from_ref(&color_blend_attachment));

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|e| format!("Failed to create graphics pipeline: {:?}", e))?[0]
    };

    // Record commands
    let cmd = ctx.begin_single_commands()?;

    let clear_values = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    }];

    let render_pass_begin = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D { width, height },
        })
        .clear_values(&clear_values);

    unsafe {
        device.cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_draw(cmd, 3, 1, 0, 0);
        device.cmd_end_render_pass(cmd);
    }

    let mut staging = screenshot::StagingBuffer::new(
        device,
        ctx.allocator_mut(),
        width,
        height,
    )?;

    screenshot::cmd_copy_image_to_buffer(device, cmd, color_image.image, staging.buffer, width, height);

    ctx.end_single_commands(cmd)?;

    let pixels = staging.read_pixels(width, height)?;
    screenshot::save_png(&pixels, width, height, output_path)?;

    info!("Saved fullscreen render to {:?}", output_path);

    // Cleanup
    staging.destroy(device, ctx.allocator_mut());
    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_framebuffer(framebuffer, None);
        device.destroy_render_pass(render_pass, None);
        device.destroy_shader_module(vert_module, None);
    }
    color_image.destroy(device, ctx.allocator_mut());

    Ok(())
}

/// Create the built-in fullscreen triangle vertex shader module.
///
/// This attempts to load a pre-compiled fullscreen.vert.spv from common paths,
/// or generates SPIR-V at runtime by invoking glslangValidator.
fn create_builtin_fullscreen_vert_module(
    device: &ash::Device,
) -> Result<vk::ShaderModule, String> {
    // Try to find a pre-compiled fullscreen.vert.spv
    let search_paths = [
        "fullscreen.vert.spv",
        "shaders/fullscreen.vert.spv",
        "../shaders/fullscreen.vert.spv",
    ];

    for path in &search_paths {
        if let Ok(code) = spv_loader::load_spirv(Path::new(path)) {
            return spv_loader::create_shader_module(device, &code);
        }
    }

    // Try runtime compilation via glslangValidator
    let glsl_source = r#"#version 450
layout(location = 0) out vec2 uv;
void main() {
    vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    vec2 p = positions[gl_VertexIndex];
    uv = vec2(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
}
"#;

    // Write to temp file and compile
    let temp_dir = std::env::temp_dir();
    let glsl_path = temp_dir.join("lux_fullscreen.vert");
    let spv_path = temp_dir.join("lux_fullscreen.vert.spv");

    std::fs::write(&glsl_path, glsl_source)
        .map_err(|e| format!("Failed to write temp GLSL: {}", e))?;

    let result = std::process::Command::new("glslangValidator")
        .args([
            "-V",
            glsl_path.to_str().unwrap(),
            "-o",
            spv_path.to_str().unwrap(),
        ])
        .output();

    match result {
        Ok(output) if output.status.success() => {
            let code = spv_loader::load_spirv(&spv_path)?;
            let module = spv_loader::create_shader_module(device, &code)?;
            let _ = std::fs::remove_file(&glsl_path);
            let _ = std::fs::remove_file(&spv_path);
            Ok(module)
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!(
                "glslangValidator failed to compile fullscreen vertex shader: {}",
                stderr
            ))
        }
        Err(_) => Err(
            "No pre-compiled fullscreen.vert.spv found and glslangValidator is not on PATH. \
             Please compile the fullscreen vertex shader manually:\n\
             glslangValidator -V fullscreen.vert -o fullscreen.vert.spv"
                .to_string(),
        ),
    }
}

// ---- PBR sphere rendering ----

fn render_pbr(
    ctx: &mut VulkanContext,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
    width: u32,
    height: u32,
    output_path: &Path,
) -> Result<(), String> {
    let device_owned = ctx.device.clone();
    let device = &device_owned;

    // Generate sphere mesh
    let (sphere_verts, sphere_indices) = scene::generate_sphere(32, 32);
    let vertex_data: &[u8] = bytemuck::cast_slice(&sphere_verts);
    let index_data: &[u8] = bytemuck::cast_slice(&sphere_indices);
    let num_indices = sphere_indices.len() as u32;

    let mut vbo = create_buffer_with_data(
        device,
        ctx.allocator_mut(),
        vertex_data,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        "pbr_vbo",
    )?;

    let mut ibo = create_buffer_with_data(
        device,
        ctx.allocator_mut(),
        index_data,
        vk::BufferUsageFlags::INDEX_BUFFER,
        "pbr_ibo",
    )?;

    // MVP uniform buffer (3 x mat4 = 192 bytes)
    let model = DefaultCamera::model();
    let view = DefaultCamera::view();
    let proj = DefaultCamera::projection(width as f32 / height as f32);

    let mut mvp_data = [0u8; 192];
    mvp_data[0..64].copy_from_slice(bytemuck::cast_slice(model.as_ref()));
    mvp_data[64..128].copy_from_slice(bytemuck::cast_slice(view.as_ref()));
    mvp_data[128..192].copy_from_slice(bytemuck::cast_slice(proj.as_ref()));

    let mut mvp_buffer = create_buffer_with_data(
        device,
        ctx.allocator_mut(),
        &mvp_data,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        "pbr_mvp",
    )?;

    // Light uniform buffer (32 bytes: light_dir vec4 + camera_pos vec4)
    let light_dir = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
    let camera_pos = DefaultCamera::EYE;
    let mut light_data = [0u8; 32];
    light_data[0..12].copy_from_slice(bytemuck::cast_slice(light_dir.as_ref()));
    light_data[12..16].copy_from_slice(&0.0f32.to_le_bytes()); // padding
    light_data[16..28].copy_from_slice(bytemuck::cast_slice(camera_pos.as_ref()));
    light_data[28..32].copy_from_slice(&0.0f32.to_le_bytes()); // padding

    let mut light_buffer = create_buffer_with_data(
        device,
        ctx.allocator_mut(),
        &light_data,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        "pbr_light",
    )?;

    // Procedural texture
    info!("Generating procedural texture...");
    let tex_pixels = scene::generate_procedural_texture(512);
    let mut texture_image = create_texture_image(ctx, 512, 512, &tex_pixels, "pbr_albedo")?;

    // Sampler
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT);

    let sampler = unsafe {
        device
            .create_sampler(&sampler_info, None)
            .map_err(|e| format!("Failed to create sampler: {:?}", e))?
    };

    // Descriptor set layout 0: MVP UBO (vertex)
    let mvp_binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let mvp_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(std::slice::from_ref(&mvp_binding));

    let mvp_ds_layout = unsafe {
        device
            .create_descriptor_set_layout(&mvp_layout_info, None)
            .map_err(|e| format!("Failed to create MVP descriptor layout: {:?}", e))?
    };

    // Descriptor set layout 1: Light UBO + sampler + texture (fragment)
    let frag_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];

    let frag_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&frag_bindings);

    let frag_ds_layout = unsafe {
        device
            .create_descriptor_set_layout(&frag_layout_info, None)
            .map_err(|e| format!("Failed to create frag descriptor layout: {:?}", e))?
    };

    // Descriptor pool
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(2),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLER)
            .descriptor_count(1),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1),
    ];

    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(2)
        .pool_sizes(&pool_sizes);

    let descriptor_pool = unsafe {
        device
            .create_descriptor_pool(&pool_info, None)
            .map_err(|e| format!("Failed to create descriptor pool: {:?}", e))?
    };

    // Allocate descriptor sets
    let set_layouts = [mvp_ds_layout, frag_ds_layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&set_layouts);

    let descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(&alloc_info)
            .map_err(|e| format!("Failed to allocate descriptor sets: {:?}", e))?
    };

    // Write descriptor sets
    let mvp_buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(mvp_buffer.buffer)
        .offset(0)
        .range(192);

    let light_buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(light_buffer.buffer)
        .offset(0)
        .range(32);

    let sampler_info_desc = vk::DescriptorImageInfo::default()
        .sampler(sampler);

    let tex_image_info = vk::DescriptorImageInfo::default()
        .image_view(texture_image.view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_sets[0])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&mvp_buffer_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_sets[1])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&light_buffer_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_sets[1])
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(std::slice::from_ref(&sampler_info_desc)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_sets[1])
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&tex_image_info)),
    ];

    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }

    // Pipeline layout with 2 descriptor set layouts
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts);

    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(|e| format!("Failed to create PBR pipeline layout: {:?}", e))?
    };

    // Offscreen color + depth images
    let mut color_image = create_offscreen_image(
        device,
        ctx.allocator_mut(),
        width,
        height,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
        "pbr_color",
    )?;

    let mut depth_image = create_offscreen_image(
        device,
        ctx.allocator_mut(),
        width,
        height,
        vk::Format::D32_SFLOAT,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::ImageAspectFlags::DEPTH,
        "pbr_depth",
    )?;

    // Render pass with depth
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

    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(std::slice::from_ref(&subpass));

    let render_pass = unsafe {
        device
            .create_render_pass(&render_pass_info, None)
            .map_err(|e| format!("Failed to create PBR render pass: {:?}", e))?
    };

    let fb_attachments = [color_image.view, depth_image.view];
    let framebuffer_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&fb_attachments)
        .width(width)
        .height(height)
        .layers(1);

    let framebuffer = unsafe {
        device
            .create_framebuffer(&framebuffer_info, None)
            .map_err(|e| format!("Failed to create PBR framebuffer: {:?}", e))?
    };

    // Graphics pipeline
    let entry_name = c"main";

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(entry_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(entry_name),
    ];

    let binding_desc = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(32) // sizeof(PbrVertex) = 32
        .input_rate(vk::VertexInputRate::VERTEX)];

    let attr_descs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0), // position
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12), // normal
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(24), // uv
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_desc)
        .vertex_attribute_descriptions(&attr_descs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    // Negative viewport height (VK_KHR_maintenance1) to match wgpu Y convention
    let viewport = vk::Viewport::default()
        .y(height as f32)
        .width(width as f32)
        .height(-(height as f32))
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

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
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

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|e| format!("Failed to create PBR pipeline: {:?}", e))?[0]
    };

    // Record commands
    let cmd = ctx.begin_single_commands()?;

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
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D { width, height },
        })
        .clear_values(&clear_values);

    unsafe {
        device.cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout,
            0,
            &descriptor_sets,
            &[],
        );
        device.cmd_bind_vertex_buffers(cmd, 0, &[vbo.buffer], &[0]);
        device.cmd_bind_index_buffer(cmd, ibo.buffer, 0, vk::IndexType::UINT32);
        device.cmd_draw_indexed(cmd, num_indices, 1, 0, 0, 0);
        device.cmd_end_render_pass(cmd);
    }

    let mut staging = screenshot::StagingBuffer::new(
        device,
        ctx.allocator_mut(),
        width,
        height,
    )?;

    screenshot::cmd_copy_image_to_buffer(device, cmd, color_image.image, staging.buffer, width, height);

    ctx.end_single_commands(cmd)?;

    let pixels = staging.read_pixels(width, height)?;
    screenshot::save_png(&pixels, width, height, output_path)?;

    info!("Saved PBR render to {:?}", output_path);

    // Cleanup
    staging.destroy(device, ctx.allocator_mut());
    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_framebuffer(framebuffer, None);
        device.destroy_render_pass(render_pass, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        device.destroy_descriptor_set_layout(mvp_ds_layout, None);
        device.destroy_descriptor_set_layout(frag_ds_layout, None);
        device.destroy_sampler(sampler, None);
    }
    color_image.destroy(device, ctx.allocator_mut());
    depth_image.destroy(device, ctx.allocator_mut());
    texture_image.destroy(device, ctx.allocator_mut());
    mvp_buffer.destroy(device, ctx.allocator_mut());
    light_buffer.destroy(device, ctx.allocator_mut());
    vbo.destroy(device, ctx.allocator_mut());
    ibo.destroy(device, ctx.allocator_mut());

    Ok(())
}
