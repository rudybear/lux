//! Rasterization renderer: scene/pipeline architecture.
//!
//! Two public entry points:
//! - `render_raster()` — render a raster scene (sphere, triangle, or glTF) through a pipeline
//! - `render_fullscreen()` — render a fullscreen fragment shader

use ash::vk;
use bytemuck;
use gpu_allocator::vulkan::{Allocation, Allocator};
use log::info;
use std::collections::HashMap;
use std::path::Path;

use crate::camera::DefaultCamera;
use crate::reflected_pipeline;
use crate::scene;
use crate::scene_manager::{self, GpuBuffer, GpuImage, IblAssets};
use crate::screenshot;
use crate::spv_loader;
use crate::vulkan_context::VulkanContext;

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
    scene_manager::create_offscreen_image(device, allocator, width, height, format, usage, aspect, name)
}

/// Create a texture image from pixel data and upload it via a staging buffer.
/// Delegates to scene_manager::create_texture_image with FRAGMENT_SHADER dst_stage.
fn create_texture_image(
    ctx: &mut VulkanContext,
    width: u32,
    height: u32,
    pixels: &[u8],
    name: &str,
) -> Result<GpuImage, String> {
    scene_manager::create_texture_image(ctx, width, height, pixels, name, vk::PipelineStageFlags::FRAGMENT_SHADER)
}

/// Render a raster scene (sphere, triangle, or glTF) through a pipeline.
///
/// Three-phase architecture:
/// 1. Upload scene geometry to GPU
/// 2. Create pipeline from shaders
/// 3. Bind scene to pipeline and execute render
pub fn render_raster(
    ctx: &mut VulkanContext,
    pipeline_base: &str,
    scene_source: &str,
    width: u32,
    height: u32,
    output_path: &Path,
    ibl_name: &str,
) -> Result<(), String> {
    info!(
        "Raster render: scene='{}', pipeline='{}', {}x{}",
        scene_source, pipeline_base, width, height
    );

    match scene_source {
        "triangle" => {
            let vert_path = format!("{}.vert.spv", pipeline_base);
            let frag_path = format!("{}.frag.spv", pipeline_base);
            let vert_code = spv_loader::load_spirv(Path::new(&vert_path))?;
            let frag_code = spv_loader::load_spirv(Path::new(&frag_path))?;
            let vert_module = spv_loader::create_shader_module(&ctx.device, &vert_code)?;
            let frag_module = spv_loader::create_shader_module(&ctx.device, &frag_code)?;

            render_triangle_scene(ctx, vert_module, frag_module, width, height, output_path)?;

            unsafe {
                ctx.device.destroy_shader_module(frag_module, None);
                ctx.device.destroy_shader_module(vert_module, None);
            }
        }
        source if source.ends_with(".glb") || source.ends_with(".gltf") => {
            // glTF scene: load mesh, use PBR pipeline
            let vert_path = format!("{}.vert.spv", pipeline_base);
            let frag_path = format!("{}.frag.spv", pipeline_base);
            let vert_code = spv_loader::load_spirv(Path::new(&vert_path))?;
            let frag_code = spv_loader::load_spirv(Path::new(&frag_path))?;
            let vert_module = spv_loader::create_shader_module(&ctx.device, &vert_code)?;
            let frag_module = spv_loader::create_shader_module(&ctx.device, &frag_code)?;

            // For glTF, use the PBR rendering path with glTF mesh data
            render_pbr_scene(ctx, vert_module, frag_module, pipeline_base, scene_source, width, height, output_path, ibl_name)?;

            unsafe {
                ctx.device.destroy_shader_module(frag_module, None);
                ctx.device.destroy_shader_module(vert_module, None);
            }
        }
        _ => {
            // Default: sphere or other procedural scenes use PBR pipeline
            let vert_path = format!("{}.vert.spv", pipeline_base);
            let frag_path = format!("{}.frag.spv", pipeline_base);
            let vert_code = spv_loader::load_spirv(Path::new(&vert_path))?;
            let frag_code = spv_loader::load_spirv(Path::new(&frag_path))?;
            let vert_module = spv_loader::create_shader_module(&ctx.device, &vert_code)?;
            let frag_module = spv_loader::create_shader_module(&ctx.device, &frag_code)?;

            render_pbr_scene(ctx, vert_module, frag_module, pipeline_base, scene_source, width, height, output_path, ibl_name)?;

            unsafe {
                ctx.device.destroy_shader_module(frag_module, None);
                ctx.device.destroy_shader_module(vert_module, None);
            }
        }
    }

    Ok(())
}

/// Render a fullscreen fragment shader.
pub fn render_fullscreen(
    ctx: &mut VulkanContext,
    pipeline_base: &str,
    width: u32,
    height: u32,
    output_path: &Path,
    _ibl_name: &str,
) -> Result<(), String> {
    info!(
        "Fullscreen render: pipeline='{}', {}x{}",
        pipeline_base, width, height
    );

    let frag_path = format!("{}.frag.spv", pipeline_base);
    let frag_code = spv_loader::load_spirv(Path::new(&frag_path))?;
    let frag_module = spv_loader::create_shader_module(&ctx.device, &frag_code)?;

    render_fullscreen_scene(ctx, frag_module, width, height, output_path)?;

    unsafe {
        ctx.device.destroy_shader_module(frag_module, None);
    }

    Ok(())
}

// ---- Triangle scene rendering ----

fn render_triangle_scene(
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

// ---- Fullscreen scene rendering ----

fn render_fullscreen_scene(
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

// ---- PBR scene rendering (sphere, glTF, or other) ----
//
// Reflection-driven: reads .vert.json and .frag.json to determine descriptor
// set layouts, pool sizes, and binding assignments. No hardcoded layouts.

/// Create a 1x1 white RGBA default texture image (used when a texture slot
/// is declared in the reflection but no actual image data is available).
fn create_default_white_texture(ctx: &mut VulkanContext) -> Result<GpuImage, String> {
    let white_pixel: [u8; 4] = [255, 255, 255, 255];
    create_texture_image(ctx, 1, 1, &white_pixel, "default_white_1x1")
}

/// Delegate to scene_manager::create_cubemap_image with FRAGMENT_SHADER dst_stage.
#[allow(dead_code)]
fn create_cubemap_image(
    ctx: &mut VulkanContext,
    face_size: u32,
    mip_count: u32,
    format: vk::Format,
    pixel_data: &[u8],
    bytes_per_pixel: u32,
    name: &str,
) -> Result<(vk::Image, vk::ImageView, Allocation), String> {
    scene_manager::create_cubemap_image(ctx, face_size, mip_count, format, pixel_data, bytes_per_pixel, name, vk::PipelineStageFlags::FRAGMENT_SHADER)
}

/// Delegate to scene_manager::create_texture_image_raw with FRAGMENT_SHADER dst_stage.
#[allow(dead_code)]
fn create_texture_image_raw(
    ctx: &mut VulkanContext,
    width: u32,
    height: u32,
    pixel_data: &[u8],
    format: vk::Format,
    name: &str,
) -> Result<GpuImage, String> {
    scene_manager::create_texture_image_raw(ctx, width, height, pixel_data, format, name, vk::PipelineStageFlags::FRAGMENT_SHADER)
}

/// Delegate to scene_manager::load_ibl_assets with FRAGMENT_SHADER dst_stage.
fn load_ibl_assets(ctx: &mut VulkanContext, ibl_name: &str) -> IblAssets {
    scene_manager::load_ibl_assets(ctx, vk::PipelineStageFlags::FRAGMENT_SHADER, ibl_name)
}

fn render_pbr_scene(
    ctx: &mut VulkanContext,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
    pipeline_base: &str,
    scene_source: &str,
    width: u32,
    height: u32,
    output_path: &Path,
    ibl_name: &str,
) -> Result<(), String> {
    let device_owned = ctx.device.clone();
    let device = &device_owned;

    // =====================================================================
    // Phase 0: Load reflection JSON and build descriptor layouts
    // =====================================================================
    let vert_json_path = format!("{}.vert.json", pipeline_base);
    let frag_json_path = format!("{}.frag.json", pipeline_base);

    let vert_refl = reflected_pipeline::load_reflection(Path::new(&vert_json_path))?;
    let frag_refl = reflected_pipeline::load_reflection(Path::new(&frag_json_path))?;

    info!(
        "Reflection loaded: vert={} frag={} descriptor_sets: vert={:?}, frag={:?}",
        vert_json_path,
        frag_json_path,
        vert_refl.descriptor_sets.keys().collect::<Vec<_>>(),
        frag_refl.descriptor_sets.keys().collect::<Vec<_>>(),
    );

    // Merge descriptor sets from both stages
    let merged = reflected_pipeline::merge_descriptor_sets(&[&vert_refl, &frag_refl]);

    // Create VkDescriptorSetLayouts from reflection
    let ds_layouts = unsafe {
        reflected_pipeline::create_descriptor_set_layouts_from_merged(device, &merged)?
    };

    // Compute pool sizes
    let (pool_sizes, max_sets) = reflected_pipeline::compute_pool_sizes(&merged);

    info!(
        "Reflection-driven descriptors: {} sets, {} pool size entries",
        max_sets,
        pool_sizes.len()
    );

    // Create descriptor pool
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(max_sets)
        .pool_sizes(&pool_sizes);

    let descriptor_pool = unsafe {
        device
            .create_descriptor_pool(&pool_info, None)
            .map_err(|e| format!("Failed to create descriptor pool: {:?}", e))?
    };

    // Allocate descriptor sets (ordered by set index)
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
            .map_err(|e| format!("Failed to allocate descriptor sets: {:?}", e))?
    };

    // Build a map: set_index -> vk::DescriptorSet
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

    // Build push constant ranges from reflection data
    let mut push_ranges: Vec<vk::PushConstantRange> = Vec::new();
    for pc in &vert_refl.push_constants {
        push_ranges.push(
            vk::PushConstantRange::default()
                .stage_flags(reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags))
                .offset(0)
                .size(pc.size),
        );
    }
    for pc in &frag_refl.push_constants {
        push_ranges.push(
            vk::PushConstantRange::default()
                .stage_flags(reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags))
                .offset(0)
                .size(pc.size),
        );
    }

    // Build push constant data buffer from reflection fields
    let mut push_constant_data: Vec<u8> = Vec::new();
    let mut push_constant_stage_flags = vk::ShaderStageFlags::empty();
    {
        let all_pcs: Vec<&reflected_pipeline::PushConstantInfo> = vert_refl
            .push_constants
            .iter()
            .chain(frag_refl.push_constants.iter())
            .collect();
        if !all_pcs.is_empty() {
            // Use the max size across all push constant blocks
            let total_size = all_pcs.iter().map(|pc| pc.size as usize).max().unwrap_or(0);
            push_constant_data.resize(total_size, 0u8);
            for pc in &all_pcs {
                push_constant_stage_flags |=
                    reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags);
                for field in &pc.fields {
                    let offset = field.offset as usize;
                    match field.name.as_str() {
                        "light_dir" => {
                            let v = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
                            let bytes = bytemuck::cast_slice::<f32, u8>(v.as_ref());
                            push_constant_data[offset..offset + bytes.len()]
                                .copy_from_slice(bytes);
                        }
                        "view_pos" => {
                            let v = glam::Vec3::new(0.0, 0.0, 3.0);
                            let bytes = bytemuck::cast_slice::<f32, u8>(v.as_ref());
                            push_constant_data[offset..offset + bytes.len()]
                                .copy_from_slice(bytes);
                        }
                        _ => {
                            info!(
                                "Unknown push constant field '{}', leaving zeroed",
                                field.name
                            );
                        }
                    }
                }
            }
        }
    }

    // Pipeline layout
    let pipeline_layout = unsafe {
        reflected_pipeline::create_pipeline_layout_from_merged(device, &ds_layouts, &push_ranges)?
    };

    // =====================================================================
    // Phase 1: Load scene geometry
    // =====================================================================
    let is_gltf = scene_source.ends_with(".glb") || scene_source.ends_with(".gltf");

    // Load glTF scene (if applicable) before we start using textures
    let mut gltf_scene = if is_gltf {
        Some(crate::gltf_loader::load_gltf(Path::new(scene_source))?)
    } else {
        None
    };

    // Determine if we need 48-byte vertices (with tangent) based on reflection stride
    let pipeline_vertex_stride = vert_refl.vertex_stride;

    // Auto-camera data computed from scene bounds
    let mut auto_eye = glam::Vec3::new(0.0, 0.0, 3.0);
    let mut auto_target = glam::Vec3::ZERO;
    let mut auto_up = glam::Vec3::new(0.0, 1.0, 0.0);
    let mut auto_far = 100.0f32;
    let mut has_scene_bounds = false;

    let (vertex_data_owned, pbr_indices) = if let Some(ref mut gltf_s) = gltf_scene {
        if gltf_s.meshes.is_empty() {
            return Err(format!("No meshes found in glTF file: {}", scene_source));
        }
        let draw_items = crate::gltf_loader::flatten_scene(gltf_s);

        if draw_items.is_empty() {
            return Err(format!("No draw items in glTF scene: {}", scene_source));
        }

        // Pack ALL meshes with world transforms
        let mut vdata: Vec<f32> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();
        let mut vertex_offset: u32 = 0;
        let mut bbox_min = glam::Vec3::splat(f32::MAX);
        let mut bbox_max = glam::Vec3::splat(f32::MIN);
        let mut positive_normals: i32 = 0;
        let mut negative_normals: i32 = 0;

        for item in &draw_items {
            let gmesh = &gltf_s.meshes[item.mesh_index];
            let world = item.world_transform;
            let upper3x3 = glam::Mat3::from_mat4(world);
            let normal_mat = upper3x3.inverse().transpose();

            for gv in &gmesh.vertices {
                let pos = world.transform_point3(glam::Vec3::from(gv.position));
                let norm = (normal_mat * glam::Vec3::from(gv.normal)).normalize();

                bbox_min = bbox_min.min(pos);
                bbox_max = bbox_max.max(pos);

                vdata.push(pos.x); vdata.push(pos.y); vdata.push(pos.z);
                vdata.push(norm.x); vdata.push(norm.y); vdata.push(norm.z);
                vdata.push(gv.uv[0]); vdata.push(gv.uv[1]);

                // Only pack tangent data when the shader expects 48-byte vertices
                if pipeline_vertex_stride >= 48 {
                    if gmesh.has_tangents {
                        let tang3 = (upper3x3 * glam::Vec3::new(gv.tangent[0], gv.tangent[1], gv.tangent[2])).normalize();
                        vdata.push(tang3.x); vdata.push(tang3.y); vdata.push(tang3.z);
                        vdata.push(gv.tangent[3]);
                    } else {
                        vdata.push(1.0); vdata.push(0.0); vdata.push(0.0); vdata.push(1.0);
                    }
                }
            }

            for &idx in &gmesh.indices {
                all_indices.push(idx + vertex_offset);
            }
            vertex_offset += gmesh.vertices.len() as u32;
        }

        // Compute auto-camera from scene bounds
        has_scene_bounds = true;
        {
            let center = (bbox_min + bbox_max) * 0.5;
            let extent = bbox_max - bbox_min;
            let max_extent = extent.x.max(extent.y).max(extent.z);

            // Find thinnest axis
            let mut thin_axis = 0usize;
            let ext = [extent.x, extent.y, extent.z];
            if ext[1] < ext[thin_axis] { thin_axis = 1; }
            if ext[2] < ext[thin_axis] { thin_axis = 2; }

            // Count normals to determine front direction
            // Floats per vertex depends on stride: 12 (48B) or 8 (32B)
            let floats_per_vert = (pipeline_vertex_stride / 4) as usize;
            let num_verts = vdata.len() / floats_per_vert;
            for i in 0..num_verts {
                let n_comp = vdata[i * floats_per_vert + 3 + thin_axis];
                if n_comp > 0.0 { positive_normals += 1; }
                else if n_comp < 0.0 { negative_normals += 1; }
            }
            // Camera goes OPPOSITE to where normals face (glTF convention:
            // model front faces -Z, camera at +Z looking along -Z).
            let view_sign: f32 = if negative_normals > positive_normals { 1.0 } else { -1.0 };

            let mut view_dir = glam::Vec3::ZERO;
            match thin_axis {
                0 => view_dir.x = view_sign,
                1 => view_dir.y = view_sign,
                _ => view_dir.z = view_sign,
            }
            let fov_y = 45.0f32.to_radians();
            let distance = max_extent * 1.2 / (2.0 * (fov_y / 2.0).tan());
            let mut eye = center + view_dir * distance;

            // Add slight elevation (15 degrees)
            let up_axis: usize = if thin_axis == 1 { 0 } else { 1 };
            let elev = 5.0f32.to_radians().sin() * distance;
            match up_axis {
                0 => eye.x += elev,
                1 => eye.y += elev,
                _ => eye.z += elev,
            }
            let mut up = glam::Vec3::ZERO;
            match up_axis {
                0 => up.x = 1.0,
                1 => up.y = 1.0,
                _ => up.z = 1.0,
            }

            auto_eye = eye;
            auto_target = center;
            auto_up = up;
            auto_far = (distance * 3.0f32).max(100.0);
            println!("[info] Auto-camera: eye=({:.2},{:.2},{:.2}) target=({:.2},{:.2},{:.2})",
                     eye.x, eye.y, eye.z, center.x, center.y, center.z);
        }

        let data: Vec<u8> = bytemuck::cast_slice(&vdata).to_vec();
        (data, all_indices)
    } else {
        let (sphere_verts, sphere_indices) = scene::generate_sphere(32, 32);

        if pipeline_vertex_stride == 48 {
            // Pad sphere PbrVertex (32 bytes) to 48 bytes with default tangent
            let default_tangent: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
            let mut padded_data: Vec<u8> = Vec::with_capacity(sphere_verts.len() * 48);
            for v in &sphere_verts {
                padded_data.extend_from_slice(bytemuck::cast_slice(&[*v]));
                padded_data.extend_from_slice(bytemuck::cast_slice(&default_tangent));
            }
            (padded_data, sphere_indices)
        } else {
            let data: Vec<u8> = bytemuck::cast_slice(&sphere_verts).to_vec();
            (data, sphere_indices)
        }
    };

    let vertex_data: &[u8] = &vertex_data_owned;
    let index_data: &[u8] = bytemuck::cast_slice(&pbr_indices);
    let num_indices = pbr_indices.len() as u32;

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

    // =====================================================================
    // Phase 2: Create uniform buffers and textures based on reflection
    // =====================================================================

    // MVP uniform buffer (192 bytes)
    let aspect = width as f32 / height as f32;
    let (model, view, proj): (glam::Mat4, glam::Mat4, glam::Mat4) = if has_scene_bounds {
        let m = glam::Mat4::IDENTITY; // transforms baked into vertices
        let v = crate::camera::look_at(auto_eye, auto_target, auto_up);
        let p = crate::camera::perspective(45.0f32.to_radians(), aspect, 0.1, auto_far);
        (m, v, p)
    } else {
        (DefaultCamera::model(), DefaultCamera::view(), DefaultCamera::projection(aspect))
    };

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

    // Light uniform buffer (32 bytes)
    let light_dir = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
    let camera_pos = if has_scene_bounds { auto_eye } else { DefaultCamera::EYE };
    let mut light_data = [0u8; 32];
    light_data[0..12].copy_from_slice(bytemuck::cast_slice(light_dir.as_ref()));
    light_data[12..16].copy_from_slice(&0.0f32.to_le_bytes());
    light_data[16..28].copy_from_slice(bytemuck::cast_slice(camera_pos.as_ref()));
    light_data[28..32].copy_from_slice(&0.0f32.to_le_bytes());

    let mut light_buffer = create_buffer_with_data(
        device,
        ctx.allocator_mut(),
        &light_data,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        "pbr_light",
    )?;

    // Sampler (shared by all texture bindings)
    let sampler_create = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT);

    let sampler = unsafe {
        device
            .create_sampler(&sampler_create, None)
            .map_err(|e| format!("Failed to create sampler: {:?}", e))?
    };

    // Get glTF material (if any) for texture extraction
    let gltf_material = gltf_scene.as_ref().and_then(|s| {
        if !s.materials.is_empty() { Some(&s.materials[0]) } else { None }
    });

    // Create default textures for missing texture slots
    let mut default_texture = create_default_white_texture(ctx)?;
    // Black texture for emissive (no emission)
    let black_pixel = vec![0u8, 0, 0, 255];
    let mut default_black_texture = create_texture_image(ctx, 1, 1, &black_pixel, "default_black")?;
    // Flat normal texture (128,128,255) = tangent-space (0,0,1)
    let normal_pixel = vec![128u8, 128, 255, 255];
    let mut default_normal_texture = create_texture_image(ctx, 1, 1, &normal_pixel, "default_normal")?;

    // Create textures from glTF images. We track them by name for binding.
    // Name -> GpuImage mapping
    let mut texture_images: HashMap<String, GpuImage> = HashMap::new();

    // Helper: try to get a glTF texture image by name, or use default
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
            "metallic_roughness_tex" => gltf_material.and_then(|m| m.metallic_roughness_image.as_ref()),
            "occlusion_tex" => gltf_material.and_then(|m| m.occlusion_image.as_ref()),
            "emissive_tex" => gltf_material.and_then(|m| m.emissive_image.as_ref()),
            _ => None,
        };

        if let Some(tex_data) = tex_image_opt {
            info!(
                "Uploading texture '{}': {}x{}",
                tex_name, tex_data.width, tex_data.height
            );
            let gpu_img = create_texture_image(
                ctx,
                tex_data.width,
                tex_data.height,
                &tex_data.pixels,
                tex_name,
            )?;
            texture_images.insert(tex_name.to_string(), gpu_img);
        }
        // If no glTF image, we will use the default texture
    }

    // For non-glTF scenes, generate a procedural texture and use it for albedo_tex / base_color_tex
    if gltf_material.is_none() {
        info!("Generating procedural texture for non-glTF scene...");
        let tex_pixels = scene::generate_procedural_texture(512);
        let proc_texture = create_texture_image(ctx, 512, 512, &tex_pixels, "procedural_albedo")?;
        texture_images.insert("albedo_tex".to_string(), proc_texture);
        // Also map base_color_tex to the same procedural texture
        let proc_texture2 = create_texture_image(ctx, 512, 512, &tex_pixels, "procedural_base_color")?;
        texture_images.insert("base_color_tex".to_string(), proc_texture2);
    }

    // Load IBL assets (specular cubemap, irradiance cubemap, BRDF LUT)
    let mut ibl_assets = load_ibl_assets(ctx, ibl_name);

    // =====================================================================
    // Phase 3: Write descriptor sets from reflection data
    // =====================================================================

    // We need to keep descriptor write infos alive during update_descriptor_sets
    // Use a Vec to hold them
    let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
    let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
    let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();

    // Pre-allocate buffer and image infos so pointers remain stable
    // Count total bindings first
    let total_bindings: usize = merged.values().map(|v| v.len()).sum();
    buffer_infos.reserve(total_bindings);
    image_infos.reserve(total_bindings);

    for (&set_idx, bindings) in &merged {
        let ds = match ds_map.get(&set_idx) {
            Some(&ds) => ds,
            None => continue,
        };

        for binding in bindings {
            let vk_type = reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);

            match vk_type {
                vk::DescriptorType::UNIFORM_BUFFER => {
                    // Match by name
                    let (buffer, range) = match binding.name.as_str() {
                        "MVP" => (mvp_buffer.buffer, binding.size as u64),
                        "Light" => (light_buffer.buffer, binding.size as u64),
                        _ => {
                            info!(
                                "Unknown UBO name '{}' at set={} binding={}, skipping",
                                binding.name, set_idx, binding.binding
                            );
                            continue;
                        }
                    };

                    let buf_info_idx = buffer_infos.len();
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
                                    &buffer_infos[buf_info_idx] as *const _,
                                    1,
                                )
                            }),
                    );
                }
                vk::DescriptorType::SAMPLER => {
                    // Use IBL-specific sampler if the binding name matches an IBL texture,
                    // otherwise use the default sampler.
                    let actual_sampler = ibl_assets
                        .sampler_for_binding(&binding.name)
                        .unwrap_or(sampler);

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
                    let is_cube = reflected_pipeline::is_cube_image_binding(&binding.binding_type);

                    if is_cube {
                        // Cubemap binding: use IBL textures or fall back to default
                        let view = ibl_assets
                            .view_for_binding(&binding.name)
                            .unwrap_or(default_texture.view);

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
                    } else if binding.name == "brdf_lut" {
                        // BRDF LUT is a 2D image, not a cubemap
                        let view = ibl_assets
                            .view_for_binding(&binding.name)
                            .unwrap_or(default_texture.view);

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
                    } else {
                        // Regular 2D texture: look up by name, fall back to correct default
                        let fallback_view = match binding.name.as_str() {
                            "emissive_tex" => default_black_texture.view,
                            "normal_tex" => default_normal_texture.view,
                            _ => default_texture.view,
                        };
                        let view = texture_images
                            .get(&binding.name)
                            .map(|img| img.view)
                            .unwrap_or(fallback_view);

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

    // =====================================================================
    // Phase 4: Create render pass, framebuffer, graphics pipeline
    // =====================================================================

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

    // Use reflection-driven vertex input (stride and attributes from vert_refl)
    let (binding_desc, attr_descs) =
        reflected_pipeline::create_reflected_vertex_input(&vert_refl);

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_desc)
        .vertex_attribute_descriptions(&attr_descs);

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

    // =====================================================================
    // Phase 5: Record commands and render
    // =====================================================================

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
        if !push_constant_data.is_empty() {
            device.cmd_push_constants(
                cmd,
                pipeline_layout,
                push_constant_stage_flags,
                0,
                &push_constant_data,
            );
        }
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

    info!("Saved raster scene render to {:?}", output_path);

    // =====================================================================
    // Cleanup
    // =====================================================================
    staging.destroy(device, ctx.allocator_mut());
    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_framebuffer(framebuffer, None);
        device.destroy_render_pass(render_pass, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        for (_, layout) in &ds_layouts {
            device.destroy_descriptor_set_layout(*layout, None);
        }
        device.destroy_sampler(sampler, None);
    }
    color_image.destroy(device, ctx.allocator_mut());
    depth_image.destroy(device, ctx.allocator_mut());
    default_texture.destroy(device, ctx.allocator_mut());
    default_black_texture.destroy(device, ctx.allocator_mut());
    default_normal_texture.destroy(device, ctx.allocator_mut());
    for (_, mut tex) in texture_images {
        tex.destroy(device, ctx.allocator_mut());
    }
    ibl_assets.destroy(device, ctx.allocator_mut());
    mvp_buffer.destroy(device, ctx.allocator_mut());
    light_buffer.destroy(device, ctx.allocator_mut());
    vbo.destroy(device, ctx.allocator_mut());
    ibo.destroy(device, ctx.allocator_mut());

    Ok(())
}

// =========================================================================
// PersistentRenderer — holds GPU resources for interactive multi-frame rendering
// =========================================================================

/// Persistent PBR renderer for interactive mode.
///
/// Holds all GPU resources (pipeline, descriptors, textures, mesh buffers)
/// across frames. Supports camera updates and re-rendering.
pub struct PersistentRenderer {
    // Pipeline
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    ds_layouts: HashMap<u32, vk::DescriptorSetLayout>,
    descriptor_sets: Vec<vk::DescriptorSet>,

    // Shader modules
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,

    // Render target (offscreen)
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    color_image: GpuImage,
    depth_image: GpuImage,
    width: u32,
    height: u32,

    // Geometry
    vbo: GpuBuffer,
    ibo: GpuBuffer,
    num_indices: u32,

    // Uniforms
    mvp_buffer: GpuBuffer,
    light_buffer: GpuBuffer,

    // Push constants
    push_constant_data: Vec<u8>,
    push_constant_stage_flags: vk::ShaderStageFlags,

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

impl PersistentRenderer {
    /// Initialize the persistent renderer with all GPU resources.
    pub fn init(
        ctx: &mut VulkanContext,
        pipeline_base: &str,
        scene_source: &str,
        width: u32,
        height: u32,
        ibl_name: &str,
    ) -> Result<Self, String> {
        let device = ctx.device.clone();

        // Load shaders
        let vert_path = format!("{}.vert.spv", pipeline_base);
        let frag_path = format!("{}.frag.spv", pipeline_base);
        let vert_code = crate::spv_loader::load_spirv(std::path::Path::new(&vert_path))?;
        let frag_code = crate::spv_loader::load_spirv(std::path::Path::new(&frag_path))?;
        let vert_module = crate::spv_loader::create_shader_module(&device, &vert_code)?;
        let frag_module = crate::spv_loader::create_shader_module(&device, &frag_code)?;

        // Phase 0: Reflection
        let vert_json_path = format!("{}.vert.json", pipeline_base);
        let frag_json_path = format!("{}.frag.json", pipeline_base);
        let vert_refl = crate::reflected_pipeline::load_reflection(std::path::Path::new(&vert_json_path))?;
        let frag_refl = crate::reflected_pipeline::load_reflection(std::path::Path::new(&frag_json_path))?;

        let merged = crate::reflected_pipeline::merge_descriptor_sets(&[&vert_refl, &frag_refl]);

        let ds_layouts = unsafe {
            crate::reflected_pipeline::create_descriptor_set_layouts_from_merged(&device, &merged)?
        };

        let (pool_sizes, max_sets) = crate::reflected_pipeline::compute_pool_sizes(&merged);

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| format!("Failed to create descriptor pool: {:?}", e))?
        };

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
                .map_err(|e| format!("Failed to allocate descriptor sets: {:?}", e))?
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

        // Push constant ranges
        let mut push_ranges: Vec<vk::PushConstantRange> = Vec::new();
        for pc in &vert_refl.push_constants {
            push_ranges.push(
                vk::PushConstantRange::default()
                    .stage_flags(crate::reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags))
                    .offset(0)
                    .size(pc.size),
            );
        }
        for pc in &frag_refl.push_constants {
            push_ranges.push(
                vk::PushConstantRange::default()
                    .stage_flags(crate::reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags))
                    .offset(0)
                    .size(pc.size),
            );
        }

        // Push constant data
        let mut push_constant_data: Vec<u8> = Vec::new();
        let mut push_constant_stage_flags = vk::ShaderStageFlags::empty();
        {
            let all_pcs: Vec<&crate::reflected_pipeline::PushConstantInfo> = vert_refl
                .push_constants.iter()
                .chain(frag_refl.push_constants.iter())
                .collect();
            if !all_pcs.is_empty() {
                let total_size = all_pcs.iter().map(|pc| pc.size as usize).max().unwrap_or(0);
                push_constant_data.resize(total_size, 0u8);
                for pc in &all_pcs {
                    push_constant_stage_flags |=
                        crate::reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags);
                    for field in &pc.fields {
                        let offset = field.offset as usize;
                        match field.name.as_str() {
                            "light_dir" => {
                                let v = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
                                let bytes = bytemuck::cast_slice::<f32, u8>(v.as_ref());
                                push_constant_data[offset..offset + bytes.len()].copy_from_slice(bytes);
                            }
                            "view_pos" => {
                                let v = glam::Vec3::new(0.0, 0.0, 3.0);
                                let bytes = bytemuck::cast_slice::<f32, u8>(v.as_ref());
                                push_constant_data[offset..offset + bytes.len()].copy_from_slice(bytes);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        let pipeline_layout = unsafe {
            crate::reflected_pipeline::create_pipeline_layout_from_merged(&device, &ds_layouts, &push_ranges)?
        };

        // Phase 1: Scene geometry
        let is_gltf = scene_source.ends_with(".glb") || scene_source.ends_with(".gltf");
        let mut gltf_scene = if is_gltf {
            Some(crate::gltf_loader::load_gltf(std::path::Path::new(scene_source))?)
        } else {
            None
        };

        let pipeline_vertex_stride = vert_refl.vertex_stride;
        let mut auto_eye = glam::Vec3::new(0.0, 0.0, 3.0);
        let mut auto_target = glam::Vec3::ZERO;
        let mut auto_up = glam::Vec3::new(0.0, 1.0, 0.0);
        let mut auto_far = 100.0f32;
        let mut has_scene_bounds = false;

        let (vertex_data_owned, pbr_indices) = if let Some(ref mut gltf_s) = gltf_scene {
            if gltf_s.meshes.is_empty() {
                return Err(format!("No meshes found in glTF file: {}", scene_source));
            }
            let draw_items = crate::gltf_loader::flatten_scene(gltf_s);
            if draw_items.is_empty() {
                return Err(format!("No draw items in glTF scene: {}", scene_source));
            }

            let mut vdata: Vec<f32> = Vec::new();
            let mut all_indices: Vec<u32> = Vec::new();
            let mut vertex_offset: u32 = 0;
            let mut vertex_positions: Vec<[f32; 3]> = Vec::new();

            for item in &draw_items {
                let gmesh = &gltf_s.meshes[item.mesh_index];
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

                    // Only pack tangent data when the shader expects 48-byte vertices
                    if pipeline_vertex_stride >= 48 {
                        if gmesh.has_tangents {
                            let tang3 = (upper3x3 * glam::Vec3::new(gv.tangent[0], gv.tangent[1], gv.tangent[2])).normalize();
                            vdata.push(tang3.x); vdata.push(tang3.y); vdata.push(tang3.z);
                            vdata.push(gv.tangent[3]);
                        } else {
                            vdata.push(1.0); vdata.push(0.0); vdata.push(0.0); vdata.push(1.0);
                        }
                    }
                }

                for &idx in &gmesh.indices {
                    all_indices.push(idx + vertex_offset);
                }
                vertex_offset += gmesh.vertices.len() as u32;
            }

            // Compute auto-camera using shared function (matches all engines)
            has_scene_bounds = true;
            {
                let (eye, target, up, far) = scene_manager::compute_auto_camera_from_draw_items(
                    &vertex_positions,
                    &draw_items,
                );
                auto_eye = eye;
                auto_target = target;
                auto_up = up;
                auto_far = far;
            }

            let data: Vec<u8> = bytemuck::cast_slice(&vdata).to_vec();
            (data, all_indices)
        } else {
            let (sphere_verts, sphere_indices) = scene::generate_sphere(32, 32);
            if pipeline_vertex_stride == 48 {
                let default_tangent: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
                let mut padded_data: Vec<u8> = Vec::with_capacity(sphere_verts.len() * 48);
                for v in &sphere_verts {
                    padded_data.extend_from_slice(bytemuck::cast_slice(&[*v]));
                    padded_data.extend_from_slice(bytemuck::cast_slice(&default_tangent));
                }
                (padded_data, sphere_indices)
            } else {
                let data: Vec<u8> = bytemuck::cast_slice(&sphere_verts).to_vec();
                (data, sphere_indices)
            }
        };

        let num_indices = pbr_indices.len() as u32;

        let vbo = create_buffer_with_data(
            &device, ctx.allocator_mut(), &vertex_data_owned,
            vk::BufferUsageFlags::VERTEX_BUFFER, "pbr_vbo",
        )?;
        let ibo = create_buffer_with_data(
            &device, ctx.allocator_mut(), bytemuck::cast_slice(&pbr_indices),
            vk::BufferUsageFlags::INDEX_BUFFER, "pbr_ibo",
        )?;

        // Phase 2: Uniforms and textures
        let aspect = width as f32 / height as f32;
        let (model, view, proj) = if has_scene_bounds {
            (
                glam::Mat4::IDENTITY,
                crate::camera::look_at(auto_eye, auto_target, auto_up),
                crate::camera::perspective(45.0f32.to_radians(), aspect, 0.1, auto_far),
            )
        } else {
            (DefaultCamera::model(), DefaultCamera::view(), DefaultCamera::projection(aspect))
        };

        let mut mvp_data = [0u8; 192];
        mvp_data[0..64].copy_from_slice(bytemuck::cast_slice(model.as_ref()));
        mvp_data[64..128].copy_from_slice(bytemuck::cast_slice(view.as_ref()));
        mvp_data[128..192].copy_from_slice(bytemuck::cast_slice(proj.as_ref()));

        let mvp_buffer = create_buffer_with_data(
            &device, ctx.allocator_mut(), &mvp_data,
            vk::BufferUsageFlags::UNIFORM_BUFFER, "pbr_mvp",
        )?;

        let light_dir = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
        let camera_pos = if has_scene_bounds { auto_eye } else { DefaultCamera::EYE };
        let mut light_data = [0u8; 32];
        light_data[0..12].copy_from_slice(bytemuck::cast_slice(light_dir.as_ref()));
        light_data[12..16].copy_from_slice(&0.0f32.to_le_bytes());
        light_data[16..28].copy_from_slice(bytemuck::cast_slice(camera_pos.as_ref()));
        light_data[28..32].copy_from_slice(&0.0f32.to_le_bytes());

        let light_buffer = create_buffer_with_data(
            &device, ctx.allocator_mut(), &light_data,
            vk::BufferUsageFlags::UNIFORM_BUFFER, "pbr_light",
        )?;

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
            if !s.materials.is_empty() { Some(&s.materials[0]) } else { None }
        });

        let default_texture = create_default_white_texture(ctx)?;
        let default_black_texture = create_texture_image(ctx, 1, 1, &[0u8, 0, 0, 255], "default_black")?;
        let default_normal_texture = create_texture_image(ctx, 1, 1, &[128u8, 128, 255, 255], "default_normal")?;

        let mut texture_images: HashMap<String, GpuImage> = HashMap::new();
        let texture_names = [
            "base_color_tex", "normal_tex", "metallic_roughness_tex",
            "occlusion_tex", "emissive_tex", "albedo_tex",
        ];
        for tex_name in &texture_names {
            let tex_image_opt = match *tex_name {
                "base_color_tex" | "albedo_tex" => gltf_material.and_then(|m| m.base_color_image.as_ref()),
                "normal_tex" => gltf_material.and_then(|m| m.normal_image.as_ref()),
                "metallic_roughness_tex" => gltf_material.and_then(|m| m.metallic_roughness_image.as_ref()),
                "occlusion_tex" => gltf_material.and_then(|m| m.occlusion_image.as_ref()),
                "emissive_tex" => gltf_material.and_then(|m| m.emissive_image.as_ref()),
                _ => None,
            };
            if let Some(tex_data) = tex_image_opt {
                let gpu_img = create_texture_image(ctx, tex_data.width, tex_data.height, &tex_data.pixels, tex_name)?;
                texture_images.insert(tex_name.to_string(), gpu_img);
            }
        }

        if gltf_material.is_none() {
            let tex_pixels = scene::generate_procedural_texture(512);
            let proc_texture = create_texture_image(ctx, 512, 512, &tex_pixels, "procedural_albedo")?;
            texture_images.insert("albedo_tex".to_string(), proc_texture);
            let proc_texture2 = create_texture_image(ctx, 512, 512, &tex_pixels, "procedural_base_color")?;
            texture_images.insert("base_color_tex".to_string(), proc_texture2);
        }

        let ibl_assets = load_ibl_assets(ctx, ibl_name);

        // Phase 3: Write descriptor sets
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
                let vk_type = crate::reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);
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
                                .buffer(buffer).offset(0).range(range),
                        );
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds).dst_binding(binding.binding)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .buffer_info(unsafe {
                                    std::slice::from_raw_parts(&buffer_infos[idx] as *const _, 1)
                                }),
                        );
                    }
                    vk::DescriptorType::SAMPLER => {
                        let actual_sampler = ibl_assets
                            .sampler_for_binding(&binding.name)
                            .unwrap_or(sampler);
                        let idx = image_infos.len();
                        image_infos.push(vk::DescriptorImageInfo::default().sampler(actual_sampler));
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds).dst_binding(binding.binding)
                                .descriptor_type(vk::DescriptorType::SAMPLER)
                                .image_info(unsafe {
                                    std::slice::from_raw_parts(&image_infos[idx] as *const _, 1)
                                }),
                        );
                    }
                    vk::DescriptorType::SAMPLED_IMAGE => {
                        let is_cube = crate::reflected_pipeline::is_cube_image_binding(&binding.binding_type);
                        let view = if is_cube {
                            ibl_assets.view_for_binding(&binding.name).unwrap_or(default_texture.view)
                        } else if binding.name == "brdf_lut" {
                            ibl_assets.view_for_binding(&binding.name).unwrap_or(default_texture.view)
                        } else {
                            let fallback = match binding.name.as_str() {
                                "emissive_tex" => default_black_texture.view,
                                "normal_tex" => default_normal_texture.view,
                                _ => default_texture.view,
                            };
                            texture_images.get(&binding.name).map(|img| img.view).unwrap_or(fallback)
                        };
                        let idx = image_infos.len();
                        image_infos.push(
                            vk::DescriptorImageInfo::default()
                                .image_view(view)
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                        );
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds).dst_binding(binding.binding)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .image_info(unsafe {
                                    std::slice::from_raw_parts(&image_infos[idx] as *const _, 1)
                                }),
                        );
                    }
                    _ => {}
                }
            }
        }

        unsafe { device.update_descriptor_sets(&writes, &[]); }

        // Phase 4: Render pass, framebuffer, pipeline
        let color_image = create_offscreen_image(
            &device, ctx.allocator_mut(), width, height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR, "pbr_color",
        )?;

        let depth_image = create_offscreen_image(
            &device, ctx.allocator_mut(), width, height,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH, "pbr_depth",
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
            .attachment(0).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_ref = vk::AttachmentReference::default()
            .attachment(1).layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

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
                .map_err(|e| format!("Failed to create render pass: {:?}", e))?
        };

        let fb_attachments = [color_image.view, depth_image.view];
        let framebuffer = unsafe {
            device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass)
                        .attachments(&fb_attachments)
                        .width(width).height(height).layers(1),
                    None,
                )
                .map_err(|e| format!("Failed to create framebuffer: {:?}", e))?
        };

        let entry_name = c"main";
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module).name(entry_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module).name(entry_name),
        ];

        let (binding_desc, attr_descs) =
            crate::reflected_pipeline::create_reflected_vertex_input(&vert_refl);

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_desc)
            .vertex_attribute_descriptions(&attr_descs);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport = vk::Viewport::default()
            .width(width as f32).height(height as f32).max_depth(1.0);
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
            .depth_test_enable(true).depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
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
                        .subpass(0)],
                    None,
                )
                .map_err(|e| format!("Failed to create pipeline: {:?}", e))?[0]
        };

        info!("PersistentRenderer initialized: {}x{}", width, height);

        Ok(PersistentRenderer {
            pipeline,
            pipeline_layout,
            descriptor_pool,
            ds_layouts,
            descriptor_sets,
            vert_module,
            frag_module,
            render_pass,
            framebuffer,
            color_image,
            depth_image,
            width,
            height,
            vbo,
            ibo,
            num_indices,
            mvp_buffer,
            light_buffer,
            push_constant_data,
            push_constant_stage_flags,
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
    pub fn update_camera(
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

    /// Record and submit rendering commands. The offscreen color_image
    /// ends up in TRANSFER_SRC_OPTIMAL layout ready for blit.
    pub fn render_frame(&self, ctx: &VulkanContext) -> Result<vk::CommandBuffer, String> {
        let device = &ctx.device;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe {
            device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| format!("Failed to allocate command buffer: {:?}", e))?[0]
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| format!("Failed to begin command buffer: {:?}", e))?;
        }

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.05, 0.05, 0.08, 1.0] },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            },
        ];

        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D { width: self.width, height: self.height },
            })
            .clear_values(&clear_values);

        unsafe {
            device.cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout,
                0, &self.descriptor_sets, &[],
            );
            if !self.push_constant_data.is_empty() {
                device.cmd_push_constants(
                    cmd, self.pipeline_layout, self.push_constant_stage_flags,
                    0, &self.push_constant_data,
                );
            }
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.vbo.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, self.ibo.buffer, 0, vk::IndexType::UINT32);
            device.cmd_draw_indexed(cmd, self.num_indices, 1, 0, 0, 0);
            device.cmd_end_render_pass(cmd);
        }

        Ok(cmd)
    }

    /// Blit the offscreen color image to a swapchain image.
    /// Assumes cmd is already recording. Transitions swapchain image layout.
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
            device.destroy_shader_module(self.vert_module, None);
            device.destroy_shader_module(self.frag_module, None);
        }

        self.color_image.destroy(&device, ctx.allocator_mut());
        self.depth_image.destroy(&device, ctx.allocator_mut());
        self.default_texture.destroy(&device, ctx.allocator_mut());
        self.default_black_texture.destroy(&device, ctx.allocator_mut());
        self.default_normal_texture.destroy(&device, ctx.allocator_mut());
        for (_, mut tex) in self.texture_images.drain() {
            tex.destroy(&device, ctx.allocator_mut());
        }
        self.ibl_assets.destroy(&device, ctx.allocator_mut());
        self.mvp_buffer.destroy(&device, ctx.allocator_mut());
        self.light_buffer.destroy(&device, ctx.allocator_mut());
        self.vbo.destroy(&device, ctx.allocator_mut());
        self.ibo.destroy(&device, ctx.allocator_mut());
    }
}

// ===========================================================================
// Renderer trait implementation for PersistentRenderer
// ===========================================================================

impl scene_manager::Renderer for PersistentRenderer {
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
        PersistentRenderer::update_camera(self, eye, target, up, fov_y, aspect, near, far);
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
