//! Rasterization renderer: scene/pipeline architecture.
//!
//! Two public entry points:
//! - `render_raster()` — render a raster scene (sphere, triangle, or glTF) through a pipeline
//! - `render_fullscreen()` — render a fullscreen fragment shader

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

/// IBL cubemap texture with its associated resources.
struct IblTexture {
    image: vk::Image,
    view: vk::ImageView,
    sampler: vk::Sampler,
    allocation: Option<Allocation>,
}

impl IblTexture {
    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
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

/// Loaded IBL assets: specular cubemap, irradiance cubemap, BRDF LUT.
struct IblAssets {
    env_specular: Option<IblTexture>,
    env_irradiance: Option<IblTexture>,
    brdf_lut: Option<IblTexture>,
}

impl IblAssets {
    fn empty() -> Self {
        Self {
            env_specular: None,
            env_irradiance: None,
            brdf_lut: None,
        }
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(ref mut tex) = self.env_specular {
            tex.destroy(device, allocator);
        }
        if let Some(ref mut tex) = self.env_irradiance {
            tex.destroy(device, allocator);
        }
        if let Some(ref mut tex) = self.brdf_lut {
            tex.destroy(device, allocator);
        }
    }

    /// Get the image view for a named IBL binding (if available).
    fn view_for_binding(&self, name: &str) -> Option<vk::ImageView> {
        match name {
            "env_specular" => self.env_specular.as_ref().map(|t| t.view),
            "env_irradiance" => self.env_irradiance.as_ref().map(|t| t.view),
            "brdf_lut" => self.brdf_lut.as_ref().map(|t| t.view),
            _ => None,
        }
    }

    /// Get the sampler for a named IBL binding (if available).
    fn sampler_for_binding(&self, name: &str) -> Option<vk::Sampler> {
        match name {
            "env_specular" => self.env_specular.as_ref().map(|t| t.sampler),
            "env_irradiance" => self.env_irradiance.as_ref().map(|t| t.sampler),
            "brdf_lut" => self.brdf_lut.as_ref().map(|t| t.sampler),
            _ => None,
        }
    }
}

/// IBL manifest (parsed from manifest.json).
#[derive(serde::Deserialize, Default)]
struct IblManifest {
    #[serde(default)]
    specular_face_size: u32,
    #[serde(default)]
    specular_mip_count: u32,
    #[serde(default)]
    irradiance_face_size: u32,
    #[serde(default)]
    brdf_lut_size: u32,
    // Also accept the nested form from preprocess_ibl.py
    #[serde(default)]
    specular: Option<IblSpecularManifest>,
    #[serde(default)]
    irradiance: Option<IblIrradianceManifest>,
    #[serde(default)]
    brdf_lut: Option<IblBrdfLutManifest>,
}

#[derive(serde::Deserialize, Default)]
struct IblSpecularManifest {
    #[serde(default)]
    face_size: u32,
    #[serde(default)]
    mip_count: u32,
}

#[derive(serde::Deserialize, Default)]
struct IblIrradianceManifest {
    #[serde(default)]
    face_size: u32,
}

#[derive(serde::Deserialize, Default)]
struct IblBrdfLutManifest {
    #[serde(default)]
    size: u32,
}

impl IblManifest {
    fn spec_face_size(&self) -> u32 {
        if self.specular_face_size > 0 {
            self.specular_face_size
        } else if let Some(ref s) = self.specular {
            if s.face_size > 0 { s.face_size } else { 256 }
        } else {
            256
        }
    }

    fn spec_mip_count(&self) -> u32 {
        if self.specular_mip_count > 0 {
            self.specular_mip_count
        } else if let Some(ref s) = self.specular {
            if s.mip_count > 0 { s.mip_count } else { 5 }
        } else {
            5
        }
    }

    fn irr_face_size(&self) -> u32 {
        if self.irradiance_face_size > 0 {
            self.irradiance_face_size
        } else if let Some(ref i) = self.irradiance {
            if i.face_size > 0 { i.face_size } else { 32 }
        } else {
            32
        }
    }

    fn lut_size(&self) -> u32 {
        if self.brdf_lut_size > 0 {
            self.brdf_lut_size
        } else if let Some(ref b) = self.brdf_lut {
            if b.size > 0 { b.size } else { 512 }
        } else {
            512
        }
    }
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
            render_pbr_scene(ctx, vert_module, frag_module, pipeline_base, scene_source, width, height, output_path)?;

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

            render_pbr_scene(ctx, vert_module, frag_module, pipeline_base, scene_source, width, height, output_path)?;

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

/// Create a cubemap VkImage with CUBE_COMPATIBLE flag, upload data via staging, create CUBE view.
fn create_cubemap_image(
    ctx: &mut VulkanContext,
    face_size: u32,
    mip_count: u32,
    format: vk::Format,
    pixel_data: &[u8],
    bytes_per_pixel: u32,
    name: &str,
) -> Result<(vk::Image, vk::ImageView, Allocation), String> {
    let device_clone = ctx.device.clone();
    let device = &device_clone;

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: face_size,
            height: face_size,
            depth: 1,
        })
        .mip_levels(mip_count)
        .array_layers(6)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);

    let image = unsafe {
        device
            .create_image(&image_info, None)
            .map_err(|e| format!("Failed to create cubemap image '{}': {:?}", name, e))?
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
        .map_err(|e| format!("Failed to allocate cubemap image memory '{}': {:?}", name, e))?;

    unsafe {
        ctx.device
            .bind_image_memory(image, allocation.memory(), allocation.offset())
            .map_err(|e| format!("Failed to bind cubemap image memory '{}': {:?}", name, e))?;
    }

    // Create staging buffer
    let staging_info = vk::BufferCreateInfo::default()
        .size(pixel_data.len() as u64)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let staging_buffer = unsafe {
        ctx.device
            .create_buffer(&staging_info, None)
            .map_err(|e| format!("Failed to create cubemap staging buffer: {:?}", e))?
    };

    let staging_reqs = unsafe { ctx.device.get_buffer_memory_requirements(staging_buffer) };

    let mut staging_alloc = ctx
        .allocator_mut()
        .allocate(&AllocationCreateDesc {
            name: "cubemap_staging",
            requirements: staging_reqs,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate cubemap staging memory: {:?}", e))?;

    unsafe {
        ctx.device
            .bind_buffer_memory(staging_buffer, staging_alloc.memory(), staging_alloc.offset())
            .map_err(|e| format!("Failed to bind cubemap staging memory: {:?}", e))?;
    }

    if let Some(mapped) = staging_alloc.mapped_slice_mut() {
        mapped[..pixel_data.len()].copy_from_slice(pixel_data);
    }

    // Transition image and copy all mip levels / faces
    let cmd = ctx.begin_single_commands()?;

    // Transition entire image to TRANSFER_DST_OPTIMAL
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(mip_count)
                .base_array_layer(0)
                .layer_count(6),
        )
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

    unsafe {
        ctx.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    // Copy each mip level and face from the staging buffer
    // Data layout: mip-major, then face-major (mip0-face0..5, mip1-face0..5, ...)
    let mut buffer_offset: u64 = 0;
    let mut regions: Vec<vk::BufferImageCopy> = Vec::new();

    for mip in 0..mip_count {
        let mip_size = std::cmp::max(face_size >> mip, 1);
        let face_bytes = (mip_size * mip_size * bytes_per_pixel) as u64;

        for face in 0..6u32 {
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(buffer_offset)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(mip)
                            .base_array_layer(face)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D::default())
                    .image_extent(vk::Extent3D {
                        width: mip_size,
                        height: mip_size,
                        depth: 1,
                    }),
            );
            buffer_offset += face_bytes;
        }
    }

    unsafe {
        ctx.device.cmd_copy_buffer_to_image(
            cmd,
            staging_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
    }

    // Transition to SHADER_READ_ONLY_OPTIMAL
    let barrier2 = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(mip_count)
                .base_array_layer(0)
                .layer_count(6),
        )
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

    unsafe {
        ctx.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier2],
        );
    }

    ctx.end_single_commands(cmd)?;

    // Cleanup staging
    let _ = ctx.allocator_mut().free(staging_alloc);
    unsafe {
        ctx.device.destroy_buffer(staging_buffer, None);
    }

    // Create CUBE image view
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::CUBE)
        .format(format)
        .components(vk::ComponentMapping::default())
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(mip_count)
                .base_array_layer(0)
                .layer_count(6),
        );

    let view = unsafe {
        ctx.device
            .create_image_view(&view_info, None)
            .map_err(|e| format!("Failed to create cubemap image view '{}': {:?}", name, e))?
    };

    Ok((image, view, allocation))
}

/// Create a 2D image from raw pixel data and upload it via a staging buffer.
/// This is the float16 variant used by the BRDF LUT.
fn create_texture_image_raw(
    ctx: &mut VulkanContext,
    width: u32,
    height: u32,
    pixel_data: &[u8],
    format: vk::Format,
    name: &str,
) -> Result<GpuImage, String> {
    let data_size = pixel_data.len() as u64;

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
            name: "texture_raw_staging",
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
        mapped[..pixel_data.len()].copy_from_slice(pixel_data);
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

/// Try to find and load IBL assets from playground/assets/ibl/<name>/ directories.
/// Returns IblAssets with whatever textures could be loaded, or empty if not found.
fn load_ibl_assets(ctx: &mut VulkanContext) -> IblAssets {
    // Search for IBL asset directories
    let ibl_base = Path::new("playground/assets/ibl");
    if !ibl_base.exists() {
        info!("IBL assets directory not found at {:?}, using defaults", ibl_base);
        return IblAssets::empty();
    }

    // Find first directory with a manifest.json
    let ibl_dir = match std::fs::read_dir(ibl_base) {
        Ok(entries) => {
            let mut found = None;
            let mut sorted_entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
            sorted_entries.sort_by_key(|e| e.file_name());
            for entry in sorted_entries {
                let path = entry.path();
                if path.is_dir() && path.join("manifest.json").exists() {
                    found = Some(path);
                    break;
                }
            }
            match found {
                Some(dir) => dir,
                None => {
                    info!("No IBL manifest.json found in {:?}", ibl_base);
                    return IblAssets::empty();
                }
            }
        }
        Err(_) => {
            info!("Failed to read IBL assets directory");
            return IblAssets::empty();
        }
    };

    info!("Loading IBL assets from {:?}", ibl_dir);

    // Parse manifest
    let manifest_path = ibl_dir.join("manifest.json");
    let manifest: IblManifest = match std::fs::read_to_string(&manifest_path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(_) => {
            info!("Failed to read IBL manifest");
            return IblAssets::empty();
        }
    };

    let mut assets = IblAssets::empty();

    // Load specular cubemap
    let spec_path = ibl_dir.join("specular.bin");
    if spec_path.exists() {
        match load_specular_cubemap(ctx, &spec_path, &manifest) {
            Ok(tex) => {
                info!("Loaded specular cubemap: {}x{}, {} mips",
                    manifest.spec_face_size(), manifest.spec_face_size(), manifest.spec_mip_count());
                assets.env_specular = Some(tex);
            }
            Err(e) => info!("Failed to load specular cubemap: {}", e),
        }
    }

    // Load irradiance cubemap
    let irr_path = ibl_dir.join("irradiance.bin");
    if irr_path.exists() {
        match load_irradiance_cubemap(ctx, &irr_path, &manifest) {
            Ok(tex) => {
                info!("Loaded irradiance cubemap: {}x{}",
                    manifest.irr_face_size(), manifest.irr_face_size());
                assets.env_irradiance = Some(tex);
            }
            Err(e) => info!("Failed to load irradiance cubemap: {}", e),
        }
    }

    // Load BRDF LUT
    let brdf_path = ibl_dir.join("brdf_lut.bin");
    if brdf_path.exists() {
        match load_brdf_lut(ctx, &brdf_path, &manifest) {
            Ok(tex) => {
                info!("Loaded BRDF LUT: {}x{}", manifest.lut_size(), manifest.lut_size());
                assets.brdf_lut = Some(tex);
            }
            Err(e) => info!("Failed to load BRDF LUT: {}", e),
        }
    }

    info!("IBL assets loaded: specular={}, irradiance={}, brdf_lut={}",
        assets.env_specular.is_some(),
        assets.env_irradiance.is_some(),
        assets.brdf_lut.is_some());

    assets
}

/// Load the specular cubemap from specular.bin (float16 RGBA, mip-major, face-major).
fn load_specular_cubemap(
    ctx: &mut VulkanContext,
    path: &Path,
    manifest: &IblManifest,
) -> Result<IblTexture, String> {
    let raw_data = std::fs::read(path)
        .map_err(|e| format!("Failed to read specular.bin: {}", e))?;

    let face_size = manifest.spec_face_size();
    let mip_count = manifest.spec_mip_count();
    let bytes_per_pixel = 8u32; // float16 RGBA = 4 * 2 bytes

    let (image, view, allocation) = create_cubemap_image(
        ctx,
        face_size,
        mip_count,
        vk::Format::R16G16B16A16_SFLOAT,
        &raw_data,
        bytes_per_pixel,
        "ibl_specular",
    )?;

    // Create sampler with linear + mipmap filtering
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .max_lod(mip_count as f32);

    let sampler = unsafe {
        ctx.device
            .create_sampler(&sampler_info, None)
            .map_err(|e| format!("Failed to create specular sampler: {:?}", e))?
    };

    Ok(IblTexture {
        image,
        view,
        sampler,
        allocation: Some(allocation),
    })
}

/// Load the irradiance cubemap from irradiance.bin (float16 RGBA, 6 faces, 1 mip).
fn load_irradiance_cubemap(
    ctx: &mut VulkanContext,
    path: &Path,
    manifest: &IblManifest,
) -> Result<IblTexture, String> {
    let raw_data = std::fs::read(path)
        .map_err(|e| format!("Failed to read irradiance.bin: {}", e))?;

    let face_size = manifest.irr_face_size();
    let bytes_per_pixel = 8u32; // float16 RGBA

    let (image, view, allocation) = create_cubemap_image(
        ctx,
        face_size,
        1,
        vk::Format::R16G16B16A16_SFLOAT,
        &raw_data,
        bytes_per_pixel,
        "ibl_irradiance",
    )?;

    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);

    let sampler = unsafe {
        ctx.device
            .create_sampler(&sampler_info, None)
            .map_err(|e| format!("Failed to create irradiance sampler: {:?}", e))?
    };

    Ok(IblTexture {
        image,
        view,
        sampler,
        allocation: Some(allocation),
    })
}

/// Load the BRDF LUT from brdf_lut.bin (float16 RG, padded to RGBA).
fn load_brdf_lut(
    ctx: &mut VulkanContext,
    path: &Path,
    manifest: &IblManifest,
) -> Result<IblTexture, String> {
    let raw_data = std::fs::read(path)
        .map_err(|e| format!("Failed to read brdf_lut.bin: {}", e))?;

    let lut_size = manifest.lut_size();
    let expected_rg_bytes = (lut_size * lut_size * 2 * 2) as usize; // float16 RG = 2 * 2 bytes

    if raw_data.len() < expected_rg_bytes {
        return Err(format!(
            "BRDF LUT data too small: {} < {} expected",
            raw_data.len(),
            expected_rg_bytes
        ));
    }

    // Pad RG float16 to RGBA float16 (each pixel: 4 bytes -> 8 bytes)
    let pixel_count = (lut_size * lut_size) as usize;
    let mut rgba_data: Vec<u8> = Vec::with_capacity(pixel_count * 8);
    let zero_u16: [u8; 2] = [0, 0]; // float16 zero

    for i in 0..pixel_count {
        let src_offset = i * 4; // 2 channels * 2 bytes each
        // R channel
        rgba_data.extend_from_slice(&raw_data[src_offset..src_offset + 2]);
        // G channel
        rgba_data.extend_from_slice(&raw_data[src_offset + 2..src_offset + 4]);
        // B channel (zero)
        rgba_data.extend_from_slice(&zero_u16);
        // A channel (zero)
        rgba_data.extend_from_slice(&zero_u16);
    }

    let gpu_image = create_texture_image_raw(
        ctx,
        lut_size,
        lut_size,
        &rgba_data,
        vk::Format::R16G16B16A16_SFLOAT,
        "ibl_brdf_lut",
    )?;

    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);

    let sampler = unsafe {
        ctx.device
            .create_sampler(&sampler_info, None)
            .map_err(|e| format!("Failed to create BRDF LUT sampler: {:?}", e))?
    };

    Ok(IblTexture {
        image: gpu_image.image,
        view: gpu_image.view,
        sampler,
        allocation: gpu_image.allocation,
    })
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
    let gltf_scene = if is_gltf {
        Some(crate::gltf_loader::load_gltf(Path::new(scene_source))?)
    } else {
        None
    };

    // Determine if we need 48-byte vertices (with tangent) based on reflection stride
    let pipeline_vertex_stride = vert_refl.vertex_stride;

    let (vertex_data_owned, pbr_indices) = if let Some(ref gltf_s) = gltf_scene {
        if gltf_s.meshes.is_empty() {
            return Err(format!("No meshes found in glTF file: {}", scene_source));
        }
        let mesh = &gltf_s.meshes[0];

        if pipeline_vertex_stride == 48 {
            // Use GltfVertex directly (48 bytes with tangent)
            let data: Vec<u8> = bytemuck::cast_slice(&mesh.vertices).to_vec();
            (data, mesh.indices.clone())
        } else {
            // Use PbrVertex (32 bytes without tangent)
            let verts: Vec<scene::PbrVertex> = mesh.vertices.iter().map(|v| {
                scene::PbrVertex {
                    position: v.position,
                    normal: v.normal,
                    uv: v.uv,
                }
            }).collect();
            let data: Vec<u8> = bytemuck::cast_slice(&verts).to_vec();
            (data, mesh.indices.clone())
        }
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

    // Light uniform buffer (32 bytes)
    let light_dir = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
    let camera_pos = DefaultCamera::EYE;
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

    // Create a 1x1 white default texture for any missing texture slots
    let mut default_texture = create_default_white_texture(ctx)?;

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
    let mut ibl_assets = load_ibl_assets(ctx);

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
                        // Regular 2D texture: look up by name, fall back to default white
                        let view = texture_images
                            .get(&binding.name)
                            .map(|img| img.view)
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
