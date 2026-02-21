//! Shared scene management: types, IBL loading, texture upload, auto-camera.
//!
//! Extracts duplicated code from `rt_renderer.rs` and `raster_renderer.rs`
//! into a single module. Both renderers use `SceneManager` for scene data
//! and implement the `Renderer` trait for unified rendering.

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use log::info;
use std::path::Path;

use crate::camera::DefaultCamera;
use crate::screenshot;
use crate::vulkan_context::VulkanContext;

// ===========================================================================
// Shared GPU resource types
// ===========================================================================

/// GPU buffer with its allocation. Defined once and used by both renderers.
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
}

impl GpuBuffer {
    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(alloc) = self.allocation.take() {
            let _ = allocator.free(alloc);
        }
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
    }
}

/// GPU image with its allocation.
pub struct GpuImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: Option<Allocation>,
}

impl GpuImage {
    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
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

/// IBL cubemap texture with its associated resources.
pub struct IblTexture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub allocation: Option<Allocation>,
}

impl IblTexture {
    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
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
pub struct IblAssets {
    pub env_specular: Option<IblTexture>,
    pub env_irradiance: Option<IblTexture>,
    pub brdf_lut: Option<IblTexture>,
}

impl IblAssets {
    pub fn empty() -> Self {
        Self {
            env_specular: None,
            env_irradiance: None,
            brdf_lut: None,
        }
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
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
    pub fn view_for_binding(&self, name: &str) -> Option<vk::ImageView> {
        match name {
            "env_specular" => self.env_specular.as_ref().map(|t| t.view),
            "env_irradiance" => self.env_irradiance.as_ref().map(|t| t.view),
            "brdf_lut" => self.brdf_lut.as_ref().map(|t| t.view),
            _ => None,
        }
    }

    /// Get the sampler for a named IBL binding (if available).
    pub fn sampler_for_binding(&self, name: &str) -> Option<vk::Sampler> {
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
pub struct IblManifest {
    #[serde(default)]
    pub specular_face_size: u32,
    #[serde(default)]
    pub specular_mip_count: u32,
    #[serde(default)]
    pub irradiance_face_size: u32,
    #[serde(default)]
    pub brdf_lut_size: u32,
    #[serde(default)]
    pub specular: Option<IblSpecularManifest>,
    #[serde(default)]
    pub irradiance: Option<IblIrradianceManifest>,
    #[serde(default)]
    pub brdf_lut: Option<IblBrdfLutManifest>,
}

#[derive(serde::Deserialize, Default)]
pub struct IblSpecularManifest {
    #[serde(default)]
    pub face_size: u32,
    #[serde(default)]
    pub mip_count: u32,
}

#[derive(serde::Deserialize, Default)]
pub struct IblIrradianceManifest {
    #[serde(default)]
    pub face_size: u32,
}

#[derive(serde::Deserialize, Default)]
pub struct IblBrdfLutManifest {
    #[serde(default)]
    pub size: u32,
}

impl IblManifest {
    pub fn spec_face_size(&self) -> u32 {
        if self.specular_face_size > 0 {
            self.specular_face_size
        } else if let Some(ref s) = self.specular {
            if s.face_size > 0 { s.face_size } else { 256 }
        } else {
            256
        }
    }

    pub fn spec_mip_count(&self) -> u32 {
        if self.specular_mip_count > 0 {
            self.specular_mip_count
        } else if let Some(ref s) = self.specular {
            if s.mip_count > 0 { s.mip_count } else { 5 }
        } else {
            5
        }
    }

    pub fn irr_face_size(&self) -> u32 {
        if self.irradiance_face_size > 0 {
            self.irradiance_face_size
        } else if let Some(ref i) = self.irradiance {
            if i.face_size > 0 { i.face_size } else { 32 }
        } else {
            32
        }
    }

    pub fn lut_size(&self) -> u32 {
        if self.brdf_lut_size > 0 {
            self.brdf_lut_size
        } else if let Some(ref b) = self.brdf_lut {
            if b.size > 0 { b.size } else { 512 }
        } else {
            512
        }
    }
}

// ===========================================================================
// Renderer trait
// ===========================================================================

/// Unified renderer trait implemented by both RTRenderer and PersistentRenderer.
///
/// Allows main.rs to handle both render paths through dynamic dispatch.
pub trait Renderer {
    /// Render a frame. For headless, this renders once. For interactive, call per frame.
    fn render(&mut self, ctx: &VulkanContext) -> Result<vk::CommandBuffer, String>;

    /// Update the camera from orbit parameters.
    fn update_camera(
        &mut self,
        eye: glam::Vec3,
        target: glam::Vec3,
        up: glam::Vec3,
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    );

    /// Blit the rendered output to a swapchain image. The command buffer should be recording.
    fn blit_to_swapchain(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        swap_image: vk::Image,
        extent: vk::Extent2D,
    );

    /// The output image (for screenshot readback).
    fn output_image(&self) -> vk::Image;

    /// The output image format.
    fn output_format(&self) -> vk::Format;

    /// Render width.
    fn width(&self) -> u32;

    /// Render height.
    fn height(&self) -> u32;

    /// Whether this renderer has auto-camera data from scene bounds.
    fn has_scene_bounds(&self) -> bool;

    /// Auto-camera eye position.
    fn auto_eye(&self) -> glam::Vec3;

    /// Auto-camera target position.
    fn auto_target(&self) -> glam::Vec3;

    /// Auto-camera up vector.
    fn auto_up(&self) -> glam::Vec3;

    /// Auto-camera far plane.
    fn auto_far(&self) -> f32;

    /// The wait stage flags for swapchain submission.
    fn wait_stage(&self) -> vk::PipelineStageFlags;

    /// Destroy all GPU resources.
    fn destroy(&mut self, ctx: &mut VulkanContext);
}

// ===========================================================================
// Shared utility functions
// ===========================================================================

/// Check whether a scene source string refers to a glTF file.
pub fn is_gltf_file(scene_source: &str) -> bool {
    scene_source.ends_with(".glb") || scene_source.ends_with(".gltf")
}

/// Compute auto-camera from scene bounds using node transforms (for glTF scenes).
///
/// This code is shared between RT and raster renderers. Both use the same
/// node-transform-based camera placement algorithm.
pub fn compute_auto_camera_from_draw_items(
    vertices_positions: &[[f32; 3]],
    draw_items: &[crate::gltf_loader::DrawItem],
) -> (glam::Vec3, glam::Vec3, glam::Vec3, f32) {
    let mut bbox_min = glam::Vec3::splat(f32::MAX);
    let mut bbox_max = glam::Vec3::splat(f32::MIN);
    for pos in vertices_positions {
        bbox_min = bbox_min.min(glam::Vec3::from(*pos));
        bbox_max = bbox_max.max(glam::Vec3::from(*pos));
    }
    let center = (bbox_min + bbox_max) * 0.5;
    let extent = bbox_max - bbox_min;

    // Extract camera direction from first draw item's world transform
    let mut cam_dir = glam::Vec3::new(0.0, 0.0, 1.0);
    let mut cam_up_vec = glam::Vec3::new(0.0, 1.0, 0.0);

    if !draw_items.is_empty() {
        let upper = glam::Mat3::from_mat4(draw_items[0].world_transform);
        let row_norms = glam::Vec3::new(
            upper.row(0).length(),
            upper.row(1).length(),
            upper.row(2).length(),
        );
        let r_approx = glam::Mat3::from_cols(
            upper.col(0) / row_norms.x.max(1e-8),
            upper.col(1) / row_norms.y.max(1e-8),
            upper.col(2) / row_norms.z.max(1e-8),
        );
        // Blender front (-Y local) and up (+Z local) in world space
        let front = r_approx.transpose() * glam::Vec3::new(0.0, -1.0, 0.0);
        let up_w = r_approx.transpose() * glam::Vec3::new(0.0, 0.0, 1.0);

        let fl = front.length();
        let ul = up_w.length();
        if fl > 0.5 && ul > 0.5 {
            let candidate_dir = front / fl;
            let candidate_up = up_w / ul;
            if candidate_dir.y.abs() < 0.9 {
                cam_dir = candidate_dir;
                cam_up_vec = candidate_up;
                if cam_dir.dot(cam_up_vec).abs() > 0.9 {
                    cam_up_vec = glam::Vec3::new(0.0, 1.0, 0.0);
                }
            }
        }
    }

    // Perpendicular extent via projection
    let view_right = (-cam_dir).cross(cam_up_vec);
    let vr_len = view_right.length();
    let (view_right, cam_up_vec) = if vr_len < 1e-6 {
        let cu = glam::Vec3::new(0.0, 0.0, 1.0);
        let vr = (-cam_dir).cross(cu);
        (vr / vr.length().max(1e-8), cu)
    } else {
        (view_right / vr_len, cam_up_vec)
    };
    let view_up = view_right.cross(-cam_dir).normalize();

    let proj_right = view_right.abs().dot(extent.abs());
    let proj_up = view_up.abs().dot(extent.abs());
    let max_perp_extent = proj_right.max(proj_up);

    let fov_y = DefaultCamera::FOV_Y_DEG.to_radians();
    let distance = (max_perp_extent / 2.0) / (fov_y / 2.0).tan() * 1.1;

    let elev_rad = 5.0f32.to_radians();
    let eye = center + distance * cam_dir + distance * elev_rad.sin() * view_up;

    let target = center;
    let up = cam_up_vec;
    let far = (distance * 3.0f32).max(100.0);

    info!(
        "Auto-camera: eye=({:.2},{:.2},{:.2}) target=({:.2},{:.2},{:.2}) distance={:.2}",
        eye.x, eye.y, eye.z, target.x, target.y, target.z, distance
    );

    (eye, target, up, far)
}

// ===========================================================================
// Buffer creation helpers
// ===========================================================================

/// Create a GPU buffer, upload data, and return the buffer + allocation.
pub fn create_buffer_with_data(
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
pub fn create_offscreen_image(
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

// ===========================================================================
// Texture upload helpers (parameterized by dst_stage)
// ===========================================================================

/// Create a texture image from pixel data and upload it via a staging buffer.
///
/// `dst_stage` controls the final pipeline barrier:
/// - For raster: `VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT`
/// - For RT: `VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR`
pub fn create_texture_image(
    ctx: &mut VulkanContext,
    width: u32,
    height: u32,
    pixels: &[u8],
    name: &str,
    dst_stage: vk::PipelineStageFlags,
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
        dst_stage,
    );

    ctx.end_single_commands(cmd)?;

    // Cleanup staging
    let _ = ctx.allocator_mut().free(staging_alloc);
    unsafe {
        ctx.device.destroy_buffer(staging_buffer, None);
    }

    Ok(image)
}

/// Create a 2D image from raw pixel data (arbitrary format) and upload via staging.
///
/// Used for BRDF LUT (float16 RGBA).
pub fn create_texture_image_raw(
    ctx: &mut VulkanContext,
    width: u32,
    height: u32,
    pixel_data: &[u8],
    format: vk::Format,
    name: &str,
    dst_stage: vk::PipelineStageFlags,
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
        dst_stage,
    );

    ctx.end_single_commands(cmd)?;

    // Cleanup staging
    let _ = ctx.allocator_mut().free(staging_alloc);
    unsafe {
        ctx.device.destroy_buffer(staging_buffer, None);
    }

    Ok(image)
}

// ===========================================================================
// Cubemap creation (parameterized by dst_stage)
// ===========================================================================

/// Create a cubemap VkImage with CUBE_COMPATIBLE flag, upload data via staging, create CUBE view.
///
/// `dst_stage` controls the final pipeline barrier.
pub fn create_cubemap_image(
    ctx: &mut VulkanContext,
    face_size: u32,
    mip_count: u32,
    format: vk::Format,
    pixel_data: &[u8],
    bytes_per_pixel: u32,
    name: &str,
    dst_stage: vk::PipelineStageFlags,
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

    screenshot::cmd_transition_image_layers(
        &ctx.device, cmd, image,
        vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER,
        mip_count, 6,
    );

    // Copy each mip level and face from the staging buffer
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
    screenshot::cmd_transition_image_layers(
        &ctx.device, cmd, image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::TRANSFER, dst_stage,
        mip_count, 6,
    );

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

// ===========================================================================
// IBL loading (parameterized by dst_stage)
// ===========================================================================

/// Try to find and load IBL assets from playground/assets/ibl/<name>/ directories.
///
/// `dst_stage` controls the final pipeline barrier for all uploaded textures.
pub fn load_ibl_assets(ctx: &mut VulkanContext, dst_stage: vk::PipelineStageFlags) -> IblAssets {
    // Search for IBL asset directories
    let ibl_base = Path::new("playground/assets/ibl");
    if !ibl_base.exists() {
        info!("IBL assets directory not found at {:?}, using defaults", ibl_base);
        return IblAssets::empty();
    }

    // Find IBL directory: prefer "pisa" then "neutral", matching C++ engine
    let ibl_dir = {
        let preferred = ["pisa", "neutral"];
        let mut found = None;
        for name in &preferred {
            let path = ibl_base.join(name);
            if path.is_dir() && path.join("manifest.json").exists() {
                found = Some(path);
                break;
            }
        }
        // Fall back to alphabetical
        if found.is_none() {
            if let Ok(entries) = std::fs::read_dir(ibl_base) {
                let mut sorted_entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
                sorted_entries.sort_by_key(|e| e.file_name());
                for entry in sorted_entries {
                    let path = entry.path();
                    if path.is_dir() && path.join("manifest.json").exists() {
                        found = Some(path);
                        break;
                    }
                }
            }
        }
        match found {
            Some(dir) => dir,
            None => {
                info!("No IBL manifest.json found in {:?}", ibl_base);
                return IblAssets::empty();
            }
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
        match load_specular_cubemap(ctx, &spec_path, &manifest, dst_stage) {
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
        match load_irradiance_cubemap(ctx, &irr_path, &manifest, dst_stage) {
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
        match load_brdf_lut(ctx, &brdf_path, &manifest, dst_stage) {
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
    dst_stage: vk::PipelineStageFlags,
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
        dst_stage,
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
    dst_stage: vk::PipelineStageFlags,
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
        dst_stage,
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
    dst_stage: vk::PipelineStageFlags,
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
        dst_stage,
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
