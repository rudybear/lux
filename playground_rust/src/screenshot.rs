//! Screenshot capture: copy offscreen image to host-visible buffer and save as PNG.

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use std::path::Path;

/// A staging buffer for reading back pixel data from the GPU.
pub struct StagingBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: u64,
}

impl StagingBuffer {
    /// Create a host-visible staging buffer large enough for width * height * 4 bytes.
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let size = (width * height * 4) as u64;

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            device
                .create_buffer(&buffer_info, None)
                .map_err(|e| format!("Failed to create staging buffer: {:?}", e))?
        };

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "screenshot_staging",
                requirements,
                location: MemoryLocation::GpuToCpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| format!("Failed to allocate staging memory: {:?}", e))?;

        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| format!("Failed to bind staging buffer memory: {:?}", e))?;
        }

        Ok(Self {
            buffer,
            allocation: Some(allocation),
            size,
        })
    }

    /// Read back the pixel data from the mapped staging buffer.
    pub fn read_pixels(&self, width: u32, height: u32) -> Result<Vec<u8>, String> {
        let alloc = self
            .allocation
            .as_ref()
            .ok_or("Staging buffer has no allocation")?;

        let mapped = alloc
            .mapped_slice()
            .ok_or("Staging buffer is not mapped")?;

        let byte_count = (width * height * 4) as usize;
        if mapped.len() < byte_count {
            return Err(format!(
                "Mapped slice too small: {} < {}",
                mapped.len(),
                byte_count
            ));
        }

        Ok(mapped[..byte_count].to_vec())
    }

    /// Destroy the staging buffer, freeing GPU memory.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(alloc) = self.allocation.take() {
            let _ = allocator.free(alloc);
        }
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
    }
}

/// Record a command to transition an image layout.
pub fn cmd_transition_image(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        )
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

/// Transition an image with custom mip level count and layer count.
pub fn cmd_transition_image_layers(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
    level_count: u32,
    layer_count: u32,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(level_count)
                .base_array_layer(0)
                .layer_count(layer_count),
        )
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);

    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

/// Record a command to copy an image to a staging buffer.
pub fn cmd_copy_image_to_buffer(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    buffer: vk::Buffer,
    width: u32,
    height: u32,
) {
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
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    unsafe {
        device.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer,
            &[region],
        );
    }
}

/// Save RGBA pixel data as a PNG file.
pub fn save_png(pixels: &[u8], width: u32, height: u32, path: &Path) -> Result<(), String> {
    let img = image::RgbaImage::from_raw(width, height, pixels.to_vec())
        .ok_or("Failed to create image from pixel data")?;

    img.save(path)
        .map_err(|e| format!("Failed to save PNG to {:?}: {}", path, e))?;

    Ok(())
}
