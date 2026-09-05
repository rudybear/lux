#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "vulkan_context.h"
#include <cstdint>
#include <string>
#include <vector>

namespace Screenshot {

// Capture the contents of a VkImage to a PNG file.
// The image must be in VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL or
// VK_IMAGE_LAYOUT_GENERAL before calling this function.
// currentLayout: the current layout of the image
void saveImageToPNG(VulkanContext& ctx,
                    VkImage image, VkFormat format,
                    uint32_t width, uint32_t height,
                    VkImageLayout currentLayout,
                    const std::string& outputPath);

// Reads back the raw bytes of a VkImage (no format conversion, no PNG
// encoding) via a staging buffer -- used for the DLSS auxiliary
// attachments (RG32F motion, R32F expected depth) and for extracting the
// alpha channel of the color attachment (docs/lux-4d-spec.md section 3).
// `bytesPerPixel` must match `format` exactly (e.g. 16 for RGBA32F won't
// apply here; 8 for RG32F, 4 for R32F/RGBA8).
std::vector<uint8_t> readImageRaw(VulkanContext& ctx,
                                   VkImage image, uint32_t width, uint32_t height,
                                   uint32_t bytesPerPixel,
                                   VkImageLayout currentLayout);

} // namespace Screenshot
