#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "vulkan_context.h"
#include <string>

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

} // namespace Screenshot
