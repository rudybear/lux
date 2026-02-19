#include "screenshot.h"
#include "stb_image_write.h"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cstring>

namespace Screenshot {

void saveImageToPNG(VulkanContext& ctx,
                    VkImage image, VkFormat format,
                    uint32_t width, uint32_t height,
                    VkImageLayout currentLayout,
                    const std::string& outputPath) {
    VkDeviceSize imageSize = width * height * 4;

    // Create staging buffer (host-visible, host-coherent)
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = imageSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkBuffer stagingBuffer;
    VmaAllocation stagingAllocation;

    VkResult result = vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocInfo,
                                      &stagingBuffer, &stagingAllocation, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create staging buffer for screenshot");
    }

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    // Transition image to TRANSFER_SRC if needed
    if (currentLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = currentLayout;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    // Copy image to staging buffer
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           stagingBuffer, 1, &region);

    ctx.endSingleTimeCommands(cmd);

    // Map and read pixels
    void* mapped;
    vmaMapMemory(ctx.allocator, stagingAllocation, &mapped);

    std::vector<uint8_t> pixels(imageSize);
    memcpy(pixels.data(), mapped, imageSize);

    vmaUnmapMemory(ctx.allocator, stagingAllocation);

    // Handle BGRA -> RGBA swizzle if format is BGRA
    if (format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB) {
        for (uint32_t i = 0; i < width * height; i++) {
            std::swap(pixels[i * 4 + 0], pixels[i * 4 + 2]);
        }
    }

    // Save as PNG
    int writeResult = stbi_write_png(outputPath.c_str(), width, height, 4,
                                     pixels.data(), width * 4);
    if (!writeResult) {
        throw std::runtime_error("Failed to write PNG: " + outputPath);
    }

    std::cout << "Saved " << width << "x" << height << " render to " << outputPath << std::endl;

    // Cleanup
    vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingAllocation);
}

} // namespace Screenshot
