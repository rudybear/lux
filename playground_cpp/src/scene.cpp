#include "scene.h"
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Scene {

void generateSphere(uint32_t stacks, uint32_t slices,
                    std::vector<Vertex>& outVertices,
                    std::vector<uint32_t>& outIndices) {
    outVertices.clear();
    outIndices.clear();

    // Generate vertices matching Python: generate_sphere()
    for (uint32_t i = 0; i <= stacks; i++) {
        float phi = static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(stacks);
        float vCoord = static_cast<float>(i) / static_cast<float>(stacks);

        for (uint32_t j = 0; j <= slices; j++) {
            float theta = 2.0f * static_cast<float>(M_PI) * static_cast<float>(j) / static_cast<float>(slices);
            float uCoord = static_cast<float>(j) / static_cast<float>(slices);

            float x = std::sin(phi) * std::cos(theta);
            float y = std::cos(phi);
            float z = std::sin(phi) * std::sin(theta);

            Vertex v;
            v.position = glm::vec3(x, y, z); // radius = 1.0
            v.normal = glm::vec3(x, y, z);   // normalized already for unit sphere
            v.uv = glm::vec2(uCoord, vCoord);
            outVertices.push_back(v);
        }
    }

    // Generate indices matching Python
    for (uint32_t i = 0; i < stacks; i++) {
        for (uint32_t j = 0; j < slices; j++) {
            uint32_t a = i * (slices + 1) + j;
            uint32_t b = a + slices + 1;

            outIndices.push_back(a);
            outIndices.push_back(b);
            outIndices.push_back(a + 1);

            outIndices.push_back(b);
            outIndices.push_back(b + 1);
            outIndices.push_back(a + 1);
        }
    }
}

void generateTriangle(std::vector<TriangleVertex>& outVertices) {
    outVertices.clear();

    // Match Python: TRIANGLE_VERTICES
    //   ( 0.0,  0.5, 0.0,   1.0, 0.0, 0.0),  # top, red
    //   (-0.5, -0.5, 0.0,   0.0, 1.0, 0.0),  # bottom-left, green
    //   ( 0.5, -0.5, 0.0,   0.0, 0.0, 1.0),  # bottom-right, blue
    outVertices.push_back({{0.0f, 0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}});
    outVertices.push_back({{-0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}});
    outVertices.push_back({{0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}});
}

std::vector<uint8_t> generateProceduralTexture(uint32_t size) {
    // Match Python: generate_procedural_texture()
    std::vector<uint8_t> pixels(size * size * 4);

    for (uint32_t y = 0; y < size; y++) {
        for (uint32_t x = 0; x < size; x++) {
            float u = static_cast<float>(x) / static_cast<float>(size);
            float v = static_cast<float>(y) / static_cast<float>(size);

            // Checker pattern with two color palettes
            int cx = static_cast<int>(u * 8.0f) % 2;
            int cy = static_cast<int>(v * 8.0f) % 2;
            int checker = cx ^ cy;

            int r, g, b;
            if (checker) {
                // Warm: terracotta / orange tones
                r = static_cast<int>(200.0f + 40.0f * std::sin(u * 12.0f));
                g = static_cast<int>(120.0f + 30.0f * std::sin(v * 10.0f));
                b = static_cast<int>(80.0f + 20.0f * std::cos(u * 8.0f + v * 6.0f));
            } else {
                // Cool: teal / blue-green tones
                r = static_cast<int>(60.0f + 30.0f * std::sin(u * 10.0f + v * 4.0f));
                g = static_cast<int>(150.0f + 40.0f * std::cos(v * 8.0f));
                b = static_cast<int>(170.0f + 50.0f * std::sin(u * 6.0f));
            }

            uint32_t idx = (y * size + x) * 4;
            pixels[idx + 0] = static_cast<uint8_t>(std::clamp(r, 0, 255));
            pixels[idx + 1] = static_cast<uint8_t>(std::clamp(g, 0, 255));
            pixels[idx + 2] = static_cast<uint8_t>(std::clamp(b, 0, 255));
            pixels[idx + 3] = 255;
        }
    }

    return pixels;
}

// Helper: create buffer with data using staging
static void createBufferWithData(VmaAllocator allocator, VkDevice device,
                                 VkCommandPool cmdPool, VkQueue queue,
                                 const void* data, VkDeviceSize size,
                                 VkBufferUsageFlags usage,
                                 VkBuffer& outBuffer, VmaAllocation& outAllocation) {
    // Create staging buffer
    VkBufferCreateInfo stagingInfo = {};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = size;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocInfo = {};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    VkBuffer stagingBuffer;
    VmaAllocation stagingAllocation;
    vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo,
                    &stagingBuffer, &stagingAllocation, nullptr);

    // Copy data to staging
    void* mapped;
    vmaMapMemory(allocator, stagingAllocation, &mapped);
    memcpy(mapped, data, size);
    vmaUnmapMemory(allocator, stagingAllocation);

    // Create device-local buffer
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateBuffer(allocator, &bufferInfo, &allocInfo,
                    &outBuffer, &outAllocation, nullptr);

    // Copy staging -> device
    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = cmdPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copyRegion = {};
    copyRegion.size = size;
    vkCmdCopyBuffer(cmd, stagingBuffer, outBuffer, 1, &copyRegion);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
    vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
}

GPUMesh uploadMesh(VmaAllocator allocator, VkDevice device,
                   VkCommandPool cmdPool, VkQueue queue,
                   const std::vector<Vertex>& vertices,
                   const std::vector<uint32_t>& indices) {
    GPUMesh mesh;
    mesh.vertexCount = static_cast<uint32_t>(vertices.size());
    mesh.indexCount = static_cast<uint32_t>(indices.size());

    VkDeviceSize vertexSize = vertices.size() * sizeof(Vertex);
    VkDeviceSize indexSize = indices.size() * sizeof(uint32_t);

    createBufferWithData(allocator, device, cmdPool, queue,
                         vertices.data(), vertexSize,
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                         mesh.vertexBuffer, mesh.vertexAllocation);

    createBufferWithData(allocator, device, cmdPool, queue,
                         indices.data(), indexSize,
                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                         mesh.indexBuffer, mesh.indexAllocation);

    return mesh;
}

GPUMesh uploadTriangleMesh(VmaAllocator allocator, VkDevice device,
                           VkCommandPool cmdPool, VkQueue queue,
                           const std::vector<TriangleVertex>& vertices) {
    GPUMesh mesh;
    mesh.vertexCount = static_cast<uint32_t>(vertices.size());
    mesh.indexCount = 0;

    VkDeviceSize vertexSize = vertices.size() * sizeof(TriangleVertex);

    createBufferWithData(allocator, device, cmdPool, queue,
                         vertices.data(), vertexSize,
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                         mesh.vertexBuffer, mesh.vertexAllocation);

    return mesh;
}

GPUTexture uploadTexture(VmaAllocator allocator, VkDevice device,
                         VkCommandPool cmdPool, VkQueue queue,
                         const std::vector<uint8_t>& pixels,
                         uint32_t width, uint32_t height) {
    GPUTexture tex;
    tex.width = width;
    tex.height = height;

    VkDeviceSize imageSize = width * height * 4;

    // Create staging buffer
    VkBufferCreateInfo stagingInfo = {};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = imageSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocInfo = {};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    VkBuffer stagingBuffer;
    VmaAllocation stagingAllocation;
    vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo,
                    &stagingBuffer, &stagingAllocation, nullptr);

    void* mapped;
    vmaMapMemory(allocator, stagingAllocation, &mapped);
    memcpy(mapped, pixels.data(), imageSize);
    vmaUnmapMemory(allocator, stagingAllocation);

    // Create image
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo imageAllocInfo = {};
    imageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(allocator, &imageInfo, &imageAllocInfo,
                   &tex.image, &tex.allocation, nullptr);

    // Transition and copy
    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = cmdPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Transition UNDEFINED -> TRANSFER_DST
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = tex.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Copy buffer -> image
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

    vkCmdCopyBufferToImage(cmd, stagingBuffer, tex.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition TRANSFER_DST -> SHADER_READ_ONLY
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
    vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);

    // Create image view
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = tex.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    vkCreateImageView(device, &viewInfo, nullptr, &tex.imageView);

    // Create sampler (linear filtering)
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    vkCreateSampler(device, &samplerInfo, nullptr, &tex.sampler);

    return tex;
}

void uploadBuffer(VmaAllocator allocator, VkDevice device,
                  VkCommandPool cmdPool, VkQueue queue,
                  const void* data, VkDeviceSize size,
                  VkBufferUsageFlags usage,
                  VkBuffer& outBuffer, VmaAllocation& outAllocation) {
    createBufferWithData(allocator, device, cmdPool, queue,
                         data, size, usage, outBuffer, outAllocation);
}

void destroyMesh(VmaAllocator allocator, GPUMesh& mesh) {
    if (mesh.vertexBuffer) vmaDestroyBuffer(allocator, mesh.vertexBuffer, mesh.vertexAllocation);
    if (mesh.indexBuffer) vmaDestroyBuffer(allocator, mesh.indexBuffer, mesh.indexAllocation);
    mesh = {};
}

void destroyTexture(VmaAllocator allocator, VkDevice device, GPUTexture& tex) {
    if (tex.sampler) vkDestroySampler(device, tex.sampler, nullptr);
    if (tex.imageView) vkDestroyImageView(device, tex.imageView, nullptr);
    if (tex.image) vmaDestroyImage(allocator, tex.image, tex.allocation);
    tex = {};
}

} // namespace Scene
