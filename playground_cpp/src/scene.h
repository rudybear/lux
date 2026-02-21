#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

// PBR sphere vertex: position + normal + UV = 32 bytes
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};
static_assert(sizeof(Vertex) == 32, "Vertex must be 32 bytes");

// Simple colored triangle vertex: position + color = 24 bytes
struct TriangleVertex {
    glm::vec3 position;
    glm::vec3 color;
};
static_assert(sizeof(TriangleVertex) == 24, "TriangleVertex must be 24 bytes");

// GPU mesh data
struct GPUMesh {
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VmaAllocation vertexAllocation = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VmaAllocation indexAllocation = VK_NULL_HANDLE;
    uint32_t indexCount = 0;
    uint32_t vertexCount = 0;
};

// GPU texture data
struct GPUTexture {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
};

namespace Scene {

// Generate a UV sphere mesh
void generateSphere(uint32_t stacks, uint32_t slices,
                    std::vector<Vertex>& outVertices,
                    std::vector<uint32_t>& outIndices);

// Generate the hardcoded colored triangle (3 vertices in clip space)
void generateTriangle(std::vector<TriangleVertex>& outVertices);

// Generate procedural checker texture (RGBA8, size x size)
std::vector<uint8_t> generateProceduralTexture(uint32_t size);

// Upload sphere mesh to GPU buffers
GPUMesh uploadMesh(VmaAllocator allocator, VkDevice device,
                   VkCommandPool cmdPool, VkQueue queue,
                   const std::vector<Vertex>& vertices,
                   const std::vector<uint32_t>& indices);

// Upload triangle mesh to GPU buffers
GPUMesh uploadTriangleMesh(VmaAllocator allocator, VkDevice device,
                           VkCommandPool cmdPool, VkQueue queue,
                           const std::vector<TriangleVertex>& vertices);

// Upload texture to GPU image
GPUTexture uploadTexture(VmaAllocator allocator, VkDevice device,
                         VkCommandPool cmdPool, VkQueue queue,
                         const std::vector<uint8_t>& pixels,
                         uint32_t width, uint32_t height);

// Upload raw data to a GPU buffer with specified usage flags
void uploadBuffer(VmaAllocator allocator, VkDevice device,
                  VkCommandPool cmdPool, VkQueue queue,
                  const void* data, VkDeviceSize size,
                  VkBufferUsageFlags usage,
                  VkBuffer& outBuffer, VmaAllocation& outAllocation);

// Cleanup GPU resources
void destroyMesh(VmaAllocator allocator, GPUMesh& mesh);
void destroyTexture(VmaAllocator allocator, VkDevice device, GPUTexture& tex);

} // namespace Scene
