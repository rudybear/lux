#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "vulkan_context.h"
#include "scene.h"
#include <string>
#include <vector>

class RTRenderer {
public:
    void init(VulkanContext& ctx,
              const std::string& rgenSpvPath,
              const std::string& rmissSpvPath,
              const std::string& rchitSpvPath,
              uint32_t width, uint32_t height);

    void render(VulkanContext& ctx);

    // Get the storage image for screenshot
    VkImage getOutputImage() const { return storageImage; }
    VkFormat getOutputFormat() const { return VK_FORMAT_R8G8B8A8_UNORM; }
    uint32_t getWidth() const { return renderWidth; }
    uint32_t getHeight() const { return renderHeight; }

    void cleanup(VulkanContext& ctx);

private:
    uint32_t renderWidth = 512;
    uint32_t renderHeight = 512;

    // Storage image (RT output)
    VkImage storageImage = VK_NULL_HANDLE;
    VmaAllocation storageAllocation = VK_NULL_HANDLE;
    VkImageView storageImageView = VK_NULL_HANDLE;

    // Acceleration structures
    VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
    VkBuffer blasBuffer = VK_NULL_HANDLE;
    VmaAllocation blasAllocation = VK_NULL_HANDLE;

    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
    VkBuffer tlasBuffer = VK_NULL_HANDLE;
    VmaAllocation tlasAllocation = VK_NULL_HANDLE;

    // Instance buffer for TLAS
    VkBuffer instanceBuffer = VK_NULL_HANDLE;
    VmaAllocation instanceAllocation = VK_NULL_HANDLE;

    // Scratch buffers
    VkBuffer blasScratchBuffer = VK_NULL_HANDLE;
    VmaAllocation blasScratchAllocation = VK_NULL_HANDLE;
    VkBuffer tlasScratchBuffer = VK_NULL_HANDLE;
    VmaAllocation tlasScratchAllocation = VK_NULL_HANDLE;

    // RT pipeline
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    // Shader Binding Table
    VkBuffer sbtBuffer = VK_NULL_HANDLE;
    VmaAllocation sbtAllocation = VK_NULL_HANDLE;
    VkStridedDeviceAddressRegionKHR raygenSBT = {};
    VkStridedDeviceAddressRegionKHR missSBT = {};
    VkStridedDeviceAddressRegionKHR hitSBT = {};
    VkStridedDeviceAddressRegionKHR callableSBT = {};

    // Camera UBO
    VkBuffer cameraBuffer = VK_NULL_HANDLE;
    VmaAllocation cameraAllocation = VK_NULL_HANDLE;

    // Mesh for BLAS
    GPUMesh sphereMesh = {};

    // Shader modules
    VkShaderModule rgenModule = VK_NULL_HANDLE;
    VkShaderModule rmissModule = VK_NULL_HANDLE;
    VkShaderModule rchitModule = VK_NULL_HANDLE;

    void createStorageImage(VulkanContext& ctx);
    void createBLAS(VulkanContext& ctx);
    void createTLAS(VulkanContext& ctx);
    void createRTPipeline(VulkanContext& ctx);
    void createSBT(VulkanContext& ctx);
    void createDescriptorSet(VulkanContext& ctx);
    void updateCameraUBO(VulkanContext& ctx);
};
