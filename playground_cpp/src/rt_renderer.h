#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "vulkan_context.h"
#include "scene.h"
#include "gltf_loader.h"
#include "reflected_pipeline.h"
#include "renderer_interface.h"
#include "scene_manager.h"
#include <string>
#include <vector>
#include <unordered_map>

class RTRenderer : public IRenderer {
public:
    void init(VulkanContext& ctx,
              const std::string& rgenSpvPath,
              const std::string& rmissSpvPath,
              const std::string& rchitSpvPath,
              uint32_t width, uint32_t height,
              SceneManager& scene);

    void render(VulkanContext& ctx) override;

    // IRenderer interface
    VkImage getOutputImage() const override { return storageImage; }
    VkFormat getOutputFormat() const override { return VK_FORMAT_R8G8B8A8_UNORM; }
    uint32_t getWidth() const override { return renderWidth; }
    uint32_t getHeight() const override { return renderHeight; }
    void cleanup(VulkanContext& ctx) override;
    void updateCamera(VulkanContext& ctx, glm::vec3 eye, glm::vec3 target,
                      glm::vec3 up, float fovY, float aspect,
                      float nearPlane, float farPlane) override;
    void blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
                          VkImage swapImage, VkExtent2D extent) override;

private:
    uint32_t renderWidth = 512;
    uint32_t renderHeight = 512;
    std::string m_pipelineBase; // for reflection JSON lookup

    // Scene manager (owned externally)
    SceneManager* m_scene = nullptr;

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

    // Reflection-driven descriptor sets
    std::unordered_map<int, VkDescriptorSetLayout> reflectedSetLayouts;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::unordered_map<int, VkDescriptorSet> reflectedDescSets;

    // Reflection data from JSON
    std::vector<ReflectionData> stageReflections;

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

    // Mesh for BLAS (owned by SceneManager, but we keep a local reference for BLAS geometry)
    GPUMesh sphereMesh = {};

    // SoA storage buffers for closest_hit vertex interpolation
    VkBuffer positionsBuffer = VK_NULL_HANDLE;
    VmaAllocation positionsAllocation = VK_NULL_HANDLE;
    VkDeviceSize positionsSize = 0;

    VkBuffer normalsBuffer = VK_NULL_HANDLE;
    VmaAllocation normalsAllocation = VK_NULL_HANDLE;
    VkDeviceSize normalsSize = 0;

    VkBuffer texCoordsBuffer = VK_NULL_HANDLE;
    VmaAllocation texCoordsAllocation = VK_NULL_HANDLE;
    VkDeviceSize texCoordsSize = 0;

    VkBuffer indexStorageBuffer = VK_NULL_HANDLE;
    VmaAllocation indexStorageAllocation = VK_NULL_HANDLE;
    VkDeviceSize indexStorageSize = 0;

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
