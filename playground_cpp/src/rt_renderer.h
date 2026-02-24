#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "vulkan_context.h"
#include "scene.h"
#include "gltf_loader.h"
#include "reflected_pipeline.h"
#include "renderer_interface.h"
#include "scene_manager.h"
#include "material_ubo.h"
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

// Per-permutation RT hit group data for multi-material rendering
struct RTHitPermutation {
    std::string suffix;             // e.g. "+normal_map+sheen"
    std::string basePath;           // full path with suffix
    VkShaderModule rchitModule = VK_NULL_HANDLE;
    ReflectionData rchitRefl;
    std::vector<int> materialIndices;  // which scene materials use this permutation
    uint32_t hitGroupIndex = 0;        // index into shader groups (offset by 2 for rgen+rmiss)
};

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

    // Material UBO (single-material mode)
    VkBuffer m_materialBuffer = VK_NULL_HANDLE;
    VmaAllocation m_materialAllocation = VK_NULL_HANDLE;

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

    // Shader modules (raygen + miss are always single; rchit varies)
    VkShaderModule rgenModule = VK_NULL_HANDLE;
    VkShaderModule rmissModule = VK_NULL_HANDLE;
    VkShaderModule rchitModule = VK_NULL_HANDLE;  // single-material fallback

    // --- Bindless RT mode (Phase 5) ---
    bool m_bindlessMode = false;
    BindlessTextureArray m_bindlessTextures;
    BindlessMaterialsSSBO m_bindlessMaterials;
    int m_bindlessTextureSet = -1;       // descriptor set index for texture array
    int m_bindlessTextureBinding = -1;   // binding index for texture array
    int m_bindlessMaterialsSet = -1;     // descriptor set index for materials SSBO
    int m_bindlessMaterialsBinding = -1; // binding index for materials SSBO
    uint32_t m_bindlessMaxCount = 1024;  // max bindless texture count

    // --- Multi-material RT fields ---
    bool m_multiMaterial = false;
    uint32_t m_geometryCount = 1;  // number of BLAS geometries (one per draw range)

    // Per-permutation hit group data
    std::vector<RTHitPermutation> m_hitPermutations;

    // Mapping: geometry index (BLAS) -> permutation index
    std::vector<int> m_geometryToPermutation;

    // Mapping: geometry index -> hit group index (for SBT)
    std::vector<uint32_t> m_geometryToHitGroup;

    // Per-material UBOs (one per material)
    std::vector<VkBuffer> m_perMaterialBuffers;
    std::vector<VmaAllocation> m_perMaterialAllocations;

    // Per-material descriptor sets (one per material, for texture + material binding)
    std::vector<VkDescriptorSet> m_perMaterialDescSets;

    void createStorageImage(VulkanContext& ctx);
    void createBLAS(VulkanContext& ctx);
    void createTLAS(VulkanContext& ctx);
    void createRTPipeline(VulkanContext& ctx);
    void createSBT(VulkanContext& ctx);
    void createDescriptorSet(VulkanContext& ctx);
    void updateCameraUBO(VulkanContext& ctx);
};
