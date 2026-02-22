#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "vulkan_context.h"
#include "scene.h"
#include "reflected_pipeline.h"
#include "renderer_interface.h"
#include "scene_manager.h"
#include "material_ubo.h"
#include "meshlet.h"
#include <string>
#include <vector>
#include <unordered_map>

// Per-permutation pipeline data for mesh shader multi-material rendering.
// The mesh shader (.mesh.spv) is SHARED across all permutations (it only reads
// geometry data), while the fragment shader (.frag.spv) differs per permutation
// (different BRDF features, texture bindings).
struct MeshPermutationPipeline {
    std::string suffix;                              // e.g. "+normal_map+sheen"
    std::string basePath;                            // full path with suffix

    VkShaderModule fragModule = VK_NULL_HANDLE;      // per-permutation frag shader
    ReflectionData fragRefl;

    VkDescriptorSetLayout materialSetLayout = VK_NULL_HANDLE;  // set 1 layout
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    // Per-material descriptor sets and UBOs for materials using this permutation
    std::vector<VkDescriptorSet> perMaterialDescSets;
    std::vector<VkBuffer> perMaterialUBOs;
    std::vector<VmaAllocation> perMaterialAllocations;
    std::vector<int> materialIndices;                // which scene materials use this
};

// A group of meshlets that share a common material.
// Built by partitioning meshlets per draw range so each group maps to a
// specific material (and therefore a specific shader permutation).
struct MeshletGroup {
    uint32_t firstMeshlet;   // offset into global meshlet descriptor array
    uint32_t meshletCount;   // how many meshlets in this group
    int materialIndex;       // scene material index
    int permutationIndex;    // index into m_meshPermutations (set during multi-pipeline setup)
};

class MeshRenderer : public IRenderer {
public:
    void init(VulkanContext& ctx, SceneManager& scene,
              const std::string& pipelineBase, uint32_t width, uint32_t height);

    void render(VulkanContext& ctx) override;

    // IRenderer interface
    VkImage getOutputImage() const override { return m_colorImage; }
    VkFormat getOutputFormat() const override { return VK_FORMAT_R8G8B8A8_UNORM; }
    uint32_t getWidth() const override { return m_width; }
    uint32_t getHeight() const override { return m_height; }
    void cleanup(VulkanContext& ctx) override;
    void updateCamera(VulkanContext& ctx, glm::vec3 eye, glm::vec3 target,
                      glm::vec3 up, float fovY, float aspect,
                      float nearPlane, float farPlane) override;
    void blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
                          VkImage swapImage, VkExtent2D extent) override;

    // Render to a swapchain image (interactive mode)
    void renderToSwapchain(VulkanContext& ctx, VkImage swapImage, VkImageView swapView,
                           VkFormat swapFormat, VkExtent2D extent,
                           VkSemaphore waitSem, VkSemaphore signalSem, VkFence fence);

private:
    uint32_t m_width = 0, m_height = 0;
    uint32_t m_totalMeshlets = 0;
    std::string m_pipelineBase;

    // Scene manager (owned externally)
    SceneManager* m_scene = nullptr;

    // Offscreen render target
    VkImage m_colorImage = VK_NULL_HANDLE;
    VmaAllocation m_colorAlloc = VK_NULL_HANDLE;
    VkImageView m_colorView = VK_NULL_HANDLE;
    VkImage m_depthImage = VK_NULL_HANDLE;
    VmaAllocation m_depthAlloc = VK_NULL_HANDLE;
    VkImageView m_depthView = VK_NULL_HANDLE;

    // Render pass & framebuffer
    VkRenderPass m_renderPass = VK_NULL_HANDLE;
    VkFramebuffer m_framebuffer = VK_NULL_HANDLE;

    // --- Single-pipeline mode (original behavior) ---
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;

    // Reflection-driven descriptor sets (single pipeline)
    std::unordered_map<int, VkDescriptorSetLayout> m_setLayouts;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    std::unordered_map<int, VkDescriptorSet> m_descriptorSets;

    // --- Multi-pipeline mode (per-material permutation selection) ---
    bool m_multiPipeline = false;
    std::vector<MeshPermutationPipeline> m_meshPermutations;
    std::vector<int> m_materialToPermutation;          // materialIndex -> permutation index
    VkDescriptorSetLayout m_sharedSet0Layout = VK_NULL_HANDLE;  // shared set 0 (meshlet data + MVP + Light)
    VkDescriptorSet m_sharedSet0 = VK_NULL_HANDLE;
    std::vector<MeshletGroup> m_meshletGroups;

    // Meshlet buffers
    VkBuffer m_meshletDescBuffer = VK_NULL_HANDLE;
    VmaAllocation m_meshletDescAlloc = VK_NULL_HANDLE;
    VkBuffer m_meshletVertBuffer = VK_NULL_HANDLE;
    VmaAllocation m_meshletVertAlloc = VK_NULL_HANDLE;
    VkBuffer m_meshletTriBuffer = VK_NULL_HANDLE;
    VmaAllocation m_meshletTriAlloc = VK_NULL_HANDLE;

    // SoA vertex buffers (same pattern as RT)
    VkBuffer m_positionsBuffer = VK_NULL_HANDLE;
    VmaAllocation m_positionsAlloc = VK_NULL_HANDLE;
    VkBuffer m_normalsBuffer = VK_NULL_HANDLE;
    VmaAllocation m_normalsAlloc = VK_NULL_HANDLE;
    VkBuffer m_texCoordsBuffer = VK_NULL_HANDLE;
    VmaAllocation m_texCoordsAlloc = VK_NULL_HANDLE;
    VkBuffer m_tangentsBuffer = VK_NULL_HANDLE;
    VmaAllocation m_tangentsAlloc = VK_NULL_HANDLE;

    // Uniform buffers
    VkBuffer m_mvpBuffer = VK_NULL_HANDLE;
    VmaAllocation m_mvpAlloc = VK_NULL_HANDLE;
    VkBuffer m_lightBuffer = VK_NULL_HANDLE;
    VmaAllocation m_lightAlloc = VK_NULL_HANDLE;
    VkBuffer m_materialBuffer = VK_NULL_HANDLE;
    VmaAllocation m_materialAllocation = VK_NULL_HANDLE;

    // Shader modules
    VkShaderModule m_meshModule = VK_NULL_HANDLE;
    VkShaderModule m_fragModule = VK_NULL_HANDLE;

    // Reflection data cached from JSON parsing
    ReflectionData m_meshReflection;
    ReflectionData m_fragReflection;

    // Buffer sizes for descriptor writes
    VkDeviceSize m_positionsSize = 0;
    VkDeviceSize m_normalsSize = 0;
    VkDeviceSize m_texCoordsSize = 0;
    VkDeviceSize m_tangentsSize = 0;
    VkDeviceSize m_meshletDescSize = 0;
    VkDeviceSize m_meshletVertSize = 0;
    VkDeviceSize m_meshletTriSize = 0;

    void createOffscreenTarget(VulkanContext& ctx);
    void createRenderPass(VulkanContext& ctx);
    void createFramebuffer(VulkanContext& ctx);
    void createPipeline(VulkanContext& ctx);
    void setupDescriptors(VulkanContext& ctx);
    void uploadMeshletData(VulkanContext& ctx);
    void uploadVertexData(VulkanContext& ctx);

    // Multi-pipeline setup (mesh shader permutations)
    void setupMeshMultiPipeline(VulkanContext& ctx, const ShaderManifest& manifest);
    void cleanupPermutations(VulkanContext& ctx);
};
