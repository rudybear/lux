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
#include <unordered_map>

// Per-permutation pipeline data for multi-material rendering
struct PermutationPipeline {
    std::string suffix;                              // e.g. "+normal_map+sheen"
    std::string basePath;                            // full path with suffix

    VkShaderModule vertModule = VK_NULL_HANDLE;
    VkShaderModule fragModule = VK_NULL_HANDLE;
    ReflectionData vertRefl;
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

class RasterRenderer : public IRenderer {
public:
    // Initialize with SceneManager
    void init(VulkanContext& ctx, SceneManager& scene, const std::string& pipelineBase,
              const std::string& renderPath, uint32_t width, uint32_t height);

    void render(VulkanContext& ctx) override;

    // IRenderer interface
    VkImage getOutputImage() const override { return offscreenImage; }
    VkFormat getOutputFormat() const override { return VK_FORMAT_R8G8B8A8_UNORM; }
    uint32_t getWidth() const override { return renderWidth; }
    uint32_t getHeight() const override { return renderHeight; }
    void cleanup(VulkanContext& ctx) override;
    void updateCamera(VulkanContext& ctx, glm::vec3 eye, glm::vec3 target,
                      glm::vec3 up, float fovY, float aspect,
                      float nearPlane, float farPlane) override;
    void blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
                          VkImage swapImage, VkExtent2D extent) override;

    // Legacy accessors (still needed for getOffscreenImage in existing code)
    VkImage getOffscreenImage() const { return offscreenImage; }
    VkFormat getOffscreenFormat() const { return VK_FORMAT_R8G8B8A8_UNORM; }

    // Depth image access (for hybrid compositing with SplatRenderer)
    VkImage getDepthImage() const { return depthImage; }
    VkFormat getDepthFormat() const { return VK_FORMAT_D32_SFLOAT; }

    // Update a material UBO at runtime (editor property editing)
    void updateMaterialUBO(VulkanContext& ctx, int materialIndex, const MaterialUBOData& data);

    // Render to a swapchain image (interactive mode)
    void renderToSwapchain(VulkanContext& ctx, VkImage swapImage, VkImageView swapView,
                           VkFormat swapFormat, VkExtent2D extent,
                           VkSemaphore waitSem, VkSemaphore signalSem, VkFence fence);

private:
    std::string m_renderPath;  // "raster", "fullscreen", "rt"
    std::string m_pipelineBase; // pipeline base path for reflection JSON lookup

    // Bindless rendering mode
    bool m_bindlessMode = false;

    // Bindless resources
    BindlessTextureArray m_bindlessTextures;
    BindlessMaterialsSSBO m_bindlessMaterials;
    uint32_t renderWidth = 512;
    uint32_t renderHeight = 512;

    // Scene manager (owned externally)
    SceneManager* m_scene = nullptr;

    // Offscreen render target
    VkImage offscreenImage = VK_NULL_HANDLE;
    VmaAllocation offscreenAllocation = VK_NULL_HANDLE;
    VkImageView offscreenImageView = VK_NULL_HANDLE;

    // Depth buffer (raster scenes with depth)
    VkImage depthImage = VK_NULL_HANDLE;
    VmaAllocation depthAllocation = VK_NULL_HANDLE;
    VkImageView depthImageView = VK_NULL_HANDLE;
    bool m_needsDepth = false;

    // Render pass and framebuffer
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;

    // --- Single-pipeline mode (triangle, fullscreen, single-material PBR) ---
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

    // Reflection-driven descriptor sets (dynamic, not hardcoded)
    std::unordered_map<int, VkDescriptorSetLayout> reflectedSetLayouts;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::unordered_map<int, VkDescriptorSet> reflectedDescSets;

    // Uniform buffers (created based on reflection)
    VkBuffer mvpBuffer = VK_NULL_HANDLE;
    VmaAllocation mvpAllocation = VK_NULL_HANDLE;
    VkBuffer lightBuffer = VK_NULL_HANDLE;
    VmaAllocation lightAllocation = VK_NULL_HANDLE;
    VkBuffer m_materialBuffer = VK_NULL_HANDLE;
    VmaAllocation m_materialAllocation = VK_NULL_HANDLE;

    // Multi-light SSBO (when shader uses SceneLight UBO + lights SSBO)
    VkBuffer m_lightsSSBO = VK_NULL_HANDLE;
    VmaAllocation m_lightsSSBOAlloc = VK_NULL_HANDLE;
    VkBuffer m_sceneLightUBO = VK_NULL_HANDLE;
    VmaAllocation m_sceneLightUBOAlloc = VK_NULL_HANDLE;
    bool m_hasMultiLight = false;

    // Shadow map resources
    static constexpr uint32_t SHADOW_MAP_SIZE = 2048;
    static constexpr uint32_t MAX_SHADOW_MAPS = 8;
    VkImage m_shadowImage = VK_NULL_HANDLE;
    VmaAllocation m_shadowImageAlloc = VK_NULL_HANDLE;
    VkImageView m_shadowImageView = VK_NULL_HANDLE;   // array view for shader sampling
    std::vector<VkImageView> m_shadowLayerViews;       // per-layer views for framebuffers
    std::vector<VkFramebuffer> m_shadowFramebuffers;
    VkRenderPass m_shadowRenderPass = VK_NULL_HANDLE;
    VkSampler m_shadowSampler = VK_NULL_HANDLE;        // comparison sampler
    VkPipelineLayout m_shadowPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_shadowPipeline = VK_NULL_HANDLE;
    VkShaderModule m_shadowVertModule = VK_NULL_HANDLE;
    VkBuffer m_shadowMatricesSSBO = VK_NULL_HANDLE;
    VmaAllocation m_shadowMatricesSSBOAlloc = VK_NULL_HANDLE;
    bool m_hasShadows = false;
    int m_numShadowMaps = 0;

    // Per-material descriptor sets and buffers (single-pipeline fallback)
    std::vector<VkDescriptorSet> m_perMaterialDescSets;
    std::vector<VkBuffer> m_perMaterialBuffers;
    std::vector<VmaAllocation> m_perMaterialAllocations;
    VkDescriptorSet m_vertexDescSet = VK_NULL_HANDLE;

    // --- Multi-pipeline mode (per-material permutation selection) ---
    bool m_multiPipeline = false;
    std::vector<PermutationPipeline> m_permutations;
    std::vector<int> m_materialToPermutation;          // materialIndex -> permutation index
    VkDescriptorSetLayout m_sharedSet0Layout = VK_NULL_HANDLE;  // shared set 0 (MVP)

    // Triangle vertex buffer
    VkBuffer triangleVB = VK_NULL_HANDLE;
    VmaAllocation triangleVBAllocation = VK_NULL_HANDLE;

    // Shader modules (single-pipeline mode)
    VkShaderModule vertModule = VK_NULL_HANDLE;
    VkShaderModule fragModule = VK_NULL_HANDLE;

    // Reflection data cached from JSON parsing (single-pipeline mode)
    ReflectionData vertReflection;
    ReflectionData fragReflection;

    // Push constant data (from reflection, filled with scene lighting)
    std::vector<uint8_t> pushConstantData;
    VkShaderStageFlags pushConstantStageFlags = 0;
    uint32_t pushConstantSize = 0;

    // Command buffer tracking for interactive mode (freed on next frame)
    VkCommandBuffer m_interactiveCmdBuffer = VK_NULL_HANDLE;

    void createOffscreenTarget(VulkanContext& ctx);
    void createRenderPass(VulkanContext& ctx);
    void createFramebuffer(VulkanContext& ctx);
    void createPipelineTriangle(VulkanContext& ctx);
    void createPipelineFullscreen(VulkanContext& ctx);
    void createPipelinePBR(VulkanContext& ctx);
    void setupPBRResources(VulkanContext& ctx);
    void setupMultiLightResources(VulkanContext& ctx);

    // Shadow map infrastructure
    void setupShadowMaps(VulkanContext& ctx);
    void renderShadowMaps(VulkanContext& ctx, VkCommandBuffer cmd);
    void cleanupShadowMaps(VulkanContext& ctx);

    // Reflection-driven descriptor setup (single pipeline)
    void setupReflectedDescriptors(VulkanContext& ctx);

    // Multi-pipeline setup
    void setupMultiPipeline(VulkanContext& ctx, const ShaderManifest& manifest);

    // Record draw commands (shared between render() and renderToSwapchain())
    void recordDrawCommands(VkCommandBuffer cmd);
};
