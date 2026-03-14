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

class DeferredRenderer : public IRenderer {
public:
    void init(VulkanContext& ctx, SceneManager& scene, const std::string& pipelineBase,
              uint32_t width, uint32_t height);

    void render(VulkanContext& ctx) override;

    // IRenderer interface
    VkImage getOutputImage() const override { return m_lightOutputImage; }
    VkFormat getOutputFormat() const override { return VK_FORMAT_R8G8B8A8_UNORM; }
    uint32_t getWidth() const override { return m_renderWidth; }
    uint32_t getHeight() const override { return m_renderHeight; }
    void cleanup(VulkanContext& ctx) override;
    void updateCamera(VulkanContext& ctx, glm::vec3 eye, glm::vec3 target,
                      glm::vec3 up, float fovY, float aspect,
                      float nearPlane, float farPlane) override;
    void blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
                          VkImage swapImage, VkExtent2D extent) override;

    // Interactive mode
    void renderToSwapchain(VulkanContext& ctx, VkImage swapImage, VkImageView swapView,
                           VkFormat swapFormat, VkExtent2D extent,
                           VkSemaphore waitSem, VkSemaphore signalSem, VkFence fence);

private:
    uint32_t m_renderWidth = 512;
    uint32_t m_renderHeight = 512;

    // Scene manager (owned externally)
    SceneManager* m_scene = nullptr;
    std::string m_pipelineBase;

    // --- G-buffer resources ---
    // RT0: RGBA8 SRGB (base color + flags)
    VkImage m_gbufRT0Image = VK_NULL_HANDLE;
    VmaAllocation m_gbufRT0Alloc = VK_NULL_HANDLE;
    VkImageView m_gbufRT0View = VK_NULL_HANDLE;

    // RT1: RGB10A2 UNORM (normals packed)
    VkImage m_gbufRT1Image = VK_NULL_HANDLE;
    VmaAllocation m_gbufRT1Alloc = VK_NULL_HANDLE;
    VkImageView m_gbufRT1View = VK_NULL_HANDLE;

    // RT2: RGBA16F (metallic, roughness, emissive, etc.)
    VkImage m_gbufRT2Image = VK_NULL_HANDLE;
    VmaAllocation m_gbufRT2Alloc = VK_NULL_HANDLE;
    VkImageView m_gbufRT2View = VK_NULL_HANDLE;

    // Depth: D32_SFLOAT
    VkImage m_gbufDepthImage = VK_NULL_HANDLE;
    VmaAllocation m_gbufDepthAlloc = VK_NULL_HANDLE;
    VkImageView m_gbufDepthView = VK_NULL_HANDLE;

    // G-buffer render pass + framebuffer
    VkRenderPass m_gbufRenderPass = VK_NULL_HANDLE;
    VkFramebuffer m_gbufFramebuffer = VK_NULL_HANDLE;

    // G-buffer pipeline
    VkShaderModule m_gbufVertModule = VK_NULL_HANDLE;
    VkShaderModule m_gbufFragModule = VK_NULL_HANDLE;
    VkPipelineLayout m_gbufPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_gbufPipeline = VK_NULL_HANDLE;

    // G-buffer reflection data
    ReflectionData m_gbufVertRefl;
    ReflectionData m_gbufFragRefl;

    // G-buffer descriptor sets (reflection-driven)
    std::unordered_map<int, VkDescriptorSetLayout> m_gbufSetLayouts;
    VkDescriptorPool m_gbufDescPool = VK_NULL_HANDLE;
    VkDescriptorSet m_gbufVertDescSet = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_gbufPerMaterialDescSets;
    std::vector<VkBuffer> m_gbufPerMaterialBuffers;
    std::vector<VmaAllocation> m_gbufPerMaterialAllocations;

    // --- Lighting pass resources ---
    // Final output image (RGBA8)
    VkImage m_lightOutputImage = VK_NULL_HANDLE;
    VmaAllocation m_lightOutputAlloc = VK_NULL_HANDLE;
    VkImageView m_lightOutputView = VK_NULL_HANDLE;

    // Lighting render pass + framebuffer
    VkRenderPass m_lightRenderPass = VK_NULL_HANDLE;
    VkFramebuffer m_lightFramebuffer = VK_NULL_HANDLE;

    // Lighting pipeline
    VkShaderModule m_lightVertModule = VK_NULL_HANDLE;
    VkShaderModule m_lightFragModule = VK_NULL_HANDLE;
    VkPipelineLayout m_lightPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_lightPipeline = VK_NULL_HANDLE;

    // Lighting reflection data
    ReflectionData m_lightVertRefl;
    ReflectionData m_lightFragRefl;

    // Lighting descriptor sets
    std::unordered_map<int, VkDescriptorSetLayout> m_lightSetLayouts;
    VkDescriptorPool m_lightDescPool = VK_NULL_HANDLE;
    std::unordered_map<int, VkDescriptorSet> m_lightDescSets;

    // G-buffer sampler for lighting pass sampling
    VkSampler m_gbufSampler = VK_NULL_HANDLE;

    // --- Uniform buffers ---
    // MVP UBO (3 x mat4 = 192 bytes)
    VkBuffer m_mvpBuffer = VK_NULL_HANDLE;
    VmaAllocation m_mvpAlloc = VK_NULL_HANDLE;

    // Light UBO (vec3 light_dir + vec3 view_pos = 32 bytes)
    VkBuffer m_lightBuffer = VK_NULL_HANDLE;
    VmaAllocation m_lightAlloc = VK_NULL_HANDLE;

    // Material UBO (144 bytes std140)
    VkBuffer m_materialBuffer = VK_NULL_HANDLE;
    VmaAllocation m_materialAlloc = VK_NULL_HANDLE;

    // DeferredCamera UBO (mat4 inv_view_proj + vec3 view_pos = 80 bytes std140)
    VkBuffer m_deferredCameraBuffer = VK_NULL_HANDLE;
    VmaAllocation m_deferredCameraAlloc = VK_NULL_HANDLE;

    // Multi-light SSBO
    VkBuffer m_lightsSSBO = VK_NULL_HANDLE;
    VmaAllocation m_lightsSSBOAlloc = VK_NULL_HANDLE;
    VkBuffer m_sceneLightUBO = VK_NULL_HANDLE;
    VmaAllocation m_sceneLightUBOAlloc = VK_NULL_HANDLE;
    bool m_hasMultiLight = false;

    // Push constant data (G-buffer pass)
    std::vector<uint8_t> m_pushConstantData;
    VkShaderStageFlags m_pushConstantStageFlags = 0;
    uint32_t m_pushConstantSize = 0;

    // Command buffer tracking for interactive mode
    VkCommandBuffer m_interactiveCmdBuffer = VK_NULL_HANDLE;

    // Private setup methods
    void createGBufferImages(VulkanContext& ctx);
    void createGBufferRenderPass(VulkanContext& ctx);
    void createGBufferFramebuffer(VulkanContext& ctx);
    void createGBufferPipeline(VulkanContext& ctx);
    void setupGBufferDescriptors(VulkanContext& ctx);

    void createLightingOutputImage(VulkanContext& ctx);
    void createLightingRenderPass(VulkanContext& ctx);
    void createLightingFramebuffer(VulkanContext& ctx);
    void createLightingPipeline(VulkanContext& ctx);
    void setupLightingDescriptors(VulkanContext& ctx);

    void setupPBRResources(VulkanContext& ctx);
    void setupMultiLightResources(VulkanContext& ctx);
    void setupDeferredCameraUBO(VulkanContext& ctx);

    // Record G-buffer draw commands
    void recordGBufferCommands(VkCommandBuffer cmd);
};
