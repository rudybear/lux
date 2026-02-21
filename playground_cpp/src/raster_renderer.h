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

    // Render to a swapchain image (interactive mode)
    void renderToSwapchain(VulkanContext& ctx, VkImage swapImage, VkImageView swapView,
                           VkFormat swapFormat, VkExtent2D extent,
                           VkSemaphore waitSem, VkSemaphore signalSem, VkFence fence);

private:
    std::string m_renderPath;  // "raster", "fullscreen", "rt"
    std::string m_pipelineBase; // pipeline base path for reflection JSON lookup
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

    // Pipeline
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

    // Triangle vertex buffer
    VkBuffer triangleVB = VK_NULL_HANDLE;
    VmaAllocation triangleVBAllocation = VK_NULL_HANDLE;

    // Shader modules
    VkShaderModule vertModule = VK_NULL_HANDLE;
    VkShaderModule fragModule = VK_NULL_HANDLE;

    // Reflection data cached from JSON parsing
    ReflectionData vertReflection;
    ReflectionData fragReflection;

    // Push constant data (from reflection, filled with scene lighting)
    std::vector<uint8_t> pushConstantData;
    VkShaderStageFlags pushConstantStageFlags = 0;
    uint32_t pushConstantSize = 0;

    void createOffscreenTarget(VulkanContext& ctx);
    void createRenderPass(VulkanContext& ctx);
    void createFramebuffer(VulkanContext& ctx);
    void createPipelineTriangle(VulkanContext& ctx);
    void createPipelineFullscreen(VulkanContext& ctx);
    void createPipelinePBR(VulkanContext& ctx);
    void setupPBRResources(VulkanContext& ctx);

    // Reflection-driven descriptor setup
    void setupReflectedDescriptors(VulkanContext& ctx);
};
