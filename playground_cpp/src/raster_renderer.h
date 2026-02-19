#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "vulkan_context.h"
#include "scene.h"
#include <string>

enum class RasterMode {
    Triangle,
    Fullscreen,
    PBR
};

class RasterRenderer {
public:
    void init(VulkanContext& ctx, RasterMode mode,
              const std::string& vertSpvPath, const std::string& fragSpvPath,
              uint32_t width, uint32_t height);

    void render(VulkanContext& ctx);

    // Get the offscreen image for screenshot
    VkImage getOffscreenImage() const { return offscreenImage; }
    VkFormat getOffscreenFormat() const { return VK_FORMAT_R8G8B8A8_UNORM; }
    uint32_t getWidth() const { return renderWidth; }
    uint32_t getHeight() const { return renderHeight; }

    // Render to a swapchain image (interactive mode)
    void renderToSwapchain(VulkanContext& ctx, VkImage swapImage, VkImageView swapView,
                           VkFormat swapFormat, VkExtent2D extent,
                           VkSemaphore waitSem, VkSemaphore signalSem, VkFence fence);

    void cleanup(VulkanContext& ctx);

private:
    RasterMode mode = RasterMode::Triangle;
    uint32_t renderWidth = 512;
    uint32_t renderHeight = 512;

    // Offscreen render target
    VkImage offscreenImage = VK_NULL_HANDLE;
    VmaAllocation offscreenAllocation = VK_NULL_HANDLE;
    VkImageView offscreenImageView = VK_NULL_HANDLE;

    // Depth buffer (PBR mode)
    VkImage depthImage = VK_NULL_HANDLE;
    VmaAllocation depthAllocation = VK_NULL_HANDLE;
    VkImageView depthImageView = VK_NULL_HANDLE;

    // Render pass and framebuffer
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;

    // Pipeline
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

    // Descriptor sets (PBR mode)
    VkDescriptorSetLayout descSetLayout0 = VK_NULL_HANDLE; // MVP
    VkDescriptorSetLayout descSetLayout1 = VK_NULL_HANDLE; // Light + texture
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet0 = VK_NULL_HANDLE;
    VkDescriptorSet descSet1 = VK_NULL_HANDLE;

    // Uniform buffers (PBR mode)
    VkBuffer mvpBuffer = VK_NULL_HANDLE;
    VmaAllocation mvpAllocation = VK_NULL_HANDLE;
    VkBuffer lightBuffer = VK_NULL_HANDLE;
    VmaAllocation lightAllocation = VK_NULL_HANDLE;

    // Mesh data
    GPUMesh mesh = {};
    GPUTexture texture = {};

    // Triangle vertex buffer
    VkBuffer triangleVB = VK_NULL_HANDLE;
    VmaAllocation triangleVBAllocation = VK_NULL_HANDLE;

    // Shader modules
    VkShaderModule vertModule = VK_NULL_HANDLE;
    VkShaderModule fragModule = VK_NULL_HANDLE;

    void createOffscreenTarget(VulkanContext& ctx);
    void createRenderPass(VulkanContext& ctx);
    void createFramebuffer(VulkanContext& ctx);
    void createPipelineTriangle(VulkanContext& ctx);
    void createPipelineFullscreen(VulkanContext& ctx);
    void createPipelinePBR(VulkanContext& ctx);
    void setupPBRResources(VulkanContext& ctx);
};
