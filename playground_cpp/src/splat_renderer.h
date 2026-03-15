#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <cstdint>

struct VulkanContext;
struct GaussianSplatData;

class SplatRenderer {
public:
    SplatRenderer() = default;
    ~SplatRenderer();

    void init(VulkanContext& ctx, const GaussianSplatData& data,
              const std::string& shaderBase, uint32_t width, uint32_t height);

    void updateCamera(glm::vec3 eye, glm::vec3 target, glm::vec3 up,
                      float fovY, float aspect, float nearPlane, float farPlane);

    void render(VulkanContext& ctx);

    void blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
                         VkImage swapImage, VkExtent2D extent);

    // Blit to swapchain in compositing mode: transitions from PRESENT_SRC
    // instead of UNDEFINED (for drawing on top of a previously-rendered frame).
    void blitToSwapchainComposite(VulkanContext& ctx, VkCommandBuffer cmd,
                                   VkImage swapImage, VkExtent2D extent);

    // Preload a background image into the splat color buffer.
    // Subsequent render() call will use LOAD instead of CLEAR so splats
    // are composited on top of the background.
    void preloadBackground(VulkanContext& ctx, VkImage srcImage, VkFormat srcFormat,
                           uint32_t srcWidth, uint32_t srcHeight);

    // Preload depth from raster pass into splat depth buffer.
    // Splats will depth-test against mesh geometry so occluded splats are hidden.
    void preloadDepth(VulkanContext& ctx, VkImage srcDepthImage,
                      uint32_t srcWidth, uint32_t srcHeight);

    void cleanup(VulkanContext& ctx);

    VkImage getOutputImage() const { return colorImage_; }
    VkFormat getOutputFormat() const { return VK_FORMAT_R8G8B8A8_UNORM; }
    uint32_t getWidth() const { return width_; }
    uint32_t getHeight() const { return height_; }

private:
    uint32_t width_ = 0, height_ = 0;
    uint32_t numSplats_ = 0;

    // Offscreen render targets
    VkImage colorImage_ = VK_NULL_HANDLE;
    VmaAllocation colorAlloc_ = VK_NULL_HANDLE;
    VkImageView colorView_ = VK_NULL_HANDLE;

    VkImage depthImage_ = VK_NULL_HANDLE;
    VmaAllocation depthAlloc_ = VK_NULL_HANDLE;
    VkImageView depthView_ = VK_NULL_HANDLE;

    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkFramebuffer framebuffer_ = VK_NULL_HANDLE;

    // Second render pass/framebuffer pair with LOAD_OP_LOAD for color compositing
    VkRenderPass renderPassLoad_ = VK_NULL_HANDLE;
    VkFramebuffer framebufferLoad_ = VK_NULL_HANDLE;
    bool hasBackground_ = false;

    // Third render pass/framebuffer pair: LOAD both color AND depth (full hybrid compositing)
    VkRenderPass renderPassLoadDepth_ = VK_NULL_HANDLE;
    VkFramebuffer framebufferLoadDepth_ = VK_NULL_HANDLE;
    bool hasBackgroundDepth_ = false;

    // Pipelines
    VkPipeline computePipeline_ = VK_NULL_HANDLE;
    VkPipeline renderPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout computeLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout renderLayout_ = VK_NULL_HANDLE;

    // Descriptor sets
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout renderSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSet computeDescSet_ = VK_NULL_HANDLE;
    VkDescriptorSet renderDescSet_ = VK_NULL_HANDLE;

    // Splat GPU buffers (VMA)
    VkBuffer posBuffer_ = VK_NULL_HANDLE;       VmaAllocation posAlloc_ = VK_NULL_HANDLE;
    VkBuffer rotBuffer_ = VK_NULL_HANDLE;       VmaAllocation rotAlloc_ = VK_NULL_HANDLE;
    VkBuffer scaleBuffer_ = VK_NULL_HANDLE;     VmaAllocation scaleAlloc_ = VK_NULL_HANDLE;
    VkBuffer opacityBuffer_ = VK_NULL_HANDLE;   VmaAllocation opacityAlloc_ = VK_NULL_HANDLE;
    std::vector<VkBuffer> shBuffers_;
    std::vector<VmaAllocation> shAllocs_;

    // Projected output buffers
    VkBuffer projCenterBuffer_ = VK_NULL_HANDLE;  VmaAllocation projCenterAlloc_ = VK_NULL_HANDLE;
    VkBuffer projConicBuffer_ = VK_NULL_HANDLE;   VmaAllocation projConicAlloc_ = VK_NULL_HANDLE;
    VkBuffer projColorBuffer_ = VK_NULL_HANDLE;   VmaAllocation projColorAlloc_ = VK_NULL_HANDLE;

    // Sort buffers (buffer A = primary, written by compute shader)
    VkBuffer sortKeysBuffer_ = VK_NULL_HANDLE;      VmaAllocation sortKeysAlloc_ = VK_NULL_HANDLE;
    VkBuffer sortedIndicesBuffer_ = VK_NULL_HANDLE;  VmaAllocation sortedIndicesAlloc_ = VK_NULL_HANDLE;

    // Visible count (atomic counter)
    VkBuffer visibleCountBuffer_ = VK_NULL_HANDLE;  VmaAllocation visibleCountAlloc_ = VK_NULL_HANDLE;

    // GPU radix sort resources
    VkBuffer sortKeysBBuffer_ = VK_NULL_HANDLE;      VmaAllocation sortKeysBAlloc_ = VK_NULL_HANDLE;
    VkBuffer sortValsBBuffer_ = VK_NULL_HANDLE;      VmaAllocation sortValsBAlloc_ = VK_NULL_HANDLE;
    VkBuffer histogramBuffer_ = VK_NULL_HANDLE;      VmaAllocation histogramAlloc_ = VK_NULL_HANDLE;
    VkBuffer partitionSumsBuffer_ = VK_NULL_HANDLE;  VmaAllocation partitionSumsAlloc_ = VK_NULL_HANDLE;

    // Sort pipelines (3 compute stages)
    VkPipeline sortHistogramPipeline_ = VK_NULL_HANDLE;
    VkPipeline sortPrefixSumPipeline_ = VK_NULL_HANDLE;
    VkPipeline sortScatterPipeline_ = VK_NULL_HANDLE;

    // Sort pipeline layouts
    VkPipelineLayout sortHistogramLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout sortPrefixSumLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout sortScatterLayout_ = VK_NULL_HANDLE;

    // Sort descriptor set layouts
    VkDescriptorSetLayout sortHistogramSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout sortPrefixSumSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout sortScatterSetLayout_ = VK_NULL_HANDLE;

    // Sort descriptor sets: [0]=A->B, [1]=B->A for histogram and scatter; single for prefix_sum
    VkDescriptorSet sortHistogramDescSets_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet sortPrefixSumDescSet_ = VK_NULL_HANDLE;
    VkDescriptorSet sortScatterDescSets_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    // Precomputed sort workgroup count
    uint32_t sortNumWg_ = 0;

    // Camera state
    glm::mat4 viewMatrix_{1.0f};
    glm::mat4 projMatrix_{1.0f};
    glm::vec3 camPos_{0.0f, 0.0f, 3.0f};
    float focalX_ = 256.0f;
    float focalY_ = 256.0f;
    uint32_t shDegree_ = 0;        // scene's actual SH degree (for push constant)
    uint32_t shaderShDegree_ = 0;  // shader's compiled SH degree (for descriptor layout)

    // Helpers
    void createOffscreenTarget(VulkanContext& ctx);
    void createRenderPass(VkDevice device);
    void createRenderPassLoad(VkDevice device);
    void createRenderPassLoadDepth(VkDevice device);
    void createFramebuffer(VkDevice device);
    void createFramebufferLoad(VkDevice device);
    void createFramebufferLoadDepth(VkDevice device);
    void createPipelines(VkDevice device, const std::string& shaderBase);
    void createSortPipelines(VkDevice device);
    void createBuffers(VulkanContext& ctx, const GaussianSplatData& data);
};
